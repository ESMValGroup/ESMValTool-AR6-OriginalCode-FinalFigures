; functions.ncl
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs: 
;
; History
;    20200917 kosaka_yu: revised to add significance in calc_trend.
;    20200914 kosaka_yu: revised to account for weighting in percentiles.
;    20200525 kosaka_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

; =================================================================
;
;                          Spatial functions
;
; =================================================================

function fillLand(x[*][*][*])
local y, z, i, j, imax, jmax, wgt, tmp, iter, yave
begin
  y = x
  imax = dimsizes(x&lon)-1
  jmax = dimsizes(x&lat)-1
  wgt = new((/3, 3/), typeof(x))
  wgt(2, 0:2) = (/ 1., 2., 1/)
  wgt(1, 0:2) = (/ 2., 4., 2/)
  wgt(0, 0:2) = (/ 1., 2., 1/)
  tmp = x(:, :, 0:2)
  do iter = 0, 3
    if (all(.not.ismissing(y))) then
      break
    end if
;print("iter = "+iter+" #missing = "+num(ismissing(y)))
    z = y
    i = 0
    tmp(:, :,   0) = z(:, :, imax)
    tmp(:, :, 1:2) = z(:, :, 0:1)
    j = 0
    y(:, j, i) = where(ismissing(z(:, j, i)), \
                       wgt_areaave2(tmp(:, j:j+1, 0:2), wgt(1:2, :), 0), \
                       z(:, j, i))
    do j = 1, jmax-1
      y(:, j, i) = where(ismissing(z(:, j, i)), \
                         wgt_areaave2(tmp(:, j-1:j+1, 0:2), wgt, 0), \
                         z(:, j, i))
    end do
    j = jmax
    y(:, j, i) = where(ismissing(z(:, j, i)), \
                       wgt_areaave2(tmp(:, j-1:j, 0:2), wgt(0:1, :), 0), \
                       z(:, j, i))
    do i = 1, imax-1
      j = 0
      y(:, j, i) = where(ismissing(z(:, j, i)), \
                         wgt_areaave2(z(:, j:j+1, i-1:i+1), wgt(1:2, :), 0), \
                         z(:, j, i))
      do j = 1, jmax-1
        y(:, j, i) = where(ismissing(z(:, j, i)), \
                           wgt_areaave2(z(:, j-1:j+1, i-1:i+1), wgt, 0), \
                           z(:, j, i))
      end do
      j = jmax
      y(:, j, i) = where(ismissing(z(:, j, i)), \
                         wgt_areaave2(z(:, j-1:j, i-1:i+1), wgt(0:1, :), 0), \
                         z(:, j, i))
    end do
    i = imax
    tmp(:, :, 0:1) = z(:, :, imax-1:imax)
    tmp(:, :,   2) = z(:, :, 0)
    j = 0
    y(:, j, i) = where(ismissing(z(:, j, i)), \
                       wgt_areaave2(tmp(:, j:j+1, 0:2), wgt(1:2, :), 0), \
                       z(:, j, i))
    do j = 1, jmax-1
      y(:, j, i) = where(ismissing(z(:, j, i)), \
                         wgt_areaave2(tmp(:, j-1:j+1, 0:2), wgt, 0), \
                         z(:, j, i))
    end do
    j = jmax
    y(:, j, i) = where(ismissing(z(:, j, i)), \
                       wgt_areaave2(tmp(:, j-1:j, 0:2), wgt(0:1, :), 0), \
                       z(:, j, i))
  end do
  if (isvar("tmp")) then
    delete(tmp)
  end if
  if (isvar("z")) then
    delete(z)
  end if
  yave = wgt_areaave(y, 1., 1., 0)
  y = where(ismissing(y), conform(y, yave, 0), y)
  return(y)
end

function apply_mask(x, ref)
local x, y, ref, stime, etime
begin
  stime = max((/x&time(0), ref&time(0)/))
  etime = min((/x&time(dimsizes(x&time)-1), ref&time(dimsizes(ref&time)-1)/))
  if (dimsizes(dimsizes(x)).eq.3) then
    y = x({stime:etime}, :, :)
    y = where(ismissing(ref({stime:etime}, :, :)), y@_FillValue, y)
  elseif (dimsizes(dimsizes(x)).eq.4) then
    y = x({stime:etime}, :, :, :)
    y = where(ismissing(ref({stime:etime}, :, :, :)), y@_FillValue, y)
  end if
  return(y)
  delete(y)
end

function global_mean(x, method)
local x, y, method, individual_hemisphere, nh, sh, gm, lat, wgty, area, lats, latn, lat_bnds, ii, ave_band
begin
  lat = x&lat
  d2r = get_d2r(typeof(lat))
  if (any(method.eq.(/"HadCRUT4", "HadCRUT5"/))) then
    if (any(lat.eq.0.)) then
      error_msg("f", DIAG_SCRIPT, "", "Grid point exactly on the equator is UNSUPPORTED.")
    end if
    wgty = latRegWgt(lat, typeof(lat), 0)
    y = x
    y(:, {:0.}, :) = y@_FillValue
    nh = wgt_areaave_Wrap(y, wgty, 1., 0)  ; Assume equatorially symmetric grid, no grid point on the equator
    y = x
    y(:, {0.:}, :) = y@_FillValue
    sh = wgt_areaave_Wrap(y, wgty, 1., 0)  ; Assume equatorially symmetric grid, no grid point on the equator
    gm = x(:, 0, 0)
    gm = (nh + sh)/2.
  elseif (method.eq."GISTEMP") then
    gm = x(:, 0, 0)
    gm = 0.
    do ii = 0, 7
      if (ii.eq.0) then
        latn = 90.
        lats = 64.2
        area = 0.05
      elseif (ii.eq.1) then
        latn = 64.2
        lats = 44.4
        area = 0.1
      elseif (ii.eq.2) then
        latn = 44.4
        lats = 23.6
        area = 0.15
      elseif (ii.eq.3) then
        latn = 23.6
        lats = 0.
        area = 0.2
      elseif (ii.eq.4) then
        latn = 0.
        lats = -23.6
        area = 0.2
      elseif (ii.eq.5) then
        latn = -23.6
        lats = -44.4
        area = 0.15
      elseif (ii.eq.6) then
        latn = -44.4
        lats = -64.2
        area = 0.1
      elseif (ii.eq.7) then
        latn = -64.2
        lats = -90.
        area = 0.05
      end if
      y = x(:, {lats:latn}, :)
      jmax = dimsizes(y&lat)
      lat_bnds = new((/jmax, 2/), typeof(y&lat))
      wgty = new(jmax, typeof(y&lat))
      do j = 1, jmax-1
        lat_bnds(j-1, 1) = (y&lat(j-1) + y&lat(j))/2.
        lat_bnds(j, 0) = lat_bnds(j-1, 1)
      end do
      if (isMonotonic(y&lat).eq.1) then
        lat_bnds(0, 0) = lats
        lat_bnds(jmax-1, 1) = latn
      else
        lat_bnds(0, 0) = latn
        lat_bnds(jmax-1, 1) = lats
      end if
      wgty = (cos(lat_bnds(:, 0)*d2r) + cos(lat_bnds(:, 1)*d2r)) \
              * abs(lat_bnds(:, 1) - lat_bnds(:, 0)) / 2.
      ave_band = wgt_areaave(y, wgty, 1., 0)
      gm = gm + ave_band * area
      delete(y)
      delete(ave_band)
      delete(wgty)
      delete(lat_bnds)
    end do
  else
    wgty = latRegWgt(lat, typeof(lat), 0)
    gm = wgt_areaave_Wrap(x, wgty, 1., 0)
  end if
  return(gm)
end

function blend_tas_tos(tas, tos, sic, sftlf, method)
;local tas, tos, sic, sftlf, lf, lftmp, ts, SAT, SST, SIC, SFTLF, stime, etime, method
local tas, tos, sic, sftlf, lf, lftmp, frac, ts, stime, etime, method
begin
;  SAT = set_yyyymm(tas)
;  SST = set_yyyymm(tos)
;  SIC = set_yyyymm(sic)
  stime = max((/tas&time(0), tos&time(0), sic&time(0)/))
  etime = min((/tas&time(dimsizes(tas&time)-1), tos&time(dimsizes(tos&time)-1), sic&time(dimsizes(sic&time)-1)/))
  lf = sftlf
  if (any(method.eq.(/"HadCRUT4", "Cowtan2015", "HadCRUT5"/))) then ; Need to check HadCRUT5 method
    lftmp = new(array_append_record(2, dimsizes(lf), 0), typeof(lf))
    lftmp(0, :, :) = lf
    lftmp(1, :, :) = 0.25
    lf = where(lf.gt.1.e-5, dim_max_n(lftmp, 0), 0.)
    delete(lftmp)
  end if  
  frac = conform(tas({stime:etime}, :, :), lf, (/1, 2/))
  ts = tas({stime:etime}, :, :)
  if (method.eq."HadCRUT4") then
    ts = tas({stime:etime}, :, :)*frac \
          + tos({stime:etime}, :, :)*(1. - frac)
  else
    ts = tas({stime:etime}, :, :)*(frac + (1. - frac)*sic({stime:etime}, :, :)) \
          + tos({stime:etime}, :, :)*(1. - frac)*(1.-sic({stime:etime}, :, :))
  end if
  return(ts)
end

; =================================================================
;
;                          Time functions
;
; =================================================================

;function set_yyyymm(x)
;local x, y, time
;begin
;  y = x
;  if (min(mod(x&time,100)).lt.1.or.max(mod(x&time,100)).gt.12) then
;    time = cd_calendar(x&time, -1)
;    y&time = time
;  end if
;  if (isatt(x&time,"calendar")) then
;    if (y&time@calendar .eq. "proleptic_gregorian") then
;      y&time@calendar = "gregorian"
;    else
;      y&time@calendar = y&time@calendar
;    end if
;  end if
;  y&time@long_name = "time"
;  y&time@standard_name = "time"
;  y&time@axid = "T"
;  return(y)
;end

function monthly2annual(x, smonth_ann, emonth_ann, syear_ann, eyear_ann)
local y, year, month, syear, eyear, ndays, ndays_min, nseasons, nseasons_min, \
      smonth1, emonth1, smonht2, emonth2, smonth3, emonth3, smonth4, emonth4, \
      tmp
begin
;  y = set_yyyymm(x)
  y = x
  year = 0
  year@calendar = y&time@calendar
  if (year@calendar.eq."proleptic_gregorian") then
    year@calendar = "gregorian"
  end if
  if (dimsizes(dimsizes(x)).eq.1) then
    ann = y({syear_ann*100+1:eyear_ann*100+1:12})
  else
    ann = y({syear_ann*100+1:eyear_ann*100+1:12}, :, :)
  end if
  syear = toint(ann&time(0)/100)
  eyear = toint(ann&time(dimsizes(ann&time)-1)/100)
  ann&time=ispan(syear, eyear, 1)
  ann = 0.
  if (dimsizes(dimsizes(x)).eq.1) then
    if (smonth_ann.le.emonth_ann) then
      ndays_min = 28*(emonth_ann-smonth_ann+1)
      do year = syear, eyear
        ndays = 0
        do month = smonth_ann, emonth_ann
          if (.not.ismissing(y({year*100+month}))) then
            ann({year}) = ann({year}) + y({year*100+month})*tofloat(days_in_month(year,month))
            ndays = ndays + days_in_month(year,month)
          end if
        end do
        if (ndays.ge.ndays_min) then  ; For 1D time series, all months are necessary
          ann({year}) = ann({year})/tofloat(ndays)
        else
          ann({year}) = ann@_FillValue
        end if
      end do
    else
      ndays_min = 28*(emonth_ann-smonth_ann+13)
      do year = syear, eyear
        ndays = 0
        do month = smonth_ann, 12
          if (.not.ismissing(y({(year-1)*100+month}))) then
            ann({year}) = ann({year}) + y({(year-1)*100+month})*tofloat(days_in_month(year-1,month))
            ndays = ndays + days_in_month(year-1,month)
          end if
        end do
        do month = 1, emonth_ann
          if (.not.ismissing(y({year*100+month}))) then
            ann({year}) = ann({year}) + y({year*100+month})*tofloat(days_in_month(year,month))
            ndays = ndays + days_in_month(year,month)
          end if
        end do
        if (ndays.ge.ndays_min) then
          ann({year}) = ann({year})/tofloat(ndays)
        else
          ann({year}) = ann@_FillValue
        end if
      end do
    end if
  else
    if (emonth_ann-smonth_ann+1.eq.12) then
      smonth1 = smonth_ann
      emonth1 = smonth1 + 2
      smonth2 = emonth1 + 1
      emonth2 = smonth2 + 2
      smonth3 = emonth2 + 1
      emonth3 = smonth3 + 2
      smonth4 = emonth3 + 1
      emonth4 = smonth4 + 2
    elseif (emonth_ann-smonth_ann+1.eq.0) then
      smonth1 = smonth_ann
      emonth1 = smonth1 + 2
      smonth2 = emonth1 + 1
      emonth2 = smonth2 + 2
      smonth3 = emonth2 + 1
      emonth3 = smonth3 + 2
      smonth4 = emonth3 + 1
      emonth4 = smonth4 + 2
      tmp = (/smonth1, emonth1, smonth2, emonth2, smonth3, emonth3, smonth4, emonth4/)
      tmp = where(tmp.gt.12, tmp-100, tmp)
      smonth1 = tmp(0)
      emonth1 = tmp(1)
      smonth2 = tmp(2)
      emonth2 = tmp(3)
      smonth3 = tmp(4)
      emonth3 = tmp(5)
      smonth4 = tmp(6)
      emonth5 = tmp(7)
      delete(tmp)
    else
      error_msg("f", DIAG_SCRIPT, "", "monthly2annual: A year must be 12-months.")
    end if
    nseasons_min = toint((emonth_ann-smonth_ann+1)/3)*3./4.  ; for 3D fields, 1 data per season (= 3months) and 3 seasons per year is necessary
    nseasons = toint(ann(0, :, :))
    do year = syear, eyear
      nseasons = 0
; season 1
      ann({year}, :, :) = where(dim_num_n(.not.ismissing(y({year*100+smonth1:year*100+emonth1}, :, :)), 0).gt.0, \
                                ann({year}, :, :) + dim_avg_n(y({year*100+smonth1:year*100+emonth1}, :, :), 0), ann({year}, :, :))
      nseasons = where(dim_num_n(.not.ismissing(y({year*100+smonth1:year*100+emonth1}, :, :)), 0).gt.0, nseasons+1, nseasons)
; season 2
      ann({year}, :, :) = where(dim_num_n(.not.ismissing(y({year*100+smonth2:year*100+emonth2}, :, :)), 0).gt.0, \
                                ann({year}, :, :) + dim_avg_n(y({year*100+smonth2:year*100+emonth2}, :, :), 0), ann({year}, :, :))
      nseasons = where(dim_num_n(.not.ismissing(y({year*100+smonth2:year*100+emonth2}, :, :)), 0).gt.0, nseasons+1, nseasons)
; season 3
      ann({year}, :, :) = where(dim_num_n(.not.ismissing(y({year*100+smonth3:year*100+emonth3}, :, :)), 0).gt.0, \
                                ann({year}, :, :) + dim_avg_n(y({year*100+smonth3:year*100+emonth3}, :, :), 0), ann({year}, :, :))
      nseasons = where(dim_num_n(.not.ismissing(y({year*100+smonth3:year*100+emonth3}, :, :)), 0).gt.0, nseasons+1, nseasons)
; season 4
      ann({year}, :, :) = where(dim_num_n(.not.ismissing(y({year*100+smonth4:year*100+emonth4}, :, :)), 0).gt.0, \
                                ann({year}, :, :) + dim_avg_n(y({year*100+smonth4:year*100+emonth4}, :, :), 0), ann({year}, :, :))
      nseasons = where(dim_num_n(.not.ismissing(y({year*100+smonth4:year*100+emonth4}, :, :)), 0).gt.0, nseasons+1, nseasons)
; average across 4 seasons
      nseasons = where(nseasons.ge.nseasons_min, nseasons, nseasons@_FillValue)
      ann({year}, :, :) = ann({year}, :, :)/tofloat(nseasons)
      ann({year}, :, :) = where(nseasons.ge.nseasons_min, ann({year}, :, :), ann@_FillValue)
    end do
    delete(nseasons)
    delete(nseasons_min)
  end if

  return(ann)
  delete(y)
  delete(ann)
;  delete(time)
end

function calc_trend(x, syr, eyr, method, annual, smon_ann, emon_ann)
local tr, tmp, dims, frac_threshold, ts, te, scale
begin
  if (annual) then
    y = monthly2annual(x, smon_ann, emon_ann, syr, eyr)
    ts = syr
    te = eyr
    scale = 1.
  else
    y = x
    ts = syr*100+1
    te = eyr*100+12
    scale = 12.
  end if
    
  if (dimsizes(dimsizes(x)).eq.1) then
    if (any(method.eq.(/"least_square",  "LeastSquare",  "Least_Square", \
                        "least_squares", "LeastSquares", "Least_Squares", \
                        "ls", "LS"/))) then
      tr = regline(tofloat(ispan(1, dimsizes(y({ts:te})), 1)), y({ts:te}))
      tmp = esacr(y({ts:te}), 1)
      edof = num(.not.ismissing(y({ts:te}))) * min((/(1.-tmp(1))/(1.+tmp(1)), 1./)) - 2.
      delete(tmp)
      sig = 1. - betainc(edof/(edof + tr@tval^2), edof/2.0, 0.5)
      delete(edof)
    elseif (any(method.eq.(/"Sen", "sen", "median_slope", "MedianSlope", \
                            "Theil-Sen"/))) then
      tmp = trend_manken(y({ts:te}), True, 0)
      tr = tmp(1)
      sig = tmp(0)
      delete(tmp)
    end if
    copy_VarAtts(y, tr)
    tr@sig = sig
  elseif (dimsizes(dimsizes(x)).eq.3) then
    frac_threshold = 0.8  ; For 3D field, at least 80% of the time series
                          ; is required for trend calculation.
    if (any(method.eq.(/"least_square",  "LeastSquare",  "Least_Square", \
                        "least_squares", "LeastSquares", "Least_Squares", \
                        "ls", "LS"/))) then
      dims = dimsizes(y({ts:te}, :, :))
;      tmp = dtrend_msg_n(tofloat(ispan(1, dims(0), 1)), y({ts:te}, :, :), True, True, 0)
;      tr = y(0, :, :)
      tr = regCoef_n(tofloat(ispan(1, dims(0), 1)), y({ts:te}, :, :), 0, 0)
      tr = where(dim_num_n(.not.ismissing(y({ts:te}, :, :)), 0).gt.dims(0)*frac_threshold, \
                 tr, tr@_FillValue)
;      tmp2 = trend_manken(y({ts:te}, :, :), False, 0)
;      sig = tr
;      sig = where(ismissing(tr), tr@_FillValue, tmp2(0, :, :))
;      delete(tmp2)
      delete(dims)
      tmp = esacr_n(y({ts:te}, :, :), 1, 0)
      edof = dim_num_n(.not.ismissing(y({ts:te}, :, :)), 0) * where(tmp(1, :, :).gt.0., (1.-tmp(1, :, :))/(1.+tmp(1, :, :)), 1.) - 2.
      delete(tmp)
      b = edof
      b = 0.5
      sig = 1. - betainc(edof/(edof + reshape(tr@tval, dimsizes(tr))^2), edof/2.0, b)
      delete(edof)
      delete(b)
    elseif (any(method.eq.(/"Sen", "sen", "median_slope", "MedianSlope", \
                            "Theil-Sen"/))) then
      dims = dimsizes(y({ts:te}, :, :))
      tmp = trend_manken(y({ts:te}, :, :), True, 0)
      tr = where(dim_num_n(.not.ismissing(y({ts:te}, :, :)), 0).gt.dims(0)*frac_threshold, \
                 tmp(1, :, :), tmp@_FillValue)
      sig = where(dim_num_n(.not.ismissing(y({ts:te}, :, :)), 0).gt.dims(0)*frac_threshold, \
                 tmp(0, :, :), tmp@_FillValue)
      delete(dims)
    end if
    copy_VarMeta(y(0, :, :), tr)
    tr@sig = ndtooned(sig)
  end if
  delete(sig)
  
  tr = tr * scale
  if (isatt(x, "units")) then
    tr@units = x@units+" per year"
  end if
;  delete(tmp)

  tr@syear_trend = syr
  tr@eyear_trend = eyr
  tr@annual_mean_before_trend = annual
  if (annual) then
    tr@smonth_annual_mean = smon_ann
    tr@emonth_annual_mean = emon_ann
  end if
  return(tr)
end

function calc_anom(x, syear, eyear)
local x, y, syear, eyear, clim, anom, time
begin
;  time = cd_calendar(x&time, -1)
  y = x
;  y&time = time
;  y = set_yyyymm(x)
;  printVarSummary(x)
;  print(x&time)
  if (dimsizes(dimsizes(x)).eq.3) then
    clim = clmMonTLL(y({syear*100+1:eyear*100+12}, :, :))
    y = calcMonAnomTLL(y, clim)
  elseif (dimsizes(dimsizes(x)).eq.1) then
    tmp = new((/dimsizes(x), 1, 1/), typeof(y))
    tmp(:, 0, 0) = y
    tmp!1 = "lat"
    tmp!2 = "lon"
    tmp&lat = (/ 0. /)
    tmp&lon = (/ 0. /)
    clim = clmMonTLL(tmp({syear*100+1:eyear*100+12}, :, :))
    ytmp = calcMonAnomTLL(tmp, clim)
    y = ytmp(:, 0, 0)
    delete(tmp)
    delete(ytmp)
  else
    error_msg("f", DIAG_SCRIPT, "", "calc_anom: Array shape is unsupported.")
  end if
  return(y)
  delete(y)
  delete(clim)
end

; =================================================================
;
;                      mean and percentile functions
;
; =================================================================

function weighted_mean(x, w)
local x, w, y
begin
  y = dim_sum_n(x*conform(x, w, 0), 0)
  y = y / dim_sum_n(where(ismissing(x), w@_FillValue, conform(x, w, 0)), 0)
  return(y)
end
  
; ==========================================================================
; percentile_val(x, x0): Obtain (x0 x 100)-th percentile value in an array x
;  x:  numeric, a 1-D array
;  x0: float or double, percentile in [0, 1]
; --------------------------------------------------------------------------
;function percentile_val(x, x0)
;local X, imax, i, val, x, x0
;begin
;  if (any(ismissing(x))) then
;    X = x(ind(.not.ismissing(x)))
;  else
;    X = x
;  end if
;  qsort(X)
;  imax = tofloat(dimsizes(X))
;  if (x0.lt.0.5/imax) then
;    val = X(0)
;  elseif (x0.gt.(imax - 0.5)/imax) then
;    val = X(toint(imax)-1)
;  else
;    i = x0*imax + 0.5
;    val = X(toint(floor(i))) + (X(toint(floor(i))+1) - X(toint(floor(i))))*(i - floor(i))
;  end if
;  return(val)
;end

; ================================================================================
; weighted_percentile(x, w, t): Obtain (t x 100)-th percentile value in an array x
;  x: numeric, a 1-D array
;  w: numeric, the same dimension size as x, weight of x
;  t: float or double, percentile in [0, 1]
; --------------------------------------------------------------------------------

function weighted_percentile(x, w, t)
local x, w, xx, ww, wsum, j, k, dims, idx
begin
  xx = x
  dims = dimsizes(xx)
  if (dimsizes(dims).eq.1) then
    idx = dim_pqsort(xx, 2)
    wsum = sum(where(ismissing(x), w@_FillValue, w))
    ww = 0.
    do j = 0, dims(0)-1
      if (.not.ismissing(xx(j))) then
        ww = ww + w(idx(j))
        if (ww/wsum.eq.t) then
          y = xx(j)
          break
        elseif (ww/wsum.gt.t) then
          y = (xx(j) + xx(j-1))/2.
          break
        end if
      end if
    end do
  else
    y = new(dims(1), typeof(x))
    idx = dim_pqsort_n(xx, 2, 0)
    do k = 0, dims(1)-1
      wsum = sum(where(ismissing(x(:, k)), w@_FillValue, w))
      ww = 0.
      do j = 0, dims(0)-1
        if (.not.ismissing(xx(j, k))) then
          ww = ww + w(idx(j, k))
          if (ww/wsum.eq.t) then
            y(k) = xx(j, k)
            break
          elseif (ww/wsum.gt.t) then
            y(k) = (xx(j, k) + xx(j-1, k))/2.
            break
          end if
        end if
      end do
    end do
  end if
  return(y)
end