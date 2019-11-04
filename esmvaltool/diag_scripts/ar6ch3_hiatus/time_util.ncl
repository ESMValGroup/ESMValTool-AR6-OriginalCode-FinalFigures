; time_util.ncl
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs: 
;
; History
;    20190310 kosaka_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

function set_yyyymm(x)
local x, y, time
begin
  y = x
  if (min(mod(x&time,100)).lt.1.or.max(mod(x&time,100)).gt.12) then
    time = cd_calendar(x&time, -1)
    y&time = time
  end if
  if (isatt(x&time,"calendar")) then
    if (y&time@calendar .eq. "proleptic_gregorian") then
      y&time@calendar = "gregorian"
    else
      y&time@calendar = y&time@calendar
    end if
  end if
  y&time@long_name = "time"
  y&time@standard_name = "time"
  y&time@axid = "T"
  return(y)
end

function monthly2annual(x, smonth_ann, emonth_ann, syear_ann, eyear_ann)
local y, year, month, syear, eyear, ndays, ndays_min, nseasons, nseasons_min, \
      smonth1, emonth1, smonht2, emonth2, smonth3, emonth3, smonth4, emonth4, \
      tmp
begin
  y = set_yyyymm(x)
  year = 0
  year@calendar = y&time@calendar
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
      tmp = dtrend_msg(tofloat(ispan(1, dimsizes(y({ts:te})), 1)), y({ts:te}), True, True)
      tr = tmp@slope
    elseif (any(method.eq.(/"Sen", "sen", "median_slope", "MedianSlope", \
                            "Theil-Sen"/))) then
      tmp = trend_manken(y({ts:te}), True, 0)
      tr = tmp(1)
    end if
    copy_VarAtts(y, tr)
  elseif (dimsizes(dimsizes(x)).eq.3) then
    frac_threshold = 0.8  ; For 3D field, at least 80% of the time series
                          ; is required for trend calculation.
    if (any(method.eq.(/"least_square",  "LeastSquare",  "Least_Square", \
                        "least_squares", "LeastSquares", "Least_Squares", \
                        "ls", "LS"/))) then
      dims = dimsizes(y({ts:te}, :, :))
      tmp = dtrend_msg_n(tofloat(ispan(1, dims(0), 1)), y({ts:te}, :, :), True, True, 0)
      tr = where(dim_num_n(.not.ismissing(y({ts:te}, :, :)), 0).gt.dims(0)*frac_threshold, \
                 reshape(tmp@slope, dims(1:2)), tmp@_FillValue)
      delete(dims)
    elseif (any(method.eq.(/"Sen", "sen", "median_slope", "MedianSlope", \
                            "Theil-Sen"/))) then
      dims = dimsizes(y({ts:te}, :, :))
      tmp = trend_manken(y({ts:te}, :, :), True, 0)
      tr = where(dim_num_n(.not.ismissing(y({ts:te}, :, :)), 0).gt.dims(0)*frac_threshold, \
                 tmp(1, :, :), tmp@_FillValue)
      delete(dims)
    end if
    copy_VarMeta(y(0, :, :), tr)
  end if

  tr = tr * scale
  if (isatt(x, "units")) then
    tr@units = x@units+" per year"
  end if
  delete(tmp)

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
