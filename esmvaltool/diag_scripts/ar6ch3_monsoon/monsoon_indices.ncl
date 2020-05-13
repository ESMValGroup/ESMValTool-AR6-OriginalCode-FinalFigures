; monsoon_indices
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs:
;
; History
;    20200513 kosaka_yu: cleand up.
;    20191030 kosa_yu: written.
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
  if (min(mod(x&time, 100)).lt.1.or.max(mod(x&time, 100)).gt.12) then
    time = cd_calendar(x&time, -1)
    delete(y&time)
    y!0 = "time"
    y&time = time
  end if
  if (isatt(x&time, "calendar")) then
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

function get_clim(x, syear, eyear, smonth, emonth)
local x, y, clim, time, month
begin
  if (.not.(smonth.eq.emonth+1 .or. all((/smonth, emonth/).eq.(/1, 12/)))) then
    error_msg("f", DIAG_SCRIPT, "", \
              "monsoon year doesn't cover 12 calendar months.")
  end if

  time = cd_calendar(x&time, -1)
  y = x
  y&time = time

  clim = y(0:11, :, :)
  clim!0 = "month"
  clim&month = ispan(1, 12, 1)
  do month = 1, 12
    if (month.lt.smonth) then
      clim({month}, :, :) \
        = dim_avg_n(y({(syear+1)*100+month:(eyear+1)*100+month:12}, :, :), 0)
    else
      clim({month}, :, :) \
        = dim_avg_n(y({syear*100+month:eyear*100+month:12}, :, :), 0)
    end if
  end do

;  if (accumulate) then
;    weight = new(12, "float")
;    do mon = 1, 12
;      weight(mon-1) = tofloat(days_in_month(2001, mon))
;    end do
;    clim = clim * conform(clim, weight, 0)
;  end if
  return(clim)
end

function summer(x, opt)
local x, y, z, smonthNH, emonthNH, smonthSH, emonthSH, weight, \
      len, yr, mon, accumulate
begin
  smonthNH = opt@smonth_NHsummer
  emonthNH = opt@emonth_NHsummer
  smonthSH = opt@smonth_SHsummer
  emonthSH = opt@emonth_SHsummer
  if (isatt(opt, "accumulate")) then
    accumulate = opt@accumulate
  else
    accumulate = False
  end if

  weight = new(12, "float")
  do mon = 1, 12
    weight(mon-1) = tofloat(days_in_month(2001, mon))
  end do

  if (x!0.eq."month".and.all(toint(x&month).eq.ispan(1, 12, 1))) then

    y := x(0, :, :)
    y = 0.
    len = 0.
    if (smonthNH.lt.emonthNH) then
      do mon = smonthNH, emonthNH
        y = y + x(mon-1, :, :)*weight(mon-1)
        len = len + weight(mon-1)
      end do
      if (.not.accumulate) then
        y = y / len
      end if
    else
      do mon = smonthNH, 12
        y = y + x(mon-1, :, :)*weight(mon-1)
        len = len + weight(mon-1)
      end do
      do mon = 1, emonthNH
        y = y + x(mon-1, :, :)*weight(mon-1)
        len = len + weight(mon-1)
      end do
      if (.not.accumulate) then
        y = y / len
      end if
    end if

    z := x(0, :, :)
    z = 0.
    len = 0.
    if (smonthSH.lt.emonthSH) then
      do mon = smonthSH, emonthSH
        z = z + x(mon-1, :, :)*weight(mon-1)
        len = len + weight(mon-1)
      end do
      if (.not.accumulate) then
        z = z / len
      end if
    else
      do mon = smonthSH, 12
        z = z + x(mon-1, :, :)*weight(mon-1)
        len = len + weight(mon-1)
      end do
      do mon = 1, emonthSH
        z = z + x(mon-1, :, :)*weight(mon-1)
        len = len + weight(mon-1)
      end do
      if (.not.accumulate) then
        z = z / len
      end if
    end if
  else
    X = set_yyyymm(x)
    syear = toint(X&time(0)/100)
    eyear = toint(X&time(dimsizes(X&time)-1)/100)
    if (smonthNH.gt.emonthNH.or.smonthSH.gt.emonthSH) then
      eyear = eyear - 1
    end if

    y := X({syear*100+smonthNH:eyear*100+smonthNH:12}, :, :)
    y&time = ispan(syear, eyear, 1)
    y = 0.
    if (smonthNH.lt.emonthNH) then
      do yr = syear, eyear
        len = 0.
        do mon = smonthNH, emonthNH
          y({yr}, :, :) = y({yr}, :, :) + X({yr*100+mon}, :, :) * weight(mon-1)
          len = len + weight(mon-1)
        end do
        if (.not.accumulate) then
          y({yr}, :, :) = y({yr}, :, :) / len
        end if
      end do
    else
      do yr = syear, eyear
        len = 0.
        do mon = smonthNH, 12
          y({yr}, :, :) = y({yr}, :, :) + X({yr*100+mon}, :, :) * weight(mon-1)
          len = len + weight(mon-1)
        end do
        do mon = 1, emonthNH
          y({yr}, :, :) \
            = y({yr}, :, :) + X({(yr+1)*100+mon}, :, :) * weight(mon-1)
          len = len + weight(mon-1)
        end do
        if (.not.accumulate) then
          y({yr}, :, :) = y({yr}, :, :) / len
        end if
      end do
    end if

    z := X({syear*100+smonthSH:eyear*100+smonthSH:12}, :, :)
    z&time = ispan(syear, eyear, 1)
    z = 0.
    if (smonthSH.lt.emonthSH) then
      do yr = syear, eyear
        len = 0.
        do mon = smonthSH, emonthSH
          z({yr}, :, :) = z({yr}, :, :) + X({yr*100+mon}, :, :) * weight(mon-1)
          len = len + weight(mon-1)
        end do
        if (.not.accumulate) then
          z({yr}, :, :) = z({yr}, :, :) / len
        end if
      end do
    else
      do yr = syear, eyear
        len = 0.
        do mon = smonthSH, 12
          z({yr}, :, :) = z({yr}, :, :) + X({yr*100+mon}, :, :) * weight(mon-1)
          len = len + weight(mon-1)
        end do
        do mon = 1, emonthSH
          z({yr}, :, :) \
            = z({yr}, :, :) + X({(yr+1)*100+mon}, :, :) * weight(mon-1)
          len = len + weight(mon-1)
        end do
        if (.not.accumulate) then
          z({yr}, :, :) = z({yr}, :, :) / len
        end if
      end do
    end if
  end if

  lat := y&lat
  if (dimsizes(dimsizes(y)).eq.2) then
    y = where(conform(y, lat, 0).lt.0., z,  y)
    y = where(conform(y, lat, 0).eq.0., 0., y)
  else
    y = where(conform(y, lat, 1).lt.0., z,  y)
    y = where(conform(y, lat, 1).eq.0., 0., y)
  end if
  return(y)
end

function summer_minus_winter(x, opt)
local x, y, z, smonthNH, emonthNH, smonthSH, emonthSH, weight, \
      len, yr, mon, accumulate
begin
  smonthNH = opt@smonth_NHsummer
  emonthNH = opt@emonth_NHsummer
  smonthSH = opt@smonth_SHsummer
  emonthSH = opt@emonth_SHsummer
  if (isatt(opt, "accumulate")) then
    accumulate = opt@accumulate
  else
    accumulate = False
  end if

  weight = new(12, "float")
  do mon = 1, 12
    weight(mon-1) = tofloat(days_in_month(2001, mon))
  end do

  if (x!0.eq."month".and.all(toint(x&month).eq.ispan(1, 12, 1))) then

    y := x(0, :, :)
    y = 0.
    len = 0.
    if (smonthNH.lt.emonthNH) then
      do mon = smonthNH, emonthNH
        y = y + x(mon-1, :, :)*weight(mon-1)
        len = len + weight(mon-1)
      end do
      if (.not.accumulate) then
        y = y / len
      end if
    else
      do mon = smonthNH, 12
        y = y + x(mon-1, :, :)*weight(mon-1)
        len = len + weight(mon-1)
      end do
      do mon = 1, emonthNH
        y = y + x(mon-1, :, :)*weight(mon-1)
        len = len + weight(mon-1)
      end do
      if (.not.accumulate) then
        y = y / len
      end if
    end if

    z := x(0, :, :)
    z = 0.
    len = 0.
    if (smonthSH.lt.emonthSH) then
      do mon = smonthSH, emonthSH
        z = z + x(mon-1, :, :)*weight(mon-1)
        len = len + weight(mon-1)
      end do
      if (.not.accumulate) then
        z = z / len
      end if
    else
      do mon = smonthSH, 12
        z = z + x(mon-1, :, :)*weight(mon-1)
        len = len + weight(mon-1)
      end do
      do mon = 1, emonthSH
        z = z + x(mon-1, :, :)*weight(mon-1)
        len = len + weight(mon-1)
      end do
      if (.not.accumulate) then
        z = z / len
      end if
    end if
  else
    X = set_yyyymm(x)
    syear = toint(X&time(0)/100)
    eyear = toint(X&time(dimsizes(X&time)-1)/100)
    if (smonthNH.gt.emonthNH.or.smonthSH.gt.emonthSH) then
      eyear = eyear - 1
    end if

    y := X({syear*100+smonthNH:eyear*100+smonthNH:12}, :, :)
    y&time = ispan(syear, eyear, 1)
    y = 0.
    if (smonthNH.lt.emonthNH) then
      do yr = syear, eyear
        len = 0.
        do mon = smonthNH, emonthNH
          y({yr}, :, :) = y({yr}, :, :) + X({yr*100+mon}, :, :) * weight(mon-1)
          len = len + weight(mon-1)
        end do
        if (.not.accumulate) then
          y({yr}, :, :) = y({yr}, :, :) / len
        end if
      end do
    else
      do yr = syear, eyear
        len = 0.
        do mon = smonthNH, 12
          y({yr}, :, :) = y({yr}, :, :) + X({yr*100+mon}, :, :) * weight(mon-1)
          len = len + weight(mon-1)
        end do
        do mon = 1, emonthNH
          y({yr}, :, :) \
            = y({yr}, :, :) + X({(yr+1)*100+mon}, :, :) * weight(mon-1)
          len = len + weight(mon-1)
        end do
        if (.not.accumulate) then
          y({yr}, :, :) = y({yr}, :, :) / len
        end if
      end do
    end if

    z := X({syear*100+smonthSH:eyear*100+smonthSH:12}, :, :)
    z&time = ispan(syear, eyear, 1)
    z = 0.
    if (smonthSH.lt.emonthSH) then
      do yr = syear, eyear
        len = 0.
        do mon = smonthSH, emonthSH
          z({yr}, :, :) = z({yr}, :, :) + X({yr*100+mon}, :, :) * weight(mon-1)
          len = len + weight(mon-1)
        end do
        if (.not.accumulate) then
          z({yr}, :, :) = z({yr}, :, :) / len
        end if
      end do
    else
      do yr = syear, eyear
        len = 0.
        do mon = smonthSH, 12
          z({yr}, :, :) = z({yr}, :, :) + X({yr*100+mon}, :, :) * weight(mon-1)
          len = len + weight(mon-1)
        end do
        do mon = 1, emonthSH
          z({yr}, :, :) \
            = z({yr}, :, :) + X({(yr+1)*100+mon}, :, :) * weight(mon-1)
          len = len + weight(mon-1)
        end do
        if (.not.accumulate) then
          z({yr}, :, :) = z({yr}, :, :) / len
        end if
      end do
    end if
  end if

  y = y - z
  lat := y&lat
  if (dimsizes(dimsizes(y)).eq.2) then
    y = where(conform(y, lat, 0).lt.0., -y, y)
    y = where(conform(y, lat, 0).eq.0., 0., y)
  else
    y = where(conform(y, lat, 1).lt.0., -y, y)
    y = where(conform(y, lat, 1).eq.0., 0., y)
  end if
  return(y)
end

function monsoon_domain(MPI, summer_minus_winter, opt)
local domain, thres_smw, thres_mpi
begin
  domain = MPI
  thres_smw = 300.
  thres_mpi = 0.5
  if (opt) then
    if (isatt(opt, "threshold_summer_minus_winter")) then
      thres_smw = opt@threshold_summer_minus_winter
    end if
    if (isatt(opt, "threshold_mpi")) then
      thres_mpi = opt@threshold_mpi
    end if
  end if
  domain = where(summer_minus_winter.gt.thres_smw, 1., 0.)
  domain = where(MPI.gt.thres_mpi, domain, 0.)
  domain@long_name = ""
  domain@units = ""
  return(domain)
end
