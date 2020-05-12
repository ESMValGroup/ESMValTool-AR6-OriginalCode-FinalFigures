; find_ensoevents
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs:
;
; History
;    20200511 kosaka_yu: cleaned to meet the code style.
;    20191024 kosaka_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/statistics.ncl"
load "$diag_scripts/shared/plot/style.ncl"
load "$diag_scripts/shared/plot/contour_maps.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")

  ; Configuration

  obs_projects = (/"OBS", "OBS6", "obs", "obs4mips", "ana4mips"/)

  if (isatt(diag_script_info, "syear")) then
    syear = diag_script_info@syear
  end if
  if (isatt(diag_script_info, "eyear")) then
    eyear = diag_script_info@eyear
  end if

  target_month = 12  ; December
  if (isatt(diag_script_info, "detect_month")) then
    target_month = diag_script_info@detect_month
  end if

  threshold = 0.75
  if (isatt(diag_script_info, "detect_threshold")) then
    threshold = diag_script_info@detect_threshold
  end if

  threshold_unit = "deg"
  if (isatt(diag_script_info, "unit_threshold")) then
    threshold_unit = diag_script_info@unit_threshold
  end if
  if (any(threshold_unit.eq.(/"degree", "degrees", "degC", "degK", \
                              "Degree", "Degrees", "DegC", "DegK", \
                              "Deg"/))) then
    threshold_unit = "deg"
  elseif (any(threshold_unit.eq.(/"sigma", "std_dev", \
                                  "stddev", "s.d."/))) then
    threshold_unit = "stdv"
  end if

  ; segment length
  annual_lags = (/-3, 4/)

  ; ========================================================================

  ; Get file list
  input_dir = diag_script_info@input_files
  paths = systemfunc("ls "+input_dir+"/ENSOindex_*.nc")

  ; ========================================================================

  work_dir = config_user_info@work_dir + "/"
  system("mkdir -p "+work_dir)

  do nn = 0, dimsizes(paths)-1

    f = addfile(paths(nn), "r")
    Nino = f->ENSOindex
    syear_data = toint(Nino&time(0))/100
    eyear_data = toint(Nino&time(dimsizes(Nino)-1))/100
    smonth_data = toint(Nino&time(0)) - syear_data*100
    emonth_data = toint(Nino&time(dimsizes(Nino)-1)) - eyear_data*100

    ; Pick up ENSO event
    idx   = Nino({syear*100+target_month:eyear*100+target_month:12})
    stdv_idx = stddev(idx)
    if (threshold_unit.eq."stdv") then
      idx = idx / stdv_idx
    end if
    idx_el = ind(idx.gt.abs(threshold))
    idx_la = ind(idx.lt.-abs(threshold))
    if (.not.all(ismissing(idx_el))) then
      years_el = toint(idx&time(idx_el))/100
    end if
    if (.not.all(ismissing(idx_la))) then
      years_la = toint(idx&time(idx_la))/100
    end if

    delete(idx)
    delete([/idx_el, idx_la/])

    ; Output

    do type = 0, 1
      if (type.eq.0) then
        if (.not.isvar("years_el")) then
          continue
        end if
        nmax = dimsizes(years_el)-1
        name = "El Nino"
        head = "ElNino"
      else
        if (.not.isvar("years_la")) then
          continue
        end if
        nmax = dimsizes(years_la)-1
        name = "La Nina"
        head = "LaNina"
      end if
      do n = 0, nmax
        if (name.eq."El Nino") then
          yr = years_el(n)
        else
          yr = years_la(n)
        end if
        ENSOidx = new((/12*(annual_lags(1)-annual_lags(0)+1)+1/), \
                      typeof(Nino), Nino@_FillValue)
        lag = tofloat(ispan(annual_lags(0)*12-1, (annual_lags(1)+1)*12-1, 1))
        lag@units = "months since 0001-01-01"
        ENSOidx!0 = "time"
        ENSOidx&time = lag
        ENSOidx@time_ref = target_month - lag(0) - 1

        i = 0
        annlag = annual_lags(0)
        mon = 12
        if (syear_data*100+smonth_data.le.(yr+annlag)*100+mon .and. \
            (yr+annlag)*100+mon.le.eyear_data*100+emonth_data) then
          ENSOidx(i) = (/Nino({(yr+annlag)*100+mon})/)
        end if
        i = i + 1
        do annlag = annual_lags(0), annual_lags(1)
          do mon = 1, 12
            if (syear_data*100+smonth_data.le.(yr+annlag)*100+mon .and. \
                (yr+annlag)*100+mon.le.eyear_data*100+emonth_data) then
              ENSOidx(i) = (/Nino({(yr+annlag)*100+mon})/)
            end if
            i = i + 1
          end do
        end do
        delete(lag)

        ENSOidx@var = "ENSOindex"
        ENSOidx@diag_script = DIAG_SCRIPT
        ENSOidx@project = Nino@project
        ENSOidx@dataset = Nino@dataset
        ENSOidx@ensemble = Nino@ensemble
        ENSOidx@esize = Nino@esize
        if (isatt(Nino, "reference_dataset")) then
          ENSOidx@reference_dataset = Nino@reference_dataset
        end if
        ENSOidx@year = yr
        ENSOidx@reference_month = target_month
        ENSOidx@long_name = "ENSO index time series for selected " \
                            + name + " event"
        ENSOidx@stdv_ref = stdv_idx

        if (any(Nino@project.eq.obs_projects)) then
          nc_filename = work_dir + head + "_" + Nino@dataset + "_" \
                        + sprinti("%0.3i", n) + ".nc"
        else
          nc_filename = work_dir + head + "_" + Nino@dataset + "_" \
                         + Nino@ensemble + "_" + sprinti("%0.3i", n) + ".nc"
          ENSOidx@ensemble = Nino@ensemble
        end if
        nc_outfile = ncdf_write(ENSOidx, nc_filename)

        delete(ENSOidx)
      end do
    end do

    delete(Nino)
    delete(f)
    if (isvar("years_el")) then
      delete(years_el)
    end if
    if (isvar("years_la")) then
      delete(years_la)
    end if
  end do

  leave_msg(DIAG_SCRIPT, "")

end
