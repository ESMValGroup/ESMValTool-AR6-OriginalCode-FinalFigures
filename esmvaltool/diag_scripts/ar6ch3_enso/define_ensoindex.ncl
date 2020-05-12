; define_ensoindex
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

  if (isatt(diag_script_info, "syear")) then
    syear_save = diag_script_info@syear
  end if
  if (isatt(diag_script_info, "eyear")) then
    eyear_save = diag_script_info@eyear
  end if

  domain = "Nino3"
  if (isatt(diag_script_info, "index")) then
    domain = diag_script_info@index
  end if

  detrend = True
  if (isatt(diag_script_info, "detrend")) then
    detrend = diag_script_info@detrend
  end if

  ; ========================================================================

  ; Get var list
  var = "ts"
  varList = metadata_att_as_array(input_file_info, "short_name")
  varList := get_unique_values(varList)

  if (all(varList.ne.var)) then
    error_msg("f", DIAG_SCRIPT, "", "variable "+var+" is missing.")
  end if
  iVarTs = ind(varList.eq.var)
;  if (isatt(variable_info[iVarTs], "reference_dataset")) then
;    refname = variable_info[iVarTs]@reference_dataset
;  end if

  ; ========================================================================

  ; prepare time filtering
  wgt = (/ 1., 2., 3., 2., 1./)
  wgt = wgt / sum(wgt)

  ; prepare index domain
  if (domain.eq."Nino3") then
    W = 210.
    E = 270.
    S = -5.
    N = 5.
  elseif (domain.eq."Nino3.4") then
    W = 190.
    E = 240.
    S = -5.
    N = 5.
  elseif (domain.eq."Nino4") then
    W = 160.
    E = 210.
    S = -5.
    N = 5.
  elseif (domain.eq."Ni«Ðo3.3_LatExt") then
    W = 220.
    E = 270.
    S = -15.
    N = 15.
  elseif (domain.eq."Nino1+2") then
    W = 270.
    E = 280.
    S = -10.
    N = 0.
  elseif (domain.eq."TP") then
    W = 120.
    E = 280.
    S = -30.
    N = 30.
  elseif (domain.eq."EP") then
    W = 120.
    E = 280.
    S = -5.
    N = 5.
  elseif (domain.eq."EP_LatExt") then
    W = 150.
    E = 270.
    S = -15.
    N = 15.
  elseif (domain.eq."EEP") then
    W = 205.
    E = 280.
    S = -5.
    N = 5.
  elseif (domain.eq."WEP") then
    W = 120.
    E = 205.
    S = -5.
    N = 5.
  else
    error_msg("f", DIAG_SCRIPT, "", "Index "+domain+" is undefined.")
  end if
  d2r = get_d2r("double")

  ; Get file list
  infoTs = select_metadata_by_name(input_file_info, var)

  do ii = 0, ListCount(infoTs) - 1
    ; Get ensemble size for later use
    esize = 0
    do i = 0, ListCount(infoTs)-1
      if (isatt(infoTs[i], "exp").and.isatt(infoTs[ii], "exp")) then
        if (all((/infoTs[i]@project, infoTs[i]@dataset, infoTs[i]@exp/).eq. \
                (/infoTs[ii]@project, infoTs[ii]@dataset, infoTs[ii]@exp/))) \
          then
          esize = esize + 1
        end if
      elseif (.not.isatt(infoTs[i], "exp").and..not.isatt(infoTs[ii], "exp")) \
        then
        if (all((/infoTs[i]@project, infoTs[i]@dataset/).eq. \
                (/infoTs[ii]@project, infoTs[ii]@dataset/))) then
          esize = esize + 1
        end if
      end if
    end do

    ; Read data
    Ts = read_data(infoTs[ii])
    ts = tofloat(Ts)
    copy_VarMeta(Ts, ts)
    delete(Ts)
    if (ts@units.eq."K") then  ; K -> degC
      ts = ts - 273.15
      ts@units = "degC"
    end if

    f = addfile(infoTs[ii]@filename, "r")
    if (isfilevar(f, "lat_bnds")) then
      lat_bnds = f->lat_bnds
    end if
    ysize = dimsizes(ts&lat)
    if (.not.isvar("lat_bnds") .or. \
        any(abs(lat_bnds(1:ysize-1, 0)-lat_bnds(0:ysize-2, 1)).gt.0.01)) then
      print(infoTs[ii]@dataset+" : Creating lat_bnds")
      lat = ts&lat
      if (isvar("lat_bnds")) then
        delete(lat_bnds)
      end if
      lat_bnds = new((/ysize, 2/), typeof(ts&lat))
      lat_bnds(1:ysize-1, 0) = (lat(0:ysize-2) + lat(1:ysize-1))/2.
      lat_bnds(0:ysize-2, 1) = (lat(0:ysize-2) + lat(1:ysize-1))/2.
      lat_bnds(0, 0) \
        = max((/lat_bnds(0, 1) - (lat_bnds(1, 1) - lat_bnds(1, 0)), -90./))
      lat_bnds(ysize-1, 1) \
        = min((/lat_bnds(ysize-1, 0) + (lat_bnds(ysize-2, 1) \
              - lat_bnds(ysize-2, 0)), 90./))
      delete(lat)
    end if

    if (isfilevar(f, "lon_bnds")) then
      lon_bnds = f->lon_bnds
    end if
    xsize = dimsizes(ts&lon)
    if (.not.isvar("lon_bnds") .or. \
        any(abs(lon_bnds(1:xsize-1, 0)-lon_bnds(0:xsize-2, 1)).gt.0.01)) then
      print(infoTs[ii]@dataset+" : Creating lon_bnds")
      lon = ts&lon
      if (isvar("lon_bnds")) then
        delete(lon_bnds)
      end if
      lon_bnds = new((/xsize, 2/), typeof(ts&lon))
      lon_bnds(1:xsize-1, 0) = (lon(0:xsize-2) + lon(1:xsize-1))/2.
      lon_bnds(0:xsize-2, 1) = (lon(0:xsize-2) + lon(1:xsize-1))/2.
      lon_bnds(0, 0) = lon_bnds(0, 1) - (lon_bnds(1, 1) - lon_bnds(1, 0))
      lon_bnds(xsize-1, 1) \
        = lon_bnds(xsize-1, 0) + (lon_bnds(xsize-2, 1) - lon_bnds(xsize-2, 0))
      delete(lon)
    end if
    delete(f)

    time = cd_calendar(ts&time, -1)
    ts&time = time
    syear_data = time(0)/100
    smonth_data = time(0)-syear_data*100
    eyear_data = time(dimsizes(time)-1)/100
    emonth_data = time(dimsizes(time)-1)-eyear_data*100
    if (.not.isatt(ts, "_FillValue")) then
      ts@_FillValue = default_fillvalue(typeof(ts))
    end if
    delete(time)

    if (isvar("syear_save")) then
      syear = syear_save
    else
      syear = syear_data + 2
    end if
    if (isvar("eyear_save")) then
      eyear = eyear_save
    else
      eyear = eyear_data - 3
    end if

    ; Anomalies
    clim = clmMonTLL(ts({syear*100+1:eyear*100+12}, :, :))
    do t = 0, dimsizes(ts&time)-1
      yr  = toint(ts&time(t) / 100)
      mon = toint(ts&time(t) - yr*100)
      ts(t, :, :) = ts(t, :, :) - clim(mon-1, :, :)
    end do
    delete(clim)

    ; Detrend
    if (detrend) then
      do mon = 1, 12
        trend = dtrend_n(ts({syear*100+mon:eyear*100+mon:12}, :, :), True, 0)
        syr = syear
        eyr = eyear
        if (smonth_data.gt.mon) then
          syr = syr + 1
        end if
        if (emonth_data.lt.mon) then
          eyr = eyr - 1
        end if
        ts({syr*100+mon:eyr*100+mon:12}, :, :) \
          = ts({syr*100+mon:eyr*100+mon:12}, :, :) \
          - conform(ts({syr*100+mon:eyr*100+mon:12}, :, :), \
                    reshape(trend@slope, dimsizes(ts(0, :, :))), (/1, 2/)) \
          * conform(ts({syr*100+mon:eyr*100+mon:12}, :, :), \
                    tofloat(ispan(syr, eyr, 1)) - (syear+eyear)/2., 0)
        delete(trend)
      end do
    end if

    ; Time smoothing
    ts = wgt_runave_n(ts, wgt, 0, 0)

    ; Spatial averaging
    j0 = ind(lat_bnds(:, 0).le.S .and. S.lt.lat_bnds(:, 1))
    j1 = ind(lat_bnds(:, 0).lt.N .and. N.le.lat_bnds(:, 1))
    i0 = ind(lon_bnds(:, 0).le.W .and. W.lt.lon_bnds(:, 1))
    if (dimsizes(i0).gt.1) then
      print("i0="+i0)
      print("W="+W)
      print("lon_bnds(0)="+lon_bnds(i0(0), :))
      print("lon_bnds(1)="+lon_bnds(i0(1), :))
      i0 := i0(0)
    end if
    i1 = ind(lon_bnds(:, 0).lt.E .and. E.le.lon_bnds(:, 1))
    if (dimsizes(i1).gt.1) then
      i1 := i1(1)
    end if
    dy = lat_bnds(:, 0)
    dy = 0.
    j = j0
    dy(j) = (lat_bnds(j, 1) - S) * (cos(lat_bnds(j, 1)*d2r) + cos(S*d2r)) / 2.
    do j = j0+1, j1-1
      dy(j) = (lat_bnds(j, 1) \
               - lat_bnds(j, 0)) * (cos(lat_bnds(j, 1)*d2r) \
                                    + cos(lat_bnds(j, 0)*d2r)) / 2.
    end do
    j = j1
    dy(j) = (N - lat_bnds(j, 0)) * (cos(N*d2r) + cos(lat_bnds(j, 0)*d2r)) / 2.
    dx = lon_bnds(:, 0)
    dx = 0.
    i = i0
    dx(i) = lon_bnds(i, 1) - W
    do i = i0+1, i1-1
      dx(i) = lon_bnds(i, 1) - lon_bnds(i, 0)
    end do
    i = i1
    dx(i) = E - lon_bnds(i, 0)
    delete([/i0, i1, j0, j1/])

    Nino = wgt_areaave_Wrap(ts, dy, dx, 0)
    delete(ts)
    delete([/dy, dx/])
    delete([/lat_bnds, lon_bnds/])

    ; Output
    work_dir = config_user_info@work_dir + "/"
    system("mkdir -p "+work_dir)

    Nino@var = "ENSOindex"
    Nino@diag_script = DIAG_SCRIPT
    Nino@project = infoTs[ii]@project
    Nino@dataset = infoTs[ii]@dataset
    Nino@esize = esize
;    Nino@reference_dataset = infoTs[ii]@reference_dataset
    Nino@long_name = "ENSO index time series"

    if (any(infoTs[ii]@project.eq.\
        (/"OBS", "OBS6", "obs", "obs4mips", "ana4mips"/))) then
      nc_filename = work_dir + "ENSOindex_" + infoTs[ii]@dataset + ".nc"
    else
      nc_filename = work_dir + "ENSOindex_" + infoTs[ii]@dataset \
                     + "_"+infoTs[ii]@ensemble + ".nc"
      Nino@ensemble = infoTs[ii]@ensemble
    end if
    nc_outfile = ncdf_write(Nino, nc_filename)

    delete(Nino)
  end do

  leave_msg(DIAG_SCRIPT, "")

end
