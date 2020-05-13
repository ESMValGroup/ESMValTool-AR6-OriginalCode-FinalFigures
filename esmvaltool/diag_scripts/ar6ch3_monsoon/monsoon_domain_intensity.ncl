; monsoon_domain_intensity
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs:
;
; History
;    20200513 kosaka_yu: cleaned up.
;    20191130 kosaka_yu: bug fix for monsoon circulation index.
;    20191030 kosaka_yu: written.
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

load "$diag_scripts/ar6ch3_monsoon/monsoon_indices.ncl"

function correct_FillValue_uv(x)
local y
begin
  y = x
  y = where(abs(x).gt.1.e+5, y@_FillValue, y)
  y = where(isnan_ieee(y), y@_FillValue, y)
  return(y)
end

begin

  enter_msg(DIAG_SCRIPT, "")

  work_dir = config_user_info@work_dir + "/"
  ; Create work dir
  system("mkdir -p " + work_dir)

  ; Configuration

  syear_clim = 1985
  eyear_clim = 2004
  if (isatt(diag_script_info, "syear_clim")) then
    syear_clim = diag_script_info@syear_clim
  end if
  if (isatt(diag_script_info, "eyear_clim")) then
    eyear_clim = diag_script_info@eyear_clim
  end if

  smonth_NHsummer = 5
  emonth_NHsummer = 9
  smonth_SHsummer = 11
  emonth_SHsummer = 3
  if (isatt(diag_script_info, "smonth_NHsummer")) then
    smonth_NHsummer = diag_script_info@smonth_NHsummer
  end if
  if (isatt(diag_script_info, "emonth_NHsummer")) then
    emonth_NHsummer = diag_script_info@emonth_NHsummer
  end if
  if (isatt(diag_script_info, "smonth_SHsummer")) then
    smonth_SHsummer = diag_script_info@smonth_SHsummer
  end if
  if (isatt(diag_script_info, "emonth_SHsummer")) then
    emonth_SHsummer = diag_script_info@emonth_SHsummer
  end if

  smonth_monsoonyr = smonth_NHsummer
  emonth_monsoonyr = smonth_monsoonyr-1
  if (emonth_monsoonyr.le.0) then
    emonth_monnsoonyr = emonth_monsoonyr+12
  end if

  varying_domain = False
  if (isatt(diag_script_info, "varying_domain")) then
    varying_domain = diag_script_info@varying_domain
  end if

  optDomain = True
  optDomain@threshold_mpi = 0.5
  optDomain@threshold_summer_minus_winter = 300.
  if (isatt(diag_script_info, "threshold_mpi")) then
    optDomain@threshold_mpi = diag_script_info@threshold_mpi
  end if
  if (isatt(diag_script_info, "threshold_summer_minus_winter")) then
    optDomain@threshold_summer_minus_winter \
      = diag_script_info@threshold_summer_minus_winter
  end if

  ; Get var list

  varList = metadata_att_as_array(input_file_info, "short_name")
  varList := get_unique_values(varList)

  if (all(varList.ne."pr")) then
    error_msg("f", DIAG_SCRIPT, "", "variable pr is missing.")
  end if

  if (any(varList.eq."uas")) then
    uvar = "uas"
  elseif (any(varList.eq."ua")) then
    uvar = "ua"
  else
    uvar = ""
  end if
  if (any(varList.eq."vas")) then
    vvar = "vas"
  elseif (any(varList.eq."va")) then
    vvar = "va"
  else
    vvar = ""
  end if

  var = "pr"
  infoPr = select_metadata_by_name(input_file_info, var)
  dataset_names = metadata_att_as_array(infoPr, "dataset")
  iPrMME     = ind(dataset_names .eq. "MultiModelMean")
  do i = 0, ListCount(infoPr)-1
    if (any(infoPr[i]@project.eq.\
            (/"obs", "OBS", "OBS6", "obs4mips", "ana4mips"/))) then
      dataset = infoPr[i]@dataset
      if (isvar("iPrObs")) then
        tmp = array_append_record(iPrObs, (/i/), 0)
        delete(iPrObs)
        iPrObs = tmp
        delete(tmp)
      else
        iPrObs = (/i/)
      end if
    end if
  end do

  if (uvar.ne."" .and. vvar.ne."") then
    infoU = select_metadata_by_name(input_file_info, uvar)
    delete(dataset_names)
    dataset_names = metadata_att_as_array(infoU, "dataset")
    iUMME    = ind(dataset_names .eq. "MultiModelMean")

    do i = 0, ListCount(infoU)-1
      if (any(infoU[i]@project.eq.\
              (/"obs", "OBS", "obs4mips", "ana4mips"/))) then
        if (isvar("iUObs")) then
          tmp = array_append_record(iUObs, (/i/), 0)
          delete(iUObs)
          iUObs = tmp
          delete(tmp)
        else
          iUObs = (/i/)
        end if
      end if
    end do

    infoV = select_metadata_by_name(input_file_info, vvar)
    delete(dataset_names)
    dataset_names = metadata_att_as_array(infoV, "dataset")
    iVMME    = ind(dataset_names .eq. "MultiModelMean")

    do i = 0, ListCount(infoV)-1
      if (any(infoV[i]@project.eq. \
              (/"obs", "OBS", "OBS6", "obs4mips", "ana4mips"/))) then
        if (isvar("iVObs")) then
          tmp = array_append_record(iVObs, (/i/), 0)
          delete(iVObs)
          iVObs = tmp
          delete(tmp)
        else
          iVObs = (/i/)
        end if
      end if
    end do

  end if

; ========================================================================

  opt = True
  opt@smonth_NHsummer = smonth_NHsummer
  opt@emonth_NHsummer = emonth_NHsummer
  opt@smonth_SHsummer = smonth_SHsummer
  opt@emonth_SHsummer = emonth_SHsummer

  optAccum = opt
  optAccum@accumulate = True

  do ii = 0, ListCount(infoPr)-1

    dataset = infoPr[ii]@dataset
    pr = read_data(infoPr[ii])

    if (any(pr@units.eq.(/"mm/s", "kg/m^2/s", "mm s-1", "kg m-2 s-1" /))) then
      pr = pr*86400.
      pr@units = "mm/day"
    end if

    prClim = get_clim(pr, syear_clim, eyear_clim, \
                      smonth_monsoonyr, emonth_monsoonyr)
    delete(pr)

    prRange = summer_minus_winter(prClim, opt)  ; mm/day
    prClimANN = dim_avg_n(prClim, 0)
    MPI = prRange
    MPI = prRange / where(prClimANN.gt.0., prClimANN, prClimANN@_FillValue)
    MPI@long_name = "Monsoon precipitation index"
    MPI@units = ""

    prRange = summer_minus_winter(prClim, optAccum)  ; mm
    prRange@var = "pr"
    prRange@units = "mm"
    prRange@diag_script = DIAG_SCRIPT
    prRange@dataset = dataset
    nc_filename = work_dir + "precip_monsoon_"+dataset+".nc"
    nc_outfile = ncdf_write(prRange, nc_filename)

    domain := monsoon_domain(MPI, prRange, optDomain)
    domain@var = "monsoon_domain"
    domain@diag_script = DIAG_SCRIPT
    domain@dataset = dataset
    nc_filename = work_dir + "domain_monsoon_"+dataset+".nc"
    nc_outfile = ncdf_write(domain, nc_filename)

    MPI@var = "MPI"
    MPI@diag_script = DIAG_SCRIPT
    MPI@dataset = dataset
    nc_filename = work_dir + "monsoon_precip_index_"+dataset+".nc"
    nc_outfile = ncdf_write(MPI, nc_filename)

    delete(prClim)
    delete(prClimANN)
    delete(prRange)
    delete(MPI)
    delete(domain)
  end do

  if (uvar.ne."" .and. vvar.ne."") then
    do ii = 0, ListCount(infoU)-1
      dataset = infoU[ii]@dataset

      u = read_data(infoU[ii])
      if (dimsizes(dimsizes(u)).gt.3) then
        utmp = u(:, {85000.}, :, :)
        delete(u)
        u = utmp
        delete(utmp)
      end if
      u = correct_FillValue_uv(u)
      uClim = get_clim(u, syear_clim, eyear_clim, \
                       smonth_monsoonyr, emonth_monsoonyr)
      uMonsoon = summer_minus_winter(uClim, opt)

      uMonsoon@var = "u"
      uMonsoon@diag_script = DIAG_SCRIPT
      uMonsoon@dataset = dataset
      uMonsoon@long_name = "Climatological summer - winter zonal wind velocity"
      nc_filename = work_dir + "u_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(uMonsoon, nc_filename)
      delete(u)
      delete(uClim)
      delete(uMonsoon)
    end do

    do ii = 0, ListCount(infoV)-1
      dataset = infoV[ii]@dataset

      v = read_data(infoV[ii])
      if (dimsizes(dimsizes(v)).gt.3) then
        vtmp = v(:, {85000.}, :, :)
        delete(v)
        v = vtmp
        delete(vtmp)
      end if
      v = correct_FillValue_uv(v)
      vClim = get_clim(v, syear_clim, eyear_clim, \
                       smonth_monsoonyr, emonth_monsoonyr)
      vMonsoon = summer_minus_winter(vClim, opt)

      vMonsoon@var = "v"
      vMonsoon@diag_script = DIAG_SCRIPT
      vMonsoon@dataset = dataset
      vMonsoon@long_name \
        = "Climatological summer - winter meridional wind velocity"
      nc_filename = work_dir + "v_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(vMonsoon, nc_filename)
      delete(v)
      delete(vClim)
      delete(vMonsoon)
    end do
  end if

; ====================== monsoon precipitation index ========================

  input_dir = config_user_info@work_dir + "/"

  do ii = 0, ListCount(infoPr) - 1
    pr = read_data(infoPr[ii])
    if (any(pr@units.eq.(/"mm/s", "kg/m^2/s", "mm s-1", "kg m-2 s-1" /))) then
      pr = pr*86400.
      pr@units = "mm/day"
    end if
    time = cd_calendar(pr&time, -1)
    pr&time = time
    syear = time(0)/100
    smonth = time(0)-syear*100
    eyear = time(dimsizes(time)-1)/100
    emonth = time(dimsizes(time)-1)-eyear*100
    if (.not.isatt(pr, "_FillValue")) then
      pr@_FillValue = default_fillvalue(typeof(pr))
    end if

    if (varying_domain) then
      opt = True
      opt@smonth_NHsummer = smonth_NHsummer
      opt@emonth_NHsummer = emonth_NHsummer
      opt@smonth_SHsummer = smonth_SHsummer
      opt@emonth_SHsummer = emonth_SHsummer

      prRange = summer_minus_winter(pr, opt)  ; mm/day
      prANN   = prRange
      do t = 0, dimsizes(prRange&time)-1
        yr = prRange&time(t)
        prANN(t, :, :) = dim_avg_n(pr({yr*100+1:yr*100+12}, :, :), 0)
      end do
      MPI = prRange
      MPI = prRange / where(prANN.gt.0., prANN, prANN@_FillValue)

      opt@accumulate = True
      prRange = summer_minus_winter(pr, opt)  ; mm
      domain := monsoon_domain(MPI, prRange, optDomain)
      delete(prRange)
      delete(prANN)
      delete(MPI)
      delete(opt)
      syear_out = toint(domain&time(0))
      eyear_out = toint(domain&time(dimsizes(domain&time)-1))
    else
      domainFile = input_dir+"domain_monsoon_"+infoPr[ii]@dataset+".nc"
      f = addfile(domainFile, "r")
      domain = f->monsoon_domain

      syear_out = syear
      eyear_out = eyear
      if (smonth_NHsummer.gt.emonth_NHsummer .or. \
          smonth_SHsummer.gt.emonth_SHsummer) then
        syear_out = syear_out + 1
      elseif (smonth.gt.min((/smonth_NHsummer, smonth_SHsummer/))) then
        syear_out = syear_out + 1
      end if
      if (emonth.lt.max((/emonth_NHsummer, emonth_SHsummer/))) then
        eyear_out = eyear_out - 1
      end if
    end if

    prSummer = pr({syear_out*100+1:eyear_out*100+12:12}, :, :)
    prSummer&time = ispan(syear_out, eyear_out, 1)
    prSummer = (/summer(pr, optAccum)/)

    if (varying_domain) then
      prSummer = where(domain.lt.0.5, prSummer@_FillValue, prSummer)
    else
      prSummer = where(conform(prSummer, domain,  (/1, 2/)).lt.0.5, \
                       prSummer@_FillValue, prSummer)
    end if
    delete(domain)
    delete(pr)

    wgty = latRegWgt(prSummer&lat, typeof(prSummer), 0)
    wgty!0 = "lat"
    wgty&lat = prSummer&lat
    GMpr = prSummer(:, 0, 0)
    GMpr = wgt_areaave(prSummer, wgty, 1., 0)

    delete(GMpr@lon)
    delete(GMpr@lat)
    GMpr = GMpr - avg(GMpr({syear_clim:eyear_clim}))

    GMpr@eyear_ref = eyear_clim
    GMpr@syear_ref = syear_clim
    GMpr@var = "GMpr"
    GMpr@long_name = "global summer monsoon precipitation anomaly"
    GMpr@diag_script = DIAG_SCRIPT
    GMpr@project = infoPr[ii]@project
    GMpr@dataset = infoPr[ii]@dataset

    work_dir = config_user_info@work_dir + "/"
    system("mkdir -p " + work_dir)
    if (isatt(infoPr[ii], "ensemble")) then
      nc_filename = work_dir + "GMprecip_summer_"\
                     + infoPr[ii]@dataset + "_" + infoPr[ii]@ensemble + ".nc"
    else
      nc_filename = work_dir + "GMprecip_summer_" + infoPr[ii]@dataset + ".nc"
    end if

    if (nc_filename.ne."") then
      nc_outfile = ncdf_write(GMpr, nc_filename)
    end if

    NHMpr = prSummer(:, 0, 0)
    NHMpr = wgt_areaave(prSummer(:, {0.:90.}, :), wgty({0.:90.}), 1., 0)

    delete(NHMpr@lon)
    delete(NHMpr@lat)
    NHMpr = NHMpr - avg(NHMpr({syear_clim:eyear_clim}))

    NHMpr@eyear_ref = eyear_clim
    NHMpr@syear_ref = syear_clim
    NHMpr@var = "NHMpr"
    NHMpr@long_name = "NH summer monsoon precipitation anomaly"
    NHMpr@diag_script = DIAG_SCRIPT
    NHMpr@project = infoPr[ii]@project
    NHMpr@dataset = infoPr[ii]@dataset

    if (isatt(infoPr[ii], "ensemble")) then
      nc_filename = work_dir + "NHMprecip_summer_" + infoPr[ii]@dataset \
                     + "_" + infoPr[ii]@ensemble + ".nc"
    else
      nc_filename = work_dir + "NHMprecip_summer_" + infoPr[ii]@dataset + ".nc"
    end if
    if (nc_filename.ne."") then
      nc_outfile = ncdf_write(NHMpr, nc_filename)
    end if

    SHMpr = prSummer(:, 0, 0)
    SHMpr = wgt_areaave(prSummer(:, {-90.:0.}, :), wgty({-90.:0.}), 1., 0)

    delete(SHMpr@lon)
    delete(SHMpr@lat)
    SHMpr = SHMpr - avg(SHMpr({syear_clim:eyear_clim}))

    SHMpr@eyear_ref = eyear_clim
    SHMpr@syear_ref = syear_clim
    SHMpr@var = "SHMpr"
    SHMpr@long_name = "SH summer monsoon precipitation anomaly"
    SHMpr@diag_script = DIAG_SCRIPT
    SHMpr@project = infoPr[ii]@project
    SHMpr@dataset = infoPr[ii]@dataset

    if (isatt(infoPr[ii], "ensemble")) then
      nc_filename = work_dir + "SHMprecip_summer_" + infoPr[ii]@dataset \
                     + "_"+infoPr[ii]@ensemble + ".nc"
    else
      nc_filename = work_dir + "SHMprecip_summer_" + infoPr[ii]@dataset + ".nc"
    end if
    if (nc_filename.ne."") then
      nc_outfile = ncdf_write(SHMpr, nc_filename)
    end if

    delete(time)
    delete(wgty)
    delete(prSummer)
    delete(GMpr)
    delete(NHMpr)
    delete(SHMpr)
  end do

; ======================= monsoon circulation index =========================

  if (uvar.ne."") then
    u = read_data(infoU[0])
    ndims = dimsizes(dimsizes(u))
    delete(u)
    if (ndims.eq.4) then
      do ii = 0, ListCount(infoU) - 1
        u = read_data(infoU[ii])
        time = cd_calendar(u&time, -1)
        u&time = time
        syear = time(0)/100
        smonth = time(0)-syear*100
        eyear = time(dimsizes(time)-1)/100
        emonth = time(dimsizes(time)-1)-eyear*100
        if (.not.isatt(u, "_FillValue")) then
          u@_FillValue = default_fillvalue(typeof(u))
        end if
        u = correct_FillValue_uv(u)
        ud = u(:, {85000.}, :, :)
        ud = u(:, {85000.}, :, :) - u(:, {20000.}, :, :)
        delete(u)
        delete(time)

        udSummer = summer(ud, opt)
        delete(ud)

        udSummer = lonFlip(udSummer)

        udNHSM = udSummer(:, {0.:20.}, {-120.:120.})
        delete(udSummer)

        wgty = latRegWgt(udNHSM&lat, typeof(udNHSM), 0)
        wgty!0 = "lat"
        wgty&lat = udNHSM&lat
        NHMu = udNHSM(:, 0, 0)
        NHMu = wgt_areaave(udNHSM, wgty, 1., 0)
        delete(wgty)

        NHMu = NHMu - avg(NHMu({syear_clim:eyear_clim}))

        NHMu@eyear_ref = eyear_clim
        NHMu@syear_ref = syear_clim
        NHMu@var = "NHMu"
        NHMu@long_name = "global summer monsoon circulation index anomaly"
        NHMu@diag_script = DIAG_SCRIPT
        NHMu@project = infoU[ii]@project
        NHMu@dataset = infoU[ii]@dataset

        work_dir = config_user_info@work_dir + "/"
        system("mkdir -p " + work_dir)
        if (isatt(infoU[ii], "ensemble")) then
          nc_filename = work_dir + "NHMcirculation_summer_" \
                         + infoU[ii]@dataset + "_" + infoU[ii]@ensemble + ".nc"
        else
          nc_filename = work_dir + "NHMcirculation_summer_" \
                         + infoU[ii]@dataset + ".nc"
        end if

        if (nc_filename.ne."") then
          nc_outfile = ncdf_write(NHMu, nc_filename)
        end if

        delete(udNHSM)
        delete(NHMu)
      end do
    end if
  end if

  leave_msg(DIAG_SCRIPT, "")

end
