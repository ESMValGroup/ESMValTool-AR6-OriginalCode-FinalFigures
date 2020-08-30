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
;    20200825 kosaka_yu: calculate multimodel means individually for CMIP5 and CMIP6.
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

  scale_monsoonPr = True
  if (isatt(diag_script_info, "scale_monsoonprecip")) then
    scale_monsoonPr = diag_script_info@scale_monsoonPr
  end if

  
  do_climPr = True
  do_climUV = True
  do_GMpr = True
  do_NHMpr = False
  do_SHMpr = False    
  do_NHMu = True
    
  obs_projects = (/"obs", "OBS", "OBS6", "obs4mips", "ana4mips"/)
  
  ; Get var list

  varList = metadata_att_as_array(input_file_info, "short_name")
  varList := get_unique_values(varList)

  if (all(varList.ne."pr")) then
    do_climPr = False
    do_GMpr = False
    do_NHMpr = False
    do_SHMpr = False
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
  if (any((/uvar, vvar/).eq."")) then
    do_climUV = False
  end if

  
  if (any((/do_climPr, do_GMpr, do_SHMpr, do_NHMpr/))) then
    var = "pr"
    infoPr = select_metadata_by_name(input_file_info, var)
    do i = 0, ListCount(infoPr)-1
      if (ismissing(infoPr[i]@project)) then
        continue
      end if
      if (any(infoPr[i]@project.eq.obs_projects)) then
        if (isvar("iPrObs")) then
          tmp = array_append_record(iPrObs, (/i/), 0)
          delete(iPrObs)
          iPrObs = tmp
          delete(tmp)
        else
          iPrObs = (/i/)
        end if
      elseif (infoPr[i]@project.eq."CMIP5".and.isStrSubset(infoPr[i]@exp, "historical")) then
        if (isvar("iPrCMIP5")) then
          tmp = array_append_record(iPrCMIP5, (/i/), 0)
          delete(iPrCMIP5)
          iPrCMIP5 = tmp
          delete(tmp)
        else
          iPrCMIP5 = (/i/)
        end if
      elseif (infoPr[i]@project.eq."CMIP6".and.isStrSubset(infoPr[i]@exp, "historical")) then
        if (isvar("iPrCMIP6")) then
          tmp = array_append_record(iPrCMIP6, (/i/), 0)
          delete(iPrCMIP6)
          iPrCMIP6 = tmp
          delete(tmp)
        else
          iPrCMIP6 = (/i/)
        end if
      elseif (isatt(infoPr[i], "exp") .and. infoPr[i]@exp.eq."amip-hist") then
        if (isvar("iPrAMIP")) then
          tmp = array_append_record(iPrAMIP, (/i/), 0)
          delete(iPrAMIP)
          iPrAMIP = tmp
          delete(tmp)
        else
          iPrAMIP = (/i/)
        end if
      end if
    end do
  end if

;  if (uvar.ne."" .and. vvar.ne."") then
  if (uvar.ne."") then
    do_NHMu = True
    infoU = select_metadata_by_name(input_file_info, uvar)

    do i = 0, ListCount(infoU)-1
      if (ismissing(infoU[i]@project)) then
        continue
      end if
      if (any(infoU[i]@project.eq.obs_projects)) then
        if (isvar("iUObs")) then
          tmp = array_append_record(iUObs, (/i/), 0)
          delete(iUObs)
          iUObs = tmp
          delete(tmp)
        else
          iUObs = (/i/)
        end if
      elseif (infoU[i]@project.eq."CMIP5".and.isStrSubset(infoU[i]@exp, "historical")) then
        if (isvar("iUCMIP5")) then
          tmp = array_append_record(iUCMIP5, (/i/), 0)
          delete(iUCMIP5)
          iUCMIP5 = tmp
          delete(tmp)
        else
          iUCMIP5 = (/i/)
        end if
      elseif (infoU[i]@project.eq."CMIP6".and.isStrSubset(infoU[i]@exp, "historical")) then
        if (isvar("iUCMIP6")) then
          tmp = array_append_record(iUCMIP6, (/i/), 0)
          delete(iUCMIP6)
          iUCMIP6 = tmp
          delete(tmp)
        else
          iUCMIP6 = (/i/)
        end if
      elseif (isatt(infoU[i], "exp") .and. infoU[i]@exp.eq."amip-hist") then
        if (isvar("iUAMIP")) then
          tmp = array_append_record(iUAMIP, (/i/), 0)
          delete(iUAMIP)
          iUAMIP = tmp
          delete(tmp)
        else
          iUAMIP = (/i/)
        end if
      end if
    end do
  else
    do_NHMu = False
  end if

  if (vvar.ne."") then
    infoV = select_metadata_by_name(input_file_info, vvar)

    do i = 0, ListCount(infoV)-1
      if (ismissing(infoV[i]@project)) then
        continue
      end if
      if (any(infoV[i]@project.eq.obs_projects)) then
        if (isvar("iVObs")) then
          tmp = array_append_record(iVObs, (/i/), 0)
          delete(iVObs)
          iVObs = tmp
          delete(tmp)
        else
          iVObs = (/i/)
        end if
      elseif (infoV[i]@project.eq."CMIP5".and.isStrSubset(infoV[i]@exp, "historical")) then
        if (isvar("iVCMIP5")) then
          tmp = array_append_record(iVCMIP5, (/i/), 0)
          delete(iVCMIP5)
          iVCMIP5 = tmp
          delete(tmp)
        else
          iVCMIP5 = (/i/)
        end if
      elseif (infoV[i]@project.eq."CMIP6".and.isStrSubset(infoV[i]@exp, "historical")) then
        if (isvar("iVCMIP6")) then
          tmp = array_append_record(iVCMIP6, (/i/), 0)
          delete(iVCMIP6)
          iVCMIP6 = tmp
          delete(tmp)
        else
          iVCMIP6 = (/i/)
        end if
      elseif (isatt(infoV[i], "exp") .and. infoV[i]@exp.eq."amip-hist") then
        if (isvar("iVAMIP")) then
          tmp = array_append_record(iVAMIP, (/i/), 0)
          delete(iVAMIP)
          iVAMIP = tmp
          delete(tmp)
        else
          iVAMIP = (/i/)
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

  if (do_climPr) then
    do ii = 0, ListCount(infoPr)-1

      dataset = infoPr[ii]@dataset
      if (isatt(infoPr[ii], "ensemble")) then
        ensemble = infoPr[ii]@ensemble
      end if
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
      prRange@syear_clim = syear_clim
      prRange@eyear_clim = eyear_clim
      prRange@smonth_NHsummer = smonth_NHsummer
      prRange@emonth_NHsummer = emonth_NHsummer
      prRange@smonth_SHsummer = smonth_SHsummer
      prRange@emonth_SHsummer = emonth_SHsummer
      if (isvar("ensemble")) then
        nc_filename = work_dir + "precip_monsoon_"+dataset+"_"+ensemble+".nc"
      else
        nc_filename = work_dir + "precip_monsoon_"+dataset+".nc"
      end if
      nc_outfile = ncdf_write(prRange, nc_filename)

      domain := monsoon_domain(MPI, prRange, optDomain)
      domain@var = "monsoon_domain"
      domain@diag_script = DIAG_SCRIPT
      domain@dataset = dataset
      domain@syear_clim = syear_clim
      domain@eyear_clim = eyear_clim
      domain@smonth_NHsummer = smonth_NHsummer
      domain@emonth_NHsummer = emonth_NHsummer
      domain@smonth_SHsummer = smonth_SHsummer
      domain@emonth_SHsummer = emonth_SHsummer
      if (isvar("ensemble")) then
        nc_filename = work_dir + "domain_monsoon_"+dataset+"_"+ensemble+".nc"
      else
        nc_filename = work_dir + "domain_monsoon_"+dataset+".nc"
      end if
      nc_outfile = ncdf_write(domain, nc_filename)

      MPI@var = "MPI"
      MPI@diag_script = DIAG_SCRIPT
      MPI@dataset = dataset
      MPI@syear_clim = syear_clim
      MPI@eyear_clim = eyear_clim
      MPI@smonth_NHsummer = smonth_NHsummer
      MPI@emonth_NHsummer = emonth_NHsummer
      MPI@smonth_SHsummer = smonth_SHsummer
      MPI@emonth_SHsummer = emonth_SHsummer
      if (isvar("ensemble")) then
        nc_filename = work_dir + "monsoon_precip_index_"+dataset+"_"+ensemble+".nc"
      else
        nc_filename = work_dir + "monsoon_precip_index_"+dataset+".nc"
      end if
      nc_outfile = ncdf_write(MPI, nc_filename)

      if (isvar("iPrCMIP5")) then
        if (any(ii.eq.iPrCMIP5)) then
          w = 1./tofloat(num(metadata_att_as_array(infoPr, "dataset").eq.infoPr[ii]@dataset))
          if (isvar("prClimCMIP5")) then
            prClimCMIP5 = prClimCMIP5 + prClim * w
            nprClimCMIP5 = nprClimCMIP5 + w
          else
            prClimCMIP5 = prClim
            prClimCMIP5 = prClim * w
            nprClimCMIP5 = w
          end if
        end if
      end if
      if (isvar("iPrCMIP6")) then
        if (any(ii.eq.iPrCMIP6)) then
          w = 1./tofloat(num(metadata_att_as_array(infoPr, "dataset").eq.infoPr[ii]@dataset))
          if (isvar("prClimCMIP6")) then
            prClimCMIP6 = prClimCMIP6 + prClim * w
            nprClimCMIP6 = nprClimCMIP6 + w
          else
            prClimCMIP6 = prClim
            prClimCMIP6 = prClim * w
            nprClimCMIP6 = w
          end if
        end if
      end if
      if (isvar("iPrAMIP")) then
        if (any(ii.eq.iPrAMIP)) then
          w = 1./tofloat(num(metadata_att_as_array(infoPr, "dataset").eq.infoPr[ii]@dataset))
          if (isvar("prClimAMIP")) then
            prClimAMIP = prClimAMIP + prClim * w
            nprClimAMIP = nprClimAMIP + w
          else
            prClimAMIP = prClim
            prClimAMIP = prClim * w
            nprClimAMIP = w
          end if
        end if
      end if
    
      delete(prClim)
      delete(prClimANN)
      delete(prRange)
      delete(MPI)
      delete(domain)
      if (isvar("ensemble")) then
        delete(ensemble)
      end if
    end do

    if (isvar("prClimCMIP5")) then
      prClimCMIP5 = prClimCMIP5 / nprClimCMIP5

      dataset = "MultiModelMeanCMIP5"

      prRange = summer_minus_winter(prClimCMIP5, opt)  ; mm/day
      prClimANN = dim_avg_n(prClimCMIP5, 0)
      MPI = prRange
      MPI = prRange / where(prClimANN.gt.0., prClimANN, prClimANN@_FillValue)
      MPI@long_name = "Monsoon precipitation index"
      MPI@units = ""

      prRange = summer_minus_winter(prClimCMIP5, optAccum)  ; mm
      prRange@var = "pr"
      prRange@units = "mm"
      prRange@diag_script = DIAG_SCRIPT
      prRange@dataset = dataset
      prRange@syear_clim = syear_clim
      prRange@eyear_clim = eyear_clim
      prRange@smonth_NHsummer = smonth_NHsummer
      prRange@emonth_NHsummer = emonth_NHsummer
      prRange@smonth_SHsummer = smonth_SHsummer
      prRange@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "precip_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(prRange, nc_filename)

      domain := monsoon_domain(MPI, prRange, optDomain)
      domain@var = "monsoon_domain"
      domain@diag_script = DIAG_SCRIPT
      domain@dataset = dataset
      domain@syear_clim = syear_clim
      domain@eyear_clim = eyear_clim
      domain@smonth_NHsummer = smonth_NHsummer
      domain@emonth_NHsummer = emonth_NHsummer
      domain@smonth_SHsummer = smonth_SHsummer
      domain@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "domain_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(domain, nc_filename)

      MPI@var = "MPI"
      MPI@diag_script = DIAG_SCRIPT
      MPI@dataset = dataset
      MPI@syear_clim = syear_clim
      MPI@eyear_clim = eyear_clim
      MPI@smonth_NHsummer = smonth_NHsummer
      MPI@emonth_NHsummer = emonth_NHsummer
      MPI@smonth_SHsummer = smonth_SHsummer
      MPI@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "monsoon_precip_index_"+dataset+".nc"
      nc_outfile = ncdf_write(MPI, nc_filename)
      delete(prClimCMIP5)
      delete(nprClimCMIP5)
      delete(prClimANN)
      delete(prRange)
      delete(MPI)
      delete(domain)
    end if

    if (isvar("prClimCMIP6")) then
      prClimCMIP6 = prClimCMIP6 / nprClimCMIP6

      dataset = "MultiModelMeanCMIP6"

      prRange = summer_minus_winter(prClimCMIP6, opt)  ; mm/day
      prClimANN = dim_avg_n(prClimCMIP6, 0)
      MPI = prRange
      MPI = prRange / where(prClimANN.gt.0., prClimANN, prClimANN@_FillValue)
      MPI@long_name = "Monsoon precipitation index"
      MPI@units = ""

      prRange = summer_minus_winter(prClimCMIP6, optAccum)  ; mm
      prRange@var = "pr"
      prRange@units = "mm"
      prRange@diag_script = DIAG_SCRIPT
      prRange@dataset = dataset
      prRange@syear_clim = syear_clim
      prRange@eyear_clim = eyear_clim
      prRange@smonth_NHsummer = smonth_NHsummer
      prRange@emonth_NHsummer = emonth_NHsummer
      prRange@smonth_SHsummer = smonth_SHsummer
      prRange@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "precip_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(prRange, nc_filename)

      domain := monsoon_domain(MPI, prRange, optDomain)
      domain@var = "monsoon_domain"
      domain@diag_script = DIAG_SCRIPT
      domain@dataset = dataset
      domain@syear_clim = syear_clim
      domain@eyear_clim = eyear_clim
      domain@smonth_NHsummer = smonth_NHsummer
      domain@emonth_NHsummer = emonth_NHsummer
      domain@smonth_SHsummer = smonth_SHsummer
      domain@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "domain_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(domain, nc_filename)

      MPI@var = "MPI"
      MPI@diag_script = DIAG_SCRIPT
      MPI@dataset = dataset
      MPI@syear_clim = syear_clim
      MPI@eyear_clim = eyear_clim
      MPI@smonth_NHsummer = smonth_NHsummer
      MPI@emonth_NHsummer = emonth_NHsummer
      MPI@smonth_SHsummer = smonth_SHsummer
      MPI@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "monsoon_precip_index_"+dataset+".nc"
      nc_outfile = ncdf_write(MPI, nc_filename)
      delete(prClimCMIP6)
      delete(nprClimCMIP6)
      delete(prClimANN)
      delete(prRange)
      delete(MPI)
      delete(domain)
    end if

    if (isvar("prClimAMIP")) then
      prClimAMIP = prClimAMIP / nprClimAMIP

      dataset = "MultiModelMeanAMIP"

      prRange = summer_minus_winter(prClimAMIP, opt)  ; mm/day
      prClimANN = dim_avg_n(prClimAMIP, 0)
      MPI = prRange
      MPI = prRange / where(prClimANN.gt.0., prClimANN, prClimANN@_FillValue)
      MPI@long_name = "Monsoon precipitation index"
      MPI@units = ""

      prRange = summer_minus_winter(prClimAMIP, optAccum)  ; mm
      prRange@var = "pr"
      prRange@units = "mm"
      prRange@diag_script = DIAG_SCRIPT
      prRange@dataset = dataset
      prRange@syear_clim = syear_clim
      prRange@eyear_clim = eyear_clim
      prRange@smonth_NHsummer = smonth_NHsummer
      prRange@emonth_NHsummer = emonth_NHsummer
      prRange@smonth_SHsummer = smonth_SHsummer
      prRange@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "precip_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(prRange, nc_filename)

      domain := monsoon_domain(MPI, prRange, optDomain)
      domain@var = "monsoon_domain"
      domain@diag_script = DIAG_SCRIPT
      domain@dataset = dataset
      domain@syear_clim = syear_clim
      domain@eyear_clim = eyear_clim
      domain@smonth_NHsummer = smonth_NHsummer
      domain@emonth_NHsummer = emonth_NHsummer
      domain@smonth_SHsummer = smonth_SHsummer
      domain@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "domain_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(domain, nc_filename)

      MPI@var = "MPI"
      MPI@diag_script = DIAG_SCRIPT
      MPI@dataset = dataset
      MPI@syear_clim = syear_clim
      MPI@eyear_clim = eyear_clim
      MPI@smonth_NHsummer = smonth_NHsummer
      MPI@emonth_NHsummer = emonth_NHsummer
      MPI@smonth_SHsummer = smonth_SHsummer
      MPI@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "monsoon_precip_index_"+dataset+".nc"
      nc_outfile = ncdf_write(MPI, nc_filename)
      delete(prClimAMIP)
      delete(nprClimAMIP)
      delete(prClimANN)
      delete(prRange)
      delete(MPI)
      delete(domain)
    end if
  end if
  
;  if (uvar.ne."" .and. vvar.ne."") then
  if (do_climUV) then
    do ii = 0, ListCount(infoU)-1
      dataset = infoU[ii]@dataset
      if (isatt(infoU[ii], "ensemble")) then
        ensemble = infoU[ii]@ensemble
      end if
      
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
      uMonsoon@syear_clim = syear_clim
      uMonsoon@eyear_clim = eyear_clim
      uMonsoon@smonth_NHsummer = smonth_NHsummer
      uMonsoon@emonth_NHsummer = emonth_NHsummer
      uMonsoon@smonth_SHsummer = smonth_SHsummer
      uMonsoon@emonth_SHsummer = emonth_SHsummer
      if (isvar("ensemble")) then
        nc_filename = work_dir + "u_monsoon_"+dataset+"_"+ensemble+".nc"
      else
        nc_filename = work_dir + "u_monsoon_"+dataset+".nc"
      end if
      nc_outfile = ncdf_write(uMonsoon, nc_filename)

      if (isvar("iUCMIP5")) then
        if (any(ii.eq.iUCMIP5)) then
          w = 1./tofloat(num(metadata_att_as_array(infoU, "dataset").eq.infoU[ii]@dataset))
          if (isvar("uClimCMIP5")) then
            uClimCMIP5 = uClimCMIP5 + uClim * w
            nuClimCMIP5 = nuClimCMIP5 + w
          else
            uClimCMIP5 = uClim
            uClimCMIP5 = uClim * w
            nuClimCMIP5 = w
          end if
        end if
      end if
      if (isvar("iUCMIP6")) then
        if (any(ii.eq.iUCMIP6)) then
          w = 1./tofloat(num(metadata_att_as_array(infoU, "dataset").eq.infoU[ii]@dataset))
          if (isvar("uClimCMIP6")) then
            uClimCMIP6 = uClimCMIP6 + uClim * w
            nuClimCMIP6 = nuClimCMIP6 + w
          else
            uClimCMIP6 = uClim
            uClimCMIP6 = uClim * w
            nuClimCMIP6 = w
          end if
        end if
      end if
      if (isvar("iUAMIP")) then
        if (any(ii.eq.iUAMIP)) then
          w = 1./tofloat(num(metadata_att_as_array(infoU, "dataset").eq.infoU[ii]@dataset))
          if (isvar("uClimAMIP")) then
            uClimAMIP = uClimAMIP + uClim * w
            nuClimAMIP = nuClimAMIP + w
          else
            uClimAMIP = uClim
            uClimAMIP = uClim * w
            nuClimAMIP = w
          end if
        end if
      end if
      delete(u)
      delete(uClim)
      delete(uMonsoon)
      if (isvar("ensemble")) then
        delete(ensemble)
      end if
    end do

    do ii = 0, ListCount(infoV)-1
      dataset = infoV[ii]@dataset
      if (isatt(infoV[ii], "ensemble")) then
        ensemble = infoV[ii]@ensemble
      end if
      
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
      vMonsoon@syear_clim = syear_clim
      vMonsoon@eyear_clim = eyear_clim
      vMonsoon@smonth_NHsummer = smonth_NHsummer
      vMonsoon@emonth_NHsummer = emonth_NHsummer
      vMonsoon@smonth_SHsummer = smonth_SHsummer
      vMonsoon@emonth_SHsummer = emonth_SHsummer
      if (isvar("ensemble")) then
        nc_filename = work_dir + "v_monsoon_"+dataset+"_"+ensemble+".nc"
      else
        nc_filename = work_dir + "v_monsoon_"+dataset+".nc"
      end if
      nc_outfile = ncdf_write(vMonsoon, nc_filename)

      if (isvar("iVMIP5")) then
        if (any(ii.eq.iVCMIP5)) then
          w = 1./tofloat(num(metadata_att_as_array(infoV, "dataset").eq.infoV[ii]@dataset))
          if (isvar("vClimCMIP5")) then
            vClimCMIP5 = vClimCMIP5 + vClim * w
            nvClimCMIP5 = nvClimCMIP5 + w
          else
            vClimCMIP5 = vClim
            vClimCMIP5 = vClim * w
            nvClimCMIP5 = w
          end if
        end if
      end if
      if (isvar("iVCMIP6")) then
        if (any(ii.eq.iVCMIP6)) then
          w = 1./tofloat(num(metadata_att_as_array(infoV, "dataset").eq.infoV[ii]@dataset))
          if (isvar("vClimCMIP6")) then
            vClimCMIP6 = vClimCMIP6 + vClim * w
            nvClimCMIP6 = nvClimCMIP6 + w
          else
            vClimCMIP6 = vClim
            vClimCMIP6 = vClim * w
            nvClimCMIP6 = w
          end if
        end if
      end if
      if (isvar("iVAMIP")) then
        if (any(ii.eq.iVAMIP)) then
          w = 1./tofloat(num(metadata_att_as_array(infoV, "dataset").eq.infoV[ii]@dataset))
          if (isvar("vClimAMIP")) then
            vClimAMIP = vClimAMIP + vClim * w
            nvClimAMIP = nvClimAMIP + w
          else
            vClimAMIP = vClim
            vClimAMIP = vClim * w
            nvClimAMIP = w
          end if
        end if
      end if
      
      delete(v)
      delete(vClim)
      delete(vMonsoon)
      if (isvar("ensemble")) then
        delete(ensemble)
      end if
    end do

    if (isvar("uClimCMIP5")) then
      uClimCMIP5 = uClimCMIP5 / nuClimCMIP5
      dataset = "MultiModelMeanCMIP5"

      uMonsoon = summer_minus_winter(uClimCMIP5, opt)
      
      uMonsoon@var = "u"
      uMonsoon@diag_script = DIAG_SCRIPT
      uMonsoon@dataset = dataset
      uMonsoon@syear_clim = syear_clim
      uMonsoon@eyear_clim = eyear_clim
      uMonsoon@smonth_NHsummer = smonth_NHsummer
      uMonsoon@emonth_NHsummer = emonth_NHsummer
      uMonsoon@smonth_SHsummer = smonth_SHsummer
      uMonsoon@emonth_SHsummer = emonth_SHsummer
      uMonsoon@long_name = "Climatological summer - winter zonal wind velocity"
      nc_filename = work_dir + "u_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(uMonsoon, nc_filename)

      delete(uClimCMIP5)
      delete(nuClimCMIP5)
      delete(uMonsoon)
    end if
    if (isvar("vClimCMIP5")) then
      vClimCMIP5 = vClimCMIP5 / nvClimCMIP5
      dataset = "MultiModelMeanCMIP5"

      vMonsoon = summer_minus_winter(vClimCMIP5, opt)

      vMonsoon@var = "v"
      vMonsoon@diag_script = DIAG_SCRIPT
      vMonsoon@dataset = dataset
      vMonsoon@long_name \
          = "Climatological summer - winter meridional wind velocity"
      vMonsoon@syear_clim = syear_clim
      vMonsoon@eyear_clim = eyear_clim
      vMonsoon@smonth_NHsummer = smonth_NHsummer
      vMonsoon@emonth_NHsummer = emonth_NHsummer
      vMonsoon@smonth_SHsummer = smonth_SHsummer
      vMonsoon@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "v_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(vMonsoon, nc_filename)

      delete(vClimCMIP5)
      delete(nvClimCMIP5)
      delete(vMonsoon)
    end if

    if (isvar("uClimCMIP6")) then
      uClimCMIP6 = uClimCMIP6 / nuClimCMIP6
      dataset = "MultiModelMeanCMIP6"

      uMonsoon = summer_minus_winter(uClimCMIP6, opt)

      uMonsoon@var = "u"
      uMonsoon@diag_script = DIAG_SCRIPT
      uMonsoon@dataset = dataset
      uMonsoon@long_name = "Climatological summer - winter zonal wind velocity"
      uMonsoon@syear_clim = syear_clim
      uMonsoon@eyear_clim = eyear_clim
      uMonsoon@smonth_NHsummer = smonth_NHsummer
      uMonsoon@emonth_NHsummer = emonth_NHsummer
      uMonsoon@smonth_SHsummer = smonth_SHsummer
      uMonsoon@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "u_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(uMonsoon, nc_filename)

      delete(uClimCMIP6)
      delete(nuClimCMIP6)
      delete(uMonsoon)
    end if
    if (isvar("vClimCMIP6")) then
      vClimCMIP6 = vClimCMIP6 / nvClimCMIP6
      dataset = "MultiModelMeanCMIP6"

      vMonsoon = summer_minus_winter(vClimCMIP6, opt)

      vMonsoon@var = "v"
      vMonsoon@diag_script = DIAG_SCRIPT
      vMonsoon@dataset = dataset
      vMonsoon@long_name \
          = "Climatological summer - winter meridional wind velocity"
      vMonsoon@syear_clim = syear_clim
      vMonsoon@eyear_clim = eyear_clim
      vMonsoon@smonth_NHsummer = smonth_NHsummer
      vMonsoon@emonth_NHsummer = emonth_NHsummer
      vMonsoon@smonth_SHsummer = smonth_SHsummer
      vMonsoon@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "v_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(vMonsoon, nc_filename)

      delete(vClimCMIP6)
      delete(nvClimCMIP6)
      delete(vMonsoon)
    end if

    if (isvar("uClimAMIP")) then
      uClimAMIP = uClimAMIP / nuClimAMIP
      dataset = "MultiModelMeanAMIP"

      uMonsoon = summer_minus_winter(uClimAMIP, opt)

      uMonsoon@var = "u"
      uMonsoon@diag_script = DIAG_SCRIPT
      uMonsoon@dataset = dataset
      uMonsoon@long_name = "Climatological summer - winter zonal wind velocity"
      uMonsoon@syear_clim = syear_clim
      uMonsoon@eyear_clim = eyear_clim
      uMonsoon@smonth_NHsummer = smonth_NHsummer
      uMonsoon@emonth_NHsummer = emonth_NHsummer
      uMonsoon@smonth_SHsummer = smonth_SHsummer
      uMonsoon@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "u_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(uMonsoon, nc_filename)

      delete(uClimAMIP)
      delete(nuClimAMIP)
      delete(uMonsoon)
    end if
    if (isvar("vClimAMIP")) then
      vClimAMIP = vClimAMIP / nvClimAMIP
      dataset = "MultiModelMeanAMIP"

      vMonsoon = summer_minus_winter(vClimAMIP, opt)

      vMonsoon@var = "v"
      vMonsoon@diag_script = DIAG_SCRIPT
      vMonsoon@dataset = dataset
      vMonsoon@long_name \
          = "Climatological summer - winter meridional wind velocity"
      vMonsoon@syear_clim = syear_clim
      vMonsoon@eyear_clim = eyear_clim
      vMonsoon@smonth_NHsummer = smonth_NHsummer
      vMonsoon@emonth_NHsummer = emonth_NHsummer
      vMonsoon@smonth_SHsummer = smonth_SHsummer
      vMonsoon@emonth_SHsummer = emonth_SHsummer
      nc_filename = work_dir + "v_monsoon_"+dataset+".nc"
      nc_outfile = ncdf_write(vMonsoon, nc_filename)

      delete(vClimAMIP)
      delete(nvClimAMIP)
      delete(vMonsoon)
    end if
  end if

; ====================== monsoon precipitation index ========================

  input_dir = config_user_info@work_dir + "/"

  if (any((/do_GMpr, do_NHMpr, do_SHMpr/))) then
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
        if (isatt(infoPr[ii], "ensemble")) then
          domainFile = input_dir+"domain_monsoon_"+infoPr[ii]@dataset+"_"+infoPr[ii]@ensemble+".nc"
        else
          domainFile = input_dir+"domain_monsoon_"+infoPr[ii]@dataset+".nc"
        end if
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

      if (do_GMpr) then
        GMpr = prSummer(:, 0, 0)
        GMpr = wgt_areaave(prSummer, wgty, 1., 0)

        delete(GMpr@lon)
        delete(GMpr@lat)
        clim = avg(GMpr({syear_clim:eyear_clim}))
        GMpr = GMpr - clim
        GMpr@long_name = "global summer monsoon precipitation anomaly"
        if (scale_monsoonPr) then
          GMpr = GMpr / clim
          GMpr@units = ""
          GMpr@long_name = "global summer monsoon precipitation anomaly scaled by climatology"
        end if

        GMpr@eyear_ref = eyear_clim
        GMpr@syear_ref = syear_clim
        GMpr@var = "GMpr"
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
        delete(GMpr)
        delete(clim)
      end if

      if (do_NHMpr) then
        NHMpr = prSummer(:, 0, 0)
        NHMpr = wgt_areaave(prSummer(:, {0.:90.}, :), wgty({0.:90.}), 1., 0)

        delete(NHMpr@lon)
        delete(NHMpr@lat)
        clim = avg(NHMpr({syear_clim:eyear_clim}))
        NHMpr = NHMpr - clim
        NHMpr@long_name = "NH summer monsoon precipitation anomaly"
        if (scale_monsoonPr) then
          NHMpr = NHMpr / clim
          NHMpr@units = ""
          NHMpr@long_name = "NH summer monsoon precipitation anomaly scaled by climatology"
        end if

        NHMpr@eyear_ref = eyear_clim
        NHMpr@syear_ref = syear_clim
        NHMpr@var = "NHMpr"
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
        delete(NHMpr)
        delete(clim)
      end if

      if (do_SHMpr) then
        SHMpr = prSummer(:, 0, 0)
        SHMpr = wgt_areaave(prSummer(:, {-90.:0.}, :), wgty({-90.:0.}), 1., 0)

        delete(SHMpr@lon)
        delete(SHMpr@lat)
        clim = avg(SHMpr({syear_clim:eyear_clim}))
        SHMpr = SHMpr - clim
        NHMpr@long_name = "NH summer monsoon precipitation anomaly"
        if (scale_monsoonPr) then
          SHMpr = SHMpr / clim
          SHMpr@units = ""
          SHMpr@long_name = "SH summer monsoon precipitation anomaly scaled by climatology"
        end if
        
        SHMpr@eyear_ref = eyear_clim
        SHMpr@syear_ref = syear_clim
        SHMpr@var = "SHMpr"
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
        delete(SHMpr)
      end if

      delete(time)
      delete(wgty)
      delete(prSummer)
    end do
  end if

; ======================= monsoon circulation index =========================

  if (do_NHMu) then
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
