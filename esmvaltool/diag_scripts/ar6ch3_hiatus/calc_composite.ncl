; calc_composite
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs: 
;
; History
;    20191102 kosaka_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/statistics.ncl"
;load "$diag_scripts/shared/plot/style.ncl"
;load "$diag_scripts/shared/plot/contour_maps.ncl"

;load "$diag_scripts/ar6ch3_hiatus/grid_util.ncl"
;load "$diag_scripts/ar6ch3_hiatus/time_util.ncl"

function check_info_consistency(info1, info2)
local consistency
begin
  consistency = True
  if (isatt(info1, "dataset")) then
    if (.not.isatt(info2, "dataset").or.info1@dataset.ne.info2@dataset) then
      consistency = False
    end if
  elseif (isatt(info2, "dataset")) then
    consistency = False
  end if
  if (isatt(info1, "ensemble")) then
    if (.not.isatt(info2, "ensemble").or.info1@ensemble.ne.info2@ensemble) then
      consistency = False
    end if
  elseif (isatt(info2, "ensemble")) then
    consistency = False
  end if
  return(consistency)
end

function get_filename(info, var)
local info, var, name, project, dataset, ensemble, experment
begin
  project    = info@project   ; obs/obs4mips/ana4mips/CMIP5/CMIP6
  dataset    = info@dataset   ; model/dataset name
  if (isatt(info, "exp")) then
    experiment = info@exp       ; historical-rcp45 etc.
  else
    experiment = ""
  end if
  if (isatt(info, "ensemble")) then
    ensemble   = info@ensemble  ; r*i*p*
  else
    ensemble   = ""
  end if
  name = var+"_"+project+"_"+dataset
  if (experiment.ne."") then
    name = name+"_"+experiment
  end if
  if (ensemble.ne."") then
    name = name+"_"+ensemble
  end if
  name = name+".nc"
  return(name)
end

begin

  enter_msg(DIAG_SCRIPT, "")

  work_dir = config_user_info@work_dir + "/"
  ; Create work dir
  system("mkdir -p " + work_dir)

; ======================================================================
;
;                           0. Configuration
;
; ======================================================================
  
; Threshold for compositing trend patterns  
  blending_GMST = True
  if (isatt(diag_script_info, "blending_GMST")) then
    blending_GMST = diag_script_info@blending_GMST
  end if
  masking_GMST = True
  if (isatt(diag_script_info, "masking_GMST")) then
    masking_GMST = diag_script_info@masking_GMST
  end if
  composite_threshold = "5_percentile"  ; "5_percentile" of cmip5/cmip6
                                        ; "10_percentile" of cmip5/cmip6
                                        ; "mean" of obs
                                        ; "median" of obs
                                        ; "max" of obs
                                        ; "specific" dataset
  if (isatt(diag_script_info, "composite_threshold")) then
    composite_threshold = diag_script_info@composite_threshold
  end if
  
  percentile_dataset = "cmip5+cmip6" ; if a percentile value; cmip5, cmip6 or cmip5+cmip6
  if (isatt(diag_script_info, "percentile_dataset")) then
    percentile_dataset = diag_script_info@percentile_dataset
  end if

  composite_dataset = "cmip5+cmip6" ; composited members; cmip5 or cmip6 of cmip5+cmip6
  if (isatt(diag_script_info, "composite_threshold")) then
    composite_dataset = diag_script_info@composite_dataset
  end if

; Configuration for composite pattern
  blending_pattern = True
  if (isatt(diag_script_info, "blending_pattern")) then
    blending_pattern = diag_script_info@blending_pattern
  end if
  masking_pattern = False
  if (isatt(diag_script_info, "masking_pattern")) then
    masking_pattern = diag_script_info@masking_pattern
  end if

  obs_pattern = "GISTEMP"
  if (isatt(diag_script_info, "obs_pattern")) then
    obs_pattern = diag_script_info@obs_pattern
  end if
  
; ======================================================================
;
;         1. Read GMST trend
;
; ======================================================================

  input_dir = diag_script_info@input_files+"/"
  if (blending_GMST) then
    blended = "blended"
  else
    blended = "unblended"
  end if
  if (masking_GMST) then
    masked = "masked"
  else
    masked = "unmasked"
  end if

  trendHiatusGMSTobs = NewList("fifo")
  
  files = systemfunc("ls "+input_dir+"trendGMST_"+blended+"_"+masked+"_hiatus_*.nc")
  do i = 0, dimsizes(files)-1
    f = addfile(files(i), "r")
    if (files(i).eq.input_dir+"trendGMST_"+blended+"_"+masked+"_hiatus_cmip6.nc") then
      trendHiatusGMSTcmip6 = f->trend
    elseif (files(i).eq.input_dir+"trendGMST_"+blended+"_"+masked+"_hiatus_cmip5.nc") then
      trendHiatusGMSTcmip5 = f->trend
    else
      tmp = f->trend
      ListAppend(trendHiatusGMSTobs, new(dimsizes(tmp), typeof(tmp)))
      i = ListCount(trendHiatusGMSTobs) - 1
      trendHiatusGMSTobs[i] = tmp
      delete(tmp)
    end if
  end do

; ======================================================================
;
;         2. Determine composite threshold
;
; ======================================================================

  if (isStrSubset(composite_threshold, "_percentile")) then
    if (pecentile_dataset.eq."cmip5+cmip6") then
      tmp = array_append_record(trendHiatusGMSTcmip5, trendHiatusGMSTcmip6, 0)
    elseif (pecentile_dataset.eq."cmip5") then
      tmp = trendHiatusGMSTcmip5
    elseif (pecentile_dataset.eq."cmip6") then
      tmp = trendHiatusGMSTcmip6
    else
      error_msg("f", DIAG_SCRIPT, "", "Unsupported: percentile_dataset = "+percentile_dataset)
    end if
    qsort(tmp)
    if (composite_threshold.eq."5_percentile") then
      factor = 0.05
    elseif (composite_threshold.eq."10_percentile") then
      factor = 0.1
    else
      error_msg("f", DIAG_SCRIPT, "", "Unsupported: composite_threshold = "+composite_threshold)
    end if
    jj = toint(tofloat(dimsizes(tmp))*0.05)
    thresholdHiatusGMSTtrend = (tmp(jj)+tmp(jj+1))/2.
    delete(tmp)
  elseif (any(composite_threshold.eq.(/"mean", "median", "max"/))) then
    tmp = new(ListCount(trendHiatusGMSTobs), "float")
    do i = 0, dimsizes(tmp)-1
      if (dimsizes(trendHiatusGMSTobs).gt.0) then
        tmp(i) = trendHiatusGMSTobs[i](0)
      else
        tmp(i) = trendHiatusGMSTobs[i]
      end if
    end do
    if (composite_threshold.eq."mean") then
      thresholdHiatusGMSTtrend = avg(tmp)
    elseif (composite_threshold.eq."median") then
      thresholdHiatusGMSTtrend = dim_median(tmp)
    elseif (composite_threshold.eq."max") then
      thresholdHiatusGMSTtrend = max(tmp)
    end if
    delete(tmp)
  else
    if (isStrSubset(composite_threshold, "HadCRUT4")) then
      name = "HadCRUT4"
    elseif (isStrSubset(composite_threshold, "CowtanWay")) then
      name = "CowtanWay"
    else
      name = composite_threshold
    end if
    do i = 0, ListCount(trendHiatusGMSTobs)-1
      if (trendHiatusGMSTobs[i]@dataset.eq.name) then
        j = i
        break
      end if
    end do
    delete(name)
    if (dimsizes(trendHiatusGMSTobs).gt.0) then
      if (isStrSubset(composite_threshold, "mean")) then
        thresholdHiatusGMSTtrend = avg(trendHiatusGMSTobs[j])
      elseif (isStrSubset(composite_threshold, "median")) then
        thresholdHiatusGMSTtrend = dim_median(trendHiatusGMSTobs[j])
      elseif (isStrSubset(composite_threshold, "max")) then
        thresholdHiatusGMSTtrend = max(trendHiatusGMSTobs[j])
      elseif (isStrSubset(composite_threshold, "percentile")) then
        error_msg("f", DIAG_SCRIPT, "", "Unsupported: composite_threshold = "+composite_threshold)
      else
        thresholdHiatusGMSTtrend = dim_median(trendHiatusGMSTobs[j])
      end if
    else
      thresholdHiatusGMSTtrend = trendHiatusGMSTobs[j]
    end if
  end if

; ======================================================================
;
;         4. Label ensemble members that are composited
;
; ======================================================================

  if (isvar("trendHiatusGMSTcmip6")) then
;    printVarSummary(trendHiatusGMSTcmip6)
;    printVarSummary(thresholdHiatusGMSTtrend)
    do_composite_cmip6 = trendHiatusGMSTcmip6.lt.thresholdHiatusGMSTtrend
    trendHiatusGMSTcmip6@do_composite = do_composite_cmip6
    trendHiatusGMSTcmip6@composite_threshold = thresholdHiatusGMSTtrend
    trendHiatusGMSTcmip6@diag_script = DIAG_SCRIPT
    trendHiatusGMSTcmip6@var = "trend"
    nc_filename = work_dir + "trendGMST_hiatus_cmip6.nc"
    nc_outfile = ncdf_write(trendHiatusGMSTcmip6, nc_filename)

    path = input_dir+"trendGMST_"+blended+"_unmasked_post_cmip6.nc"
    if (fileexists(path)) then
      f = addfile(path, "r")
      trendPostGMSTcmip6 = f->trend
      trendPostGMSTcmip6@do_composite = do_composite_cmip6
      trendPostGMSTcmip6@diag_script = DIAG_SCRIPT
      trendPostGMSTcmip6@var = "trend"
      nc_filename = work_dir + "trendGMST_post_cmip6.nc"
      nc_outfile = ncdf_write(trendPostGMSTcmip6, nc_filename)
    end if
  end if
  if (isvar("trendHiatusGMSTcmip5")) then
    do_composite_cmip5 = trendHiatusGMSTcmip5.lt.thresholdHiatusGMSTtrend
    trendHiatusGMSTcmip5@do_composite = do_composite_cmip5
    trendHiatusGMSTcmip5@composite_threshold = thresholdHiatusGMSTtrend
    trendHiatusGMSTcmip5@diag_script = DIAG_SCRIPT
    trendHiatusGMSTcmip5@var = "trend"
    nc_filename = work_dir + "trendGMST_hiatus_cmip5.nc"
    nc_outfile = ncdf_write(trendHiatusGMSTcmip5, nc_filename)

    path = input_dir+"trendGMST_"+blended+"_unmasked_post_cmip5.nc"
    if (fileexists(path)) then
      f = addfile(path, "r")
      trendPostGMSTcmip5 = f->trend
      trendPostGMSTcmip5@do_composite = do_composite_cmip5
      trendPostGMSTcmip5@diag_script = DIAG_SCRIPT
      trendPostGMSTcmip5@var = "trend"
      nc_filename = work_dir + "trendGMST_post_cmip5.nc"
      nc_outfile = ncdf_write(trendPostGMSTcmip5, nc_filename)
    end if
  end if

  ; copy obs trends
  do ii = 0, ListCount(trendHiatusGMSTobs)-1
    trendHiatusGMSTobs[ii]@diag_script = DIAG_SCRIPT
    trendHiatusGMSTobs[ii]@var = "trend"
    nc_filename = work_dir + "trendGMST_hiatus_"+trendHiatusGMSTobs[ii]@dataset+".nc"
    nc_outfile = ncdf_write(trendHiatusGMSTobs[ii], nc_filename)
  end do
  
; ======================================================================
;
;                  5. Calculate Ts trend composites
;
; ======================================================================

  if (blending_pattern) then
    blended = "blended"
  else
    blended = "unblended"
  end if
  if (masking_pattern) then
    masked = "masked"
  else
    masked = "unmasked"
  end if

; 5.1 Compositing trend patterns for the hiatus period
  
  path = input_dir+"trendPattern_"+blended+"_"+masked+"_hiatus_cmip6.nc"
  if (fileexists(path)) then
    f = addfile(path, "r")
    trendHiatusTScmip6 = f->trend
  end if
  path = input_dir+"trendPattern_"+blended+"_"+masked+"_hiatus_cmip5.nc"
  if (fileexists(path)) then
    f = addfile(path, "r")
    trendHiatusTScmip5 = f->trend
  end if

  if (any(composite_dataset.eq.(/"cmip5+cmip6", "CMIP5+CMIP6", "cmip6+cmip5", "CMIP6+CMIP5"/))) then
    dims5 = dimsizes(trendHiatusTScmip5)
    dims6 = dimsizes(trendHiatusTScmip6)
    trendHiatusTS = new((/dims5(0)+dims6(0), dims6(1), dims6(2)/), typeof(trendHiatusTScmip6))
    do i = 0, dims5(0)-1
      trendHiatusTS(i, :, :) = trendHiatusTScmip5
    end do
    do i = 0, dims6(0)-1
      trendHiatusTS(dims5(0)+i, :, :) = trendHiatusTScmip6
    end do
    do_composite = array_append_record(do_composite_cmip5, do_composite_cmip6, 0)
    datasets = new(dimsizes(do_composite), "string")
    datasets(0:dims5(0)-1) = "cmip5"
    datasets(dims5(0):)    = "cmip6"  
    delete(dims5)
    delete(dims6)
  elseif (any(composite_dataset.eq.(/"cmip6", "CMIP6"/))) then
    trendHiatusTS = trendHiatusTScmip6
    do_composite = do_composite_cmip6
    datasets = new(dimsizes(do_composite), "string")
    datasets = "cmip6"  
  elseif (any(composite_dataset.eq.(/"cmip5", "CMIP5"/))) then
    trendHiatusTS = trendHiatusTScmip5
    do_composite = do_composite_cmip5
    datasets = new(dimsizes(do_composite), "string")
    datasets = "cmip5"
  end if

  nmax = dimsizes(do_composite)      
  trendCompositeHiatus = trendHiatusTS(0, :, :)
  trendCompositeHiatus = 0.
  ncomposite = 0
  ncomposite_cmip5 = 0
  ncomposite_cmip6 = 0
  do n = 0, nmax-1
    if (do_composite(n) .and. .not.all(ismissing(trendHiatusTS(n, :, :)))) then
      trendCompositeHiatus = trendCompositeHiatus + trendHiatusTS(n, :, :)
      ncomposite = ncomposite + 1
      if (datasets(n).eq."cmip5") then
        ncomposite_cmip5 = ncomposite_cmip5 + 1
      elseif (datasets(n).eq."cmip6") then
        ncomposite_cmip6 = ncomposite_cmip6 + 1
      end if
    end if
  end do
  trendCompositeHiatus = trendCompositeHiatus / ncomposite
  trendCompositeHiatus@ncomposite = ncomposite
  trendCompositeHiatus@ncomposite_cmip6 = ncomposite_cmip6
  trendCompositeHiatus@ncomposite_cmip5 = ncomposite_cmip5

  trendCompositeHiatus@diag_script = DIAG_SCRIPT
  trendCompositeHiatus@var = "trend"
  nc_filename = work_dir + "trendPattern_hiatus_composite.nc"
  nc_outfile = ncdf_write(trendCompositeHiatus, nc_filename)

; 5.2 Compositing trend patterns for the post-hiatus period
  path = input_dir+"trendPattern_"+blended+"_unmasked_post_cmip6.nc" ; Always unmasked for post-hiatus
  if (fileexists(path)) then
    f = addfile(path, "r")
    trendPostTScmip6 = f->trend
  end if
  path = input_dir+"trendPattern_"+blended+"_unmasked_post_cmip5.nc"
  if (fileexists(path)) then
    f = addfile(path, "r")
    trendPostTScmip5 = f->trend
  end if

  if (any(composite_dataset.eq.(/"cmip5+cmip6", "CMIP5+CMIP6", "cmip6+cmip5", "CMIP6+CMIP5"/))) then
    trendPostTS = trendHiatusTS
    do i = 0, dims5(0)-1
      trendPostTS(i, :, :) = trendPostTScmip5
    end do
    do i = 0, dims6(0)-1
      trendPostTS(dims5(0)+i, :, :) = trendPostTScmip6
    end do
  elseif (any(composite_dataset.eq.(/"cmip6", "CMIP6"/))) then
    trendPostTS = trendPostTScmip6
  elseif (any(composite_dataset.eq.(/"cmip5", "CMIP5"/))) then
    trendPostTS = trendHiatusTScmip5
  end if

  trendCompositePost = trendPostTS(0, :, :)
  trendCompositePost = 0.
  ncomposite = 0
  ncomposite_cmip5 = 0
  ncomposite_cmip6 = 0
  do n = 0, nmax-1
    if (do_composite(n) .and. .not.all(ismissing(trendPostTS(n, :, :)))) then
      trendCompositePost = trendCompositePost + trendPostTS(n, :, :)
      ncomposite = ncomposite + 1
      if (datasets(n).eq."cmip5") then
        ncomposite_cmip5 = ncomposite_cmip5 + 1
      elseif (datasets(n).eq."cmip6") then
        ncomposite_cmip6 = ncomposite_cmip6 + 1
      end if
    end if
  end do
  if (ncomposite.gt.1) then
    trendCompositePost = trendCompositePost / ncomposite
  else
    trendCompositePost = trendCompositePost@_FillValue
  end if
  trendCompositePost@ncomposite = ncomposite
  trendCompositePost@ncomposite_cmip6 = ncomposite_cmip6
  trendCompositePost@ncomposite_cmip5 = ncomposite_cmip5

  trendCompositePost@diag_script = DIAG_SCRIPT
  trendCompositePost@var = "trend"
  nc_filename = work_dir + "trendPattern_post_composite.nc"
  nc_outfile = ncdf_write(trendCompositePost, nc_filename)
  
; 5.3 Copy obs trend pattern
  
  path = input_dir+"trendPattern_"+blended+"_"+masked+"_hiatus_"+obs_pattern+".nc"
  f = addfile(path, "r")
  trendHiatusTSobs = f->trend
  if (dimsizes(dimsizes(trendHiatusTSobs)).eq.3) then
    tmp = trendHiatusTSobs(0, :, :)
    delete(trendHiatusTSobs)
    trendHiatusTSobs = tmp
    delete(tmp)
  end if
  print("obs_pattern = "+obs_pattern)
  printVarSummary(trendHiatusTSobs)
    
  trendHiatusTSobs@diag_script = DIAG_SCRIPT
  trendHiatusTSobs@var = "trend"
  nc_filename = work_dir + "trendPattern_hiatus_obs.nc"
  nc_outfile = ncdf_write(trendHiatusTSobs, nc_filename)

end
