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
;    20200914 kosaka_yu: revised to weighting based on the ensemble size.
;    20200528 kosaka_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/ar6ch3_hiatus/functions.ncl"
load "$diag_scripts/ar6ch3_hiatus/io_globalmean.ncl"

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
  blend_globalmean = True
  if (isatt(diag_script_info, "blend_globalmean")) then
    blend_globalmean = diag_script_info@blend_globalmean
  end if
  mask_globalmean = True
  if (isatt(diag_script_info, "mask_globalmean")) then
    mask_globalmean = diag_script_info@mask_globalmean
  end if
  composite_threshold = "5_percentle"  ; "5_percentile" of cmip5/cmip6
                                       ; "10_percentile" of cmip5/cmip6
                                       ; "obs_mean" of obs
                                       ; "obs_median" of obs
                                       ; "obs_max" of obs
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
    composite_dataset = str_lower(diag_script_info@composite_dataset)
  end if
  
; Configuration for composite pattern
  blend_pattern = True
  if (isatt(diag_script_info, "blend_pattern")) then
    blend_pattern = diag_script_info@blend_pattern
  end if
  mask_pattern = False
  if (isatt(diag_script_info, "mask_pattern")) then
    mask_pattern = diag_script_info@mask_pattern
  end if

; ======================================================================
;
;                       1. Read GSAT/GMST trend
;
; ======================================================================

  input_dir = diag_script_info@input_files+"/"
  if (blend_globalmean) then
    GST = "GMST"
  else
    GST = "GSAT"
  end if
  if (mask_globalmean) then
    masked = "masked"
  else
    masked = "unmasked"
  end if

  ifileCMIP5 = systemfunc("ls "+input_dir+"trend"+GST+"_"+masked+"_CMIP5.txt")
  ifileCMIP6 = systemfunc("ls "+input_dir+"trend"+GST+"_"+masked+"_CMIP6.txt")

  trendGSThiatusCMIP5 = read_GMtrend_as_list(ifileCMIP5)
  trendGSThiatusCMIP6 = read_GMtrend_as_list(ifileCMIP6)

; ======================================================================
;
;            2. Get weight as the inverse of the ensemble size
;
; ======================================================================

  post_trend_CMIP5 = metadata_att_as_array(trendGSThiatusCMIP5, "trend_post")
  do ii = 0, ListCount(trendGSThiatusCMIP5)-1
    esize_all = num(trendGSThiatusCMIP5[ii]@dataset.eq.metadata_att_as_array(trendGSThiatusCMIP5, "dataset"))
    trendGSThiatusCMIP5[ii]@weight_hiatus = 1./tofloat(esize_all)
    if (ismissing(post_trend_CMIP5(ii))) then
      trendGSThiatusCMIP5[ii]@weight_post = 0.
    else
      esize_post = num(trendGSThiatusCMIP5[ii]@dataset.eq.where(ismissing(post_trend_CMIP5), "", metadata_att_as_array(trendGSThiatusCMIP5, "dataset")))
      trendGSThiatusCMIP5[ii]@weight_post = 1./tofloat(esize_post)
    end if
  end do
  delete(post_trend_CMIP5)
  post_trend_CMIP6 = metadata_att_as_array(trendGSThiatusCMIP6, "trend_post")
  do ii = 0, ListCount(trendGSThiatusCMIP6)-1
    esize_all = num(trendGSThiatusCMIP6[ii]@dataset.eq.metadata_att_as_array(trendGSThiatusCMIP6, "dataset"))
    trendGSThiatusCMIP6[ii]@weight_hiatus = 1./tofloat(esize_all)
    if (ismissing(post_trend_CMIP6(ii))) then
      trendGSThiatusCMIP6[ii]@weight_post = 0.
    else
      esize_post = num(trendGSThiatusCMIP6[ii]@dataset.eq.where(ismissing(post_trend_CMIP6), "", metadata_att_as_array(trendGSThiatusCMIP6, "dataset")))
      trendGSThiatusCMIP6[ii]@weight_post = 1./tofloat(esize_post)
    end if
  end do
  delete(post_trend_CMIP6)

; ======================================================================
;
;         3. Determine composite threshold
;
; ======================================================================

  if (isStrSubset(composite_threshold, "_percentile")) then
    array1 = list_to_array(trendGSThiatusCMIP5)
    array2 = list_to_array(trendGSThiatusCMIP6)
    array3 = metadata_att_as_array(trendGSThiatusCMIP5, "weight_hiatus")
    array4 = metadata_att_as_array(trendGSThiatusCMIP6, "weight_hiatus")
    if (pecentile_dataset.eq."cmip5+cmip6") then
      tmp = array_append_record(array1, array2, 0)
      wgt = array_append_record(array3, array4, 0)
    elseif (pecentile_dataset.eq."cmip5") then
      tmp = array1
      wgt = array3
    elseif (pecentile_dataset.eq."cmip6") then
      tmp = array2
      wgt = array4
    else
      error_msg("f", DIAG_SCRIPT, "", "Unsupported: percentile_dataset = "+percentile_dataset)
    end if
    delete([/array1, array2, array3, array4/])
;    qsort(tmp)
    if (composite_threshold.eq."5_percentile") then
      factor = 0.05
    elseif (composite_threshold.eq."10_percentile") then
      factor = 0.1
    else
      error_msg("f", DIAG_SCRIPT, "", "Unsupported: composite_threshold = "+composite_threshold)
    end if
    threshold_trendGSThiatus = weighted_percentile(tmp, wgt, factor)
    delete(tmp)
    delete(wgt)
  elseif (any(composite_threshold.eq.(/"obs_mean", "obs_median", "obs_max"/))) then
    ifileOBS   = systemfunc("ls "+input_dir+"trendGMST_*_OBS.txt")
    tmp = new(dimsizes(ifileOBS), "float")
    do ii = 0, dimsizes(ifileOBS)-1
      trendGSThiatusOBS = read_GMtrend_as_list(ifileOBS)
      if (ListCount(trendGSThiatusOBS).eq.1) then
        tmp(ii) = trendGSThiatusOBS[0]
      else
        ens = metadata_att_as_array(trendGSThiatusOBS, "ensemble")
        if (trendGSThiatusOBS[0]@dataset.eq."HadCRUT5" .and. any(ens.eq."mean")) then
          j = ind(ens.eq."mean")
        elseif (any(ens.eq."median")) then
          j = ind(ens.eq."median")
        else
          error_msg("f", DIAG_SCRIPT, "", "No mean/median in "+trendGSThiatusOBS[0]@dataset)
        end if
        tmp(ii) = trendGSThiatusOBS[j]
        delete(ens)
      end if
      delete(trendGSThiatusOBS)
    end do
    if (composite_threshold.eq."obs_mean") then
      threshold_trendGSThiatus = avg(tmp)
    elseif (composite_threshold.eq."obs_median") then
      threshold_trendGSThiatus = dim_median(tmp)
    elseif (composite_threshold.eq."obs_max") then
      threshold_trendGSThiatus = max(tmp)
    end if
    delete(tmp)
  else
    if (isStrSubset(composite_threshold, "HadCRUT4")) then
      name = "HadCRUT4"
    elseif (isStrSubset(composite_threshold, "HadCRUT5")) then
      name = "HadCRUT5"
    elseif (isStrSubset(composite_threshold, "CowtanWay")) then
      name = "CowtanWay"
    else
      name = composite_threshold
    end if
    ifileOBS   = systemfunc("ls "+input_dir+"trendGMST_"+name+"_OBS.txt")
    delete(name)
    trendGSThiatusOBS = read_GMtrend_as_list(ifileOBS)
    if (ListCount(trendGSThiatusOBS).eq.1) then
      threshold_trendGSThiatus = trendGSThiatusOBS[0]
    else
      ens = metadata_att_as_array(trendGSThiatusOBS, "ensemble")
      array = list_to_array(trendGSThiatusOBS)
      array = where(is_string_numeric(str_sub_str(ens, "ens", "")), array, array@_FillValue)
      if (isStrSubset(composite_threshold, "mean")) then
        if (any(ens.eq."mean")) then
          j = ind(ens.eq."mean")
          threshold_trendGSThiatus = trendGSThiatusOBS[j]
        else
          threshold_trendGSThiatus = avg(array)
        end if
      elseif (isStrSubset(composite_threshold, "median")) then
        if (any(ens.eq."median")) then
          j = ind(ens.eq."median")
          threshold_trendGSThiatus = trendGSThiatusOBS[j]
        else
          threshold_trendGSThiatus = dim_median(array)
        end if
      elseif (isStrSubset(composite_threshold, "max")) then
        threshold_trendGSThiatus = max(array)
      else
        error_msg("f", DIAG_SCRIPT, "", "Unsupported: composite_threshold = "+composite_threshold)
      end if
      delete([/array, ens/])
    end if
  end if

; ======================================================================
;
;             4. Label ensemble members to be composited
;
; ======================================================================

  do ii = 0, ListCount(trendGSThiatusCMIP5)-1
    if (any(composite_dataset.eq.(/"cmip5", "cmip5+cmip6"/)) .and. \
        trendGSThiatusCMIP5[ii].lt.threshold_trendGSThiatus) then
      trendGSThiatusCMIP5[ii]@composite = True
    else
      trendGSThiatusCMIP5[ii]@composite = False
    end if
  end do

  do ii = 0, ListCount(trendGSThiatusCMIP6)-1
    if (any(composite_dataset.eq.(/"cmip6", "cmip5+cmip6"/)) .and. \
        trendGSThiatusCMIP6[ii].lt.threshold_trendGSThiatus) then
      trendGSThiatusCMIP6[ii]@composite = True
    else
      trendGSThiatusCMIP6[ii]@composite = False
    end if
  end do
    
; ======================================================================
;
;                     5. Composite trend patterns
;
; ======================================================================

  if (blend_pattern) then
    blended = "blended"
  else
    blended = "unblended"
  end if
  if (mask_pattern) then
    masked = "masked"
  else
    masked = "unmasked"
  end if

  nhiatusCMIP5 = 0
  npostCMIP5   = 0
  nhiatusCMIP6 = 0
  npostCMIP6   = 0
  
  if (any(composite_dataset.eq.(/"cmip5", "cmip5+cmip6"/))) then
    do ii = 0, ListCount(trendGSThiatusCMIP5)-1
      if (trendGSThiatusCMIP5[ii]@composite) then
        path = input_dir + "HiatusTrendPattern_" + blended + "_" + masked \
                + "_CMIP5_" + trendGSThiatusCMIP5[ii]@dataset + "_" \
                + trendGSThiatusCMIP5[ii]@ensemble + ".nc"
        if (fileexists(path)) then
          f = addfile(path, "r")
          trend = f->trend
          wgt = trendGSThiatusCMIP5[ii]@weight_hiatus
          if (.not.isdefined("compositeHiatusTrendPattern")) then
            compositeHiatusTrendPattern = trend*wgt
            copy_VarCoords(trend, compositeHiatusTrendPattern)
            ncompositeHiatusTrendPattern = where(ismissing(trend), 0, wgt)
            datasetCompositeHiatus = "CMIP5 "+trendGSThiatusCMIP5[ii]@dataset+" "+trendGSThiatusCMIP5[ii]@ensemble
            compositeHiatusTrendGST = (/trendGSThiatusCMIP5[ii]/)
          else
            compositeHiatusTrendPattern = compositeHiatusTrendPattern + trend*wgt
            ncompositeHiatusTrendPattern = ncompositeHiatusTrendPattern \
                                            + where(ismissing(trend), 0, wgt)
            datasetCompositeHiatus = datasetCompositeHiatus+", CMIP5 "+trendGSThiatusCMIP5[ii]@dataset+" "+trendGSThiatusCMIP5[ii]@ensemble
            compositeHiatusTrendGST := array_append_record(compositeHiatusTrendGST, (/trendGSThiatusCMIP5[ii]/), 0)
          end if
          delete(trend)
          nhiatusCMIP5 = nhiatusCMIP5 + 1
        end if
        path = input_dir + "PostTrendPattern_" + blended + "_" + "unmasked" \
                + "_CMIP5_" + trendGSThiatusCMIP5[ii]@dataset + "_" \
                + trendGSThiatusCMIP5[ii]@ensemble + ".nc"
        if (fileexists(path)) then
          f = addfile(path, "r")
          trend = f->trend
          wgt = trendGSThiatusCMIP5[ii]@weight_post
          if (.not.isdefined("compositePostTrendPattern")) then
            compositePostTrendPattern = trend*wgt
            copy_VarCoords(trend, compositePostTrendPattern)
            ncompositePostTrendPattern = where(ismissing(trend), 0, wgt)
            datasetCompositePost = "CMIP5 "+trendGSThiatusCMIP5[ii]@dataset+" "+trendGSThiatusCMIP5[ii]@ensemble
            compositePostTrendGST = (/trendGSThiatusCMIP5[ii]@trend_post/)
          else
            compositePostTrendPattern = compositePostTrendPattern + trend*wgt
            ncompositePostTrendPattern = ncompositePostTrendPattern \
                                            + where(ismissing(trend), 0, wgt)
            datasetCompositePost = datasetCompositePost+", CMIP5 "+trendGSThiatusCMIP5[ii]@dataset+" "+trendGSThiatusCMIP5[ii]@ensemble
            compositePostTrendGST := array_append_record(compositePostTrendGST, (/trendGSThiatusCMIP5[ii]@trend_post/), 0)
          end if
          delete(trend)
          npostCMIP5 = npostCMIP5 + 1
        end if
      end if
    end do
  end if

  if (any(composite_dataset.eq.(/"cmip6", "cmip5+cmip6"/))) then
    do ii = 0, ListCount(trendGSThiatusCMIP6)-1
      if (trendGSThiatusCMIP6[ii]@composite) then
        path = input_dir + "HiatusTrendPattern_" + blended + "_" + masked \
                + "_CMIP6_CMIP_" + trendGSThiatusCMIP6[ii]@dataset + "_" \
                + trendGSThiatusCMIP6[ii]@ensemble + ".nc"
        if (fileexists(path)) then
          f = addfile(path, "r")
          trend = f->trend
          wgt = trendGSThiatusCMIP6[ii]@weight_hiatus
          if (.not.isdefined("compositeHiatusTrendPattern")) then
            compositeHiatusTrendPattern = trend*wgt
            copy_VarCoords(trend, compositeHiatusTrendPattern)            
            ncompositeHiatusTrendPattern = where(ismissing(trend), 0, wgt)
            datasetCompositeHiatus = "CMIP6 "+trendGSThiatusCMIP6[ii]@dataset+" "+trendGSThiatusCMIP6[ii]@ensemble
            compositeHiatusTrendGST = (/trendGSThiatusCMIP6[ii]/)
          else
            compositeHiatusTrendPattern = compositeHiatusTrendPattern + trend*wgt
            ncompositeHiatusTrendPattern = ncompositeHiatusTrendPattern \
                                            + where(ismissing(trend), 0, wgt)
            datasetCompositeHiatus = datasetCompositeHiatus+", CMIP6 "+trendGSThiatusCMIP6[ii]@dataset+" "+trendGSThiatusCMIP6[ii]@ensemble
            compositeHiatusTrendGST := array_append_record(compositeHiatusTrendGST, (/trendGSThiatusCMIP6[ii]/), 0)
          end if
          delete(trend)
          nhiatusCMIP6 = nhiatusCMIP6 + 1
        end if
        path = input_dir + "PostTrendPattern_" + blended + "_" + "unmasked" \
                + "_CMIP6_CMIP_" + trendGSThiatusCMIP6[ii]@dataset + "_" \
                + trendGSThiatusCMIP6[ii]@ensemble + ".nc"
        if (fileexists(path)) then
          f = addfile(path, "r")
          trend = f->trend
          wgt = trendGSThiatusCMIP6[ii]@weight_post
          if (.not.isdefined("compositePostTrendPattern")) then
            compositePostTrendPattern = trend*wgt
            copy_VarCoords(trend, compositePostTrendPattern)
            ncompositePostTrendPattern = where(ismissing(trend), 0, wgt)
            datasetCompositePost = "CMIP6 "+trendGSThiatusCMIP6[ii]@dataset+" "+trendGSThiatusCMIP6[ii]@ensemble
            compositePostTrendGST = (/trendGSThiatusCMIP6[ii]@trend_post/)
          else
            compositePostTrendPattern = compositePostTrendPattern + trend*wgt
            ncompositePostTrendPattern = ncompositePostTrendPattern \
                                            + where(ismissing(trend), 0, wgt)
            datasetCompositePost = datasetCompositePost+", CMIP6 "+trendGSThiatusCMIP6[ii]@dataset+" "+trendGSThiatusCMIP6[ii]@ensemble
            compositePostTrendGST := array_append_record(compositePostTrendGST, (/trendGSThiatusCMIP6[ii]@trend_post/), 0)
          end if
          delete(trend)
          npostCMIP6 = npostCMIP6 + 1
        end if
      end if
    end do
  end if

; ======================================================================
;
;                         6. Output files
;
; ======================================================================

  if (composite_dataset.eq."cmip5+cmip6") then
    print("Hiatus     composite: "+nhiatusCMIP5+" CMIP5 + "+nhiatusCMIP6+" CMIP6 members")
    print("PostHiatus composite: "+npostCMIP5+" CMIP5 + "+npostCMIP6+" CMIP6 members")
  elseif (composite_dataset.eq."cmip5") then
    print("Hiatus     composite: "+nhiatusCMIP5+" CMIP5 members")
    print("PostHiatus composite: "+npostCMIP5+" CMIP5 members")
  elseif (composite_dataset.eq."cmip6") then
    print("Hiatus     composite: "+nhiatusCMIP6+" CMIP6 members")
    print("PostHiatus composite: "+npostCMIP6+" CMIP6 members")
  end if
  
  if (isdefined("compositeHiatusTrendPattern")) then
    ncompositeHiatusTrendPattern@_FillValue = default_fillvalue("integer")
    compositeHiatusTrendPattern = compositeHiatusTrendPattern \
                                   / where(ncompositeHiatusTrendPattern.gt.0, \
                                           ncompositeHiatusTrendPattern, \
                                           ncompositeHiatusTrendPattern@_FillValue)
    compositeHiatusTrendPattern@diag_script = DIAG_SCRIPT
    compositeHiatusTrendPattern@var = "trend"
    compositeHiatusTrendPattern@project = composite_dataset
    if (any(composite_dataset.eq.(/"cmip5", "cmip5+cmip6"/))) then
      compositeHiatusTrendPattern@nmembers_CMIP5 = nhiatusCMIP5
    end if
    if (any(composite_dataset.eq.(/"cmip6", "cmip5+cmip6"/))) then
      compositeHiatusTrendPattern@nmembers_CMIP6 = nhiatusCMIP6
    end if
    compositeHiatusTrendPattern@threshold_trendGSThiatus = threshold_trendGSThiatus
    compositeHiatusTrendPattern@composite_threshold = composite_threshold
    if (isStrSubset(composite_threshold, "_percentile")) then
      compositeHiatusTrendPattern@threshold_dataset = threshold_dataset
    end if
    compositeHiatusTrendPattern@blend_GST = blend_globalmean
    compositeHiatusTrendPattern@mask_GST = mask_globalmean
    compositeHiatusTrendPattern@blend_pattern = blend_pattern
    compositeHiatusTrendPattern@mask_pattern = mask_pattern
    compositeHiatusTrendPattern@dataset = datasetCompositeHiatus
    compositeHiatusTrendPattern@trend_GST = compositeHiatusTrendGST
    nc_filename = work_dir + "HiatusTrendPattern_composite.nc"
    nc_outfile = ncdf_write(compositeHiatusTrendPattern, nc_filename)
  end if

  if (isdefined("compositePostTrendPattern")) then
    ncompositePostTrendPattern@_FillValue = default_fillvalue("integer")
    compositePostTrendPattern = compositePostTrendPattern \
                                 / where(ncompositePostTrendPattern.gt.0, \
                                         ncompositePostTrendPattern, \
                                         ncompositePostTrendPattern@_FillValue)
    compositePostTrendPattern@diag_script = DIAG_SCRIPT
    compositePostTrendPattern@var = "trend"
    compositePostTrendPattern@project = composite_dataset
    if (any(composite_dataset.eq.(/"cmip5", "cmip5+cmip6"/))) then
      compositePostTrendPattern@nmembers_CMIP5 = npostCMIP5
    end if
    if (any(composite_dataset.eq.(/"cmip6", "cmip5+cmip6"/))) then
      compositePostTrendPattern@nmembers_CMIP6 = npostCMIP6
    end if
    compositePostTrendPattern@threshold_trendGSThiatus = threshold_trendGSThiatus
    compositePostTrendPattern@composite_threshold = composite_threshold
    if (isStrSubset(composite_threshold, "_percentile")) then
      compositePostTrendPattern@threshold_dataset = threshold_dataset
    end if
    compositePostTrendPattern@blend_GST = blend_globalmean
    compositePostTrendPattern@mask_GST = mask_globalmean
    compositePostTrendPattern@blend_pattern = blend_pattern
    compositePostTrendPattern@mask_pattern = mask_pattern
    compositePostTrendPattern@dataset = datasetCompositePost
    compositePostTrendPattern@trend_GST = compositePostTrendGST
    nc_filename = work_dir + "PostTrendPattern_composite.nc"
    nc_outfile = ncdf_write(compositePostTrendPattern, nc_filename)
  end if
  
end