; trend
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs: 
;
; History
;    20210312 kosaka_yu: implemented error fixer of temperature unit
;    20210227 kosaka_yu: added provenance.
;    20210203 kosaka_yu: updated for the new recipe structure
;    20201109 kosaka_yu: added Kadow and NOAAGlobalTemp-Interim
;    20200917 kosaka_yu: added statistical significance to the trend maps
;    20200525 kosaka_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"
load "$diag_scripts/shared/statistics.ncl"

load "$diag_scripts/ar6ch3_hiatus/functions.ncl"
load "$diag_scripts/ar6ch3_hiatus/io_globalmean.ncl"

function get_info_match(infoRef, infoList, atts, opt)
local info, i, j, match, unique, tmp
begin
  unique = "strict"  ; strict: Allow only single match (error if multiple matches)
                     ; any: If multiple matches then return only one
                     ; no: Return all matches
  if (opt) then
    if (isatt(opt, "unique")) then
      unique = opt@unique
    end if
  end if
  info = False
  if (ListCount(infoList).le.0) then
    return(info)
  end if
  if (.not.all(isatt(infoRef, atts))) then
    return(info)
  end if
  do i = 0, ListCount(infoList)-1
    if (all(isatt(infoList[i], atts))) then
      match = True
      do j = 0, dimsizes(atts)-1
        match = match .and. (infoRef@$atts(j)$ .eq. infoList[i]@$atts(j)$)
      end do
      if (match) then
        if (info) then
          if (unique.eq."strict") then
            print("get_info_match: ERROR - unique = strict but multiple matches. STOP")
            print("Below is the reference info.")
            printVarSummary(infoRef)
            error_msg("f", DIAG_SCRIPT, "get_info_match", "")
          elseif (unique.eq."no") then
            if (isvar("info")) then  ; Correct? Or True even if info is a list?
              tmp = info
              delete(info)
              info = NewList("fifo")
              ListAppend(info, new(1, "logical"))
              info[0] = tmp
              delete(tmp)
            end if
            ListAppend(info, new(1, "logical"))
            info[ListCount(info)-1] = infoList[i]
          end if
        else
          info = infoList[i]
          if (unique.eq."any") then
            return(info)
          end if
        end if
      end if 
    end if
  end do
  return(info)
end

function concatenate_data(array1, array2)
local array, dims1, dims2, dims
begin
  dims1 = dimsizes(array1)
  dims2 = dimsizes(array2)
  dims  = dims1
  dims(0) = dims1(0) + dims2(0)
  array = new(dims, typeof(array1))
  if (dimsizes(dims1).eq.1 .and. dimsizes(dims2).eq.1) then
    array(0:dims1(0)-1) = array1
    array(dims1(0):dims(0)-1) = (/array2/)
    array&$array!0$ = array_append_record(array1&$array1!0$, array2&$array2!0$, 0)
  elseif (dimsizes(dims1).eq.3 .and. dimsizes(dims2).eq.3) then
    if (any(dims1(1:2).ne.dims2(1:2))) then
      error_msg("f", DIAG_SCRIPT, "concatenate_data", "Different array shapes.")
    end if
    array(0:dims1(0)-1, :, :) = array1
    array(dims1(0):dims(0)-1, :, :) = (/array2/)
    array&$array!0$ = array_append_record(array1&$array1!0$, array2&$array2!0$, 0)
  elseif (dimsizes(dims1).eq.4 .and. dimsizes(dims2).eq.4) then
    if (any(dims1(1:3).ne.dims2(1:3))) then
      error_msg("f", DIAG_SCRIPT, "concatenate_data", "Different array shapes.")
    end if
    array(0:dims1(0)-1, :, :, :) = array1
    array(dims1(0):dims(0)-1, :, :, :) = (/array2/)
    array&$array!0$ = array_append_record(array1&$array1!0$, array2&$array2!0$, 0)
  else
    error_msg("f", DIAG_SCRIPT, "concatenate_data", "Unsupported array shapes.")
  end if
  return(array)
end

function set_output_tail(info)
local tail
begin
  tail = info@alias
  if (any(info@dataset.eq.(/"HadCRUT4", "HadCRUT5", "CowtanWay"/))) then
    tail = tail + "_" + info@type
  end if
  return(tail)
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

; 0.1 Configuration of blending and global averaging
  
  method_GMST = "Cowtan2015" ; "HadCRUT4", "GISTEMP", "NOAAGlobalTemp", "BerkeleyEarth", "global", "GSAT"
  if (isatt(diag_script_info, "method_GMST")) then
    method_GMST = diag_script_info@method_GMST
  end if

  do_anomaly = True
  do_blend = True
  do_mask = True
  if (any(method_GMST.eq.(/"Cowtan2015", "HadCRUT4", "HadCRUT5", "GISTEMP", "NOAAGlobalTemp", "BerkeleyEarth"/))) then
    do_anomaly  = True
    do_blend = True
    do_mask  = True
    if (any(method_GMST.eq.(/"Cowtan2015", "HadCRUT4"/))) then
      masking_dataset_GMST = "HadCRUT4"
      syear_clim = 1961
      eyear_clim = 1990
    elseif (method_GMST.eq."HadCRUT5") then
      masking_dataset_GMST = "HadCRUT5"
      syear_clim = 1961
      eyear_clim = 1990
    elseif (method_GMST.eq."GISTEMP") then
      masking_dataset_GMST = "GISTEMP"
      syear_clim = 1951
      eyear_clim = 1980
    elseif (method_GMST.eq."NOAAGlobalTemp") then
      masking_dataset_GMST = "NOAAGlobalTemp"
      syear_clim = 1971
      eyear_clim = 2000
    elseif (method_GMST.eq."BerkeleyEarth") then
      syear_clim = 1951
      eyear_clim = 1980      
    end if
  elseif (method_GMST.eq."global") then
    do_anomaly  = True
    do_blend = True
    do_mask  = False
    syear_clim = 1961
    eyear_clim = 1990
  elseif (method_GMST.eq."GSAT") then
    do_anomaly  = True
    do_blend = False
    do_mask  = False
    syear_clim = 1961
    eyear_clim = 1990
  end if

  if (isatt(diag_script_info, "mask_ref_dataset_GMST")) then
    masking_dataset_GMST = diag_script_info@mask_ref_dataset_GMST
  end if
  
  masking_dataset_pattern = masking_dataset_GMST
  if (isatt(diag_script_info, "mask_ref_dataset_pattern")) then
    masking_dataset_pattern = diag_script_info@mask_ref_dataset_pattern
  end if

; 0.2 Configuration of trend
  
  syear_hiatus = 1998
  eyear_hiatus = 2013
  if (isatt(diag_script_info, "syear_hiatus")) then  ; hiatus period
    syear_hiatus = diag_script_info@syear_hiatus
  end if
  if (isatt(diag_script_info, "eyear_hiatus")) then
    eyear_hiatus = diag_script_info@eyear_hiatus
  end if

  do_post = True
  if (isatt(diag_script_info, "do_post")) then  ; post-hiatus period
    do_post = diag_script_info@do_post
  end if

  syear_post = 2013
  eyear_post = 2028
  if (isatt(diag_script_info, "syear_post")) then  ; post-hiatus period
    syear_post = diag_script_info@syear_post
  end if
  if (isatt(diag_script_info, "eyear_post")) then
    eyear_post = diag_script_info@eyear_post
  end if

  annual = True
  if (isatt(diag_script_info, "annual")) then  ; annual mean for trend?
    annual = diag_script_info@annual
  end if
  
  smonth_ann = 1
  emonth_ann = 12
  if (isatt(diag_script_info, "smonth_ann")) then  ; if annual mean, from which months to which?
    smonth_ann = diag_script_info@smonth_ann
  end if
  if (isatt(diag_script_info, "emonth_ann")) then
    emonth_ann = diag_script_info@emonth_ann
  end if

  syear_deseasonalize = syear_clim
  eyear_deseasonalize = eyear_clim
  if (.not.annual .and. .not.anomaly) then
    error_msg("f", DIAG_SCRIPT, "", "monthly trend must be based on anomaly.")    
  end if
  
  method_trend = "least_square"  ; "least_square" or "sen"
  if (isatt(diag_script_info, "method_trend")) then
    method_trend = diag_script_info@method_trend
  end if

; 0.3 Output trend of reference or other obs data for masking?
  output_ref_trend = True
  if (isatt(diag_script_info, "output_ref_trend")) then
    output_ref_trend = diag_script_info@output_ref_trend
  end if
  output_obs_trend = True
  if (isatt(diag_script_info, "output_obs_trend")) then
    output_obs_trend = diag_script_info@output_obs_trend
  end if
  
; 0.4 Output trend pattern?
  do_pattern = True
  if (isatt(diag_script_info, "do_pattern")) then
    do_pattern = diag_script_info@do_pattern
  end if
  
; 0.5 Parameters
  load "$diag_scripts/ar6ch3_hiatus/config_datasets.ncl"

; =========================================================================
;
;         1. Get dataset lists and prepare for blending and masking
;
; =========================================================================

; 1.1 Get list of files  
  varList = metadata_att_as_array(input_file_info, "short_name")
  varList := get_unique_values(varList)
  
;  if (all(varList.ne."tas")) then
;    error_msg("f", DIAG_SCRIPT, "", "variable tas is missing.")
;  end if

  var = "tas"
  if (any(var.eq.varList)) then
    infoTas = select_metadata_by_name(input_file_info, var)
    dataset_names_tas = metadata_att_as_array(infoTas, "dataset")
  else
    infoTas = NewList("fifo")
  end if
  
  var = "tasa"
  if (any(var.eq.varList)) then
    infoTasa = select_metadata_by_name(input_file_info, var)
    dataset_names_tasa = metadata_att_as_array(infoTasa, "dataset")
  else
    infoTasa = NewList("fifo")
  end if
  
  var = "tasaga"
  if (any(var.eq.varList)) then
    infoTasaga = select_metadata_by_name(input_file_info, var)
    dataset_names_tasaga = metadata_att_as_array(infoTasaga, "dataset")
  else
    infoTasaga = NewList("fifo")
  end if

  if (do_blend) then
    if (all(varList.ne."tos")) then
      error_msg("f", DIAG_SCRIPT, "", "variable tos is missing.")
    end if
    infoTos = select_metadata_by_name(input_file_info, "tos")
    
    if (all(varList.ne."sic") .and. all(varList.ne."siconc") .and. all(varList.ne."siconca")) then
      error_msg("f", DIAG_SCRIPT, "", "variable sic/siconc/siconca is missing.")
    end if

    if (any(varList.eq."sic")) then
      infoSic = select_metadata_by_name(input_file_info, "sic")
    else
      infoSic = NewList("fifo")
    end if
    if (any(varList.eq."siconc")) then
      infoSiconc = select_metadata_by_name(input_file_info, "siconc")
    else
      infoSiconc = NewList("fifo")
    end if
    if (any(varList.eq."siconca")) then
      infoSiconca = select_metadata_by_name(input_file_info, "siconca")
    else
      infoSiconca = NewList("fifo")
    end if

    if (all(varList.ne."sftlf") .and. all(varList.ne."sftof")) then
      error_msg("f", DIAG_SCRIPT, "", "variable sftlf/sftof is missing.")
    end if
    
    if (any(varList.eq."sftlf")) then
      infoSftlf = select_metadata_by_name(input_file_info, "sftlf")
    end if
    if (any(varList.eq."sftof")) then
      infoSftof = select_metadata_by_name(input_file_info, "sftof")
    end if
  end if

  atts = True
  atts@project = "OBS"
  infoObs = select_metadata_by_atts(input_file_info, atts)
  atts@project = "OBS6"
  obs_tmp = select_metadata_by_atts(input_file_info, atts)
  do iobs = 0, ListCount(obs_tmp) - 1
    ListAppend(infoObs, obs_tmp[iobs])
  end do
  delete(obs_tmp)
  atts@project = "obs"
  obs_tmp = select_metadata_by_atts(input_file_info, atts)
  do iobs = 0, ListCount(obs_tmp) - 1
    ListAppend(infoObs, obs_tmp[iobs])
  end do
  delete(obs_tmp)
  atts@project = "obs4mips"
  obs_tmp = select_metadata_by_atts(input_file_info, atts)
  do iobs = 0, ListCount(obs_tmp) - 1
    ListAppend(infoObs, obs_tmp[iobs])
  end do
  delete(obs_tmp)
  atts@project = "ana4mips"
  obs_tmp = select_metadata_by_atts(input_file_info, atts)
  do iobs = 0, ListCount(obs_tmp) - 1
    ListAppend(infoObs, obs_tmp[iobs])
  end do
  delete(obs_tmp)
  obs_datasets = metadata_att_as_array(infoObs, "dataset")
  delete(infoObs)
  delete(atts)
  
; 1.2 Get reference data for masking  
  if (do_mask) then
  ; Find reference data (e.g. HadCRUT4 median) for masking in GMST calculation
    opt = True
    opt@unique = "strict"
    infoDummy = True
    infoDummy@dataset = masking_dataset_GMST
    if (any(masking_dataset_GMST.eq.(/"HadCRUT4", "CowtanWay"/))) then
      infoDummy@type = "median"
    elseif (masking_dataset_GMST.eq."HadCRUT5") then
      infoDummy@type = "mean"
    end if
    if (isatt(infoDummy, "type")) then
      atts = (/"dataset", "type"/)
    else
      atts = (/"dataset"/)
    end if
    if (any(dataset_names_tas.eq.masking_dataset_GMST)) then
      info = get_info_match(infoDummy, infoTas, atts, opt)
    elseif (any(dataset_names_tasa.eq.masking_dataset_GMST)) then
      info = get_info_match(infoDummy, infoTasa, atts, opt)
    end if
    if (.not.info .and. masking_dataset_GMST.eq."HadCRUT5") then
      infoDummy@type = "median"
      if (any(dataset_names_tas.eq.masking_dataset_GMST)) then
        info = get_info_match(infoDummy, infoTas, atts, opt)
      elseif (any(dataset_names_tasa.eq.masking_dataset_GMST)) then
        info = get_info_match(infoDummy, infoTasa, atts, opt)
      end if
    end if
    if (.not.info) then
      error_msg("f", DIAG_SCRIPT, "", "masking_dataset_GMST = "+masking_dataset_GMST+" not found.")
    end if
    maskTsGlobalmean = read_data(info)
    maskTsGlobalmean@filename = info@filename
    delete([/infoDummy, atts, info/])
    maskTsGlobalmean&time = cd_calendar(maskTsGlobalmean&time, -1)
    do_maskGlobalmean_post = False
    if (do_post) then
      if (maskTsGlobalmean&time(dimsizes(maskTsGlobalmean&time)-1).ge.eyear_post*100+emonth_ann) then
        do_maskGlobalmean_post = True
      end if
    end if
    
  ; Find reference data (e.g. HadCRUT4 median) for masking in patterned trend calculation
    infoDummy = True
    infoDummy@dataset = masking_dataset_pattern
    if (any(masking_dataset_pattern.eq.(/"HadCRUT4", "CowtanWay"/))) then
      infoDummy@type = "median"
    elseif (masking_dataset_pattern.eq."HadCRUT5") then
      infoDummy@type = "mean"
    end if
    if (isatt(infoDummy, "type")) then
      atts = (/"dataset", "type"/)
    else
      atts = (/"dataset"/)
    end if
    if (any(dataset_names_tas.eq.masking_dataset_pattern)) then
      info = get_info_match(infoDummy, infoTas, atts, opt)
    elseif (any(dataset_names_tasa.eq.masking_dataset_pattern)) then
      info = get_info_match(infoDummy, infoTasa, atts, opt)
    end if
    if (.not.info .and. masking_dataset_pattern.eq."HadCRUT5") then
      infoDummy@type = "median"
      if (any(dataset_names_tas.eq.masking_dataset_pattern)) then
        info = get_info_match(infoDummy, infoTas, atts, opt)
      elseif (any(dataset_names_tasa.eq.masking_dataset_pattern)) then
        info = get_info_match(infoDummy, infoTasa, atts, opt)
      end if
    end if
    if (.not.info) then
      error_msg("f", DIAG_SCRIPT, "", "masking_dataset_pattern = "+masking_dataset_pattern+" not found.")
    end if
    maskTsPattern = read_data(info)
    maskTsPattern@filename = info@filename
    delete([/infoDummy, atts, info, opt/])
    maskTsPattern&time = cd_calendar(maskTsPattern&time, -1)
    do_maskPattern_post = False
    if (do_post) then
      if (maskTsPattern&time(dimsizes(maskTsPattern&time)-1).ge.eyear_post*100+emonth_ann) then
        do_maskPattern_post = True
      end if
    end if
  else
    do_maskGlobalmean_post = False
  end if

; 1.3 Prepare outputs of GSAT/GMST trends
  if (isvar("dataset_names_tas")) then
    dataset_names_tmp = dataset_names_tas
  end if
  if (isvar("dataset_names_tasa")) then
    if (isvar("dataset_names_tmp")) then
      dataset_names_tmp := array_append_record(dataset_names_tmp, dataset_names_tasa, 0)
    else
      dataset_names_tmp = dataset_names_tasa
    end if
  end if
  if (isvar("dataset_names_tasaga")) then
    if (isvar("dataset_names_tmp")) then
      dataset_names_tmp := array_append_record(dataset_names_tmp, dataset_names_tasaga, 0)
    else
      dataset_names_tmp = dataset_names_tasaga
    end if
  end if
  datasets_tmp = get_unique_values(dataset_names_tmp)
  delete(dataset_names_tmp)
  if (.not.output_ref_trend) then
    idataset_ex = ind(datasets_tmp.ne.masking_dataset_GMST.and.datasets_tmp.ne.masking_dataset_pattern)
    datasets_tmptmp = datasets_tmp(idataset_ex)
    delete(datasets_tmp)
    datasets_tmp = datasets_tmptmp
    delete([/datasets_tmptmp, idataset_ex/])
  end if
  if (.not.output_obs_trend) then
    do k = 0, dimsizes(obs_datasets)-1
      idataset_ex = ind(datasets_tmp.ne.obs_datasets(k))
      datasets_tmptmp = datasets_tmp(idataset_ex)
      delete(datasets_tmp)
      datasets_tmp = datasets_tmptmp
      delete([/datasets_tmptmp, idataset_ex/])
    end do
  end if
  
  infoDummy = True
  opt = True
  opt@unique = "any"
  do k = 0, dimsizes(datasets_tmp)-1
    infoDummy@dataset = datasets_tmp(k)
    info = get_info_match(infoDummy, infoTas, (/"dataset"/), opt)
    if (.not.info) then
      info = get_info_match(infoDummy, infoTasa, (/"dataset"/), opt)
    end if
    if (.not.info) then
      info = get_info_match(infoDummy, infoTasaga, (/"dataset"/), opt)
    end if
    if (any(info@project.eq.(/"CMIP5", "CMIP6"/))) then
      datasets_tmp(k) = info@project
    end if
    delete(info)
  end do
  datasets = get_unique_values(datasets_tmp)
  delete([/datasets_tmp, infoDummy, opt/])
  GMList = make_GMList(work_dir, do_post, do_maskGlobalmean_post, do_blend, do_mask, \
                       syear_hiatus, eyear_hiatus, syear_post, eyear_post, datasets)

; ========================================================================
;
;               Loop for each (historical) SAT/SATa/GMST data
;
; ========================================================================
  do ii = 0, ListCount(infoTas) + ListCount(infoTasa) + ListCount(infoTasaga) - 1

    if (isdefined("infoSAT")) then
      delete(infoSAT)
    end if
      
    if (ii .lt. ListCount(infoTas)) then
      infoSAT = infoTas[ii]
      SATvar = "tas"
    elseif (ii .lt. ListCount(infoTas)+ListCount(infoTasa)) then
      infoSAT = infoTasa[ii-ListCount(infoTas)]
      SATvar = "tasa"
    elseif (ii .lt. ListCount(infoTas)+ListCount(infoTasa)+ListCount(infoTasaga)) then
      infoSAT = infoTasaga[ii-ListCount(infoTas)-ListCount(infoTasa)]
      SATvar = "tasaga"
    end if

;   Skip masking reference data of output_ref_trend = False
    if (.not.output_ref_trend .and. any(infoSAT@dataset.eq.(/masking_dataset_GMST, masking_dataset_pattern/))) then
      continue
    end if
;   Skip obs data of output_obs_trend = False
    if (.not.output_obs_trend .and. any(infoSAT@dataset.eq.obs_datasets)) then
      continue
    end if
    
;   Skip if the time segment is rcp/ssp part
    if (any(infoSAT@project.eq.(/"CMIP5", "CMIP6"/)) .and. \
        .not.isStrSubset(infoSAT@exp, "historical")) then 
      continue
    end if

    if (SATvar.ne."tasaga".and. \
        (any(infoSAT@project.eq.(/"CMIP5", "CMIP6"/)).or. \
         any(infoSAT@dataset.eq.reanalyses))) then
      masking = do_mask
      blending = do_blend
    else
      masking = False
      blending = False
    end if

; Find corresponding sft?f, tos and sic* data
    if (blending .and. any(SATvar .eq. (/"tas", "tasa"/))) then
      if (isdefined("infoSST")) then
        delete(infoSST)
      end if
      if (isdefined("infoIce")) then
        delete(infoIce)
      end if
      if (isdefined("infoFrac")) then
        delete(infoFrac)
      end if

;  Find corresponding sft?f
      opt = True
      opt@unique = "any"
      if (any(infoSAT@project.eq.(/"CMIP5", "CMIP6"/))) then
        infoFrac = get_info_match(infoSAT, infoSftlf, \
                                  (/"project", "dataset", "ensemble"/), opt)
        Fracvar = "sftlf"
        if (.not.infoFrac.and.isdefined("infoSftof")) then
          infoFrac = get_info_match(infoSAT, infoSftof, \
                                    (/"project", "dataset", "ensemble"/), opt)
          Fracvar = "sftof"
        end if
        if (.not.infoFrac) then  ; Search for other ensemble members
          infoFrac = get_info_match(infoSAT, infoSftlf, \
                                    (/"project", "dataset"/), opt)
          Fracvar = "sftlf"
          if (.not.infoFrac.and.isdefined("infoSftof")) then
            infoFrac = get_info_match(infoSAT, infoSftof, \
                                      (/"project", "dataset"/), opt)
            Fracvar = "sftof"
          end if
        end if
        if (.not.infoFrac) then  ; Search for other ensemble members
          infoFrac = get_info_match(infoSAT, infoSftlf, \
                                    (/"project"/), opt)
          Fracvar = "sftlf"
          if (.not.infoFrac.and.isdefined("infoSftof")) then
            infoFrac = get_info_match(infoSAT, infoSftof, \
                                      (/"project"/), opt)
            Fracvar = "sftof"
          end if
        end if
      elseif (any(infoSAT@dataset.eq.reanalyses)) then
        infoFrac = get_info_match(infoSAT, infoSftlf, \
                                  (/"project", "dataset"/), opt)
        Fracvar = "sftlf"
        if (.not.infoFrac.and.isdefined("infoSftof")) then
          infoFrac = get_info_match(infoSAT, infoSftof, \
                                    (/"project", "dataset"/), opt)
          Fracvar = "sftof"
        end if
        if (.not.infoFrac) then
          infoObs = True
          iObsProject = 0
          do while(.not.infoFrac)
            if (iObsProject.ge.dimsizes(obs_projects)) then
              infoFrac = infoSftlf[0]
            else
              infoObs@project = obs_projects(iObsProject)
              infoFrac = get_info_match(infoObs, infoSftlf, (/"project"/), opt)
            end if
            Fracvar = "sftlf"
            if (.not.infoFrac.and.isdefined("infoSftof")) then
              infoFrac = get_info_match(infoObs, infoSftof, (/"project"/), opt)
              Fracvar = "sftof"
            end if
            iObsProject = iObsProject + 1
          end do
          delete(infoObs)
          delete(iObsProject)
        end if
      end if
      
;  Find corresponding tos
      opt = True
      opt@unique = "strict"
      if (any(infoSAT@project.eq.(/"CMIP5", "CMIP6"/))) then
        infoSST = get_info_match(infoSAT, infoTos, \
                                 (/"project", "dataset", "ensemble", "exp", "end_year"/), opt)
      elseif (any(infoSAT@dataset.eq.reanalyses)) then
        infoSST = get_info_match(infoSAT, infoTos, \
                                 (/"project", "dataset", "end_year"/), opt)
        if (.not.infoSST) then
          infoObs = True
          infoObs@end_year = infoSAT@end_year
          iObsProject = 0
          do while(.not.infoSST)
            infoObs@project = obs_projects(iObsProject)
            infoSST = get_into_match(infoObs, infoTos, (/"project", "end_year"/), opt)
            iObsProject = iObsProject + 1
          end do
          delete(infoObs)
          delete(iObsProject)
        end if
      end if
      if (infoSST) then
        SSTvar = "tos"
      end if
      
;  Find corresponding sic*
      if (infoSAT@project.eq."CMIP5") then
        infoICE = get_info_match(infoSAT, infoSic, \
                                 (/"project", "dataset", "ensemble", "exp", "end_year"/), opt)
        ICEvar = "sic"
      elseif (infoSAT@project.eq."CMIP6") then
        if (ListCount(infoSiconc).le.0) then
          infoICE = False
        else
          infoICE = get_info_match(infoSAT, infoSiconc, \
                                   (/"project", "dataset", "ensemble", "exp", "end_year"/), opt)
          ICEvar = "siconc"
        end if
        if (.not.infoICE) then
          if (ListCount(infoSiconca).le.0) then
            infoICE = False
          else
            infoICE = get_info_match(infoSAT, infoSiconca, \
                                     (/"project", "dataset", "ensemble", "exp", "end_year"/), opt)
            ICEvar = "siconca"
          end if
        end if
      elseif (any(infoSAT@dataset.eq.reanalyses)) then
        infoICE = get_info_match(infoSAT, infoSic, \
                                 (/"project", "dataset", "end_year"/), opt)
        if (.not.infoICE) then
          infoObs = True
          infoObs@end_year = infoSAT@end_year
          iObsProject = 0
          do while(.not.infoSST)
            infoObs@project = obs_projects(iObsProject)
            infoSST = get_into_match(infoObs, infoSic, (/"project", "end_year"/), opt)
            iObsProject = iObsProject + 1
          end do
          delete(infoObs)
          delete(iObsProject)
        end if
        if (infoICE) then
          ICEvar = "sic"
        end if
      end if
    end if

; Read data
    SAT = read_data(infoSAT)
    SAT&time = cd_calendar(SAT&time, -1)
    if (isdefined("infoFrac")) then
      frac = read_data(infoFrac)
      if (max(frac).gt.2.) then
        frac = frac / 100.
        frac@units = ""
      end if
      if (min(frac).lt.0. .or. max(frac).lt.0.5 .or. max(frac).gt.1.01) then
        error_msg("f", DIAG_SCRIPT, "", "something is wrong in sftlf")
      end if
      frac = where(frac.gt.1., 1., frac)
      
      if (Fracvar.eq."sftof") then
        frac = 1. - frac
      end if
    end if
    if (isdefined("infoSST")) then
      SST = read_data(infoSST)
      SST&time = cd_calendar(SST&time, -1)
    end if
    if (isdefined("infoICE")) then
      ICE = read_data(infoICE)
      ICE&time = cd_calendar(ICE&time, -1)
    end if
    
; Concatenate with extension (if any)
    opt = True
    opt@unique = "strict"
;   SAT
    if (isdefined("infoSAT2")) then
      delete(infoSAT2)
    end if
    infoDummy = infoSAT
    infoDummy@start_year = infoSAT@end_year + 1
    if (SATvar .eq. "tas") then
      infoSAT2 = get_info_match(infoDummy, infoTas, \
                                (/"project", "dataset", "ensemble", "start_year"/), opt)
    elseif (SATvar .eq. "tasa") then
      infoSAT2 = get_info_match(infoDummy, infoTasa, \
                                (/"project", "dataset", "ensemble", "start_year"/), opt)
    end if
    if (isdefined("infoSAT2")) then
      if (.not.infoSAT2) then
        infoDummy@start_year = infoSAT@end_year
        if (SATvar .eq. "tas") then
          infoSAT2 = get_info_match(infoDummy, infoTas, \
                                    (/"project", "dataset", "ensemble", "start_year"/), opt)
        elseif (SATvar .eq. "tasa") then
          infoSAT2 = get_info_match(infoDummy, infoTasa, \
                                    (/"project", "dataset", "ensemble", "start_year"/), opt)
        end if
      end if
      delete(infoDummy)
      if (infoSAT2) then
        SAT2 = read_data(infoSAT2)
        SAT2&time = cd_calendar(SAT2&time, -1)
        if (min(SAT).lt.10. .and. min(SAT2).gt.200.) then
          SAT2 = SAT2 - 273.15
        elseif (max(SAT).gt.200. .and. max(SAT2).lt.100.) then
          SAT2 = SAT2 + 273.15
        end if
        tmp  = SAT
        delete(SAT)
        SAT = concatenate_data(tmp, SAT2)
        delete(tmp)
        if (isatt(infoSAT, "exp")) then
          infoSAT@exp = infoSAT@exp+"-"+infoSAT2@exp
        end if
        infoSAT@end_year = infoSAT2@end_year
        delete(SAT2)
      end if
;      delete(infoSAT2)
    end if
;   SST
    if (isdefined("infoSST2")) then
      delete(infoSST2)
    end if
    if (isdefined("infoSST")) then
      infoDummy = infoSST
      infoDummy@start_year = infoSST@end_year + 1
      infoSST2 = get_info_match(infoDummy, infoTos, \
                                (/"project", "dataset", "ensemble", "start_year"/), opt)
      if (.not.infoSST2) then
        infoDummy@start_year = infoSST@end_year
        infoSST2 = get_info_match(infoDummy, infoTos, \
                                  (/"project", "dataset", "ensemble", "start_year"/), opt)
      end if                            
      delete(infoDummy)
      if (infoSST2) then
        SST2 = read_data(infoSST2)
        SST2&time = cd_calendar(SST2&time, -1)
        if (min(SST).lt.10. .and. min(SST2).gt.200.) then
          SST2 = SST2 - 273.15
        elseif (max(SST).gt.200. .and. max(SST2).lt.100.) then
          SST2 = SST2 + 273.15
        end if
        tmp  = SST
        delete(SST)
        SST = concatenate_data(tmp, SST2)
        delete(tmp)
        if (isatt(infoSST, "exp")) then
          infoSST@exp = infoSST@exp+"-"+infoSST2@exp
        end if
        infoSST@end_year = infoSST2@end_year
        delete(SST2)
      end if
;      delete(infoSST2)
    end if
;   ICE
    if (isdefined("infoICE2")) then
      delete(infoICE2)
    end if
    if (isdefined("infoICE")) then
      infoDummy = infoICE
      infoDummy@start_year = infoICE@end_year + 1
      if (ICEvar .eq. "sic") then
        infoICE2 = get_info_match(infoDummy, infoSic, \
                                  (/"project", "dataset", "ensemble", "start_year"/), opt)
      elseif (ICEvar .eq. "siconc") then
        infoICE2 = get_info_match(infoDummy, infoSiconc, \
                                  (/"project", "dataset", "ensemble", "start_year"/), opt)
      elseif (ICEvar .eq. "siconca") then
        infoICE2 = get_info_match(infoDummy, infoSiconca, \
                                  (/"project", "dataset", "ensemble", "start_year"/), opt)

      end if
      if (.not.infoICE2) then
        infoDummy@start_year = infoICE@end_year
        if (ICEvar .eq. "sic") then
          infoICE2 = get_info_match(infoDummy, infoSic, \
                                    (/"project", "dataset", "ensemble", "start_year"/), opt)
        elseif (ICEvar .eq. "siconc") then
          infoICE2 = get_info_match(infoDummy, infoSiconc, \
                                    (/"project", "dataset", "ensemble", "start_year"/), opt)
        elseif (ICEvar .eq. "siconca") then
          infoICE2 = get_info_match(infoDummy, infoSiconca, \
                                    (/"project", "dataset", "ensemble", "start_year"/), opt)
        end if
      end if
      delete(infoDummy)
      if (infoICE2) then
        ICE2 = read_data(infoICE2)
        ICE2&time = cd_calendar(ICE2&time, -1)
        tmp  = ICE
        delete(ICE)
        ICE = concatenate_data(tmp, ICE2)
        delete(tmp)
        if (isatt(infoICE, "exp")) then
          infoICE@exp = infoICE@exp+"-"+infoICE2@exp
        end if
        infoICE@end_year = infoICE2@end_year
        delete(ICE2)
      end if
;      delete(infoICE2)
    end if

; Scale ICE data to [0, 1]
    if (isdefined("ICE")) then
      if (all(abs(ICE).lt.0.01)) then
        error_msg("f", DIAG_SCRIPT, "", ICEvar+" = 0 everywhere in " \
                  +infoICE@dataset+" "+infoICE@ensemble)
      end if
      if (max(abs(ICE)).gt.2.) then
        ICE = ICE / 100.
        ICE@units = ""
      end if                
      if (min(ICE).lt.0. .or. max(ICE).gt.1.01) then
        error_msg("f", DIAG_SCRIPT, "", "something is wrong in "+ICEvar+" range of " \
                  +infoICE@dataset+" "+infoICE@ensemble)
      end if
      ICE = where(ICE.gt.1., 1., ICE)
    end if
    
; Convert to anomalies
    if (SATvar.eq."tas") then
      if (do_anomaly) then
        SAT = calc_anom(SAT, syear_clim, eyear_clim)
      else
        if (any(SAT@units.eq.(/"C", "degC", "degrees_C"/))) then
          SAT = SAT + 273.15
          SAT@units = "K"
        end if
      end if
    end if
    if (isdefined("SST")) then
      if (do_anomaly) then
        SST = calc_anom(SST, syear_clim, eyear_clim)
      else
        if (any(SST@units.eq.(/"C", "degC", "degrees_C"/))) then
          SST = SST + 273.15
          SST@units = "K"
        end if
      end if
    end if
 
; Trim unnecessary time segments
    if (smonth_ann.le.emonth_ann) then
      stime = syear_hiatus*100+1
    else
      stime = (syear_hiatus - 1)*100+1
    end if
    etime = eyear_post*100+12

    if (dimsizes(dimsizes(SAT)).eq.3) then
      sat = SAT({stime:etime}, :, :)
    else
      sat = SAT({stime:etime})
    end if
    delete(SAT)
    SAT = sat
    delete(sat)
    if (isvar("SST")) then
      sst = SST({stime:etime}, :, :)
      delete(SST)
      SST = sst
      delete(sst)
    end if
    if (isvar("ICE")) then
      ice = ICE({stime:etime}, :, :)
      delete(ICE)
      ICE = ice
      delete(ice)
    end if
    
; Blend
    if (blending.and.all(isdefined((/"SST", "ICE", "frac"/)))) then
      sst = fillLand(SST)   ; Fill missing values over land
      ice = fillLand(ICE)   ; Fill missing values over land      
      Ts = blend_tas_tos(SAT, sst, ice, frac, method_GMST)
      delete([/sst, ice/])
      delete([/SST, ICE, frac/])
    end if

; Calculate trend pattern and save
    if (SATvar.ne."tasaga" .and. do_pattern) then
      trendSAThiatus = calc_trend(SAT, syear_hiatus, eyear_hiatus, method_trend, \
                                  annual, smonth_ann, emonth_ann)
      trendSAThiatus@diag_script = DIAG_SCRIPT 
      trendSAThiatus@var = "trend"
      nc_filename = work_dir + "HiatusTrendPattern_unblended_unmasked_"+set_output_tail(infoSAT)+".nc"
      nc_outfile = ncdf_write(trendSAThiatus, nc_filename)      
      if (isvar("infoSAT2").and.infoSAT2) then
        log_provenance(nc_filename, "n/a", "SAT trend", "trend", "global", "other", \
                       "kosaka_yu", "", (/infoSAT@filename, infoSAT2@filename/))
      else
        log_provenance(nc_filename, "n/a", "SAT trend", "trend", "global", "other", \
                       "kosaka_yu", "", infoSAT@filename)
      end if
      delete(trendSAThiatus)

      has_post = False
      if (SAT&time(dimsizes(SAT&time)-1).ge.eyear_post*100+emonth_ann) then
        has_post = True
      end if
      if (do_post.and.has_post) then
        trendSATpost = calc_trend(SAT, syear_post, eyear_post, method_trend, \
                                  annual, smonth_ann, emonth_ann)
        trendSATpost@diag_script = DIAG_SCRIPT 
        trendSATpost@var = "trend"
        nc_filename = work_dir + "PostTrendPattern_unblended_unmasked_"+set_output_tail(infoSAT)+".nc"
        nc_outfile = ncdf_write(trendSATpost, nc_filename)      
        if (isvar("infoSAT2").and.infoSAT2) then
          log_provenance(nc_filename, "n/a", "SAT trend", "trend", "global", "other", \
                         "kosaka_yu", "", (/infoSAT@filename, infoSAT2@filename/))
        else
          log_provenance(nc_filename, "n/a", "SAT trend", "trend", "global", "other", \
                         "kosaka_yu", "", infoSAT@filename)
        end if
        delete(trendSATpost)
      end if

      if (isdefined("Ts")) then
        trendTshiatus = calc_trend(Ts, syear_hiatus, eyear_hiatus, method_trend, \
                                   annual, smonth_ann, emonth_ann)
        trendTshiatus@diag_script = DIAG_SCRIPT 
        trendTshiatus@var = "trend"
        nc_filename = work_dir + "HiatusTrendPattern_blended_unmasked_"+set_output_tail(infoSAT)+".nc"
        nc_outfile = ncdf_write(trendTshiatus, nc_filename)      
        if (isvar("infoSAT2").and.infoSAT2) then
          log_provenance(nc_filename, "n/a", "near-surface temperature trend", \
                         "trend", "global", "other", "kosaka_yu", "", \
                         (/infoSAT@filename, infoSAT2@filename, infoSST@filename, infoSST2@filename, \
                           infoICE@filename, infoICE2@filename, infoFrac@filename /))
        else
          log_provenance(nc_filename, "n/a", "near-surface temperature trend", \
                         "trend", "global", "other", "kosaka_yu", "", \
                         (/infoSAT@filename, infoSST@filename, infoICE@filename, infoFrac@filename/))
        end if
        delete(trendTshiatus)
        
        has_post = False
        if (Ts&time(dimsizes(Ts&time)-1).ge.eyear_post*100+emonth_ann) then
          has_post = True
        end if
        if (do_post.and.has_post) then
          trendTspost = calc_trend(Ts, syear_post, eyear_post, method_trend, \
                                   annual, smonth_ann, emonth_ann)
          trendTspost@diag_script = DIAG_SCRIPT 
          trendTspost@var = "trend"
          nc_filename = work_dir + "PostTrendPattern_blended_unmasked_"+set_output_tail(infoSAT)+".nc"
          nc_outfile = ncdf_write(trendTspost, nc_filename)      
          if (isvar("infoSAT2").and.infoSAT2) then
            log_provenance(nc_filename, "n/a", "near-surface temperature trend", \
                          "trend", "global", "other", "kosaka_yu", "", \
                          (/infoSAT@filename, infoSAT2@filename, infoSST@filename, infoSST2@filename, \
                            infoICE@filename, infoICE2@filename, infoFrac@filename/))
          else
            log_provenance(nc_filename, "n/a", "near-surface temperature trend", \
                           "trend", "global", "other", "kosaka_yu", "", \
                           (/infoSAT@filename, infoSST@filename, infoICE@filename, infoFrac@filename/))
          end if
          delete(trendTspost)
        end if
      end if

      if (masking) then
        SATmasked = apply_mask(SAT, maskTsPattern)
        trendSAThiatusMasked = calc_trend(SATmasked, syear_hiatus, eyear_hiatus, method_trend, \
                                          annual, smonth_ann, emonth_ann)
        trendSAThiatusMasked@diag_script = DIAG_SCRIPT 
        trendSAThiatusMasked@var = "trend"
        nc_filename = work_dir + "HiatusTrendPattern_unblended_masked_"+set_output_tail(infoSAT)+".nc"
        nc_outfile = ncdf_write(trendSAThiatusMasked, nc_filename)      
        if (isvar("infoSAT2").and.infoSAT2) then
          log_provenance(nc_filename, "n/a", "SAT trend", "trend", "global", "other", \
                         "kosaka_yu", "", (/infoSAT@filename, infoSAT2@filename, maskTsPattern@filename/))
        else
          log_provenance(nc_filename, "n/a", "SAT trend", "trend", "global", "other", \
                         "kosaka_yu", "", (/infoSAT@filename, maskTsPattern@filename/))
        end if
        delete(trendSAThiatusMasked)
                                          
        if (do_maskPattern_post) then
          has_post = False
          if (SATmasked&time(dimsizes(SATmasked&time)-1).ge.eyear_post*100+emonth_ann) then
            has_post = True
          end if
          if (do_post.and.has_post) then
            trendSATpostMasked = calc_trend(SATmasked, syear_post, eyear_post, method_trend, \
                                            annual, smonth_ann, emonth_ann)
            trendSATpostMasked@diag_script = DIAG_SCRIPT 
            trendSATpostMasked@var = "trend"
            nc_filename = work_dir + "PostTrendPattern_unblended_masked_"+set_output_tail(infoSAT)+".nc"
            nc_outfile = ncdf_write(trendSATpostMasked, nc_filename)      
            if (isvar("infoSAT2").and.infoSAT2) then
              log_provenance(nc_filename, "n/a", "SAT trend", "trend", "global", "other", \
                             "kosaka_yu", "", (/infoSAT@filename, infoSAT2@filename, maskTsPattern@filename/))
            else
              log_provenance(nc_filename, "n/a", "SAT trend", "trend", "global", "other", \
                             "kosaka_yu", "", (/infoSAT@filename, maskTsPattern@filename/))
            end if
            delete(trendSATpostMasked)
          end if
        end if
        delete(SATmasked)
      end if

      if (isdefined("Ts") .and. masking) then
        Tsmasked = apply_mask(Ts, maskTsPattern)
        trendTshiatusMasked = calc_trend(Tsmasked, syear_hiatus, eyear_hiatus, method_trend, \
                                         annual, smonth_ann, emonth_ann)
        trendTshiatusMasked@diag_script = DIAG_SCRIPT 
        trendTshiatusMasked@var = "trend"
        nc_filename = work_dir + "HiatusTrendPattern_blended_masked_"+set_output_tail(infoSAT)+".nc"
        nc_outfile = ncdf_write(trendTshiatusMasked, nc_filename)
        if (isvar("infoSAT2").and.infoSAT2) then
          log_provenance(nc_filename, "n/a", "near-surface temperature trend", \
                         "trend", "global", "other", "kosaka_yu", "", \
                         (/infoSAT@filename, infoSAT2@filename, infoSST@filename, infoSST2@filename, \
                           infoICE@filename, infoICE2@filename, infoFrac@filename, maskTsPattern@filename/))
        else
          log_provenance(nc_filename, "n/a", "near-surface temperature trend", \
                         "trend", "global", "other", "kosaka_yu", "", \
                         (/infoSAT@filename, infoSST@filename, infoICE@filename, \
                           infoFrac@filename, maskTsPattern@filename/))
        end if 
        delete(trendTshiatusMasked)

        if (do_maskPattern_post) then
          has_post = False
          if (Tsmasked&time(dimsizes(Tsmasked&time)-1).ge.eyear_post*100+emonth_ann) then
            has_post = True
          end if
          if (do_post.and.has_post) then
            trendTspostmasked = calc_trend(Tsmasked, syear_post, eyear_post, method_trend, \
                                           annual, smonth_ann, emonth_ann)
            trendTspostMasked@diag_script = DIAG_SCRIPT 
            trendTspostMasked@var = "trend"
            nc_filename = work_dir + "PostTrendPattern_blended_masked_"+set_output_tail(infoSAT)+".nc"
            nc_outfile = ncdf_write(trendSATpostMasked, nc_filename)
            if (isvar("infoSAT2").and.infoSAT2) then
              log_provenance(nc_filename, "n/a", "near-surface temperature trend", \
                             "trend", "global", "other", "kosaka_yu", "", \
                             (/infoSAT@filename, infoSAT2@filename, infoSST@filename, infoSST2@filename, \
                               infoICE@filename, infoICE2@filename, infoFrac@filename, maskTsPattern@filename/))
            else
              log_provenance(nc_filename, "n/a", "near-surface temperature trend", \
                             "trend", "global", "other", "kosaka_yu", "", \
                             (/infoSAT@filename, infoSST@filename, infoICE@filename, \
                               infoFrac@filename, maskTsPattern@filename/))
            end if 
            delete(trendTspostMasked)
          end if
        end if
        delete(Tsmasked)
      end if
    end if
    
; Calculate GMST/GSAT trends and save

;   Skip if corresponding tasaga exists
    do_globalmean = True
    if (any(SATvar.eq.(/"tas", "tasa"/)) .and. isvar("infoTasaga")) then
      opt = True
      opt@unique = "any"
      if (any(infoSAT@project.eq.(/"CMIP5", "CMIP6"/))) then
        if (get_info_match(infoSAT, infoTasaga, \
                           (/"project", "dataset", "ensemble"/), opt)) then
          do_globalmean = False
        end if
      elseif (any(infoSAT@project.eq.obs_projects)) then
        if (get_info_match(infoSAT, infoTasaga, \
                           (/"project", "dataset", "type"/), opt)) then  ; Assuming that type represents ens in HadCRUT4&5 and CowtanWay
          do_globalmean = False
        end if
      end if
    end if

;   Calculate global-mean trends and write
    if (do_globalmean) then
      if (SATvar.eq."tasaga") then
        GSAT = SAT
      else
        GSAT = global_mean(SAT, "global")
      end if
      trendGSAThiatus = calc_trend(GSAT, syear_hiatus, eyear_hiatus, method_trend, \
                                   annual, smonth_ann, emonth_ann)
      has_post = False
      if (GSAT&time(dimsizes(GSAT&time)-1).ge.eyear_post*100+emonth_ann) then
        has_post = True
      end if
      if (do_post.and.has_post) then
        trendGSATpost = calc_trend(GSAT, syear_post, eyear_post, method_trend, \
                                   annual, smonth_ann, emonth_ann)
        write_GMtrend(infoSAT, (/trendGSAThiatus, trendGSATpost/), False, False, GMList)
      else
        write_GMtrend(infoSAT, (/trendGSAThiatus/), False, False, GMList)
      end if
      if (isdefined("infoSAT2").and.infoSAT2) then
        addatt_GMtrend([/infoSAT, infoSAT2/], "filename", False, False, GMList)
      else
        addatt_GMtrend([/infoSAT/], "filename", False, False, GMList)
      end if
      delete([/GSAT, trendGSAThiatus/])
      
      if (isdefined("Ts")) then
        GMST = global_mean(Ts, "global")
        trendGMSThiatus = calc_trend(GMST, syear_hiatus, eyear_hiatus, method_trend, \
                                     annual, smonth_ann, emonth_ann)
        has_post = False
        if (GMST&time(dimsizes(GMST&time)-1).ge.eyear_post*100+emonth_ann) then
          has_post = True
        end if
        if (do_post.and.has_post) then
          trendGMSTpost = calc_trend(GMST, syear_post, eyear_post, method_trend, \
                                     annual, smonth_ann, emonth_ann)
          write_GMtrend(infoSAT, (/trendGMSThiatus, trendGMSTpost/), True, False, GMList)
        else
          write_GMtrend(infoSAT, (/trendGMSThiatus/), True, False, GMList)
        end if
        if (isdefined("infoSAT2").and.infoSAT2) then
          addatt_GMtrend([/infoSAT, infoSAT2, infoSST, infoSST2, infoICE, infoICE2, infoFrac/], \
                         "filename", True, False, GMList)
        else
          addatt_GMtrend([/infoSAT, infoSST, infoICE, infoFrac/], \
                         "filename", True, False, GMList)
        end if
        delete([/GMST, trendGMSThiatus/])
      end if

      if (masking) then
        SATmasked = apply_mask(SAT, maskTsGlobalmean)
        GSATmasked = global_mean(SATmasked, method_GMST)
        trendGSAThiatusMasked = calc_trend(GSATmasked, syear_hiatus, eyear_hiatus, method_trend, \
                                           annual, smonth_ann, emonth_ann)
        delete(SATmasked)
        has_post = False
        if (do_maskGlobalmean_post) then
          if (GSATmasked&time(dimsizes(GSATmasked&time)-1).ge.eyear_post*100+emonth_ann) then
            has_post = True
          end if
        end if
        if (do_maskGlobalmean_post.and.do_post.and.has_post) then
          trendGSATpostMasked = calc_trend(GSATmasked, syear_post, eyear_post, method_trend, \
                                           annual, smonth_ann, emonth_ann)
          write_GMtrend(infoSAT, (/trendGSAThiatusMasked, trendGSATpostMasked/), False, True, GMList)
          delete(trendGSATpostMasked)
        else
          if (isdefined("trendGSATpost")) then
            write_GMtrend(infoSAT, (/trendGSAThiatusMasked, trendGSATpost/), False, True, GMList)
            delete(trendGSATpost)
          else
            write_GMtrend(infoSAT, (/trendGSAThiatusMasked/), False, True, GMList)
          end if
        end if
        if (isdefined("infoSAT2").and.infoSAT2) then
          addatt_GMtrend([/infoSAT, infoSAT2, maskTsGlobalmean/], "filename", False, True, GMList)
        else
          addatt_GMtrend([/infoSAT, maskTsGlobalmean/], "filename", False, True, GMList)
        end if
        delete([/GSATmasked, trendGSAThiatusMasked/])
      end if

      if (isdefined("Ts") .and. masking) then
        Tsmasked = apply_mask(Ts, maskTsGlobalmean)
        GMSTmasked = global_mean(Tsmasked, method_GMST)
        trendGMSThiatusMasked = calc_trend(GMSTmasked, syear_hiatus, eyear_hiatus, method_trend, \
                                           annual, smonth_ann, emonth_ann)
        delete(Tsmasked)
        has_post = False
        if (do_maskGlobalmean_post) then
          if (GMSTmasked&time(dimsizes(GMSTmasked&time)-1).ge.eyear_post*100+emonth_ann) then
            has_post = True
          end if
        end if
        if (do_maskGlobalmean_post.and.do_post.and.has_post) then
          trendGMSTpostMasked = calc_trend(GMSTmasked, syear_post, eyear_post, method_trend, \
                                           annual, smonth_ann, emonth_ann)
          write_GMtrend(infoSAT, (/trendGMSThiatusMasked, trendGMSTpostMasked/), True, True, GMList)
          delete(trendGMSTpostMasked)
        else
          if (isdefined("trendGMSTpost")) then
            write_GMtrend(infoSAT, (/trendGMSThiatusMasked, trendGMSTpost/), True, True, GMList)
            delete(trendGMSTpost)
          else
            write_GMtrend(infoSAT, (/trendGMSThiatusMasked/), True, True, GMList)
          end if
        end if
        if (isdefined("infoSAT2").and.infoSAT2) then
          addatt_GMtrend([/infoSAT, infoSAT2, infoSST, infoSST2, infoICE, infoICE2, infoFrac, maskTsGlobalmean/], \
                         "filename", True, True, GMList)
        else
          addatt_GMtrend([/infoSAT, infoSST, infoICE, infoFrac, maskTsGlobalmean/], \
                         "filename", True, True, GMList)
        end if
        delete([/GMSTmasked, trendGMSThiatusMasked/])
      end if
      
    end if

; Clean
    delete([/infoSAT, SATvar, SAT/])
    if (isdefined("Ts")) then
      delete(Ts)
    end if
    if (isdefined("SST")) then
      delete([/infoSST, SSTvar, SST/])
    end if
    if (isdefined("ICE")) then
      delete([/infoICE, ICEvar, ICE/])
    end if
    if (isdefined("frac")) then
      delete([/infoFrac, Fracvar, frac/])
    end if
    if (isdefined("infoSAT2")) then
      delete(infoSAT2)
    end if
    if (isdefined("infoSST2")) then
      delete(infoSST2)
    end if
    if (isdefined("infoICE2")) then
      delete(infoICE2)
    end if
    
  end do

  do ii = 0, ListCount(GMList)-1
    if (isatt(GMList[ii], "ofile_GSATunmasked")) then
      log_provenance(GMList[ii]@ofile_GSATunmasked, "n/a", "GSAT trend, unmasked", \
                     "trend", "global", "other", "kosaka_yu", "", str_split(GMList[ii]@filename_GSATunmasked, ","))
    end if
    if (isatt(GMList[ii], "ofile_GMSTunmasked")) then
      log_provenance(GMList[ii]@ofile_GMSTunmasked, "n/a", "GMST trend, unmasked", \
                     "trend", "global", "other", "kosaka_yu", "", str_split(GMList[ii]@filename_GMSTunmasked, ","))
    end if
    if (isatt(GMList[ii], "ofile_GSATmasked")) then
      log_provenance(GMList[ii]@ofile_GSATmasked, "n/a", "GSAT trend, masked", \
                     "trend", "global", "other", "kosaka_yu", "", str_split(GMList[ii]@filename_GSATmasked, ","))
    end if
    if (isatt(GMList[ii], "ofile_GMSTmasked")) then
      log_provenance(GMList[ii]@ofile_GMSTmasked, "n/a", "GMST trend, masked", \
                     "trend", "global", "other", "kosaka_yu", "", str_split(GMList[ii]@filename_GMSTmasked, ","))
    end if
  end do
  
end
