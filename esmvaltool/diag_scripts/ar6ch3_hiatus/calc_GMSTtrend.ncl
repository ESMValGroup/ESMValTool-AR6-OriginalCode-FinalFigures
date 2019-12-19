; blend_globalmean
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs: 
;
; History
;    20191101 kosaka_yu: written.
;    20191202 kosaka_yu: updated for version 2.0.0b3.
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

load "$diag_scripts/ar6ch3_hiatus/grid_util.ncl"
load "$diag_scripts/ar6ch3_hiatus/time_util.ncl"
load "$diag_scripts/ar6ch3_hiatus/blend.ncl"

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

;function get_filename(info, var)
;local info, var, name, project, dataset, ensemble, experment
;begin
;  project    = info@project   ; obs/obs4mips/ana4mips/CMIP5/CMIP6
;  dataset    = info@dataset   ; model/dataset name
;  if (isatt(info, "exp")) then
;    experiment = info@exp       ; historical-rcp45 etc.
;  else
;    experiment = ""
;  end if
;  if (isatt(info, "ensemble")) then
;    ensemble   = info@ensemble  ; r*i*p*
;  else
;    ensemble   = ""
;  end if
;  name = var+"_"+project+"_"+dataset
;  if (experiment.ne."") then
;    name = name+"_"+experiment
;  end if
;  if (ensemble.ne."") then
;    name = name+"_"+ensemble
;  end if
;  name = name+".nc"
;  return(name)
;end

procedure pushObsList(ObsList, array, info)
local ens, nens, dims, dimsnew, dataset_names, i, str
begin
  ; For HadCRUT4 and CowtanWay, get ensemble number
  if (isStrSubset(info@dataset, "HadCRUT4").or.\
      isStrSubset(info@dataset, "CowtanWay")) then
    if (isStrSubset(info@filename, "_median_")) then
      ens = 0
    else
      str = str_split(info@filename, "_")
      i = str_match_ind(str, "ens")
      ens = toint(str_sub_str(str(i), "ens", ""))
      delete(str)
      delete(i)
    end if
    nens = 101

    if (ListCount(ObsList).lt.1) then
      dims = dimsizes(array)
      if (dimsizes(dims).eq.1) then
        dimsnew = nens
      else
        dimsnew = array_append_record(nens, dims, 0)
      end if
      delete(dims)
      ListAppend(ObsList, new(dimsnew, typeof(array))) ; ens = 0: median
      i = ListCount(ObsList) - 1
      ObsList[i]!0 = "ensemble"
      ObsList[i]&ensemble = ispan(0, nens-1, 1)
    else
      dataset_names = metadata_att_as_array(ObsList, "dataset")
      if (any(dataset_names.eq.info@dataset)) then
        i = ind(dataset_names.eq.info@dataset)
      else
        dims = dimsizes(array)
        if (dimsizes(dims).eq.1) then
          dimsnew = nens
        else
          dimsnew = array_append_record(nens, dims, 0)
        end if
        delete(dims)
        ListAppend(ObsList, new(dimsnew, typeof(array))) ; ens = 0: median
        i = ListCount(ObsList) - 1
        ObsList[i]!0 = "ensemble"
        ObsList[i]&ensemble = ispan(0, nens-1, 1)
      end if
      delete(dataset_names)
    end if
    if (dimsizes(dimsizes(array)).eq.2) then
      ObsList[i](ens, :, :) = array
    else
      ObsList[i](ens) = array
    end if
  else
    ListAppend(ObsList, new(dimsizes(array), typeof(array)))
    i = ListCount(ObsList) - 1
    ObsList[i] = array
    if (dimsizes(dimsizes(array)).eq.1) then
      ObsList[i]!0 = "ensemble"
      ObsList[i]&ensemble = 0
    end if
  end if
  ObsList[i]@dataset = info@dataset
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

  anomaly = True
  blending = True
  masking = True
  if (any(method_GMST.eq.(/"Cowtan2015", "HadCRUT4", "GISTEMP", "NOAAGlobalTemp"/))) then
    anomaly  = True
    blending = True
    masking  = True
    if (any(method_GMST.eq.(/"Cowtan2015", "HadCRUT4"/))) then
      masking_dataset_GMST = "HadCRUT4"
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
    end if
   elseif (any(method_GMST.eq.(/"BerkeleyEarth", "CowtanWay", "global"/))) then
    anomaly  = True
    blending = True
    masking  = False
    if (method_GMST.eq."BerkeleyEarth") then
      syear_clim = 1951
      eyear_clim = 1980
    elseif (method_GMST.eq."CowtanWay") then
      syear_clim = 1961
      eyear_clim = 1990
    elseif (method_GMST.eq."global") then
      syear_clim = 1961
      eyear_clim = 1990
    end if
  elseif (method_GMST.eq."GSAT") then
    anomaly  = True
    blending = False
    masking  = False
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

  do_post_all = True
  if (isatt(diag_script_info, "do_post")) then  ; post-hiatus period
    do_post_all = diag_script_info@do_post
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

; 0.3 Output trend pattern?
  do_pattern = True
  if (isatt(diag_script_info, "do_pattern")) then
    do_pattern = diag_script_info@do_pattern
  end if
  
; 0.4 Parameters
  obs_projects = (/"obs", "OBS", "OBS6", "obs4mips", "ana4mips"/)
  reanalyses = (/"ERA5", "ERA-Interim", "JRA-55", "JRA-25", "CFSR", "MERRA", "MERRA2", "NCEP", "NCEP2", "ERA-20C", "20CR"/)

; =========================================================================
;
;         1. Get dataset lists and prepare for blending and masking
;
; =========================================================================

; 1.1 Get list of files  
  varList = metadata_att_as_array(input_file_info, "short_name")
  varList := get_unique_values(varList)
  
  if (all(varList.ne."tas")) then
    error_msg("f", DIAG_SCRIPT, "", "variable tas is missing.")
  end if

  if (blending) then
    if (all(varList.ne."tos")) then
      error_msg("f", DIAG_SCRIPT, "", "variable tos is missing.")
    end if
    if (all(varList.ne."sic") .and. all(varList.ne."siconc") .and. all(varList.ne."siconca")) then
      error_msg("f", DIAG_SCRIPT, "", "variable sic/siconc/siconca is missing.")
    end if
  end if

  var = "tas"
  infoTas = select_metadata_by_name(input_file_info, var)
  dataset_names_tas = metadata_att_as_array(infoTas, "dataset")

  var = "tasa"
  if (any(var.eq.varList)) then
    infoTasa = select_metadata_by_name(input_file_info, var)
    dataset_names_tasa = metadata_att_as_array(infoTasa, "dataset")
  end if
  
  var = "tasaga"
  if (any(var.eq.varList)) then
    infoTasaga = select_metadata_by_name(input_file_info, var)
    dataset_names_tasaga = metadata_att_as_array(infoTasaga, "dataset")
  end if
    
; 1.2 Get dataset number of reference dataset for masking  
  if (masking) then
  ; Find reference data (e.g. HadCRUT4 median) for masking in GMST calculation
    if (any(dataset_names_tas.eq.masking_dataset_GMST)) then
      dataset_names = dataset_names_tas
      refVar = "tas"
    elseif (any(dataset_names_tasa.eq.masking_dataset_GMST)) then
      dataset_names = dataset_names_tasa
      refVar = "tasa"
    end if
    
    iTasRefGMST = ind(dataset_names .eq. masking_dataset_GMST) 
    if (dimsizes(iTasRefGMST).gt.1) then
      npaths = dimsizes(iTasRefGMST)
      tmpRefData = iTasRefGMST
      delete(iTasRefGMST)
      do i = 0, npaths-1
        if ((refVar.eq."tas".and.isStrSubset(infoTas[tmpRefData(i)]@filename, "_median_")) .or.\
            (refVar.eq."tasa".and.isStrSubset(infoTasa[tmpRefData(i)]@filename, "_median_"))) then
          iTasRefGMST = tmpRefData(i)
          break
        end if
      end do
      if (.not.isvar("iTasRefGMST")) then
        iTasRefGMST = tmpRefData(0)
      end if
      delete(tmpRefData)
    end if
    iTasRefGMST@var = refVar
    delete(dataset_names)
    
  ; Find reference data (e.g. HadCRUT4 median) for masking in patterned trend calculation
    if (isvar("infoTasa").and.any(dataset_names_tasa.eq.masking_dataset_pattern)) then
      dataset_names = dataset_names_tasa
      refVar = "tasa"
    elseif (any(dataset_names_tas.eq.masking_dataset_pattern)) then
      dataset_names = dataset_names_tas
      refVar = "tas"
    end if
    
    iTasRefPattern = ind(dataset_names .eq. masking_dataset_pattern) 
    if (dimsizes(iTasRefPattern).gt.1) then
      npaths = dimsizes(iTasRefPattern)
      tmpRefData = iTasRefPattern
      delete(iTasRefPattern)
      do i = 0, npaths-1
        if ((refVar.eq."tas".and.isStrSubset(infoTas[tmpRefData(i)]@filename, "_median_")) .or.\
            (refVar.eq."tasa".and.isStrSubset(infoTasa[tmpRefData(i)]@filename, "_median_"))) then
          iTasRefPattern = tmpRefData(i)
          break
        end if
      end do
      if (.not.isvar("iTasRefPattern")) then
        iTasRefPattern = tmpRefData(0)
      end if
      delete(tmpRefData)
    end if
    iTasRefPattern@var = refVar
    delete(dataset_names)
  end if
;  print("-----------------------")
;  print("making dataset for GMST")
;  print(infoTasa[iTasRefGMST])
;  print("-----------------------")
;  print("making dataset for Pattern")
;  print(infoTasa[iTasRefPattern])
;  print("-----------------------")
  
; 1.3 Find all observational data (needed?)
  do i = 0, ListCount(infoTas)-1
    if (any(infoTas[i]@project.eq.obs_projects)) then
      dataset = infoTas[i]@dataset
      if (isStrSubset(dataset, "-clim")) then
        continue
      end if
      if (isvar("iTasObs")) then
        tmp = array_append_record(iTasObs, (/i/), 0)
        delete(iTasObs)
        iTasObs = tmp
        delete(tmp)
      else
        iTasObs = (/i/)
      end if
    end if
  end do

  if (blending) then
; 1.4 Find correspondng tos and sic datasets for each tas dataset
    var = "tos"
    infoTos = select_metadata_by_name(input_file_info, var)
    var = "sic"
    infoSic = select_metadata_by_name(input_file_info, var)
    var = "siconc"
    infoSiconc = select_metadata_by_name(input_file_info, var)
    var = "siconca"
    infoSiconca = select_metadata_by_name(input_file_info, var)
    var = "sftlf"
    infoSftlf = select_metadata_by_name(input_file_info, var)
    var = "sftof"
    infoSftof = select_metadata_by_name(input_file_info, var)

    iTos     = new(ListCount(infoTas), "integer")
    iSic     = new(ListCount(infoTas), "integer")
    iSiconc  = new(ListCount(infoTas), "integer")
    iSiconca = new(ListCount(infoTas), "integer")
    iSftlf   = new(ListCount(infoTas), "integer")
    iSftof   = new(ListCount(infoTas), "integer")
    NameSic  = new(ListCount(infoTas), "string")
    Namelof  = new(ListCount(infoTas), "string")
    do i = 0, ListCount(infoTas)-1
      if (all(infoTas[i]@project.ne.obs_projects)) then
        project    = infoTas[i]@project
        experiment = infoTas[i]@exp
        dataset    = infoTas[i]@dataset
        ensemble   = infoTas[i]@ensemble
; find tos file
        do j = 0, ListCount(infoTos)-1
          if (infoTos[j]@project.eq.project .and. infoTos[j]@exp.eq.experiment .and. infoTos[j]@dataset.eq.dataset .and. infoTos[j]@ensemble.eq.ensemble) then
            iTos(i) = j
            break
          end if
        end do
        infoTas[i]@tos_filename = infoTos[iTos(i)]@filename
; find sea ice file
        do j = 0, ListCount(infoSic)-1
          if (infoSic[j]@project.eq.project .and. infoSic[j]@exp.eq.experiment .and. infoSic[j]@dataset.eq.dataset .and. infoSic[j]@ensemble.eq.ensemble) then
            iSic(i) = j
            NameSic(i) = "sic"
            break
          end if
        end do
        if (ismissing(iSic(i))) then
          do j = 0, ListCount(infoSiconc)-1
            if (infoSiconc[j]@project.eq.project .and. infoSiconc[j]@exp.eq.experiment .and. infoSiconc[j]@dataset.eq.dataset .and. infoSiconc[j]@ensemble.eq.ensemble) then
              iSiconc(i) = j
              NameSic(i) = "siconc"
              break
            end if
          end do
          if (ismissing(iSiconc(i))) then
            do j = 0, ListCount(infoSiconca)-1
              if (infoSiconca[j]@project.eq.project .and. infoSiconca[j]@exp.eq.experiment .and. infoSiconca[j]@dataset.eq.dataset .and. infoSiconca[j]@ensemble.eq.ensemble) then
                iSiconca(i) = j
                NameSic(i) = "siconca"
                break
              end if
            end do
            print(infoTas[i]@filename)
            infoTas[i]@sic_filename = infoSiconca[iSiconca(i)]@filename
            infoTas[i]@sic_varname  = "siconca"
          else
            infoTas[i]@sic_filename = infoSiconc[iSiconc(i)]@filename
            infoTas[i]@sic_varname  = "siconc"
          end if
        else
          infoTas[i]@sic_filename = infoSic[iSic(i)]@filename
          infoTas[i]@sic_varname  = "sic"
        end if
; find land/ocean fraction file
        do j = 0, ListCount(infoSftlf)-1
          if (infoSftlf[j]@project.eq.project .and. infoSftlf[j]@exp.eq.experiment .and. infoSftlf[j]@dataset.eq.dataset .and. infoSftlf[j]@ensemble.eq.ensemble) then
            iSftlf(i) = j
            Namelof(i) = "sftlf"
            break
          end if
        end do
        if (ismissing(iSftlf(i))) then
          do j = 0, ListCount(infoSftlf)-1
            if (infoSftlf[j]@project.eq.project .and. infoSftlf[j]@exp.eq.experiment .and. infoSftlf[j]@dataset.eq.dataset) then
              iSftlf(i) = j
              Namelof(i) = "sftlf"
              break
            end if
          end do
        end if
        if (ismissing(iSftlf(i))) then
          do j = 0, ListCount(infoSftlf)-1
            if (infoSftlf[j]@project.eq.project .and. infoSftlf[j]@dataset.eq.dataset) then
              iSftlf(i) = j
              Namelof(i) = "sftlf"
              break
            end if
          end do
        end if
        if (ismissing(iSftlf(i))) then
          do j = 0, ListCount(infoSftof)-1
            if (infoSftof[j]@project.eq.project .and. infoSftof[j]@dataset.eq.dataset) then
              iSftof(i) = j
              Namelof(i) = "sftof"
              break
            end if
          end do
          if (.not.ismissing(iSftof(i))) then
            infoTas[i]@lof_filename = infoSftof[iSftof(i)]@filename
            infoTas[i]@lof_varname  = "sftof"
          else
            do j = 0, ListCount(infoSftlf)-1
              if (infoSftlf[j]@project.eq.project) then
                iSftlf(i) = j
                Namelof(i) = "sftlf"
                break
              end if
            end do
            infoTas[i]@lof_filename = infoSftlf[iSftlf(i)]@filename
            infoTas[i]@lof_varname  = "sftlf"
          end if
        else
          infoTas[i]@lof_filename = infoSftlf[iSftlf(i)]@filename
          infoTas[i]@lof_varname  = "sftlf"
        end if
;        infoTas[i]@sftlf_filename = infoSftlf[iSftlf(i)]@filename
      else
        iTos(i)     = iTos@_FillValue
        iSic(i)     = iSic@_FillValue
        iSiconc(i)  = iSiconc@_FillValue
        iSiconca(i) = iSiconca@_FillValue
        NameSic(i)  = ""
        iSftlf(i)   = iSftlf@_FillValue
        iSftof(i)   = iSftof@_FillValue
        Namelof(i)  = ""
      end if
    end do
    
  end if
  
  projects = metadata_att_as_array(infoTas, "project")
  ncmip6 = num(projects.eq."CMIP6")
  ncmip5 = num(projects.eq."CMIP5")
  model_cmip6 = new(ncmip6, "string")
  ens_cmip6 = new(ncmip6, "string")
  model_cmip5 = new(ncmip5, "string")
  ens_cmip5 = new(ncmip5, "string")

  tas = read_data(infoTas[0])
  dims = dimsizes(tas)
  SAThiatusCMIP6   = new((/ncmip6, dims(1), dims(2)/), typeof(tas), default_fillvalue(typeof(tas)))
  SAThiatusCMIP6!0 = "ensemble"
  SAThiatusCMIP6&ensemble = ispan(1, ncmip6, 1)
  SATpostCMIP6     = SAThiatusCMIP6
  mSAThiatusCMIP6  = SAThiatusCMIP6

  TShiatusCMIP6    = SAThiatusCMIP6
  TSpostCMIP6      = SAThiatusCMIP6
  mTShiatusCMIP6   = SAThiatusCMIP6

  SAThiatusCMIP5   = new((/ncmip5, dims(1), dims(2)/), typeof(tas), default_fillvalue(typeof(tas)))
  SAThiatusCMIP5!0 = "ensemble"
  SAThiatusCMIP5&ensemble = ispan(1, ncmip5, 1)
  SATpostCMIP5     = SAThiatusCMIP5
  mSAThiatusCMIP5  = SAThiatusCMIP5

  TShiatusCMIP5    = SAThiatusCMIP5
  TSpostCMIP5      = SAThiatusCMIP5
  mTShiatusCMIP5   = SAThiatusCMIP5

  GSAThiatusCMIP6  = new(ncmip6, typeof(tas), default_fillvalue(typeof(tas)))
  GSAThiatusCMIP6!0 = "ensemble"
  GSAThiatusCMIP6&ensemble = ispan(1, ncmip6, 1)
  GSATpostCMIP6    = GSAThiatusCMIP6
  mGSAThiatusCMIP6 = GSAThiatusCMIP6

  GMSThiatusCMIP6  = GSAThiatusCMIP6
  GMSTpostCMIP6    = GSAThiatusCMIP6
  mGMSThiatusCMIP6 = GSAThiatusCMIP6

  GSAThiatusCMIP5  = new(ncmip5, typeof(tas), default_fillvalue(typeof(tas)))
  GSAThiatusCMIP5!0 = "ensemble"
  GSAThiatusCMIP5&ensemble = ispan(1, ncmip5, 1)
  GSATpostCMIP5    = GSAThiatusCMIP5
  mGSAThiatusCMIP5 = GSAThiatusCMIP5

  GMSThiatusCMIP5  = GSAThiatusCMIP5
  GMSTpostCMIP5    = GSAThiatusCMIP5
  mGMSThiatusCMIP5 = GSAThiatusCMIP5

  delete(tas)
  delete(dims)

  SAThiatusObs   = NewList("fifo")
  mSAThiatusObs  = NewList("fifo")
  TShiatusObs    = NewList("fifo")
  mTShiatusObs   = NewList("fifo")
  GSAThiatusObs  = NewList("fifo")
  mGSAThiatusObs = NewList("fifo")
  GMSThiatusObs  = NewList("fifo")
  mGMSThiatusObs = NewList("fifo")
  
; =========================================================================
;
;         2. Apply blending
;
; =========================================================================

  if (masking) then
    if (iTasRefGMST@var.eq."tas") then
      tsRefGMST = read_data(infoTas[iTasRefGMST])
    else
      tsRefGMST = read_data(infoTasa[iTasRefGMST])
    end if
    if (iTasRefPattern@var.eq."tas") then
      tsRefPattern = read_data(infoTas[iTasRefPattern])
    else
      tsRefPattern = read_data(infoTasa[iTasRefPattern])
    end if
;    printVarSummary(tsRefGMST)
    tsRefGMST&time = cd_calendar(tsRefGMST&time, -1)
    tsRefPattern&time = cd_calendar(tsRefPattern&time, -1)
  end if
;  if (isvar("ts")) then
;    delete(ts)
;  end if
;  if (isvar("tas")) then
;    delete(tas)
;  end if
;  if (isvar("GMST")) then
;    delete(GMST)
;  end if

  icmip5 = -1
  icmip6 = -1
  do ii = 0, ListCount(infoTas)-1

; 2.0 Skip the entire loop if corresponding tasa exists
    if (isvar("infoTasa")) then
      skip = False
      do j = 1, ListCount(infoTasa)-1
        if (check_info_consistency(infoTas[ii], infoTasa[j])) then
          skip = True
          break
        end if
      end do
      if (skip) then
        continue
      end if
    end if
    
; 2.1 Read tas
    tas = read_data(infoTas[ii])
    tas&time = cd_calendar(tas&time, -1)
    
; 2.2 Convert to anomaly
    if (anomaly) then
      tas = calc_anom(tas, syear_clim, eyear_clim)
    else
      if (any(tas@units.eq.(/"C", "degC", "degrees_C"/))) then
        tas = tas + 273.15
        tas@units = "K"
      end if
    end if

; 2.3 Blending
    if (blending .and. all(infoTas[ii]@project.ne.obs_projects)) then
;   2.3.1 Read tos and sic and infill land grid points
      if (.not.check_info_consistency(infoTas[ii], infoTos[iTos(ii)])) then
        error_msg("f", DIAG_SCRIPT, "", infoTas[ii]@dataset+" "+infoTas[ii]@ensemble+" doesn't match with tos.")
      end if
      tos = read_data(infoTos[iTos(ii)])
      tos&time = cd_calendar(tos&time, -1)
      if (NameSic(ii).eq."sic") then
        if (.not.check_info_consistency(infoTas[ii], infoSic[iSic(ii)])) then
          error_msg("f", DIAG_SCRIPT, "", infoTas[ii]@dataset+" "+infoTas[ii]@ensemble+" doesn't match with sic.")
        end if
        sic = read_data(infoSic[iSic(ii)])
      elseif (NameSic(ii).eq."siconc") then
        if (.not.check_info_consistency(infoTas[ii], infoSiconc[iSiconc(ii)])) then
          error_msg("f", DIAG_SCRIPT, "", infoTas[ii]@dataset+" "+infoTas[ii]@ensemble+" doesn't match with siconc.")
        end if
        sic = read_data(infoSiconc[iSiconc(ii)])
      elseif (NameSic(ii).eq."siconca") then
        if (.not.check_info_consistency(infoTas[ii], infoSiconca[iSiconca(ii)])) then
          error_msg("f", DIAG_SCRIPT, "", infoTas[ii]@dataset+" "+infoTas[ii]@ensemble+" doesn't match with siconca.")
        end if
        sic = read_data(infoSiconca[iSiconca(ii)])
      end if
      sic&time = cd_calendar(sic&time, -1)
      tos = fillLand(tos)   ; Fill missing values over land
      sic = fillLand(sic)   ; Fill missing values over land

;   2.3.2 Convert tos to anomaly if specified
      if (anomaly) then
        tos = calc_anom(tos, syear_clim, eyear_clim)
      else
        if (any(tos@units.eq.(/"C", "degC", "degrees_C"/))) then
          tos = tos + 273.15
        end if
      end if

;   2.3.3 Scale sic to [0, 1]
      if (max(sic).gt.2.) then
        sic = sic / 100.
        sic@units = ""
      end if
      if (min(sic).lt.0. .or. max(sic).lt.0.2 .or. max(sic).gt.1.01) then
        error_msg("f", DIAG_SCRIPT, "", "something is wrong in sic of " \
                   +infoTas[ii]@dataset+" "+infoTas[ii]@ensemble)
      end if
      sic = where(sic.gt.1., 1., sic)

;   2.3.4 Read sftlf
      if (Namelof(ii).eq."sftlf") then
        sftlf = read_data(infoSftlf[iSftlf(ii)])
      else
        sftof = read_data(infoSftof[iSftof(ii)])
        sftlf = sftof
        if (max(sftlf).gt.2.) then
          sftlf = 100. - sftof
        else
          sftlf = 1. - sftof
        end if
        delete(sftof)
      end if
      
;   2.3.5 Scale sftlf to [0, 1]
      if (max(sftlf).gt.2.) then
        sftlf = sftlf / 100.
        sftlf@units = ""
      end if
      if (min(sftlf).lt.0. .or. max(sftlf).lt.0.5 .or. max(sftlf).gt.1.01) then
        error_msg("f", DIAG_SCRIPT, "", "something is wrong in sftlf")
      end if
      sftlf = where(sftlf.gt.1., 1., sftlf)

;   2.3.6 Apply blending
      ts = blend_tas_tos(tas, tos, sic, sftlf, method_GMST)

      delete(tos)
      delete(sic)
      delete(sftlf)
      
      blended = True
    else
      ts = tas
      blended = False
    end if

    tas@project  = infoTas[ii]@project
    tas@dataset  = infoTas[ii]@dataset
    if (isatt(infoTas[ii], "ensemble")) then
      tas@ensemble = infoTas[ii]@ensemble
    end if
    ts@project  = infoTas[ii]@project
    ts@dataset  = infoTas[ii]@dataset
    if (isatt(infoTas[ii], "ensemble")) then
      ts@ensemble = infoTas[ii]@ensemble
    end if

; =========================================================================
;
;         3. Calculate trend pattern and save
;
; =========================================================================

    if (infoTas[ii]@project.eq."CMIP6") then
      icmip6 = icmip6 + 1
      model_cmip6(icmip6) = infoTas[ii]@dataset
      ens_cmip6(icmip6)   = infoTas[ii]@ensemble
    elseif (infoTas[ii]@project.eq."CMIP5") then
      icmip5 = icmip5 + 1
      model_cmip5(icmip5) = infoTas[ii]@dataset
      ens_cmip5(icmip5)   = infoTas[ii]@ensemble
    end if
    
    etime  = tas&time(dimsizes(tas&time)-1)
    eyear  = etime/100
    emonth = etime - eyear*100
    if (do_post_all) then
      if (annual) then
        if (etime.ge.eyear_post*100+emonth_ann) then
          do_post = True
        else
          do_post = False
        end if
      else
        if (eyear.ge.eyear_post) then
          do_post = True
        else
          do_post = False
        end if
      end if
    else
      do_post = False
    end if

    if (do_pattern) then
;   3.1 Trend pattern, unblended, unmasked
      trend1 = calc_trend(tas, syear_hiatus, eyear_hiatus, method_trend, \
                          annual, smonth_ann, emonth_ann)
      trend2 = trend1
      trend2 = trend2@_FillValue
      if (do_post) then
        trend2 = calc_trend(tas, syear_post, eyear_post, method_trend, \
                            annual, smonth_ann, emonth_ann)
      end if
      if (infoTas[ii]@project.eq."CMIP6") then
        SAThiatusCMIP6(icmip6, :, :) = trend1
        SATpostCMIP6(icmip6, :, :)   = trend2
      elseif (infoTas[ii]@project.eq."CMIP5") then
        SAThiatusCMIP5(icmip5, :, :) = trend1
        SATpostCMIP5(icmip5, :, :)   = trend2
      elseif (any(infoTas[ii]@project.eq.obs_projects)) then
        pushObsList(SAThiatusObs, trend1, infoTas[ii])
      end if
        
;   3.2 Trend pattern, unblended, masked
      tas2 = tas
      if (masking) then
        stime = syear_hiatus*100+1 ;max((/tas&time(0), tsRefPattern&time(0)/))
        etime = eyear_hiatus*100+12 ;min((/tas&time(dimsizes(ts&time)-1), tsRefPattern&time(dimsizes(tsRefPattern&time)-1)/))
        tas2({stime:etime}, :, :) = apply_mask_ref(tas({stime:etime}, :, :), \
                                                   tsRefPattern({stime:etime}, :, :))
      end if
      trend1 = calc_trend(tas2, syear_hiatus, eyear_hiatus, method_trend, \
                          annual, smonth_ann, emonth_ann)
                          delete(tas2)
      if (infoTas[ii]@project.eq."CMIP6") then
        mSAThiatusCMIP6(icmip6, :, :) = trend1
      elseif (infoTas[ii]@project.eq."CMIP5") then
        mSAThiatusCMIP5(icmip5, :, :) = trend1
      elseif (any(infoTas[ii]@project.eq.obs_projects)) then
        pushObsList(mSAThiatusObs, trend1, infoTas[ii])
      end if

      if (blended) then
;   3.3 Trend pattern, blended, unmasked
        trend1 = calc_trend(ts, syear_hiatus, eyear_hiatus, method_trend, \
                            annual, smonth_ann, emonth_ann)
        trend2 = trend2@_FillValue
        if (do_post) then
          trend2 = calc_trend(ts, syear_post, eyear_post, method_trend, \
                              annual, smonth_ann, emonth_ann)
        end if
        if (infoTas[ii]@project.eq."CMIP6") then
          TShiatusCMIP6(icmip6, :, :) = trend1
          TSpostCMIP6(icmip6, :, :)   = trend2
        elseif (infoTas[ii]@project.eq."CMIP5") then
          TShiatusCMIP5(icmip5, :, :) = trend1
          TSpostCMIP5(icmip5, :, :)   = trend2
        elseif (any(infoTas[ii]@project.eq.obs_projects)) then
          pushObsList(TShiatusObs, trend1, infoTas[ii])
        end if
      
;   3.4 Trend pattern, blended, masked
        ts2 = ts
        if (masking) then
          stime = syear_hiatus*100+1 ;max((/ts&time(0), tsRefPattern&time(0)/))
          etime = eyear_hiatus*100+12; min((/ts&time(dimsizes(ts&time)-1), tsRefPattern&time(dimsizes(tsRefPattern&time)-1)/))
          ts2({stime:etime}, :, :) = apply_mask_ref(ts({stime:etime}, :, :), \
                                                   tsRefPattern({stime:etime}, :, :))
        end if
        trend1 = calc_trend(ts2, syear_hiatus, eyear_hiatus, method_trend, \
                            annual, smonth_ann, emonth_ann)
        delete(ts2)
        if (infoTas[ii]@project.eq."CMIP6") then
          mTShiatusCMIP6(icmip6, :, :) = trend1
        elseif (infoTas[ii]@project.eq."CMIP5") then
          mTShiatusCMIP5(icmip5, :, :) = trend1
        elseif (any(infoTas[ii]@project.eq.obs_projects)) then
          pushObsList(mTShiatusObs, trend1, infoTas[ii])
        end if
      end if

      delete(trend1)
      delete(trend2)
    end if
    
; =========================================================================
;
;         4. Calculate GMST trend and save
;
; =========================================================================

;   4.0 Skip if corresponding tasaga exists
    if (isvar("infoTasaga")) then
      skip = False
      do j = 1, ListCount(infoTasaga)-1
        if (check_info_consistency(infoTas[ii], infoTasaga[j])) then
          skip = True
          break
        end if
      end do
      if (skip) then
        delete(tas)
        delete(ts)
        delete(blended)
;        if (isvar("ensObs")) then
;          delete(ensObs)
;        end if
        continue
      end if
    end if

;   4.1 Unblended, unmasked
    GMST = global_mean(tas, "global")
    GMST@project  = infoTas[ii]@project
    GMST@dataset  = infoTas[ii]@dataset
    if (isatt(infoTas[ii], "ensemble")) then
      GMST@ensemble = infoTas[ii]@ensemble
    end if

    trend1 = calc_trend(GMST, syear_hiatus, eyear_hiatus, method_trend, \
                        annual, smonth_ann, emonth_ann)
    trend2 = trend1
    trend2 = trend2@_FillValue
    if (do_post) then
      trend2 = calc_trend(GMST, syear_post, eyear_post, method_trend, \
                          annual, smonth_ann, emonth_ann)
    end if
    if (infoTas[ii]@project.eq."CMIP6") then
      GSAThiatusCMIP6(icmip6) = trend1
      if (.not.ismissing(trend2)) then
        GSATpostCMIP6(icmip6)   = trend2
      end if
    elseif (infoTas[ii]@project.eq."CMIP5") then
      GSAThiatusCMIP5(icmip5) = trend1
      if (.not.ismissing(trend2)) then
        GSATpostCMIP5(icmip5)   = trend2
      end if
    elseif (any(infoTas[ii]@project.eq.obs_projects)) then
      pushObsList(GSAThiatusObs, trend1, infoTas[ii])
    end if
    delete(GMST)
    
;   4.2 Unblended, masked
    tas2 = tas
    if (masking) then
      stime = syear_hiatus*100+1 ;max((/tas&time(0), tsRefGMST&time(0)/))
      etime = eyear_hiatus*100+12 ;min((/tas&time(dimsizes(tas&time)-1), tsRefGMST&time(dimsizes(tsRefGMST&time)-1)/))
      tas2({stime:etime}, :, :) = apply_mask_ref(tas({stime:etime}, :, :), \
                                                 tsRefGMST({stime:etime}, :, :))
    end if
    GMST = global_mean(tas2, method_GMST)
    trend1 = calc_trend(GMST, syear_hiatus, eyear_hiatus, method_trend, \
                        annual, smonth_ann, emonth_ann)
    delete(tas2)
    if (infoTas[ii]@project.eq."CMIP6") then
      mGSAThiatusCMIP6(icmip6) = trend1
    elseif (infoTas[ii]@project.eq."CMIP5") then
      mGSAThiatusCMIP5(icmip5) = trend1
    elseif (any(infoTas[ii]@project.eq.obs_projects)) then
      pushObsList(mGSAThiatusObs, trend1, infoTas[ii])
    end if
    delete(GMST)
    
    if (blended) then    
;   4.3 Blended, unmasked
      GMST = global_mean(ts, "global")
      trend1 = calc_trend(GMST, syear_hiatus, eyear_hiatus, method_trend, \
                          annual, smonth_ann, emonth_ann)
      trend2 = trend2@_FillValue
      if (do_post) then
        trend2 = calc_trend(GMST, syear_post, eyear_post, method_trend, \
                            annual, smonth_ann, emonth_ann)
      end if
      if (infoTas[ii]@project.eq."CMIP6") then
        GMSThiatusCMIP6(icmip6) = trend1
        if (.not.ismissing(trend2)) then
          GMSTpostCMIP6(icmip6)   = trend2
        end if
      elseif (infoTas[ii]@project.eq."CMIP5") then
        GMSThiatusCMIP5(icmip5) = trend1
        if (.not.ismissing(trend2)) then
          GMSTpostCMIP5(icmip5)   = trend2
        end if
      elseif (any(infoTas[ii]@project.eq.obs_projects)) then
        pushObsList(GMSThiatusObs, trend1, infoTas[ii])
      end if
      
;   4.4 Blended, masked
      ts2 = ts
      if (masking) then
        stime = syear_hiatus*100+1;max((/ts&time(0), tsRefGMST&time(0)/))
        etime = eyear_hiatus*100+12;min((/ts&time(dimsizes(ts&time)-1), tsRefGMST&time(dimsizes(tsRefGMST&time)-1)/))
        ts2({stime:etime}, :, :) = apply_mask_ref(ts({stime:etime}, :, :), \
                                                  tsRefGMST({stime:etime}, :, :))
      end if
      GMST = global_mean(ts2, method_GMST)
      trend1 = calc_trend(GMST, syear_hiatus, eyear_hiatus, method_trend, \
                          annual, smonth_ann, emonth_ann)
      delete(ts2)
      if (infoTas[ii]@project.eq."CMIP6") then
        mGMSThiatusCMIP6(icmip6) = trend1
      elseif (infoTas[ii]@project.eq."CMIP5") then
        mGMSThiatusCMIP5(icmip5) = trend1
      elseif (any(infoTas[ii]@project.eq.obs_projects)) then
        pushObsList(mGMSThiatusObs, trend1, infoTas[ii])
      end if
      delete(GMST)
    end if
    
;    delete(GMST)
    delete(trend1)
    delete(trend2)

    delete(tas)
    delete(ts)
    delete(blended)
;    if (isvar("ensObs")) then
;      delete(ensObs)
;    end if
    
  end do

  SAThiatusCMIP6@dataset  := model_cmip6
  SAThiatusCMIP6@ensemble := ens_cmip6
  SATpostCMIP6@dataset  := model_cmip6
  SATpostCMIP6@ensemble := ens_cmip6

  mSAThiatusCMIP6@dataset  := model_cmip6
  mSAThiatusCMIP6@ensemble := ens_cmip6
  
  TShiatusCMIP6@dataset  := model_cmip6
  TShiatusCMIP6@ensemble := ens_cmip6
  TSpostCMIP6@dataset  := model_cmip6
  TSpostCMIP6@ensemble := ens_cmip6
  
  mTShiatusCMIP6@dataset  := model_cmip6
  mTShiatusCMIP6@ensemble := ens_cmip6
  
  SAThiatusCMIP5@dataset  := model_cmip5
  SAThiatusCMIP5@ensemble := ens_cmip5
  SATpostCMIP5@dataset  := model_cmip5
  SATpostCMIP5@ensemble := ens_cmip5
  
  mSAThiatusCMIP5@dataset  := model_cmip5
  mSAThiatusCMIP5@ensemble := ens_cmip5
  
  TShiatusCMIP5@dataset  := model_cmip5
  TShiatusCMIP5@ensemble := ens_cmip5
  TSpostCMIP5@dataset  := model_cmip5
  TSpostCMIP5@ensemble := ens_cmip5
  
  mTShiatusCMIP5@dataset  := model_cmip5
  mTShiatusCMIP5@ensemble := ens_cmip5
  
  GSAThiatusCMIP6@dataset  := model_cmip6
  GSAThiatusCMIP6@ensemble := ens_cmip6
  GSATpostCMIP6@dataset  := model_cmip6
  GSATpostCMIP6@ensemble := ens_cmip6
  
  mGSAThiatusCMIP6@dataset  := model_cmip6
  mGSAThiatusCMIP6@ensemble := ens_cmip6
  
  GMSThiatusCMIP6@dataset  := model_cmip6
  GMSThiatusCMIP6@ensemble := ens_cmip6
  GMSTpostCMIP6@dataset  := model_cmip6
  GMSTpostCMIP6@ensemble := ens_cmip6
  
  mGMSThiatusCMIP6@dataset  := model_cmip6
  mGMSThiatusCMIP6@ensemble := ens_cmip6
  
  GSAThiatusCMIP5@dataset  := model_cmip5
  GSAThiatusCMIP5@ensemble := ens_cmip5
  GSATpostCMIP5@dataset  := model_cmip5
  GSATpostCMIP5@ensemble := ens_cmip5
  
  mGSAThiatusCMIP5@dataset  := model_cmip5
  mGSAThiatusCMIP5@ensemble := ens_cmip5
  
  GMSThiatusCMIP5@dataset  := model_cmip5
  GMSThiatusCMIP5@ensemble := ens_cmip5
  GMSTpostCMIP5@dataset  := model_cmip5
  GMSTpostCMIP5@ensemble := ens_cmip5
  
  mGMSThiatusCMIP5@dataset  := model_cmip5
  mGMSThiatusCMIP5@ensemble := ens_cmip5

  if (isvar("infoTasa")) then
    do ii = 0, ListCount(infoTasa)-1


; =========================================================================
;
;         5. Calculate trend pattern of tasa and save
;
; =========================================================================

; 5.1 Read tasa
      ts = read_data(infoTasa[ii])
      ts&time = cd_calendar(ts&time, -1)
      ts@project  = infoTasa[ii]@project
      ts@dataset  = infoTasa[ii]@dataset
      if (isatt(infoTasa[ii], "ensemble")) then
        ts@ensemble = infoTasa[ii]@ensemble
      end if
    
; 5.2 Adjust reference for anomaly in case reference period differs
;      if (anomaly) then ; No effect on trend pattern; affects GMST only
;        ts = calc_anom(ts, syear_clim, eyear_clim)
;      end if

; 5.3 Calculate trend pattern
      if (do_pattern) then
        trend1 = calc_trend(ts, syear_hiatus, eyear_hiatus, method_trend, \
                            annual, smonth_ann, emonth_ann)
        pushObsList(SAThiatusObs, trend1, infoTasa[ii])
        pushObsList(mSAThiatusObs, trend1, infoTasa[ii])
        pushObsList(TShiatusObs, trend1, infoTasa[ii])
        pushObsList(mTShiatusObs, trend1, infoTasa[ii])
        delete(trend1)
      end if
      
; =========================================================================
;
;         6. Calculate GMST trend and save
;
; =========================================================================

; 6.0 Skip if corresponding tasaga exists
      if (isvar("infoTasaga")) then
        skip = False
        do j = 1, ListCount(infoTasaga)-1
          if (check_info_consistency(infoTasa[ii], infoTasaga[j])) then
            skip = True
            break
          end if
        end do
        if (skip) then
          delete(ts)
          continue
        end if
      end if

; 6.1 Calculate GMST and its trend
      GMST = global_mean(ts, method_GMST)
      GMST@project  = infoTasa[ii]@project
      GMST@dataset  = infoTasa[ii]@dataset
      if (isatt(infoTasa[ii], "ensemble")) then
        GMST@ensemble = infoTasa[ii]@ensemble
      end if
      trend1 = calc_trend(GMST, syear_hiatus, eyear_hiatus, method_trend, \
                          annual, smonth_ann, emonth_ann)
      pushObsList(GSAThiatusObs, trend1, infoTasa[ii])
      pushObsList(mGSAThiatusObs, trend1, infoTasa[ii])
      pushObsList(GMSThiatusObs, trend1, infoTasa[ii])
      pushObsList(mGMSThiatusObs, trend1, infoTasa[ii])
      delete(trend1)
      delete(ts)
    end do
  end if

  if (isvar("infoTasaga")) then
    do ii = 0, ListCount(infoTasaga)-1
; =========================================================================
;
;         7. Calculate GMST trend and save
;
; =========================================================================
; 7.1 Read tasaga
      GMST = read_data(infoTasaga[ii])
      GMST&time = cd_calendar(GMST&time, -1)
      GMST@project  = infoTasaga[ii]@project
      GMST@dataset  = infoTasaga[ii]@dataset
      if (isatt(infoTasaga[ii], "ensemble")) then
        GMST@ensemble = infoTasaga[ii]@ensemble
      end if

; 7.2 Calculate GMST trend
      trend1 = calc_trend(GMST, syear_hiatus, eyear_hiatus, method_trend, \
                          annual, smonth_ann, emonth_ann)
      pushObsList(GSAThiatusObs, trend1, infoTasaga[ii])
      pushObsList(mGSAThiatusObs, trend1, infoTasaga[ii])
      pushObsList(GMSThiatusObs, trend1, infoTasaga[ii])
      pushObsList(mGMSThiatusObs, trend1, infoTasaga[ii])
      delete(trend1)
      delete(GMST)
    end do
  end if

; =========================================================================
;
;         8. Ouput files
;
; =========================================================================

  if (do_pattern) then
; 8.1 Model trend patterns
  
    SAThiatusCMIP6@diag_script = DIAG_SCRIPT 
    mSAThiatusCMIP6@diag_script = DIAG_SCRIPT
    TShiatusCMIP6@diag_script = DIAG_SCRIPT
    mTShiatusCMIP6@diag_script = DIAG_SCRIPT

    SAThiatusCMIP6@var = "trend"
    mSAThiatusCMIP6@var = "trend"
    TShiatusCMIP6@var = "trend"
    mTShiatusCMIP6@var = "trend"

    SATpostCMIP6@diag_script = DIAG_SCRIPT 
    TSpostCMIP6@diag_script = DIAG_SCRIPT

    SATpostCMIP6@var = "trend"
    TSpostCMIP6@var = "trend"

    nc_filename = work_dir + "trendPattern_unblended_unmasked_hiatus_cmip6.nc"
    nc_outfile = ncdf_write(SAThiatusCMIP6, nc_filename)
    nc_filename = work_dir + "trendPattern_unblended_unmasked_post_cmip6.nc"
    nc_outfile = ncdf_write(SATpostCMIP6, nc_filename)
    nc_filename = work_dir + "trendPattern_unblended_masked_hiatus_cmip6.nc"
    nc_outfile = ncdf_write(mSAThiatusCMIP6, nc_filename)
    nc_filename = work_dir + "trendPattern_blended_unmasked_hiatus_cmip6.nc"
    nc_outfile = ncdf_write(TShiatusCMIP6, nc_filename)
    nc_filename = work_dir + "trendPattern_blended_unmasked_post_cmip6.nc"
    nc_outfile = ncdf_write(TSpostCMIP6, nc_filename)
    nc_filename = work_dir + "trendPattern_blended_masked_hiatus_cmip6.nc"
    nc_outfile = ncdf_write(mTShiatusCMIP6, nc_filename)

    SAThiatusCMIP5@diag_script = DIAG_SCRIPT
    mSAThiatusCMIP5@diag_script = DIAG_SCRIPT
    TShiatusCMIP5@diag_script = DIAG_SCRIPT
    mTShiatusCMIP5@diag_script = DIAG_SCRIPT

    SAThiatusCMIP5@var = "trend"
    mSAThiatusCMIP5@var = "trend"
    TShiatusCMIP5@var = "trend"
    mTShiatusCMIP5@var = "trend"

    SATpostCMIP5@diag_script = DIAG_SCRIPT
    TSpostCMIP5@diag_script = DIAG_SCRIPT
  
    SATpostCMIP5@var = "trend"
    TSpostCMIP5@var = "trend"

    nc_filename = work_dir + "trendPattern_unblended_unmasked_hiatus_cmip5.nc"
    nc_outfile = ncdf_write(SAThiatusCMIP5, nc_filename)
    nc_filename = work_dir + "trendPattern_unblended_unmasked_post_cmip5.nc"
    nc_outfile = ncdf_write(SATpostCMIP5, nc_filename)
    nc_filename = work_dir + "trendPattern_unblended_masked_hiatus_cmip5.nc"
    nc_outfile = ncdf_write(mSAThiatusCMIP5, nc_filename)
    nc_filename = work_dir + "trendPattern_blended_unmasked_hiatus_cmip5.nc"
    nc_outfile = ncdf_write(TShiatusCMIP5, nc_filename)
    nc_filename = work_dir + "trendPattern_blended_unmasked_post_cmip5.nc"
    nc_outfile = ncdf_write(TSpostCMIP5, nc_filename)
    nc_filename = work_dir + "trendPattern_blended_masked_hiatus_cmip5.nc"
    nc_outfile = ncdf_write(mTShiatusCMIP5, nc_filename)
  end if
  
; 8.2 Model GMST/GSAT trends
  
  GSAThiatusCMIP6@diag_script = DIAG_SCRIPT 
  mGSAThiatusCMIP6@diag_script = DIAG_SCRIPT
  GMSThiatusCMIP6@diag_script = DIAG_SCRIPT
  mGMSThiatusCMIP6@diag_script = DIAG_SCRIPT

  GSAThiatusCMIP6@var = "trend"
  mGSAThiatusCMIP6@var = "trend"
  GMSThiatusCMIP6@var = "trend"
  mGMSThiatusCMIP6@var = "trend"

  GSATpostCMIP6@diag_script = DIAG_SCRIPT 
  GMSTpostCMIP6@diag_script = DIAG_SCRIPT

  GSATpostCMIP6@var = "trend"
  GMSTpostCMIP6@var = "trend"

  GSAThiatusCMIP6@blended = False
  GSAThiatusCMIP6@masked = False
  
  mGSAThiatusCMIP6@blended = False
  mGSAThiatusCMIP6@masked = True
  mGSAThiatusCMIP6@mask_reference = masking_dataset_GMST

  GMSThiatusCMIP6@blended = True
  GMSThiatusCMIP6@blend_method = method_GMST
  GMSThiatusCMIP6@masked = False

  mGMSThiatusCMIP6@blended = True
  mGMSThiatusCMIP6@blend_method = method_GMST
  mGMSThiatusCMIP6@masked = True
  mGMSThiatusCMIP6@mask_reference = masking_dataset_GMST

  GSATpostCMIP6@blended = False
  GSATpostCMIP6@masked = False

  GMSTpostCMIP6@blended = True
  GMSTpostCMIP6@blend_method = method_GMST
  GMSTpostCMIP6@masked = False
  
  nc_filename = work_dir + "trendGMST_unblended_unmasked_hiatus_cmip6.nc"
  nc_outfile = ncdf_write(GSAThiatusCMIP6, nc_filename)
  nc_filename = work_dir + "trendGMST_unblended_unmasked_post_cmip6.nc"
  nc_outfile = ncdf_write(GSATpostCMIP6, nc_filename)
  nc_filename = work_dir + "trendGMST_unblended_masked_hiatus_cmip6.nc"
  nc_outfile = ncdf_write(mGSAThiatusCMIP6, nc_filename)
  nc_filename = work_dir + "trendGMST_blended_unmasked_hiatus_cmip6.nc"
  nc_outfile = ncdf_write(GMSThiatusCMIP6, nc_filename)
  nc_filename = work_dir + "trendGMST_blended_unmasked_post_cmip6.nc"
  nc_outfile = ncdf_write(GMSTpostCMIP6, nc_filename)
  nc_filename = work_dir + "trendGMST_blended_masked_hiatus_cmip6.nc"
  nc_outfile = ncdf_write(mGMSThiatusCMIP6, nc_filename)

  GSAThiatusCMIP5@diag_script = DIAG_SCRIPT
  mGSAThiatusCMIP5@diag_script = DIAG_SCRIPT
  GMSThiatusCMIP5@diag_script = DIAG_SCRIPT
  mGMSThiatusCMIP5@diag_script = DIAG_SCRIPT

  GSAThiatusCMIP5@var = "trend"
  mGSAThiatusCMIP5@var = "trend"
  GMSThiatusCMIP5@var = "trend"
  mGMSThiatusCMIP5@var = "trend"

  GSATpostCMIP5@diag_script = DIAG_SCRIPT
  GMSTpostCMIP5@diag_script = DIAG_SCRIPT
  
  GSATpostCMIP5@var = "trend"
  GMSTpostCMIP5@var = "trend"

  GSAThiatusCMIP5@blended = False
  GSAThiatusCMIP5@masked = False
  
  mGSAThiatusCMIP5@blended = False
  mGSAThiatusCMIP5@masked = True
  mGSAThiatusCMIP5@mask_reference = masking_dataset_GMST

  GMSThiatusCMIP5@blended = True
  GMSThiatusCMIP5@blend_method = method_GMST
  GMSThiatusCMIP5@masked = False

  mGMSThiatusCMIP5@blended = True
  mGMSThiatusCMIP5@blend_method = method_GMST
  mGMSThiatusCMIP5@masked = True
  mGMSThiatusCMIP5@mask_reference = masking_dataset_GMST
  
  GSATpostCMIP5@blended = False
  GSATpostCMIP5@masked = False

  GMSTpostCMIP5@blended = True
  GMSTpostCMIP5@blend_method = method_GMST
  GMSTpostCMIP5@masked = False

  nc_filename = work_dir + "trendGMST_unblended_unmasked_hiatus_cmip5.nc"
  nc_outfile = ncdf_write(GSAThiatusCMIP5, nc_filename)
  nc_filename = work_dir + "trendGMST_unblended_unmasked_post_cmip5.nc"
  nc_outfile = ncdf_write(GSATpostCMIP5, nc_filename)
  nc_filename = work_dir + "trendGMST_unblended_masked_hiatus_cmip5.nc"
  nc_outfile = ncdf_write(mGSAThiatusCMIP5, nc_filename)
  nc_filename = work_dir + "trendGMST_blended_unmasked_hiatus_cmip5.nc"
  nc_outfile = ncdf_write(GMSThiatusCMIP5, nc_filename)
  nc_filename = work_dir + "trendGMST_blended_unmasked_post_cmip5.nc"
  nc_outfile = ncdf_write(GMSTpostCMIP5, nc_filename)
  nc_filename = work_dir + "trendGMST_blended_masked_hiatus_cmip5.nc"
  nc_outfile = ncdf_write(mGMSThiatusCMIP5, nc_filename)

; 8.3 Obs trend patterns    
  if (do_pattern) then
    do ii = 0, ListCount(SAThiatusObs)-1
      tr = SAThiatusObs[ii]
      tr@diag_script = DIAG_SCRIPT
      tr@var = "trend"
      nc_filename = work_dir + "trendPattern_unblended_unmasked_hiatus_"+tr@dataset+".nc"
      nc_outfile = ncdf_write(tr, nc_filename)
      delete(tr)
    end do

    do ii = 0, ListCount(mSAThiatusObs)-1
      tr = mSAThiatusObs[ii]
      tr@diag_script = DIAG_SCRIPT
      tr@var = "trend"
      nc_filename = work_dir + "trendPattern_unblended_masked_hiatus_"+tr@dataset+".nc"
      nc_outfile = ncdf_write(tr, nc_filename)
      delete(tr)
    end do

    do ii = 0, ListCount(TShiatusObs)-1
      tr = TShiatusObs[ii]
      tr@diag_script = DIAG_SCRIPT
      tr@var = "trend"
      nc_filename = work_dir + "trendPattern_blended_unmasked_hiatus_"+tr@dataset+".nc"
      nc_outfile = ncdf_write(tr, nc_filename)
      delete(tr)
    end do

    do ii = 0, ListCount(mTShiatusObs)-1
      tr = mTShiatusObs[ii]
      tr@diag_script = DIAG_SCRIPT
      tr@var = "trend"
      nc_filename = work_dir + "trendPattern_blended_masked_hiatus_"+tr@dataset+".nc"
      nc_outfile = ncdf_write(tr, nc_filename)
      delete(tr)
    end do
  end if

; 8.4 Obs GMST/GSAT trends
  do ii = 0, ListCount(GSAThiatusObs)-1
    tr = GSAThiatusObs[ii]
    tr@diag_script = DIAG_SCRIPT
    tr@var = "trend"
    tr@blended = False
    tr@masked = False
    nc_filename = work_dir + "trendGMST_unblended_unmasked_hiatus_"+tr@dataset+".nc"
    nc_outfile = ncdf_write(tr, nc_filename)
    delete(tr)
  end do

  do ii = 0, ListCount(mGSAThiatusObs)-1
    tr = mGSAThiatusObs[ii]
    tr@diag_script = DIAG_SCRIPT
    tr@var = "trend"
    tr@blended = False
    tr@masked = True
    tr@mask_reference = masking_dataset_GMST
    nc_filename = work_dir + "trendGMST_unblended_masked_hiatus_"+tr@dataset+".nc"
    nc_outfile = ncdf_write(tr, nc_filename)
    delete(tr)
  end do

  do ii = 0, ListCount(GMSThiatusObs)-1
    tr = GMSThiatusObs[ii]
    tr@diag_script = DIAG_SCRIPT
    tr@var = "trend"
    tr@blended = True
    tr@blend_method = method_GMST
    tr@masked = False
    nc_filename = work_dir + "trendGMST_blended_unmasked_hiatus_"+tr@dataset+".nc"
    nc_outfile = ncdf_write(tr, nc_filename)
    delete(tr)
  end do

  do ii = 0, ListCount(mGMSThiatusObs)-1
    tr = mGMSThiatusObs[ii]
    tr@diag_script = DIAG_SCRIPT
    tr@var = "trend"
    tr@blended = True
    tr@masked = True
    tr@blend_method = method_GMST
    tr@mask_reference = masking_dataset_GMST
    nc_filename = work_dir + "trendGMST_blended_masked_hiatus_"+tr@dataset+".nc"
    nc_outfile = ncdf_write(tr, nc_filename)
    delete(tr)
  end do

end
