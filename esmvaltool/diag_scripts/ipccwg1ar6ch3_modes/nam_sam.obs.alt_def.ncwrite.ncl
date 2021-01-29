load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/plot/mder.ncl"

load "$diag_scripts/ipccwg1ar6ch3_modes/functions.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")

  ; Create work directory
  out_path = config_user_info@work_dir
  system("mkdir -p " + out_path)
  
  var0 = variable_info[0]@short_name
  info_items = select_metadata_by_name(input_file_info, var0)
  datasetnames = metadata_att_as_array(info_items, "dataset")

  ;--------------------------
  ; observational datasets
  ;--------------------------
  ;obs = get_obs_list(info_items)
  atts := True
  atts@project = "OBS"
  obs = select_metadata_by_atts(info_items, atts)
  atts@project = "OBS6"
  obs_tmp = select_metadata_by_atts(info_items, atts)
  do iobs = 0, ListCount(obs_tmp) - 1
    ListAppend(obs, obs_tmp[iobs])
  end do
  atts@project = "obs"
  obs_tmp = select_metadata_by_atts(info_items, atts)
  do iobs = 0, ListCount(obs_tmp) - 1
    ListAppend(obs, obs_tmp[iobs])
  end do
  atts@project = "obs4mips"
  obs_tmp = select_metadata_by_atts(info_items, atts)
  do iobs = 0, ListCount(obs_tmp) - 1
    ListAppend(obs, obs_tmp[iobs])
  end do
  atts@project = "ana4mips"
  obs_tmp = select_metadata_by_atts(info_items, atts)
  do iobs = 0, ListCount(obs_tmp) - 1
    ListAppend(obs, obs_tmp[iobs])
  end do

  if (ListCount(obs) .lt. 1) then
    error_msg("f", DIAG_SCRIPT, "", "this diagnostic needs at least one " + \
              "obs dataset")
  end if
  obs_datasets = metadata_att_as_array(obs,  "dataset")
  dim_obs = dimsizes(obs_datasets)
  
  seas = "djf"
  if (isatt(diag_script_info, "season")) then
    seas = diag_script_info@season
  end if
  seas = str_lower(seas)
  

  eyear = 2019
;----------------------------------------------------------
  pi=4.*atan(1.0)
  rad=(pi/180.)
;----------------------------------------------------------
;
; observations section
;
  
  iobs_datasets = ispan(0, dim_obs-1, 1)
  if (isatt(diag_script_info, "reference_dataset")) then
    iref_obs_datasets = ind(obs_datasets.eq.diag_script_info@reference_dataset)
    iobs_datasets(0) = iref_obs_datasets
    iobs_datasets(iref_obs_datasets) = 0
  end if

  do ii = 0, dim_obs - 1
    iobs = iobs_datasets(ii)
    
    log_info(obs_datasets(iobs))

    ; Read observational data
    arrF := read_data(obs[iobs])
    arrF&time = cd_calendar(arrF&time,1)
    
    printVarSummary(arrF)

    if (ii.eq.0) then    ; keep for regridding and taylor diagrams calculations 
      lat = arrF&lat
      lat!0 = "lat"
      lat&lat = lat
      w = cos(rad*lat)
      w!0 = "lat"
      w&lat = lat
    end if
    do hh = 0,2     ; nam, sam, sam
      if (hh.eq.0) then
        arr := arrF
      end if
      if (hh.eq.1) then
        arr := arrF({197901:199912},:,:)
      end if
      if (hh.eq.2) then
        arr := arrF({200001:},:,:)
      end if
      arr = rmMonAnnCycTLL(arr)
      temp3 := runave_n_Wrap(arr,3,0,0)
      temp3(0,:,:) = (/ dim_avg_n(arr(:1,:,:),0) /)
      if (seas.eq."djf") then
        arr_seas := temp3(0::12,:,:)
      end if
      if (seas.eq."jja") then
        arr_seas := temp3(6::12,:,:)
      end if

      if (hh.eq.0) then   ; NAM
        finarr_nam_pc := dim_standardize(dim_standardize(dim_avg(arr_seas(:,{35},:)),0) - dim_standardize(dim_avg(arr_seas(:,{65},:)),0),0)
        finarr_nam := regCoef_n(finarr_nam_pc,arr_seas,0,0)
        copy_VarMeta(arr_seas(0,:,:),finarr_nam)
      end if
      if (hh.eq.1) then   ; SAM
        finarr_sam_pc0 := dim_standardize(dim_standardize(dim_avg(arr_seas(:,{-40},:)),0) - dim_standardize(dim_avg(arr_seas(:,{-65},:)),0),0)
        finarr_sam0 := regCoef_n(finarr_sam_pc0,arr_seas,0,0)
        copy_VarMeta(arr_seas(0,:,:),finarr_sam0)
      end if
      if (hh.eq.2) then   ; SAM
        finarr_sam_pc1 := dim_standardize(dim_standardize(dim_avg(arr_seas(:,{-40},:)),0) - dim_standardize(dim_avg(arr_seas(:,{-65},:)),0),0)
        finarr_sam1 := regCoef_n(finarr_sam_pc1,arr_seas,0,0)
        copy_VarMeta(arr_seas(0,:,:),finarr_sam1)
      end if
    end do
    if (ii.eq.0) then
      E := ispan(0, dim_obs-1, 1)
      E@datasets = str_join(datasetnames, " ")
      nam_patterns = arr_seas(:dim_obs-1,:,:)
      nam_patterns!0 = "E"
      nam_patterns&E = E
      nam_patterns = nam_patterns@_FillValue
      sam_patterns0 = nam_patterns
      sam_patterns1 = nam_patterns

      nam_timeseries = new((/dim_obs,dimsizes(finarr_nam_pc)/),float)
      nam_timeseries!0 = "E"
      nam_timeseries&E = nam_patterns&E
      nam_timeseries!1 = "time"
      nam_timeseries&time = ispan(1958,eyear,1) 
      nam_timeseries = nam_timeseries@_FillValue
      sam_timeseries0 = nam_timeseries(:,{1979:1999})
      sam_timeseries0!1 = "time2"
      sam_timeseries1 = nam_timeseries(:,{2000:})
      sam_timeseries1!1 = "time3"

      nam_pc_trends = nam_timeseries(:,0)
      sam_pc_trends0 = nam_timeseries(:,0)
      sam_pc_trends1 = nam_timeseries(:,0)
      
      nam_tay_stat = new((/dim_obs,3/),double)
      nam_tay_stat!0 = "E"
      nam_tay_stat&E = nam_patterns&E
      nam_tay_stat!1 = "stat"
      nam_tay_stat&stat = ispan(0,2,1)
      sam_tay_stat0 = nam_tay_stat
      sam_tay_stat1 = nam_tay_stat

      nam_patterns(ii,:,:)  = (/ finarr_nam /)
      sam_patterns0(ii,:,:)  = (/ finarr_sam0 /)
      sam_patterns1(i,:,:)  = (/ finarr_sam1 /)
    else
;      nam_patterns(ii,:,:)  = linint2_Wrap(finarr_nam&lon,finarr_nam&lat,finarr_nam,True,nam_patterns&lon,nam_patterns&lat,0)
      nam_patterns(ii,:,:)  = (/ finarr_nam /)
      nam_tay_stat(ii,:) = (/ taylor_stats(nam_patterns(ii,{20:},:), nam_patterns(0,{20:},:), w({20:}), 0)  /)   
  
;      sam_patterns0(ii,:,:)  = linint2_Wrap(finarr_sam0&lon,finarr_sam0&lat,finarr_sam0,True,sam_patterns0&lon,sam_patterns0&lat,0)
      sam_patterns0(ii,:,:) = (/ finarr_sam0 /)
      sam_tay_stat0(ii,:) = (/ taylor_stats(sam_patterns0(ii,{:-20},:), sam_patterns0(0,{:-20},:), w({:-20}), 0)  /)         

;      sam_patterns1(ii,:,:)  = linint2_Wrap(finarr_sam1&lon,finarr_sam1&lat,finarr_sam1,True,sam_patterns1&lon,sam_patterns1&lat,0)
      sam_patterns1(ii,:,:) = (/ finarr_sam1 /)
      sam_tay_stat1(ii,:) = (/ taylor_stats(sam_patterns1(ii,{:-20},:), sam_patterns1(0,{:-20},:), w({:-20}), 0)  /)   
    end if             
    tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_nam_pc)-1,1),finarr_nam_pc,False,True,0)
    nam_pc_trends(ii) = (/ tttt@slope*dimsizes(finarr_nam_pc)  /)            
    tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_sam_pc0)-1,1),finarr_sam_pc0,False,True,0)
    sam_pc_trends0(ii) = (/ tttt@slope*dimsizes(finarr_sam_pc0)  /) 
    tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_sam_pc1)-1,1),finarr_sam_pc1,False,True,0)
    sam_pc_trends1(ii) = (/ tttt@slope*dimsizes(finarr_sam_pc1)  /)  

    nam_timeseries(ii,:) = (/ finarr_nam_pc /)
    sam_timeseries0(ii,:) = (/ finarr_sam_pc0 /)
    sam_timeseries1(ii,:) = (/ finarr_sam_pc1 /)
    
    print("Done with "+ii+" out of "+dim_obs)
  end do

  fn = out_path + "nam_sam.alt_def.obs."+seas+".1958_1979_2000-2019.nc"
  system("rm -f " + fn)
  z = addfile(fn,"c")
  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->nam_patterns  = nam_patterns
  z->sam_patterns0 = sam_patterns0
  z->sam_patterns1 = sam_patterns1
  z->nam_timeseries  = nam_timeseries
  z->sam_timeseries0 = sam_timeseries0
  z->sam_timeseries1 = sam_timeseries1
  z->nam_tay_stat  = nam_tay_stat
  z->sam_tay_stat0 = sam_tay_stat0
  z->sam_tay_stat1 = sam_tay_stat1
  z->nam_trends    = nam_pc_trends
  z->sam_trends0   = sam_pc_trends0
  z->sam_trends1   = sam_pc_trends1
  delete(z)

end

