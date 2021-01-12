; This script calculates the NAM/SAM for observations and specified scenarios, and utilizes an
; alternative definition of the NAM/SAM using standardized zonal mean differences.
;
;----Changes required to work on a different platform------
;  This script needs to be run multiple times to create files used by nam_sam.trends.damip.summary.bar.ncl.
;  Options are found at line 24 ("djf" or "jja"), and line 25 ("historical","hist-aer",
;     "hist-GHG","hist-nat","hist-stratO3"), resulting in 10 combinations of settings. This combination of
;     options could be called within a single large do loop if one wants to alter the script.
;
; Other necessary changes:
; - The 3 observations used are specified in lines 33-35, and should be updated to the
;   local paths of these datasets. These datasets should match nam_sam.damip.ncwrite.ncl lines 35-37.

; - Line 37 update to new local path of observational NAM/SAM-metrics file (created with this script)
;
; - Lines 178/180 update to new local path of scenario NAM/SAM-metrics file (created with this script)
;
; - Line 183 update to path of local CMIP6 repository

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/plot/mder.ncl"

load "$diag_scripts/ipccwg1ar6ch3_modes/functions.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")

  var0 = variable_info[0]@short_name
  info_items = select_metadata_by_name(input_file_info, var0)
  datasetnames = metadata_att_as_array(info_items, "dataset")
  dim_MOD = ListCount(info_items)


  ;print(dim_MOD)
  print(var0)
  print(variable_info[0])
  print(variable_info[0]@exp)
  ;exp0 = variable_info[0]@exp
  ;print(exp0)
  dim_MOD = ListCount(info_items)
  dim_VAR = ListCount(variable_info)
  print("var group hahowa")
  print(variable_info[0]@variable_group)



  ;--------------------------
  ; observational datasets
  ;--------------------------
  obs = get_obs_list(info_items)
  if (ListCount(obs) .lt. 1) then
    error_msg("f", DIAG_SCRIPT, "", "this diagnostic needs at least one " + \
              "obs dataset")
  end if
  obs_datasets = metadata_att_as_array(obs,  "dataset")
  dim_obs = dimsizes(obs_datasets)


  ;------------------------------------
  ; CMIP6 datasets: historical - DAMIP
  ;------------------------------------
  atts := True
  atts@project = "CMIP6"
  cmip6 = select_metadata_by_atts(info_items, atts)

  ;checkin checkin

  if (variable_info[0]@variable_group.eq."psl_hist") then
    print("var group kayen")
  end if

  if (variable_info[0]@variable_group.eq."psl_GHG") then
    print("var group machi howa")
  end if

  ; piControl experiment
  ; else if (atts@exp = "piControl") then
    ; piControl_cmip6 = select_metadata_by_atts(cmip6, atts)
    ; piControl_cmip6_datasets = metadata_att_as_array(piControl_cmip6, "dataset")
    ; dim_cmip6_piControl = dimsizes(piControl_cmip6_datasets)
    ; scenario = piControl

  ; historical experiment
  if (variable_info[0]@variable_group.eq."psl_hist") then
     ; historical experiment
     atts@exp = "historical"
     hist_cmip6 = select_metadata_by_atts(cmip6, atts)
     hist_cmip6_datasets = metadata_att_as_array(hist_cmip6, "dataset")
     hist_cmip6_ensembles = metadata_att_as_array(hist_cmip6, "ensemble")
     dim_cmip6_hist = dimsizes(hist_cmip6_datasets)

     ; piControl experiment
     ;atts@exp = "piControl"
     ;piControl_cmip6 = select_metadata_by_atts(cmip6, atts)
     ;piControl_cmip6_datasets = metadata_att_as_array(piControl_cmip6, "dataset")
     ;dim_cmip6_piControl = dimsizes(piControl_cmip6_datasets)
     scenario = "historical"
  end if

  ; hist-stratO3 experiment
  if (variable_info[0]@variable_group.eq."psl_stratO3") then
     atts@exp = "hist-stratO3"
     hist_cmip6 = select_metadata_by_atts(cmip6, atts)
     hist_cmip6_datasets = metadata_att_as_array(hist_cmip6, "dataset")
     hist_cmip6_ensembles = metadata_att_as_array(hist_cmip6, "ensemble")
     dim_cmip6_hist = dimsizes(hist_cmip6_datasets)
     scenario = "hist-stratO3"
  end if

  ; hist-aer experiment
  if (variable_info[0]@variable_group.eq."psl_aer") then
     atts@exp = "hist-aer"
     hist_cmip6 = select_metadata_by_atts(cmip6, atts)
     hist_cmip6_datasets = metadata_att_as_array(hist_cmip6, "dataset")
     hist_cmip6_ensembles = metadata_att_as_array(hist_cmip6, "ensemble")
     dim_cmip6_hist = dimsizes(hist_cmip6_datasets)
     scenario = "hist-aer"
  end if

  ; hist-GHG experiment
  if (variable_info[0]@variable_group.eq."psl_GHG") then
     atts@exp = "hist-GHG"
     hist_cmip6 = select_metadata_by_atts(cmip6, atts)
     hist_cmip6_datasets = metadata_att_as_array(hist_cmip6, "dataset")
     hist_cmip6_ensembles = metadata_att_as_array(hist_cmip6, "ensemble")
     dim_cmip6_hist = dimsizes(hist_cmip6_datasets)
     scenario = "hist-GHG"
  end if

  ; hist-nat experiment
  if (variable_info[0]@variable_group.eq."psl_nat") then
     atts@exp = "hist-nat"
     hist_cmip6 = select_metadata_by_atts(cmip6, atts)
     hist_cmip6_datasets = metadata_att_as_array(hist_cmip6, "dataset")
     hist_cmip6_ensembles = metadata_att_as_array(hist_cmip6, "ensemble")
     dim_cmip6_hist = dimsizes(hist_cmip6_datasets)
     scenario = "hist-nat"
  end if

  ; set start and end year
  syearA = diag_script_info@start_year
  print(syearA)
  eyearA = diag_script_info@end_year
  print(eyearA)
  ntime = (eyearA - syearA + 1) * 12

  ; Create work directory
  out_path = config_user_info@work_dir
  system("mkdir -p " + out_path)

  ; Create output plot directory
  plot_dir = config_user_info@plot_dir
  system("mkdir -p " + plot_dir)

  ; Plot file type
  file_type = config_user_info@output_file_type
  if (ismissing(file_type)) then
    file_type = "ps"
  end if


  seas = diag_script_info@season  ; djf or jja
  ;scenario = atts@exp    ; historical (yes it's not DAMIP), hist-aer, hist-GHG, hist-nat and hist-stratO3
;----------------------------------------------------------
  eyear = 2014   ; 2005 or 2014
  pi=4.*atan(1.0)
  rad=(pi/180.)
;----------------------------------------------------------
; observations section
;

  ;fn = out_path + "nam_nao_sam.alt_def.obs."+seas+".1958_1979-"+eyear+".nc"
     do iobs = 0, dim_obs - 1

        log_info(obs_datasets(iobs))

        ; Read observational data
        arrF := read_data(obs[iobs])
        arrF&time = cd_calendar(arrF&time,1)

        if (iobs.eq.0) then    ; keep for regridding and taylor diagrams calculations
           lat = arrF&lat
           lat!0 = "lat"
           lat&lat = lat
           w = cos(rad*lat)
           w!0 = "lat"
           w&lat = lat
        end if
        do hh = 0,2     ; nam, nao, sam
           if (hh.le.1) then
              arr := arrF
           else
              arr := arrF({197901:},:,:)
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
           if (hh.eq.1) then   ; NAO
              finarr_nao_pc := dim_standardize(dim_standardize(arr_seas(:,{38.72},{350.86}),0) - dim_standardize(arr_seas(:,{64.15},{338.06}),0),0)
              finarr_nao := regCoef_n(finarr_nao_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_nao)
           end if
           if (hh.eq.2) then   ; SAM
              finarr_sam_pc := dim_standardize(dim_standardize(dim_avg(arr_seas(:,{-40},:)),0) - dim_standardize(dim_avg(arr_seas(:,{-65},:)),0),0)
              finarr_sam := regCoef_n(finarr_sam_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_sam)
           end if
        end do
        if (iobs.eq.0) then
           nam_patterns = arr_seas(:dimsizes(obs_datasets)-1,:,:)
           nam_patterns!0 = "E"
           nam_patterns&E = ispan(0,dimsizes(obs_datasets)-1,1)
           nam_patterns = nam_patterns@_FillValue
           sam_patterns = nam_patterns
           nao_patterns = nam_patterns

           nam_timeseries = new((/dimsizes(obs_datasets),dimsizes(finarr_nam_pc)/),float)
           nam_timeseries!0 = "E"
           nam_timeseries&E = nam_patterns&E
           nam_timeseries!1 = "time"
           nam_timeseries&time = ispan(1958,2014,1)
           nam_timeseries = nam_timeseries@_FillValue
           sam_timeseries = nam_timeseries(:,{1979:})
           sam_timeseries!1 = "time2"
           nao_timeseries = nam_timeseries

           nam_pc_trends = nam_timeseries(:,0)
           sam_pc_trends = nam_timeseries(:,0)
           nao_pc_trends = nam_timeseries(:,0)

           nam_tay_stat = new((/dimsizes(obs_datasets),3/),double)
           nam_tay_stat!0 = "E"
           nam_tay_stat&E = nam_patterns&E
           nam_tay_stat!1 = "stat"
           nam_tay_stat&stat = ispan(0,2,1)
           nao_tay_stat = nam_tay_stat
           sam_tay_stat = nam_tay_stat

           nam_patterns(iobs,:,:)  = (/ finarr_nam /)
           nao_patterns(iobs,:,:)  = (/ finarr_nao /)
           sam_patterns(iobs,:,:)  = (/ finarr_sam /)
        else
           nam_patterns(iobs,:,:)  = linint2_Wrap(finarr_nam&lon,finarr_nam&lat,finarr_nam,True,nam_patterns&lon,nam_patterns&lat,0)
           nam_tay_stat(iobs,:) = (/ taylor_stats(nam_patterns(iobs,{20:},:), nam_patterns(0,{20:},:), w({20:}), 0)  /)

           nao_patterns(iobs,:,:)  = linint2_Wrap(finarr_nao&lon,finarr_nao&lat,finarr_nao,True,nao_patterns&lon,nao_patterns&lat,0)
           nao_Flip   := lonFlip(nao_patterns)
           nao_tay_stat(iobs,:) = (/ taylor_stats(nao_Flip(iobs,{20:80},{-90:40}), nao_Flip(0,{20:80},{-90:40}), w({20:80}), 0)  /)

           sam_patterns(iobs,:,:)  = linint2_Wrap(finarr_sam&lon,finarr_sam&lat,finarr_sam,True,sam_patterns&lon,sam_patterns&lat,0)
           sam_tay_stat(iobs,:) = (/ taylor_stats(sam_patterns(iobs,{:-20},:), sam_patterns(0,{:-20},:), w({:-20}), 0)  /)
        end if
        tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_nam_pc)-1,1),finarr_nam_pc,False,True,0)
        nam_pc_trends(iobs) = (/ tttt@slope*dimsizes(finarr_nam_pc)  /)
        tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_nao_pc)-1,1),finarr_nao_pc,False,True,0)
        nao_pc_trends(iobs) = (/ tttt@slope*dimsizes(finarr_nao_pc)  /)
        tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_sam_pc)-1,1),finarr_sam_pc,False,True,0)
        sam_pc_trends(iobs) = (/ tttt@slope*dimsizes(finarr_sam_pc)  /)

        nam_timeseries(iobs,:) = (/ finarr_nam_pc /)
        nao_timeseries(iobs,:) = (/ finarr_nao_pc /)
        sam_timeseries(iobs,:) = (/ finarr_sam_pc /)

        ;print("Done with "+iobs+" our of "+nens+", "+model(iobs))
     end do

     fn = out_path + "nam_nao_sam.alt_def.obs."+seas+".1958_1979-"+eyear+".nc"
     print("fn = " + fn)
     system("rm -f " + fn)
     z = addfile(fn,"c")
     z@source = systemfunc("pwd")+"/"+get_script_name()
     z->nam_patterns = nam_patterns
     z->nao_patterns = nao_patterns
     z->sam_patterns = sam_patterns
     z->nam_timeseries = nam_timeseries
     z->nao_timeseries = nao_timeseries
     z->sam_timeseries = sam_timeseries
     z->nam_tay_stat = nam_tay_stat
     z->nao_tay_stat = nao_tay_stat
     z->sam_tay_stat = sam_tay_stat
     z->nam_trends = nam_pc_trends
     z->nao_trends = nao_pc_trends
     z->sam_trends = sam_pc_trends
     delete(z)

  z = addfile(fn,"r")
  nam_taylor_stats_obs = z->nam_tay_stat
  nao_taylor_stats_obs = z->nao_tay_stat
  sam_taylor_stats_obs = z->sam_tay_stat
  nam_trends_obs = z->nam_trends
  nao_trends_obs = z->nao_trends
  sam_trends_obs = z->sam_trends
  nam_patterns_obs = z->nam_patterns(0,:,:)
  nao_patterns_obs = z->nao_patterns(0,:,:)
  sam_patterns_obs = z->sam_patterns(0,:,:)
;-------------------------------------------------------------------------------------------------------
; NAM/NAO/SAM for historical simulations
;-------------------------------------------------------------
;
  lat := nam_patterns_obs&lat
  lat!0 = "lat"
  lat&lat = lat
  w := cos(rad*lat)  ; weights for Taylor diagram
  w!0 = "lat"
  w&lat = lat

  ;fn = out_path + "nam_nao_sam.alt_def.damip_"+scenario+"."+seas+".1958_1979-"+eyear+".nc"
     nens = dim_cmip6_hist

     print("Number of CMIP6 historical simulations found: " + nens)
     model = hist_cmip6_datasets
     emem  = hist_cmip6_ensembles
     ensemble_assign = new(nens, integer)

     enum = 1
     temp = model(0)
     do gg = 0,nens-1
        if (temp.eq.model(gg)) then   ; does the model name match what's in temp?
           ensemble_assign(gg) = enum ; if so, assign it the same number
        else
           enum = enum+1              ; if not, assign it the next number
           ensemble_assign(gg) = enum
           temp = model(gg)
        end if
     end do
     delete(temp)
     ensemble_assign@models = str_join(model+"/"+emem,",")

     do gg = 0, nens-1
        arrF := read_data(hist_cmip6[gg])
        arrF&time = cd_calendar(arrF&time,1)


        if (gg.ge.1.and.isatt(arrF,"is_all_missing")) then   ;
           continue
        end if

        do hh = 0,2     ; nam, nao, sam
           if (hh.le.1) then
              arr := arrF
           else
              arr := arrF({197901:},:,:)
           end if
           arr = rmMonAnnCycTLL(arr)

           temp := runave_n_Wrap(arr,3,0,0)
           temp(0,:,:) = (/ dim_avg_n(arr(:1,:,:),0) /)
           if (seas.eq."djf") then
              arr_seas := temp(0::12,:,:)
           end if
           if (seas.eq."jja") then
              arr_seas := temp(6::12,:,:)
           end if

           if (hh.eq.0) then   ; NAM
              finarr_nam_pc := dim_standardize(dim_standardize(dim_avg(arr_seas(:,{35},:)),0) - dim_standardize(dim_avg(arr_seas(:,{65},:)),0),0)
              finarr_nam := regCoef_n(finarr_nam_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_nam)
           end if
           if (hh.eq.1) then   ; NAO
              finarr_nao_pc := dim_standardize(dim_standardize(arr_seas(:,{38.72},{350.86}),0) - dim_standardize(arr_seas(:,{64.15},{338.06}),0),0)
              finarr_nao := regCoef_n(finarr_nao_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_nao)
           end if
           if (hh.eq.2) then   ; SAM
              finarr_sam_pc := dim_standardize(dim_standardize(dim_avg(arr_seas(:,{-40},:)),0) - dim_standardize(dim_avg(arr_seas(:,{-65},:)),0),0)
              finarr_sam := regCoef_n(finarr_sam_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_sam)
           end if
        end do

        if (gg.eq.0) then
           nao_patterns_obs_lf = lonFlip(nao_patterns_obs)
           nam_patterns := new((/nens,dimsizes(nam_patterns_obs&lat),dimsizes(nam_patterns_obs&lon)/),typeof(nam_patterns_obs))
           nam_patterns!0 = "E"
           nam_patterns&E = ispan(0,nens-1,1)
           nam_patterns!1 = "lat"
           nam_patterns&lat = nam_patterns_obs&lat
           nam_patterns!2 = "lon"
           nam_patterns&lon = nam_patterns_obs&lon
           nam_patterns = nam_patterns@_FillValue
           sam_patterns := nam_patterns
           nao_patterns := nam_patterns

           nam_timeseries := new((/nens,dimsizes(finarr_nam_pc)/),float)
           nam_timeseries!0 = "E"
           nam_timeseries&E = nam_patterns&E
           nam_timeseries!1 = "time"
           nam_timeseries&time = ispan(1958,eyear,1)
           nam_timeseries = nam_timeseries@_FillValue
           sam_timeseries := nam_timeseries(:,{1979:})
           sam_timeseries!1 = "time2"
           nao_timeseries := nam_timeseries

           nam_pc_trends := nam_timeseries(:,0)
           sam_pc_trends := nam_timeseries(:,0)
           nao_pc_trends := nam_timeseries(:,0)

           nam_tay_stat := new((/nens,3/),double)
           nam_tay_stat!0 = "E"
           nam_tay_stat&E = nam_patterns&E
           nam_tay_stat!1 = "stat"
           nam_tay_stat&stat = ispan(0,2,1)
           nao_tay_stat := nam_tay_stat
           sam_tay_stat := nam_tay_stat

           ensemble_assign!0 = "E"
           ensemble_assign&E = nam_patterns&E
        end if

        nam_patterns(gg,:,:)  = (/ linint2_Wrap(finarr_nam&lon,finarr_nam&lat,finarr_nam,True,nam_patterns&lon,nam_patterns&lat,0) /)
        nam_tay_stat(gg,:) = (/ taylor_stats(nam_patterns(gg,{20:},:), nam_patterns_obs({20:},:), w({20:}), 0)  /)

        nao_patterns(gg,:,:)  = (/ linint2_Wrap(finarr_nao&lon,finarr_nao&lat,finarr_nao,True,nao_patterns&lon,nao_patterns&lat,0) /)
        nao_Flip   := lonFlip(nao_patterns(gg,:,:))
        nao_tay_stat(gg,:) = (/ taylor_stats(nao_Flip({20:80},{-90:40}), nao_patterns_obs_lf({20:80},{-90:40}), w({20:80}), 0)  /)

        sam_patterns(gg,:,:)  = (/ linint2_Wrap(finarr_sam&lon,finarr_sam&lat,finarr_sam,True,sam_patterns&lon,sam_patterns&lat,0) /)
        sam_tay_stat(gg,:) = (/ taylor_stats(sam_patterns(gg,{:-20},:), sam_patterns_obs({:-20},:), w({:-20}), 0)  /)

        tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_nam_pc)-1,1),finarr_nam_pc,False,True,0)
        nam_pc_trends(gg) = (/ tttt@slope*dimsizes(finarr_nam_pc)  /)
        tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_nao_pc)-1,1),finarr_nao_pc,False,True,0)
        nao_pc_trends(gg) = (/ tttt@slope*dimsizes(finarr_nao_pc)  /)
        tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_sam_pc)-1,1),finarr_sam_pc,False,True,0)
        sam_pc_trends(gg) = (/ tttt@slope*dimsizes(finarr_sam_pc)  /)
        ;print(nam_pc_trends(gg)+"  "+nao_pc_trends(gg)+"  "+sam_pc_trends(gg))

        nam_timeseries(gg,:) = (/ finarr_nam_pc /)
        nao_timeseries(gg,:) = (/ finarr_nao_pc /)
        sam_timeseries(gg,:) = (/ finarr_sam_pc /)

        ;print("Done with "+paths(gg))
     end do
     if (scenario.eq."historical") then
        fn = out_path + "nam_nao_sam.alt_def."+scenario+"."+seas+".1958_1979-"+eyear+".nc"
     else
        fn = out_path + "nam_nao_sam.alt_def.damip_"+scenario+"."+seas+".1958_1979-"+eyear+".nc"
     end if
     system("rm -f " + fn)
     z = addfile(fn,"c")
     z@source = systemfunc("pwd")+"/"+get_script_name()
     z->ensemble_assign = ensemble_assign
     z->nam_patterns = nam_patterns
     z->nao_patterns = nao_patterns
     z->sam_patterns = sam_patterns
     z->nam_timeseries = nam_timeseries
     z->nao_timeseries = nao_timeseries
     z->sam_timeseries = sam_timeseries
     z->nam_tay_stat = nam_tay_stat
     z->nao_tay_stat = nao_tay_stat
     z->sam_tay_stat = sam_tay_stat
     z->nam_trends = nam_pc_trends
     z->nao_trends = nao_pc_trends
     z->sam_trends = sam_pc_trends
     delete(z)
end
