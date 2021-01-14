; This script calculates the NAM/SAM for observations, but uses
; the piControl to calculate the EOF pattern, and then projects the DAMIP runs
; onto this pattern when creating the PCTS.
;
;----Changes required to work on a different platform------
;  This script needs to be run multiple times to create files used by nam_sam.trends.damip.summary.bar.ncl.
;  Options are found at line 26 ("djf" or "jja"), and line 27 ("historical","hist-aer",
;     "hist-GHG","hist-nat","hist-stratO3"), resulting in 10 combinations of settings. This combination of
;     options could be called within a single large do loop if one wants to alter the script.
;
; Other necessary changes:
; - The 3 observations used are specified in lines 35-37, and should be updated to the
;   local paths of these datasets. These datasets should match nam_sam.damip.alt_def.ncwrite.ncl lines 33-35.

; - Line 39 update to new local path of observational NAM/SAM-metrics file (created with this script)
;
; - Line 225 update to new local path of scenario NAM/SAM-metrics file (created with this script)
;
; - Line 227 update to path of local CMIP6 repository
;
; - Line 249 update to path of local CMIP6 piControl repository
;
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

  ;if (variable_info[0]@variable_group.eq."psl_hist") then
  ;  print("var group kayen")
  ;end if

  ;if (variable_info[0]@variable_group.eq."psl_GHG") then
  ;  print("var group machi howa")
  ;end if

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
     atts@exp = "piControl"
     piControl_cmip6 = select_metadata_by_atts(cmip6, atts)
     piControl_cmip6_datasets = metadata_att_as_array(piControl_cmip6, "dataset")
     dim_cmip6_piControl = dimsizes(piControl_cmip6_datasets)
     scenario = "historical"
  end if

  ; hist-stratO3 experiment
  if (variable_info[0]@variable_group.eq."psl_stratO3") then
     atts@exp = "hist-stratO3"
     hist_cmip6 = select_metadata_by_atts(cmip6, atts)
     hist_cmip6_datasets = metadata_att_as_array(hist_cmip6, "dataset")
     hist_cmip6_ensembles = metadata_att_as_array(hist_cmip6, "ensemble")
     dim_cmip6_hist = dimsizes(hist_cmip6_datasets)

     ; piControl experiment
     atts@exp = "piControl"
     piControl_cmip6 = select_metadata_by_atts(cmip6, atts)
     piControl_cmip6_datasets = metadata_att_as_array(piControl_cmip6, "dataset")
     dim_cmip6_piControl = dimsizes(piControl_cmip6_datasets)
     scenario = "hist-stratO3"
  end if

  ; hist-aer experiment
  if (variable_info[0]@variable_group.eq."psl_aer") then
     atts@exp = "hist-aer"
     hist_cmip6 = select_metadata_by_atts(cmip6, atts)
     hist_cmip6_datasets = metadata_att_as_array(hist_cmip6, "dataset")
     hist_cmip6_ensembles = metadata_att_as_array(hist_cmip6, "ensemble")
     dim_cmip6_hist = dimsizes(hist_cmip6_datasets)

     ; piControl experiment
     atts@exp = "piControl"
     piControl_cmip6 = select_metadata_by_atts(cmip6, atts)
     piControl_cmip6_datasets = metadata_att_as_array(piControl_cmip6, "dataset")
     dim_cmip6_piControl = dimsizes(piControl_cmip6_datasets)
     scenario = "hist-aer"
  end if

  ; hist-GHG experiment
  if (variable_info[0]@variable_group.eq."psl_GHG") then
     atts@exp = "hist-GHG"
     hist_cmip6 = select_metadata_by_atts(cmip6, atts)
     hist_cmip6_datasets = metadata_att_as_array(hist_cmip6, "dataset")
     hist_cmip6_ensembles = metadata_att_as_array(hist_cmip6, "ensemble")
     dim_cmip6_hist = dimsizes(hist_cmip6_datasets)

     ; piControl experiment
     atts@exp = "piControl"
     piControl_cmip6 = select_metadata_by_atts(cmip6, atts)
     piControl_cmip6_datasets = metadata_att_as_array(piControl_cmip6, "dataset")
     dim_cmip6_piControl = dimsizes(piControl_cmip6_datasets)
     scenario = "hist-GHG"
  end if

  ; hist-nat experiment
  if (variable_info[0]@variable_group.eq."psl_nat") then
     atts@exp = "hist-nat"
     hist_cmip6 = select_metadata_by_atts(cmip6, atts)
     hist_cmip6_datasets = metadata_att_as_array(hist_cmip6, "dataset")
     hist_cmip6_ensembles = metadata_att_as_array(hist_cmip6, "ensemble")
     dim_cmip6_hist = dimsizes(hist_cmip6_datasets)

     ; piControl experiment
     atts@exp = "piControl"
     piControl_cmip6 = select_metadata_by_atts(cmip6, atts)
     piControl_cmip6_datasets = metadata_att_as_array(piControl_cmip6, "dataset")
     dim_cmip6_piControl = dimsizes(piControl_cmip6_datasets)
     scenario = "hist-nat"
  end if

  ;atts@exp = "historical"
  ;hist_cmip6 = select_metadata_by_atts(cmip6, atts)
  ;hist_cmip6_datasets = metadata_att_as_array(hist_cmip6, "dataset")
  ;hist_cmip6_ensembles = metadata_att_as_array(hist_cmip6, "ensemble")
  ;dim_cmip6_hist = dimsizes(hist_cmip6_datasets)


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
  pi=4.*atan(1.0)
  rad=(pi/180.)
  eyear = 2014    ; 2005 or 2014
;----------------------------------------------------------
; observations section
;


  ;fn = "/project/cas/asphilli/IPCC-AR6-CH3_data/nam_nao_sam.obs."+seas+".1958_1979-"+eyear+".nc"
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

           arr_seas_CW := SqrtCosWeight(arr_seas)
           arr_seas_CW_LF := lonFlip(arr_seas_CW) ; for NAO

           if (hh.eq.0) then   ; NAM
              evecv := eofunc(arr_seas_CW({lat|20:},lon|:,time|:),2,75)
              pcts  := eofunc_ts(arr_seas_CW({lat|20:},lon|:,time|:),evecv,False)

              finarr_nam_pcvar := evecv@pcvar(0)
              finarr_nam_pc := dim_standardize(pcts(0,:),0)
              finarr_nam := regCoef_n(finarr_nam_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_nam)

              if (.not.ismissing(finarr_nam({85},{5}))) then
                 if (finarr_nam({85},{5}).ge.0) then  ; arbitrary attempt to make all plots have the same sign..
                    finarr_nam = finarr_nam*-1.
                    finarr_nam_pc = finarr_nam_pc*-1.
                 end if
              end if
           end if
           if (hh.eq.1) then   ; NAO
              evecv := eofunc(arr_seas_CW_LF({lat|20:80},{lon|-90:40},time|:),2,75)
              pcts  := eofunc_ts(arr_seas_CW_LF({lat|20:80},{lon|-90:40},time|:),evecv,False)

              finarr_nao_pcvar := evecv@pcvar(0)
              finarr_nao_pc := dim_standardize(pcts(0,:),0)
              finarr_nao := regCoef_n(finarr_nao_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_nao)

              if (.not.ismissing(finarr_nao({70},{350}))) then
                 if (finarr_nao({70},{350}).ge.0) then  ; arbitrary attempt to make all plots have the same sign..
                    finarr_nao = finarr_nao*-1.
                    finarr_nao_pc = finarr_nao_pc*-1.
                 end if
              end if
           end if
           if (hh.eq.2) then   ; SAM
              evecv := eofunc(arr_seas_CW({lat|:-20},lon|:,time|:),2,75)
              pcts  := eofunc_ts(arr_seas_CW({lat|:-20},lon|:,time|:),evecv,False)

              finarr_sam_pcvar := evecv@pcvar(0)
              finarr_sam_pc := dim_standardize(pcts(0,:),0)
              finarr_sam := regCoef_n(finarr_sam_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_sam)

              if (.not.ismissing(finarr_sam({-85},{5}))) then
                 if (finarr_sam({-85},{5}).ge.0) then  ; arbitrary attempt to make all plots have the same sign..
                    finarr_sam = finarr_sam*-1.
                    finarr_sam_pc = finarr_sam_pc*-1.
                 end if
              end if
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
           nam_timeseries&time = ispan(1958,eyear,1)
           nam_timeseries = nam_timeseries@_FillValue
           sam_timeseries = nam_timeseries(:,{1979:})
           sam_timeseries!1 = "time2"
           nao_timeseries = nam_timeseries

           nam_pcvari = nam_timeseries(:,0)
           sam_pcvari = nam_pcvari
           nao_pcvari = nam_pcvari
           nam_pc_trends = nam_pcvari
           sam_pc_trends = nam_pcvari
           nao_pc_trends = nam_pcvari

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

        nam_pcvari(iobs) = (/ finarr_nam_pcvar /)
        nao_pcvari(iobs) = (/ finarr_nao_pcvar /)
        sam_pcvari(iobs) = (/ finarr_sam_pcvar /)
        ; print("Done with "+paths_obs(iobs))
     end do

     fn = out_path + "nam_nao_sam.obs."+seas+".1958_1979-"+eyear+".nc"
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
     z->nam_pc_trends = nam_pc_trends
     z->nao_pc_trends = nao_pc_trends
     z->sam_pc_trends = sam_pc_trends
     z->nam_pcvari    = nam_pcvari
     z->nao_pcvari    = nao_pcvari
     z->sam_pcvari    = sam_pcvari
     delete(z)

  z = addfile(fn,"r")
  nam_taylor_stats_obs = z->nam_tay_stat
  nao_taylor_stats_obs = z->nao_tay_stat
  sam_taylor_stats_obs = z->sam_tay_stat
  nam_trends_obs = z->nam_pc_trends
  nao_trends_obs = z->nao_pc_trends
  sam_trends_obs = z->sam_pc_trends
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
;---------------------------------------------------------------------------------------------
  ; fn = "/project/cas/asphilli/IPCC-AR6-CH3_data/nam_nao_sam.damip_"+scenario+"."+seas+".1958_1979-"+eyear+".nc"
;  if (.not.isfilepresent(fn)) then
;     fils = systemfunc("ls /project/data/cmip6/"+scenario+"/Amon/psl/*/*/*/*185001-??????.nc")
;     nens = dimsizes(fils)
;     print("Number of simulations found: "+dimsizes(fils))
;     model = new(nens,string)
;     emem  = model
;     paths := model
;     ensemble_assign = new(nens,integer)

;     do gg = 0,nens-1    ; identify model and the ensemble size
;        tt := str_split(fils(gg),"/")
;        paths(gg) = "/"+str_join(tt(:dimsizes(tt)-2),"/")+"/"
;        fname := tt(dimsizes(tt)-1)
;        fname_split := str_split(fname,"_")
;        model(gg) = fname_split(2)
;        emem(gg)  = fname_split(4)
;     end do

;    print(model)

;     pictl_paths = new(dimsizes(model),string)
;     pictl_paths = "/project/data/cmip6/piControl/Amon/ts/"+model+"

;     paths_pictl_all := piControl_list_setup("/project/data/cmip6/piControl/Amon/psl/",eyear-1958+1,1,1)
;     print(paths_pictl_all)

;     pictl_models = new(dimsizes(paths_pictl_all),string)
;     do gg = 0,dimsizes(paths_pictl_all)-1
;        tt := str_split(paths_pictl_all(gg),"/")   ; interested in the 6th text string here
;        pictl_models(gg) = tt(6)
;     end do

;     pictl_paths = new(dimsizes(model),string)
;     pictl_syear = pictl_paths
;     pictl_eyear = pictl_paths
;     do gg = 0,dimsizes(model)-1
;        w_ind := ind(model(gg).eq.pictl_models)
;        print("gg = "+gg+", w_ind = "+w_ind+", model = "+model(gg))
;        if (.not.ismissing(w_ind)) then
;            pictl_paths(gg) = paths_pictl_all(w_ind)
;           pictl_syear(gg) = paths_pictl_all@start_year(w_ind)
;           pictl_eyear(gg) = paths_pictl_all@end_year(w_ind)
;        end if
;     end do
;     print(model+"  "+pictl_paths+"  "+pictl_syear+"  "+pictl_eyear)
;---------------------------------------------------------------
     nens = dim_cmip6_hist
     print(nens)

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
        ;print("DAMIP: "+paths(gg)+"*"+model(gg)+"*"+emem(gg)+"*.nc, piControl: "+pictl_paths(gg)+"*.nc")

        ;if (ismissing(pictl_paths(gg)).or.ismissing(paths(gg))) then
        ;   continue
        ;end if

        arrF := read_data(hist_cmip6[gg])
        arrF&time = cd_calendar(arrF&time,1)
        ;if (gg.ge.1.and.isatt(arrF,"is_all_missing")) then   ;
        ;   continue
        ;end if
        arrP := read_data(piControl_cmip6[gg])
        arrP&time = cd_calendar(arrP&time,1)

        if (gg.ge.1.and.isatt(arrP,"is_all_missing")) then
           continue
        end if

        do hh = 0,2     ; nam, nao, sam
           if (hh.le.1) then
              arr := arrF
              piC := arrP
           else
              arr := arrF({197901:},:,:)
              piC := arrP(:dimsizes(arr&time)-1,:,:)
           end if
           arr = rmMonAnnCycTLL(arr)
           piC = rmMonAnnCycTLL(piC)

           temp := runave_n_Wrap(arr,3,0,0)
           temp(0,:,:) = (/ dim_avg_n(arr(:1,:,:),0) /)
           tempP := runave_n_Wrap(piC,3,0,0)
           tempP(0,:,:) = (/ dim_avg_n(piC(:1,:,:),0) /)
           if (seas.eq."djf") then
              arr_seas := temp(0::12,:,:)
              piC_seas := tempP(0::12,:,:)
           end if
           if (seas.eq."jja") then
              arr_seas := temp(6::12,:,:)
              piC_seas := tempP(6::12,:,:)
           end if
           arr_seas_CW := SqrtCosWeight(arr_seas)
           arr_seas_CW_LF := lonFlip(arr_seas_CW) ; for NAO
           piC_seas_CW := SqrtCosWeight(piC_seas)
           piC_seas_CW_LF := lonFlip(piC_seas_CW) ; for NAO

           if (hh.eq.0) then   ; NAM
              evecv := eofunc(piC_seas_CW({lat|20:},lon|:,time|:),2,75)
              pcts  := eofunc_ts(arr_seas_CW({lat|20:},lon|:,time|:),evecv,False)

              finarr_nam_pcvar := evecv@pcvar(0)
              finarr_nam_pc := dim_standardize(pcts(0,:),0)
              finarr_nam := regCoef_n(finarr_nam_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_nam)

              if (.not.ismissing(finarr_nam({85},{5}))) then
                 if (finarr_nam({85},{5}).ge.0) then  ; arbitrary attempt to make all plots have the same sign..
                    finarr_nam = finarr_nam*-1.
                    finarr_nam_pc = finarr_nam_pc*-1.
                 end if
              end if
           end if
           if (hh.eq.1) then   ; NAO
              evecv := eofunc(piC_seas_CW_LF({lat|20:80},{lon|-90:40},time|:),2,75)
              pcts  := eofunc_ts(arr_seas_CW_LF({lat|20:80},{lon|-90:40},time|:),evecv,False)

              finarr_nao_pcvar := evecv@pcvar(0)
              finarr_nao_pc := dim_standardize(pcts(0,:),0)
              finarr_nao := regCoef_n(finarr_nao_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_nao)

              if (.not.ismissing(finarr_nao({70},{350}))) then
                 if (finarr_nao({70},{350}).ge.0) then  ; arbitrary attempt to make all plots have the same sign..
                    finarr_nao = finarr_nao*-1.
                    finarr_nao_pc = finarr_nao_pc*-1.
                 end if
              end if
           end if
           if (hh.eq.2) then   ; SAM
              evecv := eofunc(piC_seas_CW({lat|:-20},lon|:,time|:),2,75)
              pcts  := eofunc_ts(arr_seas_CW({lat|:-20},lon|:,time|:),evecv,False)

              finarr_sam_pcvar := evecv@pcvar(0)
              finarr_sam_pc := dim_standardize(pcts(0,:),0)
              finarr_sam := regCoef_n(finarr_sam_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_sam)

              if (.not.ismissing(finarr_sam({-85},{5}))) then
                 if (finarr_sam({-85},{5}).ge.0) then  ; arbitrary attempt to make all plots have the same sign..
                    finarr_sam = finarr_sam*-1.
                    finarr_sam_pc = finarr_sam_pc*-1.
                 end if
              end if
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

           nam_pcvari := nam_timeseries(:,0)
           sam_pcvari := nam_pcvari
           nao_pcvari := nam_pcvari
           nam_pc_trends := nam_pcvari
           sam_pc_trends := nam_pcvari
           nao_pc_trends := nam_pcvari

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
;        print(nam_pc_trends(gg)+"  "+nao_pc_trends(gg)+"  "+sam_pc_trends(gg))

        nam_timeseries(gg,:) = (/ finarr_nam_pc /)
        nao_timeseries(gg,:) = (/ finarr_nao_pc /)
        sam_timeseries(gg,:) = (/ finarr_sam_pc /)

        nam_pcvari(gg) = (/ finarr_nam_pcvar /)
        nao_pcvari(gg) = (/ finarr_nao_pcvar /)
        sam_pcvari(gg) = (/ finarr_sam_pcvar /)
        ;print("Done with "+paths(gg))
     end do
     fn = out_path + "nam_nao_sam.damip_"+scenario+"."+seas+".1958_1979-"+eyear+".nc"
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
     z->nam_pc_trends = nam_pc_trends
     z->nao_pc_trends = nao_pc_trends
     z->sam_pc_trends = sam_pc_trends
     z->nam_pcvari    = nam_pcvari
     z->nao_pcvari    = nao_pcvari
     z->sam_pcvari    = sam_pcvari
     delete(z)
end

