; #############################################################################
; Compute NAM/NAO/SAM metrics
; Authors: Lisa Bock (DLR, Germany), Soufiane Karmouche (University of Bremen,
;          Germany), Yu Kosaka (University of Tokyo, Japan) and Adam Phillips
;          (NCAR, U.S.)
; #############################################################################
;
; Description
; Compute NAM/NAO/SAM metrics for observations, CMIP6-historical,
; CMIP5-historical simulations.
; This script creates 1 file each for observations and CMIP6-historical.
;
; Required diag_script_info attributes (diagnostic specific)
;
; Optional diag_script_info attributes (diagnostic specific)
;
; Caveats
;
; Modification history
;   20210116-Yu Kosaka: adapt Adam's updates script to the ESMValTool
;   20200727-Soufiane Karmouche: adapt Adam's scipt to the ESMValTool
;
; #############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/plot/mder.ncl"

load "$diag_scripts/ipccwg1ar6ch3_modes/functions.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")

  var0 = variable_info[0]@short_name
  info_items = select_metadata_by_name(input_file_info, var0)
  datasetnames = metadata_att_as_array(info_items, "dataset")
  dim_MOD = ListCount(info_items)


  ;--------------------------
  ; observational datasets
  ;--------------------------
;  obs = get_obs_list(info_items)
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

  ;--------------------------
  ; CMIP5 datasets
  ;--------------------------
  atts := True
  atts@project = "CMIP5"
  cmip5 = select_metadata_by_atts(info_items, atts)

  ; historical experiment
  atts@exp = "historical"
  hist_cmip5 = select_metadata_by_atts(cmip5, atts)
  atts@exp = "historical-rcp45"
  hist_cmip5_tmp = select_metadata_by_atts(cmip5, atts)
  if (ListCount(hist_cmip5_tmp).ge.1) then
    do icmip = 0, ListCount(hist_cmip5_tmp) - 1
      ListAppend(hist_cmip5, hist_cmip5_tmp[icmip])
    end do
  end if
  delete(hist_cmip5_tmp)
  
  hist_cmip5_datasets = metadata_att_as_array(hist_cmip5, "dataset")
  hist_cmip5_ensembles = metadata_att_as_array(hist_cmip5, "ensemble")
  dim_cmip5_hist = dimsizes(hist_cmip5_datasets)

  ; piControl experiment
  ; atts@exp = "piControl"
  ; piControl_cmip5 = select_metadata_by_atts(cmip5, atts)
  ; piControl_cmip5_datasets = metadata_att_as_array(piControl_cmip5, "dataset")
  ; dim_cmip5_piControl = dimsizes(piControl_cmip5_datasets)

  ;--------------------------
  ; CMIP6 datasets
  ;--------------------------
  atts@project = "CMIP6"
  delete(atts@exp)
  cmip6 = select_metadata_by_atts(info_items, atts)

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

  ; amip experiment
  atts@exp = "amip-hist"
  amip_cmip6 = select_metadata_by_atts(cmip6, atts)
  amip_cmip6_datasets = metadata_att_as_array(amip_cmip6, "dataset")
  amip_cmip6_ensembles = metadata_att_as_array(amip_cmip6, "ensemble")
  dim_cmip6_amip = dimsizes(amip_cmip6_datasets)
  
  ; set start and end year
  syearA = diag_script_info@start_year
  eyearA = diag_script_info@end_year
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


  seas = "djf"    ; djf or jja
;----------------------------------------------------------
  pi=4.*atan(1.0)
  rad=(pi/180.)
;----------------------------------------------------------
  ;--------------------------------------------------------
  ; process observational data
  ;--------------------------------------------------------

    iref_obs_datasets = ind(obs_datasets.eq.diag_script_info@reference_dataset)
    iobs_datasets = ispan(0, dim_obs-1, 1)
    iobs_datasets(0) = iref_obs_datasets
    iobs_datasets(iref_obs_datasets) = 0
    do ii = 0, dim_obs - 1
      iobs = iobs_datasets(ii)
      
        log_info(obs_datasets(iobs))
        
        ; Read observational data
        arrF := read_data(obs[iobs])
        arrF&time = cd_calendar(arrF&time,1)

        if (ii.eq.0) then    ; keep for regridding and taylor diagrams calculations
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
              finarr_namC := escorc_n(finarr_nam_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_nam)
              copy_VarMeta(arr_seas(0,:,:),finarr_namC)

              if (.not.ismissing(finarr_nam({85},{5}))) then
                 if (finarr_nam({85},{5}).ge.0) then  ; arbitrary attempt to make all plots have the same sign..
                    finarr_nam = finarr_nam*-1.
                    finarr_namC = finarr_namC*-1.
                    finarr_nam_pc = finarr_nam_pc*-1.
                 end if
              end if
              reg_fld := finarr_nam
              cor_fld := finarr_namC

              sig := arr_seas(:2,:,:)
              sig = sig@_FillValue
           end if
           if (hh.eq.1) then   ; NAO
              evecv := eofunc(arr_seas_CW_LF({lat|20:80},{lon|-90:40},time|:),2,75)
              pcts  := eofunc_ts(arr_seas_CW_LF({lat|20:80},{lon|-90:40},time|:),evecv,False)

              finarr_nao_pcvar := evecv@pcvar(0)
              finarr_nao_pc := dim_standardize(pcts(0,:),0)
              finarr_nao := regCoef_n(finarr_nao_pc,arr_seas,0,0)
              finarr_naoC := escorc_n(finarr_nao_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_nao)
              copy_VarMeta(arr_seas(0,:,:),finarr_naoC)

              if (.not.ismissing(finarr_nao({70},{350}))) then
                 if (finarr_nao({70},{350}).ge.0) then  ; arbitrary attempt to make all plots have the same sign..
                    finarr_nao = finarr_nao*-1.
                    finarr_naoC = finarr_naoC*-1.
                    finarr_nao_pc = finarr_nao_pc*-1.
                 end if
              end if
              reg_fld := finarr_nao
              cor_fld := finarr_naoC               
           end if
           if (hh.eq.2) then   ; SAM
              evecv := eofunc(arr_seas_CW({lat|:-20},lon|:,time|:),2,75)
              pcts  := eofunc_ts(arr_seas_CW({lat|:-20},lon|:,time|:),evecv,False)

              finarr_sam_pcvar := evecv@pcvar(0)
              finarr_sam_pc := dim_standardize(pcts(0,:),0)
              finarr_sam := regCoef_n(finarr_sam_pc,arr_seas,0,0)
              finarr_samC := escorc_n(finarr_sam_pc,arr_seas,0,0)
              copy_VarMeta(arr_seas(0,:,:),finarr_sam)
              copy_VarMeta(arr_seas(0,:,:),finarr_samC)

              if (.not.ismissing(finarr_sam({-85},{5}))) then
                 if (finarr_sam({-85},{5}).ge.0) then  ; arbitrary attempt to make all plots have the same sign..
                    finarr_sam = finarr_sam*-1.
                    finarr_samC = finarr_samC*-1.
                    finarr_sam_pc = finarr_sam_pc*-1.
                 end if
              end if
              reg_fld := finarr_sam
              cor_fld := finarr_samC 
           end if

;           print("Significance by rtest plus false rate detection")
           prob_rtest=(/ rtest(cor_fld,dimsizes(arr_seas&time),0) /)
           copy_VarMeta(arr_seas(0,:,:),prob_rtest)
           prob_rtest=where(arr_seas(0,:,:).eq.arr_seas@_FillValue,prob_rtest@_FillValue,prob_rtest)

;           npts=num(.not.ismissing(ndtooned(arr_seas(0,:,:))))
; 
;           siglev = 0.05
;
;           frac = fspan(1, npts, npts)
;           scale=1
;           frac = (/ (frac/npts)* scale * siglev /)
;           prob_rtest_fdr=prob_rtest
;
;           prob_rtest_1d=ndtooned(prob_rtest)
;           prob_rtest_1d_nomiss=prob_rtest_1d(ind(.not.ismissing(prob_rtest_1d)))
; 
;           sig_fdr = prob_rtest_1d_nomiss
;           ip = dim_pqsort(sig_fdr, 2)
;           sig_fdr = where(sig_fdr .gt. frac, sig_fdr@_FillValue, sig_fdr)
;           if (all(ismissing(sig_fdr))) then
;             pval_fdr = 0.
;           else
;             pval_fdr = (/ max(sig_fdr) /)
;           end if
;           prob_rtest_fdr =  (/ where(prob_rtest.gt. pval_fdr, 0, 1.) /) 
;           sig(hh,:,:) = (/ prob_rtest_fdr /)
;           delete([/prob_rtest_fdr,pval_fdr,sig_fdr,ip,prob_rtest_1d,frac,scale,siglev,npts,prob_rtest,reg_fld,cor_fld,prob_rtest_1d_nomiss/])

           siglev = 0.10
           sig(hh,:,:) = where(prob_rtest.gt.siglev, 0., 1.)
           delete([/siglev,prob_rtest,reg_fld,cor_fld/])
        end do

        if (ii.eq.0) then
           fn = out_path + "nam_nao_sam.obs."+syearA+"-"+eyearA+".nc"
           print("fn = " + fn)
           system("rm -f " + fn)
           z = addfile(fn,"c")
           z@source = systemfunc("pwd")+"/"+get_script_name()
           sig!0 = "stat"
           sig&stat = ispan(0,2,1)  
           z->pattern_significance = sig     ; represents significance for NAM, NAO and SAM obs
           
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
           ;nam_timeseries&time = nam_patterns&time
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

           nam_patterns(ii,:,:)  = (/ finarr_nam /)
           nao_patterns(ii,:,:)  = (/ finarr_nao /)
           sam_patterns(ii,:,:)  = (/ finarr_sam /)
        else
           nam_patterns(ii,:,:)  = linint2_Wrap(finarr_nam&lon,finarr_nam&lat,finarr_nam,True,nam_patterns&lon,nam_patterns&lat,0)
           nam_tay_stat(ii,:) = (/ taylor_stats(nam_patterns(ii,{20:},:), nam_patterns(0,{20:},:), w({20:}), 0)  /)

           nao_patterns(ii,:,:)  = linint2_Wrap(finarr_nao&lon,finarr_nao&lat,finarr_nao,True,nao_patterns&lon,nao_patterns&lat,0)
           nao_Flip   := lonFlip(nao_patterns)
           nao_tay_stat(ii,:) = (/ taylor_stats(nao_Flip(ii,{20:80},{-90:40}), nao_Flip(0,{20:80},{-90:40}), w({20:80}), 0)  /)

           sam_patterns(ii,:,:)  = linint2_Wrap(finarr_sam&lon,finarr_sam&lat,finarr_sam,True,sam_patterns&lon,sam_patterns&lat,0)
           sam_tay_stat(ii,:) = (/ taylor_stats(sam_patterns(ii,{:-20},:), sam_patterns(0,{:-20},:), w({:-20}), 0)  /)
        end if
        tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_nam_pc)-1,1),finarr_nam_pc,False,True,0)
        nam_pc_trends(ii) = (/ tttt@slope*dimsizes(finarr_nam_pc)  /)
        tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_nao_pc)-1,1),finarr_nao_pc,False,True,0)
        nao_pc_trends(ii) = (/ tttt@slope*dimsizes(finarr_nao_pc)  /)
        tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_sam_pc)-1,1),finarr_sam_pc,False,True,0)
        sam_pc_trends(ii) = (/ tttt@slope*dimsizes(finarr_sam_pc)  /)

        nam_timeseries(ii,:) = (/ finarr_nam_pc /)
        nao_timeseries(ii,:) = (/ finarr_nao_pc /)
        sam_timeseries(ii,:) = (/ finarr_sam_pc /)

        nam_pcvari(ii) = (/ finarr_nam_pcvar /)
        nao_pcvari(ii) = (/ finarr_nao_pcvar /)
        sam_pcvari(ii) = (/ finarr_sam_pcvar /)
        ;print("Done with "+iobs+" our of "+nens+", "+model(iobs))
     end do

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
     obs_datasets = obs_datasets(iobs_datasets)
     iref_obs_datasets = ind(obs_datasets.eq.diag_script_info@reference_dataset)
     z@datasets = obs_datasets
     delete(z)
     delete(iobs_datasets)
     ;end if
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
     obs_significance = z->pattern_significance

;-------------------------------------------------------------
; NAM/NAO/SAM for CMIP6 historical simulations
;-------------------------------------------------------------
;
  lat := nam_patterns_obs&lat
  lat!0 = "lat"
  lat&lat = lat
  w := cos(rad*lat)  ; weights for Taylor diagram
  w!0 = "lat"
  w&lat = lat

  nens = dim_cmip6_hist

  print("Number of CMIP6 historical simulations found: " + nens)
  model = hist_cmip6_datasets
  emem  = hist_cmip6_ensembles
  ensemble_assign = new(nens, integer)

  enum = 1
  temp = model(0)
  do gg = 0, nens - 1
     if (temp.eq.model(gg)) then   ; does the model name match what's in temp?
        ensemble_assign(gg) = enum ; if so, assign it the same number
     else
        enum = enum+1              ; if not, assign it the next number
        ensemble_assign(gg) = enum
        temp = model(gg)
     end if
  end do
  ensemble_assign@model = model
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
        nam_timeseries&time = ispan(1958,2014,1)
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
;    print(nam_pc_trends(gg)+"  "+nao_pc_trends(gg)+"  "+sam_pc_trends(gg))


     printVarSummary(finarr_sam_pc)
     printVarSummary(finarr_nam_pc)
     printVarSummary(finarr_nao_pc)

     nam_timeseries(gg,:) = (/ finarr_nam_pc /)
     nao_timeseries(gg,:) = (/ finarr_nao_pc /)
     sam_timeseries(gg,:) = (/ finarr_sam_pc /)

     nam_pcvari(gg) = (/ finarr_nam_pcvar /)
     nao_pcvari(gg) = (/ finarr_nao_pcvar /)
     sam_pcvari(gg) = (/ finarr_sam_pcvar /)
     print("Done with "+gg+" out of "+nens+", "+model(gg))

  end do
  fn = out_path + "nam_nao_sam.hist."+syearA+"-"+eyearA+".cmip6.nc"
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
  delete([/nam_patterns, nao_patterns, sam_patterns/])
  delete([/nam_timeseries, nao_timeseries, sam_timeseries/])
  delete([/nam_tay_stat, nao_tay_stat, sam_tay_stat/])
  delete([/nam_pc_trends, nao_pc_trends, sam_pc_trends/])
  delete([/nam_pcvari, nao_pcvari, sam_pcvari/])

  fn = out_path + "nam_nao_sam.hist."+syearA+"-"+eyearA+".cmip6.nc"  
  z = addfile(fn,"r")
  nam_taylor_stats_C6 = z->nam_tay_stat
  nao_taylor_stats_C6 = z->nao_tay_stat
  sam_taylor_stats_C6 = z->sam_tay_stat
  nam_trends_C6 = z->nam_pc_trends
  nao_trends_C6 = z->nao_pc_trends
  sam_trends_C6 = z->sam_pc_trends
  nam_patterns_C6 = z->nam_patterns
  nao_patterns_C6 = z->nao_patterns
  sam_patterns_C6 = z->sam_patterns
  ensemble_assign_C6 := z->ensemble_assign

;  cmip6_num = dimsizes(nam_trends)
  cmip6_num = dimsizes(ind(.not.ismissing(nam_trends_C6)))
  
  taylor_nam_ratio_C6  = nam_taylor_stats_C6(:,1:1)
  taylor_nam_patcor_C6 = nam_taylor_stats_C6(:,0:0)
  taylor_nam_patcorZ_C6 = 0.5*(log( (1+taylor_nam_patcor_C6) / (1-taylor_nam_patcor_C6)))    ; convert to Z-space
  taylor_nao_ratio_C6  = nao_taylor_stats_C6(:,1:1)
  taylor_nao_patcor_C6 = nao_taylor_stats_C6(:,0:0)
  taylor_nao_patcorZ_C6 = 0.5*(log( (1+taylor_nao_patcor_C6) / (1-taylor_nao_patcor_C6)))    ; convert to Z-space
  taylor_sam_ratio_C6  = sam_taylor_stats_C6(:,1:1)
  taylor_sam_patcor_C6 = sam_taylor_stats_C6(:,0:0)
  taylor_sam_patcorZ_C6 = 0.5*(log( (1+taylor_sam_patcor_C6) / (1-taylor_sam_patcor_C6)))    ; convert to Z-space

  nam_em_C6 = nam_patterns_C6(:max(ensemble_assign_C6)-1,:,:)
  nam_em_C6 = nam_em_C6@_FillValue
  nao_em_C6 = nam_em_C6
  sam_em_C6 = nam_em_C6

  nam_trends_mme_C6 = new(max(ensemble_assign_C6),double)
  sam_trends_mme_C6 = nam_trends_mme_C6
  nao_trends_mme_C6 = nam_trends_mme_C6
  tay_patcor_nam_C6  = new(max(ensemble_assign_C6),double)
  tay_ratio_nam_C6   = tay_patcor_nam_C6
  tay_patcor_nao_C6  = tay_patcor_nam_C6
  tay_ratio_nao_C6   = tay_patcor_nam_C6
  tay_patcor_sam_C6  = tay_patcor_nam_C6
  tay_ratio_sam_C6   = tay_patcor_nam_C6
  weights_C6 = new(dimsizes(nam_patterns_C6&E),float)
  do gg = 1,max(ensemble_assign_C6)         ; calculate ensemble means
     wind := ind(ensemble_assign_C6.eq.gg)
     if (dimsizes(wind).eq.1) then
        nam_trends_mme_C6(gg-1) = (/ nam_trends_C6(wind) /)
        nao_trends_mme_C6(gg-1) = (/ nao_trends_C6(wind) /)
        sam_trends_mme_C6(gg-1) = (/ sam_trends_C6(wind) /)
        
        nam_em_C6(gg-1,:,:) = (/ nam_patterns_C6(wind,:,:) /)
        nao_em_C6(gg-1,:,:) = (/ nao_patterns_C6(wind,:,:) /)
        sam_em_C6(gg-1,:,:) = (/ sam_patterns_C6(wind,:,:) /)
        tay_patcor_nam_C6(gg-1) = (/ taylor_nam_patcorZ_C6(wind,0) /)
        tay_ratio_nam_C6(gg-1) = (/ taylor_nam_ratio_C6(wind,0) /)
        tay_patcor_nao_C6(gg-1) = (/ taylor_nao_patcorZ_C6(wind,0) /)
        tay_ratio_nao_C6(gg-1) = (/ taylor_nao_ratio_C6(wind,0) /)
        tay_patcor_sam_C6(gg-1) = (/ taylor_sam_patcorZ_C6(wind,0) /)
        tay_ratio_sam_C6(gg-1) = (/ taylor_sam_ratio_C6(wind,0) /)
     else
        nam_trends_mme_C6(gg-1) = (/ dim_avg(nam_trends_C6(wind)) /)
        nao_trends_mme_C6(gg-1) = (/ dim_avg(nao_trends_C6(wind)) /)
        sam_trends_mme_C6(gg-1) = (/ dim_avg(sam_trends_C6(wind)) /)

        nam_em_C6(gg-1,:,:) = (/ dim_avg_n(nam_patterns_C6(wind,:,:),0) /)
        nao_em_C6(gg-1,:,:) = (/ dim_avg_n(nao_patterns_C6(wind,:,:),0) /)
        sam_em_C6(gg-1,:,:) = (/ dim_avg_n(sam_patterns_C6(wind,:,:),0) /)
        tay_patcor_nam_C6(gg-1) = (/ dim_avg_n(taylor_nam_patcorZ_C6(wind,:),0) /)
        tay_ratio_nam_C6(gg-1)  = (/ dim_avg_n(taylor_nam_ratio_C6(wind,:),0) /)
        tay_patcor_nao_C6(gg-1) = (/ dim_avg_n(taylor_nao_patcorZ_C6(wind,:),0) /)
        tay_ratio_nao_C6(gg-1)  = (/ dim_avg_n(taylor_nao_ratio_C6(wind,:),0) /)
        tay_patcor_sam_C6(gg-1) = (/ dim_avg_n(taylor_sam_patcorZ_C6(wind,:),0) /)
        tay_ratio_sam_C6(gg-1)  = (/ dim_avg_n(taylor_sam_ratio_C6(wind,:),0) /)
     end if
     weights_C6(wind) = 1./dimsizes(wind)
  end do
;  print(ensemble_assign+" "+weights_C6)

  nam_trends_mme_C6T = dim_avg(nam_trends_mme_C6)
  nao_trends_mme_C6T = dim_avg(nao_trends_mme_C6)
  sam_trends_mme_C6T = dim_avg(sam_trends_mme_C6)
  
  mme_nam_C6 := dim_avg_n_Wrap(nam_em_C6,0)
  mme_nao_C6 := dim_avg_n_Wrap(nao_em_C6,0)
  mme_sam_C6 := dim_avg_n_Wrap(sam_em_C6,0)

;  nao_patterns_obsLF = lonFlip(nao_patterns_obs)
;  mme_naoLF_C6 = lonFlip(mme_nao_C6)

  tay_stats_hist_nam_cmip6EM = new((/1,3/),double)
;  tay_stats_hist_nam_cmip6EM = (/ taylor_stats(mme_nam_C6({20:},:), nam_patterns_obs({20:},:), w({20:}), 0) /)
  tay_stats_hist_nao_cmip6EM = new((/1,3/),double)
;  tay_stats_hist_nao_cmip6EM = (/ taylor_stats(mme_naoLF_C6({20:80},{-90:40}), nao_patterns_obsLF({20:80},{-90:40}), w({20:80}), 0) /)
  tay_stats_hist_sam_cmip6EM = new((/1,3/),double)
;  tay_stats_hist_sam_cmip6EM = (/ taylor_stats(mme_sam_C6({:-20},:), sam_patterns_obs({:-20},:), w({:-20}), 0) /)

  tay_stats_hist_nam_cmip6EM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_nam_C6))-1)/ ((2.71828^(2*dim_avg(tay_patcor_nam_C6))+1))) /)   ; convert back from z-space
  tay_stats_hist_nam_cmip6EM(0,1) = (/ dim_avg(tay_ratio_nam_C6) /)
  tay_stats_hist_nao_cmip6EM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_nao_C6))-1)/ ((2.71828^(2*dim_avg(tay_patcor_nao_C6))+1))) /)   ; convert back from z-space
  tay_stats_hist_nao_cmip6EM(0,1) = (/ dim_avg(tay_ratio_nao_C6) /)
  tay_stats_hist_sam_cmip6EM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_sam_C6))-1)/ ((2.71828^(2*dim_avg(tay_patcor_sam_C6))+1))) /)   ; convert back from z-space
  tay_stats_hist_sam_cmip6EM(0,1) = (/ dim_avg(tay_ratio_sam_C6) /)

;---------------------------------------------------------------------------
; Calculate whether 80% of simulations match the sign of the MME

  mme_nam_sign_C6 = mme_nam_C6
  mme_nam_sign_C6 = mme_nam_sign_C6@_FillValue
  do cc = 0,dimsizes(mme_nam_C6&lat)-1        ; If more than 80% of the runs match the sign of the MME, assign value of 1. 
     do dd = 0,dimsizes(mme_nam_C6&lon)-1     ; otherwise assign a value of 0.
        if (.not.ismissing(mme_nam_C6(cc,dd))) then
           tnum = toint(num(.not.ismissing(nam_patterns_C6(:,cc,dd)))*.8)
           if (mme_nam_C6(cc,dd).ge.0) then   ; MME >= 0
              count = num(nam_patterns_C6(:,cc,dd).ge.0)
              if (count.ge.tnum) then      ; if more than 80% of the EM are >=0
                 mme_nam_sign_C6(cc,dd) = 1   ; assign value of 1
              else
                 mme_nam_sign_C6(cc,dd) = 0   ; otherwise assign value of 0
             end if
           else                            ; MME < 0
              count = num(nam_patterns_C6(:,cc,dd).lt.0)  
              if (count.ge.tnum) then      ; if more than 80% of EM are < 0
                 mme_nam_sign_C6(cc,dd) = 1   ; assign value of 1
              else
                 mme_nam_sign_C6(cc,dd) = 0   ; otherwise assign value of 0
              end if
           end if      
        end if
     end do
  end do

  mme_nao_sign_C6 = mme_nao_C6
  mme_nao_sign_C6 = mme_nao_sign_C6@_FillValue
  do cc = 0,dimsizes(mme_nao_C6&lat)-1        ; If more than 80% of the runs match the sign of the MME, assign value of 1. 
     do dd = 0,dimsizes(mme_nao_C6&lon)-1     ; otherwise assign a value of 0.
        if (.not.ismissing(mme_nao_C6(cc,dd))) then
           tnum = toint(num(.not.ismissing(nao_patterns_C6(:,cc,dd)))*.8)
           if (mme_nao_C6(cc,dd).ge.0) then   ; MME >= 0
              count = num(nao_patterns_C6(:,cc,dd).ge.0)
              if (count.ge.tnum) then      ; if more than 80% of the EM are >=0
                 mme_nao_sign_C6(cc,dd) = 1   ; assign value of 1
              else
                 mme_nao_sign_C6(cc,dd) = 0   ; otherwise assign value of 0
             end if
           else                            ; MME < 0
              count = num(nao_patterns_C6(:,cc,dd).lt.0)  
              if (count.ge.tnum) then      ; if more than 80% of EM are < 0
                 mme_nao_sign_C6(cc,dd) = 1   ; assign value of 1
              else
                 mme_nao_sign_C6(cc,dd) = 0   ; otherwise assign value of 0
              end if
           end if      
        end if
     end do
  end do

  mme_sam_sign_C6 = mme_sam_C6
  mme_sam_sign_C6 = mme_sam_sign_C6@_FillValue
  do cc = 0,dimsizes(mme_sam_C6&lat)-1        ; If more than 80% of the runs match the sign of the MME, assign value of 1. 
     do dd = 0,dimsizes(mme_sam_C6&lon)-1     ; otherwise assign a value of 0.
        if (.not.ismissing(mme_sam_C6(cc,dd))) then
           tnum = toint(num(.not.ismissing(sam_patterns_C6(:,cc,dd)))*.8)
           if (mme_sam_C6(cc,dd).ge.0) then   ; MME >= 0
              count = num(sam_patterns_C6(:,cc,dd).ge.0)
              if (count.ge.tnum) then      ; if more than 80% of the EM are >=0
                 mme_sam_sign_C6(cc,dd) = 1   ; assign value of 1
              else
                 mme_sam_sign_C6(cc,dd) = 0   ; otherwise assign value of 0
             end if
           else                            ; MME < 0
              count = num(sam_patterns_C6(:,cc,dd).lt.0)  
              if (count.ge.tnum) then      ; if more than 80% of EM are < 0
                 mme_sam_sign_C6(cc,dd) = 1   ; assign value of 1
              else
                 mme_sam_sign_C6(cc,dd) = 0   ; otherwise assign value of 0
              end if
           end if      
        end if
     end do
  end do

;-------------------------------------------------------------
; NAM/NAO/SAM for AMIP simulations
;-------------------------------------------------------------
;
  lat := nam_patterns_obs&lat
  lat!0 = "lat"
  lat&lat = lat
  w := cos(rad*lat)  ; weights for Taylor diagram
  w!0 = "lat"
  w&lat = lat

  nens = dim_cmip6_amip

  print("Number of CMIP6 AMIP simulations found: " + dim_cmip6_amip)
  model := amip_cmip6_datasets
  emem  := amip_cmip6_ensembles
  ensemble_assign_A6 := new(nens, integer)
  
  enum = 1
  temp := model(0)
  do gg = 0,nens-1
     if (temp.eq.model(gg)) then   ; does the model name match what's in temp?
        ensemble_assign_A6(gg) = enum ; if so, assign it the same number
     else
        enum = enum+1              ; if not, assign it the next number
        ensemble_assign_A6(gg) = enum
        temp = model(gg)
     end if
  end do
  delete(temp)
  ensemble_assign_A6@models = str_join(model+"/"+emem,",")

  do gg = 0, dim_cmip6_amip-1
     arrF := read_data(amip_cmip6[gg])
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
        nam_timeseries&time = ispan(1958,2014,1) 
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

        ensemble_assign_A6!0 = "E"
        ensemble_assign_A6&E = nam_patterns&E
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
;     print(nam_pc_trends(gg)+"  "+nao_pc_trends(gg)+"  "+sam_pc_trends(gg))

     nam_timeseries(gg,:) = (/ finarr_nam_pc /)
     nao_timeseries(gg,:) = (/ finarr_nao_pc /)
     sam_timeseries(gg,:) = (/ finarr_sam_pc /)

     nam_pcvari(gg) = (/ finarr_nam_pcvar /)
     nao_pcvari(gg) = (/ finarr_nao_pcvar /)
     sam_pcvari(gg) = (/ finarr_sam_pcvar /)
     print("Done with "+gg+" out of "+nens+", "+model(gg))
  end do

  fn = out_path + "nam_nao_sam.amip."+syearA+"-"+eyearA+".cmip6.nc"
  system("rm -f " + fn)
  z = addfile(fn,"c")
  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->ensemble_assign = ensemble_assign_A6
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
  delete([/nam_patterns, nao_patterns, sam_patterns/])
  delete([/nam_timeseries, nao_timeseries, sam_timeseries/])
  delete([/nam_tay_stat, nao_tay_stat, sam_tay_stat/])
  delete([/nam_pc_trends, nao_pc_trends, sam_pc_trends/])
  delete([/nam_pcvari, nao_pcvari, sam_pcvari/])
  
  y = addfile(fn,"r")
  nam_taylor_stats_A6 = y->nam_tay_stat
  nao_taylor_stats_A6 = y->nao_tay_stat
  sam_taylor_stats_A6 = y->sam_tay_stat
  nam_trends_A6 = y->nam_pc_trends
  nao_trends_A6 = y->nao_pc_trends
  sam_trends_A6 = y->sam_pc_trends
  nam_patterns_A6 = y->nam_patterns
  nao_patterns_A6 = y->nao_patterns
  sam_patterns_A6 = y->sam_patterns
  ensemble_assign_A6 := y->ensemble_assign

;  cmip6A_num = dimsizes(nam_trendsA)
  cmip6A_num = dimsizes(ind(.not.ismissing(nam_trends_A6)))

  taylor_nam_ratio_A6  = nam_taylor_stats_A6(:,1:1)
  taylor_nam_patcor_A6 = nam_taylor_stats_A6(:,0:0)
  taylor_nam_patcorZ_A6 = 0.5*(log( (1+taylor_nam_patcor_A6) / (1-taylor_nam_patcor_A6)))    ; convert to Z-space
  taylor_nao_ratio_A6  = nao_taylor_stats_A6(:,1:1)
  taylor_nao_patcor_A6 = nao_taylor_stats_A6(:,0:0)
  taylor_nao_patcorZ_A6 = 0.5*(log( (1+taylor_nao_patcor_A6) / (1-taylor_nao_patcor_A6)))    ; convert to Z-space
  taylor_sam_ratio_A6  = sam_taylor_stats_A6(:,1:1)
  taylor_sam_patcor_A6 = sam_taylor_stats_A6(:,0:0)
  taylor_sam_patcorZ_A6 = 0.5*(log( (1+taylor_sam_patcor_A6) / (1-taylor_sam_patcor_A6)))    ; convert to Z-space

  nam_em_A6 = nam_patterns_A6(:max(ensemble_assign_A6)-1,:,:)
  nam_em_A6 = nam_em_A6@_FillValue
  nao_em_A6 = nam_em_A6
  sam_em_A6 = nam_em_A6

  nam_trends_mme_A6 = new(max(ensemble_assign_A6),double)
  sam_trends_mme_A6 = nam_trends_mme_A6
  nao_trends_mme_A6 = nam_trends_mme_A6
  tay_patcor_nam_A6  = new(max(ensemble_assign_A6),double) 
  tay_ratio_nam_A6   = tay_patcor_nam_A6
  tay_patcor_nao_A6  = tay_patcor_nam_A6
  tay_ratio_nao_A6   = tay_patcor_nam_A6
  tay_patcor_sam_A6  = tay_patcor_nam_A6
  tay_ratio_sam_A6   = tay_patcor_nam_A6 
  do gg = 1,max(ensemble_assign_A6)         ; calculate ensemble means
     wind := ind(ensemble_assign_A6.eq.gg)
     if (dimsizes(wind).eq.1) then
        nam_trends_mme_A6(gg-1) = (/ nam_trends_A6(wind) /)
        nao_trends_mme_A6(gg-1) = (/ nao_trends_A6(wind) /)
        sam_trends_mme_A6(gg-1) = (/ sam_trends_A6(wind) /)

        nam_em_A6(gg-1,:,:) = (/ nam_patterns_A6(wind,:,:) /)
        nao_em_A6(gg-1,:,:) = (/ nao_patterns_A6(wind,:,:) /)
        sam_em_A6(gg-1,:,:) = (/ sam_patterns_A6(wind,:,:) /)
        tay_patcor_nam_A6(gg-1) = (/ taylor_nam_patcorZ_A6(wind,0) /)
        tay_ratio_nam_A6(gg-1) = (/ taylor_nam_ratio_A6(wind,0) /)
        tay_patcor_nao_A6(gg-1) = (/ taylor_nao_patcorZ_A6(wind,0) /)
        tay_ratio_nao_A6(gg-1) = (/ taylor_nao_ratio_A6(wind,0) /)
        tay_patcor_sam_A6(gg-1) = (/ taylor_sam_patcorZ_A6(wind,0) /)
        tay_ratio_sam_A6(gg-1) = (/ taylor_sam_ratio_A6(wind,0) /)
      else
        nam_trends_mme_A6(gg-1) = (/ dim_avg(nam_trends_A6(wind)) /)
        nao_trends_mme_A6(gg-1) = (/ dim_avg(nao_trends_A6(wind)) /)
        sam_trends_mme_A6(gg-1) = (/ dim_avg(sam_trends_A6(wind)) /)
        
        nam_em_A6(gg-1,:,:) = (/ dim_avg_n(nam_patterns_A6(wind,:,:),0) /)
        nao_em_A6(gg-1,:,:) = (/ dim_avg_n(nao_patterns_A6(wind,:,:),0) /)
        sam_em_A6(gg-1,:,:) = (/ dim_avg_n(sam_patterns_A6(wind,:,:),0) /)
        tay_patcor_nam_A6(gg-1) = (/ dim_avg_n(taylor_nam_patcorZ_A6(wind,:),0) /)
        tay_ratio_nam_A6(gg-1)  = (/ dim_avg_n(taylor_nam_ratio_A6(wind,:),0) /)
        tay_patcor_nao_A6(gg-1) = (/ dim_avg_n(taylor_nao_patcorZ_A6(wind,:),0) /)
        tay_ratio_nao_A6(gg-1)  = (/ dim_avg_n(taylor_nao_ratio_A6(wind,:),0) /)
        tay_patcor_sam_A6(gg-1) = (/ dim_avg_n(taylor_sam_patcorZ_A6(wind,:),0) /)
        tay_ratio_sam_A6(gg-1)  = (/ dim_avg_n(taylor_sam_ratio_A6(wind,:),0) /)
     end if
     print("gg = "+gg)
  end do
  nam_trends_mme_A6T = dim_avg(nam_trends_mme_A6)
  nao_trends_mme_A6T = dim_avg(nao_trends_mme_A6)
  sam_trends_mme_A6T = dim_avg(sam_trends_mme_A6)

  mme_nam_A6 = dim_avg_n_Wrap(nam_em_A6,0)
  mme_nao_A6 = dim_avg_n_Wrap(nao_em_A6,0)
  mme_sam_A6 = dim_avg_n_Wrap(sam_em_A6,0)

;  nao_patterns_obsLF = lonFlip(nao_patterns_obs)
;  mme_naoLF_A6 = lonFlip(mme_nao_A6)

  tay_stats_hist_nam_cmip6AEM = new((/1,3/),double)
;  tay_stats_hist_nam_cmip6EM = (/ taylor_stats(mme_nam_A6({20:},:), nam_patterns_obs({20:},:), w({20:}), 0) /)
  tay_stats_hist_nao_cmip6AEM = new((/1,3/),double)
;  tay_stats_hist_nao_cmip6EM = (/ taylor_stats(mme_naoLF_A6({20:80},{-90:40}), nao_patterns_obsLF({20:80},{-90:40}), w({20:80}), 0) /)
  tay_stats_hist_sam_cmip6AEM = new((/1,3/),double)
;  tay_stats_hist_sam_cmip6EM = (/ taylor_stats(mme_sam_A6({:-20},:), sam_patterns_obs({:-20},:), w({:-20}), 0) /)

  tay_stats_hist_nam_cmip6AEM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_nam_A6))-1)/ ((2.71828^(2*dim_avg(tay_patcor_nam_A6))+1))) /)   ; convert back from z-space
  tay_stats_hist_nam_cmip6AEM(0,1) = (/ dim_avg(tay_ratio_nam_A6) /)
  tay_stats_hist_nao_cmip6AEM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_nao_A6))-1)/ ((2.71828^(2*dim_avg(tay_patcor_nao_A6))+1))) /)   ; convert back from z-space
  tay_stats_hist_nao_cmip6AEM(0,1) = (/ dim_avg(tay_ratio_nao_A6) /)
  tay_stats_hist_sam_cmip6AEM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_sam_A6))-1)/ ((2.71828^(2*dim_avg(tay_patcor_sam_A6))+1))) /)   ; convert back from z-space
  tay_stats_hist_sam_cmip6AEM(0,1) = (/ dim_avg(tay_ratio_sam_A6) /)

  ;--------------------------------------------------------
  ; process CMIP5 historical data
  ;--------------------------------------------------------

  lat := nam_patterns_obs&lat
  lat!0 = "lat"
  lat&lat = lat
  w := cos(rad*lat)  ; weights for Taylor diagram
  w!0 = "lat"
  w&lat = lat

  print("Number of CMIP5 historical simulations found: " + dim_cmip5_hist)
  model := hist_cmip5_datasets
  emem  := hist_cmip5_ensembles
  ensemble_assign_C5 := new(dim_cmip5_hist, integer)

  enum = 1
  temp := model(0)

  do gg = 0, dim_cmip5_hist - 1
     if (temp.eq.model(gg)) then   ; does the model name match what's in temp?
        ensemble_assign_C5(gg) = enum ; if so, assign it the same number
     else
        enum = enum+1              ; if not, assign it the next number
        ensemble_assign_C5(gg) = enum
        temp = model(gg)
     end if
  end do

  ensemble_assign_C5@model := model
  delete(temp)
  ensemble_assign_C5@models = str_join(model+"/"+emem,",")

  do gg = 0, dim_cmip5_hist-1
    arrF := read_data(hist_cmip5[gg])
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

    if (gg.eq.0) then
       nao_patterns_obs_lf = lonFlip(nao_patterns_obs)
       nam_patterns := new((/dim_cmip5_hist,dimsizes(nam_patterns_obs&lat),dimsizes(nam_patterns_obs&lon)/),typeof(nam_patterns_obs))
       nam_patterns!0 = "E"
       nam_patterns&E = ispan(0,dim_cmip5_hist-1,1)
       nam_patterns!1 = "lat"
       nam_patterns&lat = nam_patterns_obs&lat
       nam_patterns!2 = "lon"
       nam_patterns&lon = nam_patterns_obs&lon
       nam_patterns = nam_patterns@_FillValue
       sam_patterns := nam_patterns
       nao_patterns := nam_patterns

       nam_timeseries := new((/dim_cmip5_hist,dimsizes(finarr_nam_pc)/),float)
       nam_timeseries!0 = "E"
       nam_timeseries&E = nam_patterns&E
       nam_timeseries!1 = "time"
       nam_timeseries&time = ispan(1958,2005,1)
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

       nam_tay_stat := new((/dim_cmip5_hist,3/),double)
       nam_tay_stat!0 = "E"
       nam_tay_stat&E = nam_patterns&E
       nam_tay_stat!1 = "stat"
       nam_tay_stat&stat = ispan(0,2,1)
       nao_tay_stat := nam_tay_stat
       sam_tay_stat := nam_tay_stat

       ensemble_assign_C5!0 = "E"
       ensemble_assign_C5&E := nam_patterns&E
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
    print("Done with "+gg+" out of "+(dim_cmip5_hist-1))
  end do

  fn = out_path + "nam_nao_sam.hist."+syearA+"-"+eyearA+".cmip5.nc"
  system("rm -f " + fn)
  z = addfile(fn,"c")
  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->ensemble_assign = ensemble_assign_C5
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
  delete([/nam_patterns, nao_patterns, sam_patterns/])
  delete([/nam_timeseries, nao_timeseries, sam_timeseries/])
  delete([/nam_tay_stat, nao_tay_stat, sam_tay_stat/])
  delete([/nam_pc_trends, nao_pc_trends, sam_pc_trends/])
  delete([/nam_pcvari, nao_pcvari, sam_pcvari/])
  ;delete([/z/])

  z = addfile(fn,"r")

  nam_taylor_stats_C5 = z->nam_tay_stat
  nao_taylor_stats_C5 = z->nao_tay_stat
  sam_taylor_stats_C5 = z->sam_tay_stat
  nam_patterns_C5 := z->nam_patterns
  nao_patterns_C5 := z->nao_patterns
  sam_patterns_C5 := z->sam_patterns

  taylor_nam_ratio_C5  = nam_taylor_stats_C5(:,1:1)
  taylor_nam_patcor_C5 = nam_taylor_stats_C5(:,0:0)
  taylor_nam_patcor_C5Z = 0.5*(log( (1+taylor_nam_patcor_C5) / (1-taylor_nam_patcor_C5)))    ; convert to Z-space
  taylor_nao_ratio_C5  = nao_taylor_stats_C5(:,1:1)
  taylor_nao_patcor_C5 = nao_taylor_stats_C5(:,0:0)
  taylor_nao_patcor_C5Z = 0.5*(log( (1+taylor_nao_patcor_C5) / (1-taylor_nao_patcor_C5)))    ; convert to Z-space
  taylor_sam_ratio_C5  = sam_taylor_stats_C5(:,1:1)
  taylor_sam_patcor_C5 = sam_taylor_stats_C5(:,0:0)
  taylor_sam_patcor_C5Z = 0.5*(log( (1+taylor_sam_patcor_C5) / (1-taylor_sam_patcor_C5)))    ; convert to Z-space

  ea_c5 = z->ensemble_assign

  nam_trends_C5 = z->nam_pc_trends
  nao_trends_C5 = z->nao_pc_trends
  sam_trends_C5 = z->sam_pc_trends

;  cmip5_num = dimsizes(nam_trends_C5)
  cmip5_num = dimsizes(ind(.not.ismissing(nam_trends_C5)))

  nam_em_C5 := nam_patterns_C5(:max(ea_c5)-1,:,:)
  nam_em_C5 = nam_em_C5@_FillValue
  nao_em_C5 := nam_em_C5
  sam_em_C5 := nam_em_C5

  nam_trends_mme_C5 = new(max(ea_c5),double)
  sam_trends_mme_C5 = nam_trends_mme_C5
  nao_trends_mme_C5 = nam_trends_mme_C5
  tay_patcor_nam_C5  = nam_trends_mme_C5
  tay_ratio_nam_C5   = nam_trends_mme_C5
  tay_patcor_nao_C5  = nam_trends_mme_C5
  tay_ratio_nao_C5   = nam_trends_mme_C5
  tay_patcor_sam_C5  = nam_trends_mme_C5
  tay_ratio_sam_C5   = nam_trends_mme_C5
  do gg = 1,max(ea_c5)         ; calculate CMIP5 ensemble means
     wind := ind(ea_c5.eq.gg)
     if (dimsizes(wind).eq.1) then
        nam_trends_mme_C5(gg-1) = (/ nam_trends_C5(wind) /)
        nao_trends_mme_C5(gg-1) = (/ nao_trends_C5(wind) /)
        sam_trends_mme_C5(gg-1) = (/ sam_trends_C5(wind) /)

        nam_em_C5(gg-1,:,:) = (/ nam_patterns_C5(wind,:,:) /)
        nao_em_C5(gg-1,:,:) = (/ nao_patterns_C5(wind,:,:) /)
        sam_em_C5(gg-1,:,:) = (/ sam_patterns_C5(wind,:,:) /)
        tay_patcor_nam_C5(gg-1) = (/ taylor_nam_patcor_C5Z(wind,0) /)
        tay_ratio_nam_C5(gg-1) = (/ taylor_nam_ratio_C5(wind,0) /)
        tay_patcor_nao_C5(gg-1) = (/ taylor_nao_patcor_C5Z(wind,0) /)
        tay_ratio_nao_C5(gg-1) = (/ taylor_nao_ratio_C5(wind,0) /)
        tay_patcor_sam_C5(gg-1) = (/ taylor_sam_patcor_C5Z(wind,0) /)
        tay_ratio_sam_C5(gg-1) = (/ taylor_sam_ratio_C5(wind,0) /)
     else
        nam_trends_mme_C5(gg-1) = (/ dim_avg(nam_trends_C5(wind)) /)
        nao_trends_mme_C5(gg-1) = (/ dim_avg(nao_trends_C5(wind)) /)
        sam_trends_mme_C5(gg-1) = (/ dim_avg(sam_trends_C5(wind)) /)

        nam_em_C5(gg-1,:,:) = (/ dim_avg_n(nam_patterns_C5(wind,:,:),0) /)
        nao_em_C5(gg-1,:,:) = (/ dim_avg_n(nao_patterns_C5(wind,:,:),0) /)
        sam_em_C5(gg-1,:,:) = (/ dim_avg_n(sam_patterns_C5(wind,:,:),0) /)
        tay_patcor_nam_C5(gg-1) = (/ dim_avg_n(taylor_nam_patcor_C5Z(wind,:),0) /)
        tay_ratio_nam_C5(gg-1)  = (/ dim_avg_n(taylor_nam_ratio_C5(wind,:),0) /)
        tay_patcor_nao_C5(gg-1) = (/ dim_avg_n(taylor_nao_patcor_C5Z(wind,:),0) /)
        tay_ratio_nao_C5(gg-1)  = (/ dim_avg_n(taylor_nao_ratio_C5(wind,:),0) /)
        tay_patcor_sam_C5(gg-1) = (/ dim_avg_n(taylor_sam_patcor_C5Z(wind,:),0) /)
        tay_ratio_sam_C5(gg-1)  = (/ dim_avg_n(taylor_sam_ratio_C5(wind,:),0) /)
     end if
     print("gg = "+gg)
  end do
  nam_trends_mme_C5T = dim_avg(nam_trends_mme_C5)
  nao_trends_mme_C5T = dim_avg(nao_trends_mme_C5)
  sam_trends_mme_C5T = dim_avg(sam_trends_mme_C5)

  mme_nam_C5 := dim_avg_n_Wrap(nam_em_C5,0)
  mme_nao_C5 := dim_avg_n_Wrap(nao_em_C5,0)
  mme_sam_C5 := dim_avg_n_Wrap(sam_em_C5,0)
  delete([/nam_trends_mme_C5,nao_trends_mme_C5,sam_trends_mme_C5/])

;  mme_nao_C5LF = lonFlip(mme_nao_C5)

  tay_stats_hist_nam_cmip5EM = new((/1,3/),double)
;  tay_stats_hist_nam_cmip5EM = (/ taylor_stats(mme_nam_C5({20:},:), nam_patterns_obs({20:},:), w({20:}), 0) /)
  tay_stats_hist_nao_cmip5EM = new((/1,3/),double)
;  tay_stats_hist_nao_cmip5EM = (/ taylor_stats(mme_nao_C5LF({20:80},{-90:40}), nao_patterns_obsLF({20:80},{-90:40}), w({20:80}), 0) /)
  tay_stats_hist_sam_cmip5EM = new((/1,3/),double)
;  tay_stats_hist_sam_cmip5EM = (/ taylor_stats(mme_sam_C5({:-20},:), sam_patterns_obs({:-20},:), w({:-20}), 0) /)

  tay_stats_hist_nam_cmip5EM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_nam_C5))-1)/ ((2.71828^(2*dim_avg(tay_patcor_nam_C5))+1))) /)   ; convert back from z-space
  tay_stats_hist_nam_cmip5EM(0,1) = (/ dim_avg(tay_ratio_nam_C5) /)
  tay_stats_hist_nao_cmip5EM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_nao_C5))-1)/ ((2.71828^(2*dim_avg(tay_patcor_nao_C5))+1))) /)   ; convert back from z-space
  tay_stats_hist_nao_cmip5EM(0,1) = (/ dim_avg(tay_ratio_nao_C5) /)
  tay_stats_hist_sam_cmip5EM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_sam_C5))-1)/ ((2.71828^(2*dim_avg(tay_patcor_sam_C5))+1))) /)   ; convert back from z-space
  tay_stats_hist_sam_cmip5EM(0,1) = (/ dim_avg(tay_ratio_sam_C5) /)

;=====================================================================================================
  outpath = config_user_info@plot_dir
  ofile = outpath + "nam_nao_sam"

  wks = gsn_open_wks(file_type, ofile)

  res = True
  res@mpGeophysicalLineColor = "gray42"
  res@mpGeophysicalLineThicknessF = 2.
  res@mpGridAndLimbOn = False
  res@mpFillOn = False
  res@mpOutlineOn = True
  res@mpCenterLonF = 0.
  res@mpPerimDrawOrder = "PostDraw"
  res@gsnDraw      = False
  res@gsnFrame     = False
  res@cnLevelSelectionMode = "ExplicitLevels"
  res@cnLineLabelsOn = False
  res@cnFillOn        = True
  res@cnLinesOn       = False
  res@lbLabelBarOn    = False

  res@gsnCenterStringOrthogonalPosF = 0.02
  res@gsnRightString = ""
  res@gsnLeftString = ""
  res@gsnCenterStringFontHeightF = 0.0325;0.0275
;  res@txFontHeightF = 0.018
  res@gsnPolarLabelFontHeightF = 0.0


;  res@cnLevels = fspan(-500,500,11)
  res@cnLevels = fspan(-5,5,11)
  nam_patterns_obs_map = nam_patterns_obs
  nam_patterns_obs_map = nam_patterns_obs_map / 100.  ; Pa -> hPa
  mme_nam_C6_map = mme_nam_C6
  mme_nam_C6_map = mme_nam_C6_map / 100.
  nao_patterns_obs_map = nao_patterns_obs
  nao_patterns_obs_map = nao_patterns_obs_map / 100.  ; Pa -> hPa
  mme_nao_C6_map = mme_nao_C6
  mme_nao_C6_map = mme_nao_C6_map / 100.
  sam_patterns_obs_map = sam_patterns_obs
  sam_patterns_obs_map = sam_patterns_obs_map / 100.  ; Pa -> hPa
  mme_sam_C6_map = mme_sam_C6
  mme_sam_C6_map = mme_sam_C6_map / 100.
  
;  cmap0 = read_colormap_file("$diag_scripts/ipccwg1ar6ch3_modes/AR6_colormaps_NCL/temp_div_12.rgb")
  cmap0 = read_colormap_file("$diag_scripts/shared/plot/rgb/ipcc-ar6_temperature_div_12.rgb")
  res@cnFillColors = cmap0

  res2 = True
  res2@gsnDraw  = False
  res2@gsnFrame = False
  res2@cnLinesOn = False
  res2@cnLineLabelsOn = False
  res2@cnLevelSelectionMode = "ExplicitLevels"
  res2@cnFillOn = True
  res2@cnMonoFillPattern = False
  res2@cnFillColors = (/"gray25","transparent"/)
  res2@lbLabelBarOn = False
  res2@cnInfoLabelOn = False
  res2@gsnLeftString = ""
  res2@gsnRightString = ""
  res2@gsnAddCyclic = True

  plot = new(6,graphic)
  res@gsnPolar = "NH"
  res@mpMinLatF    = 20.
  res@gsnCenterString = "(a) NAM - Obs ("+diag_script_info@reference_dataset+")"
;  plot(0) = gsn_csm_contour_map_polar(wks,nam_patterns_obs,res)
  plot(0) = gsn_csm_contour_map_polar(wks,nam_patterns_obs_map,res)
  res@gsnCenterString = "(d) NAM - Hist MME"
;  plot(3) = gsn_csm_contour_map_polar(wks,mme_nam_C6,res)
  plot(3) = gsn_csm_contour_map_polar(wks,mme_nam_C6_map,res)

  res@gsnCenterString = "(b) NAO - Obs ("+diag_script_info@reference_dataset+")"
;  plot(1) = gsn_csm_contour_map_polar(wks,nao_patterns_obs,res)
  plot(1) = gsn_csm_contour_map_polar(wks,nao_patterns_obs_map,res)
  res@gsnCenterString = "(e) NAO - Hist MME"  
;  plot(4) = gsn_csm_contour_map_polar(wks,mme_nao_C6,res)
  plot(4) = gsn_csm_contour_map_polar(wks,mme_nao_C6_map,res)

  res@gsnPolar = "SH"
  res@mpMaxLatF    = -20.
  delete(res@mpMinLatF)
  res@gsnCenterString = "(c) SAM - Obs ("+diag_script_info@reference_dataset+")"
;  plot(2) = gsn_csm_contour_map_polar(wks,sam_patterns_obs,res)
  plot(2) = gsn_csm_contour_map_polar(wks,sam_patterns_obs_map,res)
  res@gsnCenterString = "(f) SAM - Hist MME"    
;  plot(5) = gsn_csm_contour_map_polar(wks,mme_sam_C6,res)
  plot(5) = gsn_csm_contour_map_polar(wks,mme_sam_C6_map,res)

  res2@cnFillScaleF        = 0.55
  res2@cnFillPatterns = (/17,0/)
  res2@cnLevels = 0.5
  oplot = new(3,graphic)
  do gg = 0,2
     oplot(gg) = gsn_csm_contour(wks,obs_significance(gg,:,:),res2)
     overlay(plot(gg),oplot(gg))
  end do

  res2@cnFillPatterns = (/3,0/) ;(/11,0/)
  res2@cnLevels = 0.5
  res2@cnFillScaleF        = 0.7
  oplot3 = gsn_csm_contour(wks,mme_nam_sign_C6,res2)
  overlay(plot(3),oplot3)
  oplot4 = gsn_csm_contour(wks,mme_nao_sign_C6,res2)
  overlay(plot(4),oplot4)
  oplot5 = gsn_csm_contour(wks,mme_sam_sign_C6,res2)
  overlay(plot(5),oplot5)

  panres = True
  panres@gsnPanelTop = 0.94
  panres@gsnPanelBottom = 0.43
  panres@gsnPanelLabelBar = True
  panres@gsnPanelYWhiteSpacePercent = 3.0
  panres@gsnPanelXWhiteSpacePercent = 3.0
  panres@pmLabelBarHeightF = 0.05
  panres@pmLabelBarWidthF = 0.38
  panres@pmLabelBarOrthogonalPosF = 0.005  ;0.01
  panres@lbTitleOn = False
  panres@lbBoxLineColor = "gray70"
  panres@lbLabelFontHeightF = 0.010
  panres@lbBoxEndCapStyle = "TriangleBothEnds"
  panres@lbTitleOn = True
  panres@lbTitleString = "(hPa)"
  panres@lbTitlePosition  = "Bottom"              ; title position
  panres@lbTitleFontHeightF= .010                ; make title smaller
  panres@txFontHeightF = 0.018;0.022
  panres@gsnFrame = False
  gsn_panel(wks,plot,(/2,3/),panres)

;- - - - - - - - - - - - - - - - - -
; Taylor plots, panels 7-9
;
  cmap := new((/10,3/),float)
  cmap(0,:) =  (/255,255,255/)
  cmap(1,:) = (/0,0,0/)
  cmap(2,:) = (/25,51,178/)      ; dark blue, misc_seq_1_21
  cmap(3,:) = (/253, 141, 60/)   ; dark orange, misc_div_21
  cmap(4,:) = (/211,15,31/)      ; dark red, misc_div_21
  cmap(5,:) = (/131,183,214/)    ; dark cyan, temp_div (continuous)
  cmap(6,:) = (/125, 79, 20/)    ; brown, prec_div (continuous)
  cmap(7,:) = (/60, 132, 26/)    ; green, wind_seq (continuous)
  cmap(8,:) = (/153,112,171/)    ; purple, chem_div_21
  cmap(9,:) = (/252,87,44/)      ; orange-red, misc_div (continuous)
  cmap = cmap/255.
  gsn_define_colormap(wks,cmap)

  tres                 = True                     ; diagram mods desired
  colors = new(max((/dimsizes(nam_patterns_C6&E), dimsizes(nam_patterns_A6&E), dimsizes(nam_patterns_C5&E)/)),integer)
  colors = 4
  tres@Colors          = colors
;  tres@caseLabels      = model
  markers = ensemble_assign+1
  markers = 2
  tres@Markers         = markers        ; marker styles
  tres@markerTxYOffset = 0.04                     ; offset btwn marker & label
  tres@gsMarkerSizeF   = 0.006                    ; marker size
  tres@txFontHeightF   = 0.028;0.0225                    ; text size
;  tres@varLabels = ensemble_assign
  tres@stnRad         = (/ 0.5, 1.5 /)     ; additional standard radii
  tres@ccRays          = (/ 0.4, 0.9 /)     ; correllation rays
  tres@ccRays_color    = "LightGray"        ; default is "black"

  tres@centerDiffRMS   = True             ; RMS 'circles'
  tres@centerDiffRMS_color = "LightGray"    ; default is "black"/)

  tres@taylorDraw  = False                          ; don't draw
  tres@taylorFrame = False
  tres@trXMaxF = 2.0


  tplot = new(3,graphic)
  tplot_C5 = tplot
  tres@tiYAxisString = ""
  tplot(1) = taylor_diagram(wks, nao_taylor_stats_C6(:,1:1), nao_taylor_stats_C6(:,0:0), tres)
;  print(taylor_nam_patcor+" "+taylor_nam_ratio)
  delete(tres@tiYAxisString)
  tplot(0) = taylor_diagram(wks, nam_taylor_stats_C6(:,1:1), nam_taylor_stats_C6(:,0:0), tres)
  tres@tiYAxisString = ""
  tplot(2) = taylor_diagram3(wks, sam_taylor_stats_C6(:,1:1), sam_taylor_stats_C6(:,0:0), tres)

  colors2 = new((/dimsizes(nao_taylor_stats_C5(:,0))/),integer)
  colors2 = 5
  tres@Colors          :=   colors2
  markers2 = new(dimsizes(nao_taylor_stats_C5(:,0)),integer)
  markers2 = 2
  tres@Markers := markers2
  tres@tfPolyDrawOrder = "PreDraw"
  tres@tiYAxisString = ""
  tplot_C5(1) = taylor_diagram(wks, nao_taylor_stats_C5(:,1:1), nao_taylor_stats_C5(:,0:0), tres)
  delete(tres@tiYAxisString)
  tplot_C5(0) = taylor_diagram(wks, nam_taylor_stats_C5(:,1:1), nam_taylor_stats_C5(:,0:0), tres)
  tres@tiYAxisString = ""
  tplot_C5(2) = taylor_diagram3(wks, sam_taylor_stats_C5(:,1:1), sam_taylor_stats_C5(:,0:0), tres)
  do gg =0,2
     overlay(tplot_C5(gg),tplot(gg))
  end do

;  printVarSummary(taylor_nao_ratio)
  tplot2 = new(3,graphic)
  colors := new(5,string)
  marker = new(5,integer)
  colors = "black"
  marker = 16
  tres@Colors          := colors       ; marker colors
  tres@Markers         := marker
  tres@gsMarkerSizeF   = 0.008
  tplot2(1) = taylor_diagram(wks, nao_taylor_stats_obs(:,1:1), nao_taylor_stats_obs(:,0:0), tres)
  delete(tres@tiYAxisString)
  tplot2(0) = taylor_diagram(wks, nam_taylor_stats_obs(:,1:1), nam_taylor_stats_obs(:,0:0), tres)
  tres@tiYAxisString = ""
  tplot2(2) = taylor_diagram3(wks, sam_taylor_stats_obs(:,1:1), sam_taylor_stats_obs(:,0:0), tres)
  do gg =0,2
     overlay(tplot_C5(gg),tplot2(gg))
  end do

  mstring = "y"
  fontnum = 35
  xoffset = 0.0
  yoffset = 0.0
  ratioZ   = 1.0
  size    = 1.0
  angle   = 0.0
  new_marker = NhlNewMarker(wks, mstring, fontnum, xoffset, yoffset, ratioZ, size, angle)


;  tplot3a = new(3,graphic)
  tplot3b = new(3,graphic)
;  tplot4a = new(3,graphic)
  tplot4b = new(3,graphic)
;  tplot5a = new(3,graphic)
  tplot5b = new(3,graphic)
  tres@gsMarkerSizeF   = 0.008
  tres@Colors := "white"
  tres@Markers:= 16 ;new_marker
;  tplot3a(1) = taylor_diagram(wks, tay_stats_hist_nao_cmip5EM(0:0,1:1),tay_stats_hist_nao_cmip5EM(0:0,0:0), tres)
;  tplot3a(0) = taylor_diagram(wks, tay_stats_hist_nam_cmip5EM(0:0,1:1),tay_stats_hist_nam_cmip5EM(0:0,0:0), tres)
;  tplot3a(2) = taylor_diagram3(wks, tay_stats_hist_sam_cmip5EM(0:0,1:1),tay_stats_hist_sam_cmip5EM(0:0,0:0), tres)
;  do gg = 0,2
;     overlay(tplot_C5(gg),tplot3a(gg))
;  end do
  tres@Colors := 2
  tres@Markers:= 16
;  tres@gsMarkerThicknessF = 2.0
  tplot3b(1) = taylor_diagram(wks, tay_stats_hist_nao_cmip5EM(0:0,1:1),tay_stats_hist_nao_cmip5EM(0:0,0:0), tres)
  delete(tres@tiYAxisString)
  tplot3b(0) = taylor_diagram(wks, tay_stats_hist_nam_cmip5EM(0:0,1:1),tay_stats_hist_nam_cmip5EM(0:0,0:0), tres)
  tres@tiYAxisString = ""
  tplot3b(2) = taylor_diagram3(wks, tay_stats_hist_sam_cmip5EM(0:0,1:1),tay_stats_hist_sam_cmip5EM(0:0,0:0), tres)
  do gg = 0,2
     overlay(tplot_C5(gg),tplot3b(gg))
  end do

;  tres@gsMarkerThicknessF = 1.0
;  tres@Colors := "white"
;  tres@Markers:= new_marker
;  tplot4a(1) = taylor_diagram(wks, tay_stats_hist_nao_cmip6EM(0:0,1:1),tay_stats_hist_nao_cmip6EM(0:0,0:0), tres)
;  tplot4a(0) = taylor_diagram(wks, tay_stats_hist_nam_cmip6EM(0:0,1:1),tay_stats_hist_nam_cmip6EM(0:0,0:0), tres)
;  tplot4a(2) = taylor_diagram3(wks, tay_stats_hist_sam_cmip6EM(0:0,1:1),tay_stats_hist_sam_cmip6EM(0:0,0:0), tres)
;  do gg = 0,2
;     overlay(tplot_C5(gg),tplot4a(gg))
;  end do
  tres@Colors := 3
  tres@Markers:= 16
;  tres@gsMarkerThicknessF = 2.0
  tplot4b(1) = taylor_diagram(wks, tay_stats_hist_nao_cmip6EM(0:0,1:1),tay_stats_hist_nao_cmip6EM(0:0,0:0), tres)
  delete(tres@tiYAxisString)
  tplot4b(0) = taylor_diagram(wks, tay_stats_hist_nam_cmip6EM(0:0,1:1),tay_stats_hist_nam_cmip6EM(0:0,0:0), tres)
  tres@tiYAxisString = ""
  tplot4b(2) = taylor_diagram3(wks, tay_stats_hist_sam_cmip6EM(0:0,1:1),tay_stats_hist_sam_cmip6EM(0:0,0:0), tres)
  do gg = 0,2
     overlay(tplot_C5(gg),tplot4b(gg))
  end do

;  tres@gsMarkerThicknessF = 1.0
;  tres@Colors := "white"
;  tres@Markers:= 16  ;new_marker
;  tplot5a(1) = taylor_diagram(wks, tay_stats_hist_nao_cmip6EM(0:0,1:1),tay_stats_hist_nao_cmip6EM(0:0,0:0), tres)
;  tplot5a(0) = taylor_diagram(wks, tay_stats_hist_nam_cmip6EM(0:0,1:1),tay_stats_hist_nam_cmip6EM(0:0,0:0), tres)
;  tplot5a(2) = taylor_diagram3(wks, tay_stats_hist_sam_cmip6EM(0:0,1:1),tay_stats_hist_sam_cmip6EM(0:0,0:0), tres)
;  do gg = 0,2
;     overlay(tplot_C5(gg),tplot5a(gg))
;  end do
  tres@Colors := 9
  tres@Markers:= 16
;  tres@gsMarkerThicknessF = 2.0
  tplot5b(1) = taylor_diagram(wks, tay_stats_hist_nao_cmip6EM(0:0,1:1),tay_stats_hist_nao_cmip6EM(0:0,0:0), tres)
  delete(tres@tiYAxisString)
  tplot5b(0) = taylor_diagram(wks, tay_stats_hist_nam_cmip6EM(0:0,1:1),tay_stats_hist_nam_cmip6EM(0:0,0:0), tres)
  tres@tiYAxisString = ""
  tplot5b(2) = taylor_diagram3(wks, tay_stats_hist_sam_cmip6EM(0:0,1:1),tay_stats_hist_sam_cmip6EM(0:0,0:0), tres)
  do gg = 0,2
     overlay(tplot_C5(gg),tplot5b(gg))
  end do
;- - - - - - - - - - - - - - - - - -
; Linear Trend Histograms, panels 10-12
;
  nao_trends_C6 = (/ (nao_trends_C6/57.)*10. /) ; convert to hPa/decade
  nao_trends_A6 = (/ (nao_trends_A6/57.)*10. /) ; convert to hPa/decade
  nao_trends_obs = (/ (nao_trends_obs/57.)*10. /) ; convert to hPa/decade
  nam_trends_C6 = (/ (nam_trends_C6/57.)*10. /) ; convert to hPa/decade
  nam_trends_A6 = (/ (nam_trends_A6/57.)*10. /) ; convert to hPa/decade
  nam_trends_obs = (/ (nam_trends_obs/57.)*10. /) ; convert to hPa/decade
  sam_trends_C6 = (/ (sam_trends_C6/36.)*10. /) ; convert to hPa/decade
  sam_trends_A6 = (/ (sam_trends_A6/36.)*10. /) ; convert to hPa/decade
  sam_trends_obs = (/ (sam_trends_obs/36.)*10. /) ; convert to hPa/decade

  nao_trends_mme_C6T = (/ (nao_trends_mme_C6T/57.)*10. /) ; convert to hPa/decade
  nam_trends_mme_C6T = (/ (nam_trends_mme_C6T/57.)*10. /) ; convert to hPa/decade
  sam_trends_mme_C6T = (/ (sam_trends_mme_C6T/36.)*10. /) ; convert to hPa/decade
  nao_trends_mme_A6T = (/ (nao_trends_mme_A6T/57.)*10. /) ; convert to hPa/decade
  nam_trends_mme_A6T = (/ (nam_trends_mme_A6T/57.)*10. /) ; convert to hPa/decade
  sam_trends_mme_A6T = (/ (sam_trends_mme_A6T/36.)*10. /) ; convert to hPa/decade
  nao_trends_mme_C5T = (/ (nao_trends_mme_C5T/57.)*10. /) ; convert to hPa/decade
  nam_trends_mme_C5T = (/ (nam_trends_mme_C5T/57.)*10. /) ; convert to hPa/decade
  sam_trends_mme_C5T = (/ (sam_trends_mme_C5T/36.)*10. /) ; convert to hPa/decade
 
  obs_percentile("NAM trends ",nam_trends_C6,nam_trends_obs)
  obs_percentile("NAO trends ",nao_trends_C6,nao_trends_obs)
  obs_percentile("SAM trends ",sam_trends_C6,sam_trends_obs)


  hres = True
  hres@bin_center = fspan(-.375,.375,16)
  hres@bin_bounds = fspan(-.4,.4,17)
  hres@bin_spacing = hres@bin_center(1) - hres@bin_center(0)
  hres@bin_min = min(hres@bin_bounds)
  hres@bin_max  = max(hres@bin_bounds) 
  nao_h = weighted_pdfx(nao_trends_C6,weights_C6,hres)

  nam_h = weighted_pdfx(nam_trends_C6,weights_C6,hres)

  hres@bin_center := fspan(-.575,.575,22)
  hres@bin_bounds := fspan(-.6,.6,23)
  hres@bin_spacing = hres@bin_center(1) - hres@bin_center(0)
  hres@bin_min = min(hres@bin_bounds)
  hres@bin_max  = max(hres@bin_bounds) 
  sam_h = weighted_pdfx(sam_trends_C6,weights_C6,hres)

  bres = True
  bres@gsnDraw = False
  bres@gsnFrame = False
  bres@gsnYRefLine = 0.0
;  bres@gsnXRefLine = 0.0
  bres@gsnXRefLineColor = "gray50"
  bres@gsnXRefLineThicknessF = 1.5
  bres@vpWidthF = 0.65
  bres@vpHeightF = 0.48
  bres@gsnXYBarChart = True
  bres@trYMinF = 0.0
  bres@tmXBLabelFontHeightF = 0.03;0.0225
  bres@tmXTLabelFontHeightF = 0.03;0.0225
  bres@tmYLLabelFontHeightF = 0.03;0.0225
  bres@tmXBLabelDeltaF = -0.25
  bres@tmYLLabelDeltaF = -0.15
;  bres@gsnCenterStringFontHeightF = 0.0255
  bres@tiMainOn = False
  bres@xyLineThicknessF = 1.5
  bres@tiYAxisString = "% of simulations"
  bres@tiYAxisOffsetXF = 0.005
  bres@tiXAxisOffsetYF = 0.004
  bres@tiYAxisFontHeightF = 0.03
  bres@tiXAxisFontHeightF = bres@tiYAxisFontHeightF 
  bres@trXMinF = -.45
  bres@trXMaxF = .45
  bres@tmXBMode = "Explicit"
  bres@tmXBValues = fspan(-.4,.4,9)
  bres@tmXBLabels := (/"-0.4","","-0.2","","0","","0.2","","0.4"/)
  bres@gsnXYBarChartColors = cmap0(10,:)
;  bres@gsnXYBarChartOutlineOnly = True
  bres@xyLineColor = "white"
  res@gsnLeftString = "" ;syear+"-"+eyear


  bplot = new(3,graphic)
  bres@tfPolyDrawOrder = "PreDraw"
  bres@tiYAxisString = "" ;"% of simulations"
  bres@tiXAxisString = "(hPa 10yr~S~-1~N~)"
  bplot(1) = gsn_csm_xy(wks, nao_h&time, nao_h, bres)
  bres@tiYAxisString = "% of simulations"
  bplot(0) = gsn_csm_xy(wks, nam_h&time, nam_h, bres)
  bres@tiYAxisString = "" ;"% of simulations"
  bres@trXMinF = -.35
  bres@trXMaxF = .9
  bres@tmXBValues := fspan(-.6,1.1,18)
  bres@tmXBLabels := (/"-0.6","","-0.4","","-0.2","","0","","0.2","","0.4","","0.6","","0.8","","1.0",""/)
  bplot(2) = gsn_csm_xy(wks, sam_h&time, sam_h, bres)

  polyres = True
  polyres@gsLineThicknessF = 0.5
  polyres@gsLineColor = 1
  polyres@gsLineDashPattern = 2;1
  dum_refline = new(3,graphic)
  do gg = 0,2
     dum_refline(gg) = gsn_add_polyline(wks,bplot(gg),(/0,0/),(/0,100/),polyres)
  end do

  polyres@gsLineDashPattern = 0
  polyres@tfPolyDrawOrder = "PostDraw"
  polyres@gsLineThicknessF = 3.    ; add mme and obs reference lines
  polyres@gsLineColor = 5
  dum_mme2 = new(3,graphic)
  dum_mme2(1) = gsn_add_polyline(wks,bplot(1),(/nao_trends_mme_C5T,nao_trends_mme_C5T/),(/0,100/),polyres)
  dum_mme2(0) = gsn_add_polyline(wks,bplot(0),(/nam_trends_mme_C5T,nam_trends_mme_C5T/),(/0,100/),polyres)
  dum_mme2(2) = gsn_add_polyline(wks,bplot(2),(/sam_trends_mme_C5T,sam_trends_mme_C5T/),(/0,100/),polyres)

  polyres@gsLineColor = 9
  dum_mme = new(3,graphic)
;  dum_mme(1) = gsn_add_polyline(wks,bplot(1),(/dim_avg_n(nao_trends_C6,0),dim_avg_n(nao_trends_C6,0)/),(/0,100/),polyres)
;  dum_mme(0) = gsn_add_polyline(wks,bplot(0),(/dim_avg_n(nam_trends_C6,0),dim_avg_n(nam_trends_C6,0)/),(/0,100/),polyres)
;  dum_mme(2) = gsn_add_polyline(wks,bplot(2),(/dim_avg_n(sam_trends_C6,0),dim_avg_n(sam_trends_C6,0)/),(/0,100/),polyres)
  dum_mme(1) = gsn_add_polyline(wks,bplot(1),(/nao_trends_mme_C6T,nao_trends_mme_C6T/),(/0,100/),polyres)
  dum_mme(0) = gsn_add_polyline(wks,bplot(0),(/nam_trends_mme_C6T,nam_trends_mme_C6T/),(/0,100/),polyres)
  dum_mme(2) = gsn_add_polyline(wks,bplot(2),(/sam_trends_mme_C6T,sam_trends_mme_C6T/),(/0,100/),polyres)

  polyres@gsLineColor = 3
  dum_mme3 = new(3,graphic)
;  dum_mme3(1) = gsn_add_polyline(wks,bplot(1),(/dim_avg_n(nao_trends_A6,0),dim_avg_n(nao_trends_A6,0)/),(/0,100/),polyres)
;  dum_mme3(0) = gsn_add_polyline(wks,bplot(0),(/dim_avg_n(nam_trends_A6,0),dim_avg_n(nam_trends_A6,0)/),(/0,100/),polyres)
;  dum_mme3(2) = gsn_add_polyline(wks,bplot(2),(/dim_avg_n(sam_trends_A6,0),dim_avg_n(sam_trends_A6,0)/),(/0,100/),polyres)
  dum_mme3(1) = gsn_add_polyline(wks,bplot(1),(/nao_trends_mme_A6T,nao_trends_mme_A6T/),(/0,100/),polyres)
  dum_mme3(0) = gsn_add_polyline(wks,bplot(0),(/nam_trends_mme_A6T,nam_trends_mme_A6T/),(/0,100/),polyres)
  dum_mme3(2) = gsn_add_polyline(wks,bplot(2),(/sam_trends_mme_A6T,sam_trends_mme_A6T/),(/0,100/),polyres)


;  print("NAM observations: "+nam_trends_obs)
  print("NAO observations: "+nao_trends_obs)
  print("NAO Hist: "+nao_trends_C6)
  print("NAO AMIP: "+nao_trends_A6)

;  print("SAM observations: "+sam_trends_obs)
;  print("SAM AMIP Indiv: "+sam_trendsA)
;  print("SAM AMIP: "+dim_avg_n(sam_trendsA,0))

  polyres@gsLineColor = 1
  polyres@gsLineThicknessF = 1.5
  dum_obs = new((/3,6/),graphic)
  dum_obs(1,0) = gsn_add_polyline(wks,bplot(1),(/nao_trends_obs(0),nao_trends_obs(0)/),(/0,100/),polyres)   ;jra55
  dum_obs(0,0) = gsn_add_polyline(wks,bplot(0),(/nam_trends_obs(0),nam_trends_obs(0)/),(/0,100/),polyres)
  dum_obs(2,0) = gsn_add_polyline(wks,bplot(2),(/sam_trends_obs(0),sam_trends_obs(0)/),(/0,100/),polyres)

  do gg = 1,dimsizes(nao_trends_obs&E)-1
     dum_obs(1,gg) = gsn_add_polyline(wks,bplot(1),(/nao_trends_obs(gg),nao_trends_obs(gg)/),(/0,100/),polyres)
     dum_obs(0,gg) = gsn_add_polyline(wks,bplot(0),(/nam_trends_obs(gg),nam_trends_obs(gg)/),(/0,100/),polyres)
     dum_obs(2,gg) = gsn_add_polyline(wks,bplot(2),(/sam_trends_obs(gg),sam_trends_obs(gg)/),(/0,100/),polyres)
  end do

  qsort(nao_trends_C6)
  qsort(nao_trends_A6)

  print("NAO observations: "+nao_trends_obs)
  print("NAO Hist: "+nao_trends_C6)
  print("NAO AMIP: "+nao_trends_A6)

  qsort(nam_trends_C6)
  qsort(nam_trends_A6)


  print("NAM Hist: "+nam_trends_C6)
  print("NAM AMIP: "+nam_trends_A6)
  print("NAM observations: "+nam_trends_obs)

  gres = True
  gres@ItemSpacePercent = 2.
  gres@LineLabelWhiteSpacePercent = 1
  
  lineres = True
  lineres@LineLengthPercent = 2.5                         ; expressed as %, 0->100, length of line
  lineres@lgDashIndexes = 0

  textres = True
  textres@lgLabelFontHeights = 0.008

  gres@YPosPercent = 0.5
  gres@XPosPercent = 20.
  lineres@lgLineColors = 1
  lineres@lgLineThicknesses = 1.5 
  textres@lgLabels = "Observations"
  simple_legend_ndc(wks,gres,lineres,textres)

  gres@XPosPercent = 32
  lineres@lgLineColors = 5
  lineres@lgLineThicknesses = 3
  textres@lgLabels = "CMIP5 Hist MME ("+cmip5_num+")"
  simple_legend_ndc(wks,gres,lineres,textres)

  gres@XPosPercent = 48.5
  lineres@lgLineColors = 9
  textres@lgLabels = "CMIP6 Hist MME ("+cmip6_num+")"
  simple_legend_ndc(wks,gres,lineres,textres)

  gres@XPosPercent = 65
  lineres@lgLineColors = 3
  textres@lgLabels = "CMIP6 AMIP MME ("+cmip6A_num+")"
  simple_legend_ndc(wks,gres,lineres,textres)

  panres2 = True
  panres2@gsnPanelTop = 0.43
  panres2@gsnPanelBottom = 0.05
  panres2@gsnFrame = False  
  panres2@gsnPanelDebug = False
  panres2@gsnPanelYF = (/.425,.425,.425,.1675,.1675,.1675/)
  gsn_panel(wks,(/tplot_C5(0),tplot_C5(1),tplot_C5(2),bplot(0),bplot(1),bplot(2)/),(/2,3/),panres2)
;- - - - - - - - - - - - - - - - - - - - - - -
; Set titles for panels 7-12
;
  textres = True
  textres@txFontHeightF = 0.01075;0.00925

  gsn_text_ndc(wks,"(g) NAM",0.335,0.425,textres)
  gsn_text_ndc(wks,"(h) NAO",0.56,0.425,textres)
  gsn_text_ndc(wks,"(i) SAM",0.785,0.425,textres)
  gsn_text_ndc(wks,"(j) NAM (1959-2014)",0.285,0.185,textres)
  gsn_text_ndc(wks,"(k) NAO (1959-2014)",0.51,0.185,textres)
  gsn_text_ndc(wks,"(l) SAM (1979-2014)",0.735,0.185,textres)

  textres@txFont = 22
  textres@txFontHeightF = 0.018
;  gsn_text_ndc(wks,"Model Evaluation of NAM, NAO and SAM in boreal winter",0.5,0.98,textres)
  gsn_text_ndc(wks,"Model Evaluation of NAM, NAO and SAM in boreal winter",0.5,0.96,textres)

  textres@txFont = 21
  textres@txFontHeightF = 0.015
  textres@txFontColor = "gray40"
;  gsn_text_ndc(wks,"Mean sea level pressure spatial patterns",0.5,.95,textres)
  textres@txAngleF = 90.
  gsn_text_ndc(wks,"Mean sea level pressure spatial patterns",0.15,0.7,textres)
;  textres@txJust = "CenterLeft"
;  gsn_text_ndc(wks,"Taylor Statistics",0.2,0.442,textres)
  gsn_text_ndc(wks,"Taylor Statistics",0.15,0.34,textres)  
;  gsn_text_ndc(wks,"Simulated and Observed Linear Trends",0.2,0.2075,textres)
  gsn_text_ndc(wks,"Linear Trends",0.15,0.1,textres)  
;  gsn_text_ndc(wks,"Mean sea level pressure spatial patterns",0.5,.95,textres)
;  gsn_text_ndc(wks,"Taylor Statistics",0.5,0.429,textres)  
;  gsn_text_ndc(wks,"Simulated and Observed Linear Trends",0.5,0.1975,textres)  
;- - - - - - - - - - - - - - - - - - - - - - -
; Taylor legend
;
;  lres = True
;  lres@gsFillColor = (/1,1,1,1/) ;0.5/)
;  gsn_polygon_ndc(wks,(/.84,.91,.91,.84,.84/),(/.63,.63,.68,.68,.63/),lres)

  mares = True
  mares@gsMarkerIndex = 16
  mares@gsMarkerSizeF   = 0.008
  tlres = True
  tlres@txFontHeightF = 0.009;0.007
  tlres@txJust = "CenterLeft"

  yval = .2275
  mares@gsMarkerColor = 1
;  gsn_polymarker_ndc(wks,.225,yval,mares)
;  gsn_text_ndc(wks,"Observations",.2325,yval,tlres)
;  gsn_polymarker_ndc(wks,.19,yval,mares)
;  gsn_text_ndc(wks,"Observations",.197,yval,tlres)
  gsn_polymarker_ndc(wks,.18,yval,mares)
  gsn_text_ndc(wks,"Observations",.187,yval,tlres)
  mares@gsMarkerIndex = 2
  mares@gsMarkerSizeF   = 0.006
  mares@gsMarkerColor = 5
;  gsn_polymarker_ndc(wks,.305,yval,mares)
;  gsn_text_ndc(wks,"CMIP5 Run (Hist)",.3125,yval,tlres)
  gsn_polymarker_ndc(wks,.27,yval,mares)
  gsn_text_ndc(wks,"CMIP5 Hist each",.277,yval,tlres)
  mares@gsMarkerColor = 4
;  gsn_polymarker_ndc(wks,.405,yval,mares)
;  gsn_text_ndc(wks,"CMIP6 Run (Hist)",.4125,yval,tlres)
  gsn_polymarker_ndc(wks,.38,yval,mares)
  gsn_text_ndc(wks,"CMIP6 Hist each",.387,yval,tlres)
  mares@gsMarkerIndex = 16
  mares@gsMarkerColor = 2
  mares@gsMarkerThicknessF = 3.0
;  gsn_polymarker_ndc(wks,.505,yval,mares)
;  gsn_text_ndc(wks,"CMIP5 MME (Hist)",.5125,yval,tlres)
  gsn_polymarker_ndc(wks,.49,yval,mares)
  gsn_text_ndc(wks,"CMIP5 Hist MME",.497,yval,tlres)
  mares@gsMarkerColor = 9
;  gsn_polymarker_ndc(wks,.605,yval,mares)
;  gsn_text_ndc(wks,"CMIP6 MME (Hist)",.6125,yval,tlres)
  gsn_polymarker_ndc(wks,.60,yval,mares)
  gsn_text_ndc(wks,"CMIP6 Hist MME",.607,yval,tlres)
  mares@gsMarkerColor = 3
;  gsn_polymarker_ndc(wks,.705,yval,mares)
;  gsn_text_ndc(wks,"CMIP6 MME (AMIP)",.7125,yval,tlres)
  gsn_polymarker_ndc(wks,.71,yval,mares)
  gsn_text_ndc(wks,"CMIP6 AMIP MME",.717,yval,tlres)

  pmres0 = True
;  pmres0@gsLineThicknessF     = 1.          ; polymarker size
;  pmres0@gsLineColor = "gray25"
;  gsn_polyline_ndc(wks,(/.7038,.7188/),(/.475,.4805/),pmres0)
;  gsn_polyline_ndc(wks,(/.6993,.7143/),(/.4785,.484/),pmres0)
;  gsn_polyline_ndc(wks,(/.7113,.7133/),(/.483,.4775/),pmres0)
;  gsn_polyline_ndc(wks,(/.7045,.7065/),(/.4815,.476/),pmres0)
  pmres0@gsFillColor = "gray25"
  pmres0@gsFillScaleF = 0.7
  pmres0@gsFillIndex = 3 ;11
  lres = True
  lres@gsLineDashPattern = 0
  lres@gsLineThicknessF = 1.
  x1 = 0.695
  x2 = 0.715
  y1 = 0.47 ; 0.475
  y2 = 0.49 ; 0.485
  gsn_polygon_ndc(wks, (/x1, x2, x2, x1/), (/y1, y1, y2, y2/), pmres0)
  gsn_polyline_ndc(wks, (/x1, x2, x2, x1, x1/), (/y1, y1, y2, y2, y1/), lres)
  gsn_text_ndc(wks," < 80% of runs~C~ match MME sign",x2+0.005,0.48,tlres)

;  pmres = True
;  pmres@gsMarkerIndex     = 16          ; polymarker style
;  pmres@gsMarkerSizeF     = 2.          ; polymarker size
;  pmres@gsMarkerColor = pmres0@gsFillColor ; pmres0@gsLineColor
;  gsn_polymarker_ndc(wks,.225,.484,pmres)
;  gsn_polymarker_ndc(wks,.2275,.479,pmres)
;  gsn_polymarker_ndc(wks,.2225,.479,pmres)
  pmres = pmres0
  pmres@gsFillScaleF = 0.55
  pmres@gsFillIndex = 17
  x1 = 0.20
  x2 = 0.22
  y1 = 0.47 ; 0.475
  y2 = 0.49 ; 0.485
  gsn_polygon_ndc(wks, (/x1, x2, x2, x1/), (/y1, y1, y2, y2/), pmres)
  gsn_polyline_ndc(wks, (/x1, x2, x2, x1, x1/), (/y1, y1, y2, y2, y1/), lres)
;  gsn_text_ndc(wks," not significant~C~ at 5% level",x2+0.005,0.48,tlres)
  gsn_text_ndc(wks," not significant~C~ at 10% level",x2+0.005,0.48,tlres)


;  gsn_polymarker_ndc(wks,.6575,.365,mares)
;  gsn_text_ndc(wks,"Observations",.6675,.365,tlres)
;  mares@gsMarkerIndex = 2
;  mares@gsMarkerSizeF   = 0.006
;  mares@gsMarkerColor = 5
;  gsn_polymarker_ndc(wks,.6575,.355,mares)
;  gsn_text_ndc(wks,"CMIP5 Model (Hist)",.6675,.355,tlres)
;  mares@gsMarkerColor = 4
;  gsn_polymarker_ndc(wks,.6575,.345,mares)
;  gsn_text_ndc(wks,"CMIP6 Model (Hist)",.6675,.345,tlres)
;  mares@gsMarkerIndex = 16
;  mares@gsMarkerColor = 2
;  mares@gsMarkerThicknessF = 3.0
;  gsn_polymarker_ndc(wks,.6575,.335,mares)
;  gsn_text_ndc(wks,"CMIP5 MME (Hist)",.6675,.335,tlres)
;  mares@gsMarkerColor = 9
;  gsn_polymarker_ndc(wks,.6575,.325,mares)
;  gsn_text_ndc(wks,"CMIP6 MME (Hist)",.6675,.325,tlres)
;  mares@gsMarkerColor = 3
;  gsn_polymarker_ndc(wks,.6575,.315,mares)
;  gsn_text_ndc(wks,"CMIP6 MME (AMIP)",.6675,.315,tlres)
;  drawNDCGrid(wks)
  frame(wks)
  delete(wks)
  system("convert -density 216 -trim +repage -border 10 -bordercolor white -flatten "+ofile+"."+file_type+" "+ofile+".png")
end