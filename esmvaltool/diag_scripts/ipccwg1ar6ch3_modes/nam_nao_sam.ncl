; #############################################################################
; Compute NAM/NAO/SAM metrics
; Authors: Lisa Bock (DLR, Germany), Soufiane Karmouche (University of Bremen,
;          Germany) and Adam Phillips (NCAR, U.S.)
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
  obs = get_obs_list(info_items)
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
  atts@exp = "historical-rcp45"
  hist_cmip5 = select_metadata_by_atts(cmip5, atts)
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
        ;print("Done with "+iobs+" our of "+nens+", "+model(iobs))
     end do

     fn = out_path + "nam_nao_sam.obs."+syearA+"-"+eyearA+".nc"
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
     print("Done with "+gg+" our of "+nens+", "+model(gg))

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

  z = addfile(fn,"r")
  nam_taylor_stats = z->nam_tay_stat
  nao_taylor_stats = z->nao_tay_stat
  sam_taylor_stats = z->sam_tay_stat
  nam_trends = z->nam_pc_trends
  nao_trends = z->nao_pc_trends
  sam_trends = z->sam_pc_trends
  nam_patterns = z->nam_patterns
  nao_patterns = z->nao_patterns
  sam_patterns = z->sam_patterns
  ensemble_assign := z->ensemble_assign

  taylor_nam_ratio  = nam_taylor_stats(:,1:1)
  taylor_nam_patcor = nam_taylor_stats(:,0:0)
  taylor_nam_patcorZ = 0.5*(log( (1+taylor_nam_patcor) / (1-taylor_nam_patcor)))    ; convert to Z-space
  taylor_nao_ratio  = nao_taylor_stats(:,1:1)
  taylor_nao_patcor = nao_taylor_stats(:,0:0)
  taylor_nao_patcorZ = 0.5*(log( (1+taylor_nao_patcor) / (1-taylor_nao_patcor)))    ; convert to Z-space
  taylor_sam_ratio  = sam_taylor_stats(:,1:1)
  taylor_sam_patcor = sam_taylor_stats(:,0:0)
  taylor_sam_patcorZ = 0.5*(log( (1+taylor_sam_patcor) / (1-taylor_sam_patcor)))    ; convert to Z-space

  nam_em = nam_patterns(:max(ensemble_assign)-1,:,:)
  nam_em = nam_em@_FillValue
  nao_em = nam_em
  sam_em = nam_em
  tay_patcor_nam  = new(max(ensemble_assign),double)
  tay_ratio_nam   = tay_patcor_nam
  tay_patcor_nao  = tay_patcor_nam
  tay_ratio_nao   = tay_patcor_nam
  tay_patcor_sam  = tay_patcor_nam
  tay_ratio_sam   = tay_patcor_nam
  do gg = 1,max(ensemble_assign)         ; calculate ensemble means
     wind := ind(ensemble_assign.eq.gg)
     if (dimsizes(wind).eq.1) then
        nam_em(gg-1,:,:) = (/ nam_patterns(wind,:,:) /)
        nao_em(gg-1,:,:) = (/ nao_patterns(wind,:,:) /)
        sam_em(gg-1,:,:) = (/ sam_patterns(wind,:,:) /)
        tay_patcor_nam(gg-1) = (/ taylor_nam_patcorZ(wind,0) /)
        tay_ratio_nam(gg-1) = (/ taylor_nam_ratio(wind,0) /)
        tay_patcor_nao(gg-1) = (/ taylor_nao_patcorZ(wind,0) /)
        tay_ratio_nao(gg-1) = (/ taylor_nao_ratio(wind,0) /)
        tay_patcor_sam(gg-1) = (/ taylor_sam_patcorZ(wind,0) /)
        tay_ratio_sam(gg-1) = (/ taylor_sam_ratio(wind,0) /)
     else
        nam_em(gg-1,:,:) = (/ dim_avg_n(nam_patterns(wind,:,:),0) /)
        nao_em(gg-1,:,:) = (/ dim_avg_n(nao_patterns(wind,:,:),0) /)
        sam_em(gg-1,:,:) = (/ dim_avg_n(sam_patterns(wind,:,:),0) /)
        tay_patcor_nam(gg-1) = (/ dim_avg_n(taylor_nam_patcorZ(wind,:),0) /)
        tay_ratio_nam(gg-1)  = (/ dim_avg_n(taylor_nam_ratio(wind,:),0) /)
        tay_patcor_nao(gg-1) = (/ dim_avg_n(taylor_nao_patcorZ(wind,:),0) /)
        tay_ratio_nao(gg-1)  = (/ dim_avg_n(taylor_nao_ratio(wind,:),0) /)
        tay_patcor_sam(gg-1) = (/ dim_avg_n(taylor_sam_patcorZ(wind,:),0) /)
        tay_ratio_sam(gg-1)  = (/ dim_avg_n(taylor_sam_ratio(wind,:),0) /)
     end if
  end do
  mme_nam = dim_avg_n_Wrap(nam_em,0)
  mme_nao = dim_avg_n_Wrap(nao_em,0)
  mme_sam = dim_avg_n_Wrap(sam_em,0)

  nao_patterns_obsLF = lonFlip(nao_patterns_obs)
  mme_naoLF = lonFlip(mme_nam)

  tay_stats_hist_nam_cmip6EM = new((/1,3/),double)
;  tay_stats_hist_nam_cmip6EM = (/ taylor_stats(mme_nam({20:},:), nam_patterns_obs({20:},:), w({20:}), 0) /)
  tay_stats_hist_nao_cmip6EM = new((/1,3/),double)
;  tay_stats_hist_nao_cmip6EM = (/ taylor_stats(mme_naoLF({20:80},{-90:40}), nao_patterns_obsLF({20:80},{-90:40}), w({20:80}), 0) /)
  tay_stats_hist_sam_cmip6EM = new((/1,3/),double)
;  tay_stats_hist_sam_cmip6EM = (/ taylor_stats(mme_sam({:-20},:), sam_patterns_obs({:-20},:), w({:-20}), 0) /)

  tay_stats_hist_nam_cmip6EM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_nam))-1)/ ((2.71828^(2*dim_avg(tay_patcor_nam))+1))) /)   ; convert back from z-space
  tay_stats_hist_nam_cmip6EM(0,1) = (/ dim_avg(tay_ratio_nam) /)
  tay_stats_hist_nao_cmip6EM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_nao))-1)/ ((2.71828^(2*dim_avg(tay_patcor_nao))+1))) /)   ; convert back from z-space
  tay_stats_hist_nao_cmip6EM(0,1) = (/ dim_avg(tay_ratio_nao) /)
  tay_stats_hist_sam_cmip6EM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_sam))-1)/ ((2.71828^(2*dim_avg(tay_patcor_sam))+1))) /)   ; convert back from z-space
  tay_stats_hist_sam_cmip6EM(0,1) = (/ dim_avg(tay_ratio_sam) /)
;--------------------------------------------------
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
  nam_em := nam_patterns_C5(:max(ea_c5)-1,:,:)
  nam_em = nam_em@_FillValue
  nao_em := nam_em
  sam_em := nam_em

  ;--------------------------------------------------------
  ; other processing steps?
  ;--------------------------------------------------------

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
        nam_em(gg-1,:,:) = (/ nam_patterns_C5(wind,:,:) /)
        nao_em(gg-1,:,:) = (/ nao_patterns_C5(wind,:,:) /)
        sam_em(gg-1,:,:) = (/ sam_patterns_C5(wind,:,:) /)
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
        nam_em(gg-1,:,:) = (/ dim_avg_n(nam_patterns_C5(wind,:,:),0) /)
        nao_em(gg-1,:,:) = (/ dim_avg_n(nao_patterns_C5(wind,:,:),0) /)
        sam_em(gg-1,:,:) = (/ dim_avg_n(sam_patterns_C5(wind,:,:),0) /)
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
  mme_nam_C5 = dim_avg_n_Wrap(nam_em,0)
  mme_nao_C5 = dim_avg_n_Wrap(nao_em,0)
  mme_sam_C5 = dim_avg_n_Wrap(sam_em,0)
  delete([/nam_trends_mme_C5,nao_trends_mme_C5,sam_trends_mme_C5/])

  mme_nao_C5LF = lonFlip(mme_nam_C5)

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
  res@gsnCenterStringFontHeightF = 0.031
;  res@txFontHeightF = 0.018
  res@gsnPolarLabelFontHeightF = 0.0


  res@cnLevels = fspan(-500,500,11)
  cmap = read_colormap_file("/work/bd0854/b380749/student/ESMValTool/ESMValTool/esmvaltool/diag_scripts/ipccwg1ar6ch3_modes/AR6_colormaps_NCL/temp_div_12.rgb")
  res@cnFillColors = cmap

  plot = new(6,graphic)
  res@gsnPolar = "NH"
  res@mpMinLatF    = 20.
  res@gsnCenterString = "(b) NAO - Observations"
  plot(1) = gsn_csm_contour_map_polar(wks,nao_patterns_obs,res)
  res@gsnCenterString = "(e) NAO - Multimodel Ensemble"
  plot(4) = gsn_csm_contour_map_polar(wks,mme_nao,res)

  res@gsnCenterString = "(a) NAM - Observations"
  plot(0) = gsn_csm_contour_map_polar(wks,nam_patterns_obs,res)
  res@gsnCenterString = "(d) NAM - Multimodel Ensemble"
  plot(3) = gsn_csm_contour_map_polar(wks,mme_nam,res)

  res@gsnPolar = "SH"
  res@mpMaxLatF    = -20.
  delete(res@mpMinLatF)
  res@gsnCenterString = "(c) SAM - Observations"
  plot(2) = gsn_csm_contour_map_polar(wks,sam_patterns_obs,res)
  res@gsnCenterString = "(f) SAM - Multimodel Ensemble"
  plot(5) = gsn_csm_contour_map_polar(wks,mme_sam,res)

  panres = True
  panres@gsnPanelBottom = 0.43
  panres@gsnPanelLabelBar = True
  panres@gsnPanelYWhiteSpacePercent = 3.0
  panres@gsnPanelXWhiteSpacePercent = 3.0
  panres@pmLabelBarHeightF = 0.05
  panres@pmLabelBarWidthF = 0.38
  panres@pmLabelBarOrthogonalPosF = 0.01
  panres@lbTitleOn = False
  panres@lbBoxLineColor = "gray70"
  panres@lbLabelFontHeightF = 0.011
  panres@lbBoxEndCapStyle = "TriangleBothEnds"
  panres@lbTitleOn = True
  panres@lbTitleString = "Pa"
  panres@lbTitlePosition  = "Bottom"              ; title position
  panres@lbTitleFontHeightF= .010                ; make title smaller
  panres@txFontHeightF = 0.022
  panres@gsnFrame = False
  gsn_panel(wks,plot,(/2,3/),panres)

;- - - - - - - - - - - - - - - - - -
; Taylor plots, panels 7-9
;
  tres                 = True                     ; diagram mods desired
  colors = new((/dimsizes(nam_taylor_stats(:,0))/),string)
  colors = "red3"
  tres@Colors          = colors       ; marker colors
;  tres@caseLabels      = model
  markers = ensemble_assign+1
  markers = 2
  tres@Markers         = markers        ; marker styles
  tres@markerTxYOffset = 0.04                     ; offset btwn marker & label
  tres@gsMarkerSizeF   = 0.006                    ; marker size
  tres@txFontHeightF   = 0.0225                    ; text size
;  tres@varLabels = ensemble_assign
  tres@stnRad         = (/ 0.5, 1.5 /)     ; additional standard radii
  tres@ccRays          = (/ 0.4, 0.9 /)     ; correllation rays
  tres@ccRays_color    = "LightGray"        ; default is "black"

  tres@centerDiffRMS   = True             ; RMS 'circles'
  tres@centerDiffRMS_color = "LightGray"    ; default is "black"/)

  tres@taylorDraw  = False                          ; don't draw
  tres@taylorFrame = False
  tres@trXMaxF = 2.0



  ;tplot = new(3,graphic)
  ;tplot_C5 = tplot



  tplot = new(3,graphic)
  tplot_C5 = tplot
  tplot(1) = taylor_diagram(wks, nao_taylor_stats(:,1:1), nao_taylor_stats(:,0:0), tres)
;  print(taylor_nam_patcor+" "+taylor_nam_ratio)
  tplot(0) = taylor_diagram(wks, nam_taylor_stats(:,1:1), nam_taylor_stats(:,0:0), tres)
  tplot(2) = taylor_diagram3(wks, sam_taylor_stats(:,1:1), sam_taylor_stats(:,0:0), tres)

  colors2 = new((/dimsizes(nao_taylor_stats_C5(:,0))/),string)
  colors2 = "CadetBlue3"
  tres@Colors          :=   colors2
  markers2 = new(dimsizes(nao_taylor_stats_C5(:,0)),integer)
  markers2 = 2
  tres@Markers := markers2
  tres@tfPolyDrawOrder = "PreDraw"
  tplot_C5(1) = taylor_diagram(wks, nao_taylor_stats_C5(:,1:1), nao_taylor_stats_C5(:,0:0), tres)
  tplot_C5(0) = taylor_diagram(wks, nam_taylor_stats_C5(:,1:1), nam_taylor_stats_C5(:,0:0), tres)
  tplot_C5(2) = taylor_diagram3(wks, sam_taylor_stats_C5(:,1:1), sam_taylor_stats_C5(:,0:0), tres)
  do gg =0,2
     overlay(tplot_C5(gg),tplot(gg))
  end do

;  printVarSummary(taylor_nao_ratio)
  tplot2 = new(3,graphic)
  colors := new(5,string)
  marker = new(5,integer)
  colors = "black"
  marker = 1
  tres@Colors          := colors       ; marker colors
  tres@Markers         := marker
  tres@gsMarkerSizeF   = 0.028
  tplot2(1) = taylor_diagram(wks, nao_taylor_stats_obs(:,1:1), nao_taylor_stats_obs(:,0:0), tres)
  tplot2(0) = taylor_diagram(wks, nam_taylor_stats_obs(:,1:1), nam_taylor_stats_obs(:,0:0), tres)
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


  tplot3a = new(3,graphic)
  tplot3b = new(3,graphic)
  tplot4a = new(3,graphic)
  tplot4b = new(3,graphic)
  tres@gsMarkerSizeF   = 0.008
  tres@Colors := "white"
  tres@Markers:= new_marker
  tplot3a(1) = taylor_diagram(wks, tay_stats_hist_nao_cmip5EM(0:0,1:1),tay_stats_hist_nao_cmip5EM(0:0,0:0), tres)
  tplot3a(0) = taylor_diagram(wks, tay_stats_hist_nam_cmip5EM(0:0,1:1),tay_stats_hist_nam_cmip5EM(0:0,0:0), tres)
  tplot3a(2) = taylor_diagram3(wks, tay_stats_hist_sam_cmip5EM(0:0,1:1),tay_stats_hist_sam_cmip5EM(0:0,0:0), tres)
  do gg = 0,2
     overlay(tplot_C5(gg),tplot3a(gg))
  end do
  tres@Colors := "dodgerblue"
  tres@Markers:= 2
  tres@gsMarkerThicknessF = 2.0
  tplot3b(1) = taylor_diagram(wks, tay_stats_hist_nao_cmip5EM(0:0,1:1),tay_stats_hist_nao_cmip5EM(0:0,0:0), tres)
  tplot3b(0) = taylor_diagram(wks, tay_stats_hist_nam_cmip5EM(0:0,1:1),tay_stats_hist_nam_cmip5EM(0:0,0:0), tres)
  tplot3b(2) = taylor_diagram3(wks, tay_stats_hist_sam_cmip5EM(0:0,1:1),tay_stats_hist_sam_cmip5EM(0:0,0:0), tres)
  do gg = 0,2
     overlay(tplot_C5(gg),tplot3b(gg))
  end do

  tres@gsMarkerThicknessF = 1.0
  tres@Colors := "white"
  tres@Markers:= new_marker
  tplot4a(1) = taylor_diagram(wks, tay_stats_hist_nao_cmip6EM(0:0,1:1),tay_stats_hist_nao_cmip6EM(0:0,0:0), tres)
  tplot4a(0) = taylor_diagram(wks, tay_stats_hist_nam_cmip6EM(0:0,1:1),tay_stats_hist_nam_cmip6EM(0:0,0:0), tres)
  tplot4a(2) = taylor_diagram3(wks, tay_stats_hist_sam_cmip6EM(0:0,1:1),tay_stats_hist_sam_cmip6EM(0:0,0:0), tres)
  do gg = 0,2
     overlay(tplot_C5(gg),tplot4a(gg))
  end do
  tres@Colors := "orange"
  tres@Markers:= 2
  tres@gsMarkerThicknessF = 2.0
  tplot4b(1) = taylor_diagram(wks, tay_stats_hist_nao_cmip6EM(0:0,1:1),tay_stats_hist_nao_cmip6EM(0:0,0:0), tres)
  tplot4b(0) = taylor_diagram(wks, tay_stats_hist_nam_cmip6EM(0:0,1:1),tay_stats_hist_nam_cmip6EM(0:0,0:0), tres)
  tplot4b(2) = taylor_diagram3(wks, tay_stats_hist_sam_cmip6EM(0:0,1:1),tay_stats_hist_sam_cmip6EM(0:0,0:0), tres)
  do gg = 0,2
     overlay(tplot_C5(gg),tplot4b(gg))
  end do
;- - - - - - - - - - - - - - - - - -
; Linear Trend Histograms, panels 10-12
;
  nao_trends = (/ (nao_trends/57.)*10. /) ; convert to hPa/decade
  nao_trends_obs = (/ (nao_trends_obs/57.)*10. /) ; convert to hPa/decade
  nam_trends = (/ (nam_trends/57.)*10. /) ; convert to hPa/decade
  nam_trends_obs = (/ (nam_trends_obs/57.)*10. /) ; convert to hPa/decade
  sam_trends = (/ (sam_trends/36.)*10. /) ; convert to hPa/decade
  sam_trends_obs = (/ (sam_trends_obs/36.)*10. /) ; convert to hPa/decade

  nao_trends_mme_C5T = (/ (nao_trends_mme_C5T/57.)*10. /) ; convert to hPa/decade
  nam_trends_mme_C5T = (/ (nam_trends_mme_C5T/57.)*10. /) ; convert to hPa/decade
  sam_trends_mme_C5T = (/ (sam_trends_mme_C5T/36.)*10. /) ; convert to hPa/decade


  obs_percentile("NAM trends ",nam_trends,nam_trends_obs)
  obs_percentile("NAO trends ",nao_trends,nao_trends_obs)
  obs_percentile("SAM trends ",sam_trends,sam_trends_obs)

  hres = True
  hres@gsnDraw = False
  hres@gsnFrame = False
  printMinMax(nao_trends,0)

  hres@gsnHistogramBinIntervals = fspan(-.4,.4,17)
  plot = gsn_histogram(wks,nao_trends,hres)
  nao_h = plot@NumInBins*1.
  nao_h!0 = "time"
  nao_h&time = fspan(-.375,.375,16)
  nao_h = (/ (nao_h/sum(nao_h))*100. /)
  delete([/plot/])
  if (min(nao_trends).le.min(hres@gsnHistogramBinIntervals).or.max(nao_trends).gt.max(hres@gsnHistogramBinIntervals)) then
     print("NAO trends outside histogram bin range, adjust bin range")
     printMinMax(nao_trends,0)
     exit
  end if

  hres@gsnHistogramBinIntervals = fspan(-.4,.4,17)
  plot = gsn_histogram(wks,nam_trends,hres)
  nam_h = plot@NumInBins*1.
  nam_h!0 = "time"
  nam_h&time = fspan(-.375,.375,16)
  nam_h = (/ (nam_h/sum(nam_h))*100. /)
  delete([/plot/])
  if (min(nam_trends).le.min(hres@gsnHistogramBinIntervals).or.max(nam_trends).gt.max(hres@gsnHistogramBinIntervals)) then
     print("NAM trends outside histogram bin range, adjust bin range")
     printMinMax(nam_trends,0)
     exit
  end if

  printMinMax(sam_trends,0)
  delete(hres@gsnHistogramBinIntervals)
  hres@gsnHistogramBinIntervals = fspan(-.6,.6,23)
  plot = gsn_histogram(wks,sam_trends,hres)
  sam_h = plot@NumInBins*1.
  sam_h!0 = "time"
  sam_h&time = fspan(-.575,.575,22)
  sam_h = (/ (sam_h/sum(sam_h))*100. /)
  if (min(sam_trends).le.min(hres@gsnHistogramBinIntervals).or.max(sam_trends).gt.max(hres@gsnHistogramBinIntervals)) then
     print("SAM trends outside histogram bin range, adjust bin range")
     printMinMax(sam_trends,0)
     exit
  end if
  delete([/plot,hres/])

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
  bres@tmXBLabelFontHeightF = 0.0225
  bres@tmXTLabelFontHeightF = 0.0225
  bres@tmYLLabelFontHeightF = 0.0225
;  bres@gsnCenterStringFontHeightF = 0.0255
  bres@tiMainOn = False
  bres@xyLineThicknessF = 1.5
  bres@tiYAxisString = "% of simulations"
  bres@tiYAxisOffsetXF = 0.005
  bres@tiXAxisOffsetYF = 0.004
  bres@trXMinF = -.45
  bres@trXMaxF = .45
  bres@tmXBMode = "Explicit"
  bres@tmXBValues = fspan(-.4,.4,9)
  bres@tmXBLabels = bres@tmXBValues
  bres@gsnXYBarChartColors = (/0.5,0,0,.2/)  ;(/"red4"/)  ;gray40"/)
;  bres@gsnXYBarChartOutlineOnly = True
  bres@xyLineColor = "transparent"  ;"red4"
  res@gsnLeftString = "" ;syear+"-"+eyear


  bplot = new(3,graphic)
  bres@tfPolyDrawOrder = "PreDraw"
  bres@tiYAxisString = "% of simulations"
  bres@tiXAxisString = "(hPa 10yr~S~-1~N~)"
  bplot(1) = gsn_csm_xy(wks, nao_h&time, nao_h, bres)
  bplot(0) = gsn_csm_xy(wks, nam_h&time, nam_h, bres)
  bres@trXMinF = -.65
  bres@trXMaxF = .65
  bres@tmXBValues := fspan(-.6,.6,13)
  bres@tmXBLabels := (/"-0.6","","-0.4","","-0.2","","0","","0.2","","0.4","","0.6"/)
  bplot(2) = gsn_csm_xy(wks, sam_h&time, sam_h, bres)

  polyres = True
  polyres@gsLineThicknessF = 1.5
  polyres@gsLineColor = "gray30"
  polyres@gsLineDashPattern = 1
  dum_refline = new(3,graphic)
  do gg = 0,2
     dum_refline(gg) = gsn_add_polyline(wks,bplot(gg),(/0,0/),(/0,100/),polyres)
  end do

  polyres@gsLineDashPattern = 0
  polyres@tfPolyDrawOrder = "PostDraw"
  polyres@gsLineThicknessF = 3.    ; add mme and obs reference lines
  polyres@gsLineColor = "CadetBlue3"
  dum_mme2 = new(3,graphic)
  dum_mme2(1) = gsn_add_polyline(wks,bplot(1),(/nao_trends_mme_C5T,nao_trends_mme_C5T/),(/0,100/),polyres)
  dum_mme2(0) = gsn_add_polyline(wks,bplot(0),(/nam_trends_mme_C5T,nam_trends_mme_C5T/),(/0,100/),polyres)
  dum_mme2(2) = gsn_add_polyline(wks,bplot(2),(/sam_trends_mme_C5T,sam_trends_mme_C5T/),(/0,100/),polyres)

  polyres@gsLineColor = "red3"  ;"dodgerblue"
  dum_mme = new(3,graphic)
  dum_mme(1) = gsn_add_polyline(wks,bplot(1),(/dim_avg_n(nao_trends,0),dim_avg_n(nao_trends,0)/),(/0,100/),polyres)
  dum_mme(0) = gsn_add_polyline(wks,bplot(0),(/dim_avg_n(nam_trends,0),dim_avg_n(nam_trends,0)/),(/0,100/),polyres)
  dum_mme(2) = gsn_add_polyline(wks,bplot(2),(/dim_avg_n(sam_trends,0),dim_avg_n(sam_trends,0)/),(/0,100/),polyres)

  print("NAM observations: "+nam_trends_obs)
  print("NAO observations: "+nao_trends_obs)
  print("SAM observations: "+sam_trends_obs)

  polyres@gsLineColor = "black"
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

  gres = True
  gres@YPosPercent = 95.    ; expressed as %, 0->100, sets position of top border of legend
                            ;  when gres@Position is set to its default setting of "Top" (Default = 95.)
  gres@XPosPercent = 5      ; expressed as %, 0->100, sets position of left border of legend(Default = 5.)
  gres@ItemSpacePercent = 8.

  lineres = True
  lineres@lgLineColors = (/"black","red3","CadetBlue3"/)  ;"dodgerblue"/) ; line colors
  lineres@lgLineThicknesses = (/1.5,3.,3./)                        ; line thicknesses
  lineres@LineLengthPercent = 7.                         ; expressed as %, 0->100, length of line
  lineres@lgDashIndexes = (/0,0,0/)

  textres = True
  textres@lgLabels = (/"Observations","MME (CMIP6)","MME (CMIP5)"/)  ; legend labels (required)
  textres@lgLabelFontHeights = 0.02
  do gg = 2,2
     bplot(gg) = simple_legend(wks,bplot(gg),gres,lineres,textres)
  end do

  panres2 = True
  panres2@gsnPanelTop = 0.43
  panres2@gsnFrame = False
  panres2@gsnPanelDebug = False
  panres2@gsnPanelYF = (/.42,.42,.42,.175,.175,.175/)
  gsn_panel(wks,(/tplot_C5(0),tplot_C5(1),tplot_C5(2),bplot(0),bplot(1),bplot(2)/),(/2,3/),panres2)
;- - - - - - - - - - - - - - - - - - - - - - -
; Set titles for panels 7-12
;
  textres = True
  textres@txFontHeightF = 0.012
  gsn_text_ndc(wks,"(g) NAM Taylor Statistics",0.27,0.435,textres)
  gsn_text_ndc(wks,"(h) NAO Taylor Statistics",0.515,0.435,textres)
  gsn_text_ndc(wks,"(i) SAM Taylor Statistics",0.76,0.435,textres)
  gsn_text_ndc(wks,"(j) NAM Linear Trends",0.27,0.19,textres)
  gsn_text_ndc(wks,"(k) NAO Linear Trends",0.515,0.19,textres)
  gsn_text_ndc(wks,"(l) SAM Linear Trends",0.76,0.19,textres)
;- - - - - - - - - - - - - - - - - - - - - - -
; Taylor legend
;
;  lres = True
;  lres@gsFillColor = (/1,1,1,1/) ;0.5/)
;  gsn_polygon_ndc(wks,(/.84,.91,.91,.84,.84/),(/.63,.63,.68,.68,.63/),lres)

  mares = True
  mares@gsMarkerIndex = 1
  mares@gsMarkerSizeF   = 0.018
  tlres = True
  tlres@txFontHeightF = 0.006
  tlres@txJust = "CenterLeft"

  mares@gsMarkerColor = "Black"
  gsn_polymarker_ndc(wks,.825,.42,mares)
  gsn_text_ndc(wks,"Observations",.835,.42,tlres)

  mares@gsMarkerIndex = 2
  mares@gsMarkerSizeF   = 0.006
  mares@gsMarkerColor = "CadetBlue3"
  gsn_polymarker_ndc(wks,.825,.41,mares)
  gsn_text_ndc(wks,"CMIP5 Model",.835,.41,tlres)
  mares@gsMarkerColor = "Red3"
  gsn_polymarker_ndc(wks,.825,.40,mares)
  gsn_text_ndc(wks,"CMIP6 Model",.835,.40,tlres)

  mares@gsMarkerIndex = 2
  mares@gsMarkerColor = "dodgerblue"
  mares@gsMarkerThicknessF = 3.0
  gsn_polymarker_ndc(wks,.825,.39,mares)
  gsn_text_ndc(wks,"CMIP5 MME",.835,.39,tlres)
  mares@gsMarkerColor = "orange"
  gsn_polymarker_ndc(wks,.825,.38,mares)
  gsn_text_ndc(wks,"CMIP6 MME",.835,.38,tlres)

;  drawNDCGrid(wks)
  frame(wks)
  delete(wks)
  system("convert -density 216 -trim +repage -border 10 -bordercolor white -flatten "+ofile+".eps "+ofile+".png")
end

