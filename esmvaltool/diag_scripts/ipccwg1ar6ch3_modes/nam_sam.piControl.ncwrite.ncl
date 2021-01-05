; This script calculates the NAM/SAM for CMIP6 piControl simulations.
;
;----Changes required to work on a different platform------
;  This script needs to be run twice to create files used by nam_sam.trends.damip.summary.bar.ncl.
;  Option is found at line 17 ("djf" or "jja"), resulting in 2 combinations of settings. This
;  combination of options could be called within a single large do loop if one wants to alter the script.
;
; Other necessary changes:
; - Line 19 update to new local path of observational NAM/SAM-metrics file (created with nam_sam.damip.ncwrite.ncl)
;
; - Line 22 update to path of local CMIP6 piControl repository
;
; - Line 49 update to new local path of NAM/SAM CMIP6 piControl file (created in this script)
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

  ;--------------------------
  ; CMIP6 datasets
  ;--------------------------
  atts := True
  atts@project = "CMIP6"
  ;delete(atts@exp)
  cmip6 = select_metadata_by_atts(info_items, atts)

  ; piControl experiment
  atts@exp = "piControl"
  piControl_cmip6 = select_metadata_by_atts(cmip6, atts)
  piControl_cmip6_datasets = metadata_att_as_array(piControl_cmip6, "dataset")
  piControl_cmip6_ensembles = metadata_att_as_array(piControl_cmip6, "ensemble")
  dim_cmip6_piControl = dimsizes(piControl_cmip6_datasets)

    ; set start and end year
  syearA = diag_script_info@start_year
  eyearA = diag_script_info@end_year
  ntime = (eyearA - syearA + 1) * 12

  ; Create work directory
  out_path = config_user_info@work_dir
  system("mkdir -p " + out_path)



  seas = diag_script_info@season  ; djf or jja
;----------------------------------------------------------
  godir = systemfunc ("csh -c 'cd " + out_path + " ; pwd'") ; first go to work directory
  print(godir)
  ncpath = systemfunc ("ls ../../../work/diag_calc_historical/diag_historical_"+seas+"/nam_nao_sam.obs."+seas+".1958_1979-2014.n*") ; print .nc full path
  print(ncpath)
  a = addfile(ncpath ,"r")
  nam_patterns_obs = a->nam_patterns
  print(nam_patterns_obs)
;----------------------------------------------------------
;----------------------------------------------------------
  paths := piControl_cmip6_datasets
  syear = paths@start_year
  eyear = paths@end_year



  nens = dim_cmip6_piControl

  print("Number of CMIP6 piControl simulations found: " + nens)
  model = piControl_cmip6_datasets
  emem  = piControl_cmip6_ensembles
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

  ; if (seas.eq."jja") then
     do gg = 0,dimsizes(paths)-1
        arrF := read_data(piControl_cmip6[gg])
        arrF&time = cd_calendar(arrF&time,1)

        if (gg.ge.1.and.isatt(arrF,"is_all_missing")) then   ;
           continue
        end if

        do hh = 0,1     ; nam, sam
           if (hh.eq.0) then
              arr := arrF                   ; use 57 years for NAM statistics
           else
              arr := arrF(:36*12-1,:,:)     ; use 36 years for SAM statistics
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
           if (hh.eq.1) then   ; SAM
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
           nam_patterns := new((/dimsizes(paths),dimsizes(nam_patterns_obs&lat),dimsizes(nam_patterns_obs&lon)/),typeof(nam_patterns_obs))
           nam_patterns!0 = "E"
           nam_patterns&E = ispan(0,dimsizes(paths)-1,1)
           nam_patterns!1 = "lat"
           nam_patterns&lat = nam_patterns_obs&lat
           nam_patterns!2 = "lon"
           nam_patterns&lon = nam_patterns_obs&lon
           nam_patterns = nam_patterns@_FillValue
           sam_patterns := nam_patterns

           nam_timeseries := new((/dimsizes(paths),dimsizes(finarr_nam_pc)/),float)
           nam_timeseries!0 = "E"
           nam_timeseries&E = nam_patterns&E
           nam_timeseries!1 = "time"
           nam_timeseries&time = ispan(1958,2014,1)
           nam_timeseries = nam_timeseries@_FillValue
           sam_timeseries := nam_timeseries(:,{1979:})
           sam_timeseries!1 = "time2"

           nam_pcvari := nam_timeseries(:,0)
           sam_pcvari := nam_pcvari
           nam_pc_trends := nam_pcvari
           sam_pc_trends := nam_pcvari

           ensemble_assign!0 = "E"
           ensemble_assign&E = nam_patterns&E
        end if

        nam_patterns(gg,:,:)  = (/ linint2_Wrap(finarr_nam&lon,finarr_nam&lat,finarr_nam,True,nam_patterns&lon,nam_patterns&lat,0) /)

        sam_patterns(gg,:,:)  = (/ linint2_Wrap(finarr_sam&lon,finarr_sam&lat,finarr_sam,True,sam_patterns&lon,sam_patterns&lat,0) /)

        tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_nam_pc)-1,1),finarr_nam_pc,False,True,0)
        nam_pc_trends(gg) = (/ tttt@slope*dimsizes(finarr_nam_pc)  /)
        tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_sam_pc)-1,1),finarr_sam_pc,False,True,0)
        sam_pc_trends(gg) = (/ tttt@slope*dimsizes(finarr_sam_pc)  /)
;        print(nam_pc_trends(gg)+"  "+nao_pc_trends(gg)+"  "+sam_pc_trends(gg))

        nam_timeseries(gg,:) = (/ finarr_nam_pc /)
        sam_timeseries(gg,:) = (/ finarr_sam_pc /)

        nam_pcvari(gg) = (/ finarr_nam_pcvar /)
        sam_pcvari(gg) = (/ finarr_sam_pcvar /)
        print("Done with "+paths(gg))
     end do
     fn = out_path + "nam_nao_sam.piControl."+seas+".57_36yr.nc"
     system("rm -f " + fn)
     z = addfile(fn,"c")
     z@source = systemfunc("pwd")+"/"+get_script_name()
     z->ensemble_assign = ensemble_assign
     z->nam_patterns = nam_patterns
     z->sam_patterns = sam_patterns
     z->nam_timeseries = nam_timeseries
     z->sam_timeseries = sam_timeseries
     z->nam_pc_trends = nam_pc_trends
     z->sam_pc_trends = sam_pc_trends
     z->nam_pcvari    = nam_pcvari
     z->sam_pcvari    = sam_pcvari
     delete(z)
end

