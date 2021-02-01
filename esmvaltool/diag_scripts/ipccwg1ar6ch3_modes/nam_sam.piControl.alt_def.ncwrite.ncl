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
  dim_MOD = ListCount(info_items)

  ;--------------------
  ; piControl datasets
  ;--------------------
  atts := True
  atts@exp = "piControl"
  if (dim_MOD.ne.ListCount(select_metadata_by_atts(info_items, atts))) then
    error_msg("f", DIAG_SCRIPT, "", "this diagnostic can only handle piControl" + \
              " experiment")
  end if

  if (ListCount(info_items) .lt. 1) then
    error_msg("f", DIAG_SCRIPT, "", "this diagnostic needs at least one " + \
              "run")
  end if

  seas = "djf"
  if (isatt(diag_script_info, "season")) then
    seas = diag_script_info@season
  end if
  seas = str_lower(seas)


  nens = dim_MOD
  
  pictl_models = metadata_att_as_array(info_items, "dataset")
  emem = metadata_att_as_array(info_items, "ensemble")
  ensemble_assign = new(nens,integer)

  enum = 1
  temp = pictl_models(0)
  do gg = 0,dimsizes(pictl_models)-1
     if (temp.eq.pictl_models(gg)) then   ; does the model name match what's in temp?
        ensemble_assign(gg) = enum ; if so, assign it the same number
     else
        enum = enum+1              ; if not, assign it the next number
        ensemble_assign(gg) = enum
        temp = pictl_models(gg)
     end if
  end do
  ensemble_assign@models = str_join(pictl_models+"/"+emem,",")

  do gg = 0, nens-1
    arrF := read_data(info_items[gg])
    arrF&time = cd_calendar(arrF&time,1)

    if (gg.ge.1.and.isatt(arrF,"is_all_missing")) then   ; 
      continue
    end if

    do hh = 0,2     ; nam, sam, sam
      if (hh.eq.0) then
        arr := arrF                   ; use 62 years for NAM statistics
      end if
      if (hh.eq.1) then
        arr := arrF(:21*12-1,:,:)     ; use 21 years for SAM statistics
      end if
      if (hh.eq.2) then
        arr := arrF(:20*12-1,:,:)     ; use 20 years for SAM statistics
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
      if (hh.ge.1) then   ; SAM
        finarr_sam_pc := dim_standardize(dim_standardize(dim_avg(arr_seas(:,{-40},:)),0) - dim_standardize(dim_avg(arr_seas(:,{-65},:)),0),0)
        finarr_sam := regCoef_n(finarr_sam_pc,arr_seas,0,0)
        copy_VarMeta(arr_seas(0,:,:),finarr_sam)
      end if
      if (hh.eq.1) then
        finarr_sam0 := finarr_sam
        finarr_sam_pc0 := finarr_sam_pc
      end if
      if (hh.eq.2) then
        finarr_sam1 := finarr_sam
        finarr_sam_pc1 := finarr_sam_pc
      end if
    end do

    if (gg.eq.0) then
      nam_patterns := new((/nens,dimsizes(finarr_nam&lat),dimsizes(finarr_nam&lon)/),typeof(finarr_nam))
      nam_patterns!0 = "E"
      nam_patterns&E = ispan(0,nens-1,1)
      nam_patterns!1 = "lat"
      nam_patterns&lat = finarr_nam&lat
      nam_patterns!2 = "lon"
      nam_patterns&lon = finarr_nam&lon
      nam_patterns = nam_patterns@_FillValue
      sam_patterns0 := nam_patterns
      sam_patterns1 := nam_patterns

      nam_timeseries := new((/nens,dimsizes(finarr_nam_pc)/),float)
      nam_timeseries!0 = "E"
      nam_timeseries&E = nam_patterns&E
      nam_timeseries!1 = "time"
      nam_timeseries&time = ispan(1958,2019,1) 
      nam_timeseries = nam_timeseries@_FillValue
      sam_timeseries0 := nam_timeseries(:,{1979:1999})
      sam_timeseries0!1 = "time2"
      sam_timeseries1 := nam_timeseries(:,{2000:})
      sam_timeseries1!1 = "time3"

      nam_pc_trends := nam_timeseries(:,0)
      sam_pc_trends0 := nam_timeseries(:,0)
      sam_pc_trends1 := nam_timeseries(:,0)

      ensemble_assign!0 = "E"
      ensemble_assign&E = nam_timeseries&E
    end if

    nam_patterns(gg,:,:)  = (/ linint2_Wrap(finarr_nam&lon,finarr_nam&lat,finarr_nam,True,nam_patterns&lon,nam_patterns&lat,0) /)
         
    sam_patterns0(gg,:,:)  = (/ linint2_Wrap(finarr_sam0&lon,finarr_sam0&lat,finarr_sam0,True,sam_patterns0&lon,sam_patterns0&lat,0) /)

    sam_patterns1(gg,:,:)  = (/ linint2_Wrap(finarr_sam1&lon,finarr_sam1&lat,finarr_sam1,True,sam_patterns1&lon,sam_patterns1&lat,0) /)

    tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_nam_pc)-1,1),finarr_nam_pc,False,True,0)
    nam_pc_trends(gg) = (/ tttt@slope*dimsizes(finarr_nam_pc)  /)            
    tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_sam_pc0)-1,1),finarr_sam_pc0,False,True,0)
    sam_pc_trends0(gg) = (/ tttt@slope*dimsizes(finarr_sam_pc0)  /)  
    tttt := dtrend_msg_n(ispan(0,dimsizes(finarr_sam_pc1)-1,1),finarr_sam_pc1,False,True,0)
    sam_pc_trends1(gg) = (/ tttt@slope*dimsizes(finarr_sam_pc1)  /)  

    nam_timeseries(gg,:) = (/ finarr_nam_pc /)
    sam_timeseries0(gg,:) = (/ finarr_sam_pc0 /)
    sam_timeseries1(gg,:) = (/ finarr_sam_pc1 /)
    print("Done with "+gg+" out of "+nens)
  end do

  fn = out_path + "nam_sam.alt_def.piControl."+seas+".62_21_20yr.nc"
  system("rm -f " + fn)  
  z = addfile(fn,"c")
  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->ensemble_assign = ensemble_assign
  z->nam_patterns = nam_patterns
  z->sam_patterns0 = sam_patterns0
  z->sam_patterns1 = sam_patterns1
  z->nam_timeseries = nam_timeseries
  z->sam_timeseries0 = sam_timeseries0
  z->sam_timeseries1 = sam_timeseries1
  z->nam_trends = nam_pc_trends
  z->sam_trends0 = sam_pc_trends0
  z->sam_trends1 = sam_pc_trends1
  delete(z)

end

