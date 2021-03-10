load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/plot/mder.ncl"

load "$diag_scripts/ipccwg1ar6ch3_modes/functions.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")

  syear_a = 1958
  eyear_a = 2019
  syear_b = 1979
  eyear_b = 1999
  syear_c = 2000
  eyear_c = 2019
  if (isatt(diag_script_info, "syear_nam")) then
    syear_a := diag_script_info@syear_nam
  end if
  if (isatt(diag_script_info, "eyear_nam")) then
    eyear_a := diag_script_info@eyear_nam
  end if
  if (isatt(diag_script_info, "syear_sam0")) then
    syear_b := diag_script_info@syear_sam0
  end if
  if (isatt(diag_script_info, "eyear_sam0")) then
    eyear_b := diag_script_info@eyear_sam0
  end if
  if (isatt(diag_script_info, "syear_sam1")) then
    syear_c := diag_script_info@syear_sam1
  end if
  if (isatt(diag_script_info, "eyear_sam1")) then
    eyear_c := diag_script_info@eyear_sam1
  end if
  
  ; Create work directory
  out_path = config_user_info@work_dir
  system("mkdir -p " + out_path)

  var0 = variable_info[0]@short_name
  info_items = select_metadata_by_name(input_file_info, var0)
  datasetnames = metadata_att_as_array(info_items, "dataset")
  dim_MOD = ListCount(info_items)

  scenario = "hist-GHG"
  if (isatt(diag_script_info, "scenario")) then
    scenario = diag_script_info@scenario
  end if
  
  seas = "djf"
  if (isatt(diag_script_info, "season")) then
    seas = diag_script_info@season
  end if
  seas = str_lower(seas)

;----------------------------------------------------------
  pi=4.*atan(1.0)
  rad=(pi/180.)

;-------------------------------------------------------------
; NAM/SAM for DAMIP simulations
;-------------------------------------------------------------
;
;  lat := nam_patterns_obs&lat
;  lat!0 = "lat"
;  lat&lat = lat
;  w := cos(rad*lat)  ; weights for Taylor diagram
;  w!0 = "lat"
;  w&lat = lat

  nens = dim_MOD
  
  models = metadata_att_as_array(info_items, "dataset")
  emem = metadata_att_as_array(info_items, "ensemble")
  ensemble_assign = new(nens,integer)

  enum = 1
  temp = models(0)
  do gg = 0,dimsizes(models)-1
     if (temp.eq.models(gg)) then   ; does the model name match what's in temp?
        ensemble_assign(gg) = enum ; if so, assign it the same number
     else
        enum = enum+1              ; if not, assign it the next number
        ensemble_assign(gg) = enum
        temp = models(gg)
     end if
  end do
  ensemble_assign@models = str_join(models+"/"+emem,",")

  do gg = 0, nens-1
    arrF := read_data(info_items[gg])
    arrF&time := cd_calendar(arrF&time,-1)
    syr_data = arrF&time(0)/100
    eyr_data = arrF&time(dimsizes(arrF&time)-1)/100
    
;    if (gg.ge.1.and.isatt(arrF,"is_all_missing")) then   ; 
;      continue
;    end if

    do hh = 0,2     ; nam, sam, sam
      if (hh.eq.0) then
        syr = syear_a
        eyr = eyear_a
      end if
      if (hh.ge.1) then
        syr = min((/syear_b, syear_c/))
        eyr = max((/eyear_b, eyear_c/))
      end if
      arr := new((/3, eyr-syr+1, dimsizes(arrF&lat), dimsizes(arrF&lon)/), typeof(arrF))  ; (mon, yr, lat, lon)
      arr!0 = "month"
      arr!1 = "year"
      arr!2 = arrF!1
      arr!3 = arrF!2
      if (seas.eq."djf") then
        smo = 0
        emo = 2
      elseif (seas.eq."jja") then
        smo = 6
        emo = 8
      end if
      arr&month = ispan(smo, emo, 1)
      arr&year = ispan(syr, eyr, 1)
      arr&$arr!2$ = arrF&$arrF!1$
      arr&$arr!3$ = arrF&$arrF!2$
      do mon = smo, emo
        if (mon.lt.1) then
          mo = mon + 12
          syrF = syr - 1
          eyrF = eyr - 1
        else
          mo = mon
          syrF = syr
          eyrF = eyr
        end if
        syrO = syr
        eyrO = eyr
        if (syrF.lt.syr_data) then
          syrO = syr + syr_data - syrF
          syrF = syr_data
        end if
        if (eyrF.gt.eyr_data) then
          eyrO = eyr + eyr_data - eyrF
          eyrF = eyr_data
        end if
        arr({mon}, {syrO:eyrO}, :, :) = (/arrF({syrF*100+mo:eyrF*100+mo:12}, :, :)/)
      end do
      zm := dim_avg_n_Wrap(arr, 3) ; (mon, yr, lat)
      zm = dim_standardize_n(zm, 0, 1)  ; standardize along "year" dimension for each calendar month
      zm_seas := dim_avg_n_Wrap(zm, 0) ; (yr, lat) seasonal mean

      arr_seas := dim_avg_n_Wrap(arr, 0) ; (yr, lat, lon) seasonal mean

      delete([/arr, zm/])

      if (hh.eq.0) then   ; NAM
        finarr_nam_pc := zm_seas({syear_a:eyear_a}, {35}) - zm_seas({syear_a:eyear_a}, {65})
        finarr_nam := regCoef_n(dim_standardize(finarr_nam_pc, 0),arr_seas({syear_a:eyear_a}, :, :),0,0)
        copy_VarMeta(arr_seas(0,:,:),finarr_nam)
      end if
      if (hh.eq.1) then   ; SAM
        finarr_sam_pc0 := zm_seas({syear_b:eyear_b}, {-40}) - zm_seas({syear_b:eyear_b}, {-65})
        finarr_sam0 := regCoef_n(dim_standardize(finarr_sam_pc0, 0),arr_seas({syear_b:eyear_b}, :, :),0,0)
        copy_VarMeta(arr_seas(0,:,:),finarr_sam0)
      end if
      if (hh.eq.2) then   ; SAM
        finarr_sam_pc1 := zm_seas({syear_c:eyear_c}, {-40}) - zm_seas({syear_c:eyear_c}, {-65})
        finarr_sam1 := regCoef_n(dim_standardize(finarr_sam_pc1, 0),arr_seas({syear_c:eyear_c}, :, :),0,0)
        copy_VarMeta(arr_seas(0,:,:),finarr_sam1)
      end if
    end do
    delete(arrF)
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
      nam_timeseries&time = ispan(syear_a,eyear_a,1) 
      nam_timeseries = nam_timeseries@_FillValue
      sam_timeseries0 := new((/nens,dimsizes(finarr_sam_pc0)/),float)
      sam_timeseries0!0 = "E"
      sam_timeseries0&E = sam_patterns0&E
      sam_timeseries0!1 = "time2"
      sam_timeseries0&time2 = ispan(syear_b,eyear_b,1) 
      sam_timeseries0 = sam_timeseries0@_FillValue
      sam_timeseries1 := new((/nens,dimsizes(finarr_sam_pc1)/),float)
      sam_timeseries1!0 = "E"
      sam_timeseries1&E = sam_patterns1&E
      sam_timeseries1!1 = "time3"
      sam_timeseries1&time3 = ispan(syear_c,eyear_c,1) 
      sam_timeseries1 = sam_timeseries1@_FillValue
      
      nam_pc_trends := nam_timeseries(:,0)
      sam_pc_trends0 := nam_timeseries(:,0)
      sam_pc_trends1 := nam_timeseries(:,0)

      ensemble_assign!0 = "E"
      ensemble_assign&E = nam_patterns&E
    end if

;    nam_patterns(gg,:,:)  = (/ linint2_Wrap(finarr_nam&lon,finarr_nam&lat,finarr_nam,True,nam_patterns&lon,nam_patterns&lat,0) /)
    nam_patterns(gg,:,:)  = (/ finarr_nam /)
         
;    sam_patterns0(gg,:,:)  = (/ linint2_Wrap(finarr_sam0&lon,finarr_sam0&lat,finarr_sam0,True,sam_patterns0&lon,sam_patterns0&lat,0) /)
    sam_patterns0(gg,:,:)  = (/ finarr_sam0 /)

;    sam_patterns1(gg,:,:)  = (/ linint2_Wrap(finarr_sam1&lon,finarr_sam1&lat,finarr_sam1,True,sam_patterns1&lon,sam_patterns1&lat,0) /)
    sam_patterns1(gg,:,:)  = (/ finarr_sam1 /)

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

  fn = out_path + "nam_sam.alt_def.damip_"+scenario+"."+seas+".nc"
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

  log_provenance(fn, "n/a", \
                 "Patterns, time series & trends of NAM/SAM defined with the latitudinal SLP difference in DAMIP", \
                 (/"anomaly", "var", "trend"/), (/"nhext", "shext"/), "other", "kosaka_yu", "", \
                 metadata_att_as_array(info_items, "filename"))
  
end

