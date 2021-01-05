; #############################################################################
; Compute AMV metrics
; Authors: Lisa Bock (DLR, Germany), Soufiane Karmouche (University of Bremen,
;          Germany) and Adam Phillips (NCAR, U.S.)
; #############################################################################
;
; Description
; Compute AMV metrics for observations, CMIP6-historical, CMIP6-picontrol,
; CMIP5-historical and CMIP5-piControl simulations.
; This script creates 1 file each for observations, CMIP6-historical,
; CMIP6-piControl, CMIP5-historical and CMIP5-piControl.
;
; Required diag_script_info attributes (diagnostic specific)
;
; Optional diag_script_info attributes (diagnostic specific)
;
; Caveats
;
; Modification history
;   20200703-bock_lisa: adapt Adam's scipt to the ESMValTool
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
  atts@exp = "piControl"
  piControl_cmip5 = select_metadata_by_atts(cmip5, atts)
  piControl_cmip5_datasets = metadata_att_as_array(piControl_cmip5, "dataset")
  dim_cmip5_piControl = dimsizes(piControl_cmip5_datasets)

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
  atts@exp = "piControl"
  piControl_cmip6 = select_metadata_by_atts(cmip6, atts)
  piControl_cmip6_datasets = metadata_att_as_array(piControl_cmip6, "dataset")
  dim_cmip6_piControl = dimsizes(piControl_cmip6_datasets)

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

  ; setting constants
  ;fca = 1./ 157.       ; 13yr low pass
  fca = 1./ 121.       ; 10 year low pass
  fcb = 0.
  ihp = 0
  nsigma = 1
  nwgt = min((/217, ntime - 1/))
  pi=4.*atan(1.0)
  rad=(pi/ 180.)

  ; create low pass filter
  wgt = new(nwgt, float)
  wgt = filwgts_lancos(nwgt, ihp, fca, fcb, nsigma)

  ;--------------------------------------------------------
  ; process observational data
  ;--------------------------------------------------------

  do iobs = 0, dim_obs - 1

    log_info(obs_datasets(iobs))

    ; Read observational data
    arr = read_data(obs[iobs])
    arr = where(arr.le.-1.8, -1.8, arr)    ; set all values below -1.8 to -1.8

    arr = lonFlip(arr)

    arr = rmMonAnnCycTLL(arr)
    coswgt  := cos(rad*arr&lat)
    coswgt!0 = "lat"
    coswgt&lat= arr&lat
    natl_aa :=   wgt_areaave(arr(:, {0:60}, {-80:0}), coswgt({0.:60.}), 1.0, 0)
    global_aa := wgt_areaave(arr(:, {-60:60}, :), coswgt({-60.:60.}), 1.0, 0)

    ; timeseries plot
    finarr := new((/2, dimsizes(natl_aa)/), typeof(natl_aa), -999.)
    finarr!1 = "time"
    finarr&time = arr&time
    finarr(0,:) = (/natl_aa - global_aa/)
    finarr(1,:) = (/wgt_runave_n_Wrap(finarr(0, :), wgt, 1, 0)/)

    arr = arr - conform(arr, global_aa, 0)

    finreg := (/regCoef(dim_standardize(finarr(0, :), 0),\
                arr(lat|:, lon|:, time|:))/)
    copy_VarMeta(arr(0, :, :), finreg)
    do hh = 0, 2
       finreg = (/smth9(finreg, 0.5, 0.25, True)/)
    end do
    finregLP := (/regCoef_n(dim_standardize(finarr(1, :), 0), \
                   wgt_runave_n_Wrap(arr, wgt, 1, 0), 0, 0)/)
    copy_VarMeta(arr(0, :, :), finregLP)

    if (iobs.eq.0) then
       fn = out_path + "amv.obs."+syearA+"-"+eyearA+".nc"
       print("fn = " + fn)
       system("rm -f " + fn)
       z = addfile(fn,"c")
       z@source = systemfunc("pwd")+"/"+get_script_name()
       z->amv_pattern_obs0 = finregLP

       amv_pattern_obs = arr(:dimsizes(obs_datasets)-1,:,:)
       amv_pattern_obs = amv_pattern_obs@_FillValue
       delete(amv_pattern_obs&time)
       amv_pattern_obs!0 = "E"
       amv_pattern_obs&E = ispan(1,dimsizes(obs_datasets),1)
       amv_pattern_obs(iobs,:,:) = (/ finregLP /)

       amv_timeseries = new((/dimsizes(obs_datasets),(eyearA-syearA+1)*12/),typeof(finarr))
       amv_timeseries!0 = "E"
       amv_timeseries&E = ispan(1,dimsizes(obs_datasets),1)
       amv_timeseries!1 = "time"
       amv_timeseries&time = arr&time
       amv_timeseries_raw = amv_timeseries
       tay_stat = new((/dimsizes(obs_datasets),3/),double)
       tay_stat!0 = "E"
       tay_stat&E = ispan(1,dimsizes(obs_datasets),1)
       tay_stat!1 = "stat"
       tay_stat&stat = ispan(0,2,1)
       lat := finregLP&lat
       lat!0 = "lat"
       lat&lat = lat
       w = cos(rad*lat({0:60}))  ; weights for Taylor diagram
    else
       amv_pattern_obs(iobs,:,:) = (/ linint2_Wrap(finregLP&lon,finregLP&lat, \
                                      finregLP,True,amv_pattern_obs&lon, \
                                      amv_pattern_obs&lat,0) /)
       tay_stat(iobs,:) = (/ taylor_stats(amv_pattern_obs(iobs,{0:60},{-80:0}), \
                             amv_pattern_obs(0,{0:60},{-80:0}), w, 0) /)
    end if
    amv_timeseries_raw(iobs,:) = (/ finarr(0,:) /)
    amv_timeseries(iobs,:) = (/ finarr(1,:) /)
    delete(arr)
  end do

  z->amv_pattern_obs = amv_pattern_obs
  z->amv_timeseries_raw = amv_timeseries_raw
  z->amv_timeseries = amv_timeseries
  z->tay_stats = tay_stat
  delete([/z/])

  z = addfile(fn,"r")
  amv_obs = z->amv_pattern_obs0
  amv_tseries_lp_obs = z->amv_timeseries
  amv_tseries_obs = z->amv_timeseries_raw
  tay_stats_obs = z->tay_stats


  ;--------------------------------------------------------
  ; process CMIP6 historical data
  ;--------------------------------------------------------

  lat := amv_obs&lat
  lat!0 = "lat"
  lat&lat = lat
  w = cos(rad*lat({0:60}))  ; weights for Taylor diagram

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

  do gg = 0, nens - 1
    arr := read_data(hist_cmip6[gg])

    arr = where(arr.le.-1.8,-1.8,arr)    ; set all values below -1.8 to -1.8
    ; mask out land (this is redundant for data that is already masked)
    d = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
    lsm := landsea_mask(d->LSMASK,arr&lat,arr&lon)
    arr = mask(arr,conform(arr,lsm,(/1,2/)).ge.1,False)

    arr = lonFlip(arr)

    arr = rmMonAnnCycTLL(arr)
    coswgt  := cos(rad*arr&lat)
    coswgt!0 = "lat"
    coswgt&lat= arr&lat
    natl_aa :=   wgt_areaave(arr(:,{0:60},{-80:0}),coswgt({0.:60.}),1.0,0)
    global_aa := wgt_areaave(arr(:,{-60:60},:),coswgt({-60.:60.}),1.0,0)
    finarr := new((/2,dimsizes(natl_aa)/),typeof(natl_aa),-999.)    ; timeseries plot
    finarr!1 = "time"
    finarr&time = arr&time
    finarr(0,:) = (/ natl_aa - global_aa /)
    finarr(1,:) = (/ wgt_runave_n_Wrap(finarr(0,:),wgt,1,0) /)

    arr = arr - conform(arr, global_aa, 0)

    finreg := (/ regCoef(dim_standardize(finarr(0,:),0),arr(lat|:,lon|:,time|:))  /)
    copy_VarMeta(arr(0,:,:),finreg)
    do hh = 0,2
       finreg = (/ smth9(finreg,0.5,0.25,True) /)
    end do
    finregLP := (/regCoef_n(dim_standardize(finarr(1, :) ,0), \
                  wgt_runave_n_Wrap(arr, wgt, 1, 0), 0, 0)/)
    copy_VarMeta(arr(0,:,:),finregLP)

    if (gg.eq.0) then
       amv_pattern = new((/nens,dimsizes(amv_obs&lat), \
                           dimsizes(amv_obs&lon)/),typeof(amv_obs))
       amv_pattern!0 = "nens"
       amv_pattern&nens = ispan(0,nens-1,1)
       amv_pattern!1 = "lat"
       amv_pattern&lat = amv_obs&lat
       amv_pattern!2 = "lon"
       amv_pattern&lon = amv_obs&lon

       amv_timeseries := new((/nens,(eyearA-syearA+1)*12/),typeof(finarr))
       amv_timeseries!0 = "nens"
       amv_timeseries&nens = amv_pattern&nens
       amv_timeseries!1 = "time"
       amv_timeseries&time = arr&time
       amv_timeseries_raw := amv_timeseries
       tay_stat := new((/nens,3/),double)
       tay_stat!0 = "nens"
       tay_stat&nens = amv_pattern&nens
       tay_stat!1 = "stat"
       tay_stat&stat = ispan(0,2,1)
    end if
    amv_pattern(gg,:,:) = (/linint2_Wrap(finregLP&lon, finregLP&lat, finregLP, \
                            True, amv_obs&lon, amv_obs&lat, 0)/)
    tay_stat(gg,:) = (/taylor_stats(amv_pattern(gg, {0:60}, {-80:0}), \
                       amv_obs({0:60}, {-80:0}), w, 0) /)
    amv_timeseries_raw(gg,:) = (/ finarr(0,:) /)
    amv_timeseries(gg,:) = (/ finarr(1,:) /)
    print("Done with "+gg+" our of "+nens+", "+model(gg))
  end do

  fn = out_path + "amv.hist."+syearA+"-"+eyearA+".cmip6.nc"
  system("rm -f " + fn)
  z = addfile(fn,"c")
  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->amv_pattern = amv_pattern
  z->amv_timeseries_raw = amv_timeseries_raw
  z->amv_timeseries = amv_timeseries
  z->tay_stats = tay_stat
  z->ensemble_assign = ensemble_assign
  delete([/z/])

  z = addfile(fn,"r")
  amv_hist = z->amv_pattern
  amv_tseries_lp_hist = z->amv_timeseries
  amv_tseries_hist = z->amv_timeseries_raw
  tay_stats_hist = z->tay_stats
  ensemble_assign = z->ensemble_assign


  ;--------------------------------------------------------
  ; process CMIP6 piControl data
  ;--------------------------------------------------------

  do gg = 0, dim_cmip6_piControl - 1
     arr := read_data(piControl_cmip6[gg])

     arr = where(arr.le.-1.8,-1.8,arr)    ; set all values below -1.8 to -1.8
     ; mask out land (this is redundant for data that is already masked)
     d = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
     lsm := landsea_mask(d->LSMASK,arr&lat,arr&lon)
     arr = mask(arr,conform(arr,lsm,(/1,2/)).ge.1,False)

     arr = lonFlip(arr)

     arr = rmMonAnnCycTLL(arr)
     coswgt  := cos(rad*arr&lat)
     coswgt!0 = "lat"
     coswgt&lat= arr&lat
     natl_aa :=   wgt_areaave(arr(:,{0:60},{-80:0}),coswgt({0.:60.}),1.0,0)
     global_aa := wgt_areaave(arr(:,{-60:60},:),coswgt({-60.:60.}),1.0,0)
     finarr := new((/2,dimsizes(natl_aa)/),typeof(natl_aa),-999.)    ; timeseries plot
     finarr!1 = "time"
     finarr&time = arr&time
     finarr(0,:) = (/ natl_aa - global_aa /)
     finarr(1,:) = (/ wgt_runave_n_Wrap(finarr(0,:),wgt,1,0) /)

     arr = arr - conform(arr, global_aa, 0)

     finreg := (/regCoef(dim_standardize(finarr(0, :), 0), \
                         arr(lat|:, lon|:, time|:))/)
     copy_VarMeta(arr(0,:,:),finreg)
     do hh = 0,2
        finreg = (/ smth9(finreg,0.5,0.25,True) /)
     end do
     finregLP := (/regCoef_n(dim_standardize(finarr(1 ,:), 0), \
                   wgt_runave_n_Wrap(arr, wgt, 1, 0), 0, 0)/)
     copy_VarMeta(arr(0,:,:),finregLP)

     if (gg.eq.0) then
        amv_pattern := new((/dim_cmip6_piControl, dimsizes(amv_obs&lat), \
                            dimsizes(amv_obs&lon)/), typeof(amv_obs))
        amv_pattern!0 = "nens"
        amv_pattern&nens = ispan(0,dim_cmip6_piControl-1,1)
        amv_pattern!1 = "lat"
        amv_pattern&lat = amv_obs&lat
        amv_pattern!2 = "lon"
        amv_pattern&lon = amv_obs&lon

        amv_timeseries := new((/dim_cmip6_piControl, \
                             (eyearA - syearA + 1) * 12/), typeof(finarr))
        amv_timeseries!0 = "nens"
        amv_timeseries&nens = amv_pattern&nens
        amv_timeseries!1 = "time"
        amv_timeseries&time = arr&time
        amv_timeseries_raw := amv_timeseries
        tay_stat := new((/dim_cmip6_piControl,3/),double)
        tay_stat!0 = "nens"
        tay_stat&nens = amv_pattern&nens
        tay_stat!1 = "stat"
        tay_stat&stat = ispan(0,2,1)
     end if
     amv_pattern(gg, :, :) = (/linint2_Wrap(finregLP&lon, finregLP&lat, \
                               finregLP, True, amv_obs&lon, amv_obs&lat, 0)/)
     tay_stat(gg, :) = (/taylor_stats(amv_pattern(gg, {0:60}, {-80:0}), \
                         amv_obs({0:60}, {-80:0}), w, 0)/)
     amv_timeseries_raw(gg,:) = (/ finarr(0,:) /)
     amv_timeseries(gg,:) = (/ finarr(1,:) /)
     print("Done with "+gg+" out of "+(dim_cmip6_piControl-1))
  end do

  fn = out_path + "amv.piControl.115yr.cmip6.nc"
  system("rm -f " + fn)
  z = addfile(fn,"c")
  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->amv_pattern = amv_pattern
  z->amv_timeseries_raw = amv_timeseries_raw
  z->amv_timeseries = amv_timeseries
  z->tay_stats = tay_stat
  delete([/z/])

  z = addfile(fn,"r")
  amv_tseries_piControl = z->amv_timeseries_raw
  amv_tseries_lp_piControl = z->amv_timeseries


  ;--------------------------------------------------------
  ; process CMIP5 historical data
  ;--------------------------------------------------------

  lat := amv_obs&lat
  lat!0 = "lat"
  lat&lat = lat
  w = cos(rad*lat({0:60}))  ; weights for Taylor diagram


  print("Number of CMIP5 historical simulations found: " + dim_cmip5_hist)
  model := hist_cmip5_datasets
  emem  := hist_cmip5_ensembles
  ensemble_assign_C5 := new(dim_cmip5_hist, integer)

  enum = 1
  temp = model(0)
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

  do gg = 0, dim_cmip5_hist - 1
    arr := read_data(hist_cmip5[gg])

    arr = where(arr.le.-1.8,-1.8,arr)    ; set all values below -1.8 to -1.8
    ; mask out land (this is redundant for data that is already masked)
    d = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
    lsm := landsea_mask(d->LSMASK,arr&lat,arr&lon)
    arr = mask(arr,conform(arr,lsm,(/1,2/)).ge.1,False)

    arr = lonFlip(arr)

    arr = rmMonAnnCycTLL(arr)
    coswgt  := cos(rad*arr&lat)
    coswgt!0 = "lat"
    coswgt&lat= arr&lat
    natl_aa :=   wgt_areaave(arr(:,{0:60},{-80:0}),coswgt({0.:60.}),1.0,0)
    global_aa := wgt_areaave(arr(:,{-60:60},:),coswgt({-60.:60.}),1.0,0)
    finarr := new((/2,dimsizes(natl_aa)/),typeof(natl_aa),-999.)    ; timeseries plot
    finarr!1 = "time"
    finarr&time = arr&time
    finarr(0,:) = (/ natl_aa - global_aa /)
    finarr(1,:) = (/ wgt_runave_n_Wrap(finarr(0,:),wgt,1,0) /)

    arr = arr - conform(arr, global_aa, 0)

    finreg := (/ regCoef(dim_standardize(finarr(0,:),0),arr(lat|:,lon|:,time|:))  /)
    copy_VarMeta(arr(0,:,:),finreg)
    do hh = 0,2
       finreg = (/ smth9(finreg,0.5,0.25,True) /)
    end do
    finregLP := (/regCoef_n(dim_standardize(finarr(1, :) ,0), \
                  wgt_runave_n_Wrap(arr, wgt, 1, 0), 0, 0)/)
    copy_VarMeta(arr(0,:,:),finregLP)

    if (gg.eq.0) then
       amv_pattern := new((/dim_cmip5_hist,dimsizes(amv_obs&lat), \
                           dimsizes(amv_obs&lon)/),typeof(amv_obs))
       amv_pattern!0 = "dim_cmip5_hist"
       amv_pattern&dim_cmip5_hist = ispan(0,dim_cmip5_hist-1,1)
       amv_pattern!1 = "lat"
       amv_pattern&lat = amv_obs&lat
       amv_pattern!2 = "lon"
       amv_pattern&lon = amv_obs&lon

       amv_timeseries := new((/dim_cmip5_hist,(eyearA-syearA+1)*12/),typeof(finarr))
       amv_timeseries!0 = "dim_cmip5_hist"
       amv_timeseries&dim_cmip5_hist = amv_pattern&dim_cmip5_hist
       amv_timeseries!1 = "time"
       amv_timeseries&time = arr&time
       amv_timeseries_raw := amv_timeseries
       tay_stat := new((/dim_cmip5_hist,3/),double)
       tay_stat!0 = "dim_cmip5_hist"
       tay_stat&dim_cmip5_hist = amv_pattern&dim_cmip5_hist
       tay_stat!1 = "stat"
       tay_stat&stat = ispan(0,2,1)
    end if
    amv_pattern(gg,:,:) = (/linint2_Wrap(finregLP&lon, finregLP&lat, finregLP, \
                            True, amv_obs&lon, amv_obs&lat, 0)/)
    tay_stat(gg,:) = (/taylor_stats(amv_pattern(gg, {0:60}, {-80:0}), \
                       amv_obs({0:60}, {-80:0}), w, 0) /)
    amv_timeseries_raw(gg,:) = (/ finarr(0,:) /)
    amv_timeseries(gg,:) = (/ finarr(1,:) /)
    print("Done with "+gg+" our of "+dim_cmip5_hist+", "+model(gg))
  end do

  fn = out_path + "amv.hist."+syearA+"-"+eyearA+".cmip5.nc"
  system("rm -f " + fn)
  z = addfile(fn,"c")
  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->amv_pattern = amv_pattern
  z->amv_timeseries_raw = amv_timeseries_raw
  z->amv_timeseries = amv_timeseries
  z->tay_stats = tay_stat
  z->ensemble_assign = ensemble_assign_C5
  delete([/z/])

  z = addfile(fn,"r")
  amv_hist_C5 = z->amv_pattern
  amv_tseries_lp_hist_C5 = z->amv_timeseries
  amv_tseries_hist_C5 = z->amv_timeseries_raw
  ea_c5 = z->ensemble_assign

  taylor_amv_ratio_C5  = z->tay_stats(:,1:1)
  taylor_amv_patcor_C5 = z->tay_stats(:,0:0)
  ; convert to Z-space
  taylor_amv_patcor_C5T = 0.5 * (log((1 + taylor_amv_patcor_C5) / \
                                 (1 - taylor_amv_patcor_C5)))

  ;--------------------------------------------------------
  ; process CMIP5 piControl data
  ;--------------------------------------------------------

  do gg = 0, dim_cmip5_piControl - 1
     arr := read_data(piControl_cmip5[gg])

     arr = where(arr.le.-1.8,-1.8,arr)    ; set all values below -1.8 to -1.8
     ; mask out land (this is redundant for data that is already masked)
     d = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
     lsm := landsea_mask(d->LSMASK,arr&lat,arr&lon)
     arr = mask(arr,conform(arr,lsm,(/1,2/)).ge.1,False)

     arr = lonFlip(arr)

     arr = rmMonAnnCycTLL(arr)
     coswgt  := cos(rad*arr&lat)
     coswgt!0 = "lat"
     coswgt&lat= arr&lat
     natl_aa :=   wgt_areaave(arr(:,{0:60},{-80:0}),coswgt({0.:60.}),1.0,0)
     global_aa := wgt_areaave(arr(:,{-60:60},:),coswgt({-60.:60.}),1.0,0)
     finarr := new((/2,dimsizes(natl_aa)/),typeof(natl_aa),-999.)    ; timeseries plot
     finarr!1 = "time"
     finarr&time = arr&time
     finarr(0,:) = (/ natl_aa - global_aa /)
     finarr(1,:) = (/ wgt_runave_n_Wrap(finarr(0,:),wgt,1,0) /)

     arr = arr - conform(arr, global_aa, 0)

     finreg := (/regCoef(dim_standardize(finarr(0, :), 0), \
                         arr(lat|:, lon|:, time|:))/)
     copy_VarMeta(arr(0,:,:),finreg)
     do hh = 0,2
        finreg = (/ smth9(finreg,0.5,0.25,True) /)
     end do
     finregLP := (/regCoef_n(dim_standardize(finarr(1 ,:), 0), \
                   wgt_runave_n_Wrap(arr, wgt, 1, 0), 0, 0)/)
     copy_VarMeta(arr(0,:,:),finregLP)

     if (gg.eq.0) then
        amv_pattern := new((/dim_cmip5_piControl, dimsizes(amv_obs&lat), \
                            dimsizes(amv_obs&lon)/), typeof(amv_obs))
        amv_pattern!0 = "nens"
        amv_pattern&nens = ispan(0,dim_cmip5_piControl-1,1)
        amv_pattern!1 = "lat"
        amv_pattern&lat = amv_obs&lat
        amv_pattern!2 = "lon"
        amv_pattern&lon = amv_obs&lon

        amv_timeseries := new((/dim_cmip5_piControl, \
                             (eyearA - syearA + 1) * 12/), typeof(finarr))
        amv_timeseries!0 = "nens"
        amv_timeseries&nens = amv_pattern&nens
        amv_timeseries!1 = "time"
        amv_timeseries&time = arr&time
        amv_timeseries_raw := amv_timeseries
        tay_stat := new((/dim_cmip5_piControl,3/),double)
        tay_stat!0 = "nens"
        tay_stat&nens = amv_pattern&nens
        tay_stat!1 = "stat"
        tay_stat&stat = ispan(0,2,1)
     end if
     amv_pattern(gg, :, :) = (/linint2_Wrap(finregLP&lon, finregLP&lat, \
                               finregLP, True, amv_obs&lon, amv_obs&lat, 0)/)
     tay_stat(gg, :) = (/taylor_stats(amv_pattern(gg, {0:60}, {-80:0}), \
                         amv_obs({0:60}, {-80:0}), w, 0)/)
     amv_timeseries_raw(gg,:) = (/ finarr(0,:) /)
     amv_timeseries(gg,:) = (/ finarr(1,:) /)
     print("Done with "+gg+" out of "+(dim_cmip5_piControl-1))
  end do

  fn = out_path + "amv.piControl.115yr.cmip5.nc"
  system("rm -f " + fn)
  z = addfile(fn,"c")
  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->amv_pattern = amv_pattern
  z->amv_timeseries_raw = amv_timeseries_raw
  z->amv_timeseries = amv_timeseries
  z->tay_stats = tay_stat
  delete([/z/])

  z = addfile(fn,"r")
  amv_tseries_piControl_C5 = z->amv_timeseries_raw
  amv_tseries_lp_piControl_C5 = z->amv_timeseries


  ;--------------------------------------------------------
  ; other processing steps?
  ;--------------------------------------------------------

  amv_hist_em5 = amv_hist_C5(:max(ea_c5)-1,:,:)
  amv_hist_em5 = amv_hist_em5@_FillValue
  amv_tseries_lp_hist_em5 = amv_tseries_lp_hist_C5(:max(ea_c5)-1,:)
  amv_tseries_lp_hist_em5 = amv_tseries_lp_hist_em5@_FillValue
  tay_ratio_C5 = taylor_amv_ratio_C5(:max(ea_c5)-1,0)
  tay_ratio_C5 = tay_ratio_C5@_FillValue
  tay_patcor_C5 = taylor_amv_patcor_C5T(:max(ea_c5)-1,0)
  tay_patcor_C5 = tay_patcor_C5@_FillValue
  do gg = 1,max(ea_c5)         ; calculate CMIP5 ensemble means
    wind := ind(ea_c5.eq.gg)
    if (dimsizes(wind).eq.1) then
      amv_hist_em5(gg-1,:,:) = (/ amv_hist_C5(wind,:,:) /)
      amv_tseries_lp_hist_em5(gg-1,:) = (/ amv_tseries_lp_hist_C5(wind,:) /)
      tay_ratio_C5(gg-1) = (/ taylor_amv_ratio_C5(wind,0) /)
      tay_patcor_C5(gg-1) = (/ taylor_amv_patcor_C5T(wind,0) /)
    else
      amv_hist_em5(gg-1,:,:) = (/ dim_avg_n(amv_hist_C5(wind,:,:),0) /)
      amv_tseries_lp_hist_em5(gg-1,:) = (/ dim_avg_n(amv_tseries_lp_hist_C5(wind,:),0) /)
      tay_ratio_C5(gg-1) = (/ dim_avg_n(taylor_amv_ratio_C5(wind,0),0) /)
      tay_patcor_C5(gg-1) = (/ dim_avg_n(taylor_amv_patcor_C5T(wind,0),0) /)
    end if
  end do
  mme_amv_C5 = dim_avg_n_Wrap(amv_hist_em5,0)
  mme_amv_tseries_lp_hist_C5 = dim_avg_n_Wrap(amv_tseries_lp_hist_em5,0)
  delete([/amv_hist_em5,amv_tseries_lp_hist_em5/])

  tay_stats_hist_cmip5EM = new((/1,3/),double)
  ; tay_stats_hist_cmip5EM = (/ taylor_stats(mme_amv_C5({0:60},{-80:0}), \
  ;                              amv_obs({0:60},{-80:0}), w, 0) /)
  ; delete([/mme_amv_C5/])
  ; convert back from z-spac
  tay_stats_hist_cmip5EM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_C5))-1)/ \
                                 ((2.71828^(2*dim_avg(tay_patcor_C5))+1))) /)
  tay_stats_hist_cmip5EM(0,1) = (/ dim_avg(tay_ratio_C5) /)

  ;--------------------------------------------------------

  tay_patcor_C6 = tay_stats_hist(:,0)
  tay_ratio_C6  = tay_stats_hist(:,1)
  tay_patcor_C6 = 0.5*(log( (1+tay_patcor_C6) / (1-tay_patcor_C6)))    ; convert to Z-space

  amv_hist_em = amv_hist(:max(ensemble_assign)-1,:,:)
  amv_hist_em = amv_hist_em@_FillValue
  amv_tseries_lp_hist_em = amv_tseries_lp_hist(:max(ensemble_assign)-1,:)
  amv_tseries_lp_hist_em = amv_tseries_lp_hist_em@_FillValue
  tay_ratio = tay_ratio_C6(:max(ensemble_assign)-1)
  tay_ratio = tay_ratio@_FillValue
  tay_patcor = tay_patcor_C6(:max(ensemble_assign)-1)
  tay_patcor = tay_patcor@_FillValue
  do gg = 1,max(ensemble_assign)         ; calculate ensemble means
    wind := ind(ensemble_assign.eq.gg)
    if (dimsizes(wind).eq.1) then
      amv_hist_em(gg-1,:,:) = (/ amv_hist(wind,:,:) /)
      amv_tseries_lp_hist_em(gg-1,:) = (/ amv_tseries_lp_hist(wind,:) /)
      tay_ratio(gg-1) = (/ tay_ratio_C6(wind) /)
      tay_patcor(gg-1) = (/ tay_patcor_C6(wind) /)
    else
      amv_hist_em(gg-1,:,:) = (/ dim_avg_n(amv_hist(wind,:,:),0) /)
      amv_tseries_lp_hist_em(gg-1,:) = (/ dim_avg_n(amv_tseries_lp_hist(wind,:),0) /)
      tay_ratio(gg-1) = (/ dim_avg(tay_ratio_C6(wind)) /)
      tay_patcor(gg-1) = (/ dim_avg(tay_patcor_C6(wind)) /)
    end if
  end do
  mme_amv = dim_avg_n_Wrap(amv_hist_em,0)
  mme_amv_tseries_lp_hist = dim_avg_n_Wrap(amv_tseries_lp_hist_em,0)

  tay_stats_hist_cmip6EM = new((/1,3/),double)
  ; tay_stats_hist_cmip6EM = (/ taylor_stats(mme_amv({0:60},{-80:0}), \
  ;                              amv_obs({0:60},{-80:0}), w, 0) /)
  ; convert back from z-space
  tay_stats_hist_cmip6EM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor))-1)/ \
                                   ((2.71828^(2*dim_avg(tay_patcor))+1))) /)
  tay_stats_hist_cmip6EM(0,1) = (/ dim_avg(tay_ratio) /)


  ;--------------------------------------------------------
  ; Observations box plot statistics
  ;--------------------------------------------------------

  autocor_amv_tseries_lp_obs = esacr(amv_tseries_lp_obs,10*12)
  lag1_amv_tseries_lp_obs = autocor_amv_tseries_lp_obs(:,12)
  lag10_amv_tseries_lp_obs = autocor_amv_tseries_lp_obs(:,12*10-1)
  delete(autocor_amv_tseries_lp_obs)

  autocor_amv_tseries_obs = esacr(amv_tseries_obs,10*12)
  lag1_amv_tseries_obs = autocor_amv_tseries_obs(:,12)
  lag10_amv_tseries_obs = autocor_amv_tseries_obs(:,12*10-1)
  delete(autocor_amv_tseries_obs)

  ;--------------------------------------------------------
  ; CMIP6 historical autocorrelation box plot statistics
  ;--------------------------------------------------------

  temp_ac := esacr(amv_tseries_lp_hist,10*12)
  lag10_amv_tseries_lp_hist_B = boxplot_stat(temp_ac(:,12*10-1))
  temp_ac := esacr(amv_tseries_hist,10*12)
  lag1_amv_tseries_hist_B = boxplot_stat(temp_ac(:,12))

  ;--------------------------------------------------------
  ; CMIP6 piControl autocorrelation box plot statistics
  ;--------------------------------------------------------

  temp_ac := esacr(amv_tseries_lp_piControl,10*12)
  lag10_amv_tseries_lp_piControl_B = boxplot_stat(temp_ac(:,12*10-1))
  temp_ac := esacr(amv_tseries_piControl,10*12)
  lag1_amv_tseries_piControl_B = boxplot_stat(temp_ac(:,12))

  ;--------------------------------------------------------
  ; CMIP6 piControl / historical and observational standard
  ; deviation box plot statistics
  ;--------------------------------------------------------

  stddev_amv_tseries_lp_obs         = dim_stddev(amv_tseries_lp_obs)
  stddev_amv_tseries_lp_hist_B      = boxplot_stat(dim_stddev(amv_tseries_lp_hist))
  stddev_amv_tseries_lp_piControl_B = boxplot_stat(dim_stddev(amv_tseries_lp_piControl))
  stddev_amv_tseries_obs            = dim_stddev(amv_tseries_obs)
  stddev_amv_tseries_hist_B         = boxplot_stat(dim_stddev(amv_tseries_hist))
  stddev_amv_tseries_piControl_B    = boxplot_stat(dim_stddev(amv_tseries_piControl))

  ;--------------------------------------------------------
  ; CMIP5 historical autocorrelation box plot statistics
  ;--------------------------------------------------------

  temp_ac := esacr(amv_tseries_lp_hist_C5,10*12)
  lag10_amv_tseries_lp_hist_B_C5 = boxplot_stat(temp_ac(:,12*10-1))
  temp_ac := esacr(amv_tseries_hist_C5,10*12)
  lag1_amv_tseries_hist_B_C5 = boxplot_stat(temp_ac(:,12))

  ;--------------------------------------------------------
  ; CMIP5 piControl autocorrelation box plot statistics
  ;--------------------------------------------------------

  temp_ac := esacr(amv_tseries_lp_piControl_C5,10*12)
  lag10_amv_tseries_lp_piControl_B_C5 = boxplot_stat(temp_ac(:,12*10-1))
  temp_ac := esacr(amv_tseries_piControl_C5,10*12)
  lag1_amv_tseries_piControl_B_C5 = boxplot_stat(temp_ac(:,12))

  ;--------------------------------------------------------
  ; CMIP5 piControl / historical and observational standard
  ; deviation box plot statistics
  ;--------------------------------------------------------

  stddev_amv_tseries_lp_hist_B_C5      =  boxplot_stat(dim_stddev(amv_tseries_lp_hist_C5))
  stddev_amv_tseries_lp_piControl_B_C5 =  boxplot_stat(dim_stddev(amv_tseries_lp_piControl_C5))
  stddev_amv_tseries_hist_B_C5      =  boxplot_stat(dim_stddev(amv_tseries_hist_C5))
  stddev_amv_tseries_piControl_B_C5 =  boxplot_stat(dim_stddev(amv_tseries_piControl_C5))


  ;--------------------------------------------------------
  ; Preparation for box plots
  ;--------------------------------------------------------

  yval_ac = new((/8,5/),float)
  yval_ac(0,:) = (/ lag1_amv_tseries_piControl_B_C5 /)
  yval_ac(1,:) = (/ lag1_amv_tseries_piControl_B /)
  yval_ac(2,:) = (/ lag1_amv_tseries_hist_B_C5 /)
  yval_ac(3,:) = (/ lag1_amv_tseries_hist_B /)

  yval_ac(4,:) = (/ lag10_amv_tseries_lp_piControl_B_C5 /)
  yval_ac(5,:) = (/ lag10_amv_tseries_lp_piControl_B /)
  yval_ac(6,:) = (/ lag10_amv_tseries_lp_hist_B_C5 /)
  yval_ac(7,:) = (/ lag10_amv_tseries_lp_hist_B /)

  yval_sd = new((/8,5/),float)
  yval_sd(0,:) = (/ stddev_amv_tseries_piControl_B_C5 /)
  yval_sd(1,:) = (/ stddev_amv_tseries_piControl_B /)
  yval_sd(2,:) = (/ stddev_amv_tseries_hist_B_C5 /)
  yval_sd(3,:) = (/ stddev_amv_tseries_hist_B /)

  yval_sd(4,:) = (/ stddev_amv_tseries_lp_piControl_B_C5 /)
  yval_sd(5,:) = (/ stddev_amv_tseries_lp_piControl_B /)
  yval_sd(6,:) = (/ stddev_amv_tseries_lp_hist_B_C5 /)
  yval_sd(7,:) = (/ stddev_amv_tseries_lp_hist_B /)


  ;--------------------------------------------------------
  ; Preparing timeseries plot
  ;--------------------------------------------------------

  hist_range = amv_tseries_lp_hist(:1,:)
  hist_range = hist_range@_FillValue
  do gg = 60,dimsizes(amv_tseries_lp_hist&time)-61
    temp = dim_stddev_n(amv_tseries_lp_hist(:,gg),0)
    hist_range(0,gg) = (/ mme_amv_tseries_lp_hist(gg)+(temp*2.) /)
    hist_range(1,gg) = (/ mme_amv_tseries_lp_hist(gg)-(temp*2.) /)
    delete(temp)
  end do

  hist_range_C5 = amv_tseries_lp_hist_C5(:1,:)
  hist_range_C5 = hist_range_C5@_FillValue
  do gg = 60,dimsizes(amv_tseries_lp_hist_C5&time)-61
    temp = dim_stddev_n(amv_tseries_lp_hist_C5(:,gg),0)
    hist_range_C5(0,gg) = (/ mme_amv_tseries_lp_hist_C5(gg)+(temp*2.) /)
    hist_range_C5(1,gg) = (/ mme_amv_tseries_lp_hist_C5(gg)-(temp*2.) /)
    delete(temp)
  end do

  timeseries = new((/6,dimsizes(amv_tseries_hist&time)/),float)
  timeseries(0,:) = (/ amv_tseries_lp_obs(0,:) /)
  timeseries(1,:dimsizes(mme_amv_tseries_lp_hist_C5)-61) = \
    (/ mme_amv_tseries_lp_hist_C5(:dimsizes(mme_amv_tseries_lp_hist_C5)-61) /)
  timeseries(2,:) = (/ mme_amv_tseries_lp_hist /)

  timeseries(:,:59) = timeseries@_FillValue
  timeseries(:,dimsizes(amv_tseries_hist&time)-60:) = timeseries@_FillValue

;  amv_tseries_lp_piControl_mme = dim_avg_n_Wrap(amv_tseries_lp_piControl,0)

  timeseries(3,:) = 0.   ; zero line
;  timeseries(4,:) = dim_avg(amv_tseries_lp_piControl_mme)+ \
;                      (2*dim_stddev(amv_tseries_lp_piControl_mme))
;  timeseries(5,:) = dim_avg(amv_tseries_lp_piControl_mme)- \
;                      (2*dim_stddev(amv_tseries_lp_piControl_mme))


  ;--------------------------------------------------------
  ;Plotting
  ;--------------------------------------------------------

  outpath = config_user_info@plot_dir
  ofile = outpath + "amv"

  wks = gsn_open_wks(file_type, ofile)

  res = True
  res@mpProjection = "Robinson"
  res@mpGeophysicalLineColor = "gray42"
  res@mpPerimOn    = False
  res@mpGridLatSpacingF =  90            ; change latitude  line spacing
  res@mpGridLonSpacingF = 180.           ; change longitude line spacing
  res@mpGridLineColor   = "transparent"  ; trick ncl into drawing perimeter
  res@mpGridAndLimbOn   = True           ; turn on lat/lon lines
  res@mpFillOn = False
  res@mpCenterLonF = 0.
  res@mpOutlineOn = True
  res@gsnDraw      = False
  res@gsnFrame     = False
  res@cnLevelSelectionMode = "ExplicitLevels"
  res@cnLineLabelsOn = False
  res@cnFillOn        = True
  res@cnLinesOn       = False
  res@lbLabelBarOn    = False

  res@gsnLeftStringOrthogonalPosF = -0.03
  res@gsnLeftStringParallelPosF = .005
  res@gsnRightStringOrthogonalPosF = -0.03
  res@gsnRightStringParallelPosF = 0.96
  res@gsnRightString = ""
  res@gsnLeftString = ""
  res@gsnLeftStringFontHeightF = 0.018
  res@gsnCenterStringFontHeightF = 0.022
  res@gsnRightStringFontHeightF = 0.018

;  res@cnLevels = fspan(-.4,.4,9)
  res@cnLevels = fspan(-.5,.5,11)
;  cmap = read_colormap_file("AR6_colormaps_NCL/AR6_Temp_10.rgb")
  cmap = read_colormap_file("$diag_scripts/shared/plot/rgb/ipcc-ar6_temperature_div_12.rgb")
  res@cnFillColors = cmap

  plot = new(2,graphic)
  res@gsnCenterString = "(a) AMV - Observations"
  plot(0) = gsn_csm_contour_map(wks,amv_obs,res)
  res@gsnCenterString = "(b) AMV - Multimodel Ensemble"
  plot(1) = gsn_csm_contour_map(wks,mme_amv,res)

  panres = True
  panres@gsnPanelBottom = 0.41
  panres@gsnPanelRight = 0.7
  panres@gsnPanelLabelBar = True
  panres@gsnPanelYWhiteSpacePercent = 3.0
  panres@pmLabelBarHeightF = 0.05
  panres@pmLabelBarWidthF = 0.38
  panres@lbTitleOn = True
  panres@lbTitleString = "~S~o~N~C SD~S~-1~N~"
  panres@lbTitlePosition  = "Bottom"              ; title position
  panres@lbTitleFontHeightF= .010                ; make title smaller
  panres@lbBoxLineColor = "gray70"
  panres@lbLabelFontHeightF = .010
  panres@lbTitleFontHeightF= .008
  panres@txFontHeightF = 0.022
  panres@gsnFrame = False
  panres@lbBoxEndCapStyle = "TriangleBothEnds"
  gsn_panel(wks,plot,(/1,2/),panres)


  ;--------------------------------------------------------
  ;Plotting taylor diagram, panel 3
  ;--------------------------------------------------------

  tres                 = True                     ; diagram mods desired
  colors = new(dimsizes(amv_tseries_hist&nens),string)
  colors = "red3"
  tres@Colors          = colors       ; marker colors
  markers = ensemble_assign
  markers = 2
  tres@Markers         = markers ; ensemble_assign+1        ; marker styles
  tres@gsMarkerSizeF   = 0.006                    ; marker size
  tres@txFontHeightF   = 0.0225                    ; text size
  tres@stnRad         = (/ 2., 3. /)     ; additional standard radii
  tres@ccRays          = (/ 0.4, 0.9 /)     ; correllation rays
  tres@ccRays_color    = "LightGray" ;(/.9,.9,.9,.9/)<--Does not work     ; default is "black"

  tres@centerDiffRMS   = True             ; RMS 'circles'
  tres@centerDiffRMS_color = "LightGray" ;(/.9,.9,.9,.1/)  ; default is "black"/)

  tres@taylorDraw  = False                          ; don't draw
  tres@taylorFrame = False
  tres@trXMaxF = 2.0

  patcor = tay_stats_hist(:,0:0)
  ratio  = tay_stats_hist(:,1:1)

  ; mask the few values off of the Taylor plot
  ratio = where(ratio.ge.3.5,ratio@_FillValue,ratio)
  taylor_amv_ratio_C5 = where(taylor_amv_ratio_C5.ge.3.5,taylor_amv_ratio_C5@_FillValue,taylor_amv_ratio_C5)

  tplot = taylor_diagram3(wks, ratio, patcor, tres)

  colors2 = new((/dimsizes(taylor_amv_ratio_C5(:,0))/),string)
  colors2 = "CadetBlue3"
  tres@Colors          :=   colors2
  markers2 = new(dimsizes(taylor_amv_ratio_C5(:,0)),integer)
  markers2 = 2
  tres@Markers := markers2
  tres@tfPolyDrawOrder = "PreDraw"

  tplot_C5 = taylor_diagram3(wks, taylor_amv_ratio_C5, taylor_amv_patcor_C5, tres)
  overlay(tplot_C5,tplot)

  colors := new(dim_obs,string)
  marker = new(dim_obs,integer)
  colors = "black"
  marker = 1
  tres@Colors          := colors       ; marker colors
  tres@Markers         := marker
  tres@gsMarkerSizeF   = 0.028
  more_amv_obs_patcor = tay_stats_obs(:,0:0)
  more_amv_obs_ratio  = tay_stats_obs(:,1:1)
  print(more_amv_obs_patcor+" "+more_amv_obs_ratio)
  tplot2 = taylor_diagram3(wks, more_amv_obs_ratio, more_amv_obs_patcor, tres)
  overlay(tplot_C5,tplot2)




  mstring = "y"
  fontnum = 35
  xoffset = 0.0
  yoffset = 0.0
  ratioZ   = 1.0
  size    = 1.0
  angle   = 0.0

  new_marker = NhlNewMarker(wks, mstring, fontnum, xoffset, yoffset, ratioZ, size, angle)

  tres@gsMarkerSizeF   = 0.008
  tres@Colors := "white"
  tres@Markers:= new_marker
  tplot3a = taylor_diagram3(wks, tay_stats_hist_cmip5EM(0:0,1:1),tay_stats_hist_cmip5EM(0:0,0:0), tres)
  overlay(tplot_C5,tplot3a)
  tres@Colors := "dodgerblue"
  tres@Markers:= 2
  tres@gsMarkerThicknessF = 3.0
  tplot3b = taylor_diagram3(wks, tay_stats_hist_cmip5EM(0:0,1:1),tay_stats_hist_cmip5EM(0:0,0:0), tres)
  overlay(tplot_C5,tplot3b)


  print(tay_stats_hist_cmip5EM)
  print(tay_stats_hist_cmip6EM)
  tres@gsMarkerThicknessF = 1.0
  tres@Colors := "white"
  tres@Markers:= new_marker
  tplot4a = taylor_diagram3(wks, tay_stats_hist_cmip6EM(0:0,1:1),tay_stats_hist_cmip6EM(0:0,0:0), tres)
  overlay(tplot_C5,tplot4a)
  tres@Colors := "orange"
  tres@Markers:= 2
  tres@gsMarkerThicknessF = 3.0
  tplot4b = taylor_diagram3(wks, tay_stats_hist_cmip6EM(0:0,1:1),tay_stats_hist_cmip6EM(0:0,0:0), tres)
  overlay(tplot_C5,tplot4b)

  panres@gsnPanelLabelBar = False
  panres@gsnPanelLeft = 0.67
  delete(panres@gsnPanelRight)
  panres@gsnPanelTop = 0.81
  panres@gsnPanelBottom = 0.60
  gsn_panel(wks,tplot_C5,(/1,1/),panres)


  ;--------------------------------------------------------
  ;Plotting box plots, panel 4 and 5
  ;--------------------------------------------------------

  bres            = True
  bres@vpWidthF = 0.6
  bres@vpHeightF = 0.45
  bres@trXMinF = 0.
  bres@trXMaxF = 5.
  bres@trYMinF = -1.
  bres@trYMaxF = 1.
  bres@tmXBLabels = (/"Control","Control","Control","Control","Historical","Historical","Historical","Historical"/) ; labels for each box
  bres@tmXBOn = False
  bres@tmYLLabelFontHeightF = 0.024
  bres@tmYLMode = "Explicit"
  bres@tmYLValues = (/-1,-0.5,0,0.5,1/)
  bres@tmYLLabels = bres@tmYLValues
  bres@tmYLMinorValues = fspan(-1,1,21)
  bres@tiMainFont = "Helvetica"
  bres@tiMainString = ""   ;"Autocorrelation"
;  bres@tiMainFontHeightF = 0.032
;  bres@tiMainOffsetYF = 0.00
  bres@tiYAxisString = "( )"
  bres@tiYAxisFontHeightF = 0.035

  llres                   = True
  llres@gsLineThicknessF  = 1.5                 ; line thickness

  opti          = True
  opti@boxWidth = .3				; Width of box (x units)
  opti@boxColors = (/"CadetBlue3","red3","CadetBlue3","red3","CadetBlue3","red3","CadetBlue3","red3"/)

  txres = True
  txres@txFontHeightF = 0.02

  plres = True
  plres@gsLineColor = "gray50"
  plres@gsLineThicknessF = 1.0

  polyres = True
  polyres@gsFillColor = (/0.75,0.75,0.75,1/)  ;.3/)
;- - - - - - - - - - - - - - - - - - - - - - - -
; Draw blank box plots containing gray boxes representing observational spread, panels 4-5
;   - Blank plots were drawn due to draw order issues when trying to plot boxplots and the gray boxes.
;
  plot_blank = new(2,graphic)
  bres@tiYAxisFontColor = "white"
  bres@tmYLLabelFontColor = "white"
  plot_blank(0) = blank_boxplot(wks,bres)
  bres@tiYAxisString = "(~S~o~N~C)"
  bres@trYMinF = -0.065
  bres@trYMaxF = 0.4
  bres@tmYLValues := (/0.0,0.1,0.2,0.3,.4/)
  bres@tmYLLabels := bres@tmYLValues
  bres@tmYLMinorValues := fspan(0,1.5,16)
  plot_blank(1) = blank_boxplot(wks,bres)

  print("Lag1 AMV Obs: "+lag1_amv_tseries_obs)
  print("Lag10 AMV-LP Obs: "+lag10_amv_tseries_lp_obs)
  print("Standard deviation AMV Obs: "+ stddev_amv_tseries_obs)
  print("Standard deviation AMV-LP Obs: "+ stddev_amv_tseries_lp_obs)

  dumA = new(4,graphic)
  dumA(0) = gsn_add_polygon(wks,plot_blank(0),(/-1,2.25,2.25,-1.,-1./),(/min(lag1_amv_tseries_obs),min(lag1_amv_tseries_obs),max(lag1_amv_tseries_obs),max(lag1_amv_tseries_obs),min(lag1_amv_tseries_obs)/),polyres)
  dumA(1) = gsn_add_polygon(wks,plot_blank(0),(/2.75,4.75,4.75,2.75,2.75/),(/min(lag10_amv_tseries_lp_obs),min(lag10_amv_tseries_lp_obs),max(lag10_amv_tseries_lp_obs),max(lag10_amv_tseries_lp_obs),min(lag10_amv_tseries_lp_obs)/),polyres)
  dumA(2) = gsn_add_polygon(wks,plot_blank(1),(/-1,2.25,2.25,-1.,-1./),(/min(stddev_amv_tseries_obs),min(stddev_amv_tseries_obs),max(stddev_amv_tseries_obs),max(stddev_amv_tseries_obs),min(stddev_amv_tseries_obs)/),polyres)
  dumA(3) = gsn_add_polygon(wks,plot_blank(1),(/2.75,4.75,4.75,2.75,2.75/),(/min(stddev_amv_tseries_lp_obs),min(stddev_amv_tseries_lp_obs),max(stddev_amv_tseries_lp_obs),max(stddev_amv_tseries_lp_obs),min(stddev_amv_tseries_lp_obs)/),polyres)

  panres@gsnPanelTop = 0.57
  panres@gsnPanelBottom = 0.39
  panres@gsnPanelRight = 0.45
  panres@gsnPanelXWhiteSpacePercent = 7.0
  delete(panres@gsnPanelLeft)
  gsn_panel(wks,plot_blank,(/1,2/),panres)
;- - - - - - - - - - - - - - - - - - - - - - - -
; Draw box plots, panels 4-5
;
  bres@tiYAxisFontColor = "black"
  bres@tmYLLabelFontColor = "black"
  bres@tiYAxisString = "( )"
  bres@trYMinF = -1.
  bres@trYMaxF = 1.
  bres@tmYLValues := (/-1,-0.5,0,0.5,1/)
  bres@tmYLLabels := bres@tmYLValues
  bres@tmYLMinorValues := fspan(-1,1,21)

  dum = new(7,graphic)
  dum2 = dum
  bplot = new(2,graphic)

  opti@boxFill_Colors = (/"transparent","transparent","CadetBlue3","red3","transparent","transparent","CadetBlue3","red3"/)
  opti@boxFill_midColor = (/"CadetBlue3","red3","white","white","CadetBlue3","red3","white","white"/)
  bplot(0) = boxplot2(wks,(/0.5,1,1.5,2, 3,3.5,4,4.5/),yval_ac,opti,bres,llres)
  dum(0) = gsn_add_text(wks,bplot(0),"piControl",.75,bres@trYMinF+0.15,txres)
  dum(1) = gsn_add_text(wks,bplot(0),"Historical",1.75,bres@trYMinF+0.155,txres)
  dum(2) = gsn_add_text(wks,bplot(0),"piControl",3.25,bres@trYMinF+0.15,txres)
  dum(3) = gsn_add_text(wks,bplot(0),"Historical",4.25,bres@trYMinF+0.155,txres)
  dum(4) = gsn_add_polyline(wks,bplot(0),(/-5,5/),(/0.,0./),plres)
  dum(5) = gsn_add_text(wks,bplot(0),"Unsmoothed",1.25,1,txres)
  dum(6) = gsn_add_text(wks,bplot(0),"Low-Pass",3.75,1,txres)

  bres@tiYAxisString = "(~S~o~N~C)"
  bres@trYMinF = -0.065
  bres@trYMaxF = 0.4
  bres@tmYLValues := (/0.0,0.1,0.2,0.3,.4/)
  bres@tmYLLabels := bres@tmYLValues
  bres@tmYLMinorValues := fspan(0,1.5,16)
  bres@tiMainString = "" ;"Standard Deviation (~S~o~N~C)"
  bplot(1) = boxplot2(wks,(/0.5,1,1.5,2,3,3.5,4,4.5/),yval_sd,opti,bres,llres)

  dum2(0) = gsn_add_text(wks, bplot(1), "piControl", .75, \
                         bres@trYMinF+0.03, txres)
  dum2(1) = gsn_add_text(wks, bplot(1), "Historical", 1.75, \
                         bres@trYMinF+0.032, txres)
  dum2(2) = gsn_add_text(wks, bplot(1), "piControl", 3.25, \
                         bres@trYMinF+0.03, txres)
  dum2(3) = gsn_add_text(wks, bplot(1), "Historical", 4.25, \
                         bres@trYMinF+0.032, txres)
  dum2(4) = gsn_add_polyline(wks, bplot(1), (/-5, 5/), (/0., 0./), plres)
  dum2(5) = gsn_add_text(wks, bplot(1), "Unsmoothed", 1.25, 0.4, txres)
  dum2(6) = gsn_add_text(wks, bplot(1), "Low-Pass", 3.75, 0.4, txres)

  gsn_panel(wks,bplot,(/1,2/),panres)

  ;--------------------------------------------------------
  ;Plotting timeseries plot, panel 6
  ;--------------------------------------------------------

  mres = True
  mres@gsnDraw = False
  mres@gsnFrame = False
  mres@vpWidthF = 0.8
  mres@vpHeightF = 0.26
  mres@trXMinF = 1899.5
  mres@trXMaxF = 2015.5
  mres@trYMinF = -.4
  mres@trYMaxF = .4

  mres@gsnCenterString = ""
  mres@gsnLeftString = ""
  mres@gsnRightString = ""
  mres@xyMonoDashPattern = True

  mres2 = mres
  mres@xyMonoLineColor    = False             ; want colored lines
  ; colors chosen, last line = line at Y = 0
  mres@xyLineColors       = (/"black", "CadetBlue3", "red3", "gray50", "gray30", "gray30"/)
  mres@xyLineThicknesses	 = (/2.5, 2.5, 2.5, 1.5, 1.5, 1.5/)
  mres@xyMonoDashPattern = False
  mres@xyDashPatterns = (/0, 0, 0, 0, 1, 1/)

  mres2@gsnXYFillColors = "LightBlue"  ;"CadetBlue3"
  mres2@xyLineColor     = -1
  mres2@tiYAxisString = "(~S~o~N~C)"
  mres2@tiYAxisFontHeightF = 0.02
  mres2@tiYAxisOffsetYF = -0.005

  bot_plot = gsn_csm_xy(wks, fspan(syearA, 2005.91666, \
                        dimsizes(hist_range_C5&time)), hist_range_C5, mres2)
  delete(mres2@gsnXYFillColors)
  mres2@gsnXYFillColors = (/.7, 0, 0, .3/)  ;"LightPink"
  mid_plot = gsn_csm_xy(wks, fspan(syearA, eyearA+.91666, \
                        dimsizes(hist_range&time)), hist_range, mres2)

  tplot = gsn_csm_xy(wks, fspan(syearA, eyearA+.91666, \
                     dimsizes(hist_range&time)), timeseries, mres)
  overlay(bot_plot, mid_plot)
  overlay(bot_plot, tplot)


  panres@gsnPanelTop = 0.58
  panres@gsnPanelBottom = 0.4
  panres@gsnPanelLeft = 0.4
  delete(panres@gsnPanelXWhiteSpacePercent)
  delete(panres@gsnPanelRight)
  gsn_panel(wks, bot_plot, (/1, 1/), panres)

  ;--------------------------------------------------------
  ; Set titles for panels 3-6
  ;--------------------------------------------------------

  textres = True
  textres@txFontHeightF = 0.012
  gsn_text_ndc(wks, "(c) AMV Taylor Statistics", 0.83, 0.82, textres)
  gsn_text_ndc(wks, "(d) Autocorrelation", 0.115, 0.555, textres)
  gsn_text_ndc(wks, "(e) Standard Deviation", 0.345, 0.555, textres)
  gsn_text_ndc(wks, "(f) AMV Timeseries", 0.73, 0.585, textres)


  ;--------------------------------------------------------
  ; Boxplot legends
  ;--------------------------------------------------------

  gres = True
  gres@YPosPercent = 40.75    ; expressed as %, 0->100, sets position of top border of legend
  gres@LineLabelWhiteSpacePercent = .5
  gres@ItemSpacePercent = .9

  lineres = True
  lineres@lgLineColors = "CadetBlue3"  ; line colors
  lineres@lgLineThicknesses = 1.5                        ; line thicknesses
  lineres@LineLengthPercent = 2.                         ; expressed as %, 0->100, length of line
  lineres@lgDashIndexes = 0

  textres = True
  textres@lgLabels = (/"CMIP5", "CMIP6"/)  ; legend labels (required)
  textres@lgLabelFontHeights = 0.006
  gres@XPosPercent = 15.
  lineres@lgLineColors := (/"CadetBlue3", "Red3"/)
  simple_legend_ndc(wks, gres, lineres, textres)
  gres@XPosPercent = 37.5
  simple_legend_ndc(wks, gres, lineres, textres)

  gres@YPosPercent = 40.25
  gres@XPosPercent = 6.5
  lineres@lgLineThicknesses = 10.
  textres@lgLabels := " Observational~C~ Range"   ; ("+nobs+")"
  lineres@lgLineColors :=  polyres@gsFillColor
  simple_legend_ndc(wks, gres, lineres, textres)

  gres@XPosPercent = 29.
  simple_legend_ndc(wks, gres, lineres, textres)

  ;--------------------------------------------------------
  ; Timeseries legend
  ;--------------------------------------------------------

  lineres@LineLengthPercent = 1.5
  lineres@lgLineThicknesses = 2.5
  gres@YPosPercent = 44
  lineres@lgLineColors := "black"
  gres@XPosPercent = 53
  textres@lgLabels = "Observations"
  simple_legend_ndc(wks, gres, lineres, textres)

  lineres@lgLineColors = "CadetBlue3"
  textres@lgLabels = "CMIP5 MME"
  gres@XPosPercent = 61.
  simple_legend_ndc(wks, gres, lineres, textres)

  gres@XPosPercent = 69
  lineres@lgLineColors = "Red3"
  textres@lgLabels = "CMIP6 MME"
  simple_legend_ndc(wks, gres, lineres, textres)

  lineres@lgLineThicknesses = 9.5
  gres@XPosPercent = 77
  lineres@lgLineColors :=  "LightBlue"
  textres@lgLabels = " CMIP5 Range"
  simple_legend_ndc(wks, gres, lineres, textres)

  gres@XPosPercent = 85.5
  lineres@lgLineColors :=  (/.7, 0, 0, .3/)
  textres@lgLabels = " CMIP6 Range"
  simple_legend_ndc(wks, gres, lineres, textres)

  ;--------------------------------------------------------
  ; Taylor legend
  ;--------------------------------------------------------

  ; lres = True
  ; lres@gsFillColor = (/1,1,1,1/) ;0.5/)
  ; gsn_polygon_ndc(wks, (/.84, .91, .91, .84, .84/), \
  ;                 (/.63, .63, .68, .68, .63/), lres)

  mares = True
  mares@gsMarkerIndex = 1
  mares@gsMarkerSizeF   = 0.018
  tlres = True
  tlres@txFontHeightF = 0.006
  tlres@txJust = "CenterLeft"

  mares@gsMarkerColor = "Black"
  gsn_polymarker_ndc(wks, .90, .805, mares)
  gsn_text_ndc(wks, "Observations", .905, .805, tlres)

  mares@gsMarkerIndex = 2
  mares@gsMarkerSizeF   = 0.006
  mares@gsMarkerColor = "CadetBlue3"
  gsn_polymarker_ndc(wks, .90, .795, mares)
  gsn_text_ndc(wks, "CMIP5 Model", .905, .795, tlres)
  mares@gsMarkerColor = "Red3"
  gsn_polymarker_ndc(wks, .90, .785, mares)
  gsn_text_ndc(wks, "CMIP6 Model", .905, .785, tlres)

  mares@gsMarkerIndex = 2
  mares@gsMarkerColor = "dodgerblue"
  mares@gsMarkerThicknessF = 3.0
  gsn_polymarker_ndc(wks, .90, .775, mares)
  gsn_text_ndc(wks, "CMIP5 MME", .905, .775, tlres)
  mares@gsMarkerColor = "orange"
  gsn_polymarker_ndc(wks, .90, .765, mares)
  gsn_text_ndc(wks, "CMIP6 MME", .905, .765, tlres)

  ; drawNDCGrid(wks)
  frame(wks)
  delete(wks)

  log_info(" wrote " + ofile + "." + file_type)

end
