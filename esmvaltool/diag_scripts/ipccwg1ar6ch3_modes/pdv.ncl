; #############################################################################
; Compute PDV metrics
; Authors: Lisa Bock (DLR, Germany), Soufiane Karmouche (University of Bremen,
;          Germany), Yu Kosaka (University of Tokyo, Japan) and Adam Phillips
;          (NCAR, U.S.)
; #############################################################################
;
; Description
; Compute PDV metrics for observations, CMIP6-historical, CMIP6-picontrol,
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
;   20210306-Yu Kosaka: added provenance
;   20210301-Yu Kosaka: revised figure details
;   20210205-Yu Kosaka: minor revision in figure legend
;   20210131-Yu Kosaka: revised from monthly to annual mean data
;   20210117-Yu Kosaka: adapt Adam's updates script to the ESMValTool
;   20200715-bock_lisa: adapt Adam's scipt to the ESMValTool
;
; #############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/plot/mder.ncl"

load "$diag_scripts/ipccwg1ar6ch3_modes/functions.ncl"

function monthly2annual(x)
local x, y, ii
begin
  y = x(::12, :, :)
  dims = dimsizes(y)
  do ii = 0, dims(0)-1
    y(ii, :, :) = dim_avg_n(x(12*ii:12*(ii+1)-1, :, :), 0)
  end do
  return(y)
end

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
  ntime = eyearA - syearA + 1

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

  outpath = config_user_info@plot_dir
  ofile = outpath + "pdv"
  
  ; setting constants
  fca = 1./ 10.       ; 10 year low pass
  fcb = 0.
  ihp = 0
  nsigma = 1
  nwgt = min((/11, ntime - 1/))
  pi=4.*atan(1.0)
  rad=(pi/ 180.)

  ; create low pass filter
  wgt = new(nwgt, float)
  wgt = filwgts_lancos(nwgt, ihp, fca, fcb, nsigma)

  ;--------------------------------------------------------
  ; process observational data
  ;--------------------------------------------------------

  iref_obs_datasets = ind(obs_datasets.eq.diag_script_info@reference_dataset)
  iobs_datasets = ispan(0, dim_obs-1, 1)
  iobs_datasets(0) = iref_obs_datasets
  iobs_datasets(iref_obs_datasets) = 0
  do ii = 0, dim_obs - 1
    iobs = iobs_datasets(ii)
    
    obs_datasets(ii) = obs[iobs]@dataset 
    log_info(obs_datasets(ii))

    ; Read observational data
    arr = read_data(obs[iobs])
    arr = where(arr.le.-1.8, -1.8, arr)    ; set all values below -1.8 to -1.8

    arr = rmMonAnnCycTLL(arr)
    tmp := monthly2annual(arr)
    delete(arr)
    arr = tmp
    delete(tmp)
        
    coswgt  := cos(rad*arr&lat)
    coswgt!0 = "lat"
    coswgt&lat= arr&lat
    tpi := (/wgt_areaave(arr(:,{-10:10},{170:270}),coswgt({-10.:10.}),1.0,0) - \
             (0.5*(wgt_areaave(arr(:,{25:45},{140:215}),coswgt({25.:45.}),1.0,0)+wgt_areaave(arr(:,{-50:-15},{150:200}),coswgt({-50.:-15.}),1.0,0))) /)
    tpi!0 = "time"
    tpi&time = arr&time
    tpi_lp := wgt_runave_n_Wrap(tpi,wgt,1,0)   ; create tpi-low pass

    arr = lonFlip(arr)
    arr_lp := wgt_runave_n_Wrap(arr,wgt,1,0)

    ipo := (/ regCoef(dim_standardize(tpi_lp,0),arr_lp(lat|:,lon|:,time|:)) /)
    copy_VarMeta(arr_lp(0,:,:),ipo)

    if (ii.eq.0) then
      reg_fld=arr_lp(0,:,:)
      reg_fld=(/regCoef_n(dim_standardize(tpi_lp,0),arr_lp,0,0)/)
      cor_fld=reg_fld
      cor_fld=(/escorc(dim_standardize(tpi_lp,0),arr_lp(lat|:,lon|:,time|:))/)

      prob_rtest=cor_fld
      prob_rtest=(/ rtest(cor_fld,dimsizes(tpi_lp)/10,0) /)
      prob_rtest=where(arr_lp(0,:,:).eq.arr_lp@_FillValue,prob_rtest@_FillValue,prob_rtest)

      siglev = 0.10

      sig = where(prob_rtest.gt.siglev, 0., 1.)
           
      fn = out_path + "pdv.obs."+syearA+"-"+eyearA+".nc"
      print("fn = " + fn)
      system("rm -f " + fn)
      z = addfile(fn,"c")
      z@source = systemfunc("pwd")+"/"+get_script_name()
      z->ipo_pattern_obs0 = ipo    ; write ersstv5 pattern to file

      ipo_pattern_obs = arr(:dimsizes(obs_datasets)-1,:,:)
      ipo_pattern_obs = ipo_pattern_obs@_FillValue
      delete(ipo_pattern_obs&time)
      ipo_pattern_obs!0 = "E"
      ipo_pattern_obs&E = ispan(1,dimsizes(obs_datasets),1)
      ipo_pattern_sig_obs = ipo_pattern_obs
      ipo_pattern_obs(ii,:,:) = (/ ipo /)
      ipo_pattern_sig_obs(ii,:,:) = (/ sig /)
      delete(sig)
           
      ipo_obs_flip = ipo   ; for taylor stats
      ipo_obs_flip = lonFlip(ipo)
           
      tpi_write = new((/dimsizes(obs_datasets),(eyearA-syearA+1)/),typeof(ipo))
      tpi_write!0 = "E"
      tpi_write&E = ispan(1,dimsizes(obs_datasets),1)
      tpi_write!1 = "time"
      tpi_write&time = tpi&time
      tpi_lp_write = tpi_write
      tay_stat = new((/dimsizes(obs_datasets),3/),double)
      tay_stat!0 = "E"
      tay_stat&E = ispan(1,dimsizes(obs_datasets),1)
      tay_stat!1 = "stat"
      tay_stat&stat = ispan(0,2,1)
      lat := ipo&lat
      lat!0 = "lat"
      lat&lat = lat
      w = cos(rad*lat({-40:60}))  ; weights for Taylor diagram
    else
      ipo_flip := ipo
      ipo_flip = lonFlip(ipo_flip)
      tay_stat(ii,:) = (/ taylor_stats(ipo_flip({-40:60},{110:290}), ipo_obs_flip({-40:60},{110:290}), w, 0) /)
      delete(ipo_flip)
      ipo_pattern_obs(ii,:,:) = (/ ipo /)
    end if
    tpi_write(ii,:)      = (/ tpi /)
    tpi_lp_write(ii,:)   = (/ tpi_lp /)
    delete(arr)
  end do

  z->tpi_lp = tpi_lp_write
  z->tpi = tpi_write
  z->tay_stats = tay_stat
  z->ipo_pattern_obs = ipo_pattern_obs
  z->ipo_pattern_obs_signif = ipo_pattern_sig_obs
  obs_datasets = obs_datasets(iobs_datasets)
  iref_obs_datasets = ind(obs_datasets.eq.diag_script_info@reference_dataset)
  z@datasets = obs_datasets
  delete([/z/])
  delete(iobs_datasets)
  delete([/tpi_lp_write, tpi_write, tay_stat, ipo_pattern_obs, ipo_pattern_sig_obs/])

  log_provenance(fn, ofile+"."+file_type, \
                 "Observed patterns, time series, and taylor statitics of PDV", \
                 (/"anomaly","var"/), (/"global"/), (/"geo", "taylor", "box", "times"/), \
                 "kosaka_yu", "", metadata_att_as_array(obs, "filename"))
  
  z = addfile(fn,"r")
  ipo_obs = z->ipo_pattern_obs(0,:,:)
  tpi_lp_obs = z->tpi_lp
  tpi_obs = z->tpi
  tay_stats_obs = z->tay_stats
  sig = z->ipo_pattern_obs_signif

  ;--------------------------------------------------------
  ; process CMIP6 historical data
  ;--------------------------------------------------------

  lat := ipo_obs&lat
  lat!0 = "lat"
  lat&lat = lat
  w = cos(rad*lat({-40:60}))  ; weights for Taylor diagram

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


  do gg = 0,nens-1
     arr := read_data(hist_cmip6[gg])

     arr = where(arr.le.-1.8,-1.8,arr)    ; set all values below -1.8 to -1.8
     d = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")   ; mask out land (this is redundant for data that is already masked)
     lsm := landsea_mask(d->LSMASK,arr&lat,arr&lon)
     arr = mask(arr,conform(arr,lsm,(/1,2/)).ge.1,False)
     
     arr = rmMonAnnCycTLL(arr)
     tmp := monthly2annual(arr)
     delete(arr)
     arr = tmp
     delete(tmp)

     coswgt  := cos(rad*arr&lat)
     coswgt!0 = "lat"
     coswgt&lat= arr&lat
     tpi := (/  wgt_areaave(arr(:,{-10:10},{170:270}),coswgt({-10.:10.}),1.0,0) - \
     (0.5*(wgt_areaave(arr(:,{25:45},{140:215}),coswgt({25.:45.}),1.0,0)+wgt_areaave(arr(:,{-50:-15},{150:200}),coswgt({-50.:-15.}),1.0,0))) /)
     tpi!0 = "time"
     tpi&time = arr&time
     tpi_lp := wgt_runave_n_Wrap(tpi,wgt,1,0)   ; create tpi-low pass
     arr = lonFlip(arr)
     arr_lp := wgt_runave_n_Wrap(arr,wgt,1,0)   ; create tpi-low pass

     ipo := (/ regCoef(dim_standardize(tpi_lp,0),arr_lp(lat|:,lon|:,time|:)) /)
     copy_VarMeta(arr_lp(0,:,:),ipo)

     if (gg.eq.0) then
       finarr = new((/nens,dimsizes(ipo_obs&lat),dimsizes(ipo_obs&lon)/),typeof(ipo))
       finarr!0 = "nens"
       finarr&nens = ispan(0,nens-1,1)
       finarr!1 = "lat"
       finarr&lat = ipo_obs&lat
       finarr!2 = "lon"
       finarr&lon = ipo_obs&lon

       tpi_write := new((/nens,(eyearA-syearA+1)/),typeof(ipo))
       tpi_write!0 = "nens"
       tpi_write&nens = finarr&nens
       tpi_write!1 = "time"
       tpi_write&time = arr_lp&time

       ;print(getvardimnames(tpi) + " : tpi : " + dimsizes(tpi))
       ;print(getvardimnames(ipo) + " : ipo : " + dimsizes(ipo))
       ;print(getvardimnames(tpi_write) + " : tpi_write : " + dimsizes(tpi_write))
       ;print(getvardimnames(tpi_lp) + " : tpi_lp : " + dimsizes(tpi_lp))
       ;print(getvardimnames(tpi_lp_write) + " : tpi_lp_write : " + dimsizes(tpi_lp_write))

       tpi_lp_write := tpi_write
       tay_stat := new((/nens,3/),double)
       tay_stat!0 = "nens"
       tay_stat&nens = finarr&nens
       tay_stat!1 = "stat"
       tay_stat&stat = ispan(0,2,1)

       ensemble_assign!0 = "nens"
       ensemble_assign&nens = finarr&nens
     end if
     finarr(gg,:,:) = (/ipo/)
     finarr_flip = ipo
     finarr_flip = lonFlip(finarr_flip)
     tay_stat(gg,:) = (/ taylor_stats(finarr_flip({-40:60},{110:290}), ipo_obs_flip({-40:60},{110:290}), w, 0) /)
     delete(finarr_flip)
     tpi_write(gg,:) = (/ tpi /)
     tpi_lp_write(gg,:) = (/ tpi_lp /)
     delete(arr)
     print("Done with "+gg+" out of "+nens+", "+model(gg))
  end do

  fn = out_path + "pdv.hist."+syearA+"-"+eyearA+".cmip6.nc"
  system("rm -f " + fn)
  z = addfile(fn,"c")
  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->ipo_model_pattern = finarr
  z->tpi      = tpi_write
  z->tpi_lp   = tpi_lp_write
  z->tay_stats = tay_stat
  z->ensemble_assign = ensemble_assign
  delete([/z/])
  delete([/finarr, tpi_write, tpi_lp_write, tay_stat, ensemble_assign/])
  
  log_provenance(fn, ofile+"."+file_type, \
                 "Patterns, time series, and taylor statitics of PDV in CMIP6 historical", \
                 (/"anomaly","var"/), (/"global"/), (/"geo", "taylor", "box", "times"/), \
                 "kosaka_yu", "", metadata_att_as_array(hist_cmip6, "filename"))

  z = addfile(fn,"r")
  ipo_hist = z->ipo_model_pattern
  tpi_hist = z->tpi
  tpi_lp_hist = z->tpi_lp
  tay_stats_hist = z->tay_stats
  ensemble_assign := z->ensemble_assign

;------------------------------------------------------------------------------
; Compare variance of mme vs. concatenated runs
;
  tpi_filter = tpi_lp_hist

  tpi_filter_em = tpi_lp_hist(:max(ensemble_assign)-1,:)
  tpi_filter_em = tpi_filter_em @_FillValue
  do gg = 1,max(ensemble_assign)         ; calculate ensemble means
     wind := ind(ensemble_assign.eq.gg)
     if (dimsizes(wind).eq.1) then
        tpi_filter_em(gg-1,:) = (/ tpi_lp_hist(wind,:) /)
     else
        tpi_filter_em(gg-1,:) = (/ dim_avg_n(tpi_lp_hist(wind,:),0) /)
     end if
  end do
  mme_tpi_filter = dim_avg_n_Wrap(tpi_filter_em,0)

  temp = ndtooned(tpi_lp_hist)
  snval = decimalPlaces(dim_variance(mme_tpi_filter)/dim_variance(temp)*100.,0,True)+"%"
  
  ;--------------------------------------------------------
  ; process CMIP6 piControl data
  ;--------------------------------------------------------

  do gg = 0, dim_cmip6_piControl - 1
     arr := read_data(piControl_cmip6[gg])

     arr = where(arr.le.-1.8,-1.8,arr)    ; set all values below -1.8 to -1.8
     d = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")   ; mask out land (this is redundant for data that is already masked)
     lsm := landsea_mask(d->LSMASK,arr&lat,arr&lon)
     arr = mask(arr,conform(arr,lsm,(/1,2/)).ge.1,False)
     
     arr = rmMonAnnCycTLL(arr)
     tmp := monthly2annual(arr)
     delete(arr)
     arr = tmp
     delete(tmp)
     
     coswgt  := cos(rad*arr&lat)
     coswgt!0 = "lat"
     coswgt&lat= arr&lat
     tpi := (/  wgt_areaave(arr(:,{-10:10},{170:270}),coswgt({-10.:10.}),1.0,0) - \
     (0.5*(wgt_areaave(arr(:,{25:45},{140:215}),coswgt({25.:45.}),1.0,0)+wgt_areaave(arr(:,{-50:-15},{150:200}),coswgt({-50.:-15.}),1.0,0))) /)
     tpi!0 = "time"
     tpi&time = arr&time
     tpi_lp := wgt_runave_n_Wrap(tpi,wgt,1,0)   ; create tpi-low pass
     arr = lonFlip(arr)
     arr_lp := wgt_runave_n_Wrap(arr,wgt,1,0)   ; create tpi-low pass

     ipo := (/ regCoef(dim_standardize(tpi_lp,0),arr_lp(lat|:,lon|:,time| :)) /)
     copy_VarMeta(arr_lp(0,:,:),ipo)

     if (gg.eq.0) then
        finarr := new((/dim_cmip6_piControl,dimsizes(ipo_obs&lat),dimsizes(ipo_obs&lon)/),typeof(ipo))
        finarr!0 = "nens"
        finarr&nens = ispan(0,dim_cmip6_piControl-1,1)
        finarr!1 = "lat"
        finarr&lat = ipo_obs&lat
        finarr!2 = "lon"
        finarr&lon = ipo_obs&lon

        tpi_write := new((/dim_cmip6_piControl, (eyearA-syearA+1)/),typeof(ipo))
        tpi_write!0 = "nens"
        tpi_write&nens = finarr&nens
        tpi_write!1 = "time"
        tpi_write&time = arr_lp&time

        tpi_lp_write := tpi_write
        tay_stat := new((/dim_cmip6_piControl,3/),double)
        tay_stat!0 = "nens"
        tay_stat&nens = finarr&nens
        tay_stat!1 = "stat"
        tay_stat&stat = ispan(0,2,1)
     end if
     finarr(gg,:,:) = (/ipo/)
     finarr_flip = ipo
     finarr_flip = lonFlip(finarr_flip)
     tay_stat(gg,:) = (/ taylor_stats(finarr_flip({-40:60},{110:290}), ipo_obs_flip({-40:60},{110:290}), w, 0) /)
     delete(finarr_flip)
     tpi_write(gg,:) = (/ tpi /)
     tpi_lp_write(gg,:) = (/ tpi_lp /)
     delete(arr)
     print("Done with "+gg+" out of "+(dim_cmip6_piControl-1))
  end do

  fn = out_path + "pdv.piControl.115yr.cmip6.nc"
  system("rm -f " + fn)
  z = addfile(fn,"c")

  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->ipo_model_pattern = finarr
  z->tpi      = tpi_write
  z->tpi_lp   = tpi_lp_write
  z->tay_stats = tay_stat
  ipo_piControl = finarr
  tpi_piControl = tpi_write
  tpi_lp_piControl = tpi_lp_write
  tay_stats_piControl = tay_stat
  delete([/tpi_write,finarr,tpi_lp_write,tay_stat/])

  log_provenance(fn, ofile+"."+file_type, \
                 "Patterns, time series, and taylor statitics of PDV in CMIP6 piControl", \
                 (/"anomaly","var"/), (/"global"/), (/"geo", "taylor", "box", "times"/), \
                 "kosaka_yu", "", metadata_att_as_array(piControl_cmip6, "filename"))
                 
  z = addfile(fn,"r")
  tpi_piControl = z->tpi
  tpi_lp_piControl = z->tpi_lp
  
  ;--------------------------------------------------------
  ; process CMIP5 historical data
  ;--------------------------------------------------------

  lat := ipo_obs&lat
  lat!0 = "lat"
  lat&lat = lat
  w = cos(rad*lat({-40:60}))  ; weights for Taylor diagram


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

  do gg = 0, dim_cmip5_hist - 1
    arr := read_data(hist_cmip5[gg])

    arr = where(arr.le.-1.8,-1.8,arr)    ; set all values below -1.8 to -1.8
    ; mask out land (this is redundant for data that is already masked)
    d = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
    lsm := landsea_mask(d->LSMASK,arr&lat,arr&lon)
    arr = mask(arr,conform(arr,lsm,(/1,2/)).ge.1,False)
    
    arr = rmMonAnnCycTLL(arr)
    tmp := monthly2annual(arr)
    delete(arr)
    arr = tmp
    delete(tmp)
    
    coswgt  := cos(rad*arr&lat)
    coswgt!0 = "lat"
    coswgt&lat= arr&lat
    tpi := (/  wgt_areaave(arr(:,{-10:10},{170:270}),coswgt({-10.:10.}),1.0,0) - \
    (0.5*(wgt_areaave(arr(:,{25:45},{140:215}),coswgt({25.:45.}),1.0,0)+wgt_areaave(arr(:,{-50:-15},{150:200}),coswgt({-50.:-15.}),1.0,0))) /)
    tpi!0 = "time"
    tpi&time = arr&time
    tpi_lp := wgt_runave_n_Wrap(tpi,wgt,1,0)   ; create tpi-low pass
    arr = lonFlip(arr)
    arr_lp := wgt_runave_n_Wrap(arr,wgt,1,0)   ; create tpi-low pass
    ipo := (/ regCoef(dim_standardize(tpi_lp,0),arr_lp(lat|:,lon|:,time|:)) /)
    copy_VarMeta(arr_lp(0,:,:),ipo)

    if (gg.eq.0) then
      finarr := new((/dim_cmip5_hist,dimsizes(ipo_obs&lat),dimsizes(ipo_obs&lon)/),typeof(ipo))
      finarr!0 = "dim_cmip5_hist"
      finarr&dim_cmip5_hist = ispan(0,dim_cmip5_hist-1,1)
      finarr!1 = "lat"
      finarr&lat = ipo_obs&lat
      finarr!2 = "lon"
      finarr&lon = ipo_obs&lon

      tpi_write := new((/dim_cmip5_hist,dimsizes(arr&time)/),typeof(ipo))
      tpi_write!0 = "dim_cmip5_hist"
      tpi_write&dim_cmip5_hist = finarr&dim_cmip5_hist
      tpi_write!1 = "time"
      tpi_write&time = arr_lp&time
      tpi_lp_write := tpi_write
      tay_stat = new((/dim_cmip5_hist,3/),double)
      tay_stat!0 = "dim_cmip5_hist"
      tay_stat&dim_cmip5_hist = finarr&dim_cmip5_hist
      tay_stat!1 = "stat"
      tay_stat&stat = ispan(0,2,1)

    end if
    finarr(gg,:,:) = (/ipo/)
    finarr_flip = ipo
    finarr_flip = lonFlip(finarr_flip)
    tay_stat(gg,:) = (/ taylor_stats(finarr_flip({-40:60},{110:290}), ipo_obs_flip({-40:60},{110:290}), w, 0) /)
    delete(finarr_flip)
    tpi_write(gg,:) = (/ tpi /)
    tpi_lp_write(gg,:) = (/ tpi_lp /)
    delete(arr)
    print("Done with "+gg+" out of "+dim_cmip5_hist+", "+model(gg))
  end do

  fn = out_path + "pdv.hist."+syearA+"-"+eyearA+".cmip5.nc"
  system("rm -f " + fn)
  z = addfile(fn,"c")
  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->ipo_model_pattern = finarr
  z->tpi      = tpi_write
  z->tpi_lp   = tpi_lp_write
  z->tay_stats = tay_stat
  z->ensemble_assign = ensemble_assign_C5
  delete([/z/])
  delete([/finarr, tpi_write, tpi_lp_write, tay_stat, ensemble_assign_C5/])

  log_provenance(fn, ofile+"."+file_type, \
                 "Patterns, time series, and taylor statitics of PDV in CMIP5 historical", \
                 (/"anomaly","var"/), (/"global"/), (/"geo", "taylor", "box", "times"/), \
                 "kosaka_yu", "", metadata_att_as_array(hist_cmip5, "filename"))
                 
  z = addfile(fn,"r")
  tpi_lp_hist_C5 = z->tpi_lp
  tpi_hist_C5    = z->tpi
  ipo_hist_C5 = z->ipo_model_pattern
  ea_c5 = z->ensemble_assign

  taylor_ipo_ratio_C5  = z->tay_stats(:,1:1)
  taylor_ipo_patcor_C5 = z->tay_stats(:,0:0)
  taylor_ipo_patcor_C5T = 0.5*(log( (1+taylor_ipo_patcor_C5) / (1-taylor_ipo_patcor_C5)))    ; convert to Z-space

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
     
     arr = rmMonAnnCycTLL(arr)
     tmp := monthly2annual(arr)
     delete(arr)
     arr = tmp
     delete(tmp)
     
     coswgt  := cos(rad*arr&lat)
     coswgt!0 = "lat"
     coswgt&lat= arr&lat
     tpi := (/  wgt_areaave(arr(:,{-10:10},{170:270}),coswgt({-10.:10.}),1.0,0) - \
     (0.5*(wgt_areaave(arr(:,{25:45},{140:215}),coswgt({25.:45.}),1.0,0)+wgt_areaave(arr(:,{-50:-15},{150:200}),coswgt({-50.:-15.}),1.0,0))) /)
     tpi!0 = "time"
     tpi&time = arr&time
     tpi_lp := wgt_runave_n_Wrap(tpi,wgt,1,0)   ; create tpi-low pass
     arr = lonFlip(arr)
     arr_lp := wgt_runave_n_Wrap(arr,wgt,1,0)   ; create tpi-low pass

     ipo := (/ regCoef(dim_standardize(tpi_lp,0),arr_lp(lat|:,lon|:,time|:)) /)
     copy_VarMeta(arr_lp(0,:,:),ipo)


     if (gg.eq.0) then
        finarr := new((/dim_cmip5_piControl,dimsizes(ipo_obs&lat),dimsizes(ipo_obs&lon)/),typeof(ipo))
        finarr!0 = "nens"
        finarr&nens = ispan(0,dim_cmip5_piControl-1,1)
        finarr!1 = "lat"
        finarr&lat = ipo_obs&lat
        finarr!2 = "lon"
        finarr&lon = ipo_obs&lon

        tpi_write := new((/dim_cmip5_piControl,dimsizes(arr_lp&time)/),typeof(ipo))
        tpi_write!0 = "nens"
        tpi_write&nens = finarr&nens
        tpi_write!1 = "time"
        tpi_write&time = arr_lp&time
        tpi_lp_write := tpi_write
        tay_stat := new((/dim_cmip5_piControl,3/),double)
        tay_stat!0 = "nens"
        tay_stat&nens = finarr&nens
        tay_stat!1 = "stat"
        tay_stat&stat = ispan(0,2,1)
     end if

     finarr(gg,:,:) = (/ipo/)
     finarr_flip = ipo
     finarr_flip = lonFlip(finarr_flip)
     tay_stat(gg,:) = (/ taylor_stats(finarr_flip({-40:60},{110:290}), ipo_obs_flip({-40:60},{110:290}), w, 0) /)
     delete(finarr_flip)
     tpi_write(gg,:) = (/ tpi /)
     tpi_lp_write(gg,:) = (/ tpi_lp /)
     delete(arr)
     print("Done with "+gg+" out of "+(dim_cmip5_piControl-1))
  end do

  fn = out_path + "pdv.piControl.115yr.cmip5.nc"
  system("rm -f " + fn)
  z = addfile(fn,"c")

  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->ipo_model_pattern = finarr
  z->tpi      = tpi_write
  z->tpi_lp   = tpi_lp_write
  z->tay_stats = tay_stat
  delete([/z/])

  log_provenance(fn, ofile+"."+file_type, \
                 "Patterns, time series, and taylor statitics of PDV in CMIP5 piControl", \
                 (/"anomaly","var"/), (/"global"/), (/"geo", "taylor", "box", "times"/), \
                 "kosaka_yu", "", metadata_att_as_array(piControl_cmip5, "filename"))
                 
  z = addfile(fn,"r")
  tpi_piControl_C5 = z->tpi
  tpi_lp_piControl_C5 = z->tpi_lp


  ;--------------------------------------------------------
  ; other processing steps?
  ;--------------------------------------------------------

  ipo_hist_em5 = ipo_hist_C5(:max(ea_c5)-1,:,:)
  ipo_hist_em5 = ipo_hist_em5@_FillValue
  tpi_lp_hist_em5 = tpi_lp_hist_C5(:max(ea_c5)-1,:)
  tpi_lp_hist_em5 = tpi_lp_hist_em5@_FillValue
  tay_ratio_C5 = taylor_ipo_ratio_C5(:max(ea_c5)-1,0)
  tay_ratio_C5 = tay_ratio_C5@_FillValue
  tay_patcor_C5 = taylor_ipo_patcor_C5(:max(ea_c5)-1,0)
  tay_patcor_C5 = tay_patcor_C5@_FillValue

  weights_C5  = new(dim_cmip5_hist,float)
  do gg = 1,max(ea_c5)         ; calculate CMIP5 ensemble means
     wind := ind(ea_c5.eq.gg)
     if (dimsizes(wind).eq.1) then
        ipo_hist_em5(gg-1,:,:) = (/ ipo_hist_C5(wind,:,:) /)
        tpi_lp_hist_em5(gg-1,:) = (/ tpi_lp_hist_C5(wind,:) /)
        tay_ratio_C5(gg-1) = (/ taylor_ipo_ratio_C5(wind,0) /)
        tay_patcor_C5(gg-1) = (/ taylor_ipo_patcor_C5T(wind,0) /)
     else
        ipo_hist_em5(gg-1,:,:) = (/ dim_avg_n(ipo_hist_C5(wind,:,:),0) /)
        tpi_lp_hist_em5(gg-1,:) = (/ dim_avg_n(tpi_lp_hist_C5(wind,:),0) /)
        tay_ratio_C5(gg-1) = (/ dim_avg_n(taylor_ipo_ratio_C5(wind,0),0) /)
        tay_patcor_C5(gg-1) = (/ dim_avg_n(taylor_ipo_patcor_C5T(wind,0),0) /)
     end if
     weights_C5(wind) = 1./dimsizes(wind)
  end do
    
  mme_ipo_C5 = dim_avg_n_Wrap(ipo_hist_em5,0)
  mme_tpi_lp_hist_C5 = dim_avg_n_Wrap(tpi_lp_hist_em5,0)
  delete([/ipo_hist_em5,tpi_lp_hist_em5/])

  tay_stats_hist_cmip5EM = new((/1,3/),double)
  delete([/mme_ipo_C5/])
  tay_stats_hist_cmip5EM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor_C5))-1)/ ((2.71828^(2*dim_avg(tay_patcor_C5))+1))) /)   ; convert back from z-space
  tay_stats_hist_cmip5EM(0,1) = (/ dim_avg(tay_ratio_C5) /)
;------------------------------------------------------------------------------
; Compare variance of mme vs. concatenated runs for CMIP5
;
  tpi_filter_em_C5 = tpi_lp_hist_C5(:max(ea_c5)-1,:)
  tpi_filter_em_C5 = tpi_filter_em_C5@_FillValue
  do gg = 1,max(ea_c5)         ; calculate ensemble means
     wind := ind(ea_c5.eq.gg)
     if (dimsizes(wind).eq.1) then
        tpi_filter_em_C5(gg-1,:) = (/ tpi_lp_hist_C5(wind,:) /)
     else
        tpi_filter_em_C5(gg-1,:) = (/ dim_avg_n(tpi_lp_hist_C5(wind,:),0) /)
     end if
  end do
  mme_tpi_filter_C5 = dim_avg_n_Wrap(tpi_filter_em_C5,0)

  temp := ndtooned(tpi_lp_hist_C5)
  snval_C5 = decimalPlaces(dim_variance(mme_tpi_filter_C5)/dim_variance(temp)*100.,0,True)+"%"
  ;--------------------------------------------------------

  tay_patcor_C6 = tay_stats_hist(:,0)
  tay_ratio_C6  = tay_stats_hist(:,1)
  tay_patcor_C6 = 0.5*(log( (1+tay_patcor_C6) / (1-tay_patcor_C6)))    ; convert to Z-space

  ipo_hist_em = ipo_hist(:max(ensemble_assign)-1,:,:)
  ipo_hist_em = ipo_hist_em@_FillValue
  tpi_lp_hist_em = tpi_lp_hist(:max(ensemble_assign)-1,:)
  tpi_lp_hist_em = tpi_lp_hist_em @_FillValue
  tay_ratio = tay_ratio_C6(:max(ensemble_assign)-1)
  tay_ratio = tay_ratio@_FillValue
  tay_patcor = tay_patcor_C6(:max(ensemble_assign)-1)
  tay_patcor = tay_patcor@_FillValue
  weights = new(dimsizes(ipo_hist&nens),float)
  nensmem = tay_patcor_C6(:max(ensemble_assign)-1)
  do gg = 1,max(ensemble_assign)         ; calculate ensemble means
     wind := ind(ensemble_assign.eq.gg)
     if (dimsizes(wind).eq.1) then
        ipo_hist_em(gg-1,:,:) = (/ ipo_hist(wind,:,:) /)
        tpi_lp_hist_em(gg-1,:) = (/ tpi_lp_hist(wind,:) /)
        tay_ratio(gg-1) = (/ tay_ratio_C6(wind) /)
        tay_patcor(gg-1) = (/ tay_patcor_C6(wind) /)
     else
        ipo_hist_em(gg-1,:,:) = (/ dim_avg_n(ipo_hist(wind,:,:),0) /)
        tpi_lp_hist_em(gg-1,:) = (/ dim_avg_n(tpi_lp_hist(wind,:),0) /)
        tay_ratio(gg-1) = (/ dim_avg(tay_ratio_C6(wind)) /)
        tay_patcor(gg-1) = (/ dim_avg(tay_patcor_C6(wind)) /)
     end if
     weights(wind) = 1./dimsizes(wind)
     nensmem(gg-1) = dimsizes(wind)
  end do
  tpi_lp_hist_em@num_ensemble_members = nensmem

  mme_ipo = dim_avg_n_Wrap(ipo_hist_em,0)
  mme_ipo_sign = mme_ipo
  mme_ipo_sign = mme_ipo_sign@_FillValue

  do cc = 0,dimsizes(mme_ipo&lat)-1        ; If more than 80% of the runs agree on sign, assign value of 1. 
     do dd = 0,dimsizes(mme_ipo&lon)-1     ; otherwise assign a value of 0.
        if (.not.ismissing(mme_ipo(cc,dd))) then
           tnum = toint(num(.not.ismissing(ipo_hist(:,cc,dd)))*.8)
           countp = num(ipo_hist(:,cc,dd).gt.0)
           countn = num(ipo_hist(:,cc,dd).lt.0)
           if (countp.gt.tnum.or.countn.gt.tnum) then      ; if more than 80% of the members are >0 or <0
             mme_ipo_sign(cc,dd) = 1   ; assign value of 1
           else
             mme_ipo_sign(cc,dd) = 0   ; otherwise assign value of 0
           end if
         end if
     end do
  end do
   
  mme_tpi_lp_hist = dim_avg_n_Wrap(tpi_lp_hist_em,0)

  tay_stats_hist_cmip6EM = new((/1,3/),double)
  tay_stats_hist_cmip6EM(0,0) = (/ ((2.71828^(2*dim_avg(tay_patcor))-1)/ ((2.71828^(2*dim_avg(tay_patcor))+1))) /)   ; convert back from z-space
  tay_stats_hist_cmip6EM(0,1) = (/ dim_avg(tay_ratio) /)


  ;--------------------------------------------------------
  ; Observations box plot statistics
  ;--------------------------------------------------------

  autocor_tpi_lp_obs = esacr(tpi_lp_obs,10)
  lag1_tpi_lp_obs = autocor_tpi_lp_obs(:,1)
  lag10_tpi_lp_obs = autocor_tpi_lp_obs(:,10)
  delete(autocor_tpi_lp_obs)

  autocor_tpi_obs = esacr(tpi_obs,10)
  lag1_tpi_obs = autocor_tpi_obs(:,1)
  lag10_tpi_obs = autocor_tpi_obs(:,10)
  delete(autocor_tpi_obs)

  ;--------------------------------------------------------
  ; CMIP6 historical autocorrelation box plot statistics
  ;--------------------------------------------------------

  temp_ac := esacr(tpi_lp_hist,10)
  lag10_tpi_lp_hist_B = (/weighted_percentile(temp_ac(:,10),weights,.05),weighted_percentile(temp_ac(:,10),weights,.25),weighted_mean(temp_ac(:,10),weights),weighted_percentile(temp_ac(:,10),weights,.75),weighted_percentile(temp_ac(:,10),weights,.95)/)
  
  temp_acZ := weighted_mean(0.5*(log( (1.+temp_ac(:, 10)) / (1.-temp_ac(:, 10)))), weights)   ; convert to Z-space
  lag10_tpi_lp_hist_B(2) = (/ ((2.71828^(2*temp_acZ)-1)/ ((2.71828^(2*temp_acZ)+1))) /)   ; convert back from z-space
  delete(temp_acZ)
  
  temp_ac := esacr(tpi_hist,10)
  lag1_tpi_hist_B = (/weighted_percentile(temp_ac(:,1),weights,.05),weighted_percentile(temp_ac(:,1),weights,.25),weighted_mean(temp_ac(:,1),weights),weighted_percentile(temp_ac(:,1),weights,.75),weighted_percentile(temp_ac(:,1),weights,.95)/)

  temp_acZ := weighted_mean(0.5*(log( (1.+temp_ac(:, 1)) / (1.-temp_ac(:, 1)))), weights)   ; convert to Z-space
  lag1_tpi_hist_B(2) = (/ ((2.71828^(2*temp_acZ)-1)/ ((2.71828^(2*temp_acZ)+1))) /)   ; convert back from z-space
  delete(temp_acZ)

  
  ;--------------------------------------------------------
  ; CMIP6 piControl autocorrelation box plot statistics
  ; (no need to do weighting for statistics, 1 piControl per model)
  ;--------------------------------------------------------

  temp_ac := esacr(tpi_lp_piControl,10)
  lag10_tpi_lp_piControl_B = boxplot_stat2(temp_ac(:,10))
  temp_acZ := avg(0.5*(log( (1.+temp_ac(:, 10)) / (1.-temp_ac(:, 10)))))   ; convert to Z-space
  lag10_tpi_lp_piControl_B(2) = (/ ((2.71828^(2*temp_acZ)-1)/ ((2.71828^(2*temp_acZ)+1))) /)   ; convert back from z-space
  delete(temp_acZ)

  temp_ac := esacr(tpi_piControl,10)
  lag1_tpi_piControl_B = boxplot_stat2(temp_ac(:,1))
  temp_acZ := avg(0.5*(log( (1.+temp_ac(:, 1)) / (1.-temp_ac(:, 1)))))   ; convert to Z-space
  lag1_tpi_piControl_B(2) = (/ ((2.71828^(2*temp_acZ)-1)/ ((2.71828^(2*temp_acZ)+1))) /)   ; convert back from z-space
  delete(temp_acZ)

  ;--------------------------------------------------------
  ; CMIP6 piControl / historical and observational standard
  ; deviation box plot statistics
  ;--------------------------------------------------------

  stddev_tpi_lp_obs         = dim_stddev(tpi_lp_obs)
  temp_sd := dim_stddev(tpi_lp_hist)
  stddev_tpi_lp_hist_B = (/weighted_percentile(temp_sd,weights,.05),weighted_percentile(temp_sd,weights,.25),weighted_mean(temp_sd,weights),weighted_percentile(temp_sd,weights,.75),weighted_percentile(temp_sd,weights,.95)/)
  stddev_tpi_lp_piControl_B = boxplot_stat2(dim_stddev(tpi_lp_piControl))
  stddev_tpi_lp_piControl_B(2) = avg(dim_stddev(tpi_lp_piControl))
  
  stddev_tpi_obs            = dim_stddev(tpi_obs)
  temp_sd := dim_stddev(tpi_hist)
  stddev_tpi_hist_B         = (/weighted_percentile(temp_sd,weights,.05),weighted_percentile(temp_sd,weights,.25),weighted_mean(temp_sd,weights),weighted_percentile(temp_sd,weights,.75),weighted_percentile(temp_sd,weights,.95)/)
  stddev_tpi_piControl_B    = boxplot_stat2(dim_stddev(tpi_piControl))
  stddev_tpi_piControl_B(2) = avg(dim_stddev(tpi_piControl))


  
  var_tpi_obs = dim_variance(tpi_obs)
  dof_obs := dim_num(.not.ismissing(tpi_obs))
  var_tpi_hist := variance(tpi_hist)
  dof_hist := num(.not.ismissing(tpi_hist))
  do ii = 0, dimsizes(var_tpi_obs)-1
    fsig = ftest(var_tpi_obs(ii), dof_obs(ii), var_tpi_hist, dof_hist, 0)
    print("CMIP6 historical, unfiltered, "+obs_datasets(ii)+": p = "+fsig)
  end do
  
  var_tpi_obs = dim_variance(tpi_lp_obs)
  dof_obs := dim_num(.not.ismissing(tpi_lp_obs))/10.
  var_tpi_hist := variance(tpi_lp_hist)
  dof_hist := num(.not.ismissing(tpi_lp_hist))/10.
  do ii = 0, dimsizes(var_tpi_obs)-1
    fsig = ftest(var_tpi_obs(ii), dof_obs(ii), var_tpi_hist, dof_hist, 0)
    print("CMIP6 historical, filtered,   "+obs_datasets(ii)+": p = "+fsig)
  end do

  
  ;--------------------------------------------------------
  ; CMIP5 historical autocorrelation box plot statistics
  ;--------------------------------------------------------

  temp_ac := esacr(tpi_lp_hist_C5,10)
  lag10_tpi_lp_hist_B_C5 = (/weighted_percentile(temp_ac(:,10),weights_C5,.05),weighted_percentile(temp_ac(:,10),weights_C5,.25),weighted_mean(temp_ac(:,10),weights_C5),weighted_percentile(temp_ac(:,10),weights_C5,.75),weighted_percentile(temp_ac(:,10),weights_C5,.95)/)

  temp_acZ := weighted_mean(0.5*(log( (1.+temp_ac(:, 10)) / (1.-temp_ac(:, 10)))), weights_C5)   ; convert to Z-space
  lag10_tpi_lp_hist_B_C5(2) = (/ ((2.71828^(2*temp_acZ)-1)/ ((2.71828^(2*temp_acZ)+1))) /)   ; convert back from z-space
  delete(temp_acZ)
  
  temp_ac := esacr(tpi_hist_C5,10)
  lag1_tpi_hist_B_C5 = (/weighted_percentile(temp_ac(:,1),weights_C5,.05),weighted_percentile(temp_ac(:,1),weights_C5,.25),weighted_mean(temp_ac(:,1),weights_C5),weighted_percentile(temp_ac(:,1),weights_C5,.75),weighted_percentile(temp_ac(:,1),weights_C5,.95)/)

  temp_acZ := weighted_mean(0.5*(log( (1.+temp_ac(:, 1)) / (1.-temp_ac(:, 1)))), weights_C5)   ; convert to Z-space
  lag1_tpi_hist_B_C5(2) = (/ ((2.71828^(2*temp_acZ)-1)/ ((2.71828^(2*temp_acZ)+1))) /)   ; convert back from z-space
  delete(temp_acZ)
  
  ;--------------------------------------------------------
  ; CMIP5 piControl autocorrelation box plot statistics
  ;--------------------------------------------------------

  temp_ac := esacr(tpi_lp_piControl_C5,10)
  lag10_tpi_lp_piControl_B_C5 = boxplot_stat2(temp_ac(:,10))
  temp_acZ := avg(0.5*(log( (1.+temp_ac(:, 10)) / (1.-temp_ac(:, 10)))))   ; convert to Z-space
  lag10_tpi_lp_piControl_B_C5(2) = (/ ((2.71828^(2*temp_acZ)-1)/ ((2.71828^(2*temp_acZ)+1))) /)   ; convert back from z-space
  delete(temp_acZ)
  
  temp_ac := esacr(tpi_piControl_C5,10)
  lag1_tpi_piControl_B_C5 = boxplot_stat2(temp_ac(:,1))
  temp_acZ := avg(0.5*(log( (1.+temp_ac(:, 1)) / (1.-temp_ac(:, 1)))))   ; convert to Z-space
  lag1_tpi_piControl_B_C5(2) = (/ ((2.71828^(2*temp_acZ)-1)/ ((2.71828^(2*temp_acZ)+1))) /)   ; convert back from z-space
  delete(temp_acZ)

  ;--------------------------------------------------------
  ; CMIP5 piControl / historical and observational standard
  ; deviation box plot statistics
  ;--------------------------------------------------------

  temp_sd := dim_stddev(tpi_lp_hist_C5)
  stddev_tpi_lp_hist_B_C5 = (/weighted_percentile(temp_sd,weights_C5,.05),weighted_percentile(temp_sd,weights_C5,.25),weighted_mean(temp_sd,weights_C5),weighted_percentile(temp_sd,weights_C5,.75),weighted_percentile(temp_sd,weights_C5,.95)/)
  stddev_tpi_lp_piControl_B_C5 =  boxplot_stat2(dim_stddev(tpi_lp_piControl_C5))
  stddev_tpi_lp_piControl_B_C5(2) =  avg(dim_stddev(tpi_lp_piControl_C5))

  temp_sd := dim_stddev(tpi_hist_C5)
  stddev_tpi_hist_B_C5 =(/weighted_percentile(temp_sd,weights_C5,.05),weighted_percentile(temp_sd,weights_C5,.25),weighted_mean(temp_sd,weights_C5),weighted_percentile(temp_sd,weights_C5,.75),weighted_percentile(temp_sd,weights_C5,.95)/)
  stddev_tpi_piControl_B_C5 =  boxplot_stat2(dim_stddev(tpi_piControl_C5))
  stddev_tpi_piControl_B_C5(2) =  avg(dim_stddev(tpi_piControl_C5))


  
  var_tpi_obs = dim_variance(tpi_obs)
  dof_obs := dim_num(.not.ismissing(tpi_obs))
  var_tpi_hist := variance(tpi_hist_C5)
  dof_hist := num(.not.ismissing(tpi_hist_C5))
    print("CMIP5 historical, unfiltered : var = "+var_tpi_hist)
  do ii = 0, dimsizes(var_tpi_obs)-1
    fsig = ftest(var_tpi_obs(ii), dof_obs(ii), var_tpi_hist, dof_hist, 0)
    print("CMIP5 historical, unfiltered, "+obs_datasets(ii)+": var = "+var_tpi_obs(ii)+" p = "+fsig)
  end do

  var_tpi_obs = dim_variance(tpi_lp_obs)
  dof_obs := dim_num(.not.ismissing(tpi_lp_obs))/10.
  var_tpi_hist := variance(tpi_lp_hist_C5)
  dof_hist := num(.not.ismissing(tpi_lp_hist_C5))/10.
  print("CMIP5 historical, filtered   : var = "+var_tpi_hist)
  do ii = 0, dimsizes(var_tpi_obs)-1
    fsig = ftest(var_tpi_obs(ii), dof_obs(ii), var_tpi_hist, dof_hist, 0)
    print("CMIP5 historical, filtered,   "+obs_datasets(ii)+": var = "+var_tpi_obs(ii)+" p = "+fsig)
  end do


  
  ;--------------------------------------------------------
  ; Preparation for box plots
  ;--------------------------------------------------------

  yval_ac = new((/8,5/),float)
  yval_ac(0,:) = (/ lag1_tpi_piControl_B_C5 /)
  yval_ac(1,:) = (/ lag1_tpi_piControl_B /)
  yval_ac(2,:) = (/ lag1_tpi_hist_B_C5 /)
  yval_ac(3,:) = (/ lag1_tpi_hist_B /)

  yval_ac(4,:) = (/ lag10_tpi_lp_piControl_B_C5 /)
  yval_ac(5,:) = (/ lag10_tpi_lp_piControl_B /)
  yval_ac(6,:) = (/ lag10_tpi_lp_hist_B_C5 /)
  yval_ac(7,:) = (/ lag10_tpi_lp_hist_B /)

  yval_sd = new((/8,5/),float)
  yval_sd(0,:) = (/ stddev_tpi_piControl_B_C5 /)
  yval_sd(1,:) = (/ stddev_tpi_piControl_B /)
  yval_sd(2,:) = (/ stddev_tpi_hist_B_C5 /)
  yval_sd(3,:) = (/ stddev_tpi_hist_B /)

  yval_sd(4,:) = (/ stddev_tpi_lp_piControl_B_C5 /)
  yval_sd(5,:) = (/ stddev_tpi_lp_piControl_B /)
  yval_sd(6,:) = (/ stddev_tpi_lp_hist_B_C5 /)
  yval_sd(7,:) = (/ stddev_tpi_lp_hist_B /)


  ;--------------------------------------------------------
  ; Preparing timeseries plot
  ;--------------------------------------------------------

  nmod_crit = 1  ; 1 or more models needed to count for box_whisker_2  (was at one time = 3)
  
  hist_range = tpi_lp_hist(:1,:)
  hist_range = hist_range@_FillValue
  mme_range = tpi_lp_hist(:2,:)
  mme_range = mme_range@_FillValue
  do gg = 5,dimsizes(tpi_lp_hist&time)-6
     hist_range(0,gg) = (/weighted_percentile(tpi_lp_hist(:,gg),weights,.95) /)  ; individual members 5%
     hist_range(1,gg) = (/weighted_percentile(tpi_lp_hist(:,gg),weights,.05) /)   ; individual members 95%

     temparr := box_whisker_2(tpi_lp_hist_em(:,gg),nmod_crit)
     mme_range(0,gg) = (/ temparr(2) /)    ; ensemble mme median with nmod_crit applied (and thus different from mme_tpi_lp_hist which has no criteria and is a mean)
     mme_range(1,gg) = (/ temparr(1) /)    ; ensemble 5%
     mme_range(2,gg) = (/ temparr(3) /)    ; ensemble 95%
  end do

  hist_range_C5 = tpi_lp_hist_C5(:1,:)
  hist_range_C5 = hist_range_C5@_FillValue
  do gg = 5,dimsizes(tpi_lp_hist_C5&time)-6
     hist_range_C5(0,gg) = (/weighted_percentile(tpi_lp_hist_C5(:,gg),weights_C5,.95) /)
     hist_range_C5(1,gg) = (/weighted_percentile(tpi_lp_hist_C5(:,gg),weights_C5,.05) /)  
  end do

  timeseries = new((/8,dimsizes(tpi_hist&time)/),float)
;  timeseries(:2,:) = (/ tpi_lp_obs(:2,:) /)
  timeseries(:dim_obs-1,:) = (/ tpi_lp_obs(:dim_obs-1,:) /)
  
  timeseries(3,:dimsizes(mme_tpi_lp_hist_C5)-6) = (/ mme_tpi_lp_hist_C5(:dimsizes(mme_tpi_lp_hist_C5)-6) /)
  timeseries(4,:) = (/ mme_tpi_lp_hist /)

  timeseries(:,:4) = timeseries@_FillValue
  timeseries(:,dimsizes(tpi_hist&time)-5:) = timeseries@_FillValue

  timeseries(5,:) = 0.

  timeseries(6:,:) = (/ mme_range(1:,:) /)    ; mme_tpi_lp_hist 5% / 95%
  delete(temparr)
  
;-----------------------------------------------------
  cor = escorc(dim_avg_n(timeseries(:2,:),0),timeseries(4,:))
  corval = decimalPlaces(cor,2,True)   ;+"%"
;  prob = rtest(cor,dimsizes(mme_tpi_lp_hist),0)

;  sx = equiv_sample_size(dim_avg_n(timeseries(:2,:),0), 0.05, 0)
;  sy = equiv_sample_size(timeseries(4,:), 0.05, 0)

;  prob = ftest(dim_variance(dim_avg_n(timeseries(:2,:),0)), dimsizes(timeseries(0,:)), dim_variance(timeseries(4,:)), dimsizes(timeseries(4,:)), 0)

;  corR = regline_stats(dim_avg_n(timeseries(:2,:),0),timeseries(4,:))
;  print(corR)
;  if (prob.lt.0.05) then
;     sig = "**"
;  else if (prob.lt.0.10) then
;     sig = "*"
;  end if

;  print(cor + " "+prob)
  ;------------------------------------------------------------
; repeat for CMIP5
  dimZ = dimsizes(mme_tpi_lp_hist_C5)
  corC5 = escorc(dim_avg_n(timeseries(:2,:dimZ-1),0),mme_tpi_lp_hist_C5)
  probC5 = rtest(corC5,dimsizes(timeseries(0,:)),0)
  prob2C5 = ftest(dim_variance(dim_avg_n(timeseries(:2,:dimZ-1),0)), dimsizes(timeseries(0,:dimZ-1))/10, dim_variance(mme_tpi_lp_hist_C5), dimsizes(mme_tpi_lp_hist_C5)/10, 0)
;  print(corC5+" "+probC5+" "+prob2C5)
  corR_C5 = regline_stats(dim_avg_n(timeseries(:2,:dimZ-1),0),mme_tpi_lp_hist_C5)
;  print(corR)
  corval_C5 = decimalPlaces(corC5,2,True)   
;  print(corval_C5 + " "+probC5+" "+snval_C5)
  
  ;--------------------------------------------------------
  ;Plotting
  ;--------------------------------------------------------

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
  res@mpCenterLonF = 210.
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
  res@gsnAddCyclic = True
  
  res@cnLevels = fspan(-.4,.4,11)
  cmap = read_colormap_file("$diag_scripts/shared/plot/rgb/ipcc-ar6_temperature_div_12.rgb")
  res@cnFillColors = cmap

  res2 = True
  res2@gsnDraw  = False
  res2@gsnFrame = False
  res2@cnLinesOn = False
  res2@cnLineLabelsOn = False
  res2@cnLevelSelectionMode = "ExplicitLevels"
  res2@cnFillOn = True
  res2@cnMonoFillPattern = False
  res2@gsnAddCyclic = True
  res2@cnFillColors = (/"gray25","transparent"/)
  res2@lbLabelBarOn = False
  res2@cnInfoLabelOn = False
  res2@gsnLeftString = ""
  res2@gsnRightString = ""
  
  plot = new(2,graphic)
  res@gsnCenterString = "(a) Observations"
  plot(0) = gsn_csm_contour_map(wks,ipo_obs,res)
;  res@gsnCenterString = "(b) CMIP6 Hist MME"
  res@gsnCenterString = "(b) CMIP6 Hist MME mean"
  plot(1) = gsn_csm_contour_map(wks,mme_ipo,res)

;  res2@cnFillScaleF        = 1. ; 0.4
;  res2@cnFillPatterns = (/17,0/)
;  res2@cnLevels = 0.5
;  oplot = gsn_csm_contour(wks,sig(0,:,:),res2)
;  overlay(plot(0),oplot)
  res_marker = True
  res_marker@gsMarkerIndex = 5
  res_marker@gsMarkerSizeF = 0.0014
  res_marker@gsMarkerThicknessF = 0.5
  res_marker@gsMarkerOpacityF = 0.6
  xdummy = new(product(dimsizes(sig(0, :, :))), graphic)
  idum = 0
  do j = 0, dimsizes(sig&lat)-1, 2
    if (mod(j,4).eq.0) then
      i0 = 0
    else
      i0 = 1
    end if
    do i = i0, dimsizes(sig&lon)-1, 2
      if (.not.ismissing(sig(0, j, i)).and.sig(0, j, i).lt.0.5) then
        xdummy(idum) = gsn_add_polymarker(wks, plot(0), sig&lon(i), \
                                          sig&lat(j), res_marker)
        idum = idum + 1
      end if
    end do
  end do
  
  res2@cnFillPatterns = (/3,0/)
  res2@cnLevels = 0.5
  res2@cnFillScaleF        = 0.8  ; 0.7
  oplot1 = gsn_csm_contour(wks,mme_ipo_sign,res2)
  overlay(plot(1),oplot1)
  
  panres = True
  panres@gsnPanelBottom = 0.41
  panres@gsnPanelRight = 0.7
  panres@gsnPanelLabelBar = True
  panres@gsnPanelYWhiteSpacePercent = 3.0
  panres@pmLabelBarHeightF = 0.05
  panres@pmLabelBarWidthF = 0.38
  panres@lbBoxLineColor = "gray70"
  panres@lbLabelFontHeightF = 0.011
  panres@lbBoxEndCapStyle = "TriangleBothEnds"
  panres@txFontHeightF = 0.022
  panres@gsnFrame = False
  panres@pmLabelBarOrthogonalPosF = 0.017
  panres@lbTitleOn = True
  panres@lbTitleString = "(~S~o~N~C)"
  panres@lbTitlePosition  = "Top"              ; title position
  panres@lbTitleFontHeightF= .010                ; make title smaller
  gsn_panel(wks,plot,(/1,2/),panres)


  ;--------------------------------------------------------
  ;Plotting taylor diagram, panel 3
  ;--------------------------------------------------------
  cmap := new((/12,3/),float)
  cmap(0,:) =  (/255,255,255/)
  cmap(1,:) = (/0,0,0/)
  cmap(2,:) = (/253, 141, 60/)   ; dark orange, misc_div_21  amip
  cmap(3,:) = (/211,15,31/)      ; dark red, misc_div_21   historical
  cmap(4,:) = (/125, 79, 20/)    ; brown, prec_div (continuous)  hist-GHG
  cmap(5,:) = (/131,183,214/)    ; dark cyan, temp_div (continuous)  hist-aer
  cmap(6,:) = (/153,112,171/)    ; purple, chem_div_21  hist-stratO3
  cmap(7,:) = (/60, 132, 26/)    ; green, wind_seq (continuous)  hist-nat
  cmap(8,:) = (/25,51,178/)      ; dark blue, misc_seq_1_21 piControl
  cmap(9,:) = (/252,87,44/)      ; orange-red, misc_div (continuous) 
  cmap(10,:) = (/127,127,127/)
  cmap(11,:) = (/76,76,76/)
  cmap = cmap/255.
  gsn_define_colormap(wks,cmap)
  
  tres                 = True                     ; diagram mods desired
  colors = new(dimsizes(tpi_hist&nens),integer)
  colors = 3
  tres@Colors          = colors       ; marker colors
  markers = ensemble_assign
  markers = 2
  tres@Markers         = markers  ;ensemble_assign+1        ; marker styles
  tres@markerTxYOffset = 0.04                     ; offset btwn marker & label
  tres@gsMarkerSizeF   = 0.006                    ; marker size
  tres@txFontHeightF   = 0.0225                    ; text size
  tres@stnRad         = (/ 0.5, 1.5 /)     ; additional standard radii
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
  tres@txFontHeightF   = 0.028
  tplot = taylor_diagram(wks, ratio, patcor, tres)

  colors2 = new((/dimsizes(taylor_ipo_ratio_C5(:,0))/),integer)
  colors2 = 5
  tres@Colors          :=   colors2
  markers2 = new(dimsizes(taylor_ipo_ratio_C5(:,0)),integer)
  markers2 = 2
  tres@Markers := markers2
  tres@tfPolyDrawOrder = "PreDraw"
  tplot_C5 = taylor_diagram(wks, taylor_ipo_ratio_C5, taylor_ipo_patcor_C5, tres)
  overlay(tplot_C5,tplot)

  colors := new(dimsizes(obs_datasets),string)
  marker = new(dimsizes(obs_datasets),integer)
  colors = "black"
  marker = 1
  tres@Colors          := colors       ; marker colors
  tres@Markers         := marker
  tres@gsMarkerSizeF   = 0.028
  more_ipo_obs_patcor = tay_stats_obs(:,0:0)
  more_ipo_obs_ratio  = tay_stats_obs(:,1:1)
  print(more_ipo_obs_patcor+" "+more_ipo_obs_ratio)
  tplot2 = taylor_diagram(wks, more_ipo_obs_ratio, more_ipo_obs_patcor, tres)

  tres@gsMarkerSizeF   = 0.008
  tres@Colors := 8
  tres@Markers:= 16
  tplot3b = taylor_diagram(wks, tay_stats_hist_cmip5EM(0:0,1:1),tay_stats_hist_cmip5EM(0:0,0:0), tres)
  overlay(tplot_C5,tplot3b)

  tres@gsMarkerThicknessF = 2.0
  tres@Colors := 9
  tres@Markers:= 16
  tplot4b = taylor_diagram(wks, tay_stats_hist_cmip6EM(0:0,1:1),tay_stats_hist_cmip6EM(0:0,0:0), tres)
  overlay(tplot_C5,tplot4b)

  overlay(tplot_C5,tplot2)
  panres@gsnPanelLabelBar = False
  panres@gsnPanelLeft = 0.665
  panres@gsnPanelRight = 0.995
  panres@gsnPanelTop = 0.81
  panres@gsnPanelBottom = 0.60
  gsn_panel(wks,tplot_C5,(/1,1/),panres)


  ;--------------------------------------------------------
  ;Plotting box plots, panel 4 and 5
  ;--------------------------------------------------------

  bres            = True                         ; plot mods desired
  bres@vpWidthF = 0.6
  bres@vpHeightF = 0.45
  bres@trXMinF = 0.
  bres@trXMaxF = 5.
  bres@trYMinF = -1.
  bres@trYMaxF = 1.
  bres@tmXBLabels = (/"Control","Control","Control","Control","Historical","Historical","Historical","Historical"/) ; labels for each box
  bres@tmXBOn = False
  bres@tmYLLabelFontHeightF = 0.03 ;0.024
  bres@tmYLMode = "Explicit"
  bres@tmYLValues = (/-1,-0.5,0,0.5,1/)
  bres@tmYLLabels = bres@tmYLValues
  bres@tmYLMinorValues = fspan(-1,1,21)
  bres@tiMainFont = "Helvetica"
  bres@tiMainString = ""  ;"Autocorrelation"
  bres@tiYAxisString = "" ; "( )"
  bres@tiYAxisFontHeightF = 0.035

  llres                   = True
  llres@gsLineThicknessF  = 1.5                 ; line thickness

  opti          = True
  opti@boxWidth = .3				; Width of box (x units)
  opti@boxColors = (/5,3,5,3,5,3,5,3/)  

  txres = True
  txres@txFontHeightF = 0.028 ;0.02

  plres = True
  plres@gsLineColor = 1
  plres@gsLineThicknessF = .75
  plres@gsLineDashPattern = 5

  polyres = True
  polyres@gsFillColor = (/0.75,0.75,0.75,1/)   ;.3/)

;- - - - - - - - - - - - - - - - - - - - - - - -
; Draw blank box plots containing gray boxes representing observational spread, panels 4-5
;   - Blank plots were drawn due to draw order issues when trying to plot boxplots and the gray boxes.
;
  plot_blank = new(2,graphic)
  bres@tiYAxisFontColor = "white"
  bres@tmYLLabelFontColor = "white"
  plot_blank(0) = blank_boxplot(wks,bres)
  bres@tiYAxisString = "(~S~o~N~C)"
  bres@trYMinF = -0.3
  bres@trYMaxF = 1.2
  bres@tmYLValues := fspan(0,1.5,6)
  bres@tmYLLabels := bres@tmYLValues
  bres@tmYLMinorValues := fspan(0,1.5,16)
  plot_blank(1) = blank_boxplot(wks,bres)

  print("Lag1 PDV Obs: "+lag1_tpi_obs)
  print("Lag10 PDV-LP Obs: "+lag10_tpi_lp_obs)
  print("Standard deviation PDV Obs: "+ stddev_tpi_obs)
  print("Standard deviation PDV-LP Obs: "+ stddev_tpi_lp_obs)

  plres3 = True
  plres3@gsLineThicknessF = 1.5
  plres3@gsLineColor = "black"
  dpl = (/0,2,3/)
  dumA = new((/4,3/),graphic)
;  do gg = 0,2
  do gg = 0,dim_obs-1
     plres3@gsLineDashPattern = dpl(gg)
     dumA(0,gg) = gsn_add_polyline(wks,plot_blank(0),(/0.25,2.25/),(/lag1_tpi_obs(gg),lag1_tpi_obs(gg)/),plres3)
     dumA(1,gg) = gsn_add_polyline(wks,plot_blank(0),(/2.75,4.75/),(/lag10_tpi_lp_obs(gg),lag10_tpi_lp_obs(gg)/),plres3)
     dumA(2,gg) = gsn_add_polyline(wks,plot_blank(1),(/0.25,2.25/),(/stddev_tpi_obs(gg),stddev_tpi_obs(gg)/),plres3)
     dumA(3,gg) = gsn_add_polyline(wks,plot_blank(1),(/2.75,4.75/),(/stddev_tpi_lp_obs(gg),stddev_tpi_lp_obs(gg)/),plres3)
  end do

  panres@gsnPanelTop = 0.595
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
  bres@tiYAxisString = "" ;"( )"
  bres@trYMinF = -1.
  bres@trYMaxF = 1.
  bres@tmYLValues := (/-1,-0.5,0,0.5,1/)
  bres@tmYLLabels := bres@tmYLValues
  bres@tmYLMinorValues := fspan(-1,1,21)

  bplot = new(2,graphic)
  opti@boxFill_Colors = (/-1,-1,5,3,-1,-1,5,3/)
  opti@boxFill_midColor = (/5,3,0,0,5,3,0,0/)

  bplot(0) = boxplot2(wks,(/0.5,1,1.5,2, 3,3.5,4,4.5/),yval_ac,opti,bres,llres)

  dum = new(7,graphic)
  dum2 = dum
;  dum(0) = gsn_add_text(wks,bplot(0),"piControl",.75,bres@trYMinF+0.15,txres)
;  dum(1) = gsn_add_text(wks,bplot(0),"Historical",1.75,bres@trYMinF+0.155,txres)
;  dum(2) = gsn_add_text(wks,bplot(0),"piControl",3.25,bres@trYMinF+0.15,txres)
;  dum(3) = gsn_add_text(wks,bplot(0),"Historical",4.25,bres@trYMinF+0.155,txres)
  dum(4) = gsn_add_polyline(wks,bplot(0),(/-5,5/),(/0.,0./),plres)
  dum(5) = gsn_add_text(wks,bplot(0),"annual",1.25,bres@trYMinF+0.1525,txres)
  dum(6) = gsn_add_text(wks,bplot(0),"10-yr low pass",3.75,bres@trYMinF+0.1525,txres)

  bres@tiYAxisString = "(~S~o~N~C)"
  bres@trYMinF = -0.3
  bres@trYMaxF = 1.2
  bres@tmYLValues := fspan(0,1.5,6)
  bres@tmYLLabels := bres@tmYLValues
  bres@tmYLMinorValues := fspan(0,1.5,16)
  bres@tiMainString = ""    ;Standard Deviation (~S~o~N~C)"
  bplot(1) = boxplot2(wks,(/0.5,1,1.5,2,3,3.5,4,4.5/),yval_sd,opti,bres,llres)
;  dum2(0) = gsn_add_text(wks,bplot(1),"piControl",.75,bres@trYMinF+0.15,txres)
;  dum2(1) = gsn_add_text(wks,bplot(1),"Historical",1.75,bres@trYMinF+0.155,txres)
;  dum2(2) = gsn_add_text(wks,bplot(1),"piControl",3.25,bres@trYMinF+0.15,txres)
;  dum2(3) = gsn_add_text(wks,bplot(1),"Historical",4.25,bres@trYMinF+0.155,txres)
  dum2(4) = gsn_add_polyline(wks,bplot(1),(/-5,5/),(/0.,0./),plres)
  dum2(5) = gsn_add_text(wks,bplot(1),"annual",1.25,bres@trYMinF+0.1525,txres)
  dum2(6) = gsn_add_text(wks,bplot(1),"10-yr low pass",3.75,bres@trYMinF+0.1525,txres)
  gsn_panel(wks,bplot,(/1,2/),panres)

;- - - - - - - - - - - - - - - 
; set up box plot legend
;   
  bres@trYMinF = -1.   
  bres@trYMaxF = 1.
  bres@trXMinF = 0.
  bres@trXMaxF = 2.0
  bres@tmYLOn = False
  bres@tmXBOn = False
  bres@tmYLBorderOn = False
  bres@tmXBBorderOn = False
  bres@vpXF = 0.18 ;.23
  bres@vpYF = .435
  bres@vpWidthF = .06
  bres@vpHeightF = .075
  bres@tiYAxisOn = False
  bres@tmXBLabels := (/"Historical","piControl"/)
  opti@boxFill_Colors := (/-1,1/)  ; (/-1, 3/)
  opti@boxFill_midColor := (/1,0/) ; (/5,0/)
  opti@boxColors := (/1,1/)        ; (/5,3/)  

  yval_lg = new((/2,5/),"float")
  yval_lg(0,:) = (/-.5,-.25,0,.25,.5/)
  yval_lg(1,:) = yval_lg(0,:)

  bplg = boxplot2(wks,(/0.5,1.5/),yval_lg,opti,bres,llres)
  draw(bplg)

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
  mres@trYMinF = -.8
  mres@trYMaxF = .8

  mres@gsnCenterString = ""
  mres@gsnLeftString = ""
  mres@gsnRightString = ""
  mres@xyMonoDashPattern = True

  mres2 = mres
  mres@xyMonoLineColor    = False             ; want colored lines
  mres@xyLineColors       = (/1,1,1,5,3,1,3,3/) ; colors chosen, last line = line at Y = 0
  mres@xyLineThicknesses	 = (/1.5,1.5,1.5,2.5,2.5,.5,1.,1./)
  mres@xyMonoDashPattern = False
  mres@xyDashPatterns = (/0,2,3,0,0,1,11,11/)

;  mres2@gsnXYFillColors = (/131,183,214./) /255.    ; dark cyan, temp_div (continuous)  hist-aer
  mres2@gsnXYFillColors := (/131./255.,183./255.,214./255.,0.6/)   ; dark cyan, temp_div (continuous)  hist-aer
  mres2@xyLineColor     = -1
  mres2@tiYAxisString = "(~S~o~N~C)"
  mres2@tiYAxisFontHeightF = 0.02
  mres2@tiYAxisOffsetYF = -0.005

  bot_plot = gsn_csm_xy(wks,ispan(syearA,2005,1),hist_range_C5,mres2)
  delete(mres2@gsnXYFillColors)
  mres2@gsnXYFillColors = (/.7,0,0,.3/)  ;"LightPink"
  mid_plot = gsn_csm_xy(wks,ispan(syearA,eyearA,1),hist_range,mres2)
  tplot = gsn_csm_xy(wks,ispan(syearA,eyearA,1),timeseries,mres)
  overlay(bot_plot,mid_plot)
  overlay(bot_plot,tplot)

  panres@gsnPanelTop = 0.57
  panres@gsnPanelBottom = 0.4
  panres@gsnPanelLeft = 0.37
  delete(panres@gsnPanelXWhiteSpacePercent)
  delete(panres@gsnPanelRight)
  gsn_panel(wks,bot_plot,(/1,1/),panres)
  ;--------------------------------------------------------
  ; Set titles for panels 3-6
  ;--------------------------------------------------------
  textres = True
  textres@txFontHeightF = 0.012
  gsn_text_ndc(wks,"(c) Taylor Statistics",0.84,0.822,textres)
  gsn_text_ndc(wks,"(d) Autocorrelation",0.125,0.56,textres)
  gsn_text_ndc(wks,"(e) Standard Deviation",0.355,0.56,textres)
  gsn_text_ndc(wks,"(f)",0.55,0.575,textres)

  ;--------------------------------------------------------
  ; Boxplot legends
  ;--------------------------------------------------------

  gres = True
  gres@YPosPercent = 40.25    ; expressed as %, 0->100, sets position of top border of legend
  gres@LineLabelWhiteSpacePercent = .5
  gres@ItemSpacePercent = .9

  lineres = True
  lineres@lgLineColors = 5  ; line colors
  lineres@lgLineThicknesses = 1.5                        ; line thicknesses
  lineres@LineLengthPercent = 2.5                        ; expressed as %, 0->100, length of line
  lineres@lgDashIndexes = 0

  x1 = 0.08
  y1 = 0.403
  lres = True
  lres@gsLineThicknessF = 1.5
  lres@gsLineDashSegLenF = 0.075
  txres = True
  txres@txFontHeightF = 0.007
  txres@txJust = "CenterLeft"
  do gg = 0, dimsizes(obs_datasets)-1
    lres@gsLineDashPattern = dpl(gg)
    gsn_polyline_ndc(wks, (/x1,x1+0.03/),(/y1-gg*0.0125,y1-gg*0.0125/), lres)
    gsn_text_ndc(wks, obs_datasets(gg), x1+0.035, y1-gg*0.0125, txres)
  end do
  ;--------------------------------------------------------
  ; Timeseries legend
  ;--------------------------------------------------------

  x1 = 0.52
  y1 = 0.393
  lres = True
  lres@gsLineThicknessF = 1.5
  lres@gsLineDashSegLenF = 0.075
  txres = True
  txres@txFontHeightF = 0.007
  txres@txJust = "CenterLeft"
  do gg = 0, dimsizes(obs_datasets)-1
    lres@gsLineDashPattern = dpl(gg)
    gsn_polyline_ndc(wks, (/x1,x1+0.03/),(/y1-gg*0.0125,y1-gg*0.0125/), lres)
    gsn_text_ndc(wks, obs_datasets(gg), x1+0.035, y1-gg*0.0125, txres)
  end do

  x1 = 0.62
  lres@gsLineDashPattern = 0
  lres@gsLineColor = 5
  lres@gsLineThicknessF = 2.5
  gsn_polyline_ndc(wks, (/x1,x1+0.03/),(/y1,y1/), lres)
;  gsn_text_ndc(wks, "CMIP5 Hist MME", x1+0.035, y1, txres)
  gsn_text_ndc(wks, "CMIP5 Hist MME mean", x1+0.035, y1, txres)

  
  lres = True   ; draw box showing observational range
  lres@gsFillColor := 5
  lres@gsFillOpacityF = 0.6
  gsn_polygon_ndc(wks,(/x1,x1+0.03,x1+0.03,x1,x1/),(/y1-0.01,y1-0.01,y1-0.02,y1-0.02,y1-0.01/),lres)
  gsn_polygon_ndc(wks,(/x1,x1+0.03,x1+0.03,x1,x1/),(/y1-0.01,y1-0.01,y1-0.02,y1-0.02,y1-0.01/),lres)
  gsn_text_ndc(wks, "CMIP5 Hist 5-95th range", x1+0.035, y1-0.016, txres)

  x1 = 0.78
  lres@gsLineColor = 3
  gsn_polyline_ndc(wks, (/x1,x1+0.03/),(/y1,y1/), lres)
;  gsn_text_ndc(wks, "CMIP6 Hist MME", x1+0.035, y1, txres)
  gsn_text_ndc(wks, "CMIP6 Hist MME mean", x1+0.035, y1, txres)
  lres@gsLineDashPattern = 5
  lres@gsLineThicknessF = 1.
  gsn_polyline_ndc(wks, (/x1,x1+0.03/),(/y1-0.0125,y1-0.0125/), lres)
  gsn_text_ndc(wks, "CMIP6 Hist 5-95% CI", x1+0.035, y1-0.013, txres)
  lres@gsFillColor = 3
  lres@gsFillOpacityF = 0.3
  gsn_polygon_ndc(wks,(/x1,x1+0.03,x1+0.03,x1,x1/),(/y1-0.02,y1-0.02,y1-0.03,y1-0.03,y1-0.02/),lres)
  gsn_polygon_ndc(wks,(/x1,x1+0.03,x1+0.03,x1,x1/),(/y1-0.02,y1-0.02,y1-0.03,y1-0.03,y1-0.02/),lres)
  gsn_text_ndc(wks, "CMIP6 Hist 5-95th range", x1+0.035, y1-0.026, txres)
  delete(lres@gsLineColor)
  delete(lres@gsLineThicknessF)

  ;--------------------------------------------------------
  ; Taylor legend
  ;--------------------------------------------------------

  mares = True
  mares@gsMarkerIndex = 1
  mares@gsMarkerSizeF   = 0.018
  tlres = True
  tlres@txFontHeightF = 0.0065 ;0.007;0.006
  tlres@txJust = "CenterLeft"

  mares@gsMarkerColor = "Black"
  gsn_polymarker_ndc(wks,.788,.808,mares)
  gsn_text_ndc(wks,"Observations",.793,.808,tlres)

  mares@gsMarkerIndex = 2
  mares@gsMarkerSizeF   = 0.006
  mares@gsMarkerColor = 5
  gsn_polymarker_ndc(wks,.858,.808,mares)
  gsn_text_ndc(wks,"CMIP5 Hist member",.863,.808,tlres)
  
  mares@gsMarkerColor = 3
  gsn_polymarker_ndc(wks,.858,.798,mares)
  gsn_text_ndc(wks,"CMIP6 Hist member",.863,.798,tlres)

  mares@gsMarkerIndex = 1
  mares@gsMarkerColor = 8
  mares@gsMarkerSizeF = 0.018
  gsn_polymarker_ndc(wks,.858,.788,mares)
;  gsn_text_ndc(wks,"CMIP5 Hist MME",.863,.788,tlres)
  gsn_text_ndc(wks,"CMIP5 Hist MME mean",.863,.788,tlres)
  mares@gsMarkerColor = 9
  gsn_polymarker_ndc(wks,.858,.778,mares)
;  gsn_text_ndc(wks,"CMIP6 Hist MME",.863,.778,tlres)
  gsn_text_ndc(wks,"CMIP6 Hist MME mean",.863,.778,tlres)

;- - - - - - - - - - - - - - - - - - - - - - - 
; Boxplot legend, additional text
;
  gsn_text_ndc(wks,"r="+corval+",  S/N="+snval,.825,.44,tlres)
  gsn_text_ndc(wks,"piControl  Historical",0.174,0.365,tlres)
  gsn_text_ndc(wks,"- 95%",.225,.417,tlres)
  gsn_text_ndc(wks,"- 75%",.235,.408,tlres)
  gsn_text_ndc(wks,"- Mean",.235,.398,tlres)
  gsn_text_ndc(wks,"- 25%",.235,.388,tlres)
  gsn_text_ndc(wks,"- 5%",.225,.379,tlres)

  tlres@txFontColor = 5
  gsn_text_ndc(wks,"CMIP5 (Hist, "+count_unique_values(ea_c5)+" models, "+num(.not.ismissing(dim_stddev(tpi_lp_hist_C5)))+" runs)",0.28,.412,tlres)
  gsn_text_ndc(wks,"CMIP5 (piControl, "+num(.not.ismissing(dim_stddev(tpi_lp_piControl_C5)))+" models)",0.28,0.400,tlres)
  tlres@txFontColor = 3
  gsn_text_ndc(wks,"CMIP6 (Hist, "+count_unique_values(ensemble_assign)+" models, "+num(.not.ismissing(dim_stddev(tpi_lp_hist)))+" runs)",0.28,0.388,tlres)
  gsn_text_ndc(wks,"CMIP6 (piControl, "+num(.not.ismissing(dim_stddev(tpi_lp_piControl)))+" models)",0.28,0.376,tlres)
  delete(tlres@txFontColor)

  textres = True 
  textres@txFont = 22
  textres@txFontHeightF = 0.018
  gsn_text_ndc(wks,"Model Evaluation of the Pacific Decadal Variability (PDV - 1900-2014)",0.5,0.87,textres)

  textres@txFont = 21
  textres@txFontHeightF = 0.015
  textres@txFontColor = "gray40"
  gsn_text_ndc(wks,"Sea surface temperature spatial patterns",0.5,.84,textres)
  gsn_text_ndc(wks,"Persistence and Amplitude",0.25,0.58,textres)  
  gsn_text_ndc(wks,"Temporal Evolution",0.71,0.58,textres) 


  tlres@txFontHeightF = 0.008
  pmres0 = True
  pmres0@gsFillColor = "gray25"
  pmres0@gsFillScaleF = 0.8  ; 0.7
;  pmres0@gsFillIndex = 11
  pmres0@gsFillIndex = 3
  lres = True
  lres@gsLineDashPattern = 0
  lres@gsLineThicknessF = 1.
  x1 = 0.547
  x2 = 0.571
  y1 = 0.612
  y2 = 0.627
  gsn_polygon_ndc(wks, (/x1, x2, x2, x1/), (/y1, y1, y2, y2/), pmres0)
  gsn_polyline_ndc(wks, (/x1, x2, x2, x1, x1/), (/y1, y1, y2, y2, y1/), lres)
  gsn_text_ndc(wks," < 80% of runs~C~ agree on sign",x2+0.004,0.62,tlres)

;  pmres = pmres0
;  pmres@gsFillScaleF = 0.8  ; 0.55
;  pmres@gsFillIndex = 17
  x1 = 0.047
  x2 = 0.071
;  gsn_polygon_ndc(wks, (/x1, x2, x2, x1/), (/y1, y1, y2, y2/), pmres)
  gsn_polyline_ndc(wks, (/x1, x2, x2, x1, x1/), (/y1, y1, y2, y2, y1/), lres)
  do j = 0, 2
    if (any(j.eq.(/0,2/))) then
      do i = 0, 2
        gsn_polymarker_ndc(wks, x1+(i+0.5)*(x2-x1)/3., y1+(j+0.5)*(y2-y1)/3., res_marker)
      end do
    else
      do i = 0, 1
        gsn_polymarker_ndc(wks, x1+(i+1.)*(x2-x1)/3., y1+(j+0.5)*(y2-y1)/3., res_marker)
      end do
    end if
  end do
  gsn_text_ndc(wks," not significant~C~ at 10% level",x2+0.004,0.62,tlres)

  frame(wks)
  delete(wks)

end
