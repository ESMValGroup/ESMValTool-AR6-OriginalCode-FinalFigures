; #############################################################################
; Plot GSAT in piControl and obs and GSAT trend histogram
; Authors: Yu Kosaka (University of Tokyo, Japan) and Adam Phillips (NCAR, U.S.)
; #############################################################################
;
; Description
;
; Caveats
;
; Modification history
;   20210206-Yu Kosaka: changed from linear trend to period difference
;   20210202-Yu Kosaka: revised figure details
;   20210131-Yu Kosaka: adapt Adam's script to the ESMValTool
;
; #############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/plot/mder.ncl"

load "$diag_scripts/ipccwg1ar6ch3_modes/functions.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")

  var_model = "tas"
  var_obs = (/"tasa", "tasaga"/)

  ; models
  info_items = select_metadata_by_name(input_file_info, var_model)

  atts = True
  atts@project = "CMIP6"
  cmip6 = select_metadata_by_atts(info_items, atts)

  ; historical experiment
  atts@exp = "historical-ssp245"
  hist_cmip6 = select_metadata_by_atts(cmip6, atts)
  hist_cmip6_datasets = metadata_att_as_array(hist_cmip6, "dataset")
  hist_cmip6_ensembles = metadata_att_as_array(hist_cmip6, "ensemble")
;  dim_cmip6_hist = dimsizes(hist_cmip6_datasets)

  ; piControl experiment
  atts@exp = "piControl"
  piControl_cmip6 = select_metadata_by_atts(cmip6, atts)
  piControl_cmip6_datasets = metadata_att_as_array(piControl_cmip6, "dataset")
  piControl_cmip6_ensembles = metadata_att_as_array(piControl_cmip6, "ensemble")
;  dim_cmip6_piControl = dimsizes(piControl_cmip6_datasets)

  ; obs
  atts := True
  atts@project = "OBS"
  obs = select_metadata_by_atts(input_file_info, atts)
  atts@project = "OBS6"
  obs_tmp = select_metadata_by_atts(input_file_info, atts)
  do iobs = 0, ListCount(obs_tmp) - 1
    ListAppend(obs, obs_tmp[iobs])
  end do
  delete(obs_tmp)
  atts@project = "obs"
  obs_tmp = select_metadata_by_atts(input_file_info, atts)
  do iobs = 0, ListCount(obs_tmp) - 1
    ListAppend(obs, obs_tmp[iobs])
  end do
  delete(obs_tmp)
  atts@project = "obs4mips"
  obs_tmp = select_metadata_by_atts(input_file_info, atts)
  do iobs = 0, ListCount(obs_tmp) - 1
    ListAppend(obs, obs_tmp[iobs])
  end do
  delete(obs_tmp)
  atts@project = "ana4mips"
  obs_tmp = select_metadata_by_atts(input_file_info, atts)
  do iobs = 0, ListCount(obs_tmp) - 1
    ListAppend(obs, obs_tmp[iobs])
  end do
  delete(obs_tmp)
  if (ListCount(obs) .lt. 1) then
    error_msg("f", DIAG_SCRIPT, "", "this diagnostic needs at least one " + \
              "obs dataset")
  end if
  obs_datasets = metadata_att_as_array(obs,  "dataset")
;  dim_obs = dimsizes(obs_datasets)

  imax = dimsizes(obs_datasets)-1
  if (any(obs_datasets.eq."Kadow")) then
    ii = ind(obs_datasets.eq."Kadow")
    if (ii.lt.imax) then
      obs_datasets(ii) = obs_datasets(imax)
      obs_datasets(imax) = "Kadow"
      obs_tmp = obs[imax]
      obs[imax] = obs[ii]
      obs[ii] = obs_tmp
      delete(obs_tmp)
    end if
    imax = imax - 1
  end if
  if (any(obs_datasets.eq."CowtanWay")) then
    ii = ind(obs_datasets.eq."CowtanWay")
    if (ii.lt.imax) then
      obs_datasets(ii) = obs_datasets(imax)
      obs_datasets(imax) = "CowtanWay"
      obs_tmp = obs[imax]
      obs[imax] = obs[ii]
      obs[ii] = obs_tmp
      delete(obs_tmp)
    end if
    imax = imax - 1
  end if
  
    
  

  ; configuration
  seas = "ann"    ; ann

  syear_early = 1850
  eyear_early = 1900
  syear_recent = 2010
  eyear_recent = 2019
  if (isatt(diag_script_info, "syear_early")) then
    syear_early = diag_script_info@syear_early
  end if
  if (isatt(diag_script_info, "eyear_early")) then
    eyear_early = diag_script_info@eyear_early
  end if
  if (isatt(diag_script_info, "syear_recent")) then
    syear_recent = diag_script_info@syear_recent
  end if
  if (isatt(diag_script_info, "eyear_recent")) then
    eyear_recent = diag_script_info@eyear_recent
  end if
  
  thres = 0.75
  if (isatt(diag_script_info, "threshold")) then
    thres = diag_script_info@threshold
  end if


  
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

;----------------------------------------------------------
  pi=4.*atan(1.0)
  rad=(pi/180.)

; ==================== Read piControl ====================
  nens = ListCount(piControl_cmip6)

  ensemble_assign = new(nens,integer)
  enum = 1
  temp = piControl_cmip6_datasets(0)
  do gg = 0, nens-1
    if (temp.eq.piControl_cmip6_datasets(gg)) then   ; does the model name match what's in temp?
      ensemble_assign(gg) = enum ; if so, assign it the same number
    else
      enum = enum+1              ; if not, assign it the next number
      ensemble_assign(gg) = enum
      temp = piControl_cmip6_datasets(gg)
    end if
  end do
  ensemble_assign@models = str_join(piControl_cmip6_datasets+"/"+piControl_cmip6_ensembles,",")

  do gg = 0, nens-1
    
    arr := read_data(piControl_cmip6[gg])
    arr&time = cd_calendar(arr&time,1)
    numyr = dimsizes(arr&time)/12
    
    if (gg.eq.0) then
      finarr = new((/nens, numyr/),float)
      finarr!0 = "E"
      finarr&E = ispan(0, nens-1,1)
      finarr!1 = "time"
      finarr&time = ispan(0,numyr-1,1)

;      finarr_trend = new((/nens,numyr-120/),float)   ; -120 = number of observational years
      finarr_trend = new((/nens,numyr-(eyear_recent-syear_early+1)/),float)  ; -(eyear_recent-syear_early+1) = number of historical-sssp years
      finarr_trend!0 = "E"
      finarr_trend&E = ispan(0,nens-1,1)
      finarr_trend!1 = "time2"
      finarr_trend&time2 = ispan(0,numyr-(eyear_recent-syear_early+1)-1,1)
    end if

    if (gg.ge.1.and.isatt(arr,"is_all_missing")) then   ; 
      continue
    end if

    arr = rmMonAnnCycTLL(arr)
    temp := runave_n_Wrap(arr,12,0,0)
    if (seas.eq."ann") then
      arr_seas := temp(5::12,:,:)
    end if
    arr_seas = dtrend_msg_n(ispan(0,dimsizes(arr_seas&time)-1,1),arr_seas,False,False,0)
    delete(temp)
    
    coswgt:=cos(rad*arr_seas&lat)
    coswgt!0 = "lat" 
    coswgt&lat= arr_seas&lat
    arr_aa := wgt_areaave(arr_seas,coswgt,1.0,0) 

    do ii = 0,dimsizes(arr_aa)-(eyear_recent-syear_early+1)-1
      is = ii
      ie = is + eyear_early - syear_early
      js = is + syear_recent - syear_early
      je = js + eyear_recent - syear_recent
      finarr_trend(gg,ii) = avg(arr_aa(js:je)) - avg(arr_aa(is:ie))
    end do
    finarr(gg,:) = (/ arr_aa /)
;    print("Done with "+paths(gg)+" "+syearP(gg)+" "+eyearP(gg))
  end do
;  finarr_trend = finarr_trend*10.   ; scale to trends per decade
  finarr_trend@units = "~S~o~N~C"
  
  fn = out_path + "piControl_cmip6_gsat.nc"  
  system("rm -f "+fn)
  z = addfile(fn,"c")
  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->ensemble_assign = ensemble_assign
  z->picontrol_tas_aa = finarr
  z->picontrol_tas_aa_runtrend = finarr_trend
  delete(z)
  delete([/ensemble_assign, finarr, finarr_trend/])
  
  z = addfile(fn,"r")
  model_tseries = z->picontrol_tas_aa
  model_runtrend = z->picontrol_tas_aa_runtrend
  ea = z->ensemble_assign
  
; ==================== Read historical ====================
  nens = ListCount(hist_cmip6)

  ensemble_assign = new(nens,integer)
  enum = 1
  temp = hist_cmip6_datasets(0)
  do gg = 0, nens-1
    if (temp.eq.hist_cmip6_datasets(gg)) then   ; does the model name match what's in temp?
      ensemble_assign(gg) = enum ; if so, assign it the same number
    else
      enum = enum+1              ; if not, assign it the next number
      ensemble_assign(gg) = enum
      temp = hist_cmip6_datasets(gg)
    end if
  end do
  ensemble_assign@models = str_join(hist_cmip6_datasets+"/"+hist_cmip6_ensembles,",")

  do gg = 0, nens-1
    arr := read_data(hist_cmip6[gg])
    arr&time = cd_calendar(arr&time,1)

    if (gg.eq.0) then
      syear = toint(arr&time(0)/100)
      eyear = toint(arr&time(dimsizes(arr&time)-1)/100)
      finarr = new((/nens,eyear-syear+1/),float)
      finarr!0 = "E"
      finarr&E = ispan(0,nens-1,1)
      finarr!1 = "time"
      finarr&time = ispan(syear,eyear,1)

      finarr_trend = new((/nens/),float)   
      finarr_trend!0 = "E"
      finarr_trend&E = ispan(0,nens-1,1)
    end if

    if (gg.ge.1.and.isatt(arr,"is_all_missing")) then   ; 
      continue
    end if

    arr = rmMonAnnCycTLL(arr)
    temp := runave_n_Wrap(arr,12,0,0)
    if (seas.eq."ann") then
      arr_seas := temp(5::12,:,:)
      arr_seas&time = toint(arr_seas&time/100)
      print(arr_seas&time(0)+"-"+arr_seas&time(dimsizes(arr_seas&time)-1))
    end if
    delete(temp)

    coswgt:=cos(rad*arr_seas&lat)
    coswgt!0 = "lat" 
    coswgt&lat= arr_seas&lat
    arr_aa := wgt_areaave_Wrap(arr_seas,coswgt,1.0,0) 
    delete(coswgt)
     
;    tttt := dtrend_msg(ispan(0,dimsizes(arr_aa)-1,1),arr_aa,False,True)
    finarr_trend(gg) = avg(arr_aa({syear_recent:eyear_recent})) - avg(arr_aa({syear_early:eyear_early}))

    finarr(gg,:) = (/ arr_aa /)
;     print("Done with "+paths(gg)+"*"+model(gg)+"*"+emem(gg)+"*.nc "+scenario_paths(gg)+"*.nc")
  end do
;  finarr_trend = finarr_trend*10.   ; scale to trends per decade
;  finarr_trend@units = "C per decade"
  finarr_trend@units = "~S~o~N~C"
  
  fn = out_path + "historical_cmip6_gsat.nc"
  system("rm -f "+fn)
  z = addfile(fn, "c")
  z@source = systemfunc("pwd")+"/"+get_script_name()
  z->ensemble_assign = ensemble_assign
  z->histssp_tas_aa = finarr
  z->histssp_tas_aa_trend = finarr_trend
  delete(z)
  delete([/ensemble_assign, finarr, finarr_trend/])
  
  z = addfile(fn, "r")
  model_trendHS = z->histssp_tas_aa_trend
  eaHS          = z->ensemble_assign

  eaHS = where(ismissing(model_trendHS),eaHS@_FillValue,eaHS)

  weights = new(dimsizes(model_trendHS),float)
  do gg = 1,max(eaHS)         ; calculate ensemble means
     wind := ind(eaHS.eq.gg)
     if (.not.ismissing(wind(0))) then
        weights(wind) = 1./dimsizes(wind)
     end if
  end do
;  print(eaHS+" "+weights)

  hres = True
;  hres@bin_center = fspan(-.1975,.1975,80)
;  hres@bin_bounds = fspan(-.2,.2,81)
  hres@bin_center = fspan(-1.975,1.975,80)
  hres@bin_bounds = fspan(-2.,2.,81)
  hres@bin_spacing = hres@bin_center(1) - hres@bin_center(0)
  hres@bin_min = min(hres@bin_bounds)
  hres@bin_max  = max(hres@bin_bounds) 
  histssp_h = weighted_pdfx(model_trendHS,weights,hres)
  histssp_h&time = hres@bin_center
;  histssp_mean = weighted_mean(model_trendHS,weights)
;  histssp_5th = weighted_percentile(model_trendHS,weights, 0.05)
;  histssp_95th = weighted_percentile(model_trendHS,weights, 0.95)
  if (min(model_trendHS).lt.min(hres@bin_bounds).or.max(model_trendHS).gt.max(hres@bin_bounds)) then    ; check to make sure values lie within bin_bounds. Can remove if bin_bounds will not change?
     printMinMax(model_trendHS,0)
     print("trends outside of bins, must reset")
     exit
  end if
  delete(hres)

;-----------------------------------------------------------------
; Compute area averages/trends for observations
  nobs = ListCount(obs)

  obs_tseries = new((/nobs,eyear_recent-syear_early+1/),float)
  obs_tseries!0 = "dataset"
  obs_tseries&dataset = ispan(1, nobs, 1)
  obs_tseries!1 = "year"
  obs_tseries&year = ispan(syear_early, eyear_recent, 1)
  do gg = 0, nobs-1
    arr := read_data(obs[gg])
    arr&time = cd_calendar(arr&time,1)
    syear_tmp = toint(arr&time(0)/100)
    eyear_tmp = toint(arr&time(dimsizes(arr&time)-1)/100)

    if (obs[gg]@short_name.eq."tas") then
      arr = rmMonAnnCycTLL(arr)
    end if
    
    temp := runave_n_Wrap(arr,12,0,0)
    if (any(obs[gg]@short_name.eq.(/"tas", "tasa"/))) then
      if (seas.eq."ann") then
        arr_seas := temp(5::12, :, :)
      end if
      coswgt:=cos(rad*arr_seas&lat)
      coswgt!0 = "lat" 
      coswgt&lat= arr_seas&lat
      tseries := wgt_areaave_Wrap(arr_seas,coswgt,1.0,0) 
      delete(coswgt)
    else
      if (seas.eq."ann") then
        tseries := temp(5::12)
      end if
    end if
    tseries&time = toint(tseries&time/100)
;    print(tseries&time(0)+"-"+tseries&time(dimsizes(tseries&time)-1))
    tseries = tseries - avg(tseries({1961:1990}))
    
    obs_tseries(gg, {syear_tmp:eyear_tmp}) = (/ tseries /)
    delete([/arr, temp, tseries/])
  end do
  obs_tseries@datasets = str_join(obs_datasets, " ")
  
  obs_trends = new(nobs,float)
;  do gg = 0, nobs-1
;    tttt := dtrend_msg(ispan(syear, eyear, 1), obs_tseries(gg, :), False, True)
;    obs_trends(gg) = (/ tttt@slope /)
;  end do
  obs_trends = dim_avg_n(obs_tseries(:, {syear_recent:eyear_recent}), 1) - dim_avg_n(obs_tseries(:, {syear_early:eyear_early}), 1)
  obs_trends@datasets = str_join(obs_datasets, " ")
  
;  obs_trends = obs_trends*10.   ; scale to trends per decade
;  print(obs_trends)
print("==============================")
;---------------------------------------------
  fna = "$diag_scripts/ipccwg1ar6ch3_gsat_pi/cmip6_color_bymodel.rgb"
  c := asciiread(fna,(/numAsciiRow(fna)/),"string")
  cmap0 = new((/dimsizes(c),3/),float)
  dash0 = new((/dimsizes(c)/),integer)
  mname = new(dimsizes(c),string)
  do gg = 0,dimsizes(c)-1
    tt := str_split(str_squeeze(c(gg))," ")
    cmap0(gg,:) = (/tofloat(tt(1)),tofloat(tt(2)),tofloat(tt(3))/)
    mname(gg) = tt(0)
  end do
  dash0 = 0
  do gg = 1, dimsizes(c)-1
    do kk = 0, gg-1
      if (all(cmap0(gg,:).eq.cmap0(kk,:))) then
        dash0(gg) = dash0(gg) + 1
      end if
    end do
  end do
   
  cmap0 = cmap0/255.
  ea_models := str_split(ea@models,",")

  cmap1 = new((/dimsizes(ea),3/),float)
  dash1 = new((/dimsizes(ea)/),integer)
  mname_used = new(dimsizes(ea),string)
  do gg = 0,dimsizes(ea)-1
     tt := str_split(ea_models(gg),"/")
     mname_used(gg) = tt(0)
     wind := ind(tt(0).eq.mname)
     cmap1(gg,:) = cmap0(wind,:)
     dash1(gg) = dash0(wind)
;     print(tt(0)+" "+cmap1(gg,:))
  end do

;===========================================================================================
  ofile = plot_dir + "gsat_pi+hist+obs"
  wks = gsn_open_wks(file_type,ofile)

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

  mres = True
  mres@gsnDraw = False
  mres@gsnFrame = False
  mres@vpWidthF = 0.8
  mres@vpHeightF = 0.35
  mres@trYMinF = -.8000001
  mres@trYMaxF = 1.2
  mres@tiXAxisFontHeightF = 0.018
  mres@tiYAxisFontHeightF = 0.018
  mres@tiXAxisString = "(year)"
  mres@tiYAxisString = "(~S~o~N~C)"
  mres@tiYAxisOffsetXF = 0.005
  mres@tiXAxisOffsetYF = 0.004
  
  mres@gsnCenterString = ""
  mres@gsnLeftString = ""
  mres@gsnRightString = ""
  mres@xyMonoDashPattern = False
  mres@xyLineDashSegLenF = 0.08
  
  mres@gsnYRefLine = 0.0
  mres@gsnYRefLineColor = 1

  mres@xyMonoLineColor    = False            
  mres@tmXBLabelFontHeightF = 0.018
  mres@tmYLLabelFontHeightF = mres@tmXBLabelFontHeightF 
  mres@tmXBLabelDeltaF = -0.2
  mres@tmYLLabelDeltaF = -0.2
  mres@tmXTOn = False
  mres@tmYROn = False
  mres@trXMinF = 0.
  mres@trXMaxF = 501.

  plot = new(3,graphic)
  sd = dim_stddev_Wrap(model_tseries)
  sd2 = sd
  qsort(sd2)

  threshold = sd2(round(dimsizes(sd2)*thres,3)-1)
  lowb = (/0,threshold+.001/)
  highb = (/threshold,50/)
  do gg = 0,1
     w_ind := ind(sd.ge.lowb(gg).and.sd.le.highb(gg))
;     print(w_ind)
;     print(mname_used(w_ind))
     do hh = 0,dimsizes(w_ind)-1
       print(""+stddev(model_tseries(w_ind(hh),:)))
     end do
     if (gg.eq.0) then
       mres@xyLineThicknessF := 1.
       mres@xyLineColors := cmap1(w_ind,:)   ;        print(mres@xyLineColors*255.)
       mres@xyDashPatterns := dash1(w_ind)
       plot(gg) = gsn_csm_xy(wks,ispan(1,500,1),model_tseries(w_ind,:),mres)
       nruns_a = num(.not.ismissing(w_ind))
     else
;       mres@xyMonoLineColor    = True       
;       mres@xyLineColor = (/.75,.75,.75/)
;       plot(gg) = gsn_csm_xy(wks,ispan(1,500,1),model_tseries(w_ind,:),mres)
;       mres@xyMonoLineColor    = False         
       mres@xyLineThicknessF := 1.
       xyLineColors = cmap1(w_ind,:)
       mres@xyLineColors := xyLineColors
       xyDashPatterns = dash1(w_ind)
       mres@xyDashPatterns := xyDashPatterns
;       mres@xyLineThicknessF = 3.0
       mnames = mname_used(w_ind)
       plot(gg) = gsn_csm_xy(wks,ispan(1,500,1),model_tseries(w_ind,:),mres)
;       oplot = gsn_csm_xy(wks,ispan(1,500,1),runave(model_tseries(w_ind,:),10,0),mres)
;       overlay(plot(gg),oplot)
       nruns_b = num(.not.ismissing(w_ind))
     end if
  end do

  mres@xyLineThicknessF = 1.0
  mres@vpWidthF = mres@vpWidthF*((2020-syear_early+1)/500.)
  mres@trXMinF = syear_early
  mres@trXMaxF = 2020
  mres@tmXBMode = "Explicit"
  mres@tmXBValues = (/1860,1900,1940,1980,2020/)
  mres@tmXBLabels = mres@tmXBValues
  mres@tmXBMinorValues = ispan(1850,2020,10)
  delete(mres@tiXAxisString)
  mres@xyMonoLineColor = False
  mres@xyMonoDashPattern = False
  obsDashPatterns = (/0,1,2,0,1/)
  mres@xyDashPatterns := obsDashPatterns(0:dimsizes(obs_datasets)-1)
  obsColors = (/"black", "black", "black", "grey60", "grey60"/)
  mres@xyLineColors := obsColors(0:dimsizes(obs_datasets)-1)
  mres@xyLineDashSegLenF = 0.15
  plot(2) = gsn_csm_xy(wks,ispan(syear,eyear,1),obs_tseries,mres)

  panres = True
  panres@gsnPanelTop = 0.93
  panres@gsnPanelBottom = 0.05
  panres@gsnFrame = False
  panres@gsnPanelDebug = True
  panres@gsnPanelXF = (/0.2718,0.2718,0.2718+0.454463-0.155426/)
  panres@gsnPanelYF = (/-1,0.62,-1/)
  panres@gsnPanelYWhiteSpacePercent = 8. ;10.0
  panres@gsnPanelDebug = True
  gsn_panel(wks,plot,(/3,1/),panres)
;----------------------------
  hres = True     
  hres@gsnDraw = False
  hres@gsnFrame = False
;  printVarSummary(model_runtrend)
;  printMinMax(model_runtrend,0)

;  hres@gsnHistogramBinIntervals = fspan(-.1,.1,41)
  hres@gsnHistogramBinIntervals = fspan(-1.,1.,41)
  plot = gsn_histogram(wks,ndtooned(model_runtrend),hres)
  aa_h = plot@NumInBins*1.
  aa_h!0 = "time"
;  aa_h&time = fspan(-.0975,.0975,40)
  aa_h&time = fspan(-.975,.975,40)
  aa_h = (/ (aa_h/sum(aa_h))*100. /)
  delete([/plot/])
  if (min(model_runtrend).le.min(hres@gsnHistogramBinIntervals).or.max(model_runtrend).gt.max(hres@gsnHistogramBinIntervals)) then   ; can remove if gsnHistogramBinInterval setting will not change?
     print("AA trends outside histogram bin range, adjust bin range")
     printMinMax(model_runtrend,0)
     exit
  end if

;  dims = dimsizes(model_runtrend)
;  wtmp = new(dims, "float")
;  do gg = 0, dims(0)-1
;    wtmp(gg, :) = 1./tofloat(num(.not.ismissing(model_runtrend(gg, :))))
;  end do
;  model_runtrend_mean = weighted_mean(ndtooned(model_runtrend), ndtooned(wtmp))
;  model_runtrend_5th = weighted_percentile(ndtooned(model_runtrend), ndtooned(wtmp), 0.05)
;  model_runtrend_95th = weighted_percentile(ndtooned(model_runtrend), ndtooned(wtmp), 0.95)
;  delete(wtmp)
;  delete(dims)
  
;  cmap0 := read_colormap_file("../AR6_colormaps_NCL/temp_div_12.rgb")
  cmap0 := read_colormap_file("$diag_scripts/shared/plot/rgb/ipcc-ar6_temperature_div_12.rgb")
  bres = True
  bres@gsnDraw = False
  bres@gsnFrame = False
;  bres@gsnYRefLine = 0.0
;  bres@gsnXRefLineColor = "gray50"
;  bres@gsnXRefLineThicknessF = 1.5
  bres@vpWidthF = 0.24; 0.27
  bres@vpHeightF = 0.198828  ;0.199674
  bres@vpXF = 0.2718
  bres@vpYF = 0.318137  ;0.307954
  bres@gsnXYBarChart = True
  bres@tmXBLabelFontHeightF = 0.01 ;0.0117
  bres@tmXTLabelFontHeightF = 0.01 ;0.0117
  bres@tmYLLabelFontHeightF = 0.01 ;0.0117
  bres@tmXBLabelDeltaF = -0.2
  bres@tmYLLabelDeltaF = -0.2
  bres@tmXTOn = False
  bres@tmYROn = False
  bres@tiMainOn = False  
  bres@xyLineThicknessF = 1.5
;  bres@tiYAxisString = "% of simulations"
  bres@tiYAxisOffsetXF = 0.005
  bres@tiXAxisOffsetYF = 0.004
  bres@trYMinF = 0.0
  bres@trYMaxF = 28.0
  bres@trXMinF = -0.6 ;-.08
  bres@trXMaxF = 1.8 ;.16
  bres@tmXBMode = "Explicit"
  bres@tmXBLabels = (/-0.5, 0., 0.5, 1., 1.5, 2./)
  bres@gsnXYBarChartColors = cmap(8,:)
  bres@gsnXYBarChartFillOpacityF = 0.3
  bres@xyLineColor = "transparent"  ;"white"
  bres@gsnLeftString = "" ;syear+"-"+eyear

  bres@tfPolyDrawOrder = "PreDraw"
  bres@tiYAxisString = "(% of segments)"
;  bres@tiXAxisString = "(~S~o~N~C 10yr~S~-1~N~)"
  bres@tiXAxisString = "(~S~o~N~C)"
  bres@gsnXYBarChartWidth = .005
  bplot_ov = gsn_csm_xy(wks, aa_h&time, aa_h, bres)  
  
  bres@gsnXYBarChartColors = cmap(3,:)
  bres@gsnXYBarChartWidth = .005
  bres@gsnXYBarChartFillOpacityF = 0.3
  bplot = gsn_csm_xy(wks,histssp_h&time,histssp_h,bres)
  overlay(bplot,bplot_ov)

;  polyres := True
;  polyres@gsLineThicknessF = 1.
;  polyres@gsLineColor = cmap(8,:)
;  polyres@gsLineDashPattern = 0
;  dum_pimean = gsn_add_polyline(wks,bplot,(/model_runtrend_mean,model_runtrend_mean/),(/0,100/),polyres)
;  polyres@gsLineDashPattern = 2
;;  polyres@gsLineDashSegLenF = 0.08
;  dum_pi5th = gsn_add_polyline(wks,bplot,(/model_runtrend_5th,model_runtrend_5th/),(/0,100/),polyres)
;  dum_pi95th = gsn_add_polyline(wks,bplot,(/model_runtrend_95th,model_runtrend_95th/),(/0,100/),polyres)
;
;  polyres@gsLineColor = cmap(3,:)
;  polyres@gsLineDashPattern = 0
;  dum_histmean = gsn_add_polyline(wks,bplot,(/histssp_mean,histssp_mean/),(/0,100/),polyres)
;  polyres@gsLineDashPattern = 2
;;  polyres@gsLineDashSegLenF = 0.08
;  dum_hist5th = gsn_add_polyline(wks,bplot,(/histssp_5th,histssp_5th/),(/0,100/),polyres)
;  dum_hist95th = gsn_add_polyline(wks,bplot,(/histssp_95th,histssp_95th/),(/0,100/),polyres)
  
;- - - - - - - - - - - - - - -  -- - - - - 
  polyres := True
  polyres@gsLineThicknessF = 0.5
  polyres@gsLineColor = "grey50"
  polyres@gsLineDashPattern = 2
  polyres@gsLineDashSegLenF = 0.08
  dum_refline = gsn_add_polyline(wks,bplot,(/0,0/),(/0,100/),polyres)
  delete(polyres@gsLineDashSegLenF)
  
  polyres@gsLineColor := 1
  polyres@gsLineDashPattern = 0
  polyres@tfPolyDrawOrder = "PostDraw"
  polyres@gsLineThicknessF = 1.5    ; add mme and obs reference lines
  dum_obs = new(nobs,graphic)
  do gg = 0, nobs-1
    polyres@gsLineDashPattern = mres@xyDashPatterns(gg)
    polyres@gsLineColor := mres@xyLineColors(gg)
    polyres@gsLineDashSegLenF = 0.1
    dum_obs(gg) = gsn_add_polyline(wks,bplot,(/obs_trends(gg),obs_trends(gg)/),(/0,100/),polyres)
  end do
  draw(bplot)
;---------------------------------------------------------------------
  textres = True 
  textres@txFont = 22
  textres@txFontHeightF = 0.013
  gsn_text_ndc(wks,"Internal variabiliy of annual GSAT versus observed changes",0.5,0.94,textres)

  textres@txFont = 21
  textres@txFontHeightF = 0.011
  textres@txFontColor = "gray40"
;  gsn_text_ndc(wks,"Temporal variation of GSAT in piControl ("+nruns_a+" models)",0.5,.92,textres)
  gsn_text_ndc(wks,"Temporal variation of GSAT in piControl ("+nruns_a+" models)",0.5,.915,textres)
;  gsn_text_ndc(wks,"Temporal variation of GSAT in piControl ("+nruns_b+" models)",0.5,0.635,textres)
  gsn_text_ndc(wks,"Temporal variation of GSAT in piControl ("+nruns_b+" models)",0.5,0.6291,textres)  
;  gsn_text_ndc(wks,"Observed and simulated 120-yr trends~C~        in piControl and historical runs",0.4,0.3375,textres)
  gsn_text_ndc(wks,"Changes from 1850-1900 to 2010-2019~C~ & corresponding changes in piControl",0.395,0.3375,textres) 
  gsn_text_ndc(wks,"    Observed ~C~GMST changes",0.65,0.3375,textres) 
;- - - - - - - - - - - - - - - - - - - - - - - 
; Timeseries b) legend
;
  gres = True
  gres@LineLabelWhiteSpacePercent = .75
  gres@ItemSpacePercent = 1.25

  lineres = True
  lineres@lgLineThicknesses = 1. ; 3.0                    ; line thicknesses
  lineres@LineLengthPercent = 2.5                         ; expressed as %, 0->100, length of line
;  lineres@lgDashIndexes = 0
;  lineres@lgMonoLineDashSegLen = True
  
  textres = True
  textres@lgLabelFontHeights = 0.006

  xpos = (/32,32,32,42,42,42,52,52,52,63,63,63/)
  ypos = (/61,59.5,58,61,59.5,58,61,59.5,58,61,59.5,578/)
  do gg = 0,dimsizes(mnames)-1
     gres@YPosPercent = ypos(gg)
     gres@XPosPercent = xpos(gg)
     textres@lgLabels = mnames(gg)
     lineres@lgDashIndexes = (/xyDashPatterns(gg)/)
     lineres@lgLineColors = xyLineColors(gg,:)
     lineres@lgLineDashSegLens = (/0.08/)
     simple_legend_ndc(wks,gres,lineres,textres)
  end do

  delete([/gres,lineres,textres/])
;- - - - - - - - - - - - - - - - - - - - - - - 
; Timeseries d) legend
;
  gres = True
  gres@LineLabelWhiteSpacePercent = .75
  gres@ItemSpacePercent = 1.75

  lineres = True
  lineres@lgLineThicknesses = 1.5                        ; line thicknesses
  lineres@LineLengthPercent = 3. ;2.5                         ; expressed as %, 0->100, length of line
;  lineres@lgDashIndexes = 0

  textres = True
  textres@lgLabelFontHeights = 0.010

  gres@YPosPercent = 8.
  if (dimsizes(obs_datasets).gt.3) then
    gres@XPosPercent = 50.5 ;62.5 
    lineres@lgDashIndexes := obsDashPatterns(0:2) ;(/0,2,3/)
    lineres@lgLineColors := obsColors(0:2)
    textres@lgLabels := obs_datasets(0:2) ;str_split(obs_tseries@datasets," ")
    simple_legend_ndc(wks,gres,lineres,textres)
    gres@XPosPercent = 64. ;62.5 
    lineres@lgDashIndexes := obsDashPatterns(3:dimsizes(obs_datasets)-1) ;(/0,2,3/)
    lineres@lgLineColors := obsColors(3:dimsizes(obs_datasets)-1)
    textres@lgLabels := obs_datasets(3:dimsizes(obs_datasets)-1) ;str_split(obs_tseries@datasets," ")
    simple_legend_ndc(wks,gres,lineres,textres)
  else
    gres@XPosPercent = 56. ;62.5 
    lineres@lgDashIndexes := obsDashPatterns(0:dimsizes(obs_datasets)-1) ;(/0,2,3/)
    lineres@lgLineColors := obsColors(0:dimsizes(obs_datasets)-1)
    textres@lgLabels := obs_datasets ;str_split(obs_tseries@datasets," ")
    simple_legend_ndc(wks,gres,lineres,textres)
  end if
  
;- - - - - - - - - - - - - - - - - - - - - - - 
; Bar chart legend
;
  lres = True   ; draw box showing observational range
  lres@gsFillOpacityF = bres@gsnXYBarChartFillOpacityF
  lres@gsFillColor := cmap(8,:)
  gsn_polygon_ndc(wks,(/.27,.285,.285,.27,.27/),(/.05,.05,.07,.07,.05/),lres)
  gsn_polygon_ndc(wks,(/.27,.285,.285,.27,.27/),(/.05,.05,.07,.07,.05/),lres)  
  textres@txFontHeightF = 0.010
  textres@txFontColor = "black"
  textres@txJust = "CenterLeft"
  gsn_text_ndc(wks,"piControl",.29,.06,textres)

;  lres@gsLineColor := cmap(8,:)
;  lres@gsLineDashPattern = 0
;  gsn_polyline_ndc(wks,(/.275,.275/),(/.04,.06/),lres)  
;  lres@gsLineDashPattern = 2
;  gsn_polyline_ndc(wks,(/.255,.255/),(/.04,.06/),lres)
;  gsn_polyline_ndc(wks,(/.295,.295/),(/.04,.06/),lres)  
;  textres2 = textres
;  textres2@txFontHeightF = 0.007
;  textres2@txAngleF = 90.
;  textres2@txJust = "CenterRight"
;  gsn_text_ndc(wks,"mean",.275,.038,textres2)
;  gsn_text_ndc(wks,"5th", .255,.038,textres2)
;  gsn_text_ndc(wks,"95th",.295,.038,textres2)

  
  cmap7 = new(4,float)
  cmap7(:2) = cmap(3,:)
  cmap7(3)  = 0.75 
  lres@gsFillColor := cmap7
;  gsn_polygon_ndc(wks,(/.36,.375,.375,.36,.36/),(/.05,.05,.07,.07,.05/),lres)
;  gsn_polygon_ndc(wks,(/.36,.375,.375,.36,.36/),(/.05,.05,.07,.07,.05/),lres)  
;  gsn_text_ndc(wks,"historical-ssp245",.38,.06,textres)
  gsn_polygon_ndc(wks,(/.39,.405,.405,.39,.39/),(/.05,.05,.07,.07,.05/),lres)
  gsn_polygon_ndc(wks,(/.39,.405,.405,.39,.39/),(/.05,.05,.07,.07,.05/),lres)  
  gsn_text_ndc(wks,"historical-ssp245",.41,.06,textres)

;  lres@gsLineColor := cmap(3,:)
;  lres@gsLineDashPattern = 0
;  gsn_polyline_ndc(wks,(/.415,.415/),(/.04,.06/),lres)  
;  lres@gsLineDashPattern = 2
;  gsn_polyline_ndc(wks,(/.395,.395/),(/.04,.06/),lres)
;  gsn_polyline_ndc(wks,(/.435,.435/),(/.04,.06/),lres)  
;
;  gsn_text_ndc(wks,"mean",.415,.038,textres2)
;  gsn_text_ndc(wks,"5th", .395,.038,textres2)
;  gsn_text_ndc(wks,"95th",.435,.038,textres2)

;- - - - - - - - - - - - - - - - - - - - - - - 
; Panel numbering
; 
  textres@txFontHeightF = 0.014
  gsn_text_ndc(wks,"(a)",.2818,.884804,textres)
  gsn_text_ndc(wks,"(b)",.2818,.6,textres)
  gsn_text_ndc(wks,"(c)",.2818,.298137,textres)
  gsn_text_ndc(wks,"(d)",.580837,.298137,textres)
  frame(wks)
  delete(wks)
  system("convert -density 216 -trim +repage -border 10 -bordercolor white -flatten "+ofile+"."+file_type+" "+ofile+".png")
end

