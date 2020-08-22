; draw_hadleycell_trend
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs:
;
; History
;    20200814 kosaka_yu: merge find_ensoevents.ncl
;                        follow the ENSO metric package at
;                        https://github.com/CLIVAR-PRP/ENSO_metrics/wiki/EnsoDuration
;    20200812 kosaka_yu: (c,d): show individual ensemble members instead of compositing all members
;                        and plot them for individual models instead of modelling centers
;                        flip x and y axes to plot ~50 models or more
;    20200511 kosaka_yu: cleaned to meet the code style;
;                        removed bottom two panels that are removed from SOD
;    20200105 kosaka_yu: show shading instead of lines;
;                        use duration based on composite
;    20191012 kosaka_yu: updated for ESMValTool 2.0.0b2
;    20190316 kosa_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/statistics.ncl"
load "$diag_scripts/shared/plot/style.ncl"
load "$diag_scripts/shared/plot/contour_maps.ncl"
load "$diag_scripts/shared/plot/contourplot.ncl"

load "$diag_scripts/ar6ch3_enso/functions.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")

  obs_projects = (/"OBS", "OBS6", "obs", "obs4mips", "ana4mips"/)

  if (isatt(diag_script_info, "syear")) then
    syear0 = diag_script_info@syear
  end if
  if (isatt(diag_script_info, "eyear")) then
    eyear0 = diag_script_info@eyear
  end if
  
  target_month = 12  ; December
  if (isatt(diag_script_info, "detect_month")) then
    target_month = diag_script_info@detect_month
  end if

  detect_threshold = 0.75
  if (isatt(diag_script_info, "detect_threshold")) then
    detect_threshold = diag_script_info@detect_threshold
  end if

  detect_threshold_unit = "stdv"
  if (isatt(diag_script_info, "unit_detect_threshold")) then
    detect_threshold_unit = diag_script_info@unit_detect_threshold
  end if
  if (any(detect_threshold_unit.eq.(/"degree", "degrees", "degC", "degK", \
                                     "Degree", "Degrees", "DegC", "DegK", \
                                     "Deg"/))) then
    detect_threshold_unit = "deg"
  elseif (any(detect_threshold_unit.eq.(/"sigma", "std_dev", \
                                         "stddev", "s.d."/))) then
    detect_threshold_unit = "stdv"
  end if

  duration_threshold = 0.5
  if (isatt(diag_script_info, "duration_threshold")) then
    duration_threshold = diag_script_info@duration_threshold
  end if
  duration_threshold_unit = "stdv"
  if (isatt(diag_script_info, "unit_duration_threshold")) then
    duration_threshold_unit = diag_script_info@unit_duration_threshold
  end if
  if (any(duration_threshold_unit.eq.(/"degree", "degrees", "degC", "degK", \
                                       "Degree", "Degrees", "DegC", "DegK", \
                                       "Deg"/))) then
    duration_threshold_unit = "deg"
  elseif (any(duration_threshold_unit.eq.(/"sigma", "std_dev", \
                                           "stddev", "s.d."/))) then
    duration_threshold_unit = "stdv"
  end if

  plot_each_cmip5 = False
  plot_each_cmip6 = True

  ; segment length for composite
  lag_range = (/-2, 3/) ; years

  ; segment length for evaluating duration
  duration_range = (/-1, 2/) ; years

; =============================
  
  ; Get file list
  input_dir = diag_script_info@input_files
  paths = systemfunc("ls "+input_dir+"/ENSOindex_*.nc")

  nobs = 0
  nruns_cmip5 = 0
  nruns_cmip6 = 0
  
  do nn = 0, dimsizes(paths)-1
    f = addfile(paths(nn), "r")
    Nino = f->ENSOindex
    type = typeof(Nino)
    if (any(Nino@project.eq.obs_projects)) then
      nobs = nobs + 1
    elseif (Nino@project.eq."CMIP5") then
      nruns_cmip5 = nruns_cmip5 + 1
    elseif (Nino@project.eq."CMIP6") then
      nruns_cmip6 = nruns_cmip6 + 1
    end if
    delete(Nino)
  end do

  if (nobs.gt.0) then
    obs = new(nobs, "string")
    ts_el_obs = new((/nobs, (lag_range(1)-lag_range(0)+1)*12/), type)
    ts_la_obs = new((/nobs, (lag_range(1)-lag_range(0)+1)*12/), type)
    ts_el_obs!1 = "time"
    ts_la_obs!1 = "time"
    ts_el_obs&time = ispan(lag_range(0)*12+1, (lag_range(1)+1)*12, 1)
    ts_la_obs&time = ispan(lag_range(0)*12+1, (lag_range(1)+1)*12, 1)
    duration_el_obs = new(nobs, "float")
    duration_la_obs = new(nobs, "float")
  end if

  if (nruns_cmip5.gt.0) then
    models_cmip5 = new(nruns_cmip5, "string")
    runs_cmip5 = new(nruns_cmip5, "string")
    esizes_cmip5 = new(nruns_cmip5, "integer")
    ts_el_cmip5 = new((/nruns_cmip5, (lag_range(1)-lag_range(0)+1)*12/), type)
    ts_la_cmip5 = new((/nruns_cmip5, (lag_range(1)-lag_range(0)+1)*12/), type)
    ts_el_cmip5!1 = "time"
    ts_la_cmip5!1 = "time"
    ts_el_cmip5&time = ispan(lag_range(0)*12+1, (lag_range(1)+1)*12, 1)
    ts_la_cmip5&time = ispan(lag_range(0)*12+1, (lag_range(1)+1)*12, 1)
    duration_el_cmip5 = new(nruns_cmip5, "float")
    duration_la_cmip5 = new(nruns_cmip5, "float")
  end if

  if (nruns_cmip6.gt.0) then
    models_cmip6 = new(nruns_cmip6, "string")
    runs_cmip6 = new(nruns_cmip6, "string")
    esizes_cmip6 = new(nruns_cmip6, "integer")
    ts_el_cmip6 = new((/nruns_cmip6, (lag_range(1)-lag_range(0)+1)*12/), type)
    ts_la_cmip6 = new((/nruns_cmip6, (lag_range(1)-lag_range(0)+1)*12/), type)
    ts_el_cmip6!1 = "time"
    ts_la_cmip6!1 = "time"
    ts_el_cmip6&time = ispan(lag_range(0)*12+1, (lag_range(1)+1)*12, 1)
    ts_la_cmip6&time = ispan(lag_range(0)*12+1, (lag_range(1)+1)*12, 1)
    duration_el_cmip6 = new(nruns_cmip6, "float")
    duration_la_cmip6 = new(nruns_cmip6, "float")
  end if

  time_composite = new((lag_range(1)-lag_range(0)+1)*12, "integer") ; for ts array in the loop
  do yr = lag_range(0), lag_range(1)
    time_composite((yr-lag_range(0))*12:(yr-lag_range(0))*12+11) = yr*100 + ispan(1, 12, 1)
  end do
  time_duration = new((duration_range(1)-duration_range(0)+1)*12, "integer") ; for ts array in the loop
  do yr = duration_range(0), duration_range(1)
    time_duration((yr-duration_range(0))*12:(yr-duration_range(0))*12+11) = yr*100 + ispan(1, 12, 1)
  end do
  
  iobs = -1
  irun_cmip5 = -1
  irun_cmip6 = -1
  do nn = 0, dimsizes(paths)-1
    f = addfile(paths(nn), "r")
    Nino = f->ENSOindex
    if (any(Nino@project.eq.obs_projects)) then
      iobs = iobs + 1
      obs(iobs) = Nino@dataset
    elseif (Nino@project.eq."CMIP5") then
      irun_cmip5 = irun_cmip5 + 1
      models_cmip5(irun_cmip5) = Nino@dataset
      runs_cmip5(irun_cmip5) = Nino@ensemble
      esizes_cmip5(irun_cmip5) = Nino@esize
    elseif (Nino@project.eq."CMIP6") then
      irun_cmip6 = irun_cmip6 + 1
      models_cmip6(irun_cmip6) = Nino@dataset
      runs_cmip6(irun_cmip6) = Nino@ensemble
      esizes_cmip6(irun_cmip6) = Nino@esize
    end if
  
    syear_data = toint(Nino&time(0))/100
    eyear_data = toint(Nino&time(dimsizes(Nino)-1))/100
    smonth_data = toint(Nino&time(0)) - syear_data*100
    emonth_data = toint(Nino&time(dimsizes(Nino)-1)) - eyear_data*100

    if (isdefined("syear0")) then
      syear = syear0
    else
      syear = syear_data - min((/lag_range(0), duration_range(0)/))
    end if
    if (isdefined("eyear0")) then
      eyear = eyear0
    else
      eyear = eyear_data - max((/lag_range(1), duration_range(1)/))
    end if

    ; Select ENSO event
    idx   = Nino({syear*100+target_month:eyear*100+target_month:12})
    stdv_idx = stddev(idx)
    if (detect_threshold_unit.eq."stdv") then
      idx = idx / stdv_idx
    end if
    idx_el = ind(idx.gt.abs(detect_threshold))
    idx_la = ind(idx.lt.-abs(detect_threshold))
    if (.not.all(ismissing(idx_el))) then
      years_el = toint(idx&time(idx_el))/100
      nevents_el = dimsizes(years_el)
    else
;      print(idx+" "+detect_threshold)
;      exit
      nevents_el = 0
    end if
    if (.not.all(ismissing(idx_la))) then
      years_la = toint(idx&time(idx_la))/100
      nevents_la = dimsizes(years_la)
    else
;      print(idx+" "+detect_threshold)
;      exit
      nevents_la = 0
    end if
    delete(idx)
    delete(stdv_idx)
    delete(idx_el)
    delete(idx_la)


    ; Get index time series and composite the events
    if (nevents_el.gt.0) then
      ts = new((/nevents_el, (lag_range(1)-lag_range(0)+1)*12/), type)
      ts!1 = "time"
      ts&time = time_composite
      do ii = 0, nevents_el-1
        t0 = max((/(years_el(ii) + lag_range(0))*100+1, syear_data*100+smonth_data/))
        t1 = min((/(years_el(ii) + lag_range(1))*100+12, eyear_data*100+emonth_data/))
        s0 = t0 - years_el(ii)*100
        s1 = t1 - years_el(ii)*100
        ts(ii, {s0:s1}) = (/Nino({t0:t1})/)
      end do
      if (any(Nino@project.eq.obs_projects)) then
        ts_el_obs(iobs, :) = dim_avg_n(ts, 0)
      elseif (Nino@project.eq."CMIP5") then
        ts_el_cmip5(irun_cmip5, :) = dim_avg_n(ts, 0)
      elseif (Nino@project.eq."CMIP6") then
        ts_el_cmip6(irun_cmip6, :) = dim_avg_n(ts, 0)
      end if
      delete(ts)
    end if
    if (nevents_la.gt.0) then
      ts = new((/nevents_la, (lag_range(1)-lag_range(0)+1)*12/), type)
      ts!1 = "time"
      ts&time = time_composite
      do ii = 0, nevents_la-1
        t0 = max((/(years_la(ii) + lag_range(0))*100+1, syear_data*100+smonth_data/))
        t1 = min((/(years_la(ii) + lag_range(1))*100+12, eyear_data*100+emonth_data/))
        s0 = t0 - years_la(ii)*100
        s1 = t1 - years_la(ii)*100
        ts(ii, {s0:s1}) = (/Nino({t0:t1})/)
      end do
      if (any(Nino@project.eq.obs_projects)) then
        ts_la_obs(iobs, :) = dim_avg_n(ts, 0)
      elseif (Nino@project.eq."CMIP5") then
        ts_la_cmip5(irun_cmip5, :) = dim_avg_n(ts, 0)
      elseif (Nino@project.eq."CMIP6") then
        ts_la_cmip6(irun_cmip6, :) = dim_avg_n(ts, 0)
      end if      
      delete(ts)
    end if

    ; Get duration and average across events
    threshold = duration_threshold
    if (duration_threshold_unit.eq."stdv") then
      threshold = duration_threshold * stddev(Nino({syear*100+1:eyear*100+12}))
    end if

    if (nevents_el.gt.0) then
      durations = new(nevents_el, "float")
      ts = new((/(duration_range(1)-duration_range(0)+1)*12/), type)
      ts!0 = "time"
      ts&time = time_duration
      do ii = 0, nevents_el-1
        t0 = max((/(years_el(ii) + duration_range(0))*100+1, syear_data*100+smonth_data/))
        t1 = min((/(years_el(ii) + duration_range(1))*100+12, eyear_data*100+emonth_data/))
        s0 = t0 - years_el(ii)*100
        s1 = t1 - years_el(ii)*100
        ts({s0:s1}) = (/Nino({t0:t1})/)
        durations(ii) = 0.
        j0 = ind(time_duration.eq.target_month)
        do j = j0, 0, 1
          if (.not.ismissing(ts(j)).and.ts(j).gt.threshold) then
            durations(ii) = durations(ii) + 1.
          else
            break
          end if
        end do
        do j = j0+1, dimsizes(ts)-1
          if (.not.ismissing(ts(j)).and.ts(j).gt.threshold) then
            durations(ii) = durations(ii) + 1.
          else
            break
          end if
        end do
      end do
      if (any(Nino@project.eq.obs_projects)) then
        duration_el_obs(iobs) = avg(durations)
;print("El Nino: "+years_el+" "+durations+" months")
;print("Mean El Nino duration = "+duration_el_obs(iobs)+" months")
      elseif (Nino@project.eq."CMIP5") then
        duration_el_cmip5(irun_cmip5) = avg(durations)
      elseif (Nino@project.eq."CMIP6") then
        duration_el_cmip6(irun_cmip6) = avg(durations)
      end if
      delete(ts)
      delete(durations)
    end if
    if (nevents_la.gt.0) then
      durations = new(nevents_la, "float")
      ts = new((/(duration_range(1)-duration_range(0)+1)*12/), type)
      ts!0 = "time"
      ts&time = time_duration
      do ii = 0, nevents_la-1
        t0 = max((/(years_la(ii) + duration_range(0))*100+1, syear_data*100+smonth_data/))
        t1 = min((/(years_la(ii) + duration_range(1))*100+12, eyear_data*100+emonth_data/))
        s0 = t0 - years_la(ii)*100
        s1 = t1 - years_la(ii)*100
        ts({s0:s1}) = (/Nino({t0:t1})/)
        durations(ii) = 0.
        j0 = ind(time_duration.eq.target_month)
        do j = j0, 0, 1
          if (.not.ismissing(ts(j)).and.ts(j).lt.-threshold) then
;if (any(Nino@project.eq.obs_projects)) then
;print(years_la(ii)*100+time_duration(j)+" "+ts(j)+" vs "+(-threshold))
;end if
            durations(ii) = durations(ii) + 1.
          else
            break
          end if
        end do
        do j = j0+1, dimsizes(ts)-1
          if (.not.ismissing(ts(j)).and.ts(j).lt.-threshold) then
;if (any(Nino@project.eq.obs_projects)) then
;print(years_la(ii)*100+time_duration(j)+" "+ts(j)+" vs "+(-threshold))
;end if
            durations(ii) = durations(ii) + 1.
          else
            break
          end if
        end do
      end do
      if (any(Nino@project.eq.obs_projects)) then
        duration_la_obs(iobs) = avg(durations)
;print("La Nina: "+years_la+" "+durations+" months")
;print("Mean La Nina duration = "+duration_la_obs(iobs)+" months")
      elseif (Nino@project.eq."CMIP5") then
        duration_la_cmip5(irun_cmip5) = avg(durations)
      elseif (Nino@project.eq."CMIP6") then
        duration_la_cmip6(irun_cmip6) = avg(durations)
      end if      
      delete(ts)
      delete(durations)
    end if

    delete(Nino)
    if (isdefined("years_el")) then
      delete(years_el)
    end if
    if (isdefined("years_la")) then
      delete(years_la)
    end if
  end do

  if (nruns_cmip5.gt.0) then
    ts_el_cmip5_mean = weighted_mean(ts_el_cmip5, 1./tofloat(esizes_cmip5))
    ts_el_cmip5_5th = weighted_percentile(ts_el_cmip5, 1./tofloat(esizes_cmip5), 0.05)
    ts_el_cmip5_95th = weighted_percentile(ts_el_cmip5, 1./tofloat(esizes_cmip5), 0.95)
    ts_la_cmip5_mean = weighted_mean(ts_la_cmip5, 1./tofloat(esizes_cmip5))
    ts_la_cmip5_5th = weighted_percentile(ts_la_cmip5, 1./tofloat(esizes_cmip5), 0.05)
    ts_la_cmip5_95th = weighted_percentile(ts_la_cmip5, 1./tofloat(esizes_cmip5), 0.95)
    duration_el_cmip5_mean = weighted_mean(duration_el_cmip5, 1./tofloat(esizes_cmip5))
    duration_el_cmip5_5th  = weighted_percentile(duration_el_cmip5, 1./tofloat(esizes_cmip5), 0.05)
    duration_el_cmip5_25th = weighted_percentile(duration_el_cmip5, 1./tofloat(esizes_cmip5), 0.25)
    duration_el_cmip5_75th = weighted_percentile(duration_el_cmip5, 1./tofloat(esizes_cmip5), 0.75)
    duration_el_cmip5_95th = weighted_percentile(duration_el_cmip5, 1./tofloat(esizes_cmip5), 0.95)
    duration_la_cmip5_mean = weighted_mean(duration_la_cmip5, 1./tofloat(esizes_cmip5))
    duration_la_cmip5_5th  = weighted_percentile(duration_la_cmip5, 1./tofloat(esizes_cmip5), 0.05)
    duration_la_cmip5_25th = weighted_percentile(duration_la_cmip5, 1./tofloat(esizes_cmip5), 0.25)
    duration_la_cmip5_75th = weighted_percentile(duration_la_cmip5, 1./tofloat(esizes_cmip5), 0.75)
    duration_la_cmip5_95th = weighted_percentile(duration_la_cmip5, 1./tofloat(esizes_cmip5), 0.95)
  end if

  if (nruns_cmip6.gt.0) then
    ts_el_cmip6_mean = weighted_mean(ts_el_cmip6, 1./tofloat(esizes_cmip6))
    ts_el_cmip6_5th = weighted_percentile(ts_el_cmip6, 1./tofloat(esizes_cmip6), 0.05)
    ts_el_cmip6_95th = weighted_percentile(ts_el_cmip6, 1./tofloat(esizes_cmip6), 0.95)
    ts_la_cmip6_mean = weighted_mean(ts_la_cmip6, 1./tofloat(esizes_cmip6))
    ts_la_cmip6_5th = weighted_percentile(ts_la_cmip6, 1./tofloat(esizes_cmip6), 0.05)
    ts_la_cmip6_95th = weighted_percentile(ts_la_cmip6, 1./tofloat(esizes_cmip6), 0.95)
    duration_el_cmip6_mean = weighted_mean(duration_el_cmip6, 1./tofloat(esizes_cmip6))
    duration_el_cmip6_5th  = weighted_percentile(duration_el_cmip6, 1./tofloat(esizes_cmip6), 0.05)
    duration_el_cmip6_25th = weighted_percentile(duration_el_cmip6, 1./tofloat(esizes_cmip6), 0.25)
    duration_el_cmip6_75th = weighted_percentile(duration_el_cmip6, 1./tofloat(esizes_cmip6), 0.75)
    duration_el_cmip6_95th = weighted_percentile(duration_el_cmip6, 1./tofloat(esizes_cmip6), 0.95)
    duration_la_cmip6_mean = weighted_mean(duration_la_cmip6, 1./tofloat(esizes_cmip6))
    duration_la_cmip6_5th  = weighted_percentile(duration_la_cmip6, 1./tofloat(esizes_cmip6), 0.05)
    duration_la_cmip6_25th = weighted_percentile(duration_la_cmip6, 1./tofloat(esizes_cmip6), 0.25)
    duration_la_cmip6_75th = weighted_percentile(duration_la_cmip6, 1./tofloat(esizes_cmip6), 0.75)
    duration_la_cmip6_95th = weighted_percentile(duration_la_cmip6, 1./tofloat(esizes_cmip6), 0.95)
  end if

; ================================= Plot ================================
  
  wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                "enso_lifecycle")
  panel_width = 0.4
  dummy   = new(1000, "graphic")
  idummy = 0
  
; ========================= CMIP5, CMIP6 and obs ========================

  ts_plots = new(2, "graphic")
  vpX0 = 0.55 - panel_width
  vpX1 = 0.58
  vpY  = 0.95
  panel_height = panel_width * 0.56
  
  lag = ts_el_obs&$ts_el_obs!1$
  lagpoly = array_append_record(lag, lag(::-1), 0)
  tsize = dimsizes(lag)
  
  rests := True
  rests@gsnDraw = False
  rests@gsnFrame = False
  rests@vpWidthF = panel_width
  rests@vpHeightF = panel_height
  rests@xyDashPattern = 0
  rests@xyMonoLineColor = False
  rests@gsnYRefLine = 0.
  rests@trXMinF = min(lag)
  rests@trXMaxF = max(lag)
  rests@trYMinF = -2.4
  rests@trYMaxF = 2.4
  rests@tmXTOn = False
  rests@tmXBMode          = "Explicit"
  rests@tmXBLabelFontHeightF = 0.012
  rests@tmXBMajorThicknessF = 0.
  rests@tmXBMajorLengthF = 0.006
  rests@tmXBMinorLengthF = 0.006
  rests@tmXBMajorOutwardLengthF = 0.006
  rests@tmXBMinorOutwardLengthF = 0.006
  rests@tmXBValues = (/-18, -6, 6, 18, 30, 42/)
  rests@tmXBLabels = (/"year(-2)", "year(-1)", "year(0)", "year(1)", "year(2)", "year(3)"/)
  rests@tmXBMinorValues   = lag
  rests@tmYLMajorLengthF = 0.01
  rests@tmYLMajorOutwardLengthF = 0.01
  rests@tmXBLabelDeltaF = -0.6
  rests@tmYLLabelDeltaF = -0.5
  rests@tmYLLabelFontHeightF = 0.014
  rests@tmYLPrecision = 1
  rests@tmYROn = False
  rests@tmLabelAutoStride = False
  rests@tiYAxisFontHeightF = rests@tmYLLabelFontHeightF
  rests@tiYAxisOffsetXF = 0.012
  rests@tiYAxisString = "(~S~o~N~C)"
  
  resp = True
  resp@gsFillOpacityF = 0.1
  resp@gsLineThicknessF = 0.4
  resp@gsLineColor = "transparent"

  txres = True
  txres@txFontHeightF = 0.02
  txres@txJust = "centerleft"

  rests@tiMainFontHeightF = 0.018
  rests@tiMainOffsetYF = -0.015
  rests@xyLineThicknessF = 0.01

  rests@tiMainString = "(a) El Nino evolution"
  rests@vpXF = vpX0
  rests@vpYF = vpY
  ts_plots(0) \
    = gsn_csm_xy(wks, (/rests@trXMinF, rests@trXMaxF/), (/0., 0./), rests)
  rests@tiMainString = "(b) La Nina evolution"
  rests@vpXF = vpX1
  rests@tmYLLabelsOn = False
  rests@tiYAxisOn = False
  ts_plots(1) \
    = gsn_csm_xy(wks, (/rests@trXMinF, rests@trXMaxF/), (/0., 0./), rests)


  lres = True
  lres@gsLineThicknessF = 0.1
  lres@gsLineDashPattern =  2
  lres@gsLineOpacityF = 1.
  lres@gsLineColor := "grey50"
  do i = 0, 8
    dummy(idummy) = gsn_add_polyline(wks, ts_plots(0), (/-23.5+12.*i, -23.5+12*i/), \
                                     (/rests@trYMinF, rests@trYMaxF/), lres)
    idummy = idummy + 1
    dummy(idummy) = gsn_add_polyline(wks, ts_plots(1), (/-23.5+12.*i, -23.5+12*i/), \
                                     (/rests@trYMinF, rests@trYMaxF/), lres)
    idummy = idummy + 1
  end do

  rests@xyDashPattern = 0
  rests@xyLineThicknessF = 0.25
  res = True
  res@project = "CMIP6"
  color_cmip6 = get_color(res)
  res@project = "CMIP5"
  color_cmip5 = get_color(res)

  resp = True
  resp@gsFillOpacityF = 0.1
  if (nruns_cmip5.gt.0) then
    resp@gsFillColor := color_cmip5
    ts_el_cmip5_range = new((/2, tsize/), typeof(ts_el_cmip5_5th))
    ts_el_cmip5_range(0, :) = ts_el_cmip5_5th
    ts_el_cmip5_range(1, :) = ts_el_cmip5_95th
    dummy(idummy) = gsn_add_polygon(wks, ts_plots(0), \
                                    lagpoly, make_poly(ts_el_cmip5_range), resp)
    idummy = idummy + 1
  end if
  if (nruns_cmip6.gt.0) then
    resp@gsFillColor := color_cmip6
    ts_el_cmip6_range = new((/2, tsize/), typeof(ts_el_cmip6_5th))
    ts_el_cmip6_range(0, :) = ts_el_cmip6_5th
    ts_el_cmip6_range(1, :) = ts_el_cmip6_95th
    dummy(idummy) = gsn_add_polygon(wks, ts_plots(0), \
                                    lagpoly, make_poly(ts_el_cmip6_range), resp)
    idummy = idummy + 1
  end if

  lres = True
  lres@gsLineThicknessF = 3.
  lres@gsLineDashPattern =  0
  lres@gsLineOpacityF = 1.
  lres@gsLineColor := color_cmip5
  dummy(idummy) = gsn_add_polyline(wks, ts_plots(0), lag, ts_el_cmip5_mean, lres)
  idummy = idummy + 1
  lres@gsLineColor := color_cmip6
  dummy(idummy) = gsn_add_polyline(wks, ts_plots(0), lag, ts_el_cmip6_mean, lres)
  idummy = idummy + 1

  if (nruns_cmip5.gt.0) then
    resp@gsFillColor := color_cmip5
    ts_la_cmip5_range = new((/2, tsize/), typeof(ts_la_cmip5_5th))
    ts_la_cmip5_range(0, :) = ts_la_cmip5_5th
    ts_la_cmip5_range(1, :) = ts_la_cmip5_95th
    dummy(idummy) = gsn_add_polygon(wks, ts_plots(1), \
                                    lagpoly, make_poly(ts_la_cmip5_range), resp)
    idummy = idummy + 1
  end if
  if (nruns_cmip6.gt.0) then
    resp@gsFillColor := color_cmip6
    ts_la_cmip6_range = new((/2, tsize/), typeof(ts_la_cmip6_5th))
    ts_la_cmip6_range(0, :) = ts_la_cmip6_5th
    ts_la_cmip6_range(1, :) = ts_la_cmip6_95th
    dummy(idummy) = gsn_add_polygon(wks, ts_plots(1), \
                                    lagpoly, make_poly(ts_la_cmip6_range), resp)
    idummy = idummy + 1
  end if
  
  lres@gsLineColor := color_cmip5
  dummy(idummy) = gsn_add_polyline(wks, ts_plots(1), lag, ts_la_cmip5_mean, lres)
  idummy = idummy + 1
  lres@gsLineColor := color_cmip6
  dummy(idummy) = gsn_add_polyline(wks, ts_plots(1), lag, ts_la_cmip6_mean, lres)
  idummy = idummy + 1

  
  x1 = 0.03        ; for legends
  x2 = 0.065       ; for legends
  x = (x1 + x2)/2. ; for legends

  txres@txFontHeightF = 0.01
  txres@txJust = "BottomCenter"
  if (nruns_cmip5.gt.0) then
    resp@gsFillColor := color_cmip5
    y1 = vpY - 0.228
    y2 = y1 - 0.03
    gsn_polygon_ndc(wks, (/x1, x2, x2, x1/), (/y1, y1, y2, y2/), resp)
    y = (y1 + y2)/2.
    lres@gsLineColor := color_cmip5
    gsn_polyline_ndc(wks, (/x1, x2/), (/y, y/), lres)
    txres@txFontColor := color_cmip5
    gsn_text_ndc(wks, "CMIP5", x, y1+0.005, txres)
  end if
  if (nruns_cmip6.gt.0) then
    resp@gsFillColor := color_cmip6
    y1 = vpY - 0.09
    y2 = y1 - 0.03
    gsn_polygon_ndc(wks, (/x1, x2, x2, x1/), (/y1, y1, y2, y2/), resp)
    y = (y1 + y2)/2.
    lres@gsLineColor := color_cmip6
    gsn_polyline_ndc(wks, (/x1, x2/), (/y, y/), lres)
    txres@txFontColor := color_cmip6
    gsn_text_ndc(wks, "CMIP6", x, y1+0.005, txres)
    txres@txFontHeightF = 0.008
    txres@txJust = "CenterRight"
    txres@txFontColor := "black"
    gsn_text_ndc(wks, "95th", x1-0.002, y1, txres)
    gsn_text_ndc(wks, "mean", x1-0.002, y, txres)
    gsn_text_ndc(wks, "5th", x1-0.002, y2, txres)
  end if
  
  res_obs = True
  txres@txFontHeightF = 0.01
  txres@txJust = "TopCenter"
  do ii = 0, nobs-1
    res_obs@project = "OBS"
    res_obs@dataset = obs(ii)
    lres@gsLineColor = get_color(res_obs)
    lres@gsLineDashPattern =  get_lineindex(res_obs)
    lres@gsLineThicknessF = 1.5
    lres@gsLineOpacityF = 1.
    dummy(idummy) = gsn_add_polyline(wks, ts_plots(0), \
                                     lag, ts_el_obs(ii, :), lres)
    idummy = idummy + 1
    dummy(idummy) = gsn_add_polyline(wks, ts_plots(1), \
                                     lag, ts_la_obs(ii, :), lres)
    idummy = idummy + 1

    y = vpY - ii*0.03
    gsn_polyline_ndc(wks, (/x1, x2/), (/y, y/), lres)
    y = y - 0.006
    txres@txFontColor := get_color(res_obs)
    gsn_text_ndc(wks, obs(ii), x, y, txres)
  end do
  
  draw(ts_plots(0))
  draw(ts_plots(1))

; =========================== Individual Models =============================

  nraws = 0
  if (isvar("duration_el_cmip6_mean")) then
    nraws = nraws + 1
  end if
  if (isvar("duration_el_cmip5_mean")) then
    nraws = nraws + 1
  end if
  if (plot_each_cmip6) then
    nraws = nraws + count_unique_values(models_cmip6)
  end if
  if (plot_each_cmip5) then
    nraws = nraws + count_unique_values(models_cmip5)
  end if

  raw = ispan(0, nraws-1, 1)
  names := new(nraws, "string")
  names!0 = "raw"
  names&raw = raw
  iraw = 0
  if (isvar("duration_el_cmip6_mean")) then
    names(iraw) = "CMIP6"
    iraw = iraw + 1
  end if
  if (isvar("duration_el_cmip5_mean")) then
    names(iraw) = "CMIP5"
    iraw = iraw + 1
  end if
  if (plot_each_cmip6) then
    names(iraw:iraw+count_unique_values(models_cmip6)-1) = (/get_unique_values(models_cmip6)/)
    iraw = iraw + count_unique_values(models_cmip6)
  end if
  if (plot_each_cmip5) then
    names(iraw:iraw+count_unique_values(models_cmip5)-1) = (/get_unique_values(models_cmip5)/)
    iraw = iraw + count_unique_values(models_cmip5)
  end if
  
  mme_plots = new(2, "graphic")
  
  res := True
  res@gsnDraw = False
  res@gsnFrame = False
  res@xyCurveDrawOrder = "PreDraw"
  res@trYMinF = raw(0) - 0.6
  res@trYMaxF = raw(nraws-1) + 0.6
  res@trYReverse = True
  res@vpWidthF = panel_width
  res@vpHeightF = nraws * 0.0115
  res@vpYF = vpY - panel_height - 0.065

  res@tmXMajorGrid = True
  res@tmYMajorGrid = True
  res@tmXMajorGridLineDashPattern = 0
  res@tmYMajorGridLineDashPattern = 0
  res@tmXMajorGridLineColor = "grey"
  res@tmYMajorGridLineColor = "grey"
  res@tmXMajorGridThicknessF = 0.005
  res@tmYMajorGridThicknessF = 0.
  res@tmGridDrawOrder = "PreDraw"
  
  res@tmYROn = False
  res@tmYLMode = "Explicit"
  res@tmYLValues = raw
  res@tmYLLabels = names
  res@tmYLLabelJust = "CenterRight"
  res@tmYLLabelFontHeightF = 0.01
  res@tmYLLabelDeltaF = rests@tmYLLabelDeltaF
  res@tmYLMajorLengthF = 0.008
  res@tmYLMajorOutwardLengthF = res@tmYLMajorLengthF

  duration_max = max((/max(duration_el_cmip6), max(duration_el_cmip5), \
                       max(duration_la_cmip6), max(duration_la_cmip5)/))
  duration_min = max((/min(duration_el_cmip6), min(duration_el_cmip5), \
                       min(duration_la_cmip6), min(duration_la_cmip5)/))
  if (duration_max.lt.27) then
    res@trXMaxF = 28.
  elseif (duration_max.lt.29) then
    res@trXMaxF = 30.
  elseif (duration_max.lt.31) then
    res@trXMaxF = 32.
  elseif (duration_max.lt.33) then
    res@trXMaxF = 34.
  elseif (duration_max.lt.35) then
    res@trXMaxF = 36.
  else
    res@trXMaxF = 38.
  end if
  if (duration_min.gt.11) then
    res@trXMinF = 10.
  elseif (duration_min.gt.9) then
    res@trXMinF = 8.
  else
    res@trXMinF = 6.
  end if
  res@tmXTOn = False
  res@tmXTMode          = "Explicit"
  res@tmXTValues        = (/12, 18, 24, 30, 36/)
  res@tmXTLabels        = res@tmXTValues
  res@tmXBMode          = res@tmXTMode
  res@tmXBLabelFontHeightF = rests@tmYLLabelFontHeightF
  res@tmXBValues        = res@tmXTValues
  res@tmXBLabels        = res@tmXTLabels
  res@tmXBMajorLengthF = 0.008
  res@tmXBMajorOutwardLengthF = res@tmXBMajorLengthF
  res@tmXBLabelDeltaF = rests@tmXBLabelDeltaF
  res@tiXAxisFontHeightF = res@tmXBLabelFontHeightF
  res@tiXAxisOffsetYF = -0.004
  res@tiXAxisString = "(months)"
  
  res@tiMainFontHeightF = rests@tiMainFontHeightF
  res@tiMainOffsetYF = 0.
  res@tiDeltaF = 0.5
  
  mkres := True
  mkres@gsMarkerThicknessF = 2.0
  mkres@gsMarkerSizeF    = 0.006
  mkres@tfPolyDrawOrder = "PostDraw"
  lnres := True
  lnres@gsLineThicknessF = 2.
  lnres@tfPolyDrawOrder = "PostDraw"
  polyres := True
  polyres@tfPolyDrawOrder = "PostDraw"
  txres := True
  txres@txFontHeightF = 0.012
  
  res_obs@project = "OBS"
  res_obs@dataset = obs(0)
  res@xyLineColor = get_color(res_obs)
  res@xyDashPattern = get_lineindex(res_obs)
  res@xyLineThicknessF = 2.

  res@vpXF = vpX0
  res@tiMainString = "(c) Mean El Nino duration (month)"
  mme_plots(0) = gsn_csm_xy(wks, (/duration_el_obs(0), duration_el_obs(0)/), \
                            (/duration_el_obs@_FillValue, duration_el_obs@_FillValue/), res)
  res@vpXF = vpX1
  res@tmYLLabelsOn = False
  res@tiMainString = "(d) Mean La Nina duration (month)"
  mme_plots(1) = gsn_csm_xy(wks, (/duration_la_obs(0), duration_la_obs(0)/), \
                            (/duration_la_obs@_FillValue, duration_la_obs@_FillValue/), res)

  do j = 0, nobs-1
    res_obs@project = "OBS"
    res_obs@dataset = obs(j)
    lnres@gsLineColor = get_color(res_obs)
    lnres@gsLineDashPattern =  get_lineindex(res_obs)
    dummy(idummy) = gsn_add_polyline(wks, mme_plots(0), (/duration_el_obs(j), duration_el_obs(j)/), \
                                     (/res@trYMinF, res@trYMaxF/), lnres)
    idummy = idummy + 1
    dummy(idummy) = gsn_add_polyline(wks, mme_plots(1), (/duration_la_obs(j), duration_la_obs(j)/), \
                                     (/res@trYMinF, res@trYMaxF/), lnres)
    idummy = idummy + 1
  end do
  
  w = 0.4  ; half height of MME bars
  dsres := True
  if (any(names.eq."CMIP6")) then
    iraw = ind(names.eq."CMIP6")
    dsres@project = "CMIP6"
    polyres@gsFillColor = get_color(dsres)
    dummy(idummy) \
      = gsn_add_polygon(wks, mme_plots(0), \
                        (/duration_el_cmip6_25th, duration_el_cmip6_75th, \
                          duration_el_cmip6_75th, duration_el_cmip6_25th/), \
                        (/raw(iraw)-w, raw(iraw)-w, \
                          raw(iraw)+w, raw(iraw)+w/), polyres)
    idummy = idummy + 1
    dummy(idummy) \
      = gsn_add_polygon(wks, mme_plots(1), \
                        (/duration_la_cmip6_25th, duration_la_cmip6_75th, \
                          duration_la_cmip6_75th, duration_la_cmip6_25th/), \
                        (/raw(iraw)-w, raw(iraw)-w, \
                          raw(iraw)+w, raw(iraw)+w/), polyres)
    idummy = idummy + 1

    lnres@gsLineColor := get_color(dsres)
    dummy(idummy) = gsn_add_polyline(wks, mme_plots(0), \
                                     (/duration_el_cmip6_5th, duration_el_cmip6_95th/), \
                                     (/raw(iraw), raw(iraw)/), lnres)
    idummy = idummy + 1
    dummy(idummy) = gsn_add_polyline(wks, mme_plots(1), \
                                     (/duration_la_cmip6_5th, duration_la_cmip6_95th/), \
                                     (/raw(iraw), raw(iraw)/), lnres)
    idummy = idummy + 1
    lnres@gsLineColor := "white"
    dummy(idummy) = gsn_add_polyline(wks, mme_plots(0), \
                                     (/duration_el_cmip6_mean, duration_el_cmip6_mean/), \
                                     (/raw(iraw)-w, raw(iraw)+w/), lnres)
    idummy = idummy + 1
    dummy(idummy) = gsn_add_polyline(wks, mme_plots(1), \
                                     (/duration_la_cmip6_mean, duration_la_cmip6_mean/), \
                                     (/raw(iraw)-w, raw(iraw)+w/), lnres)
    idummy = idummy + 1
  end if
  if (any(names.eq."CMIP5")) then
    iraw = ind(names.eq."CMIP5")
    dsres@project = "CMIP5"
    polyres@gsFillColor = get_color(dsres)
    dummy(idummy) \
      = gsn_add_polygon(wks, mme_plots(0), \
                        (/duration_el_cmip5_25th, duration_el_cmip5_75th, \
                          duration_el_cmip5_75th, duration_el_cmip5_25th/), \
                        (/raw(iraw)-w, raw(iraw)-w, \
                          raw(iraw)+w, raw(iraw)+w/), polyres)
    idummy = idummy + 1
    dummy(idummy) \
      = gsn_add_polygon(wks, mme_plots(1), \
                        (/duration_la_cmip5_25th, duration_la_cmip5_75th, \
                          duration_la_cmip5_75th, duration_la_cmip5_25th/), \
                        (/raw(iraw)-w, raw(iraw)-w, \
                          raw(iraw)+w, raw(iraw)+w/), polyres)
    idummy = idummy + 1

    lnres@gsLineColor := get_color(dsres)
    dummy(idummy) = gsn_add_polyline(wks, mme_plots(0), \
                                     (/duration_el_cmip5_5th, duration_el_cmip5_95th/), \
                                     (/raw(iraw), raw(iraw)/), lnres)
    idummy = idummy + 1
    dummy(idummy) = gsn_add_polyline(wks, mme_plots(1), \
                                     (/duration_la_cmip5_5th, duration_la_cmip5_95th/), \
                                     (/raw(iraw), raw(iraw)/), lnres)
    idummy = idummy + 1
    lnres@gsLineColor := "white"
    dummy(idummy) = gsn_add_polyline(wks, mme_plots(0), \
                                     (/duration_el_cmip5_mean, duration_el_cmip5_mean/), \
                                     (/raw(iraw)-w, raw(iraw)+w/), lnres)
    idummy = idummy + 1
    dummy(idummy) = gsn_add_polyline(wks, mme_plots(1), \
                                     (/duration_la_cmip5_mean, duration_la_cmip5_mean/), \
                                     (/raw(iraw)-w, raw(iraw)+w/), lnres)
    idummy = idummy + 1
  end if

  dsres := True
  if (plot_each_cmip6) then
    do ii = 0, dimsizes(duration_el_cmip6)-1
      if (any(names.eq.models_cmip6(ii))) then
        iraw = ind(names.eq.models_cmip6(ii))
        dsres@dataset = models_cmip6(ii)
        dsres@project = "CMIP6"
        mkres@gsMarkerColor    = get_color(dsres)
        mkres@gsMarkerIndex    = get_markerindex(dsres)
        dummy(idummy) = gsn_add_polymarker(wks, mme_plots(0), \
                                           duration_el_cmip6(ii), raw(iraw), mkres)
        idummy = idummy + 1
        dummy(idummy) = gsn_add_polymarker(wks, mme_plots(1), \
                                           duration_la_cmip6(ii), raw(iraw), mkres)
        idummy = idummy + 1
      end if
    end do
  end if
  if (plot_each_cmip5) then
    do ii = 0, dimsizes(duration_el_cmip5)-1
      if (any(names.eq.models_cmip5(ii))) then
        iraw = ind(names.eq.models_cmip5(ii))
        dsres@dataset = models_cmip5(ii)
        dsres@project = "CMIP5"
        mkres@gsMarkerColor    = get_color(dsres)
        mkres@gsMarkerIndex    = get_markerindex(dsres)
        dummy(idummy) = gsn_add_polymarker(wks, mme_plots(0), \
                                           duration_el_cmip5(ii), raw(iraw), mkres)
        idummy = idummy + 1
        dummy(idummy) = gsn_add_polymarker(wks, mme_plots(1), \
                                           duration_la_cmip5(ii), raw(iraw), mkres)
        idummy = idummy + 1
      end if
    end do
  end if
  
  ; Legend

  x1 = 0.03
  x2 = 0.065
  y1 = vpY-0.13
  y2 = y1 - 0.011
  polyres@gsFillColor := color_cmip6
  gsn_polygon_ndc(wks, (/x1, x2, x2, x1/), (/y1, y1, y2, y2/), polyres)
  x0 = x1 - 0.015
  x3 = x2 + 0.015
  y = (y1 + y2) / 2.
  lnres@gsLineThicknessF = 2.
  lnres@gsLineColor := color_cmip6
  gsn_polyline_ndc(wks, (/x0, x3/), (/y, y/), lnres)
  x = (x1 + x2) / 2.
  lnres@gsLineColor := "white"
  gsn_polyline_ndc(wks, (/x, x/), (/y1, y2/), lnres)
  y = y - 0.002
  txres := True
  txres@txFontHeightF = 0.008
  txres@txJust = "TopCenter"  
  gsn_text_ndc(wks, "5th", x0, y, txres)
  gsn_text_ndc(wks, "95th", x3, y, txres)
  y = y2 - 0.006
  gsn_text_ndc(wks, "25th", x1, y, txres)
  gsn_text_ndc(wks, "75th", x2, y, txres)
  y = y2 - 0.018
  gsn_text_ndc(wks, "mean", x, y, txres)
  lnres@gsLineThicknessF = 0.5
  lnres@gsLineColor := "black"
  gsn_polyline_ndc(wks, (/x, x/), (/y+0.005, y2/), lnres)

  y1 = vpY-0.268
  y2 = y1 - 0.011
  polyres@gsFillColor := color_cmip5
  gsn_polygon_ndc(wks, (/x1, x2, x2, x1/), (/y1, y1, y2, y2/), polyres)
  x0 = x1 - 0.015
  x3 = x2 + 0.015
  y = (y1 + y2) / 2.
  lnres@gsLineThicknessF = 2.
  lnres@gsLineColor := color_cmip5
  gsn_polyline_ndc(wks, (/x0, x3/), (/y, y/), lnres)
  x = (x1 + x2) / 2.
  lnres@gsLineColor := "white"
  gsn_polyline_ndc(wks, (/x, x/), (/y1, y2/), lnres)
  
  x = 0.015
  y = vpY - 0.175
  gsn_polymarker_ndc(wks, x, y, mkres)
  txres@txJust = "CenterLeft"
  txres@txFontColor = color_cmip6
  gsn_text_ndc(wks, "Ensemble", x+0.008, y, txres)
  gsn_text_ndc(wks, " member", x+0.008, y-0.012, txres)

  x1 = 0.001
  x2 = 0.094
  y1 = vpY + 0.015
  y2 = vpY - 0.285
  lnres@gsLineThicknessF = 0.5
  lnres@gsLineColor := "black"
  gsn_polyline_ndc(wks, (/x1, x1, x2, x2, x1/), (/y1, y2, y2, y1, y1/), lnres)
  
  ; Figure title

  txres := True
  txres@txFontHeightF = 0.02
  txres@txFont = "helvetica-bold"
  txres@txJust = "TopCenter"  
  gsn_text_ndc(wks, "ENSO time evolution", 0.55, 0.9999, txres)
    
  draw(mme_plots(0))
  draw(mme_plots(1))

end
