; draw_seasonality
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs:
;
; History
;    20200905 kosaka_yu: refined the drawing part
;    20200811 kosaka_yu: (b): show individual ensemble members instead of compositing all members
;                        and plot them for individual models instead of modelling centers
;                        flip x and y axes to in plot ~50 models or more
;    20200511 kosaka_yu: cleaned to meet the code style.
;    20200105 kosaka_yu: show shading instead of lines
;    20191024 kosaka_yu: written.
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

function expand_season(x)
local y, dims
begin
  if (dimsizes(dimsizes(x)).eq.1) then
    y = new(14, typeof(x))
    y(1:12) = x
    y(0)  = (x(0)+x(11))/2.
    y(13) = (x(0)+x(11))/2.
  elseif (dimsizes(dimsizes(x)).eq.2) then
    dims = dimsizes(x)
    y = new((/dims(0), 14/), typeof(x))
    y(:, 1:12) = x
    y(:, 0)    = (x(:, 0)+x(:, 11))/2.
    y(:, 13)   = (x(:, 0)+x(:, 11))/2.
    delete(dims)
  end if
  return(y)
end

begin

  enter_msg(DIAG_SCRIPT, "")

  obs_projects = (/"OBS", "OBS6", "obs4mips", "ana4mips"/)

  syear = 1960
  eyear = 2014
  if (isatt(diag_script_info, "syear")) then
    syear = diag_script_info@syear
  end if
  if (isatt(diag_script_info, "eyear")) then
    eyear = diag_script_info@eyear
  end if

  detrend = True
  if (isatt(diag_script_info, "detrend")) then
    detrend = diag_script_info@detrend
  end if

  plot_each_cmip5 = False
  plot_each_cmip6 = True
  
; ==========================================================================
;                                Read files
; ==========================================================================
  
  ; Get file list
  input_dir = diag_script_info@input_files
  paths = systemfunc("ls "+input_dir+"/ENSOindex_*.nc")

  ; ========================================================================

  ; Get a list of models and obs
  nobs = 0
  nruns = 0
  do ii = 0, dimsizes(paths) - 1
    f = addfile(paths(ii), "r")
    ts = f->ENSOindex
    if (any(ts@project.eq.obs_projects)) then
      nobs = nobs + 1
    else
      nruns = nruns + 1
    end if
    delete(ts)
    delete(f)
  end do

  ; Read data and calculate seasonality for each run/obs
  stdv = new((/nruns, 12/), "float")
  seasonality = new(nruns, "float")
  projects = new(nruns, "string")
  models = new(nruns, "string")
  runs = new(nruns, "string")
  stdv_obs = new((/nobs, 12/), "float")
  seasonality_obs = new(nobs, "float")
  obs = new(nobs, "string")

  iobs = 0
  irun = 0
  do ii = 0, dimsizes(paths) - 1
    f = addfile(paths(ii), "r")
    ts = f->ENSOindex
    ts&time := toint(ts&time)
    syear_data = ts&time(0)/100
    smonth_data = ts&time(0)-syear_data*100
    eyear_data = ts&time(dimsizes(ts&time)-1)/100
    emonth_data = ts&time(dimsizes(ts&time)-1)-eyear_data*100

    ts_stdv  = new(12, typeof(ts))
    do mon = 1, 12
      if (syear.gt.syear_data) then
        t0 = syear*100+mon
      else
        if (mon.lt.smonth_data) then
          t0 = (syear_data+1)*100 + mon
        else
          t0 = syear_data*100 + mon
        end if
      end if
      if (eyear.lt.eyear_data) then
        t1 = eyear*100+mon
      else
        if (mon.lt.emonth_data) then
          t1 = (eyear_data-1)*100 + mon
        else
          t1 = eyear_data*100 + mon
        end if
      end if
      ts_stdv(mon-1)  = stddev(ts({t0:t1:12}))
    end do

    t0 = max((/syear, syear_data/))*100 + 12
    t1 = (min((/eyear, eyear_data/))-1)*100 + 12
    ndj = ts({t0:t1:12})
    t0 = max((/syear, syear_data/))*100 + 4
    t1 = (min((/eyear, eyear_data/))-1)*100 + 4
    mam = ts({t0:t1:12})
    do yr = max((/syear, syear_data/)), min((/eyear, eyear_data/))-1
      ndj({yr*100+12}) = avg(ts({yr*100+11:yr*100+101}))
      mam({yr*100+4}) = avg(ts({yr*100+3:yr*100+5}))
    end do

    if (any(ts@project.eq.obs_projects)) then
      stdv_obs(iobs, :) = (/ts_stdv/)
      seasonality_obs(iobs) = stddev(ndj)/stddev(mam)
      obs(iobs) = ts@dataset
      iobs = iobs + 1
    else
      stdv(irun, :) = (/ts_stdv/)
      seasonality(irun) = stddev(ndj)/stddev(mam)
      projects(irun) = ts@project
      models(irun) = ts@dataset
      runs(irun) = ts@ensemble
      irun = irun + 1
    end if
    delete(ndj)
    delete(mam)
    delete(ts_stdv)
    delete(ts)
  end do

  ; ========================================================================

  ; Calculate multimodel statistics
  ; MME mean based on mean of std dev instead of sqrt of mean variance
  nruns_cmip5 = num(projects.eq."CMIP5")
  nruns_cmip6 = num(projects.eq."CMIP6")
  if (nruns_cmip5.gt.0) then
    stdv_cmip5 = new((/nruns_cmip5, 12/), "float")
    esize_cmip5 = new(nruns_cmip5, "integer")
    seasonality_cmip5 = new(nruns_cmip5, "float")
  end if
  if (nruns_cmip6.gt.0) then
    stdv_cmip6 = new((/nruns_cmip6, 12/), "float")
    esize_cmip6 = new(nruns_cmip6, "integer")
    seasonality_cmip6 = new(nruns_cmip6, "float")
  end if
  irun_cmip5 = 0
  irun_cmip6 = 0
  do ii = 0, nruns-1
    esize = num(projects(ii)+"-"+models(ii).eq.projects+"-"+models)
    if (projects(ii).eq."CMIP5") then
      stdv_cmip5(irun_cmip5, :) = stdv(ii, :)
      esize_cmip5(irun_cmip5) = esize
      seasonality_cmip5(irun_cmip5) = seasonality(ii)
      irun_cmip5 = irun_cmip5 + 1
    elseif (projects(ii).eq."CMIP6") then
      stdv_cmip6(irun_cmip6, :) = stdv(ii, :)
      esize_cmip6(irun_cmip6) = esize
      seasonality_cmip6(irun_cmip6) = seasonality(ii)
      irun_cmip6 = irun_cmip6 + 1
    end if
  end do

  if (nruns_cmip5.gt.0) then
    stdv_cmip5_mean = weighted_mean(stdv_cmip5, 1./tofloat(esize_cmip5))
    stdv_cmip5_5th = weighted_percentile(stdv_cmip5, 1./tofloat(esize_cmip5), 0.05)
    stdv_cmip5_95th = weighted_percentile(stdv_cmip5, 1./tofloat(esize_cmip5), 0.95)
    seasonality_cmip5_mean = weighted_mean(seasonality_cmip5, 1./tofloat(esize_cmip5))
    seasonality_cmip5_5th  = weighted_percentile(seasonality_cmip5, 1./tofloat(esize_cmip5), 0.05)
    seasonality_cmip5_25th = weighted_percentile(seasonality_cmip5, 1./tofloat(esize_cmip5), 0.25)
    seasonality_cmip5_75th = weighted_percentile(seasonality_cmip5, 1./tofloat(esize_cmip5), 0.75)
    seasonality_cmip5_95th = weighted_percentile(seasonality_cmip5, 1./tofloat(esize_cmip5), 0.95)
  end if

  if (nruns_cmip6.gt.0) then
    stdv_cmip6_mean = weighted_mean(stdv_cmip6, 1./tofloat(esize_cmip6))
    stdv_cmip6_5th = weighted_percentile(stdv_cmip6, 1./tofloat(esize_cmip6), 0.05)
    stdv_cmip6_95th = weighted_percentile(stdv_cmip6, 1./tofloat(esize_cmip6), 0.95)
    seasonality_cmip6_mean = weighted_mean(seasonality_cmip6, 1./tofloat(esize_cmip6))
    seasonality_cmip6_5th  = weighted_percentile(seasonality_cmip6, 1./tofloat(esize_cmip6), 0.05)
    seasonality_cmip6_25th = weighted_percentile(seasonality_cmip6, 1./tofloat(esize_cmip6), 0.25)
    seasonality_cmip6_75th = weighted_percentile(seasonality_cmip6, 1./tofloat(esize_cmip6), 0.75)
    seasonality_cmip6_95th = weighted_percentile(seasonality_cmip6, 1./tofloat(esize_cmip6), 0.95)
  end if

; ==========================================================================
;                               Draw figure
; ==========================================================================

  wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                "enso_seasonality")
  dummy   = new(500, "graphic")
  idummy = 0

  panel_width = 0.4

  res_cmip = True
  res_cmip@dataset = "CMIP5"
  res_cmip@project = "CMIP5"
  color_cmip5 = get_color(res_cmip)
  res_cmip@dataset = "CMIP6"
  res_cmip@project = "CMIP6"
  color_cmip6 = get_color(res_cmip)
  
  ; Legend

  vpX = 0.5 - panel_width / 2.
  vpY = 0.94 ; rests@vpYF
  xLegend1 = vpX - 0.155 ;rests@vpXF - 0.155
  xLegend2 = xLegend1 + 0.093 ;0.094
  yLegend1 = vpY + 0.015
  yLegend2 = yLegend1 - 0.3 ;vpY - 0.285
  lnres = True
  lnres@gsLineThicknessF = 0.5
  lnres@gsLineColor := "black"
  gsn_polyline_ndc(wks, (/xLegend1, xLegend1, xLegend2, xLegend2, xLegend1/), \
                   (/yLegend1, yLegend2, yLegend2, yLegend1, yLegend1/), lnres)
  
; =========================== CMIP5, CMIP6 and obs ============================
  
  month = (/ 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 12.5/)
  monthpoly = array_append_record(month, month(::-1), 0)

  ; for legend box
;  x0 = 2.5
;  x3 = 8.35
;  y0 = 1.57
;  y3 = 2.
  
  rests := True
  rests@gsnDraw = False
  rests@gsnFrame = False
  rests@xyCurveDrawOrder = "PreDraw"
  rests@vpWidthF = panel_width
  rests@vpHeightF = 0.2
  rests@vpXF = vpX ; 0.5 - panel_width / 2.
  rests@vpYF = vpY ; 0.94
  
  rests@xyDashPattern = 0
  rests@xyMonoLineColor = False
  rests@gsnYRefLine = 0.
  rests@trXMinF = month(0)
  rests@trXMaxF = month(dimsizes(month)-1)
  rests@trYMinF = 0.2
  rests@trYMaxF = 2.05
  rests@tmXTOn = False
  rests@tmYROn = False
  rests@tmYLMode = "Explicit"
  rests@tmXBMode = "Explicit"
  rests@tmXBLabelFontHeightF = 0.014
;  rests@tmXBMajorThicknessF = 0.
;  rests@tmXBMinorThicknessF = 2.
  rests@tmXBLabelDeltaF = -0.6
  rests@tmYLLabelFontHeightF = 0.014
  rests@tmYLLabelDeltaF = -0.7
  rests@tmYLValues = (/0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2./)
  rests@tmYLLabels = rests@tmYLValues
  rests@tiYAxisFontHeightF = rests@tmYLLabelFontHeightF
  rests@tiYAxisOffsetXF = -0.0  ; -0.012
  rests@tiYAxisString = "(~S~o~N~C)"
  rests@tmXBValues = ispan(1, 12, 1)
  rests@tmXBLabels = (/"J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"/)
;  rests@tmXBMinorValues = ispan(1, 12, 1)
  rests@tmLabelAutoStride = False
  rests@tiMainFontHeightF = 0.018
  rests@tiDeltaF = 0.5
;  rests@tiMainOffsetYF = -0.005
  
  rests@xyLineThicknessF = 0.01
  rests@xyLineThicknessF = 0.01
  rests@tiMainString = "(a) Std deviation of the ENSO index"
  seasonality_plot = gsn_csm_xy(wks, (/rests@trXMinF, rests@trXMaxF/), \
                                (/0., 0./), rests)

  resp = True
  resp@tfPolyDrawOrder = "PreDraw"
  resp@gsLineThicknessF = 0.4
  resp@gsLineColor = "transparent"

  txres = True
  txres@txFontHeightF = 0.01
  txres@txJust = "centerleft"

  resp@gsFillColor = color_cmip5
  resp@gsFillOpacityF = 0.1
  stdv_cmip5_range = new((/2, 12/), typeof(stdv_cmip5_mean))
  stdv_cmip5_range(0, :) = stdv_cmip5_95th
  stdv_cmip5_range(1, :) = stdv_cmip5_5th
  dummy(idummy) = gsn_add_polygon(wks, seasonality_plot, monthpoly, \
                                  make_poly(expand_season(stdv_cmip5_range)), resp)
  idummy = idummy + 1
  resp@gsFillColor = color_cmip6
  stdv_cmip6_range = new((/2, 12/), typeof(stdv_cmip6_mean))
  stdv_cmip6_range(0, :) = stdv_cmip6_95th
  stdv_cmip6_range(1, :) = stdv_cmip6_5th
  dummy(idummy) = gsn_add_polygon(wks, seasonality_plot, monthpoly, \
                                  make_poly(expand_season(stdv_cmip6_range)), resp)
  idummy = idummy + 1

  rests2 = True
  rests2@tfPolyDrawOrder = "PostDraw"
  rests2@gsnDraw = False
  rests2@gsnFrame = False
  rests2@xyDashPattern = 0
  rests2@xyLineThicknessF = 0.25

  rests2@xyLineColor = color_cmip6
  rests2@xyDashPattern =  get_lineindex(res_cmip)
  rests2@xyLineThicknessF = 3.
  rests2@xyLineOpacityF = 1.
  plot = gsn_csm_xy(wks, month, expand_season(stdv_cmip6_mean), rests2)
  overlay(seasonality_plot, plot)
  rests2@xyLineColor = color_cmip5
  rests2@xyDashPattern =  get_lineindex(res_cmip)
  rests2@xyLineThicknessF = 3.
  plot = gsn_csm_xy(wks, month, expand_season(stdv_cmip5_mean), rests2)
  overlay(seasonality_plot, plot)

  ; for legend
  x1 = xLegend1 + 0.03
  x2 = x1 + 0.035
  x = (x1 + x2)/2.
  
  res_obs = True
  lres = True
  lres@gsLineThicknessF = 1.5
  do i = 0, nobs-1
    res_obs@project = "OBS"
    res_obs@dataset = obs(i)
    lres@gsLineColor = get_color(res_obs)
    lres@gsLineDashPattern = get_lineindex(res_obs)
    dummy(idummy) = gsn_add_polyline(wks, seasonality_plot, \
                                     month, expand_season(stdv_obs(i, :)), lres)
    idummy = idummy + 1

    ; Legend
    y = vpY - i*0.03
    gsn_polyline_ndc(wks, (/x1, x2/), (/y, y/), lres)
    y = y - 0.006
    txres@txFontColor := get_color(res_obs)
    txres@txJust = "TopCenter"
    gsn_text_ndc(wks, obs(i), x, y, txres)
  end do

  ; Legend

  txres@txFontHeightF = 0.01
  txres@txJust = "BottomCenter"
  resp@gsFillColor := color_cmip5

  y1 = vpY - 0.228
  y2 = y1 - 0.03
  gsn_polygon_ndc(wks, (/x1, x2, x2, x1/), (/y1, y1, y2, y2/), resp)
  y = (y1 + y2)/2.
  lres@gsLineColor := color_cmip5
  gsn_polyline_ndc(wks, (/x1, x2/), (/y, y/), lres)
  txres@txFontColor := color_cmip5
  gsn_text_ndc(wks, "CMIP5", x, y1+0.005, txres)

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
  
  draw(seasonality_plot)

; ============================ Individual Models =============================

  models_cmip5 = get_unique_values(models(ind(projects.eq."CMIP5")))
  models_cmip6 = get_unique_values(models(ind(projects.eq."CMIP6")))
  
  nraws = 0
  if (isvar("seasonality_cmip6_mean")) then
    nraws = nraws + 1
  end if
  if (isvar("seasonality_cmip5_mean")) then
    nraws = nraws + 1
  end if
  if (plot_each_cmip6) then
    nraws = nraws + dimsizes(models_cmip6)
  end if
  if (plot_each_cmip5) then
    nraws = nraws + dimsizes(models_cmip5)
  end if

  raw = ispan(0, nraws-1, 1)
  names = new(nraws, "string")
  names!0 = "raw"
  names&raw = raw
  iraw = 0
  if (isvar("seasonality_cmip6_mean")) then
    names(iraw) = "CMIP6"
    iraw = iraw + 1
  end if
  if (isvar("seasonality_cmip5_mean")) then
    names(iraw) = "CMIP5"
    iraw = iraw + 1
  end if
  if (plot_each_cmip6) then
    names(iraw:iraw+dimsizes(models_cmip6)-1) = (/models_cmip6/)
    iraw = iraw + dimsizes(models_cmip6)
  end if
  if (plot_each_cmip5) then
    names(iraw:iraw+dimsizes(models_cmip5)-1) = (/models_cmip5/)
    iraw = iraw + dimsizes(models_cmip5)
  end if

  res := True
  res@gsnDraw = False
  res@gsnFrame = False
  res@xyCurveDrawOrder = "PreDraw"
  res@trYMinF = raw(0) - 0.6
  res@trYMaxF = raw(nraws-1) + 0.6
  res@trYReverse = True
  res@vpWidthF = panel_width
  res@vpHeightF = nraws * 0.0115
  res@vpXF = 0.5 - panel_width / 2.
  res@vpYF = rests@vpYF - rests@vpHeightF - 0.08

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
  res@tmYLLabelDeltaF = -0.5 ;rests@tmYLLabelDeltaF
  res@tmYLMajorLengthF = 0.008
  res@tmYLMajorOutwardLengthF = res@tmYLMajorLengthF
  
  res@trXMaxF = 2.5
  res@trXMinF = 0.5
  res@tmXTOn = False
  res@tmXTMode          = "Explicit"
  res@tmXTValues        = (/0., 0.5, 1., 1.5, 2., 2.5, 3./)
  res@tmXTLabels        = res@tmXTValues
  res@tmXBMode          = res@tmXTMode
  res@tmXBLabelFontHeightF = rests@tmXBLabelFontHeightF
  res@tmXBValues        = res@tmXTValues
  res@tmXBLabels        = res@tmXTLabels
  res@tmXBMajorLengthF = 0.008
  res@tmXBMajorOutwardLengthF = res@tmXBMajorLengthF
  res@tmXBLabelDeltaF = rests@tmXBLabelDeltaF

  res@tiMainFontHeightF = rests@tiMainFontHeightF
  res@tiMainOffsetYF = 0.005
  res@tiDeltaF = rests@tiDeltaF
  
  mkres = True
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
  res@tiMainString = "(b) Seasonality metric (NDJ/MAM ratio)"
  mme_plot = gsn_csm_xy(wks, (/seasonality_obs(0), seasonality_obs(0)/), \
                        (/seasonality_obs@_FillValue, seasonality_obs@_FillValue/), res)
  do j = 0, nobs-1
    res_obs@project = "OBS"
    res_obs@dataset = obs(j)
    lnres@gsLineColor = get_color(res_obs)
    lnres@gsLineDashPattern =  get_lineindex(res_obs)
    dummy(idummy) = gsn_add_polyline(wks, mme_plot, (/seasonality_obs(j), seasonality_obs(j)/), \
                                     (/res@trYMinF, res@trYMaxF/), lnres)
    idummy = idummy + 1
  end do
  
  w = 0.4  ; half height of MME bars
  dsres := True
  if (any(names.eq."CMIP6")) then
    iraw = ind(names.eq."CMIP6")
    polyres@gsFillColor = color_cmip6
    dummy(idummy) \
      = gsn_add_polygon(wks, mme_plot, \
                        (/seasonality_cmip6_25th, seasonality_cmip6_75th, \
                          seasonality_cmip6_75th, seasonality_cmip6_25th/), \
                        (/raw(iraw)-w, raw(iraw)-w, \
                          raw(iraw)+w, raw(iraw)+w/), polyres)
    idummy = idummy + 1
    lnres@gsLineColor := color_cmip6
    dummy(idummy) = gsn_add_polyline(wks, mme_plot, \
                                     (/seasonality_cmip6_5th, seasonality_cmip6_95th/), \
                                     (/raw(iraw), raw(iraw)/), lnres)
    idummy = idummy + 1
    lnres@gsLineColor := "white"
    dummy(idummy) = gsn_add_polyline(wks, mme_plot, \
                                     (/seasonality_cmip6_mean, seasonality_cmip6_mean/), \
                                     (/raw(iraw)-w, raw(iraw)+w/), lnres)
    idummy = idummy + 1
  end if
  if (any(names.eq."CMIP5")) then
    iraw = ind(names.eq."CMIP5")
    polyres@gsFillColor = color_cmip5
    dummy(idummy) \
      = gsn_add_polygon(wks, mme_plot, \
                        (/seasonality_cmip5_25th, seasonality_cmip5_75th, \
                          seasonality_cmip5_75th, seasonality_cmip5_25th/), \
                        (/raw(iraw)-w, raw(iraw)-w, \
                          raw(iraw)+w, raw(iraw)+w/), polyres)
    idummy = idummy + 1
    lnres@gsLineColor := color_cmip5
    dummy(idummy) = gsn_add_polyline(wks, mme_plot, \
                                     (/seasonality_cmip5_5th, seasonality_cmip5_95th/), \
                                     (/raw(iraw), raw(iraw)/), lnres)
    idummy = idummy + 1
    lnres@gsLineColor := "white"
    dummy(idummy) = gsn_add_polyline(wks, mme_plot, \
                                     (/seasonality_cmip5_mean, seasonality_cmip5_mean/), \
                                     (/raw(iraw)-w, raw(iraw)+w/), lnres)
    idummy = idummy + 1
  end if

  dsres := True
  do ii = 0, nruns-1
    if (any(names.eq.models(ii))) then
      iraw = ind(names.eq.models(ii))
      dsres@dataset = models(ii)
      dsres@project = projects(ii)
      mkres@gsMarkerColor    = get_color(dsres)
      mkres@gsMarkerIndex    = get_markerindex(dsres)
      dummy(idummy) = gsn_add_polymarker(wks, mme_plot, \
                                         seasonality(ii), raw(iraw), mkres)
      idummy = idummy + 1
    end if
  end do

  ; Legend

;  vpY = rests@vpYF
;  xLegend1 = rests@vpXF - 0.155
;  xLegend2 = xLegend1 + 0.093 ;0.094
;  yLegend1 = vpY + 0.015
;  yLegend2 = yLegend1 - 0.3 ;vpY - 0.285
;  lnres@gsLineThicknessF = 0.5
;  lnres@gsLineColor := "black"
;  gsn_polyline_ndc(wks, (/xLegend1, xLegend1, xLegend2, xLegend2, xLegend1/), \
;                   (/yLegend1, yLegend2, yLegend2, yLegend1, yLegend1/), lnres)
  
  x1 = xLegend1 + 0.03
  x2 = x1 + 0.035
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
  
  x = x1 - 0.015
  y = vpY - 0.175
  gsn_polymarker_ndc(wks, x, y, mkres)
  txres@txJust = "CenterLeft"
  txres@txFontColor = color_cmip6
  gsn_text_ndc(wks, "Ensemble", x+0.008, y, txres)
  gsn_text_ndc(wks, " member", x+0.008, y-0.012, txres)

  ; Figure title

  txres := True
  txres@txFontHeightF = 0.02
  txres@txFont = "helvetica-bold"
  txres@txJust = "TopCenter"  
  gsn_text_ndc(wks, "ENSO seasonality", 0.5, 0.9999, txres)
    
  draw(mme_plot)

end
