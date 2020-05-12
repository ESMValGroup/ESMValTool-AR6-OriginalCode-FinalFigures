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

load "$diag_scripts/ar6ch3_enso/modelling_center.ncl"
load "$diag_scripts/ar6ch3_enso/setting_mme.ncl"

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

;  disabled the shading
;  add_shading = False

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

  ; ========================================================================

  ; Get file list
  input_dir = diag_script_info@input_files
  paths = systemfunc("ls "+input_dir+"/ENSOindex_*.nc")

  ; ========================================================================

  ; Get a list of models and obs
  nobs = 0
  nmodels = 0
  do ii = 0, dimsizes(paths) - 1
    f = addfile(paths(ii), "r")
    ts = f->ENSOindex
    if (any(ts@project.eq.obs_projects)) then
      nobs = nobs + 1
    else
      nmodels = nmodels + 1
    end if
    delete(ts)
    delete(f)
  end do
  obs_all    = new(nobs,    "string")
  models_all = new(nmodels, "string")

  i = 0
  j = 0
  do ii = 0, dimsizes(paths) - 1
    f = addfile(paths(ii), "r")
    ts = f->ENSOindex
    if (any(ts@project.eq.obs_projects)) then
      obs_all(i) = ts@dataset
      i = i + 1
    else
      models_all(j) = ts@dataset
      j = j + 1
    end if
    delete(ts)
    delete(f)
  end do
  models = get_unique_values(models_all)
  nmodels = dimsizes(models)
  projects = models
  projects = ""
  obs = get_unique_values(obs_all)
  nobs = dimsizes(obs)

  ; Read data
  mean = new((/nmodels, 12/), "float")
  var  = new((/nmodels, 12/), "float")
  nyrs = new((/nmodels, 12/), "integer")
  ndj_mean = new(nmodels, "float")
  ndj_var  = new(nmodels, "float")
  nyrs_ndj = new(nmodels, "integer")
  mam_mean = new(nmodels, "float")
  mam_var = new(nmodels, "float")
  nyrs_mam = new(nmodels, "integer")
  mean_obs = new((/nobs, 12/), "float")
  var_obs  = new((/nobs, 12/), "float")
  nyrs_obs = new((/nobs, 12/), "integer")
  ndj_mean_obs = new(nobs, "float")
  ndj_var_obs  = new(nobs, "float")
  nyrs_ndj_obs = new(nobs, "integer")
  mam_mean_obs = new(nobs, "float")
  mam_var_obs = new(nobs, "float")
  nyrs_mam_obs = new(nobs, "integer")
  mean = 0.
  var  = 0.
  nyrs = 0
  ndj_mean = 0.
  ndj_var = 0.
  nyrs_ndj = 0
  mam_mean = 0.
  mam_var = 0.
  nyrs_mam = 0
  mean_obs = 0.
  var_obs  = 0.
  nyrs_obs = 0
  ndj_mean_obs = 0.
  ndj_var_obs = 0.
  nyrs_ndj_obs = 0
  mam_mean_obs = 0.
  mam_var_obs = 0.
  nyrs_mam_obs = 0
  do ii = 0, dimsizes(paths) - 1
    f = addfile(paths(ii), "r")
    ts = f->ENSOindex
    ts&time := toint(ts&time)
    syear_data = ts&time(0)/100
    smonth_data = ts&time(0)-syear_data*100
    eyear_data = ts&time(dimsizes(ts&time)-1)/100
    emonth_data = ts&time(dimsizes(ts&time)-1)-eyear_data*100

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

    x  = new(12, typeof(ts))
    xx = new(12, typeof(ts))
    nx = new(12, "integer")
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
          t0 = (eyear_data-1)*100 + mon
        else
          t0 = eyear_data*100 + mon
        end if
      end if
      x(mon-1)  = sum(ts({t0:t1:12}))
      xx(mon-1) = sum(ts({t0:t1:12})*ts({t0:t1:12}))
      nx(mon-1) = num(.not.ismissing(ts({t0:t1:12})))
    end do

    if (any(ts@project.eq.obs_projects)) then
      i = ind(ts@dataset.eq.obs)
      mean_obs(i, :) = mean_obs(i, :) + tofloat(x)
      var_obs(i, :)  = var_obs(i, :) + tofloat(xx)
      nyrs_obs(i, :) = nyrs_obs(i, :) + nx
      ndj_mean_obs(i) = ndj_mean_obs(i) + tofloat(sum(ndj))
      ndj_var_obs(i)  = ndj_var_obs(i)  + tofloat(sum(ndj ^ 2))
      nyrs_ndj_obs(i) = nyrs_ndj_obs(i) + num(.not.ismissing(ndj))
      mam_mean_obs(i) = mam_mean_obs(i) + tofloat(sum(mam))
      mam_var_obs(i)  = mam_var_obs(i)  + tofloat(sum(mam ^ 2))
      nyrs_mam_obs(i) = nyrs_mam_obs(i) + num(.not.ismissing(mam))
    else
      i = ind(ts@dataset.eq.models)
      projects(i) = ts@project
      mean(i, :)  = mean(i, :) + tofloat(x)
      var(i, :)   = var(i, :)  + tofloat(xx)
      nyrs(i, :) = nyrs(i, :) + nx
      ndj_mean(i) = ndj_mean(i) + tofloat(sum(ndj))
      ndj_var(i)  = ndj_var(i)  + tofloat(sum(ndj ^ 2))
      nyrs_ndj(i) = nyrs_ndj(i) + num(.not.ismissing(ndj))
      mam_mean(i) = mam_mean(i) + tofloat(sum(mam))
      mam_var(i)  = mam_var(i)  + tofloat(sum(mam ^ 2))
      nyrs_mam(i) = nyrs_mam(i) + num(.not.ismissing(mam))
    end if
    delete(x)
    delete(xx)
    delete(nx)
    delete(ndj)
    delete(mam)
    delete(ts)
  end do
  mean_obs = mean_obs / tofloat(nyrs_obs)
  var_obs  = var_obs / tofloat(nyrs_obs - 1) \
    - mean_obs ^ 2*tofloat(nyrs_obs)/tofloat(nyrs_obs-1)
  stdv_obs = sqrt(var_obs)

  ndj_mean_obs = ndj_mean_obs / tofloat(nyrs_ndj_obs)
  ndj_var_obs  = ndj_var_obs / tofloat(nyrs_ndj_obs - 1) \
    - ndj_mean_obs ^ 2*tofloat(nyrs_ndj_obs)/tofloat(nyrs_ndj_obs-1)
  mam_mean_obs = mam_mean_obs / tofloat(nyrs_mam_obs)
  mam_var_obs  = mam_var_obs / tofloat(nyrs_mam_obs - 1) \
    - mam_mean_obs ^ 2*tofloat(nyrs_mam_obs)/tofloat(nyrs_mam_obs-1)
  seasonality_obs = sqrt(ndj_var_obs/mam_var_obs)

  mean = mean / tofloat(nyrs)
  var  = var / tofloat(nyrs - 1) - mean ^ 2*tofloat(nyrs)/tofloat(nyrs-1)
  stdv = sqrt(var)
  ndj_mean = ndj_mean / tofloat(nyrs_ndj)
  ndj_var  = ndj_var / tofloat(nyrs_ndj - 1) \
    - ndj_mean ^ 2*tofloat(nyrs_ndj)/tofloat(nyrs_ndj - 1)
  mam_mean = mam_mean / tofloat(nyrs_mam)
  mam_var  = mam_var / tofloat(nyrs_mam - 1) \
    - mam_mean ^ 2*tofloat(nyrs_mam)/tofloat(nyrs_mam - 1)
  seasonality = sqrt(ndj_var/mam_var)

  stdv_cmip5mean = new(12, "float")
  stdv_cmip6mean = new(12, "float")
  stdv_cmip5mean = 0.
  stdv_cmip6mean = 0.
  nmodels_cmip5 = 0
  nmodels_cmip6 = 0
  do i = 0, nmodels-1
    if (projects(i).eq."CMIP5") then
      if (isvar("stdv_cmip5")) then
        tmp = stdv_cmip5
        delete(stdv_cmip5)
        dims = dimsizes(tmp)
        stdv_cmip5 = new((/dims(0)+1, dims(1)/), typeof(tmp))
        stdv_cmip5(0:dims(0)-1, :) = tmp
        stdv_cmip5(dims(0), :) = stdv(i, :)
        delete(tmp)
        delete(dims)
      else
        stdv_cmip5 = stdv(0:0, :)
        stdv_cmip5(0, :) = stdv(i, :)
      end if
      stdv_cmip5mean = stdv_cmip5mean + stdv(i, :)
      nmodels_cmip5 = nmodels_cmip5+1
    elseif (projects(i).eq."CMIP6") then
      if (isvar("stdv_cmip6")) then
        tmp = stdv_cmip6
        delete(stdv_cmip6)
        dims = dimsizes(tmp)
        stdv_cmip6 = new((/dims(0)+1, dims(1)/), typeof(tmp))
        stdv_cmip6(0:dims(0)-1, :) = tmp
        stdv_cmip6(dims(0), :) = stdv(i, :)
        delete(tmp)
        delete(dims)
      else
        stdv_cmip6 = stdv(0:0, :)
        stdv_cmip6(0, :) = stdv(i, :)
      end if
      stdv_cmip6mean = stdv_cmip6mean + stdv(i, :)
      nmodels_cmip6 = nmodels_cmip6+1
    end if
  end do
  stdv_cmip5mean = stdv_cmip5mean / nmodels_cmip5
  stdv_cmip6mean = stdv_cmip6mean / nmodels_cmip6

  wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                "enso_seasonality")

; =========================== CMIP5, CMIP6 and obs ============================

  dummy   = new(500, "graphic")
  idummy = 0

  month = (/ 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 12.5/)
  monthpoly = array_append_record(month, month(::-1), 0)

  rests := True
  rests@gsnDraw = False
  rests@gsnFrame = False
  rests@vpWidthF = 0.5
  rests@vpHeightF = 0.3
  rests@xyDashPattern = 0
  rests@xyMonoLineColor = False
  rests@gsnYRefLine = 0.
  rests@trXMinF = month(0)
  rests@trXMaxF = month(dimsizes(month)-1)
  rests@trYMinF = 0.2
  rests@trYMaxF = 2.0
  rests@tmXTOn = False
  rests@tmYLMode = "Explicit"
  rests@tmXBMode = "Explicit"
  rests@tmXBLabelFontHeightF = 0.014
  rests@tmXBMajorThicknessF = 0.
  rests@tmYLLabelFontHeightF = 0.014
  rests@tmYLValues = (/0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2./)
  rests@tmYLLabels = rests@tmYLValues
  rests@tmXBValues = (/1, 3, 5, 7, 9, 11/)
  rests@tmXBLabels = (/"Jan", "Mar", "May", "Jul", "Sep", "Nov"/)
  rests@tmXBMinorValues = ispan(1, 12, 1)
  rests@tmLabelAutoStride = False

  resp = True
  resp@gsFillOpacityF = 0.1
  resp@gsLineThicknessF = 0.4
  resp@gsLineColor = "transparent"

  txres = True
  txres@txFontHeightF = 0.012
  txres@txJust = "centerleft"

  rests@xyLineThicknessF = 0.01
  rests@tiMainFontHeightF = 0.02
  rests@tiMainOffsetYF = -0.01
  rests@xyLineThicknessF = 0.01
  rests@tiMainString = "(a) ENSO seasonality"
  seasonality_plot = gsn_csm_xy(wks, (/rests@trXMinF, rests@trXMaxF/), \
                                (/0., 0./), rests)

  rests@xyDashPattern = 0
  rests@xyLineThicknessF = 0.25
  res_cmip = True
  res_cmip@dataset = "CMIP5"
  res_cmip@project = "CMIP5"
  resp@gsFillColor = get_color(res_cmip)
  resp@gsFillOpacityF = 0.2
  dummy(idummy) = gsn_add_polygon(wks, seasonality_plot, monthpoly, \
                                  make_poly(expand_season(stdv_cmip5)), resp)
  idummy = idummy + 1
  res_cmip@dataset = "CMIP6"
  res_cmip@project = "CMIP6"
  resp@gsFillColor = get_color(res_cmip)
  dummy(idummy) = gsn_add_polygon(wks, seasonality_plot, monthpoly, \
                                  make_poly(expand_season(stdv_cmip6)), resp)
  idummy = idummy + 1

  res_cmip@project = "CMIP6"
  rests@xyLineColor = get_color(res_cmip)
  rests@xyDashPattern =  get_lineindex(res_cmip)
  rests@xyLineThicknessF = 3.
  rests@xyLineOpacityF = 1.
  plot = gsn_csm_xy(wks, month, expand_season(stdv_cmip6mean), rests)
  overlay(seasonality_plot, plot)
  res_cmip@project = "CMIP5"
  rests@xyLineColor = get_color(res_cmip)
  rests@xyDashPattern =  get_lineindex(res_cmip)
  rests@xyLineThicknessF = 3.
  plot = gsn_csm_xy(wks, month, expand_season(stdv_cmip5mean), rests)
  overlay(seasonality_plot, plot)

  res_obs = True
  do i = 0, nobs-1
    res_obs@project = "OBS"
    res_obs@dataset = obs(i)
    rests@xyLineColor = get_color(res_obs)
    rests@xyDashPattern =  get_lineindex(res_obs)
    rests@xyLineThicknessF = 1.5
    plot = gsn_csm_xy(wks, month, expand_season(stdv_obs(i, :)), rests)
    overlay(seasonality_plot, plot)

    lres = True
    lres@gsLineColor = rests@xyLineColor
    lres@gsLineDashPattern = rests@xyDashPattern
    lres@gsLineThicknessF = rests@xyLineThicknessF
    dummy(idummy) = gsn_add_polyline(wks, seasonality_plot, \
                                     (/1.+i*3., 1+i*3.+0.8/), \
                                     (/rests@trYMaxF, rests@trYMaxF/)-0.1, \
                                     lres)
    idummy = idummy + 1
    txres@txFontColor = get_color(res_obs)
    dummy(idummy) = gsn_add_text(wks, seasonality_plot, obs(i), \
                                 1.+i*3.+0.9, rests@trYMaxF-0.1, txres)
    idummy = idummy + 1
  end do

  res_cmip@project = "CMIP6"
  x = 4.2+3.*(nobs-1)
  resp@gsLineOpacityF = 0.2
  resp@gsFillColor := get_color(res_cmip)
  dummy(idummy) = gsn_add_polygon(wks, seasonality_plot, \
                                  (/x, x, x+0.9, x+0.9/), \
                                  (/rests@trYMaxF-0.03, rests@trYMaxF-0.17, \
                                    rests@trYMaxF-0.17, rests@trYMaxF-0.03/), \
                                  resp)
  idummy = idummy + 1
  lres@gsLineColor = get_color(res_cmip)
  lres@gsLineDashPattern = get_lineindex(res_cmip)
  lres@gsLineThicknessF = 3.
  dummy(idummy) = gsn_add_polyline(wks, seasonality_plot, (/x, x+0.9/), \
                                   (/rests@trYMaxF, rests@trYMaxF/)-0.1, lres)
  idummy = idummy + 1

  txres@txFontColor = get_color(res_cmip)
  dummy(idummy) = gsn_add_text(wks, seasonality_plot, "CMIP6", x+1., \
                               rests@trYMaxF-0.1, txres)

  idummy = idummy + 1
  res_cmip@project = "CMIP5"
  x = 7.+3.*(nobs-1)
  resp@gsLineOpacityF = 0.2
  resp@gsFillColor := get_color(res_cmip)
  dummy(idummy) = gsn_add_polygon(wks, seasonality_plot, \
                                  (/x, x, x+0.9, x+0.9/), \
                                  (/rests@trYMaxF-0.03, rests@trYMaxF-0.17, \
                                    rests@trYMaxF-0.17, rests@trYMaxF-0.03/), \
                                  resp)
  idummy = idummy + 1
  lres@gsLineColor = get_color(res_cmip)
  lres@gsLineDashPattern = get_lineindex(res_cmip)
  lres@gsLineThicknessF = 3.
  dummy(idummy) = gsn_add_polyline(wks, seasonality_plot, (/x, x+0.9/), \
                                   (/rests@trYMaxF, rests@trYMaxF/)-0.1, lres)
  idummy = idummy + 1
  txres@txFontColor = get_color(res_cmip)
  dummy(idummy) = gsn_add_text(wks, seasonality_plot, "CMIP5", \
                               x+1., rests@trYMaxF-0.1, txres)
  idummy = idummy + 1

  pres = True
  pres@gsnFrame = False
  pres@gsnMaximize = False
  pres@gsnPanelTop = 0.9
  pres@gsnPanelBottom = 0.5
  pres@gsnPanelLeft = 0.25
  pres@gsnPanelRight = 0.75
  gsn_panel(wks, seasonality_plot, (/1, 1/), pres)

; ============================ Individual Models =============================

  do ii = 0, nmodels-1
    if (isvar("centers")) then
      tmp = centers
      delete(centers)
      center = get_modelling_center(models(ii))
      centers = array_append_record(tmp, center(0), 0)
      delete(tmp)
      delete(center)
    else
      centers = get_modelling_center(models(ii))
    end if
  end do
  centers := get_unique_values(centers)
  sqsort(centers)
  ncenters = dimsizes(centers)

  ncols = ncenters
  if (any(projects.eq."CMIP6")) then
    ncols = ncols + 1
    if (any(projects.eq."CMIP5")) then
      icolcmip6 = -2
    else
      icolcmip6 = -1
    end if
  end if
  if (any(projects.eq."CMIP5")) then
    ncols = ncols + 1
    icolcmip5 = -1
  end if

  col = ispan(ncenters-ncols, ncenters-1, 1)
  names = new(ncols, "string")
  names!0 = "col"
  names&col = col
  names({0:}) = (/centers/)
  if (any(projects.eq."CMIP6")) then
    names({icolcmip6}) = "CMIP6"
  end if
  if (any(projects.eq."CMIP5")) then
    names({icolcmip5}) = "CMIP5"
  end if

  res := True
  res@gsnDraw = False
  res@gsnFrame = False
  res@trXMinF = col(0) - 0.5
  res@trXMaxF = col(ncols-1) + 0.5
  res@vpWidthF = 0.5
  res@vpHeightF = 0.2
  res@tmXTOn = False
  res@tmXMajorGrid = True
  res@tmYMajorGrid = True
  res@tmXMajorGridLineDashPattern = 0
  res@tmYMajorGridLineDashPattern = 0
  res@tmXMajorGridLineColor = "grey"
  res@tmYMajorGridLineColor = "grey"
  res@tmXMajorGridThicknessF = 0.005
  res@tmYMajorGridThicknessF = 0.

  res@tmXBMode = "Explicit"
  res@tmXBValues = col
  res@tmXBLabels = names
  res@tmXBLabelAngleF = 60.  ; 45.
  res@tmXBLabelJust = "CenterRight"
  res@tmXBLabelFontHeightF = 0.01

  mkres = True
  mkres@gsMarkerThicknessF = 2.0
  mkres@gsMarkerSizeF    = 0.01
  lnres = True
  lnres@gsLineThicknessF = 2.

  res@trYMaxF = 2.5
  res@trYMinF = 0.5
  res@tmYLMode          = "Explicit"
  res@tmYLLabelFontHeightF = res@tmXBLabelFontHeightF
  res@tmYLValues        = (/0., 0.5, 1., 1.5, 2., 2.5, 3./)
  res@tmYLLabels        = res@tmYLValues

  res_obs@project = "OBS"
  res_obs@dataset = obs(0)
  res@xyLineColor = get_color(res_obs)
  res@xyDashPattern = get_lineindex(res_obs)
  res@xyLineThicknessF = 2.
  res@tiMainString = "(b) Seasonality metric"
  mme_plot = gsn_csm_xy(wks, (/res@trXMinF, res@trXMaxF/), \
                        (/seasonality_obs(0), seasonality_obs(0)/), res)
  do j = 1, nobs-1
    res_obs@project = "OBS"
    res_obs@dataset = obs(j)
    res@xyLineColor = get_color(res_obs)
    res@xyDashPattern =  get_lineindex(res_obs)
    plot = gsn_csm_xy(wks, (/res@trXMinF, res@trXMaxF/), \
                      (/seasonality_obs(j), seasonality_obs(j)/), res)
    overlay(mme_plot, plot)
  end do

  w = 0.3  ; width of bars for MME
  nmodel_cmip5 = 0
  nmodel_cmip6 = 0
  do ii = 0, nmodels-1
    center = get_modelling_center(models(ii))
    i = ind(center(0).eq.centers)
    res_cmip@dataset = models(ii)
    res_cmip@project = projects(ii)
    mkres@gsMarkerColor    = get_color(res_cmip)
    mkres@gsMarkerIndex    = get_markerindex(res_cmip)
    dummy(idummy) = gsn_add_polymarker(wks, mme_plot, i, \
                                       seasonality(ii), mkres)
    idummy = idummy + 1
    delete(center)
    if (projects(ii).eq."CMIP6") then
      nmodel_cmip6 = nmodel_cmip6 + 1
      if (isvar("seasonality_cmip6")) then
        tmp = seasonality_cmip6
        delete(seasonality_cmip6)
        seasonality_cmip6 = new(nmodel_cmip6, typeof(seasonality))
        seasonality_cmip6 = array_append_record(tmp, seasonality(ii), 0)
        delete(tmp)
      else
        seasonality_cmip6 = seasonality(ii)
      end if
    end if
    if (projects(ii).eq."CMIP5") then
      nmodel_cmip5 = nmodel_cmip5 + 1
      if (isvar("seasonality_cmip5")) then
        tmp = seasonality_cmip5
        delete(seasonality_cmip5)
        seasonality_cmip5 = new(nmodel_cmip5, typeof(seasonality))
        seasonality_cmip5 = array_append_record(tmp, seasonality(ii), 0)
        delete(tmp)
      else
        seasonality_cmip5 = seasonality(ii)
      end if
    end if
  end do
  if (nmodel_cmip6.gt.2) then
    median = dim_median(seasonality_cmip6)
    qsort(seasonality_cmip6)
    x25  = round(0.25*dimsizes(seasonality_cmip6), 3) - 1
    x75  = round(0.75*dimsizes(seasonality_cmip6), 3) - 1
    percentile25 = seasonality_cmip6(x25)
    percentile75 = seasonality_cmip6(x75)
    xmax = max(seasonality_cmip6)
    xmin = min(seasonality_cmip6)
    median@project = "CMIP6"
    polyres = True
    polyres@gsFillColor = get_color(median)
    dummy(idummy) \
      = gsn_add_polygon(wks, mme_plot, \
                        (/icolcmip6-w, icolcmip6-w, \
                          icolcmip6+w, icolcmip6+w/), \
                        (/percentile25, percentile75, \
                          percentile75, percentile25/), polyres)
    idummy = idummy + 1
    lnres@gsLineColor := get_color(median)
    dummy(idummy) = gsn_add_polyline(wks, mme_plot, \
                                     (/icolcmip6, icolcmip6/), \
                                     (/xmin, xmax/), lnres)
    idummy = idummy + 1
    lnres@gsLineColor := "white"
    dummy(idummy) = gsn_add_polyline(wks, mme_plot, \
                                     (/icolcmip6-w, icolcmip6+w/), \
                                     (/median, median/), lnres)
    idummy = idummy + 1
    delete(seasonality_cmip6)
  end if
  if (nmodel_cmip5.gt.2) then
    median = dim_median(seasonality_cmip5)
    qsort(seasonality_cmip5)
    x25  = round(0.25*dimsizes(seasonality_cmip5), 3) - 1
    x75  = round(0.75*dimsizes(seasonality_cmip5), 3) - 1
    percentile25 = seasonality_cmip5(x25)
    percentile75 = seasonality_cmip5(x75)
    xmax = max(seasonality_cmip5)
    xmin = min(seasonality_cmip5)
    median@project = "CMIP5"
    polyres = True
    polyres@gsFillColor = get_color(median)
    dummy(idummy) \
      = gsn_add_polygon(wks, mme_plot, \
                        (/icolcmip5-w, icolcmip5-w, \
                          icolcmip5+w, icolcmip5+w/), \
                        (/percentile25, percentile75, \
                          percentile75, percentile25/), polyres)
    idummy = idummy + 1
    lnres@gsLineColor := get_color(median)
    dummy(idummy) = gsn_add_polyline(wks, mme_plot, \
                                     (/icolcmip5, icolcmip5/), \
                                     (/xmin, xmax/), lnres)
    idummy = idummy + 1
    lnres@gsLineColor := "white"
    dummy(idummy) = gsn_add_polyline(wks, mme_plot, \
                                     (/icolcmip5-w, icolcmip5+w/), \
                                     (/median, median/), lnres)
    idummy = idummy + 1
    delete(seasonality_cmip5)
  end if

  pres@gsnPanelTop = 0.5
  pres@gsnPanelBottom = 0.1
  pres@gsnPanelLeft = 0.2
  pres@gsnPanelRight = 0.8
  gsn_panel(wks, mme_plot, (/1, 1/), pres)

end
