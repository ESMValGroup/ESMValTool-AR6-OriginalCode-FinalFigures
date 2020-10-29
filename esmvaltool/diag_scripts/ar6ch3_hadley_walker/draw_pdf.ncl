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
;    20201024 kosaka_yu: added ERA-Interim..
;    20201019 kosaka_yu: modified the legend format.
;    20200930 kosaka_yu: enabled ensemble obs/reanalysis datasets.
;    20200918 kosaka_yu: adjusted parameter for confidence intervals.
;    20200918 kosaka_yu: revised to show confidence intervals.
;    20200915 kosaka_yu: merged to include Walker circulation figures.
;    20200908 kosaka_yu: revised figure details
;    20200903 kosaka_yu: changed from pdf to box-whisker plots
;    20200514 kosaka_yu: cleaned up.
;    20191224 kosaka_yu: modified to handle piControl & historical experiments
;    20191111 kosaka_yu: modified to handle cmip5 and cmip6 individually
;    20191017 kosaka_yu: updated for ESMValTool 2.0.0b2
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
load "$diag_scripts/ar6ch3_hadley_walker/percentiles.ncl"

function get_statistics(x[*])
local weight, esize, xx, y, nmodels, nmodels_ens, i, m, n
begin
  ; weighted mean and percentile ranges
  weight = x
  weight = 1./tofloat(x@esize)
  y = new(8, typeof(x))
;  y(0) = min(x)
;  y(1) = weighted_percentile(x, weight, 0.05)
  y(0) = weighted_percentile(x, weight, 0.05)
  y(1) = weighted_percentile(x, weight, 0.25)
  y(2) = weighted_mean(x, weight)
  y(3) = weighted_percentile(x, weight, 0.75)
  y(4) = weighted_percentile(x, weight, 0.95)
;  y(3) = weighted_percentile(x, weight, 0.95)
;  y(4) = max(x)
  y@nruns = dimsizes(x)

  confint_threshold = True  ; False
  
  if (isatt(x, "model_id")) then
    nmodels = count_unique_values(x@model_id)
    y@nmodels = nmodels

    models = get_unique_values(x@model_id)

; unweighted average of individual models
  ;delete(xx)
    xx = new(nmodels, typeof(x))
    nmodels_ens = 0
    do n = 0, nmodels-1
      if (num(.not.ismissing(x(ind(x@model_id.eq.models(n))))).ge.3) then
        xx(n) = avg(x(ind(x@model_id.eq.models(n))))
        nmodels_ens = nmodels_ens + 1
      end if
    end do
    y@means = xx

    if (confint_threshold) then
  ; unweighted confidence interval
      y(5) = avg(xx) + cdft_t(0.05, nmodels_ens-1)*stddev(xx)/sqrt(tofloat(nmodels_ens))
      y(6) = avg(xx)
      y(7) = avg(xx) + cdft_t(0.95, nmodels_ens-1)*stddev(xx)/sqrt(tofloat(nmodels_ens))
    else
  ; unweighted confidence interval
      delete(xx)
      esize = min(x@esize)
      ;esize = max((/min(x@esize), 3/))
      xx = new((/nmodels, esize/), typeof(x))
      do n = 0, nmodels-1
        i = ind(x@model_id.eq.models(n))
        if (dimsizes(i).ge.esize) then
          xx(n, :) = x(i(0:esize-1))
        end if
        delete(i)
      end do
      n = dimsizes(ndtooned(xx))
      y(5) = avg(xx) + cdft_t(0.05, n-1)*stddev(xx)/sqrt(tofloat(n))
      y(6) = y(2)
      y(7) = avg(xx) + cdft_t(0.95, n-1)*stddev(xx)/sqrt(tofloat(n))
    end if
    delete(xx)
  else
    esize = dimsizes(x)
    y(5) = avg(x) + cdft_t(0.05, esize-1)*stddev(x)/sqrt(tofloat(esize))
    y(6) = avg(x)
    y(7) = avg(x) + cdft_t(0.95, esize-1)*stddev(x)/sqrt(tofloat(esize))
  end if
    
  return(y)
end

procedure box_whisker(wks, plot, x, w, percentiles, type, color, add_exp, dum, i)
local lres, plres, txres, y, add_exp, mkres, type
begin
  lres = True
  lres@gsLineColor := color
  lres@gsLineDashPattern = 0
  lres@gsLineThicknessF = 2.
  plres = True

  if (type.eq."percentile") then
    dum(i) = gsn_add_polyline(wks, plot, (/x, x/), \
                              (/percentiles(0), percentiles(4)/), lres)
    i = i + 1
    plres@gsFillColor := "white"
    dum(i) = gsn_add_polygon(wks, plot, (/x-w/2., x-w/2., x+w/2., x+w/2./), \
                             (/percentiles(1), percentiles(3), \
                               percentiles(3), percentiles(1)/), plres)
    i = i + 1                
    dum(i) = gsn_add_polyline(wks, plot, (/x-w/2., x-w/2., x+w/2., x+w/2., x-w/2./), \
                              (/percentiles(1), percentiles(3), percentiles(3), \
                                percentiles(1), percentiles(1)/), lres)
    i = i + 1
    dum(i) = gsn_add_polyline(wks, plot, (/x-w/2., x+w/2./), \
                              (/percentiles(2), percentiles(2)/), lres)
    i = i + 1
  elseif (type.eq."confidence") then
    plres@gsFillColor := color
    dum(i) = gsn_add_polygon(wks, plot, (/x-w/2., x-w/2., x+w/2., x+w/2./), \
                             (/percentiles(5), percentiles(7), \
                               percentiles(7), percentiles(5)/), plres)
    i = i + 1
    lres@gsLineColor := "white"
    lres@gsLineThicknessF = 1.
    dum(i) = gsn_add_polyline(wks, plot, (/x-w/2.+0.02, x+w/2.-0.02/), \
                              (/percentiles(6), percentiles(6)/), lres)
    i = i + 1
    
    if (isatt(percentiles, "means")) then
      mkres = True
      mkres@gsMarkerIndex = 16
      mkres@gsMarkerSizeF =  0.005
      do j = 0, dimsizes(percentiles@means)-1
        if (.not.ismissing(percentiles@means(j))) then
          dum(i) = gsn_add_polymarker(wks, plot, x, \
                                      percentiles@means(j), mkres)
          i = i + 1
        end if
      end do
    end if
  end if
  
  if (add_exp) then
    y = 0.
    txres = True
    txres@txFontHeightF = 0.013
    txres@txFontColor := color
    if (percentiles@hemisphere.eq."NH") then
;      if (any(str_squeeze(percentiles@exp).eq.(/"piControl", "GHG", "strat~C~ O3"/))) then
      if (any(str_squeeze(percentiles@exp).eq.(/"AMIP", "GHG", "strat~C~ O3"/))) then
        txres@txJust = "BottomCenter"
        if (type.eq."percentile") then
          y = percentiles(4) + 0.02
        elseif (type.eq."confidence") then
          y = max((/percentiles(7), max(percentiles@means)/)) + 0.02
        end if
      else
        txres@txJust = "TopCenter"
        if (type.eq."percentile") then
          y = percentiles(0) - 0.02
        elseif (type.eq."confidence") then
          y = min((/percentiles(5), min(percentiles@means)/)) - 0.02
        end if
      end if
    else
      if (percentiles@exp.eq."AMIP") then
        txres@txJust = "TopCenter"
        if (type.eq."percentile") then
          y = percentiles(0) - 0.02
        elseif (type.eq."confidence") then
          y = min((/percentiles(5), min(percentiles@means)/)) - 0.02
        end if
      else
        txres@txJust = "BottomCenter"
        if (type.eq."percentile") then
          y = percentiles(4) + 0.02
        elseif (type.eq."confidence") then
          y = max((/percentiles(7), max(percentiles@means)/)) + 0.02
        end if
      end if
    end if
    dum(i) = gsn_add_text(wks, plot, percentiles@exp, x, y, txres)
    i = i + 1
  end if
end

begin

  enter_msg(DIAG_SCRIPT, "")

; ========================== Configuration =========================

  layout = (/(/"NH_edge_trend_ANN", "NH_edge_trend_DJF"/), \
             (/"SH_edge_trend_ANN", "SH_edge_trend_DJF"/)/)
  if (isatt(diag_script_info, "layout")) then
    layout := diag_script_info@layout
  end if

  scale = 10.  ; trend in "per [scale] years"

  input_dirs = diag_script_info@input_files
  ndirs = dimsizes(input_dirs)

  dims = dimsizes(layout)
  if (dimsizes(dims).eq.1) then
    tmp = dims
    delete(dims)
    dims = (/1, tmp/)
    delete(tmp)
    tmp = layout
    delete(layout)
    layout = new(dims, "string")
    layout(0, :) = tmp
    delete(tmp)
  elseif (dimsizes(dims).gt.2) then
    error_msg("f", DIAG_SCRIPT, "", \
              "layout must be either 1D or 2D.")    
  end if
  dirs = new(dims, "string")
  dirs = ""
  
  do ii = 0, dims(0)-1
    do jj = 0, dims(1)-1
      do n = 0, ndirs-1
        if (isStrSubset(input_dirs(n), layout(ii, jj))) then
          dirs(ii, jj) = input_dirs(n)
          break
        end if
      end do
    end do
  end do
  if (any(dirs.eq."")) then
    error_msg("f", DIAG_SCRIPT, "", \
              "Failed to identify the path corresponding to layout.")
  end if

  obslist = (/"ERA5", "JRA-55", "MERRA2", "CFSR", "ERA-Interim", "HadSLP2", "ICOADS", "ERA-20C", "20CRv3"/)
  obslist@color = (/"black", "black", "black", "black", "black", "grey60", "grey60", "grey60", "grey60"/)
  obslist@pattern = (/0, 3, 1, 4, 2, 0, 3, 1, 2/)
;  color_obs = (/"black", "black", "black", "black", "black", "black", "black"/)
;  pattern_obs = (/3, 0, 1, 2, 4, 5, 6/)
  obsenslist = (/"20CRv2", "20CRv2c", "20CRv3", "CERA-20C"/)
  obsenslist@pattern = (/3, 3, 3, 2/)
  obsenslist@color = (/"grey60", "grey60", "grey60", "grey60"/)

  
  color_cmip5hist = (/204., 35., 35. /)/255. ;(/37., 81., 204./)/255.
  color_cmip5pi = (/150., 150., 150./)/255.
  color_cmip6hist = (/204., 35., 35./)/255.
  color_cmip6pi = (/27., 55., 170. /)/255.
  color_cmip6amip = (/251., 141., 70./)/255.
  color_cmip6GHG = (/122., 78., 27. /)/255.
  color_cmip6aer = (/133., 183., 213./)/255.
  color_cmip6nat = (/63., 131., 35./)/255.
  color_cmip6stratO3 = (/152., 114., 170./)/255.

; ========================== Prepare drawing =========================
  
  wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                "hadley_walker_trends")
  output_dir = config_user_info@plot_dir + "/"
  system("mkdir -p "+output_dir)

  plots = new(dims, "graphic")
  dummy = new(1000, "graphic")
  idummy = 0

  res := True
  res@gsnDraw = False
  res@gsnFrame = False
  res@tmYROn = False
  res@tmXTOn = False
  res@tmXBOn = False
  res@tmXBLabels = False
  res@tmYLPrecision = 2
  res@vpHeightF = 0.35
;  res@gsnXRefLine = 0.3
;  res@gsnYRefLineThicknessF = 0.5
  res@gsnYRefLine = 0.
  res@gsnYRefLineThicknessF = 0.5
  res@gsnYRefLineColor = "black"
    
  lres = True
  lres@gsLineThicknessF = 3.
  lres@gsLineDashSegLenF = 0.1
  lres@tfPolyDrawOrder = "PostDraw"

  txres = True
  txres@txJust = "CenterCenter"
  txres@txFontHeightF = 0.013       

  polyres = True
  polyres@gsFillColor = "grey80"
  polyres@gsFillOpacityF = 0.5
  plres = True
  
  alphabet = str_split_by_length("abcdefghijklmnopqrstuvwxyz", 1)

  ipanel = 0
  do ii = 0, dims(0)-1
    do jj = 0, dims(1)-1

; ======================= Get data and percentiles =====================

      print("Reading files in "+dirs(ii, jj))
      paths = systemfunc("ls "+dirs(ii, jj)+"/*.nc")
      if (dimsizes(paths).eq.0) then
        continue
      end if

      if (isStrSubset(layout(ii, jj), "NH_")) then
        hemisphere = "NH"
      elseif (isStrSubset(layout(ii, jj), "SH_")) then
        hemisphere = "SH"
      else
        hemisphere = ""
      end if
      if (isStrSubset(layout(ii, jj), "_ANN")) then
        season = "annual mean"
      elseif (isStrSubset(layout(ii, jj), "_DJF")) then
        if (hemisphere.eq."NH") then
          season = "winter"
        elseif (hemisphere.eq."SH") then
          season = "summer"
        else
          season = "DJF"
        end if
      elseif (isStrSubset(layout(ii, jj), "_MAM")) then
        if (hemisphere.eq."NH") then
          season = "spring"
        elseif (hemisphere.eq."SH") then
          season = "autumn"
        else
          season = "MAM"
        end if
      elseif (isStrSubset(layout(ii, jj), "_JJA")) then
        if (hemisphere.eq."NH") then
          season = "summer"
        elseif (hemisphere.eq."SH") then
          season = "winter"
        else
          season = "JJA"
        end if
      elseif (isStrSubset(layout(ii, jj), "_SON")) then
        if (hemisphere.eq."NH") then
          season = "autumn"
        elseif (hemisphere.eq."SH") then
          season = "spring"
        else
          season = "SON"
        end if
      else
        season = ""
      end if
      if (season.eq."") then
        season = "annual mean"
      end if
      
      ncmip5 = 0
      ncmip6 = 0
      ncmip6_single = 0
      range = 0.
      do k = 0, dimsizes(paths)-1
        f = addfile(paths(k), "r")
        tmppath = str_split(paths(k), "/")
        fname = tmppath(dimsizes(tmppath)-1)
        delete(tmppath)
        if (isfilevar(f, "lat")) then
          varname = "lat"
        elseif (isfilevar(f, "dslp")) then
          varname = "dslp"
        end if
        data = f->$varname$
        if (varname.eq."dslp".and.data@units.eq."Pa") then
          data = data /100.
          data@units = "hPa"
        end if
        data = data * scale
        if (isStrSubset(fname, "_cmip").and..not.isStrSubset(fname, "pi.nc")) then
          syear = toint(data@syear)
          eyear = toint(data@eyear)
        end if
        if (isStrSubset(paths(k), "_cmip5hist.nc")) then
;print("cmip5 hist")
          cmip5hist = get_statistics(data)
          cmip5hist@project = "CMIP5"
          cmip5hist@exp = "    all  ~C~ forcing"
          cmip5hist@hemisphere = hemisphere
          range = max((/range, max(abs(cmip5hist))/))
          ncmip5 = ncmip5 + 1
        elseif (isStrSubset(paths(k), "_cmip5pi.nc")) then
;print("cmip5 piControl")
          cmip5pi = get_statistics(data)
          cmip5pi@project = "CMIP5"
          cmip5pi@exp = "piControl"
          cmip5pi@hemisphere = hemisphere
          range = max((/range, max(abs(cmip5pi))/))
          ncmip5 = ncmip5 + 1
        elseif (isStrSubset(paths(k), "_cmip6hist.nc")) then
;print("cmip6 hist")
          cmip6hist = get_statistics(data)
          cmip6hist@project = "CMIP6"
          cmip6hist@exp = "    all  ~C~ forcing"
          cmip6hist@hemisphere = hemisphere
          range = max((/range, max(abs(cmip6hist))/))
          ncmip6 = ncmip6 + 1
        elseif (isStrSubset(paths(k), "_cmip6pi.nc")) then
;print("cmip6 piControl")
          cmip6pi = get_statistics(data)
          cmip6pi@project = "CMIP6"
          cmip6pi@exp = "piControl"
          cmip6pi@hemisphere = hemisphere
          range = max((/range, max(abs(cmip6pi))/))
          ncmip6 = ncmip6 + 1
        elseif (isStrSubset(paths(k), "_cmip6amip.nc") .or. \
                isStrSubset(paths(k), "_cmip6amip-hist.nc")) then
;print("cmip6 amip")
          cmip6amip = get_statistics(data)
          cmip6amip@project = "CMIP6"
          cmip6amip@exp = "AMIP"
          cmip6amip@hemisphere = hemisphere
          if (isStrSubset(paths(k), "_cmip6amip.nc")) then
            cmip6amip@activity = "DECK"
          else
            cmip6amip@activity = "GMMIP"
          end if
          range = max((/range, max(abs(cmip6amip))/))
          ncmip6 = ncmip6 + 1
        elseif (isStrSubset(paths(k), "_cmip6GHG.nc")) then
;print("cmip6 GHG")
          cmip6GHG = get_statistics(data)
          cmip6GHG@project = "CMIP6"
          cmip6GHG@exp = "GHG"
          cmip6GHG@hemisphere = hemisphere
          range = max((/range, max(abs(cmip6GHG))/))
          ncmip6 = ncmip6 + 1
          ncmip6_single = ncmip6_single + 1
        elseif (isStrSubset(paths(k), "_cmip6aer.nc")) then
;print("cmip6 aer")
          cmip6aer = get_statistics(data)
          cmip6aer@project = "CMIP6"
          cmip6aer@exp = "aerosol"
          cmip6aer@hemisphere = hemisphere
          range = max((/range, max(abs(cmip6aer))/))
          ncmip6 = ncmip6 + 1
          ncmip6_single = ncmip6_single + 1
        elseif (isStrSubset(paths(k), "_cmip6nat.nc")) then
;print("cmip6 nat")
          cmip6nat = get_statistics(data)
          cmip6nat@project = "CMIP6"
          cmip6nat@exp = "natural"
          cmip6nat@hemisphere = hemisphere
          range = max((/range, max(abs(cmip6nat))/))
          ncmip6 = ncmip6 + 1
          ncmip6_single = ncmip6_single + 1
        elseif (isStrSubset(paths(k), "_cmip6stratO3.nc")) then
;print("cmip6 stratO3")
          cmip6stratO3 = get_statistics(data)
          cmip6stratO3@project = "CMIP6"
          cmip6stratO3@exp = "strat~C~ O3  "
          cmip6stratO3@hemisphere = hemisphere
          range = max((/range, max(abs(cmip6stratO3))/))
          ncmip6 = ncmip6 + 1
          ncmip6_single = ncmip6_single + 1
        else
          if (any(data@dataset.eq.obsenslist)) then
            if (isvar("ObsEns")) then
              tmp = ObsEns
              delete(ObsEns)
              dimstmp = dimsizes(tmp)
              if (dimsizes(dimstmp).eq.1) then
                ObsEns = new((/2, dimstmp(0)/), typeof(tmp))
                ObsEns(0, :) = tmp
                ObsEns(1, :) = get_statistics(data)
              else
                ObsEns = new((/dimstmp(0)+1, dimstmp(1)/), typeof(tmp))
                ObsEns(0:dimstmp(0)-1, :) = tmp
                ObsEns(dimstmp(0), :) = get_statistics(data)
              end if
              ObsEns@dataset := array_append_record(tmp@dataset, data@dataset, 0)
              delete(tmp)
              delete(dimstmp)
            else
              tmp = get_statistics(data)
              ObsEns = new((/1, dimsizes(tmp)/), typeof(tmp))
              ObsEns(0, :) = tmp
              ObsEns@dataset = data@dataset
              delete(tmp)
            end if
          else
            if (isvar("Obs")) then
              tmp = Obs
              delete(Obs)
              Obs = array_append_record(tmp, data, 0)
              delete(Obs@dataset)
              Obs@dataset = array_append_record(tmp@dataset, data@dataset, 0)
              delete(tmp)
            else
              Obs = data
            end if
            range = max((/range, abs(data)/))
          end if
        end if
        delete(data)
      end do
      nexp = ncmip5 + ncmip6
      if (ncmip5.gt.0) then
        nexp = nexp + 1
      end if
      
      delete(paths)
      
; ======================= Draw figure =====================

      res@tmYLLabelFontHeightF = 0.02
      res@tmYLLabelDeltaF = -0.5
      res@tiMainOffsetYF = -0.028
      res@tiMainFontHeightF = 0.022
      if (varname.eq."lat") then
        if (range.gt.0.06*scale) then
          res@trYMaxF = 0.1*scale
;        else
;          res@trYMaxF = 0.05*scale
;        end if
        elseif (range.gt.0.04*scale) then
          res@trYMaxF = 0.065*scale
        else
          res@trYMaxF = 0.045*scale
        end if
      elseif (varname.eq."dslp") then
        if (range.gt.0.02*scale) then
          res@trYMaxF = 0.038*scale
        elseif (range.gt.0.005*scale) then
;          res@trYMaxF = 0.014*scale
          res@trYMaxF = 0.012*scale
        else
          res@trYMaxF = 0.1*scale
        end if
      end if
      res@trYMinF = -res@trYMaxF
      res@tiYAxisFontHeightF = 0.02
      res@tiYAxisOffsetXF = -0.005

;      res@gsnCenterStringFontHeightF = 0.016
      res@gsnLeftStringFontHeightF = 0.022
      res@gsnLeftString = "("+alphabet(ipanel)+")"
      res@gsnLeftStringParallelPosF = -0.1
      res@gsnLeftStringOrthogonalPosF = 0.03
      if (all((/hemisphere, season/).ne."")) then
;        res@gsnCenterString = hemisphere+" "+season
        res@tiMainString = hemisphere+" "+season
      else
        if (season.eq."annual mean") then
;          res@gsnCenterString = "Annual mean"
          res@tiMainString = "Annual mean"
        else
;          res@gsnCenterString = season
          res@tiMainString = season
        end if
      end if
      res@tiMainString = res@tiMainString +" "+syear+"-"+eyear
;      if (ipanel.eq.0) then
      if (jj.eq.0) then
        if (varname.eq."lat") then
          if (scale.eq.1.) then
            res@tiYAxisString = "(degree/yr; positive if northward)"
          else
            res@tiYAxisString = " (degree / "+scale+" years; ~C~positive if northward)"
          end if
        elseif (varname.eq."dslp") then
          if (scale.eq.1.) then
            res@tiYAxisString = "(hPa/yr; positive if strengthening)"
          else
            res@tiYAxisString = "      (hPa / "+scale+" years; ~C~positive if strengthening)"
          end if
        end if
      else
        res@tiYAxisString = ""
      end if
      res@vpWidthF = 0.4

      w = 0.7
      xcmip5 = -0.4
      xcmip6 = 1.
      if (ncmip5.gt.0) then
        res@trXMinF = xcmip5 -tofloat(ncmip5-1) - 0.7
      else
        res@trXMinF = 0.5
      end if
      res@trXMaxF =  tofloat(ncmip6+1) + 0.1
      if (isvar("cmip6hist")) then
        res@trXMaxF = res@trXMaxF + 1.
      end if
      xcmip6_divide = tofloat(ncmip6 - ncmip6_single) + 0.7
      
      res@xyLineColor = "transparent"
      plots(ii, jj) = gsn_csm_xy(wks, (/res@trXMinF, res@trXMaxF/), (/0., 0./), res)

      lres@gsLineThicknessF = 0.5
      lres@gsLineDashPattern = 0
      lres@gsLineColor = "black"
      dummy(idummy) = gsn_add_polyline(wks, plots(ii, jj), (/xcmip6_divide, xcmip6_divide/), \
                                       (/res@trYMinF, res@trYMaxF/), lres)
      idummy = idummy + 1

      if (ncmip5.gt.0) then
        dummy(idummy) = gsn_add_polygon(wks, plots(ii, jj), (/res@trXMinF, res@trXMinF, 0.3, 0.3/), \
                                        (/res@trYMinF, res@trYMaxF, res@trYMaxF, res@trYMinF/), polyres)
        idummy = idummy + 1
      end if
      
      if (ii.eq.0 .and. jj.eq.0) then
        add_expname = True
      else
        add_expname = False
      end if
      
      if (isvar("ObsEns")) then
        dimstmp = dimsizes(ObsEns)
        do n = 0, dimstmp(0)-1
          iobs = ind(obsenslist.eq.ObsEns@dataset(n))
          plres@gsFillColor = obsenslist@color(iobs)
          plres@gsFillIndex = obsenslist@pattern(iobs)
          dummy(idummy) = gsn_add_polygon(wks, plots(ii, jj), \
                                          (/res@trXMinF, xcmip6_divide, \
                                            xcmip6_divide, res@trXMinF/), \
                                          (/ObsEns(n, 0), ObsEns(n, 0), \
                                            ObsEns(n, 6), ObsEns(n, 6)/), \
                                          plres)  ; max-min range
          idummy = idummy + 1
          if (plres@gsFillIndex.ne.0) then
            lres@gsLineThicknessF = 1.
            lres@gsLineColor = obsenslist@color(iobs)
            lres@gsLineDashPattern = 0
            dummy(idummy) = gsn_add_polyline(wks, plots(ii, jj), \
                                            (/res@trXMinF, xcmip6_divide, \
                                              xcmip6_divide, res@trXMinF, res@trXMinF/), \
                                            (/ObsEns(n, 0), ObsEns(n, 0), \
                                              ObsEns(n, 6), ObsEns(n, 6), ObsEns(n, 0)/), \
                                            lres)  ; max-min range
            idummy = idummy + 1
          end if
          if (.not.isdefined("legend") .or. \
              all(metadata_att_as_array(legend, "name").ne.ObsEns@dataset(n))) then
            if (.not.isdefined("legend")) then
              legend = NewList("fifo")
            end if
            dataset = ObsEns@dataset(n)
            dataset@name = ObsEns@dataset(n)
            dataset@type = "ensemble_obs"
            dataset@color = plres@gsFillColor
            dataset@pattern = plres@gsFillIndex
            ListAppend(legend, new(1, "string"))
            k = ListCount(legend)-1
            legend[k] = dataset
            delete(dataset)
          end if
        end do
        delete(ObsEns)
        delete(dimstmp)
      end if

      if (isvar("Obs")) then
        do n = 0, dimsizes(Obs)-1
          iobs = ind(obslist.eq.Obs@dataset(n))
          lres@gsLineColor = obslist@color(iobs) ;color_obs(n)
          lres@gsLineDashPattern = obslist@pattern(iobs) ; pattern_obs(n)
          lres@gsLineThicknessF = 2.
;          dummy(idummy) = gsn_add_polyline(wks, plots(ii, jj), (/res@trXMinF, res@trXMaxF/), \
          dummy(idummy) = gsn_add_polyline(wks, plots(ii, jj), (/res@trXMinF, xcmip6_divide/), \
                                           (/Obs(n), Obs(n)/), lres)
          idummy = idummy + 1
          if (.not.isdefined("legend") .or. \
              all(metadata_att_as_array(legend, "name").ne.Obs@dataset(n))) then
            if (.not.isdefined("legend")) then
              legend = NewList("fifo")
            end if
            dataset = Obs@dataset(n)
            dataset@name = Obs@dataset(n)
            dataset@type = "single_obs"
            dataset@color = lres@gsLineColor
            dataset@pattern = lres@gsLineDashPattern
            dataset@thick = lres@gsLineThicknessF
            if (isatt(lres, "gsLineDashSegLenF")) then
              dataset@seg = lres@gsLineDashSegLenF
            end if
            ListAppend(legend, new(1, "string"))
            k = ListCount(legend)-1
            legend[k] = dataset
            delete(dataset)
          end if
        end do
        delete(Obs)
      end if

      if (isvar("cmip5pi")) then
        box_whisker(wks, plots(ii, jj), xcmip5, w, cmip5pi, "percentile", \
                    color_cmip5pi, add_expname, dummy, idummy)
      
        if (.not.isdefined("legend") .or. \
            all(metadata_att_as_array(legend, "name").ne."cmip5pi")) then
          if (.not.isdefined("legend")) then
            legend = NewList("fifo")
          end if
          dataset = "CMIP5 piControl"
          dataset@name = "cmip5pi"
          dataset@type = "ensemble"
          dataset@nmodels = cmip5pi@nmodels
          dataset@nruns = cmip5pi@nruns
          dataset@color = color_cmip5pi
          ListAppend(legend, new(1, "string"))
          k = ListCount(legend)-1
          legend[k] = dataset
          delete(dataset)
        end if
        
        delete(cmip5pi)
        xcmip5 = xcmip5 - 1.
      end if
      if (isvar("cmip5hist")) then
        box_whisker(wks, plots(ii, jj), xcmip5, w, cmip5hist, "percentile", \
                    color_cmip5hist, add_expname, dummy, idummy)

        if (.not.isdefined("legend") .or. \
            all(metadata_att_as_array(legend, "name").ne."cmip5hist")) then
          if (.not.isdefined("legend")) then
            legend = NewList("fifo")
          end if
          dataset = "CMIP5 historical-RCP4.5"
          dataset@name = "cmip5hist"
          dataset@type = "ensemble"
          dataset@nmodels = cmip5hist@nmodels
          dataset@nruns = cmip5hist@nruns
          dataset@color = color_cmip5hist
          ListAppend(legend, new(1, "string"))
          k = ListCount(legend)-1
          legend[k] = dataset
          delete(dataset)
        end if
        
        delete(cmip5hist)
        xcmip5 = xcmip5 - 1.
      end if
      if (isvar("cmip6hist")) then
        box_whisker(wks, plots(ii, jj), xcmip6, w, cmip6hist, "percentile", \
                    color_cmip6hist, add_expname, dummy, idummy)
        ;delete(cmip6hist)
        xcmip6 = xcmip6 + 1.
      end if
      if (isvar("cmip6amip")) then
        box_whisker(wks, plots(ii, jj), xcmip6, w, cmip6amip, "percentile", \
                    color_cmip6amip, add_expname, dummy, idummy)

        if (.not.isdefined("legend") .or. \
            all(metadata_att_as_array(legend, "name").ne."cmip6amip")) then
          if (.not.isdefined("legend")) then
            legend = NewList("fifo")
          end if
          dataset = "AMIP" ; "CMIP6 AMIP"
          dataset@name = "cmip6amip"
          dataset@type = "ensemble"
          dataset@nmodels = cmip6amip@nmodels
          dataset@nruns = cmip6amip@nruns
          dataset@color = color_cmip6amip
          dataset@activity = cmip6amip@activity
          ListAppend(legend, new(1, "string"))
          k = ListCount(legend)-1
          legend[k] = dataset
          delete(dataset)
        else
          k = ind(metadata_att_as_array(legend, "name").eq."cmip6amip")
          if (all(legend[k]@activity.ne.cmip6amip@activity)) then
            legend[k]@nmodels2 = cmip6amip@nmodels
            legend[k]@nruns2 = cmip6amip@nruns
            legend[k]@activity := array_append_record(legend[k]@activity, cmip6amip@activity, 0)
          end if
        end if
        delete(cmip6amip)
        xcmip6 = xcmip6 + 1.
      end if
      if (isvar("cmip6pi")) then
        box_whisker(wks, plots(ii, jj), xcmip6, w, cmip6pi, "percentile", \
                    color_cmip6pi, add_expname, dummy, idummy)
      
        if (.not.isdefined("legend") .or. \
            all(metadata_att_as_array(legend, "name").ne."cmip6pi")) then
          if (.not.isdefined("legend")) then
            legend = NewList("fifo")
          end if
          dataset = "piControl"  ; "CMIP6 piControl"
          dataset@name = "cmip6pi"
          dataset@type = "ensemble"
          dataset@nmodels = cmip6pi@nmodels
          dataset@nruns = cmip6pi@nruns
          dataset@color = color_cmip6pi
          ListAppend(legend, new(1, "string"))
          k = ListCount(legend)-1
          legend[k] = dataset
          delete(dataset)
        else
          k = ind(metadata_att_as_array(legend, "name").eq."cmip6pi")
          dataset = legend[k]
          dataset@nruns := array_append_record(legend[k]@nruns, cmip6pi@nruns, 0)
          legend[k] = dataset
          delete(dataset)
        end if
        
        delete(cmip6pi)
        xcmip6 = xcmip6 + 1.
      end if
      if (xcmip6.gt.xcmip6_divide) then
        xcmip6 = xcmip6 + 0.4
      end if
      if (isvar("cmip6hist")) then
        box_whisker(wks, plots(ii, jj), xcmip6, w, cmip6hist, "confidence", \
                    color_cmip6hist, add_expname, dummy, idummy)
        if (.not.isdefined("legend") .or. \
            all(metadata_att_as_array(legend, "name").ne."cmip6hist")) then
          if (.not.isdefined("legend")) then
            legend = NewList("fifo")
          end if
          dataset = "historical"  ; "CMIP6 historical"
          dataset@name = "cmip6hist"
          dataset@type = "ensemble"
          dataset@nmodels = cmip6hist@nmodels
          dataset@nruns = cmip6hist@nruns
          dataset@color = color_cmip6hist
          ListAppend(legend, new(1, "string"))
          k = ListCount(legend)-1
          legend[k] = dataset
          delete(dataset)
        end if
        delete(cmip6hist)
        xcmip6 = xcmip6 + 1.
      end if
      
      if (isvar("cmip6GHG")) then
        box_whisker(wks, plots(ii, jj), xcmip6, w, cmip6GHG, "confidence", \
                    color_cmip6GHG, add_expname, dummy, idummy)

        if (.not.isdefined("legend") .or. \
            all(metadata_att_as_array(legend, "name").ne."cmip6GHG")) then
          if (.not.isdefined("legend")) then
            legend = NewList("fifo")
          end if
          dataset = "hist-GHG"  ;"CMIP6 hist-GHG"
          dataset@name = "cmip6GHG"
          dataset@type = "ensemble"
          dataset@nmodels = cmip6GHG@nmodels
          dataset@nruns = cmip6GHG@nruns
          dataset@color = color_cmip6GHG
          ListAppend(legend, new(1, "string"))
          k = ListCount(legend)-1
          legend[k] = dataset
          delete(dataset)
        end if
        
        delete(cmip6GHG)
        xcmip6 = xcmip6 + 1.
      end if
      if (isvar("cmip6aer")) then
        box_whisker(wks, plots(ii, jj), xcmip6, w, cmip6aer, "confidence", \
                    color_cmip6aer, add_expname, dummy, idummy)

        if (.not.isdefined("legend") .or. \
            all(metadata_att_as_array(legend, "name").ne."cmip6aer")) then
          if (.not.isdefined("legend")) then
            legend = NewList("fifo")
          end if
          dataset = "hist-aer"  ;"CMIP6 hist-aer"
          dataset@name = "cmip6aer"
          dataset@type = "ensemble"
          dataset@nmodels = cmip6aer@nmodels
          dataset@nruns = cmip6aer@nruns
          dataset@color = color_cmip6aer
          ListAppend(legend, new(1, "string"))
          k = ListCount(legend)-1
          legend[k] = dataset
          delete(dataset)
        end if
        
        delete(cmip6aer)
        xcmip6 = xcmip6 + 1.
      end if
      if (isvar("cmip6stratO3")) then
        box_whisker(wks, plots(ii, jj), xcmip6, w, cmip6stratO3, "confidence", \
                    color_cmip6stratO3, add_expname, dummy, idummy)

        if (.not.isdefined("legend") .or. \
            all(metadata_att_as_array(legend, "name").ne."cmip6stratO3")) then
          if (.not.isdefined("legend")) then
            legend = NewList("fifo")
          end if
          dataset = "hist-stratO3"  ;"CMIP6 hist-stratO3"
          dataset@name = "cmip6stratO3"
          dataset@type = "ensemble"
          dataset@nmodels = cmip6stratO3@nmodels
          dataset@nruns = cmip6stratO3@nruns
          dataset@color = color_cmip6stratO3
          ListAppend(legend, new(1, "string"))
          k = ListCount(legend)-1
          legend[k] = dataset
          delete(dataset)
        end if
        
        delete(cmip6stratO3)
        xcmip6 = xcmip6 + 1.
      end if
      if (isvar("cmip6nat")) then
        box_whisker(wks, plots(ii, jj), xcmip6, w, cmip6nat, "confidence", \
                    color_cmip6nat, add_expname, dummy, idummy)

        if (.not.isdefined("legend") .or. \
            all(metadata_att_as_array(legend, "name").ne."cmip6nat")) then
          if (.not.isdefined("legend")) then
            legend = NewList("fifo")
          end if
          dataset = "hist-nat"  ; "CMIP6 hist-nat"
          dataset@name = "cmip6nat"
          dataset@type = "ensemble"
          dataset@nmodels = cmip6nat@nmodels
          dataset@nruns = cmip6nat@nruns
          dataset@color = color_cmip6nat
          ListAppend(legend, new(1, "string"))
          k = ListCount(legend)-1
          legend[k] = dataset
          delete(dataset)
        end if
        
        delete(cmip6nat)
        xcmip6 = xcmip6 + 1.
      end if
;      if (isvar("Obs")) then
;        do n = 0, dimsizes(Obs)-1
;          lres@gsLineColor = color_obs(n)
;          lres@gsLineDashPattern = pattern_obs(n)
;          lres@gsLineThicknessF = 2.
;          dummy(idummy) = gsn_add_polyline(wks, plots(ii, jj), (/res@trXMinF, res@trXMaxF/), \
;                                           (/Obs(n), Obs(n)/), lres)
;          idummy = idummy + 1
;          if (.not.isdefined("legend") .or. \
;              all(metadata_att_as_array(legend, "name").ne.Obs@dataset(n))) then
;            if (.not.isdefined("legend")) then
;              legend = NewList("fifo")
;            end if
;            dataset = Obs@dataset(n)
;            dataset@name = Obs@dataset(n)
;            dataset@type = "single"
;            dataset@color = lres@gsLineColor
;            dataset@pattern = lres@gsLineDashPattern
;            dataset@thick = lres@gsLineThicknessF
;            if (isatt(lres, "gsLineDashSegLenF")) then
;              dataset@seg = lres@gsLineDashSegLenF
;            end if
;            ListAppend(legend, new(1, "string"))
;            k = ListCount(legend)-1
;            legend[k] = dataset
;            delete(dataset)
;          end if
;        end do
;        delete(Obs)
;      end if

      txres@txFontColor = "black"
      if (hemisphere.eq."SH") then
        y = res@trYMaxF * 0.98
        txres@txJust = "TopCenter"
        x = (xcmip6_divide + res@trXMinF)/2.
        dummy(idummy) = gsn_add_text(wks, plots(ii, jj), "Comparison with obs", x, y, txres)
        idummy = idummy + 1
        x = xcmip6_divide/2. + 0.2
        dummy(idummy) = gsn_add_text(wks, plots(ii, jj), "CMIP6", x, y-res@trYMaxF/10., txres)
        idummy = idummy + 1
        if (ncmip5.gt.0) then  ; .and.add_expname) then
          txres@txFontColor = "grey30"
          x = xcmip5 + tofloat(ncmip5)/2. + 0.5
          dummy(idummy) = gsn_add_text(wks, plots(ii, jj), "CMIP5", x, y-res@trYMaxF/10., txres)
          idummy = idummy + 1
        end if
      else
        y = res@trYMinF * 0.98
        txres@txJust = "BottomCenter"
        x = (xcmip6_divide + res@trXMinF)/2.
        dummy(idummy) = gsn_add_text(wks, plots(ii, jj), "Comparison with obs", x, y+res@trYMaxF/10., txres)
        idummy = idummy + 1
        x = xcmip6_divide/2. + 0.2
        dummy(idummy) = gsn_add_text(wks, plots(ii, jj), "CMIP6", x, y, txres)
        idummy = idummy + 1
        if (ncmip5.gt.0) then  ; .and.add_expname) then
          txres@txFontColor = "grey30"
          x = xcmip5 + tofloat(ncmip5)/2. + 0.5
          dummy(idummy) = gsn_add_text(wks, plots(ii, jj), "CMIP5", x, y, txres)
          idummy = idummy + 1
        end if
      end if

      txres@txFontColor = "black"
      if (ncmip6_single.gt.0) then
        x = (res@trXMaxF + xcmip6_divide)/2.
        dummy(idummy) = gsn_add_text(wks, plots(ii, jj), "Forcing contribution~C~        in CMIP6", x, y, txres)
        idummy = idummy + 1
      end if
      ipanel = ipanel + 1
            
    end do
  end do

  pres = True
  pres@gsnDraw = True
  pres@gsnFrame = False
  pres@gsnPanelMainFontHeightF = 0.016
  pres@gsnPanelMainFont = "helvetica-bold"
  pres@gsnPanelMainString = "Trend of Hadley cell's subtropical edge latitude"
;  pres@gsnPanelFigureStrings= "("+alphabet(0:dimsizes(ndtooned(plots))-1)+")"
;  pres@gsnPanelFigureStringsFontHeightF = 0.016
;  pres@gsnPanelFigureStringsPerimOn = False
;  pres@gsnPanelFigureStringsBackgroundFillColor = "transparent"
;  pres@amJust   = "TopLeft"
;  pres@amOrthogonalPosF = -0.59
;  pres@amParallelPosF = -0.5
;  gsn_panel(wks, ndtooned(plots), dims, pres)
  pres@gsnPanelTop = 0.8
  pres@gsnPanelBottom = 0.5
  pres@gsnPanelMainPosYF = pres@gsnPanelTop-0.015
  gsn_panel(wks, plots(0, :), (/1, 3/), pres)
  pres@gsnPanelTop = 0.5
  pres@gsnPanelBottom = 0.2
  pres@gsnPanelMainPosYF = pres@gsnPanelTop-0.015
  pres@gsnPanelMainString = "Trend of Walker circulation strength"
  gsn_panel(wks, plots(1, :), (/1, 3/), pres)

  txres       := True
  txres@txJust = "CenterCenter"
  txres@txFontHeightF = 0.018
  txres@txFont = "helvetica"
  gsn_text_ndc(wks, "Tropical overturning circulation changes",  0.5, 0.82, txres)

; ======================= Add legend =====================

  if (isdefined("legend")) then
    txres       := True
    fontHeightL = 0.011
    fontHeightS = 0.008
    
    x0 = 0.33
    y0 = 0.16
    h = 0.035
    if (any(str_match_bool_ic(metadata_att_as_array(legend, "name"), "cmip5")) .or. \
        any(str_match_bool_ic(metadata_att_as_array(legend, "name"), "cmip6"))) then
       w = w / tofloat(dims(1)*(nexp+1)) * 0.8
       h1 = 0.02
       h2 = h
       lres := True
       lres@gsLineThicknessF = 2.
       lres@gsLineColor := "grey50"
       lres@gsLineDashPattern = 0
       plres := True
       plres@gsFillColor := "white" ;lres@gsLineColor
       gsn_polyline_ndc(wks, (/x0, x0/), (/y0-h2, y0+h2/), lres)
;       gsn_polyline_ndc(wks, (/x0-w/3., x0+w/3./), (/y0+h2, y0+h2/), lres)
;       gsn_polyline_ndc(wks, (/x0-w/3., x0+w/3./), (/y0-h2, y0-h2/), lres)
       gsn_polygon_ndc(wks, (/x0-w/2., x0-w/2., x0+w/2., x0+w/2./), \
                            (/y0-h1, y0+h1, y0+h1, y0-h1/), plres)
       gsn_polyline_ndc(wks, (/x0-w/2., x0+w/2./), (/y0, y0/), lres)
       gsn_polyline_ndc(wks, (/x0-w/2., x0-w/2., x0+w/2., x0+w/2., x0-w/2./), \
                             (/y0-h1, y0+h1, y0+h1, y0-h1, y0-h1/), lres)

       txres@txFontColor = "black"
       txres@txFontHeightF = fontHeightS
       txres@txJust = "CenterLeft"
       txres@txAngleF = 0.
;       gsn_text_ndc(wks, "max",  x0+0.002, y0+h2, txres)
;       gsn_text_ndc(wks, "95th", x0+w/2.+0.002, y0+h1, txres)
       gsn_text_ndc(wks, "95th",  x0+0.002, y0+h2, txres)
       gsn_text_ndc(wks, "75th", x0+w/2.+0.002, y0+h1, txres)
       gsn_text_ndc(wks, "mean", x0+w/2.+0.002, y0, txres)
       gsn_text_ndc(wks, "25th", x0+w/2.+0.002, y0-h1, txres)
       gsn_text_ndc(wks, "5th", x0+0.002, y0-h2, txres)
;       gsn_text_ndc(wks, "5th", x0+w/2.+0.002, y0-h1, txres)
;       gsn_text_ndc(wks, "min", x0+0.002, y0-h2, txres)
       txres@txJust = "BottomCenter"
       txres@txAngleF = 90.
       gsn_text_ndc(wks, "Ensemble range", x0-w/2.-0.014, y0, txres)
       gsn_text_ndc(wks, "(all members)", x0-w/2.-0.002, y0, txres)
       
       x = x0 + w*3.7
       plres@gsFillColor = lres@gsLineColor
       gsn_polygon_ndc(wks, (/x-w/2., x-w/2., x+w/2., x+w/2./), \
                            (/y0-h1/1.5, y0+h1/1.5, y0+h1/1.5, y0-h1/1.5/), plres)
       lres@gsLineColor := "white"
       lres@gsLineThicknessF = 1.
       gsn_polyline_ndc(wks, (/x-w/2.+0.0005, x+w/2.-0.0005/), (/y0, y0/), lres)
;       lres@gsLineThicknessF = 1.
;       lres@gsLineColor := plres@gsFillColor
;       gsn_polyline_ndc(wks, (/x-w/2., x-w/2., x+w/2., x+w/2., x-w/2./), \
;                             (/y0-h1/1.5, y0+h1/1.5, y0+h1/1.5, y0-h1/1.5, y0-h1/1.5/), lres)
       txres@txJust = "TopCenter"
       txres@txAngleF = 90.
       gsn_text_ndc(wks, "5-95%", x+w/2.+0.003, y0, txres)
       gsn_text_ndc(wks, "confidence int.", x+w/2.+0.015, y0, txres)
;;       gsn_text_ndc(wks, "of mean", x+w/2.+0.027, y0, txres)
       gsn_text_ndc(wks, "(models with   ", x+w/2.+0.027, y0, txres)
       gsn_text_ndc(wks, "  ~F34~3~F21~ 3 runs)", x+w/2.+0.039, y0, txres)
;       gsn_text_ndc(wks, "(equal ensemble size", x+w/2.+0.027, y0, txres)
;       gsn_text_ndc(wks, "   across models)", x+w/2.+0.039, y0, txres)
       mkres = True
       mkres@gsMarkerIndex = 16
       mkres@gsMarkerSizeF =  0.005
       gsn_polymarker_ndc(wks, (/x, x/), (/y0-h1/4., y0+h1/3./), mkres)
       lres@gsLineThicknessF = 0.5
       lres@gsLineColor = "black"
       gsn_polyline_ndc(wks, (/x, x-w/3./), (/y0+h1/3., y0+h1/1.5+0.005/), lres)
       txres@txAngleF = 0.
       txres@txJust = "BottomCenter"
       gsn_text_ndc(wks, "mean of", x-w/3., y0+h1/1.5+0.03, txres)
       gsn_text_ndc(wks, "each", x-w/3., y0+h1/1.5+0.018, txres)
       gsn_text_ndc(wks, "model", x-w/3., y0+h1/1.5+0.006, txres)

       
       lres@gsLineThicknessF = 1.
       txres@txFontHeightF = fontHeightL
       txres@txJust = "TopLeft"
       m = 0
       do n = 0, ListCount(legend)-1
         if (legend[n]@type .eq. "ensemble") then
           x = x0 + 0.14 + m/5 * 0.32
           y = y0 + h*1.2 - mod(m, 5)*h*0.55
           txres@txFontColor := legend[n]@color
           if (isStrSubset(legend[n], "piControl").and.count_unique_values(legend[n]@nruns).gt.1) then
             name = legend[n]+" ("+legend[n]@nmodels+" models, "
             do j = 0, dimsizes(legend[n]@nruns)-1
               if (legend[n]@nruns(j).gt.0) then
                 if (j.gt.0) then
                   name = name + ", "
                 end if
                 if (strlen(name).gt.38.and..not.isStrSubset(name, "~C~")) then
                   name = name+"~C~                 "
                 end if
                 name = name + alphabet(j)
                 do i = j+1, dimsizes(legend[n]@nruns)-1
                   if (legend[n]@nruns(i).eq.legend[n]@nruns(j)) then
                     name = name + ","+alphabet(i)
                     legend[n]@nruns(i) = -1
                   end if
                 end do
                 name = name + ": "+legend[n]@nruns(j)
               end if
             end do
             name = name + " segments)"
           elseif (isatt(legend[n], "nmodels2")) then
             name = legend[n]+" (a-c: "+legend[n]@nmodels+" models, "+legend[n]@nruns+" runs,~C~"\
                    +"           e-f: "+legend[n]@nmodels2+" models, "+legend[n]@nruns2+" runs)"
           else
             if (isStrSubset(legend[n], "piControl")) then
               name = legend[n]+" ("+legend[n]@nmodels+" models, "+legend[n]@nruns+" segments)"
             else
               name = legend[n]+" ("+legend[n]@nmodels+" models, "+legend[n]@nruns+" runs)"
             end if
           end if
           gsn_text_ndc(wks, name, x, y, txres)
           m = m + 1
           if (isStrSubset(name, "~C~")) then
             m = m + 1
           end if
         end if
       end do
     end if
     
     m = 0
     do n = 0, ListCount(legend)-1
       if (legend[n]@type .eq."single_obs") then
         x1 = x0 - 0.05 - m/4 * 0.14
         x2 = x1 - 0.044
         y = y0 + h*0.9 - mod(m, 4)*h*0.6
         legend[n]@x1 = x1
         legend[n]@x2 = x2
         legend[n]@y = y
         m = m + 1
       end if
     end do
     do n = 0, ListCount(legend)-1
       if (legend[n]@type .eq."ensemble_obs") then
         x1 = x0 - 0.05 - m/4 * 0.14
         x2 = x1 - 0.044
         y = y0 + h*0.9 - mod(m, 4)*h*0.6
         legend[n]@x1 = x1
         legend[n]@x2 = x2
         legend[n]@y = y
         m = m + 1
       end if
     end do
     do n = 0, ListCount(legend)-1
       if (any(legend[n]@type .eq. (/"single_obs", "ensemble_obs"/))) then
         x1 = legend[n]@x1
         x2 = legend[n]@x2
         y  = legend[n]@y
         if (legend[n]@type.eq."single_obs") then
           lres@gsLineColor := legend[n]@color
           lres@gsLineThicknessF = legend[n]@thick
           lres@gsLineDashPattern = legend[n]@pattern
           if (isatt(legend[n], "seg")) then
             lres@gsLineDashSegLenF = legend[n]@seg
           end if
           gsn_polyline_ndc(wks, (/x1, x2/), (/y, y/), lres)
         elseif (legend[n]@type.eq."ensemble_obs") then
           plres@gsFillColor := legend[n]@color
           plres@gsFillIndex = legend[n]@pattern
           gsn_polygon_ndc(wks, (/x1, x2, x2, x1/), (/y-h*0.2, y-h*0.2, y+h*0.2, y+h*0.2/), plres)
           if (plres@gsFillIndex.ne.0) then
             lres@gsLineThicknessF = 1.
             lres@gsLineColor = obsenslist@color(iobs)
             lres@gsLineDashPattern = 0
             gsn_polyline_ndc(wks, (/x1, x2, x2, x1, x1/), \
                              (/y-h*0.2, y-h*0.2, y+h*0.2, y+h*0.2, y-h*0.2/), lres)
           end if
           txres@txFontColor := legend[n]@color
           txres@txJust = "CenterCenter"
           txres@txFontHeightF = fontHeightS
           txres@txBackgroundFillColor = "white"
           txres@txPerimColor = "transparent"
           txres@txPerimSpaceF = 0.1
           gsn_text_ndc(wks, "max", (x1+x2)/2., y+h*0.2, txres)
           gsn_text_ndc(wks, "min", (x1+x2)/2., y-h*0.2, txres)
         end if
         x = x2 - 0.01
         x = x2 - 0.005
         txres@txFontColor := legend[n]@color
         txres@txJust = "CenterRight"
         txres@txFontHeightF = fontHeightL
         txres@txBackgroundFillColor = "transparent"
         name = legend[n]
         gsn_text_ndc(wks, name, x, y, txres)
       end if
     end do
  end if
  
end
