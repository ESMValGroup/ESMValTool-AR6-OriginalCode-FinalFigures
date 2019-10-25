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
;    20190316 kosa_yu: written.
;    20191017 kosaka_yu: updated for ESMValTool 2.0.0b2
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

function pdf_polygon(pdf)
local polypdf
begin
  imax = dimsizes(pdf)
;  if (do_histogram) then
    polypdf = new((/2, 2*imax+2/), typeof(pdf))
    if (isatt(pdf, "bin_bounds")) then
      do i = 0, imax
        polypdf(0, 2*i  ) = pdf@bin_bounds(i)
        polypdf(0, 2*i+1) = pdf@bin_bounds(i)
      end do
    else
      polypdf(0, 0:1) = pdf@bin_center(0) - (pdf@bin_center(1) - pdf@bin_center(0)) / 2.
      do i = 1, imax-2
        polypdf(0, 2*i  ) = (pdf@bin_center(i) + pdf@bin_center(i-1)) / 2.
        polypdf(0, 2*i+1) = (pdf@bin_center(i) + pdf@bin_center(i-1)) / 2.
      end do
      polypdf(0, 2*imax:2*imax+1) = pdf@bin_center(imax-1) - (pdf@bin_center(imax-1) - pdf@bin_center(imax-2)) / 2.
    end if
    polypdf(1, 0) = 0.
    do i = 0, imax-1
      polypdf(1, 2*i+1) = pdf(i)
      polypdf(1, 2*i+2) = pdf(i)
    end do
    polypdf(1, 2*imax+1) = 0.
;  else
;    polypdf = new((/2, imax+2/), typeof(pdf))
;    polypdf(0, 0) = pdf@bin_center(0)
;    polypdf(0, 1:imax) = pdf@bin_center
;    polypdf(1, imax+1) = pdf@bin_center(imax-1)
;    polypdf(1, 0) = 0.
;    polypdf(1, 1:imax) = pdf
;    polypdf(1, imax+1) = 0.
;  end if
  return(polypdf)
end
      
begin

  enter_msg(DIAG_SCRIPT, "")

  if (isatt(diag_script_info, "layout")) then
    layout = diag_script_info@layout
  end if
  
  scale = 10.  ; trend in "per [scale] years" 

  input_dirs = diag_script_info@input_files
  ndirs = dimsizes(input_dirs)
  hemisphere = new(ndirs, "string")
  season     = new(ndirs, "string")

  do n = 0, ndirs-1
    if (isStrSubset(input_dirs(n), "NH_")) then
      hemisphere(n) = "NH"
    elseif (isStrSubset(input_dirs(n), "SH_")) then
      hemisphere(n) = "SH"
    else
      error_msg("f", DIAG_SCRIPT, "", \
                "Failed to detect hemisphere.")
    end if
    if (isStrSubset(input_dirs(n), "_DJF/")) then
      season(n) = "DJF"
    elseif (isStrSubset(input_dirs(n), "_MAM/")) then
      season(n) = "MAM"
    elseif (isStrSubset(input_dirs(n), "_JJA/")) then
      season(n) = "JJA"
    elseif (isStrSubset(input_dirs(n), "_SON/")) then
      season(n) = "SON"
    elseif (isStrSubset(input_dirs(n), "_ANN/")) then
      season(n) = "ANN"
    else
      error_msg("f", DIAG_SCRIPT, "", \
                "Failed to detect season.")
    end if    
  end do

  hemispheres = get_unique_values(hemisphere)
  seasons     = get_unique_values(season)
  nhemispheres = dimsizes(hemispheres)
  nseasons = dimsizes(seasons)

  sqsort(hemispheres)
  j = 0
  do i = 0, nseasons-1
    do j = i+1, nseasons-1
      if (seasons(j).eq."ANN") then
        seasons(j) = seasons(i)
        seasons(i) = "ANN"
        break
      elseif (seasons(j).eq."DJF") then
        seasons(j) = seasons(i)
        seasons(i) = "DJF"
        break
      elseif (seasons(j).eq."MAM") then
        seasons(j) = seasons(i)
        seasons(i) = "MAM"
        break
      elseif (seasons(j).eq."JJA") then
        seasons(j) = seasons(i)
        seasons(i) = "JJA"
        break
      elseif (seasons(j).eq."SON") then
        seasons(j) = seasons(i)
        seasons(i) = "SON"
        break
      end if
    end do
  end do
  if (.not.isvar("layout")) then
    layout = (/nhemispheres, nseasons/)
  end if

  layout_i = new(ndirs, "integer")
  layout_j = new(ndirs, "integer")
  if (all(layout.eq.(/nhemispheres, nseasons/))) then
    do n = 0, ndirs-1
      layout_i(n) = ind(hemisphere(n).eq.hemispheres)
      layout_j(n) = ind(season(n).eq.seasons)
    end do
  elseif (any(layout.eq.1)) then
    layout_ij = new(ndirs, "integer")
    rank      = new(ndirs, "integer")
    do n = 0, ndirs - 1
      if (hemisphere(n).eq."NH") then
        rank(n) = 10
      elseif (hemisphere(n).eq."SH") then
        rank(n) = 20
      end if
      if (season(n).eq."ANN") then
        rank(n) = rank(n) + 1
      elseif (season(n).eq."DJF") then
        rank(n) = rank(n) + 2
      elseif (season(n).eq."MAM") then
        rank(n) = rank(n) + 3
      elseif (season(n).eq."JJA") then
        rank(n) = rank(n) + 4
      elseif (season(n).eq."SON") then
        rank(n) = rank(n) + 5
      end if
    end do
    do n = 0, ndirs-1
      layout_ij(minind(rank)) = n
      rank(minind(rank)) = rank@_FillValue
    end do
    delete(rank)
    if (layout(0).eq.1) then
      layout_i = 0
      layout_j = layout_ij
    elseif (layout(1).eq.1) then
      layout_i = layout_ij
      layout_j = 0
    end if
    delete(layout_ij)
  end if
  
  wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                "hadleycell_pdfs")

  pdfs = new(layout, "graphic")
  lines = new(100, "graphic")
  iline = 0
  titles = new(layout, "string")
  
  color_obs = (/ "blue3", "darkgreen", "darkslategray", "darkseagreen" /)
  color_pdf = "pink2"
  color_mme = "brown4"
  
  res = True
  res@gsnDraw = False
  res@gsnFrame = False
  res@tmYROn = False
  res@tmXTOn = False
  res@tmYLLabelFontHeightF = 0.016
  res@tmXBLabelFontHeightF = 0.016
  res@tmYLPrecision = 2
  res@tmXBPrecision = 2
  res@tmXBLabelDeltaF = -0.5
  res@tmYLLabelDeltaF = -0.5
  res@vpHeightF = 0.35
  res@vpWidthF = 0.4
  res@tiMainOffsetYF = -0.01
  res@tiMainFontThicknessF = 2.
  res@gsnXYBarChart = True
  res@gsnRightXRefLineColor = color_pdf
  res@xyLineColors = (/"transparent", "transparent"/)
  res@gsnXRefLine = 0.
  res@gsnYRefLine = 0.
  res@gsnYRefLineThicknessF = 0.
  res@gsnYRefLineColor = "transparent"
  
  lres = True
;  lres@gsnDraw = False
;  lres@gsnFrame = False
  lres@gsLineThicknessF = 2.

  refres = True
  refres@gsLineColor = "grey20"
  refres@gsLineThicknessF = 0.5
  refres@gsLineDashPattern = 1
  refres@gsLineDashSegLenF = 0.02

  txres         = True
;  txres@gsnDraw = False
;  txres@gsnFrame = False
  txres@txFontHeightF = 0.01
  txres@txJust = "CenterLeft"

  plotLegend = True
; ============================== DRAW PDFs ==============================
  
  varname = "lat"
  do ndir = 0, ndirs-1
print(""+input_dirs(ndir))
    paths = systemfunc("ls "+input_dirs(ndir)+"/*.nc")
    if (dimsizes(paths).eq.0) then
      continue
    end if

    do i = 0, dimsizes(paths)-1
      f = addfile(paths(i), "r")
      if (isStrSubset(paths(i), "_MME.nc")) then
        MME = f->$varname$
      elseif (isStrSubset(paths(i), "MMEmean.nc")) then
        MMEmean = f->$varname$
      else
        if (isvar("Obs")) then
          tmp = Obs
          delete(Obs)
          data = f->$varname$
          Obs = array_append_record(tmp, data, 0)
          delete(Obs@dataset)
          Obs@dataset = array_append_record(tmp@dataset, data@dataset, 0)
          delete(tmp)
          delete(data)
        else
          Obs = f->$varname$
        end if
      end if
    end do

    MME = MME * scale
    MMEmean = MMEmean * scale
    Obs = Obs * scale

; -------------------------------------
; Calculate PDF
      
    nbins = 40   ; For probability distribution
  
    resPDF = True
    resPDF@bin_min = -0.1 * scale
    resPDF@bin_max =  0.1 * scale
    
    PDF = pdfx(MME, nbins, resPDF)
    PDF = PDF / tofloat(dimsizes(MME))
    delete(MME)
    PDF@long_name = ""
      
; -------------------------------------
; Plot PDF

    res@trYMinF = -0.1*scale
    res@trYMaxF = 0.1*scale
    res@tmYLMode = "Manual"
    res@tmYLTickSpacingF = 0.05*scale
    res@trXMinF = 0.
    res@trXMaxF = 8. / tofloat(nbins)
    res@tmXBMaxTicks = 3

    res@tmXBLabelFontHeightF = 0.016
    res@tmYLLabelFontHeightF = 0.016

    i = layout_i(ndir)
    j = layout_j(ndir)
    pdfs(i, j) = gsn_csm_xy(wks, PDF, PDF@bin_center, res)
    lres@gsLineThicknessF = 2.
    lres@gsLineColor = color_mme
    lines(iline) = gsn_add_polyline(wks, pdfs(i, j), \
                                      (/res@trXMinF, res@trXMaxF/), \
                                      (/MMEmean, MMEmean/), lres)
    iline = iline + 1

    if (plotLegend) then
      y = 0.5 - (max(layout_i)+1)*0.15
      x = 0.05
      gsn_polyline_ndc(wks, (/x, x+0.03/), (/y, y/), lres)
      gsn_text_ndc(wks, "MME mean", x+0.04, y, txres)
    end if
    do ii = 0, dimsizes(Obs)-1
      lres@gsLineThicknessF = 0.5
      lres@gsLineColor = color_obs(ii)
      lines(iline) = gsn_add_polyline(wks, pdfs(i, j), \
                                      (/res@trXMinF, res@trXMaxF/), \
                                      (/Obs(ii), Obs(ii)/), lres)
      iline = iline + 1
      if (plotLegend) then
        x = 0.05 + 0.9 / tofloat(dimsizes(Obs)+1) * tofloat(ii+1)
;          lines(iline) = gsn_polyline_ndc(wks, (/x, x+0.03/), (/0.2, 0.2/), lres)
;          iline = iline + 1
;          lines(iline) = gsn_text_ndc(wks, Obs@dataset(ii), x+0.04, 0.2, txres)
;          iline = iline + 1
        gsn_polyline_ndc(wks, (/x, x+0.03/), (/y, y/), lres)
        gsn_text_ndc(wks, Obs@dataset(ii), x+0.04, y, txres)
      end if
    end do
    plotLegend = False
    lines(iline) = gsn_add_polyline(wks, pdfs(i, j), \
                                    (/res@trXMinF, res@trXMaxF/), \
                                    (/0., 0./), refres)
    iline = iline + 1

    titles(i, j) = hemisphere(ndir)+" "+season(ndir)
      
    delete(PDF)
    delete(MMEmean)
    delete(Obs)
    delete(paths)
  end do

  titles := ndtooned(titles)
  alphabet = str_split_by_length("abcdefghijklmnopqrstuvwxyz", 1)
  do i = 0, dimsizes(titles)-1
    titles(i) = "("+alphabet(i)+") "+titles(i)
  end do
  
  pres = True
;  pres@gsnMaximize = True
  pres@gsnPanelXWhiteSpacePercent = 1.
  pres@gsnPanelYWhiteSpacePercent = 8.
  pres@gsnPanelFigureStrings = titles
  pres@gsnPanelFigureStringsFontHeightF = 0.018
  pres@gsnPanelFigureStringsPerimOn = False
  pres@amJust   = "TopCenter"
  pres@amOrthogonalPosF = -0.68
  pres@amParallelPosF = 0.
  gsn_panel(wks, ndtooned(pdfs), layout, pres)

end
