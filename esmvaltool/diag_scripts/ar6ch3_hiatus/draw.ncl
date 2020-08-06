; draw_hiatus
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs:
;
; History
;    20200531 kosaka_yu: written.
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

function addline_PDF(x, y, x0)
local z, ix
begin
  z = new((/2, 2/), typeof(x))
;  z@_FillValue = y@_FillValue
  z = z@_FillValue
  ix = ind_nearest_coord(x0, x, 0)
  if (x(ix).gt.x0) then
    ix = ix - 1
  end if
  if (x(ix).eq.x(ix+1)) then
    ix = ix+1
  end if
  if (ix.lt.0.or.ix.ge.dimsizes(x)) then
    return(z)
  end if
;print("ix = "+ix)
  z(1, 1)  = ((x(ix+1) - x0) * y(ix) + (x0 - x(ix)) * y(ix+1)) / (x(ix+1) - x(ix))
  z(1, 0) = 0.
  z(0, 1) = x0
  z(0, 0) = x0
  return(z)
end

function pdf_polygon(pdf, do_histogram)
local polypdf
begin
  imax = dimsizes(pdf)
  if (do_histogram) then
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
  else
    polypdf = new((/2, imax+2/), typeof(pdf))
    polypdf(0, 0) = pdf@bin_center(0)
    polypdf(0, 1:imax) = pdf@bin_center
    polypdf(1, imax+1) = pdf@bin_center(imax-1)
    polypdf(1, 0) = 0.
    polypdf(1, 1:imax) = pdf
    polypdf(1, imax+1) = 0.  
  end if
  return(polypdf)
end

function tail_PDF(x, y, thres)
local z, ix, z0
begin
  z = new((/2, dimsizes(x)/), typeof(x))
;  z@_FillValue = y@_FillValue
  z = z@_FillValue
  z(1, :) = where(x.lt.thres, y, z@_FillValue)
  z(0, :) = x 
  ix = ind_nearest_coord(thres, x, 0)
  if (x(ix).gt.thres) then
    ix = ix - 1
  elseif (x(ix).eq.x(ix+1)) then
    ix = ix + 1
  end if
  if (ix.lt.0.or.ix.ge.dimsizes(x)) then
    return(z)
  end if
  z0 = addline_PDF(x, y, thres)
  z(1, ix+1)  = z0(1, 1)
  z(1, ix+2:) = z0(1, 0)
  z(0, ix+1) = thres
  z(0, ix+2) = thres
  z(0, ix+3:) = z@_FillValue
  return(z)
end

function Res_drawPDF(color, line_thickness, fill_pattern, fill_opacity)
local res
begin
  res = True
  res@gsLineColor := color
  res@gsLineThicknessF = line_thickness
  res@gsFillColor := color
  res@gsFillIndex = fill_pattern
  res@gsFillOpacityF = fill_opacity
  return(res)
end

;function modify_tail(pdf)
;local imax, newpdf
;begin
;  imax = dimsizes(pdf)
;  newpdf = new((/2, imax+2/), typeof(pdf))
;  newpdf(0, 0) = pdf@bin_center(0)
;  newpdf(0, 1:imax) = pdf@bin_center
;  newpdf(1, imax+1) = pdf@bin_center(imax-1)
;  newpdf(1, 0) = 0.
;  newpdf(1, 1:imax) = pdf
;  newpdf(1, imax+1) = 0.
;  return(newpdf)
;end

begin

  enter_msg(DIAG_SCRIPT, "")

; ======================================================================
;
;                           0. Configuration
;
; ======================================================================

  scale = 10.  ; trend in "per [scale] years" 
  long_name = "GMST trend"
  unit = "K/decade"
  input_dir = diag_script_info@input_files + "/"
  
  individual_cmip = True
  if (isatt(diag_script_info, "pdf_individual_cmip")) then  ;
    individual_cmip = diag_script_info@pdf_individual_cmip
  end if
  
  pdf_style = "smooth"
  if (isatt(diag_script_info, "pdf_style")) then  ; pdf
    pdf_style = diag_script_info@pdf_style
  end if
  if (pdf_style.eq."smooth") then
    do_kde_n_test = True
  else
    do_kde_n_test = False
  end if
  if (pdf_style.eq."histogram") then
    do_histogram = True
  else
    do_histogram = False
  end if
  
;  color_cmip = "orange"
;  color_cmip6 = "coral2"
;  color_cmip5 = "goldenrod2"
;  color_cmip_mean = "white"
;  color_cmip_selected = "orange4"
;  color_cmip6_selected = "coral4"
;  color_cmip5_selected = "darkorange4"
;  color_obs_ens = (/ "darkorchid", "deepskyblue3"/)
;  color_obs_single = (/ "blue3", "darkgreen", "darkslategray", "darkseagreen" /)
;  pdf_opacity = 0.5
;  color_cmip = "seagreen"
;  color_cmip6 = "deepskyblue2"
;  color_cmip5 = "green3" ;"lightseagreen"
;  color_cmip_mean = "white"
;  color_cmip_selected = "navyblue"
;  color_cmip6_selected = "navyblue"
;  color_cmip5_selected = "navyblue"
;  color_obs_ens = (/ "mediumorchid", "mediumturquoise"/)
;  color_obs_single = (/ "red3", "gold4", "orangered4", "coral4" /)
;  pdf_opacity = 0.2

  color_cmip = "seagreen"
  color_cmip6 = (/204., 35., 35./)/255. 
  color_cmip5 = (/37., 81., 204./)/255. 
  color_cmip_mean = color_cmip
  color_cmip6_mean = color_cmip6
  color_cmip5_mean = color_cmip5
  color_cmip_selected = "navyblue"
  color_cmip6_selected = "navyblue"
  color_cmip5_selected = "navyblue"
  color_obs_ens = (/ "grey40", "black"/)
  color_obs_single = (/ "blue2", "green4", "purple3", "red" /)
  pdf_opacity = 0.2

  periphery_thickness = 2.
  line_thickness = 3.

; ======================================================================
;
;              1. Read GMST trends and composite trend patterns
;
; ======================================================================

  do ii = 0, dimsizes(input_dir)-1
    tmp = systemfunc("ls "+input_dir(ii))
    tmp = input_dir(ii) + tmp
    if (isvar("files")) then
      files := array_append_record(files, tmp, 0)
    else
      files = tmp
    end if
    delete(tmp)
  end do
    
  do ii = 0, dimsizes(files)-1
    if (isStrSubset(files(ii), "HiatusTrendPattern_composite.nc")) then
      f = addfile(files(ii), "r")
      TstrendCMIPhiatus = f->trend
      TstrendCMIPhiatus = TstrendCMIPhiatus * scale      
    elseif (isStrSubset(files(ii), "PostTrendPattern_composite.nc")) then
      f = addfile(files(ii), "r")
      TstrendCMIPpost = f->trend
      TstrendCMIPpost = TstrendCMIPpost * scale      
    end if
  end do

  if (TstrendCMIPhiatus@blend_GST.eq.1) then
    GST = "GMST"
  else
    GST = "GSAT"
  end if
  if (TstrendCMIPhiatus@mask_GST.eq.1) then
    masking = "masked"
  else
    masking = "unmasked"
  end if
  composite_threshold = TstrendCMIPhiatus@composite_threshold
  composite_threshold_GSTtrend_composite = TstrendCMIPhiatus@threshold_trendGSThiatus * scale

  exit
  
  do ii = 0, dimsizes(files)-1
    f = addfile(files(ii), "r")
    if (isStrSubset(files(ii), "trend"+GST+"_"+masking+"_CMIP6.txt")) then
      GSTtrendCMIP6hiatus = f->trend
      GSTtrendCMIP6hiatus = GSTtrendCMIP6hiatus * scale
      thresGSTtrendCMIP6 = GSTtrendCMIP6hiatus@composite_threshold * scale
      syear_hiatus = GMSTtrendCMIP6Hiatus@syear_trend
      eyear_hiatus = GMSTtrendCMIP6Hiatus@eyear_trend
    elseif (isStrSubset(files(ii), "trendGMST_post_cmip6.nc")) then
      GMSTtrendCMIP6Post = f->trend
      GMSTtrendCMIP6Post = GMSTtrendCMIP6Post * scale
      syear_post = GMSTtrendCMIP6Post@syear_trend
      eyear_post = GMSTtrendCMIP6Post@eyear_trend
    elseif (isStrSubset(files(ii), "trendGMST_hiatus_cmip5.nc")) then
      GMSTtrendCMIP5Hiatus = f->trend
      GMSTtrendCMIP5Hiatus = GMSTtrendCMIP5Hiatus * scale
      thresGMSTtrendCMIP5 = GMSTtrendCMIP5Hiatus@composite_threshold * scale
      syear_hiatus = GMSTtrendCMIP5Hiatus@syear_trend
      eyear_hiatus = GMSTtrendCMIP5Hiatus@eyear_trend
    elseif (isStrSubset(files(ii), "trendGMST_post_cmip5.nc")) then
      GMSTtrendCMIP5Post = f->trend
      GMSTtrendCMIP5Post = GMSTtrendCMIP5Post * scale
      syear_post = GMSTtrendCMIP5Post@syear_trend
      eyear_post = GMSTtrendCMIP5Post@eyear_trend
    elseif (isStrSubset(files(ii), "trendGMST_hiatus_")) then
      if (.not.isdefined("GMSTtrendObsHiatus")) then
        GMSTtrendObsHiatus = NewList("fifo")
      end if
      tmp = f->trend
      tmp = tmp * scale
      ListAppend(GMSTtrendObsHiatus, new(dimsizes(tmp), typeof(tmp)))
      i = ListCount(GMSTtrendObsHiatus)-1
      GMSTtrendObsHiatus[i] = tmp
      syear_hiatus = tmp@syear_trend
      eyear_hiatus = tmp@eyear_trend
      delete(tmp)
    else
      error_msg("f", DIAG_SCRIPT, "", "Unrecognized file: "+files(ii))
    end if
  end do

; ======================================================================
;
;                       2. Make PDFs of GMST trends
;
; ======================================================================
  
  if (do_kde_n_test) then
    nbins = 50 ; For probability distribution
    nbinsobs = 100
  else
    nbins = 25   ; For probability distribution
    nbinsobs = 40
  end if
    
  resPDF = True
  resPDF@bin_min = -0.02 * scale
  resPDF@bin_max =  0.08 * scale
  
  resPDFobs = True
  resPDFobs@bin_min = -0.00 * scale
  resPDFobs@bin_max =  0.02 * scale
  if (isdefined("GMSTtrendObsHiatus")) then
    do ii = 0, ListCount(GMSTtrendObsHiatus)-1
      if (dimsizes(GMSTtrendObsHiatus[ii]).gt.1) then
        if (.not.isdefined("PDFhiatusObs")) then
          PDFhiatusObs = NewList("fifo")
        end if
;        ListAppend(PDFhiatusObs, new(nbins, "double"))
        ListAppend(PDFhiatusObs, new(nbinsobs, "double"))
        i = ListCount(PDFhiatusObs)-1
        if (do_kde_n_test) then
          bins = fspan(resPDFobs@bin_min, resPDFobs@bin_max, nbinsobs)
          PDFhiatusObs[i] = kde_n_test(GMSTtrendObsHiatus[ii], bins, 0)
          PDFhiatusObs[i]@bin_center = bins
          delete(bins)
        else
;          PDFhiatusObs[i] = pdfx(GMSTtrendObsHiatus[ii], nbins, resPDF)
          PDFhiatusObs[i] = pdfx(GMSTtrendObsHiatus[ii], nbinsobs, resPDFobs)
          PDFhiatusObs[i] = PDFhiatusObs[i] / 100.
        end if
;        PDFhiatusObs[i] = PDFhiatusObs[i] / max(PDFhiatusObs[i])
        PDFhiatusObs[i] = PDFhiatusObs[i] / PDFhiatusObs[i]@bin_spacing
        PDFhiatusObs[i]@dataset = GMSTtrendObsHiatus[ii]@dataset
      else
        if (.not.isdefined("SingleValhiatusObs")) then
          SingleValhiatusObs = NewList("fifo")
        end if
        ListAppend(SingleValhiatusObs, new(1, "double"))
        i = ListCount(SingleValhiatusObs)-1
        SingleValhiatusObs[i] = GMSTtrendObsHiatus[ii]
        SingleValhiatusObs[i]@dataset = GMSTtrendObsHiatus[ii]@dataset
      end if
    end do
  end if

  if (individual_cmip) then
    if (isvar("GMSTtrendCMIP5Hiatus")) then
      if (do_kde_n_test) then
        bins = fspan(resPDF@bin_min, resPDF@bin_max, nbins)
        PDFhiatusCMIP5 = kde_n_test(GMSTtrendCMIP5Hiatus, bins, 0)
        PDFhiatusCMIP5@bin_center = bins
        delete(bins)
      else
        PDFhiatusCMIP5 = pdfx(GMSTtrendCMIP5Hiatus, nbins, resPDF)
        PDFhiatusCMIP5 = PDFhiatusCMIP5 / 100.
      end if
;      PDFhiatusCMIP5 = PDFhiatusCMIP5 / max(PDFhiatusCMIP5)
      PDFhiatusCMIP5 = PDFhiatusCMIP5 / PDFhiatusCMIP5@bin_spacing
      PDFhiatusCMIP5@mean = avg(GMSTtrendCMIP5Hiatus)
    end if
    if (isvar("GMSTtrendCMIP5Post")) then
      if (do_kde_n_test) then
        bins = fspan(resPDF@bin_min, resPDF@bin_max, nbins)
        PDFpostCMIP5 = kde_n_test(GMSTtrendCMIP5Post, bins, 0)
        PDFpostCMIP5@bin_center = bins
        delete(bins)
      else
        PDFpostCMIP5   = pdfx(GMSTtrendCMIP5Post, nbins, resPDF)
        PDFpostCMIP5 = PDFpostCMIP5 / 100.
      end if
;      PDFpostCMIP5   = PDFpostCMIP5 / max(PDFpostCMIP5)
      PDFpostCMIP5   = PDFpostCMIP5 / PDFpostCMIP5@bin_spacing
      PDFpostCMIP5@mean = avg(GMSTtrendCMIP5Post)
    end if
    if (isvar("GMSTtrendCMIP6Hiatus")) then
      if (do_kde_n_test) then
        bins = fspan(resPDF@bin_min, resPDF@bin_max, nbins)
        PDFhiatusCMIP6 = kde_n_test(GMSTtrendCMIP6Hiatus, bins, 0)
        PDFhiatusCMIP6@bin_center = bins
        delete(bins)
      else
        PDFhiatusCMIP6 = pdfx(GMSTtrendCMIP6Hiatus, nbins, resPDF)
        PDFhiatusCMIP6 = PDFhiatusCMIP6 / 100.
      end if
;      PDFhiatusCMIP6 = PDFhiatusCMIP6 / max(PDFhiatusCMIP6)
      PDFhiatusCMIP6 = PDFhiatusCMIP6 / PDFhiatusCMIP6@bin_spacing
      PDFhiatusCMIP6@mean = avg(GMSTtrendCMIP6Hiatus)
    end if
    if (isvar("GMSTtrendCMIP6Post")) then
;      if (num(.not.ismissing(GMSTtrendCMIP6Post)).gt.num(.not.ismissing(GMSTtrendCMIP6Hiatus))/2) then
        if (do_kde_n_test) then
          bins = fspan(resPDF@bin_min, resPDF@bin_max, nbins)
          PDFpostCMIP6 = kde_n_test(GMSTtrendCMIP6Post, bins, 0)
          PDFpostCMIP6@bin_center = bins
          delete(bins)
        else
          PDFpostCMIP6   = pdfx(GMSTtrendCMIP6Post, nbins, resPDF)
          PDFpostCMIP6   = PDFpostCMIP6 / 100.
        end if
;        PDFpostCMIP6   = PDFpostCMIP6 / max(PDFpostCMIP6)
        PDFpostCMIP6   = PDFpostCMIP6 / PDFpostCMIP6@bin_spacing
        PDFpostCMIP6@mean = avg(GMSTtrendCMIP6Post)
;      end if
    end if
  else
    if (isvar("GMSTtrendCMIP5Hiatus").and.isvar("GMSTtrendCMIP6Hiatus")) then
      GMSTtrendCMIPHiatus = new(dimsizes(GMSTtrendCMIP5Hiatus)+dimsizes(GMSTtrendCMIP6Hiatus), typeof(GMSTtrendCMIP5Hiatus))
      GMSTtrendCMIPHiatus!0 = "ensemble"
      GMSTtrendCMIPHiatus&ensemble = ispan(1, dimsizes(GMSTtrendCMIPHiatus), 1)
      GMSTtrendCMIPHiatus = (/array_append_record(GMSTtrendCMIP5Hiatus, GMSTtrendCMIP6Hiatus, 0)/)
      copy_VarAtts(GMSTtrendCMIP5Hiatus, GMSTtrendCMIPHiatus)
      delete(GMSTtrendCMIPHiatus@do_composite)
      GMSTtrendCMIPHiatus@do_composite = array_append_record(GMSTtrendCMIP5Hiatus@do_composite, \
                                                             GMSTtrendCMIP6Hiatus@do_composite, 0)
;      printVarSummary(GMSTtrendCMIP5Hiatus)
;      printVarSummary(GMSTtrendCMIP6Hiatus)
;      printVarSummary(GMSTtrendCMIPHiatus)
    elseif (isvar("GMSTtrendCMIP5Hiatus")) then
      GMSTtrendCMIPHiatus = GMSTtrendCMIP5Hiatus
    elseif (isvar("GMSTtrendCMIP6Hiatus")) then
      GMSTtrendCMIPHiatus = GMSTtrendCMIP6Hiatus
    end if
    if (isvar("GMSTtrendCMIPHiatus")) then
      if (do_kde_n_test) then
        bins = fspan(resPDF@bin_min, resPDF@bin_max, nbins)
        PDFhiatus = kde_n_test(GMSTtrendCMIPHiatus, bins, 0)
        PDFhiatus@bin_center = bins
        delete(bins)
      else
        PDFhiatus = pdfx(GMSTtrendCMIPHiatus, nbins, resPDF)
        PDFhiatus = PDFhiatus / 100.
      end if
;      PDFhiatus = PDFhiatus / max(PDFhiatus)
      PDFhiatus = PDFhiatus / PDFhiatus@bin_spacing
      PDFhiatus@mean = avg(GMSTtrendCMIPHiatus)
    end if
    if (isvar("GMSTtrendCMIP5Post").and.isvar("GMSTtrendCMIP6Post")) then
      GMSTtrendCMIPPost = new(dimsizes(GMSTtrendCMIP5Post)+dimsizes(GMSTtrendCMIP6Post), typeof(GMSTtrendCMIP5Post))
      GMSTtrendCMIPPost!0 = "ensemble"
      GMSTtrendCMIPPost&ensemble = ispan(1, dimsizes(GMSTtrendCMIPPost), 1)
      GMSTtrendCMIPPost = (/array_append_record(GMSTtrendCMIP5Post, GMSTtrendCMIP6Post, 0)/)
      copy_VarAtts(GMSTtrendCMIP5Post, GMSTtrendCMIPPost)
      delete(GMSTtrendCMIPPost@do_composite)
      GMSTtrendCMIPPost@do_composite = array_append_record(GMSTtrendCMIP5Post@do_composite, \
                                                           GMSTtrendCMIP6Post@do_composite, 0)
    elseif (isvar("GMSTtrendCMIP5Post")) then
      GMSTtrendCMIPPost = GMSTtrendCMIP5Post
    elseif (isvar("GMSTtrendCMIP6Post")) then
      GMSTtrendCMIPPost = GMSTtrendCMIP6Post
    end if
    if (isvar("GMSTtrendCMIPPost")) then
      if (do_kde_n_test) then
        bins = fspan(resPDF@bin_min, resPDF@bin_max, nbins)
        PDFpost = kde_n_test(GMSTtrendCMIPPost, bins, 0)
        PDFpost@bin_center = bins
        delete(bins)
      else
        PDFpost = pdfx(GMSTtrendCMIPPost, nbins, resPDF)
        PDFpost = PDFpost / 100.
      end if
;      PDFpost = PDFpost / max(PDFpost)
      PDFpost = PDFpost / PDFpost@bin_spacing
      PDFpost@mean = avg(GMSTtrendCMIPPost)
    end if
  end if

  if (isvar("tmp")) then
    delete(tmp)
  end if

; ======================================================================
;
;                            3. Plot PDFs
;
; ======================================================================

  if (isvar("thresGMSTtrendCMIP5").and.isvar("thresGMSTtrendCMIP6")) then
    if (thresGMSTtrendCMIP5.eq.thresGMSTtrendCMIP6) then
      nthres = 1
      thresGMSTtrend = thresGMSTtrendCMIP5
    else
      nthres = 2
    end if
  elseif (isvar("thresGMSTtrendCMIP5").xor.isvar("thresGMSTtrendCMIP6")) then
    nthres = 1
  end if

  wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                "hiatus_and_posthiatus_pdfs")

  pdfs = new(2, "graphic")
  dummy = new(100, "graphic")
  res = True
  res@gsnDraw = False
  res@gsnFrame = False
  res@trYMinF = 0.
  res@trYMaxF = 7.
  res@trXMinF = -0.02*scale
  res@trXMaxF = 0.06*scale
  res@tmYROn = False
  res@tmXTOn = False
  res@tmYLLabelFontHeightF = 0.016
  res@tmXBLabelFontHeightF = 0.016
  res@tmYLPrecision = 1
  res@tmXBPrecision = 1
  res@vpXF = 0.09
  res@vpYF = 0.9
  res@vpHeightF = 0.35
  res@vpWidthF = 0.4
  res@tiMainString = "(a) "+syear_hiatus+"-"+eyear_hiatus+" GMST trend"
  res@tiMainOffsetYF = -0.01
  res@tiMainFontThicknessF = 2.
  res@tiYAxisString = "Probability Density [("+unit+")~S~-1~N~]"
  res@tiYAxisFontHeightF = 0.015
  txres         = True
  txres@gsnDraw = False
  txres@gsnFrame = False
  txres@txFontHeightF = 0.01
  txres@txJust = "TopRight"
  
; 3.1 Plot hiatus PDF
  xaxis = (/-1., 1./)
  yaxis = (/0., 0./)
  xaxis@long_name = long_name+"["+unit+"]"
  pdfs(0) = gsn_csm_xy(wks, (/-1., 1./), (/ 0., 0./), res)
  idummy = 0
  ilabel = 0
  xlabel = 0.48
  ylabel = 0.89
  if (isvar("PDFhiatus")) then  ; combined cmip5+cmip6
    ; CMIP PDF fill
    resp = Res_drawPDF(color_cmip, periphery_thickness, 0, pdf_opacity)
;    dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
;                                    PDFhiatus@bin_center, PDFhiatus, resp)
;    tmp = modify_tail(PDFhiatus)
    polyPDFhiatus = pdf_polygon(PDFhiatus, do_histogram)
    dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
                                    polyPDFhiatus(0, :), polyPDFhiatus(1, :), resp)
    idummy = idummy + 1
;    delete(tmp)
    ; CMIP PDF mean
    tmp = addline_PDF(polyPDFhiatus(0, :), polyPDFhiatus(1, :), PDFhiatus@mean)
    resp = Res_drawPDF(color_cmip_mean, periphery_thickness, 0, 1.)
    dummy(idummy) = gsn_add_polyline(wks, pdfs(0), \
                                     tmp(0, :), tmp(1, :), resp)
    delete(tmp)
    idummy = idummy + 1
    ; CMIP PDF periphery
    resp = Res_drawPDF(color_cmip, periphery_thickness, 0, 1.)
    dummy(idummy) = gsn_add_polyline(wks, pdfs(0), \
                                     polyPDFhiatus(0, :), polyPDFhiatus(1, :), resp)
    idummy = idummy + 1
    if (nthres.eq.1) then
      tmp = tail_PDF(polyPDFhiatus(0, :), polyPDFhiatus(1, :), thresGMSTtrend)
      resp = Res_drawPDF(color_cmip_selected, periphery_thickness, 0, pdf_opacity)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
                                      tmp(0, :), tmp(1, :), resp)
      idummy = idummy + 1
      delete(tmp)
    elseif (nthres.eq.2) then
      polyPDFhiatusCMIP5 = pdf_polygon(PDFhiatusCMIP5, do_histogram)
      tmp = tail_PDF(polyPDFhiatusCMIP5(0, :), polyPDFhiatusCMIP5(1, :), thresGMSTtrendCMIP5)
      resp = Res_drawPDF(color_cmip5_selected, periphery_thickness, 0, pdf_opacity)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
                                      tmp(0, :), tmp(1, :), resp)
      idummy = idummy + 1
;      delete(tmp)
;      delete(polyPDFhiatusCMIP5)
      polyPDFhiatusCMIP6 = pdf_polygon(PDFhiatusCMIP6, do_histogram)
      tmp = tail_PDF(polyPDFhiatusCMIP6(0, :), polyPDFhiatusCMIP6(1, :), thresGMSTtrendCMIP6)
      resp = Res_drawPDF(color_cmip6_selected, periphery_thickness, 0, pdf_opacity)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
                                      tmp(0, :), tmp(1, :), resp)
      idummy = idummy + 1
;      delete(tmp)
;      delete(polyPDFhiatusCMIP6)
    end if
    delete(polyPDFhiatus)
    txres@txFontColor := color_cmip    
    gsn_text_ndc(wks, "cmip: "+num(.not.ismissing(GMSTtrendCMIPHiatus)), xlabel, ylabel-ilabel*0.015, txres)
    ilabel = ilabel + 1
  else   ; cmip5 and cmip6 individually
    if (isvar("PDFhiatusCMIP5")) then
      ; CMIP5 PDF fill
      resp = Res_drawPDF(color_cmip5, periphery_thickness, 0, 0.);pdf_opacity) ; outline only
;      dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
;                                      PDFhiatusCMIP5@bin_center, PDFhiatusCMIP5, resp)
;      tmp = modify_tail(PDFhiatusCMIP5)
      polyPDFhiatusCMIP5 = pdf_polygon(PDFhiatusCMIP5, do_histogram)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
                                      polyPDFhiatusCMIP5(0, :), polyPDFhiatusCMIP5(1, :), resp)
      idummy = idummy + 1
;      delete(tmp)
      ; CMIP5 PDF mean
      tmp = addline_PDF(polyPDFhiatusCMIP5(0, :), polyPDFhiatusCMIP5(1, :), PDFhiatusCMIP5@mean)
      resp = Res_drawPDF(color_cmip5_mean, periphery_thickness, 0, 1.)
      dummy(idummy) = gsn_add_polyline(wks, pdfs(0), \
                                       tmp(0, :), tmp(1, :), resp)
      delete(tmp)
      idummy = idummy + 1
      ; CMIP5 PDF periphery
      resp = Res_drawPDF(color_cmip5, periphery_thickness, 0, 1.)
      dummy(idummy) = gsn_add_polyline(wks, pdfs(0), \
                                       polyPDFhiatusCMIP5(0, :), polyPDFhiatusCMIP5(1, :), resp)
      idummy = idummy + 1
      if (nthres.eq.2.or..not.isvar("PDFhiatusCMIP6")) then
        tmp = tail_PDF(polyPDFhiatusCMIP5(0, :), polyPDFhiatusCMIP5(1, :), thresGMSTtrendCMIP5)
        resp = Res_drawPDF(color_cmip5_selected, periphery_thickness, 0, pdf_opacity)
        dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
                                        tmp(0, :), tmp(1, :), resp)
        idummy = idummy + 1
        delete(tmp)
      end if
      txres@txFontColor := color_cmip5
      gsn_text_ndc(wks, "cmip5: "+num(.not.ismissing(GMSTtrendCMIP5Hiatus)), xlabel, ylabel-ilabel*0.015, txres)
      ilabel = ilabel + 1
    end if
    if (isvar("PDFhiatusCMIP6")) then
      ; CMIP6 PDF fill
      resp = Res_drawPDF(color_cmip6, periphery_thickness, 0, pdf_opacity)
;      dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
;                                      PDFhiatusCMIP6@bin_center, PDFhiatusCMIP6, resp)
;      tmp = modify_tail(PDFhiatusCMIP6)
      polyPDFhiatusCMIP6 = pdf_polygon(PDFhiatusCMIP6, do_histogram)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
                                      polyPDFhiatusCMIP6(0, :), polyPDFhiatusCMIP6(1, :), resp)
      idummy = idummy + 1
;      delete(tmp)
      ; CMIP6 PDF mean
      tmp = addline_PDF(polyPDFhiatusCMIP6(0, :), polyPDFhiatusCMIP6(1, :), PDFhiatusCMIP6@mean)
      resp = Res_drawPDF(color_cmip6_mean, periphery_thickness, 0, 1.)
      dummy(idummy) = gsn_add_polyline(wks, pdfs(0), \
                                       tmp(0, :), tmp(1, :), resp)
      delete(tmp)
      idummy = idummy + 1
      ; CMIP6 PDF periphery
      resp = Res_drawPDF(color_cmip6, periphery_thickness, 0, 1.)
      dummy(idummy) = gsn_add_polyline(wks, pdfs(0), \
                                       polyPDFhiatusCMIP6(0, :), polyPDFhiatusCMIP6(1, :), resp)
      idummy = idummy + 1
      if (nthres.eq.2.or..not.isvar("PDFhiatusCMIP5")) then
        tmp = tail_PDF(polyPDFhiatusCMIP6(0, :), polyPDFhiatusCMIP6(1, :), thresGMSTtrendCMIP6)
        resp = Res_drawPDF(color_cmip6_selected, periphery_thickness, 0, pdf_opacity)
        dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
                                        tmp(0, :), tmp(1, :), resp)
        idummy = idummy + 1
        delete(tmp)
      end if
      txres@txFontColor := color_cmip6
      gsn_text_ndc(wks, "cmip6: "+num(.not.ismissing(GMSTtrendCMIP6Hiatus)), xlabel, ylabel-ilabel*0.015, txres)
      ilabel = ilabel + 1
    end if
    if (nthres.eq.1.and.isvar("PDFhiatusCMIP5").and.isvar("PDFhiatusCMIP6")) then
      xx = polyPDFhiatusCMIP6(0, :)  ;; assume that cmip5 and cmip6 share common bins
      yy = (/ polyPDFhiatusCMIP6(1, :) /)
      yy = where(polyPDFhiatusCMIP5(1, :).gt.polyPDFhiatusCMIP6(1, :), polyPDFhiatusCMIP5(1, :), polyPDFhiatusCMIP6(1, :))
      tmp = tail_PDF(xx, yy, thresGMSTtrend)
      resp = Res_drawPDF(color_cmip_selected, periphery_thickness, 0, pdf_opacity)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
                                      tmp(0, :), tmp(1, :), resp)
      delete(tmp)
      idummy = idummy + 1
      delete(xx)
      delete(yy)
    end if
  end if
  if (isdefined("PDFhiatusObs")) then
    j = 0
    do ii = 0, ListCount(PDFhiatusObs)-1
      polyPDFhiatusObs = pdf_polygon(PDFhiatusObs[ii], do_histogram)
      resp = Res_drawPDF(color_obs_ens(j), periphery_thickness, 3, 1.)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
                                      polyPDFhiatusObs(0, :), polyPDFhiatusObs(1, :), resp)
      idummy = idummy + 1
      dummy(idummy) = gsn_add_polyline(wks, pdfs(0), \
                                       polyPDFhiatusObs(0, :), polyPDFhiatusObs(1, :), resp)
      idummy = idummy + 1
      delete(polyPDFhiatusObs)
      txres@txFontColor := color_obs_ens(j)
      gsn_text_ndc(wks, PDFhiatusObs[ii]@dataset, xlabel, ylabel-ilabel*0.015, txres)
      ilabel = ilabel + 1
      j = j + 1
    end do
  end if
  if (isdefined("SingleValhiatusObs")) then
    j = 0
    do ii = 0, ListCount(SingleValhiatusObs)-1
      resp = Res_drawPDF(color_obs_single(j), line_thickness, 0, 1.)
      dummy(idummy) = gsn_add_polyline(wks, pdfs(0), \
                                      (/SingleValhiatusObs[ii], SingleValhiatusObs[ii]/), \
                                      (/res@trYMinF, res@trYMaxF/), resp)
      idummy = idummy + 1
      txres@txFontColor := color_obs_single(j)
      gsn_text_ndc(wks, SingleValhiatusObs[ii]@dataset, xlabel, ylabel-ilabel*0.015, txres)
      ilabel = ilabel + 1
      j = j + 1
    end do
  end if

; 3.2 Plot post-hiatus PDF

  res@tiXAxisString = long_name+" ["+unit+"]"
  res@tiXAxisFontHeightF = 0.015
  res@vpYF = 0.44
  res@tiMainString = "(b) "+syear_post+"-"+eyear_post+" GMST trend"
  pdfs(1) = gsn_csm_xy(wks, (/-1., 1./), (/ 0., 0./), res)
  ilabel = 0
  ylabel = 0.43
  if (isvar("PDFpost")) then  ; combined cmip5+cmip6
    ; CMIP PDF fill
    resp = Res_drawPDF(color_cmip, periphery_thickness, 0, pdf_opacity)
;    dummy(idummy) = gsn_add_polygon(wks, pdfs(1), \
;                                    PDFpost@bin_center, PDFpost, resp)
;    tmp = modify_tail(PDFpost)
    polyPDFpost = pdf_polygon(PDFpost, do_histogram)
    dummy(idummy) = gsn_add_polygon(wks, pdfs(1), \
                                    polyPDFpost(0, :), polyPDFpost(1, :), resp)
    idummy = idummy + 1
;    delete(tmp)
    ; CMIP PDF mean
    tmp = addline_PDF(polyPDFpost(0, :), polyPDFpost(1, :), PDFpost@mean)
    resp = Res_drawPDF(color_cmip_mean, periphery_thickness, 0, 1.)
    dummy(idummy) = gsn_add_polyline(wks, pdfs(1), \
                                     tmp(0, :), tmp(1, :), resp)
    delete(tmp)
    idummy = idummy + 1
    txres@txFontColor := color_cmip    
    gsn_text_ndc(wks, "cmip: "+num(.not.ismissing(GMSTtrendCMIPPost)), xlabel, ylabel-ilabel*0.015, txres)
    ilabel = ilabel + 1
  else   ; cmip5 and cmip6 individually
    if (isvar("PDFpostCMIP5")) then
      ; CMIP5 PDF fill
      resp = Res_drawPDF(color_cmip5, periphery_thickness, 0, 0.); pdf_opacity)
;      dummy(idummy) = gsn_add_polygon(wks, pdfs(1), \
;                                      PDFpostCMIP5@bin_center, PDFpostCMIP5, resp)
;      tmp = modify_tail(PDFpostCMIP5)
      polyPDFpostCMIP5 = pdf_polygon(PDFpostCMIP5, do_histogram)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(1), \
                                      polyPDFpostCMIP5(0, :), polyPDFpostCMIP5(1, :), resp)
      idummy = idummy + 1
;      delete(tmp)
      ; CMIP5 PDF mean
      tmp = addline_PDF(polyPDFpostCMIP5(0, :), polyPDFpostCMIP5(1, :), PDFpostCMIP5@mean)
      resp = Res_drawPDF(color_cmip5_mean, periphery_thickness, 0, 1.)
      dummy(idummy) = gsn_add_polyline(wks, pdfs(1), \
                                       tmp(0, :), tmp(1, :), resp)
      delete(tmp)
      idummy = idummy + 1
      txres@txFontColor := color_cmip5
      gsn_text_ndc(wks, "cmip5: "+num(.not.ismissing(GMSTtrendCMIP5Post)), xlabel, ylabel-ilabel*0.015, txres)
      ilabel = ilabel + 1
    end if
    if (isvar("PDFpostCMIP6")) then
      ; CMIP6 PDF fill
      resp = Res_drawPDF(color_cmip6, periphery_thickness, 0, pdf_opacity)
;      dummy(idummy) = gsn_add_polygon(wks, pdfs(1), \
;                                      PDFpostCMIP6@bin_center, PDFpostCMIP6, resp)
;      tmp = modify_tail(PDFpostCMIP6)
      polyPDFpostCMIP6 = pdf_polygon(PDFpostCMIP6, do_histogram)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(1), \
                                      polyPDFpostCMIP6(0, :), polyPDFpostCMIP6(1, :), resp)
      idummy = idummy + 1
;      delete(tmp)
      ; CMIP6 PDF mean
      tmp = addline_PDF(polyPDFpostCMIP6(0, :), polyPDFpostCMIP6(1, :), PDFpostCMIP6@mean)
      resp = Res_drawPDF(color_cmip6_mean, periphery_thickness, 0, 1.)
      dummy(idummy) = gsn_add_polyline(wks, pdfs(1), \
                                       tmp(0, :), tmp(1, :), resp)
      delete(tmp)
      idummy = idummy + 1
      txres@txFontColor := color_cmip6
      gsn_text_ndc(wks, "cmip6: "+num(.not.ismissing(GMSTtrendCMIP6Post)), xlabel, ylabel-ilabel*0.015, txres)
      ilabel = ilabel + 1
    end if
  end if
  if (isvar("GMSTtrendCMIPPost")) then
    do ii = 0, dimsizes(GMSTtrendCMIPPost)-1
      if (ismissing(GMSTtrendCMIPPost(ii))) then
        continue
      end if
      if (GMSTtrendCMIPPost@do_composite(ii).and.isvar("PDFpost")) then
        tmp = addline_PDF(polyPDFpost(0, :), polyPDFpost(1, :), GMSTtrendCMIPPost(ii))
        resp = Res_drawPDF(color_cmip_selected, line_thickness, 0, 1.)
        dummy(idummy) = gsn_add_polyline(wks, pdfs(1), \
                                         tmp(0, :), tmp(1, :), resp)
        idummy = idummy + 1
        delete(tmp)
      end if
    end do
  else
    if (isvar("GMSTtrendCMIP5Post")) then
      do ii = 0, dimsizes(GMSTtrendCMIP5Post)-1
        if (ismissing(GMSTtrendCMIP5Post(ii))) then
          continue
        end if
        if (GMSTtrendCMIP5Post@do_composite(ii).and.isvar("PDFpostCMIP5")) then
          tmp = addline_PDF(polyPDFpostCMIP5(0, :), polyPDFpostCMIP5(1, :), GMSTtrendCMIP5Post(ii))
          resp = Res_drawPDF(color_cmip5_selected, line_thickness, 0, 1.)
          dummy(idummy) = gsn_add_polyline(wks, pdfs(1), \
                                           tmp(0, :), tmp(1, :), resp)
          idummy = idummy + 1
          delete(tmp)
        end if
      end do
    end if
    if (isvar("GMSTtrendCMIP6Post")) then
      do ii = 0, dimsizes(GMSTtrendCMIP6Post)-1
        if (ismissing(GMSTtrendCMIP6Post(ii))) then
          continue
        end if
        if (GMSTtrendCMIP6Post@do_composite(ii).and.isvar("PDFpostCMIP6")) then
          tmp = addline_PDF(polyPDFpostCMIP6(0, :), polyPDFpostCMIP6(1, :), GMSTtrendCMIP6Post(ii))
          resp = Res_drawPDF(color_cmip6_selected, line_thickness, 0, 1.)
          dummy(idummy) = gsn_add_polyline(wks, pdfs(1), \
                                           tmp(0, :), tmp(1, :), resp)
          idummy = idummy + 1
          delete(tmp)
        end if
      end do
    end if
  end if
  if (isvar("PDFpost")) then  ; combined cmip5+cmip6
    resp = Res_drawPDF(color_cmip, periphery_thickness, 0, 1.)
    dummy(idummy) = gsn_add_polyline(wks, pdfs(1), \
                                     polyPDFpost(0, :), polyPDFpost(1, :), resp)
    idummy = idummy + 1
  else   ; cmip5 and cmip6 individually
    if (isvar("PDFpostCMIP5")) then
      resp = Res_drawPDF(color_cmip5, periphery_thickness, 0, 1.)
      dummy(idummy) = gsn_add_polyline(wks, pdfs(1), \
                                       polyPDFpostCMIP5(0, :), polyPDFpostCMIP5(1, :), resp)
      idummy = idummy + 1
    end if
    if (isvar("PDFpostCMIP6")) then
      resp = Res_drawPDF(color_cmip6, periphery_thickness, 0, 1.)
      dummy(idummy) = gsn_add_polyline(wks, pdfs(1), \
                                       polyPDFpostCMIP6(0, :), polyPDFpostCMIP6(1, :), resp)
      idummy = idummy + 1
    end if
  end if

; ======================================================================
;
;                          4. Read patterned trends
;
; ======================================================================

  path = input_dir+"trendPattern_hiatus_composite.nc"
  if (fileexists(path)) then
    f = addfile(path, "r")
    TStrendCMIPHiatus = f->trend
    TStrendCMIPHiatus = TStrendCMIPHiatus * scale
  end if
  path = input_dir+"trendPattern_post_composite.nc"
  if (fileexists(path)) then
    f = addfile(path, "r")
    TStrendCMIPPost = f->trend
    TStrendCMIPPost = TStrendCMIPPost * scale
  end if
  path = input_dir+"trendPattern_hiatus_obs.nc"
  if (fileexists(path)) then
    f = addfile(path, "r")
    TStrendObsHiatus = f->trend
    TStrendObsHiatus = TStrendObsHiatus * scale
  end if

; -------------------------------------
; Plot
  patterns = new(3, "graphic")
  resTs = True
  resTs@res_gsnDraw = False
  resTs@res_gsnFrame = False
  resTs@res_gsnMaximize = False
  resTs@res_gsnAddCyclic = True
  resTs@res_cnFillOn = True
  resTs@res_cnFillMode = "AreaFill"
  resTs@res_cnLinesOn = False
  resTs@res_cnLineLabelsOn = False
  resTs@res_cnInfoLabelOn = False
  resTs@res_mpProjection = "Robinson"
  resTs@res_mpMinLonF = 30.
  resTs@res_mpMaxLonF = 390.
  resTs@res_mpCenterLonF = 210.
  resTs@res_mpGeophysicalLineColor := "grey30"
  resTs@res_mpGridAndLimbOn = True
  resTs@res_mpGridLineColor := "transparent"
;  resTs@res_mpGridLatSpacingF = 90.
;  resTs@res_mpGridLonSpacingF = 720.
  resTs@res_cnLevelSelectionMode = "ExplicitLevels"
  resTs@res_lbLabelBarOn = False
  resTs@res_lbLabelFontHeightF = 0.016
  resTs@res_pmLabelBarOrthogonalPosF = 0.06
  resTs@res_lbTitleOn = True
  resTs@res_lbTitleString = "Near-surface temperature trend ["+unit+"]"
  resTs@res_lbTitlePosition = "Bottom"
  resTs@res_lbTitleFontHeightF = 0.016
  resTs@res_lbTitleOffsetF = 0.1
  resTs@res_tiMainFontHeightF = 0.017
  resTs@res_tiMainOffsetYF = -0.012
;  resTs@res_tmXTOn = False
;  resTs@res_tmYROn = False
 ; resTs@res_tmXBLabelFontHeightF = 0.008
 ; resTs@res_tmYLLabelFontHeightF = 0.008
;  resTs@res_gsnLeftString = ""
;  resTs@res_gsnRightString = ""
  resTs@res_mpPerimLineThicknessF = 0.
  resTs@res_mpLimbLineThicknessF = 0.
  resTs@res_mpPerimOn = False
  resTs@res_tmBorderThicknessF = 1.
  resTs@res_tmXBMajorThicknessF =  resTs@res_tmBorderThicknessF
  resTs@res_tmYLMajorThicknessF =  resTs@res_tmBorderThicknessF
  resTs@res_tmXBMinorThicknessF =  resTs@res_tmBorderThicknessF/2.
;  resTs@res_cnLevels = (/ -1., -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1. /)
;  resTs@res_cnFillPalette = "ncl_default"
  resTs@res_cnLevels = (/ -1.3, -0.9, -0.6, -0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.9, 1.3 /)
  resTs@res_cnFillPalette = "temp_diff_18lev"

  txres@txFontColor := "black"
  
  obs_data = TStrendObsHiatus@dataset

  resTs@res_tiMainString = "(c) "+syear_hiatus+"-"+eyear_hiatus+" "+obs_data+" trend"
  resTs@res_vpXF = 0.52
  resTs@res_vpYF = 0.91
  resTs@res_vpHeightF = 0.23
  resTs@res_vpWidthF = 0.46
  copy_VarAtts(resTs, TStrendObsHiatus)
  patterns( 0 ) = contour_map(wks, TStrendObsHiatus, "")

  resTs@res_tiMainString = "(d) "+syear_hiatus+"-"+eyear_hiatus+" trend composite"
  resTs@res_vpYF = 0.63
  copy_VarAtts(resTs, TStrendCMIPHiatus)
  patterns( 1 ) = contour_map(wks, TStrendCMIPHiatus, "")
  gsn_text_ndc(wks, "cmip5: "+TStrendCMIPHiatus@ncomposite_cmip5, 0.98, 0.63, txres)
  gsn_text_ndc(wks, "cmip6: "+TStrendCMIPHiatus@ncomposite_cmip6, 0.98, 0.615, txres)

  resTs@res_tiMainString = "(e) "+syear_post+"-"+eyear_post+" trend composite"
  resTs@res_vpYF = 0.35
  resTs@res_lbLabelBarOn = True
  copy_VarAtts(resTs, TStrendCMIPPost)
  patterns( 2 ) = contour_map(wks, TStrendCMIPPost, "")
  gsn_text_ndc(wks, "cmip5: "+TStrendCMIPPost@ncomposite_cmip5, 0.98, 0.35, txres)
  gsn_text_ndc(wks, "cmip6: "+TStrendCMIPPost@ncomposite_cmip6, 0.98, 0.335, txres)

  draw(pdfs)
  draw(patterns)
  frame(wks)

end
