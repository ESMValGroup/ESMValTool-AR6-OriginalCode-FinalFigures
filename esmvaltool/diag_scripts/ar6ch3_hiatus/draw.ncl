; draw
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs:
;
; History
;    20210312 kosaka_yu: fixed color palette setting
;    20210227 kosaka_yu: added netcdf output and provenance.
;    20210215 kosaka_yu: revised figure details
;    20210204 kosaka_yu: updated for the new recipe structure
;    20201109 kosaka_yu: added Kadow and NOAAGlobalTemp-Interim.
;    20200924 kosaka_yu: revised figure details.
;    20200917 kosaka_yu: revised figure details.
;    20200910 kosaka_yu: revised.
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

load "$diag_scripts/ar6ch3_hiatus/io_globalmean.ncl"

function weighted_mean(x, w)
local x, w, y
begin
  y = dim_sum_n(x*conform(x, w, 0), 0)
  y = y / dim_sum_n(where(ismissing(x), w@_FillValue, conform(x, w, 0)), 0)
  return(y)
end

function weighted_percentile(x, w, t)
local x, w, xx, ww, wsum, j, k, dims, idx
begin
  xx = x
  dims = dimsizes(xx)
  if (dimsizes(dims).eq.1) then
    idx = dim_pqsort(xx, 2)
    wsum = sum(where(ismissing(x), w@_FillValue, w))
    ww = 0.
    do j = 0, dims(0)-1
      if (.not.ismissing(xx(j))) then
        ww = ww + w(idx(j))
        if (ww/wsum.eq.t) then
          y = xx(j)
          break
        elseif (ww/wsum.gt.t) then
          if (j.eq.0) then
            y = xx(j)
          else
            y = (xx(j) + xx(j-1))/2.
          end if
          break
        end if
      end if
    end do
  else
    y = new(dims(1), typeof(x))
    idx = dim_pqsort_n(xx, 2, 0)
    do k = 0, dims(1)-1
      wsum = sum(where(ismissing(x(:, k)), w@_FillValue, w))
      ww = 0.
      do j = 0, dims(0)-1
        if (.not.ismissing(xx(j, k))) then
          ww = ww + w(idx(j, k))
          if (ww/wsum.eq.t) then
            y(k) = xx(j, k)
            break
          elseif (ww/wsum.gt.t) then
            y(k) = (xx(j, k) + xx(j-1, k))/2.
            break
          end if
        end if
      end do
    end do
  end if
  return(y)
end

function weighted_pdfx(x, w, res)
local x, w, res, nbins, pdf, bounds, i, n
begin
  nbins = dimsizes(res@bin_center)
  pdf = new(nbins, "double")
  bounds = res@bin_bounds
  pdf@nbins = nbins
  pdf@bin_spacing = res@bin_spacing
  pdf@bin_bound_min = res@bin_min
  pdf@bin_bound_max = res@bin_max
  pdf@bin_center = res@bin_center
  pdf@bin_bounds = bounds
  pdf = 0.
  do i = 0, dimsizes(x)-1
    if (ismissing(x(i))) then
      continue
    end if
    do n = 0, nbins-1
      if (bounds(n).le.x(i).and.x(i).lt.bounds(n+1)) then
        pdf(n) = pdf(n) + w(i)
        break
      end if
    end do
  end do
  pdf = pdf / sum(where(ismissing(x), 0., w)) * 100.
  return(pdf)
end
  
  
function addline_PDF(x, y, x0)
local z, ix
begin
  z = new((/2, 2/), typeof(x))
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
  z(1, 1)  = ((x(ix+1) - x0) * y(ix) + (x0 - x(ix)) * y(ix+1)) / (x(ix+1) - x(ix))
  z(1, 0) = 0.
  z(0, 1) = x0
  z(0, 0) = x0
  return(z)
end

function pdf_polygon(pdf)
local polypdf, i, imax
begin
  imax = dimsizes(pdf)
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
  return(polypdf)
end

function tail_PDF(x, y, thres)
local z, ix, z0
begin
  z = new((/2, dimsizes(x)/), typeof(x))
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

begin

  enter_msg(DIAG_SCRIPT, "")

; ======================================================================
;
;                           0. Configuration
;
; ======================================================================

  load "$diag_scripts/ar6ch3_hiatus/config_datasets.ncl"
  
  scale = 10.  ; trend in "per [scale] years" 
  long_name = "GMST trend"
  unit = "~S~o~N~C (10 years)~S~-1~N~"
  input_dir = diag_script_info@input_files + "/"

  obs_dataset_pattern = "HadCRUT5_mean"
  if (isatt(diag_script_info, "obs_dataset_pattern")) then
    obs_dataset_pattern = diag_script_info@obs_dataset_pattern
  end if
  if (any(obs_dataset_pattern.eq.(/"HadCRUT5", "HadCRUT5mean", "HadCRUT5 mean"/))) then
    obs_dataset_pattern = "HadCRUT5_mean"
  elseif (any(obs_dataset_pattern.eq.(/"CowtanWay", "CowtanWay median"/))) then
    obs_dataset_pattern = "CowtanWay_median"
  end if
    
;  individual_cmip = True
;  if (isatt(diag_script_info, "pdf_individual_cmip")) then  ;
;    individual_cmip = diag_script_info@pdf_individual_cmip
;  end if
  
;  pdf_style = "smooth"
;  if (isatt(diag_script_info, "pdf_style")) then  ; pdf
;    pdf_style = diag_script_info@pdf_style
;  end if
;  if (pdf_style.eq."smooth") then
;    do_kde_n_test = True
;  else
;    do_kde_n_test = False
;  end if
;  if (pdf_style.eq."histogram") then
;    do_histogram = True
;  else
;    do_histogram = False
;  end if
  
  color_cmip = "seagreen"
  color_cmip6 = (/204., 35., 35./)/255. 
  color_cmip5 = "grey80"
  color_cmip_mean = color_cmip
  color_cmip6_mean = color_cmip6
  color_cmip5_mean = color_cmip5
  color_cmip_selected = "orange4"
  color_cmip6_selected = "darkorchid4"
  color_cmip5_selected = "orange4"
  color_cmip5_legend = "grey50"
  color_cmip6_legend = color_cmip6
  color_obs_ens = (/ "grey40", "black"/)
  color_obs_single = tofloat((/(/48, 79, 191/), (/36, 147, 126/), \
                               (/127, 68, 170/), (/237, 128, 55/), (/54, 156, 232/)/))/255.
  opacity_pdf_cmip5 = 0.7
  opacity_pdf_cmip6 = 0.15 ;0.2
  opacity_selected_pdf = 0.25
  opacity_selected_line = 0.25
  
  periphery_thickness = 1.
  line_thickness = 1.5

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
      if (isdefined("paths_all")) then
        tmp = paths_all
        delete(paths_all)
        paths_all = array_append_record(tmp, files(ii), 0)
        delete(tmp)
      else
        paths_all = files(ii)
      end if
      f = addfile(files(ii), "r")
      TstrendCMIPHiatus = f->trend
      TstrendCMIPHiatus = TstrendCMIPHiatus * scale
      TstrendCMIPHiatus@units = str_sub_str(TstrendCMIPHiatus, "year", toint(scale)+" years")
    elseif (isStrSubset(files(ii), "PostTrendPattern_composite.nc")) then
      if (isdefined("paths_all")) then
        tmp = paths_all
        delete(paths_all)
        paths_all = array_append_record(tmp, files(ii), 0)
        delete(tmp)
      else
        paths_all = files(ii)
      end if
      f = addfile(files(ii), "r")
      TstrendCMIPPost = f->trend
      TstrendCMIPPost = TstrendCMIPPost * scale
      TstrendCMIPPost@units = str_sub_str(TstrendCMIPPost, "year", toint(scale)+" years")
    end if
  end do

  if (TstrendCMIPHiatus@blend_GST.eq.1) then
    GST = "GMST"
    blending = "blended"
  else
    GST = "GSAT"
    blending = "unblended"
  end if
  if (TstrendCMIPHiatus@mask_GST.eq.1) then
    masking = "masked"
  else
    masking = "unmasked"
  end if
  composite_threshold = TstrendCMIPHiatus@composite_threshold
  threshold_GSTtrend_composite = TstrendCMIPHiatus@threshold_trendGSThiatus * scale
  composite_project = TstrendCMIPHiatus@project
  if (isatt(TstrendCMIPHiatus, "nmembers_CMIP6")) then
    nmembers_composite_cmip6 = TstrendCMIPHiatus@nmembers_CMIP6
  end if
  if (isatt(TstrendCMIPHiatus, "nmembers_CMIP5")) then
    nmembers_composite_cmip5 = TstrendCMIPHiatus@nmembers_CMIP5
  end if
  composite_member_names = str_left_strip(str_split(TstrendCMIPHiatus@dataset, ","))


  ; Read GMST
  do ii = 0, dimsizes(files)-1
    filename = str_get_field(files(ii), str_fields_count(files(ii), "/"), "/")
    if (filename.eq. "trendGMST_"+masking+"_CMIP5.txt") then
      if (isvar("ifileCMIP5")) then
        ifileCMIP5 := array_append_record(ifileCMIP5, files(ii), 0)
      else
        ifileCMIP5 = files(ii)
      end if
    elseif (filename.eq. "trendGMST_"+masking+"_CMIP6.txt") then
      if (isvar("ifileCMIP6")) then
        ifileCMIP6 := array_append_record(ifileCMIP6, files(ii), 0)
      else
        ifileCMIP6 = files(ii)
      end if
    end if
    delete(filename)
  end do
  if (isdefined("paths_all")) then
    tmp = paths_all
    delete(paths_all)
    paths_all = array_append_record(tmp, array_append_record(ifileCMIP5, ifileCMIP6, 0), 0)
    delete(tmp)
  else
    paths_all = array_append_record(ifileCMIP5, ifileCMIP6, 0)
  end if
  GMSTtrendCMIP5 = read_GMtrend_as_list(ifileCMIP5)
  GMSTtrendCMIP6 = read_GMtrend_as_list(ifileCMIP6)
  delete([/ifileCMIP5, ifileCMIP6/])

  ; Read GSAT
  do ii = 0, dimsizes(files)-1

    filename = str_get_field(files(ii), str_fields_count(files(ii), "/"), "/")
    if (filename.eq. "trendGSAT_"+masking+"_CMIP5.txt") then
      if (isvar("ifileCMIP5")) then
        ifileCMIP5 := array_append_record(ifileCMIP5, files(ii), 0)
      else
        ifileCMIP5 = files(ii)
      end if
    elseif (filename.eq. "trendGSAT_"+masking+"_CMIP6.txt") then
      if (isvar("ifileCMIP6")) then
        ifileCMIP6 := array_append_record(ifileCMIP6, files(ii), 0)
      else
        ifileCMIP6 = files(ii)
      end if
    end if
    delete(filename)
  end do
  if (isdefined("paths_all")) then
    tmp = paths_all
    delete(paths_all)
    paths_all = array_append_record(tmp, array_append_record(ifileCMIP5, ifileCMIP6, 0), 0)
    delete(tmp)
  else
    paths_all = array_append_record(ifileCMIP5, ifileCMIP6, 0)
  end if
  GSATtrendCMIP5 = read_GMtrend_as_list(ifileCMIP5)
  GSATtrendCMIP6 = read_GMtrend_as_list(ifileCMIP6)
  delete([/ifileCMIP5, ifileCMIP6/])

  if (GST.eq."GMST") then
    cmip5_member_names = "CMIP5 "+metadata_att_as_array(GMSTtrendCMIP5, "dataset") \
                         +" "+metadata_att_as_array(GMSTtrendCMIP5, "ensemble")
    cmip6_member_names = "CMIP6 "+metadata_att_as_array(GMSTtrendCMIP6, "dataset") \
                         +" "+metadata_att_as_array(GMSTtrendCMIP6, "ensemble")
  elseif (GST.eq."GSAT") then
    cmip5_member_names = "CMIP5 "+metadata_att_as_array(GSATtrendCMIP5, "dataset") \
                         +" "+metadata_att_as_array(GSATtrendCMIP5, "ensemble")
    cmip6_member_names = "CMIP6 "+metadata_att_as_array(GSATtrendCMIP6, "dataset") \
                         +" "+metadata_att_as_array(GSATtrendCMIP6, "ensemble")
  end if

  ; Read obs
  do ii = 0, dimsizes(files)-1
    filename = str_get_field(files(ii), str_fields_count(files(ii), "/"), "/")
    if ((/isStrSubset(filename, "trendGMST_").or.isStrSubset(filename, "trendGSAT_")/) \
        .and.isStrSubset(filename, "_OBS.txt")) then
      if (isdefined("paths_all")) then
        tmp = paths_all
        delete(paths_all)
        paths_all = array_append_record(tmp, files(ii), 0)
        delete(tmp)
      else
        paths_all = files(ii)
      end if
      tmp = read_GMtrend_as_list(files(ii))
      if (any(tmp[0]@dataset.eq.reanalyses)) then
        if (.not.isStrSubset(filename, "trend"+GST).or. \
            .not.isStrSubset(filename, "_"+masking+"_")) then
          delete(tmp)
          continue
        end if
      end if
      if (.not.isdefined("GSTtrendObs")) then
        GSTtrendObs = NewList("fifo")
      end if
      if (ListCount(tmp).gt.1) then
        ensemble = metadata_att_as_array(tmp, "ensemble")
        x = list_to_array(tmp)
        y = x(ind(ensemble.ne."mean".and.ensemble.ne."median"))
        atts = getvaratts(tmp[0])
        do j = 0, dimsizes(atts)-1
          y@$atts(j)$ = tmp[0]@$atts(j)$
        end do
        if (any(ensemble.eq."mean")) then
          y@mean = x(ind(ensemble.eq."mean"))
        end if
        if (any(ensemble.eq."median")) then
          y@median = x(ind(ensemble.eq."median"))
        end if
        delete(x)
        delete(ensemble)
        delete(atts)
      else
        y = tmp[0]
        y@dataset = tmp[0]@dataset
      end if
      if (isStrSubset(filename, "trendGMST_")) then
        y@GST = "GMST"
      elseif (isStrSubset(filename, "trendGSAT_")) then
        y@GST = "GSAT"
      end if
      ListAppend(GSTtrendObs, new(dimsizes(y), typeof(y)))
      i = ListCount(GSTtrendObs)-1
      GSTtrendObs[i] = y
      delete(y)
      delete(tmp)
    end if
    delete(filename)
  end do

; ======================================================================
;
;                       2. Make PDFs of GST trends
;
; ======================================================================

  nbinsobs = 40
  resPDFobs = True
  resPDFobs@bin_min = tofloat(0.00 * scale)
  resPDFobs@bin_max = tofloat(0.02 * scale)
  if (isdefined("GSTtrendObs")) then
    do ii = 0, ListCount(GSTtrendObs)-1
      GSTtrendObs[ii] = GSTtrendObs[ii] * scale
      GSTtrendObs[ii]@units = "K per "+toint(scale)+" years"
      if (dimsizes(GSTtrendObs[ii]).gt.1) then
        if (.not.isdefined("PDFhiatusObs")) then
          PDFhiatusObs = NewList("fifo")
        end if
        ListAppend(PDFhiatusObs, new(nbinsobs, "double"))
        i = ListCount(PDFhiatusObs)-1
        PDFhiatusObs[i] = pdfx(GSTtrendObs[ii], nbinsobs, resPDFobs)
        PDFhiatusObs[i] = PDFhiatusObs[i] / 100.
        PDFhiatusObs[i] = PDFhiatusObs[i] / PDFhiatusObs[i]@bin_spacing
        PDFhiatusObs[i]@dataset = GSTtrendObs[ii]@dataset
        PDFhiatusObs[i]@GST = GSTtrendObs[ii]@GST
      else
        if (.not.isdefined("SingleValhiatusObs")) then
          SingleValhiatusObs = NewList("fifo")
        end if
        ListAppend(SingleValhiatusObs, new(1, "double"))
        i = ListCount(SingleValhiatusObs)-1
        SingleValhiatusObs[i] = GSTtrendObs[ii]
        SingleValhiatusObs[i]@dataset = GSTtrendObs[ii]@dataset
        SingleValhiatusObs[i]@GST = GSTtrendObs[ii]@GST
      end if
    end do
  end if

  nbins = 25 ; 20
  resPDF = True
  resPDF@bin_min = -0.02 * scale
  resPDF@bin_max = 0.08 * scale
  resPDF@bin_spacing = (resPDF@bin_max - resPDF@bin_min) / tofloat(nbins)
  resPDF@bin_center = fspan(resPDF@bin_min + resPDF@bin_spacing/2., \
                            resPDF@bin_max - resPDF@bin_spacing/2., \
                            nbins)
  resPDF@bin_bounds = fspan(resPDF@bin_min, resPDF@bin_max, nbins + 1)

  print("trend (K/decade):     hiatus    post-hiatus")
  if (isvar("GMSTtrendCMIP5")) then
    GMSTtrendCMIP5Hiatus = list_to_array(GMSTtrendCMIP5)
    GMSTtrendCMIP5Hiatus = GMSTtrendCMIP5Hiatus * scale
    GMSTtrendCMIP5Hiatus@units = "K per "+toint(scale)+" years"
    model = metadata_att_as_array(GMSTtrendCMIP5, "dataset")
    model@_FillValue = ""
    model = where(ismissing(GMSTtrendCMIP5Hiatus), model@_FillValue, model)
    esize = new(dimsizes(model), "integer")
    do i = 0, dimsizes(model)-1
      if (ismissing(model(i))) then
        esize(i) = esize@_FillValue
      else
        esize(i) = num(model.eq.model(i))
      end if
    end do
    weight = 1./tofloat(esize)

    if (GST.eq."GMST") then
      PDFhiatusCMIP5 = weighted_pdfx(GMSTtrendCMIP5Hiatus, weight, resPDF)
      PDFhiatusCMIP5 = PDFhiatusCMIP5 / 100.
      PDFhiatusCMIP5 = PDFhiatusCMIP5 / PDFhiatusCMIP5@bin_spacing
      PDFhiatusCMIP5@mean = weighted_mean(GMSTtrendCMIP5Hiatus, weight)
      PDFhiatusCMIP5@nmodels = count_unique_values(model)
      PDFhiatusCMIP5@nruns = num(.not.ismissing(GMSTtrendCMIP5Hiatus))
      PDFhiatusCMIP5@GST = "GMST"
    else
      AltMeanHiatusCMIP5 = weighted_mean(GMSTtrendCMIP5Hiatus, weight)
      AltMeanHiatusCMIP5@GST = "GMST"
    end if
    delete(weight)
    delete(model)
    hiatus_period = GMSTtrendCMIP5[0]@hiatus_period
    
    GMSTtrendCMIP5Post = metadata_att_as_array(GMSTtrendCMIP5, "trend_post")
    GMSTtrendCMIP5Post = GMSTtrendCMIP5Post * scale
    GMSTtrendCMIP5Post@units = "K per "+toint(scale)+" years"
    model = metadata_att_as_array(GMSTtrendCMIP5, "dataset")
    model@_FillValue = ""
    model = where(ismissing(GMSTtrendCMIP5Post), model@_FillValue, model)
    esize = 0
    do i = 0, dimsizes(model)-1
      if (ismissing(model(i))) then
        esize(i) = esize@_FillValue
      else
        esize(i) = num(model.eq.model(i))
      end if
    end do
    weight = 1./tofloat(esize)

    if (GST.eq."GMST") then
      PDFpostCMIP5 = weighted_pdfx(GMSTtrendCMIP5Post, weight, resPDF)
      PDFpostCMIP5 = PDFpostCMIP5 / 100.
      PDFpostCMIP5 = PDFpostCMIP5 / PDFpostCMIP5@bin_spacing
      PDFpostCMIP5@mean = weighted_mean(GMSTtrendCMIP5Post, weight)
      PDFpostCMIP5@nmodels = count_unique_values(model)
      PDFpostCMIP5@nruns = num(.not.ismissing(GMSTtrendCMIP5Post))
      PDFpostCMIP5@GST = "GMST"
      if (any(composite_project.eq.(/"cmip5", "cmip5+cmip6"/))) then
        GMSTtrendCMIP5Post_selected = GMSTtrendCMIP5Post 
        do ii = 0, dimsizes(cmip5_member_names)-1
          if (all(cmip5_member_names(ii).ne.composite_member_names)) then
            GMSTtrendCMIP5Post_selected(ii) = GMSTtrendCMIP5Post_selected@_FillValue
          end if
        end do
        PDFpostCMIP5selected = weighted_pdfx(GMSTtrendCMIP5Post_selected, weight, resPDF)
        PDFpostCMIP5selected = PDFpostCMIP5selected / 100.
        PDFpostCMIP5selected = PDFpostCMIP5selected / PDFpostCMIP5selected@bin_spacing
        PDFpostCMIP5selected = PDFpostCMIP5selected \
                                * sum(where(ismissing(GMSTtrendCMIP5Post_selected), 0., weight)) \
                                / sum(where(ismissing(GMSTtrendCMIP5Post), 0., weight))
      end if
    else
      AltMeanPostCMIP5 = weighted_mean(GMSTtrendCMIP5Post, weight)
      AltMeanPostCMIP5@GST = "GMST"
    end if
    delete(weight)
    delete(esize)
    delete(model)
    
    do ii = 0, ListCount(GMSTtrendCMIP5)-1
      if (isatt(GMSTtrendCMIP5[ii], "post_period")) then
        post_period = GMSTtrendCMIP5[ii]@post_period
        break
      end if
    end do
  end if

  if (isvar("GSATtrendCMIP5")) then
    GSATtrendCMIP5Hiatus = list_to_array(GSATtrendCMIP5)
    GSATtrendCMIP5Hiatus = GSATtrendCMIP5Hiatus * scale
    GSATtrendCMIP5Hiatus@units = "K per "+toint(scale)+" years"
    model = metadata_att_as_array(GSATtrendCMIP5, "dataset")
    model@_FillValue = ""
    model = where(ismissing(GSATtrendCMIP5Hiatus), model@_FillValue, model)
    esize = new(dimsizes(model), "integer")
    do i = 0, dimsizes(model)-1
      if (ismissing(model(i))) then
        esize(i) = esize@_FillValue
      else
        esize(i) = num(model.eq.model(i))
      end if
    end do
    weight = 1./tofloat(esize)

    if (GST.eq."GSAT") then
      PDFhiatusCMIP5 = weighted_pdfx(GSATtrendCMIP5Hiatus, weight, resPDF)
      PDFhiatusCMIP5 = PDFhiatusCMIP5 / 100.
      PDFhiatusCMIP5 = PDFhiatusCMIP5 / PDFhiatusCMIP5@bin_spacing
      PDFhiatusCMIP5@mean = weighted_mean(GSATtrendCMIP5Hiatus, weight)
      PDFhiatusCMIP5@nmodels = count_unique_values(model)
      PDFhiatusCMIP5@nruns = num(.not.ismissing(GSATtrendCMIP5Hiatus))
      PDFhiatusCMIP5@GST = "GSAT"
    else
      AltMeanHiatusCMIP5 = weighted_mean(GSATtrendCMIP5Hiatus, weight)
      AltMeanHiatusCMIP5@GST = "GSAT"
    end if
    delete(weight)
    delete(model)
    if (.not.isdefined("hiatus_period")) then
      hiatus_period = GSATtrendCMIP5[0]@hiatus_period
    end if
    
    GSATtrendCMIP5Post = metadata_att_as_array(GSATtrendCMIP5, "trend_post")
    GSATtrendCMIP5Post = GSATtrendCMIP5Post * scale
    GSATtrendCMIP5Post@units = "K per "+toint(scale)+" years"
    model = metadata_att_as_array(GSATtrendCMIP5, "dataset")
    model@_FillValue = ""
    model = where(ismissing(GSATtrendCMIP5Post), model@_FillValue, model)
    esize = 0
    do i = 0, dimsizes(model)-1
      if (ismissing(model(i))) then
        esize(i) = esize@_FillValue
      else
        esize(i) = num(model.eq.model(i))
      end if
    end do
    weight = 1./tofloat(esize)

    if (GST.eq."GSAT") then
      PDFpostCMIP5 = weighted_pdfx(GSATtrendCMIP5Post, weight, resPDF)
      PDFpostCMIP5 = PDFpostCMIP5 / 100.
      PDFpostCMIP5 = PDFpostCMIP5 / PDFpostCMIP5@bin_spacing
      PDFpostCMIP5@mean = weighted_mean(GSATtrendCMIP5Post, weight)
      PDFpostCMIP5@nmodels = count_unique_values(model)
      PDFpostCMIP5@nruns = num(.not.ismissing(GSATtrendCMIP5Post))
      PDFpostCMIP5@GST = "GSAT"
      if (any(composite_project.eq.(/"cmip5", "cmip5+cmip6"/))) then
        GSATtrendCMIP5Post_selected = GSATtrendCMIP5Post 
        do ii = 0, dimsizes(cmip5_member_names)-1
          if (all(cmip5_member_names(ii).ne.composite_member_names)) then
            GSATtrendCMIP5Post_selected(ii) = GSATtrendCMIP5Post_selected@_FillValue
          end if
        end do
        PDFpostCMIP5selected = weighted_pdfx(GSATtrendCMIP5Post_selected, weight, resPDF)
        PDFpostCMIP5selected = PDFpostCMIP5selected / 100.
        PDFpostCMIP5selected = PDFpostCMIP5selected / PDFpostCMIP5selected@bin_spacing
        PDFpostCMIP5selected = PDFpostCMIP5selected \
                                * sum(where(ismissing(GSATtrendCMIP5Post_selected), 0., weight)) \
                                / sum(where(ismissing(GSATtrendCMIP5Post), 0., weight))
      end if
    else
      AltMeanPostCMIP5 = weighted_mean(GSATtrendCMIP5Post, weight)
      AltMeanPostCMIP5@GST = "GSAT"
    end if
    delete(weight)
    delete(esize)
    delete(model)

    if (.not.isdefined("post_period")) then
      do ii = 0, ListCount(GSATtrendCMIP5)-1
        if (isatt(GSATtrendCMIP5[ii], "post_period")) then
          post_period = GSATtrendCMIP5[ii]@post_period
          break
        end if
      end do
    end if
  end if

  print("CMIP5 MME mean "+PDFhiatusCMIP5@GST+": "+PDFhiatusCMIP5@mean+"   "+PDFpostCMIP5@mean)
  print("CMIP5 MME mean "+AltMeanHiatusCMIP5@GST+": "+AltMeanHiatusCMIP5+"   "+AltMeanPostCMIP5)
  
  if (isvar("GMSTtrendCMIP6")) then
    GMSTtrendCMIP6Hiatus = list_to_array(GMSTtrendCMIP6)
    GMSTtrendCMIP6Hiatus = GMSTtrendCMIP6Hiatus * scale
    GMSTtrendCMIP6Hiatus@units = "K per "+toint(scale)+" years"
    model = metadata_att_as_array(GMSTtrendCMIP6, "dataset")
    model@_FillValue = ""
    model = where(ismissing(GMSTtrendCMIP6Hiatus), model@_FillValue, model)
    esize = new(dimsizes(model), "integer")
    do i = 0, dimsizes(model)-1
      if (ismissing(model(i))) then
        esize(i) = esize@_FillValue
      else
        esize(i) = num(model.eq.model(i))
      end if
    end do
    weight = 1./tofloat(esize)
    if (GST.eq."GMST") then
      PDFhiatusCMIP6 = weighted_pdfx(GMSTtrendCMIP6Hiatus, weight, resPDF)
      PDFhiatusCMIP6 = PDFhiatusCMIP6 / 100.
      PDFhiatusCMIP6 = PDFhiatusCMIP6 / PDFhiatusCMIP6@bin_spacing
      PDFhiatusCMIP6@mean = weighted_mean(GMSTtrendCMIP6Hiatus, weight)
      PDFhiatusCMIP6@nmodels = count_unique_values(model)
      PDFhiatusCMIP6@nruns = num(.not.ismissing(GMSTtrendCMIP6Hiatus))
      PDFhiatusCMIP6@GST = "GMST"
    else
      AltMeanHiatusCMIP6 = weighted_mean(GMSTtrendCMIP6Hiatus, weight)
      AltMeanHiatusCMIP6@GST = "GMST"
    end if
    delete(weight)
    delete(model)
    if (.not.isvar("hiatus_period")) then
      hiatus_period = GMSTtrendCMIP6[0]@hiatus_period
    end if
    
    GMSTtrendCMIP6Post = metadata_att_as_array(GMSTtrendCMIP6, "trend_post")
    GMSTtrendCMIP6Post = GMSTtrendCMIP6Post * scale
    GMSTtrendCMIP6Post@units = "K per "+toint(scale)+" years"
    model = metadata_att_as_array(GMSTtrendCMIP6, "dataset")
    model@_FillValue = ""
    model = where(ismissing(GMSTtrendCMIP6Post), model@_FillValue, model)
    esize = 0
    do i = 0, dimsizes(model)-1
      if (ismissing(model(i))) then
        esize(i) = esize@_FillValue
      else
        esize(i) = num(model.eq.model(i))
      end if
    end do
    weight = 1./tofloat(esize)

    if (GST.eq."GMST") then
      PDFpostCMIP6 = weighted_pdfx(GMSTtrendCMIP6Post, weight, resPDF)
      PDFpostCMIP6 = PDFpostCMIP6 / 100.
      PDFpostCMIP6 = PDFpostCMIP6 / PDFpostCMIP6@bin_spacing
      PDFpostCMIP6@mean = weighted_mean(GMSTtrendCMIP6Post, weight)
      PDFpostCMIP6@nmodels = count_unique_values(model)
      PDFpostCMIP6@nruns = num(.not.ismissing(GMSTtrendCMIP6Post))
      PDFpostCMIP6@GST = "GMST"
      GMSTtrendCMIP6Post_composited = GMSTtrendCMIP6Post

      if (any(composite_project.eq.(/"cmip6", "cmip5+cmip6"/))) then
        GMSTtrendCMIP6Post_selected = GMSTtrendCMIP6Post 
        do ii = 0, dimsizes(cmip6_member_names)-1
          if (all(cmip6_member_names(ii).ne.composite_member_names)) then
            GMSTtrendCMIP6Post_selected(ii) = GMSTtrendCMIP6Post_selected@_FillValue
          end if
        end do
        PDFpostCMIP6selected = weighted_pdfx(GMSTtrendCMIP6Post_selected, weight, resPDF)
        PDFpostCMIP6selected = PDFpostCMIP6selected / 100.
        PDFpostCMIP6selected = PDFpostCMIP6selected / PDFpostCMIP6selected@bin_spacing
        PDFpostCMIP6selected = PDFpostCMIP6selected \
                                * sum(where(ismissing(GMSTtrendCMIP6Post_selected), 0., weight)) \
                                / sum(where(ismissing(GMSTtrendCMIP6Post), 0., weight))
      end if
    else
      AltMeanPostCMIP6 = weighted_mean(GMSTtrendCMIP6Post, weight)
      AltMeanPostCMIP6@GST = "GMST"
    end if
    delete(weight)
    delete(esize)
    delete(model)

    if (.not.isvar("post_period")) then
      do ii = 0, ListCount(GMSTtrendCMIP6)-1
        if (isatt(GMSTtrendCMIP6[ii], "post_period")) then
          post_period = GMSTtrendCMIP6[ii]@post_period
          break
        end if
      end do
    end if
  end if

  if (isvar("GSATtrendCMIP6")) then
    GSATtrendCMIP6Hiatus = list_to_array(GSATtrendCMIP6)
    GSATtrendCMIP6Hiatus = GSATtrendCMIP6Hiatus * scale
    GSATtrendCMIP6Hiatus@units = "K per "+toint(scale)+" years"
    model = metadata_att_as_array(GSATtrendCMIP6, "dataset")
    model@_FillValue = ""
    model = where(ismissing(GSATtrendCMIP6Hiatus), model@_FillValue, model)
    esize = new(dimsizes(model), "integer")
    do i = 0, dimsizes(model)-1
      if (ismissing(model(i))) then
        esize(i) = esize@_FillValue
      else
        esize(i) = num(model.eq.model(i))
      end if
    end do
    weight = 1./tofloat(esize)
    if (GST.eq."GSAT") then
      PDFhiatusCMIP6 = weighted_pdfx(GSATtrendCMIP6Hiatus, weight, resPDF)
      PDFhiatusCMIP6 = PDFhiatusCMIP6 / 100.
      PDFhiatusCMIP6 = PDFhiatusCMIP6 / PDFhiatusCMIP6@bin_spacing
      PDFhiatusCMIP6@mean = weighted_mean(GSATtrendCMIP6Hiatus, weight)
      PDFhiatusCMIP6@nmodels = count_unique_values(model)
      PDFhiatusCMIP6@nruns = num(.not.ismissing(GSATtrendCMIP6Hiatus))
      PDFhiatusCMIP6@GST = "GSAT"
    else
      AltMeanHiatusCMIP6 = weighted_mean(GSATtrendCMIP6Hiatus, weight)
      AltMeanHiatusCMIP6@GST = "GSAT"
    end if
    delete(weight)
    delete(model)
    if (.not.isvar("hiatus_period")) then
      hiatus_period = GSATtrendCMIP6[0]@hiatus_period
    end if
    
    GSATtrendCMIP6Post = metadata_att_as_array(GSATtrendCMIP6, "trend_post")
    GSATtrendCMIP6Post = GSATtrendCMIP6Post * scale
    GSATtrendCMIP6Post@units = "K per "+toint(scale)+" years"
    model = metadata_att_as_array(GSATtrendCMIP6, "dataset")
    model@_FillValue = ""
    model = where(ismissing(GSATtrendCMIP6Post), model@_FillValue, model)
    esize = 0
    do i = 0, dimsizes(model)-1
      if (ismissing(model(i))) then
        esize(i) = esize@_FillValue
      else
        esize(i) = num(model.eq.model(i))
      end if
    end do
    weight = 1./tofloat(esize)

    if (GST.eq."GSAT") then
      PDFpostCMIP6 = weighted_pdfx(GSATtrendCMIP6Post, weight, resPDF)
      PDFpostCMIP6 = PDFpostCMIP6 / 100.
      PDFpostCMIP6 = PDFpostCMIP6 / PDFpostCMIP6@bin_spacing
      PDFpostCMIP6@mean = weighted_mean(GSATtrendCMIP6Post, weight)
      PDFpostCMIP6@nmodels = count_unique_values(model)
      PDFpostCMIP6@nruns = num(.not.ismissing(GSATtrendCMIP6Post))
      PDFpostCMIP6@GST = "GSAT"
      GSATtrendCMIP6Post_composited = GSATtrendCMIP6Post

      if (any(composite_project.eq.(/"cmip6", "cmip5+cmip6"/))) then
        GSATtrendCMIP6Post_selected = GSATtrendCMIP6Post 
        do ii = 0, dimsizes(cmip6_member_names)-1
          if (all(cmip6_member_names(ii).ne.composite_member_names)) then
            GSATtrendCMIP6Post_selected(ii) = GSATtrendCMIP6Post_selected@_FillValue
          end if
        end do
        PDFpostCMIP6selected = weighted_pdfx(GSATtrendCMIP6Post_selected, weight, resPDF)
        PDFpostCMIP6selected = PDFpostCMIP6selected / 100.
        PDFpostCMIP6selected = PDFpostCMIP6selected / PDFpostCMIP6selected@bin_spacing
        PDFpostCMIP6selected = PDFpostCMIP6selected \
                                * sum(where(ismissing(GSATtrendCMIP6Post_selected), 0., weight)) \
                                / sum(where(ismissing(GSATtrendCMIP6Post), 0., weight))
      end if
    else
      AltMeanPostCMIP6 = weighted_mean(GSATtrendCMIP6Post, weight)
      AltMeanPostCMIP6@GST = "GSAT"
    end if
    delete(weight)
    delete(esize)
    delete(model)

    if (.not.isvar("post_period")) then
      do ii = 0, ListCount(GSATtrendCMIP6)-1
        if (isatt(GSATtrendCMIP6[ii], "post_period")) then
          post_period = GSATtrendCMIP6[ii]@post_period
          break
        end if
      end do
    end if
  end if

  print("CMIP6 MME mean "+PDFhiatusCMIP6@GST+": "+PDFhiatusCMIP6@mean+"   "+PDFpostCMIP6@mean)
  print("CMIP6 MME mean "+AltMeanHiatusCMIP6@GST+": "+AltMeanHiatusCMIP6+"   "+AltMeanPostCMIP6)
    
; ======================================================================
;
;                            3. Plot PDFs
;
; ======================================================================

  ; 3 trend patterns
  ;vpX = (/0.09, 0.09, 0.52, 0.52, 0.52/)
  ;vpW = (/0.4, 0.4, 0.46, 0.46, 0.46/)
  ;vpY = (/0.9, 0.44, 0.91, 0.63, 0.35/)
  ;vpH = (/0.35, 0.35, 0.23, 0.23, 0.23/)
  ; 2 trend patterns
  vpX = (/0.09, 0.09, 0.46, 0.46/)
  vpW = (/0.35, 0.35, 0.54, 0.54/)
  vpY = (/0.84, 0.44, 0.84, 0.48/)
  vpH = (/0.30, 0.30, 0.27, 0.27/)
 
  wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                "hiatus_and_posthiatus_pdfs")

  pdfs = new(2, "graphic")
  dummy = new(100, "graphic")
  res = True
  res@gsnDraw = False
  res@gsnFrame = False
  res@trYMinF = 0.
  res@trYMaxF = 7.5
  res@trXMinF = -0.02*scale
  res@trXMaxF = 0.08*scale ; 0.06*scale
  res@tmYROn = False
  res@tmXTOn = False
  res@tmYLLabelFontHeightF = 0.016
  res@tmXBLabelFontHeightF = 0.016
  res@tmYLPrecision = 1
  res@tmXBPrecision = 1
  res@tmYLLabelJust = "CenterCenter"
  res@tmYLLabelDeltaF = -0.25
  res@tmXBLabelDeltaF = -0.5
  res@gsnXRefLine = 0.
  res@gsnXRefLineDashPattern = 2
  res@gsnXRefLineThicknessF = 0.25
;  res@gsnXRefLineColor =
  res@vpXF = vpX(0)
  res@vpYF = vpY(0)
  res@vpHeightF = vpH(0)
  res@vpWidthF = vpW(0)
  res@tiMainFontHeightF = 0.017
  res@tiMainFont = "helvetica" ;"helvetica-bold"
  res@tiMainString = "(a) "+hiatus_period ;+" GMST trend"
  res@tiMainOffsetYF = 0.01 ;0.
  res@tiMainFontThicknessF = 2.
  res@tiYAxisString = "Normalized Density" ;"Probability Density [("+unit+")~S~-1~N~]"
  res@tiYAxisFontHeightF = 0.015
  txres        := True
  txres@gsnDraw = False
  txres@gsnFrame = False
  txres@txFontHeightF = 0.009
  txres@txJust = "TopRight"
  
; 3.1 Plot hiatus PDF
  xaxis = (/-1., 1./)
  yaxis = (/0., 0./)
  xaxis@long_name = long_name+"("+unit+")"
  pdfs(0) = gsn_csm_xy(wks, (/-1., 1./), (/ 0., 0./), res)
  idummy = 0
  ilabel = 0.
  xlabelR = res@trXMaxF - 0.015
  xlabelL = res@trXMinF + 0.015
  xlabel = xlabelR
  ylabel = res@trYMaxF - 0.2
  ylabeli = 0.35

  ; CMIP5 PDF fill
  if (isvar("PDFhiatusCMIP5")) then
    polyPDFhiatusCMIP5 = pdf_polygon(PDFhiatusCMIP5)
    resp = Res_drawPDF(color_cmip5, periphery_thickness, 0, opacity_pdf_cmip5)
    dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
                                    polyPDFhiatusCMIP5(0, :), polyPDFhiatusCMIP5(1, :), resp)
    idummy = idummy + 1
  end if
  ; CMIP6 PDF fill
  if (isvar("PDFhiatusCMIP6")) then
    polyPDFhiatusCMIP6 = pdf_polygon(PDFhiatusCMIP6)
    resp = Res_drawPDF(color_cmip6, periphery_thickness, 0, opacity_pdf_cmip6)
    dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
                                    polyPDFhiatusCMIP6(0, :), polyPDFhiatusCMIP6(1, :), resp)
    idummy = idummy + 1
  end if
  if (isvar("PDFhiatusCMIP5")) then
    ; CMIP5 PDF periphery
;    resp = Res_drawPDF(color_cmip5, periphery_thickness, 0, 1.)
;    dummy(idummy) = gsn_add_polyline(wks, pdfs(0), \
;                                     polyPDFhiatusCMIP5(0, :), polyPDFhiatusCMIP5(1, :), resp)
;    idummy = idummy + 1
    ; CMIP5 PDF mean
    polyres := True
    xmarker = new(5, "float")
    ymarker = new(5, "float")
    datatondc(pdfs(0), PDFhiatusCMIP5@mean, res@trYMaxF, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.008
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.008
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) + 0.01
    ymarker(2) = ymarker(0) + 0.02
    ymarker(3) = ymarker(0) + 0.01
    ymarker(4) = ymarker(0)
    if (PDFhiatusCMIP5@GST.eq."GMST") then
      polyres@gsLineColor = color_cmip5
      gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
    else
      polyres@gsFillColor = color_cmip5
      gsn_polygon_ndc(wks, xmarker(0:3), ymarker(0:3), polyres)
    end if
    delete(xmarker)
    delete(ymarker)
  end if
  if (isvar("AltMeanHiatusCMIP5")) then
    polyres := True
    xmarker = new(5, "float")
    ymarker = new(5, "float")
    datatondc(pdfs(0), AltMeanHiatusCMIP5, res@trYMaxF, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.008
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.008
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) + 0.01
    ymarker(2) = ymarker(0) + 0.02
    ymarker(3) = ymarker(0) + 0.01
    ymarker(4) = ymarker(0)
    if (AltMeanHiatusCMIP5@GST.eq."GMST") then
      polyres@gsLineColor = color_cmip5
      gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
    else
      polyres@gsFillColor = color_cmip5
      gsn_polygon_ndc(wks, xmarker(0:3), ymarker(0:3), polyres)
    end if
    delete(xmarker)
    delete(ymarker)
  end if
  if (isvar("PDFhiatusCMIP6")) then
    ; CMIP6 PDF periphery
;    resp = Res_drawPDF(color_cmip6, periphery_thickness, 0, 1.)
;    dummy(idummy) = gsn_add_polyline(wks, pdfs(0), \
;                                     polyPDFhiatusCMIP6(0, :), polyPDFhiatusCMIP6(1, :), resp)
;    idummy = idummy + 1
    ; CMIP6 PDF mean
    polyres := True
    xmarker = new(5, "float")
    ymarker = new(5, "float")
    datatondc(pdfs(0), PDFhiatusCMIP6@mean, res@trYMaxF, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.008
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.008
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) + 0.01
    ymarker(2) = ymarker(0) + 0.02
    ymarker(3) = ymarker(0) + 0.01
    ymarker(4) = ymarker(0)
    if (PDFhiatusCMIP6@GST.eq."GMST") then
      polyres@gsLineColor = color_cmip6
      gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
    else
      polyres@gsFillColor = color_cmip6
      gsn_polygon_ndc(wks, xmarker(0:3), ymarker(0:3), polyres)
    end if
    delete(xmarker)
    delete(ymarker)
  end if
  if (isvar("AltMeanHiatusCMIP6")) then
    polyres := True
    xmarker = new(5, "float")
    ymarker = new(5, "float")
    datatondc(pdfs(0), AltMeanHiatusCMIP6, res@trYMaxF, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.008
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.008
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) + 0.01
    ymarker(2) = ymarker(0) + 0.02
    ymarker(3) = ymarker(0) + 0.01
    ymarker(4) = ymarker(0)
    if (AltMeanHiatusCMIP6@GST.eq."GMST") then
      polyres@gsLineColor = color_cmip6
      gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
    else
      polyres@gsFillColor = color_cmip6
      gsn_polygon_ndc(wks, xmarker(0:3), ymarker(0:3), polyres)
    end if
    delete(xmarker)
    delete(ymarker)
  end if
  ; PDF selected members
  if (composite_project.eq."cmip6") then
    tmp = tail_PDF(polyPDFhiatusCMIP6(0, :), polyPDFhiatusCMIP6(1, :), threshold_GSTtrend_composite)
    resp = Res_drawPDF(color_cmip6_selected, periphery_thickness, 0, opacity_selected_pdf)
  elseif (composite_project.eq."cmip5") then
    tmp = tail_PDF(polyPDFhiatusCMIP5(0, :), polyPDFhiatusCMIP5(1, :), threshold_GSTtrend_composite)
    resp = Res_drawPDF(color_cmip5_selected, periphery_thickness, 0, opacity_selected_pdf)
  elseif (composite_project.eq."cmip5+cmip6") then
    tmp = tail_PDF(polyPDFhiatusCMIP6(0, :), dim_max_n((/polyPDFhiatusCMIP5(1, :), polyPDFhiatusCMIP6(1, :)/), 0), threshold_GSTtrend_composite)
    resp = Res_drawPDF(color_cmip_selected, periphery_thickness, 0, opacity_selected_pdf)
  end if
  if (isvar("tmp")) then
    dummy(idummy) = gsn_add_polygon(wks, pdfs(0), tmp(0, :), tmp(1, :), resp)
    idummy = idummy + 1
    delete(tmp)
  end if
  ; Legend CMIP6
  if (isvar("PDFhiatusCMIP6")) then
    txres@txFontOpacityF = 1.
    txres@txFontColor := color_cmip6_legend
    dummy(idummy) = gsn_add_text(wks, pdfs(0), "CMIP6: "+PDFhiatusCMIP6@nmodels+" models,     ~C~             "+PDFhiatusCMIP6@nruns+" members ", xlabel, ylabel-ilabel*ylabeli, txres)
    idummy = idummy + 1
    ilabel = ilabel + 2.
    polyres := True
    polyres@gsFillColor = color_cmip6
    polyres@gsFillOpacityF = 1.
    x = xlabel-0.05
    y = ylabel-ilabel*ylabeli
    xmarker = new(5, "float")
    ymarker = new(5, "float")
    datatondc(pdfs(0), x+0.03, y, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.005
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.005
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) - 0.00625
    ymarker(2) = ymarker(0) - 0.0125
    ymarker(3) = ymarker(0) - 0.00625
    ymarker(4) = ymarker(0)
    gsn_polygon_ndc(wks, xmarker(0:3), ymarker(0:3), polyres)
    dummy(idummy) = gsn_add_text(wks, pdfs(0), "GSAT ensemble mean", x, y, txres)
    idummy = idummy + 1
    ilabel = ilabel + 1.
    
    y = ylabel-ilabel*ylabeli
    datatondc(pdfs(0), x+0.03, y, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.005
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.005
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) - 0.00625
    ymarker(2) = ymarker(0) - 0.0125
    ymarker(3) = ymarker(0) - 0.00625
    ymarker(4) = ymarker(0)
    polyres@gsLineColor = color_cmip6
    gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
    dummy(idummy) = gsn_add_text(wks, pdfs(0), "GMST ensemble mean", x, y, txres)
    idummy = idummy + 1
    ilabel = ilabel + 1.
    delete([/xmarker, ymarker/])
    
    polyres@gsFillOpacityF = opacity_pdf_cmip6
    y = ylabel-ilabel*ylabeli
    xmarker = (/x+0.01, x+0.05, x+0.05, x+0.01/)
    ymarker = (/y, y, y-ylabeli*0.75, y-ylabeli*0.75/)
    dummy(idummy) = gsn_add_polygon(wks, pdfs(0), xmarker, ymarker, polyres)
    idummy = idummy + 1
    dummy(idummy) = gsn_add_text(wks, pdfs(0), GST+" histogram", x, ylabel-ilabel*ylabeli, txres)
    idummy = idummy + 1
    if (isStrSubset(composite_project, "cmip6")) then
      polyres@gsFillColor := color_cmip6_selected
      polyres@gsFillOpacityF = opacity_selected_pdf
      xmarker(1) = xmarker(0)+0.01
      xmarker(2) = xmarker(1)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(0), xmarker, ymarker, polyres)
      idummy = idummy + 1    
      ilabel = ilabel + 1.
      txres@txFontOpacityF = 0.7; opacity_pdf_cmip6
      dummy(idummy) = gsn_add_text(wks, pdfs(0), "selected members", xlabel, ylabel-ilabel*ylabeli, txres)
      idummy = idummy + 1
      txres@txFontOpacityF = txres@txFontOpacityF * opacity_selected_pdf / opacity_pdf_cmip6 ; opacity_selected_pdf
      txres@txFontColor := color_cmip6_selected
      dummy(idummy) = gsn_add_text(wks, pdfs(0), "selected members", xlabel, ylabel-ilabel*ylabeli, txres)
      txres@txFontOpacityF = 1.
      idummy = idummy + 1
      polyres@gsLineColor := color_cmip6_selected
      x = (xmarker(0)+xmarker(1))/2.
      y = ylabel-ilabel*ylabeli
      dummy(idummy) = gsn_add_polyline(wks, pdfs(0), (/x, x-0.01/), (/y+ylabeli/2., y/), polyres)
      idummy = idummy + 1    
      ilabel = ilabel + 1.
    end if
    ilabel = ilabel + 0.5
    delete(xmarker)
    delete(ymarker)
  end if
  ; Legend CMIP5
  if (isvar("PDFhiatusCMIP5")) then
    txres@txFontColor := color_cmip5_legend
    txres@txFontOpacityF = 1.
    dummy(idummy) = gsn_add_text(wks, pdfs(0), "CMIP5: "+PDFhiatusCMIP5@nmodels+" models,     ~C~             "+PDFhiatusCMIP5@nruns+" members ", xlabel, ylabel-ilabel*ylabeli, txres)
    idummy = idummy + 1
    ilabel = ilabel + 2.
    polyres := True
    polyres@gsFillColor = color_cmip5
    polyres@gsFillOpacityF = 1.
    x = xlabel-0.05
    y = ylabel-ilabel*ylabeli
    xmarker = new(5, "float")
    ymarker = new(5, "float")
    datatondc(pdfs(0), x+0.03, y, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.005
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.005
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) - 0.00625
    ymarker(2) = ymarker(0) - 0.0125
    ymarker(3) = ymarker(0) - 0.00625
    ymarker(4) = ymarker(0)
    gsn_polygon_ndc(wks, xmarker, ymarker, polyres)
    dummy(idummy) = gsn_add_text(wks, pdfs(0), "GSAT ensemble mean", x, y, txres)
    idummy = idummy + 1
    ilabel = ilabel + 1.

    y = ylabel-ilabel*ylabeli
    datatondc(pdfs(0), x+0.03, y, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.005
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.005
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) - 0.00625
    ymarker(2) = ymarker(0) - 0.0125
    ymarker(3) = ymarker(0) - 0.00625
    ymarker(4) = ymarker(0)
    polyres@gsLineColor = color_cmip5
    gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
    dummy(idummy) = gsn_add_text(wks, pdfs(0), "GMST ensemble mean", x, y, txres)
    idummy = idummy + 1
    ilabel = ilabel + 1.
    delete([/xmarker, ymarker/])
    
    polyres@gsFillOpacityF = opacity_pdf_cmip5
    y = ylabel-ilabel*ylabeli
    xmarker = (/x+0.01, x+0.05, x+0.05, x+0.01/)
    ymarker = (/y, y, y-ylabeli*0.75, y-ylabeli*0.75/)
    dummy(idummy) = gsn_add_polygon(wks, pdfs(0), xmarker, ymarker, polyres)
    idummy = idummy + 1
    dummy(idummy) = gsn_add_text(wks, pdfs(0), GST+" histogram", x, y, txres)
    idummy = idummy + 1
    if (isStrSubset(composite_project, "cmip5")) then
      polyres@gsFillColor := color_cmip5_selected
      polyres@gsFillOpacityF = opacity_selected_pdf
      xmarker(1) = xmarker(0)+0.01
      xmarker(2) = xmarker(1)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(0), xmarker, ymarker, polyres)
      idummy = idummy + 1    
      ilabel = ilabel + 1.
      txres@txFontOpacityF = 0.7  ; opacity_pdf_cmip5
      dummy(idummy) = gsn_add_text(wks, pdfs(0), "selected members", xlabel, ylabel-ilabel*ylabeli, txres)
      idummy = idummy + 1
      txres@txFontOpacityF = txres@txFontOpacityF * opacity_selected_pdf / opacity_pdf_cmip5 ; opacity_selected_pdf
      txres@txFontColor := color_cmip5_selected
      dummy(idummy) = gsn_add_text(wks, pdfs(0), "selected members", xlabel, ylabel-ilabel*ylabeli, txres)
      txres@txFontOpacityF = 1.
      idummy = idummy + 1
      polyres@gsLineColor = color_cmip5_selected
      x = (xmarker(0)+xmarker(1))/2.
      y = ylabel-ilabel*ylabeli
      dummy(idummy) = gsn_add_polyline(wks, pdfs(0), (/x, x-0.01/), (/y+ylabeli/2., y/), polyres)
      idummy = idummy + 1    
      ilabel = ilabel + 1.
    end if
    ilabel = ilabel + 0.5
    delete(xmarker)
    delete(ymarker)
  end if
  ; Legend Obs
  xlabel = xlabelL
  ilabel = 0.
  txres@txJust = "TopLeft"
  if (isdefined("PDFhiatusObs").or.isdefined("SingleValhiatusObs")) then
    txres@txFontColor := "black"
    dummy(idummy) = gsn_add_text(wks, pdfs(0), "Observations", xlabel, ylabel-ilabel*ylabeli, txres)
    idummy = idummy + 1
    ilabel = ilabel + 1
  end if
  if (isdefined("PDFhiatusObs")) then
    j = 0
    do ii = 0, ListCount(PDFhiatusObs)-1
      polyPDFhiatusObs = pdf_polygon(PDFhiatusObs[ii])
      resp = Res_drawPDF(color_obs_ens(j), periphery_thickness, 3, 1.)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(0), \
                                      polyPDFhiatusObs(0, :), polyPDFhiatusObs(1, :), resp)
      idummy = idummy + 1
      dummy(idummy) = gsn_add_polyline(wks, pdfs(0), \
                                       polyPDFhiatusObs(0, :), polyPDFhiatusObs(1, :), resp)
      idummy = idummy + 1
      delete(polyPDFhiatusObs)
      resp@gsFillScaleF = 0.7
      x = xlabel+0.05
      y = ylabel-ilabel*ylabeli
      xmarker = (/x-0.01, x-0.05, x-0.05, x-0.01/)
      ymarker = (/y, y, y-ylabeli*0.75, y-ylabeli*0.75/)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(0), xmarker, ymarker, resp)
      idummy = idummy + 1
      xmarker2 = array_append_record(xmarker, xmarker(0), 0)
      ymarker2 = array_append_record(ymarker, ymarker(0), 0)
      dummy(idummy) = gsn_add_polyline(wks, pdfs(0), xmarker2, ymarker2, resp)
      idummy = idummy + 1
      txres@txFontColor := color_obs_ens(j)
      dummy(idummy) = gsn_add_text(wks, pdfs(0), PDFhiatusObs[ii]@dataset, x, y, txres)
      idummy = idummy + 1
      ilabel = ilabel + 1.
      j = j + 1
      delete(xmarker)
      delete(ymarker)
      delete(xmarker2)
      delete(ymarker2)
    end do
  end if
  if (isdefined("SingleValhiatusObs")) then
    polyres := True
    xmarker = new(4, "float")
    ymarker = new(4, "float")
    j = 0
    do ii = 0, ListCount(SingleValhiatusObs)-1
      polyres@gsFillColor := color_obs_single(j, :)
      polyres@gsLineColor := color_obs_single(j, :)
      datatondc(pdfs(0), tofloat(SingleValhiatusObs[ii]), res@trYMaxF, xmarker(0), ymarker(0))
      xmarker(1) = xmarker(0) - 0.007
      xmarker(2) = xmarker(0) + 0.007
      xmarker(3) = xmarker(0)
      ymarker(1) = ymarker(0) + 0.02
      ymarker(2) = ymarker(1)
      ymarker(3) = ymarker(0)
      if (any(SingleValhiatusObs[ii]@GST.eq."GSAT")) then ; Draw filled triangles first
        gsn_polygon_ndc(wks, xmarker(0:2), ymarker(0:2), polyres)
      end if
      x = xlabel+0.05
      y = ylabel-ilabel*ylabeli
      datatondc(pdfs(0), x-0.03, y, xmarker(0), ymarker(0))
      xmarker(1) = xmarker(0) 
      xmarker(2) = xmarker(0) + 0.004
      xmarker(0) = xmarker(0) - 0.004
      xmarker(3) = xmarker(0)
      ymarker(1) = ymarker(0) - 0.0125
      ymarker(2) = ymarker(0)
      ymarker(3) = ymarker(0)
      if (any(SingleValhiatusObs[ii]@GST.eq."GMST")) then
        gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
      else
        gsn_polygon_ndc(wks, xmarker(0:2), ymarker(0:2), polyres)
      end if
      txres@txFontColor := color_obs_single(j, :)
      dataset_name = SingleValhiatusObs[ii]@dataset
      if (dataset_name.eq."NOAAGlobalTemp") then
        dataset_name = "NOAA~C~GlobalTemp"
        ilabel = ilabel + 1
      elseif (dataset_name.eq."NOAAGlobalTemp-Interim") then
        dataset_name = "NOAA~C~GlobalTemp~C~-Interim"
        ilabel = ilabel + 2
      end if
      dummy(idummy) = gsn_add_text(wks, pdfs(0), dataset_name, x, y, txres)
      ilabel = ilabel + 1
      j = j + 1
    end do

    j = 0
    do ii = 0, ListCount(SingleValhiatusObs)-1
      polyres@gsFillColor := color_obs_single(j, :)
      polyres@gsLineColor := color_obs_single(j, :)
      datatondc(pdfs(0), tofloat(SingleValhiatusObs[ii]), res@trYMaxF, xmarker(0), ymarker(0))
      xmarker(1) = xmarker(0) - 0.007
      xmarker(2) = xmarker(0) + 0.007
      xmarker(3) = xmarker(0)
      ymarker(1) = ymarker(0) + 0.02
      ymarker(2) = ymarker(1)
      ymarker(3) = ymarker(0)
      if (any(SingleValhiatusObs[ii]@GST.eq."GMST")) then ; Superpose open triangles
        gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
      end if
      j = j + 1
    end do
    delete(xmarker)
    delete(ymarker)
  end if

; 3.2 Plot post-hiatus PDF

  res@tiXAxisString = long_name+" ("+unit+")"
  res@tiXAxisFontHeightF = 0.015
  res@vpXF = vpX(1)
  res@vpYF = vpY(1)
  res@vpHeightF = vpH(1)
  res@vpWidthF = vpW(1)
  res@tiMainString = "(b) "+post_period; +" GMST trend"
  res@tiMainOffsetYF = 0.01 ;0.
  pdfs(1) = gsn_csm_xy(wks, (/-1., 1./), (/ 0., 0./), res)
  ilabel = 0.
  xlabel = xlabelR
  txres@txJust = "TopRight"
  ; CMIP5 PDF fill
  if (isvar("PDFpostCMIP5")) then
    polyPDFpostCMIP5 = pdf_polygon(PDFpostCMIP5)
    resp = Res_drawPDF(color_cmip5, periphery_thickness, 0, opacity_pdf_cmip5)
    dummy(idummy) = gsn_add_polygon(wks, pdfs(1), \
                                    polyPDFpostCMIP5(0, :), polyPDFpostCMIP5(1, :), resp)
    idummy = idummy + 1
  end if
  ; CMIP6 PDF fill
  if (isvar("PDFpostCMIP6")) then
    polyPDFpostCMIP6 = pdf_polygon(PDFpostCMIP6)
    resp = Res_drawPDF(color_cmip6, periphery_thickness, 0, opacity_pdf_cmip6)
    dummy(idummy) = gsn_add_polygon(wks, pdfs(1), \
                                    polyPDFpostCMIP6(0, :), polyPDFpostCMIP6(1, :), resp)
    idummy = idummy + 1
  end if
  ; PDF selected members CMIP5
  if (any(composite_project.eq.(/"cmip5", "cmip5+cmip6"/))) then
;    do ii = 0, dimsizes(cmip5_member_names)-1
;      if (any(cmip5_member_names(ii).eq.composite_member_names) .and. \
;          .not.ismissing(GMSTtrendCMIP5Post(ii))) then
;        tmp = addline_PDF(polyPDFpostCMIP5(0, :), polyPDFpostCMIP5(1, :), GMSTtrendCMIP5Post(ii))
;        resp = Res_drawPDF(color_cmip5_selected, line_thickness, 0, opacity_selected_line)
;        dummy(idummy) = gsn_add_polyline(wks, pdfs(1), \
;                                         tmp(0, :), tmp(1, :), resp)
;        idummy = idummy + 1
;        delete(tmp)
;      end if
;    end do
    polyPDFpostCMIP5selected = pdf_polygon(PDFpostCMIP5selected)
    resp = Res_drawPDF(color_cmip5_selected, periphery_thickness, 0, opacity_selected_pdf)
    dummy(idummy) = gsn_add_polygon(wks, pdfs(1), \
                                    polyPDFpostCMIP5selected(0, :), polyPDFpostCMIP5selected(1, :), resp)
    idummy = idummy + 1
  end if
  ; PDF selected members CMIP5
  if (any(composite_project.eq.(/"cmip6", "cmip5+cmip6"/))) then
;    do ii = 0, dimsizes(cmip6_member_names)-1
;      if (any(cmip6_member_names(ii).eq.composite_member_names) .and. \
;          .not.ismissing(GMSTtrendCMIP6Post(ii))) then
;        tmp = addline_PDF(polyPDFpostCMIP6(0, :), polyPDFpostCMIP6(1, :), GMSTtrendCMIP6Post(ii))
;        resp = Res_drawPDF(color_cmip6_selected, line_thickness, 0, opacity_selected_line)
;        dummy(idummy) = gsn_add_polyline(wks, pdfs(1), \
;                                         tmp(0, :), tmp(1, :), resp)
;        idummy = idummy + 1
;        delete(tmp)
;      end if
;    end do
    polyPDFpostCMIP6selected = pdf_polygon(PDFpostCMIP6selected)
    resp = Res_drawPDF(color_cmip6_selected, periphery_thickness, 0, opacity_selected_pdf)
    dummy(idummy) = gsn_add_polygon(wks, pdfs(1), \
                                    polyPDFpostCMIP6selected(0, :), polyPDFpostCMIP6selected(1, :), resp)
    idummy = idummy + 1
  end if

  if (isvar("PDFpostCMIP5")) then
    ; CMIP5 PDF periphery
;    resp = Res_drawPDF(color_cmip5, periphery_thickness, 0, 1.)
;    dummy(idummy) = gsn_add_polyline(wks, pdfs(1), \
;                                     polyPDFpostCMIP5(0, :), polyPDFpostCMIP5(1, :), resp)
;    idummy = idummy + 1
    ; CMIP5 PDF mean
    polyres := True
    xmarker = new(5, "float")
    ymarker = new(5, "float")
    datatondc(pdfs(1), PDFpostCMIP5@mean, res@trYMaxF, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.008
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.008
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) + 0.01
    ymarker(2) = ymarker(0) + 0.02
    ymarker(3) = ymarker(0) + 0.01
    ymarker(4) = ymarker(0)
    if (PDFpostCMIP5@GST.eq."GMST") then
      polyres@gsLineColor = color_cmip5
      gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
    else
      polyres@gsFillColor = color_cmip5
      gsn_polygon_ndc(wks, xmarker(0:3), ymarker(0:3), polyres)
    end if
    delete(xmarker)
    delete(ymarker)
  end if
  if (isvar("AltMeanPostCMIP5")) then
    ; CMIP5 PDF mean
    polyres := True
    xmarker = new(5, "float")
    ymarker = new(5, "float")
    datatondc(pdfs(1), AltMeanPostCMIP5, res@trYMaxF, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.008
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.008
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) + 0.01
    ymarker(2) = ymarker(0) + 0.02
    ymarker(3) = ymarker(0) + 0.01
    ymarker(4) = ymarker(0)
    if (AltMeanPostCMIP5@GST.eq."GMST") then
      polyres@gsLineColor = color_cmip5
      gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
    else
      polyres@gsFillColor = color_cmip5
      gsn_polygon_ndc(wks, xmarker(0:3), ymarker(0:3), polyres)
    end if
    delete(xmarker)
    delete(ymarker)
  end if
  if (isvar("PDFpostCMIP6")) then
    ; CMIP6 PDF periphery
;    resp = Res_drawPDF(color_cmip6, periphery_thickness, 0, 1.)
;    dummy(idummy) = gsn_add_polyline(wks, pdfs(1), \
;                                     polyPDFpostCMIP6(0, :), polyPDFpostCMIP6(1, :), resp)
;    idummy = idummy + 1
    ; CMIP6 PDF mean
    polyres := True
    xmarker = new(5, "float")
    ymarker = new(5, "float")
    datatondc(pdfs(1), PDFpostCMIP6@mean, res@trYMaxF, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.008
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.008
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) + 0.01
    ymarker(2) = ymarker(0) + 0.02
    ymarker(3) = ymarker(0) + 0.01
    ymarker(4) = ymarker(0)
    if (PDFpostCMIP6@GST.eq."GMST") then
      polyres@gsLineColor = color_cmip6
      gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
    else
      polyres@gsFillColor = color_cmip6
      gsn_polygon_ndc(wks, xmarker(0:3), ymarker(0:3), polyres)
    end if
    delete(xmarker)
    delete(ymarker)
  end if
  if (isvar("AltMeanPostCMIP6")) then
    ; CMIP5 PDF mean
    polyres := True
    xmarker = new(5, "float")
    ymarker = new(5, "float")
    datatondc(pdfs(1), AltMeanPostCMIP6, res@trYMaxF, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.008
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.008
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) + 0.01
    ymarker(2) = ymarker(0) + 0.02
    ymarker(3) = ymarker(0) + 0.01
    ymarker(4) = ymarker(0)
    if (AltMeanPostCMIP6@GST.eq."GMST") then
      polyres@gsLineColor = color_cmip6
      gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
    else
      polyres@gsFillColor = color_cmip6
      gsn_polygon_ndc(wks, xmarker(0:3), ymarker(0:3), polyres)
    end if
    delete(xmarker)
    delete(ymarker)
  end if
  ; Legend CMIP6
  if (isvar("PDFpostCMIP6")) then
    txres@txFontOpacityF = 1.
    txres@txFontColor := color_cmip6_legend
    dummy(idummy) = gsn_add_text(wks, pdfs(1), "CMIP6: "+PDFpostCMIP6@nmodels+" models,     ~C~             "+PDFpostCMIP6@nruns+" members ", xlabel, ylabel-ilabel*ylabeli, txres)
    idummy = idummy + 1
    ilabel = ilabel + 2.
    polyres := True
    polyres@gsFillColor = color_cmip6
    polyres@gsFillOpacityF = 1.
    x = xlabel-0.05
    y = ylabel-ilabel*ylabeli
    xmarker = new(5, "float")
    ymarker = new(5, "float")
    datatondc(pdfs(1), x+0.03, y, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.005
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.005
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) - 0.00625
    ymarker(2) = ymarker(0) - 0.0125
    ymarker(3) = ymarker(0) - 0.00625
    ymarker(4) = ymarker(0)
    gsn_polygon_ndc(wks, xmarker(0:3), ymarker(0:3), polyres)
    dummy(idummy) = gsn_add_text(wks, pdfs(1), "GSAT ensemble mean", x, ylabel-ilabel*ylabeli, txres)
    idummy = idummy + 1
    ilabel = ilabel + 1.

    y = ylabel-ilabel*ylabeli
    datatondc(pdfs(1), x+0.03, y, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.005
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.005
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) - 0.00625
    ymarker(2) = ymarker(0) - 0.0125
    ymarker(3) = ymarker(0) - 0.00625
    ymarker(4) = ymarker(0)
    polyres@gsLineColor = color_cmip6
    gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
    dummy(idummy) = gsn_add_text(wks, pdfs(1), "GMST ensemble mean", x, ylabel-ilabel*ylabeli, txres)
    idummy = idummy + 1
    ilabel = ilabel + 1.
    delete([/xmarker, ymarker/])
    
    polyres@gsFillOpacityF = opacity_pdf_cmip6
    y = ylabel-ilabel*ylabeli
    xmarker = (/x+0.01, x+0.05, x+0.05, x+0.01/)
    ymarker = (/y, y, y-ylabeli*0.75, y-ylabeli*0.75/)
    dummy(idummy) = gsn_add_polygon(wks, pdfs(1), xmarker, ymarker, polyres)
    idummy = idummy + 1
    dummy(idummy) = gsn_add_text(wks, pdfs(1), GST+" histogram", x, ylabel-ilabel*ylabeli, txres)
    idummy = idummy + 1
    if (isStrSubset(composite_project, "cmip6")) then
      polyres@gsFillColor := color_cmip6_selected
      polyres@gsFillOpacityF = opacity_selected_pdf
      xmarker(0) = xmarker(0)+0.01
      xmarker(1) = xmarker(1)-0.01
      xmarker(2) = xmarker(2)-0.01
      xmarker(3) = xmarker(3)+0.01
      ymarker(0) = ymarker(2)+(ymarker(0)-ymarker(2 ))*0.45
      ymarker(1) = ymarker(0)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(1), xmarker, ymarker, polyres)
      idummy = idummy + 1    
      ilabel = ilabel + 1.
      txres@txFontOpacityF = 0.7
      dummy(idummy) = gsn_add_text(wks, pdfs(1), "selected in (a)", xlabel, ylabel-ilabel*ylabeli, txres)
      idummy = idummy + 1
      txres@txFontOpacityF = txres@txFontOpacityF * opacity_selected_pdf / opacity_pdf_cmip6 ; opacity_selected_pdf
      txres@txFontColor := color_cmip6_selected
      dummy(idummy) = gsn_add_text(wks, pdfs(1), "selected in (a)", xlabel, ylabel-ilabel*ylabeli, txres)
      txres@txFontOpacityF = 1.
      idummy = idummy + 1
      polyres@gsLineColor := color_cmip6_selected
      x = (xmarker(0)+xmarker(1))/2.
      y = ymarker(2)+(ymarker(0)-ymarker(2))*0.5
      dummy(idummy) = gsn_add_polyline(wks, pdfs(1), (/x, x-0.015/), (/y, ylabel-ilabel*ylabeli/), polyres)
      idummy = idummy + 1    
      ilabel = ilabel + 1.
    end if
    ilabel = ilabel + 0.5
    delete(xmarker)
    delete(ymarker)
  end if
  ; Legend CMIP5
  if (isvar("PDFpostCMIP5")) then
    txres@txFontOpacityF = 1.
    txres@txFontColor := color_cmip5_legend
    dummy(idummy) = gsn_add_text(wks, pdfs(1), "CMIP5: "+PDFpostCMIP5@nmodels+" models,     ~C~             "+PDFpostCMIP5@nruns+" members ", xlabel, ylabel-ilabel*ylabeli, txres)
    idummy = idummy + 1
    ilabel = ilabel + 2.
    polyres := True
    polyres@gsFillColor = color_cmip5
    polyres@gsFillOpacityF = 1.
    x = xlabel-0.05
    y = ylabel-ilabel*ylabeli
    xmarker = new(5, "float")
    ymarker = new(5, "float")
    datatondc(pdfs(1), x+0.03, y, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.005
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.005
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) - 0.00625
    ymarker(2) = ymarker(0) - 0.0125
    ymarker(3) = ymarker(0) - 0.00625
    ymarker(4) = ymarker(0)
    gsn_polygon_ndc(wks, xmarker(0:3), ymarker(0:3), polyres)
    dummy(idummy) = gsn_add_text(wks, pdfs(1), "GSAT ensemble mean", x, ylabel-ilabel*ylabeli, txres)
    idummy = idummy + 1
    ilabel = ilabel + 1.
    
    y = ylabel-ilabel*ylabeli
    datatondc(pdfs(1), x+0.03, y, xmarker(0), ymarker(0))
    xmarker(1) = xmarker(0) - 0.005
    xmarker(2) = xmarker(0)
    xmarker(3) = xmarker(0) + 0.005
    xmarker(4) = xmarker(0)
    ymarker(1) = ymarker(0) - 0.00625
    ymarker(2) = ymarker(0) - 0.0125
    ymarker(3) = ymarker(0) - 0.00625
    ymarker(4) = ymarker(0)
    polyres@gsLineColor = color_cmip5
    gsn_polyline_ndc(wks, xmarker, ymarker, polyres)
    dummy(idummy) = gsn_add_text(wks, pdfs(1), "GMST ensemble mean", x, ylabel-ilabel*ylabeli, txres)
    idummy = idummy + 1
    ilabel = ilabel + 1.
    delete([/xmarker, ymarker/])
    
    polyres@gsFillOpacityF = opacity_pdf_cmip5
    y = ylabel-ilabel*ylabeli
    xmarker = (/x+0.01, x+0.05, x+0.05, x+0.01/)
    ymarker = (/y, y, y-ylabeli*0.75, y-ylabeli*0.75/)
    dummy(idummy) = gsn_add_polygon(wks, pdfs(1), xmarker, ymarker, polyres)
    idummy = idummy + 1
    dummy(idummy) = gsn_add_text(wks, pdfs(1), GST+" histogram", x, ylabel-ilabel*ylabeli, txres)
    idummy = idummy + 1
    ilabel = ilabel + 1.
    if (isStrSubset(composite_project, "cmip5")) then
      y = ylabel-ilabel*ylabeli
      ymarker = (/y, y, y-ylabeli*0.75, y-ylabeli*0.75/)
      dummy(idummy) = gsn_add_polygon(wks, pdfs(1), xmarker, ymarker, polyres)
      idummy = idummy + 1
      polyres@gsFillColor := color_cmip5_selected
      polyres@gsFillOpacityF = opacity_selected_pdf
      dummy(idummy) = gsn_add_polygon(wks, pdfs(1), xmarker, ymarker, polyres)
      idummy = idummy + 1    
      txres@txFontOpacityF = 0.7
      dummy(idummy) = gsn_add_text(wks, pdfs(1), "selected in (a)", x, ylabel-ilabel*ylabeli, txres)
      idummy = idummy + 1
      txres@txFontOpacityF = txres@txFontOpacityF * opacity_selected_pdf / opacity_pdf_cmip5 ; opacity_selected_pdf
      txres@txFontColor := color_cmip5_selected
      dummy(idummy) = gsn_add_text(wks, pdfs(1), "selected in (a)", x, ylabel-ilabel*ylabeli, txres)
      txres@txFontOpacityF = 1.
      idummy = idummy + 1
      ilabel = ilabel + 1.8
    end if
    ilabel = ilabel + 0.5
    delete(xmarker)
    delete(ymarker)
  end if

; ======================================================================
;
;                          4. Draw patterned trends
;
; ======================================================================

  do ii = 0, dimsizes(input_dir)-1
    path = input_dir(ii)+"HiatusTrendPattern_unblended_unmasked_OBS_"+obs_dataset_pattern+".nc"
    if (fileexists(path)) then
      if (isdefined("paths_all")) then
        tmp = paths_all
        delete(paths_all)
        paths_all = array_append_record(tmp, path, 0)
        delete(tmp)
      else
        paths_all = path
      end if
      f = addfile(path, "r")
      TstrendObsHiatus = f->trend
      TstrendObsHiatus = TstrendObsHiatus * scale
      TstrendObsHiatus@units = str_sub_str(TstrendObsHiatus@units, "per year", "per "+toint(scale)+" years")
      break
    end if
  end do

  if (isatt(TstrendObsHiatus, "sig")) then
    draw_sig = True
    sigTstrendObsHiatus = TstrendObsHiatus
    sigTstrendObsHiatus = reshape(TstrendObsHiatus@sig, dimsizes(TstrendObsHiatus))
    sigTstrendObsHiatus@long_name = ""
    sigTstrendObsHiatus@units = ""
  else
    draw_sig = False
  end if
  
; -------------------------------------
; Plot obs trend pattern
  patterns = new(2, "graphic")
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
  resTs@res_mpOceanFillColor = "LightGray"
  resTs@res_mpLandFillColor = "LightGray"
  resTs@res_mpInlandWaterFillColor = "LightGray"
  resTs@res_lbLabelBarOn = False
  resTs@res_lbLabelFontHeightF = 0.016
  resTs@res_pmLabelBarOrthogonalPosF = 0.12
  resTs@res_lbTitleOn = True
  resTs@res_lbTitleString = "Near-surface temperature trend ("+unit+")"
  resTs@res_lbTitlePosition = "Bottom"
  resTs@res_lbTitleFontHeightF = 0.016
  resTs@res_lbTitleOffsetF = 0.1
  resTs@res_lbLabelFontHeightF = 0.013
  resTs@res_lbBoxLinesOn = True
  resTs@res_lbBoxSeparatorLinesOn = False
  resTs@res_lbRightMarginF = -0.1
  resTs@res_lbLeftMarginF = -0.1
  resTs@res_tiMainFont = "helvetica"
  resTs@res_tiMainFontHeightF = 0.017
  resTs@res_tiMainOffsetYF = -0.012
  resTs@res_mpPerimLineThicknessF = 0.
  resTs@res_mpLimbLineThicknessF = 0.
  resTs@res_mpPerimOn = False
  resTs@res_tmBorderThicknessF = 1.5
  resTs@res_tmXBMajorThicknessF =  resTs@res_tmBorderThicknessF
  resTs@res_tmYLMajorThicknessF =  resTs@res_tmBorderThicknessF
  resTs@res_tmXBMinorThicknessF =  resTs@res_tmBorderThicknessF/2.
  resTs@res_cnLevelSelectionMode = "ManualLevels"
  resTs@res_cnMinLevelValF = -1.05 ;-0.95 ; -1.05
  resTs@res_cnMaxLevelValF = 1.05 ;0.95 ; 1.05
  resTs@res_cnLevelSpacingF = 0.1
  pal = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
                           + "ipcc-ar6_temperature_div.rgb")
;  resTs@res_cnFillPalette = "temp_div"
  resTs@res_cnFillPalette = pal
  resTs@res_lbAutoManage = False
  resTs@res_lbLabelAutoStride = True
  
  txres@txJust = "TopRight"

  sres = True
  sres@gsnDraw = False
  sres@gsnFrame = False
  sres@cnFillOn       = True
  sres@cnLinesOn      = False
  sres@cnLineLabelsOn = False
  sres@cnInfoLabelOn  = False
  sres@cnMonoFillPattern = False
  sres@cnMonoFillColor = True
  sres@cnMonoFillScale = True
  sres@cnFillScaleF = 0.8
  sres@cnFillOpacityF = 0.4
  sres@cnLevelSelectionMode = "ExplicitLevels"
  sres@cnLevels = (/ 0.9 /)
  sres@cnFillPatterns = (/ 17, -1 /)
  sres@cnFillOpacityF = 0.1
  sres@cnFillColor = "grey30"
  sres@cnFillScaleF = 1.4
  sres@cnInfoLabelOn = False
  sres@lbLabelBarOn   = False
  sres@gsnAddCyclic = True
  
  resTs@res_tiMainString = "(c) "+str_sub_str(obs_dataset_pattern, "_", " ")
  resTs@res_vpXF = vpX(2)
  resTs@res_vpYF = vpY(2)
  resTs@res_vpHeightF = vpH(2)
  resTs@res_vpWidthF = vpW(2)
  copy_VarAtts(resTs, TstrendObsHiatus)
  patterns( 0 ) = contour_map(wks, TstrendObsHiatus, "")
  if (draw_sig) then
;    plot = gsn_csm_contour(wks, sigTstrendObsHiatus, sres)
;    overlay( patterns( 0 ), plot )

    xdummy = new(product(dimsizes(sigTstrendObsHiatus)), "graphic")
    idum = 0
    markerres = True
    markerres@gsMarkerIndex = 5
    markerres@gsMarkerSizeF = 0.002
    markerres@gsMarkerOpacityF = 0.4
    do j = 0, dimsizes(TstrendObsHiatus&lat)-1
      do i = 0, dimsizes(TstrendObsHiatus&lon)-1
        if (.not.ismissing(sigTstrendObsHiatus(j,i)).and.sigTstrendObsHiatus(j,i).lt.0.9) then
          xdummy(idum) = gsn_add_polymarker(wks, patterns(0), TstrendObsHiatus&lon(i), \
                                            TstrendObsHiatus&lat(j), markerres)
          idum = idum + 1
        end if
      end do
    end do
        
    plres = True
    x2 = resTs@res_vpXF + resTs@res_vpWidthF*0.98
    x1 = x2 - resTs@res_vpWidthF * 0.07
    x3 = x2
    x4 = x1
    y3 = resTs@res_vpYF - resTs@res_vpHeightF*0.93
    y2 = y3 - resTs@res_vpHeightF * 0.07
    y1 = y2
    y4 = y3
    lres = True
    lres@gsLineDashPattern = 0
    lres@gsLineThicknessF = 1.
    gsn_polyline_ndc(wks, (/x1, x2, x3, x4, x1/), (/y1, y2, y3, y4, y1/), lres)
;    plres = True
;    plres@gsFillScaleF = sres@cnFillScaleF
;    plres@gsFillIndex = sres@cnFillPatterns(0)
;    plres@gsFillOpacityF = sres@cnFillOpacityF
;    plres@gsFillColor = sres@cnFillColor
;    gsn_polygon_ndc(wks, (/x1, x2, x3, x4/), (/y1, y2, y3, y4/), plres)
    do i = 0, 3
      do j = 0, 2
        x = x1 + (i+0.5)*(x2-x1)/4.
        y = y2 + (j+0.5)*(y3-y2)/3. 
        gsn_polymarker_ndc(wks, x, y, markerres)
      end do
    end do
    txres@txFontColor := "black"
    txres@txJust = "TopRight"
    gsn_text_ndc(wks, "not significant~C~  at "+toint((1.-sres@cnLevels(0))*100.)+"% level", x2, y2-0.005, txres)
  end if

; -------------------------------------
; Plot composite trend pattern
  resTs@res_tiMainString = "(d) "+str_upper(composite_project)+" composite"
  resTs@res_vpXF = vpX(3)
  resTs@res_vpYF = vpY(3)
  resTs@res_vpHeightF = vpH(3)
  resTs@res_vpWidthF = vpW(3)
  resTs@res_lbLabelBarOn = True
  copy_VarAtts(resTs, TstrendCMIPHiatus)
  patterns( 1 ) = contour_map(wks, TstrendCMIPHiatus, "")
  
  composite_projects = str_get_field(composite_member_names, 1, " ")
  composite_models = str_get_field(composite_member_names, 2, " ")
  composite_models@_FillValue = ""

  ilabel = 0
  xlabel = resTs@res_vpXF+resTs@res_vpWidthF
  ylabel = resTs@res_vpYF
  ylabeli = 0.015
  if (isatt(TstrendCMIPHiatus, "nmembers_CMIP6")) then
    nmodels = count_unique_values(where(composite_projects.eq."CMIP6", \
                                  composite_models, composite_models@_FillValue))
    if (isatt(TstrendCMIPHiatus, "nmembers_CMIP5")) then
      txres@txFontOpacityF = 0.7
      txres@txFontColor := color_cmip6_legend
      gsn_text_ndc(wks, "CMIP6: "+nmodels+" models", xlabel, ylabel-ilabel*ylabeli, txres)
      txres@txFontColor := color_cmip6_selected
      txres@txFontOpacityF = txres@txFontOpacityF * opacity_selected_pdf / opacity_pdf_cmip6
      gsn_text_ndc(wks, "CMIP6: "+nmodels+" models", xlabel, ylabel-ilabel*ylabeli, txres)
    else
      txres@txFontOpacityF = 0.7
      txres@txFontColor := color_cmip6_legend
      gsn_text_ndc(wks, nmodels+" models", xlabel, ylabel-ilabel*ylabeli, txres)
      txres@txFontColor := color_cmip6_selected
      txres@txFontOpacityF = txres@txFontOpacityF * opacity_selected_pdf / opacity_pdf_cmip6
      gsn_text_ndc(wks, nmodels+" models", xlabel, ylabel-ilabel*ylabeli, txres)
    end if
    ilabel = ilabel + 1
    txres@txFontOpacityF = 0.7
    txres@txFontColor := color_cmip6_legend
    gsn_text_ndc(wks, TstrendCMIPHiatus@nmembers_CMIP6+" runs", xlabel, ylabel-ilabel*ylabeli, txres)
    txres@txFontColor := color_cmip6_selected
    txres@txFontOpacityF = txres@txFontOpacityF * opacity_selected_pdf / opacity_pdf_cmip6
    gsn_text_ndc(wks, TstrendCMIPHiatus@nmembers_CMIP6+" runs", xlabel, ylabel-ilabel*ylabeli, txres)
    ilabel = ilabel + 1
  end if
  if (isatt(TstrendCMIPHiatus, "nmembers_CMIP5")) then
    nmodels = count_unique_values(where(composite_projects.eq."CMIP5", \
                                  composite_models, composite_models@_FillValue))
    if (isatt(TstrendCMIPHiatus, "nmembers_CMIP6")) then
      txres@txFontOpacityF = 0.7
      txres@txFontColor := color_cmip5_legend
      gsn_text_ndc(wks, "CMIP5: "+nmodels+" models", xlabel, ylabel-ilabel*ylabeli, txres)
      txres@txFontColor := color_cmip5_selected
      txres@txFontOpacityF = txres@txFontOpacityF * opacity_selected_pdf / opacity_pdf_cmip5
      gsn_text_ndc(wks, "CMIP5: "+nmodels+" models", xlabel, ylabel-ilabel*ylabeli, txres)
    else
      txres@txFontOpacityF = 0.7
      txres@txFontColor := color_cmip5_legend
      gsn_text_ndc(wks, nmodels+" models", xlabel, ylabel-ilabel*ylabeli, txres)
      txres@txFontColor := color_cmip5_selected
      txres@txFontOpacityF = txres@txFontOpacityF * opacity_selected_pdf / opacity_pdf_cmip5
      gsn_text_ndc(wks, nmodels+" models", xlabel, ylabel-ilabel*ylabeli, txres)
    end if
    ilabel = ilabel + 1
    txres@txFontOpacityF = 0.7
    txres@txFontColor := color_cmip5_legend
    gsn_text_ndc(wks, TstrendCMIPHiatus@nmembers_CMIP5+" runs", xlabel, ylabel-ilabel*ylabeli, txres)
    txres@txFontColor := color_cmip5_selected
    txres@txFontOpacityF = txres@txFontOpacityF * opacity_selected_pdf / opacity_pdf_cmip5
    gsn_text_ndc(wks, TstrendCMIPHiatus@nmembers_CMIP5+" runs", xlabel, ylabel-ilabel*ylabeli, txres)
    ilabel = ilabel + 1
  end if

  txres@txFontColor := "black"
  txres@txFontHeightF = 0.02
  txres@txFont = "helvetica"
  txres@txJust = "TopCenter"
  gsn_text_ndc(wks, "Decadal modulation of the surface global warming trend", 0.5, max(vpY)+0.11, txres)
  txres@txFont = "helvetica-bold"
  txres@txFontHeightF = 0.018
  gsn_text_ndc(wks, "Distribution of GMST/GSAT trend", (vpX(0)+vpX(3))/2.-0.01, vpY(0)+0.075, txres)
  gsn_text_ndc(wks, "Pattern of trend over 1998-2012", (vpX(3)+1.)/2., vpY(0)+0.075, txres)
  draw(pdfs)
  draw(patterns)
  frame(wks)


; Output netcdf
  if (config_user_info@write_netcdf) then
    out_dir =  config_user_info@work_dir + "/"
    system("mkdir -p "+out_dir)
    nc_outfile = out_dir+"hiatus_and_posthiatus_trends.nc"
    if (fileexists(nc_outfile)) then
      system("rm -f "+nc_outfile)
    end if
    nc_outfile@existing = "append"

    do ii = 0, ListCount(GSTtrendObs)-1
      GSTtrendObs[ii]@var = GSTtrendObs[ii]@GST+"_trend_"+GSTtrendObs[ii]@dataset+"_hiatus"
      GSTtrendObs[ii]@diag_script = DIAG_SCRIPT
      GSTtrendObs[ii]!0 = "ensemble_"+GSTtrendObs[ii]@dataset
      GSTtrendObs[ii]&$GSTtrendObs[ii]!0$ = ispan(1, dimsizes(GSTtrendObs[ii]), 1)
      delete(GSTtrendObs[ii]@trend_post)
      delete(GSTtrendObs[ii]@eyear_post)
      delete(GSTtrendObs[ii]@syear_post)
      delete(GSTtrendObs[ii]@post_period)
      ncdf_outfile = ncdf_write(GSTtrendObs[ii], nc_outfile)
    end do
    
    GMSTtrendCMIP5Hiatus!0 = "ensemble_cmip5"
    GMSTtrendCMIP5Hiatus&ensemble_cmip5 = ispan(1, dimsizes(GMSTtrendCMIP5Hiatus), 1)
    GMSTtrendCMIP5Hiatus@var = "GMST_trend_CMIP5_hiatus"
    GMSTtrendCMIP5Hiatus@ensemble = str_join(metadata_att_as_array(GMSTtrendCMIP5, "dataset") \
                                    +"/"+metadata_att_as_array(GMSTtrendCMIP5, "ensemble"), ",")
    GMSTtrendCMIP5Hiatus@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(GMSTtrendCMIP5Hiatus, nc_outfile)

    GMSTtrendCMIP5Post!0 = "ensemble_cmip5"
    GMSTtrendCMIP5Post&ensemble_cmip5 = ispan(1, dimsizes(GMSTtrendCMIP5Post), 1)
    GMSTtrendCMIP5Post@var = "GMST_trend_CMIP5_posthiatus"
    GMSTtrendCMIP5Post@ensemble = str_join(metadata_att_as_array(GMSTtrendCMIP5, "dataset") \
                                  +"/"+metadata_att_as_array(GMSTtrendCMIP5, "ensemble"), ",")
    GMSTtrendCMIP5Post@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(GMSTtrendCMIP5Post, nc_outfile)

    GMSTtrendCMIP6Hiatus!0 = "ensemble_cmip6"
    GMSTtrendCMIP6Hiatus&ensemble_cmip6 = ispan(1, dimsizes(GMSTtrendCMIP6Hiatus), 1)
    GMSTtrendCMIP6Hiatus@var = "GMST_trend_CMIP6_hiatus"
    GMSTtrendCMIP6Hiatus@ensemble = str_join(metadata_att_as_array(GMSTtrendCMIP6, "dataset") \
                                     +"/"+metadata_att_as_array(GMSTtrendCMIP6, "ensemble"), ",")
    GMSTtrendCMIP6Hiatus@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(GMSTtrendCMIP6Hiatus, nc_outfile)

    GMSTtrendCMIP6Post!0 = "ensemble_cmip6"
    GMSTtrendCMIP6Post&ensemble_cmip6 = ispan(1, dimsizes(GMSTtrendCMIP6Post), 1)
    GMSTtrendCMIP6Post@var = "GMST_trend_CMIP6_posthiatus"
    GMSTtrendCMIP6Post@ensemble = str_join(metadata_att_as_array(GMSTtrendCMIP6, "dataset") \
                                   +"/"+metadata_att_as_array(GMSTtrendCMIP6, "ensemble"), ",")
    GMSTtrendCMIP6Post@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(GMSTtrendCMIP6Post, nc_outfile)

    GSATtrendCMIP5Hiatus!0 = "ensemble_cmip5"
    GSATtrendCMIP5Hiatus&ensemble_cmip5 = ispan(1, dimsizes(GSATtrendCMIP5Hiatus), 1)
    GSATtrendCMIP5Hiatus@var = "GSAT_trend_CMIP5_hiatus"
    GSATtrendCMIP5Hiatus@ensemble = str_join(metadata_att_as_array(GSATtrendCMIP5, "dataset") \
                                    +"/"+metadata_att_as_array(GSATtrendCMIP5, "ensemble"), ",")
    GSATtrendCMIP5Hiatus@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(GSATtrendCMIP5Hiatus, nc_outfile)
    
    GSATtrendCMIP5Post!0 = "ensemble_cmip5"
    GSATtrendCMIP5Post&ensemble_cmip5 = ispan(1, dimsizes(GSATtrendCMIP5Post), 1)
    GSATtrendCMIP5Post@var = "GSAT_trend_CMIP5_posthiatus"
    GSATtrendCMIP5Post@ensemble = str_join(metadata_att_as_array(GSATtrendCMIP5, "dataset") \
                                   +"/"+metadata_att_as_array(GSATtrendCMIP5, "ensemble"), ",")
    GSATtrendCMIP5Post@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(GSATtrendCMIP5Post, nc_outfile)

    GSATtrendCMIP6Hiatus!0 = "ensemble_cmip6"
    GSATtrendCMIP6Hiatus&ensemble_cmip6 = ispan(1, dimsizes(GSATtrendCMIP6Hiatus), 1)
    GSATtrendCMIP6Hiatus@var = "GSAT_trend_CMIP6_hiatus"
    GSATtrendCMIP6Hiatus@ensemble = str_join(metadata_att_as_array(GSATtrendCMIP6, "dataset") \
                                     +"/"+metadata_att_as_array(GSATtrendCMIP6, "ensemble"), ",")
    GSATtrendCMIP6Hiatus@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(GSATtrendCMIP6Hiatus, nc_outfile)

    GSATtrendCMIP6Post!0 = "ensemble_cmip6"
    GSATtrendCMIP6Post&ensemble_cmip6 = ispan(1, dimsizes(GSATtrendCMIP6Post), 1)
    GSATtrendCMIP6Post@var = "GSAT_trend_CMIP6_posthiatus"
    GSATtrendCMIP6Post@ensemble = str_join(metadata_att_as_array(GSATtrendCMIP6, "dataset") \
                                  +"/"+metadata_att_as_array(GSATtrendCMIP6, "ensemble"), ",")
    GSATtrendCMIP6Post@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(GSATtrendCMIP6Post, nc_outfile)

    tmp = (/TstrendObsHiatus/)
    copy_VarCoords(TstrendObsHiatus, tmp)
    tmp@var = "Ts_trend_"+obs_dataset_pattern+"_hiatus"
    tmp@diag_script = DIAG_SCRIPT
    tmp@sig = TstrendObsHiatus@sig
    tmp@units = TstrendObsHiatus@units
    ncdf_outfile = ncdf_write(tmp, nc_outfile)

    tmp := (/TstrendCMIPHiatus/)
    copy_VarCoords(TstrendCMIPHiatus, tmp)
    tmp@ensemble = str_sub_str(str_sub_str(str_sub_str(str_sub_str(TstrendCMIPHiatus@dataset, "CMIP6 ", ""), "CMIP5", ""), ", ", ","), " ", "/")
    tmp@var = "Ts_trend_"+str_upper(str_sub_str(TstrendCMIPHiatus@project,"+","_"))+"_composite_hiatus"
    tmp@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(tmp, nc_outfile)
  end if

; Log provenance
  log_provenance(nc_outfile, wks@fullname, \
                 "near surface temperature trends for the early 21c slow warming and post-slow warming periods", \
                 "trend", "global", (/"histogram", "geo"/), "kosaka_yu", "", paths_all)

  leave_msg(DIAG_SCRIPT, "")
  
end
