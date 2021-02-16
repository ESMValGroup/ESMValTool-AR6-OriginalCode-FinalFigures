; #############################################################################
; WRAPPER SCRIPT FOR COLLECTING AND PLOTTING TIMESERIES
; Author: Lisa Bock (DLR, Germany)
; IPCC AR6
; #############################################################################
; Description
;    Collects timeseries and their statistics previously calculated by
;    tsline.ncl and passes them to a new tsline plot
;
; Required diag_script_info attributes
;    styleset: as in diag_scripts/shared/plot/style.ncl functions
;
; Optional diag_script_info attributes
;    ref_start: start year of reference period for anomalies
;    ref_end: end year of reference period for anomalies
;    region: name of domain
;    plot_units: variable unit for plotting
;    y-min: set min of y-axis
;    y-max: set max of y-axis
;    order: order in which experiments should be plotted
;    run_ave: if not equal 0 than calculate running mean over this number
;             of years
;    header: if true, region name as header     
;    stat_shading: if true: shading of statistic range
;    ref_shading: if true: shading of reference period
;
; Caveats
;
; Modification history
;    20190204-A_bock_lisa: written.
;
; #############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/latlon.ncl"
load "$diag_scripts/shared/statistics.ncl"
load "$diag_scripts/shared/scaling.ncl"
load "$diag_scripts/shared/ensemble.ncl"

load "$diag_scripts/shared/plot/style.ncl"
load "$diag_scripts/shared/plot/xy_line.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")
  diag_script_base = basename(DIAG_SCRIPT)

  var_names = metadata_att_as_array(variable_info, "short_name")

  var0 = variable_info[0]@short_name
  info_var = select_metadata_by_name(input_file_info, var0)
  units0 = info_var[0]@units
  datasets = metadata_att_as_array(info_var, "dataset")
  dim_MOD = ListCount(info_var)

  if (diag_script_info@blending) then
    var = "gmst"
    units0 = "degC"
  else
    var = "gsat"
  end if

;  if(any(var_names .eq. "tasStderr")) then
;    obs_stderr = True
;    var1 = "tasStderr"
;    info_items1 = select_metadata_by_name(input_file_info, var1)
;  else
;    obs_stderr = False
;  end if

  ; Create output plot directory
  plot_dir = config_user_info@plot_dir
  system("mkdir -p " + plot_dir)
  aux_dir = diag_script_info@auxiliary_data_dir

  ; Check file type
  file_type = config_user_info@output_file_type
  if(ismissing(file_type)) then
    file_type = "ps"
  end if

  ; Set default values for non-required diag_script_info attributes
  set_default_att(diag_script_info, "region", "Global")
  set_default_att(diag_script_info, "run_ave", 0)
  set_default_att(diag_script_info, "header", True)
  set_default_att(diag_script_info, "stat_shading", False)
  set_default_att(diag_script_info, "ref_shading", False)

  ; Determine time range
  start_year = diag_script_info@start_year
  end_year = diag_script_info@end_year
  all_years = ispan(start_year, end_year, 1)

  ; Create time coordinate
  ntime = end_year - start_year + 1
  time = new(ntime, integer)
  time = ispan(start_year, end_year, 1)

  ref_start = diag_script_info@ref_start
  ref_end = diag_script_info@ref_end


  ; -------------------------------------------
  ; Read additional reference datasets
  ; -------------------------------------------

  if (dim_MOD .gt. 0) then

    data_ref = new((/dim_MOD, ntime/), double)
    data_ref!0 = "dataset"
    data_ref!1 = "time"
    data_ref&dataset = datasets
    data_ref&time = time

    ; Loop over datasets
    do imod = 0, dim_MOD - 1

      log_info("Process dataset: " + datasets(imod))

      ; Read data
      A0 = read_data(info_var[imod])

      ndim = dimsizes(A0)

      ; calculate anomaly for each grid point
      start_year_0 = info_var[imod]@start_year
      end_year_0 = info_var[imod]@end_year
      if (start_year_0 .gt. diag_script_info@ref_start) then
        ref_start = start_year_0
      else
        ref_start = diag_script_info@ref_start
      end if
      if (ref_start .ne. 1961 .or. ref_end .ne. 1990) then
        tmp = time_operations(A0, ref_start, ref_end, "average", \
                              "monthlyclim", True)
        do i = 0, ndim(0) - 1
          A0(i, :, :) = A0(i, :, :) - tmp(mod(i, 12), :, :)
        end do
      end if

      ; calculate field mean
      A0_mean := area_operations(A0, -90., 90., \
                                 0., 360., "average", True)

      ; yearly average
      diag = time_operations(A0_mean, -1, -1, "average", \
                             "yearly", True)


      ; Match time coordinate
      date = diag&year
      idx1 = ind(date(0).eq.data_ref&time)
      idx2 = ind(date(dimsizes(date) - 1).eq.data_ref&time)
      delete(date)

      data_ref(imod, idx1:idx2)  = (/diag/)

      delete(A0)
      delete(A0_mean)
      delete(diag)

    end do

  end if
 
  ; -----------------------------------------------
  ; Read pre-calculated multi model statistics
  ; -----------------------------------------------

  input_files = diag_script_info@input_files + "/" + \
    "tsline_gmst_anom" + "_stat.nc"
    ;"tsline_mm_tas_anom" + "_stat.nc"
  input_files := tostring(input_files)

  ; Filter out non-existing files
  input_files := input_files(ind(isfilepresent(input_files)))

  nfiles = dimsizes(input_files)

  dim_data   = (/nfiles, 3, ntime/)
  data_mean  = new(dim_data, double)
  data_mean!0 = "experiment"
  data_mean!1 = "stat"
  data_mean!2 = "time"
  data_mean&experiment = new(nfiles, string, "exp")
  data_mean&stat = (/"mean", "stat1", "stat2"/)
  data_mean&time = time
  data_mean@var = var

  ; Loop over files in the list, read and append data
  do ii = 0, nfiles - 1

    log_info("Read in " + input_files(ii))

    var_mean = var + "_mean"
    var_stddev = var + "_stddev"

    data_temp = ncdf_read(input_files(ii), var_mean)
    data_stat = ncdf_read(input_files(ii), var_stddev)
    data_stat1 = ncdf_read(input_files(ii), var+"_5quantile")
    data_stat2 = ncdf_read(input_files(ii), var+"_95quantile")

    log_info("Experiment: " + data_temp@experiment)

    if(isatt(diag_script_info, "order")) then
      ii_n = ind(data_temp@experiment .eq. diag_script_info@order)
      if (all(ismissing(ii_n))) then
        error_msg("f", DIAG_SCRIPT, "", "Collected experiment " + \
                  data_temp@experiment + " does not appear in " + \
                  "diag_script_info@order")
      elseif (ii_n .gt. nfiles-1)
        error_msg("f", DIAG_SCRIPT, "", "diag_script_info@order not " \
                  + "well defined (less experiments available)")
      end if
    else
      ii_n = ii
    end if

    ; Match time coordinate
    date = data_temp&time
    idx1 = ind(date(0).eq.data_mean&time)
    idx2 = ind(date(dimsizes(date) - 1).eq.data_mean&time)
    delete(date)

    data_mean(ii_n, 0, idx1:idx2)  = (/data_temp(:)/)
    ;data_mean(ii_n, 1, idx1:idx2)  = (/data_temp(:)-data_stat(:)/)
    ;data_mean(ii_n, 2, idx1:idx2)  = (/data_temp(:)+data_stat(:)/)
    data_mean(ii_n, 1, idx1:idx2)  = (/data_stat1(:)/)
    data_mean(ii_n, 2, idx1:idx2)  = (/data_stat2(:)/)
    data_mean&experiment(ii_n) = data_temp@experiment
    data_mean@units = units0

    delete(data_temp)
    delete(data_stat)
    delete(data_stat1)
    delete(data_stat2)

  end do

  ;; print statistics for Nicolas

  ;idexp = ind("CMIP6_historical-ssp245" .eq. data_mean&experiment)

  ;idx1 = ind(1850 .eq. data_mean&time)
  ;idx2 = ind(1900 .eq. data_mean&time)
  ;tmp := dim_avg(data_mean(idexp, :, idx1:idx2))
  ;print("1850 - 1900: " + tmp(0) + " (" + tmp(1) + ", " + tmp(2) + ")")

  ;idx1 = ind(2010 .eq. data_mean&time)
  ;idx2 = ind(2018 .eq. data_mean&time)
  ;tmp := dim_avg(data_mean(idexp, :, idx1:idx2))
  ;print("2010 - 2018: " + tmp(0) + " (" + tmp(1) + ", " + tmp(2) + ")")

  ; values for right inset
  idx1 = ind(1995 .eq. data_mean&time)
  idx2 = ind(2014 .eq. data_mean&time)
  mean_models:= dim_avg(data_mean(:, 0, idx1:idx2))
  mean_ref:= dim_avg(data_ref(:, idx1:idx2))
  ;mean_models := array_append_record(mean_models, data_ref, 0)

  ; -------------------------------------------
  ; NetCDF Output
  ; -------------------------------------------


  do i = 0, dimsizes(data_mean&experiment) - 1
    if (any(data_mean&experiment(i) .eq. (/"CMIP6_historical", "CMIP6_historical-ssp245"/))) then
      data_mean&experiment(i) = "CMIP6"
    elseif (any(data_mean&experiment(i) .eq. (/"CMIP5_historical", "CMIP5_historical-rcp45"/))) 
      data_mean&experiment(i) = "CMIP5"
    end if
  end do

  out_path = config_user_info@work_dir
  system("mkdir -p " + out_path)
  out_path1 = out_path + "tsline_collect_" + var0 + ".nc"
  data_mean@ncdf = out_path
  data_mean@diag_script = DIAG_SCRIPT
  data_mean@var = var
  ncdf_outfile = ncdf_write(data_mean, out_path1)

  if (isatt(diag_script_info, "ref")) then
    out_path1 = out_path + "tsline_collect_" + var0 + "_ref.nc"
    data_ref@ncdf = out_path
    data_ref@diag_script = DIAG_SCRIPT
    data_ref@var = var
    ncdf_outfile = ncdf_write(data_ref, out_path1)
  end if

  ; -------------------------------------------
  ; Plotting
  ; -------------------------------------------

  ; Define workstation
  outfile = config_user_info@plot_dir + data_mean@var + "_" + \
    str_sub_str(diag_script_info@region, " ", "_") + \
    "_multimodel_anom_" + start_year + "-" + end_year
  wks = gsn_open_wks(file_type, outfile)


  ; Select colors and other plotting attributes
  ; (see ./diag_scripts/lib/ncl/style.ncl)
  colors = project_style(info_var, diag_script_info, "colors")
  dashes = project_style(info_var, diag_script_info, "dashes")
  thicks = project_style(info_var, diag_script_info, "thicks")
  annots = project_style(info_var, diag_script_info, "annots")

  ; ************************************************
  ; plotting parameters
  ; ************************************************

  plot = new(1, graphic)

  res           = True
  res@gsnDraw   = False
  res@gsnFrame  = False

  res@vpXF      = 0.05
  res@vpYF      = 0.7
  res@vpHeightF = 0.4
  res@vpWidthF  = 0.7
  res@pmLegendDisplayMode = "Never"
  res@tmYRLabelFontHeightF = 0.016
  res@tmYLLabelFontHeightF = 0.016
  res@tiXAxisFontHeightF = 0.016
  res@tiYAxisFontHeightF = 0.016
  res@tmXBMajorOutwardLengthF = 0.006
  res@tmYLMajorOutwardLengthF = 0.006
  res@tmXBMinorOutwardLengthF = 0.003
  res@tmYLMinorOutwardLengthF = 0.003
  res@tmXBMajorLengthF = 0.006
  res@tmYLMajorLengthF = 0.006
  res@tmXBMinorLengthF = 0.003
  res@tmYLMinorLengthF = 0.003

  if (isatt(diag_script_info, "ref")) then
    min_tmp = (/min(data_mean), min(data_ref)/)
    max_tmp = (/max(data_mean), max(data_ref)/)
  else
    min_tmp = (/min(data_mean), min(data_stat)/)
    max_tmp = (/max(data_mean), max(data_stat)/)
  end if
  res@trYMinF  = min(min_tmp) - 0.05 * (max(max_tmp) - min(min_tmp))
  res@trYMaxF  = max(max_tmp) + 0.05 * (max(max_tmp) - min(min_tmp))
  res@tiYAxisOn = True
  res@tiXAxisString = "Year"
  res@gsnStringFontHeightF = 0.016

  xmin = start_year
  xmax = end_year
  res@trXMinF = xmin
  res@trXMaxF = xmax
  res@tmXBMode = "Explicit"
  if (xmax - xmin.gt.20) then
    res@tmXBValues = ispan(xmin, xmax, 10)
    res@tmXBLabels = ispan(xmin, xmax, 10)
    res@tmXBMinorValues =  ispan(xmin, xmax, 5)
  else
    res@tmXBValues = ispan(xmin, xmax, 5)
    res@tmXBLabels = ispan(xmin, xmax, 5)
    res@tmXBMinorValues =  ispan(xmin, xmax, 1)
  end if
   res@tmXBLabelAngleF = 45
   res@tmXBLabelJust     = "CenterRight"
  if (isatt(diag_script_info, "y_min")) then
    res@trYMinF = diag_script_info@y_min
  end if
  if (isatt(diag_script_info, "y_max")) then
    res@trYMaxF = diag_script_info@y_max
    ;res@trYMaxF = 1.2
  end if

  res@tmXBMode         = "Manual"
  res@tmXBTickSpacingF = 20

  res@tiMainString = ""
  ;if (isatt(diag_script_info, "title")) then
  ;  res@tiMainString = diag_script_info@title
  ;else
  ;  res@tiMainString = ""
  ;end if

  res@tiYAxisString = "Change in Temperature (~F34~0~F~ C)"


  ; ---------------------------------------------------------------------
  ; add multi model mean of different experiments
  ; ---------------------------------------------------------------------
  ; number of different experiments
  nexp     = dim_data(0)

  cmap = read_colormap_file("$diag_scripts/shared/plot/rgb/ipcc-ar6_cmip_line.rgb")
  res@xyDashPatterns    = (/0, 0, 0, 0, 0, 0/)
  res@xyLineThicknesses = (/5, 5, 5, 5, 5, 5/)
  res@xyLineColors = cmap(::-1,:)

  if (nexp .gt. 6) then
    error_msg("w", scriptname, funcname, "Color palette not defined for " \
              + "more than " + nexp + " experiments")
  end if

  ;res@pmLegendDisplayMode = "Always"
  res@pmLegendDisplayMode = "Never"
  res@xyExplicitLegendLabels = data_mean&experiment
  res@lgBoxMinorExtentF      = 0.2       ; Shorten the legend lines

  plot(0) = gsn_csm_xy(wks, data_ref&time, data_mean(:,0,:), res)  ; create plot

  res@pmLegendDisplayMode = "Never"

  ; ---------------------------------------------------------------------
  ; Add multi model statistics (5% and 95% quantile)
  ; ---------------------------------------------------------------------
  if (isatt(diag_script_info, "stat_shading")) then
    if (diag_script_info@stat_shading .ne. False) then
      res_stat = True
      res_stat@gsnXYFillOpacities = 0.2
      ;cmap = read_colormap_file("$diag_scripts/shared/plot/rgb/ipcc-ar6_damip_shading.rgb")
      cmap = read_colormap_file("$diag_scripts/shared/plot/rgb/ipcc-ar6_cmip_shading.rgb")
      copy_VarMeta(res, res_stat)
      res_stat@gsnXYFillColors = cmap(2,:)
      delete(res_stat@xyLineColors)
      ; We don't want the line, so make it transparent.
      ;res_stat@xyLineColor = -1
      res_stat@xyLineColor = cmap(2,:)
      delete(res_stat@xyLineThicknesses)
      res_stat@xyLineThicknesses = (/1, 1/)
      shading_plot = gsn_csm_xy(wks, data_ref&time, data_mean(0, 1:2, :), \
      ;shading_plot = gsn_csm_xy(wks, data_ref&time, data_stat(0:1, 0, :), \
                                res_stat)
      overlay(plot(0), shading_plot)
      if(nexp .ge. 2) then
        res_stat@gsnXYFillColors = cmap(1,:)
        res_stat@xyLineColor = cmap(1,:)
        shading_plot = gsn_csm_xy(wks, data_ref&time, data_mean(1, 1:2, :), \
                                  res_stat)
        overlay(plot(0), shading_plot)
      end if
      if(nexp .ge. 3) then
        res_stat@gsnXYFillColors = cmap(0,:)
        res_stat@xyLineColor = cmap(0,:)
        shading_plot = gsn_csm_xy(wks, data_ref&time, data_mean(2, 1:2, :), \
                                  res_stat)
        overlay(plot(0), shading_plot)
      end if
      if(nexp .ge. 4) then
        res_stat@gsnXYFillColors = cmap(3,:)
        res_stat@xyLineColor = cmap(3,:)
        shading_plot = gsn_csm_xy(wks, data_ref&time, data_mean(3, 1:2, :), \
                                  res_stat)
        overlay(plot(0), shading_plot)
      end if
      if(nexp .gt. 4) then
        error_msg("w", scriptname, funcname, "Color palette not defined for " \
                  + nexp + " experiments")
      end if
    end if
  end if

  ; ---------------------------------------------------------------------
  ; add reference datasets
  ; ---------------------------------------------------------------------
  if (isatt(diag_script_info, "ref")) then
    delete(res@xyDashPatterns)
    delete(res@xyLineThicknesses)
    delete(res@xyLineColors)
    res@xyDashPatterns    = dashes
    ;res@xyLineThicknesses = thicks
    res@xyLineThicknesses = (/3, 3, 3, 3, 3/)   ; make lines thicker
    res@xyLineColors      = colors  ; change line color

    ref_p = gsn_csm_xy(wks, data_ref&time, data_ref, res)  ; create plot
    overlay(plot(0), ref_p)
  end if

  if (diag_script_info@ref_shading) then

    ; ---------------------------------------------------------------------
    ; yellow shading of reference period
    ; ---------------------------------------------------------------------
    gsres                = True
    gsres@gsFillColor    = "Grey70" ;"yellow"
    gsres@gsFillOpacityF = 0.1
    xbox = (/ref_start, ref_end, ref_end, ref_start, ref_start/)
    ybox = (/res@trYMinF, res@trYMinF, res@trYMaxF, res@trYMaxF, res@trYMinF/)
    newplot00 = gsn_add_polygon(wks, plot(0), xbox, ybox, gsres)

    txtres               = True
    txtres@txFont        = "helvetica-bold"
    txtres@txFontColor   = "Grey40"
    txtres@txFontHeightF = 0.013
    ref_txt = gsn_add_text(wks, plot, "reference period", \
                           0.5*(ref_start + ref_end), \
                           res@trYMinF + 0.05 * (res@trYMaxF - res@trYMinF), \
                           txtres)

  end if


  ; ---------------------------------------------------------------------
  ; Add emulators
  ; ---------------------------------------------------------------------

  ;timeseries

  em_means = new((/17, ntime/), double)
  em_means!0 = "experiment"
  em_means!1 = "time"
  em_means&time = time

  ;f_em = aux_dir + "/20201711-1141_emulator-timeseries-fig3-3.csv"
  f_em = aux_dir + "/20201711-1141_emulator-output-mean-median-5-95-likely-range-fig3-3.csv"

  field_em = asciiread(f_em,-1,"string")

  delim = ","

  do i = 0, ntime-1
    em_means(:, i) = todouble(str_get_field(field_em, i+9, delim))
  end do

  res@xyLineColors := (/"yellow", "yellow","yellow","yellow"/)
  res@xyDashPatterns    := (/0,0,0,0/)
  res@xyLineThicknesses = (/1, 1, 1, 1, 3/)   ; make lines thicker

  em_p = gsn_csm_xy(wks, em_means&time, em_means(5:8, :), res)  ; create plot
  overlay(plot(0), em_p)

  ; ---------------------------------------------------------------------
  ; Add emulators PDF
  ; ---------------------------------------------------------------------

  em_pdf = new((/200/), double)
  ;em_pdf!0 = "emulator"
  em_pdf!0 = "temperature"

  f_em = aux_dir + "/20201711-1141_emulator-pdfs-fig3-3.csv"

  field_em := asciiread(f_em,-1,"string")

  tmp := new((/9,200/), double)

  do i = 0, 199
    tmp(:, i) = (/todouble(str_get_field(field_em, i+2, delim))/)
  end do

  em_pdf( :) = (/tmp(1, :)/)
  em_pdf&temperature = tmp(2,:)

  res0 = True
  res0@gsnDraw   = False
  res0@gsnFrame  = False
  res0@vpHeightF = 0.5
  res0@vpWidthF  = 0.05

  res0@xyLineColors := (/"yellow", "yellow","yellow","yellow"/)
  res0@xyDashPatterns    := (/0,0,0,0/)
  res0@xyLineThicknesses = (/1, 1, 1, 1, 3/)   ; make lines thicker

  res0@tmXTOn = False
  res0@tmXBOn = False
  res0@tmYROn = True
  res0@tmYLLabelsOn = False
  res0@tmYRLabelsOn = True
  res0@tmYRLabelFontHeightF = 0.016
  res0@tmYLLabelFontHeightF = 0.016
  res0@tiYAxisFontHeightF = 0.016
  res0@tiYAxisSide   = "Right"
  res0@tiYAxisAngleF = 90.
  res0@tiYAxisOn = True
  res0@gsnStringFontHeightF = 0.016

  res0@trYMinF = res@trYMinF
  res0@trYMaxF = res@trYMaxF
  res0@trXMinF = 0.
  res0@trXMaxF = 5.

  res0@tiYAxisString = "1995-2014 Mean Temperature rel. to 1850-1900 (~F34~0~F~ C)"

  plot2 = gsn_csm_xy(wks, em_pdf, em_pdf&temperature, res0)

  em_pdf( :) = (/tmp(3, :)/)
  em_pdf&temperature = tmp(4,:)

  plotu = gsn_csm_xy(wks, em_pdf, em_pdf&temperature, res0)
  overlay(plot2, plotu)

  em_pdf( :) = (/tmp(5, :)/)
  em_pdf&temperature = tmp(6,:)

  plotv = gsn_csm_xy(wks, em_pdf, em_pdf&temperature, res0)
  overlay(plot2, plotv)

  em_pdf( :) = (/tmp(7, :)/)
  em_pdf&temperature = tmp(8,:)

  plotw = gsn_csm_xy(wks, em_pdf, em_pdf&temperature, res0)
  overlay(plot2, plotw)

  res@xyDashPatterns    := (/0, 0, 0, 0, 0, 0/)
  res@xyLineThicknesses := (/5, 5, 5, 5, 5, 5/)
  cmap = read_colormap_file("$diag_scripts/shared/plot/rgb/ipcc-ar6_cmip_line.rgb")
  res@xyLineColors := cmap(::-1,:)
  var := fspan(0., 5., 3)
  tmp := new((/2, 3/), double)
  tmp(:, 0) = mean_models(:)
  tmp(:, 1) = mean_models(:)
  tmp(:, 2) = mean_models(:)
  plotx = gsn_csm_xy(wks, var, tmp, res)
  overlay(plot2, plotx)

  res@xyDashPatterns    := dashes
  res@xyLineThicknesses := (/3, 3, 3, 3, 3/)   ; make lines thicker
  res@xyLineColors      := colors  ; change line color
  tmp := new((/4, 3/), double)
  tmp(:, 0) = mean_ref(:)
  tmp(:, 1) = mean_ref(:)
  tmp(:, 2) = mean_ref(:)
  ploty = gsn_csm_xy(wks, var, tmp, res)
  overlay(plot2, ploty)


  getvalues plot2
    "trXMinF"   : bxmin
    "trYMinF"   : bymin
    "trXMaxF"   : bxmax
    "trYMaxF"   : bymax
  end getvalues

  xbox1 = (/bxmin,bxmax,bxmax,bxmin,bxmin/)
  ybox1 = (/bymin,bymin,bymax,bymax,bymin/)

  gnres = True
  gnres@gsFillColor     = "Gray20"
  gnres@gsFillOpacityF = 0.1
  gnres@tfPolyDrawOrder = "PreDraw"
  plot@$unique_string("box")$ = gsn_add_polygon(wks,plot2,xbox1,ybox1,gnres)
 
  newplot = gsn_attach_plots(plot(0), plot2, res, res0)


  ; ---------------------------------------------------------------------
  ; Draw some lines to create a legend
  ; ---------------------------------------------------------------------
  res_lines                   = True      ; polyline mods desired
  res_lines@tfPolyDrawOrder   = "Predraw"
  res_lines@gsLineColor       = "grey"    ; line color
  res_lines@gsLineThicknessF  = 1.        ; line thicker
  res_lines@gsLineDashPattern = 1.        ; dash pattern

  xx = (/res@trXMinF, res@trXMaxF/)
  yy = (/0.0, 0.0/)
  dum0 = gsn_add_polyline(wks, plot(0), xx, yy, res_lines)

  ; ***********************************************
  ; legend resources
  ; ***********************************************

  lgres                    = True
  lgres@lgItemType         = "Lines"   ; show lines only (default)
  lgres@lgLabelFontHeightF = .06       ; set the legend label font thickness
  lgres@vpWidthF           = 0.2      ; width of legend (NDC)
  lgres@vpHeightF          = 0.15      ; height of legend (NDC)
  lgres@lgPerimColor       = "gray"    ; draw the box perimeter in orange
  lgres@lgPerimThicknessF  = 1.0       ; thicken the box perimeter
  ;lgres@lgBoxBackground    = "white"
  lgres@lgPerimFill        = 0
  lgres@lgPerimFillColor   = "white"

  delete(cmap)
  cmap = read_colormap_file("$diag_scripts/shared/plot/rgb/ipcc-ar6_cmip_line.rgb")
  tmp := tostring(cmap)
  tmp1 = (/"(/" + tmp(2,0) + "," + tmp(2,1) + "," + tmp(2,2) + "/)" , \
           "(/" + tmp(1,0) + "," + tmp(1,1) + "," + tmp(1,2) + "/)" /)
           ;"(/" + tmp(1,0) + "," + tmp(1,1) + "," + tmp(1,2) + "/)" , \
           ;"(/" + tmp(0,0) + "," + tmp(0,1) + "," + tmp(0,2) + "/)" /)
  lgres@lgLineColors      = array_append_record(colors, tmp1, 0)
  lgres@lgDashIndexes     = array_append_record(dashes, (/0,0/), 0)
  ;lgres@lgDashIndexes     = array_append_record(dashes, (/0,0,0/), 0)
  ;lgres@lgLineThicknesses = thicks + 3
  lgres@lgLineThicknessF = 3
  labels = array_append_record(annots, data_mean&experiment, 0)

  ;emulators
  lgres@lgLineColors      := array_append_record(lgres@lgLineColors, "yellow", 0)
  lgres@lgDashIndexes     := array_append_record(lgres@lgDashIndexes, (/0/), 0)
  labels := array_append_record(labels, "Emulators", 0)

  nitems = dimsizes(labels)
  lgres@lgItemOrder = ispan(nitems - 1, 0, 1)

  ; Create legend
  lbid = gsn_create_legend(wks, nitems, labels, lgres)

  amres = True
  amres@amParallelPosF   = - 0.3 ;-0.38  ;0.81
  amres@amOrthogonalPosF = - 0.25 ;-0.3  ;0.0
  annoid1 = gsn_add_annotation(plot(0), lbid, amres)


  resP = True
  resP@gsnMaximize = True
  resP@gsnPaperOrientation = "portrait"
  resP@gsnPaperMargin =  0.8

  ;if (isatt(diag_script_info, "panel_num")) then
  ;  resP@gsnPanelFigureStrings = diag_script_info@panel_num
  ;else
  ;  resP@gsnPanelFigureStrings = (/"a)"/)
  ;end if
  ;resP@gsnPanelFigureStringsPerimOn = False
  ;resP@gsnPanelFigureStringsBackgroundFillColor ="transparent"
  ;resP@gsnPanelFigureStringsJust = "TopLeft"
  ;resP@gsnPanelFigureStringsFontHeightF = 0.016

  gsn_panel(wks, plot, (/1, 1/), resP)

  log_info(" wrote " + outfile + "." + file_type)

;  ; ***************************************
;  ; add meta data to plot (for reporting)
;  ; ***************************************
;
;  if (diag_script_info@region .eq. "Global") then
;    domain = "global"
;  else
;    domain = "reg"
;  end if
;
;  statistics = "anomaly"
;
;  caption = "Multi model mean time series of the " + statistics + \
;             " for variable " + varname + ", similar to IPCC AR5, fig. 10.21."
;  contrib_authors = (/""bock_lisa"/)
;
;  ; Call provenance logger
;  log_provenance(ncdf_outfile, \
;                 outfile + "." + file_type, \
;                 caption, \
;                 statistics, \
;                 domain, \
;                 "times", \
;                 contrib_authors, \
;                 (/"flato13ipcc", "jones13jgr"/), \
;                 metadata_att_as_array(input_file_info, "filename"))


  leave_msg(DIAG_SCRIPT, "")

end
