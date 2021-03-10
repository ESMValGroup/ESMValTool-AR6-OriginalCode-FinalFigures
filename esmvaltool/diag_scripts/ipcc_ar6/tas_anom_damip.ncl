; #############################################################################
; PLOTS TIME SERIES
; Authors: Lisa Bock (DLR, Germany), Sabrina Wenzel (DLR, Germany) and
;         Irene Cionni (ENEA, Italy)
; #############################################################################
;
; Description
;    This script is for plotting a time series of the global mean (anomaly).
;
; Required diag_script_info attributes (diagnostic specific)
;
; Optional diag_script_info attributes (diagnostic specific)
;    ref_start: start year of reference period for anomalies
;    ref_end: end year of reference period for anomalies
;    ref_mask: if true, model fields will be masked by reference fields
;    region: name of domain
;    plot_units: variable unit for plotting
;    y-min: set min of y-axis
;    y-max: set max of y-axis
;    run_ave: if not equal 0 than calculate running mean over this number
;             of years
;    header: if true, region name as header
;
; Caveats
;
; Modification history
;    20190911-bock_lisa: included method of Jones et al., 2013 and
;                        added provenance
;    20181112-bock_lisa: code rewritten for ESMValTool v2.0
;    20170623-lauer_axel: added tags for reporting
;    20160905-bock_lisa: implemented multi-model mean and variable refence
;                        period for anomaly with refence mean value
;    20151027-lauer_axel: moved call to 'write_references' to the beginning
;                         of the code
;    20150622-wenzel_sabrina: added optional anomaly calculation, choise of
;                             area opperations (sum, average) and detrending of
;                             time series.
;    20150420-righi_mattia: adapted to new structure. Some of the original
;                           features not ported yet (seasonal average,
;                            smoothing, multi-model mean, etc.).
;    20??????-cionni_irene: written.
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

  ; Check required diag_script_info attributes
  exit_if_missing_atts(diag_script_info, (/"panels", "start_year", "end_year"/))

  ; Set default values for non-required diag_script_info attributes
  set_default_att(diag_script_info, "anomaly", True)
  set_default_att(diag_script_info, "ref_mask", False)
  set_default_att(diag_script_info, "region", "Global")
  ;set_default_att(diag_script_info, "header", False)

  if (isatt(diag_script_info, "title")) then
    header = diag_script_info@title
  else
    header = ""
  end if

  ; Create output directories
  plot_dir = config_user_info@plot_dir
  system("mkdir -p " + plot_dir)
  out_path = config_user_info@work_dir
  system("mkdir -p " + out_path)

  if (diag_script_info@anomaly) then
    anom = "anom"
    if (isatt(diag_script_info, "ref_start") .and. \
        isatt(diag_script_info, "ref_end")) then
      ref_start = diag_script_info@ref_start
      ref_end   = diag_script_info@ref_end
    else
      error_msg("f", DIAG_SCRIPT, "", \
                "period for reference years is not defined " + \
                "(needed for anomaly)")
    end if
  else
    anom = "abs"
  end if

  ; Plot file type
  file_type = config_user_info@output_file_type
  if (ismissing(file_type)) then
    file_type = "ps"
  end if

  ; Determine time range
  start_year = diag_script_info@start_year
  end_year = diag_script_info@end_year
  all_years = ispan(start_year, end_year, 1)

  ; Create time coordinate
  ntime = end_year - start_year + 1
  time = new(ntime, integer)
  time = ispan(start_year, end_year, 1)


  ; *************************************
  ; General settings for the plot
  ; *************************************

  ; Define workstation
  outfile = config_user_info@plot_dir + "tas_" + anom + "_damip_" \
            + header + "_" + start_year + "-" + end_year
  wks = gsn_open_wks(file_type, outfile)

  ; Set resources
  res = True
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

  xmin = start_year
  xmax = end_year
  res@trXMinF = xmin
  res@trXMaxF = xmax
  ; res@trXMaxF = xmax + 0.25 * (xmax - xmin)
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

  res@tmXBMode         = "Manual"
  res@tmXBTickSpacingF = 20

  if (isatt(diag_script_info, "y_min")) then
    res@trYMinF = diag_script_info@y_min
  end if
  if (isatt(diag_script_info, "y_max")) then
    res@trYMaxF = diag_script_info@y_max
  end if
  res@tiYAxisOn = True
  res@tiXAxisString = ""
  res@gsnStringFontHeightF = 0.016



  ; ------------------------------------------------------------------
  ; Panel settings
  ; ------------------------------------------------------------------

  nvar = ListCount(variable_info)

  panels = diag_script_info@panels
  reference = diag_script_info@reference
  tmp := dimsizes(panels)
  if(dimsizes(tmp).gt.1) then
    npanels = tmp(0)
    nexp = tmp(1)
  else 
    npanels = 1
    nexp = tmp(0)
  end if
  delete(tmp)
  plots = new((/npanels/), graphic)


  ; ------------------------------------------------------------------
  ; Loop over panels
  ; ------------------------------------------------------------------

  do ivar = 0, npanels-1

    ; ------------------------------------------------------------------
    ; Read reference dataset
    ; ------------------------------------------------------------------

    var = reference(ivar)

    log_info("processing " + var)

    ;  Retrieve metadata item
    atts = True
    atts@variable_group = var
    info_ref = select_metadata_by_atts(input_file_info, atts)
    delete(atts)

    ref_dataset = metadata_att_as_array(info_ref, "dataset")

    A_ref = read_data(info_ref[0])

    ; Anomaly
    if (anom .eq. "anom") then
      tmp = time_operations(A_ref, ref_start, ref_end, "average", \
                            "monthlyclim", True)
      ndim := dimsizes(A_ref)
      do i = 0, ndim(0) - 1
        A_ref(i, :, :) = A_ref(i, :, :) - tmp(mod(i, 12), :, :)
      end do
      delete(tmp)
    end if

    ; yearly average
    tmp := time_operations(A_ref, -1, -1, "average", \
                                "yearly", True)

    ; calculate field mean
    diag_ref := area_operations(tmp, -90., 90., \
                             0., 360., "average", True)

    ref_arr = new((/ntime/), double)
    ref_arr!0 = "time"
    ref_arr&time = time
    ref_arr@_FillValue = 1e+20
    ref_arr = ref_arr@_FillValue

    ; Match time coordinate
    date = diag_ref&year
    idx1 = ind(date(0).eq.ref_arr&time)
    idx2 = ind(date(dimsizes(date) - 1).eq.ref_arr&time)
    ref_arr(idx1:idx2) = (/diag_ref/)
    copy_VarAtts(diag_ref, ref_arr)
    delete(diag_ref)
    delete(date)

    ; ------------------------------------------------------------------
    ; Read model datasets
    ; ------------------------------------------------------------------

    ; create array for statistics
    model_arr_stat = new((/5, 3, ntime/), double)
    model_arr_stat!0 = "exp"
    model_arr_stat!1 = "stat"
    model_arr_stat!2 = "time"
    model_arr_stat&exp = (/"exp1", "exp2", "exp3", "exp4", "ref"/)
    model_arr_stat&time = time
    model_arr_stat&stat = (/"mean", "min", "max"/)
    ;model_arr_stat@_FillValue = 1e+20
    ;model_arr_stat = model_arr_stat@_FillValue

    model_arr_stat(4, 0, :) = (/ref_arr/)

    do iexp = 0, nexp - 1

      if(npanels .gt. 1) then
        var = panels(ivar, iexp)
      else
        var = panels(iexp)
      end if

      log_info("processing " + var)

      ;  Retrieve metadata item
      atts = True
      atts@variable_group = var
      info_items = select_metadata_by_atts(input_file_info, atts)
      delete(atts)

      var0 = info_items[0]@short_name
      exp0 = info_items[0]@exp
      ;project0 = input_file_info[0]@project
      datasetnames := metadata_att_as_array(info_items, "dataset")
      ensembles := metadata_att_as_array(info_items, "ensemble")
      dim_MOD = ListCount(info_items)

      ; get multi-model mean index
      mmm_ind = ind(datasetnames .eq. "MultiModelMean")

      ; Create model array
      model_arr = new((/dim_MOD, ntime/), double)
      model_arr!0 = "model"
      model_arr!1 = "time"
      model_arr&model = datasetnames
      model_arr&time = time
      model_arr@_FillValue = 1e+20
      model_arr = model_arr@_FillValue

      ; Loop over models
      do imod = 0, dim_MOD - 1

        log_info("Process dataset: " + datasetnames(imod))

        ; Read data
        A0 = read_data(info_items[imod])
        dnames = getVarDimNames(A0)

        ; Convert units for plotting (if required)
        if (isatt(diag_script_info, "plot_units")) then
          A0 = convert_units(A0, diag_script_info@plot_units)
        end if

        ; Anomaly
        if (anom .eq. "anom") then
          if (ref_start.lt.start_year) then
            ref_start = start_year
          end if
          if ((ref_start.lt.start_year) .or. (ref_end.gt.end_year) .or. \
              (ref_end.lt.ref_start)) then
            error_msg("f", DIAG_SCRIPT, "", \
                      "period for reference years is not properly defined")
          end if
        end if

        ; masking with reference dataset
        if (diag_script_info@ref_mask) then
          ; Determine start/end year
          start_year = info_items[imod]@start_year
          end_year = info_items[imod]@end_year
          A_ref_mask = time_operations(A_ref, start_year, \
                                       end_year, "extract", "", 0)
          A0 = where(A_ref_mask.eq.A_ref_mask@_FillValue, \
                     A_ref_mask@_FillValue, A0)
          delete(A_ref_mask)
        end if

        ; Anomaly
        if (anom .eq. "anom") then
          tmp := time_operations(A0, ref_start, ref_end, "extract", "", 0)
          A0_monavg_ref = time_operations(tmp, ref_start, ref_end, "average", \
                                          "monthlyclim", True)
          delete(tmp)

          ; calculate anomaly for each grid point
          do i = 0, dimsizes(A0&time) - 1
            A0(i, :, :) = A0(i, :, :) - A0_monavg_ref(mod(i, 12), :, :)
          end do

          delete(A0_monavg_ref)

        end if

        ; Calculate time average of dataset
        A0_timavg = time_operations(A0, -1, -1, "average", \
                                    "yearly", True)
        delete(A0)

        ; calculate global mean of complete time series
        procmod = area_operations(A0_timavg, -90., 90., 0., 360., \
                                  "average", True)
        delete(A0_timavg)

        ; Match time coordinate
        date = procmod&year
        idx1 = ind(date(0).eq.model_arr&time)
        idx2 = ind(date(dimsizes(date) - 1).eq.model_arr&time)
        model_arr(imod, idx1:idx2) = (/procmod/)
        if (imod.eq.0) then
          copy_VarAtts(procmod, model_arr)
        end if
        delete(procmod)
        delete(date)

      end do


      ; *************************************
      ; calculate means over ensemble members
      ; *************************************

      ndim := dimsizes(datasetnames)
      ensemble_assign := new(ndim, integer)
      ensemble_size := new(ndim, integer)

      enum = 1
      temp = datasetnames(0)
      do gg = 0, ndim - 1
        if (temp.eq.datasetnames(gg)) then ; does the model name match what's in temp?
          ensemble_assign(gg) = enum ; if so, assign it the same number
        else
          enum = enum + 1              ; if not, assign it the next number
          ensemble_assign(gg) = enum
          temp = datasetnames(gg)
        end if
      end do   
      ensemble_assign@model = datasetnames
      delete(temp)
      ensemble_assign@models = str_join(datasetnames+"/"+ensembles,",")

      model_arr_em := model_arr(:max(ensemble_assign)-1,:)
      model_arr_em = model_arr_em@_FillValue
      datasetnames_em := datasetnames(:max(ensemble_assign)-1)

      do gg = 1,max(ensemble_assign)         ; calculate ensemble means
        wind := ind(ensemble_assign.eq.gg)
        ensemble_size(wind) = dimsizes(wind)
        if (dimsizes(wind).eq.1) then
          model_arr_em(gg-1,:) = (/ model_arr(wind,:) /)
          datasetnames_em(gg-1) = datasetnames(wind)
        else
          model_arr_em(gg-1,:) = (/ dim_avg_n(model_arr(wind,:),0) /)
          datasetnames_em(gg-1) = datasetnames(wind(0))
        end if
      end do


      ; *************************************
      ; calculate multi-model mean
      ; *************************************

      ind_mod := ind(datasetnames_em.ne."MultiModelMean")
      model_arr(mmm_ind, :) = dim_avg_n(model_arr_em(ind_mod,:),0) 


      ; *************************************
      ; calculate 5% and 95% quantiles
      ; *************************************

      q_05 = new(ntime, double)
      q_95 = new(ntime, double)

      do itime = 0, ntime -1
        if (any(.not.ismissing(model_arr(:, itime)))) then
          model_num = num(.not.ismissing(model_arr_em(ind_mod, itime)))
          ind_m := ind(.not.ismissing(model_arr(:, itime)) .and. \
                       datasetnames.ne."MultiModelMean") 
          ind_sort := dim_pqsort(model_arr(ind_m, itime), 1)
          count = 0.
          do imodel = 0, dimsizes(ind_m) - 1
            count = count + 1./(model_num*ensemble_size(ind_m(ind_sort(imodel))))
            if(count .gt. 0.95) then
              break  
            end if
            if(ismissing(q_05(itime)) .and. count .gt. 0.05) then
              q_05(itime) =  model_arr(ind_m(ind_sort(imodel)), itime)
            end if
            q_95(itime) =  model_arr(ind_m(ind_sort(imodel)), itime)
          end do
        end if
      end do

      model_arr_stat&exp(iexp) =  var
      model_arr_stat(iexp, 0, :) = (/model_arr(mmm_ind, :)/)
      model_arr_stat(iexp, 1, :) = (/q_05/)
      model_arr_stat(iexp, 2, :) = (/q_95/)
      
      delete(model_arr)

    end do


    ; *************************************
    ; output to NetCDF
    ; *************************************

    out_path1 = out_path + "tas_" + anom + "_damip_" + header + "_" + \
                ivar + ".nc"
    model_arr_stat@ncdf = out_path
    model_arr_stat@diag_script = DIAG_SCRIPT
    model_arr_stat@var = var0
    ncdf_outfile = ncdf_write(model_arr_stat, out_path1)


    ; *************************************
    ; Plotting
    ; *************************************

    cmap = read_colormap_file("$diag_scripts/shared/plot/rgb/ipcc-ar6_damip_line.rgb")

    res@xyDashPatterns    := (/0, 0, 0, 0/)
    res@xyLineThicknesses := (/3, 3, 3, 3/)
    res@xyLineColors := cmap(:,:)

    ymax = max(model_arr_stat)
    ymin = min(model_arr_stat)
    res@trYMaxF := max((/0.2, ymax + 0.05 * (ymax - ymin)/))
    res@trYMinF := min((/-0.2, ymin - 0.05 * (ymax - ymin)/))

    if (isatt(diag_script_info, "header")) then
      res@tiMainString = diag_script_info@header(ivar)
    else
      res@tiMainString = ""
    end if


    if (isatt(diag_script_info, "plot_units")) then
      units = diag_script_info@plot_units
    else
      units = info_items[0]@units
    end if

    if (ivar .eq. 0) then
      res@tiYAxisString = "Change in Temperature (~F34~0~F~ C)"
    else
      res@tiYAxisString = ""
    end if

    plots(ivar) = gsn_csm_xy(wks, time, model_arr_stat(0:4, 0, :), res)  ; create plot

    res@pmLegendDisplayMode = "Never"

    
    ; ---------------------------------------------------------------------
    ; Add multi model statistics
    ; ---------------------------------------------------------------------
    res_stat = True
    res_stat@gsnXYFillOpacities = 0.1
    cmap = read_colormap_file("$diag_scripts/shared/plot/rgb/ipcc-ar6_damip_shading.rgb")
    copy_VarMeta(res, res_stat)
    delete(res_stat@xyLineColors)
    res_stat@gsnXYFillColors = cmap(3,:)
    res_stat@xyLineColor := cmap(3,:)
    res_stat@xyLineThicknesses := (/1, 1/)
    shading_plot = gsn_csm_xy(wks, time, model_arr_stat(3, 1:2, :), \
                              res_stat)
    overlay(plots(ivar), shading_plot)

    res_stat@gsnXYFillColors = cmap(2,:)
    res_stat@xyLineColor = cmap(2,:)
    shading_plot = gsn_csm_xy(wks, time, model_arr_stat(2, 1:2, :), \
                              res_stat)
    overlay(plots(ivar), shading_plot)

    res_stat@gsnXYFillColors = cmap(1,:)
    res_stat@xyLineColor = cmap(1,:)
    shading_plot = gsn_csm_xy(wks, time, model_arr_stat(1, 1:2, :), \
                              res_stat)
    overlay(plots(ivar), shading_plot)

    res_stat@gsnXYFillColors = cmap(0,:)
    res_stat@xyLineColor = cmap(0,:)
    shading_plot = gsn_csm_xy(wks, time, model_arr_stat(0, 1:2, :), \
                              res_stat)
    overlay(plots(ivar), shading_plot)

    ; ---------------------------------------------------------------------
    ; add reference datasets
    ; ---------------------------------------------------------------------
    res@xyDashPatterns    := (/0/)
    res@xyLineThicknesses := (/3/) ; thicks  ; make 2nd lines thicker
    res@xyLineColors      := "black"  ; change line color

    ref_p = gsn_csm_xy(wks, time, ref_arr, res)  ; create plot
    overlay(plots(ivar), ref_p)


    ; ---------------------------------------------------------------------
    ; Create legend
    ; ---------------------------------------------------------------------
    ;if (ivar .eq. 1 .and. var .eq. "tas_hist-aer_global") then
    ;  res_text                    = True                  ; text mods desired
    ;  res_text@txFontHeightF      = 0.02                 ; change text size
    ;  res_text@txJust             = "CenterLeft"          ; text justification

    ;  res_lines                   = True                  ; polyline mods desired
    ;  res_lines@gsLineDashPattern = 0.                    ; solid line
    ;  res_lines@gsLineThicknessF  = 4.                    ; line thicker
    ;  cmap = read_colormap_file("$diag_scripts/shared/plot/rgb/ipcc-ar6_damip_line.rgb")
    ;  res_lines@gsLineColor       = cmap(0,:)                 ; line color
    ;  xx = (/1880,1895/)
    ;  yy = (/3.0,3.0/)
    ;  dum1 = gsn_add_polyline(wks,plots(ivar),xx,yy,res_lines)
    ;  dum2 = gsn_add_text(wks,plots(ivar),"Anthropogenic + Natural",\
    ;                      1905, yy(0),res_text)
    ;  
    ;  yy = (/2.5,2.5/)
    ;  res_lines@gsLineColor       = cmap(1,:)
    ;  dum3 = gsn_add_polyline(wks,plots(ivar),xx,yy,res_lines)                ; add polyline
    ;  dum4 = gsn_add_text(wks,plots(ivar),"Natural",1905,yy(0),res_text)       ; add text
    ;  
    ;  yy = (/2.0,2.0/)
    ;  res_lines@gsLineColor       = cmap(2,:)
    ;  dum5 = gsn_add_polyline(wks,plots(ivar),xx,yy,res_lines)                ; add polyline
    ;  dum6 = gsn_add_text(wks,plots(ivar),"Greenhouse Gases",1905,yy(0),res_text)       ; add text
    ;  
    ;  yy = (/1.5,1.5/)
    ;  res_lines@gsLineColor       = cmap(3,:)
    ;  dum7 = gsn_add_polyline(wks,plots(ivar),xx,yy,res_lines)                ; add polyline
    ;  dum8 = gsn_add_text(wks,plots(ivar),"Aerosols",1905,yy(0),res_text)       ; add text
    ;  
    ;  yy = (/1.0,1.0/)
    ;  res_lines@gsLineColor       := "black"                                ; change to black
    ;  dum9 = gsn_add_polyline(wks,plots(ivar),xx,yy,res_lines)                ; add polyline
    ;  dum10 = gsn_add_text(wks,plots(ivar),"Observations",1905,yy(0),res_text) ; add text
    ;end if


  end do


  ; Plotting panels

  pres = True

  pres@gsnPanelMainString = "Anomaly of Near-Surface Air Temperature"

  pres@gsnPanelRowSpec = True
  pres@gsnPanelCenter  = False

  ;pres@gsnPanelFigureStrings = (/"a)","b)","c)","d)","e)","f)", "g)", "h)", "i)", "j)"/)
  pres@gsnPanelFigureStrings = diag_script_info@panel_figure_strings
  pres@gsnPanelFigureStringsPerimOn = False
  pres@gsnPanelFigureStringsBackgroundFillColor ="transparent"
  pres@gsnPanelFigureStringsJust = "TopLeft"
  pres@gsnPanelFigureStringsFontHeightF = 0.014

  gsn_panel(wks, plots, (/3,3,3,1/), pres)

  log_info(" wrote " + outfile + "." + file_type)


  ; ***************************************
  ; add meta data to plot (for reporting)
  ; ***************************************

  domain = "global"

  if (anom .eq. "anom") then
    statistics = "anomaly"
  else
    statistics = "mean"
  end if

  caption = "Time series of the " + statistics + " for near-surface " \
    + "air temperature."
  contrib_authors = (/"bock_lisa"/)

  ; Call provenance logger
  log_provenance(ncdf_outfile, \
                 outfile + "." + file_type, \
                 caption, \
                 statistics, \
                 domain, \
                 "times", \
                 contrib_authors, \
                 (/"flato13ipcc", "jones13jgr"/), \
                 metadata_att_as_array(input_file_info, "filename"))

  leave_msg(DIAG_SCRIPT, "")

end