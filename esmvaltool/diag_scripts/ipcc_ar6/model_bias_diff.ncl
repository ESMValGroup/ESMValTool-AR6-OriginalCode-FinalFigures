; MODEL_BIAS
; ############################################################################
; Author: Axel Lauer (DLR, Germany), Lisa Bock (DLR, Germany)
; PROJECT-NAME IPCC-AR6
; ############################################################################
; Description
;   Calculates the multi-model mean bias, absolute difference, relative 
;   difference and root mean square difference of annual mean 2-d cloud 
;   variables compared with a reference dataset (observations).
;
; Required diag_script_info attributes (diagnostic specific)
;   none
;
; Optional diag_script_info attributes (diagnostic specific)
;   projection:    map projection, e.g., Mollweide, Mercator
;
; Required variable_info attributes (variable specific)
;   reference_dataset: name of reference datatset
;   modelgroup_1: names of models in group 1
;   modelgroup_2: names of models in group 2
;
; Optional variable_info attributes (variable specific)
;   long_name: description of variable
;
; Caveats
;   none
;
; Modification history
;   20190312-A_bock_ls: added calculation of RMSD and IPCC AR6 color scheme
;   20190222-A_laue_ax: added output of provenance (v2.0)
;   20181119-A_laue_ax: adapted code to multi-variable capable framework
;   20180923-A_laue_ax: added writing of results to netcdf
;   20180914-A_laue_ax: code rewritten for ESMValTool v2.0
;   20170620-A_laue_ax: added tags for reporting
;   20160901-A_laue_ax: added regridding option 1 deg x 1 deg
;   20151027-A_laue_ax: moved call to 'write_references' to the beginning
;                       of the code
;   20150428-A-laue_ax: written.
;
; ############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/scaling.ncl"
load "$diag_scripts/shared/statistics.ncl"
load "$diag_scripts/shared/plot/style.ncl"
load "$diag_scripts/shared/plot/contour_maps.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")

  var0 = variable_info[0]@short_name
  info0 = select_metadata_by_name(input_file_info, var0)
  dim_MOD = ListCount(info0)
  if (isatt(variable_info[0], "reference_dataset")) then
    refname = variable_info[0]@reference_dataset
  end if
  names = metadata_att_as_array(info0, "dataset")
  infiles = metadata_att_as_array(info0, "filename")

  log_info("++++++++++++++++++++++++++++++++++++++++++")
  log_info(DIAG_SCRIPT + " (var: " + var0 + ")")
  log_info("++++++++++++++++++++++++++++++++++++++++++")

  ; Check for required diag_script_info attributes
  exit_if_missing_atts(diag_script_info, (/"groups", "modelgroup_1", "modelgroup_2"/))
 
  ; Set default values for non-required diag_script_info attributes
  set_default_att(diag_script_info, "stippandhatch", False)
  set_default_att(diag_script_info, "projection", "Robinson")

  ; make sure path for (mandatory) netcdf output exists

  work_dir = config_user_info@work_dir + "/"
  ; Create work dir
  system("mkdir -p " + work_dir)

end

begin
  ; ========================================================================
  ; ========================== initialization ==============================
  ; ========================================================================

  ; check for reference dataset definition

  if (.not.(isvar("refname"))) then
    error_msg("f", DIAG_SCRIPT, "", "no reference dataset defined in recipe")
  end if

  stippandhatch = diag_script_info@stippandhatch

  ; get reference dataset

  ref_ind = ind(names .eq. refname)
  if (ismissing(ref_ind)) then
    error_msg("f", DIAG_SCRIPT, "", "reference dataset (" \
              + refname + ") is missing")
  end if

  ; basename of diag_script
  diag_script_base = basename(DIAG_SCRIPT)

  ; ========================================================================
  ; =========================== Read data ==================================
  ; ========================================================================

  ; read data

  A0 = read_data(info0[ref_ind])
  refdata = time_operations(A0, -1, -1, "average", "annualclim", True)
  if (dimsizes(A0&time).gt.12) then
    ref_mon = time_operations(A0, -1, -1, "average", "monthlyclim", True)
  else
    ref_mon = A0
  end if
  delete(A0)

  dim = dimsizes(refdata)
  dim_MOD1 = dimsizes(diag_script_info@modelgroup_1)
  dim_MOD2 = dimsizes(diag_script_info@modelgroup_2)
  data1 = new((/dim_MOD1, dim(0), dim(1)/) ,float)
  data2 = new((/dim_MOD2, dim(0), dim(1)/) ,float)
  data_mon1 = new((/dim_MOD1, 12, dim(0), dim(1)/) ,float)
  data_mon2 = new((/dim_MOD2, 12, dim(0), dim(1)/) ,float)
  data_ivar1 = new((/dim_MOD1, dim(0), dim(1)/) ,float)
  data_ivar2 = new((/dim_MOD2, dim(0), dim(1)/) ,float)

  do imod = 0, dim_MOD - 1

    if(imod .ne. ref_ind) then
      log_info("processing " + names(imod))

      if (isvar("A0")) then
        delete(A0)
      end if

      if (any(diag_script_info@modelgroup_1 .eq. names(imod))) then

        ii = ind(diag_script_info@modelgroup_1 .eq. names(imod))

        A0 = read_data(info0[imod])

        data1(ii,:,:) = time_operations(A0, 1995, 2014, "average", "annualclim", True)
        data_mon1(ii,:,:,:) = time_operations(A0, -1, -1, "average", "monthlyclim", True)

        if (stippandhatch) then
          tmp = time_operations(A0, info0[imod]@start_year, \
                                info0[imod]@start_year + 29, \
                                "average", "yearly", True)
          tmp = dtrend_quadratic_msg_n(tmp, False, False, 0)
          data_ivar1(ii, :, :) = dim_stddev_n(tmp, 0)
        end if

      elseif (any(diag_script_info@modelgroup_2 .eq. names(imod))) 

        ii = ind(diag_script_info@modelgroup_2 .eq. names(imod))

        A0 = read_data(info0[imod])

        data2(ii,:,:) = time_operations(A0, 1995, 2014, "average", "annualclim", True)
        data_mon2(ii,:,:,:) = time_operations(A0, -1, -1, "average", "monthlyclim", True)

        if (stippandhatch) then
          tmp = time_operations(A0, info0[imod]@start_year, \
                                info0[imod]@start_year + 29, \
                                "average", "yearly", True)
          tmp = dtrend_quadratic_msg_n(tmp, False, False, 0)
          data_ivar2(ii, :, :) = dim_stddev_n(tmp, 0)
        end if

      else

        error_msg("f", DIAG_SCRIPT, "", "model " + names(imod) + "is not part of one predefined modelgroups")

      end if

      delete(A0)

    end if

  end do

  ; -----------------------------------------------------------
  ; read precalculated natural variability for multi-model-mean
  ; -----------------------------------------------------------

  if (stippandhatch) then
;    log_info("Read precalculated natural variability for multi-model")
;
;    input_dir1 = str_match(diag_script_info@input_files, diag_script_info@groups(0)) 
;    natvar_file1 = addfile(input_dir1 + "/" + "IAV_mmm_piControl_" + \
;                          var0 + "_annualclim.nc", "r")
;    natvar_mmm1 = natvar_file1->iav
;    tmp = natvar_mmm1
;    delete(natvar_mmm1)
;    natvar_mmm1 = rm_single_dims(tmp)
;    delete(tmp)
;
;    input_dir2 = str_match(diag_script_info@input_files, diag_script_info@groups(1)) 
;    natvar_file2 = addfile(input_dir2 + "/" + "IAV_mmm_piControl_" + \
;                          var0 + "_annualclim.nc", "r")
;    natvar_mmm2 = natvar_file2->iav
;    tmp = natvar_mmm2
;    delete(natvar_mmm2)
;    natvar_mmm2 = rm_single_dims(tmp)
;    delete(tmp)
  end if

  ; -----------------------------------------------------------

  ; convert units for plotting
  if (var0.eq."pr") then
    ; kg m-2 s-1 --> mm day-1
    refdata = convert_units(refdata, "mm/day")
    data1 = convert_units(data1, "mm/day")
    data2 = convert_units(data2, "mm/day")
    ref_mon = convert_units(ref_mon, "mm/day")
    data_mon1 = convert_units(data_mon1, "mm/day")
    data_mon2 = convert_units(data_mon2, "mm/day")
  elseif (var0.eq."tas") then
    data1 = convert_units(data1, "degC")
    data2 = convert_units(data2, "degC")
    data_mon1 = convert_units(data_mon1, "degC")
    data_mon2 = convert_units(data_mon2, "degC")
    refdata = convert_units(refdata, "degC")
  elseif (var0.eq."tos" .or. var0.eq."thetao") then
    if (data1@units .eq. "K") then
      data1 = convert_units(data1, "degC")
      data2 = convert_units(data2, "degC")
      data_mon1 = convert_units(data_mon1, "degC")
      data_mon2 = convert_units(data_mon2, "degC")
    end if
    if (refdata@units .eq. "K") then
      refdata = convert_units(refdata, "degC")
      ref_mon = convert_units(ref_mon, "degC")
    end if
  elseif (var0.eq."so") then
    refdata = refdata * 1000.
  end if


  ; ========================================================================
  ; =========================== Calculations ===============================
  ; ========================================================================

  ; calculate mmm of each group
  mmdata1 = dim_avg_n_Wrap(data1, 0)
  mmdata2 = dim_avg_n_Wrap(data2, 0)

  ; differences between multi-model mean and reference data set
  ; (multi-model bias)

  diff1 = mmdata1 - refdata
  diff2 = mmdata2 - refdata
  mmdata1@diag_script = DIAG_SCRIPT
  copy_VarMeta(mmdata1, diff1)

  if any(var0 .eq. (/"tas","tos","thetao"/)) then
    diff1@units = "~F34~0~F~ C"
    diff1@res_cnLevels = ispan(-6, 6, 1)
  elseif (var0 .eq. "pr") then
    diff1@res_cnLevels = ispan(-30, 30, 5) * 0.1
  elseif (var0 .eq. "so") then
    diff1@units = "0.001"
    diff1@res_cnLevels = ispan(-30, 30, 5) * 0.1
    ;diff@res_cnLevels = ispan(-8, 8, 1)
  end if
  copy_VarMeta(mmdata1, mmdata2)
  copy_VarMeta(diff1, diff2)

  diff1_i    = new((/dim_MOD1, dim(0), dim(1)/) ,float)
  diff2_i    = new((/dim_MOD2, dim(0), dim(1)/) ,float)
  ivar1_i    = new((/dim_MOD1, dim(0), dim(1)/) ,float)
  ivar2_i    = new((/dim_MOD2, dim(0), dim(1)/) ,float)
  ;absdiff_i = new((/dim_MOD-2, dim(0), dim(1)/) ,float)
  ;rmsdiff_i = new((/dim_MOD-2, dim(0), dim(1)/) ,float)
  ;reldiff_i = new((/dim_MOD-2, dim(0), dim(1)/) ,float)

  if (stippandhatch .or. plot_rms_diff .or. plot_abs_diff .or. plot_rel_diff) then

    do imod=0,dim_MOD-1

      if(imod .ne. ref_ind) then
        log_info("processing " + names(imod))
 
        if (any(diag_script_info@modelgroup_1 .eq. names(imod))) then 
       
          ii = ind(diag_script_info@modelgroup_1 .eq. names(imod))

          ; difference
          diff1_i(ii,:,:) = data1(ii,:,:) - refdata

        elseif (any(diag_script_info@modelgroup_2 .eq. names(imod)))

          ii = ind(diag_script_info@modelgroup_2 .eq. names(imod))

          ; difference
          diff2_i(ii,:,:) = data2(ii,:,:) - refdata

        end if

        ;;absolute difference
        ;absdiff_i(ii,:,:) = abs(diff_i(ii,:,:))

        ;; root mean square error
        ;rmsdiff_i(ii,:,:) = calculate_metric(data_mon(ii,:,:,:), ref_mon, "RMSDxy")

        ;; relative difference
        ;; replace "epsilon" values with missing value
        ;datam = where(abs(data).lt.1.e-6, data@_FillValue, data)
        ;reldiff_i(ii,:,:) = 100.0 * (diff_i(ii,:,:) / datam(ii,:,:))

      end if

    end do

  end if

  

  ;; multi-model mean of absolute difference between models and reference
  ;; data set 

  ;absdiff = dim_avg_n_Wrap(absdiff_i, 0)
  ;copy_VarMeta(diff, absdiff)

  ;rmsdiff = dim_avg_n_Wrap(rmsdiff_i, 0)
  ;copy_VarMeta(diff, rmsdiff)

  ;reldiff = dim_avg_n_Wrap(reldiff_i, 0)
  ;copy_VarMeta(diff, reldiff) 

  ;rmsd = calculate_metric(refdata, mmdata, "RMSD")
  ;bias = area_operations(diff, -90., 90., 0., 360., "average", True)
  ;mean = area_operations(mmdata, -90., 90., 0., 360., "average", True)

  ; ----------------------------------------------
  ; Determine sigma and sign for significance
  ; ----------------------------------------------
  if (stippandhatch) then
    ; first panel
    var_pos_signif = dim_num_n(where(diff1_i .gt. 0., 1., \
                                     diff1@_FillValue) .eq. 1., 0)
    var_neg_signif = dim_num_n(where(diff1_i .lt. 0., 1., \
                                     diff1@_FillValue) .eq. 1., 0)
    var_pos_signif_tmp = \
      var_pos_signif / where((var_pos_signif + var_neg_signif) \
                             .ne. 0., var_pos_signif + \
                             var_neg_signif, diff1@_FillValue)
    var_neg_signif_tmp = \
      var_neg_signif / where((var_pos_signif + var_neg_signif) \
                             .ne. 0., var_pos_signif + \
                             var_neg_signif, diff1@_FillValue)
    signif_and_sign = where(var_pos_signif_tmp .ge. 0.8 .or. \
                            var_neg_signif_tmp .ge. 0.8, 1., 0.)
;    signif_and_sign = where(ismissing(signif_and_sign), 0.,\
;                            signif_and_sign)
    ; variablity threshold: sqrt(2)*1.645
    natvar_mmm1 = rm_single_dims(dim_avg_n(data_ivar1(:, :, :), 0) * sqrt(2.))
    var_thres = abs(1.645 * natvar_mmm1) 
    tmp := new(dimsizes(diff1_i), "float")
    tmp = 0
    do imod = 0, dim_MOD1 - 1
      tmp(imod, :, :) = where(abs(diff1_i(imod, :, :)) .gt. var_thres(:, :), 1, 0)
    end do
    var_pos_signif := dim_sum_n(tmp, 0)
    delete(tmp)
    var_pos_signif_tmp := var_pos_signif / (dim_MOD1 - 2)
    sigma_and_sign = where(var_pos_signif_tmp .ge. 0.66 , 1., 0.)

    to_plot_not_robust_1 = where(sigma_and_sign .eq. 0, 1., 0.)
    ;to_plot_not_robust_1 = var_pos_signif_tmp

    to_plot_confl_sign_1 = where(signif_and_sign .eq. 0, sigma_and_sign, 0.)
    ;to_plot_confl_sign_1 = signif_and_sign

    copy_VarMeta(diff1, natvar_mmm1)
    copy_VarMeta(diff1, to_plot_confl_sign_1)
    copy_VarMeta(diff1, to_plot_not_robust_1)

    ; second panel
    var_pos_signif = dim_num_n(where(diff1_i .gt. 0., 1., \
                                     diff1@_FillValue) .eq. 1., 0)
    var_neg_signif = dim_num_n(where(diff1_i .lt. 0., 1., \
                                     diff1@_FillValue) .eq. 1., 0)
    var_pos_signif_tmp = \
      var_pos_signif / where((var_pos_signif + var_neg_signif) \
                             .ne. 0., var_pos_signif + \
                             var_neg_signif, diff1@_FillValue)
    var_neg_signif_tmp = \
      var_neg_signif / where((var_pos_signif + var_neg_signif) \
                             .ne. 0., var_pos_signif + \
                             var_neg_signif, diff1@_FillValue)
    signif_and_sign = where(var_pos_signif_tmp .ge. 0.75 .or. \
                            var_neg_signif_tmp .ge. 0.75, 1., 0.)
    signif_and_sign = where(ismissing(signif_and_sign), 0.,\
                            signif_and_sign)
    ; variablity threshold: sqrt(2)*1.645
    natvar_mmm2 = rm_single_dims(dim_avg_n(data_ivar2(:, :, :), 0) * sqrt(2.))
    var_thres = abs(1.645 * natvar_mmm2) 
    tmp := new(dimsizes(diff1_i), "float")
    tmp = 0
    do imod = 0, dim_MOD2 - 1
      tmp(imod, :, :) = where(abs(diff1_i(imod, :, :)) .gt. var_thres(:, :), 1, 0)
    end do
    var_pos_signif := dim_sum_n(tmp, 0)
    delete(tmp)
    var_pos_signif_tmp := var_pos_signif / (dim_MOD2 - 2)
    sigma_and_sign = where(var_pos_signif_tmp .ge. 0.66 , 1., 0.)

    to_plot_not_robust_2 = where(sigma_and_sign .eq. 0, 1., 0.)

    to_plot_confl_sign_2 = where(signif_and_sign .eq. 0, sigma_and_sign, 0.)

    copy_VarMeta(diff1, natvar_mmm2)
    copy_VarMeta(diff2, to_plot_confl_sign_2)
    copy_VarMeta(diff2, to_plot_not_robust_2)

    ; improvement
    diff1_i = abs(diff1_i) - abs(diff2_i)    

    improve_signif = dim_num_n(where(diff1_i .lt. 0., 1., \
                                     diff1@_FillValue) .eq. 1., 0)
    not_improve_signif = dim_num_n(where(diff1_i .gt. 0., 1., \
                                     diff1@_FillValue) .eq. 1., 0)
    improve_signif_tmp = \
      improve_signif / where((improve_signif + not_improve_signif) \
                             .ne. 0., improve_signif + \
                             not_improve_signif, diff1@_FillValue)
    to_plot_improve = where(improve_signif_tmp .ge. 0.75 .or. \
                            improve_signif_tmp .ge. 0.75, 0., 1.)

    copy_VarMeta(diff1, to_plot_improve)

  end if


  ; ========================================================================
  ; ============================= plotting =================================
  ; ========================================================================

  ;climofiles = new(2, string)
  ;climofiles(0) = infiles(mm_ind)
  ;climofiles(1) = infiles(ref_ind)

  diff1@res_gsnMaximize  = True  ; use full page for the plot
  diff1@res_cnFillOn  = True  ; color plot desired
  diff1@res_cnLineLabelsOn  = False  ; contour lines
  diff1@res_cnLinesOn  = False
  diff1@res_tiMainOn  = True
  diff1@res_gsnLeftStringFontHeightF = 0.015
  diff1@res_gsnRightStringFontHeightF = 0.015
  diff1@res_cnLevelSelectionMode  = "ExplicitLevels"
  diff1@res_mpOutlineOn  = True
  if (.not.isatt(diff1, "res_cnLevels")) then
    diff1@res_cnLevels = fspan(min(diff1), max(diff1), 20)
  end if
  diff1@res_mpFillOn  = False
  diff1@res_lbLabelBarOn  = True
  diff1@res_gsnRightString  = ""
  diff1@res_gsnLeftString   = ""
  diff1@res_mpFillDrawOrder  = "PostDraw"  ; draw map fill last
  diff1@res_cnMissingValFillColor  = "Gray"
  diff1@res_tmYLLabelsOn  = False
  diff1@res_tmYLOn  = False
  diff1@res_tmYRLabelsOn  = False
  diff1@res_tmYROn  = False
  diff1@res_tmXBLabelsOn  = False
  diff1@res_tmXBOn  = False
  diff1@res_tmXTLabelsOn  = False
  diff1@res_tmXTOn  = False
  diff1@res_cnInfoLabelOn  = False  ; turn off cn info label
  diff1@res_mpProjection  = diag_script_info@projection

  diff1@var = var0  ; Overwrite existing entry
  if (isatt(variable_info[0], "long_name")) then
    diff1@var_long_name = variable_info[0]@long_name
  end if
  diff1@var_units  = diff1@units

  plots = new((/5/), graphic)

  ; --------------------------------------------------------------------
  ; plot contour map

  diff1@res_gsnDraw  = False  ; Do not draw yet
  diff1@res_gsnFrame  = False  ; Don't advance frame.
  diff1@res_mpPerimOn  = False

  diff1@res_mpGeophysicalLineColor = "gray42"
  diff1@res_mpPerimOn    = False
  diff1@res_mpGridLineColor   = "transparent"
  diff1@res_mpGridAndLimbOn   = True
  diff1@res_mpOutlineOn = True

  diff1@res_lbTitleString  = "(" + diff1@units + ")"
  diff1@res_lbTitlePosition  = "Bottom"

  diff1@res_lbLabelFontHeightF  = 0.014
  diff1@res_lbTopMarginF  = 0.1
  diff1@res_lbTitleFontHeightF  = 0.014

  diff1@res_lbBoxEndCapStyle = "TriangleBothEnds"
  diff1@res_lbBoxLineColor = "gray70"

  diff1@res_tiMainFontHeightF  = 0.016

  diff1@res_gsnRightStringParallelPosF = 0.8

  if (var0.eq."tas") then
    ;;mmdata@res_tiMainString = "Near-Surface Temperature ~C~   Multi-Model Mean"
    ;mmdata@res_cnLevels = ispan(-35, 35, 5)

    pal = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
                             + "ipcc-ar6_temperature_div.rgb")
    ;mmdata@res_cnFillPalette = pal
    ;diff@res_tiMainString = "Near-Surface Temperature ~C~  Multi-Model Mean Bias"
    diff1@res_cnFillPalette = pal
    ;if (plot_abs_diff) then
    ;  pal3 = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
    ;                            + "ipcc-ar6_temperature_div.rgb")
    ;end if
    ;if (plot_rel_diff) then
    ;  pal4 = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
    ;                            + "ipcc-ar6_temperature_10.rgb")
    ;end if
    ;if (plot_rms_diff) then
    ;  pal5 = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
    ;                            + "ipcc-ar6_temperature_seq.rgb")
    ;end if
  elseif (var0.eq."pr") then
    ;mmdata@res_tiMainString = "Precipitation ~C~Multi Model Mean"
    ;mmdata@res_cnLevels = ispan(1, 10, 1)

    pal = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
                             + "ipcc-ar6_precipitation_div.rgb")
    ;mmdata@res_cnFillPalette = pal
    ;diff@res_tiMainString = "Precipitation ~C~Multi Model Mean Bias"
    pal2 = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
                              + "ipcc-ar6_precipitation_div.rgb")
    diff1@res_cnFillPalette = pal2
    ;if (plot_abs_diff) then
    ;  pal3 = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
    ;                            + "ipcc-ar6_precipitation_10.rgb")
    ;end if
    ;if (plot_rel_diff) then
    ;  pal4 = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
    ;                            + "ipcc-ar6_precipitation_10.rgb")
    ;end if
    ;if (plot_rms_diff) then
    ;  pal5 = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
    ;                            + "ipcc-ar6_precipitation_div.rgb")
    ;end if
  elseif (var0.eq."tos" .or. var0.eq."thetao") then
    ;mmdata@res_tiMainString = "Sea Surface Temperature ~C~Multi Model Mean"
    ;mmdata@res_cnLevels = ispan(-30, 30, 5)

    pal = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
                             + "ipcc-ar6_temperature_div.rgb")
    ;mmdata@res_cnFillPalette = pal
    ;diff@res_tiMainString = "Sea Surface Temperature ~C~Multi Model Mean Bias"
    diff1@res_cnFillPalette = pal
    ;if (plot_abs_diff) then
    ;  pal3 = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
    ;                            + "ipcc-ar6_temperature_10.rgb")
    ;end if
    ;if (plot_rel_diff) then
    ;  pal4 = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
    ;                            + "ipcc-ar6_temperature_10.rgb")
    ;end if
    ;if (plot_rms_diff) then
    ;  pal5 = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
    ;                            + "ipcc-ar6_temperature_seq.rgb")
    ;end if
  elseif (var0.eq."so") then
    ;mmdata@res_tiMainString = "Sea Surface Salinity ~C~Multi Model Mean"
    ;mmdata@res_cnLevels = ispan(310, 370, 5) * 0.1
    ;mmdata@res_cnLevels = ispan(26, 37, 1)

    ;pal = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
    ;                         + "ipcc-ar6_misc_seq_1.rgb")
    ;mmdata@res_cnFillPalette = pal
    ;diff@res_tiMainString = "Sea Surface Salinity ~C~Multi Model Mean Bias"
    pal2 = read_colormap_file("$diag_scripts/shared/plot/rgb/" \
                              + "ipcc-ar6_misc_div.rgb")
    diff1@res_cnFillPalette = pal2
  ;else
  ;  mmdata@res_cnLevels = fspan(min(mmdata), max(mmdata), 20)
  end if

  copy_VarMeta(diff1, diff2)

  if(isatt(diag_script_info, "caption")) then
    diff1@res_tiMainString = diag_script_info@caption(0)
    diff2@res_tiMainString = diag_script_info@caption(1)
  else
    diff1@res_tiMainString = "Multi-Model Mean Bias 1"
    diff2@res_tiMainString = "Multi-Model Mean Bias 2"
  end if

  plotsperline = (/2, 0/)
  plotind = (/0, 1/)  ; mmm and mean bias are always plotted

  ;; absolute differences

  ;if (plot_abs_diff) then
  ;  copy_VarMeta(diff, absdiff)

  ;  absdiff@res_tiMainString = "Multi Model Mean of Absolute Error"

  ;  iadd = 2
  ;  itmp = array_append_record(plotind, iadd, 0)
  ;  delete(plotind)
  ;  plotind = itmp
  ;  delete(itmp)
  ;  plotsperline(1) = plotsperline(1) + 1
  ;end if

  ;; root mean square differences

  ;if (plot_rms_diff) then
  ;  copy_VarMeta(diff, rmsdiff)
  ;  if (isvar("pal5")) then
  ;    rmsdiff@res_cnFillPalette = pal5
  ;    delete(rmsdiff@res_cnLevels)
  ;    rmsdiff@res_cnLevels = ispan(1,12,1) * 0.5
  ;    rmsdiff@res_lbBoxEndCapStyle = "TriangleHighEnd"
  ;  end if

  ;  if(isatt(diag_script_info, "rmsd_caption")) then
  ;     rmsdiff@res_tiMainString = diag_script_info@rmsd_caption
  ;  else
  ;    rmsdiff@res_tiMainString = "Multi Model Mean of Root Mean Square Error"
  ;  end if

  ;  iadd = 3
  ;  itmp = array_append_record(plotind, iadd, 0)
  ;  delete(plotind)
  ;  plotind = itmp
  ;  delete(itmp)
  ;  plotsperline(1) = plotsperline(1) + 1
  ;end if

  ;; relative differences

  ;if (plot_rel_diff) then
  ;  copy_VarMeta(diff, reldiff)
  ;  delete(reldiff@res_cnLevels)
  ;  reldiff@res_cnLevels = fspan(-80.0, 80.0, 9)
  ;  reldiff@res_tiMainString = "Multi-Model Mean of Relative Error"
  ;  reldiff@units = "%"
  ;  reldiff@res_lbTitleString = "(" + reldiff@units + ")"
  ;  if (isvar("pal4")) then
  ;    delete(reldiff@res_cnFillColors)
  ;    reldiff@res_cnFillColors = pal4
  ;  end if

  ;  iadd = 4
  ;  itmp = array_append_record(plotind, iadd, 0)
  ;  delete(plotind)
  ;  plotind = itmp
  ;  delete(itmp)
  ;  plotsperline(1) = plotsperline(1) + 1
  ;end if

  ;; add global mean, bias and rmsd value
  ;mmdata@res_gsnRightString = "mean = " + sprintf("%6.3f", mean)
  ;diff@res_gsnRightString = "mean = " + sprintf("%6.3f", bias)
  ;rmsdiff@res_gsnRightString = "mean = " + sprintf("%6.3f", rmsd)
 

  ; panelling resources

  pres  = True
  pres@gsnPanelCenter  = False
  pres@gsnPanelRowSpec = True  ; tell panel what order to plot
  pres@gsnPanelYWhiteSpacePercent = 5
  pres@gsnPanelXWhiteSpacePercent = 5
  if (isatt(diag_script_info, "panel_num")) then
    pres@gsnPanelFigureStrings = diag_script_info@panel_num
  else
    pres@gsnPanelFigureStrings = (/"a)","b)","c)","d)"/)
  end if
  pres@gsnPanelFigureStringsPerimOn = False
  pres@gsnPanelFigureStringsBackgroundFillColor ="transparent"
  pres@gsnPanelFigureStringsJust = "TopLeft"
  pres@gsnPanelFigureStringsFontHeightF = 0.016

  if (isatt(diag_script_info, "panel_title")) then
    pres@gsnPanelMainString = diag_script_info@panel_title
  end if

  ; --------------------------------------------------------------------
  ; create workspace

  if (isvar("wks")) then
    delete(wks)
  end if

  wks = get_wks("dummy_for_wks", DIAG_SCRIPT, "model_bias_diff_" + var0 \
                + "_annualclim_" + input_file_info[0]@project )

  plotfile = wks@fullname

  ;plots(0) = contour_map(wks, mmdata, var0)
  plots(0) = contour_map(wks, diff1, var0)
  plots(1) = contour_map(wks, diff2, var0)
  ;if (plot_abs_diff) then
  ;  plots(2) = contour_map(wks, absdiff, var0)
  ;end if
  ;if (plot_rms_diff) then
  ;  plots(3) = contour_map(wks, rmsdiff, var0)
  ;end if
  ;if (plot_rel_diff) then
  ;  plots(4) = contour_map(wks, reldiff, var0)
  ;end if

  ; add significance
  if (stippandhatch) then
    ; resources dots
    ress = True
    ress@gsnDraw = False
    ress@gsnFrame = False
    ress@gsnAddCyclic = True
    ress@cnLinesOn = False
    ress@cnLineLabelsOn = False
    ress@cnLevelSelectionMode = "ExplicitLevels"
    ress@cnLevels = (/.3, 1.3/)
    ress@cnInfoLabelOn = False
    ress@tiMainOn = False
    ress@cnFillOn = True
    ress@cnFillMode = "AreaFill"
    ress@cnMonoFillPattern = False
    ress@cnFillPatterns = (/-1, 6, 6/)
    ress@cnMonoFillColor = True
    ress@lbLabelBarOn = False
    ress@gsnLeftString = ""
    ress@gsnRightString = ""
    ress@cnFillDotSizeF       = 0.001
    ress@cnFillScaleF        = 1.        ; add extra density

    ; plot cross-hatching
    if (max(to_plot_confl_sign_1(:, :)) .gt. \
        min(to_plot_confl_sign_1(:, :))) then
      plot1 = gsn_csm_contour(wks, to_plot_confl_sign_1(:, :), ress)
      overlay(plots(0), plot1)
    end if
    if (max(to_plot_confl_sign_2(:, :)) .gt. \
        min(to_plot_confl_sign_2(:, :))) then
      plot2 = gsn_csm_contour(wks, to_plot_confl_sign_2(:, :), ress)
      overlay(plots(1), plot2)
    end if

    ; resources hatching
    resb = True
    resb@gsnDraw = False
    resb@gsnFrame = False
    resb@gsnAddCyclic = True
    resb@cnLinesOn = False
    resb@cnLineLabelsOn = False
    resb@cnLevelSelectionMode = "ExplicitLevels"
    resb@cnLevels = (/.3, 1.3/)
    resb@cnInfoLabelOn = False
    resb@tiMainOn = False
    resb@cnFillOn = True
    resb@cnFillMode = "AreaFill"
    resb@cnRasterSmoothingOn = False
    resb@cnMonoFillPattern = False
    resb@cnFillPatterns = (/-1, 4, 4/)
    resb@cnMonoFillColor = True
    resb@lbLabelBarOn = False
    resb@gsnLeftString = ""
    resb@gsnRightString = ""
    resb@cnFillScaleF = 0.5

    ; plot hatching
    if (max(to_plot_not_robust_1(:, :)) .gt. \
        min(to_plot_not_robust_1(:, :))) then
      plot3 = gsn_csm_contour(wks, to_plot_not_robust_1(:, :), resb)
      overlay(plots(0), plot3)
    end if
    if (max(to_plot_not_robust_2(:, :)) .gt. \
        min(to_plot_not_robust_2(:, :))) then
      plot4 = gsn_csm_contour(wks, to_plot_not_robust_2(:, :), resb)
      overlay(plots(1), plot4)
    end if

    ; mark improvements in panel 0
    ;resb@cnFillPatterns = (/-1, 3, 3/)
    resb@cnFillPatterns = (/-1, 17, 17/)
    resb@cnFillScaleF = 1.
    ; plot stiplling
    if all(to_plot_improve(:, :) .eq. 1) then
      to_plot_improve(0, 0) = 0
    end if
    if (max(to_plot_improve(:, :)) .gt. \
        min(to_plot_improve(:, :))) then
      plot5 = gsn_csm_contour(wks, to_plot_improve(:, :), resb)
      overlay(plots(0), plot5)
    end if


  end if

  gsn_panel(wks, plots(plotind), plotsperline, pres)

  ; add meta data to plot (for reporting)

  caption = "Multi model values, from top left to bottom right: " \
            + "mean, bias"
  ;if (plot_abs_diff) then
  ;  caption = caption + ", absolute error"
  ;end if
  ;if (plot_rms_diff) then
  ;  caption = caption + ", root mean square error"
  ;end if
  ;if (plot_rel_diff) then
  ;  caption = caption + ", relative error"
  ;end if

  ; ###########################################
  ; # output to netCDF                        #
  ; ###########################################

  nc_filename = work_dir + "model_bias_diff_" + var0 + "_annualclim_" \
                + input_file_info[0]@project + ".nc"
  nc_filename@existing = "overwrite"

  ;mmdata@var = var0 + "_mean"
  ;mmdata@long_name = var0 + " (multi-model mean)"
  ;nc_outfile = ncdf_write(mmdata, nc_filename)

  ;nc_filename@existing = "append"

  ;refdata@var = var0 + "_ref"
  ;refdata@long_name = var0 + " (reference data)"
  ;nc_outfile = ncdf_write(refdata, nc_filename)

  diff1@var = var0 + "_bias_1"
  diff1@long_name = var0 + " (multi-model bias)"
  nc_outfile = ncdf_write(diff1, nc_filename)

  nc_filename@existing = "append"

  diff2@var = var0 + "_bias_2"
  diff2@long_name = var0 + " (multi-model bias)"
  nc_outfile = ncdf_write(diff2, nc_filename)

  ;if (isvar("absdiff")) then
  ;  absdiff@var = var0 + "_abs_bias"
  ;  absdiff@long_name = var0 + " (multi-model absolute bias)"
  ;  nc_outfile = ncdf_write(absdiff, nc_filename)
  ;end if

  ;if (isvar("rmsdiff")) then
  ;  rmsdiff@var = var0 + "_rms_bias"
  ;  rmsdiff@long_name = var0 + " (multi-model root mean square difference)"
  ;  nc_outfile = ncdf_write(rmsdiff, nc_filename)

  ;  if (plot_rms_diff .and. stippandhatch) then
  ;    to_plot_not_signif_rmsd@diag_script = (/DIAG_SCRIPT/)
  ;    to_plot_not_signif_rmsd@var = "not_signif_rmsd"
  ;    to_plot_not_signif_rmsd@var_long_name = "not significant: rmsd " \
  ;                                       + "smaller than 1*natvar"
  ;    to_plot_not_signif@units     = 1
  ;    ncdf_outfile = ncdf_write(to_plot_not_signif_rmsd, nc_filename)
  ;  end if
  ;end if

  ;if (isvar("reldiff")) then
  ;  reldiff@var = var0 + "_rel_bias"
  ;  reldiff@long_name = var0 + " (multi-model relative bias)"
  ;  reldiff@units = reldiff@units
  ;  nc_outfile = ncdf_write(reldiff, nc_filename)
  ;end if

  ;if (stippandhatch) then
  ;  to_plot_signif@diag_script = (/DIAG_SCRIPT/)
  ;  to_plot_signif@var = "signif"
  ;  to_plot_signif@var_long_name = "significance: mean change larger than " \
  ;                                 + "2*natvar and 90% of models have the " \
  ;                                 + "same sign" 
  ;  to_plot_signif@units     = 1
  ;  ncdf_outfile = ncdf_write(to_plot_signif, nc_filename) 

  ;  to_plot_not_signif@diag_script = (/DIAG_SCRIPT/)
  ;  to_plot_not_signif@var = "not_signif"
  ;  to_plot_not_signif@var_long_name = "not significant: mean change " \
  ;                                     + "smaller than 1*natvar"
  ;  to_plot_not_signif@units     = 1
  ;  ncdf_outfile = ncdf_write(to_plot_not_signif, nc_filename)

  ;end if

  if (stippandhatch) then
    natvar_mmm1@diag_script = (/DIAG_SCRIPT/)
    natvar_mmm1@var = "natvar_mmm_1"
    natvar_mmm1@var_long_name = "natural variability"
    natvar_mmm1@units     = 1
    ncdf_outfile = ncdf_write(natvar_mmm1, nc_filename) 

    to_plot_confl_sign_1@diag_script = (/DIAG_SCRIPT/)
    to_plot_confl_sign_1@var = "confl_sign_1"
    to_plot_confl_sign_1@var_long_name = "more than 66% of models show change " \
                                       + "greater than variability threshold " \
                                       + "and <80% agree on sign of change"
    to_plot_confl_sign_1@units     = 1
    ncdf_outfile = ncdf_write(to_plot_confl_sign_1, nc_filename) 

    to_plot_not_robust_1@diag_script = (/DIAG_SCRIPT/)
    to_plot_not_robust_1@var = "not_robust_1"
    to_plot_not_robust_1@var_long_name = "more than 66% of models show change " \
                                       + "greater than variability threshold"
    to_plot_not_robust_1@units     = 1
    ncdf_outfile = ncdf_write(to_plot_not_robust_1, nc_filename)

    natvar_mmm2@diag_script = (/DIAG_SCRIPT/)
    natvar_mmm2@var = "natvar_mmm_2"
    natvar_mmm2@var_long_name = "natural variability"
    natvar_mmm2@units     = 1
    ncdf_outfile = ncdf_write(natvar_mmm2, nc_filename) 

    to_plot_confl_sign_2@diag_script = (/DIAG_SCRIPT/)
    to_plot_confl_sign_2@var = "confl_sign_2"
    to_plot_confl_sign_2@var_long_name = "more than 66% of models show change " \
                                       + "greater than variability threshold " \
                                       + "and <80% agree on sign of change"
    to_plot_confl_sign_2@units     = 1
    ncdf_outfile = ncdf_write(to_plot_confl_sign_2, nc_filename) 

    to_plot_not_robust_2@diag_script = (/DIAG_SCRIPT/)
    to_plot_not_robust_2@var = "not_robust_2"
    to_plot_not_robust_2@var_long_name = "more than 66% of models show change " \
                                       + "greater than variability threshold"
    to_plot_not_robust_2@units     = 1
    ncdf_outfile = ncdf_write(to_plot_not_robust_2, nc_filename)

    to_plot_improve@diag_script = (/DIAG_SCRIPT/)
    to_plot_improve@var = "improve"
    to_plot_improve@var_long_name = "improvement"
    to_plot_improve@units     = 1
    ncdf_outfile = ncdf_write(to_plot_improve, nc_filename) 
  end if

  ; ------------------------------------------------------------------------
  ; write provenance to netcdf output and plot file(s) (mean)
  ; ------------------------------------------------------------------------

  authors = (/"bock_lisa"/)
  statistics = (/"clim", "diff"/)
  domain = ("global")
  plottype = ("geo")
  references = (/"bock20jgr"/)
  prov_caption = caption + " for variable " + var0 \
    + ", reference = " + names(ref_ind) + "."
  infiles = metadata_att_as_array(info0, "filename")

;  log_provenance(nc_filename, plotfile, prov_caption, statistics, \
;                 domain, plottype, authors, references, infiles)

  leave_msg(DIAG_SCRIPT, "")

end
