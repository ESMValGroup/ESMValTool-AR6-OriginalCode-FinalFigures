; #############################################################################
; PLOTS TIME SERIES
; Author: Irene Cionni (ENEA, Italy), Veronika Eyring (DLR, Germany)
; ESMVal project
; #############################################################################
;
; Description
;    This script is based on the original CCMVal script tsline.ncl and has
;    been further developed as part of the ESMValTool.
;
; Required diag_script_info attributes (diagnostic specific)
;    styleset: as in diag_scripts/shared/plot/style.ncl functions
;
; Optional diag_script_info attributes (diagnostic specific)
;    time_avg: type of time average (currently only "yearly" and "monthly" are
;              available).
;    ts_anomaly: calculates anomalies with respect to the defined period
;    ref_start: start year of reference period for anomalies
;    ref_end: end year of reference period for anomalies
;    ref_value: if true, right panel with mean values is attached
;    ref_mask: if true, model fields will be masked by reference fields
;    plot_units: variable unit for plotting
;    write_stat: if true, write multi model statistics in nc-file
;    stat: statistics for multi model nc-file (MinMax,5-95,10-90)
;    y-min: set min of y-axis
;    y-max: set max of y-axis
;    mean_nh_sh: if true, calculate first NH and SH mean
;    run_ave: if not equal 0 than calculate running mean over this number 
;             of years
;
; Caveats
;
; Modification history
;    20181112-A_bock_ls: code rewritten for ESMValTool v2.0
;    20170623-A_laue_ax: added tags for reporting
;    20160905-A-Bock_li: implemented multi-model mean and variable refence
;                        period for anomaly with refence mean value.
;    20151027-A_laue_ax: moved call to 'write_references' to the beginning
;                        of the code
;    20150622-A_wenz_sa: added optional anomaly calculation, choise of
;                        area opperations (sum, average) and detrending of
;                        time series.
;    20150420-A_righ_ma: adapted to new structure. Some of the original
;                        features not ported yet (seasonal average, smoothing,
;                        multi-model mean, etc.).
;    20??????-A_cion_ir: written.
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

  var0 = variable_info[0]@short_name
  field_type0 = variable_info[0]@field
  exp0 = variable_info[0]@exp
  project0 = input_file_info[0]@project
  info_items = select_metadata_by_name(input_file_info, var0)
  datasetnames = metadata_att_as_array(info_items, "dataset")
  dim_MOD = ListCount(info_items)
  dim_VAR = ListCount(variable_info)

  ; Create output plot directory
  plot_dir = config_user_info@plot_dir
  system("mkdir -p " + plot_dir)

  ; Check field type
  plot_fields = (/"T2Ms"/)
  ind_f = any(field_type0 .eq. plot_fields)
  if (all(ismissing(ind_f))) then
    error_msg("f", DIAG_SCRIPT, "", "can't use field " + field_type0)
  end if
  delete(plot_fields)
  delete(ind_f)

  ; Plot file type
  file_type = config_user_info@output_file_type
  if (ismissing(file_type)) then
    file_type = "ps"
  end if

  ; Time averaging
  if (isatt(diag_script_info, "time_avg")) then
    time_avg = diag_script_info@time_avg
    if (all(time_avg.ne.(/"yearly", "monthly"/))) then
        error_msg("f", DIAG_SCRIPT, "", \
                  "time averaging option " + time_avg + " not yet " + \
                  "implemented")
    end if
  else
    time_avg = "monthly"
  end if

  ; Check for required settings
  exit_if_missing_atts(diag_script_info, "styleset")

  ; Set default values for non-required diag_script_info attributes
  set_default_att(diag_script_info, "time_avg", "monthly")
  set_default_att(diag_script_info, "ts_anomaly", "noanom")
  set_default_att(diag_script_info, "ref_value", True)
  set_default_att(diag_script_info, "ref_mask", False)
  set_default_att(diag_script_info, "region", "Global")
  set_default_att(diag_script_info, "write_stat", False)
  set_default_att(diag_script_info, "stat", "MinMax")
  set_default_att(diag_script_info, "mean_nh_sh", False)
  set_default_att(diag_script_info, "run_ave", 0)

  ; Determine time range
  start_year = min(metadata_att_as_array(info_items, "start_year"))
  end_year = max(metadata_att_as_array(info_items, "end_year"))
  all_years = ispan(start_year, end_year, 1)

  ; Create time coordinate
  if (time_avg.eq."monthly") then
    ntime = 12 * (end_year - start_year + 1)
    time = new(ntime, integer)
    do yy = start_year, end_year
        do mm = 1, 12
            time(12 * (yy - start_year) + mm - 1) = 100 * yy + mm
        end do
    end do
  else if (time_avg.eq."yearly") then
    ntime = end_year - start_year + 1
    time = new(ntime, integer)
    time = ispan(start_year, end_year, 1)
  end if
  end if

  ; get multi-model mean index
  mmm_ind = ind(datasetnames .eq. "MultiModelMean")

  ; Set index of the reference (and alternative) dataset
  if (isatt(variable_info[0], "reference_dataset")) then
    ref_ind = ind(datasetnames.eq.variable_info[0]@reference_dataset)
    A_ref = read_data(info_items[ref_ind])
  end if

;  ; individual case for HadCRUT4 observations
;  ; FIX-ME: mean value comes from climatology file (absolute.nc).
;  ; There are no missing values as in the anomaly data.
;  clim_ind = -1
;  if (any(datasetnames.eq."HadCRUT4-clim")) then
;    clim_ind = ind(datasetnames.eq."HadCRUT4-clim")
;    ind_wo_clim = ind(datasetnames.ne."HadCRUT4-clim")
;  end if

  ; Anomaly
  if (isatt(diag_script_info, "ts_anomaly")) then
    anom = diag_script_info@ts_anomaly
    if (anom .eq. "anom") then
      if (isatt(diag_script_info, "ref_start") .and. \
          isatt(diag_script_info, "ref_end")) then
        ref_start = diag_script_info@ref_start
        ref_end   = diag_script_info@ref_end
        if (ref_start.lt.start_year) then
          ref_start = start_year
        end if
        if ((ref_start.lt.start_year) .or. (ref_end.gt.end_year) .or. \
            (ref_end.lt.ref_start)) then
          error_msg("f", DIAG_SCRIPT, "", \
                    "period for reference years is not properly defined")
        end if
      else
        error_msg("f", DIAG_SCRIPT, "", \
                  "period for reference years is not defined " + \
                  "(needed for anomaly)")
      end if

      anom_ref = new((/dim_MOD/), double)
      anom_ref!0 = "model"
      anom_ref&model = datasetnames
    end if
  else
    anom = "noanom"
  end if

  ; if true, right panel with mean values is attached
  if (isatt(diag_script_info, "ref_value")) then
    ref_value = diag_script_info@ref_value
  end if

  ; TODO
  model_arr_stddev = 0

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

    ;if(imod.ne.clim_ind) then

      ; Read data
      A0 = read_data(info_items[imod])
      dnames = getVarDimNames(A0)

      ; Convert units for plotting (if required)
      if (isatt(diag_script_info, "plot_units")) then
        A0 = convert_units(A0, diag_script_info@plot_units)
      end if

      ; time, lat, lon
      if (field_type0.eq."T2Ms") then

        if (isatt(variable_info[0], "reference_dataset")) then

          ;; save field with no masking for mean reference value
          ;if (anom .eq. "anom" .and. ref_value .eq. True)
          ;  A0_nomask = A0
          ;end if

          if(diag_script_info@ref_mask .and. imod .ne. ref_ind) then
            ; Determine start/end year
            start_year = info_items[imod]@start_year
            end_year = info_items[imod]@end_year
            A_ref_mask = time_operations(A_ref, start_year, \
                                         end_year, "extract", "", 0)
            A0 = where(A_ref_mask.eq.A_ref_mask@_FillValue, \
                       A_ref_mask@_FillValue, A0)
            delete(A_ref_mask)
          end if

        end if

        ; Anomaly
        if(anom .eq. "anom") then
          ; calculate mean of ref period if 0.5 data points are available
          tmp = time_operations(A0, 1961, 1990, "extract", "", 0)
          date := cd_calendar(tmp&time, 0)
          year := date(:, 0)
          month := date(:, 1)
          weights = days_in_month(toint(year), toint(month))
          ;limit  = toint(0.5 * dimsizes(tmp&time))
          limit  = 1
          A0_timavg_ref = dim_avg_wgt_n_Wrap(tmp, weights, limit, 0)
          delete(tmp)
          delete(date)

          do i = 0, dimsizes(A0&time) - 1
            A0(i, :, :) = A0(i, :, :) - A0_timavg_ref(:, :)
          end do

          ;if (ref_value.eq.True)
          ;  A0_timavg_ref_nomask = time_operations(A0_nomask, ref_start, \
          ;                                         ref_end, "average",   \
          ;                                         "annualclim", True)
          ;  anom_ref(imod) = area_operations(A0_timavg_ref_nomask, -90., 90., \
          ;                                   0., 360., "average", True)
          ;  delete(A0_timavg_ref_nomask)
          ;  delete(A0_nomask)
          ;end if

          anom_ref_tmp = area_operations(A0_timavg_ref, -90., 90., \
                                         0., 360., "average", True)
          if (diag_script_info@mean_nh_sh) then
            ; calculate global mean first for each hemisphere

            tmp1 = area_operations(A0_timavg_ref, -90., 0., 0., 360., \
                                   "average", True)
            tmp2 = area_operations(A0_timavg_ref, 0., 90., 0., 360., \
                                   "average", True)
            anom_ref_tmp = (tmp1 + tmp2) /2.
            delete(tmp1)
            delete(tmp2)
          end if
          delete(A0_timavg_ref)

        end if

        procmod = area_operations(A0, -90., 90., 0., 360., \
                                  "average", True)

        if (diag_script_info@mean_nh_sh) then
          ; calculate global mean first for each hemisphere

          tmp1 = area_operations(A0, -90., 0., 0., 360., \
                                 "average", True)
          tmp2 = area_operations(A0, 0., 90., 0., 360., \
                                 "average", True)
          procmod = (tmp1 + tmp2) /2.
          delete(tmp1)
          delete(tmp2)
        end if

        if(anom .eq. "anom") then
          procmod = procmod + anom_ref_tmp

          anom_ref_new = time_operations(procmod, ref_start, ref_end, "average", \
                                         "annualclim", True)
          procmod = procmod - anom_ref_new

          if (ref_value.eq.True)
            anom_ref(imod) = anom_ref_new
          end if

          delete(anom_ref_new)
          delete(anom_ref_tmp)
        end if

      end if

      if (.not.isdefined("procmod")) then
        error_msg("f", diag_script_info, "", "cannot process this " + \
                  "field type " + field_type0)
      end if
      delete(A0)

      ; Detrend
      if (isatt(diag_script_info, "ts_detrend")) then
        detr = diag_script_info@ts_detrend
        tmp = runave_Wrap(procmod, 2, 0)
        delete(procmod)
        procmod = tmp
        delete(tmp)
      else
        detr = "nodetr"
      end if

      ; Calculate time average
      if (time_avg.ne."monthly") then
        A0_timavg = time_operations(procmod, -1, -1, "average", \
                                    time_avg, True)
        if(diag_script_info@run_ave .gt. 0) then
          A0_timavg = runave_Wrap(A0_timavg, diag_script_info@run_ave, 1)
        end if
      else
        A0_timavg = procmod
      end if
      delete(procmod)
      procmod = A0_timavg
      delete(A0_timavg)

      ; Match time coordinate
      if (time_avg.eq."monthly") then
        date = cd_calendar(procmod&time, -1)
      else if (time_avg.eq."yearly") then
        date = procmod&year
      end if
      end if
      idx1 = ind(date(0).eq.model_arr&time)
      idx2 = ind(date(dimsizes(date) - 1).eq.model_arr&time)
      model_arr(imod, idx1:idx2) = (/procmod/)
      if (imod.eq.0) then
        copy_VarAtts(procmod, model_arr)
      end if
      delete(procmod)
      delete(date)

    ;end if

  end do

  ; calculate multi model statistics
  if (diag_script_info@write_stat) then
    if (isatt(variable_info[0], "reference_dataset")) then
      ;if (clim_ind .ne. -1) then
      ;  ind_mod = ind(datasetnames.ne."MultiModelMean" .and. \
      ;                datasetnames.ne.variable_info[0]@reference_dataset .and. \
      ;                datasetnames.ne."HadCRUT4-clim")
      ;else
        ind_mod = ind(datasetnames.ne."MultiModelMean" .and. \
                      datasetnames.ne.variable_info[0]@reference_dataset)
      ;end if
    else
      ind_mod = ind(datasetnames.ne."MultiModelMean")
    end if

    model_arr_stat = new((/3, ntime/), double)
    model_arr_stat!0 = "stat"
    model_arr_stat!1 = "time"
    model_arr_stat&time = time
    model_arr_stat&stat = (/"mean", "min", "max"/)

    model_arr_stat(0, :) = (/model_arr(mmm_ind, :)/)
    do i = 0, ntime-1
      statb = stat_dispersion(model_arr(ind_mod, i), False)
      if (diag_script_info@stat .eq. "MinMax") then
        ; min
        model_arr_stat(1,i) = (/statb(2)/)
        ; max
        model_arr_stat(2, i) = (/statb(14)/)
      elseif (diag_script_info@stat .eq. "5-95") then
        ; 5% quantile
        model_arr_stat(0, i) = statb(24)
        ; 95% quantile
        model_arr_stat(0,i) = statb(25)
      elseif (diag_script_info@stat .eq. "10-90") then
        ; 10% quantile
        model_arr_stat(1, i) = (/statb(3)/)
        ; 90% quantile
        model_arr_stat(2, i) = (/statb(13)/)
      end if
    end do
  end if

  ; individual case for HadCRUT4 observations
  ; FIX-ME: mean value comes from climatology file (absolute.nc).
  ; There are no missing values as in the anomaly data.
  ;if (isatt(variable_info[0], "reference_dataset") .and. variable_info[0]@reference_dataset .eq. "HadCRUT4") then
  ;  if(ref_start .eq. 1961 .and. ref_end .eq. 1990) then
  ;    anom_ref(ref_ind) = 14.0
  ;  else
  ;    anom_ref(ref_ind) = anom_ref@_FillValue
  ;  end if
  ;end if
;  if (ref_value .and. any(datasetnames.eq."HadCRUT4-clim")) then
;    ; Read data
;    A0 = read_data(info_items[clim_ind])
;    dnames = getVarDimNames(A0)
;
;    ; Convert units for plotting (if required)
;    if (isatt(diag_script_info, "plot_units")) then
;      A0 = convert_units(A0, diag_script_info@plot_units)
;    end if
;
;    if (ref_value.eq.True)
;      A0_timavg_clim = time_operations(A0, 1990, 1990, "average", \
;                                       "annualclim", True)
;      clim_ref = area_operations(A0_timavg_clim, -90., 90., 0., 360., \
;                                 "average", True)
;      anom_ref(ref_ind) = clim_ref
;
;      tmp = model_arr(ind_wo_clim, :)
;      delete(model_arr)
;      model_arr = tmp
;      delete(tmp)
;      tmp = anom_ref(ind_wo_clim)
;      delete(anom_ref)
;      anom_ref = tmp
;      delete(tmp)
;    end if
;  end if

  ; Convert time coordinate to years (required by the plot routine)
  if (time_avg.eq."monthly") then
    year = model_arr&time / 100
    xmin = min(year)
    xmax = max(year)
    month = model_arr&time - 100 * year
    time = todouble(year + month/12.)
    delete(model_arr&time)
    model_arr&time = time
    delete(time)
    delete(year)
    delete(month)
  else if (time_avg.eq."yearly")
    xmin = min(model_arr&time)
    xmax = max(model_arr&time)
    tmp = todouble(model_arr&time)
    delete(model_arr&time)
    model_arr&time = tmp
    delete(tmp)
  end if
  end if

  ; Optional output to NetCDF
  if (config_user_info@write_netcdf.eq."True") then
    out_path = config_user_info@work_dir
    system("mkdir -p " + out_path)
    out_path1 = out_path + "tsline_" + var0 + "_" + anom + "_" + detr + ".nc"
      ; + start_year + "-" + end_year + ".nc"
    model_arr@ncdf = out_path
    model_arr@experiment = project0 + "_" + exp0
    model_arr@diag_script = DIAG_SCRIPT
    model_arr@var = var0
    ncdf_outfile = ncdf_write(model_arr, out_path1)
    ; statistics
    if (diag_script_info@write_stat) then
      out_path2 = out_path + "tsline_mm_" + var0 + "_stat.nc"
      model_arr_stat@ncdf = out_path
      model_arr_stat@experiment = project0 + "_" + exp0
      model_arr_stat@diag_script = DIAG_SCRIPT
      model_arr_stat@var = var0
      ncdf_outfile = ncdf_write(model_arr_stat, out_path2)
    end if
  end if

  ; Define workstation
  outfile = config_user_info@plot_dir + var0 + "_" \
    + str_sub_str(diag_script_info@region, " ", "_") \
    + "_" + project0 + "_" + exp0 + "_" + anom + "_" + detr + "_" \
    + start_year + "-" + end_year
  wks = gsn_open_wks(file_type, outfile)

  ; Set resources
  res = True
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
  if (isatt(diag_script_info, "y_min")) then
    res@trYMinF = diag_script_info@y_min
  end if
  if (isatt(diag_script_info, "y_max")) then
    res@trYMaxF = diag_script_info@y_max
  end if

  if (diag_script_info@region .eq. "Global") then
    res@tiMainString = ""
  else
    res@tiMainString = diag_script_info@region
  end if

  if (isatt(diag_script_info, "ts_anomaly")) then
    res@tiYAxisString = var0 + " Anomaly" + " [" + model_arr@units + "]"
  else
    res@tiYAxisString = var0 + " [" + model_arr@units + "]"
  end if

  res0 = True
  ref_start = diag_script_info@ref_start
  ref_end = diag_script_info@ref_end
  res0@tiYAxisString = tostring(ref_start) + "-" + tostring(ref_end) \
    + " " + var0 + " Mean" + " [" + model_arr@units + "]"

  if(anom .eq. "anom") then
    xy_line_anom(wks, anom_ref, model_arr, model_arr&time, \
                 model_arr_stddev, ref_start, ref_end, res, res0, info_items)
  else
    xy_line(wks, model_arr, model_arr&time, model_arr_stddev, res, info_items)
  end if

  log_info(" wrote " + outfile + "." + file_type)

;  ; add meta data to plot (for reporting)
;
;  climofiles = new(dim_MOD, string)
;  climofiles = input_file_info@filename
;
;  domain = "DM_global"
;
;  if (anom .eq. "anom") then
;    stat = "ST_anomaly"
;    captionadd = "anomalies"
;  else
;    stat = "ST_mean"
;    captionadd = "means"
;  end if
;
;  alltags = array_append_record(tags, (/"PT_time", stat, domain/), 0)
;
;  if (diag_script_info@multi_model_mean.eq."y") then
;    allmodelnames = array_append_record(dataset_info@dataset, \
;                                        (/"multi-model-mean"/), 0)
;  else
;    allmodelnames = dataset_info@dataset
;  end if
;
;  caption = "Time series of the " + captionadd + " for variable " \
;    + variables(0) + ", similar to IPCC AR5, fig. 9.8."
;
;  id = DIAG_SCRIPT + "_" + variables(0)
;
;  contrib_authors = (/"A_cion_ir", "A_righ_ma", "A_wenz_sa", "A_bock_ls"/)
;
;  ; ESMValMD(wks@fullname, alltags, caption, id, variables(0), \
;  ;          allmodelnames, climofiles, DIAG_SCRIPT, contrib_authors)

  leave_msg(DIAG_SCRIPT, "")

end
