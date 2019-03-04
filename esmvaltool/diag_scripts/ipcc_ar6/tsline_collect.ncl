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
;    time_avg: type of time average (currently only "yearly" and "monthly" are
;              available).
;    ts_anomaly: calculates anomalies with respect to the defined period
;    ref_start: start year of reference period for anomalies
;    ref_end: end year of reference period for anomalies
;    ref_value: if true, right panel with mean values is attached
;    plot_units: variable unit for plotting
;
; Caveats
;
; Modification history
;    20190204-A_bock_ls: written.
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

  var0 = variable_info[0]@short_name
;  field_type0 = variable_info[0]@field
  info_items = select_metadata_by_name(input_file_info, var0)
  units0 = info_items[0]@units
;  datasetnames = metadata_att_as_array(info_items, "dataset")
;  dim_MOD = ListCount(info_items)
;  dim_VAR = ListCount(variable_info)

  ; Create output plot directory
  plot_dir = config_user_info@plot_dir
  system("mkdir -p " + plot_dir)

  ; Check file type
  file_type = config_user_info@output_file_type
  if(ismissing(file_type)) then
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

  ; Set default values for non-required diag_script_info attributes
  set_default_att(diag_script_info, "time_avg", "monthly")
  set_default_att(diag_script_info, "ts_anomaly", "noanom")
  set_default_att(diag_script_info, "ref_value", False)
  set_default_att(diag_script_info, "region", "global")

  ; Determine time range
;  start_year = min(metadata_att_as_array(info_items, "start_year"))
;  end_year = max(metadata_att_as_array(info_items, "end_year"))
  start_year = diag_script_info@start_year
  end_year = diag_script_info@end_year
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

  ; Create array for collecting two additional datasets
  dim_data = (/2*3, ntime/)
  data_all = new(dim_data, double)

  if (isatt(diag_script_info, "ref")) then
    dim_ref = (/dimsizes(diag_script_info@ref), ntime/)
    data_ref = new(dim_ref, double)
  end if

  ; Anomaly
  if (isatt(diag_script_info, "ts_anomaly")) then
    anom = diag_script_info@ts_anomaly
    if (anom .eq. "anom") then
      if (isatt(diag_script_info, "ref_start") .and. \
          isatt(diag_script_info, "ref_end")) then
        ref_start = diag_script_info@ref_start
        ref_end   = diag_script_info@ref_end
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
      anom_ref = 0.

    end if
  else
    anom = "noanom"
  end if

  if (isatt(diag_script_info, "ts_detrend")) then
    detr = diag_script_info@ts_detrend
  else
    detr = "nodetr"
  end if

  ; -----------------------------------------------
  ; Read pre-calculated multi model statistics
  ; -----------------------------------------------

  input_files = diag_script_info@input_files + "/" + \
    "tsline_mm_" + var0 + "_stat.nc"
  input_files := tostring(input_files)

  ; Filter out non-existing files
  input_files := input_files(ind(isfilepresent(input_files)))

  nfiles = dimsizes(input_files)

  dim_data   = (/nfiles, ntime/)
  data_mean  = new(dim_data, double)
  data_mean!0 = "experiment"
  data_mean&experiment = new(nfiles, string, "exp")
  data_mean!1 = "time"
  data_mean@var = var0
  data_mean@units = diag_script_info@plot_units
  data_stat = new((/2, nfiles, ntime/), double)

  ; Loop over files in the list, read and append data
  do ii = 0, nfiles - 1

    log_info("Read in " + input_files(ii))

    data_temp = ncdf_read(input_files(ii), var0)

    data_mean(ii, :)  = (/data_temp(0, :)/)
    data_mean&experiment(ii) = data_temp@experiment
    ;data_mean&experiment(ii) = stringtochar(data_temp@experiment)
    data_mean&time = data_temp&time

    data_stat(0, ii, :) = data_temp(1, :)
    data_stat(1, ii, :) = data_temp(2, :)

    delete(data_temp)

  end do

  ; -------------------------------------------
  ; Read observational data
  ; -------------------------------------------

  if (isatt(diag_script_info, "ref")) then
    input_files_ref = diag_script_info@input_files + "/" + \
      "tsline_" + var0 + "_" + anom + "_" + detr + ".nc"
    input_files_ref := tostring(input_files_ref)

    ; Filter out non-existing files
    input_files_ref := input_files_ref(ind(isfilepresent(input_files_ref)))

    data_temp = ncdf_read(input_files_ref(0), var0)

    do i = 0, dimsizes(diag_script_info@ref)-1

      ind_ref = ind(data_temp&model .eq. diag_script_info@ref(i))
      if (all(ismissing(ind_ref))) then
        error_msg("f", DIAG_SCRIPT, "", "observational data set not available")
      else
        data_ref(i,  :) = data_temp(ind_ref, :)
      end if

    end do

    delete(data_temp)
  end if

  ; -------------------------------------------
  ; Plotting
  ; -------------------------------------------

  ; Define workstation
  outfile = config_user_info@plot_dir + data_mean@var + "_" + \
    diag_script_info@region + "_multimodel_" + anom + \
    "_" + detr + "_" + start_year + "-" + end_year
  wks = gsn_open_wks(file_type, outfile)

  ; Set resources
  res = True
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

  res@tiMainString = diag_script_info@region

  if (isatt(diag_script_info, "ts_anomaly")) then
    res@tiYAxisString = data_mean@var + " Anomaly" + " [" + data_mean@units + "]"
  else
    res@tiYAxisString = data_mean@var + " [" + data_mean@units + "]"
  end if

  res0 = True
  ref_start = diag_script_info@ref_start
  ref_end = diag_script_info@ref_end
  res0@tiYAxisString = tostring(ref_start) + "-" + tostring(ref_end) \
    + " " + data_mean@var + " Mean" + " [" + data_mean@units + "]"

printVarSummary(data_mean)

  if (isatt(diag_script_info, "ref")) then
    xy_line_collect(wks, data_mean, data_ref, data_ref&time, \
                   data_stat, ref_start, ref_end, res, res0, info_items)
  else
    xy_line_collect(wks, data_mean, 0, data_mean&time, \
                    data_stat, ref_start, ref_end, res, res0, info_items)
  end if

  log_info(" wrote " + outfile + "." + file_type)

  leave_msg(DIAG_SCRIPT, "")

end
