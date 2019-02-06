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

load "./interface_scripts/interface.ncl"

load "./diag_scripts/shared/latlon.ncl"
load "./diag_scripts/shared/statistics.ncl"
load "./diag_scripts/shared/scaling.ncl"
load "./diag_scripts/shared/ensemble.ncl"

load "./diag_scripts/shared/plot/style.ncl"
load "./diag_scripts/shared/plot/xy_line.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")
  diag_script_base = basename(DIAG_SCRIPT)

;  var0 = variable_info[0]@short_name
;  field_type0 = variable_info[0]@field
;  info_items = select_metadata_by_name(input_file_info, var0)
;  datasetnames = metadata_att_as_array(info_items, "dataset")
;  dim_MOD = ListCount(info_items)
;  dim_VAR = ListCount(variable_info)

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
  ;data_all!0 = "experiments"
  ;data_all&experiments = diag_script_info@experiments
  dim_ref = (/dimsizes(diag_script_info@ref), ntime/)
  data_ref = new(dim_ref, double)
  data_ref!0 = "models"
  data_ref&models = diag_script_info@ref

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
     "tsline_mm_stat.nc"
  input_files := tostring(input_files)

  ; Filter out non-existing files
  input_files := input_files(ind(isfilepresent(input_files)))

  ; Loop over files in the list, read and append data
  do ii = 0, dimsizes(input_files) - 1

    data_temp = ncdf_read(input_files(ii), "tas")

    ;; Create array for collecting all datasets
    ;if (.not.isdefined("data_all")) then
    ;  ;dim_temp = dimsizes(data_temp&time)
    ;  ;dim_data = (/dimsizes(input_files), 3, dim_temp/)
    ;  ;data_all = new(dim_data, double)
    ;  data_all(0, :, :) = data_temp
    ;  ;data_all!0 = "experiments"
    ;  ;data_all&experiments = diag_script_info@experiments
    ;end if

    ;; Append data
    ;data_all(ii, :, :) = (/data_temp/)

printVarSummary(data_all)
printVarSummary(data_temp)
    if (ii.eq.0) then
      data_all(0:2, :) = data_temp
    else
      ; Append data
      data_all(3:5, :) = (/data_temp/)
    end if

    delete(data_temp)

  end do

  ; -------------------------------------------
  ; Read observational data
  ; -------------------------------------------

  input_files_ref = diag_script_info@input_files + "/" + \
                    "tsline_" + anom + "_" + detr + ".nc"
  input_files_ref := tostring(input_files_ref)

print("input_files_ref = " + input_files_ref)
print("ind(isfilepresent(input_files_ref)) = " + ind(isfilepresent(input_files_ref)))

  ; Filter out non-existing files
  input_files_ref := input_files_ref(ind(isfilepresent(input_files_ref)))

  ; ; Loop over files in the list, read and append data
  ; do ii = 0, dimsizes(input_files_ref) - 1

  data_temp = ncdf_read(input_files_ref(0), "tas")

printVarSummary(data_ref)
printVarSummary(data_temp)
print("data_temp&model = " + data_temp&model)

  do i = 0, dimsizes(data_ref&models)-1

print("data_ref&models(i) = " + data_ref&models(i))
    ind_ref = ind(data_temp&model .eq. data_ref&models(i))
    if (all(ismissing(ind_ref))) then
      error_msg("f", DIAG_SCRIPT, "", "observational data set not available")
    else
      data_ref(i,  :) = (/data_temp(ind_ref, :)/)
    end if

  end do

printVarSummary(data_ref)

  delete(data_temp)

;  ; Attach plotting options
;  copy_VarAtts(diag_script_info, data_all)
;  if (isatt(diag_script_info, "title")) then
;    data_all@res_tiMainString = diag_script_info@title
;  end if
;  if (diag_script_info@metric.eq."taylor") then
;    data_all@res_varLabels = data_all&models
;    data_all@res_caseLabels = data_all&diagnostics
;  end if
;
;  ; Create outfile directory
;  system("mkdir -p " + config_user_info@plot_dir)
;  if (dimsizes(data_all&diagnostics).gt.1) then
;    diags = data_all&diagnostics(0) + "_to_" + \
;      data_all&diagnostics(dimsizes(data_all&diagnostics) - 1)
;  else
;    diags = str_concat(data_all&diagnostics)
;  end if
;
;  ; Define outfile name
;  outfile_plot = config_user_info@plot_dir + diags
;  if (isatt(data_all, "metric")) then
;    outfile_plot = outfile_plot + "_" + data_all@metric
;  end if
;
;  ; Create workspace
;  wks = gsn_open_wks(file_type, outfile_plot)
;  if (diag_script_info@metric.eq."taylor") then
;    wks@legendfile = outfile_plot + "_legend"
;  end if
;
;  ; Call plotting function
;  if (diag_script_info@metric.eq."taylor") then
;    plot = taylor_plot(wks, data_all, "grade")
;  else
;    plot = portrait_plot(wks, data_all, "grade")
;  end if

;  log_info(" gv " + outfile_plot + "." + file_type)

  leave_msg(DIAG_SCRIPT, "")

end
