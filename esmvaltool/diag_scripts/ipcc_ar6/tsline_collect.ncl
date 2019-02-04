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

load "./diag_scripts/shared/plot/style.ncl"
load "./diag_scripts/shared/plot/xy_line.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")
  diag_script_base = basename(DIAG_SCRIPT)

  ; Check file type
  file_type = config_user_info@output_file_type
  if(ismissing(file_type)) then
    file_type = "ps"
  end if



  ; Set default values for non-required diag_script_info attributes
  set_default_att(diag_script_info, "time_avg", "monthly")
  set_default_att(diag_script_info, "ts_anomaly", "noanom")
  set_default_att(diag_script_info, "ref_value", False)

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


  input_files = diag_script_info@input_files + "/" + \
    "tsline_" + var0 + "_" + anom + "_" + detr + \
    "_mm_stat_" + start_year + "_" + end_year + ".nc"
  input_files := tostring(input_files)

print("input_files1 = " + input_files)

  ; Filter out non-existing files
  input_files := input_files(ind(isfilepresent(input_files)))

print("input_files2 = " + input_files)

#  ; Loop over files in the list, read and append data
#  do ii = 0, dimsizes(input_files) - 1
#
#    f = addfile(input_files(ii), "r")
#    curr_file = tostring(f->temp_list)
#    data_temp = ncdf_read(curr_file, "grade")
#
#    ; Change to 3 dimensional
#    if (dimsizes(dimsizes(data_temp)).eq.2) then
#
#      dim_temp = array_append_record(dimsizes(data_temp), 2, 0)
#      temp = new(dim_temp, typeof(data_temp))
#      temp(:, :, 0) = data_temp
#      data_temp := temp
#      delete(temp)
#
#    end if
#
#    ; Create array for collecting all datasets
#    if (.not.isdefined("data_all")) then
#      dim_temp = dimsizes(data_temp)
#      dim_data = (/dimsizes(input_files), dim_temp(1), 2/)
#      data_all = new(dim_data, float)
#      data_all(0, :, :) = data_temp
#      data_all!1 = "models"
#      data_all&models = data_temp&models
#    end if
#
#    ; Make sure dataset coordinate is consistent
#    consistent = False
#    if (dimsizes(data_temp&models).eq.dimsizes(data_all&models)) then
#      if (all(data_temp&models.eq.data_all&models)) then
#        consistent = True
#      end if
#    end if
#
#    ; Append data
#    if (consistent) then
#      data_all(ii, :, :) = (/data_temp/)
#      copy_VarCoords(data_temp, data_all(ii:ii, :, :))
#    else
#      do imod2 = 0, dimsizes(data_temp&models) - 1
#        if (.not.any(data_temp&models(imod2).eq.data_all&models)) then
#          ; Append record for dataset(imod)
#          data_new = extend_var_at(data_all, 1, dimsizes(data_all&models))
#          data_new(ii, dimsizes(data_all&models), :) = \
#            (/data_temp(0, imod2, :)/)
#          data_new&models(dimsizes(data_all&models)) = \
#            (/data_temp&models(imod2)/)
#          data_all := data_new
#          delete(data_new)
#        else
#          ; Loop over datasets of data
#          do imod = 0, dimsizes(data_all&models) - 1
#            ; if no data dataset is similar to curreny entry, write data entry
#            if (data_all&models(imod).eq. data_temp&models(imod2)) then
#                data_all(ii, imod, :) = (/data_temp(0, imod2, :)/)
#                copy_VarCoords(data_temp(0:0, imod2, :), \
#                               data_all(ii:ii, imod, :))
#            end if
#          end do
#        end if
#      end do
#    end if
#    delete(data_temp)
#  end do
#
#  ; Reduce size if all entries have only one reference
#  if (all(ismissing(data_all(:, :, 1)))) then
#    data_new = data_all(:, :, 0)
#    delete(data_all)
#    data_all = data_new
#    delete(data_new)
#  end if
#  delete(data_all@var)
#  delete(data_all@title)
#  delete(data_all@ncdf_dir)
#  if (isatt(data_all, "reference")) then
#    delete(data_all@reference)
#  end if
#
#  ; Sort datasets in alphabetical order, excluding multi-model mean/median
#  ; which are placed at the beginning
#  if (isatt(diag_script_info, "sort")) then
#    if (diag_script_info@sort) then
#      idx = ind(data_all&models.eq."MultiModelMean" .or. \
#                data_all&models.eq."MultiModelMedian")
#      pid = sort_alphabetically(data_all&models, idx, "begin")
#      if (dimsizes(dimsizes(data_all)).eq.3) then
#        data_all := data_all(:, pid, :)
#      else
#        data_all := data_all(:, pid)
#      end if
#      delete(pid)
#    end if
#  end if
#
#  ; Attach plotting options
#  copy_VarAtts(diag_script_info, data_all)
#  if (isatt(diag_script_info, "title")) then
#    data_all@res_tiMainString = diag_script_info@title
#  end if
#  if (diag_script_info@metric.eq."taylor") then
#    data_all@res_varLabels = data_all&models
#    data_all@res_caseLabels = data_all&diagnostics
#  end if
#
#  ; Create outfile directory
#  system("mkdir -p " + config_user_info@plot_dir)
#  if (dimsizes(data_all&diagnostics).gt.1) then
#    diags = data_all&diagnostics(0) + "_to_" + \
#      data_all&diagnostics(dimsizes(data_all&diagnostics) - 1)
#  else
#    diags = str_concat(data_all&diagnostics)
#  end if
#
#  ; Define outfile name
#  outfile_plot = config_user_info@plot_dir + diags
#  if (isatt(data_all, "metric")) then
#    outfile_plot = outfile_plot + "_" + data_all@metric
#  end if
#
#  ; Create workspace
#  wks = gsn_open_wks(file_type, outfile_plot)
#  if (diag_script_info@metric.eq."taylor") then
#    wks@legendfile = outfile_plot + "_legend"
#  end if
#
#  ; Call plotting function
#  if (diag_script_info@metric.eq."taylor") then
#    plot = taylor_plot(wks, data_all, "grade")
#  else
#    plot = portrait_plot(wks, data_all, "grade")
#  end if

  log_info(" gv " + outfile_plot + "." + file_type)

  leave_msg(DIAG_SCRIPT, "")

end
