; #############################################################################
;  PLOTS TIME SERIES
;  Author: Lisa Bock (lisa.bock@dlr.de)
;  ESMVal project
; #############################################################################
; 
;  Description
;     Global climate variability of surface temperature as represented by the 
;     standard deviation of zonal-mean values
;     This script has been developed to reproduce Fig. 9.33 in 
;     IPCC AR5 Chapter 9.
; 
;  Required variable_info attributes (variable specific)
; 
;  Required diag_script_info attributes (diagnostics specific)
; 
;  Optional diag_script_info attributes (diagnostic specific)
; 
;  Modification history
;  20190304-A_bock_li: ported from v1 to v2
;  20160905-A_bock_li: written.
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
  info_items = select_metadata_by_name(input_file_info, var0)
  datasetnames = metadata_att_as_array(info_items, "dataset")
  dim_MOD = ListCount(info_items)
  dim_VAR = ListCount(variable_info)

  if (dim_VAR .gt. 1) then
    var1 = variable_info[1]@short_name
    info_items1 = select_metadata_by_name(input_file_info, var1)
    datasetnames1 = metadata_att_as_array(info_items1, "dataset")
    dim_MOD1 = ListCount(info_items1)
    do i = 0, dim_MOD1-1
      ListAppend(info_items, info_items1[i])
    end do
    datasetnames := array_append_record(datasetnames, datasetnames1, 0)
    dim_MOD = dim_MOD + dim_MOD1
  end if

  ; Create output plot directory
  plot_dir = config_user_info@plot_dir
  system("mkdir -p " + plot_dir)

  ; Plot file type
  file_type = config_user_info@output_file_type
  if (ismissing(file_type)) then
    file_type = "ps"
  end if

  ; Set index of the reference (and alternative) dataset
  if (isatt(variable_info[0], "reference_dataset")) then
    ref_ind = ind(datasetnames.eq.variable_info[0]@reference_dataset)
    A_ref = read_data(info_items[ref_ind])
  else
    A_ref = read_data(info_items[0])
  end if

  ; Determine time range
  start_year = min(metadata_att_as_array(info_items, "start_year"))
  end_year = max(metadata_att_as_array(info_items, "end_year"))
  all_years = ispan(start_year, end_year, 1)

  ; Create model array
  model_arr = new((/dim_MOD, dimsizes(A_ref&lat)/), float)
  model_arr!0 = "model"
  model_arr!1 = "lat"
  model_arr&model = datasetnames
  model_arr&lat = A_ref&lat

  ; Loop over models
  do imod = 0, dim_MOD - 1

    ; Read data
    A0 = read_data(info_items[imod])

    ; Convert units for plotting (if required)
    if (isatt(diag_script_info, "plot_units")) then
      A0 = convert_units(A0, diag_script_info@plot_units)
    end if

    procmod = dim_avg_Wrap(A0)
    delete(A0)

    A0_timavg = time_operations(procmod, -1, -1, "average", "yearly", True)
    delete(procmod)

    ;; calculate zonal standard deviation
    A0_stddev = dim_stddev_n_Wrap(A0_timavg,0)
    delete(A0_timavg)

    if (dimsizes(A0_stddev&lat).ne.dimsizes(A_ref&lat)) then
      dnames = getVarDimNames(A0_stddev)
      dx = ind(dnames.eq."lat")
      var_hreg = linint1_n_Wrap(A0_stddev&lat, A0_stddev, False, \
                               A_ref&lat, 0, dx)
    else
      var_hreg = A0_stddev
      copy_VarAtts(A0_stddev, var_hreg)
    end if

    model_arr(imod, :) = (/var_hreg/)
    if (imod.eq.0) then
        copy_VarAtts(var_hreg, model_arr)
    end if
    delete(A0_stddev)

  end do


  ; Output to NetCDF
  out_path = config_user_info@work_dir
  system("mkdir -p " + out_path)
  out_path = out_path + var0 + "_std_dev_zonmean.nc"
  model_arr@ncdf = out_path
  model_arr@diag_script = DIAG_SCRIPT
  model_arr@var = var0
  ncdf_outfile = ncdf_write(model_arr, out_path)

  ; Define workstation
  outfile = config_user_info@plot_dir + var0 + "_std_dev_zonmean.nc" 
  wks = gsn_open_wks(file_type, outfile)

  ; Set resources
  res = True

  res@tiMainString = "Zonal standard deviation ~C~ ~Z75~" + variable_info[0]@long_name
  res@tiXAxisString = "Latitude"

  if (model_arr@units .eq. "degC") then
    units = "~F34~0~F~ C"
  else
    units = model_arr@units
  end if
  res@tiYAxisString = "(" + units + ")"

  xy_line(wks, model_arr, model_arr&lat, 0, res, info_items)

  log_info(" wrote " + outfile + "." + file_type)

  leave_msg(DIAG_SCRIPT, "")

end
