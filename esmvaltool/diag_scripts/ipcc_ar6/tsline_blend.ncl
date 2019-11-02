; #############################################################################
; PLOTS TIME SERIES
; Authors: Lisa Bock (DLR, Germany)
; #############################################################################
;
; Description
;    This script is for plotting a time series of the global mean (anomaly)
;    surface temperature (GMST) with blending and masking from Cowtan 2015.
;
; Required diag_script_info attributes (diagnostic specific)
;    styleset: as in diag_scripts/shared/plot/style.ncl functions
;
; Optional diag_script_info attributes (diagnostic specific)
;    time_avg: type of time average (currently only "yearly" and "monthly" are
;              available).
;    ts_anomaly: calculates anomalies with respect to the defined period;
;                for each gird point by removing the mean for the given
;                calendar month (requiring at least 50% of the data to be
;                non-missing)
;    ref_start: start year of reference period for anomalies
;    ref_end: end year of reference period for anomalies
;    ref_value: if true, right panel with mean values is attached
;    ref_mask: if true, model fields will be masked by reference fields
;    region: name of domain
;    plot_units: variable unit for plotting
;    y-min: set min of y-axis
;    y-max: set max of y-axis
;    run_ave: if not equal 0 than calculate running mean over this number
;             of years
;    header: if true, region name as header
;    write_stat: if true, write multi model statistics in nc-file
;    stat: statistics for multi model nc-file (MinMax,5-95,10-90)
;
; Caveats
;
; Modification history
;    20191030-bock_lisa: written.
;
; #############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/latlon.ncl"
load "$diag_scripts/shared/statistics.ncl"
load "$diag_scripts/shared/scaling.ncl"
load "$diag_scripts/shared/ensemble.ncl"

load "$diag_scripts/shared/plot/style.ncl"
load "$diag_scripts/shared/plot/xy_line.ncl"


undef("areas")
function areas(grid)

begin

  area = new(grid, float)
  do i = 0, grid - 1
    area(i) = ((sin(get_d2r(180.*(i+1)/grid-90.)) \
              - sin(get_d2r(180.*(i)/grid-90.)) ) \
              / sin(get_d2r(180./grid)))
  end do
  return area

end


begin

  enter_msg(DIAG_SCRIPT, "")

  var0 = variable_info[3]@short_name
  exp0 = variable_info[3]@exp
print("var0 = " + var0)
  project0 = input_file_info[0]@project
  info_items = select_metadata_by_name(input_file_info, var0)
  dim_VAR = ListCount(variable_info)

  info_tas = select_metadata_by_name(input_file_info, "tas")
  info_tos = select_metadata_by_name(input_file_info, "tos")
  info_sic = select_metadata_by_name(input_file_info, "siconc")
  info_sftlf = select_metadata_by_name(input_file_info, "sftlf")

  datasetnames = metadata_att_as_array(info_tas, "dataset")
  datasetnames := array_append_record(datasetnames, "MultiModelMean", 0)
  dim_MOD = ListCount(info_tas)

  log_info("++++++++++++++++++++++++++++++++++++++++++")
  log_info(DIAG_SCRIPT) 
  log_info("++++++++++++++++++++++++++++++++++++++++++")

  ; Create output plot directory
  plot_dir = config_user_info@plot_dir
  system("mkdir -p " + plot_dir)

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
  set_default_att(diag_script_info, "run_ave", 0)
  set_default_att(diag_script_info, "header", False)
  set_default_att(diag_script_info, "volcanoes", False)
  set_default_att(diag_script_info, "write_stat", False)
  set_default_att(diag_script_info, "stat", "MinMax")

  ; Determine time range
  start_year = min(metadata_att_as_array(info_tas, "start_year"))
  end_year = max(metadata_att_as_array(info_tas, "end_year"))
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
  elseif (time_avg.eq."yearly") then
    ntime = end_year - start_year + 1
    time = new(ntime, integer)
    time = ispan(start_year, end_year, 1)
  end if

  ; Set index of the reference dataset and read it
;  if (isatt(variable_info[0], "reference_dataset")) then
;  if (isatt(metadata_att_as_array(info_tas, "reference_dataset"))) then
print("ref_dataset = " + metadata_att_as_array(info_tas, "reference_dataset"))
    ref_dataset = metadata_att_as_array(info_tas, "reference_dataset")
    ref_ind = ind(datasetnames.eq.ref_dataset(0))
    tas_ref = read_data(info_tas[ref_ind])
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

      anom_ref = new((/dim_MOD+1/), double)
      anom_ref!0 = "model"
      anom_ref&model = datasetnames
    end if
  else
    anom = "noanom"
  end if

  ; TODO
  model_arr_stddev = 0

  ; Create model array
  model_arr = new((/dim_MOD+1, ntime/), double)
  model_arr!0 = "model"
  model_arr!1 = "time"
  model_arr&model = datasetnames
;  model_arr&model = array_append_record(datasetnames, "MultiModelMean", 0)
print("model_arr&model = " + model_arr&model)
  model_arr&time = time
  model_arr@_FillValue = 1e+20
  model_arr = model_arr@_FillValue
  model_arr@units = "degC"
  delete(time)

  ; get multi-model mean index
  mmm_ind = ind(datasetnames .eq. "MultiModelMean")
print("mmm_ind = " + mmm_ind)

  ; Loop over models
  do imod = 0, dim_MOD - 1

    log_info("Process dataset: " + datasetnames(imod))

    ; Read data
    tas0 = read_data(info_tas[imod])
    dnames = getVarDimNames(tas0)
    ndim = dimsizes(tas0)

    ; Convert units from K to degC
    tas0 = convert_units(tas0, "degC")

    if (imod .ne. ref_ind) then

      ; Read data
      tos0 = read_data(info_tos[imod])
      sic0 = read_data(info_sic[imod])
      sftlf0 = read_data(info_sftlf[imod])
      ndim_sftlf = dimsizes(sftlf0)

      ; force missing cellsto open water/land and scale if stored as percentage
      sic0 = where(sic0.lt.0.0, 0.0, sic0)
      sic0 = where(sic0.gt.100.0, 0.0, sic0)
      if (max(sic0) .gt. 90.0) then
        sic0 = 0.01 * sic0
      end if

      sftof0 = 1. - sftlf0
      sftof0 = where(sftof0.lt.0.0, 0.0, sftof0)
      sftof0 = where(sftof0.gt.100.0, 0.0, sftof0)
      if (max(sftof0) .gt. 90.0) then
        sftof0 = 0.01 * sftof0
      end if

      ; combine land/ice masks
      do m = 0, ndim(0) - 1
        sic0(m,:,:) = (1.0 - sic0(m,:,:)) * sftof0(:,:) 
      end do

      ; deal with missing tos through sic
      sic0 = where(tos0 .lt. -500.00, 0.0, sic0)
      sic0 = where(tos0 .gt. 500.00, 0.0, sic0)

      ; prepare missing
      tos0 = where(abs(tos0) .gt. 500.0, tos0@_FillValue, tos0)

    end if

    printVarSummary(tas0)

    ; calculate tas anomaly for each grid point
    ;base_mon = time_operations(tas0, ref_start, ref_end, "average", \
    ;                                  "monthlyclim", True)
    base_mon = time_operations(tas0, 1961, 1990, "average", \
                                      "monthlyclim", True)
    tas0_anom = tas0
    do i = 0, ndim(0) - 1
      tas0_anom(i, :, :) = tas0(i, :, :) - base_mon(mod(i, 12), :, :)
    end do

    if (imod .ne. ref_ind) then

      ; calculate tos anomaly for each grid point
      ;base_mon = time_operations(tos0, ref_start, ref_end, "average", \
      ;                                  "monthlyclim", True)
      base_mon = time_operations(tos0, 1961, 1990, "average", \
                                        "monthlyclim", True)
      tos0_anom = tos0
      do i = 0, ndim(0) - 1
        tos0_anom(i, :, :) = tos0(i, :, :) - base_mon(mod(i, 12), :, :)
      end do

      ; blend
      tos0 = tas0 * (1.0 - sic0) + tos0 * sic0
      tos0_anom = tas0_anom * (1.0 - sic0) + tos0_anom * sic0

      ; deal with any remaining NaNs
      tos0 = where(tos0.eq.tos0@_FillValue, tas0, tos0)
      tos0_anom = where(tos0_anom.eq.tos0@_FillValue, tas0_anom, tos0_anom)

      ; masking with reference dataset
      ; Determine start/end year
      start_year = info_tas[imod]@start_year
      end_year = info_tas[imod]@end_year
      tas_ref_mask = time_operations(tas_ref, start_year, \
                                       end_year, "extract", "", 0)
      ;tas0 = where(tas_ref_mask.eq.tas_ref_mask@_FillValue, \
      ;             tas_ref_mask@_FillValue, tos0)
      tas0_anom = where(tas_ref_mask.eq.tas_ref_mask@_FillValue, \
                   tas_ref_mask@_FillValue, tos0_anom)

    end if

    ; calculate field mean
    gmst_mon = area_operations(tas0, -90., 90., \
                               0., 360., "average", True) 
    gmst_mon_anom = area_operations(tas0_anom, -90., 90., \
                               0., 360., "average", True) 

    ; yearly average
    if (time_avg .eq. "yearly") then
      diag = time_operations(gmst_mon_anom, -1, -1, "average", \
                             "yearly", True)
    else
      diag = gmst_mon_anom
    end if

    ref_diag = time_operations(gmst_mon, 1961, 1990, "average", \
                               "annualclim", True)

    if (ref_start .ne. 1961 .or. ref_end .ne. 1990) then
      diag = diag + ref_diag
      ;ref_diag = time_operations(diag, ref_start, ref_end, "average", \
      ;                           "annualclim", True)
      ref_diag = avg(diag(ref_start-start_year:ref_end-start_year))
      diag = diag - ref_diag
    end if

    ; save mean ref value
    anom_ref(imod) = ref_diag

    model_arr(imod,:) = (/diag/)


;    ; Running Mean
;    if(diag_script_info@run_ave .gt. 0) then
;      run_ave = "running_mean"
;      tmp = runave_Wrap(procmod, 12 * diag_script_info@run_ave, 1)
;      delete(procmod)
;      procmod = tmp
;      delete(tmp)
;    else
;      run_ave = ""
;    end if

;    ; Match time coordinate
;    if (time_avg.eq."monthly") then
;      date = cd_calendar(procmod&time, -1)
;    elseif (time_avg.eq."yearly") then
;      date = procmod&year
;    end if
;    idx1 = ind(date(0).eq.model_arr&time)
;    idx2 = ind(date(dimsizes(date) - 1).eq.model_arr&time)
;    model_arr(imod, idx1:idx2) = (/procmod/)
;    if (imod.eq.0) then
;      copy_VarAtts(procmod, model_arr)
;    end if
;    delete(procmod)
;    delete(date)

  end do

  ; individual case for HadCRUT4 observations
  ; mean value comes from climatology file (absolute.nc).
  ; There are no missing values as in the anomaly data.
  ;if (anom .eq. "anom" .and. isatt(variable_info[0], "reference_dataset") \
  ;    .and. variable_info[0]@reference_dataset .eq. "HadCRUT4") then
  if (anom .eq. "anom" .and. ref_dataset(0).eq."HadCRUT4") then
    if(ref_start .eq. 1961 .and. ref_end .eq. 1990) then
      anom_ref(ref_ind) = 14.0
    else
      anom_ref(ref_ind) = anom_ref@_FillValue
    end if
  end if

  ; calculate multi model statistics
  ;if (diag_script_info@write_stat) then
    ;if (isatt(variable_info[0], "reference_dataset")) then
      ind_mod = ind(datasetnames.ne."MultiModelMean" .and. \
                    datasetnames.ne.ref_dataset(0))
                    ;datasetnames.ne.variable_info[0]@reference_dataset)
    ;else
    ;  ind_mod = ind(datasetnames.ne."MultiModelMean")
    ;end if

    model_arr_stat = new((/3, ntime/), double)
    model_arr_stat!0 = "stat"
    model_arr_stat!1 = "time"
    model_arr_stat&time = model_arr&time
    model_arr_stat&stat = (/"mean", "min", "max"/)

    ;model_arr_stat(0, :) = (/model_arr(mmm_ind, :)/)
;    tmp = new((/dim_MOD+1, ntime/), float)
;    copy_VarMeta(model_arr,tmp)
;printVarSummary(tmp)
    do i = 0, ntime-1
      statb = stat_dispersion(model_arr(ind_mod, i), False)
      model_arr_stat(0,i) = (/statb(0)/)
      if (diag_script_info@stat .eq. "MinMax") then
        ; min
        model_arr_stat(1,i) = (/statb(2)/)
        ; max
        model_arr_stat(2, i) = (/statb(14)/)
      elseif (diag_script_info@stat .eq. "5-95") then
        ; 5% quantile
        model_arr_stat(0, i) = (/statb(24)/)
        ; 95% quantile
        model_arr_stat(0,i) = (/statb(25)/)
      elseif (diag_script_info@stat .eq. "10-90") then
        ; 10% quantile
        model_arr_stat(1, i) = (/statb(3)/)
        ; 90% quantile
        model_arr_stat(2, i) = (/statb(13)/)
      end if
;      tmp(0:dim_MOD-1,i) = model_arr(:,i)
;printVarSummary(tmp)
;      tmp(dim_MOD,i) = (/statb(0)/)
;printVarSummary(tmp)
;      model_arr(dim_MOD,i) = (/statb(0)/)
    end do
  ;end if
;  model_arr := tmp
;  model_arr&model := array_append_record(model_arr&model, "MultiModelMean", 0)
  model_arr(mmm_ind, :) = dim_avg_n(model_arr(ind_mod,:),0) 
  anom_ref(mmm_ind) = avg(anom_ref(ind_mod))
print("model_arr(mmm_ind,:) = " + model_arr(dim_MOD,:))

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
  elseif (time_avg.eq."yearly") then
    xmin = min(model_arr&time)
    xmax = max(model_arr&time)
    tmp = todouble(model_arr&time)
    delete(model_arr&time)
    model_arr&time = tmp
    delete(tmp)
  end if

  ; *************************************
  ; output to NetCDF
  ; *************************************
  out_path = config_user_info@work_dir
  system("mkdir -p " + out_path)
  out_path1 = out_path + "tsline_" + var0 + "_" + anom + ".nc"
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

  ; Define workstation
  outfile = config_user_info@plot_dir + var0 + "_" \
    + str_sub_str(diag_script_info@region, " ", "_") \
    + "_" + project0 + "_" + exp0 + "_" + anom + "_" \
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

  res@tmXBMode         = "Manual"
  res@tmXBTickSpacingF = 20

  if (diag_script_info@header) then
    res@tiMainString = diag_script_info@region
  else
    res@tiMainString = ""
  end if

  if (isatt(variable_info[0], "long_name")) then
    if (var0 .eq. "tas") then
      varname = "Temperature"
    else
      varname = variable_info[0]@long_name
    end if
  else
    varname = var0
  end if

  if (model_arr@units .eq. "degC") then
    units = "~F34~0~F~ C"
  else
    units = model_arr@units
  end if

  if (anom .eq. "anom") then
    res@tiYAxisString = varname + " Anomaly" + " (" + units + ")"
  else
    res@tiYAxisString = varname + " (" + units + ")"
  end if

  res0 = True
  ref_start = diag_script_info@ref_start
  ref_end = diag_script_info@ref_end
  res0@tiYAxisString = tostring(ref_start) + "-" + tostring(ref_end) \
    + " Mean " + varname + " (" + units + ")"

print(info_items)

  if (anom .eq. "anom") then
    xy_line_anom(wks, anom_ref, model_arr, model_arr&time, \
                 model_arr_stddev, ref_start, ref_end, res, res0, info_items)
  else
    xy_line(wks, model_arr, model_arr&time, model_arr_stddev, res, info_items)
  end if

  log_info(" wrote " + outfile + "." + file_type)

  ; ***************************************
  ; add meta data to plot (for reporting)
  ; ***************************************

  if (diag_script_info@region .eq. "Global") then
    domain = "global"
  else
    domain = "reg"
  end if

  if (anom .eq. "anom") then
    statistics = "anomaly"
  else
    statistics = "mean"
  end if

  caption = "Time series of the " + statistics + " for Surface Temperature " \
            + "with blending and masking as Cowtan et al., 2015."
  contrib_authors = (/"bock_lisa"/)

  ; Call provenance logger
  log_provenance(ncdf_outfile, \
                 outfile + "." + file_type, \
                 caption, \
                 statistics, \
                 domain, \
                 "times", \
                 contrib_authors, \
                 (/"flato13ipcc", "cowtan15grl"/), \
                 metadata_att_as_array(input_file_info, "filename"))

  leave_msg(DIAG_SCRIPT, "")

end
