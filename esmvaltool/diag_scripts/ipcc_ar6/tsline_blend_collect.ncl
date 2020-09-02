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

  if (var0 .eq. "tasa") then
    var = "gmst"
    units0 = "degC"
  else
    var = var0
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
      if (time_avg .eq. "yearly") then
        diag = time_operations(A0_mean, -1, -1, "average", \
                                "yearly", True)
      else
        diag = A0_mean
      end if


      ; Match time coordinate
      if (time_avg.eq."monthly") then
        date = cd_calendar(diag&time, -1)
      elseif (time_avg.eq."yearly") then
        date = diag&year
      end if
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
    "tsline_gmst_" + anom + "_stat.nc"
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
    if (time_avg.eq."monthly") then
      date = cd_calendar(data_temp&time, -1)
    elseif (time_avg.eq."yearly") then
      date = data_temp&time
    end if
    idx1 = ind(date(0).eq.data_mean&time)
    idx2 = ind(date(dimsizes(date) - 1).eq.data_mean&time)
    delete(date)

    data_mean(ii_n, 0, idx1:idx2)  = (/data_temp(:)/)
    data_mean(ii_n, 1, idx1:idx2)  = (/data_temp(:)-data_stat(:)/)
    data_mean(ii_n, 2, idx1:idx2)  = (/data_temp(:)+data_stat(:)/)
    data_mean&experiment(ii_n) = data_temp@experiment
    data_mean@units = units0

    delete(data_temp)
    delete(data_stat)

  end do

;  ; -------------------------------------------
;  ; Read extern observational uncertainty
;  ; -------------------------------------------
;
;  if (obs_stderr) then
;
;    data_ref_stderr = new ((/2, ntime/), double)
;    data_ref_stderr!0 = "statistics"
;    data_ref_stderr!1 = "time"
;    data_ref_stderr&time = time
;
;;    data_temp = read_data(info_items1[0])
;
;;    data_ref_stderr(0, :) = data_ref(0, :) - data_temp(:)
;;    data_ref_stderr(1, :) = data_ref(0, :) + data_temp(:)
;
;    fdata = "/work/bd0854/b309059/data/HadCRUT4/HadCRUT.4.6.0.0.annual_ns_avg.txt"
;    data   = asciiread(fdata,(/171,12/),"float")
;  
;    data_temp = data(:, 1) - data(:, 10)
;print(data_temp)
;    data_ref_stderr(0, :) = data_ref(0, :) - (/data_temp(0:(ntime-1))/)
;    data_temp = data(:, 11) - data(:, 1)
;print(data_temp)
;    data_ref_stderr(1, :) = data_ref(0, :) + (/data_temp(0:(ntime-1))/)
;
;  else
;
    data_ref_stderr = 0
;
;  end if

 

  ; -------------------------------------------
  ; NetCDF Output
  ; -------------------------------------------


  do i = 0, dimsizes(data_mean&experiment) - 1
    if (data_mean&experiment(i) .eq. "CMIP6_historical") then
      data_mean&experiment(i) = "CMIP6"
    elseif (data_mean&experiment(i) .eq. "CMIP5_historical") then
      data_mean&experiment(i) = "CMIP5"
    end if
  end do

  out_path = config_user_info@work_dir
  system("mkdir -p " + out_path)
  out_path1 = out_path + "tsline_collect_" + var0 + ".nc"
  data_mean@ncdf = out_path
  data_mean@diag_script = DIAG_SCRIPT
  data_mean@var = var0
  ncdf_outfile = ncdf_write(data_mean, out_path1)

  if (isatt(diag_script_info, "ref")) then
    out_path1 = out_path + "tsline_collect_" + var0 + "_ref.nc"
    data_ref@ncdf = out_path
    data_ref@diag_script = DIAG_SCRIPT
    data_ref@var = var0
    ncdf_outfile = ncdf_write(data_ref, out_path1)
  end if

  ; -------------------------------------------
  ; Plotting
  ; -------------------------------------------

  ; Define workstation
  outfile = config_user_info@plot_dir + data_mean@var + "_" + \
    str_sub_str(diag_script_info@region, " ", "_") + \
    "_multimodel_" + anom + "_" + start_year + "-" + end_year
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
  if (isatt(diag_script_info, "y_min")) then
    res@trYMinF = diag_script_info@y_min
  end if
  if (isatt(diag_script_info, "y_max")) then
    res@trYMaxF = diag_script_info@y_max
    ;res@trYMaxF = 1.2
  end if

  ;if (xmin .eq. 1850) then
  ;  res@tmXBMode         = "Manual"
  ;  res@tmXBTickSpacingF = 50
  ;elseif (xmin .eq. 1950) then
    res@tmXBMode         = "Manual"
    res@tmXBTickSpacingF = 20
  ;end if

  if (isatt(diag_script_info, "title")) then
    res@tiMainString = diag_script_info@title
  else
    res@tiMainString = ""
  end if

  ;res@tiMainString = diag_script_info@region

  if (isatt(variable_info[0], "long_name")) then
    if (var0 .eq. "tasa") then
      varname = "Temperature"
    elseif (var0 .eq. "siconc")
      varname = "Sea Ice"
    else
      varname = variable_info[0]@long_name
    end if
  else
    varname = var0
  end if

  if (data_mean@units .eq. "degC") then
    units = "~F34~0~F~ C"
  else
    units = data_mean@units
  end if

  if (anom .eq. "anom") then
    res@tiYAxisString = "Change in " + varname + " (" + units + ")"
  else
    res@tiYAxisString = varname + " (" + units + ")"
  end if

  res0 = True
  ref_start = diag_script_info@ref_start
  ref_end = diag_script_info@ref_end

  if (isatt(diag_script_info, "ref")) then
    xy_line_collect(wks, data_mean, data_ref, data_ref&time, \
                   data_ref_stderr, ref_start, ref_end, res, res0, info_var)
  else
    xy_line_collect(wks, data_mean, 0, data_mean&time, \
                    0, ref_start, ref_end, res, res0, info_var)
  end if

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
;  if (anom .eq. "anom") then
;    statistics = "anomaly"
;  else
;    statistics = "mean"
;  end if
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
