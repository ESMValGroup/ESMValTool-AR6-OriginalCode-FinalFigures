; Annular_modes_EOF
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;    This script applies the EOF analysis to a horizontal field over 20N-90N
;    and 20S-90S in observations and individual model ensemble members.
;    Intended to be applied for sea level pressure.
;    Outputs: PC1 time series and regressed anomalies onto PC1s
;             Figures show multi-model means of regression patterns.
;
; History
;    20190218 kosa_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;    season, season_nam, season_sam : "monthly", "DJF", "MAM", "JJA", or "SON"
;                                     Default: "monthly"
;                                     Priority: season_nam, season_sam > season
;    syear_eof, eyear_eof: First and last years for the EOF analysis.
;                          Expansion coefficients are calculated throughout
;                          the input data
;    movingave_index:      Integer, size of the moving average window to be
;                          applied to the PC1s (mainly for visualizing)
;
; ############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/statistics.ncl"
load "$diag_scripts/shared/plot/style.ncl"
load "$diag_scripts/shared/plot/contour_maps.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")

  tmp = metadata_att_as_array(input_file_info, "short_name")
  variables = get_unique_values(tmp)
  delete(tmp)
  n = dimsizes(variables)

  if (n.gt.1) then
    log_info(DIAG_SCRIPT + ": WARNING - Only one variable supported. " \
             + "Using only first variable.")
  end if
  mainvarind = 0

  var = variable_info[mainvarind]@short_name
  info = select_metadata_by_name(input_file_info, var)
  ndatasets = ListCount(info)
  log_info(DIAG_SCRIPT + ": Info : "+ndatasets+" datasets will be used.")
  if (isatt(variable_info[mainvarind], "reference_dataset")) then
    refname = variable_info[mainvarind]@reference_dataset
  end if
  names = metadata_att_as_array(info, "dataset")
  infiles = metadata_att_as_array(info, "filename")
  nmodels = dim_num(names.ne.refname)
  imodels = new(nmodels, "integer")
  i = 0
  do n = 0, ndatasets-1
    if (names(n).ne.refname) then
      imodels(i) = n
      i = i + 1
    end if
  end do

; ========================================================================
; ========================= write references =============================
; ========================================================================

;  Set default values for non-required diag_script_info attributes
  set_default_att(diag_script_info, "projection", "PolarStereographic")

;  season specification: "monthly": monthly anomalies for all calendar month,
;                        "DJF": seasonal mean anomalies for DJF
;                        "MAM":                             MAM
;                        "JJA":                             JJA
;                        "SON":                             SON
;                        "ANN": annual mean anomalies

  season_nam = "monthly"
  season_sam = "monthly"

  if (isatt(diag_script_info, "season")) then  ; common season for NAM & SAM
    season_info = diag_script_info@season
    if (season_info.ne."monthly") then
      season_nam = season_info
      season_sam = season_info
    end if
  end if

  if (isatt(diag_script_info, "season_nam")) then  ; NAM season
    season_info = diag_script_info@season_nam
    if (season_info.ne."monthly") then
      season_nam = season_info
    end if
  end if
  if (isatt(diag_script_info, "season_sam")) then  ; SAM season
    season_info = diag_script_info@season_sam
    if (season_info.ne."monthly") then
      season_sam = season_info
    end if
  end if

  ave_window = 0
  if (isatt(diag_script_info, "movingave_index")) then  ; smoothing indices
    moving_ave = diag_script_info@movingave_index
    if (isnumeric(moving_ave)) then
      if (moving_ave.lt.0) then
        error_msg("f", DIAG_SCRIPT, "", \
                  "movingave_index must be a non-negative integer or FALSE")
      else
        ave_window = moving_ave
      end if
    elseif(islogical(moving_ave)) then
      if (moving_ave) then
        error_msg("f", DIAG_SCRIPT, "", \
                  "movingave_index must be a non-negative integer or FALSE")
      else
        ave_window = 0
      end if
    else
      error_msg("f", DIAG_SCRIPT, "", \
                "movingave_index must be a non-negative integer or FALSE")
    end if
  end if

; make sure path for (optional) netcdf output exists

  if (config_user_info@write_netcdf.eq."True") then
    write_nc = True
    work_dir = config_user_info@work_dir + "/"
    ; Create work dir
    system("mkdir -p " + work_dir)
  else
    write_nc = False
  end if

  if (config_user_info@write_plots.eq."True") then
    write_plots = True
  else
    write_plots = False
  end if

; ========================================================================
; ========================== initialization ==============================
; ========================================================================

  diag_script_base = basename(DIAG_SCRIPT)

  ; check for reference dataset definition
  if (.not.(isvar("refname"))) then
    error_msg("f", DIAG_SCRIPT, "", "no reference dataset defined in recipe")
  end if

  ; set reference dataset

  iref = ind(names .eq. refname)
  if (ismissing(iref)) then
    error_msg("f", DIAG_SCRIPT, "", "reference dataset (" \
              + refname + ") is missing")
  end if

  ; set models

  imodels = ind(names .ne. refname)
  if (all(ismissing(imodels))) then
    error_msg("f", DIAG_SCRIPT, "", "no model dataset found")
  end if

  ; check maximum overlapping period

  if (isvar("data")) then
    delete(data)
  end if
  if (isvar("time")) then
    delete(time)
  end if
  do idata = 0, ndatasets-1
    data = read_data(info[idata])
    time = cd_calendar(data&time, -1)
    delete(data)
    if (idata.eq.0) then
      stime = time(0)
      etime = time(dimsizes(time)-1)
      stime_entire = time(0)
      etime_entire = time(dimsizes(time)-1)
    else
      stime = max((/ time(0), stime /))
      etime = min((/ time(dimsizes(time)-1), etime /))
      stime_entire = min((/ time(0), stime_entire /))
      etime_entire = max((/ time(dimsizes(time)-1), etime_entire /))
    end if
    delete(time)
  end do

  do ihemisphere = 0, 1
    if (ihemisphere .eq. 0) then
      hemisphere = "NH"
      lat_domain = (/ 20., 90. /)
      log_info("++++++++++++++++++++++++++++++++++++++++++")
      log_info(DIAG_SCRIPT + " NAM")
      log_info("++++++++++++++++++++++++++++++++++++++++++")
      season = season_nam
    elseif(ihemisphere .eq. 1) then
      hemisphere = "SH"
      lat_domain = (/ -90., -20. /)
      log_info("++++++++++++++++++++++++++++++++++++++++++")
      log_info(DIAG_SCRIPT + " SAM")
      log_info("++++++++++++++++++++++++++++++++++++++++++")
      season = season_sam
    end if

    if (isatt(diag_script_info, "syear_eof")) then
      syear_eof = diag_script_info@syear_eof
    end if
    if (isatt(diag_script_info, "eyear_eof")) then
      eyear_eof = diag_script_info@eyear_eof
    end if

  ; check if maximum overlapping period covers specified EOF period, or
  ; if EOF period is not specified, it is set as the maximum available period

  ; Maximum overlapping period
    syear  = stime/100
    smonth = stime - syear*100
    eyear  = etime/100
    emonth = etime - eyear*100

    if (isvar("syear_eof")) then
      if (season.eq."DJF") then
        stime_eof = (syear_eof-1)*100+12
      elseif(season.eq."MAM") then
        stime_eof = syear_eof*100+3
      elseif(season.eq."JJA") then
        stime_eof = syear_eof*100+6
      elseif(season.eq."SON") then
        stime_eof = syear_eof*100+9
      else
        stime_eof = syear_eof*100+1
      end if
      if (stime_eof.lt.stime) then
        error_msg("f", DIAG_SCRIPT, "", \
                  "Data don't cover specified EOF period.")
      end if
    else  ; If not specified, begining of the overlapping period
      syear_eof = syear
      if (season.eq."DJF") then
        syear_eof = syear + 1
      elseif(season.eq."MAM") then
        if (smonth.le.3) then
          syear_eof = syear
        else
          syear_eof = syear + 1
        end if
      elseif(season.eq."JJA") then
        if (smonth.le.6) then
          syear_eof = syear
        else
          syear_eof = syear + 1
        end if
      elseif(season.eq."SON") then
        if (smonth.le.9) then
          syear_eof = syear
        else
          syear_eof = syear + 1
        end if
      else
        if (smonth.eq.1) then
          syear_eof = syear
        else
          syear_eof = syear + 1
        end if
      end if
    end if
    if (isvar("eyear_eof")) then
      if (season.eq."DJF") then
        etime_eof = eyear_eof*100+2
      elseif(season.eq."MAM") then
        etime_eof = eyear_eof*100+5
      elseif(season.eq."JJA") then
        etime_eof = eyear_eof*100+8
      elseif(season.eq."SON") then
        etime_eof = eyear_eof*100+11
      else
        etime_eof = eyear_eof*100+12
      end if
      if (etime_eof.gt.etime) then
        error_msg("f", DIAG_SCRIPT, "", \
                  "Data don't cover specified EOF period.")
      end if
    else  ; If not specified, end of the overlapping period
      eyear_eof = eyear
      if (season.eq."DJF") then
        if (emonth.ge.2) then
          eyear_eof = eyear
        else
          eyear_eof = eyear - 1
        end if
      elseif(season.eq."MAM") then
        if (emonth.ge.5) then
          eyear_eof = eyear
        else
          eyear_eof = eyear - 1
        end if
      elseif(season.eq."JJA") then
        if (emonth.ge.8) then
          eyear_eof = eyear
        else
          eyear_eof = eyear - 1
        end if
      elseif(season.eq."SON") then
        if (emonth.ge.11) then
          eyear_eof = eyear
        else
          eyear_eof = eyear - 1
        end if
      else
        if (emonth.eq.12) then
          eyear_eof = eyear
        else
          eyear_eof = eyear - 1
        end if
      end if
    end if

  ; The entire data period
    syear_entire  = stime_entire/100
    smonth_entire = stime_entire - syear_entire*100
    eyear_entire  = etime_entire/100
    emonth_entire = etime_entire - eyear_entire*100

    if (season.eq."monthly") then
      if (smonth_entire.gt.1) then
        syear_entire = syear_entire + 1
      end if
      if (emonth_entire.lt.12) then
        eyear_entire = eyear_entire - 1
      end if
    elseif(season.eq."DJF") then
      syear_entire = syear_entire + 1
      if (emonth_entire.lt.2) then
        eyear_entire = eyear_entire - 1
      end if
    elseif(season.eq."MAM") then
      if (smonth_entire.gt.3) then
        syear_entire = syear_entire + 1
      end if
      if (emonth_entire.lt.5) then
        eyear_entire = eyear_entire - 1
      end if
    elseif(season.eq."JJA") then
      if (smonth_entire.gt.6) then
        syear_entire = syear_entire + 1
      end if
      if (emonth_entire.lt.8) then
        eyear_entire = eyear_entire - 1
      end if
    elseif(season.eq."SON") then
      if (smonth_entire.gt.9) then
        syear_entire = syear_entire + 1
      end if
      if (emonth_entire.lt.11) then
        eyear_entire = eyear_entire - 1
      end if
    end if

; ========================================================================
; =========================== calculations ===============================
; ========================================================================

    do idata = 0, ndatasets-1
      data = read_data(info[idata])
      units = data@units
      if (any(units.eq.(/ "Pa", "Pascals" /))) then
        data = data / 100.
        units = "hPa"
        data@units = units
      end if
      time = cd_calendar(data&time, -1)
      data&time = time
      syear = time(0)/100
      smonth = time(0) - syear*100
      eyear = time(dimsizes(time)-1)/100
      emonth = time(dimsizes(time)-1) - eyear*100
      delete(time)
      if (season.eq."monthly") then
        clim = clmMonTLL(data({(syear_eof)*100+1:(eyear_eof)*100+12}, :, :))
        if (smonth.gt.1) then
          syear = syear + 1
        end if
        if (emonth.lt.12) then
          eyear = eyear - 1
        end if
        anom = calcMonAnomTLL(data({syear*100+1:eyear*100+12}, :, :), clim)
        delete(clim)
      else
        if (season.eq."DJF") then
          syear = syear + 1
          if (emonth.lt.2) then
            eyear = eyear - 1
          end if
          anom = data({syear*100+1:eyear*100+1:12}, :, :)
          anom&time = ispan(syear, eyear, 1)
          do yr = syear, eyear
            anom({yr}, :, :) = \
                           dim_avg_n(data({(yr-1)*100+12:yr*100+2}, :, :), 0)
          end do
        elseif(season.eq."MAM") then
          if (smonth.gt.3) then
            syear = syear + 1
          end if
          if (emonth.lt.5) then
            eyear = eyear - 1
          end if
          anom = data({syear*100+4:eyear*100+4:12}, :, :)
          anom&time = ispan(syear, eyear, 1)
          do yr = syear, eyear
            anom({yr}, :, :) = dim_avg_n(data({yr*100+3:yr*100+5}, :, :), 0)
          end do
        elseif(season.eq."JJA") then
          if (smonth.gt.6) then
            syear = syear + 1
          end if
          if (emonth.lt.8) then
            eyear = eyear - 1
          end if
          anom = data({syear*100+7:eyear*100+7:12}, :, :)
          anom&time = ispan(syear, eyear, 1)
          do yr = syear, eyear
            anom({yr}, :, :) = dim_avg_n(data({yr*100+6:yr*100+8}, :, :), 0)
          end do
        elseif(season.eq."SON") then
          if (smonth.gt.9) then
            syear = syear + 1
          end if
          if (emonth.lt.11) then
            eyear = eyear - 1
          end if
          anom = data({syear*100+10:eyear*100+10:12}, :, :)
          anom&time = ispan(syear, eyear, 1)
          do yr = syear, eyear
            anom({yr}, :, :) = dim_avg_n(data({yr*100+9:yr*100+11}, :, :), 0)
          end do
        elseif(season.eq."ANN") then
          if (smonth.gt.1) then
            syear = syear + 1
          end if
          if (emonth.lt.12) then
            eyear = eyear - 1
          end if
          anom = data({syear*100+1:eyear*100+1:12}, :, :)
          anom&time = ispan(syear, eyear, 1)
          do yr = syear, eyear
            anom({yr}, :, :) = dim_avg_n(data({yr*100+1:yr*100+12}, :, :), 0)
          end do
        end if
        anom = anom \
          - conform(anom, \
                    dim_avg_n(anom({syear_eof:eyear_eof}, :, :), 0), (/1, 2/))
      end if
      delete(data)

      log_info(DIAG_SCRIPT + " performing EOF")
      wgt = anom(0, :, :)
      wgt = 1.
      wgt = NewCosWeight(wgt)
      sqrtwgt = wgt
      sqrtwgt = sqrt(wgt)
      Anom = anom
      Anom = anom * conform(anom, sqrtwgt, (/ 1, 2 /))
      ; EOF for syear_eof:eyear_eof only
      if (max(anom&time).gt.9999) then  ; Need check
        EOF = eofunc_n_Wrap(Anom({syear_eof*100+1:eyear_eof*100+12},\
                                 {lat_domain(0):lat_domain(1)}, :), 2, True, 0)
      else
        EOF = eofunc_n_Wrap(Anom({syear_eof:eyear_eof},\
                                 {lat_domain(0):lat_domain(1)}, :), 2, True, 0)
      end if
      ; Adjust sign
      if (hemisphere .eq. "NH") then
        sign = wgt_areaave2(EOF(0, {70.:90.}, :), wgt({70.:90.}, :), 0)
      else
        sign = wgt_areaave2(EOF(0, {-90.:-70.}, :), wgt({-90.:-70.}, :), 0)
      end if

      if (sign.gt.0.) then
        EOF = -EOF
      end if
      ; PC for the entire record
      PC = eofunc_ts_n_Wrap(Anom(:, {lat_domain(0):lat_domain(1)}, :), \
                            EOF, True, 0)
      ; Normalize so that the PC segment over the EOF period is standardized
      if (max(anom&time).gt.9999) then  ; Need check
        PC_eofperiod  = PC(:, {syear_eof*100+1:eyear_eof*100+12})
      else
        PC_eofperiod  = PC(:, {syear_eof:eyear_eof})
      end if
      PC = PC - conform(PC, dim_avg_n(PC_eofperiod, 1), 0)
      PC_eofperiod = PC_eofperiod \
        - conform(PC_eofperiod, dim_avg_n(PC_eofperiod, 1), 0)
      PC = PC / conform(PC, dim_stddev_n(PC_eofperiod, 1), 0)
      PC_eofperiod = PC_eofperiod \
        / conform(PC_eofperiod, dim_stddev_n(PC_eofperiod, 1), 0)

      ; Obtain regression anomalies instead of EOF pattern
      if (max(anom&time).gt.9999) then  ; Need check
        reg = regCoef_n(PC_eofperiod(0, :), \
                        anom({syear_eof*100+1:eyear_eof*100+12}, :, :), 0, 0)
      else
        reg = regCoef_n(PC_eofperiod(0, :), \
                        anom({syear_eof:eyear_eof}, :, :), 0, 0)
      end if
      copy_VarCoords(anom(0, :, :), reg)

      ; Save
      log_info(DIAG_SCRIPT + " saving data")
      Index = PC(0, :)
      ; AM = EOF(0, :, :)
      ; AM = AM / sqrtwgt({lat_domain(0):lat_domain(1)}, :)
      AM = reg
      delete(EOF)
      delete(PC)
      delete(sqrtwgt)
      delete(wgt)
      delete(anom)
      delete(Anom)
      delete(reg)

      if (ave_window.gt.0) then
        if (mod(ave_window, 2).eq.0) then
          wgt = new(ave_window+1, "float")
          wgt = 1.
          wgt(0) = 0.5
          wgt(ave_window) = 0.5
        else
          wgt = new(ave_window, "float")
          wgt = 1.
        end if
        Index = wgt_runave_n_Wrap(Index, wgt, 0, 0)
        delete(wgt)
      end if

      if (hemisphere .eq. "NH") then
        if (names(idata).eq.refname) then
          NAMref = AM
          NAMindexref = Index
        else
          if (isdefined("NAM")) then
            NAM({idata}, :, :) = AM
            if (max(Index&time).gt.9999) then  ; Need check
              NAMindex({idata}, {syear*100+1:eyear*100+12}) = (/ Index /)
            else
              NAMindex({idata}, {syear:eyear}) = (/ Index /)
            end if
          else
            dims = dimsizes(AM)
            NAM = new((/ nmodels, dims(0), dims(1) /), typeof(AM))
            ; Assuming that all models have been regridded to a common grid
            delete(dims)
            NAM!0 = "model"
            NAM&model = imodels
            imodel = 0
            NAM(imodel, :, :) = AM
            if (max(Index&time).gt.9999) then  ; Need check
              time = new((eyear_entire-syear_entire+1)*12, "integer")
              i = 0
              do yr = syear_entire, eyear_entire
                do mo = 1, 12
                  time(i) = yr*100+mo
                  i = i + 1
                end do
              end do
              NAMindex = new((/ nmodels, dimsizes(time) /), typeof(Index))
              NAMindex!0 = "model"
              NAMindex!1 = "time"
              NAMindex&model = imodels
              NAMindex&time = time
              delete(time)
              NAMindex(imodel, {syear*100+1:eyear*100+12}) = (/ Index /)
            else
              time = ispan(syear_entire, eyear_entire, 1)
              NAMindex = new((/ nmodels, dimsizes(time) /), typeof(Index))
              NAMindex!0 = "model"
              NAMindex!1 = "time"
              NAMindex&model = imodels
              NAMindex&time = time
              delete(time)
              NAMindex(imodel, {syear:eyear}) = (/ Index /)
            end if
          end if
        end if
      else
        if (names(idata).eq.refname) then
          SAMref = AM
          SAMindexref = Index
        else
          if (isdefined("SAM")) then
            SAM({idata}, :, :) = AM
            if (max(Index&time).gt.9999) then  ; Need check
              SAMindex({idata}, {syear*100+1:eyear*100+12}) = (/ Index /)
            else
              SAMindex({idata}, {syear:eyear}) = (/ Index /)
            end if
          else
            dims = dimsizes(AM)
            SAM = new((/ nmodels, dims(0), dims(1) /), typeof(AM))
            ; Assuming that all models have been regridded to a common grid
            delete(dims)
            SAM!0 = "model"
            SAM&model = imodels
            imodel = 0
            SAM(imodel, :, :) = AM
            if (max(Index&time).gt.9999) then  ; Need check
              time = new((eyear_entire-syear_entire+1)*12, "integer")
              i = 0
              do yr = syear_entire, eyear_entire
                do mo = 1, 12
                  time(i) = yr*100+mo
                  i = i + 1
                end do
              end do
              SAMindex = new((/ nmodels, dimsizes(time) /), typeof(Index))
              SAMindex!0 = "model"
              SAMindex!1 = "time"
              SAMindex&model = imodels
              SAMindex&time = time
              delete(time)
              SAMindex(imodel, {syear*100+1:eyear*100+12}) = (/ Index /)
            else
              time = ispan(syear_entire, eyear_entire, 1)
              SAMindex = new((/ nmodels, dimsizes(time) /), typeof(Index))
              SAMindex!0 = "model"
              SAMindex!1 = "time"
              SAMindex&model = imodels
              SAMindex&time = time
              delete(time)
              SAMindex(imodel, {syear:eyear}) = (/ Index /)
            end if
          end if
        end if
      end if

      delete(AM)
      delete(Index)
    end do  ; idata
  end do  ; ihemisphere

  ; ========================================================================
  ; ============================= plotting =================================
  ; ========================================================================
  if (write_plots) then
    log_info(DIAG_SCRIPT + " plotting NAM and SAM patterns")
    pattern = True
    pattern@diag_script = DIAG_SCRIPT
    pattern@res = True

    pattern@res_gsnMaximize     = False      ; use full page for the plot
    pattern@res_cnFillOn        = True      ; color plot desired
    pattern@res_cnLineLabelsOn  = False     ; contour lines
    pattern@res_cnLinesOn       = False
    pattern@res_tiMainOn        = False
    pattern@res_mpPerimOn       = False
    pattern@res_mpGridLineColor = -1
    pattern@res_mpGridAndLimbOn = True

    pattern@res_gsnLeftStringFontHeightF = 0.015
    pattern@res_cnLevelSelectionMode     = "ExplicitLevels"
    pattern@res_mpOutlineOn     = True
    pattern@res_mpGeophysicalLineColor = "grey30"
    pattern@res_mpPerimOn       = True
    pattern@res_mpFillOn        = False

    pattern@res_lbLabelBarOn          = False
    pattern@res_gsnRightString        = ""
    pattern@res_mpFillDrawOrder       = "PostDraw"    ; draw map fill last
    pattern@res_cnMissingValFillColor = "Gray"
    pattern@res_tmYLLabelsOn          = False
    pattern@res_tmYLOn                = False
    pattern@res_tmYRLabelsOn          = False
    pattern@res_tmYROn                = False
    pattern@res_tmXBLabelsOn          = False
    pattern@res_tmXBOn                = False
    pattern@res_tmXTLabelsOn          = False
    pattern@res_tmXTOn                = False
    pattern@res_cnInfoLabelOn         = False    ; turn off cn info label

    pattern@res_cnLevelSelectionMode = "ExplicitLevels"
    pattern@res_cnLevels = tofloat(ispan(-5, 5, 1))

    pattern@var = var
    if (.not. isatt(variable_info[mainvarind], "long_name")) then
      variable_info[mainvarind]@long_name = ""
    end if
    if (.not. isatt(variable_info[mainvarind], "units")) then
      variable_info[mainvarind]@units = ""
    end if
    pattern@var_long_name = variable_info[mainvarind]@long_name
    pattern@var_units     = variable_info[mainvarind]@units

    plots = new((/2, 3/), graphic)

  ; --------------------------------------------------------------------
  ; create workspace

    if (isvar("wks")) then
      delete(wks)
    end if

    if (season_nam.eq.season_sam) then
      wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                    "nam_sam_"+refname+"_vs_mme_"+season_nam)
    else
      wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                    "nam_sam_"+refname+"_vs_mme_"+season_nam+"4NAM_"\
                    + season_sam+"4SAM")
    end if
  ; --------------------------------------------------------------------

    pattern@res_gsnDraw       = False  ; do not draw yet
    pattern@res_gsnFrame      = False  ; don't advance frame
    pattern@res_gsnAddCyclic = True

  ; plot contour map
    pattern@res_gsnPolar = "NH"
    pattern@res_mpMinLatF = 20
    copy_VarAtts(pattern, NAMref)
    copy_VarAtts(pattern, NAM)

    pattern@res_gsnPolar = "SH"
    delete(pattern@res_mpMinLatF)
    pattern@res_mpMaxLatF = -20
    copy_VarAtts(pattern, SAMref)
    copy_VarAtts(pattern, SAM)

    diag_script_info@region = "Arctic"
    diag_script_info@min_lat = 20
    plots(0, 0) = contour_map_polar(wks, NAMref, var)
    plots(0, 1) = contour_map_polar(wks, dim_avg_n_Wrap(NAM, 0), var)

    diag_script_info@region = "Antarctic"
    delete(diag_script_info@min_lat)
    diag_script_info@max_lat = -20
    plots(1, 0) = contour_map_polar(wks, SAMref, var)
    plots(1, 1) = contour_map_polar(wks, dim_avg_n_Wrap(SAM, 0), var)

    pres = True
    pres@gsnDraw           = True
    pres@gsnFrame          = False
    pres@gsnPanelLabelBar  = True
    pres@pmLabelBarOrthogonalPosF = 0.0
    pres@lbTitleString = var + " anomaly (" + units + ")"
    pres@lbTitlePosition = "Bottom"
    pres@lbLabelFontHeightF = 0.012
    pres@lbAutoManage    = False
    pres@lbTitleFontHeightF = 0.012
    pres@gsnPanelRight    = 0.5
    pres@gsnPanelYWhiteSpacePercent = 10.
    pres@gsnMaximize = False
    pres@gsnOrientation = "landscape"
    pres@gsnPanelFigureStringsPerimOn     = False
    pres@gsnPanelFigureStringsFontHeightF = 0.01
    pres@amJust                           = "TopLeft"
    pres@amOrthogonalPosF                 = -0.65
    pres@gsnPanelFigureStrings = (/ "(a) NAM ("+refname+")", \
                                    "(b) NAM (MME mean)", \
                                    "(d) SAM ("+refname+")", \
                                    "(e) SAM (MME mean)" /)
    gsn_panel(wks, ndtooned(plots(0:1, 0:1)), (/2, 2/), pres)

    log_info(DIAG_SCRIPT + " plotting NAM and SAM indices")
    res                   = True
    res@xyDashPattern     = 0
    res@xyMonoLineColor   = False
    res@gsnYRefLine       = 0.
    res@xyLineThicknesses = 0.5
    res@xyLineColor       = "grey50"
    res@tmYLLabelFontHeightF = 0.01
    res@gsnDraw           = False
    res@gsnFrame          = False
    range = max((/max(abs(NAMindex)), max(abs(NAMindexref)), \
                 max(abs(SAMindex)), max(abs(SAMindexref))/))
    if (range.gt.4.) then
      res@trYMinF = -5.
      res@trYMaxF = 5.
    elseif(range.gt.3) then
      res@trYMinF = -4.
      res@trYMaxF = 4.
    elseif(range.gt.2) then
      res@trYMinF = -3.
      res@trYMaxF = 3.
    elseif(range.gt.1) then
      res@trYMinF = -2.
      res@trYMaxF = 2.
    else
      res@trYMinF = -1.
      res@trYMaxF = 1.
    end if
    res@tmXTOn = False
    res@tmYROn = False
    res@vpWidthF        = 0.4
    res@vpHeightF       = 0.2
    res@vpXF            = 0.55
    if (max(NAMindex&time).gt.999) then
      NAMtime = new(dimsizes(NAMindex&time), "float")
      i = 0
      do yr = toint(NAMindex&time(0))/100, \
        toint(NAMindex&time(dimsizes(NAMindex&time)-1))/100
        do mo = 1, 12
          NAMtime(i) = tofloat(yr)+tofloat(mo-1)/12.
          i = i + 1
        end do
      end do
      NAMtimeref = new(dimsizes(NAMindexref&time), "float")
      i = 0
      do yr = toint(NAMindexref&time(0))/100, \
        toint(NAMindexref&time(dimsizes(NAMindexref&time)-1))/100
        do mo = 1, 12
          NAMtimeref(i) = tofloat(yr)+tofloat(mo-1)/12.
          i = i + 1
        end do
      end do
    else
      NAMtime = NAMindex&time
      NAMtimeref = NAMindexref&time
    end if
    if (max(SAMindex&time).gt.9999) then
      SAMtime = new(dimsizes(SAMindex&time), "float")
      i = 0
      do yr = toint(SAMindex&time(0))/100, \
        toint(SAMindex&time(dimsizes(SAMindex&time)-1))/100
        do mo = 1, 12
          SAMtime(i) = tofloat(yr)+tofloat(mo-1)/12.
          i = i + 1
        end do
      end do
      SAMtimeref = new(dimsizes(SAMindexref&time), "float")
      i = 0
      do yr = toint(SAMindexref&time(0))/100, \
        toint(SAMindexref&time(dimsizes(SAMindexref&time)-1))/100
        do mo = 1, 12
          SAMtimeref(i) = tofloat(yr)+tofloat(mo-1)/12.
          i = i + 1
        end do
      end do
    else
      SAMtime = SAMindex&time
      SAMtimeref = SAMindexref&time
    end if
    NAMindex@long_name = ""
    SAMindex@long_name = ""
    NAMindexref@long_name = ""
    SAMindexref@long_name = ""
    res@vpYF            = 0.8
    res@tmXBLabelFontHeightF = 0.
    res@tiMainFontHeightF = 0.01
    res@tiMainOffsetYF = -0.008
    res@tiMainString = "(c) NAM index"
    if (ave_window.gt.0) then
      if (season_nam.eq."monthly") then
        res@tiMainString = res@tiMainString+" ("+ave_window+"mon moving ave)"
      else
        res@tiMainString = res@tiMainString+" ("+ave_window+"yr moving ave)"
      end if
    end if
    plots(0, 2) = gsn_csm_xy(wks, NAMtime, NAMindex, res)
    res@vpYF            = 0.52
    res@tmXBLabelFontHeightF = res@tmYLLabelFontHeightF
    res@tiMainString = "(f) SAM index"
    if (ave_window.gt.0) then
      if (season_nam.eq."monthly") then
        res@tiMainString = res@tiMainString+" ("+ave_window+"mon moving ave)"
      else
        res@tiMainString = res@tiMainString+" ("+ave_window+"yr moving ave)"
      end if
    end if
    plots(1, 2) = gsn_csm_xy(wks, SAMtime, SAMindex, res)
    res@xyLineColor = "red"
    delete(res@xyLineThicknesses)
    res@xyLineThicknessF = 2.
    plot = gsn_csm_xy(wks, NAMtimeref, NAMindexref, res)
    overlay(plots(0, 2), plot)
    plot = gsn_csm_xy(wks, SAMtimeref, SAMindexref, res)
    overlay(plots(1, 2), plot)

    res@xyLineColor = "black"
    plot = gsn_csm_xy(wks, NAMtime, dim_avg_n_Wrap(NAMindex, 0), res)
    overlay(plots(0, 2), plot)
    plot = gsn_csm_xy(wks, SAMtime, dim_avg_n_Wrap(SAMindex, 0), res)
    overlay(plots(1, 2), plot)

    delete(NAMtime)
    delete(SAMtime)

    draw(plots(:, 2))
    frame(wks)
    log_info("Wrote " + wks@fullname)

    ; add meta data to plot (for reporting)

    caption = ""
    id = DIAG_SCRIPT + "_" + var + "_" + season

    contrib_authors = "kosa_yu"

  end if  ; if write_plots

  ; ###########################################
  ; # Optional output to netCDF               #
  ; ###########################################

  if (write_nc) then
    NAM@var = var
    SAM@var = var
    NAMref@var = var
    SAMref@var = var
    NAM@diag_script = DIAG_SCRIPT
    SAM@diag_script = DIAG_SCRIPT
    NAMref@diag_script = DIAG_SCRIPT
    SAMref@diag_script = DIAG_SCRIPT

    NAMindex@var = var
    SAMindex@var = var
    NAMindexref@var = var
    SAMindexref@var = var
    NAMindex@diag_script = DIAG_SCRIPT
    SAMindex@diag_script = DIAG_SCRIPT
    NAMindexref@diag_script = DIAG_SCRIPT
    SAMindexref@diag_script = DIAG_SCRIPT

    nc_filename = work_dir + "NAM_pattern_models.nc"
    nc_outfile = ncdf_write(NAM, nc_filename)
    nc_filename = work_dir + "NAM_pattern_ref.nc"
    nc_outfile = ncdf_write(NAMref, nc_filename)
    nc_filename = work_dir + "SAM_pattern_models.nc"
    nc_outfile = ncdf_write(SAM, nc_filename)
    nc_filename = work_dir + "SAM_pattern_ref.nc"
    nc_outfile = ncdf_write(SAMref, nc_filename)

    nc_filename = work_dir + "NAM_index_models.nc"
    nc_outfile = ncdf_write(NAMindex, nc_filename)
    nc_filename = work_dir + "NAM_index_ref.nc"
    nc_outfile = ncdf_write(NAMindexref, nc_filename)
    nc_filename = work_dir + "SAM_index_models.nc"
    nc_outfile = ncdf_write(SAMindex, nc_filename)
    nc_filename = work_dir + "SAM_index_ref.nc"
    nc_outfile = ncdf_write(SAMindexref, nc_filename)
  end if

  leave_msg(DIAG_SCRIPT, "")

end
