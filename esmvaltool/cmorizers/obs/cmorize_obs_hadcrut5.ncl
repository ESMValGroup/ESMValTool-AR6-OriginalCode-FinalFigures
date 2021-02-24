; #############################################################################
; ESMValTool CMORizer for HadCRUT data
; #############################################################################
;
; Tier
;    Tier 2: other freely-available dataset.
;
; Source
;
; Last access
;    20210207
;
; Download and processing instructions
;
; Caveats
;
; Modification history
;    20210207-kosaka_yu: updated for v5.0.1.0
;    20191223-kosaka_yu: written.
;
; #############################################################################
loadscript(getenv("esmvaltool_root") + \
           "/esmvaltool/cmorizers/obs/interface.ncl")

begin

  ; Script name (for logger)
  DIAG_SCRIPT = "cmorize_obs_hadcrut5.ncl"

  ; Source name
  OBSNAME = "HadCRUT5"

  ; Tier
  TIER = 2

  ; Period
  YEAR1 = 1850
  YEAR2 = 2020

  ; Selected variable (standard name)
  VAR1 = "tas"
  VAR2 = "tasa"
  VAR3 = "tasaga"
  
  ; MIP
  MIP = "Amon"

  ; Frequency
  FREQ = "mon"

  ; CMOR table
  CMOR_TABLE1 = getenv("cmor_tables") + \
    "/cmip5/Tables/CMIP5_Amon"
  CMOR_TABLE2 = getenv("cmor_tables") + \
    "/custom/CMOR_tasa.dat"
  CMOR_TABLE3 = getenv("cmor_tables") + \
    "/custom/CMOR_tasaga.dat"

  ; Version
  VERSION = "5.0.1.0"

  ; Type
;  TYPE1 = "ground"
;  TYPE2 = "ground"

  ; Global attributes
  SOURCE = "https://crudata.uea.ac.uk/cru/data/temperature/"
  REF1 =  "Morice, C. P. et al. 2021, J. Geophys. Res. Atmosphere, doi: 10.1029/2019JD032361"
  REF2 =  REF1
  REF3 =  REF1
  COMMENT1 = "Temperature time-series calculated from the anomaly " + \
    "time-series by adding the temperature climatology for 1961-1990"
  COMMENT2 = "Temperature anomaly with respect to the period 1961-1990"
  COMMENT3 = "Global mean temperature anomaly with respect to the period 1961-1990"

  nens = 200
  len = (YEAR2 - YEAR1 + 1)*12
end

begin

  ; Read absolute temperature
;  fname2 = input_dir_path + "absolute.nc"
;  setfileoption("nc", "MissingToFillValue", False)
;  f2 = addfile(fname2, "r")
    
;  tmp = f2->tem
;  clim = tofloat(tmp * tmp@scale_factor) + 273.15
;  copy_VarCoords(tmp, clim)
;  delete(tmp)

  ; Swap latitudes
;  if (isMonotonic(clim&lat).eq.-1) then
;    clim = clim(:, ::-1, :)
;  end if

;  log_info("  Climatology range: " + min(clim) + \
;           " K to " + max(clim) + " K")

  do ens = 0, nens
    ; Read anomaly
    if (ens.eq.0) then
      fname1 = input_dir_path + "HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc"
      vname  = "tas_mean"
    else
      fname1 = input_dir_path + "ensemble_members/HadCRUT.5.0.1.0.analysis.anomalies."+ens+".nc"
      vname  = "tas"
    end if
    setfileoption("nc", "MissingToFillValue", True)
    f1 = addfile(fname1, "r")

    anomaly = f1->$vname$(0:len-1, :, :)

  ; Swap latitudes
    if (isMonotonic(anomaly&latitude).eq.-1) then
      anomaly = anomaly(:, ::-1, :)
    end if

    log_info("  Anomaly range: " + min(anomaly) + \
             " K to " + max(anomaly) + " K")

    output1 = tofloat(anomaly)
    copy_VarMeta(anomaly, output1)
    output2 = output1
    dims = dimsizes(output1)

  ; Add absolute temperature to anomaly
;    do yr = 0, dims(0) / 12 - 1
;      m1 = yr * 12
;      m2 = m1 + 11
;      output1(m1:m2, :, :) = where(.not.ismissing(clim), \
;                                   anomaly(m1:m2, :, :) + clim, \
;                                   tofloat(anomaly@_FillValue))
;    end do

  ; Format coordinates
    output1!0 = "time"
    output1!1 = "lat"
    output1!2 = "lon"
    format_coords(output1, YEAR1 + "0101", YEAR2 + "1231", FREQ)
    output2!0 = "time"
    output2!1 = "lat"
    output2!2 = "lon"
    format_coords(output2, YEAR1 + "0101", YEAR2 + "1231", FREQ)

    if (ens.eq.0) then
      time_save = output1&time
    end if
    
  ; Calculate coordinate bounds
    bounds1 = guess_coord_bounds(output1, FREQ)
    bounds2 = guess_coord_bounds(output2, FREQ)

  ; Set variable attributes
    tmp = format_variable(output1, VAR1, CMOR_TABLE1)
    delete(output1)
    output1 = tmp
    delete(tmp)
    tmp = format_variable(output2, VAR2, CMOR_TABLE2)
    delete(output2)
    output2 = tmp
    delete(tmp)

  ; Add height coordinate
    output1@coordinates = "height"
    height = 2.d
    height!0 = "ncl_scalar"
    height@units = "m"
    height@axis = "Z"
    height@positive = "up"
    height@long_name = "height"
    height@standard_name = "height"

  ; Set global attributes
    gAtt1 = set_global_atts(OBSNAME, TIER, SOURCE, REF1, COMMENT1)
    gAtt2 = set_global_atts(OBSNAME, TIER, SOURCE, REF2, COMMENT2)

    if (ens.eq.0) then
      ENS = "mean"
    else
      ENS = "ens"+sprinti("%0.3i", ens)
    end if
    
  ; Write temperature time-series
;    DATESTR = YEAR1 + "01-" + YEAR2 + "12"
;    fout = output_dir_path + \
;    str_join((/"OBS", OBSNAME, ENS, VERSION, \
;               MIP, VAR1, DATESTR/), "_") + ".nc"
;    write_nc(fout, VAR1, output1, bounds1, gAtt1)
;    w = addfile(fout, "w")
;    w->height = height
;    delete(w)
;    delete(gAtt1)
;    delete(bounds1)
;    delete(output1)

  ; Write temperature anomaly time-series
    DATESTR = YEAR1 + "01-" + YEAR2 + "12"
    fout = output_dir_path + \
    str_join((/"OBS", OBSNAME, ENS, VERSION, \
               MIP, VAR2, DATESTR/), "_") + ".nc"
    write_nc(fout, VAR2, output2, bounds2, gAtt2)
    w = addfile(fout, "w")
    delete(w)
    delete(gAtt2)
    delete(bounds2)
    delete(output2)

    delete(anomaly)
  end do
  
end

begin

  ; Read file
  do ens = 0, nens
    if (ens.eq.0) then
      fname = input_dir_path + "HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.nc"
      vname = "tas_mean"
      f = addfile(fname, "r")
      anomaly = f->$vname$(0:len-1)
    else
      fname = input_dir_path + "HadCRUT.5.0.1.0.analysis.ensemble_series.global.monthly.nc"
      vname = "tas"
      f = addfile(fname, "r")
      anomaly = f->$vname$(ens-1, 0:len-1)
    end if
  
    log_info("  Anomaly range: " + min(anomaly) + \
             " K to " + max(anomaly) + " K")

    output = tofloat(anomaly)
             
    output!0 = "time"
    output&time = time_save

  ; Format coordinates
    format_coords(output, YEAR1 + "0101", YEAR2 + "1231", FREQ)
    
  ; Calculate coordinate bounds
    bounds = guess_coord_bounds(output, FREQ)
    print(bounds)
    
  ; Set variable attributes
    tmp = format_variable(output, VAR3, CMOR_TABLE3)
    delete(output)
    output = tmp
    delete(tmp)

  ; Add height coordinate
    output@coordinates = "height"
    height = 2.d
    height!0 = "ncl_scalar"
    height@units = "m"
    height@axis = "Z"
    height@positive = "up"
    height@long_name = "height"
    height@standard_name = "height"

  ; Set global attributes
    gAtt = set_global_atts(OBSNAME, TIER, SOURCE, REF3, COMMENT3)

  ; Write temperature anomaly time-series
    DATESTR = YEAR1 + "01-" + YEAR2 + "12"
    if (ens.eq.0) then
      ENS = "mean"
    else
      ENS = "ens"+sprinti("%0.3i", ens)
    end if
    fout = output_dir_path + \
    str_join((/"OBS", OBSNAME, ENS, VERSION, \
               MIP, VAR3, DATESTR/), "_") + ".nc"
    write_nc(fout, VAR3, output, bounds, gAtt)
    w = addfile(fout, "w")
    w->height = height
    delete(w)
    delete(gAtt)
    delete(bounds)
    delete(output)
  end do

end
