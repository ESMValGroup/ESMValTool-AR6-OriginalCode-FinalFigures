; #############################################################################
; ESMValTool CMORizer for HadCRUT data
; #############################################################################
;
; Tier
;    Tier 2: other freely-available dataset.
;
; Source
;    https://crudata.uea.ac.uk/cru/data/temperature/
;
; Last access
;    20190208
;
; Download and processing instructions
;    Download the dataset "HadCRUT4" (median temperature anomalies) and
;    the dataset "Absolute" (absolute temperatures for the base period
;    1961-90 on a 5x5 grid).
;
; Caveats
;    In contrast to the HadCRUT3 reformat script which produces temperature
;    anomalies (relative to the 1961-90 climatology), this script calculates
;    absolute tas by adding the climatology ("absolute.nc") to the anomalies
;    ("HadCRUT.4.6.0.0.median.nc"). It creates 2 output, one with the
;    temperature time-series and one with the anomaly time-series
;
; Modification history
;    20200625-bock_lisa: add standard error
;    20190916-righi_mattia: remove no-longer used climatology output.
;    20190229-righi_mattia: added output for anomaly (tasa).
;    20190208-righi_mattia: added output for climatology and adapted to v2.
;    20180222-lauer_axel: bug fix (added swapping of latitudes if needed).
;    20160203-lauer_axel: written.
;
; #############################################################################
loadscript(getenv("esmvaltool_root") + \
           "/esmvaltool/cmorizers/obs/interface.ncl")

begin

  ; Script name (for logger)
  DIAG_SCRIPT = "cmorize_obs_hadcrut4.ncl"

  ; Source name
  OBSNAME = "HadCRUT4"

  ; Tier
  TIER = 2

  ; Period
  YEAR1 = 1850
  MONTH1 = "01"
  DAY1 = "01"
  YEAR2 = 2020
  MONTH2 = "04"
  DAY2 = "30"

  ; Selected variable (standard name)
  VAR = "tas"

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
    "/custom/CMOR_tasStderr.dat"

  ; Version
  VERSION = "1"

  ; Type
  TYPE = "ground"

  ; Global attributes
  SOURCE = "https://crudata.uea.ac.uk/cru/data/temperature/"
  REF =  "morice12jgr"
  COMMENT1 = "Temperature time-series calculated from the anomaly " + \
    "time-series by adding the temperature climatology for 1961-1990"
  COMMENT2 = "Temperature anomaly with respect to the period 1961-1990"
  COMMENT3 = "HadCRUT4 surface temperature ensemble data - square " + \
    "root of covariance matrix diagonal"

end

begin

  ; Read file
  fname1 = input_dir_path + "HadCRUT.4.6.0.0.median.nc"
  fname2 = input_dir_path + "absolute.nc"
  fname3 = input_dir_path + "HadCRUT.4.6.0.0.uncorrelated_supplementary.nc"
  f1 = addfile(fname1, "r")
  setfileoption("nc", "MissingToFillValue", False)
  f2 = addfile(fname2, "r")
  f3 = addfile(fname3, "r")

  ; Read anomaly
  anomaly = f1->temperature_anomaly

  ; Read absolute temperature
  tmp = f2->tem
  clim = tofloat(tmp * tmp@scale_factor) + 273.15
  copy_VarCoords(tmp, clim)
  delete(tmp)

  ; Read standard error
  stderr = f3->standard_error

  ; Swap latitudes
  if (isMonotonic(anomaly&latitude).eq.-1) then
    anomaly = anomaly(:, ::-1, :)
  end if

  if (isMonotonic(clim&lat).eq.-1) then
    clim = clim(:, ::-1, :)
  end if

  if (isMonotonic(stderr&latitude).eq.-1) then
    stderr = stderr(:, ::-1, :)
  end if

  log_info("  Climatology range: " + min(clim) + \
           " K to " + max(clim) + " K")
  log_info("  Anomaly range: " + min(anomaly) + \
           " K to " + max(anomaly) + " K")
  log_info("  Standard error range: " + min(stderr) + \
           " K to " + max(stderr) + " K")

  output1 = anomaly
  output2 = anomaly
  output3 = stderr
  dims = dimsizes(output1)

  ; Add absolute temperature to anomaly
  do yr = 0, dims(0) / 12 - 1
    m1 = yr * 12
    m2 = m1 + 11
    output1(m1:m2, :, :) = where(.not.ismissing(clim), \
                                 anomaly(m1:m2, :, :) + clim, \
                                 tofloat(anomaly@_FillValue))
  end do

  ; Format coordinates
  output1!0 = "time"
  output1!1 = "lat"
  output1!2 = "lon"
  format_coords(output1, YEAR1 + MONTH1 + DAY1, YEAR2 + MONTH2 + DAY2, FREQ)
  output2!0 = "time"
  output2!1 = "lat"
  output2!2 = "lon"
  format_coords(output2, YEAR1 + MONTH1 + DAY1, YEAR2 + MONTH2 + DAY2, FREQ)
  output3!0 = "time"
  output3!1 = "lat"
  output3!2 = "lon"
  format_coords(output3, YEAR1 + MONTH1 + DAY1, YEAR2 + MONTH2 + DAY2, FREQ)

  ; Calculate coordinate bounds
  bounds1 = guess_coord_bounds(output1, FREQ)
  bounds2 = guess_coord_bounds(output2, FREQ)
  bounds3 = guess_coord_bounds(output3, FREQ)

  ; Set variable attributes
  tmp = format_variable(output1, VAR, CMOR_TABLE1)
  delete(output1)
  output1 = tmp
  delete(tmp)
  tmp = format_variable(output2, "tasa", CMOR_TABLE2)
  delete(output2)
  output2 = tmp
  delete(tmp)
  tmp = format_variable(output3, "tasStderr", CMOR_TABLE3)
  delete(output3)
  output3 = tmp
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
  gAtt1 = set_global_atts(OBSNAME, TIER, SOURCE, REF, COMMENT1)
  gAtt2 = set_global_atts(OBSNAME, TIER, SOURCE, REF, COMMENT2)
  gAtt3 = set_global_atts(OBSNAME, TIER, SOURCE, REF, COMMENT3)

  ; Write temperature time-series
  DATESTR = YEAR1 + MONTH1 + "-" + YEAR2 + MONTH2
  fout = output_dir_path + \
    str_join((/"OBS", OBSNAME, TYPE, VERSION, \
               MIP, VAR, DATESTR/), "_") + ".nc"
  write_nc(fout, VAR, output1, bounds1, gAtt1)
  w = addfile(fout, "w")
  w->height = height
  delete(w)
  delete(gAtt1)
  delete(bounds1)
  delete(output1)

  ; Write temperature anomaly time-series
  DATESTR = YEAR1 + MONTH1 + "-" + YEAR2 + MONTH2
  fout = output_dir_path + \
    str_join((/"OBS", OBSNAME, TYPE, VERSION, \
               MIP, "tasa", DATESTR/), "_") + ".nc"
  write_nc(fout, "tasa", output2, bounds2, gAtt2)
  w = addfile(fout, "w")
  delete(w)
  delete(gAtt2)
  delete(bounds2)
  delete(output2)

  ; Write standard_error of temperature anomaly time-series
  DATESTR = YEAR1 + "01-" + YEAR2 + "12"
  fout = output_dir_path + \
    str_join((/"OBS", OBSNAME, TYPE, VERSION, \
               MIP, "tasStderr", DATESTR/), "_") + ".nc"
  write_nc(fout, "tasStderr", output3, bounds3, gAtt3)
  w = addfile(fout, "w")
  delete(w)
  delete(gAtt3)
  delete(bounds3)
  delete(output3)

end
