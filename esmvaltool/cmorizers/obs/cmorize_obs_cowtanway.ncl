; #############################################################################
; ESMValTool CMORizer for Cowtan&Way data
; #############################################################################
;
; Tier
;    Tier 2: other freely-available dataset.
;
; Source
;    https://www-users.york.ac.uk/~kdc3/papers/coverage2013/series.html
;
; Last access
;    20210207
;
; Download and processing instructions
;    Download the Version 2.0 long reconstruction -> HadCRUT4 infilled by kriging
;    -> Download "Global mean/uncertainty: monthly",
;                "Gridded data: netcdf" and "Ensmble: monthly"
;
; Modification history
;    20210207-kosaka_yu: updated to 2020.
;    20191101-kosaka_yu: written.
;
; #############################################################################
loadscript(getenv("esmvaltool_root") + \
           "/esmvaltool/cmorizers/obs/interface.ncl")

begin

  ; Script name (for logger)
  DIAG_SCRIPT = "cmorize_obs_cowtanway.ncl"

  ; Source name
  OBSNAME = "CowtanWay"

  ; Tier
  TIER = 2

  ; Period
  YEAR1 = 1850
  YEAR2 = 2020

  ; Selected variable (standard name)
  VAR1 = "tasa"
  VAR2 = "tasaga"

  ; MIP
  MIP = "Amon"

  ; Frequency
  FREQ = "mon"

  ; CMOR table
  CMOR_TABLE1= getenv("cmor_tables") + \
    "/custom/CMOR_tasa.dat"
  CMOR_TABLE2= getenv("cmor_tables") + \
    "/custom/CMOR_tasaga.dat"
  
  ; Version
  VERSION = "version2.0"

  ; Type
;  TYPE1 = "ground"
;  TYPE2 = "ground"

  ; Global attributes
  SOURCE = "https://www-users.york.ac.uk/~kdc3/papers/coverage2013/series.html"
  REF1 =  "Cowtan and Way, Quart. J. Roy. Met. Soc., doi:10.1002/qj.2297, 2014"
  REF2 =  REF1
  COMMENT1 = "Temperature anomaly with respect to the period 1961-1990, ensemble median, infilled by kriging"
  COMMENT2 = "Temperature anomaly with respect to the period 1961-1990, infilled by kriging, global average"

end

begin

  ; Read file
  fname = input_dir_path + "had4_krig_v2_0_0.nc"
  f = addfile(fname, "r")

  ; Read anomaly
  len = (YEAR2 - YEAR1 + 1)*12
  anomaly = f->temperature_anomaly(0:len-1, :, :)

  ; Swap latitudes
  if (isMonotonic(anomaly&latitude).eq.-1) then
    anomaly = anomaly(:, ::-1, :)
  end if

  log_info("  Anomaly range: " + min(anomaly) + \
           " K to " + max(anomaly) + " K")

  output = anomaly
  dims = dimsizes(output)
  delete(anomaly)
  
  ; Format coordinates
  output!0 = "time"
  output!1 = "lat"
  output!2 = "lon"
  format_coords(output, YEAR1 + "0101", YEAR2 + "1231", FREQ)
  time_save = output&time
  
  ; Calculate coordinate bounds
  bounds = guess_coord_bounds(output, FREQ)

  ; Set variable attributes
  tmp = format_variable(output, VAR1, CMOR_TABLE1)
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
  gAtt = set_global_atts(OBSNAME, TIER, SOURCE, REF1, COMMENT1)

  ; Write temperature anomaly time-series
  DATESTR = YEAR1 + "01-" + YEAR2 + "12"
  fout = output_dir_path + \
    str_join((/"OBS", OBSNAME, "median", VERSION, \
               MIP, VAR1, DATESTR/), "_") + ".nc"
  write_nc(fout, VAR1, output, bounds, gAtt)
  w = addfile(fout, "w")
  w->height = height
  delete(w)
  delete(gAtt)
  delete(bounds)
  delete(output)

; ============================================================

  ; Read file
  nens = 100
  anomaly = new((/len, nens+1/), "float")
  fname = input_dir_path + "had4_krig_v2_0_0.txt"
  tmp = asciiread(fname, (/len, 5/), "float")
  anomaly(:, 0) = tmp(:, 1)
  delete(tmp)
  
  fname = input_dir_path + "had4_krig_ensemble_v2_0_0.txt"
  tmp = asciiread(fname, (/len, nens+1/), "float")
  anomaly(:, 1:100) = tmp(:, 1:100)
  delete(tmp)
  
  log_info("  Anomaly range: " + min(anomaly) + \
           " K to " + max(anomaly) + " K")

  anomaly!0 = "time"
  anomaly&time = time_save
           
  do ens = 0, 100
    output = anomaly(:, ens)

  ; Format coordinates
    format_coords(output, YEAR1 + "0101", YEAR2 + "1231", FREQ)
    
  ; Calculate coordinate bounds
    bounds = guess_coord_bounds(output, FREQ)
    print(bounds)
    
  ; Set variable attributes
    tmp = format_variable(output, VAR2, CMOR_TABLE2)
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
    gAtt = set_global_atts(OBSNAME, TIER, SOURCE, REF2, COMMENT2)

  ; Write temperature anomaly time-series
    DATESTR = YEAR1 + "01-" + YEAR2 + "12"
    if (ens.eq.0) then
      ENS = "median"
    else
      ENS = "ens"+sprinti("%0.3i", ens)
    end if
    fout = output_dir_path + \
    str_join((/"OBS", OBSNAME, ENS, VERSION, \
               MIP, VAR2, DATESTR/), "_") + ".nc"
    write_nc(fout, VAR2, output, bounds, gAtt)
    w = addfile(fout, "w")
    w->height = height
    delete(w)
    delete(gAtt)
    delete(bounds)
    delete(output)
  end do

end
