; #############################################################################
; ESMValTool CMORizer for BerkeleyEarth data
; #############################################################################
;
; Tier
;    Tier 2: other freely-available dataset.
;
; Source
;    http://berkeleyearth.org/data/
;
; Last access
;    20210207
;
; Download and processing instructions
;    Download "Gridded Data -> Monthly Land + Ocean
;                -> 1deg x 1deg Latitude-Longitude Grid"
;            and
;             "Time Series Data -> Land + Ocean (1850 - Recent)
;                -> Monthly Global Average Temperature"
;
; Caveats
;    Produces temperature anomalies (relative to the 1951-80 climatology)
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
  DIAG_SCRIPT = "cmorize_obs_berkeleyearth.ncl"

  ; Source name
  OBSNAME = "BerkeleyEarth"

  ; Tier
  TIER = 2

  ; Period
  YEAR1 = 1850
  YEAR2 = 2020

  ; Selected variable (standard name)
  VAR1 = "tas"
  VAR2 = "tasa"
  VAR3 = "tasaga"

  ; MIPS
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
  VERSION = "1"

  ; Type
  TYPE = "ground"
  
  ; Global attributes
  SOURCE = "http://berkeleyearth.org"
  REF1 =  "Rohde, et al., " + \
          "Geoinformatics & Geostatistics: An Overview, " +\
         "doi:10.4172/2327-4581.1000101, 2013"
  REF2 =  REF1
  REF3 =  REF1
  COMMENT1 = "Temperature time-series calculated from the anomaly " + \
  "time-series by adding the temperature climatology for 1951-1980"
  COMMENT2 = "Temperature anomaly with respect to the period 1951-1980"
  COMMENT3 = "Global mean temperature anomaly with respect to the period 1951-1980"

end

begin

  ; Read file
  fname = input_dir_path + "Land_and_Ocean_LatLong1.nc"
  f = addfile(fname, "r")    
  ; Read anomaly
  len = (YEAR2 - YEAR1 + 1)*12
  anomaly = f->temperature(0:len-1, :, :)
  anomaly@_FillValue = default_fillvalue(typeof(anomaly))
  replace_ieeenan(anomaly, anomaly@_FillValue, 0)
  
  anomaly!1 = "lat"
  anomaly!2 = "lon"
  time := anomaly&time
  time@units = "years since 0000-1-1"
  time := cd_convert(time, "days since 1950-1-1")
  anomaly&time = time
  
  
  ; Read climatology
  clim = f->climatology
  clim!1 = "lat"
  clim!2 = "lon"
  
  ; Swap latitudes
  if (isMonotonic(anomaly&lat).eq.-1) then
    anomaly = anomaly(:, ::-1, :)
  end if
  if (isMonotonic(clim&lat).eq.-1) then
    clim = clim(:, ::-1, :)
  end if

  log_info("  Anomaly range: " + min(anomaly) + \
           " K to " + max(anomaly) + " K")

  output1 = anomaly
  output2 = anomaly
  dims = dimsizes(output1)

  ; Add absolute temperature to anomaly
  do yr = 0, dims(0) / 12 - 1
    m1 = yr * 12
    m2 = m1 + 11
    output1(m1:m2, :, :) = where(.not.ismissing(clim), \
                                 anomaly(m1:m2, :, :) + clim, \
                                 anomaly@_FillValue)
  end do
  
  ; Format coordinates
  output1!0 = "time"
  output1!1 = "lat"
  output1!2 = "lon"
  format_coords(output1, YEAR1 + "0101", YEAR2 + "1231", FREQ)
  output2!0 = "time"
  output2!1 = "lat"
  output2!2 = "lon"
  format_coords(output2, YEAR1 + "0101", YEAR2 + "1231", FREQ)
  
  time_save = output1&time
  
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
    
  ; Write temperature time-series
  DATESTR = YEAR1 + "01-" + YEAR2 + "12"
  fout = output_dir_path + \
  str_join((/"OBS", OBSNAME, TYPE, VERSION, \
           MIP, VAR1, DATESTR/), "_") + ".nc"
  write_nc(fout, VAR1, output1, bounds1, gAtt1)
  w = addfile(fout, "w")
  w->height = height
  delete(w)
  delete(gAtt1)
  delete(output1)
  delete(bounds1)

  fout = output_dir_path + \
  str_join((/"OBS", OBSNAME, TYPE, VERSION, \
           MIP, VAR2, DATESTR/), "_") + ".nc"
  write_nc(fout, VAR2, output2, bounds2, gAtt2)
  w = addfile(fout, "w")
  w->height = height
  delete(w)
  delete(gAtt2)
  delete(output2)
  delete(bounds2)
    
end

begin

  ; Read file
  fname = input_dir_path + "Land_and_Ocean_complete.txt"
  tmp = readAsciiTable(fname, 1, "string", (/77, 0/))
  data = tmp(:, 0)
  delete(tmp)
  output3 = new(len, "float")
  do i = 0, len-1
    tmp = str_split(data(i), " ")
    year = toint(tmp(0))
    month = toint(tmp(1))
    if (YEAR1.le.year.and.year.le.YEAR2) then
      if (is_string_numeric(tmp(2))) then
        output3((year-YEAR1)*12+month-1) = tofloat(tmp(2))
      end if
    end if
    delete(tmp)
  end do
  output3!0 = "time"
  output3&time = time_save
  delete(data)
  
  log_info("  Anomaly range: " + min(output3) + \
           " K to " + max(output3) + " K")

  output3!0 = "time"
  output3&time = time_save

  ; Format coordinates
  format_coords(output3, YEAR1 + "0101", YEAR2 + "1231", FREQ)
    
  ; Calculate coordinate bounds
  bounds = guess_coord_bounds(output3, FREQ)
    
  ; Set variable attributes
  tmp = format_variable(output3, VAR3, CMOR_TABLE3)
  delete(output3)
  output3 = tmp
  delete(tmp)

  ; Add height coordinate
  output3@coordinates = "height"
  height = 2.d
  height!0 = "ncl_scalar"
  height@units = "m"
  height@axis = "Z"
  height@positive = "up"
  height@long_name = "height"
  height@standard_name = "height"

  ; Set global attributes
  gAtt3 = set_global_atts(OBSNAME, TIER, SOURCE, REF3, COMMENT3)

  ; Write temperature anomaly time-series
  DATESTR = YEAR1 + "01-" + YEAR2 + "12"
  fout = output_dir_path + \
  str_join((/"OBS", OBSNAME, TYPE, VERSION, \
           MIP, VAR3, DATESTR/), "_") + ".nc"
  write_nc(fout, VAR3, output3, bounds, gAtt3)
  w = addfile(fout, "w")
  w->height = height
  delete(w)
  delete(gAtt3)
  delete(output3)

end
