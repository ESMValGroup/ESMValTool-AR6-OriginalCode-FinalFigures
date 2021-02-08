; #############################################################################
; ESMValTool CMORizer for GISTEMP data
; #############################################################################
;
; Tier
;    Tier 2: other freely-available dataset.
;
; Source
;    https://data.giss.nasa.gov/gistemp/
;
; Last access
;    20210207
;
; Download and processing instructions
;    Download "Global-mean monthly, seasonal, and annual means
;                -> CSV"
;            and
;             "Gridded Monthly Temperature Anomaly Data
;                -> Compressed NetCDF Files (regular 2x2x2 gr
;                   -> Land-Ocean Temperature Index, ERSSTv5, 1200km smoothing"  
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
  DIAG_SCRIPT = "cmorize_obs_gistemp.ncl"

  ; Source name
  OBSNAME = "GISTEMP"

  ; Tier
  TIER = 2

  ; Period
  YEAR1 = 1880
  YEAR2 = 2020

  ; Selected variable (standard name)
  VAR1 = "tasa"
  VAR2 = "tasaga"

  ; MIPS
  MIP = "Amon"

    ; Frequency
  FREQ = "mon"

  ; CMOR table
  CMOR_TABLE1 = getenv("cmor_tables") + \
    "/custom/CMOR_tasa.dat"
  CMOR_TABLE2 = getenv("cmor_tables") + \
    "/custom/CMOR_tasaga.dat"

  ; Version
  VERSION = "v4"

  ; Type
  TYPE = "ground"
  
  ; Global attributes
  SOURCE = "https://data.giss.nasa.gov/gistemp/"
  REF1 =  "Lenssen et al., J. Geophys. Res., doi:10.1029/2018JD029522, 2019"
  REF2 =  REF1
  COMMENT1 = "Temperature anomaly with respect to the period 1951-1980"
  COMMENT2 = "Global mean temperature anomaly with respect to the period 1951-1980"

end

begin

  ; Read file
  fname = input_dir_path + "gistemp1200_GHCNv4_ERSSTv5.nc"
  f = addfile(fname, "r")
    
  ; Read anomaly
  time = f->time
  yyyymm = cd_calendar(time, -1)
  ts = ind(yyyymm.eq.YEAR1*100+1)
  te = ind(yyyymm.eq.YEAR2*100+12)
  anomaly = short2flt(f->tempanomaly(ts:te, :, :))
  FillVal = tofloat(anomaly@_FillValue)
  delete(anomaly@_FillValue)
  delete(anomaly@scale_factor)
  anomaly@_FillValue = 1.e+22
  anomaly = where(anomaly.eq.FillVal, anomaly@_FillValue, anomaly)

  ; Swap latitudes
  if (isMonotonic(anomaly&lat).eq.-1) then
    anomaly = anomaly(:, ::-1, :)
  end if

  log_info("  Anomaly range: " + min(anomaly) + \
           " K to " + max(anomaly) + " K")

  output1 = anomaly
  dims = dimsizes(output1)

  ; Format coordinates
  output1!0 = "time"
  output1!1 = "lat"
  output1!2 = "lon"
  format_coords(output1, YEAR1 + "0101", YEAR2 + "1231", FREQ)
    
  ; Format time coordinate
;  ctime = time_attrib(output1&time, YEAR1, YEAR2, "M")
;  delete(output1&time)
;  output1&time = ctime
;  delete(ctime)
  time_save = output1&time
  
  ; Calculate coordinate bounds
  bounds = guess_coord_bounds(output1, FREQ)

  ; Set variable attributes
  tmp = format_variable(output1, VAR1, CMOR_TABLE1)
  delete(output1)
  output1 = tmp
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
    
  ; Write temperature time-series
  DATESTR = YEAR1 + "01-" + YEAR2 + "12"
  fout = output_dir_path + \
  str_join((/"OBS", OBSNAME, TYPE, VERSION, \
           MIP, VAR1, DATESTR/), "_") + ".nc"
  write_nc(fout, VAR1, output1, bounds, gAtt1)
  w = addfile(fout, "w")
  w->height = height
  delete(w)
  delete(gAtt1)
  delete(output1)
  delete(bounds)
    
end

begin

  ; Read file
  fname = input_dir_path + "GLB.Ts+dSST.csv"
  tmp = readAsciiTable(fname, 1, "string", (/2, 1/))
  data = tmp(:, 0)
  delete(tmp)
  output2 = new((YEAR2 - YEAR1 + 1)*12, "float")
  do i = 0, dimsizes(data)-1
    tmp = str_split(data(i), ",")
    year = toint(tmp(0))
    if (YEAR1.le.year.and.year.le.YEAR2) then
      do month = 1, 12
        if (is_string_numeric(tmp(month))) then
          output2((year-YEAR1)*12+month-1) = tofloat(tmp(month))
        end if
      end do
    end if
    delete(tmp)
  end do
  output2!0 = "time"
  output2&time = time_save
  delete(data)
  
  log_info("  Anomaly range: " + min(output2) + \
           " K to " + max(output2) + " K")

  output2!0 = "time"
  output2&time = time_save

  ; Format coordinates
  format_coords(output2, YEAR1 + "0101", YEAR2 + "1231", FREQ)
    
  ; Calculate coordinate bounds
  bounds = guess_coord_bounds(output2, FREQ)
    
  ; Set variable attributes
  tmp = format_variable(output2, VAR2, CMOR_TABLE2)
  delete(output2)
  output2 = tmp
  delete(tmp)

  ; Add height coordinate
  output2@coordinates = "height"
  height = 2.d
  height!0 = "ncl_scalar"
  height@units = "m"
  height@axis = "Z"
  height@positive = "up"
  height@long_name = "height"
  height@standard_name = "height"

  ; Set global attributes
  gAtt2 = set_global_atts(OBSNAME, TIER, SOURCE, REF2, COMMENT2)

  ; Write temperature anomaly time-series
  DATESTR = YEAR1 + "01-" + YEAR2 + "12"
  fout = output_dir_path + \
  str_join((/"OBS", OBSNAME, TYPE, VERSION, \
           MIP, VAR2, DATESTR/), "_") + ".nc"
  write_nc(fout, VAR2, output2, bounds, gAtt2)
  w = addfile(fout, "w")
  w->height = height
  delete(w)
  delete(gAtt2)
  delete(bounds)
  delete(output2)

end