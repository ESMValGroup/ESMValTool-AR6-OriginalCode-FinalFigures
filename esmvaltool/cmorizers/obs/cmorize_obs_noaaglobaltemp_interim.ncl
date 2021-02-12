; #############################################################################
; ESMValTool CMORizer for NOAAGlobalTemp-Interim data
; #############################################################################
;
; Tier
;    Tier 2: other freely-available dataset.
;
; Source
;    
;
; Last access
;    20201104
;
; Download and processing instructions
;    Download 
;    ftp://ftp.ncdc.noaa.gov/pub/data/cmb/ersst/v5/2020.grl.dat/interim/temp.ano.merg5.asc
;    
; Caveats
;    Produces temperature anomalies
;
; Modification history
;    20201105 kosaka_yu: written.
;
; #############################################################################
loadscript(getenv("esmvaltool_root") + \
           "/esmvaltool/cmorizers/obs/interface.ncl")

begin

  ; Script name (for logger)
  DIAG_SCRIPT = "cmorize_obs_noaaglobaltemp_interim.ncl"

  ; Source name
  OBSNAME = "NOAAGlobalTemp-Interim"

  ; Tier
  TIER = 2

  ; Period
  YEAR1 = 1850
  YEAR2 = 2018

  ; Selected variable (standard name)
  VAR1 = "tasa"
;  VAR2 = "tasaga"

  ; MIPS
  MIP = "Amon"

    ; Frequency
  FREQ = "mon"

  ; CMOR table
  CMOR_TABLE1 = getenv("cmor_tables") + \
    "/custom/CMOR_tasa.dat"
;  CMOR_TABLE2 = getenv("cmor_tables") + \
;    "/custom/CMOR_tasaga.dat"

  ; Version
  VERSION = "v5"

  ; Type
  TYPE = "ground"
  
  ; Global attributes
  SOURCE = ""
  REF1 =  ""
;  REF2 =  REF1
  COMMENT1 = "Temperature anomaly"  ; with respect to the period 1971-2000"
;  COMMENT2 = "Global mean temperature anomaly" ; with respect to the period 1971-2000"

end

begin

  ; Read file
  fname = input_dir_path + "temp.ano.merg5.asc"

  ; Read anomaly
;  time = f->time
;  yyyymm = cd_calendar(time, -1)
;  ts = ind(yyyymm.eq.YEAR1*100+1)
;  te = ind(yyyymm.eq.YEAR2*100+12)
;  anomaly = f->anom(ts:te, 0, :, :)

  xsize = 72
  ysize = 36
  lon = fspan(2.5, 357.5, xsize)
  lon@long_name = "longitude"
  lon@units = "degrees_east"
  lat = fspan(-87.5, 87.5, ysize)
  lat@long_name = "latitude"
  lat@units = "degrees_north"
  tsize = (YEAR2 - YEAR1 + 1)*12
  yyyymm = yyyymm_time(YEAR1, YEAR2, "integer")
  year = toint(yyyymm/100)
  month = toint(yyyymm)-year*100
  day = year
  hour = year
  minute = year
  second = year
  day = 1
  hour = 0
  minute = 0
  second = 0
  units = "days since "+YEAR1+"-1-1 00:00:00"
  opt = 0
  opt@calendar = "standard"
  time = cd_inv_calendar(year, month, day, hour, minute, second, units, opt)

  anomaly = new((/tsize, ysize, xsize/), "float")
  anomaly!0 = "time"
  anomaly!1 = "lat"
  anomaly!2 = "lon"
  anomaly&time = time
  anomaly&lat = lat
  anomaly&lon = lon

  tmp = readAsciiTable(fname, 1, "string", 0)
  dims = dimsizes(tmp)
  nraws = dims(0)
  n = 0
  do while (n.lt.nraws)
    if (str_fields_count(tmp(n, 0), " ").eq.2) then
      yr = toint(str_get_field(tmp(n, 0), 2, " "))
      mo = toint(str_get_field(tmp(n, 0), 1, " "))
      t = ind(yyyymm.eq.yr*100+mo)

      n = n + 1
      y = 0
      do while (n.lt.nraws.and.str_fields_count(tmp(n, 0), " ").gt.2)
        if (YEAR1.le.yr .and. yr.le.YEAR2) then
          anomaly(t, y, :) = tofloat(str_split(tmp(n, 0), " "))
        end if
        y = y + 1
        n = n + 1
      end do
    end if
  end do
  delete(tmp)
  delete(dims)

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
