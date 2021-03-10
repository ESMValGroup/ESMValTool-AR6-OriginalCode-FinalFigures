; #############################################################################
; ESMValTool CMORizer for HadSLP2
; #############################################################################
;
; Tier
;    Tier 2: other freely-available dataset.
;
; Source
;    https://www.metoffice.gov.uk/hadobs/hadslp2/index.html
;
; Last access
;    20200915
;
; Download and processing instructions
;    Download HadSLP2 and the variance-reduced HadSLP2r
;    Download from
;    https://www.metoffice.gov.uk/hadobs/hadslp2/data/download.html
;    wget https://www.metoffice.gov.uk/hadobs/hadslp2/data/hadslp2.asc.gz
;    wget https://www.metoffice.gov.uk/hadobs/hadslp2/data/HadSLP2r_lowvar_200501-201212.nc
;
; Modification history
;    20200915-yu_kosaka: written.
;
; #############################################################################
loadscript(getenv("esmvaltool_root") + \
           "/esmvaltool/cmorizers/obs/interface.ncl")

begin

  ; Script name (for logger)
  DIAG_SCRIPT = "cmorize_obs_hadslp2.ncl"

  ; Source name
  OBSNAME = "HadSLP2"

  ; Tier
  TIER = 2

  ; Period
  ;YEAR1 = 1850
  ;YEAR2 = 2012

  ; Selected variable (standard name)
  VAR = (/"psl"/)

  ; Name in the raw data
  NAME = (/"Press"/)

  ; Subdirectory
  SUBDIR = (/""/)

  ; Expected units (according to CMOR standard)
  EXPUNITS = (/"Pa"/)

  ; MIP
  MIP = (/"Amon"/)

  ; Frequency
  FREQ = (/"mon"/)

  ; CMOR tables
  CMOR_TABLE = getenv("cmor_tables") + "/cmip5/Tables/CMIP5_" + MIP

  ; Type
  TYPE = "ground"

  ; Version
  VERSION = "2r"

  ; Global attributes
  SOURCE = "https://www.metoffice.gov.uk/hadobs/hadslp2/index.html"
  REF = "Allan, R. J. and Ansell, T. J., J. Climate, " + \
    "doi:10.1175/JCLI3937.1, 2006"
  COMMENT = ""

end

begin

  xsize = 72
  ysize = 37
  
  ; Loop over variables
  vv = 0

  log_info("Processing " + VAR(vv) + " (" + MIP(vv) + ")")

  fname = input_dir_path + "hadslp2.asc"
  lines  = readAsciiTable(fname, 1, "string", 0)

  tsize1 = 12*(2004-1850+1)
  tmp1 = new((/tsize1, ysize, xsize/), "float")
  header = new(2, "integer")
  year1 = new(tsize1, "integer")
  month1 = new(tsize1, "integer")

  n = 0
  do t = 0, tsize1-1
    header = toint(str_split(lines(n, 0), " "))
    year1(t) = header(0)
    month1(t) = header(1)
    n = n + 1
    do y = 0, ysize-1
      tmp1(t, y, :) = tofloat(reshape(str_split(lines(n, 0), " "), (/xsize/)))
      n = n + 1
    end do
  end do
  tmp1 = tmp1 * 0.01
  tmp1@_FillValue = -999.9

  fname = input_dir_path + "HadSLP2r_lowvar_200501-201212.nc"
  f = addfile(fname, "r")
  tmp2 = f->$NAME(vv)$
  dims = dimsizes(tmp2)
  tsize2 = dims(0)*dims(1)
  year2 = new(tsize2, "integer")
  month2 = new(tsize2, "integer")
  yr = 2005
  mo = 1
  do t = 0, tsize2-1
    month2(t) = mo
    year2(t) = yr
    mo = mo + 1
    if (mo.gt.12) then
      mo = mo - 12
      yr = yr + 1
    end if
  end do

  tmp = new((/tsize1+tsize2, ysize, xsize/), "float")
  tmp!0 = "time"
  tmp!1 = "lat"
  tmp!2 = "lon"
  tmp(0:tsize1-1, :, 0:xsize/2-1) = (/tmp1(:, ::-1, xsize/2:)/)
  tmp(0:tsize1-1, :, xsize/2:) = (/tmp1(:, ::-1, 0:xsize/2-1)/)
  t = tsize1
  do yr = year2(0), year2(dimsizes(year2)-1)
    do mo = 1, 12
      tmp(t, :, 0:xsize/2-1) = (/tmp2(yr-year2(0), mo-1, ::-1, xsize/2:)/)
      tmp(t, :, xsize/2:) = (/tmp2(yr-year2(0), mo-1, ::-1, 0:xsize/2-1)/)
      t = t + 1
    end do
  end do
  delete(tmp1)
  delete(tmp2)

  output = tmp
  
  ; Convert units
  output@units = "Pa"
  output = output * 100.

  output!0 = "time"
  output!1 = "lat"
  output!2 = "lon"

  year = array_append_record(year1, year2, 0)
  month = array_append_record(month1, month2, 0)	
  day = month
  day = 1
  hour = day
  hour = 0	
  minute = hour
  second = hour
  time = cd_inv_calendar(year, month, day, hour, minute, second, "days since "+year(0)+"-1-1", 0)  
  lon = tofloat(ispan(0, 355, 5))
  lat = tofloat(ispan(-90, 90, 5))
  output&time = time
  output&lat = lat
  output&lon = lon

  ; Format coordinates
printVarSummary(output)
print(year1(0)+" "+year2(tsize2-1))
  format_coords(output, year1(0) + "0101", year2(tsize2-1) + "1231", FREQ(vv))

  ; Set variable attributes
  tmp = format_variable(output, VAR(vv), CMOR_TABLE(vv))
  delete(output)
  output = tmp
  delete(tmp)

  ; Calculate coordinate bounds
  bounds = guess_coord_bounds(output, FREQ(vv))

  ; Set global attributes
  gAtt = set_global_atts(OBSNAME, TIER, SOURCE, REF, COMMENT)

  ; Output file
  DATESTR = year1(0) + "01-" + year2(tsize2-1) + "12"
  fout = output_dir_path + \
         str_join((/"OBS", OBSNAME, TYPE, VERSION, \
                   MIP(vv), VAR(vv), DATESTR/), "_") + ".nc"

  ; Write variable
  write_nc(fout, VAR(vv), output, bounds, gAtt)
  delete(gAtt)
  delete(output)
  delete(bounds)

end
