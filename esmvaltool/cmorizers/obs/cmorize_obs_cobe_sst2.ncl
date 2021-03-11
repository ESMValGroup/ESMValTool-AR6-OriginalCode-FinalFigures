; #############################################################################
; ESMValTool CMORizer for COBE-SST2 data
; #############################################################################
;
; Tier
;    Tier 2: other freely-available dataset.
;
; Source
;    https://psl.noaa.gov/data/gridded/data.cobe2.html
;
; Last access
;    20210310
;
; Download and processing instructions
;    Download sst.mon.mean.nc
;
; Modification history
;    20210310-kosaka_yu: written.
;
; ############################################################################
loadscript(getenv("esmvaltool_root") + \
"/esmvaltool/cmorizers/obs/interface.ncl")

begin

                                ; Script name (for logger)
  DIAG_SCRIPT = "cmorize_obs_cobe_sst2.ncl"

                                ; Source name
  OBSNAME = "COBE-SST2"

                                ; Tier
  TIER = 2

                                ; Period
  YEAR1 = 1850
  YEAR2 = 2019

                                ; Selected variable (standard name)
  VAR1 = "tos"
  VAR2 = "ts"

                                ; Name in the raw data
  NAME = "sst"

                                ; MIP
  MIP1 = "Omon"
  MIP2 = "Amon"

                                ; Frequency
  FREQ = "mon"

                                ; CMOR table
  CMOR_TABLE1 = getenv("cmor_tables") + "/cmip5/Tables/CMIP5_" + MIP1
  CMOR_TABLE2 = getenv("cmor_tables") + "/cmip5/Tables/CMIP5_" + MIP2

                                ; Type
  TYPE = "reanaly"

                                ; Version
  VERSION = "1"

                                ; Global attributes
  SOURCE = "https://psl.noaa.gov/data/gridded/data.cobe2.html"
  REF = "Hirahara, S., Ishii, M., and Y. Fukuda, 2014. J. Climate, doi: 10.1175/JCLI-D-12-00837.1"
  COMMENT = ""

end

begin

                                ; Loop over variables

  log_info("Processing " + VAR1 + " (" + MIP1 + ")")

                                ; Read variables
  fname = input_dir_path + "sst.mon.mean.nc"
  f = addfile(fname, "r")
  output = f->$NAME$

                                ; Convert units
  if (NAME.eq."sst") then
    output = output + 273.15  ; [degC] --> [K]
    output@units = "K"
  end if

                                ; Extract time period
  date = cd_calendar(output&time, 0)
  idx = ind(date(:, 0).ge.YEAR1 .and. date(:, 0).le.YEAR2)
  output := output(idx, :, :)

                                ; Format coordinates
  output!0 = "time"
  output!1 = "lat"
  output!2 = "lon"
  format_coords(output, YEAR1 + "0101", YEAR2 + "1231", FREQ)

                                ; Calculate coordinate bounds
  bounds = guess_coord_bounds(output, FREQ)

                                ; Set global attributes
  gAtt = set_global_atts(OBSNAME, TIER, SOURCE, REF, COMMENT)

; Output tos

                                ; Set variable attributes
  output1 = format_variable(output, VAR1, CMOR_TABLE1)

                                ; Output file
  DATESTR = YEAR1 + "01-" + YEAR2 + "12"
  fout = output_dir_path + \
  str_join((/"OBS", OBSNAME, TYPE, VERSION, \
  MIP1, VAR1, DATESTR/), "_") + ".nc"

                                ; Write variable
  write_nc(fout, VAR1, output1, bounds, gAtt)
  delete(output1)

; Output ts

  log_info("Processing " + VAR2 + " (" + MIP2 + ")")

                                ; Set variable attributes
  output2 = format_variable(output, VAR2, CMOR_TABLE2)

                                ; Output file
  DATESTR = YEAR1 + "01-" + YEAR2 + "12"
  fout = output_dir_path + \
  str_join((/"OBS", OBSNAME, TYPE, VERSION, \
  MIP2, VAR2, DATESTR/), "_") + ".nc"

                                ; Write variable
  write_nc(fout, VAR2, output2, bounds, gAtt)
  delete(output2)

  delete(gAtt)
  delete(bounds)

end