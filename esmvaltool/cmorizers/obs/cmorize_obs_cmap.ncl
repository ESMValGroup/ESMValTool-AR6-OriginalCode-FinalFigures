; #############################################################################
; REFORMAT SCRIPT FOR CMAP DATA
; #############################################################################
;
; Tier
;    Tier 2: other freely-available dataset.
;
; Source
;    https://www.esrl.noaa.gov/psd/data/gridded/data.cmap.html
;
; Last access
;    2019/10/11
;
; Download and processing instructions
;    Download "precip.mon.mean.nc" of Standard Monthly
;
; Modification history
;    20190226 kosaka_yu: written
;    20191011 kosaka_yu: updated for ESMValTool 2.0.0b2
;
; ############################################################################
loadscript(getenv("esmvaltool_root") + \
           "/esmvaltool/cmorizers/obs/interface.ncl")


begin

  ; Script name (for logger)
  DIAG_SCRIPT = "cmorize_obs_cmap.ncl"

  ; Source name
  OBSNAME = "CMAP"

  ; Tier
  TIER = 2

  ; Period
  YEAR1 = 1979
  YEAR2 = 2018

  ; MIPS
  MIPS = (/"Amon"/)

  ; FREQ
  FREQ = (/"mon"/)
  
  ; Selected variable (standard name)
  VARS = (/"pr"/)

  ; Name in the raw data
  NAME = (/"precip"/)

  ; CMOR table
  CMOR_TABLE = getenv("esmvaltool_root") + \
    "/cmor/tables/cmip5/Tables/CMIP5_" + MIPS

end

begin

  ; Loop over variables
  do vID = 0, dimsizes(VARS) - 1

    log_info("Processing " + VARS(vID))

    fname = input_dir_path + NAME(vID) + ".mon.mean.nc"
    f = addfile(fname, "r")
    output = f->$NAME(vID)$

    if (isStrSubset(NAME(vID), "precip")) then
      output = output / 86400.  ; convert from mm/day to kg m-2 s-1
      output@units = "kg m-2 s-1"
    end if

    date = cd_calendar(output&time, 0)
    idx = ind(date(:, 0).ge.YEAR1 .and. date(:, 0).le.YEAR2)
    output := output(idx, :, :)

    ; Format time coordinate
    output!0 = "time"
    output!1 = "lat"
    output!2 = "lon"
    format_coords(output, YEAR1 + "0101", YEAR2 + "1231", FREQ(vID))
        
    ; Set variable attributes
    tmp = format_variable(output, VARS(vID), CMOR_TABLE(vID))
    delete(output)
    output = tmp
    delete(tmp)

    ; Calculate coordinate bounds
    bounds = guess_coord_bounds(output, FREQ(vID))

    ; Set global attributes
    gAtt = set_global_atts( \
      OBSNAME, TIER, \
      "https://www.esrl.noaa.gov/psd/data/gridded/data.cmap.html", \
      "Xie, P., and P.A. Arkin, Bull. Amer. Meteor. Soc., "\
      +"doi:10.1175/1520-0477(1997)078<2539:GPAYMA>2.0.CO;2, 1997", "")

    ; Outfile
    fout = output_dir_path + "OBS_" + OBSNAME + "_L3_V1902_" + MIPS(vID) + \  ;; L3 correct?
      "_" + VARS(vID) + "_" + YEAR1 + "01-" + YEAR2 + "12.nc"

    ; Write variable
    write_nc(fout, VARS(vID), output, bounds, gAtt)
    delete(gAtt)
    delete(output)

  end do

end
