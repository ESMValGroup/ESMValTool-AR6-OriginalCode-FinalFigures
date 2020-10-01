; #############################################################################
; ESMValTool CMORizer for CERA-20C data
; #############################################################################
;
; Tier
;    Tier 3: restricted datasets (i.e., dataset which requires a registration
;    to be retrieved or provided upon request to the respective contact or PI).
;
; Source
;    https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/cera-20c
;
; Last access
;    20200924
;
; Download and processing instructions
; Download with scripts based on ECMWF WebAPI.
;  surfacex/
;  msl.1901-2010.ens{0-9}.nc
;  u10.1901-2010.ens{0-9}.nc
;  v10.1901-2010.ens{0-9}.nc
;
; Modification history
;    20200930-yu_kosaka: written based on cmorize_obs_era_20c.ncl.
;
; #############################################################################
loadscript(getenv("esmvaltool_root") + \
           "/esmvaltool/cmorizers/obs/interface.ncl")

begin

  ; Script name (for logger)
  DIAG_SCRIPT = "cmorize_obs_cera_20c.ncl"

  ; Source name
  OBSNAME = "CERA-20C"

  ; Tier
  TIER = 3

  ; Period
  YEAR1 = 1901
  YEAR2 = 2010

  ; Selected variable (standard name)
  VAR = (/"psl", "uas", "vas"/)

  ; Name in the raw data
  NAME = (/"msl", "u10", "v10"/)

  ; Name in the file name
  FNAME = (/"msl", "10u", "10v"/)

  ; Subdirectory
  SUBDIR = (/"surface/", "surface/", "surface/"/)

  ; Expected units (according to CMOR standard)
  EXPUNITS = (/"Pa", "m/s", "m/s"/)

  ; MIP
  MIP = (/"Amon", "Amon", "Amon"/)

  ; Frequency
  FREQ = (/"mon", "mon", "mon"/)

  ; CMOR tables
  CMOR_TABLE = getenv("cmor_tables") + "/cmip5/Tables/CMIP5_" + MIP

  ; Type
  TYPE = "reanaly"

  ; Version
  VERSION = "1"

  ; Ensemble
  ENS1 = 0
  ENS2 = 9

  ; Global attributes
  SOURCE = "https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/cera-20c"
  REF = "Laloyaux, P. et al. J. Adv. Modeling Earth Systems," + \
    "doi:10.1029/2018MS001273, 2018"
  COMMENT = ""

end

begin

  ; Loop over variables
  do vv = 0, dimsizes(VAR) - 1

    log_info("Processing " + VAR(vv) + " (" + MIP(vv) + ")")

    do ens = ENS1, ENS2

      if (isStrSubset(MIP(vv), "mon")) then
        fname = input_dir_path + SUBDIR(vv) + FNAME(vv) + ".1901-2010.ens" + ens + ".nc"
      end if

      f = addfile(fname, "r")
      tmp = f->$NAME(vv)$
      delete(fname)
      delete(f)
      fill_start = num(ismissing(tmp))

      ; Extract time range
      tmp&time@calendar = "standard"
      date = cd_calendar(tmp&time, 1)
      idx1 = ind(date.eq.YEAR1*100+1)
      idx2 = ind(date.eq.YEAR2*100+12)	
      if (dimsizes(dimsizes(tmp)).eq.4) then
        output = tmp(idx1:idx2, :, :, :)
      else
        output = tmp(idx1:idx2, :, :)
      end if
      delete(idx1)
      delete(idx2)
      delete(tmp)
      delete(date)

      ; Unpack variable according to metadata information
      if (isatt(output, "scale_factor") .or. \
          isatt(output, "add_offset")) then
        tmp = output * output@scale_factor + output@add_offset
        copy_VarMeta(output, tmp)
        delete(output)
        output = tmp
        delete(tmp)
      end if

      ; Convert units
      if (any(VAR(vv).eq.(/"ta", "tas"/)) .and. \
          output@units.eq."degC") then
        output = output + 273.15
        output@units = "K"
      end if
      if (VAR(vv).eq."hus" .and. output@units.eq."grams/kg") then
        output = output / 1000.
        output@units = "1"
      end if
      if (any(VAR(vv).eq.(/"uas", "vas"/))) then
        output@units = EXPUNITS(vv)
      end if
      if (output@units.ne.EXPUNITS(vv)) then
        error_msg("f", DIAG_SCRIPT, "", \
                  "possibly wrong input units for " + VAR(vv))
      end if

      rank = dimsizes(dimsizes(output))
      output!0 = "time"
      if (rank.eq.4) then
        output!1 = "plev"
        output!2 = "lat"
        output!3 = "lon"
        output&plev = output&plev * 100.  ; [mb] --> [Pa]
      elseif (rank.eq.3)
        output!1 = "lat"
        output!2 = "lon"
      end if

      ; Format coordinates
      format_coords(output, YEAR1 + "0101", YEAR2 + "1231", FREQ(vv))

      ; Set variable attributes
      tmp = format_variable(output, VAR(vv), CMOR_TABLE(vv))
      delete(output)
      output = tmp
      delete(tmp)

      ; Calculate coordinate bounds
      bounds = guess_coord_bounds(output, FREQ(vv))

      ; Set global attributes
      gAtt = set_global_atts(OBSNAME, TIER, SOURCE, REF, COMMENT)

      ; Check fill values
      fill_end = num(ismissing(output))
      if (fill_start.ne.fill_end) then
        error_msg("f", DIAG_SCRIPT, "", \
                  "missing values lost during conversion")
      end if
      delete(fill_start)
      delete(fill_end)

      ; Output file
      DATESTR = YEAR1 + "01-" + YEAR2 + "12"
      ENS = "ens" + ens
      fout = output_dir_path + \
        str_join((/"OBS", OBSNAME, TYPE, ENS, \
                   MIP(vv), VAR(vv), DATESTR/), "_") + ".nc"

      ; Add height coordinate to tas variable (required by the new backend)
      if (VAR(vv).eq."tas") then
        output@coordinates = "height"
      end if

      ; Write variable
      write_nc(fout, VAR(vv), output, bounds, gAtt)
      delete(gAtt)
      delete(output)
      delete(bounds)

      ; Add height coordinate to tas variable (required by the new backend)
      if (VAR(vv).eq."tas") then
        height = 2.d
        height!0 = "ncl_scalar"
        height@units = "m"
        height@axis = "Z"
        height@positive = "up"
        height@long_name = "height"
        height@standard_name = "height"
        w = addfile(fout, "w")
        w->height = height
        delete(w)
      end if

    end do

  end do

end
