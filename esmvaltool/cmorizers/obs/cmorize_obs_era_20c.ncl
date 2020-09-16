; #############################################################################
; ESMValTool CMORizer for ERA-20C data
; #############################################################################
;
; Tier
;    Tier 3: restricted datasets (i.e., dataset which requires a registration
;    to be retrieved or provided upon request to the respective contact or PI).
;
; Source
;    https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era-20c
;
; Last access
;    20190904
;
; Download and processing instructions
; Download with scripts based on ECMWF WebAPI.
;  surface1/monthly_analysis/
;  msl_1900-2010.nc
;  u10_1900-2010.nc
;  v10_1900-2010.nc
;
; Modification history
;    20200915-yu_kosaka: written based on cmorize_obs_ncep.ncl.
;
; #############################################################################
loadscript(getenv("esmvaltool_root") + \
           "/esmvaltool/cmorizers/obs/interface.ncl")

begin

  ; Script name (for logger)
  DIAG_SCRIPT = "cmorize_obs_era_20c.ncl"

  ; Source name
  OBSNAME = "ERA-20C"

  ; Tier
  TIER = 3

  ; Period
  YEAR1 = 1900
  YEAR2 = 2010

  ; Selected variable (standard name)
  VAR = (/"psl", "uas", "vas"/)

  ; Name in the raw data
  NAME = (/"msl", "u10", "v10"/)

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

  ; Global attributes
  SOURCE = "https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era-20c"
  REF = "Poli, P. et al., J. Climate " + \
    "doi:10.1175/JCLI-D-15-0556.1, 2016"
  COMMENT = ""

end

begin

  ; Loop over variables
  do vv = 0, dimsizes(VAR) - 1

    log_info("Processing " + VAR(vv) + " (" + MIP(vv) + ")")

    do yy = YEAR1, YEAR2

      if (isStrSubset(MIP(vv), "mon")) then
        fname = input_dir_path + SUBDIR(vv) + NAME(vv) + "_1900-2010.nc"
      end if

      if (isStrSubset(MIP(vv), "day")) then
        flist = systemfunc("ls " + input_dir_path + SUBDIR(vv) + NAME(vv) + \
                           ".*" + yy + ".nc")
        fname = flist(0)
        delete(flist)
      end if

      f = addfile(fname, "r")
      tmp = f->$NAME(vv)$
      delete(fname)
      delete(f)
      fill_start = num(ismissing(tmp))

      ; Extract time range
      tmp&time@calendar = "standard"
      date = cd_calendar(tmp&time, 0)
      idx = ind(date(:, 0).eq.yy)
      if (dimsizes(dimsizes(tmp)).eq.4) then
        output = tmp(idx, :, :, :)
      else
        output = tmp(idx, :, :)
      end if
      delete(idx)
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
      format_coords(output, yy + "0101", yy + "1231", FREQ(vv))

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
      DATESTR = yy + "01-" + yy + "12"
      fout = output_dir_path + \
        str_join((/"OBS", OBSNAME, TYPE, VERSION, \
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
