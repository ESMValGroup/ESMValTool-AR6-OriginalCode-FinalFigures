; #############################################################################
; ESMValTool CMORizer for ERA5 data
; #############################################################################
;
; Tier
;    Tier 3:  restricted datasets (i.e., dataset which requires a registration
;    to be retrieved or provided upon request to the respective contact or PI).
;
; Source
;    https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels
;    https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels
;
; Last access
;    20191013
;
; Download and processing instructions
;    Download via CDS for each variable and each year, and save as [var].yyyy.nc.
;
; Modification history
;    20201112-kosaka_yu: added preliminary backward extension of ERA5.
;    20201112-kosaka_yu: added tos and siconc.
;    20191223-kosaka_yu: written.
;
; #############################################################################
loadscript(getenv("esmvaltool_root") + "/esmvaltool/cmorizers/obs/interface.ncl")

begin

  ; Script name (for logger)
  DIAG_SCRIPT = "cmorize_obs_era5.ncl"

  ; Source name
  OBSNAME = "ERA5"

  ; Tier
  TIER = 3

  ; Period
  YEAR1 = 1950
  YEAR2 = 2018

  ; Selected variable (standard name)
  VAR = (/ \
          "tas", "psl", "uas", "vas", "tos", "sic" /)

  ; Name in the raw data
  NAME = (/ \
          "t2m", "msl", "u10", "v10", "sst", "siconc"/)

  ; Variable in the path of the raw data
  PNAME = NAME

  ; Expected units (according to CMOR standard)
  EXPUNITS = (/ \
               "K", "Pa", "m s**-1", "m s**-1", "degC", "%"/)

  ; MIP
  MIP = (/ \
          "Amon", "Amon", "Amon", "Amon", "Omon", "OImon"/)

  ; Frequency
  FREQ = (/ \
           "mon", "mon", "mon", "mon", "mon", "mon"/)

  ; CMOR tables
  CMOR_TABLE = getenv("cmor_tables") + "/cmip5/Tables/CMIP5_" + MIP

  ; Type
  TYPE = "reanaly"

  ; Version
  VERSION = "1"

  ; Global attributes
  SOURCE = "https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels"
  REF = ""
  COMMENT = ""

end

begin

  ; Loop over variables
  do vv = 0, dimsizes(VAR) - 1

    log_info("Processing " + VAR(vv) + " (" + MIP(vv) + ")")

;    if (isStrSubset(MIP(vv), "mon")) then
;      fname = input_dir_path + "era5_" + PNAME(vv) + "_"+YEAR1+"-"+YEAR2+"_monthly.nc"
;    elseif (isStrSubset(MIP(vv), "day")) then
;      fname = input_dir_path + "era5_" + PNAME(vv) + "_"+YEAR1+"-"+YEAR2+"_hourly.nc"
;    end if
;
;    f = addfile(fname, "r")
;    date = cd_calendar(f->time, 0)

    do yr = YEAR1, YEAR2

      if (isStrSubset(MIP(vv), "mon")) then
        fname = input_dir_path + PNAME(vv) + "." + yr + ".nc"
      elseif (isStrSubset(MIP(vv), "day")) then
        fname = input_dir_path + PNAME(vv) + "."+yr+".hourly.nc"
      end if

      f = addfile(fname, "r")
      date = cd_calendar(f->time, 0)

      ; Extract time range
      idx = ind(date(:, 0).eq.yr)

      dims = getfilevardims(f, NAME(vv))
      if (dimsizes(dims).eq.3) then
        output = f->$NAME(vv)$(idx, :, :)
      elseif (dimsizes(dims).eq.4) then
        output = f->$NAME(vv)$(idx, :, :, :)
      end if
      fill_start = num(ismissing(output))
      delete(dims)
      delete(idx)

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
      if (VAR(vv).eq."tos" .and. \
        output@units.eq."K") then
        output = output - 273.15
        output@units = "degC"
      end if
      if (VAR(vv).eq."sic" .and. \
        output@units.eq."(0 - 1)") then
        output = output * 100.
        output@units = "%"
      end if
      if (VAR(vv).eq."hus" .and. output@units.eq."grams/kg") then
        output = output / 1000.
        output@units = "1"
      end if
      if (VAR(vv).eq."zg" .and. any(output@units.eq.(/"m**2 s**-2", "m^2 s^-2", "m^2/s^2"/))) then
        output = output / 9.81
        output@units = "m"
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
        plev = tofloat(output&plev)
        plev = plev * 100.  ; [mb] --> [Pa]
        delete(output&plev)
        output&plev = plev
      elseif (rank.eq.3)
        output!1 = "lat"
        output!2 = "lon"
      end if

      ; Format coordinates
      format_coords(output, yr + "0101", yr + "1231", FREQ(vv))

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
      DATESTR = yr + "01-" + yr + "12"
      fout = output_dir_path + \
      str_join((/"OBS", OBSNAME, TYPE, VERSION, \
      MIP(vv), VAR(vv), DATESTR/), "_") + ".nc"

      ; Add height coordinate to tas variable (required by the new backend)
      if (any(VAR(vv).eq.(/"tas", "uas", "vas"/))) then
        output@coordinates = "height"
      end if

      ; Write variable
      write_nc(fout, VAR(vv), output, bounds, gAtt)
      delete(gAtt)
      delete(output)
      delete(bounds)

      ; Add height coordinate to tas variable (required by the new backend)
      if (any(VAR(vv).eq.(/"tas", "uas", "vas"/))) then
        if (VAR(vv).eq."tas") then
          height = 2.d
        else
          height = 10.d
        end if
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
;      if (VAR(vv).eq."sic") then
;        type = "sea_ice"
;        type@long_name = "Sea Ice area type"
;        type@standard_name = "area_type"
;        w = addfile(fout, "w")
;        w->type = type
;        delete(w)
;      end if
    end do
    delete(fname)
    delete(f)
    delete(date)
  end do

end