; #############################################################################
; ESMValTool CMORizer for NOAA 20CR data
; #############################################################################
;
; Tier
;    Tier 2: other freely-available dataset.
;
; Source
;    https://www.psl.noaa.gov/data/gridded/data.20thC_ReanV3.html
;
; Last access
;    20200915
;
; Download and processing instructions
;    To facilitate the download, the links to the ftp server are provided.
;    Since the filenames are sometimes identical across different
;    save the data in the following subdirectories in input_dir_path.
;
;    Subdirectory ensemble_mean/pressure/:
;      ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV3/Monthlies/prsSI-MO/
;
;    Subdirectory ensemble_mean/surface/:
;      ftp://ftp2.psl.noaa.gov/Datasets/20thC_ReanV3/Monthlies/miscSI-MO/
;        prmsl.mon.mean.nc
;      ftp://ftp2.psl.noaa.gov/Datasets/20thC_ReanV3/Monthlies/10mSI-MO/
;        uwnd.10m.mon.mean.nc
;        vwnd.10m.mon.mean.nc
;;
;    Select the section "Pressure" and "Surface" and download the variables
;    listed below. Since raw data on pressure levels and for surface have the
;    same file and variable name, save the data in two different subdirectories
;    "pressure" and "surface" in input_dir_path.
;
;   Subdirectory every_member/pressure/:
;
;   Subdirectory every_member/surface/:
;    wget https://portal.nersc.gov/archive/home/projects/incite11/www/20C_Reanalysis_version_3/everymember_anal_netcdf/mnmean/PRMSL/PRMSL_{1806..2015}_mnmean.tar
;    wget https://portal.nersc.gov/archive/home/projects/incite11/www/20C_Reanalysis_version_3/everymember_anal_netcdf/mnmean/UGRD10m/UGRD10m_{1806..2015}_mnmean.tar
;    wget https://portal.nersc.gov/archive/home/projects/incite11/www/20C_Reanalysis_version_3/everymember_anal_netcdf/mnmean/UGRD10m/UGRD10m_{1806..2015}_mnmean.tar
;
;    Specify the time range of the data as YEAR1-YEAR2 below, considering only
;    complete years (Jan to Dec).
;
; Modification history
;    20200925-yu_kosaka: revised to add every_member.
;    20200915-yu_kosaka: written based on cmorize_obs_ncep.ncl.	
;
; #############################################################################
loadscript(getenv("esmvaltool_root") + \
           "/esmvaltool/cmorizers/obs/interface.ncl")

begin

  ; Script name (for logger)
  DIAG_SCRIPT = "cmorize_obs_20crv3.ncl"

  ; Source name
  OBSNAME = "20CRv3"

  ; Tier
  TIER = 2

  ; Period
  YEAR1 = 1836
  YEAR2 = 2015

  ; Ensemble
  RUN1 = 1
  RUN2 = 80

  ; Selected variable (standard name)
  VAR = (/"psl", "uas", "vas"/)

  ; Name in the raw data
  NAME = (/"prmsl", "uwnd", "vwnd"/)

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
  VERSION = "3"

  ; Global attributes
  SOURCE = "https://www.psl.noaa.gov/data/20thC_Rean/"
  REF = "Slivinski, L. C. et al., Q. J. R. Meteorol. Soc., " + \
    "doi:10.1002/qj.3598, 2019"
  COMMENT = ""

end

begin

  ; Loop over variables
  do vv = 0, dimsizes(VAR) - 1

  do rr = RUN1-1, RUN2
  
    if (rr.eq.RUN1-1) then
      ens = "mean"
    else
      ens = sprinti("%0.3i", rr)
    end if

    log_info("Processing " + VAR(vv) + " (" + MIP(vv) + ") " + ens)

    do yy = YEAR1, YEAR2

      if (isStrSubset(MIP(vv), "mon")) then
        if (ens.eq."mean") then
	  name = NAME(vv)
          if (any(NAME(vv).eq.(/"uwnd", "vwnd"/)).and.SUBDIR(vv).eq."surface/") then
            fname = input_dir_path + "ensemble_mean/" + SUBDIR(vv) + NAME(vv) + ".10m.mon.mean.nc"
          else
            fname = input_dir_path + "ensemble_mean/" + SUBDIR(vv) + NAME(vv) + ".mon.mean.nc"
          end if
	else
          if (NAME(vv).eq."uwnd" .and. SUBDIR(vv).eq."surface/") then
	    name = "UGRD10m"
          elseif (NAME(vv).eq."vwnd" .and. SUBDIR(vv).eq."surface/") then
	    name = "VGRD10m"
          elseif (NAME(vv).eq."prmsl") then
  	    name = "PRMSL"
	  end if
	  fname = input_dir_path + "every_member/" + SUBDIR(vv) + name + "/" + yy + "/" + name + "." + yy + ".mnmean_mem"+ens+".nc"
        end if
      end if

      if (isStrSubset(MIP(vv), "day")) then
        flist = systemfunc("ls " + input_dir_path + SUBDIR(vv) + NAME(vv) + \
                           ".*" + yy + ".nc")
        fname = flist(0)
        delete(flist)
      end if

      f = addfile(fname, "r")
      tmp = f->$name$
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
      if (output@units.eq."m s**-1") then
        output@units = "m/s"
      end if
      if (output@units.ne.EXPUNITS(vv)) then
        error_msg("f", DIAG_SCRIPT, "", \
                  "possibly wrong input units for " + VAR(vv))
      end if

      rank = dimsizes(dimsizes(output))
      output!0 = "time"
      if (rank.eq.4 .and. SUBDIR(vv).eq."surface/") then
        tmp = output(:, 0, :, :)
        delete(output)
        output = tmp
        delete(tmp)
        rank = 3 
      end if
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
;        str_join((/"OBS", OBSNAME, TYPE, VERSION, \
        str_join((/"OBS", OBSNAME, TYPE, ens, \
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

  end do
end
