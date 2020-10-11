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
;        uwnd.mon.mean.nc
;        vwnd.mon.mean.nc
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
;    Subdirectory every_member/pressure/:
;     for lev in 1000 850 500 200
;     do
;      wget https://portal.nersc.gov/archive/home/projects/incite11/www/20C_Reanalysis_version_3/everymember_anal_netcdf/mnmean/UGRD${lev}/UGRD${lev}_{1806..2015}_mnmean.tar
;      wget https://portal.nersc.gov/archive/home/projects/incite11/www/20C_Reanalysis_version_3/everymember_anal_netcdf/mnmean/VGRD${lev}/VGRD${lev}_{1806..2015}_mnmean.tar
;    done
;
;    Subdirectory every_member/surface/:
;     wget https://portal.nersc.gov/archive/home/projects/incite11/www/20C_Reanalysis_version_3/everymember_anal_netcdf/mnmean/PRMSL/PRMSL_{1806..2015}_mnmean.tar
;     wget https://portal.nersc.gov/archive/home/projects/incite11/www/20C_Reanalysis_version_3/everymember_anal_netcdf/mnmean/UGRD10m/UGRD10m_{1806..2015}_mnmean.tar
;     wget https://portal.nersc.gov/archive/home/projects/incite11/www/20C_Reanalysis_version_3/everymember_anal_netcdf/mnmean/UGRD10m/UGRD10m_{1806..2015}_mnmean.tar
;
;    Specify the time range of the data as YEAR1-YEAR2 below, considering only
;    complete years (Jan to Dec).
;
; Modification history
;    20201003-yu_kosaka: added pressure level ua and va.
;    20200924-yu_kosaka: revised to add every_member.
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
  VAR = (/"psl", "uas", "vas", "ua", "va"/)

  ; Name in the raw data
  NAME_ensmean = (/"prmsl", "uwnd", "vwnd", "uwnd", "vwnd"/)
  NAME_members = (/"PRMSL", "UGRD10m", "VGRD10m", "UGRD", "VGRD"/)

  ; Subdirectory
  SUBDIR = (/"surface/", "surface/", "surface/", "pressure/", "pressure/"/)

  ; Expected units (according to CMOR standard)
  EXPUNITS = (/"Pa", "m/s", "m/s", "m/s", "m/s"/)

  ; MIP
  MIP = (/"Amon", "Amon", "Amon", "Amon", "Amon"/)

  ; Frequency
  FREQ = (/"mon", "mon", "mon", "mon", "mon"/)

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

      if (isStrSubset(MIP(vv), "mon")) then
        if (ens.eq."mean") then
	  name = NAME_ensmean(vv)
          if (any(name.eq.(/"uwnd", "vwnd"/)).and.SUBDIR(vv).eq."surface/") then
            fname = input_dir_path + "ensemble_mean/" + SUBDIR(vv) + name + ".10m.mon.mean.nc"
          else
            fname = input_dir_path + "ensemble_mean/" + SUBDIR(vv) + name + ".mon.mean.nc"
          end if
	else
          name = NAME_members(vv)
          if (SUBDIR(vv).eq."surface/") then
            fname = systemfunc("ls "+input_dir_path + "every_member/" + SUBDIR(vv) + name + "/{"+YEAR1+".."+YEAR2+"}/" + name + ".*.mnmean_mem"+ens+".nc")
          else
            fname_head = input_dir_path + "every_member/" + SUBDIR(vv) + "/*/" + name 
            fname_tail = ".*.mnmean_mem"+ens+".nc"
          end if
        end if
      end if

      if (isvar("fname")) then
        if (dimsizes(fname).eq.1) then
          f = addfile(fname, "r")
          tmp = f->$name$
        else
          f = addfiles(fname, "r")
          ListSetType(f, "cat")
          tmp = f[:]->$name$
        end if
        delete(fname)
        delete(f)
      else
        fname_head_YEAR1 = str_sub_str(fname_head, "*", YEAR1)
        fname_tail_YEAR1 = str_sub_str(fname_tail, "*", YEAR1)
        fname = systemfunc("ls "+fname_head_YEAR1+"*"+fname_tail_YEAR1)
        zsize = dimsizes(fname)
        lev = new(zsize, "integer")
        do z = 0, zsize - 1
          lev(z) = toint(str_sub_str(str_sub_str(fname(z), fname_head_YEAR1, ""), fname_tail_YEAR1, ""))
        end do
        qsort(lev)
        lev = lev(::-1)
        delete(fname)
        do z = 0, zsize - 1
          fname = systemfunc("ls "+str_sub_str(fname_head, "*", "{"+YEAR1+".."+YEAR2+"}")+lev(z)+fname_tail)
          f = addfiles(fname, "r")
          ListSetType(f, "cat")
          if (z.eq.0) then
            tmptmp = f[:]->$name$
            dims = dimsizes(tmptmp)
            dims(1) = zsize
            tmp = new(dims, typeof(tmptmp))
printVarSummary(tmp)
            delete(dims)
            delete(tmptmp)
          end if
          tmptmp = f[:]->$name$
printVarSummary(tmptmp)
          tmp(:, z:z, :, :) = tmptmp
          delete(tmptmp)
          delete(fname)
          delete(f)
        end do
        delete(lev)
      end if
      fill_start = num(ismissing(tmp))

      ; Extract time range
      tmp&time@calendar = "standard"
      date = cd_calendar(tmp&time, -1)
      idx0 = ind(date.eq.YEAR1*100+1)
      idx1 = ind(date.eq.YEAR2*100+12)
      if (dimsizes(dimsizes(tmp)).eq.4) then
        output = tmp(idx0:idx1, :, :, :)
      else
        output = tmp(idx0:idx1, :, :)
      end if
      delete(idx0)
      delete(idx1)
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
;        output&plev = output&plev * 100.  ; [mb] --> [Pa]
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
end
