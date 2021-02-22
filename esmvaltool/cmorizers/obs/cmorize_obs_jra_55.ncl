; #############################################################################
; ESMValTool CMORizer for JRA-55 data
; #############################################################################
;
; Tier
;    Tier 3:  restricted datasets (i.e., dataset which requires a registration
;    to be retrieved or provided upon request to the respective contact or PI).
;
; Source
;    https://jra.kishou.go.jp/JRA-55/index_en.html
;
; Last access
;    20210102
;
; Download and processing instructions
;    Download from http://search.diasjp.net/en/dataset/JRA55
;     Hist/Monthly/anl_p125
;     Hist/Monthly/anl_surf125
;    keeping the above directory structure
;
; Modification history
;    20210128-kosaka_yu: written.
;
; #############################################################################
loadscript(getenv("esmvaltool_root") + "/esmvaltool/cmorizers/obs/interface.ncl")
load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_inv_string.ncl"

begin

  ; Script name (for logger)
  DIAG_SCRIPT = "cmorize_obs_jra_55.ncl"

  ; Source name
  OBSNAME = "JRA-55"

  ; Tier
  TIER = 3

  ; Period
  YEAR1 = 1958
  YEAR2 = 2020

  ; Selected variable (standard name)
  VAR = (/ "hur", "hus", "ta", "ua", "va", "wap", "zg", \
           "hurs", "huss", "tas", "ps", "psl", "uas", "vas" /)


  ; Variable in the path of the raw data
  PNAME = (/ "rh", "spfh", "tmp", "ugrd", "vgrd", "vvel", "hgt",  \
             "rh2m", "spfh2m", "tmp2m", "pres", "prmsl", "ugrd10m", "vgrd10m"/)

  ; Variable in the grib file
  NAME = (/ "RH_GDS0_ISBL_S123", "SPFH_GDS0_ISBL_S123", "TMP_GDS0_ISBL_S123", \
            "UGRD_GDS0_ISBL_S123", "VGRD_GDS0_ISBL_S123", "VVEL_GDS0_ISBL_S123", "HGT_GDS0_ISBL_S123", \
            "RH_GDS0_HTGL_S123", "SPFH_GDS0_HTGL_S123", "TMP_GDS0_HTGL_S123", \
            "PRES_GDS0_SFC_S123", "PRMSL_GDS0_MSL_S123", "UGRD_GDS0_HTGL_S123", "VGRD_GDS0_HTGL_S123" /)

  LATNAME = (/ "g0_lat_1", "g0_lat_1", "g0_lat_1", "g0_lat_1", "g0_lat_1", "g0_lat_1", "g0_lat_1", \
               "g0_lat_0", "g0_lat_0", "g0_lat_0", "g0_lat_0", "g0_lat_0", "g0_lat_0", "g0_lat_0" /)

  LONNAME = (/ "g0_lon_2", "g0_lon_2", "g0_lon_2", "g0_lon_2", "g0_lon_2", "g0_lon_2", "g0_lon_2", \
               "g0_lon_1", "g0_lon_1", "g0_lon_1", "g0_lon_1", "g0_lon_1", "g0_lon_1", "g0_lon_1" /)

  LEVNAME = (/ "lv_ISBL0", "lv_ISBL0", "lv_ISBL0", "lv_ISBL0", "lv_ISBL0", "lv_ISBL0", "lv_ISBL0", \
               "", "", "", "", "", "", "" /)
  
  ; Variable category in the path
  CATEGORY = (/"anl_p125", "anl_p125", "anl_p125", "anl_p125", "anl_p125", "anl_p125", "anl_p125", \
               "anl_surf125", "anl_surf125", "anl_surf125", "anl_surf125", "anl_surf125", "anl_surf125", "anl_surf125"/)

  ; Unit in the girb file
  UNITS = (/ "%", "1", "K", "m s-1", "m s-1", "Pa s-1", "m", \ 
             "%", "1", "K", "Pa", "Pa", "m s-1", "m s-1"/)
               
  ; Expected units (according to CMOR standard)
  EXPUNITS = (/ "%", "1", "K", "m s-1", "m s-1", "Pa s-1", "m", \
                "%", "1", "K", "Pa", "Pa", "m s-1", "m s-1"/)

  ; MIP
  MIP = (/ "Amon", "Amon", "Amon", "Amon", "Amon", "Amon", "Amon", \
           "Amon", "Amon", "Amon", "Amon", "Amon", "Amon", "Amon"/)

  ; Frequency
  FREQ = (/"mon", "mon", "mon", "mon", "mon", "mon", "mon", \
           "mon", "mon", "mon", "mon", "mon", "mon", "mon"/)

  ; CMOR tables
  CMOR_TABLE = getenv("cmor_tables") + "/cmip5/Tables/CMIP5_" + MIP

  ; Type
  TYPE = "reanaly"

  ; Version
  VERSION = "1"

  ; Global attributes
  SOURCE = "http://search.diasjp.net/en/dataset/JRA55"
  REF = "Kobayashi, S. et al., J. Meteor. Soc. Japan, 93, 5-48, doi:10.2151/jmsj.2015-001"
  COMMENT = ""

end

begin

  ; Loop over variables
  do vv = 0, dimsizes(VAR) - 1

    if (FREQ(vv).ne."mon") then
      error_msg("f", DIAG_SCRIPT, "", \
                "Only monthly data are supported currently.")
    end if
    
    log_info("Processing " + VAR(vv) + " (" + MIP(vv) + ")")

    if (CATEGORY(vv).eq."anl_p125") then
      fname_head = input_dir_path + "Hist/Monthly/anl_p125/anl_p125_"+PNAME(vv)+"."
    else
      fname_head = input_dir_path + "Hist/Monthly/"+CATEGORY(vv)+"/"+CATEGORY(vv)+"."
    end if

    fname = fname_head + sprinti("%0.4i", YEAR1) + "01"
    f = addfile(fname, "r")
    tmp = f->$NAME(vv)$

    output = new(array_append_record(12*(YEAR2-YEAR1+1), dimsizes(tmp), 0), typeof(tmp))
    output!0 = "time"
    output&time = cd_inv_string(yyyymm_time(YEAR1, YEAR2, "string") , "%Y%M")
    if (dimsizes(dimsizes(tmp)).eq.3) then
      output!1 = "plev"
      output!2 = "lat"
      output!3 = "lon"
      output&plev = tofloat(tmp&$LEVNAME(vv)$) * 100.
    else
      output!1 = "lat"
      output!2 = "lon"
    end if
    output&lat = tmp&$LATNAME(vv)$
    output&lon = tmp&$LONNAME(vv)$
    output@units = UNITS(vv)

    delete(tmp)

    t = 0
    do yr = YEAR1, YEAR2

      do mo = 1, 12
        
        fname = fname_head + sprinti("%0.4i", yr) + sprinti("%0.2i", mo)

        f = addfile(fname, "r")
        if (dimsizes(dimsizes(output)).eq.4) then
          output(t, :, :, :) = (/f->$NAME(vv)$/)
        else
          output(t, :, :) = (/f->$NAME(vv)$/)
        end if

        delete(fname)
        delete(f)
        t = t + 1
      end do
    end do
        
    if (output@units.ne.EXPUNITS(vv)) then
      error_msg("f", DIAG_SCRIPT, "", \
                "possibly wrong input units for " + VAR(vv))
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
;    fill_end = num(ismissing(output))
;    if (fill_start.ne.fill_end) then
;      error_msg("f", DIAG_SCRIPT, "", \
;                "missing values lost during conversion")
;    end if
;    delete(fill_start)
;    delete(fill_end)

    ; Output file
    DATESTR = YEAR1 + "01-" + YEAR2 + "12"
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

  end do

end
