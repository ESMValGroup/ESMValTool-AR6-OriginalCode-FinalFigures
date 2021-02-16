; trend_cell_edges
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs:
;
; History
;    20200901 kosaka_yu: added AMIP and DAMIP simulations; split piControl into
;                        time segments of specified length automatically
;    20200514 kosaka_yu: cleaned up.
;    20191224 kosaka_yu: modified to handle piControl & historical experiments
;    20191111 kosaka_yu: modified to handle cmip5 and cmip6 individually
;    20191017 kosaka_yu: updated for ESMValTool 2.0.0b2
;    20190316 kosa_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/statistics.ncl"
load "$diag_scripts/shared/plot/style.ncl"
load "$diag_scripts/shared/plot/contour_maps.ncl"

function zero_crossing(x, hemisphere, sign, lat_uncertainty)
local edge, y, lat, t, edge, j, jeq, jmax, jsubpolar, jpolar, j0, j1, jj, \
      subpolar_boudnary, polar_boundary
begin
  lat = x&lat
  jeq = ind_nearest_coord(0., lat, 0)
  jmax = dimsizes(lat)-1
  delete(lat)
  if (hemisphere.eq."NH") then
    if (isMonotonic(x&lat).eq.-1) then
      y = x(:, 0:jeq:-1)
    else
      y = x(:, jeq:jmax)
    end if
  else
    if (isMonotonic(x&lat).eq.-1) then
      y = x(:, jeq:jmax)
    else
      y = x(:, 0:jeq:-1)
    end if
  end if
  lat = y&lat
  jmax = dimsizes(lat)-1
  y = sign/abs(sign)*y

  edge = new((/dimsizes(x&year)/), "float", default_fillvalue("float"))
  edge!0 = "year"
  edge&year = x&year
  edge = edge@_FillValue

  subpolar_boundary = 30.
  polar_boundary = 60.

  jsubpolar = ind_nearest_coord(subpolar_boundary, abs(lat), 0)
  jpolar    = ind_nearest_coord(polar_boundary,    abs(lat), 0)

  do t = 0, dimsizes(x&year)-1
    ; Search edge (where y changes from - to +)
    j0 = minind(y(t, 0:jsubpolar))
    j1 = maxind(y(t, j0:jpolar))
    j1 = j0 + j1

    do j = j0+1, j1
      if (y(t, j-1).lt.0. .and. y(t, j).gt.0.) then
        edge(t) = tofloat((y(t, j)*lat(j-1) - y(t, j-1)*lat(j)) \
                          / (y(t, j) - y(t, j-1)))
      elseif (y(t, j-1).lt.0. .and. y(t, j).eq.0. .and. y(t, j+1).gt.0.) then
        edge(t) = tofloat(lat(j))
      end if
      if (.not.ismissing(edge(t))) then
        do jj = j+1, jmax
          if (abs(lat(jj)).le.abs(edge(t))+lat_uncertainty .and. \
              y(t, jj-1).lt.0. .and. y(t, jj).gt.0.) then
            edge(t) = edge@_FillValue  ; based on Grise et al (2018 JClim)
            break
          end if
        end do
        break
      end if
    end do
  end do

  return(edge)
end

function calc_trend(x, trend_method)
local dims, tmp, frac_threshold, tr
begin
  frac_threshold = 0.8  ; At least 80% of the time series must be non-missing
  if (any(trend_method.eq.(/"least_square",  "LeastSquare",  "Least_Square", \
                            "least_squares", "LeastSquares", "Least_Squares", \
                            "ls", "LS"/))) then
    dims = dimsizes(x)
    tmp = dtrend_msg_n(tofloat(ispan(1, dims(0), 1)), x, True, True, 0)
    tr = where(dim_num_n(.not.ismissing(x), 0).gt.dims(0)*frac_threshold, \
               tmp@slope, tmp@_FillValue)
    delete(dims)
  elseif (any(trend_method.eq.(/"Sen", "sen", "median_slope", "MedianSlope", \
                                "Theil-Sen"/))) then
    tmp = trend_manken(x, True, 0)
    tr = where(dim_num_n(.not.ismissing(x), 0).gt.dims(0)*frac_threshold, \
               tmp(1, :), tmp@_FillValue)
  end if
  copy_VarMeta(x(0), tr)

  delete(tmp)
  return(tr)
end

procedure write_data(x, branch, dataset)
local nc_filename, work_dir, y
begin
  work_dir = config_user_info@work_dir + "/"
  system("mkdir -p " + work_dir)

  y = x
  y!0 = "ensemble"
  y&ensemble = ispan(1, dimsizes(y), 1)
    nc_filename = work_dir + branch + "edge_" \
      + x@season + "_"

  if (isStrSubset(dataset, "pi")) then
    nc_filename = nc_filename + (x@eyear - x@syear + 1)+ "yrs"
  else
    nc_filename = nc_filename + x@syear + "-" + x@eyear
  end if
  nc_filename = nc_filename + "_" + dataset + ".nc"

  if (nc_filename.ne."") then
    nc_outfile = ncdf_write(y, nc_filename)
  end if
  delete(y)
end

begin

  enter_msg(DIAG_SCRIPT, "")

  hemisphere = "NH"
  if (isatt(diag_script_info, "hemisphere")) then
    hemisphere = diag_script_info@hemisphere
  end if
  if (all(hemisphere.ne.(/"NH", "SH"/))) then
    error_msg("f", DIAG_SCRIPT, "", \
              "'hemisphere' must be either 'NH' or 'SH'.")
  end if

  method = "uas"
  if (isatt(diag_script_info, "method")) then
    method = diag_script_info@method
  end if

  trend_method = "least_square"
  if (isatt(diag_script_info, "trend_method")) then
    trend_method = diag_script_info@trend_method
  end if

  syear_trend = 1980
  eyear_trend = 2014
  if (isatt(diag_script_info, "syear_trend")) then
    syear_trend = diag_script_info@syear_trend
  end if
  if (isatt(diag_script_info, "eyear_trend")) then
    eyear_trend = diag_script_info@eyear_trend
  end if
  
; ========================================================================

  if (any(method.eq.(/"mpsi", "MPSI", "PSI500"/))) then
    method = "mpsi"
    var = "va"
  elseif (any(method.eq.(/"USFC", "UAS", "Usfc", "usfc", "uas"/))) then
    method = "uas"
    var = "uas"
  else
    error_msg("f", DIAG_SCRIPT, "", \
              "Currently, 'method' must be chosen from ['mpsi', 'usfc'].")
  end if
  varList := metadata_att_as_array(variable_info, "short_name")
  if (all(varList.ne.var)) then
    error_msg("f", DIAG_SCRIPT, "", "variable "+var+" is missing.")
  end if

  projList = metadata_att_as_array(input_file_info, "project")
  iObs = ind(projList.eq."OBS" .or. projList.eq."OBS6" .or. \
             projList.eq."obs" .or. \
             projList.eq."ana4mips" .or. projList.eq."obs4mips")

  ndatasets = ListCount(input_file_info)

; ========================================================================

  do ii = 0, ndatasets - 1
    if (input_file_info[ii]@short_name.ne.var) then
      continue
    end if
    data = read_data(input_file_info[ii])

    f = addfile(input_file_info[ii]@filename, "r")
    if (isfilevar(f, "season_year")) then
      year = f->season_year
      seasons = f->clim_season
      season = str_upper(str_squeeze(seasons(0, :)))
      delete(seasons)
    else
      year = f->year
;      month_number_bnds = f->month_number_bnds
;      if (all(month_number_bnds(0, :).eq.(/1, 12/))) then
        season = "ANN"
;      else
;        error_msg("f", DIAG_SCRIPT, "", \
;                  input_file_info[ii]@filename+": Cannot detect season.")
;      end if
;      delete(month_number_bnds)
    end if
    delete(f)

    data!0 = "year"
    data&year = year

    if (isatt(input_file_info[ii], "exp") .and. input_file_info[ii]@exp.eq."piControl") then
      nyears = eyear_trend - syear_trend + 1
      nseg = dimsizes(year) / nyears
      ts = new(nseg, "integer")
      te = new(nseg, "integer")
      do n = 0, nseg-1
        ts(n) = nyears*n
        te(n) = ts(n) + nyears - 1
      end do
    else
      nseg = 1
      ts = (/ind(year.eq.syear_trend)/)
      te = (/ind(year.eq.eyear_trend)/)
    end if

    do n = 0, nseg-1
      syear = year(ts(n))
      eyear = year(te(n))
    
      if (method.eq."mpsi") then
        ; trim plev < 5. for zonal_mpsi function
        v = data(ts(n):te(n), {100499.:501.}, :)
        va = conform_dims(array_append_record(dimsizes(v), (/1/), 0), \
                        v, (/0, 1, 2/))
        copy_VarCoords_1(v, va)
        va!2 = v!2
        va&$va!2$ = v&$v!2$
        delete(v)

        ; calculate zonal streamfunction
        ps = va(:, 0, :, :)
        ps = 100000.  ; dummy 1000 hPa
        psi = zonal_mpsi_Wrap(va(:, ::-1, :, :), va&lat, va&plev(::-1), ps)
        delete(va)
        delete(ps)

        ; get psi@500hPa
        if (any(psi&plev.eq.50000.)) then
          psi500 = psi(tst, {50000.}, :)
        else
          psi500 = linint1_n_Wrap(psi&plev, psi, False, (/50000./), 0, 1)
        end if
        if (dimsizes(dimsizes(psi500)).eq.3) then
          tmp = psi500
          delete(psi500)
          psi500 = tmp(:, 0, :)  ; psi500(time, lat)
          delete(tmp)
        end if
        delete(psi)

        ; get Hadley cell edges
        lat_uncertainty = 20.
        if (hemisphere.eq."NH") then
          edge \  ; where psi500 changes from + to -
          = zero_crossing(psi500, hemisphere, -1, lat_uncertainty)
        elseif (hemisphere.eq."SH") then
          edge \  ; where psi500 changes from - to +
          = zero_crossing(psi500, hemisphere, 1, lat_uncertainty)
        end if
        delete(psi500)
      elseif (method.eq."uas") then
        ; get Hadley cell edges
        lat_uncertainty = 0.
        edge \  ; where uas changes from - to +
          = zero_crossing(data(ts(n):te(n), :), hemisphere, 1, lat_uncertainty)
      end if

      ; calculate trends
      trend_edge = calc_trend(edge, trend_method)
      delete(edge)

      if (any(ii.eq.iObs)) then
        ; output obs trends
        trendObs = trend_edge
        trendObs@hemisphere = hemisphere
        trendObs@season = season
        trendObs@syear = syear
        trendObs@eyear = eyear
        trendObs@dataset = input_file_info[ii]@dataset
        trendObs@var = "lat"
        trendObs@long_name = "trend of "+hemisphere+" Hadley cell edge latitude"
        trendObs@diag_script = DIAG_SCRIPT

        write_data(trendObs, hemisphere, input_file_info[ii]@dataset)

        delete(trendObs)
      else
        ; save model trends
        esize = num(metadata_att_as_array(input_file_info, "project").eq.input_file_info[ii]@project.and.\
                    metadata_att_as_array(input_file_info, "exp").eq.input_file_info[ii]@exp.and.\
                    metadata_att_as_array(input_file_info, "dataset").eq.input_file_info[ii]@dataset)

        if (input_file_info[ii]@project.eq."CMIP5" .and. \
            isStrSubset(input_file_info[ii]@exp, "historical")) then
          if (isvar("trend_cmip5hist")) then
            tmp = trend_cmip5hist
            delete(trend_cmip5hist)
            trend_cmip5hist = array_append_record(tmp, trend_edge, 0)
            trend_cmip5hist@esize := array_append_record(tmp@esize, esize, 0)
            delete(tmp)
          else
            trend_cmip5hist = (/trend_edge/)
            trend_cmip5hist@esize = esize
            trend_cmip5hist@hemisphere = hemisphere
            trend_cmip5hist@season = season
            trend_cmip5hist@syear = syear
            trend_cmip5hist@eyear = eyear
            trend_cmip5hist@var = "lat"
            trend_cmip5hist@long_name \
              = "trend of "+hemisphere+" Hadley cell edge latitude"
            trend_cmip5hist@diag_script = DIAG_SCRIPT
          end if

        elseif (input_file_info[ii]@project.eq."CMIP5" .and. \
                input_file_info[ii]@exp.eq."piControl") then
          if (isvar("trend_cmip5pi")) then
            tmp = trend_cmip5pi
            delete(trend_cmip5pi)
            trend_cmip5pi = array_append_record(tmp, trend_edge, 0)
            trend_cmip5pi@esize := array_append_record(tmp@esize, nseg, 0)
            delete(tmp)
          else
            trend_cmip5pi = (/trend_edge/)
            trend_cmip5pi@esize = nseg
            trend_cmip5pi@hemisphere = hemisphere
            trend_cmip5pi@season = season
            trend_cmip5pi@syear = syear
            trend_cmip5pi@eyear = eyear
            trend_cmip5pi@var = "lat"
            trend_cmip5pi@long_name \
              = "trend of "+hemisphere+" Hadley cell edge latitude"
            trend_cmip5pi@diag_script = DIAG_SCRIPT
          end if

        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                isStrSubset(input_file_info[ii]@exp, "historical")) then
          if (isvar("trend_cmip6hist")) then
            tmp = trend_cmip6hist
            delete(trend_cmip6hist)
            trend_cmip6hist = array_append_record(tmp, trend_edge, 0)
            trend_cmip6hist@esize := array_append_record(tmp@esize, esize, 0)
            delete(tmp)
          else
            trend_cmip6hist = (/trend_edge/)
            trend_cmip6hist@esize = esize
            trend_cmip6hist@hemisphere = hemisphere
            trend_cmip6hist@season = season
            trend_cmip6hist@syear = syear
            trend_cmip6hist@eyear = eyear
            trend_cmip6hist@var = "lat"
            trend_cmip6hist@long_name \
              = "trend of "+hemisphere+" Hadley cell edge latitude"
            trend_cmip6hist@diag_script = DIAG_SCRIPT
          end if

        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                input_file_info[ii]@exp.eq."piControl") then
          if (isvar("trend_cmip6pi")) then
            tmp = trend_cmip6pi
            delete(trend_cmip6pi)
            trend_cmip6pi = array_append_record(tmp, trend_edge, 0)
            trend_cmip6pi@esize := array_append_record(tmp@esize, nseg, 0)
            delete(tmp)
          else
            trend_cmip6pi = (/trend_edge/)
            trend_cmip6pi@esize = nseg
            trend_cmip6pi@hemisphere = hemisphere
            trend_cmip6pi@season = season
            trend_cmip6pi@syear = syear
            trend_cmip6pi@eyear = eyear
            trend_cmip6pi@var = "lat"
            trend_cmip6pi@long_name \
              = "trend of "+hemisphere+" Hadley cell edge latitude"
            trend_cmip6pi@diag_script = DIAG_SCRIPT
          end if

        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                any(input_file_info[ii]@exp.eq.(/"amip", "amip-hist"/))) then
          if (isvar("trend_cmip6amip")) then
            tmp = trend_cmip6amip
            delete(trend_cmip6amip)
            trend_cmip6amip = array_append_record(tmp, trend_edge, 0)
            trend_cmip6amip@esize := array_append_record(tmp@esize, esize, 0)
            delete(tmp)
          else
            trend_cmip6amip = (/trend_edge/)
            trend_cmip6amip@esize = esize
            trend_cmip6amip@hemisphere = hemisphere
            trend_cmip6amip@season = season
            trend_cmip6amip@syear = syear
            trend_cmip6amip@eyear = eyear
            trend_cmip6amip@var = "lat"
            trend_cmip6amip@long_name \
              = "trend of "+hemisphere+" Hadley cell edge latitude"
            trend_cmip6amip@diag_script = DIAG_SCRIPT
          end if
        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                input_file_info[ii]@exp.eq."hist-GHG") then
          if (isvar("trend_cmip6GHG")) then
            tmp = trend_cmip6GHG
            delete(trend_cmip6GHG)
            trend_cmip6GHG = array_append_record(tmp, trend_edge, 0)
            trend_cmip6GHG@esize := array_append_record(tmp@esize, esize, 0)
            delete(tmp)
          else
            trend_cmip6GHG = (/trend_edge/)
            trend_cmip6GHG@esize = esize
            trend_cmip6GHG@hemisphere = hemisphere
            trend_cmip6GHG@season = season
            trend_cmip6GHG@syear = syear
            trend_cmip6GHG@eyear = eyear
            trend_cmip6GHG@var = "lat"
            trend_cmip6GHG@long_name \
              = "trend of "+hemisphere+" Hadley cell edge latitude"
            trend_cmip6GHG@diag_script = DIAG_SCRIPT
          end if
        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                input_file_info[ii]@exp.eq."hist-aer") then
          if (isvar("trend_cmip6aer")) then
            tmp = trend_cmip6aer
            delete(trend_cmip6aer)
            trend_cmip6aer = array_append_record(tmp, trend_edge, 0)
            trend_cmip6aer@esize := array_append_record(tmp@esize, esize, 0)
            delete(tmp)
          else
            trend_cmip6aer = (/trend_edge/)
            trend_cmip6aer@esize = esize
            trend_cmip6aer@hemisphere = hemisphere
            trend_cmip6aer@season = season
            trend_cmip6aer@syear = syear
            trend_cmip6aer@eyear = eyear
            trend_cmip6aer@var = "lat"
            trend_cmip6aer@long_name \
              = "trend of "+hemisphere+" Hadley cell edge latitude"
            trend_cmip6aer@diag_script = DIAG_SCRIPT
          end if
        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                input_file_info[ii]@exp.eq."hist-nat") then
          if (isvar("trend_cmip6nat")) then
            tmp = trend_cmip6nat
            delete(trend_cmip6nat)
            trend_cmip6nat = array_append_record(tmp, trend_edge, 0)
            trend_cmip6nat@esize := array_append_record(tmp@esize, esize, 0)
            delete(tmp)
          else
            trend_cmip6nat = (/trend_edge/)
            trend_cmip6nat@esize = esize
            trend_cmip6nat@hemisphere = hemisphere
            trend_cmip6nat@season = season
            trend_cmip6nat@syear = syear
            trend_cmip6nat@eyear = eyear
            trend_cmip6nat@var = "lat"
            trend_cmip6nat@long_name \
              = "trend of "+hemisphere+" Hadley cell edge latitude"
            trend_cmip6nat@diag_script = DIAG_SCRIPT
          end if
        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                input_file_info[ii]@exp.eq."hist-stratO3") then
          if (isvar("trend_cmip6stratO3")) then
            tmp = trend_cmip6stratO3
            delete(trend_cmip6stratO3)
            trend_cmip6stratO3 = array_append_record(tmp, trend_edge, 0)
            trend_cmip6stratO3@esize := array_append_record(tmp@esize, esize, 0)
            delete(tmp)
          else
            trend_cmip6stratO3 = (/trend_edge/)
            trend_cmip6stratO3@esize = esize
            trend_cmip6stratO3@hemisphere = hemisphere
            trend_cmip6stratO3@season = season
            trend_cmip6stratO3@syear = syear
            trend_cmip6stratO3@eyear = eyear
            trend_cmip6stratO3@var = "lat"
            trend_cmip6stratO3@long_name \
              = "trend of "+hemisphere+" Hadley cell edge latitude"
            trend_cmip6stratO3@diag_script = DIAG_SCRIPT
          end if
          
        end if
      end if
      delete(trend_edge)
    end do
    delete(data)
    delete(year)
    delete(ts)
    delete(te)
  end do
  
  ; output model trends
  if (isvar("trend_cmip5hist")) then
    write_data(trend_cmip5hist, hemisphere, "cmip5hist")
  end if
  if (isvar("trend_cmip5pi")) then
    write_data(trend_cmip5pi, hemisphere, "cmip5pi")
  end if
  if (isvar("trend_cmip6hist")) then
    write_data(trend_cmip6hist, hemisphere, "cmip6hist")
  end if
  if (isvar("trend_cmip6pi")) then
    write_data(trend_cmip6pi, hemisphere, "cmip6pi")
  end if
  if (isvar("trend_cmip6amip")) then
    write_data(trend_cmip6amip, hemisphere, "cmip6amip")
  end if
  if (isvar("trend_cmip6GHG")) then
    write_data(trend_cmip6GHG, hemisphere, "cmip6GHG")
  end if
  if (isvar("trend_cmip6aer")) then
    write_data(trend_cmip6aer, hemisphere, "cmip6aer")
  end if
  if (isvar("trend_cmip6nat")) then
    write_data(trend_cmip6nat, hemisphere, "cmip6nat")
  end if
  if (isvar("trend_cmip6stratO3")) then
    write_data(trend_cmip6stratO3, hemisphere, "cmip6stratO3")
  end if
  
  leave_msg(DIAG_SCRIPT, "")

end
