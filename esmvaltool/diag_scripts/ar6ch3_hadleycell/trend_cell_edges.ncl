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
;    20190316 kosa_yu: written.
;    20191017 kosaka_yu: updated for ESMValTool 2.0.0b2
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
local edge, y, lat, t, edge, j, jeq, jmax, jsubpolar, jpolar, j0, j1, jj, subpolar_boudnary, polar_boundary
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
;  print(hemisphere+" lat from "+lat(0)+" to "+lat(dimsizes(lat)-1))
  y = sign/abs(sign)*y
  
  edge = new((/dimsizes(x&year)/), "float", default_fillvalue("float"))
  edge!0 = "year"
  edge&year = x&year
  edge = edge@_FillValue

  subpolar_boundary = 30.
  polar_boundary = 60.

  jsubpolar = ind_nearest_coord(subpolar_boundary, abs(lat), 0)
  jpolar    = ind_nearest_coord(polar_boundary,    abs(lat), 0)
;  print("jsubpolar = "+jsubpolar+" jpolar = "+jpolar)
  
  do t = 0, dimsizes(x&year)-1
    ; Search edge (where y changes from - to +)
    j0 = minind(y(t, 0:jsubpolar))
    j1 = maxind(y(t, j0:jpolar))
    j1 = j0 + j1
;    print("j0 = "+j0+" j1 = "+j1)
    
    do j = j0+1, j1
      if (y(t, j-1).lt.0. .and. y(t, j).gt.0.) then
        edge(t) = tofloat((y(t, j)*lat(j-1) - y(t, j-1)*lat(j))/(y(t, j) - y(t, j-1)))
      elseif (y(t, j-1).lt.0. .and. y(t, j).eq.0. .and. y(t, j+1).gt.0.) then
        edge(t) = tofloat(lat(j))
      end if
      if (.not.ismissing(edge(t))) then
        do jj = j+1, jmax
          if (abs(lat(jj)).le.abs(edge(t))+lat_uncertainty .and. y(t, jj-1).lt.0. .and. y(t, jj).gt.0.) then ; based on Grise et al (2018 J Climate)
            edge(t) = edge@_FillValue
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
  frac_threshold = 0.8  ; At least 80% of the time series is needed for trend calculation.
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
  nc_filename = work_dir + branch + "edge_"+x@season+"_"+x@syear+"-"+x@eyear+"_"+dataset+".nc"

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

  MME = "all"
  if (isatt(diag_script_info, "MME")) then
    MME = diag_script_info@MME
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
  iObs = ind(projList.eq."OBS" .or. projList.eq."ana4mips" .or. projList.eq."obs4mips")

  if (MME.eq."all") then
    iMME = ind(projList.eq."CMIP5".or.projList.eq."CMIP6")
  else
    ensList = metadata_att_as_array(input_file_info, "ensemble")
    do i = 0, dimsizes(ensList)-1
      if (any(projList(i).eq.(/"CMIP5", "CMIP6"/))) then
        if (ensList(i).eq.MME .or. isStrSubset(ensList(i), MME+"f")) then
          if (isvar("iMME")) then
            tmp = iMME
            delete(iMME)
            iMME = array_append_record(tmp, i, 0)
            delete(tmp)
          else
            iMME = (/ i /)
          end if
        end if
      end if
    end do
  end if
  
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
      month_number_bnds = f->month_number_bnds
      if (all(month_number_bnds(0, :).eq.(/1, 12/))) then
        season = "ANN"
      else
        error_msg("f", DIAG_SCRIPT, "", \
                  input_file_info[ii]@filename+": Cannot detect season.")
      end if
    end if
    delete(f)

    data!0 = "year"
    data&year = year
    
    syear = year(0)
    eyear = year(dimsizes(year)-1)
    delete(year)
    
    if (method.eq."mpsi") then
      ;trim plev < 5. for zonal_mpsi function
      ;data = trim_for_mpsi500(data)
      v = data(:, {100499.:501.}, :)
      va = conform_dims(array_append_record(dimsizes(v), (/1/), 0), v, (/0, 1, 2/))
      copy_VarCoords_1(v, va)
      va!2 = v!2
      va&$va!2$ = v&$v!2$
      delete(v)
      delete(data)
      
      ;calculate zonal streamfunction
      ps = va(:, 0, :, :)
      ps = 100000. ; dummy 1000 hPa
      psi = zonal_mpsi_Wrap(va(:, ::-1, :, :), va&lat, va&plev(::-1), ps)
      delete(va)
      delete(ps)

      ;get psi@500hPa
      if (any(psi&plev.eq.50000.)) then
        psi500 = psi(:, {50000.}, :)
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

      ;get Hadley cell edges
      lat_uncertainty = 20.
      if (hemisphere.eq."NH") then
        edge = zero_crossing(psi500, hemisphere, -1, lat_uncertainty) ; where psi500 changes from + to -
      elseif (hemisphere.eq."SH") then
        edge = zero_crossing(psi500, hemisphere, 1, lat_uncertainty) ; where psi500 changes from - to +
      end if
      delete(psi500)
    elseif (method.eq."uas") then
      ;get Hadley cell edges
      lat_uncertainty = 0.
      edge = zero_crossing(data, hemisphere, 1, lat_uncertainty) ; where uas changes from - to +
      delete(data)
    end if
      
;    print("------- "+hemisphere+" edge -------")
;    print(""+edge)
    
    ;calculate trends
    trend_edge = calc_trend(edge, trend_method)
    delete(edge)

    if (any(ii.eq.iObs)) then
      ;output obs trends
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
      ;save model trends
      if (isvar("trend") ) then
        tmp = trend
        delete(trend)
        trend = array_append_record(tmp, trend_edge, 0)
        delete(tmp)
      else
        trend = (/trend_edge/)
        trend@hemisphere = hemisphere
        trend@season = season
        trend@syear = syear
        trend@eyear = eyear
        trend@var = "lat"
        trend@long_name = "trend of "+hemisphere+" Hadley cell edge latitude"
        trend@diag_script = DIAG_SCRIPT
      end if
      if (any(ii.eq.iMME)) then
        if (isvar("trendMME") ) then
          tmp = trendMME
          delete(trendMME)
          trendMME = array_append_record(tmp, trend_edge, 0)
          if (isatt(trendMME, "dataset")) then
            delete(trendMME@dataset)
          end if
          trendMME@dataset = array_append_record(tmp@dataset, input_file_info[ii]@dataset, 0)
          delete(tmp)
        else
          trendMME = (/trend_edge/)
          trendMME@season = season
          trendMME@syear = syear
          trendMME@eyear = eyear
          trendMME@var = "lat"
          trendMME@long_name = "MME mean trend of "+hemisphere+" Hadley cell edge latitude"
          trendMME@MME = MME
          trendMME@dataset = input_file_info[ii]@dataset
          trendMME@diag_script = DIAG_SCRIPT
        end if
      end if
    end if
    delete(trend_edge)
  end do

  ;output model trends
  if (isvar("trend")) then
    write_data(trend, hemisphere, "MME")
  end if
  if (isvar("trendMME")) then
    write_data(dim_avg_Wrap(trendMME), hemisphere, "MMEmean")
  end if

  leave_msg(DIAG_SCRIPT, "")

end