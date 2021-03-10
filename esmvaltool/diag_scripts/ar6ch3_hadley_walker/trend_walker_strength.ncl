; trend_walker_edges
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs:
;
; History
;    20210224 kosaka_yu: added provenance.
;    20200929 kosaka_yu: support ensemble reanalyses.
;    20200918 kosaka_yu: output model id as attributes.
;    20200915 kosaka_yu: written.
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
;  copy_VarMeta(x(0), tr)
  tr@units = x@units+"/yr"
  tr@_FillValue = x@_FillValue
  
  delete(tmp)
  return(tr)
end

function write_data(x, dataset)
local nc_filename, nc_outfile, work_dir, y
begin
  work_dir = config_user_info@work_dir + "/"
  system("mkdir -p " + work_dir)

  y = x
  y!0 = "ensemble"
  y&ensemble = ispan(1, dimsizes(y), 1)
    nc_filename = work_dir + "WalkerStrength_" \
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
  return(nc_outfile)
end

begin

  enter_msg(DIAG_SCRIPT, "")

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
  
  var = "psl"
  varList := metadata_att_as_array(variable_info, "short_name")
  if (all(varList.ne.var)) then
    error_msg("f", DIAG_SCRIPT, "", "variable "+var+" is missing.")
  end if

  projList = metadata_att_as_array(input_file_info, "project")
  iObs = ind(projList.eq."OBS" .or. projList.eq."OBS6" .or. \
             projList.eq."obs" .or. \
             projList.eq."ana4mips" .or. projList.eq."obs4mips")

  ; East box
  W1 = 200.
  E1 = 280.
  S1 = -5.
  N1 = 5.
  ; West box
  W2 = 80.
  E2 = 160.
  S2 = -5.
  N2 = 5.

  d2r = get_d2r("double")

  ndatasets = ListCount(input_file_info)

  dataList = get_unique_values(metadata_att_as_array(input_file_info, "dataset"))
  
; ========================================================================

  do ii = 0, ndatasets - 1
    if (input_file_info[ii]@short_name.ne.var) then
      continue
    end if
    tmp = read_data(input_file_info[ii])
    data = todouble(tmp)
    copy_VarMeta(tmp, data)
    delete(tmp)
    
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
      ts = new(nseg, "integer")
      te = new(nseg, "integer")
      syear = year(0)
      eyear = year(dimsizes(year)-1)
      if (any(ii.eq.iObs).and.(syear.gt.syear_trend.or.eyear.lt.eyear_trend)) then
        nyears_missing = eyear_trend - min((/eyear, eyear_trend/)) + \
                         max((/syear, syear_trend/)) - syear_trend
        if (nyears_missing.lt.2) then
          ts(0) = (/ind(year.eq.max((/syear, syear_trend/)))/)
          te(0) = (/ind(year.eq.min((/eyear, eyear_trend/)))/)
        else
          delete(year)
          delete(data)
          delete(ts)
          delete(te)
          continue
        end if
      else
        ts(0) = (/ind(year.eq.syear_trend)/)
        te(0) = (/ind(year.eq.eyear_trend)/)
      end if
    end if

    ; Get bounds of grid boxes
    if (isfilevar(f, "lat_bnds")) then
      lat_bnds = f->lat_bnds
    end if
    ysize = dimsizes(data&lat)
    if (.not.isvar("lat_bnds") .or. \
      any(abs(lat_bnds(1:ysize-1, 0)-lat_bnds(0:ysize-2, 1)).gt.0.01)) then
      print(input_file_info[ii]@dataset+" : Creating lat_bnds")
      lat = data&lat
      if (isvar("lat_bnds")) then
        delete(lat_bnds)
      end if
      lat_bnds = new((/ysize, 2/), typeof(data&lat))
      lat_bnds(1:ysize-1, 0) = (lat(0:ysize-2) + lat(1:ysize-1))/2.
      lat_bnds(0:ysize-2, 1) = (lat(0:ysize-2) + lat(1:ysize-1))/2.
      lat_bnds(0, 0) \
        = max((/lat_bnds(0, 1) - (lat_bnds(1, 1) - lat_bnds(1, 0)), -90./))
      lat_bnds(ysize-1, 1) \
        = min((/lat_bnds(ysize-1, 0) + (lat_bnds(ysize-2, 1) \
          - lat_bnds(ysize-2, 0)), 90./))
      delete(lat)
    end if

    if (isfilevar(f, "lon_bnds")) then
      lon_bnds = f->lon_bnds
    end if
    xsize = dimsizes(data&lon)
    if (.not.isvar("lon_bnds") .or. \
      any(abs(lon_bnds(1:xsize-1, 0)-lon_bnds(0:xsize-2, 1)).gt.0.01)) then
      print(input_file_info[ii]@dataset+" : Creating lon_bnds")
      lon = data&lon
      if (isvar("lon_bnds")) then
        delete(lon_bnds)
      end if
      lon_bnds = new((/xsize, 2/), typeof(data&lon))
      lon_bnds(1:xsize-1, 0) = (lon(0:xsize-2) + lon(1:xsize-1))/2.
      lon_bnds(0:xsize-2, 1) = (lon(0:xsize-2) + lon(1:xsize-1))/2.
      lon_bnds(0, 0) = lon_bnds(0, 1) - (lon_bnds(1, 1) - lon_bnds(1, 0))
      lon_bnds(xsize-1, 1) \
        = lon_bnds(xsize-1, 0) + (lon_bnds(xsize-2, 1) - lon_bnds(xsize-2, 0))
      delete(lon)
    end if
    delete(f)

    ; Get weights for area averaging
    ; Eastern box
    j0 = ind(lat_bnds(:, 0).le.S1 .and. S1.lt.lat_bnds(:, 1))
    j1 = ind(lat_bnds(:, 0).lt.N1 .and. N1.le.lat_bnds(:, 1))
    i0 = ind(lon_bnds(:, 0).le.W1 .and. W1.lt.lon_bnds(:, 1))
    if (dimsizes(i0).gt.1) then
      i0 := i0(0)
    end if
    i1 = ind(lon_bnds(:, 0).lt.E1 .and. E1.le.lon_bnds(:, 1))
    if (dimsizes(i1).gt.1) then
      i1 := i1(1)
    end if
    dy1 = lat_bnds(:, 0)
    dy1 = 0.
    j = j0
    dy1(j) = (lat_bnds(j, 1) - S1) * (cos(lat_bnds(j, 1)*d2r) + cos(S1*d2r)) / 2.
    do j = j0+1, j1-1
      dy1(j) = (lat_bnds(j, 1) - lat_bnds(j, 0)) \
                * (cos(lat_bnds(j, 1)*d2r) + cos(lat_bnds(j, 0)*d2r)) / 2.
    end do
    j = j1
    dy1(j) = (N1 - lat_bnds(j, 0)) * (cos(N1*d2r) + cos(lat_bnds(j, 0)*d2r)) / 2.
    dx1 = lon_bnds(:, 0)
    dx1 = 0.
    i = i0
    dx1(i) = lon_bnds(i, 1) - W1
    do i = i0+1, i1-1
      dx1(i) = lon_bnds(i, 1) - lon_bnds(i, 0)
    end do
    i = i1
    dx1(i) = E1 - lon_bnds(i, 0)
    delete([/i0, i1, j0, j1/])
    ; Western box
    j0 = ind(lat_bnds(:, 0).le.S2 .and. S2.lt.lat_bnds(:, 1))
    j1 = ind(lat_bnds(:, 0).lt.N2 .and. N2.le.lat_bnds(:, 1))
    i0 = ind(lon_bnds(:, 0).le.W2 .and. W2.lt.lon_bnds(:, 1))
    if (dimsizes(i0).gt.1) then
      i0 := i0(0)
    end if
    i1 = ind(lon_bnds(:, 0).lt.E2 .and. E2.le.lon_bnds(:, 1))
    if (dimsizes(i1).gt.1) then
      i1 := i1(1)
    end if
    dy2 = lat_bnds(:, 0)
    dy2 = 0.
    j = j0
    dy2(j) = (lat_bnds(j, 1) - S2) * (cos(lat_bnds(j, 1)*d2r) + cos(S2*d2r)) / 2.
    do j = j0+1, j1-1
      dy2(j) = (lat_bnds(j, 1) - lat_bnds(j, 0)) \
                * (cos(lat_bnds(j, 1)*d2r) + cos(lat_bnds(j, 0)*d2r)) / 2.
    end do
    j = j1
    dy2(j) = (N2 - lat_bnds(j, 0)) * (cos(N2*d2r) + cos(lat_bnds(j, 0)*d2r)) / 2.
    dx2 = lon_bnds(:, 0)
    dx2 = 0.
    i = i0
    dx2(i) = lon_bnds(i, 1) - W2
    do i = i0+1, i1-1
      dx2(i) = lon_bnds(i, 1) - lon_bnds(i, 0)
    end do
    i = i1
    dx2(i) = E2 - lon_bnds(i, 0)
    delete([/i0, i1, j0, j1/])

    delete(lat_bnds)
    delete(lon_bnds)
    
    do n = 0, nseg-1
      syear = year(ts(n))
      eyear = year(te(n))
    
      ; get psl zonal gradient
      east = wgt_areaave_Wrap(data(ts(n):te(n), :, :), dy1, dx1, 0)
      west = wgt_areaave_Wrap(data(ts(n):te(n), :, :), dy2, dx2, 0)
      dslp = east
      dslp = east - west
      if (typeof(dslp).eq."double") then
        tmp = dslp
        delete(dslp)
        dslp = tofloat(tmp)
        copy_VarMeta(tmp, dslp)
        delete(tmp)
      end if
      delete(east)
      delete(west)

      
      ; calculate trends
      trend_dslp = calc_trend(dslp, trend_method)
      delete(dslp)

      if (any(ii.eq.iObs)) then
        esize = num(metadata_att_as_array(input_file_info, "dataset").eq.input_file_info[ii]@dataset)
        if (esize.eq.1) then
          ; output obs trends
          trendObs = trend_dslp
          trendObs@season = season
          trendObs@syear = syear
          trendObs@eyear = eyear
          trendObs@dataset = input_file_info[ii]@dataset
          trendObs@var = "dslp"
          trendObs@long_name = "trend of Walker circulation strength"
          trendObs@diag_script = DIAG_SCRIPT

          nc_filename = write_data(trendObs, input_file_info[ii]@dataset)
          log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                         "trend", "eq", "other", "kosaka_yu", "", input_file_info[ii]@dataset)
          delete(trendObs)
        else
          if (input_file_info[ii]@dataset.eq."20CRv3") then
            if (isvar("trend_20crv3")) then
              tmp = trend_20crv3
              delete(trend_20crv3)
              trend_20crv3 = array_append_record(tmp, trend_dslp, 0)
              trend_20crv3@esize := array_append_record(tmp@esize, esize, 0)
              delete(tmp)
              tmp = paths_20crv3
              delete(paths_20crv3)
              paths_20crv3 = array_append_record(tmp, input_file_info[ii]@filename, 0)
              delete(tmp)
            else
              trend_20crv3 = (/trend_dslp/)
              trend_20crv3@esize = esize
              trend_20crv3@season = season
              trend_20crv3@syear = syear
              trend_20crv3@eyear = eyear
              trend_20crv3@dataset = input_file_info[ii]@dataset
              trend_20crv3@var = "dslp"
              trend_20crv3@long_name \
                = "trend of Walker circulation strength"
              trend_20crv3@diag_script = DIAG_SCRIPT
              paths_20crv3 = (/input_file_info[ii]@filename/)
            end if
          elseif (input_file_info[ii]@dataset.eq."20CRv2c") then
            if (isvar("trend_20crv2c")) then
              tmp = trend_20crv2c
              delete(trend_20crv2c)
              trend_20crv2c = array_append_record(tmp, trend_dslp, 0)
              trend_20crv2c@esize := array_append_record(tmp@esize, esize, 0)
              delete(tmp)
              tmp = paths_20crv2c
              delete(paths_20crv2c)
              paths_20crv2c = array_append_record(tmp, input_file_info[ii]@filename, 0)
              delete(tmp)
            else
              trend_20crv2c = (/trend_dslp/)
              trend_20crv2c@esize = esize
              trend_20crv2c@season = season
              trend_20crv2c@syear = syear
              trend_20crv2c@eyear = eyear
              trend_20crv2c@dataset = input_file_info[ii]@dataset
              trend_20crv2c@var = "dslp"
              trend_20crv2c@long_name \
                = "trend of Walker circulation strength"
              trend_20crv2c@diag_script = DIAG_SCRIPT
              paths_20crv2c = (/input_file_info[ii]@filename/)
            end if
          elseif (input_file_info[ii]@dataset.eq."20CRv2") then
            if (isvar("trend_20crv2")) then
              tmp = trend_20crv2
              delete(trend_20crv2)
              trend_20crv2 = array_append_record(tmp, trend_dslp, 0)
              trend_20crv2@esize := array_append_record(tmp@esize, esize, 0)
              delete(tmp)
              tmp = paths_20crv2
              delete(paths_20crv2)
              paths_20crv2 = array_append_record(tmp, input_file_info[ii]@filename, 0)
              delete(tmp)
            else
              trend_20crv2 = (/trend_dslp/)
              trend_20crv2@esize = esize
              trend_20crv2@season = season
              trend_20crv2@syear = syear
              trend_20crv2@eyear = eyear
              trend_20crv2@dataset = input_file_info[ii]@dataset
              trend_20crv2@var = "dslp"
              trend_20crv2@long_name \
                = "trend of Walker circulation strength"
              trend_20crv2@diag_script = DIAG_SCRIPT
              paths_20crv2 = (/input_file_info[ii]@filename/)
            end if
          elseif (input_file_info[ii]@dataset.eq."CERA-20C") then
            if (isvar("trend_cera20c")) then
              tmp = trend_cera20c
              delete(trend_cera20c)
              trend_cera20c = array_append_record(tmp, trend_dslp, 0)
              trend_cera20c@esize := array_append_record(tmp@esize, esize, 0)
              delete(tmp)
              tmp = paths_cera20c
              delete(paths_cera20c)
              paths_cera20c = array_append_record(tmp, input_file_info[ii]@filename, 0)
              delete(tmp)
            else
              trend_cera20c = (/trend_dslp/)
              trend_cera20c@esize = esize
              trend_cera20c@season = season
              trend_cera20c@syear = syear
              trend_cera20c@eyear = eyear
              trend_cera20c@dataset = input_file_info[ii]@dataset
              trend_cera20c@var = "dslp"
              trend_cera20c@long_name \
                = "trend of Walker circulation strength"
              trend_cera20c@diag_script = DIAG_SCRIPT
              paths_cera20c = (/input_file_info[ii]@filename/)
            end if
          end if
        end if
      else
        ; save model trends
        esize = num(metadata_att_as_array(input_file_info, "project").eq.input_file_info[ii]@project.and.\
                    metadata_att_as_array(input_file_info, "exp").eq.input_file_info[ii]@exp.and.\
                    metadata_att_as_array(input_file_info, "dataset").eq.input_file_info[ii]@dataset)
        model_id = ind(input_file_info[ii]@dataset.eq.dataList)
        if (input_file_info[ii]@project.eq."CMIP5" .and. \
            isStrSubset(input_file_info[ii]@exp, "historical")) then
          if (isvar("trend_cmip5hist")) then
            tmp = trend_cmip5hist
            delete(trend_cmip5hist)
            trend_cmip5hist = array_append_record(tmp, trend_dslp, 0)
            trend_cmip5hist@esize := array_append_record(tmp@esize, esize, 0)
            trend_cmip5hist@model_id := array_append_record(tmp@model_id, model_id, 0)
            trend_cmip5hist@dataset := tmp@dataset+","+input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            delete(tmp)
            tmp = paths_cmip5hist
            delete(paths_cmip5hist)
            paths_cmip5hist = array_append_record(tmp, input_file_info[ii]@filename, 0)
            delete(tmp)
          else
            trend_cmip5hist = (/trend_dslp/)
            trend_cmip5hist@dataset = input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            trend_cmip5hist@model_id = model_id
            trend_cmip5hist@esize = esize
            trend_cmip5hist@season = season
            trend_cmip5hist@syear = syear
            trend_cmip5hist@eyear = eyear
            trend_cmip5hist@var = "dslp"
            trend_cmip5hist@long_name \
              = "trend of Walker circulation strength"
            trend_cmip5hist@diag_script = DIAG_SCRIPT
            paths_cmip5hist = (/input_file_info[ii]@filename/)
          end if
        elseif (input_file_info[ii]@project.eq."CMIP5" .and. \
                input_file_info[ii]@exp.eq."piControl") then
          if (isvar("trend_cmip5pi")) then
            tmp = trend_cmip5pi
            delete(trend_cmip5pi)
            trend_cmip5pi = array_append_record(tmp, trend_dslp, 0)
            trend_cmip5pi@esize := array_append_record(tmp@esize, nseg, 0)
            trend_cmip5pi@model_id := array_append_record(tmp@model_id, model_id, 0)
            trend_cmip5pi@dataset := tmp@dataset+","+input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            delete(tmp)
            tmp = paths_cmip5pi
            delete(paths_cmip5pi)
            paths_cmip5pi = array_append_record(tmp, input_file_info[ii]@filename, 0)
            delete(tmp)
          else
            trend_cmip5pi = (/trend_dslp/)
            trend_cmip5pi@dataset = input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            trend_cmip5pi@model_id = model_id
            trend_cmip5pi@esize = nseg
            trend_cmip5pi@season = season
            trend_cmip5pi@syear = syear
            trend_cmip5pi@eyear = eyear
            trend_cmip5pi@var = "dslp"
            trend_cmip5pi@long_name \
              = "trend of Walker circulation strength"
            trend_cmip5pi@diag_script = DIAG_SCRIPT
            paths_cmip5pi = (/input_file_info[ii]@filename/)
          end if
        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                isStrSubset(input_file_info[ii]@exp, "historical")) then
          if (isvar("trend_cmip6hist")) then
            tmp = trend_cmip6hist
            delete(trend_cmip6hist)
            trend_cmip6hist = array_append_record(tmp, trend_dslp, 0)
            trend_cmip6hist@esize := array_append_record(tmp@esize, esize, 0)
            trend_cmip6hist@model_id := array_append_record(tmp@model_id, model_id, 0)
            trend_cmip6hist@dataset := tmp@dataset+","+input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            delete(tmp)
            tmp = paths_cmip6hist
            delete(paths_cmip6hist)
            paths_cmip6hist = array_append_record(tmp, input_file_info[ii]@filename, 0)
            delete(tmp)
          else
            trend_cmip6hist = (/trend_dslp/)
            trend_cmip6hist@dataset = input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            trend_cmip6hist@model_id = model_id
            trend_cmip6hist@esize = esize
            trend_cmip6hist@season = season
            trend_cmip6hist@syear = syear
            trend_cmip6hist@eyear = eyear
            trend_cmip6hist@var = "dslp"
            trend_cmip6hist@long_name \
              = "trend of Walker circulation strength"
            trend_cmip6hist@diag_script = DIAG_SCRIPT
            paths_cmip6hist = (/input_file_info[ii]@filename/)
          end if
        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                input_file_info[ii]@exp.eq."piControl") then
          if (isvar("trend_cmip6pi")) then
            tmp = trend_cmip6pi
            delete(trend_cmip6pi)
            trend_cmip6pi = array_append_record(tmp, trend_dslp, 0)
            trend_cmip6pi@esize := array_append_record(tmp@esize, nseg, 0)
            trend_cmip6pi@model_id := array_append_record(tmp@model_id, model_id, 0)
            trend_cmip6pi@dataset := tmp@dataset+","+input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            delete(tmp)
            tmp = paths_cmip6pi
            delete(paths_cmip6pi)
            paths_cmip6pi = array_append_record(tmp, input_file_info[ii]@filename, 0)
            delete(tmp)
          else
            trend_cmip6pi = (/trend_dslp/)
            trend_cmip6pi@dataset = input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            trend_cmip6pi@model_id = model_id
            trend_cmip6pi@esize = nseg
            trend_cmip6pi@season = season
            trend_cmip6pi@syear = syear
            trend_cmip6pi@eyear = eyear
            trend_cmip6pi@var = "dslp"
            trend_cmip6pi@long_name \
              = "trend of Walker circulation strength"
            trend_cmip6pi@diag_script = DIAG_SCRIPT
            paths_cmip6pi = (/input_file_info[ii]@filename/)
          end if
        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                input_file_info[ii]@exp.eq."amip") then
          if (isvar("trend_cmip6amip")) then
            tmp = trend_cmip6amip
            delete(trend_cmip6amip)
            trend_cmip6amip = array_append_record(tmp, trend_dslp, 0)
            trend_cmip6amip@esize := array_append_record(tmp@esize, esize, 0)
            trend_cmip6amip@model_id := array_append_record(tmp@model_id, model_id, 0)
            trend_cmip6amip@dataset := tmp@dataset+","+input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            delete(tmp)
            tmp = paths_cmip6amip
            delete(paths_cmip6amip)
            paths_cmip6amip = array_append_record(tmp, input_file_info[ii]@filename, 0)
            delete(tmp)
          else
            trend_cmip6amip = (/trend_dslp/)
            trend_cmip6amip@dataset = input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            trend_cmip6amip@model_id = model_id
            trend_cmip6amip@esize = esize
            trend_cmip6amip@season = season
            trend_cmip6amip@syear = syear
            trend_cmip6amip@eyear = eyear
            trend_cmip6amip@var = "dslp"
            trend_cmip6amip@long_name \
              = "trend of Walker circulation strength"
            trend_cmip6amip@diag_script = DIAG_SCRIPT
            paths_cmip6amip = (/input_file_info[ii]@filename/)
          end if
        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                input_file_info[ii]@exp.eq."amip-hist") then
          if (isvar("trend_cmip6amiphist")) then
            tmp = trend_cmip6amiphist
            delete(trend_cmip6amiphist)
            trend_cmip6amiphist = array_append_record(tmp, trend_dslp, 0)
            trend_cmip6amiphist@esize := array_append_record(tmp@esize, esize, 0)
            trend_cmip6amiphist@model_id := array_append_record(tmp@model_id, model_id, 0)
            trend_cmip6amiphist@dataset := tmp@dataset+","+input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            delete(tmp)
            tmp = paths_cmip6amiphist
            delete(paths_cmip6amiphist)
            paths_cmip6amiphist = array_append_record(tmp, input_file_info[ii]@filename, 0)
            delete(tmp)
          else
            trend_cmip6amiphist = (/trend_dslp/)
            trend_cmip6amiphist@dataset = input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            trend_cmip6amiphist@model_id = model_id
            trend_cmip6amiphist@esize = esize
            trend_cmip6amiphist@season = season
            trend_cmip6amiphist@syear = syear
            trend_cmip6amiphist@eyear = eyear
            trend_cmip6amiphist@var = "dslp"
            trend_cmip6amiphist@long_name \
              = "trend of Walker circulation strength"
            trend_cmip6amiphist@diag_script = DIAG_SCRIPT
            paths_cmip6amiphist = (/input_file_info[ii]@filename/)
          end if
        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                input_file_info[ii]@exp.eq."hist-GHG") then
          if (isvar("trend_cmip6GHG")) then
            tmp = trend_cmip6GHG
            delete(trend_cmip6GHG)
            trend_cmip6GHG = array_append_record(tmp, trend_dslp, 0)
            trend_cmip6GHG@esize := array_append_record(tmp@esize, esize, 0)
            trend_cmip6GHG@model_id := array_append_record(tmp@model_id, model_id, 0)
            trend_cmip6GHG@dataset := tmp@dataset+","+input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            delete(tmp)
            tmp = paths_cmip6GHG
            delete(paths_cmip6GHG)
            paths_cmip6GHG = array_append_record(tmp, input_file_info[ii]@filename, 0)
            delete(tmp)
          else
            trend_cmip6GHG = (/trend_dslp/)
            trend_cmip6GHG@dataset = input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            trend_cmip6GHG@model_id = model_id
            trend_cmip6GHG@esize = esize
            trend_cmip6GHG@season = season
            trend_cmip6GHG@syear = syear
            trend_cmip6GHG@eyear = eyear
            trend_cmip6GHG@var = "dslp"
            trend_cmip6GHG@long_name \
              = "trend of Walker circulation strength"
            trend_cmip6GHG@diag_script = DIAG_SCRIPT
            paths_cmip6GHG = (/input_file_info[ii]@filename/)
          end if
        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                input_file_info[ii]@exp.eq."hist-aer") then
          if (isvar("trend_cmip6aer")) then
            tmp = trend_cmip6aer
            delete(trend_cmip6aer)
            trend_cmip6aer = array_append_record(tmp, trend_dslp, 0)
            trend_cmip6aer@esize := array_append_record(tmp@esize, esize, 0)
            trend_cmip6aer@model_id := array_append_record(tmp@model_id, model_id, 0)
            trend_cmip6aer@dataset := tmp@dataset+","+input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            delete(tmp)
            tmp = paths_cmip6aer
            delete(paths_cmip6aer)
            paths_cmip6aer = array_append_record(tmp, input_file_info[ii]@filename, 0)
            delete(tmp)
          else
            trend_cmip6aer = (/trend_dslp/)
            trend_cmip6aer@dataset = input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            trend_cmip6aer@model_id = model_id
            trend_cmip6aer@esize = esize
            trend_cmip6aer@season = season
            trend_cmip6aer@syear = syear
            trend_cmip6aer@eyear = eyear
            trend_cmip6aer@var = "dslp"
            trend_cmip6aer@long_name \
              = "trend of Walker circulation strength"
            trend_cmip6aer@diag_script = DIAG_SCRIPT
            paths_cmip6aer = (/input_file_info[ii]@filename/)
          end if
        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                input_file_info[ii]@exp.eq."hist-nat") then
          if (isvar("trend_cmip6nat")) then
            tmp = trend_cmip6nat
            delete(trend_cmip6nat)
            trend_cmip6nat = array_append_record(tmp, trend_dslp, 0)
            trend_cmip6nat@esize := array_append_record(tmp@esize, esize, 0)
            trend_cmip6nat@model_id := array_append_record(tmp@model_id, model_id, 0)
            trend_cmip6nat@dataset := tmp@dataset+","+input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            delete(tmp)
            tmp = paths_cmip6nat
            delete(paths_cmip6nat)
            paths_cmip6nat = array_append_record(tmp, input_file_info[ii]@filename, 0)
            delete(tmp)
          else
            trend_cmip6nat = (/trend_dslp/)
            trend_cmip6nat@dataset = input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            trend_cmip6nat@model_id = model_id
            trend_cmip6nat@esize = esize
            trend_cmip6nat@season = season
            trend_cmip6nat@syear = syear
            trend_cmip6nat@eyear = eyear
            trend_cmip6nat@var = "dslp"
            trend_cmip6nat@long_name \
              = "trend of Walker circulation strength"
            trend_cmip6nat@diag_script = DIAG_SCRIPT
            paths_cmip6nat = (/input_file_info[ii]@filename/)
          end if
        elseif (input_file_info[ii]@project.eq."CMIP6" .and. \
                input_file_info[ii]@exp.eq."hist-stratO3") then
          if (isvar("trend_cmip6stratO3")) then
            tmp = trend_cmip6stratO3
            delete(trend_cmip6stratO3)
            trend_cmip6stratO3 = array_append_record(tmp, trend_dslp, 0)
            trend_cmip6stratO3@esize := array_append_record(tmp@esize, esize, 0)
            trend_cmip6stratO3@model_id := array_append_record(tmp@model_id, model_id, 0)
            trend_cmip6stratO3@dataset := tmp@dataset+","+input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            delete(tmp)
            tmp = paths_cmip6stratO3
            delete(paths_cmip6stratO3)
            paths_cmip6stratO3 = array_append_record(tmp, input_file_info[ii]@filename, 0)
            delete(tmp)
          else
            trend_cmip6stratO3 = (/trend_dslp/)
            trend_cmip6stratO3@dataset = input_file_info[ii]@dataset+"/"+input_file_info[ii]@ensemble
            trend_cmip6stratO3@model_id = model_id
            trend_cmip6stratO3@esize = esize
            trend_cmip6stratO3@season = season
            trend_cmip6stratO3@syear = syear
            trend_cmip6stratO3@eyear = eyear
            trend_cmip6stratO3@var = "dslp"
            trend_cmip6stratO3@long_name \
              = "trend of Walker circulation strength"
            trend_cmip6stratO3@diag_script = DIAG_SCRIPT
            paths_cmip6stratO3 = (/input_file_info[ii]@filename/)
          end if
          
        end if
      end if
      delete(trend_dslp)
    end do
    delete(data)
    delete(year)
    delete(ts)
    delete(te)
    delete(dy1)
    delete(dx1)
    delete(dy2)
    delete(dx2)
  end do
  
  ; output ensemble obs/reanalysis
  if (isvar("trend_20crv3")) then
    nc_filename = write_data(trend_20crv3, "20CRv3")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_20crv3)
  end if
  if (isvar("trend_20crv2c")) then
    nc_filename = write_data(trend_20crv2c, "20CRv2c")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_20crv2c)
  end if
  if (isvar("trend_20crv2")) then
    nc_filename = write_data(trend_20crv2, "20CRv2")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_20crv2)
  end if
  if (isvar("trend_cera20c")) then
    nc_filename = write_data(trend_cera20c, "CERA-20C")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_cera20c)
  end if
  
  ; output model trends
  if (isvar("trend_cmip5hist")) then
    nc_filename = write_data(trend_cmip5hist, "cmip5hist")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_cmip5hist)
  end if
  if (isvar("trend_cmip5pi")) then
    nc_filename = write_data(trend_cmip5pi, "cmip5pi")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_cmip5pi)
  end if
  if (isvar("trend_cmip6hist")) then
    nc_filename = write_data(trend_cmip6hist, "cmip6hist")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_cmip6hist)
  end if
  if (isvar("trend_cmip6pi")) then
    nc_filename = write_data(trend_cmip6pi, "cmip6pi")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_cmip6pi)
  end if
  if (isvar("trend_cmip6amip")) then
    nc_filename = write_data(trend_cmip6amip, "cmip6amip")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_cmip6amip)
  end if
  if (isvar("trend_cmip6amiphist")) then
    nc_filename = write_data(trend_cmip6amiphist, "cmip6amip-hist")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_cmip6amiphist)
  end if
  if (isvar("trend_cmip6GHG")) then
    nc_filename = write_data(trend_cmip6GHG, "cmip6GHG")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_cmip6GHG)
  end if
  if (isvar("trend_cmip6aer")) then
    nc_filename = write_data(trend_cmip6aer, "cmip6aer")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_cmip6aer)
  end if
  if (isvar("trend_cmip6nat")) then
    nc_filename = write_data(trend_cmip6nat, "cmip6nat")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_cmip6nat)
  end if
  if (isvar("trend_cmip6stratO3")) then
    nc_filename = write_data(trend_cmip6stratO3, "cmip6stratO3")
    log_provenance(nc_filename, "n/a", "trend of Walker circulation strength", \
                   "trend", "eq", "other", "kosaka_yu", "", paths_cmip6stratO3)
  end if
  
  leave_msg(DIAG_SCRIPT, "")

end
