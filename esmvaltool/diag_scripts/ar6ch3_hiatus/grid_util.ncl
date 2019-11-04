; grid_util.ncl
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs: 
;
; History
;    20190319 kosaka_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

function regrid(x, target, opt)
local method, y, loni, lati, lono, lato, regrid_scheme, dx, dy, xsize, ysize, cres, \
      irectilinear, orectilinear, regopt
begin
  regrid_scheme = "linear"
  if (opt) then
    if (isatt(opt, "regrid_scheme")) then
      method = opt@regrid_scheme
    end if
  end if
  loni = x&lon
  lati = x&lat

  if (isstring(target)) then
    cres = str_split(target, "x")
    if (is_string_numeric(cres(0))) then
      dx = tofloat(cres(0))
    else
      error_msg("f", DIAG_SCRIPT, "", "regrid: Cannot read the target grid :"+target)
    end if
    if (is_string_numeric(cres(1))) then
      dy = tofloat(cres(1))
    else
      error_msg("f", DIAG_SCRIPT, "", "regrid: Cannot read the target grid :"+target)
    end if
    xsize = toint(360./dx)
    ysize = toint(180./dy)
    lono = fspan(dx/2., 360.-dx/2., xsize)
    lato = fspan(-90.+dy/2., 90.-dy/2., ysize)
    if (opt) then
      if (isatt(opt, "lon_offset")) then
        lono = lono + opt@lon_offset
      end if
      if (isatt(opt, "lat_offset")) then
        lato = lato + opt@lat_offset
      end if
    end if
    lono@standard_name = "longitude"
    lono@units = "degrees_east"
    lono@axis = "X"
    lato@standard_name = "latitude"
    lato@units = "degrees_north"
    lato@axis = "Y"
  else
    lono = target&lon
    lato = target&lat
  end if

  if (dimsizes(dimsizes(loni)).eq.1 .and. dimsizes(dimsizes(lati)).eq.1) then
    irectilinear = True
  else
    irectilinear = False
  end if
  if (dimsizes(dimsizes(lono)).eq.1 .and. dimsizes(dimsizes(lato)).eq.1) then
    orectilinear = True
  else
    orectilinear = False
  end if
  
  if (regrid_scheme.eq."area_weighted") then
    if (irectilinear.and.orectilinear) then
      y = area_conserve_remap_Wrap(loni, lati, x, lono, lato, False)
    else
      error_msg("w", DIAG_SCRIPT, "", "regrid: area_weighted regridding of " \
                                      +"non-rectilinear grid is unsupported.")
      y = lono
      copy_VarAtts(x, y)
      y = y@_FillValue
;      y!0 = lat
;      y!1 = lon
;      y&lat = lato
;      y&lon = lono
    end if
  elseif (regrid_scheme.eq."linear") then
    if (irectilinear.and.orectilinear) then
      y = linint2_Wrap(loni, lati, x, True, lono, lato, 0)
    elseif (orectilinear) then
      y = linint2_points_Wrap(loni, lati, x, True, lono, lato, 0)
    else
      error_msg("w", DIAG_SCRIPT, "", "regrid: linear regridding to " \
                                      +"non-rectilinear grid is unsupported.")
      y = lato
      copy_VarAtts(x, y)
      y = y@_FillValue
;      y!0 = "lat"
;      y!1 = "lon"
;      y&lat = lato
;      y&lon = lono
    end if
  elseif (regrid_scheme.eq."nearest") then
    regopt = True
    regopt@method = 1
    y = triple2grid(loni, lati, x, lono, lato, regopt)
  else
    error_msg("w", DIAG_SCRIPT, "", "regrid: Cannot apply the requested regridding scheme :" \
              +regrid_scheme)
    y = x
    y = y@_FillValue
  end if
  return(y)
end

function fillLand(x[*][*][*])
local y, z, i, j, imax, jmax, wgt, tmp, iter, yave
begin
  y = x
  imax = dimsizes(x&lon)-1
  jmax = dimsizes(x&lat)-1
  wgt = new((/3, 3/), typeof(x))
  wgt(2, 0:2) = (/ 1., 2., 1/)
  wgt(1, 0:2) = (/ 2., 4., 2/)
  wgt(0, 0:2) = (/ 1., 2., 1/)
  tmp = x(:, :, 0:2)
  do iter = 0, 3
    if (all(.not.ismissing(y))) then
      break
    end if
;print("iter = "+iter+" #missing = "+num(ismissing(y)))
    z = y
    i = 0
    tmp(:, :,   0) = z(:, :, imax)
    tmp(:, :, 1:2) = z(:, :, 0:1)
    j = 0
    y(:, j, i) = where(ismissing(z(:, j, i)), \
                       wgt_areaave2(tmp(:, j:j+1, 0:2), wgt(1:2, :), 0), \
                       z(:, j, i))
    do j = 1, jmax-1
      y(:, j, i) = where(ismissing(z(:, j, i)), \
                         wgt_areaave2(tmp(:, j-1:j+1, 0:2), wgt, 0), \
                         z(:, j, i))
    end do
    j = jmax
    y(:, j, i) = where(ismissing(z(:, j, i)), \
                       wgt_areaave2(tmp(:, j-1:j, 0:2), wgt(0:1, :), 0), \
                       z(:, j, i))
    do i = 1, imax-1
      j = 0
      y(:, j, i) = where(ismissing(z(:, j, i)), \
                         wgt_areaave2(z(:, j:j+1, i-1:i+1), wgt(1:2, :), 0), \
                         z(:, j, i))
      do j = 1, jmax-1
        y(:, j, i) = where(ismissing(z(:, j, i)), \
                           wgt_areaave2(z(:, j-1:j+1, i-1:i+1), wgt, 0), \
                           z(:, j, i))
      end do
      j = jmax
      y(:, j, i) = where(ismissing(z(:, j, i)), \
                         wgt_areaave2(z(:, j-1:j, i-1:i+1), wgt(0:1, :), 0), \
                         z(:, j, i))
    end do
    i = imax
    tmp(:, :, 0:1) = z(:, :, imax-1:imax)
    tmp(:, :,   2) = z(:, :, 0)
    j = 0
    y(:, j, i) = where(ismissing(z(:, j, i)), \
                       wgt_areaave2(tmp(:, j:j+1, 0:2), wgt(1:2, :), 0), \
                       z(:, j, i))
    do j = 1, jmax-1
      y(:, j, i) = where(ismissing(z(:, j, i)), \
                         wgt_areaave2(tmp(:, j-1:j+1, 0:2), wgt, 0), \
                         z(:, j, i))
    end do
    j = jmax
    y(:, j, i) = where(ismissing(z(:, j, i)), \
                       wgt_areaave2(tmp(:, j-1:j, 0:2), wgt(0:1, :), 0), \
                       z(:, j, i))
  end do
  if (isvar("tmp")) then
    delete(tmp)
  end if
  if (isvar("z")) then
    delete(z)
  end if
  yave = wgt_areaave(y, 1., 1., 0)
  y = where(ismissing(y), conform(y, yave, 0), y)
  return(y)
end

function global_mean(x, individual_hemisphere)
local x, individual_hemisphere, nh, sh, gm, lat, wgty
begin
  lat = x&lat
  if (any(lat.eq.0.)) then
      error_msg("f", DIAG_SCRIPT, "", "Grid point exactly on the equator is UNSUPPORTED.")
  end if
  wgty = latRegWgt(lat, typeof(x), 0)
  if (individual_hemisphere) then
    y = x
    y( :, {:0.}, :) = y@_FillValue
    nh = wgt_areaave_Wrap(y, wgty, 1., 0)  ; Assume equatorially symmetric grid, no grid point on the equator
    y = x
    y( :, {0.:}, :) = y@_FillValue
    sh = wgt_areaave_Wrap(y, wgty, 1., 0)  ; Assume equatorially symmetric grid, no grid point on the equator
    gm = (nh + sh)/2.
  else
    gm = wgt_areaave_Wrap(x, wgty, 1., 0)
  end if
  return(gm)
end
