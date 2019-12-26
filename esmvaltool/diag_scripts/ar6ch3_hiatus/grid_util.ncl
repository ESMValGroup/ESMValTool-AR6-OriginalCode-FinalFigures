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
;    20191201 kosaka_yu: modified to enable various global averaging.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

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

function global_mean(x, method)
local x, y, method, individual_hemisphere, nh, sh, gm, lat, wgty, area, lats, latn, lat_bnds, ii, ave_band
begin
  lat = x&lat
  d2r = get_d2r(typeof(lat))
  if (any(method.eq.(/"HadCRUT4", "HadCRUT5"/)) then
    if (any(lat.eq.0.)) then
      error_msg("f", DIAG_SCRIPT, "", "Grid point exactly on the equator is UNSUPPORTED.")
    end if
    wgty = latRegWgt(lat, typeof(lat), 0)
    y = x
    y(:, {:0.}, :) = y@_FillValue
    nh = wgt_areaave_Wrap(y, wgty, 1., 0)  ; Assume equatorially symmetric grid, no grid point on the equator
    y = x
    y(:, {0.:}, :) = y@_FillValue
    sh = wgt_areaave_Wrap(y, wgty, 1., 0)  ; Assume equatorially symmetric grid, no grid point on the equator
    gm = x(:, 0, 0)
    gm = (nh + sh)/2.
  elseif (method.eq."GISTEMP") then
    gm = x(:, 0, 0)
    gm = 0.
    do ii = 0, 7
      if (ii.eq.0) then
        latn = 90.
        lats = 64.2
        area = 0.05
      elseif (ii.eq.1) then
        latn = 64.2
        lats = 44.4
        area = 0.1
      elseif (ii.eq.2) then
        latn = 44.4
        lats = 23.6
        area = 0.15
      elseif (ii.eq.3) then
        latn = 23.6
        lats = 0.
        area = 0.2
      elseif (ii.eq.4) then
        latn = 0.
        lats = -23.6
        area = 0.2
      elseif (ii.eq.5) then
        latn = -23.6
        lats = -44.4
        area = 0.15
      elseif (ii.eq.6) then
        latn = -44.4
        lats = -64.2
        area = 0.1
      elseif (ii.eq.7) then
        latn = -64.2
        lats = -90.
        area = 0.05
      end if
      y = x(:, {lats:latn}, :)
      jmax = dimsizes(y&lat)
      lat_bnds = new((/jmax, 2/), typeof(y&lat))
      wgty = new(jmax, typeof(y&lat))
      do j = 1, jmax-1
        lat_bnds(j-1, 1) = (y&lat(j-1) + y&lat(j))/2.
        lat_bnds(j, 0) = lat_bnds(j-1, 1)
      end do
      if (isMonotonic(y&lat).eq.1) then
        lat_bnds(0, 0) = lats
        lat_bnds(jmax-1, 1) = latn
      else
        lat_bnds(0, 0) = latn
        lat_bnds(jmax-1, 1) = lats
      end if
      wgty = (cos(lat_bnds(:, 0)*d2r) + cos(lat_bnds(:, 1)*d2r)) \
              * abs(lat_bnds(:, 1) - lat_bnds(:, 0)) / 2.
      ave_band = wgt_areaave(y, wgty, 1., 0)
      gm = gm + ave_band * area
      delete(y)
      delete(ave_band)
      delete(wgty)
      delete(lat_bnds)
    end do
  else
    wgty = latRegWgt(lat, typeof(lat), 0)
    gm = wgt_areaave_Wrap(x, wgty, 1., 0)
  end if
  return(gm)
end
