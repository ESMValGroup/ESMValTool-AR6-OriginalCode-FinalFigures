function weighted_mean(x, w)
local x, w, y
begin
  y = dim_sum_n(x*conform(x, w, 0), 0)
  y = y / dim_sum_n(where(ismissing(x), w@_FillValue, conform(x, w, 0)), 0)
  return(y)
end
  
function weighted_percentile(x, w, t)
local x, w, xx, ww, wsum, j, k, dims, idx
begin
  xx = x
  dims = dimsizes(xx)
  if (dimsizes(dims).eq.1) then
    idx = dim_pqsort(xx, 2)
    wsum = sum(where(ismissing(x), w@_FillValue, w))
    ww = 0.
    do j = 0, dims(0)-1
      if (.not.ismissing(xx(j))) then
        ww = ww + w(idx(j))
        if (ww/wsum.eq.t) then
          y = xx(j)
          break
        elseif (ww/wsum.gt.t) then
          y = (xx(j) + xx(j-1))/2.
          break
        end if
      end if
    end do
  else
    y = new(dims(1), typeof(x))
    idx = dim_pqsort_n(xx, 2, 0)
    do k = 0, dims(1)-1
      wsum = sum(where(ismissing(x(:, k)), w@_FillValue, w))
      ww = 0.
      do j = 0, dims(0)-1
        if (.not.ismissing(xx(j, k))) then
          ww = ww + w(idx(j, k))
          if (ww/wsum.eq.t) then
            y(k) = xx(j, k)
            break
          elseif (ww/wsum.gt.t) then
            y(k) = (xx(j, k) + xx(j-1, k))/2.
            break
          end if
        end if
      end do
    end do
  end if
  return(y)
end

function make_poly(x)
local dims, y, xx, x5, x95, yy
begin
  dims = dimsizes(x)
  y = new((/2, dims(1)/), typeof(x))
  if (dims(0).eq.2) then
    y = x
  elseif (dims(0).gt.2) then
    do t = 0, dims(1)-1
      xx = x(:, t)
      qsort(xx)
      x5  = round(0.05*dims(0), 3) - 1
      x95 = round(0.95*dims(0), 3) - 1
      y(0, t) = xx(x5)
      y(1, t) = xx(x95)
    end do
    delete(xx)
  end if
  yy = array_append_record(y(0, :), y(1, ::-1), 0)
  delete(y)
  return(yy)
end

function get_color(x)
local color
begin
  if (x@project.eq."CMIP6") then
    color = (/204., 35., 35./)/255.  ; "deeppink4"
  elseif (x@project.eq."CMIP5") then
    color = (/37., 81., 204./)/255.  ; "deepskyblue"
  elseif (any(x@project.eq.(/"OBS", "OBS6", "obs", \
                             "obs4mips", "ana4mips"/))) then
    color = (/0., 0., 0./)
  end if
  return(color)
end

function get_lineindex(x)
local lidx
begin
  if (any(x@project.eq.(/"CMIP5", "CMIP6"/))) then
    lidx = 0
  elseif (any(x@project.eq.(/"OBS", "OBS6", "obs", \
                             "obs4mips", "ana4mips"/))) then
    if (x@dataset.eq."HadISST") then
      lidx = 0
    elseif (x@dataset.eq."ERSSTv5") then
      lidx = 1
    end if
  end if
  return(lidx)
end

function get_markerindex(x)
local midx
begin
  if (x@project.eq."CMIP6") then
    midx = 16
  elseif (x@project.eq."CMIP5") then
    midx = 4
  else
    midx = 6
  end if
  return(midx)
end
