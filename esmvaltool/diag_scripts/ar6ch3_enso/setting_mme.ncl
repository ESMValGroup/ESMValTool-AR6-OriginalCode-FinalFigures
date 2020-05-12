function make_poly(x)
local dims, y, xx, x5, x95, yy
begin
  dims = dimsizes(x)
  y = new((/2, dims(1)/), typeof(x))
  do t = 0, dims(1)-1
    xx = x(:, t)
    qsort(xx)
    x5  = round(0.05*dims(0), 3) - 1
    x95 = round(0.95*dims(0), 3) - 1
    y(0, t) = xx(x5)
    y(1, t) = xx(x95)
  end do
  delete(xx)
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
