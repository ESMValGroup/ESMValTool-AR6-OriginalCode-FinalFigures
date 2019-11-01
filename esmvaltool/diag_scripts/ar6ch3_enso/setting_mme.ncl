function make_poly(x)
local xave, xstdv, x0, x1, y
begin
  xave  = dim_avg_n(x, 0)
  xstdv = dim_stddev_n(x, 0)
  x0 = xave + xstdv
  x1 = xave - xstdv
  y = array_append_record(x0, x1(::-1), 0)
  return(y)
end

function get_color(x)
local color
begin
  if (x@project.eq."CMIP6") then
    color = (/204., 35., 35./)/255. ; "deeppink4"
  elseif (x@project.eq."CMIP5") then
    color = (/37., 81., 204./)/255. ; "deepskyblue"
  elseif (any(x@project.eq.(/"OBS", "OBS6", "obs", "obs4mips", "ana4mips"/))) then
    color = (/0., 0., 0./)
  end if
  return(color)
end

function get_lineindex(x)
local lidx
begin
  if (x@project.eq."CMIP5") then
    lidx = 0 ;1
  else
    lidx = 0
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
