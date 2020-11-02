function get_color(x)
  local color, x
begin
  if (x.eq."CMIP6") then
    color = (/204., 35., 35./)/255.
  elseif (x.eq."CMIP5") then
    color = (/37., 81., 204./)/255.
  else
    if (x.eq."MIROC-ES2L") then
      color = (/184., 95., 182./)/255.
    elseif (x.eq."MRI-ESM2-0") then
      color = (/173., 255., 47./)/255.
    elseif (x.eq."EC-Earth3-Veg-LR") then
      color = (/124., 99., 184./)/255.
    else
      color = (/204., 35., 35./)/255.  ; "deeppink4"
    end if
  end if
  return(color)
end