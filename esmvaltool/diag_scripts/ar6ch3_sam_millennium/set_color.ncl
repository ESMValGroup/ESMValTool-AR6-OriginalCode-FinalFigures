function get_color(x)
  local color, x
begin
  if (x.eq."CMIP6") then
    color = (/204., 35., 35./)/255.  ; "deeppink4"
  elseif (x.eq."CMIP5") then
    color = (/37., 81., 204./)/255.  ; "deepskyblue"
  else
    if (x.eq."MIROC-ES2L") then
      color = (/204., 35., 35./)/255.  ; "deeppink4"
    elseif (x.eq."MRI-ESM2-0") then
      color = (/204., 35., 35./)/255.  ; "deeppink4"
    else
      color = (/204., 35., 35./)/255.  ; "deeppink4"
    end if
  end if
  return(color)
end