; ****************************************************************************
; map.ncl
;
; Description:
; Generates a global map with shapefiles for Fig. 3.9
;
; Creator: Lisa Bock (lisa.bock@dlr.de)
; Creation Date: 05 February 2021
; ****************************************************************************

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

;----------------------------------------------------------------------
; This function creates a cylindrical equidistant map of Australia
; so you you can add polylines, polygons, or point data to it later.
;----------------------------------------------------------------------
function create_map(wks,title)
local a, res2
begin
  res2               = True
  res2@gsnMaximize   = True
  res2@gsnDraw       = False
  res2@gsnFrame      = False

  res2@mpOutlineOn   = True
  ;res2@mpFillOn      = False  
  res2@mpDataBaseVersion = "MediumRes"

  res2@mpGeophysicalLineColor = "gray42"
  res2@mpGeophysicalLineThicknessF = 4. ;6.
  res2@mpPerimOn    = False
  res2@mpGridLineColor   = "transparent"
  res2@mpGridAndLimbOn   = True
  res2@mpGridLineThicknessF = 6.
  res2mpOutlineOn = True
  ;res2@mpGridLatSpacingF     = 180
  ;res2@mpGridLonSpacingF     = 360


;---Turn on fancier tickmark labels.
;  res2@pmTickMarkDisplayMode = "Always"

;;---Zoom in on area of interest
;  res2@mpLimitMode           = "LatLon"
;  res2@mpMinLatF             = -45
;  res2@mpMaxLatF             =  -6
;  res2@mpMinLonF             = 110
;  res2@mpMaxLonF             = 155

  res2@mpProjection  = "Robinson"

  res2@tiMainString          = title

;  res2@mpOceanFillColor       = "grey80" ;"lightblue1"            ; array index in color map
;  res2@mpLandFillColor        = "grey36" ;"burlywood1"          ; for land, ocean, and inland h20
;  res2@mpInlandWaterFillColor = "grey80" ;"lightblue1"

;---Create map.
  map = gsn_csm_map(wks,res2)

  return(map)
end

;----------------------------------------------------------------------
; Main code
;----------------------------------------------------------------------
begin
;
; This shapefile was obtained from:
;
;  http://e-atlas.org.au/content/au-ga-river-basins-1997
;
  europe = "/work/bd0854/b309059/esmvaltool/auxiliary_data/europe.shp"
  africa = "/work/bd0854/b309059/esmvaltool/auxiliary_data/africa.shp"
  asia = "/work/bd0854/b309059/esmvaltool/auxiliary_data/asia.shp"
  australasia = "/work/bd0854/b309059/esmvaltool/auxiliary_data/australasia.shp"
  antarctica = "/work/bd0854/b309059/esmvaltool/auxiliary_data/antarctica.shp"
  polar = "/work/bd0854/b309059/esmvaltool/auxiliary_data/polar.shp"
  arctica = "/work/bd0854/b309059/esmvaltool/auxiliary_data/antarctica.shp"
  north_america = "/work/bd0854/b309059/esmvaltool/auxiliary_data/north_america.shp"
  central_america = "/work/bd0854/b309059/esmvaltool/auxiliary_data/central_america.shp"
  south_america = "/work/bd0854/b309059/esmvaltool/auxiliary_data/south_america.shp"
  north_atlantic = "/work/bd0854/b309059/esmvaltool/auxiliary_data/north_atlantic.shp"
  south_atlantic = "/work/bd0854/b309059/esmvaltool/auxiliary_data/south_atlantic.shp"
  north_pacific = "/work/bd0854/b309059/esmvaltool/auxiliary_data/north_pacific.shp"
  south_pacific = "/work/bd0854/b309059/esmvaltool/auxiliary_data/south_pacific.shp"
  southern_ocean = "/work/bd0854/b309059/esmvaltool/auxiliary_data/southern_ocean.shp"
  indian_ocean = "/work/bd0854/b309059/esmvaltool/auxiliary_data/indian_ocean.shp"

;--- Open workstation.
  wks = gsn_open_wks("png","shapefiles")

;---Create the map
  map = create_map(wks,"")

;---Attach the polylines
  pres             = True
  pres@gsLineColor = "blue"
  ;pres@gsLineColor = "magenta4"
  ;pres@gsLineOpacityF = 0.6
  pres@gsLineThicknessF = 8.
  ;pres@gsFillColor = "magenta4"
  ;pres@gsFillOpacityF = 0.2
  poly1 = gsn_add_shapefile_polylines(wks,map,europe,pres)
  poly2 = gsn_add_shapefile_polylines(wks,map,africa,pres)
  poly3 = gsn_add_shapefile_polylines(wks,map,asia,pres)
  poly4 = gsn_add_shapefile_polylines(wks,map,australasia,pres)
  poly5 = gsn_add_shapefile_polylines(wks,map,north_america,pres)
  ;poly6 = gsn_add_shapefile_polylines(wks,map,central_america,pres)
  poly7 = gsn_add_shapefile_polylines(wks,map,south_america,pres)
  poly8 = gsn_add_shapefile_polylines(wks,map,antarctica,pres)
  poly9 = gsn_add_shapefile_polylines(wks,map,arctica,pres)
  ;poly10 = gsn_add_shapefile_polylines(wks,map,polar,pres)
  ;poly10 = gsn_add_shapefile_polygons(wks,map,europe,pres)
  ;poly20 = gsn_add_shapefile_polygons(wks,map,africa,pres)
  ;poly30 = gsn_add_shapefile_polygons(wks,map,asia,pres)
  ;poly40 = gsn_add_shapefile_polygons(wks,map,australasia,pres)
  ;poly50 = gsn_add_shapefile_polygons(wks,map,north_america,pres)
  ;poly60 = gsn_add_shapefile_polygons(wks,map,south_america,pres)
  ;poly70 = gsn_add_shapefile_polygons(wks,map,antarctica,pres)

;  ;pres@gsLineColor = "cyan3"
;  ;pres@gsFillColor = "cyan3"
;  poly11 = gsn_add_shapefile_polylines(wks,map,north_atlantic,pres)
;  poly12 = gsn_add_shapefile_polylines(wks,map,south_atlantic,pres)
;  poly13 = gsn_add_shapefile_polylines(wks,map,north_pacific,pres)
;  poly14 = gsn_add_shapefile_polylines(wks,map,south_pacific,pres)
;  poly15 = gsn_add_shapefile_polylines(wks,map,indian_ocean,pres)
;  poly16 = gsn_add_shapefile_polylines(wks,map,southern_ocean,pres)
;  ;poly110 = gsn_add_shapefile_polygons(wks,map,north_atlantic,pres)
;  ;poly120 = gsn_add_shapefile_polygons(wks,map,south_atlantic,pres)
;  ;poly130 = gsn_add_shapefile_polygons(wks,map,north_pacific,pres)
;  ;poly140 = gsn_add_shapefile_polygons(wks,map,south_pacific,pres)
;  ;poly150 = gsn_add_shapefile_polygons(wks,map,indian_ocean,pres)
;  ;poly160 = gsn_add_shapefile_polygons(wks,map,southern_ocean,pres)
;
;  pres@gsLineDashPattern = 1
;  poly21 = gsn_add_polyline(wks, map, (/0., 180./), (/60., 60./), pres)
;  poly22 = gsn_add_polyline(wks, map, (/-180., 0./), (/60., 60./), pres)

;---Drawing the map will also draw the attached polylines.
  draw(map)
  frame(wks)
end
