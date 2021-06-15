; ****************************************************************************
; Fig_3_41_collect.ncl
;
; Description:
; Generates the final figure Fig. 3.41 with all input files of tas, pr, sic
; and ohc data
;
; Creator: Lisa Bock (lisa.bock@dlr.de)
; Creation Date: 10 March 2021
; ****************************************************************************

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"


;----------------------------------------------------------------------
; This function creates a cylindrical equidistant map of Australia
; so you you can add polylines, polygons, or point data to it later.
;----------------------------------------------------------------------
function create_map(wks, title, res2)
local a
begin
  res2@gsnDraw       = False
  res2@gsnFrame      = False

  res2@vpWidthF              =  0.25
  res2@vpHeightF             =  0.25

  res2@mpOutlineOn   = True

  res2@mpGeophysicalLineColor = "gray12"
  res2@mpGeophysicalLineThicknessF = 1. ;6.
  res2@mpPerimOn    = False
  res2@mpGridLineColor   = "transparent"
  res2@mpGridAndLimbOn   = True
  res2@mpGridLineThicknessF = 6.
  res2mpOutlineOn = True

  res2@mpProjection  = "Robinson"

;---Create map.
  map = gsn_csm_map(wks,res2)

  return(map)
end

begin

  start_year = 1850
  end_year = 2020
  ntime = end_year - start_year + 1
  time = new(ntime, integer)
  time = ispan(start_year, end_year, 1)

  ; *************************************
  ; General settings for the plot
  ; *************************************

  ; Define workstation
  outfile = "Fig_3_41"
  wks = gsn_open_wks("pdf", outfile)

  ; Set resources
  res = True
  res@gsnDraw   = False
  res@gsnFrame  = False

  res@vpXF      = 0.05
  res@vpYF      = 0.7
  res@vpHeightF = 0.4
  res@vpWidthF  = 0.7
  res@pmLegendDisplayMode = "Never"
  res@tmYRLabelFontHeightF = 0.016
  res@tmYLLabelFontHeightF = 0.016
  res@tiXAxisFontHeightF = 0.016
  res@tiYAxisFontHeightF = 0.016
  res@tmXBMajorOutwardLengthF = 0.006
  res@tmYLMajorOutwardLengthF = 0.006
  res@tmXBMinorOutwardLengthF = 0.003
  res@tmYLMinorOutwardLengthF = 0.003
  res@tmXBMajorLengthF = 0.006
  res@tmYLMajorLengthF = 0.006
  res@tmXBMinorLengthF = 0.003
  res@tmYLMinorLengthF = 0.003

  res@tiYAxisOn = True
  res@tiXAxisString = ""
  res@gsnStringFontHeightF = 0.016

  header = (/"land", "global", "ocean", "north_america", "asia", "arctica", "global", "south_america", \
             "australasia", "antarctica", "60N-90N", "europe", "antarctica", "africa", "arctic"/)
  title = (/"Land", "Global", "Ocean", "North America", "Asia", "Arctic", "Global", "Central and South America", \
             "Australasia", "Antarctic", "60N-90N", "Europe and North Africa", "Antarctic", "Africa", "Arctic"/)
  var = (/"tas", "tas", "ohc", "tas", "tas", "siconc", "pr", "tas", "tas", "siconc", "pr", "tas", "tas", \
          "tas", "tas"/)

  ; -----------------------------------------------

  input_files = "/work/bd0854/b309059/esmvaltool_output/Chapter_3/FGD/fig_3_41/"\
                + (/"tas_anom_damip_global_1.nc", \
                    "tas_anom_damip_global_0.nc", \
                    "ohc_damip.nc", \
                    "tas_anom_damip_america_europe_0.nc", \
                    "tas_anom_damip_africa_asia_1.nc", \
                    "tsline_collect_siconc_nh.nc", \
                    "precip_anom_0.nc", \
                    "tas_anom_damip_america_europe_1.nc", \
                    "tas_anom_damip_africa_asia_2.nc", \
                    "tsline_collect_siconc_sh.nc", \
                    "precip_anom_1.nc", \
                    "tas_anom_damip_america_europe_2.nc", \
                    "tas_anom_damip_antarctica_0.nc", \
                    "tas_anom_damip_africa_asia_0.nc", \
                    "tas_anom_damip_arctic_0.nc"/)

  nfiles = dimsizes(input_files)

  dim_data   = (/nfiles, 3, 3, ntime/)
  data  = new(dim_data, double)
  data!0 = "var"
  data!1 = "exp"
  data!2 = "stat"
  data!3 = "time"
  data&exp = (/"historical", "hist-nat", "ref"/)
  data&stat = (/"mean", "stat1", "stat2"/)
  data&time = time
  data = data@_FillValue

  plots  = new((/nfiles/), graphic)
  poly   = new((/nfiles/), graphic)
  annoid = new((/nfiles/), graphic)

  ; Loop over files in the list, read and append data
  do ii = 0, nfiles - 1

    print("Read in " + input_files(ii))

    f = addfile(input_files(ii), "r")

    if (var(ii) .eq. "tas") then
      data_temp = f->tas
    elseif (var(ii) .eq. "siconc")
      data_temp = f->siconc
    elseif (var(ii) .eq. "pr")
      data_temp = f->pr
    elseif (var(ii) .eq. "ohc")
      data_temp = f->ohc
    end if

    ; Match time coordinate
    if (var(ii) .eq. "ohc") then
      date = f->time
      ;data_temp = 10e-21 * data_temp
    else
      date = data_temp&time
    end if
    idx1 = ind(date(0).eq.data&time)
    idx2 = ind(date(dimsizes(date) - 1).eq.data&time)
    delete(date)

    data(ii, 0:1, :, idx1:idx2) = (/data_temp(0:1, :, :)/)

    if (var(ii) .eq. "tas") then
      data(ii, 2, :, idx1:idx2) = (/data_temp(4, :, :)/)
    else
      data(ii, 2, :, idx1:idx2) = (/data_temp(2, :, :)/)
    end if

    ;if (any(var(ii) .eq. (/"tas", "siconc"/))) then
    if (any(var(ii) .eq. (/"tas", "pr", "siconc"/))) then
    ;if (any(var(ii) .eq. (/"tas"/))) then
      ; --------------------------------------------------------------------
      ; Low pass filter
      ; --------------------------------------------------------------------

      print("smoothing applied: " + var(ii))

      ;;create filter
      ;nwt = 9      ; number of weights
      ;fca = 0.2
      ;nsigma = 1.
      ;wgt = filwgts_lanczos (nwt, 0, fca, 0, nsigma) 

      ;;apply filter
      ;data(ii, :, :, :) = wgt_runave_n_Wrap(data(ii, :, :, :), wgt, 1, 2)

      ;running mean
      tmp = data(ii, :, :, :)
      ; 3 yrs running mean
      tmp = runave_n_Wrap(data(ii, :, :, :), 3, 0, 2)
      data(ii, :, :, 1:(ntime-2)) = tmp(:,:,1:(ntime-2))
      ; 5 yrs running mean
      ;tmp = runave_n_Wrap(data(ii, :, :, :), 5, 0, 2)
      ;data(ii, :, :, 2:(ntime-3)) = tmp(:,:,2:(ntime-3))
      delete(tmp)

    end if

    delete(data_temp)

    ; *************************************
    ; Plotting
    ; *************************************

    cmap = read_colormap_file("diag_scripts/shared/plot/rgb/ipcc-ar6_damip_line.rgb")

    res@xyDashPatterns    := (/0, 0, 0/)
    res@xyLineThicknesses := (/3, 3, 3/)
    cmap(2,:) = namedcolor2rgba("black")
    res@xyLineColors := cmap(0:2, :)

    ymax = max(data(ii, :, : , :))
    ymin = min(data(ii, :, : , :))
    res@trYMaxF := max((/0.2, ymax + 0.05 * (ymax - ymin)/))
    res@trYMinF := min((/-0.2, ymin - 0.05 * (ymax - ymin)/))
    if (var(ii) .eq. "siconc") then
      res@trYMaxF := 6.
    elseif (var(ii) .eq. "tas")
      res@trYMaxF := 3.7
      res@trYMinF := -1.5
    end if

    res@tmXBMode = "Explicit"
    res@tmXBMode         = "Manual"

    if (var(ii) .eq. "pr") then
      xmin = 1950
      xmax = 2014
      res@trXMinF = xmin
      res@trXMaxF = xmax
      res@tmXBValues := ispan(xmin, xmax, 5)
      res@tmXBLabels := ispan(xmin, xmax, 5)
      res@tmXBMinorValues :=  ispan(xmin, xmax, 5)
      res@tmXBTickSpacingF = 20
    elseif (var(ii) .eq. "siconc")
      xmin = 1980
      xmax = 2015
      res@trXMinF = 1979
      res@trXMaxF = 2014 ;end_year
      res@tmXBMode         = "Explicit"
      res@tmXBValues := ispan(xmin, xmax, 10)
      res@tmXBLabels := ispan(xmin, xmax, 10)
      res@tmXBMinorValues :=  ispan(xmin, xmax, 2)
      ;res@tmXBTickSpacingF = 10
    else
      xmin = start_year
      xmax = end_year
      res@trXMinF = xmin
      res@trXMaxF = xmax
      res@tmXBValues := ispan(xmin, xmax, 10)
      res@tmXBLabels := ispan(xmin, xmax, 10)
      res@tmXBMinorValues :=  ispan(xmin, xmax, 5)
      res@tmXBTickSpacingF = 50
    end if

    res@tiMainString = title(ii)


    if (var(ii) .eq. "tas") then
      res@tiYAxisString = "Change in Temperature (~F34~0~F~ C)"
    elseif (var(ii) .eq. "siconc")
      res@tiYAxisString = "Change in Sea Ice (10~S~6~N~ km~S~2~N~)"
    elseif (var(ii) .eq. "pr")
      res@tiYAxisString = "Change in Precipitation (mm/day)"
    elseif (var(ii) .eq. "ohc")
      res@tiYAxisString = "Change in Ocean Heat Content (ZJ)"
    else
      res@tiYAxisString = ""
    end if

    plots(ii) = gsn_csm_xy(wks, time, data(ii, :, 0, :), res)  ; create plot

    res@pmLegendDisplayMode = "Never"

    
    ; ---------------------------------------------------------------------
    ; Add multi model statistics
    ; ---------------------------------------------------------------------
    res_stat = True
    res_stat@gsnXYFillOpacities = 0.1
    cmap = read_colormap_file("diag_scripts/shared/plot/rgb/ipcc-ar6_damip_shading.rgb")
    copy_VarMeta(res, res_stat)
    delete(res_stat@xyLineColors)
    res_stat@xyLineThicknesses := (/1, 1/)

    res_stat@gsnXYFillColors = cmap(1,:)
    res_stat@xyLineColor = cmap(1,:)
    shading_plot = gsn_csm_xy(wks, time, data(ii, 1, 1:2, :), \
                              res_stat)
    overlay(plots(ii), shading_plot)

    res_stat@gsnXYFillColors = cmap(0,:)
    res_stat@xyLineColor = cmap(0,:)
    shading_plot = gsn_csm_xy(wks, time, data(ii, 0, 1:2, :), \
                              res_stat)
    overlay(plots(ii), shading_plot)


    ;-- attach small map
    amres                     =  True
    amres@amJust              = "TopLeft"                ;-- labelbar justification
    amres@amParallelPosF      = -0.5                     ;-- move labelbar to the right of plot
    amres@amOrthogonalPosF    = -0.5                     ;-- move labelbar to the top of plot

    res2 = True

    map = create_map(wks, "", res2)

    ;---Attach the polylines
    pres             = True
    pres@gsLineColor = "blue"
    ;pres@gsLineOpacityF = 0.6
    pres@gsLineThicknessF = 2.
    pres@gsFillColor = "blue"
    pres@gsFillOpacityF = 0.5

    if (var(ii) .eq. "pr") then

      res3 = True
      copy_VarMeta(res2, res3)

      a  = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc","r")
      lsdata = a->LSMASK
      lat = a->lat
      lon = a->lon
      lsm = landsea_mask(lsdata, lat, lon)

      if(header(ii) .eq. "60N-90N") then
        lsm(0:149, :) = lsm@_FillValue
      end if

      res3@cnLevelSelectionMode = "ExplicitLevels" ; explicitly set the levels via cnLevels
      res3@cnLevels         = (/1.,2.,3.,4./)      ; set the levels
      res3@cnFillOn         = True                 ; turn on color fill
      res3@cnFillPalette    =(/"white", "blue", "blue", "blue", "blue"/)
      res3@cnFillOpacityF = 0.5
      res3@lbLabelBarOn = False

      delete(map)

      ;---Create map.
      map = gsn_csm_contour_map(wks, lsm, res3)

      ;poly(ii) = gsn_add_polygon(wks, map, (/-180, 180, 180,-180,-180/), (/60,60,90,90,60/), pres)

      ;-- atach the labelbar to the plot
      annoid(ii) = gsn_add_annotation(plots(ii), map, amres)

    elseif ((var(ii) .eq. "tas" .and. all(header(ii) .ne. (/"land", "global"/))) \
          .or. var(ii) .eq. "siconc")
    
      shape = "/work/bd0854/b309059/esmvaltool/auxiliary_data/" + header(ii) + ".shp"

      poly(ii) = gsn_add_shapefile_polygons(wks,map,shape,pres)

      ;-- atach the labelbar to the plot
      annoid(ii) = gsn_add_annotation(plots(ii), map, amres)

    end if

    delete(map)

  end do


  ; Plotting panels

  pres = True

  pres@gsnPanelMainString = "Anomaly"

  pres@gsnPanelRowSpec = True
  pres@gsnPanelCenter  = False

  gsn_panel(wks, plots, (/3,4,4,2,2/), pres)

end
