; #############################################################################
; DIAGNOSTIC SCRIPT for reproducing Friedlingstein 14 fig 1
; Author: Bettina Gier (Uni Bremen & DLR, Germany)
; CRESCENDO project
; #############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"
load "$diag_scripts/shared/latlon.ncl"
load "$diag_scripts/shared/statistics.ncl"
load "$diag_scripts/shared/scaling.ncl"

begin
  enter_msg(DIAG_SCRIPT, "")
  dim_VAR = ListCount(variable_info)
end

begin
  ; References: Le Quere 2016
  diag_script_info = True
  diag_script_info@reference = "LeQuere2018"
  if diag_script_info@reference.eq."LeQuere2018" then
    ref_osinks = (/(/1.0, 0.5/), (/1.3, 0.5/), (/1.7, 0.5/), (/1.9, 0.5/), (/2.1, 0.5/), (/2.4, 0.5/)/)
    ref_osinks!0 = "time"
    ; Decadal means, center coordinate
    ref_osinks&time = (/1965, 1975, 1985, 1995, 2005, 2012/)

    ref_lnsinks = (/(/1.4, 0.7/), (/2.4, 0.6/), (/2.0, 0.6/), (/2.5, 0.5/), (/2.9, 0.8/), (/3.0, 0.8/)/)
    ref_lnsinks!0 = "time"
    ; Decadal means, center coordinate
    ref_lnsinks&time = (/1965, 1975, 1985, 1995, 2005, 2012/)

    ref_eluc = (/(/1.4, 0.7/), (/1.1, 0.7/), (/1.2, 0.7/), (/1.3, 0.7/), (/1.2, 0.7/), (/1.3, 0.7/)/)

    ref_lsinks = ref_lnsinks
    ref_lsinks = ref_lnsinks - ref_eluc
    ; Do correct error propagation (ref_lnsink error was ref_lsinks err^2 + ref_eluc^2)
    ref_lsinks(:,1) = sqrt(abs(ref_lnsinks(:,1)^2 - ref_eluc(:,1)^2))
  else if diag_script_info@reference.eq."LeQuere2013" then
    ref_osinks = (/(/1.2, 0.5/), (/1.5, 0.5/), (/1.9, 0.5/), (/2.2, 0.4/), (/2.4, 0.5/), (/2.5, 0.5/)/)
    ref_osinks!0 = "time"
    ; Decadal means, center coordinate
    ref_osinks&time = (/1965, 1975, 1985, 1995, 2005, 2007/)

    ref_lnsinks = (/(/1.7, 0.7/), (/1.7, 0.8/), (/1.6, 0.8/), (/2.7, 0.8/), (/2.4, 0.8/), (/2.5, 0.5/)/)
    ref_lnsinks!0 = "time"
    ; Decadal means, center coordinate
    ref_lnsinks&time = (/1964, 1974, 1984, 1994, 2004, 2006/)

    ref_eluc = (/(/1.5, 0.5/), (/1.3, 0.5/), (/1.4, 0.5/), (/1.6, 0.5/), (/1.0, 0.5/), (/1.0, 0.5/)/)

    ref_lsinks = ref_lnsinks
    ref_lsinks = ref_lnsinks - ref_eluc
    ; Do correct error propagation (ref_lnsink error was ref_lsinks err^2 + ref_eluc^2)
    ref_lsinks(:,1) = sqrt(abs(ref_lnsinks(:,1)^2 - ref_eluc(:,1)^2))
    ref_lsinks&time = ref_lnsinks&time + 1
  end if
  end if
  ref_co2 =  397.12 ; value for 2014!
  ; 2005 = 379. ; TODO: Implement actual co2 global ref data


  ; Plot file type
  file_type = config_user_info@output_file_type
  if (ismissing(file_type)) then
    file_type = "ps"
  end if

  ; Create Plot dir
  plot_dir = config_user_info@plot_dir
  system("mkdir -p " + plot_dir)


  ; N-Cycle Models
  n_cycle_mods = (/"BNU-ESM", "CESM1-BGC", "NorESM1-ME", \
                   "UKESM1-0-LL", "NorESM2-LM", \
                   "EC-Earth3-Veg", "CESM2", "CESM2-WACCM", \
                   "SAM0-UNICON", "MIROC-ES2L", "MPI-ESM1-2-LR"/)
  ; Models with prescribed land use change
  pluc_mods = (/"bcc-csm1-1", "inmcm4", "MRI-ESM1"/)

  ; IPCC colors
  black = (/0., 0., 0./)
  lightblue = (/112./255., 160./255., 205./255./)
  orange = (/196./255., 121./255., 0./)
  orange_shade =  (/204./255., 174./255., 113./255./)
  lightgrey = (/178./255., 178./255., 178./255./)
  darkblue = (/0., 52./255., 102./255./)
  darkgreen = (/0., 79./255., 0./)
end

begin
  ; Color Picker Function
  undef("color_picker_co2")
  function color_picker_co2(models)
  ;
  ; Arguments:
  ;   models: List of models to pick colors for
  ; Description:
  ;   Returns array of colors based on model parameters
  ;   like Nitrogen Cycle or prescribed land use change
  ; Modification history
  ;   20190220_A_gier_bettina: written.
  local n_cycle_mods, pluc_mods, colors
  begin
  ; FOR COLORS
  ; TODO: Complete list + add CMIP6 models
  ; N-Cycle Models
  n_cycle_mods = (/"BNU-ESM", "CESM1-BGC", "NorESM1-ME", \
                   "UKESM1-0-LL", "NorESM2-LM", \
                   "EC-Earth3-Veg", "CESM2", "CESM2-WACCM", \
                   "SAM0-UNICON", "GFDL-ESM4"/)
  ; Models with prescribed land use change
  pluc_mods = (/"bcc-csm1-1", "inmcm4", "MRI-ESM1"/)

  ; IPCC colors
  black = (/0., 0., 0./)
  lightblue = (/112./255., 160./255., 205./255./)
  orange = (/196./255., 121./255., 0./)
  orange_shade =  (/204./255., 174./255., 113./255./)
  lightgrey = (/178./255., 178./255., 178./255./)
  darkblue = (/0., 52./255., 102./255./)
  darkgreen = (/0., 79./255., 0./)
  black_shade = (/128./255., 128./255., 128./255/)
  darkblue_shade = (/67./255., 147./255., 195./255./)
  lightblue_shade = (/92./255., 174./255., 178./255./)
  darkgreen_shade = (/223./255.,  237./255., 195./255./)

  ; Set up array to hold colors
  colors = new((/dimsizes(models), 3/), float)

  do idat = 0, dimsizes(models) - 1
    if models(idat).eq."BCC-CSM2-MR" then
      colors(idat, :) = darkgreen
    else if models(idat).eq."CanESM5" then
      colors(idat, :) = black
    else if models(idat).eq."NorESM2-LM" then
      colors(idat, :) = lightblue_shade
    else if models(idat).eq."CNRM-ESM2-1" then
      colors(idat, :) = black_shade
    else if models(idat).eq."GFDL-ESM4" then
      colors(idat, :) = lightgrey
    else if models(idat).eq."UKESM1-0-LL" then
      colors(idat, :) = lightblue
    else if models(idat).eq."CESM2" then
      colors(idat, :) = darkblue_shade
    else
      colors(idat, :) = lightblue_shade
    end if
    end if
    end if
    end if
    end if
    end if
    end if
  end do
  ;do idat = 0, dimsizes(models) - 1
  ;  if any(n_cycle_mods.eq.models(idat)) then
  ;    colors(idat, :) = (/112/255., 160/255., 205/255./)
  ;  else if any(pluc_mods.eq.models(idat)) then
  ;    colors(idat, :) = (/196/255., 121/255., 0/255./)
  ;  else
  ;    colors(idat, :) = (/0/255., 0/266., 0/255./)
  ;  end if
  ;  end if
  ;end do

  return(colors)
  end

end

begin
  ;leave_msg(DIAG_SCRIPT, "")
  ; Loop over variables
  do ivar = 0, dim_VAR - 1

    var = variable_info[ivar]@short_name
    info_items = select_metadata_by_name(input_file_info, var)
    datasetnames = metadata_att_as_array(info_items, "dataset")
    start_years = metadata_att_as_array(info_items, "start_year")
    start_year = min(start_years)
    end_years = metadata_att_as_array(info_items, "end_year")
    end_year = max(end_years)

    if var.eq."co2" then
      sp_opt = "average"
    else
      sp_opt = "sum"
    end if

    temp_array = new((/dimsizes(datasetnames), end_year - start_year + 1/), float)
    temp_array!0 = "datasets"
    temp_array&datasets = datasetnames
    temp_array!1 = "time"
    temp_array&time = ispan(start_year, end_year, 1); + 0.5

    ; Loop over models
    do idat = 0, dimsizes(datasetnames) - 1
      ;if var.eq."nbp" .and. datasetnames(idat).eq."MPI-ESM-LR" then
      ;  continue
      ;end if

      ; Read in data
      tmp_mod = read_data(info_items[idat])
      data_yearly = time_operations(tmp_mod, info_items[idat]@start_year, info_items[idat]@end_year, \
                                    "average", "yearly", True)
       if var.eq."co2" .and. datasetnames(idat).eq."CanESM5" then
        data_yearly = data_yearly * 1e-6 ;Gives units in ppmv, correcting here as not ESGF data yet!
       end if
      ; Compute Index
      idx1 = ind(start_years(idat).eq.temp_array&time)
      idx2 = ind(end_years(idat).eq.temp_array&time)
      if dimsizes(dimsizes(tmp_mod)).ge.2 then
        temp_array(idat, idx1:idx2) = tofloat(area_operations(data_yearly, -90., 90., \
                                           0., 360., sp_opt, True))
      else
        temp_array(idat, idx1:idx2) = tofloat(data_yearly)
;tofloat(time_operations(tmp_mod, info_items[idat]@start_year, \
;                               info_items[idat]@end_year, "average", "yearly", True))
      end if
      copy_VarAtts(tmp_mod, temp_array)

       ;if var.eq."fgco2" then
       ; print(datasetnames(idat))
       ; print(avg(data_yearly))
       ;end if

      delete(tmp_mod)
      delete(data_yearly)
    end do ; End loop over models

    if var.eq."co2" then
      co2_arr = temp_array
      co2_arr = convert_units(co2_arr, "ppm")
    else if var.eq."fgco2" then
      fgco2_arr = temp_array
      fgco2_arr = convert_units(fgco2_arr, "PgC y-1")
      fgco2_arr = runave_n_Wrap(fgco2_arr, 10, 0, 1)
    else if var.eq."nbp" then
      nbp_arr = temp_array
      nbp_arr = convert_units(nbp_arr, "PgC y-1")
      nbp_arr = runave_n_Wrap(nbp_arr, 10, 0, 1)

    else
      error_msg("f", DIAG_SCRIPT, "", "variable " + var + "not supported. " \
                + "Currently supported: co2, fgco2, nbp.")
    end if
    end if
    end if
    delete(temp_array)
    delete([/datasetnames, var, info_items, start_years, end_years/])

  end do ; End loop over variables

  ; TODO: MAKE THIS NICER! READ OBS DATA
  test = addfile("/work/bd0854/b309137/data_ext/Tier2/ESRL/OBS_ESRL_ground_GLOB_T0M_co2_198001-201812.nc", "r")
  co2_glob_full = test->co2
  delete(test)
  co2_glob = time_operations(co2_glob_full, 1980, 2014, "average", "yearly", True)

  ; --------------------- PLOTTING --------------------------------------------

  outfile = plot_dir + "carbon_global_" + start_year + "-" + end_year
  wks = gsn_open_wks(file_type, outfile)
  ; ----------- CO2 Plot ---------------
  res = True
  res@gsnDraw = False
  res@gsnFrame = False
  res@tiXAxisString = ""
  res@tiYAxisString = "Atmospheric CO~B~2~N~ (ppm)"
  res@trXMinF = start_year
  res@trXMaxF = end_year + 1 + (0.05 * (end_year - start_year))
  res@tmXBFormat = "f" ; Don't put unnecessary 0's
  res@tmYLFormat = "f"
  res@vpHeightF = 0.6
  res@vpWidthF = 0.8
  res@xyMonoDashPattern = True
  res@xyDashPattern = 0
  res@xyLineThicknessF = 2.
  res@xyLineColors = color_picker_co2(co2_arr&datasets)

  plot_co2 = gsn_csm_xy(wks, co2_arr&time, co2_arr, res)
  ; Add obs
  lnres = True
  lnres@gsLineColor = orange
  lnres@gsLineThicknessF = 3.
  plot_co2_obs = gsn_add_polyline(wks,plot_co2,co2_glob&year,co2_glob,lnres)


  ; Add Obs to CO2 Plot
  resM = True
  resM@gsMarkerColor = orange
  resM@gsMarkerIndex = 16
  resM@gsClipOn = False
  resM@gsMarkerSizeF = 0.015

  txres = True
  txres@txFontHeightF = 0.04
  txres@txFontColor   = orange
  txres@txJust = "CenterLeft"

  plot_co2@obs = gsn_add_text(wks, plot_co2, "   ~F37~Z", res@trXMaxF, ref_co2(0), txres)

  ; ------------ FGCO2 Plot
  res@tiYAxisString = "Ocean carbon flux (PgCy~S~-1~N~)"
  delete(res@xyLineColors)
  res@xyLineColors = color_picker_co2(fgco2_arr&datasets)
  res@trYMinF = min((/min(fgco2_arr), min(ref_osinks(:,0)-ref_osinks(:,1))/))
  res@trYMaxF = max((/max(fgco2_arr), max(ref_osinks(:,0)+ref_osinks(:,1))/))
  plot_fgco2 = gsn_csm_xy(wks, fgco2_arr&time, fgco2_arr, res)
  ; Add Observations:
  gsres = True
  gsres@gsFillColor = orange

  do iobs = 0, dimsizes(ref_osinks(:,0)) - 1
    if ref_osinks&time(iobs).gt.end_year then
      continue
    end if
    wdth = 0.01 * (res@trXMaxF - res@trXMinF)
    xbox = (/ref_osinks&time(iobs) - wdth, ref_osinks&time(iobs) + wdth, \
             ref_osinks&time(iobs) + wdth, ref_osinks&time(iobs) - wdth, \
             ref_osinks&time(iobs) - wdth/)
    ybox = (/ref_osinks(iobs, 0) - ref_osinks(iobs, 1), \
             ref_osinks(iobs, 0) - ref_osinks(iobs, 1), \
             ref_osinks(iobs, 0) + ref_osinks(iobs, 1), \
             ref_osinks(iobs, 0) + ref_osinks(iobs, 1), \
             ref_osinks(iobs, 0) - ref_osinks(iobs, 1)/)
    plot_fgco2@$unique_string("osink")$ = gsn_add_polygon(wks, plot_fgco2, \
                  xbox, ybox, gsres)
  end do

  ; ------ NBP Plot
  res@tiXAxisString = "Year"
  res@tiYAxisString = "Land carbon flux (PgCy~S~-1~N~)"
  delete(res@xyLineColors)
  res@xyLineColors = color_picker_co2(nbp_arr&datasets)
  res@trYMinF = min((/min(nbp_arr), min(ref_lnsinks(:,0)-ref_lnsinks(:,1))/))
  res@trYMaxF = max((/max(nbp_arr), max(ref_lnsinks(:,0)+ref_lnsinks(:,1))/))

  plot_nbp = gsn_csm_xy(wks, nbp_arr&time, nbp_arr, res)

  ; Add Observations:
  gsres = True
  gsres@gsFillColor = orange

  do iobs = 0, dimsizes(ref_lsinks(:,0)) - 1
    if ref_lsinks&time(iobs).gt.end_year then
      continue
    end if
    xbox = (/ref_lsinks&time(iobs) - wdth, ref_lsinks&time(iobs) + wdth, \
             ref_lsinks&time(iobs) + wdth, ref_lsinks&time(iobs) - wdth, \
             ref_lsinks&time(iobs) - wdth/)
    ybox = (/ref_lsinks(iobs, 0) - ref_lsinks(iobs, 1), \
             ref_lsinks(iobs, 0) - ref_lsinks(iobs, 1), \
             ref_lsinks(iobs, 0) + ref_lsinks(iobs, 1), \
             ref_lsinks(iobs, 0) + ref_lsinks(iobs, 1), \
             ref_lsinks(iobs, 0) - ref_lsinks(iobs, 1)/)
    plot_nbp@$unique_string("lsink")$ = gsn_add_polygon(wks, plot_nbp, \
                  xbox, ybox, gsres)
  end do
  plot_nbp@lsink_txt = gsn_add_text(wks, plot_nbp, "   F~B~L", res@trXMaxF, \
                       ref_lsinks(dimsizes(ref_lsinks(:,0))-1, 0), txres)

  gsres@gsFillColor = orange_shade

  do iobs = 0, dimsizes(ref_lsinks(:,0)) - 1
    if ref_lnsinks&time(iobs).gt.end_year then
      continue
    end if
    xbox = (/ref_lnsinks&time(iobs) - wdth, ref_lnsinks&time(iobs) + wdth, \
             ref_lnsinks&time(iobs) + wdth, ref_lnsinks&time(iobs) - wdth, \
             ref_lnsinks&time(iobs) - wdth/)
    ybox = (/ref_lnsinks(iobs, 0) - ref_lnsinks(iobs, 1), \
             ref_lnsinks(iobs, 0) - ref_lnsinks(iobs, 1), \
             ref_lnsinks(iobs, 0) + ref_lnsinks(iobs, 1), \
             ref_lnsinks(iobs, 0) + ref_lnsinks(iobs, 1), \
             ref_lnsinks(iobs, 0) - ref_lnsinks(iobs, 1)/)
    plot_nbp@$unique_string("lnsink")$ = gsn_add_polygon(wks, plot_nbp, \
                  xbox, ybox, gsres)
  end do
  txres@txFontColor   = gsres@gsFillColor
  plot_nbp@lnsink_txt = gsn_add_text(wks, plot_nbp, "   F~B~LN", res@trXMaxF, \
                       ref_lnsinks(dimsizes(ref_lnsinks(:,0))-1, 0), txres)


  ; --- PANELING
  pres = True
  pres@gsnMaximize = True
  pres@gsnFrame = False
  ;pres@gsnPanelRowSpec = True
  ;pres@gsnPanelCenter = False
  pres@gsnPanelTop = 0.95
  pres@gsnPanelBottom = 0.05
  pres@gsnPanelFigureStringsJust = "TopLeft"
  pres@gsnPanelFigureStrings            = (/"(a)","(b)", "(c)"/)           ; add strings to panel
  pres@gsnPanelFigureStringsPerimOn = "False"
  pres@gsnPanelFigureStringsFontHeightF = 0.012                   ; make figure strings slightly bigger

  gsn_panel(wks, (/plot_co2, plot_fgco2, plot_nbp/), (/3, 1/), pres)

  ;; Add Legend

  ; Initialize with OBS

  ; list of all models
  model_list = co2_arr&datasets
  do imod = 0, dimsizes(fgco2_arr&datasets) - 1
    if .not. any(fgco2_arr&datasets(imod) .eq. model_list) then
      model_list := array_append_record(model_list, fgco2_arr&datasets(imod), 0)
    end if
  end do
  do imod = 0, dimsizes(nbp_arr&datasets) - 1
    if .not. any(nbp_arr&datasets(imod) .eq. model_list) then
      model_list := array_append_record(model_list, nbp_arr&datasets(imod), 0)
    end if
  end do

  do imod = 0, dimsizes(model_list) - 1
    if any(model_list(imod) .eq. n_cycle_mods) then
      if (.not.isdefined("nmods")) then
        nmods = model_list(imod)
      else
        nmod_new = array_append_record(nmods, model_list(imod), 0)
        delete(nmods)
        nmods = nmod_new
        delete(nmod_new)
      end if
    else
      if (.not.isdefined("omods")) then
        omods = model_list(imod)
      else
        omods_new = array_append_record(omods, model_list(imod), 0)
        delete(omods)
        omods = omods_new
        delete(omods_new)
      end if
    end if
  end do

  my_labels = new(dimsizes(model_list)+3, string)
  my_labels(0) = "OBS"
  my_labels(1) = "N-cycle models"
  my_labels(2:1+dimsizes(nmods)) = nmods
  my_labels(2+dimsizes(nmods)) = "Other Models"
  my_labels(3+ dimsizes(nmods):) = omods

  my_colors = new((/dimsizes(my_labels), 3/), float)
  my_colors(0, :) = orange
  my_colors(1, :) = darkblue
  my_colors(2:1+dimsizes(nmods), :) = color_picker_co2(nmods)
  my_colors(2+dimsizes(nmods), :) = black
  my_colors(3+dimsizes(nmods):dimsizes(model_list)+2, :) = color_picker_co2(omods)

  my_opacities = new(dimsizes(my_labels), float)
  my_opacities = 1.
  my_opacities(1) = 0.
  my_opacities(2+dimsizes(nmods)) = 0.

  my_lt = new(dimsizes(my_labels), float)
  my_lt = 5.

  label_colors = new(dimsizes(my_labels), string)
  label_colors = "black"
  label_colors(1) = "blue"
  label_colors(2+dimsizes(nmods)) = "darkgrey"


  gres = True
  gres@YPosPercent = 93.5               ; expressed as %, 0->100, sets position of bottom border of legend
                                      ;  when gres@Position="Bottom" (Default = 95.)
  gres@XPosPercent = 75.               ; expressed as %, 0->100, sets position of left border of legend
                                      ;  (Default = 5.)
  gres@Position = "Top"            ; YPosPercent setting reflects the "Top" or "Bottom" of legend
                                      ;  (Default="Top")
  gres@ItemSpacePercent = 2.          ; expressed as %, 0->100, space between legend items (Default = 5.)
  gres@LineLabelWhiteSpacePercent = 2 ; expressed as %, 0->100, space between line and label (Default = 3.)


  lineres = True
  lineres@lgLineColors = my_colors ; line colors
  lineres@lgLineOpacity = my_opacities              ; line opacities
  lineres@lgLineThicknesses = my_lt         ; line thicknesses
  ;lineres@lgDashIndexes = (/0,1,14/)                   ; line types
  lineres@LineLengthPercent = 5                        ; expressed as %, 0->100, length of line

  textres = True
  textres@lgLabels = my_labels ; labels (req)
  textres@lgLabelFontHeight = 0.008                          ; label font heights
  textres@lgLabelColors = "black"                  ; label font colors
 ; textres@lgLineOpacity = my_opacities                                    ; 0->1, label opacities

  simple_legend_ndc(wks,gres,lineres,textres)


  frame(wks)


end
