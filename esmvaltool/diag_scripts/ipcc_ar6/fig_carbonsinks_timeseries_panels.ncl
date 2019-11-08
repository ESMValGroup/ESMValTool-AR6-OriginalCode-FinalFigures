; #############################################################################
; DIAGNOSTIC SCRIPT for reproducing IPCC ch. 9 fig. 9.6
; Author: Bettina Gier (Uni Bremen & DLR, Germany)
; CRESCENDO project
; #############################################################################
;
; Description
;    Calculates centred pattern correlations for annual mean climatologies
;    and plots them. Like IPCC ch. 9 fig 9.6
;
; Required diag_script_info attributes (diagnostics specific)
;
; Optional diag_script_info attributes (diagnostic specific)
;
; Required variable_info attributes (variable specific)
;     none
;
; Optional variable_info attributes (variable specific)
;     none
;
; Caveats
;
;
; Modification history
;    20190213-A_gier_be: written.
;
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
  ref_co2 = 379. ; TODO: Implement actual co2 global ref data


  ; Plot file type
  file_type = config_user_info@output_file_type
  if (ismissing(file_type)) then
    file_type = "ps"
  end if

  ; Create Plot dir
  plot_dir = config_user_info@plot_dir
  system("mkdir -p " + plot_dir)
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
  ;   like Nitrogen Cycle or presricbed land use change
  ; Modification history
  ;   20190220_A_gier_be: written.
  local n_cycle_mods, pluc_mods, colors
  begin
  ; FOR COLORS
  ; TODO: Complete list + add CMIP6 models
  ; N-Cycle Models
  n_cycle_mods = (/"BNU-ESM", "CESM1-BGC", "NorESM1-ME", \
                   "UKESM1-0-LL", "NorESM2-LM"/)
  ; Models with prescribed land use change
  pluc_mods = (/"bcc-csm1-1", "inmcm4", "MRI-ESM1"/)

  ; Set up array to hold colors
  colors = new((/dimsizes(models), 3/), float)

  do idat = 0, dimsizes(models) - 1
    if any(n_cycle_mods.eq.models(idat)) then
      colors(idat, :) = (/112/255., 160/255., 205/255./)
    else if any(pluc_mods.eq.models(idat)) then
      colors(idat, :) = (/196/255., 121/255., 0/255./)
    else
      colors(idat, :) = (/0/255., 0/266., 0/255./)
    end if
    end if
  end do

  return(colors)

  end
end

begin
  leave_msg(DIAG_SCRIPT, "")

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
      if var.eq."nbp" .and. datasetnames(idat).eq."MPI-ESM-LR" then
        continue
      end if

      ; Read in data
      tmp_mod = read_data(info_items[idat])
      data_yearly = time_operations(tmp_mod, info_items[idat]@start_year, info_items[idat]@end_year, \
                                    "average", "yearly", True)
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
  res@xyLineColors = color_picker_co2(co2_arr&datasets)

  plot_co2 = gsn_csm_xy(wks, co2_arr&time, co2_arr, res)

  ; Add Obs to CO2 Plot
  resM = True
  resM@gsMarkerColor = "red"
  resM@gsMarkerIndex = 16
  resM@gsClipOn = False
  resM@gsMarkerSizeF = 0.015

  txres = True
  txres@txFontHeightF = 0.04
  txres@txFontColor   = "red"
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
  gsres@gsFillColor = "red"

  do iobs = 0, dimsizes(ref_osinks(:,0)) - 1
    if ref_osinks&time(iobs).gt.end_year then
      continue
    end if
    print("in loop: " + iobs)
    wdth = 0.01 * (res@trXMaxF - res@trXMinF)
    xbox = (/ref_osinks&time(iobs) - wdth, ref_osinks&time(iobs) + wdth, \
             ref_osinks&time(iobs) + wdth, ref_osinks&time(iobs) - wdth, \
             ref_osinks&time(iobs) - wdth/)
    ybox = (/ref_osinks(iobs, 0) - ref_osinks(iobs, 1), \
             ref_osinks(iobs, 0) - ref_osinks(iobs, 1), \
             ref_osinks(iobs, 0) + ref_osinks(iobs, 1), \
             ref_osinks(iobs, 0) + ref_osinks(iobs, 1), \
             ref_osinks(iobs, 0) - ref_osinks(iobs, 1)/)
    print("x values " + xbox(0) + " to " + xbox(1))
    print("y values " + ybox(0) + " to " + ybox(2))
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
  gsres@gsFillColor = "red"

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

  gsres@gsFillColor = "indianred1"

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

  frame(wks)


end
