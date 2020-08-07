; draw_samindex_millenium.ncl
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs: 
;
; History
;    20191120 kosaka_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/statistics.ncl"
load "$diag_scripts/shared/plot/style.ncl"
load "$diag_scripts/shared/plot/contour_maps.ncl"
load "$diag_scripts/ar6ch3_sam_millennium/set_color.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")

  ; Configuration

  syear = 1000
  eyear = 2020
  syear_data = 950
  eyear_data = 2050
  
  if (isatt(diag_script_info, "syear_ref")) then
    syear_ref = diag_script_info@syear_ref
  end if
  if (isatt(diag_script_info, "eyear_ref")) then
    eyear_ref = diag_script_info@eyear_ref
  end if

  runmean = 7
  if (isatt(diag_script_info, "nyears_running_mean")) then
    runmean = diag_script_info@nyears_running_mean
  end if

  cutoff_period = 140.
  if (isatt(diag_script_info, "cutoff_period")) then
    cutoff_period = diag_script_info@cutoff_period
  end if

  normalize = True
  
  ; ========================================================================

  ; Get var list
  var = "psl"
  varList = metadata_att_as_array(input_file_info, "short_name")
  varList := get_unique_values(varList)
  
  if (all(varList.ne.var)) then
    error_msg("f", DIAG_SCRIPT, "", "variable "+var+" is missing.")
  end if
  iVar = ind(varList.eq.var)

; ========================= READ CMIP DATA =============================

  ; Get file list
  info = select_metadata_by_name(input_file_info, var)

  ; Get list of unique integrations
  models = metadata_att_as_array(info, "dataset")
  project = metadata_att_as_array(info, "project")
  ensemble = metadata_att_as_array(info, "ensemble")
  modelens = project + "_" + models + "_"+ ensemble
  integrations = get_unique_values(modelens)
  nintegrations = dimsizes(integrations)
  
  ; Define array
  sam = new((/nintegrations, eyear_data-syear_data+1/), "float")
  sam!1 = "time"
  sam&time = ispan(syear_data, eyear_data, 1)
  north = sam
  south = sam
  cmip_phase = new((/nintegrations/), "string")
  model_name = new((/nintegrations/), "string")
  
  do ii = 0, ListCount(info) - 1
    ; Read data
    psl = read_data(info[ii])
    slp = tofloat(psl)
    copy_VarMeta(psl, slp)
    delete(psl)
    
    time = cd_calendar(slp&time, -1)
    syr = time(0)/100
    eyr = time(dimsizes(time)-1)/100
    delete(time)
    time = ispan(syr, eyr, 1)
    slp&time = time
    delete(time)

    yr0 = max((/syr, syear_data/))
    yr1 = min((/eyr, eyear_data/))

    int = ind(info[ii]@project+"_"+info[ii]@dataset+"_"+info[ii]@ensemble.eq.integrations)
    
    north(int, {yr0:yr1}) = (/slp({yr0:yr1}, {-40.})/)
    south(int, {yr0:yr1}) = (/slp({yr0:yr1}, {-65.})/)
    delete(slp)
    cmip_phase(int) = info[ii]@project
    model_name(int) = info[ii]@dataset
  end do
  
  north = (north - conform(north, dim_avg_n(north(:, {syear_ref:eyear_ref}), 1), 0)) \
;  north = (north - avg(north(:, {syear_ref:eyear_ref}))) \
             / conform(north, dim_stddev_n(north(:, {syear_ref:eyear_ref}), 1), 0)
  south = (south - conform(south, dim_avg_n(south(:, {syear_ref:eyear_ref}), 1), 0)) \
;  south = (south - avg(south(:, {syear_ref:eyear_ref}))) \
             / conform(south, dim_stddev_n(south(:, {syear_ref:eyear_ref}), 1), 0) 
  sam = north - south
  delete(north)
  delete(south)

  if (normalize) then
    sam = sam - conform(sam, dim_avg_n(sam(:, {syear_ref:eyear_ref}), 1), 0)
    sam = sam / conform(sam, dim_stddev_n(sam(:, {syear_ref:eyear_ref}), 1), 0)
  end if

; ======================== READ RECONSTRUCTIONS ========================

  script = get_script_name()
;  n = str_index_of_substr(script, "/", -1)
  dirs = str_split(script, "/")
  dirs = dirs+"/"
  dir = "/"+str_concat(dirs(0:dimsizes(dirs)-2))
  delete(dirs)
  delete(script)
  file_abram = dir+"abram2014sam/abram2014sam.txt"
  file_villalba = dir+"villalba2012sam/villalba2012sam.txt"
  file_datwyler_ann = dir+"datwyler2017/Reconstructions_Annual_LC.txt"
  file_datwyler_djf = dir+"datwyler2017/Reconstructions_DJF_LC.txt"
  delete(dir)

  do n = 0, 3
    if (n.eq.0) then
      path = file_abram
      ncol = 1
      delimiter = str_get_tab()
    elseif (n.eq.1) then
      path = file_villalba
      ncol = 3
      delimiter = str_get_tab()
    elseif (n.eq.2) then
      path = file_datwyler_ann
      ncol = 2
      delimiter = ";"
    elseif (n.eq.3) then
      path = file_datwyler_djf
      ncol = 2
      delimiter = ";"
    end if
    lines = asciiread(path, -1, "string")
    nlines = dimsizes(lines)
    do ii = 0, nlines-1
      if (str_is_blank(lines(ii))) then
        continue
      end if
      if (isStrSubset(lines(ii), "#")) then
;        print(str_index_of_substr(str_squeeze(lines(ii)), "#", 1))
;        if (str_index_of_substr(str_squeeze(lines(ii)), "#", 1).le.1) then
        continue
;        end if
      elseif (isStrSubset(lines(ii), "age_AD").or.isStrSubset(lines(ii), "Year")) then
        continue
      end if
;      print(""+data(ii))
      vals = str_split_csv(lines(ii), delimiter, 3)
      year = toint(vals(0, 0))
      val  = tofloat(vals(0, ncol))
;      print(year+" "+val)
      delete(vals)

      if (.not.isvar("data")) then
        data = new((/1, 2/), "float")
        data(0, 0) = year
        data(0, 1) = val
      else
        tmp = data
        delete(data)
        dims = dimsizes(tmp)
        data = new((/dims(0)+1, dims(1)/), typeof(tmp))
        data(0:dims(0)-1, :) = tmp
        data(dims(0), 0) = year
        data(dims(0), 1) = val
        delete(tmp)
        delete(dims)
      end if
    end do
    if (isMonotonic(data(:, 0)).eq.-1) then
      data = data(::-1, :)
    end if

    if (normalize) then
      ts = ind(data(:, 0).eq.syear_ref)
      te = ind(data(:, 0).eq.eyear_ref)
      data(:, 1) = data(:, 1) - avg(data(ts:te, 1))
      data(:, 1) = data(:, 1) / stddev(data(ts:te, 1))
    end if
    
    dims = dimsizes(data)
    syr = toint(data(0, 0))
    eyr = toint(data(dims(0)-1, 0))
    delete(dims)
    if (path.eq.file_abram) then
      sam_abram = sam(0, :)
      sam_abram = sam_abram@_FillValue
      sam_abram({syr:eyr}) = (/data(:, 1)/)
      sam_abram = sam_abram - avg(sam_abram({syear_ref:eyear_ref}))
      sam_abram@source = "Abram et al. (2014)"
      sam_abram@color = (/36., 147., 126./)/255.
    elseif (path.eq.file_villalba) then
      sam_villalba = sam(0, :)
      sam_villalba = sam_villalba@_FillValue
      sam_villalba({syr:eyr}) = (/data(:, 1)/)
      sam_villalba = sam_villalba - avg(sam_villalba({syear_ref:eyear_ref}))
      sam_villalba@source = "Villalba et al. (2012)"
      sam_villalba@color = (/237., 128., 55./)/255.
    elseif (path.eq.file_datwyler_ann) then
      sam_datwyler_ann = sam(0, :)
      sam_datwyler_ann = sam_datwyler_ann@_FillValue
      sam_datwyler_ann({syr:eyr}) = (/data(:, 1)/)
      sam_datwyler_ann = sam_datwyler_ann - avg(sam_datwyler_ann({syear_ref:eyear_ref}))
      sam_datwyler_ann@source = "Datwyler et al. (2018)"
      sam_datwyler_ann@color = (/127., 68., 170./)/255.
    elseif (path.eq.file_datwyler_djf) then
      sam_datwyler_djf = sam(0, :)
      sam_datwyler_djf = sam_datwyler_djf@_FillValue
      sam_datwyler_djf({syr:eyr}) = (/data(:, 1)/)
      sam_datwyler_djf = sam_datwyler_djf - avg(sam_datwyler_djf({syear_ref:eyear_ref}))
      sam_datwyler_djf@source = sam_datwyler_ann@source
      sam_datwyler_djf@color = sam_datwyler_ann@color
    end if
    delete(data)
    delete(lines)
  end do

; ============================== FILTERING =============================
      
  ; Running mean
  wgt = new(runmean, "float")
  wgt = 1.
  wgt = wgt / sum(wgt)
  sam_runmean = wgt_runave_n_Wrap(sam, wgt, 0, 1)
  sam_abram_runmean = wgt_runave_n_Wrap(sam_abram, wgt, 0, 0)
  sam_villalba_runmean = wgt_runave_n_Wrap(sam_villalba, wgt, 0, 0)
  sam_datwyler_ann_runmean = wgt_runave_n_Wrap(sam_datwyler_ann, wgt, 0, 0)
  sam_datwyler_djf_runmean = wgt_runave_n_Wrap(sam_datwyler_djf, wgt, 0, 0)
  delete(wgt)
  
  ; Lowpass filter
  runmean = toint(cutoff_period / 2)
  even = False
  if (mod(runmean, 2).eq.0) then
    even = True
    runmean = runmean + 1
  end if
  wgt = new(runmean, "float")
  wgt = 1.
  if (even) then
    wgt(0) = 0.5
    wgt(runmean-1) = 0.5
  end if
  wgt = wgt / sum(wgt)
  sam_lowpass = wgt_runave_n_Wrap(sam, wgt, 0, 1)
  sam_abram_lowpass = wgt_runave_n_Wrap(sam_abram, wgt, 0, 0)
  sam_villalba_lowpass = wgt_runave_n_Wrap(sam_villalba, wgt, 0, 0)
  sam_datwyler_ann_lowpass = wgt_runave_n_Wrap(sam_datwyler_ann, wgt, 0, 0)
  sam_datwyler_djf_lowpass = wgt_runave_n_Wrap(sam_datwyler_djf, wgt, 0, 0)
  delete(wgt)

  if (num(cmip_phase.eq."CMIP5").gt.1) then
    sam_lowpass_cmip5mean = sam_lowpass(0, :)
    sam_lowpass_cmip5mean = \
      where(dim_num_n(ismissing(sam_lowpass(ind(cmip_phase.eq."CMIP5"), :)), 0).gt.0, \
            sam_lowpass@_FillValue, dim_avg_n(sam_lowpass(ind(cmip_phase.eq."CMIP5"), :), 0))
  end if
  if (num(cmip_phase.eq."CMIP6").gt.1) then
    sam_lowpass_cmip6mean = sam_lowpass(0, :)
    sam_lowpass_cmip6mean = \
      where(dim_num_n(ismissing(sam_lowpass(ind(cmip_phase.eq."CMIP6"), :)), 0).gt.0, \
            sam_lowpass@_FillValue, dim_avg_n(sam_lowpass(ind(cmip_phase.eq."CMIP6"), :), 0))
  end if

; =============================== Draw ===============================

  wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                "sam_millenium")
  
  rests := True
;  rests@xyMonoLineColor   = False
;  rests@gsnYRefLine       = 0.
  rests@tmYLLabelFontHeightF = 0.01
  rests@tmXBLabelFontHeightF = 0.01
  rests@gsnDraw           = False
  rests@gsnFrame          = False
;  rests@gsnMaximize       = True
  rests@trXMinF           = tofloat(syear)
  rests@trXMaxF           = tofloat(eyear)
  rests@tmXTOn            = False
  rests@tmYLMode          = "Manual"
  rests@tmYLTickSpacingF  = 2.
  rests@tmYLTickStartF    = -4.
  rests@tmYLTickEndF      = 4.
  rests@tmYRMode          = rests@tmYLMode
  rests@tmYRTickSpacingF  = rests@tmYLTickSpacingF
  rests@tmYRTickStartF    = rests@tmYLTickStartF
  rests@tmYRTickEndF      = rests@tmYLTickEndF
  rests@vpWidthF          = 0.6
  rests@vpHeightF         = 0.7
  rests@tiMainFontHeightF = 0.015
  rests@tiMainOffsetYF = -0.008

  rests@xyDashPattern     = 0
  rests@xyLineThicknessF = 0.5

  restsL = rests
  restsL@trYMaxF           = 3.5 ;2.
  restsL@trYMinF           = -13.
  restsL@xyLineColor       = sam_abram@color
  restsL@tiMainString = "SAM indices in the last millenium"
  restsR = rests
  restsR@trYMaxF           = 7.5 ;6.
  restsR@trYMinF           = -9.
  restsR@xyLineColor       = sam_villalba@color
  restsR2 = rests
  restsR2@trYMaxF           = 13.5 ;12.
  restsR2@trYMinF           = -3.
  colors = new((/nintegrations, 3/), "float")
  color_cmip5 = (/0.3, 0.3, 0.3/)
  do ii = 0, nintegrations-1
    if (cmip_phase(ii).eq."CMIP5") then
      colors(ii, :) = color_cmip5
    elseif (cmip_phase(ii).eq."CMIP6") then
      colors(ii, :) = get_color(model_name(ii))
    end if
  end do
  ;restsR2@xyLineColor       = "grey"
  restsR2@xyLineColors      = colors
  restsR2@xyLineOpacityF    = 0.4
  restsR2@tmYRTickEndF      = 4.

  plot = gsn_csm_xy3(wks, sam_abram_runmean&time, sam_abram_runmean, sam_villalba_runmean, sam_runmean, restsL, restsR, restsR2)

  dummy = new(nintegrations+18, "graphic")
  polyres = True
  polyres@gsLineThicknessF = 0.5
  polyres@gsLineColor       = "grey30"
  polyres@gsLineDashPattern = 1

  txres := True
  txres@txFontHeightF = 0.015
  txres@txJust = "CenterLeft"

  dummy(0) = gsn_add_polyline(wks, plot, (/syear, eyear/), (/0., 0./), polyres)
  dummy(1) = gsn_add_polyline(wks, plot@xy2, (/syear, eyear/), (/0., 0./), polyres)
  dummy(2) = gsn_add_polyline(wks, plot@xy3, (/syear, eyear/), (/0., 0./), polyres)

  polyres@gsLineDashPattern = 0
  polyres@gsLineThicknessF  = 0.5
  polyres@gsLineColor      := sam_datwyler_ann@color
  dummy(3) = gsn_add_polyline(wks, plot, sam_datwyler_ann_runmean&time, sam_datwyler_ann_runmean, polyres)
  
  polyres@gsLineThicknessF = 2.5
  polyres@gsLineColor     := sam_abram@color
  dummy(4) = gsn_add_polyline(wks, plot, sam_abram_lowpass&time, sam_abram_lowpass, polyres)
  txres@txFontColor := polyres@gsLineColor
  dummy(5) = gsn_add_text(wks, plot, sam_abram@source, 1420., 0.5, txres)

  polyres@gsLineColor     := sam_datwyler_ann@color
  dummy(6) = gsn_add_polyline(wks, plot, sam_datwyler_ann_lowpass&time, sam_datwyler_ann_lowpass, polyres)
;  txres@txFontColor := polyres@gsLineColor
;  dummy(7) = gsn_add_text(wks, plot, sam_datwyler_ann@source, 1070., -1.3, txres)

  polyres@gsLineThicknessF = 0.5
  polyres@gsLineColor     := sam_datwyler_djf@color
  dummy(8) = gsn_add_polyline(wks, plot@xy2, sam_datwyler_djf_runmean&time, sam_datwyler_djf_runmean, polyres)
  
  polyres@gsLineThicknessF = 2.5
  polyres@gsLineColor     := sam_villalba@color
  dummy(9) = gsn_add_polyline(wks, plot@xy2, sam_villalba_lowpass&time, sam_villalba_lowpass, polyres)
  txres@txFontColor := polyres@gsLineColor
  dummy(10) = gsn_add_text(wks, plot@xy2, sam_villalba@source, 1100., -3.1, txres)

  polyres@gsLineColor     := sam_datwyler_djf@color
  dummy(11) = gsn_add_polyline(wks, plot@xy2, sam_datwyler_djf_lowpass&time, sam_datwyler_djf_lowpass, polyres)
  txres@txFontColor := polyres@gsLineColor
  dummy(12) = gsn_add_text(wks, plot@xy2, sam_datwyler_djf@source, 1550., 0.4, txres)

  
  polyres@gsLineThicknessF = 0.75
  do n = 0, nintegrations-1
    if (cmip_phase(n).eq."CMIP5") then
      polyres@gsLineColor := color_cmip5
    elseif (cmip_phase(n).eq."CMIP6") then
      polyres@gsLineColor := get_color(model_name(n))
    end if
    dummy(12+n) = gsn_add_polyline(wks, plot@xy3, sam_lowpass&time, sam_lowpass(n, :), polyres)
  end do
  if (isvar("sam_lowpass_cmip5mean")) then
    polyres@gsLineThicknessF = 4.
    polyres@gsLineColor     := "white"
    dummy(12+nintegrations) = gsn_add_polyline(wks, plot@xy3, sam_lowpass_cmip5mean&time, \
                                               sam_lowpass_cmip5mean, polyres)
    polyres@gsLineThicknessF = 2.5
    polyres@gsLineColor     := color_cmip5
    dummy(13+nintegrations) = gsn_add_polyline(wks, plot@xy3, sam_lowpass_cmip5mean&time, \
                                               sam_lowpass_cmip5mean, polyres)
    txres@txFontColor := polyres@gsLineColor
    dummy(14+nintegrations) = gsn_add_text(wks, plot@xy3, "CMIP5", 1900., -2.0, txres)
  end if
  if (isvar("sam_lowpass_cmip6mean")) then
    polyres@gsLineThicknessF = 4.
    polyres@gsLineColor     := "white"
    dummy(15+nintegrations) = gsn_add_polyline(wks, plot@xy3, sam_lowpass_cmip6mean&time, \
                                               sam_lowpass_cmip6mean, polyres)
    polyres@gsLineThicknessF = 2.5
    polyres@gsLineColor     := get_color("CMIP6")
    dummy(16+nintegrations) = gsn_add_polyline(wks, plot@xy3, sam_lowpass_cmip6mean&time, \
                                               sam_lowpass_cmip6mean, polyres)
    txres@txFontColor := polyres@gsLineColor
    dummy(17+nintegrations) = gsn_add_text(wks, plot@xy3, "CMIP6", 1900., -2.6, txres)
  end if
  
  draw(plot)
  frame(wks)
  
  leave_msg(DIAG_SCRIPT, "")

end
