; draw_global_monsoon
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs:
;
; History
;    20200514 kosaka_yu: enabled specifying plots to draw; cleaned up the codes
;    20191030 kosaka_yu: written.
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
load "$diag_scripts/shared/plot/contourplot.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")

  moving_ave = 0
  if (isatt(diag_script_info, "moving_ave")) then
    moving_ave = diag_script_info@moving_ave
  end if
  scale_precip_ann = True
  if (isatt(diag_script_info, "scale_precip_ann")) then
    scale_precip_ann = diag_script_info@scale_precip_ann
  end if

  input_dir = diag_script_info@input_files + "/"

  colors_satellite = (/"purple", "orange", "brown"/)
  colors_insitu = (/"green", "blue", "darkorchid4", "deepping4"/)
  colors_reanal = (/"blue3", "darkgreen", "darkslategray", "darkseagreen"/)

  domainFiles = systemfunc("ls "+input_dir+"domain_monsoon_*.nc")
  if (scale_precip_ann) then
    prFiles = systemfunc("ls "+input_dir+"monsoon_precip_index_*.nc")
  else
    prFiles = systemfunc("ls "+input_dir+"precip_monsoon_*.nc")
  end if
  uFiles = systemfunc("ls "+input_dir+"u_monsoon_*.nc")
  vFiles = systemfunc("ls "+input_dir+"v_monsoon_*.nc")

; if domainFiles and prFiles exist...
  refnamePr = "GPCP-SG"
  if (isatt(diag_script_info, "dataset_obs_pr")) then
    refnamePr = diag_script_info@dataset_obs_pr
  end if
  refnameUV = "ERA-Interim"
  if (isatt(diag_script_info, "dataset_obs_uv")) then
    refnameUV = diag_script_info@dataset_obs_uv
  end if

  domainFileRef  = input_dir+"domain_monsoon_"+refnamePr+".nc"
  if (scale_precip_ann) then
    prFileRef    = input_dir+"monsoon_precip_index_"+refnamePr+".nc"
  else
    prFileRef    = input_dir+"precip_monsoon_"+refnamePr+".nc"
  end if
  uFileRef       = input_dir+"u_monsoon_"+refnameUV+".nc"
  vFileRef       = input_dir+"v_monsoon_"+refnameUV+".nc"

  domainFileMME  = input_dir+"domain_monsoon_MultiModelMean.nc"
  if (scale_precip_ann) then
    prFileMME    = input_dir+"monsoon_precip_index_MultiModelMean.nc"
  else
    prFileMME    = input_dir+"precip_monsoon_MultiModelMean.nc"
  end if
  uFileMME       = input_dir+"u_monsoon_MultiModelMean.nc"
  vFileMME       = input_dir+"v_monsoon_MultiModelMean.nc"

  drawMap = True
  if (isatt(diag_script_info, "plot_map")) then
    drawMap = diag_script_info@plot_map
  end if

  f = addfile(domainFileRef, "r")
  domainRef = f->monsoon_domain
  domainRef@long_name = ""
  f = addfile(domainFileMME, "r")
  domainMME = f->monsoon_domain
  domainMME@long_name = ""

  f = addfile(prFileRef, "r")
  if (scale_precip_ann) then
    prRef = f->MPI
  else
    prRef = f->pr
  end if
  prRef@long_name = ""
  prRef@units = ""
  refnamePr = prRef@dataset
  f = addfile(prFileMME, "r")
  if (scale_precip_ann) then
    prMME = f->MPI
  else
    prMME = f->pr
  end if
  prMME@long_name = ""
  prMME@units = ""
  MMEnamePr = prMME@dataset

  if (drawMap .and. \
      all(fileexists((/uFileRef, uFileMME, vFileRef, vFileMME/)))) then
    drawUV = True
  else
    drawUV = False
  end if

  if (drawUV) then
    f = addfile(uFileRef, "r")
    uRef = f->u
    uunits = uRef@units
    uRef@long_name = ""
    uRef@units = ""
    refnameUV = uRef@dataset
    f = addfile(uFileMME, "r")
    uMME = f->u
    uMME@long_name = ""
    uMME@units = ""

    f = addfile(vFileRef, "r")
    vRef = f->v
    vRef@long_name = ""
    vRef@units = ""
    f = addfile(vFileMME, "r")
    vMME = f->v
    vMME@long_name = ""
    vMME@units = ""
  end if

  if (isdefined("wks")) then
    delete(wks)
  end if

  if (drawUV) then
    if (moving_ave.gt.0) then
      wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                    "global_monsoon_" + refnamePr + "+" \
                    + refnameUV + "_movingave" + moving_ave + "yrs")
    else
      wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                    "global_monsoon_" + refnamePr + "+" + refnameUV)
    end if
  else
    if (moving_ave.gt.0) then
      wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                    "global_monsoon_" + refnamePr + \
                    "_movingave" + moving_ave + "yrs")
    else
      wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                    "global_monsoon_" + refnamePr)
    end if
  end if

  if (drawMap) then
    resPr = True
    resPr@res_gsnDraw = False
    resPr@res_gsnFrame = False
    resPr@res_gsnMaximize = False
    resPr@res_gsnAddCyclic = True
    resPr@res_cnFillOn = True
    resPr@res_cnFillMode = "AreaFill"
    resPr@res_cnLinesOn = False
    resPr@res_cnLineLabelsOn = False
    resPr@res_cnInfoLabelOn = False
    resPr@res_mpMinLonF = -180.
    resPr@res_mpMaxLonF = 180.
    resPr@res_mpCenterLonF = 0.
    resPr@res_mpGeophysicalLineColor = "grey30"
;    resPr@res_mpGridAndLimbOn = True
;    resPr@res_mpGridLatSpacingF = 90.
;    resPr@res_mpGridLonSpacingF = 720.
    resPr@res_cnLevelSelectionMode = "ExplicitLevels"
    resPr@res_lbLabelBarOn = False
    resPr@res_tiMainFontHeightF = 0.018
    resPr@res_tiMainOffsetYF = -0.008
    resPr@res_tmXTOn = False
    resPr@res_tmYROn = False
    resPr@res_tmXBLabelFontHeightF = 0.008
    resPr@res_tmYLLabelFontHeightF = 0.008
    resPr@res_gsnLeftString = ""
    resPr@res_gsnRightString = ""
    resPr@res_mpPerimLineThicknessF = 0.
    resPr@res_mpLimbLineThicknessF = 0.
    resPr@res_mpPerimOn = False
    resPr@res_tmBorderThicknessF = 1.
    resPr@res_tmXBMajorThicknessF =  resPr@res_tmBorderThicknessF
    resPr@res_tmYLMajorThicknessF =  resPr@res_tmBorderThicknessF
    resPr@res_tmXBMinorThicknessF =  resPr@res_tmBorderThicknessF/2.
    resPr@res_tmYLMinorThicknessF =  resPr@res_tmBorderThicknessF/2.

    if (scale_precip_ann) then
      resPr@res_cnLevels = (/-2.5, -2., -1.5, -1., -0.5, \
                            0.5, 1., 1.5, 2., 2.5/)
    else
      resPr@res_cnLevels = (/-1000, -800, -600, -400, -200, \
                            200, 400, 600, 800, 1000/)
    end if
    resPr@res_cnFillPalette = "AR6_Precip_11"

    if (drawUV) then
      resUV := True
      resUV@gsnDraw = False
      resUV@gsnFrame = False
      resUV@gsnMaximize = False
      resUV@gsnAddCyclic = True
      resUV@vcGlyphStyle = "CurlyVector"  ; "FillArrow"
      resUV@vcFillArrowsOn = True
      resUV@vcFillArrowHeadMinFracXF  = 0.2
      resUV@vcFillArrowWidthF = 0.08
      resUV@vcFillArrowEdgeThicknessF = 1.
      resUV@vcFillArrowEdgeColor = "white"
      resUV@vcFillArrowFillColor = "black"
      resUV@vcRefAnnoSide = "Bottom"
      resUV@vcRefAnnoOrthogonalPosF = 0.2
      resUV@vcRefAnnoParallelPosF = 1.
      resUV@vcRefAnnoPerimOn = False
      resUV@vcRefLengthF = 0.035
      resUV@vcMinDistanceF = 0.02
      resUV@vcRefAnnoString2 = ""
      resUV@vcRefMagnitudeF = 10.
      resUV@vcMinMagnitudeF = 2.
      resUV@vcRefAnnoString1 = resUV@vcRefMagnitudeF + " " + uunits
    end if

    resDomain = True
    resDomain@res_gsnDraw = False
    resDomain@res_gsnFrame = False
    resDomain@res_gsnMaximize = False
    resDomain@res_gsnAddCyclic = True
    resDomain@res_cnFillOn = True
    resDomain@res_cnLinesOn = True
    resDomain@res_cnLineLabelsOn = False
    resDomain@res_cnInfoLabelOn = False
    resDomain@res_cnLevelSelectionMode = "ExplicitLevels"
    resDomain@res_cnLevels = (/ 0.5 /)
    resDomain@res_cnLineThicknessF = 2.
    resDomain@res_cnLineColor = "brown"
    resDomain@res_cnMonoFillPattern = False
    resDomain@res_cnMonoFillColor = True
    resDomain@res_cnMonoFillScale = True
    resDomain@res_cnFillColor = resDomain@res_cnLineColor
    resDomain@res_cnFillScaleF = 0.6
    resDomain@res_cnFillPatterns = (/-1, 3/)
    resDomain@res_lbLabelBarOn = False
    resDomain@res_gsnLeftString = ""
    resDomain@res_gsnRightString = ""

    plots := new((/ 2, 2 /), "graphic")

    copy_VarAtts(resPr, prRef)
    if (drawUV) then
      prRef@res_tiMainString = refnamePr+" & "+refnameUV
    else
      prRef@res_tiMainString = refnamePr
    end if
    prRef@res_mpMinLatF = 0.
    prRef@res_mpMaxLatF = 60.
    prRef@res_tmXBOn = False
    plots(0, 0) = contour_map(wks, prRef, "pr")
    copy_VarAtts(resDomain, domainRef)
    plot = contourplot(wks, domainRef, "domain")
    overlay(plots(0, 0), plot)

    copy_VarAtts(resPr, prMME)
    prMME@res_tiMainString = "MME mean"
    prMME@res_mpMinLatF = 0.
    prMME@res_mpMaxLatF = 60.
    prMME@res_tmXBOn = False
    plots(0, 1) = contour_map(wks, prMME, "pr")
    copy_VarAtts(resDomain, domainMME)
    plot = contourplot(wks, domainMME, "domain")
    overlay(plots(0, 1), plot)

    prRef@res_tiMainString = ""
    prRef@res_mpMinLatF = -60.
    prRef@res_mpMaxLatF = 0.
    prRef@res_tmXBOn = True
    plots(1, 0) = contour_map(wks, prRef, "pr")
    plot = contourplot(wks, domainRef, "domain")
    overlay(plots(1, 0), plot)

    prMME@res_tiMainString = ""
    prMME@res_mpMinLatF = -60.
    prMME@res_mpMaxLatF = 0.
    prMME@res_tmXBOn = True
    plots(1, 1) = contour_map(wks, prMME, "pr")
    plot = contourplot(wks, domainMME, "domain")
    overlay(plots(1, 1), plot)

    if (isdefined("uRef").and.isdefined("vRef").and. \
        isdefined("uMME").and.isdefined("vMME")) then
      resUV@vcRefAnnoOn = False
      plot = gsn_csm_vector(wks, uRef, vRef, resUV)
      overlay(plots(0, 0), plot)
      plot = gsn_csm_vector(wks, uMME, vMME, resUV)
      overlay(plots(0, 1), plot)
      plot = gsn_csm_vector(wks, uRef, vRef, resUV)
      overlay(plots(1, 0), plot)
      resUV@vcRefAnnoOn = True
      plot = gsn_csm_vector(wks, uMME, vMME, resUV)
      overlay(plots(1, 1), plot)
    end if

    pres                   = True
    pres@gsnFrame          = False
    pres@gsnPanelBottom    = 0.4
    pres@gsnPanelLabelBar  = True
;    pres@gsnMaximize = True
    pres@gsnPanelYWhiteSpacePercent = 0.
    if (drawUV) then
      pres@pmLabelBarOrthogonalPosF = 0.025
    else
      pres@pmLabelBarOrthogonalPosF = -0.01
    end if
    gsn_panel(wks, ndtooned(plots), (/2, 2/), pres)
  end if

end

; ============================================================

begin

  if (moving_ave.gt.0) then
    if (mod(moving_ave, 2).eq.0) then
      wgt = new(moving_ave+1, "float")
      wgt = 1.
      wgt(0) = 0.5
      wgt(moving_ave) = 0.5
    else
      wgt = new(moving_ave, "float")
      wgt = 1.
    end if
    wgt = wgt / tofloat(moving_ave)
  end if

  drawGMpr = True
  if (isatt(diag_script_info, "plot_globalprecipitation")) then
    drawGMpr = diag_script_info@plot_globalprecipitation
  end if
  drawNHpr = True
  if (isatt(diag_script_info, "plot_NHprecipitation")) then
    drawNHpr = diag_script_info@plot_NHprecipitation
  end if
  drawNHcirc = True
  if (isatt(diag_script_info, "plot_NHcirculation")) then
    drawNHcirc = diag_script_info@plot_NHcirculation
  end if

  if (drawGMpr.or.drawNHpr) then
    GMfiles = systemfunc("ls "+input_dir+"GMpr*.nc")

    do ii = 0, dimsizes(GMfiles)-1
      f = addfile(GMfiles(ii), "r")
      tmp = f->GMpr
      if (moving_ave.gt.0) then
        tmp = wgt_runave_n(tmp, wgt, 0, 0)
      end if

      NHMfile = str_sub_str(GMfiles(ii), "GMpr", "NHMpr")
      f = addfile(NHMfile, "r")
      tmpNH = f->NHMpr
      if (moving_ave.gt.0) then
        tmpNH = wgt_runave_n(tmpNH, wgt, 0, 0)
      end if

      if (any(tmp@project \
              .eq.(/"obs", "OBS", "OBS6", "obs4mips", "ana4mips"/))) then
        if (.not.isdefined("gObs")) then
          gObs = NewList("fifo")
        end if
        ListAppend(gObs, new(dimsizes(tmp), typeof(tmp)))
        i = ListCount(gObs)-1
        gObs[i] = tmp

        if (.not.isdefined("nObs")) then
          nObs = NewList("fifo")
        end if
        ListAppend(nObs, new(dimsizes(tmpNH), typeof(tmpNH)))
        i = ListCount(nObs)-1
        nObs[i] = tmpNH
      else
        if (.not.isdefined("gCMIP")) then
          gCMIP = NewList("fifo")
        end if
        ListAppend(gCMIP, new(dimsizes(tmp), typeof(tmp)))
        i = ListCount(gCMIP)-1
        gCMIP[i] = tmp

        if (.not.isdefined("nCMIP")) then
          nCMIP = NewList("fifo")
        end if
        ListAppend(nCMIP, new(dimsizes(tmpNH), typeof(tmpNH)))
        i = ListCount(nCMIP)-1
        nCMIP[i] = tmpNH
      end if
      delete(tmp)
      delete(tmpNH)
    end do

    syear = 999999
    eyear = -999999
    nobs = ListCount(gObs)
    do ii = 0, nobs-1
      syear = min((/toint(gObs[ii]&time(0)), syear/))
      eyear = max((/toint(gObs[ii]&time(dimsizes(gObs[ii]&time)-1)), eyear/))
    end do
    GMprObs = new((/ nobs, eyear-syear+1 /), "float")
    GMprObs!0 = "dataset"
    GMprObs!1 = "time"
    GMprObs&dataset = ispan(1, nobs, 1)
    GMprObs&time = ispan(syear, eyear, 1)
    NHMprObs = new((/ nobs, eyear-syear+1 /), "float")
    NHMprObs!0 = "dataset"
    NHMprObs!1 = "time"
    NHMprObs&dataset = ispan(1, nobs, 1)
    NHMprObs&time = ispan(syear, eyear, 1)
    icolor_satellite = 0
    icolor_insitu = 0
    colors   = new(nobs, "string")
    patterns = new(nobs, "integer")
    do ii = 0, nobs-1
      syr = gObs[ii]&time(0)
      eyr = gObs[ii]&time(dimsizes(gObs[ii]&time)-1)
      GMprObs(ii, {syr:eyr}) = (/ gObs[ii] /)

      syr = nObs[ii]&time(0)
      eyr = nObs[ii]&time(dimsizes(nObs[ii]&time)-1)
      NHMprObs(ii, {syr:eyr}) = (/ nObs[ii] /)

      if (any((/isStrSubset(gObs[ii]@dataset, "GPCP"), \
              isStrSubset(gObs[ii]@dataset, "CMAP"), \
              isStrSubset(gObs[ii]@dataset, "TRMM")/))) then
        colors(ii) = colors_satellite(icolor_satellite)
        icolor_satellite = icolor_satellite+1
      else
        colors(ii) = colors_insitu(icolor_insitu)
        icolor_insitu = icolor_insitu+1
      end if
      patterns(ii) = 0
      if (ii.eq.0) then
        GMprObs@dataset := (/ gObs[ii]@dataset /)
      else
        GMprObs@dataset := array_append_record(GMprObs@dataset, \
                                               gObs[ii]@dataset, 0)
      end if
    end do
    delete(gObs)
    delete(nObs)

    syear = 999999
    eyear = -999999
    nmodels = ListCount(gCMIP)
    do ii = 0, nmodels-1
      syear = min((/toint(gCMIP[ii]&time(0)), syear/))
      eyear = max((/toint(gCMIP[ii]&time(dimsizes(gCMIP[ii]&time)-1)), eyear/))
    end do
    GMprModels = new((/ nmodels, eyear-syear+1 /), "float")
    GMprModels!0 = "model"
    GMprModels!1 = "time"
    GMprModels&model = ispan(1, nmodels, 1)
    GMprModels&time = ispan(syear, eyear, 1)
    do ii = 0, nmodels-1
      syr = gCMIP[ii]&time(0)
      eyr = gCMIP[ii]&time(dimsizes(gCMIP[ii]&time)-1)
      GMprModels(ii, {syr:eyr}) = (/ gCMIP[ii] /)
    end do
    GMprModels@units = gCMIP[0]@units

    NHMprModels = new((/ nmodels, eyear-syear+1 /), "float")
    NHMprModels!0 = "model"
    NHMprModels!1 = "time"
    NHMprModels&model = ispan(1, nmodels, 1)
    NHMprModels&time = ispan(syear, eyear, 1)
    do ii = 0, nmodels-1
      syr = nCMIP[ii]&time(0)
      eyr = nCMIP[ii]&time(dimsizes(nCMIP[ii]&time)-1)
      NHMprModels(ii, {syr:eyr}) = (/ nCMIP[ii] /)
    end do
    NHMprModels@units = nCMIP[0]@units
    delete(gCMIP)
    delete(nCMIP)

    GMprMME = dim_avg_n_Wrap(GMprModels, 0)
    GMprMME = where(dim_num_n(ismissing(GMprModels), 0).le.nmodels/2, \
                    GMprMME, GMprMME@_FillValue)
                    NHMprMME = dim_avg_n_Wrap(NHMprModels, 0)
    NHMprMME = where(dim_num_n(ismissing(NHMprModels), 0).le.nmodels/2, \
                     NHMprMME, NHMprMME@_FillValue)
  end if

  if (drawNHcirc) then
    NHMufiles = systemfunc("ls "+input_dir+"NHMcirculation_summer_*.nc")

    if (all(ismissing(NHMufiles))) then
      drawNHcirc = False
    else
      do ii = 0, dimsizes(NHMufiles)-1
        f = addfile(NHMufiles(ii), "r")
        tmp = f->NHMu
        if (moving_ave.gt.0) then
          tmp = wgt_runave_n(tmp, wgt, 0, 0)
        end if

        if (any(tmp@project \
                .eq.(/"obs", "OBS", "OBS6", "obs4mips", "ana4mips"/))) then
          if (.not.isdefined("uObs")) then
            uObs = NewList("fifo")
          end if
          ListAppend(uObs, new(dimsizes(tmp), typeof(tmp)))
          i = ListCount(uObs)-1
          uObs[i] = tmp
        else
          if (.not.isdefined("uCMIP")) then
            uCMIP = NewList("fifo")
          end if
          ListAppend(uCMIP, new(dimsizes(tmp), typeof(tmp)))
          i = ListCount(uCMIP)-1
          uCMIP[i] = tmp
        end if
        delete(tmp)
      end do

      syear = 999999
      eyear = -999999
      nobsu = ListCount(uObs)
      do ii = 0, nobsu-1
        syear = min((/toint(uObs[ii]&time(0)), syear/))
        eyear = max((/toint(uObs[ii]&time(dimsizes(uObs[ii]&time)-1)), eyear/))
      end do
      NHMuObs = new((/ nobsu, eyear-syear+1 /), "float")
      NHMuObs!0 = "dataset"
      NHMuObs!1 = "time"
      NHMuObs&dataset = ispan(1, nobsu, 1)
      NHMuObs&time = ispan(syear, eyear, 1)

      icolor_reanal = 0
      colorsu   = new(nobsu, "string")
      patternsu = new(nobsu, "integer")
      do ii = 0, nobsu-1
        syr = uObs[ii]&time(0)
        eyr = uObs[ii]&time(dimsizes(uObs[ii]&time)-1)
        NHMuObs(ii, {syr:eyr}) = (/ uObs[ii] /)

        colorsu(ii) =   colors_reanal(icolor_reanal)
        icolor_reanal = icolor_reanal+1
        patternsu(ii) = 0
        if (ii.eq.0) then
          NHMuObs@dataset := (/ uObs[ii]@dataset /)
        else
          NHMuObs@dataset \
            := array_append_record(NHMuObs@dataset, uObs[ii]@dataset, 0)
        end if
      end do
      delete(uObs)

      syear = 999999
      eyear = -999999
      nmodels = ListCount(uCMIP)
      do ii = 0, nmodels-1
        syear = min((/toint(uCMIP[ii]&time(0)), syear/))
        eyear = max((/toint(uCMIP[ii]&time(dimsizes(uCMIP[ii]&time)-1)), \
                    eyear/))
      end do
      NHMuModels = new((/ nmodels, eyear-syear+1 /), "float")
      NHMuModels!0 = "model"
      NHMuModels!1 = "time"
      NHMuModels&model = ispan(1, nmodels, 1)
      NHMuModels&time = ispan(syear, eyear, 1)
      do ii = 0, nmodels-1
        syr = uCMIP[ii]&time(0)
        eyr = uCMIP[ii]&time(dimsizes(uCMIP[ii]&time)-1)
        NHMuModels(ii, {syr:eyr}) = (/ uCMIP[ii] /)
      end do
      delete(uCMIP)

      NHMuMME = dim_avg_n_Wrap(NHMuModels, 0)
      NHMuMME = where(dim_num_n(ismissing(NHMuModels), 0).le.nmodels/2, \
                      NHMuMME, NHMuMME@_FillValue)
    end if
  end if

  plotts = new(4, "graphic")
  iplots = 0
  if (drawGMpr) then
    iplotsGMpr = iplots
    iplots = iplots + 1
  end if
  if (drawNHpr) then
    iplotsNHpr = iplots
    iplots = iplots + 1
  end if
  if (drawNHcirc) then
    iplotsNHcirc = iplots
    iplots = iplots + 1
  end if
  if (iplots.eq.1) then
    xLegendRef = 0.5
    yLegendRef = 0.6
  elseif (iplots.eq.2) then
    xLegendRef = 0.5
    yLegendRef = 0.4
  elseif (iplots.eq.3) then
    xLegendRef = 0.5
    yLegendRef = 0.3
  end if

  rests := True
  rests@xyDashPattern     = 0
  rests@xyMonoLineColor   = False
  rests@gsnYRefLine       = 0.
  rests@tmYLLabelFontHeightF = 0.01
  rests@tmXBLabelFontHeightF = 0.01
  rests@gsnDraw           = False
  rests@gsnFrame          = False

  if (drawGMpr) then
    minyear = min((/min(GMprObs&time), min(GMprMME&time), \
                  min(GMprModels&time)/))
    maxyear = max((/max(GMprObs&time), max(GMprMME&time), \
                  max(GMprModels&time)/))
  elseif (drawNHpr) then
    minyear = min((/min(NHMprObs&time), min(NHMprMME&time), \
                  min(NHMprModels&time)/))
    maxyear = max((/max(NHMprObs&time), max(NHMprMME&time), \
                  max(NHMprModels&time)/))
  elseif (drawNHcirc) then
    minyear = min((/min(NHMuObs&time), min(NHMuMME&time), \
                  min(NHMuModels&time)/))
    maxyear = max((/max(NHMuObs&time), max(NHMuMME&time), \
                  max(NHMuModels&time)/))
  end if
  minyear = toint(minyear/5)*5
  if (.not.mod(maxyear, 5).eq.0) then
    maxyear = (toint(maxyear/5) + 1)*5
  end if
  rests@trXMinF = 1880  ; minyear
  rests@trXMaxF = maxyear
  rests@tmXTOn = False
  rests@tmYROn = False
  rests@vpWidthF = 0.6
  rests@vpHeightF = 0.2
;  rests@vpXF = 0.2
;  rests@vpYF = 0.4
  rests@tiMainFontHeightF = 0.015
  rests@tiMainOffsetYF = -0.008

  txres := True
  txres@txFontHeightF = 0.01
  txres@txJust = "CenterLeft"
  lres := True
  yLegend = yLegendRef - 0.15  ; 0.32

  rests@xyLineThicknessF = 0.5
  rests@xyLineColor      = (/204., 35., 35./)/255.  ; (/37, 81., 204./)/255.
  rests@xyLineOpacityF   = 0.5
  if (drawGMpr) then
    range = max((/max(abs(GMprObs)), max(abs(GMprMME)), \
                max(abs(GMprModels))/))
  elseif (drawNHpr) then
    range = max((/max(abs(NHMprObs)), max(abs(NHMprMME)), \
                max(abs(NHMprModels))/))
  end if

  if (drawGMpr.or.drawNHpr) then
    if (range.gt.80.) then
      rests@trYMaxF = 100.
    elseif (range.gt.70.) then
      rests@trYMaxF = 80.
    elseif (range.gt.60.) then
      rests@trYMaxF = 70.
    elseif (range.gt.50.) then
      rests@trYMaxF = 60.
    else
      rests@trYMaxF = 50.
    end if
    rests@trYMinF = -rests@trYMaxF
  end if

  if (drawGMpr) then
    if (moving_ave.gt.0) then
      rests@tiMainString = "Global summer monsoon precipitation (" \
                           + GMprModels@units + ", " + moving_ave \
                           + "-yr moving ave)"
    else
      rests@tiMainString = "Global summer monsoon precipitation (" \
                           + GMprModels@units + ")"
    end if
    GMprModels@long_names = ""
    plotts(iplotsGMpr) = gsn_csm_xy(wks, GMprModels&time, GMprModels, rests)
  end if

  if (drawNHpr) then
    if (moving_ave.gt.0) then
      rests@tiMainString = "NH summer monsoon precipitation (" \
                         + NHMprModels@units + ", " \
                         + moving_ave + "-yr moving ave)"
    else
      rests@tiMainString = "NH summer monsoon precipitation (" \
                           + NHMprModels@units + ")"
    end if
    NHMprModels@long_names = ""
    plotts(iplotsNHpr) = gsn_csm_xy(wks, NHMprModels&time, NHMprModels, rests)
  end if

;  if (isvar("NHMuModels")) then
  if (drawNHcirc) then
    restsu = rests
    range = max((/max(abs(NHMuObs)), max(abs(NHMuMME)), \
                max(abs(NHMuModels))/))
    if (range.gt.3.) then
      restsu@trYMaxF = 3.5
    elseif (range.gt.2.8) then
      restsu@trYMaxF = 3.
    elseif (range.gt.2.6) then
      restsu@trYMaxF = 2.8
    elseif (range.gt.2.4) then
      restsu@trYMaxF = 2.6
    elseif (range.gt.2.2) then
      restsu@trYMaxF = 2.4
    elseif (range.gt.2.) then
      restsu@trYMaxF = 2.2
    elseif (range.gt.1.8) then
      restsu@trYMaxF = 2.
    elseif (range.gt.1.6) then
      restsu@trYMaxF = 1.8
    else
      restsu@trYMaxF = 1.6
    end if
    restsu@trYMinF = -restsu@trYMaxF
    if (moving_ave.gt.0) then
      restsu@tiMainString = "NH summer monsoon circulation (m/s, " \
                            + moving_ave + "-yr moving ave)"
    else
      restsu@tiMainString = "NH summer monsoon circulationn (m/s)"
    end if
    NHMuModels@long_names = ""
    plotts(iplotsNHcirc) = gsn_csm_xy(wks, NHMuModels&time, NHMuModels, restsu)
  end if

  lres@gsLineThicknessF = rests@xyLineThicknessF
  lres@gsLineColor      = rests@xyLineColor
  lres@gsLineOpacityF   = rests@xyLineOpacityF
  gsn_polyline_ndc(wks, (/0.64, 0.67/), (/yLegend, yLegend/), lres)
  gsn_text_ndc(wks, "individual runs", 0.68, yLegend, txres)

  rests@tiMainString = ""
  rests@xyLineThicknessF = 2.5
  rests@xyLineOpacityF   = 1.

  if (drawGMpr) then
    GMprMME@long_names = ""
    plot = gsn_csm_xy(wks, GMprMME&time, GMprMME, rests)
    overlay(plotts(iplotsGMpr), plot)
  end if

  if (drawNHpr) then
    NHMprMME@long_names = ""
    plot = gsn_csm_xy(wks, NHMprMME&time, NHMprMME, rests)
    overlay(plotts(iplotsNHpr), plot)
  end if

  if (drawNHcirc) then
    restsu@tiMainString = ""
    restsu@xyLineThicknessF = 2.5
    restsu@xyLineOpacityF   = 1.
    NHMuMME@long_names = ""
    plot = gsn_csm_xy(wks, NHMuMME&time, NHMuMME, restsu)
    overlay(plotts(iplotsNHcirc), plot)
  end if

  lres@gsLineThicknessF = rests@xyLineThicknessF
  lres@gsLineColor      = rests@xyLineColor
  lres@gsLineOpacityF   = rests@xyLineOpacityF
  gsn_polyline_ndc(wks, (/0.51, 0.54/), (/yLegend, yLegend/), lres)
  gsn_text_ndc(wks, "MME mean", 0.55, yLegend, txres)

  if (isatt(rests, "xyLineColor")) then
    delete(rests@xyLineColor)
  end if
  if (isatt(rests, "xyDashPattern")) then
    delete(rests@xyDashPattern)
  end if

  if (drawGMpr) then
    rests@xyLineThicknessF  = 1.5
    rests@xyLineColors     := colors(0:nobs-1)
    rests@xyDashPatterns   := patterns(0:nobs-1)
    GMprObs@long_names = ""
    plot = gsn_csm_xy(wks, GMprObs&time, (/GMprObs/), rests)
    overlay(plotts(iplotsGMpr), plot)
  end if

  if (drawNHpr) then
    rests@xyLineThicknessF  = 1.5
    rests@xyLineColors     := colors(0:nobs-1)
    rests@xyDashPatterns   := patterns(0:nobs-1)
    NHMprObs@long_names = ""
    plot = gsn_csm_xy(wks, NHMprObs&time, NHMprObs, rests)
    overlay(plotts(iplotsNHpr), plot)
  end if

  if (drawNHcirc) then
    delete(restsu@xyLineColor)
    delete(restsu@xyDashPattern)
    restsu@xyLineThicknessF = 1.5
    restsu@xyLineColors    := colorsu(0:nobsu-1)
    restsu@xyDashPatterns  := patternsu(0:nobsu-1)
    NHMuObs@long_names = ""
    plot = gsn_csm_xy(wks, NHMuObs&time, (/NHMuObs/), restsu)
    overlay(plotts(iplotsNHcirc), plot)
  end if

  yLegend = yLegendRef - 0.17
  if (drawGMpr.or.drawNHpr) then
    do ii = 0, dimsizes(colors)-1
      lres@gsLineThicknessF  := rests@xyLineThicknessF
      lres@gsLineColor       := rests@xyLineColors(ii)
      lres@gsLineDashPattern := rests@xyDashPatterns(ii)
      xLegend = xLegendRef + 0.13*ii + 0.01
      gsn_polyline_ndc(wks, (/xLegend, xLegend+0.03/), (/yLegend, yLegend/), \
                       lres)
      if (drawGMpr) then
        gsn_text_ndc(wks, GMprObs@dataset(ii), xLegend+0.04, yLegend, txres)
      else
        gsn_text_ndc(wks, NHprObs@dataset(ii), xLegend+0.04, yLegend, txres)
      end if
    end do
    yLegend = yLegend - 0.02
  end if

  if (drawNHcirc) then
    do ii = 0, dimsizes(colorsu)-1
      lres@gsLineThicknessF := restsu@xyLineThicknessF
      lres@gsLineColor      := restsu@xyLineColors(ii)
      lres@gsLineDashPattern := restsu@xyDashPatterns(ii)
      xLegend = xLegendRef + 0.13*ii + 0.01
      gsn_polyline_ndc(wks, (/xLegend, xLegend+0.03/), (/yLegend, yLegend/), \
                       lres)
      gsn_text_ndc(wks, NHMuObs@dataset(ii), xLegend+0.04, yLegend, txres)
    end do
  end if

  pres                   = True
  pres@gsnFrame          = False
  pres@gsnPanelTop       = 0.67
  pres@gsnPanelBottom    = 0.
  pres@gsnPanelLabelBar  = False
  pres@gsnMaximize       = True
  pres@gsnPanelYWhiteSpacePercent = 0.
  pres@pmLabelBarOrthogonalPosF = 0.025
  gsn_panel(wks, plotts, (/2, 2/), pres)
  frame(wks)

end
