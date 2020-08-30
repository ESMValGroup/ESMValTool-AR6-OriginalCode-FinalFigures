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
;    20200824 kosaka_yu: changed figure layouts.
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
load "$diag_scripts/ar6ch3_monsoon/percentiles.ncl"

function rm_last_slash(x)
local y, n
begin
  n = strlen(x)
  if (str_get_cols(x, n-1, n-1).eq."/") then
    y = str_get_cols(x, 0, n-2)
  else
    y = x
  end if
  return(y)
end

begin

  enter_msg(DIAG_SCRIPT, "")

  colors_satellite = (/"purple", "orange", "brown"/)
  colors_satellite = (/"black", "black", "black"/)
  patterns_satellite = (/14, 16, 6/)
  seglens_satellite = (/0.05, 0.06, 0.15 /)
  colors_insitu = (/"green", "blue", "darkorchid4", "deepping4"/)
  colors_insitu = (/"black", "black", "black", "black"/)
  patterns_insitu = (/0, 3, 2, 1/)
  seglens_insitu = (/0.15, 0.1, 0.05, 0.15/)
  
  colors_reanal = (/"blue3", "darkgreen", "darkslategray", "darkseagreen"/)
  colors_reanal = (/"black", "black", "black", "black"/)
  patterns_reanal = (/0, 1, 14, 7/)
  seglens_reanal = (/0.15, 0.1, 0.05, 0.15 /)

  color_CMIP6 = (/204., 35., 35./)/255.
  color_CMIP5 = (/37., 81., 204./)/255.
  color_AMIP  = (/236., 156., 46./)/255.  ; (/236., 209., 81./)/255.
  
  moving_ave = 0
  if (isatt(diag_script_info, "moving_ave")) then
    moving_ave = diag_script_info@moving_ave
  end if
  scale_precip_map_ann = True
  if (isatt(diag_script_info, "scale_precip_map_ann")) then
    scale_precip_map_ann = diag_script_info@scale_precip_map_ann
  end if

  if (isatt(diag_script_info, "diagnostic_map")) then
    diagnostic_map = diag_script_info@diagnostic_map
  else
    diagnostic_map = ""
  end if
  refPr_map = "GPCP-SG"
  if (isatt(diag_script_info, "dataset_obs_pr_map")) then
    refPr_map = diag_script_info@dataset_obs_pr_map
  end if
  refUV_map = "ERA5"
  if (isatt(diag_script_info, "dataset_obs_uv_map")) then
    refUV_map = diag_script_info@dataset_obs_uv_map
  end if    
  project_map = "CMIP6"
  if (isatt(diag_script_info, "project_map")) then
    project_map = diag_script_info@project_map
  end if    
  
  if (isatt(diag_script_info, "diagnostic_precip")) then
    diagnostic_precip = diag_script_info@diagnostic_precip
  else
    diagnostic_precip = ""
  end if

  if (isatt(diag_script_info, "diagnostic_circulation")) then
    diagnostic_circulation = diag_script_info@diagnostic_circulation
  else
    diagnostic_circulation = ""
  end if
  
; =======================================================
  
  if (all(diagnostic_map .eq. "")) then
    input_dir_map = ""
  else
    input_dir_map = new(dimsizes(diagnostic_map), "string")
    do n = 0, dimsizes(input_dir_map)-1
      diagnostic_map_tmp = rm_last_slash(diagnostic_map(n))
      do i = 0, dimsizes(diag_script_info@input_files)-1
        tmp_input_path = rm_last_slash(diag_script_info@input_files(i))
        k = str_index_of_substr(tmp_input_path, diagnostic_map_tmp, -1)
        if (.not.ismissing(k)) then
          if (dimsizes(k).gt.1) then
            j = k(dimsizes(k)-1)
          else
            j = k
          end if
          if (j.eq.strlen(tmp_input_path)-strlen(diagnostic_map_tmp)) then
            input_dir_map(n) = tmp_input_path + "/"
          end if
          delete(j)
        end if
        delete(k)
      end do
    end do
    if (all(input_dir_map.eq."")) then
      error_msg("f", DIAG_SCRIPT, "draw_global_monsoon", "Cannot find input_dir_map.")
    end if
  end if
  if (all(diagnostic_precip .eq. "")) then
    input_dir_precip = ""
  else
    input_dir_precip = new(dimsizes(diagnostic_precip), "string")
    do n = 0, dimsizes(input_dir_precip)-1
      diagnostic_precip_tmp = rm_last_slash(diagnostic_precip(n))
      do i = 0, dimsizes(diag_script_info@input_files)-1
        tmp_input_path = rm_last_slash(diag_script_info@input_files(i))
        k = str_index_of_substr(tmp_input_path, diagnostic_precip_tmp, -1)
        if (.not.ismissing(k)) then
          if (dimsizes(k).gt.1) then
            j = k(dimsizes(k)-1)
          else
            j = k
          end if
          if (j.eq.strlen(tmp_input_path)-strlen(diagnostic_precip_tmp)) then
            input_dir_precip(n) = tmp_input_path + "/"
          end if
          delete(j)
        end if
        delete(k)
      end do
    end do
    if (all(input_dir_precip.eq."")) then
      error_msg("f", DIAG_SCRIPT, "draw_global_monsoon", "Cannot find input_dir_precip.")
    end if
  end if
  if (all(diagnostic_circulation .eq. "")) then
    input_dir_circulation = ""
  else
    input_dir_circulation = new(dimsizes(diagnostic_circulation), "string")
    do n = 0, dimsizes(diagnostic_circulation)-1
      diagnostic_circulation_tmp = rm_last_slash(diagnostic_circulation(n))
      do i = 0, dimsizes(diag_script_info@input_files)-1
        tmp_input_path = rm_last_slash(diag_script_info@input_files(i))
        k = str_index_of_substr(tmp_input_path, diagnostic_circulation_tmp, -1)
        if (.not.ismissing(k)) then
          if (dimsizes(k).gt.1) then
            j = k(dimsizes(k)-1)
          else
            j = k
          end if
          if (j.eq.strlen(tmp_input_path)-strlen(diagnostic_circulation_tmp)) then
            input_dir_circulation(n) = tmp_input_path + "/"
          end if
          delete(j)
        end if
        delete(k)
      end do
    end do
    if (all(input_dir_circulation.eq."")) then
      error_msg("f", DIAG_SCRIPT, "draw_global_monsoon", "Cannot find input_dir_circulation.")
    end if
  end if

end

; ===================================================================

begin

  if (.not.all(input_dir_map .eq. "")) then
    if (dimsizes(input_dir_map).eq.1) then
        domainFileRef  = input_dir_map+"domain_monsoon_"+refPr_map+".nc"
        if (scale_precip_map_ann) then
          prFileRef    = input_dir_map+"monsoon_precip_index_"+refPr_map+".nc"
        else
          prFileRef    = input_dir_map+"precip_monsoon_"+refPr_map+".nc"
        end if
        uFileRef       = input_dir_map+"u_monsoon_"+refUV_map+".nc"
        vFileRef       = input_dir_map+"v_monsoon_"+refUV_map+".nc"

        domainFileMME  = input_dir_map+"domain_monsoon_MultiModelMean"+project_map+".nc"
        if (scale_precip_map_ann) then
          prFileMME    = input_dir_map+"monsoon_precip_index_MultiModelMean"+project_map+".nc"
        else
          prFileMME    = input_dir_map+"precip_monsoon_MultiModelMean"+project_map+".nc"
        end if
        uFileMME       = input_dir_map+"u_monsoon_MultiModelMean"+project_map+".nc"
        vFileMME       = input_dir_map+"v_monsoon_MultiModelMean"+project_map+".nc"
    else
      do n = 0, dimsizes(input_dir_map)-1
        if (.not.isvar("domainFileRef")) then
          tmppath = input_dir_map(n)+"domain_monsoon_"+refPr_map+".nc"
          if (fileexists(tmppath)) then
            domainFileRef = tmppath
          end if
        end if
        if (.not.isvar("prFileRef")) then
          if (scale_precip_map_ann) then
            tmppath = input_dir_map(n)+"monsoon_precip_index_"+refPr_map+".nc"
          else
            tmppath = input_dir_map(n)+"precip_monsoon_"+refPr_map+".nc"
          end if
          if (fileexists(tmppath)) then
            prFileRef = tmppath
          end if
        end if
        if (.not.isvar("uFileRef")) then
          tmppath = input_dir_map(n)+"u_monsoon_"+refUV_map+".nc"
          if (fileexists(tmppath)) then
            uFileRef = tmppath
          end if
        end if
        if (.not.isvar("vFileRef")) then
          tmppath = input_dir_map(n)+"v_monsoon_"+refUV_map+".nc"
          if (fileexists(tmppath)) then
            vFileRef = tmppath
          end if
        end if

        if (.not.isvar("domainFileMME")) then
          tmppath = input_dir_map(n)+"domain_monsoon_MultiModelMean"+project_map+".nc"
          if (fileexists(tmppath)) then
            domainFileMME = tmppath
          end if
        end if
        if (.not.isvar("prFileMME")) then
          if (scale_precip_map_ann) then
            tmppath = input_dir_map(n)+"monsoon_precip_index_MultiModelMean"+project_map+".nc"
          else
            tmppath = input_dir_map(n)+"precip_monsoon_MultiModelMean"+project_map+".nc"
          end if
          if (fileexists(tmppath)) then
            prFileMME = tmppath
          end if
        end if
        if (.not.isvar("uFileMME")) then
          tmppath = input_dir_map(n)+"u_monsoon_MultiModelMean"+project_map+".nc"
          if (fileexists(tmppath)) then
            uFileMME = tmppath
          end if
        end if
        if (.not.isvar("vFileMME")) then
          tmppath = input_dir_map(n)+"v_monsoon_MultiModelMean"+project_map+".nc"
          if (fileexists(tmppath)) then
            vFileMME = tmppath
          end if
        end if
      end do
    end if
    
    f = addfile(domainFileRef, "r")
    domainRef = f->monsoon_domain
    domainRef@long_name = ""
    f = addfile(domainFileMME, "r")
    domainMME = f->monsoon_domain
    domainMME@long_name = ""

    syear_clim = domainRef@syear_clim
    eyear_clim = domainRef@eyear_clim
      
;    drawPr = False
;    if (fileexists(prFileRef).and.fileexists(prFileMME)) then
;      drawPr = True
      f = addfile(prFileRef, "r")
      if (scale_precip_map_ann) then
        prRef = f->MPI
      else
        prRef = f->pr
      end if
      prRef@long_name = ""
      prRef@units = ""
      refnamePr = prRef@dataset
      f = addfile(prFileMME, "r")
      if (scale_precip_map_ann) then
        prMME = f->MPI
      else
        prMME = f->pr
      end if
      prMME@long_name = ""
      prMME@units = ""
      MMEnamePr = prMME@dataset
;    end if
    
    drawUV = False
    if (fileexists(uFileRef).and.fileexists(vFileRef).and. \
        fileexists(uFileMME).and.fileexists(vFileMME)) then
      drawUV = True
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
      if (moving_ave.gt.1) then
        wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                      "global_monsoon_" + refnamePr + "+" \
                      + refnameUV + "_movingave" + moving_ave + "yrs")
      else
        wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                      "global_monsoon_" + refnamePr + "+" + refnameUV)
      end if
    else
      if (moving_ave.gt.1) then
        wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                      "global_monsoon_" + refnamePr + \
                      "_movingave" + moving_ave + "yrs")
      else
        wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                      "global_monsoon_" + refnamePr)
      end if
    end if

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

    if (scale_precip_map_ann) then
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
      resUV@vcRefAnnoFontHeightF = 0.015
      resUV@vcRefAnnoArrowSpaceF = 0.15
      resUV@vcRefAnnoArrowMinOffsetF = 0.15
      resUV@vcRefAnnoOrthogonalPosF = 0.2
      resUV@vcRefAnnoParallelPosF = 1.
      resUV@vcRefAnnoPerimOn = True ; False
      resUV@vcRefLengthF = 0.035
      resUV@vcMinDistanceF = 0.02
      resUV@vcRefAnnoString2 = "Summer minus Winter~C~ 850hPa wind velocity"
      resUV@vcRefMagnitudeF = 10.
      resUV@vcMinMagnitudeF = 2.
      resUV@vcRefAnnoString1 = resUV@vcRefMagnitudeF + " m s~S~-1~N~"
;      resUV@vcRefAnnoString1 = resUV@vcRefMagnitudeF + " " + uunits
    end if

    domainRefBound = domainRef
    domainRef_tmp = smth9_Wrap(domainRef, 0.5, 0.25, True)
    domainRef = where(domainRef.le.0.5, domainRef, domainRef_tmp)
    delete(domainRef_tmp)
      
    domainMMEBound = domainMME
    domainMME_tmp = smth9_Wrap(domainMME, 0.5, 0.25, True)
    domainMME = where(domainMME.le.0.5, domainMME, domainMME_tmp)    
    delete(domainMME_tmp)
      
    resDomain = True
    resDomain@res_gsnDraw = False
    resDomain@res_gsnFrame = False
    resDomain@res_gsnMaximize = False
    resDomain@res_gsnAddCyclic = True
    resDomain@res_cnFillOn = True
    resDomain@res_cnLinesOn = False
    resDomain@res_cnLineLabelsOn = False
    resDomain@res_cnInfoLabelOn = False
    resDomain@res_cnLevelSelectionMode = "ExplicitLevels"
;    resDomain@res_cnLevels = (/ 0.5 /)
    resDomain@res_cnLevels = (/ 0.5, 0.9 /)
    resDomain@res_cnLineThicknessF = 2.
    resDomain@res_cnLineColor = "brown"
    resDomain@res_cnMonoFillPattern = False
    resDomain@res_cnMonoFillColor = True
    resDomain@res_cnMonoFillScale = True
    resDomain@res_cnFillColor = resDomain@res_cnLineColor
    resDomain@res_cnFillScaleF = 0.4
;    resDomain@res_cnFillPatterns = (/-1, 3/)
    resDomain@res_cnFillPatterns = (/-1, 3, -1/)
    resDomain@res_lbLabelBarOn = False
    resDomain@res_gsnLeftString = ""
    resDomain@res_gsnRightString = ""
    resDomain@res_tiMainString = ""

    resDomainBound = resDomain
    resDomainBound@res_cnFillOn = False
    resDomainBound@res_cnLinesOn = True
    resDomainBound@res_cnLevels := (/ 0.5 /)
        
    plots := new((/ 2, 2 /), "graphic")

    copy_VarAtts(resPr, prRef)
    prRef@res_gsnStringFontHeightF = 0.022
    if (drawUV) then
      prRef@res_tiMainString = ""  ;refnamePr+" & "+refnameUV
      if (refnamePr.eq."GPCP-SG") then
        prRef@res_gsnCenterString = "GPCP & "+refnameUV
      else
        prRef@res_gsnCenterString = refnamePr+" & "+refnameUV
      end if
    else
      prRef@res_tiMainString = ""  ;refnamePr
      prRef@res_gsnCenterString = refnamePr
    end if
    prRef@res_mpMinLatF = 0.
    prRef@res_mpMaxLatF = 60.
    prRef@res_tmXBOn = False
    prRef@res_gsnLeftString = "(a)"
    plots(0, 0) = contour_map(wks, prRef, "pr")

    copy_VarAtts(resDomain, domainRef)
    plot = contourplot(wks, domainRef, "domain")
    overlay(plots(0, 0), plot)
    copy_VarAtts(resDomainBound, domainRefBound)
    plot = contourplot(wks, domainRefBound, "domain")
    overlay(plots(0, 0), plot)
    
    copy_VarAtts(resPr, prMME)
    prMME@res_gsnStringFontHeightF = prRef@res_gsnStringFontHeightF
    prMME@res_tiMainString = ""  ;"MME mean"
    prMME@res_gsnCenterString = "CMIP6 MME mean"
    prMME@res_gsnLeftString = "(b)"
    prMME@res_mpMinLatF = 0.
    prMME@res_mpMaxLatF = 60.
    prMME@res_tmXBOn = False
    plots(0, 1) = contour_map(wks, prMME, "pr")

    copy_VarAtts(resDomain, domainMME)
    plot = contourplot(wks, domainMME, "domain")
    overlay(plots(0, 1), plot)
    copy_VarAtts(resDomainBound, domainMMEBound)
    plot = contourplot(wks, domainMMEBound, "domain")
    overlay(plots(0, 1), plot)

    prRef@res_tiMainString = ""
    prRef@res_gsnCenterString = ""
    prRef@res_gsnLeftString = ""
    prRef@res_mpMinLatF = -60.
    prRef@res_mpMaxLatF = 0.
    prRef@res_tmXBOn = True
    plots(1, 0) = contour_map(wks, prRef, "pr")
    plot = contourplot(wks, domainRef, "domain")
    overlay(plots(1, 0), plot)
    plot = contourplot(wks, domainRefBound, "domain")
    overlay(plots(1, 0), plot)

    prMME@res_tiMainString = ""
    prMME@res_gsnCenterString = ""
    prMME@res_gsnLeftString = ""
    prMME@res_mpMinLatF = -60.
    prMME@res_mpMaxLatF = 0.
    prMME@res_tmXBOn = True
    plots(1, 1) = contour_map(wks, prMME, "pr")
    plot = contourplot(wks, domainMME, "domain")
    overlay(plots(1, 1), plot)
    plot = contourplot(wks, domainMMEBound, "domain")
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
    pres@gsnPanelMainFont  = "helvetica-bold"
    pres@gsnPanelMainFontHeightF = 0.018
    pres@gsnPanelMainPosYF = 0.79
    pres@gsnPanelMainString = "Global monsoon domain and intensity"
    pres@gsnPanelBottom    = 0.4
    pres@gsnPanelLeft      = 0.02
    pres@gsnPanelRight     = 0.98
;    pres@gsnPanelDebug = True
    pres@gsnPanelYF        = (/0.75, 0.75, 0.665, 0.665/)
    pres@gsnPanelLabelBar  = True    
    pres@lbBoxMinorExtentF = 0.6
    pres@lbTitlePosition   = "Bottom"
    pres@lbTitleOffsetF    = 0.25
    pres@lbTitleFontHeightF = 0.012
    if (scale_precip_map_ann) then
      pres@lbTitleString   = "(Summer minus Winter) / (Annual mean) ratio of precipitation rate" 
    else
      pres@lbTitleString   = "Summer minus winter precipitation rate (mm day~S~-1~N~)" 
    end if
    if (drawUV) then
      pres@pmLabelBarOrthogonalPosF = 0.03 ; 0.025
    else
      pres@pmLabelBarOrthogonalPosF = -0.01
    end if
    gsn_panel(wks, ndtooned(plots), (/2, 2/), pres)

    x1 = 0.05
    x2 = 0.14
    y1 = 0.53
    y2 = 0.56
    x = (x1 + x2)/2.
    y = y1 - 0.012
    plres = True
    plres@gsLineThicknessF = resDomain@res_cnLineThicknessF
    plres@gsLineColor = resDomain@res_cnLineColor
    gsn_polyline_ndc(wks, (/x1,x1,x2,x2,x1/), (/y1,y2,y2,y1,y1/), plres)
    pgres = True
    pgres@gsFillIndex = resDomain@res_cnFillPatterns(1)
    pgres@gsFillColor = plres@gsLineColor
    pgres@gsFillScaleF = resDomain@res_cnFillScaleF
    x3 = x1 + 0.005
    x4 = x2 - 0.005
    y3 = y1 + 0.005
    y4 = y2 - 0.005
    gsn_polygon_ndc(wks, (/x1,x1,x2,x2,x1,x3,x3,x4,x4,x3/), (/y1,y2,y2,y1,y1,y3,y4,y4,y3,y3/), pgres)
    txres = True
    txres@txFontHeightF = 0.013
    txres@txFontColor = plres@gsLineColor
    txres@txJust = "CenterCenter"
    gsn_text_ndc(wks, "Monsoon domain", x, y, txres)

  end if

end

; ============================================================

begin

  if (moving_ave.gt.1) then
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

  if (all(input_dir_precip.eq."")) then
    drawPr = False
  else
    drawPr = True
  end if
  if (all(input_dir_circulation.eq."")) then
    drawCirc = False
  else
    drawCirc = True
  end if

  if (drawPr) then
    if (dimsizes(input_dir_precip).gt.1) then
      paths = ""
      do n = 0, dimsizes(input_dir_precip)-1
        paths = paths+input_dir_precip(n)+"GMprecip_summer_*.nc "
      end do
    else
      paths = input_dir_precip+"GMprecip_summer_*.nc"
    end if
    Prfiles = systemfunc("ls "+paths)

    do ii = 0, dimsizes(Prfiles)-1
      f = addfile(Prfiles(ii), "r")
      tmp = f->GMpr
      if (moving_ave.gt.1) then
        tmp = wgt_runave_n(tmp, wgt, 0, 0)
      end if
;      if (isatt(tmp, "experiment_id")) then
;        print(tmp@dataset+" "+tmp@experiment_id)
;      else
;        print(tmp@dataset)
;      end if
      if (any(tmp@project \
              .eq.(/"obs", "OBS", "OBS6", "obs4mips", "ana4mips"/))) then
        if (.not.isdefined("gObs")) then
          gObs = NewList("fifo")
        end if
        ListAppend(gObs, new(dimsizes(tmp), typeof(tmp)))
        i = ListCount(gObs)-1
        gObs[i] = tmp
      elseif (tmp@project.eq."CMIP6".and.isStrSubset(tmp@experiment_id, "historical")) then
        if (.not.isdefined("gCMIP6")) then
          gCMIP6 = NewList("fifo")
        end if
        ListAppend(gCMIP6, new(dimsizes(tmp), typeof(tmp)))
        i = ListCount(gCMIP6)-1
        gCMIP6[i] = tmp
      elseif (tmp@project.eq."CMIP5".and.isStrSubset(tmp@experiment_id, "historical")) then
        if (.not.isdefined("gCMIP5")) then
          gCMIP5 = NewList("fifo")
        end if
        ListAppend(gCMIP5, new(dimsizes(tmp), typeof(tmp)))
        i = ListCount(gCMIP5)-1
        gCMIP5[i] = tmp
      elseif (tmp@experiment_id.eq."amip-hist") then
        if (.not.isdefined("gAMIP")) then
          gAMIP = NewList("fifo")
        end if
        ListAppend(gAMIP, new(dimsizes(tmp), typeof(tmp)))
        i = ListCount(gAMIP)-1
        gAMIP[i] = tmp
      end if
      delete(tmp)
    end do

    syear = 999999
    eyear = -999999
    nobs = ListCount(gObs)
    do ii = 0, nobs-1
      syear = min((/toint(gObs[ii]&time(0)), syear/))
      eyear = max((/toint(gObs[ii]&time(dimsizes(gObs[ii]&time)-1)), eyear/))
    end do
    GMprObs = new((/ nobs, eyear-syear+1 /), "float")
    GMprObs!0 = "dset"
    GMprObs!1 = "time"
    GMprObs&dset = ispan(1, nobs, 1)
    GMprObs&time = ispan(syear, eyear, 1)
    icolor_satellite = 0
    icolor_insitu = 0
    colors   = new(nobs, "string")
    patterns = new(nobs, "integer")
    seglens  = new(nobs, "float")
    do ii = 0, nobs-1
      syr = gObs[ii]&time(0)
      eyr = gObs[ii]&time(dimsizes(gObs[ii]&time)-1)
      GMprObs(ii, {syr:eyr}) = (/ gObs[ii] /)
      if (gObs[ii]@units.eq."") then
        GMprObs(ii, :) = GMprObs(ii, :) * 100.
        GMprObs@units = "%"
      else
        GMprObs@units = gObs[ii]@units
      end if

      if (any((/isStrSubset(gObs[ii]@dataset, "GPCP"), \
              isStrSubset(gObs[ii]@dataset, "CMAP"), \
              isStrSubset(gObs[ii]@dataset, "TRMM")/))) then
        colors(ii) = colors_satellite(icolor_satellite)
        patterns(ii) = patterns_satellite(icolor_satellite)
        seglens(ii)  = seglens_satellite(icolor_satellite)
        icolor_satellite = icolor_satellite+1
      else
        colors(ii) = colors_insitu(icolor_insitu)
        patterns(ii) = patterns_insitu(icolor_insitu)
        seglens(ii)  = seglens_insitu(icolor_insitu)
        icolor_insitu = icolor_insitu+1
      end if
      if (ii.eq.0) then
        GMprObs@dataset := (/ gObs[ii]@dataset /)
      else
        GMprObs@dataset := array_append_record(GMprObs@dataset, \
                                               gObs[ii]@dataset, 0)
      end if
    end do
    delete(gObs)

    syear = 999999
    eyear = -999999
    nruns = ListCount(gCMIP6)
    if (nruns.gt.0) then
      do ii = 0, nruns-1
        syear = min((/toint(gCMIP6[ii]&time(0)), syear/))
        eyear = max((/toint(gCMIP6[ii]&time(dimsizes(gCMIP6[ii]&time)-1)), eyear/))
      end do
      GMprCMIP6 = new((/ nruns, eyear-syear+1 /), "float")
      GMprCMIP6!0 = "ensemble"
      GMprCMIP6!1 = "time"
      GMprCMIP6&ensemble = ispan(1, nruns, 1)
      GMprCMIP6&time = ispan(syear, eyear, 1)
      esizeGMprCMIP6 = new(nruns, "integer")
      esizeGMprCMIP6!0 = "ensemble"
      esizeGMprCMIP6&ensemble = GMprCMIP6&ensemble
      do ii = 0, nruns-1
        syr = gCMIP6[ii]&time(0)
        eyr = gCMIP6[ii]&time(dimsizes(gCMIP6[ii]&time)-1)
        GMprCMIP6(ii, {syr:eyr}) = (/ gCMIP6[ii] /)
        esizeGMprCMIP6 = num(metadata_att_as_array(gCMIP6, "dataset").eq.gCMIP6[ii]@dataset)
      end do
      if (gCMIP6[0]@units.eq."") then
        GMprCMIP6 = GMprCMIP6 * 100.
        GMprCMIP6@units = "%"
      else
        GMprCMIP6@units = gCMIP6[0]@units
      end if
      
      GMprCMIP6mean  = GMprCMIP6(0, :)
      GMprCMIP6_5th  = GMprCMIP6mean
      GMprCMIP6_95th = GMprCMIP6mean
      GMprCMIP6mean  = (/weighted_mean(GMprCMIP6, 1./tofloat(esizeGMprCMIP6))/)
      GMprCMIP6_5th  = (/weighted_percentile(GMprCMIP6, 1./tofloat(esizeGMprCMIP6), 0.05)/)
      GMprCMIP6_95th = (/weighted_percentile(GMprCMIP6, 1./tofloat(esizeGMprCMIP6), 0.95)/)

      nruns_CMIP6_pr = nruns
      nmodels_CMIP6_pr = count_unique_values(metadata_att_as_array(gCMIP6, "dataset"))
      delete(gCMIP6)
      delete(esizeGMprCMIP6)
    end if

    syear = 999999
    eyear = -999999
    nruns = ListCount(gCMIP5)
    if (nruns.gt.0) then
      do ii = 0, nruns-1
        syear = min((/toint(gCMIP5[ii]&time(0)), syear/))
        eyear = max((/toint(gCMIP5[ii]&time(dimsizes(gCMIP5[ii]&time)-1)), eyear/))
      end do
      GMprCMIP5 = new((/ nruns, eyear-syear+1 /), "float")
      GMprCMIP5!0 = "ensemble"
      GMprCMIP5!1 = "time"
      GMprCMIP5&ensemble = ispan(1, nruns, 1)
      GMprCMIP5&time = ispan(syear, eyear, 1)
      esizeGMprCMIP5 = new(nruns, "integer")
      esizeGMprCMIP5!0 = "ensemble"
      esizeGMprCMIP5&ensemble = GMprCMIP5&ensemble
      do ii = 0, nruns-1
        syr = gCMIP5[ii]&time(0)
        eyr = gCMIP5[ii]&time(dimsizes(gCMIP5[ii]&time)-1)
        GMprCMIP5(ii, {syr:eyr}) = (/ gCMIP5[ii] /)
        esizeGMprCMIP5 = num(metadata_att_as_array(gCMIP5, "dataset").eq.gCMIP5[ii]@dataset)
      end do
      if (gCMIP5[0]@units.eq."") then
        GMprCMIP5 = GMprCMIP5 * 100.
        GMprCMIP5@units = "%"
      else
        GMprCMIP5@units = gCMIP5[0]@units
      end if
      
      GMprCMIP5mean  = GMprCMIP5(0, :)
      GMprCMIP5_5th  = GMprCMIP5mean
      GMprCMIP5_95th = GMprCMIP5mean
      GMprCMIP5mean  = (/weighted_mean(GMprCMIP5, 1./tofloat(esizeGMprCMIP5))/)
      GMprCMIP5_5th  = (/weighted_percentile(GMprCMIP5, 1./tofloat(esizeGMprCMIP5), 0.05)/)
      GMprCMIP5_95th = (/weighted_percentile(GMprCMIP5, 1./tofloat(esizeGMprCMIP5), 0.95)/)

      nruns_CMIP5_pr = nruns
      nmodels_CMIP5_pr = count_unique_values(metadata_att_as_array(gCMIP5, "dataset"))
      delete(gCMIP5)
      delete(esizeGMprCMIP5)
    end if

    syear = 999999
    eyear = -999999
    nruns = ListCount(gAMIP)
    if (nruns.gt.0) then
      do ii = 0, nruns-1
        syear = min((/toint(gAMIP[ii]&time(0)), syear/))
        eyear = max((/toint(gAMIP[ii]&time(dimsizes(gAMIP[ii]&time)-1)), eyear/))
      end do
      GMprAMIP = new((/ nruns, eyear-syear+1 /), "float")
      GMprAMIP!0 = "ensemble"
      GMprAMIP!1 = "time"
      GMprAMIP&ensemble = ispan(1, nruns, 1)
      GMprAMIP&time = ispan(syear, eyear, 1)
      esizeGMprAMIP = new(nruns, "integer")
      esizeGMprAMIP!0 = "ensemble"
      esizeGMprAMIP&ensemble = GMprAMIP&ensemble
      do ii = 0, nruns-1
        syr = gAMIP[ii]&time(0)
        eyr = gAMIP[ii]&time(dimsizes(gAMIP[ii]&time)-1)
        GMprAMIP(ii, {syr:eyr}) = (/ gAMIP[ii] /)
        esizeGMprAMIP = num(metadata_att_as_array(gAMIP, "dataset").eq.gAMIP[ii]@dataset)
      end do
      GMprAMIP@units = gAMIP[0]@units
      if (gAMIP[0]@units.eq."") then
        GMprAMIP = GMprAMIP * 100.
        GMprAMIP@units = "%"
      else
        GMprAMIP@units = gAMIP[0]@units
      end if
      
      GMprAMIPmean  = GMprAMIP(0, :)
      GMprAMIP_5th  = GMprAMIPmean
      GMprAMIP_95th = GMprAMIPmean
      GMprAMIPmean  = (/weighted_mean(GMprAMIP, 1./tofloat(esizeGMprAMIP))/)
      GMprAMIP_5th  = (/weighted_percentile(GMprAMIP, 1./tofloat(esizeGMprAMIP), 0.05)/)
      GMprAMIP_95th = (/weighted_percentile(GMprAMIP, 1./tofloat(esizeGMprAMIP), 0.95)/)

      nruns_AMIP_pr = nruns
      nmodels_AMIP_pr = count_unique_values(metadata_att_as_array(gAMIP, "dataset"))
      delete(gAMIP)
      delete(esizeGMprAMIP)
    end if
  end if

  if (drawCirc) then
    if (dimsizes(input_dir_circulation).gt.1) then
      paths = ""
      do n = 0, dimsizes(input_dir_circulation)-1
        paths = paths+input_dir_circulation(n)+"NHMcirculation_summer_*.nc "
      end do
    else
      paths = input_dir_circulation+"NHMcirculation_summer_*.nc"
    end if
    NHMufiles = systemfunc("ls "+paths)

    if (all(ismissing(NHMufiles))) then
      drawCirc = False
    else
      do ii = 0, dimsizes(NHMufiles)-1
        f = addfile(NHMufiles(ii), "r")
        tmp = f->NHMu
        if (moving_ave.gt.1) then
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
        elseif (tmp@project.eq."CMIP6".and.isStrSubset(tmp@experiment_id, "historical")) then
          if (.not.isdefined("uCMIP6")) then
            uCMIP6 = NewList("fifo")
          end if
          ListAppend(uCMIP6, new(dimsizes(tmp), typeof(tmp)))
          i = ListCount(uCMIP6)-1
          uCMIP6[i] = tmp
        elseif (tmp@project.eq."CMIP5".and.isStrSubset(tmp@experiment_id, "historical")) then
          if (.not.isdefined("uCMIP5")) then
            uCMIP5 = NewList("fifo")
          end if
          ListAppend(uCMIP5, new(dimsizes(tmp), typeof(tmp)))
          i = ListCount(uCMIP5)-1
          uCMIP5[i] = tmp
        elseif (tmp@experiment_id.eq."amip-hist") then
          if (.not.isdefined("uAMIP")) then
            uAMIP = NewList("fifo")
          end if
          ListAppend(uAMIP, new(dimsizes(tmp), typeof(tmp)))
          i = ListCount(uAMIP)-1
          uAMIP[i] = tmp
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
      NHMuObs!0 = "dset"
      NHMuObs!1 = "time"
      NHMuObs&dset = ispan(1, nobsu, 1)
      NHMuObs&time = ispan(syear, eyear, 1)

      icolor_reanal = 0
      colorsu   = new(nobsu, "string")
      patternsu = new(nobsu, "integer")
      seglensu  = new(nobsu, "float")
      do ii = 0, nobsu-1
        syr = uObs[ii]&time(0)
        eyr = uObs[ii]&time(dimsizes(uObs[ii]&time)-1)
        NHMuObs(ii, {syr:eyr}) = (/ uObs[ii] /)

        colorsu(ii)   = colors_reanal(icolor_reanal)
        patternsu(ii) = patterns_reanal(icolor_reanal)
        seglensu(ii)  = seglens_reanal(icolor_reanal)
        icolor_reanal = icolor_reanal+1
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
      nruns = ListCount(uCMIP6)
      if (nruns.gt.0) then
        do ii = 0, nruns-1
          syear = min((/toint(uCMIP6[ii]&time(0)), syear/))
          eyear = max((/toint(uCMIP6[ii]&time(dimsizes(uCMIP6[ii]&time)-1)), \
                      eyear/))
        end do
        NHMuCMIP6 = new((/ nruns, eyear-syear+1 /), "float")
        NHMuCMIP6!0 = "ensemble"
        NHMuCMIP6!1 = "time"
        NHMuCMIP6&ensemble = ispan(1, nruns, 1)
        NHMuCMIP6&time = ispan(syear, eyear, 1)
        esizeNHMuCMIP6 = new(nruns, "integer")
        esizeNHMuCMIP6!0 = "ensemble"
        esizeNHMuCMIP6&ensemble = NHMuCMIP6&ensemble
        do ii = 0, nruns-1
          syr = uCMIP6[ii]&time(0)
          eyr = uCMIP6[ii]&time(dimsizes(uCMIP6[ii]&time)-1)
          NHMuCMIP6(ii, {syr:eyr}) = (/ uCMIP6[ii] /)
          esizeNHMuCMIP6 = num(metadata_att_as_array(uCMIP6, "dataset").eq.uCMIP6[ii]@dataset)
        end do

        NHMuCMIP6mean  = NHMuCMIP6(0, :)
        NHMuCMIP6_5th  = NHMuCMIP6mean
        NHMuCMIP6_95th = NHMuCMIP6mean
        NHMuCMIP6mean  = (/weighted_mean(NHMuCMIP6, 1./tofloat(esizeNHMuCMIP6))/)
        NHMuCMIP6_5th  = (/weighted_percentile(NHMuCMIP6, 1./tofloat(esizeNHMuCMIP6), 0.05)/)
        NHMuCMIP6_95th = (/weighted_percentile(NHMuCMIP6, 1./tofloat(esizeNHMuCMIP6), 0.95)/)

        nruns_CMIP6_u = nruns
        nmodels_CMIP6_u = count_unique_values(metadata_att_as_array(uCMIP6, "dataset"))
        delete(uCMIP6)
        delete(esizeNHMuCMIP6)
      end if

      syear = 999999
      eyear = -999999
      nruns = ListCount(uCMIP5)
      if (nruns.gt.0) then
        do ii = 0, nruns-1
          syear = min((/toint(uCMIP5[ii]&time(0)), syear/))
          eyear = max((/toint(uCMIP5[ii]&time(dimsizes(uCMIP5[ii]&time)-1)), \
                      eyear/))
        end do
        NHMuCMIP5 = new((/ nruns, eyear-syear+1 /), "float")
        NHMuCMIP5!0 = "ensemble"
        NHMuCMIP5!1 = "time"
        NHMuCMIP5&ensemble = ispan(1, nruns, 1)
        NHMuCMIP5&time = ispan(syear, eyear, 1)
        esizeNHMuCMIP5 = new(nruns, "integer")
        esizeNHMuCMIP5!0 = "ensemble"
        esizeNHMuCMIP5&ensemble = NHMuCMIP5&ensemble
        do ii = 0, nruns-1
          syr = uCMIP5[ii]&time(0)
          eyr = uCMIP5[ii]&time(dimsizes(uCMIP5[ii]&time)-1)
          NHMuCMIP5(ii, {syr:eyr}) = (/ uCMIP5[ii] /)
          esizeNHMuCMIP5 = num(metadata_att_as_array(uCMIP5, "dataset").eq.uCMIP5[ii]@dataset)
        end do

        NHMuCMIP5mean  = NHMuCMIP5(0, :)
        NHMuCMIP5_5th  = NHMuCMIP5mean
        NHMuCMIP5_95th = NHMuCMIP5mean
        NHMuCMIP5mean  = (/weighted_mean(NHMuCMIP5, 1./tofloat(esizeNHMuCMIP5))/)
        NHMuCMIP5_5th  = (/weighted_percentile(NHMuCMIP5, 1./tofloat(esizeNHMuCMIP5), 0.05)/)
        NHMuCMIP5_95th = (/weighted_percentile(NHMuCMIP5, 1./tofloat(esizeNHMuCMIP5), 0.95)/)

        nruns_CMIP5_u = nruns
        nmodels_CMIP5_u = count_unique_values(metadata_att_as_array(uCMIP5, "dataset"))
        delete(uCMIP5)
        delete(esizeNHMuCMIP5)
      end if

      syear = 999999
      eyear = -999999
      nruns = ListCount(uAMIP)
      if (nruns.gt.0) then
        do ii = 0, nruns-1
          syear = min((/toint(uAMIP[ii]&time(0)), syear/))
          eyear = max((/toint(uAMIP[ii]&time(dimsizes(uAMIP[ii]&time)-1)), \
                      eyear/))
        end do
        NHMuAMIP = new((/ nruns, eyear-syear+1 /), "float")
        NHMuAMIP!0 = "ensemble"
        NHMuAMIP!1 = "time"
        NHMuAMIP&ensemble = ispan(1, nruns, 1)
        NHMuAMIP&time = ispan(syear, eyear, 1)
        esizeNHMuAMIP = new(nruns, "integer")
        esizeNHMuAMIP!0 = "ensemble"
        esizeNHMuAMIP&ensemble = NHMuAMIP&ensemble
        do ii = 0, nruns-1
          syr = uAMIP[ii]&time(0)
          eyr = uAMIP[ii]&time(dimsizes(uAMIP[ii]&time)-1)
          NHMuAMIP(ii, {syr:eyr}) = (/ uAMIP[ii] /)
          esizeNHMuAMIP = num(metadata_att_as_array(uAMIP, "dataset").eq.uAMIP[ii]@dataset)
        end do

        NHMuAMIPmean  = NHMuAMIP(0, :)
        NHMuAMIP_5th  = NHMuAMIPmean
        NHMuAMIP_95th = NHMuAMIPmean
        NHMuAMIPmean  = (/weighted_mean(NHMuAMIP, 1./tofloat(esizeNHMuAMIP))/)
        NHMuAMIP_5th  = (/weighted_percentile(NHMuAMIP, 1./tofloat(esizeNHMuAMIP), 0.05)/)
        NHMuAMIP_95th = (/weighted_percentile(NHMuAMIP, 1./tofloat(esizeNHMuAMIP), 0.95)/)

        nruns_AMIP_u = nruns
        nmodels_AMIP_u = count_unique_values(metadata_att_as_array(uAMIP, "dataset"))
        delete(uAMIP)
        delete(esizeNHMuAMIP)
      end if
    end if
  end if

  plotts = new(2, "graphic")
  dummy = new(40, "graphic")
  idummy = 0
  
  rests := True
  rests@xyDashPattern     = 0
  rests@xyMonoLineColor   = False
  rests@gsnYRefLine       = 0.
  rests@tmYLLabelFontHeightF = 0.015
  rests@tmXBLabelFontHeightF = 0.015
  rests@tmYLLabelDeltaF   = -0.3
  rests@tmXBLabelDeltaF   = -0.3
  rests@gsnDraw           = False
  rests@gsnFrame          = False

  rests@trXMinF = 1880
  rests@trXMaxF = 2020
  rests@tmXTOn = False
  rests@tmYROn = False
  rests@vpWidthF = 0.6
  rests@vpHeightF = 0.2
  rests@tiMainFontHeightF = 0.015
  rests@tiMainOffsetYF = -0.008
  rests@gsnStringFontHeightF = 0.018
  
  txres := True

  rests@xyLineThicknessF = 0.5
  rests@xyLineColor      = (/204., 35., 35./)/255.  ; (/37, 81., 204./)/255.
  rests@xyLineOpacityF   = 0.5

  resp := True
;  resp@tfPolyDrawOrder = "Draw"
  resp@gsFillOpacityF = 0.1

  resl := True
;  resl@tfPolyDrawOrder = "Draw"

  x2 = 0.57
  x3 = 0.61
  xc = (x2 + x3)/2.
  wLegend = 0.065
  iLegend = 0.02
  yLegend1 = 0.23
  yLegend2 = 0.26
  yLegendC = (yLegend1 + yLegend2)/2.
  yLegend  = yLegendC - 0.004
  done_CMIP5legend = False
  done_CMIP6legend = False
  done_AMIPlegend = False
  if (drawPr) then
    range = max(abs(GMprObs))
    if (isvar("GMprCMIP6mean")) then
      range = max((/range, max(abs(GMprCMIP6_5th))/))
      range = max((/range, max(abs(GMprCMIP6_95th))/))
      pr_units = GMprCMIP6mean@units
    end if
    if (isvar("GMprCMIP5mean")) then
      range = max((/range, max(abs(GMprCMIP5_5th))/))
      range = max((/range, max(abs(GMprCMIP5_95th))/))
      pr_units = GMprCMIP5mean@units
    end if
    if (isvar("GMprAMIPmean")) then
      range = max((/range, max(abs(GMprAMIP_5th))/))
      range = max((/range, max(abs(GMprAMIP_95th))/))
      pr_units = GMprAMIPmean@units
    end if
    ranges := (/2., 3., 4., 5., 6., 7., 8., 10., 12., 15., 20., 30., 40., 50., 60., 70., 80., 100./)
    rests@trYMaxF = min(where(ranges.lt.range, 9999., ranges))
    rests@trYMinF = -rests@trYMaxF
    rests@gsnCenterString = "Global summer monsoon precipitation"
    if (dimsizes(input_dir_precip).eq.1) then
      if (isStrSubset(str_lower(input_dir_precip), "land")) then
        rests@gsnCenterString = "Global land summer monsoon precipitation"
      end if
    else
      do n = 0, dimsizes(input_dir_precip)-1
        if (isStrSubset(str_lower(input_dir_precip(n)), "land")) then
          rests@gsnCenterString = "Global land summer monsoon precipitation"
          break
        end if
      end do
    end if
    if (rests@trYMaxF.eq.tofloat(toint(rests@trYMaxF))) then
      rests@gsnCenterStringOrthogonalPosF = 0.02
    else
      rests@gsnCenterStringOrthogonalPosF = 0.06
    end if
    rests@gsnLeftStringOrthogonalPosF = rests@gsnCenterStringOrthogonalPosF  ; - 0.02
    rests@gsnLeftString = "(c)"
    if (pr_units.eq."%") then
      rests@tiYAxisString = "(%)"
    else
      rests@tiYAxisString = "(mm day~S~-1~N~)"
    end if
    
    plotts(0) = gsn_csm_xy(wks, (/rests@trXMinF, rests@trXMaxF/), \
                           (/GMprCMIP6mean@_FillValue, GMprCMIP6mean@_FillValue/), rests)
    if (isvar("syear_clim").and.isvar("eyear_clim")) then
      x := (/syear_clim, syear_clim, eyear_clim, eyear_clim/)
      y := (/rests@trYMinF, rests@trYMaxF, rests@trYMaxF, rests@trYMinF/)
      resp@gsFillColor := "grey75"
      dummy(idummy) = gsn_add_polygon(wks, plotts(0), x, y, resp)
      idummy = idummy + 1
    end if
                           
    if (isvar("GMprCMIP5mean")) then
      x := array_append_record(GMprCMIP5_5th&time, GMprCMIP5_95th&time(::-1), 0)
      y := array_append_record(GMprCMIP5_5th, GMprCMIP5_95th(::-1), 0)
      resp@gsFillColor := color_CMIP5
      dummy(idummy) = gsn_add_polygon(wks, plotts(0), x, y, resp)
      idummy = idummy + 1

      resl@gsLineColor := resp@gsFillColor
      resl@gsLineThicknessF = 2.5
      resl@gsLineDashPattern = 0
      dummy(idummy) = gsn_add_polyline(wks, plotts(0), GMprCMIP5mean&time, GMprCMIP5mean, resl)
      idummy = idummy + 1

      x4 = x3 + wLegend
      gsn_polygon_ndc(wks, (/x3,x3,x4,x4/), (/yLegend1,yLegend2,yLegend2,yLegend1/), resp)
      gsn_polyline_ndc(wks, (/x3, x4/), (/yLegendC, yLegendC/), resl)
      txres@txJust = "TopCenter"
      txres@txFontColor := resp@gsFillColor
      txres@txFontHeightF = 0.012
      gsn_text_ndc(wks, "CMIP5", (x3+x4)/2., yLegend1-0.004, txres)
      if (all((/nruns_CMIP5_pr, nmodels_CMIP5_pr/).eq.(/nruns_CMIP5_u, nmodels_CMIP5_u/))) then
        txres@txFontHeightF = 0.008
        gsn_text_ndc(wks, nmodels_CMIP5_pr+"models, "+nruns_CMIP5_pr+"runs", (x3+x4)/2., yLegend1-0.02, txres)
      end if
      done_CMIP5legend = True
      x3 = x4 + iLegend
    end if

    if (isvar("GMprCMIP6mean")) then
      x := array_append_record(GMprCMIP6_5th&time, GMprCMIP6_95th&time(::-1), 0)
      y := array_append_record(GMprCMIP6_5th, GMprCMIP6_95th(::-1), 0)
      resp@gsFillColor := color_CMIP6
      dummy(idummy) = gsn_add_polygon(wks, plotts(0), x, y, resp)
      idummy = idummy + 1

      resl@gsLineColor := resp@gsFillColor
      resl@gsLineThicknessF = 2.5
      resl@gsLineDashPattern = 0
      dummy(idummy) = gsn_add_polyline(wks, plotts(0), GMprCMIP6mean&time, GMprCMIP6mean, resl)
      idummy = idummy + 1

      x1 = x2 - wLegend
      gsn_polygon_ndc(wks, (/x1,x1,x2,x2/), (/yLegend1,yLegend2,yLegend2,yLegend1/), resp)
      gsn_polyline_ndc(wks, (/x1, x2/), (/yLegendC, yLegendC/), resl)
      txres@txJust = "TopCenter"
      txres@txFontColor := resp@gsFillColor
      txres@txFontHeightF = 0.012
      gsn_text_ndc(wks, "CMIP6", (x1+x2)/2., yLegend1-0.004, txres)
      if (all((/nruns_CMIP6_pr, nmodels_CMIP6_pr/).eq.(/nruns_CMIP6_u, nmodels_CMIP6_u/))) then
        txres@txFontHeightF = 0.008
        gsn_text_ndc(wks, nmodels_CMIP6_pr+"models, "+nruns_CMIP6_pr+"runs", (x1+x2)/2., yLegend1-0.02, txres)
      end if
      txres@txFontColor := "black"
      txres@txFontHeightF = 0.01
;      txres@txJust = "CenterLeft"
;      gsn_text_ndc(wks, "95th", x2+0.004, yLegend2, txres)
;      gsn_text_ndc(wks, "mean", x2+0.004, yLegendC, txres)
;      gsn_text_ndc(wks, "5th",  x2+0.004, yLegend1, txres)
      txres@txJust = "CenterCenter"
      gsn_text_ndc(wks, "95th", xc, yLegend2, txres)
      gsn_text_ndc(wks, "mean", xc, yLegendC, txres)
      gsn_text_ndc(wks, "5th",  xc, yLegend1, txres)
      done_CMIP6legend = True
      x2 = x1 - iLegend
    end if

    if (isvar("GMprAMIPmean")) then
      x := array_append_record(GMprAMIP_5th&time, GMprAMIP_95th&time(::-1), 0)
      y := array_append_record(GMprAMIP_5th, GMprAMIP_95th(::-1), 0)
      resp@gsFillColor := color_AMIP
      dummy(idummy) = gsn_add_polygon(wks, plotts(0), x, y, resp)
      idummy = idummy + 1

      resl@gsLineColor := resp@gsFillColor
      resl@gsLineThicknessF = 2.5
      resl@gsLineDashPattern = 0
      dummy(idummy) = gsn_add_polyline(wks, plotts(0), GMprAMIPmean&time, GMprAMIPmean, resl)
      idummy = idummy + 1

;      x4 = x3 + wLegend
;      gsn_polygon_ndc(wks, (/x3,x3,x4,x4/), (/yLegend1,yLegend2,yLegend2,yLegend1/), resp)
;      gsn_polyline_ndc(wks, (/x3, x4/), (/yLegendC, yLegendC/), resl)
;      txres@txJust = "TopCenter"
;      txres@txFontColor := resp@gsFillColor
;      txres@txFontHeightF = 0.012
;      gsn_text_ndc(wks, "AMIP", (x3+x4)/2., yLegend1-0.004, txres)
;      if (all((/nruns_AMIP_pr, nmodels_AMIP_pr/).eq.(/nruns_AMIP_u, nmodels_AMIP_u/))) then
;        txres@txFontHeightF = 0.008
;        gsn_text_ndc(wks, nmodels_AMIP_pr+"models, "+nruns_AMIP_pr+"runs", (x3+x4)/2., yLegend1-0.02, txres)
;      end if
;      done_AMIPlegend = True
;      x3 = x4 + iLegend

      x1 = x2 - wLegend - 0.01
      x2 = x2 - 0.01
      gsn_polygon_ndc(wks, (/x1,x1,x2,x2/), (/yLegend1,yLegend2,yLegend2,yLegend1/), resp)
      gsn_polyline_ndc(wks, (/x1, x2/), (/yLegendC, yLegendC/), resl)
      txres@txJust = "TopCenter"
      txres@txFontColor := resp@gsFillColor
      txres@txFontHeightF = 0.012
      gsn_text_ndc(wks, "AMIP", (x1+x2)/2., yLegend1-0.004, txres)
      if (all((/nruns_AMIP_pr, nmodels_AMIP_pr/).eq.(/nruns_AMIP_u, nmodels_AMIP_u/))) then
        txres@txFontHeightF = 0.008
        gsn_text_ndc(wks, nmodels_AMIP_pr+"models, "+nruns_AMIP_pr+"runs", (x1+x2)/2., yLegend1-0.02, txres)
      end if
      done_AMIPlegend = True
      x2 = x1 - iLegend
    end if

    if (isvar("GMprObs")) then
      resl@gsLineThicknessF = 1.5
      do n = 0, nobs-1
        resl@gsLineColor := colors(n)
        resl@gsLineDashPattern = patterns(n)
        resl@gsLineDashSegLenF = seglens(n)
        dummy(idummy) = gsn_add_polyline(wks, plotts(0), GMprObs&time, GMprObs(n, :), resl)
        idummy = idummy + 1

        x1 = x2 - wLegend
        gsn_polyline_ndc(wks, (/x1, x2/), (/yLegendC, yLegendC/), resl)
        txres@txJust = "TopCenter"
        txres@txFontColor := resl@gsLineColor
        txres@txFontHeightF = 0.012
        if (GMprObs@dataset(n).eq."GPCP-SG") then
          gsn_text_ndc(wks, "GPCP", (x1+x2)/2., yLegend, txres)
        else
          gsn_text_ndc(wks, GMprObs@dataset(n), (x1+x2)/2., yLegend, txres)
        end if
        x2 = x1 - iLegend
        delete(resl@gsLineDashSegLenF)
      end do
    end if
  end if

  if (drawCirc) then
    restsu = rests
;    restsu@tmYLPrecision = 1
;    restsu@tmYLLabelJust = "CenterCenter"
    range = max(abs(NHMuObs))
    if (isvar("NHMuCMIP6mean")) then
      range = max((/range, max(abs(NHMuCMIP6_5th))/))
      range = max((/range, max(abs(NHMuCMIP6_95th))/))
    end if
    if (isvar("NHMuCMIP5mean")) then
      range = max((/range, max(abs(NHMuCMIP5_5th))/))
      range = max((/range, max(abs(NHMuCMIP5_95th))/))
    end if
    if (isvar("NHMuAMIPmean")) then
      range = max((/range, max(abs(NHMuAMIP_5th))/))
      range = max((/range, max(abs(NHMuAMIP_95th))/))
    end if
    
    ranges := (/1.5, 1.8, 2., 2.2, 2.4, 2.6, 2.8, 3., 3.5, 4., 6., 8., 10., 15., 20./)
    restsu@trYMaxF = min(where(ranges.lt.range, 9999., ranges))
    restsu@trYMinF = -restsu@trYMaxF
    restsu@gsnCenterString = "NH summer monsoon circulation"
    if (restsu@trYMaxF.eq.tofloat(toint(restsu@trYMaxF))) then
      restsu@gsnCenterStringOrthogonalPosF = 0.02
    else
      restsu@gsnCenterStringOrthogonalPosF = 0.06
    end if
    restsu@gsnLeftString = "(d)"
    restsu@gsnLeftStringOrthogonalPosF = restsu@gsnCenterStringOrthogonalPosF - 0.02
    restsu@tiYAxisString = "(m s~S~-1~N~)"
    plotts(1) = gsn_csm_xy(wks, (/rests@trXMinF, rests@trXMaxF/), (/NHMuCMIP6mean@_FillValue, NHMuCMIP6mean@_FillValue/), restsu)

    if (isvar("syear_clim").and.isvar("eyear_clim")) then
      x := (/syear_clim, syear_clim, eyear_clim, eyear_clim/)
      y := (/rests@trYMinF, rests@trYMaxF, rests@trYMaxF, rests@trYMinF/)
      resp@gsFillColor := "grey75"
      dummy(idummy) = gsn_add_polygon(wks, plotts(1), x, y, resp)
      idummy = idummy + 1
    end if
    
    if (isvar("NHMuCMIP5mean")) then
      x := array_append_record(NHMuCMIP5_5th&time, NHMuCMIP5_95th&time(::-1), 0)
      y := array_append_record(NHMuCMIP5_5th, NHMuCMIP5_95th(::-1), 0)
      resp@gsFillColor := color_CMIP5
      dummy(idummy) = gsn_add_polygon(wks, plotts(1), x, y, resp)
      idummy = idummy + 1
    
      resl@gsLineColor := resp@gsFillColor
      resl@gsLineThicknessF = 2.5
      resl@gsLineDashPattern = 0
      dummy(idummy) = gsn_add_polyline(wks, plotts(1), NHMuCMIP5mean&time, NHMuCMIP5mean, resl)
      idummy = idummy + 1

      if (.not.done_CMIP5legend) then
        x4 = x3 + wLegend
        gsn_polygon_ndc(wks, (/x1,x1,x2,x2/), (/yLegend1,yLegend2,yLegend2,yLegend1/), resp)
        gsn_polyline_ndc(wks, (/x1, x2/), (/yLegendC, yLegendC/), resl)
        txres@txJust = "TopCenter"
        txres@txFontColor := resp@gsFillColor
        txres@txFontHeightF = 0.012
        gsn_text_ndc(wks, "CMIP5", (x1+x2)/2., yLegend1-0.004, txres)
        done_CMIP5legend = True
        x3 = x4 + iLegend
      end if
    end if
    if (isvar("NHMuCMIP6mean")) then
      x := array_append_record(NHMuCMIP6_5th&time, NHMuCMIP6_95th&time(::-1), 0)
      y := array_append_record(NHMuCMIP6_5th, NHMuCMIP6_95th(::-1), 0)
      resp@gsFillColor := color_CMIP6 
      dummy(idummy) = gsn_add_polygon(wks, plotts(1), x, y, resp)
      idummy = idummy + 1
    
      resl@gsLineColor := resp@gsFillColor
      resl@gsLineThicknessF = 2.5
      resl@gsLineDashPattern = 0
      dummy(idummy) = gsn_add_polyline(wks, plotts(1), NHMuCMIP6mean&time, NHMuCMIP6mean, resl)
      idummy = idummy + 1
      
      if (.not.done_CMIP6legend) then
        x1 = x2 - wLegend
        gsn_polygon_ndc(wks, (/x1,x1,x2,x2/), (/yLegend1,yLegend2,yLegend2,yLegend1/), resp)
        gsn_polyline_ndc(wks, (/x1, x2/), (/yLegendC, yLegendC/), resl)
        txres@txJust = "TopCenter"
        txres@txFontColor := resp@gsFillColor
        txres@txFontHeightF = 0.012
        gsn_text_ndc(wks, "CMIP6", (x1+x2)/2., yLegend1-0.004, txres)
        txres@txFontColor := "black"
        txres@txFontHeightF = 0.01
;        txres@txJust = "CenterLeft"
;        gsn_text_ndc(wks, "95th", x2+0.004, yLegend2, txres)
;        gsn_text_ndc(wks, "mean", x2+0.004, yLegendC, txres)
;        gsn_text_ndc(wks, "5th",  x2+0.004, yLegend1, txres)
        txres@txJust = "CenterCenter"
        gsn_text_ndc(wks, "95th", xc, yLegend2, txres)
        gsn_text_ndc(wks, "mean", xc, yLegendC, txres)
        gsn_text_ndc(wks, "5th",  xc, yLegend1, txres)
        done_CMIP6legend = True
        x2 = x1 - iLegend
      end if
    end if
    if (isvar("NHMuAMIPmean")) then
      x := array_append_record(NHMuAMIP_5th&time, NHMuAMIP_95th&time(::-1), 0)
      y := array_append_record(NHMuAMIP_5th, NHMuAMIP_95th(::-1), 0)
      resp@gsFillColor := color_AMIP
      dummy(idummy) = gsn_add_polygon(wks, plotts(1), x, y, resp)
      idummy = idummy + 1
    
      resl@gsLineColor := resp@gsFillColor
      resl@gsLineThicknessF = 2.5
      resl@gsLineDashPattern = 0
      dummy(idummy) = gsn_add_polyline(wks, plotts(1), NHMuAMIPmean&time, NHMuAMIPmean, resl)
      idummy = idummy + 1
      
      if (.not.done_AMIPlegend) then
        x4 = x3 + wLegend
        gsn_polygon_ndc(wks, (/x3,x3,x4,x4/), (/yLegend1,yLegend2,yLegend2,yLegend1/), resp)
        gsn_polyline_ndc(wks, (/x3, x4/), (/yLegendC, yLegendC/), resl)
        txres@txJust = "TopCenter"
        txres@txFontColor := resp@gsFillColor
        txres@txFontHeightF = 0.012
        gsn_text_ndc(wks, "AMIP", (x3+x4)/2., yLegend1-0.004, txres)
        txres@txFontHeightF = 0.008
        gsn_text_ndc(wks, nmodels_AMIP_pr+"models, "+nruns_AMIP_pr+"runs", (x1+x2)/2., yLegend1-0.02, txres)
        done_AMIPlegend = True
        x3 = x4 + iLegend
      end if
    end if
    if (isvar("NHMuObs")) then
      resl@gsLineThicknessF = 1.5
      do n = 0, nobsu-1
        resl@gsLineColor := colorsu(n)
        resl@gsLineDashPattern = patternsu(n)
        resl@gsLineDashSegLenF = seglensu(n)
        dummy(idummy) = gsn_add_polyline(wks, plotts(1), NHMuObs&time, NHMuObs(n, :), resl)
        idummy = idummy + 1
        
        x4 = x3 + wLegend
        gsn_polyline_ndc(wks, (/x3, x4/), (/yLegendC, yLegendC/), resl)
        txres@txJust = "TopCenter"
        txres@txFontColor := resl@gsLineColor
        txres@txFontHeightF = 0.012
        gsn_text_ndc(wks, NHMuObs@dataset(n), (x3+x4)/2., yLegend, txres)
        x3 = x4 + iLegend
        delete(resl@gsLineDashSegLenF)
      end do
    end if
  end if

  pres                  := True
  pres@gsnFrame          = False
  pres@gsnPanelTop       = 0.63
  pres@gsnPanelBottom    = 0.1
  pres@gsnPanelLabelBar  = False
  pres@gsnMaximize       = True
;  pres@gsnPanelDebug = True
;  pres@gsnPanelXF        = (/0.04, 0.54/)
  pres@pmLabelBarOrthogonalPosF = 0.025
  gsn_panel(wks, plotts, (/1, 2/), pres)
  frame(wks)

end
