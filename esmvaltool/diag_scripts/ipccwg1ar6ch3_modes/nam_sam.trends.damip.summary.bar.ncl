; This script plots results from nam_sam.piControl.ncwrite.ncl, nam_sam.piControl.alt_def.ncwrite.ncl,
; nam_sam.damip.ncwrite.ncl and nam_sam.damip.alt_def.ncwrite.ncl. ALl four of those scripts
; must be called multiple times to create the 28 files required by this script.
;
;----Changes required to work on a different platform------
;
; - Line 39 update to directory path of local files created by 4 scripts cited above.
; - Line 40 update to path of NAM/SAM observational DJF file (created in nam_sam.damip.ncwrite.ncl)
; - Line 43 update to path of NAM/SAM observational JJA file (created in nam_sam.damip.ncwrite.ncl)
; - Line 47 update to path of NAM/SAM piControl DJF file (created in nam_sam.piControl.ncwrite.ncl)
; - Line 51 update to path of NAM/SAM piControl JJA file (created in nam_sam.piControl.ncwrite.ncl)
; - Line 55 update to path of NAM/SAM historical DJF file (created in nam_sam.damip.ncwrite.ncl)
; - Line 58 update to path of NAM/SAM historical JJA file (created in nam_sam.damip.ncwrite.ncl)
; - Line 62 update to path of NAM/SAM hist-stratO3 DJF file (created in nam_sam.damip.ncwrite.ncl)
; - Line 65 update to path of NAM/SAM hist-stratO3 DJF file (created in nam_sam.damip.ncwrite.ncl)
; - Line 69 update to path of NAM/SAM hist-aer DJF file (created in nam_sam.damip.ncwrite.ncl)
; - Line 72 update to path of NAM/SAM hist-aer DJF file (created in nam_sam.damip.ncwrite.ncl)
; - Line 76 update to path of NAM/SAM hist-GHG DJF file (created in nam_sam.damip.ncwrite.ncl)
; - Line 79 update to path of NAM/SAM hist-GHG DJF file (created in nam_sam.damip.ncwrite.ncl)
; - Line 83 update to path of NAM/SAM hist-nat DJF file (created in nam_sam.damip.ncwrite.ncl)
; - Line 86 update to path of NAM/SAM hist-nat DJF file (created in nam_sam.damip.ncwrite.ncl)
; - Line 132 update to path of NAM/SAM alt_def observational DJF file (created in nam_sam.damip.alt_def.ncwrite.ncl)
; - Line 135 update to path of NAM/SAM alt_def observational JJA file (created in nam_sam.damip.alt_def.ncwrite.ncl)
; - Line 139 update to path of NAM/SAM alt_def piControl DJF file (created in nam_sam.piControl.alt_def.ncwrite.ncl)
; - Line 142 update to path of NAM/SAM alt_def piControl JJA file (created in nam_sam.piControl.alt_def.ncwrite.ncl)
; - Line 146 update to path of NAM/SAM alt_def historical DJF file (created in nam_sam.damip.alt_def.ncwrite.ncl)
; - Line 149 update to path of NAM/SAM alt_def historical JJA file (created in nam_sam.damip.alt_def.ncwrite.ncl)
; - Line 153 update to path of NAM/SAM alt_def hist-stratO3 DJF file (created in nam_sam.damip.alt_def.ncwrite.ncl)
; - Line 156 update to path of NAM/SAM alt_def hist-stratO3 DJF file (created in nam_sam.damip.alt_def.ncwrite.ncl)
; - Line 160 update to path of NAM/SAM alt_def hist-aer DJF file (created in nam_sam.damip.alt_def.ncwrite.ncl)
; - Line 163 update to path of NAM/SAM alt_def hist-aer DJF file (created in nam_sam.damip.alt_def.ncwrite.ncl)
; - Line 167 update to path of NAM/SAM alt_def hist-GHG DJF file (created in nam_sam.damip.alt_def.ncwrite.ncl)
; - Line 170 update to path of NAM/SAM alt_def hist-GHG DJF file (created in nam_sam.damip.alt_def.ncwrite.ncl)
; - Line 174 update to path of NAM/SAM alt_def hist-nat DJF file (created in nam_sam.damip.alt_def.ncwrite.ncl)
; - Line 177 update to path of NAM/SAM alt_def hist-nat DJF file (created in nam_sam.damip.alt_def.ncwrite.ncl)
;

; Updated: #Soufiane Karmouche 15/09/2020



load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/plot/mder.ncl"

load "$diag_scripts/ipccwg1ar6ch3_modes/functions.ncl"

begin

  ; Create work directory
  out_path = config_user_info@work_dir

  system("mkdir -p " + out_path)

  ; Create output plot directory
  plot_dir = config_user_info@plot_dir
  system("mkdir -p " + plot_dir)

  ; Plot file type
  file_type = config_user_info@output_file_type
  if (ismissing(file_type)) then
    file_type = "ps"
  end if

  print(out_path)
  ;print("out_path" + out_path)

  curdir= systemfunc ("pwd")
  ;print(curdir)

  godir = systemfunc ("csh -c 'cd " + out_path + " ; pwd'") ; first go to work directory

  ncpath= systemfunc ("ls ../../../work/diag_calc_historical/diag_historical_djf/nam_nao_sam.obs.djf.1958_1979-2014.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_obs_djf = a->nam_pc_trends
  sam_obs_djf = a->sam_pc_trends
  ncpath= systemfunc ("ls ../../../work/diag_calc_historical/diag_historical_jja/nam_nao_sam.obs.jja.1958_1979-2014.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_obs_jja = a->nam_pc_trends
  sam_obs_jja = a->sam_pc_trends

  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.piControl.djf.57_36yr.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_ctl_djf = a->nam_pc_trends
  sam_ctl_djf = a->sam_pc_trends

  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.piControl.jja.57_36yr.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_ctl_jja = a->nam_pc_trends
  sam_ctl_jja = a->sam_pc_trends

  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.damip_historical.djf.1958_1979-2014.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_hist_djf = a->nam_pc_trends
  sam_hist_djf = a->sam_pc_trends
  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.damip_historical.jja.1958_1979-2014.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_hist_jja = a->nam_pc_trends
  sam_hist_jja = a->sam_pc_trends

  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.damip_hist-stratO3.djf.1958_1979-2014.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_histstrat_djf = a->nam_pc_trends
  sam_histstrat_djf = a->sam_pc_trends
  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.damip_hist-stratO3.jja.1958_1979-2014.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_histstrat_jja = a->nam_pc_trends
  sam_histstrat_jja = a->sam_pc_trends

  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.damip_hist-aer.djf.1958_1979-2014.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_histaer_djf = a->nam_pc_trends
  sam_histaer_djf = a->sam_pc_trends
  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.damip_hist-aer.jja.1958_1979-2014.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_histaer_jja = a->nam_pc_trends
  sam_histaer_jja = a->sam_pc_trends

  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.damip_hist-GHG.djf.1958_1979-2014.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_histghg_djf = a->nam_pc_trends
  sam_histghg_djf = a->sam_pc_trends
  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.damip_hist-GHG.jja.1958_1979-2014.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_histghg_jja = a->nam_pc_trends
  sam_histghg_jja = a->sam_pc_trends

  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.damip_hist-nat.djf.1958_1979-2014.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_histnat_djf = a->nam_pc_trends
  sam_histnat_djf = a->sam_pc_trends
  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.damip_hist-nat.jja.1958_1979-2014.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_histnat_jja = a->nam_pc_trends
  sam_histnat_jja = a->sam_pc_trends
;-------------------------------------------------------------------------
; values are in units of per 57yrs (NAM) and 36 years (SAM), will convert
; to be in terms of per decade

  yarrN = new((/12,5/),float)
  yarrN(0,:) = (/ boxplot_stat(nam_ctl_djf) /)
  yarrN(1,:) = (/ boxplot_stat(nam_hist_djf) /)
  yarrN(2,:) = (/ boxplot_stat(nam_histaer_djf) /)
  yarrN(3,:) = (/ boxplot_stat(nam_histghg_djf) /)
  yarrN(4,:) = (/ boxplot_stat(nam_histnat_djf) /)
  yarrN(5,:) = (/ boxplot_stat(nam_histstrat_djf) /)

  yarrN(6,:) = (/ boxplot_stat(nam_ctl_jja) /)
  yarrN(7,:) = (/ boxplot_stat(nam_hist_jja) /)
  yarrN(8,:) = (/ boxplot_stat(nam_histaer_jja) /)
  yarrN(9,:) = (/ boxplot_stat(nam_histghg_jja) /)
  yarrN(10,:) = (/ boxplot_stat(nam_histnat_jja) /)
  yarrN(11,:) = (/ boxplot_stat(nam_histstrat_jja) /)

  yarrS = new((/12,5/),float)
  yarrS(0,:) = (/ boxplot_stat(sam_ctl_djf) /)
  yarrS(1,:) = (/ boxplot_stat(sam_hist_djf) /)
  yarrS(2,:) = (/ boxplot_stat(sam_histaer_djf) /)
  yarrS(3,:) = (/ boxplot_stat(sam_histghg_djf) /)
  yarrS(4,:) = (/ boxplot_stat(sam_histnat_djf) /)
  yarrS(5,:) = (/ boxplot_stat(sam_histstrat_djf) /)

  yarrS(6,:) = (/ boxplot_stat(sam_ctl_jja) /)
  yarrS(7,:) = (/ boxplot_stat(sam_hist_jja) /)
  yarrS(8,:) = (/ boxplot_stat(sam_histaer_jja) /)
  yarrS(9,:) = (/ boxplot_stat(sam_histghg_jja) /)
  yarrS(10,:) = (/ boxplot_stat(sam_histnat_jja) /)
  yarrS(11,:) = (/ boxplot_stat(sam_histstrat_jja) /)

  yarrN = (/ (yarrN/57.)*10 /)
  yarrS = (/ (yarrS/36.)*10 /)
  nam_obs_djf = (/ (nam_obs_djf/57.)*10 /)
  nam_obs_jja = (/ (nam_obs_jja/57.)*10 /)
  sam_obs_djf = (/ (sam_obs_djf/36.)*10 /)
  sam_obs_jja = (/ (sam_obs_jja/36.)*10 /)
;-------------------------------------------------------------------------
; Read in alternative defition trends
;-------------------------------------------------------------------------
  godir = systemfunc ("csh -c 'cd " + out_path + " ; pwd'") ; first go to work directory

  ncpath= systemfunc ("ls ../../../work/diag_calc_historical/diag_alt_def_historical_djf/nam_nao_sam.alt_def.obs.djf.1958_1979-2014.n*") ; print .nc full path
  a = addfile(ncpath ,"r")
  nam_obs_djfA = a->nam_trends
  print(nam_obs_djfA)
  sam_obs_djfA = a->sam_trends
  ncpath= systemfunc ("ls ../../../work/diag_calc_historical/diag_alt_def_historical_jja/nam_nao_sam.alt_def.obs.jja.1958_1979-2014.n*")
  a = addfile(ncpath ,"r")
  nam_obs_jjaA = a->nam_trends
  print(nam_obs_jjaA)
  sam_obs_jjaA = a->sam_trends


  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.alt_def.piControl.djf.57_36yr.n*")
  a = addfile(ncpath ,"r")
  nam_ctl_djfA = a->nam_trends
  sam_ctl_djfA = a->sam_trends
  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.alt_def.piControl.jja.57_36yr.n*")
  a = addfile(ncpath ,"r")
  nam_ctl_jjaA = a->nam_trends
  print(nam_ctl_jjaA)
  sam_ctl_jjaA = a->sam_trends

  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.alt_def.historical.djf.1958_1979-2014.n*")
  a = addfile(ncpath ,"r")
  nam_hist_djfA = a->nam_trends
  sam_hist_djfA = a->sam_trends
  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.alt_def.historical.jja.1958_1979-2014.n*")
  a = addfile(ncpath ,"r")
  nam_hist_jjaA = a->nam_trends
  sam_hist_jjaA = a->sam_trends

  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.alt_def.damip_hist-stratO3.djf.1958_1979-2014.n*")
  a = addfile(ncpath ,"r")
  nam_histstrat_djfA = a->nam_trends
  sam_histstrat_djfA = a->sam_trends
  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.alt_def.damip_hist-stratO3.jja.1958_1979-2014.n*")
  a = addfile(ncpath ,"r")
  nam_histstrat_jjaA = a->nam_trends
  sam_histstrat_jjaA = a->sam_trends

  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.alt_def.damip_hist-aer.djf.1958_1979-2014.n*")
  a = addfile(ncpath ,"r")
  nam_histaer_djfA = a->nam_trends
  sam_histaer_djfA = a->sam_trends
  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.alt_def.damip_hist-aer.jja.1958_1979-2014.n*")
  a = addfile(ncpath ,"r")
  nam_histaer_jjaA = a->nam_trends
  print(nam_histaer_jjaA)
  sam_histaer_jjaA = a->sam_trends

  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.alt_def.damip_hist-GHG.djf.1958_1979-2014.n*")
  a = addfile(ncpath ,"r")
  nam_histghg_djfA = a->nam_trends
  print(nam_histghg_djfA)
  sam_histghg_djfA = a->sam_trends
  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.alt_def.damip_hist-GHG.jja.1958_1979-2014.n*")
  a = addfile(ncpath ,"r")
  nam_histghg_jjaA = a->nam_trends
  print(nam_histghg_jjaA)
  sam_histghg_jjaA = a->sam_trends

  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.alt_def.damip_hist-nat.djf.1958_1979-2014.n*")
  a = addfile(ncpath ,"r")
  nam_histnat_djfA = a->nam_trends
  sam_histnat_djfA = a->sam_trends
  ncpath= systemfunc ("ls ../../../work/*/*/nam_nao_sam.alt_def.damip_hist-nat.jja.1958_1979-2014.n*")
  a = addfile(ncpath ,"r")
  nam_histnat_jjaA = a->nam_trends
  sam_histnat_jjaA = a->sam_trends

;-------------------------------------------------------------------------
  nobs       = num(.not.ismissing(nam_obs_djfA))  ;dimsizes(nam_obs_djfA)
  nctl       = num(.not.ismissing(nam_ctl_djfA))  ;dimsizes(nam_ctl_djfA)
  nhist      = num(.not.ismissing(nam_hist_djfA))   ;dimsizes(nam_hist_djfA)
  nhiststrat = num(.not.ismissing(nam_histstrat_djfA))    ;dimsizes(nam_histstrat_djfA)
  nhistaer   = num(.not.ismissing(nam_histaer_djfA))    ;dimsizes(nam_histaer_djfA)
  nhistghg   = num(.not.ismissing(nam_histghg_djfA))    ;dimsizes(nam_histghg_djfA)
  nhistnat   = num(.not.ismissing(nam_histnat_djfA))    ;dimsizes(nam_histnat_djfA)
;-------------------------------------------------------------------------
; values are in units of per 57yrs (NAM) and 36 years (SAM), will convert
; to be in terms of per decade

  zarrN = new((/12,5/),float)
  zarrN(0,:) = (/ boxplot_stat(nam_ctl_djfA) /)
  zarrN(1,:) = (/ boxplot_stat(nam_hist_djfA) /)
  zarrN(2,:) = (/ boxplot_stat(nam_histaer_djfA) /)
  zarrN(3,:) = (/ boxplot_stat(nam_histghg_djfA) /)
  zarrN(4,:) = (/ boxplot_stat(nam_histnat_djfA) /)
  zarrN(5,:) = (/ boxplot_stat(nam_histstrat_djfA) /)

  zarrN(6,:) = (/ boxplot_stat(nam_ctl_jjaA) /)
  zarrN(7,:) = (/ boxplot_stat(nam_hist_jjaA) /)
  zarrN(8,:) = (/ boxplot_stat(nam_histaer_jjaA) /)
  zarrN(9,:) = (/ boxplot_stat(nam_histghg_jjaA) /)
  zarrN(10,:) = (/ boxplot_stat(nam_histnat_jjaA) /)
  zarrN(11,:) = (/ boxplot_stat(nam_histstrat_jjaA) /)

  zarrS = new((/12,5/),float)
  zarrS(0,:) = (/ boxplot_stat(sam_ctl_djfA) /)
  zarrS(1,:) = (/ boxplot_stat(sam_hist_djfA) /)
  zarrS(2,:) = (/ boxplot_stat(sam_histaer_djfA) /)
  zarrS(3,:) = (/ boxplot_stat(sam_histghg_djfA) /)
  zarrS(4,:) = (/ boxplot_stat(sam_histnat_djfA) /)
  zarrS(5,:) = (/ boxplot_stat(sam_histstrat_djfA) /)

  zarrS(6,:) = (/ boxplot_stat(sam_ctl_jjaA) /)
  zarrS(7,:) = (/ boxplot_stat(sam_hist_jjaA) /)
  zarrS(8,:) = (/ boxplot_stat(sam_histaer_jjaA) /)
  zarrS(9,:) = (/ boxplot_stat(sam_histghg_jjaA) /)
  zarrS(10,:) = (/ boxplot_stat(sam_histnat_jjaA) /)
  zarrS(11,:) = (/ boxplot_stat(sam_histstrat_jjaA) /)

  zarrN = (/ (zarrN/57.)*10 /)
  zarrS = (/ (zarrS/36.)*10 /)
  nam_obs_djfA = (/ (nam_obs_djfA/57.)*10 /)
  nam_obs_jjaA = (/ (nam_obs_jjaA/57.)*10 /)
  sam_obs_djfA = (/ (sam_obs_djfA/36.)*10 /)
  sam_obs_jjaA = (/ (sam_obs_jjaA/36.)*10 /)
;========================================================================================
  outpath = config_user_info@plot_dir
  ofile = outpath + "nam_sam_trends_damip"
  wks = gsn_open_wks(file_type, ofile)


  bres            = True                         ; plot mods desired
  bres@vpWidthF = 0.6
  bres@vpHeightF = 0.45
  bres@trXMinF = -0.5
  bres@trXMaxF = 14.
  bres@trYMinF = -.45
  bres@trYMaxF = .45
  bres@tmXBLabels = (/"","","","","","","","","","","",""/) ; labels for each box
  bres@tmXBOn = False
  bres@tmXBLabelsOn = True
  bres@tmYLLabelFontHeightF = 0.024
  bres@tmYLMode = "Explicit"
  bres@tmYLValues = (/-0.45,-0.3,-0.15,0,0.15,0.3,0.45,0.6/)

  bres@tmYLLabels = (/"-0.45","-0.30","-0.15","0","0.15","0.30","0.45","0.60"/)
  bres@tmYLMinorValues = fspan(-1,1,41)
  bres@tiMainFont = "Helvetica"
  bres@tiMainString = "Autocorrelation"
  bres@tiMainFontHeightF = 0.028
  bres@tiMainOffsetYF = -0.01
  llres                   = True
  llres@gsLineThicknessF  = 1.25                 ; line thickness
  llres@gsLineDashPattern = 0   ;5
  opti          = True
  opti@boxWidth = .3				; Width of box (x units)

  gsn_define_colormap(wks,"srip_reanalysis")
  opti@boxColors = (/4,16,15,2,10,7, 4,16,15,2,10,7/)
  polyres = True
  polyres@gsMarkerColor = "black"
  polyres@gsMarkerThicknessF = 2.5
  polyres@gsMarkerIndex = 1
  txres = True
  txres@txFontHeightF = 0.025
  plres = True
  plres@gsLineColor = (/0,0,0,0.7/)
  plres@gsLineThicknessF = 1.
  plres@tfPolyDrawOrder = "PostDraw"
  plres@gsLineDashPattern = 0

  plres2 = True
  plres2@gsLineColor = "gray50"   ;(/0.1,0.1,0.1,0.3/)  ;"gray50"
  plres2@gsLineThicknessF = 2.
  plres2@tfPolyDrawOrder = "PostDraw"

  polyres = True
  polyres@gsFillColor = (/0.75,0.75,0.75,1/)
  polyres@tfPolyDrawOrder = "PostDraw"


  plot_blank = new(2,graphic)
  bres@tiMainFontColor = "white"
  bres@tmYLLabelFontColor = "white"
  bres@tiMainString = "(a) NAM"
  plot_blank(0) = blank_boxplot(wks,bres)
  bres@tmYLValues := (/-0.9,-0.6,-0.3,0,0.3,0.6,0.9/)
  bres@tmYLLabels := bres@tmYLValues
  bres@tmYLMinorValues := fspan(-1,1,21)
  bres@trYMinF = -.6
  bres@trYMaxF = .9
  bres@tiMainString = "(b) SAM"
  plot_blank(1) = blank_boxplot(wks,bres)

  print("NAM DJF Obs: "+nam_obs_djfA)
  print("NAM JJA Obs: "+nam_obs_jjaA)
  print("SAM DJF Obs: "+sam_obs_djfA)
  print("SAM JJA Obs: "+sam_obs_jjaA)

  dum = new(4,graphic)
  dum(0) = gsn_add_polygon(wks,plot_blank(0),(/0.2,6.3,6.3,0.2,0.2/),(/min(nam_obs_djfA),min(nam_obs_djfA),max(nam_obs_djfA),max(nam_obs_djfA),min(nam_obs_djfA)/),polyres)
  dum(1) = gsn_add_polygon(wks,plot_blank(0),(/7.2,13.3,13.3,7.2,7.2/),(/min(nam_obs_jjaA),min(nam_obs_jjaA),max(nam_obs_jjaA),max(nam_obs_jjaA),min(nam_obs_jjaA)/),polyres)
  dum(2) = gsn_add_polygon(wks,plot_blank(1),(/0.2,6.3,6.3,0.2,0.2/),(/min(sam_obs_djfA),min(sam_obs_djfA),max(sam_obs_djfA),max(sam_obs_djfA),min(sam_obs_djfA)/),polyres)
  dum(3) = gsn_add_polygon(wks,plot_blank(1),(/7.2,13.3,13.3,7.2,7.2/),(/min(sam_obs_jjaA),min(sam_obs_jjaA),max(sam_obs_jjaA),max(sam_obs_jjaA),min(sam_obs_jjaA)/),polyres)

  panres = True
  panres@txFontHeightF = 0.022
  panres@gsnPanelXWhiteSpacePercent = 7.0
  panres@gsnFrame = False
  gsn_panel(wks,plot_blank,(/1,2/),panres)

;--- Create 1st panel-----
  bplot = new(2,graphic)
  bplot2 = bplot
  bres@tiMainFontColor = "black"
  bres@tmYLLabelFontColor = "black"
  bres@trYMinF = -.45
  bres@trYMaxF = .45
  bres@tmYLValues := (/-0.45,-0.3,-0.15,0,0.15,0.3,0.45,0.6/)
  bres@tmYLLabels := (/"-0.45","-0.30","-0.15","0","0.15","0.30","0.45","0.60"/)
  bres@tmYLMinorValues := fspan(-1,1,41)
  bres@tiMainString = "(a) NAM"

  bplot(0) = boxplot2(wks,(/1,2,3,4,5,6,8,9,10,11,12,13/),yarrN,opti,bres,llres)


  dum2 = new(3,graphic)
  dum3 = dum

  opti@boxFill_Colors = opti@boxColors
  opti@boxFill_midColor = "white"
  bplot2(0) = boxplot2(wks,(/1,2,3,4,5,6,8,9,10,11,12,13/)-.5,zarrN,opti,bres,llres)

  dum2(0) = gsn_add_text(wks,bplot2(0),"DJF",3.25,bres@trYMinF+0.1,txres)
  dum2(1) = gsn_add_text(wks,bplot2(0),"JJA",10.25,bres@trYMinF+0.1,txres)
  dum2(2) = gsn_add_polyline(wks,bplot2(0),(/-15,15/),(/0.,0./),plres2)
  overlay(bplot(0),bplot2(0))

;--- Create 2nd panel-----

  bres@tmYLValues := (/-0.9,-0.6,-0.3,0,0.3,0.6,0.9/)
  bres@tmYLLabels := bres@tmYLValues
  bres@tmYLMinorValues := fspan(-1,1,21)
  bres@trYMinF = -.6
  bres@trYMaxF = .9
  bres@tiMainString = "(b) SAM"
  delete([/opti@boxFill_Colors,opti@boxFill_midColor/])
  bplot(1) = boxplot2(wks,(/1,2,3,4,5,6,8,9,10,11,12,13/),yarrS,opti,bres,llres)

  opti@boxFill_Colors = opti@boxColors
  opti@boxFill_midColor = "white"

  bplot2(1) = boxplot2(wks,(/1,2,3,4,5,6,8,9,10,11,12,13/)-.5,zarrS,opti,bres,llres)

  dum3(0) = gsn_add_text(wks,bplot2(1),"DJF",3.25,bres@trYMinF+0.17,txres)
  dum3(1) = gsn_add_text(wks,bplot2(1),"JJA",10.25,bres@trYMinF+0.17,txres)
  dum3(2) = gsn_add_polyline(wks,bplot2(1),(/-15,15/),(/0.,0./),plres2)
  overlay(bplot(1),bplot2(1))

;---------------------------------------------
; Create legend
;
  gres = True
  gres@YPosPercent = 96.    ; expressed as %, 0->100, sets position of top border of legend
                            ;  when gres@Position is set to its default setting of "Top" (Default = 95.)
  gres@XPosPercent = 40.      ; expressed as %, 0->100, sets position of left border of legend(Default = 5.)
  gres@LineLabelWhiteSpacePercent = 2.
  gres@ItemSpacePercent = 5.5

  lineres = True
  lineres@lgLineColors = opti@boxColors(:2)   ; line colors
  lineres@lgLineThicknesses = llres@gsLineThicknessF                       ; line thicknesses
  lineres@LineLengthPercent = 5.                         ; expressed as %, 0->100, length of line

  textres = True
  textres@lgLabels = (/"Pi-control ("+nctl+")","Historical ("+nhist+")","Hist-aer ("+nhistaer+")"/)  ; legend labels (required)
  textres@lgLabelFontHeights = 0.015
  bplot(1) = simple_legend(wks,bplot(1),gres,lineres,textres)
  gres@XPosPercent = 72.
  textres@lgLabels = (/"Hist-GHG ("+nhistghg+")","Hist-nat ("+nhistnat+")","Hist-stratO3 ("+nhiststrat+")"/)
  lineres@lgLineColors = opti@boxColors(3:5)
  bplot(1) = simple_legend(wks,bplot(1),gres,lineres,textres)

  gres@YPosPercent = 93.5
  gres@XPosPercent = 8.
  lineres@lgLineThicknesses = 12.
  textres@lgLabels := " Observational~C~ Range"   ; ("+nobs+")"
  lineres@lgLineColors :=  polyres@gsFillColor
  bplot(1) = simple_legend(wks,bplot(1),gres,lineres,textres)

  gsn_panel(wks,bplot,(/1,2/),panres)
  frame(wks)
  delete(wks)
  system("convert -density 216 -trim +repage -border 10 -bordercolor white "+ofile+".eps "+ofile+".png")
end
