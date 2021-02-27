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
;    20210227 kosaka_yu: revised SAM index definition.
;    20210221 kosaka_yu: added file output and provenance.
;    20210129 kosaka_yu: removed DJF.
;    20201029 kosaka_yu: removed MME mean.
;    20200916 kosaka_yu: revised.
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

  restandardize = True ;False
  
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
  sam!0 = "ensemble"
  sam&ensemble = ispan(1, nintegrations, 1)
  sam!1 = "year"
  sam&year = ispan(syear_data, eyear_data, 1)
  cmip_phase = new((/nintegrations/), "string")
  model_name = new((/nintegrations/), "string")
  ensemble_name = new((/nintegrations/), "string")

  north = new((/nintegrations, eyear_data-syear_data+1, 12/), "float")
  north!0 = "ensemble"
  north&ensemble = ispan(1, nintegrations, 1)
  north!1 = "year"
  north&year = ispan(syear_data, eyear_data, 1)
  north!2 = "month"
  north&month = ispan(1, 12, 1)
  south = north
  
  do ii = 0, ListCount(info) - 1
    ; Read data
    psl = read_data(info[ii])
    
    time = cd_calendar(psl&time, -1)
    syr = time(0)/100
    eyr = time(dimsizes(time)-1)/100
    smo = time(0) - syr*100
    emo = time(dimsizes(time)-1) - eyr*100
    
    slp = new((/eyr-syr+1, 12, dimsizes(psl(0, :))/), typeof(psl))
    copy_VarCoords(psl(0, :), slp(0, 0, :))
    slp!0 = "year"
    slp&year = ispan(syr, eyr, 1)
    slp!1 = "month"
    slp&month = ispan(1, 12, 1)

    do yr = syr, eyr
      do mo = 1, 12
        if ((yr.eq.syr.and.mo.lt.smo).or.(yr.eq.eyr.and.mo.gt.emo)) then
          continue
        end if
        slp({yr}, {mo}, :) = (/psl(ind(time.eq.yr*100+mo), :)/)
      end do
    end do
    delete(time)
    delete(psl)

    yr0 = max((/syr, syear_data/))
    yr1 = min((/eyr, eyear_data/))

    int = ind(info[ii]@project+"_"+info[ii]@dataset+"_"+info[ii]@ensemble.eq.integrations)
    
    north(int, {yr0:yr1}, :) = (/slp({yr0:yr1}, :, {-40.})/)
    south(int, {yr0:yr1}, :) = (/slp({yr0:yr1}, :, {-65.})/)
    cmip_phase(int) = info[ii]@project
    model_name(int) = info[ii]@dataset
    ensemble_name(int) = info[ii]@ensemble

    delete(slp)
  end do

  north = north - conform(north, dim_avg_n(north(:, {syear_ref:eyear_ref}, :), 1), (/0, 2/))
  north = north / conform(north, dim_stddev_n(north(:, {syear_ref:eyear_ref}, :), 1), (/0, 2/))
  south = south - conform(south, dim_avg_n(south(:, {syear_ref:eyear_ref}, :), 1), (/0, 2/))
  south = south / conform(south, dim_stddev_n(south(:, {syear_ref:eyear_ref}, :), 1), (/0, 2/))
    
  sam = dim_avg_n(north, 2) - dim_avg_n(south, 2)
  delete(north)
  delete(south)

  if (restandardize) then
    sam = sam - conform(sam, dim_avg_n(sam(:, {syear_ref:eyear_ref}), 1), 0)
    sam = sam / conform(sam, dim_stddev_n(sam(:, {syear_ref:eyear_ref}), 1), 0)
  end if
  
; ======================== READ RECONSTRUCTIONS ========================

  script = get_script_name()
  dirs = str_split(script, "/")
  dirs = dirs+"/"
  diag_script_dir = "/"+str_concat(dirs(0:dimsizes(dirs)-2))
  delete(dirs)
  delete(script)
  dir_abram = "abram2014/"
  dir_datwyler = "datwyler2017/"
  file_abram = "abram2014sam.txt"
  file_datwyler_ann = "Reconstructions_Annual_LC.txt"

  address_abram = "ftp://ftp.ncdc.noaa.gov/pub/data/paleo/contributions_by_author/abram2014/"+file_abram
  address_datwyler_ann = "https://www1.ncdc.noaa.gov/pub/data/paleo/reconstructions/datwyler2017/"+file_datwyler_ann
  
  do n = 0, 1
    if (n.eq.0) then
      path = diag_script_dir+dir_abram+file_abram
      download_path = address_abram
      save_dir = dir_abram
      save_file = file_abram
      ncol = 1
      delimiter = str_get_tab()
    elseif (n.eq.1) then
      path = diag_script_dir+dir_datwyler+file_datwyler_ann
      download_path = address_datwyler_ann
      save_dir = dir_datwyler
      save_file = file_datwyler_ann
      ncol = 2
      delimiter = ";"
    end if
    if (.not.fileexists(path)) then
      path = config_user_info@work_dir+save_dir+save_file
      if (.not.fileexists(path)) then
        system("mkdir -p "+config_user_info@work_dir+save_dir)
        system("wget -P "+config_user_info@work_dir+save_dir+" "+download_path)
      end if
    end if
    if (isvar("file_reconstruction")) then
      file_reconstruction := array_append_record(file_reconstruction, path, 0)
    else
      file_reconstruction = path
    end if
    
    lines = asciiread(path, -1, "string")
    nlines = dimsizes(lines)
    do ii = 0, nlines-1
      if (str_is_blank(lines(ii))) then
        continue
      end if
      if (isStrSubset(lines(ii), "#")) then
        continue
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

    if (restandardize) then
      ts = ind(data(:, 0).eq.syear_ref)
      te = ind(data(:, 0).eq.eyear_ref)
      data(:, 1) = data(:, 1) - avg(data(ts:te, 1))
      data(:, 1) = data(:, 1) / stddev(data(ts:te, 1))
    end if
    
    dims = dimsizes(data)
    syr = toint(data(0, 0))
    eyr = toint(data(dims(0)-1, 0))
    delete(dims)
    if (isStrSubset(path, file_abram)) then
      sam_abram = sam(0, :)
      sam_abram = sam_abram@_FillValue
      sam_abram({syr:eyr}) = (/data(:, 1)/)
      sam_abram = sam_abram - avg(sam_abram({syear_ref:eyear_ref}))
      sam_abram@source = "Abram et al. (2014)"
      sam_abram@color = (/68., 113., 212./)/255. ;(/36., 147., 126./)/255.
    elseif (isStrSubset(path, file_datwyler_ann)) then
      sam_datwyler_ann = sam(0, :)
      sam_datwyler_ann = sam_datwyler_ann@_FillValue
      sam_datwyler_ann({syr:eyr}) = (/data(:, 1)/)
      sam_datwyler_ann = sam_datwyler_ann - avg(sam_datwyler_ann({syear_ref:eyear_ref}))
      sam_datwyler_ann@source = "Da~H-13V2F35~H~FV-2H3~twyler et al. (2018)"
      sam_datwyler_ann@color = (/197., 90., 17./)/255.
    end if
    delete(data)
    delete(lines)
  end do

; ============================== FILTERING =============================

  sam@project = str_join(cmip_phase, ",")
  sam@dataset = str_join(model_name + "/" + ensemble_name, ",")
  
  ; Running mean
  wgt = new(runmean, "float")
  wgt = 1.
  wgt = wgt / sum(wgt)
  sam_runmean = wgt_runave_n_Wrap(sam, wgt, 0, 1)
  sam_abram_runmean = wgt_runave_n_Wrap(sam_abram, wgt, 0, 0)
  sam_datwyler_ann_runmean = wgt_runave_n_Wrap(sam_datwyler_ann, wgt, 0, 0)
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
  sam_datwyler_ann_lowpass = wgt_runave_n_Wrap(sam_datwyler_ann, wgt, 0, 0)
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
  dummy = new(500, "graphic")
  idummy = 0
  
  rests := True
  rests@tmYLLabelFontHeightF = 0.015
  rests@tmYRLabelFontHeightF = rests@tmYLLabelFontHeightF
  rests@tmXBLabelFontHeightF = 0.015
  rests@gsnDraw           = False
  rests@gsnFrame          = False
  rests@trXMinF           = tofloat(syear)
  rests@trXMaxF           = tofloat(eyear)
  rests@tmXTOn            = False
  rests@tmYLMode          = "Explicit"
  rests@vpWidthF          = 0.6
  rests@vpHeightF         = 0.3
  rests@tiMainFontHeightF = 0.015
  rests@tiMainOffsetYF = -0.008

  rests@tiMainString = "SAM index in the last millenium"
  rests@tiMainFontHeightF = 0.02
  
  rests = rests
  rests@trYMinF = -4.5
  rests@trYMaxF = 4.
  rests@tmYLOn = True
  rests@tmYLLabelsOn = True
  rests@tmYLValues = (/-6, -4, -2, 0, 2, 4, 6/)
  rests@tmYLLabels =   rests@tmYLValues
  rests@tmYLMinorValues = (/-5., -3., -1., 1., 3., 5. /)
  
  rests@xyLineColor = (/0.5, 0.5, 0.5/)
  rests@xyDashPattern     = 0
  rests@xyLineThicknessF  = 0.5
  plot0 = gsn_csm_xy(wks, (/syear, eyear/), (/0, 0/), rests)

  txres := True
  txres@txFontHeightF = 0.02
  txres@txJust = "CenterLeft"
  dummy(idummy) = gsn_add_text(wks, plot0, "(a)", syear+20, rests@trYMaxF - 0.7, txres)
  idummy = idummy + 1
  
  rests@trYMinF = -3.5
  rests@trYMaxF = 3.5
  rests@vpHeightF = rests@vpHeightF * 7. / 8.5
  rests@tmYLLabelsOn = True
  rests@tmYRLabelsOn = False
  rests@tiMainString = ""
  plot1 = gsn_csm_xy(wks, (/syear, eyear/), (/0, 0/), rests)
  dummy(idummy) = gsn_add_text(wks, plot1, "(b)", syear+20, rests@trYMaxF - 0.7, txres)
  idummy = idummy + 1
  
  lres = True
  lres@gsLineDashPattern = 0
  lres@gsLineThicknessF = 0.5
  lres@gsLineColor = sam_abram@color
  dummy(idummy) = gsn_add_polyline(wks, plot0, sam_abram_runmean&year, sam_abram_runmean, lres)
  idummy = idummy + 1
  lres@gsLineColor    = sam_datwyler_ann@color
  dummy(idummy) = gsn_add_polyline(wks, plot0, sam_datwyler_ann_runmean&year, sam_datwyler_ann_runmean, lres)
  idummy = idummy + 1

  lres@gsLineThicknessF = 2.5
  lres@gsLineColor = sam_abram@color
  dummy(idummy) = gsn_add_polyline(wks, plot0, sam_abram_lowpass&year, sam_abram_lowpass, lres)
  idummy = idummy + 1
  lres@gsLineColor = sam_datwyler_ann@color
  dummy(idummy) = gsn_add_polyline(wks, plot0, sam_datwyler_ann_lowpass&year, sam_datwyler_ann_lowpass, lres)
  idummy = idummy + 1

  color_cmip5 = (/0.5, 0.5, 0.5/)
  do ii = 0, nintegrations-1
    if (cmip_phase(ii).eq."CMIP5") then
      lres@gsLineColor    = color_cmip5
      lres@gsLineThicknessF = 0.5
      lres@gsLineOpacityF = 0.5
    elseif (cmip_phase(ii).eq."CMIP6") then
      lres@gsLineColor    = get_color(model_name(ii))
      lres@gsLineThicknessF = 0.5
      lres@gsLineOpacityF = 1.
    end if
    dummy(idummy) = gsn_add_polyline(wks, plot1, sam_runmean&year, sam_runmean(ii, :), lres)
    idummy = idummy + 1
  end do
  lres@gsLineOpacityF = 1.
  do ii = 0, nintegrations-1
    if (cmip_phase(ii).eq."CMIP5") then
      lres@gsLineColor := color_cmip5
      lres@gsLineThicknessF = 1.5
    elseif (cmip_phase(ii).eq."CMIP6") then
      lres@gsLineColor := get_color(model_name(ii))
      lres@gsLineThicknessF = 2.5
    end if
    dummy(idummy) = gsn_add_polyline(wks, plot1, sam_lowpass&year, sam_lowpass(ii, :), lres)
    idummy = idummy + 1
  end do

  
  txres@txFontHeightF = 0.015  
  txres@txFontColor = sam_abram@color
  dummy(idummy) = gsn_add_text(wks, plot0, sam_abram@source, 1420., 0.5, txres)
  idummy = idummy + 1
  
  txres@txFontColor = sam_datwyler_ann@color
  dummy(idummy) = gsn_add_text(wks, plot0, sam_datwyler_ann@source, 1670., -3.5, txres)
  idummy = idummy + 1
  

  txres@txJust = "TopLeft"
  legend = get_unique_values(where(cmip_phase.ne."CMIP5", model_name, "CMIP5"))
  nlegend = dimsizes(legend)
  x = syear + 50.
  dx = tofloat(eyear-syear-100)/40.
  do n = 0, nlegend-1
    if (legend(n).ne."CMIP5") then
      txres@txFontColor := get_color(legend(n))  
      dummy(idummy) = gsn_add_text(wks, plot1, legend(n), x, -2.8, txres)
      idummy = idummy + 1
      x = x + dx*(strlen(legend(n))*0.6+5.)
    end if
  end do
  if (any(legend.eq."CMIP5")) then
    txres@txFontColor := color_cmip5
    dummy(idummy) = gsn_add_text(wks, plot1, "CMIP5", x, -2.8, txres)
    idummy = idummy + 1
    x = x + dx*strlen("CMIP5")
  end if

  res1                     = True
  res2                     = True
  res1@gsnMaximize         = False
  res2@gsnAttachPlotsXAxis = True

  amid = gsn_attach_plots(plot0,plot1,res1,res2)

  draw(plot0)
  frame(wks)
  
; =============================== Write Output ===============================
  outpath = config_user_info@work_dir + "sam_millennium.nc"
  if (config_user_info@write_netcdf) then
    system("mkdir -p " + config_user_info@work_dir)
    delete(sam_runmean@wgt_runave_op_ncl)
    delete(sam_lowpass@wgt_runave_op_ncl)
    delete(sam_abram_runmean@wgt_runave_op_ncl)
    delete(sam_abram_lowpass@wgt_runave_op_ncl)
    delete(sam_datwyler_ann_runmean@wgt_runave_op_ncl)
    delete(sam_datwyler_ann_lowpass@wgt_runave_op_ncl)
    delete(sam_abram_runmean@ensemble)
    delete(sam_abram_lowpass@ensemble)
    delete(sam_datwyler_ann_runmean@ensemble)
    delete(sam_datwyler_ann_lowpass@ensemble)
    delete(sam_abram_runmean@color)
    delete(sam_abram_lowpass@color)
    delete(sam_datwyler_ann_runmean@color)
    delete(sam_datwyler_ann_lowpass@color)
    delete(sam_abram_runmean@source)
    delete(sam_abram_lowpass@source)
    delete(sam_datwyler_ann_runmean@source)
    delete(sam_datwyler_ann_lowpass@source)
    sam_abram_runmean@reference = "Abram et al. (2014)"
    sam_abram_lowpass@reference = "Abram et al. (2014)"
    sam_datwyler_ann_runmean@reference = "Datwyler et al. (2018)"
    sam_datwyler_ann_lowpass@reference = "Datwyler et al. (2018)"
    sam_runmean@filter = runmean+"-year running mean"
    sam_abram_runmean@filter = runmean+"-year running mean"
    sam_datwyler_ann_runmean@filter = runmean+"-year running mean"
    sam_lowpass@filter = cutoff_period + "-year low-pass filtered"
    sam_abram_lowpass@filter = cutoff_period + "-year low-pass filtered"
    sam_datwyler_ann_lowpass@filter = cutoff_period + "-year low-pass filtered"
    
    sam_runmean@var = "sam_cmip_runmean"
    sam_runmean@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(sam_runmean, outpath)
    outpath@existing = "append" 
    sam_lowpass@var = "sam_cmip_lowpass"
    sam_lowpass@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(sam_lowpass, outpath)
    sam_abram_runmean@var = "sam_abram_runmean"
    sam_abram_runmean@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(sam_abram_runmean, outpath)
    sam_abram_lowpass@var = "sam_abram_lowpass"
    sam_abram_lowpass@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(sam_abram_lowpass, outpath)
    sam_datwyler_ann_runmean@var = "sam_datwyler_runmean"
    sam_datwyler_ann_runmean@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(sam_datwyler_ann_runmean, outpath)
    sam_datwyler_ann_lowpass@var = "sam_datwyler_lowpass"
    sam_datwyler_ann_lowpass@diag_script = DIAG_SCRIPT
    ncdf_outfile = ncdf_write(sam_datwyler_ann_lowpass, outpath)
  end if
  

; ------------------------------------------------------------------------
; write provenance to netcdf output and plot file
; ------------------------------------------------------------------------
;
  plotfile = wks@fullname
  authors = "kosaka_yu"
  statistics = "var"
  domain = (/"shmidlat"/)
  plottype = "times"
  references = (/"abram14natcc", "datwyler18climdyn"/)
  caption = "Filtered time series of annual SAM indices from reconstructions and model simulations."
  infiles = metadata_att_as_array(info, "filename")
  infiles := array_append_record(infiles, file_reconstruction, 0)
  
  log_provenance(outpath, plotfile, caption, statistics, \
                 domain, plottype, authors, references, infiles)

  leave_msg(DIAG_SCRIPT, "")

end
