; #############################################################################
; Plot NAM/SAM trends
; Authors: Yu Kosaka (University of Tokyo, Japan) and Adam Phillips (NCAR, U.S.)
; #############################################################################
;
; Description
;
; Modification history
;   20210306-Yu Kosaka: added netcdf output and provenance
;   20210216-Yu Kosaka: revised figure details
;   20210212-Yu Kosaka: revised to get trend periods from input files
;   20210129-Yu Kosaka: adapt Adam's script to the ESMValTool
;
; #############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/shared/plot/mder.ncl"

load "$diag_scripts/ipccwg1ar6ch3_modes/functions.ncl"

begin

  ; Create output plot directory
  plot_dir = config_user_info@plot_dir
  system("mkdir -p " + plot_dir)

  ; Plot file type
  file_type = config_user_info@output_file_type
  if (ismissing(file_type)) then
    file_type = "ps"
  end if

  idirs = diag_script_info@input_files
  
;-------------------------------------------------------------------------
; Read in alternative defition trends
;-------------------------------------------------------------------------
  do ii = 0, dimsizes(idirs)-1
    print("dir: "+idirs(ii))
    if (isStrSubset(idirs(ii), "diag_obs_")) then
      ncpaths := systemfunc ("ls "+idirs(ii))
      do jj = 0, dimsizes(ncpaths)-1
        if (isStrSubset(str_lower(ncpaths(jj)), "_djf").or.isStrSubset(str_lower(ncpaths(jj)), ".djf.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_obs_djfA = a->nam_trends
          sam_obs_djfA_0 = a->sam_trends0
          sam_obs_djfA_1 = a->sam_trends1
        elseif (isStrSubset(str_lower(ncpaths(jj)), "_jja").or.isStrSubset(str_lower(ncpaths(jj)), ".jja.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_obs_jjaA = a->nam_trends
          sam_obs_jjaA_0 = a->sam_trends0
          sam_obs_jjaA_1 = a->sam_trends1
          obs_datasets = str_split(a->E@datasets, " ")
        end if
      end do
    elseif (isStrSubset(idirs(ii), "diag_pi_")) then
      ncpaths := systemfunc ("ls "+idirs(ii))
      do jj = 0, dimsizes(ncpaths)-1
        if (isStrSubset(str_lower(ncpaths(jj)), "_djf").or.isStrSubset(str_lower(ncpaths(jj)), ".djf.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_ctl_djfA = a->nam_trends
          sam_ctl_djfA_0 = a->sam_trends0
          sam_ctl_djfA_1 = a->sam_trends1
        elseif (isStrSubset(str_lower(ncpaths(jj)), "_jja").or.isStrSubset(str_lower(ncpaths(jj)), ".jja.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_ctl_jjaA = a->nam_trends
          sam_ctl_jjaA_0 = a->sam_trends0
          sam_ctl_jjaA_1 = a->sam_trends1
        end if
      end do
    elseif (isStrSubset(idirs(ii), "diag_historical").or.isStrSubset(idirs(ii), "diag_hist_")) then
      ncpaths := systemfunc ("ls "+idirs(ii))
      do jj = 0, dimsizes(ncpaths)-1
        if (isStrSubset(str_lower(ncpaths(jj)), "_djf").or.isStrSubset(str_lower(ncpaths(jj)), ".djf.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_hist_djfA = a->nam_trends
          sam_hist_djfA_0 = a->sam_trends0
          sam_hist_djfA_1 = a->sam_trends1
        elseif (isStrSubset(str_lower(ncpaths(jj)), "_jja").or.isStrSubset(str_lower(ncpaths(jj)), ".jja.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_hist_jjaA = a->nam_trends
          sam_hist_jjaA_0 = a->sam_trends0
          sam_hist_jjaA_1 = a->sam_trends1
          ea_hist = a->ensemble_assign
        end if
      end do
    elseif (isStrSubset(idirs(ii), "diag_hist-stratO3_")) then
      ncpaths := systemfunc ("ls "+idirs(ii))
      do jj = 0, dimsizes(ncpaths)-1
        if (isStrSubset(str_lower(ncpaths(jj)), "_djf").or.isStrSubset(str_lower(ncpaths(jj)), ".djf.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_histstrat_djfA = a->nam_trends
          sam_histstrat_djfA_0 = a->sam_trends0
          sam_histstrat_djfA_1 = a->sam_trends1
        elseif (isStrSubset(str_lower(ncpaths(jj)), "_jja").or.isStrSubset(str_lower(ncpaths(jj)), ".jja.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_histstrat_jjaA = a->nam_trends
          sam_histstrat_jjaA_0 = a->sam_trends0
          sam_histstrat_jjaA_1 = a->sam_trends1
          ea_histstrat = a->ensemble_assign
        end if
      end do
    elseif (isStrSubset(idirs(ii), "diag_hist-aer_")) then
      ncpaths := systemfunc ("ls "+idirs(ii))
      do jj = 0, dimsizes(ncpaths)-1
        if (isStrSubset(str_lower(ncpaths(jj)), "_djf").or.isStrSubset(str_lower(ncpaths(jj)), ".djf.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_histaer_djfA = a->nam_trends
          sam_histaer_djfA_0 = a->sam_trends0
          sam_histaer_djfA_1 = a->sam_trends1
        elseif (isStrSubset(str_lower(ncpaths(jj)), "_jja").or.isStrSubset(str_lower(ncpaths(jj)), ".jja.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_histaer_jjaA = a->nam_trends
          sam_histaer_jjaA_0 = a->sam_trends0
          sam_histaer_jjaA_1 = a->sam_trends1
          ea_histaer = a->ensemble_assign
        end if
      end do
    elseif (isStrSubset(idirs(ii), "diag_hist-GHG_")) then
      ncpaths := systemfunc ("ls "+idirs(ii))
      do jj = 0, dimsizes(ncpaths)-1
        if (isStrSubset(str_lower(ncpaths(jj)), "_djf").or.isStrSubset(str_lower(ncpaths(jj)), ".djf.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_histghg_djfA = a->nam_trends
          sam_histghg_djfA_0 = a->sam_trends0
          sam_histghg_djfA_1 = a->sam_trends1
        elseif (isStrSubset(str_lower(ncpaths(jj)), "_jja").or.isStrSubset(str_lower(ncpaths(jj)), ".jja.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_histghg_jjaA = a->nam_trends
          sam_histghg_jjaA_0 = a->sam_trends0
          sam_histghg_jjaA_1 = a->sam_trends1
          ea_histghg = a->ensemble_assign
        end if
      end do
    elseif (isStrSubset(idirs(ii), "diag_hist-nat_")) then
      ncpaths := systemfunc ("ls "+idirs(ii))
      do jj = 0, dimsizes(ncpaths)-1
        if (isStrSubset(str_lower(ncpaths(jj)), "_djf").or.isStrSubset(str_lower(ncpaths(jj)), ".djf.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_histnat_djfA = a->nam_trends
          sam_histnat_djfA_0 = a->sam_trends0
          sam_histnat_djfA_1 = a->sam_trends1
        elseif (isStrSubset(str_lower(ncpaths(jj)), "_jja").or.isStrSubset(str_lower(ncpaths(jj)), ".jja.")) then
          if (isvar("paths_all")) then
            paths_all := array_append_record(paths_all, (/idirs(ii)+"/"+ncpaths(jj)/), 0)
          else
            paths_all := (/idirs(ii)+"/"+ncpaths(jj)/)
          end if
          a = addfile(idirs(ii)+"/"+ncpaths(jj),"r")
          nam_histnat_jjaA = a->nam_trends
          sam_histnat_jjaA_0 = a->sam_trends0
          sam_histnat_jjaA_1 = a->sam_trends1
          ea_histnat = a->ensemble_assign
        end if
      end do
    end if
  end do

  syear_a = nam_obs_djfA@syear
  eyear_a = nam_obs_djfA@eyear
  syear_b = sam_obs_djfA_0@syear
  eyear_b = sam_obs_djfA_0@eyear
  syear_c = sam_obs_djfA_1@syear
  eyear_c = sam_obs_djfA_1@eyear

  nam_yrs = tofloat(eyear_a - syear_a + 1)
  sam_yrs0 = tofloat(eyear_b - syear_b + 1)
  sam_yrs1 = tofloat(eyear_c - syear_c + 1)
  
;-------------------------------------------------------------------------
  ea_hist = where(ismissing(nam_hist_djfA),ea_hist@_FillValue,ea_hist)  ; set ea_* arrays to _FillValue when a trend isn't there (for weighting purposes)
  ea_histstrat = where(ismissing(nam_histstrat_djfA),ea_histstrat@_FillValue,ea_histstrat)
  ea_histaer = where(ismissing(nam_histaer_djfA),ea_histaer@_FillValue,ea_histaer)
  ea_histghg = where(ismissing(nam_histghg_djfA),ea_histghg@_FillValue,ea_histghg)
  ea_histnat = where(ismissing(nam_histnat_djfA),ea_histnat@_FillValue,ea_histnat)

  wgt_hist      = calc_weights_by_ens_size(ea_hist)      ; for weighted_percentile calls below
  wgt_histstrat = calc_weights_by_ens_size(ea_histstrat)  
  wgt_histaer   = calc_weights_by_ens_size(ea_histaer)  
  wgt_histghg   = calc_weights_by_ens_size(ea_histghg)  
  wgt_histnat   = calc_weights_by_ens_size(ea_histnat)  

  em_nam_hist_djfA = calc_ensmean_1d(nam_hist_djfA,ea_hist)    ; form ensemble means, and keep track of how many ensemble members (returned in @num_ensemble_members)
  em_sam_hist_djfA_0 = calc_ensmean_1d(sam_hist_djfA_0,ea_hist) 
  em_sam_hist_djfA_1 = calc_ensmean_1d(sam_hist_djfA_1,ea_hist)
  em_nam_hist_jjaA = calc_ensmean_1d(nam_hist_jjaA,ea_hist)
  em_sam_hist_jjaA_0 = calc_ensmean_1d(sam_hist_jjaA_0,ea_hist)
  em_sam_hist_jjaA_1 = calc_ensmean_1d(sam_hist_jjaA_1,ea_hist)

  em_nam_histstrat_djfA = calc_ensmean_1d(nam_histstrat_djfA,ea_histstrat)
  em_sam_histstrat_djfA_0 = calc_ensmean_1d(sam_histstrat_djfA_0,ea_histstrat)
  em_sam_histstrat_djfA_1 = calc_ensmean_1d(sam_histstrat_djfA_1,ea_histstrat)
  em_nam_histstrat_jjaA = calc_ensmean_1d(nam_histstrat_jjaA,ea_histstrat)
  em_sam_histstrat_jjaA_0 = calc_ensmean_1d(sam_histstrat_jjaA_0,ea_histstrat)
  em_sam_histstrat_jjaA_1 = calc_ensmean_1d(sam_histstrat_jjaA_1,ea_histstrat)

  em_nam_histaer_djfA = calc_ensmean_1d(nam_histaer_djfA,ea_histaer)
  em_sam_histaer_djfA_0 = calc_ensmean_1d(sam_histaer_djfA_0,ea_histaer)
  em_sam_histaer_djfA_1 = calc_ensmean_1d(sam_histaer_djfA_1,ea_histaer)
  em_nam_histaer_jjaA = calc_ensmean_1d(nam_histaer_jjaA,ea_histaer)
  em_sam_histaer_jjaA_0 = calc_ensmean_1d(sam_histaer_jjaA_0,ea_histaer)
  em_sam_histaer_jjaA_1 = calc_ensmean_1d(sam_histaer_jjaA_1,ea_histaer)

  em_nam_histghg_djfA = calc_ensmean_1d(nam_histghg_djfA,ea_histghg)
  em_sam_histghg_djfA_0 = calc_ensmean_1d(sam_histghg_djfA_0,ea_histghg)
  em_sam_histghg_djfA_1 = calc_ensmean_1d(sam_histghg_djfA_1,ea_histghg)
  em_nam_histghg_jjaA = calc_ensmean_1d(nam_histghg_jjaA,ea_histghg)
  em_sam_histghg_jjaA_0 = calc_ensmean_1d(sam_histghg_jjaA_0,ea_histghg)
  em_sam_histghg_jjaA_1 = calc_ensmean_1d(sam_histghg_jjaA_1,ea_histghg)

  em_nam_histnat_djfA = calc_ensmean_1d(nam_histnat_djfA,ea_histnat)
  em_sam_histnat_djfA_0 = calc_ensmean_1d(sam_histnat_djfA_0,ea_histnat)
  em_sam_histnat_djfA_1 = calc_ensmean_1d(sam_histnat_djfA_1,ea_histnat)
  em_nam_histnat_jjaA = calc_ensmean_1d(nam_histnat_jjaA,ea_histnat)
  em_sam_histnat_jjaA_0 = calc_ensmean_1d(sam_histnat_jjaA_0,ea_histnat)
  em_sam_histnat_jjaA_1 = calc_ensmean_1d(sam_histnat_jjaA_1,ea_histnat)
;-------------------------------------------------------------------------
  nobs       = num(.not.ismissing(nam_obs_djfA))
  nhist      = count_unique_values(ea_hist)+" models, "+num(.not.ismissing(nam_hist_djfA))+" runs"   ;dimsizes(nam_hist_djfA)
  nctl       = num(.not.ismissing(nam_ctl_djfA))+" models, "+num(.not.ismissing(nam_ctl_djfA))+" segments"  ;dimsizes(nam_ctl_djfA)

  nmod_crit = 3  ; 3 or more models needed to count for box_whisker_2
  
  tarr = set_min_numEMsize(em_nam_hist_djfA,nmod_crit) 
;  print(tarr+" "+tarr@num_ensemble_members)
  nhistF      = num(.not.ismissing(tarr))+" models, "+sum(tarr@num_ensemble_members)+" runs"  
  tarr := set_min_numEMsize(em_nam_histstrat_djfA,nmod_crit) 
  nhiststrat = num(.not.ismissing(tarr))+" models, "+sum(tarr@num_ensemble_members)+" runs"
  tarr := set_min_numEMsize(em_nam_histaer_djfA,nmod_crit) 
  nhistaer   = num(.not.ismissing(tarr))+" models, "+sum(tarr@num_ensemble_members)+" runs"
  tarr := set_min_numEMsize(em_nam_histghg_djfA,nmod_crit) 
  nhistghg   = num(.not.ismissing(tarr))+" models, "+sum(tarr@num_ensemble_members)+" runs"
  tarr := set_min_numEMsize(em_nam_histnat_djfA,nmod_crit) 
  nhistnat   = num(.not.ismissing(tarr))+" models, "+sum(tarr@num_ensemble_members)+" runs"
;-------------------------------------------------------------------------
; values are in units of per trend period, will convert
; to be in terms of per decade

  wgt_ctl := new(dimsizes(nam_ctl_djfA),float)   ; set weights for piControl (= to 1)
  wgt_ctl = 1

  zarrN = new((/14,5/),float)
  zarrN(0,:) = (/ box_whisker_1b(nam_hist_djfA,wgt_hist) /)  
  zarrN(1,:) = (/ box_whisker_1b(nam_ctl_djfA,wgt_ctl) /)  
  zarrN(2,:) = (/ box_whisker_2(em_nam_hist_djfA,nmod_crit) /)
  zarrN(3,:) = (/ box_whisker_2(em_nam_histghg_djfA,nmod_crit) /)
  zarrN(4,:) = (/ box_whisker_2(em_nam_histaer_djfA,nmod_crit) /)
  zarrN(5,:) = (/ box_whisker_2(em_nam_histstrat_djfA,nmod_crit) /)
  zarrN(6,:) = (/ box_whisker_2(em_nam_histnat_djfA,nmod_crit) /)  
  
  zarrN(7,:)  = (/ box_whisker_1b(nam_hist_jjaA,wgt_hist) /)  
  zarrN(8,:) = (/ box_whisker_1b(nam_ctl_jjaA,wgt_ctl) /)  
  zarrN(9,:) = (/ box_whisker_2(em_nam_hist_jjaA,nmod_crit) /)
  zarrN(10,:) = (/ box_whisker_2(em_nam_histghg_jjaA,nmod_crit) /)
  zarrN(11,:) = (/ box_whisker_2(em_nam_histaer_jjaA,nmod_crit) /)
  zarrN(12,:) = (/ box_whisker_2(em_nam_histstrat_jjaA,nmod_crit) /)
  zarrN(13,:) = (/ box_whisker_2(em_nam_histnat_jjaA,nmod_crit) /)  

  zarrS0 = new((/14,5/),float)
  zarrS0(0,:) = (/ box_whisker_1b(sam_hist_djfA_0,wgt_hist) /)  
  zarrS0(1,:) = (/ box_whisker_1b(sam_ctl_djfA_0,wgt_ctl) /)  
  zarrS0(2,:) = (/ box_whisker_2(em_sam_hist_djfA_0,nmod_crit) /)
  zarrS0(3,:) = (/ box_whisker_2(em_sam_histghg_djfA_0,nmod_crit) /)
  zarrS0(4,:) = (/ box_whisker_2(em_sam_histaer_djfA_0,nmod_crit) /)
  zarrS0(5,:) = (/ box_whisker_2(em_sam_histstrat_djfA_0,nmod_crit) /)
  zarrS0(6,:) = (/ box_whisker_2(em_sam_histnat_djfA_0,nmod_crit) /)  
  
  zarrS0(7,:)  = (/ box_whisker_1b(sam_hist_jjaA_0,wgt_hist) /)  
  zarrS0(8,:) = (/ box_whisker_1b(sam_ctl_jjaA_0,wgt_ctl) /)  
  zarrS0(9,:) = (/ box_whisker_2(em_sam_hist_jjaA_0,nmod_crit) /)
  zarrS0(10,:) = (/ box_whisker_2(em_sam_histghg_jjaA_0,nmod_crit) /)
  zarrS0(11,:) = (/ box_whisker_2(em_sam_histaer_jjaA_0,nmod_crit) /)
  zarrS0(12,:) = (/ box_whisker_2(em_sam_histstrat_jjaA_0,nmod_crit) /)
  zarrS0(13,:) = (/ box_whisker_2(em_sam_histnat_jjaA_0,nmod_crit) /)  

  zarrS1 = new((/14,5/),float)
  zarrS1(0,:) = (/ box_whisker_1b(sam_hist_djfA_1,wgt_hist) /)  
  zarrS1(1,:) = (/ box_whisker_1b(sam_ctl_djfA_1,wgt_ctl) /)  
  zarrS1(2,:) = (/ box_whisker_2(em_sam_hist_djfA_1,nmod_crit) /)
  zarrS1(3,:) = (/ box_whisker_2(em_sam_histghg_djfA_1,nmod_crit) /)
  zarrS1(4,:) = (/ box_whisker_2(em_sam_histaer_djfA_1,nmod_crit) /)
  zarrS1(5,:) = (/ box_whisker_2(em_sam_histstrat_djfA_1,nmod_crit) /)
  zarrS1(6,:) = (/ box_whisker_2(em_sam_histnat_djfA_1,nmod_crit) /)  
  
  zarrS1(7,:)  = (/ box_whisker_1b(sam_hist_jjaA_1,wgt_hist) /)  
  zarrS1(8,:) = (/ box_whisker_1b(sam_ctl_jjaA_1,wgt_ctl) /)  
  zarrS1(9,:) = (/ box_whisker_2(em_sam_hist_jjaA_1,nmod_crit) /)
  zarrS1(10,:) = (/ box_whisker_2(em_sam_histghg_jjaA_1,nmod_crit) /)
  zarrS1(11,:) = (/ box_whisker_2(em_sam_histaer_jjaA_1,nmod_crit) /)
  zarrS1(12,:) = (/ box_whisker_2(em_sam_histstrat_jjaA_1,nmod_crit) /)
  zarrS1(13,:) = (/ box_whisker_2(em_sam_histnat_jjaA_1,nmod_crit) /)  

  zarrN = (/ (zarrN/nam_yrs)*10 /)
  zarrS0 = (/ (zarrS0/sam_yrs0)*10 /)  
  zarrS1 = (/ (zarrS1/sam_yrs1)*10 /)

  nam_obs_djfA = (/ (nam_obs_djfA/nam_yrs)*10 /)
  nam_obs_jjaA = (/ (nam_obs_jjaA/nam_yrs)*10 /)
  sam_obs_djfA_0 = (/ (sam_obs_djfA_0/sam_yrs0)*10 /)
  sam_obs_jjaA_0 = (/ (sam_obs_jjaA_0/sam_yrs0)*10 /)
  sam_obs_djfA_1 = (/ (sam_obs_djfA_1/sam_yrs1)*10 /)
  sam_obs_jjaA_1 = (/ (sam_obs_jjaA_1/sam_yrs1)*10 /)
;========================================================================================

  ofile = plot_dir + "/nam_sam_trends_damip"
  wks = gsn_open_wks(file_type, ofile)

  bres            = True                         ; plot mods desired
  bres@vpWidthF = 0.6
  bres@vpHeightF = 0.45
  bres@trXMinF = -0.5
  bres@trXMaxF = 16.
  bres@tmXBLabels = (/"","","","","","","","","","","","","",""/) ; labels for each box  
  bres@tmXBOn = False
  bres@tmXBLabelsOn = True
  bres@tmYLLabelFontHeightF = 0.017
  bres@tiMainFont = "Helvetica"
  bres@tiMainString = ""
  bres@tiMainFontHeightF = 0.0225
  bres@tiMainOffsetYF = -0.03
  llres                   = True			
  llres@gsLineThicknessF  = 1.25                 ; line thickness 
  llres@gsLineDashPattern = 0   ;5
  opti          = True			
  opti@boxWidth = .6				; Width of box (x units)

  cmap = new((/16,3/),float)
  cmap(0,:) =  (/255,255,255/)
  cmap(1,:) = (/0,0,0/)
  cmap(2,:) = (/253, 141, 60/)   ; dark orange, misc_div_21  amip
  cmap(3,:) = (/211,15,31/)      ; dark red, misc_div_21   historical
  cmap(4,:) = (/125, 79, 20/)    ; brown, prec_div (continuous)  hist-GHG
  cmap(5,:) = (/131,183,214/)    ; dark cyan, temp_div (continuous)  hist-aer
  cmap(6,:) = (/153,112,171/)    ; purple, chem_div_21  hist-stratO3
  cmap(7,:) = (/60, 132, 26/)    ; green, wind_seq (continuous)  hist-nat
  cmap(8,:) = (/25,51,178/)      ; dark blue, misc_seq_1_21 piControl
  cmap(9:,:) = cmap(2:8,:)
  cmap = cmap/255.
  gsn_define_colormap(wks,cmap)
;  opti@boxColors = ispan(2,15,1)
;  boxColors = (/2,3,8,3,4,5,6,7/)
  boxColors1 = (/3,8,3,8/)
  boxColors2 = (/3,4,5,6,7,3,4,5,6,7/)

  txres = True
  txres@txFontHeightF = 0.020
  plres = True
  plres@gsLineColor = "gray15"  
  plres@gsLineThicknessF = 0.5
  plres@tfPolyDrawOrder = "PostDraw"
  plres@gsLineDashPattern = 2

  plres2 = True
  plres2@gsLineColor = "gray15"
  plres2@gsLineThicknessF = 1.
  plres2@tfPolyDrawOrder = "PostDraw"

  plres3 = True
  plres3@gsLineThicknessF = 2.0
  plres3@gsLineColor = "gray50"

  plot_blank = new(4,graphic)
  bres@trYMinF = -0.3
  bres@trYMaxF = 0.3
  bres@tmYLMode = "Explicit"
  bres@tmYLValues = (/-.4,-.3,-.2,-.1,0,.1,.2,.3,.4/)
  bres@tmYLLabels = (/"-0.4","-0.3","-0.2","-0.1","0","0.1","0.2","0.3","0.4"/)
  bres@tiYAxisString = "(decade~S~-1~N~)"
  bres@tiMainFontColor = "white"
  bres@tiYAxisFontColor = "white"
  bres@tmYLLabelFontColor = "white"
  bres@tiMainString = "(a) NAM ("+syear_a+"-"+eyear_a+")"
  plot_blank(1) = blank_boxplot(wks,bres)
  bres@tmYLValues := (/-1.6,-1.2,-0.8,-0.4,0.,0.4,0.8,1.2,1.6/) ; (/-2.,-1.5,-1.,-0.5,0.,0.5,1,1.5,2./)
  bres@tmYLLabels := bres@tmYLValues
  bres@trYMinF = -1.2
  bres@trYMaxF = 1.2
  bres@tiMainString = "(b) SAM ("+syear_b+"-"+eyear_b+")"
  plot_blank(2) = blank_boxplot(wks,bres)

  bres@tiMainString = "(c) SAM ("+syear_c+"-"+eyear_c+")"
  plot_blank(3) = blank_boxplot(wks,bres)

;  print("NAM DJF Obs: "+nam_obs_djfA)
;  print("NAM JJA Obs: "+nam_obs_jjaA)
;  print("SAM DJF Obs-0: "+sam_obs_djfA_0)
;  print("SAM JJA Obs-0: "+sam_obs_jjaA_0)
;  print("SAM DJF Obs-1: "+sam_obs_djfA_1)
;  print("SAM JJA Obs-1: "+sam_obs_jjaA_1)

  dum = new(24,graphic)

  cntr = 0
  do gg = 0,nobs-1
     plres3@gsLineDashPattern = gg
     dum(cntr) = gsn_add_polyline(wks,plot_blank(1),(/0.5,2.5/),(/nam_obs_djfA(gg),nam_obs_djfA(gg)/),plres3)
     dum(cntr+1) = gsn_add_polyline(wks,plot_blank(1),(/8.5,10.5/),(/nam_obs_jjaA(gg),nam_obs_jjaA(gg)/),plres3)

     dum(cntr+2) = gsn_add_polyline(wks,plot_blank(2),(/0.5,2.5/),(/sam_obs_djfA_0(gg),sam_obs_djfA_0(gg)/),plres3)
     dum(cntr+3) = gsn_add_polyline(wks,plot_blank(2),(/8.5,10.5/),(/sam_obs_jjaA_0(gg),sam_obs_jjaA_0(gg)/),plres3)

     dum(cntr+4) = gsn_add_polyline(wks,plot_blank(3),(/0.5,2.5/),(/sam_obs_djfA_1(gg),sam_obs_djfA_1(gg)/),plres3)
     dum(cntr+5) = gsn_add_polyline(wks,plot_blank(3),(/8.5,10.5/),(/sam_obs_jjaA_1(gg),sam_obs_jjaA_1(gg)/),plres3)
     cntr = cntr+6
  end do

  panres = True
  panres@txFontHeightF = 0.022
  panres@gsnPanelXWhiteSpacePercent = 5.0
  panres@gsnPanelYWhiteSpacePercent = 3.0
  panres@gsnFrame = False
  gsn_panel(wks,plot_blank,(/2,2/),panres)

;--- Create 1st panel-----
  bplot = new(4,graphic)
  bplot2 = bplot
  bres@tiMainFontColor = "black"
  bres@tiYAxisFontColor = "black"
  bres@tmYLLabelFontColor = "black"
  bres@trYMinF = -0.3
  bres@trYMaxF = 0.3
  bres@tmYLValues := (/-.4,-.3,-.2,-.1,0,.1,.2,.3,.4/)
  bres@tmYLLabels := (/"-0.4","-0.3","-0.2","-0.1","0","0.1","0.2","0.3","0.4"/)
  bres@tiMainString = "(a) NAM ("+syear_a+"-"+eyear_a+")"

  bres@tmXBLabels := (/"","","",""/)
  opti@boxColors  := boxColors1
  bplot(1) = boxplot2(wks,(/1,2,9,10/),zarrN((/0,1,7,8/),:),opti,bres,llres)	


  dum2 = new(4,graphic)
  dum3 = dum
  dum4 = dum

  bres@tmXBLabels := (/"","","","","","","","","",""/)

  opti@boxFill_midColor = "white"
  opti@boxColors := boxColors2
  opti@boxFill_Colors := opti@boxColors
  bplot2(1) = boxplot2(wks,(/3,4,5,6,7,11,12,13,14,15/),zarrN((/2,3,4,5,6,9,10,11,12,13/),:),opti,bres,llres)	

  dum2(0) = gsn_add_text(wks,bplot2(1),"DJF",4.,bres@trYMinF+0.0425,txres)
  dum2(1) = gsn_add_text(wks,bplot2(1),"JJA",12.,bres@trYMinF+0.0425,txres)
  dum2(2) = gsn_add_polyline(wks,bplot2(1),(/-15,15.75/),(/0.,0./),plres2)
  dum2(3) = gsn_add_polyline(wks,bplot2(1),(/8,8/),(/-2.,bres@trYMaxF*0.85/),plres)
  overlay(bplot(1),bplot2(1))
;--- Create 2nd panel-----

  bres@tmXBLabels := (/"","","",""/)
  bres@tmYLValues := (/-1.6,-1.2,-0.8,-0.4,0.,0.4,0.8,1.2,1.6/) ; (/-2.,-1.5,-1.,-0.5,0.,0.5,1.,1.5,2./)
  bres@tmYLLabels := bres@tmYLValues
  bres@trYMinF = -1.2
  bres@trYMaxF = 1.2
  bres@tiMainString = "(b) SAM ("+syear_b+"-"+eyear_b+")"
  delete([/opti@boxFill_Colors, opti@boxFill_midColor/])
  opti@boxColors  := boxColors1
  bplot(2) = boxplot2(wks,(/1,2,9,10/),zarrS0((/0,1,7,8/),:),opti,bres,llres)	

  bres@tmXBLabels := (/"","","","","","","","","",""/)

  opti@boxFill_midColor = "white"
  opti@boxColors := boxColors2
  opti@boxFill_Colors := opti@boxColors
  bplot2(2) = boxplot2(wks,(/3,4,5,6,7,11,12,13,14,15/),zarrS0((/2,3,4,5,6,9,10,11,12,13/),:),opti,bres,llres)	

  dum3(0) = gsn_add_text(wks,bplot2(2),"DJF",4.,bres@trYMinF+0.14,txres)
  dum3(1) = gsn_add_text(wks,bplot2(2),"JJA",12.,bres@trYMinF+0.14,txres)
  dum3(2) = gsn_add_polyline(wks,bplot2(2),(/-15,15.75/),(/0.,0./),plres2)
  dum3(3) = gsn_add_polyline(wks,bplot2(2),(/8,8/),(/-3.,bres@trYMaxF*0.85/),plres)
  overlay(bplot(2),bplot2(2))

;--- Create 3rd panel-----

  bres@tmXBLabels := (/"","","",""/)
  bres@tiMainString = "(c) SAM ("+syear_c+"-"+eyear_c+")"
  delete([/opti@boxFill_Colors,opti@boxFill_midColor/])
  opti@boxColors  := boxColors1
  bplot(3) = boxplot2(wks,(/1,2,9,10/),zarrS1((/0,1,7,8/),:),opti,bres,llres)	

  bres@tmXBLabels := (/"","","","","","","","","",""/)

  opti@boxFill_midColor = "white"
  opti@boxColors := boxColors2
  opti@boxFill_Colors := opti@boxColors
  bplot2(3) = boxplot2(wks,(/3,4,5,6,7,11,12,13,14,15/),zarrS1((/2,3,4,5,6,9,10,11,12,13/),:),opti,bres,llres)	

  dum4(0) = gsn_add_text(wks,bplot2(3),"DJF",4.,bres@trYMinF+0.14,txres)
  dum4(1) = gsn_add_text(wks,bplot2(3),"JJA",12.,bres@trYMinF+0.14,txres)
  dum4(2) = gsn_add_polyline(wks,bplot2(3),(/-15,15.75/),(/0.,0./),plres2)
  dum4(3) = gsn_add_polyline(wks,bplot2(3),(/8,8/),(/-3.,bres@trYMaxF*0.85/),plres)
  overlay(bplot(3),bplot2(3))
;---------------------------------------------
; Add ensemble means (as dots)

  polymres = True
  polymres@gsMarkerIndex     = 16          ; polymarker style
  polymres@gsMarkerSizeF     = 4.          ; polymarker size
  polymres@tfPolyDrawOrder = "PostDraw"

  dum5 = new(10,graphic)
  xarr := new(dimsizes(em_nam_hist_djfA),float)
  xarr = 3
  dum5(0) = gsn_add_polymarker(wks,bplot2(1),xarr,(set_min_numEMsize(em_nam_hist_djfA,nmod_crit)/nam_yrs)*10.,polymres)
  xarr := new(dimsizes(em_nam_histghg_djfA),float)
  xarr = 4
  dum5(1) = gsn_add_polymarker(wks,bplot2(1),xarr,(set_min_numEMsize(em_nam_histghg_djfA,nmod_crit)/nam_yrs)*10.,polymres)
  xarr := new(dimsizes(em_nam_histaer_djfA),float)
  xarr = 5
  dum5(2) = gsn_add_polymarker(wks,bplot2(1),xarr,(set_min_numEMsize(em_nam_histaer_djfA,nmod_crit)/nam_yrs)*10.,polymres)
  xarr := new(dimsizes(em_nam_histstrat_djfA),float)
  xarr = 6
  dum5(3) = gsn_add_polymarker(wks,bplot2(1),xarr,(set_min_numEMsize(em_nam_histstrat_djfA,nmod_crit)/nam_yrs)*10.,polymres)
  xarr := new(dimsizes(em_nam_histnat_djfA),float)
  xarr = 7
  dum5(4) = gsn_add_polymarker(wks,bplot2(1),xarr,(set_min_numEMsize(em_nam_histnat_djfA,nmod_crit)/nam_yrs)*10.,polymres)

  xarr := new(dimsizes(em_nam_hist_jjaA),float)
  xarr = 11
  dum5(5) = gsn_add_polymarker(wks,bplot2(1),xarr,(set_min_numEMsize(em_nam_hist_jjaA,nmod_crit)/nam_yrs)*10.,polymres)
  xarr := new(dimsizes(em_nam_histghg_jjaA),float)
  xarr = 12
  dum5(6) = gsn_add_polymarker(wks,bplot2(1),xarr,(set_min_numEMsize(em_nam_histghg_jjaA,nmod_crit)/nam_yrs)*10.,polymres)
  xarr := new(dimsizes(em_nam_histaer_jjaA),float)
  xarr = 13
  dum5(7) = gsn_add_polymarker(wks,bplot2(1),xarr,(set_min_numEMsize(em_nam_histaer_jjaA,nmod_crit)/nam_yrs)*10.,polymres)
  xarr := new(dimsizes(em_nam_histstrat_jjaA),float)
  xarr = 14
  dum5(8) = gsn_add_polymarker(wks,bplot2(1),xarr,(set_min_numEMsize(em_nam_histstrat_jjaA,nmod_crit)/nam_yrs)*10.,polymres)
  xarr := new(dimsizes(em_nam_histnat_jjaA),float)
  xarr = 15
  dum5(9) = gsn_add_polymarker(wks,bplot2(1),xarr,(set_min_numEMsize(em_nam_histnat_jjaA,nmod_crit)/nam_yrs)*10.,polymres)

; - - - Panel 2 - - - 
  dum6 = new(10,graphic)
  xarr := new(dimsizes(em_sam_hist_djfA_0),float)
  xarr = 3
  dum6(0) = gsn_add_polymarker(wks,bplot2(2),xarr,(set_min_numEMsize(em_sam_hist_djfA_0,nmod_crit)/sam_yrs0)*10.,polymres)
  xarr := new(dimsizes(em_sam_histghg_djfA_0),float)
  xarr = 4
  dum6(1) = gsn_add_polymarker(wks,bplot2(2),xarr,(set_min_numEMsize(em_sam_histghg_djfA_0,nmod_crit)/sam_yrs0)*10.,polymres)
  xarr := new(dimsizes(em_sam_histaer_djfA_0),float)
  xarr = 5
  dum6(2) = gsn_add_polymarker(wks,bplot2(2),xarr,(set_min_numEMsize(em_sam_histaer_djfA_0,nmod_crit)/sam_yrs0)*10.,polymres)
  xarr := new(dimsizes(em_sam_histstrat_djfA_0),float)
  xarr = 6
  dum6(3) = gsn_add_polymarker(wks,bplot2(2),xarr,(set_min_numEMsize(em_sam_histstrat_djfA_0,nmod_crit)/sam_yrs0)*10.,polymres)
  xarr := new(dimsizes(em_sam_histnat_djfA_0),float)
  xarr = 7
  dum6(4) = gsn_add_polymarker(wks,bplot2(2),xarr,(set_min_numEMsize(em_sam_histnat_djfA_0,nmod_crit)/sam_yrs0)*10.,polymres)
  xarr := new(dimsizes(em_sam_hist_jjaA_0),float)
  xarr = 11
  dum6(5) = gsn_add_polymarker(wks,bplot2(2),xarr,(set_min_numEMsize(em_sam_hist_jjaA_0,nmod_crit)/sam_yrs0)*10.,polymres)
  xarr := new(dimsizes(em_sam_histghg_jjaA_0),float)
  xarr = 12
  dum6(6) = gsn_add_polymarker(wks,bplot2(2),xarr,(set_min_numEMsize(em_sam_histghg_jjaA_0,nmod_crit)/sam_yrs0)*10.,polymres)
  xarr := new(dimsizes(em_sam_histaer_jjaA_0),float)
  xarr = 13
  dum6(7) = gsn_add_polymarker(wks,bplot2(2),xarr,(set_min_numEMsize(em_sam_histaer_jjaA_0,nmod_crit)/sam_yrs0)*10.,polymres)
  xarr := new(dimsizes(em_sam_histstrat_jjaA_0),float)
  xarr = 14
  dum6(8) = gsn_add_polymarker(wks,bplot2(2),xarr,(set_min_numEMsize(em_sam_histstrat_jjaA_0,nmod_crit)/sam_yrs0)*10.,polymres)
  xarr := new(dimsizes(em_sam_histnat_jjaA_0),float)
  xarr = 15
  dum6(9) = gsn_add_polymarker(wks,bplot2(2),xarr,(set_min_numEMsize(em_sam_histnat_jjaA_0,nmod_crit)/sam_yrs0)*10.,polymres)

; - - - Panel 3 - - - 
  dum7 = new(10,graphic)
  xarr := new(dimsizes(em_sam_hist_djfA_1),float)
  xarr = 3
  dum7(0) = gsn_add_polymarker(wks,bplot2(3),xarr,(set_min_numEMsize(em_sam_hist_djfA_1,nmod_crit)/sam_yrs1)*10.,polymres)
  xarr := new(dimsizes(em_sam_histghg_djfA_1),float)
  xarr = 4
  dum7(1) = gsn_add_polymarker(wks,bplot2(3),xarr,(set_min_numEMsize(em_sam_histghg_djfA_1,nmod_crit)/sam_yrs1)*10.,polymres)
  xarr := new(dimsizes(em_sam_histaer_djfA_1),float)
  xarr = 5
  dum7(2) = gsn_add_polymarker(wks,bplot2(3),xarr,(set_min_numEMsize(em_sam_histaer_djfA_1,nmod_crit)/sam_yrs1)*10.,polymres)
  xarr := new(dimsizes(em_sam_histstrat_djfA_1),float)
  xarr = 6
  dum7(3) = gsn_add_polymarker(wks,bplot2(3),xarr,(set_min_numEMsize(em_sam_histstrat_djfA_1,nmod_crit)/sam_yrs1)*10.,polymres)
  xarr := new(dimsizes(em_sam_histnat_djfA_1),float)
  xarr = 7
  dum7(4) = gsn_add_polymarker(wks,bplot2(3),xarr,(set_min_numEMsize(em_sam_histnat_djfA_1,nmod_crit)/sam_yrs1)*10.,polymres)

  xarr := new(dimsizes(em_sam_hist_jjaA_1),float)
  xarr = 11
  dum7(5) = gsn_add_polymarker(wks,bplot2(3),xarr,(set_min_numEMsize(em_sam_hist_jjaA_1,nmod_crit)/sam_yrs1)*10.,polymres)
  xarr := new(dimsizes(em_sam_histghg_jjaA_1),float)
  xarr = 12
  dum7(6) = gsn_add_polymarker(wks,bplot2(3),xarr,(set_min_numEMsize(em_sam_histghg_jjaA_1,nmod_crit)/sam_yrs1)*10.,polymres)
  xarr := new(dimsizes(em_sam_histaer_jjaA_1),float)
  xarr = 13
  dum7(7) = gsn_add_polymarker(wks,bplot2(3),xarr,(set_min_numEMsize(em_sam_histaer_jjaA_1,nmod_crit)/sam_yrs1)*10.,polymres)
  xarr := new(dimsizes(em_sam_histstrat_jjaA_1),float)
  xarr = 14
  dum7(8) = gsn_add_polymarker(wks,bplot2(3),xarr,(set_min_numEMsize(em_sam_histstrat_jjaA_1,nmod_crit)/sam_yrs1)*10.,polymres)
  xarr := new(dimsizes(em_sam_histnat_jjaA_1),float)
  xarr = 15
  dum7(9) = gsn_add_polymarker(wks,bplot2(3),xarr,(set_min_numEMsize(em_sam_histnat_jjaA_1,nmod_crit)/sam_yrs1)*10.,polymres)

;---------------------------------------------
; Create legend
;
  gres = True
  gres@YPosPercent = 64.5    ; expressed as %, 0->100, sets position of top border of legend 
                            ;  when gres@Position is set to its default setting of "Top" (Default = 95.)
  gres@XPosPercent = 19.      ; expressed as %, 0->100, sets position of left border of legend(Default = 5.)
  gres@LineLabelWhiteSpacePercent = 1.
  gres@ItemSpacePercent = 2.

  lineres = True
  lineres@lgLineColors = boxColors1(:1)   ; line colors
  lineres@lgLineThicknesses = 2.                       ; line thicknesses
  lineres@LineLengthPercent = 5.                         ; expressed as %, 0->100, length of line
  
  textres = True
;  textres@lgLabels = (/"Historical ("+nhist+")","piControl ("+nctl+")"/)  ; legend labels (required)
  textres@lgLabelFontHeights = 0.010
;  simple_legend_ndc(wks,gres,lineres,textres)

;  gres@YPosPercent = 59.5 
;  textres@lgLabels := (/"Hist-GHG ("+nhistghg+")", "Hist-aer ("+nhistaer+")","Hist-stratO3 ("+nhiststrat+")","Hist-nat ("+nhistnat+")"/)
;  lineres@lgLineColors := boxColors2(1:4)
;  simple_legend_ndc(wks,gres,lineres,textres)

  gres@YPosPercent = 72.
  obsColors = (/plres3@gsLineColor,plres3@gsLineColor,plres3@gsLineColor/)
  obsIndexes = (/0,1,2/)
  lineres@lgLineColors := obsColors(0:nobs-1)
  lineres@lgDashIndexes := obsIndexes(0:nobs-1)

  textres@lgLabels := where(obs_datasets.eq."NCEP", "NCEP/NCAR", obs_datasets)
  simple_legend_ndc(wks,gres,lineres,textres)

  txres@txFontHeightF = 0.010
  txres@txJust = "CenterLeft"
  txres@txFontColor = boxColors1(0)
  gsn_text_ndc(wks,"Historical (unfilled box, "+nhist+")",gres@XPosPercent/100.,.66,txres)
  txres@txFontColor = boxColors1(1)
  gsn_text_ndc(wks,"piControl ("+nctl+")",gres@XPosPercent/100.,.64,txres)

  txres@txFontColor = boxColors1(0)
  gsn_text_ndc(wks,"Historical (filled box, "+nhistF+")",gres@XPosPercent/100.,.60,txres)
  txres@txFontColor = boxColors2(1)
  gsn_text_ndc(wks,"Hist-GHG ("+nhistghg+")",gres@XPosPercent/100.,.58,txres)
  txres@txFontColor = boxColors2(2)
  gsn_text_ndc(wks,"Hist-aer ("+nhistaer+")",gres@XPosPercent/100.,.56,txres)
  txres@txFontColor = boxColors2(3)
  gsn_text_ndc(wks,"Hist-stratO3 ("+nhiststrat+")",gres@XPosPercent/100.,.54,txres)
  txres@txFontColor = boxColors2(4)
  gsn_text_ndc(wks,"Hist-nat ("+nhistnat+")",gres@XPosPercent/100.,.52,txres)

  txres@txFont = 22
  txres@txFontHeightF = 0.018
  txres@txFontColor := "black"
  gsn_text_ndc(wks,"  Attribution of observed seasonal~C~trends of annular modes to forcings",0.05,0.78,txres)

  bres@trYMinF = -1.   ; set up box plot legend 
  bres@trYMaxF = 1.
  bres@trXMinF = 0.
  bres@trXMaxF = 2.
  bres@tmYLOn = False
  bres@tmXBOn = False
  bres@tmYLBorderOn = False
  bres@tmXBBorderOn = False
  bres@vpXF = .12
  bres@vpYF = .71
  bres@vpWidthF = .055
  bres@vpHeightF = .125
  bres@tiYAxisOn = False
  bres@tiMainOn = False
  bres@tmXBLabels := (/"",""/)
  opti@boxFill_Colors := (/"white","white"/)
  opti@boxFill_midColor := (/"gray50","gray50"/)
  opti@boxColors := (/"gray50","gray50"/)

  yval_lg = new((/2,5/),"float")
  yval_lg(0,:) = (/-.5,-.25,0,.25,.5/)
  bplg = boxplot2(wks,(/0.5,0.5/),yval_lg,opti,bres,llres)
  draw(bplg)
  bres@vpYF = .6217
  opti@boxFill_Colors := (/"gray50","gray50"/)
  opti@boxFill_midColor := (/"white","white"/)
  opti@boxColors := (/"gray50","gray50"/)        ; (/5,3/)  
  yval_lg(0,:) = (/-.25,-.25,0,.25,.25/)
  bplg2 = boxplot2(wks,(/0.5,0.5/),yval_lg,opti,bres,llres)
  draw(bplg2)

  tlres = True
  tlres@txFontHeightF = 0.007
  tlres@txJust = "CenterLeft"
  gsn_text_ndc(wks,"- 95th",.14,.6793,tlres)
  gsn_text_ndc(wks,"- 75th",.15,.6633,tlres)
  gsn_text_ndc(wks,"- Mean",.15,.6483,tlres)
  gsn_text_ndc(wks,"- 25th",.15,.6328,tlres)
  gsn_text_ndc(wks,"- 5th",.14,.6170,tlres)

  gsn_text_ndc(wks,"- 95%",.15,.5755,tlres)
  gsn_text_ndc(wks,"- Mean",.15,.56,tlres)
  gsn_text_ndc(wks,"- 5%",.15,.5445,tlres)

  tlres@txAngleF = 90
  tlres@txJust = "CenterCenter"
  gsn_text_ndc(wks,"Ensemble Range",.105,.6483,tlres)
  gsn_text_ndc(wks,"(all models)",.115,.6483,tlres)

  gsn_text_ndc(wks,"Confidence Interval",.095,.56,tlres)
  gsn_text_ndc(wks,"(models with ~F34~3~F21~ "+nmod_crit,.105,.56,tlres)
  gsn_text_ndc(wks,"runs only)",.115,.56,tlres)

  gsn_panel(wks,bplot,(/2,2/),panres)
  frame(wks)

; Output netcdf file
  if (config_user_info@write_netcdf) then
    out_dir =  config_user_info@work_dir + "/"
    system("mkdir -p "+out_dir)
    nc_outfile = out_dir+"NAM_SAM_detection_attribution.nc"
    if (fileexists(nc_outfile)) then
      system("rm -f "+nc_outfile)
    end if
    nc_outfile@existing = "append"

    nam_obs_djfA@diag_script = DIAG_SCRIPT
    nam_obs_djfA!0 = "obs_dataset"
    nam_obs_djfA@var = "NAM_obs_DJF_"+syear_a+"_"+eyear_a
    nam_obs_djfA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_obs_djfA, nc_outfile)
    sam_obs_djfA_0!0 = "obs_dataset"
    sam_obs_djfA_0@var = "SAM_obs_DJF_"+syear_b+"_"+eyear_b
    sam_obs_djfA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_obs_djfA_0, nc_outfile)
    sam_obs_djfA_1!0 = "obs_dataset"
    sam_obs_djfA_1@var = "SAM_obs_DJF_"+syear_c+"_"+eyear_c
    sam_obs_djfA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_obs_djfA_1, nc_outfile)

    nam_obs_jjaA!0 = "obs_dataset"
    nam_obs_jjaA@var = "NAM_obs_JJA_"+syear_a+"_"+eyear_a
    nam_obs_jjaA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_obs_jjaA, nc_outfile)
    sam_obs_jjaA_0!0 = "obs_dataset"
    sam_obs_jjaA_0@var = "SAM_obs_JJA_"+syear_b+"_"+eyear_b
    sam_obs_jjaA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_obs_jjaA_0, nc_outfile)
    sam_obs_jjaA_1!0 = "obs_dataset"
    sam_obs_jjaA_1@var = "SAM_obs_JJA_"+syear_c+"_"+eyear_c
    sam_obs_jjaA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_obs_jjaA_1, nc_outfile)

    nam_ctl_djfA = nam_ctl_djfA/nam_yrs*10
    nam_ctl_djfA!0 = "model_ctl"
    nam_ctl_djfA@var = "NAM_piControl_DJF_"+toint(nam_yrs)+"yrs"
    nam_ctl_djfA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_ctl_djfA, nc_outfile)
    sam_ctl_djfA_0 = sam_ctl_djfA_0/sam_yrs0*10
    sam_ctl_djfA_0!0 = "model_ctl"
    sam_ctl_djfA_0@var = "SAM_piControl_DJF_"+syear_b+"_"+eyear_b
    sam_ctl_djfA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_ctl_djfA_0, nc_outfile)
    sam_ctl_djfA_1 = sam_ctl_djfA_1/sam_yrs1*10
    sam_ctl_djfA_1!0 = "model_ctl"
    sam_ctl_djfA_1@var = "SAM_piControl_DJF_"+syear_c+"_"+eyear_c
    sam_ctl_djfA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_ctl_djfA_1, nc_outfile)

    nam_ctl_jjaA = nam_ctl_jjaA/nam_yrs*10
    nam_ctl_jjaA!0 = "model_ctl"
    nam_ctl_jjaA@var = "NAM_piControl_JJA_"+toint(nam_yrs)+"yrs"
    nam_ctl_jjaA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_ctl_jjaA, nc_outfile)
    sam_ctl_jjaA_0 = sam_ctl_jjaA_0/sam_yrs0*10
    sam_ctl_jjaA_0!0 = "model_ctl"
    sam_ctl_jjaA_0@var = "SAM_piControl_JJA_"+syear_b+"_"+eyear_b
    sam_ctl_jjaA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_ctl_jjaA_0, nc_outfile)
    sam_ctl_jjaA_1 = sam_ctl_jjaA_1/sam_yrs1*10
    sam_ctl_jjaA_1!0 = "model_ctl"
    sam_ctl_jjaA_1@var = "SAM_piControl_JJA_"+syear_c+"_"+eyear_c
    sam_ctl_jjaA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_ctl_jjaA_1, nc_outfile)

    nam_hist_djfA = nam_hist_djfA/nam_yrs*10
    nam_hist_djfA!0 = "ensemble_hist"
    nam_hist_djfA@var = "NAM_hist_DJF_"+syear_a+"_"+eyear_a
    nam_hist_djfA@weight = wgt_hist
    nam_hist_djfA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_hist_djfA, nc_outfile)
    sam_hist_djfA_0 = sam_hist_djfA_0/sam_yrs0*10
    sam_hist_djfA_0!0 = "ensemble_hist"
    sam_hist_djfA_0@var = "SAM_hist_DJF_"+syear_b+"_"+eyear_b
    sam_hist_djfA_0@weight = wgt_hist
    sam_hist_djfA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_hist_djfA_0, nc_outfile)
    sam_hist_djfA_1 = sam_hist_djfA_1/sam_yrs1*10
    sam_hist_djfA_1!0 = "ensemble_hist"
    sam_hist_djfA_1@var = "SAM_hist_DJF_"+syear_c+"_"+eyear_c
    sam_hist_djfA_1@weight = wgt_hist
    sam_hist_djfA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_hist_djfA_1, nc_outfile)

    nam_hist_jjaA = nam_hist_jjaA/nam_yrs*10
    nam_hist_jjaA!0 = "ensemble_hist"
    nam_hist_jjaA@var = "NAM_hist_JJA_"+syear_a+"_"+eyear_a
    nam_hist_jjaA@weight = wgt_hist
    nam_hist_jjaA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_hist_jjaA, nc_outfile)
    sam_hist_jjaA_0 = sam_hist_jjaA_0/sam_yrs0*10
    sam_hist_jjaA_0!0 = "ensemble_hist"
    sam_hist_jjaA_0@var = "SAM_hist_JJA_"+syear_b+"_"+eyear_b
    sam_hist_jjaA_0@weight = wgt_hist
    sam_hist_jjaA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_hist_jjaA_0, nc_outfile)
    sam_hist_jjaA_1 = sam_hist_jjaA_1/sam_yrs1*10
    sam_hist_jjaA_1!0 = "ensemble_hist"
    sam_hist_jjaA_1@var = "SAM_hist_JJA_"+syear_c+"_"+eyear_c
    sam_hist_jjaA_1@weight = wgt_hist
    sam_hist_jjaA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_hist_jjaA_1, nc_outfile)

    nam_histghg_djfA = nam_histghg_djfA/nam_yrs*10
    nam_histghg_djfA!0 = "ensemble_GHG"
    nam_histghg_djfA@var = "NAM_GHG_DJF_"+syear_a+"_"+eyear_a
    nam_histghg_djfA@weight = wgt_histghg
    nam_histghg_djfA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_histghg_djfA, nc_outfile)
    sam_histghg_djfA_0 = sam_histghg_djfA_0/sam_yrs0*10
    sam_histghg_djfA_0!0 = "ensemble_GHG"
    sam_histghg_djfA_0@var = "SAM_GHG_DJF_"+syear_b+"_"+eyear_b
    sam_histghg_djfA_0@weight = wgt_histghg
    sam_histghg_djfA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histghg_djfA_0, nc_outfile)
    sam_histghg_djfA_1 = sam_histghg_djfA_1/sam_yrs1*10
    sam_histghg_djfA_1!0 = "ensemble_GHG"
    sam_histghg_djfA_1@var = "SAM_GHG_DJF_"+syear_c+"_"+eyear_c
    sam_histghg_djfA_1@weight = wgt_histghg
    sam_histghg_djfA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histghg_djfA_1, nc_outfile)

    nam_histghg_jjaA = nam_histghg_jjaA/nam_yrs*10
    nam_histghg_jjaA!0 = "ensemble_GHG"
    nam_histghg_jjaA@var = "NAM_GHG_JJA_"+syear_a+"_"+eyear_a
    nam_histghg_jjaA@weight = wgt_histghg
    nam_histghg_jjaA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_histghg_jjaA, nc_outfile)
    sam_histghg_jjaA_0 = sam_histghg_jjaA_0/sam_yrs0*10
    sam_histghg_jjaA_0!0 = "ensemble_GHG"
    sam_histghg_jjaA_0@var = "SAM_GHG_JJA_"+syear_b+"_"+eyear_b
    sam_histghg_jjaA_0@weight = wgt_histghg
    sam_histghg_jjaA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histghg_jjaA_0, nc_outfile)
    sam_histghg_jjaA_1 = sam_histghg_jjaA_1/sam_yrs1*10
    sam_histghg_jjaA_1!0 = "ensemble_GHG"
    sam_histghg_jjaA_1@var = "SAM_GHG_JJA_"+syear_c+"_"+eyear_c
    sam_histghg_jjaA_1@weight = wgt_histghg
    sam_histghg_jjaA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histghg_jjaA_1, nc_outfile)
    
    nam_histaer_djfA = nam_histaer_djfA/nam_yrs*10
    nam_histaer_djfA!0 = "ensemble_aer"
    nam_histaer_djfA@var = "NAM_aer_DJF_"+syear_a+"_"+eyear_a
    nam_histaer_djfA@weight = wgt_histaer
    nam_histaer_djfA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_histaer_djfA, nc_outfile)
    sam_histaer_djfA_0 = sam_histaer_djfA_0/sam_yrs0*10
    sam_histaer_djfA_0!0 = "ensemble_aer"
    sam_histaer_djfA_0@var = "SAM_aer_DJF_"+syear_b+"_"+eyear_b
    sam_histaer_djfA_0@weight = wgt_histaer
    sam_histaer_djfA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histaer_djfA_0, nc_outfile)
    sam_histaer_djfA_1 = sam_histaer_djfA_1/sam_yrs1*10
    sam_histaer_djfA_1!0 = "ensemble_aer"
    sam_histaer_djfA_1@var = "SAM_aer_DJF_"+syear_c+"_"+eyear_c
    sam_histaer_djfA_1@weight = wgt_histaer
    sam_histaer_djfA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histaer_djfA_1, nc_outfile)

    nam_histaer_jjaA = nam_histaer_jjaA/nam_yrs*10
    nam_histaer_jjaA!0 = "ensemble_aer"
    nam_histaer_jjaA@var = "NAM_aer_JJA_"+syear_a+"_"+eyear_a
    nam_histaer_jjaA@weight = wgt_histaer
    nam_histaer_jjaA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_histaer_jjaA, nc_outfile)
    sam_histaer_jjaA_0 = sam_histaer_jjaA_0/sam_yrs0*10
    sam_histaer_jjaA_0!0 = "ensemble_aer"
    sam_histaer_jjaA_0@var = "SAM_aer_JJA_"+syear_b+"_"+eyear_b
    sam_histaer_jjaA_0@weight = wgt_histaer
    sam_histaer_jjaA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histaer_jjaA_0, nc_outfile)
    sam_histaer_jjaA_1 = sam_histaer_jjaA_1/sam_yrs1*10
    sam_histaer_jjaA_1!0 = "ensemble_aer"
    sam_histaer_jjaA_1@var = "SAM_aer_JJA_"+syear_c+"_"+eyear_c
    sam_histaer_jjaA_1@weight = wgt_histaer
    sam_histaer_jjaA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histaer_jjaA_1, nc_outfile)
    
    nam_histnat_djfA = nam_histnat_djfA/nam_yrs*10
    nam_histnat_djfA!0 = "ensemble_nat"
    nam_histnat_djfA@var = "NAM_nat_DJF_"+syear_a+"_"+eyear_a
    nam_histnat_djfA@weight = wgt_histnat
    nam_histnat_djfA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_histnat_djfA, nc_outfile)
    sam_histnat_djfA_0 = sam_histnat_djfA_0/sam_yrs0*10
    sam_histnat_djfA_0!0 = "ensemble_nat"
    sam_histnat_djfA_0@var = "SAM_nat_DJF_"+syear_b+"_"+eyear_b
    sam_histnat_djfA_0@weight = wgt_histnat
    sam_histnat_djfA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histnat_djfA_0, nc_outfile)
    sam_histnat_djfA_1 = sam_histnat_djfA_1/sam_yrs1*10
    sam_histnat_djfA_1!0 = "ensemble_nat"
    sam_histnat_djfA_1@var = "SAM_nat_DJF_"+syear_c+"_"+eyear_c
    sam_histnat_djfA_1@weight = wgt_histnat
    sam_histnat_djfA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histnat_djfA_1, nc_outfile)

    nam_histnat_jjaA = nam_histnat_jjaA/nam_yrs*10
    nam_histnat_jjaA!0 = "ensemble_nat"
    nam_histnat_jjaA@var = "NAM_nat_JJA_"+syear_a+"_"+eyear_a
    nam_histnat_jjaA@weight = wgt_histnat
    nam_histnat_jjaA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_histnat_jjaA, nc_outfile)
    sam_histnat_jjaA_0 = sam_histnat_jjaA_0/sam_yrs0*10
    sam_histnat_jjaA_0!0 = "ensemble_nat"
    sam_histnat_jjaA_0@var = "SAM_nat_JJA_"+syear_b+"_"+eyear_b
    sam_histnat_jjaA_0@weight = wgt_histnat
    sam_histnat_jjaA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histnat_jjaA_0, nc_outfile)
    sam_histnat_jjaA_1 = sam_histnat_jjaA_1/sam_yrs1*10
    sam_histnat_jjaA_1!0 = "ensemble_nat"
    sam_histnat_jjaA_1@var = "SAM_nat_JJA_"+syear_c+"_"+eyear_c
    sam_histnat_jjaA_1@weight = wgt_histnat
    sam_histnat_jjaA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histnat_jjaA_1, nc_outfile)

    nam_histstrat_djfA = nam_histstrat_djfA/nam_yrs*10
    nam_histstrat_djfA!0 = "ensemble_stratO3"
    nam_histstrat_djfA@var = "NAM_stratO3_DJF_"+syear_a+"_"+eyear_a
    nam_histstrat_djfA@weight = wgt_histstrat
    nam_histstrat_djfA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_histstrat_djfA, nc_outfile)
    sam_histstrat_djfA_0 = sam_histstrat_djfA_0/sam_yrs0*10
    sam_histstrat_djfA_0!0 = "ensemble_stratO3"
    sam_histstrat_djfA_0@var = "SAM_stratO3_DJF_"+syear_b+"_"+eyear_b
    sam_histstrat_djfA_0@weight = wgt_histstrat
    sam_histstrat_djfA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histstrat_djfA_0, nc_outfile)
    sam_histstrat_djfA_1 = sam_histstrat_djfA_1/sam_yrs1*10
    sam_histstrat_djfA_1!0 = "ensemble_stratO3"
    sam_histstrat_djfA_1@var = "SAM_stratO3_DJF_"+syear_c+"_"+eyear_c
    sam_histstrat_djfA_1@weight = wgt_histstrat
    sam_histstrat_djfA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histstrat_djfA_1, nc_outfile)

    nam_histstrat_jjaA = nam_histstrat_jjaA/nam_yrs*10
    nam_histstrat_jjaA!0 = "ensemble_stratO3"
    nam_histstrat_jjaA@var = "NAM_stratO3_JJA_"+syear_a+"_"+eyear_a
    nam_histstrat_jjaA@weight = wgt_histstrat
    nam_histstrat_jjaA@units = "decade^-1"
    ncdf_outfile = ncdf_write(nam_histstrat_jjaA, nc_outfile)
    sam_histstrat_jjaA_0 = sam_histstrat_jjaA_0/sam_yrs0*10
    sam_histstrat_jjaA_0!0 = "ensemble_stratO3"
    sam_histstrat_jjaA_0@var = "SAM_stratO3_JJA_"+syear_b+"_"+eyear_b
    sam_histstrat_jjaA_0@weight = wgt_histstrat
    sam_histstrat_jjaA_0@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histstrat_jjaA_0, nc_outfile)
    sam_histstrat_jjaA_1 = sam_histstrat_jjaA_1/sam_yrs1*10
    sam_histstrat_jjaA_1!0 = "ensemble_stratO3"
    sam_histstrat_jjaA_1@var = "SAM_stratO3_JJA_"+syear_c+"_"+eyear_c
    sam_histstrat_jjaA_1@weight = wgt_histstrat
    sam_histstrat_jjaA_1@units = "decade^-1"
    ncdf_outfile = ncdf_write(sam_histstrat_jjaA_1, nc_outfile)
  end if

  log_provenance(nc_outfile, ofile+"."+file_type, "NAM/SAM trends", \
                 (/"trend"/), (/"nhext", "shext"/), "box", "kosaka_yu", "", paths_all)

end

