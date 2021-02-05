; summary_percentile
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs:
;
; History
;    20200914 kosaka_yu: revised to account for weighting in percentiles.
;    20191203 kosaka_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

load "$diag_scripts/ar6ch3_hiatus/functions.ncl"
load "$diag_scripts/ar6ch3_hiatus/io_globalmean.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")

  scale = 10. ; deg/decade

  input_dirs = diag_script_info@input_files + "/"
  ndirs = dimsizes(input_dirs)
  output_dir = config_user_info@plot_dir + "/"
  system("mkdir -p "+output_dir)
  system("rm -f "+output_dir+"percentiles_GMSTtrend_hiatus_*.txt")

  do n = 0, ndirs - 1
    diagnostic = str_get_field(input_dirs(n), str_fields_count(input_dirs(n), "/"), "/")
    ipos = str_index_of_substr(diagnostic, "mask", -1) + 4
    if (ismissing(ipos)) then
      ipos = str_index_of_substr(diagnostic, "ref", -1) + 3
    end if
    if (ismissing(ipos)) then
      ipos = str_index_of_substr(diagnostic, "_", -1) + 1
    end if
    if (isvar("obs_datasets")) then
      obs_datasets := array_append_record(obs_datasets, str_get_field(str_insert(diagnostic, " ", ipos), 2, " "), 0)
    else
      obs_datasets = str_get_field(str_insert(diagnostic, " ", ipos), 2, " ")
    end if
    delete(ipos)
    delete(diagnostic)
  end do
  obs_datasets := get_unique_values(obs_datasets)
;  print(obs_datasets+"")
  
  do ii = 0, dimsizes(obs_datasets)-1
    ofile = output_dir + "percentiles_GMSTtrend_hiatus_ref"+obs_datasets(ii)+".txt"

    do n = 0, ndirs - 1
      path_obs = input_dirs(n)+"trendGMST_"+obs_datasets(ii)+"_OBS.txt"
      if (fileexists(path_obs)) then
        trend_obs_list = read_GMtrend_as_list(path_obs)
        if (ListCount(trend_obs_list).gt.1) then
          trend_obs_array = list_to_array(trend_obs_list)
          ens = metadata_att_as_array(trend_obs_list, "ensemble")
          if (any(ens.eq."mean")) then
            trend_obs = trend_obs_array(ind(ens.eq."mean"))
          elseif (any(ens.eq."median")) then
            trend_obs = trend_obs_array(ind(ens.eq."median"))
          end if
          delete(trend_obs_array)
          delete(ens)
        else
          trend_obs = trend_obs_list[0]
        end if
        trend_obs = trend_obs * scale
        delete(trend_obs_list)
      else
        continue
      end if
    end do

    fmt = "%8.5f"
    if (isvar("trend_obs")) then
      hlist = obs_datasets(ii)+": "+sprintf(fmt, trend_obs)
      write_table(ofile, "w", [/hlist/], "%s")
    else
      continue
    end if

    hlist = "                        min      2.5%      5%      10%    median     mean     90%      95%     97.5%     max "
    write_table(ofile, "a", [/hlist/], "%s")
    
    do jj = 0, 1
      if (jj.eq.0) then
        cmip = "CMIP6"
      elseif (jj.eq.1) then
        cmip = "CMIP5"
      end if
        
      do kk = 0, 3
        if (kk.eq.0) then
          masking = "masked"
          blending = "blended"
        elseif (kk.eq.1) then
          masking = "masked"
          blending = "unblended"
        elseif (kk.eq.2) then
          masking = "unmasked"
          blending = "blended"
        elseif (kk.eq.3) then
          masking = "unmasked"
          blending = "unblended"
        end if

        do n = 0, ndirs-1
          if (.not.isStrSubset(input_dirs(n), "_ref"+obs_datasets(ii)+"/")) then
            continue
          end if
        
          if (blending.eq."blended") then
            ipath = input_dirs(n)+"trendGMST_"+masking+"_"+cmip+".txt"
          else
            ipath = input_dirs(n)+"trendGSAT_"+masking+"_"+cmip+".txt"
          end if
          if (fileexists(ipath)) then
            if (isvar("paths")) then
              pathtmp = array_append_record(paths, ipath, 0)
              delete(paths)
              paths = pathtmp
              delete(pathtmp)
            else
              paths = ipath
            end if
          end if
        end do

        if (isvar("paths")) then
          trend_cmip_list = read_GMtrend_as_list(paths)
          trend_cmip = list_to_array(trend_cmip_list)
          trend_cmip = trend_cmip * scale
          weight_cmip = trend_cmip
          weight_cmip = 0.
          do ll = 0, ListCount(trend_cmip_list)-1
            esize = num(trend_cmip_list[ll]@dataset.eq.metadata_att_as_array(trend_cmip_list, "dataset"))
            weight_cmip(ll) = 1./tofloat(esize)
          end do
          delete(trend_cmip_list)
        end if

        if (kk.eq.0) then
          hlist = cmip+" ("+dimsizes(trend_cmip)+" members)"
          write_table(ofile, "a", [/hlist/], "%s")
        end if
        
        xlist = [/"", ""/]
        if (blending.eq."blended") then
          xlist[0] = "blended, "
        else
          xlist[0] = "unblended, "
        end if
        if (masking.eq."masked") then
          xlist[0] = xlist[0] + "masked:"
        else
          xlist[0] = xlist[0] + "unmasked:"
        end if
        xlist[1] = sprintf(fmt, min(trend_cmip))+" " \
                   +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.025))+" " \
                   +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.05))+" " \
                   +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.1))+" " \
                   +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.5))+" " \
                   +sprintf(fmt, weighted_mean(trend_cmip, weight_cmip))+" " \
                   +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.9))+" " \
                   +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.95))+" " \
                   +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.975))+" " \
                   +sprintf(fmt, max(trend_cmip))
        write_table(ofile, "a", xlist, "%-21s%s")

        delete(paths)
        if (isvar("trend_cmip")) then
          delete(trend_cmip)
        end if
        if (isvar("weight_cmip")) then
          delete(weight_cmip)
        end if
      end do
      
      hlist = "-------------------------------------------------------------------"
      write_table(ofile, "a", [/hlist/], "%s")
      hlist = ""
      write_table(ofile, "a", [/hlist/], "%s")

    end do

  end do  
end
