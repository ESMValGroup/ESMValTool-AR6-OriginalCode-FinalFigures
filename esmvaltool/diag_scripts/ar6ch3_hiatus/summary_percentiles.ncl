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
  
  do n = 0, ndirs-1
    do ii = 0, 3
      if (ii.eq.0) then
        masking = "masked"
        blending = "blended"
      elseif (ii.eq.1) then
        masking = "masked"
        blending = "unblended"
      elseif (ii.eq.2) then
        masking = "unmasked"
        blending = "blended"
      elseif (ii.eq.3) then
        masking = "unmasked"
        blending = "unblended"
      end if

      if (blending.eq."blended") then
        path_cmip5 = input_dirs(n)+"trendGMST_"+masking+"_CMIP5.txt"
        path_cmip6 = input_dirs(n)+"trendGMST_"+masking+"_CMIP6.txt"
      else
        path_cmip5 = input_dirs(n)+"trendGSAT_"+masking+"_CMIP5.txt"
        path_cmip6 = input_dirs(n)+"trendGSAT_"+masking+"_CMIP6.txt"
      end if
      if (fileexists(path_cmip5)) then
        trend_cmip5_list = read_GMtrend_as_list(path_cmip5)
        trend_cmip5 = list_to_array(trend_cmip5_list)
        trend_cmip5 = trend_cmip5 * scale
        weight_cmip5 = trend_cmip5
        weight_cmip5 = 0.
        do ii = 0, ListCount(trend_cmip5_list)-1
          esize = num(trend_cmip5_list[ii]@dataset.eq.metadata_att_as_array(trend_cmip5_list, "dataset"))
          weight_cmip5(ii) = 1./tofloat(esize)
        end do
        delete(trend_cmip5_list)
      end if
      if (fileexists(path_cmip6)) then
        trend_cmip6_list = read_GMtrend_as_list(path_cmip6)
        trend_cmip6 = list_to_array(trend_cmip6_list)
        trend_cmip6 = trend_cmip6 * scale
        weight_cmip6 = trend_cmip6
        weight_cmip6 = 0.
        do ii = 0, ListCount(trend_cmip6_list)-1
          esize = num(trend_cmip6_list[ii]@dataset.eq.metadata_att_as_array(trend_cmip6_list, "dataset"))
          weight_cmip6(ii) = 1./tofloat(esize)
        end do
        delete(trend_cmip6_list)
      end if

      diagnostic = str_get_field(input_dirs(n), str_fields_count(input_dirs(n), "/"), "/")
      ipos = str_index_of_substr(diagnostic, "mask", -1) + 4
      if (ismissing(ipos)) then
        ipos = str_index_of_substr(diagnostic, "ref", -1) + 3
      end if
      if (ismissing(ipos)) then
        ipos = str_index_of_substr(diagnostic, "_", -1) + 1
      end if
      obs_dataset = str_get_field(str_insert(diagnostic, " ", ipos), 2, " ")
      delete(ipos)
      delete(diagnostic)
        
      path_obs = input_dirs(n)+"trendGMST_"+obs_dataset+"_OBS.txt"
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

      ofile = output_dir + "percentiles_GMSTtrend_hiatus_"+blending+"_"+masking+".txt"
      fmt = "%8.5f"
      if (isvar("trend_obs")) then
        hlist = obs_dataset+": "+sprintf(fmt, trend_obs)
        write_table(ofile, "a", [/hlist/], "%s")
      end if

      if (blending.eq."blended") then
        hlist = "blended, "
      else
        hlist = "unblended, "
      end if
      if (masking.eq."masked") then
        hlist = hlist + "masked: reference = "+obs_dataset
      else
        hlist = hlist + "unmasked"
      end if
      write_table(ofile, "a", [/hlist/], "%s")
      
      if (isvar("trend_cmip6").or.isvar("trend_cmip5")) then
        hlist = "           min      2.5%     3.75%     5%      10%    median     mean     90%      95%     97.5%     max "
        write_table(ofile, "a", [/hlist/], "%s")
        if (isvar("trend_cmip6")) then
          hlist = "cmip6:   "+sprintf(fmt, min(trend_cmip6))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip6, weight_cmip6, 0.025))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip6, weight_cmip6, 0.0375))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip6, weight_cmip6, 0.05))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip6, weight_cmip6, 0.1))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip6, weight_cmip6, 0.5))+" " \
                             +sprintf(fmt, weighted_mean(trend_cmip6, weight_cmip6))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip6, weight_cmip6, 0.9))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip6, weight_cmip6, 0.95))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip6, weight_cmip6, 0.975))+" " \
                             +sprintf(fmt, max(trend_cmip6))
          write_table(ofile, "a", [/hlist/], "%s")
        end if
        if (isvar("trend_cmip5")) then
          hlist = "cmip5:   "+sprintf(fmt, min(trend_cmip5))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip5, weight_cmip5, 0.025))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip5, weight_cmip5, 0.0375))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip5, weight_cmip5, 0.05))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip5, weight_cmip5, 0.1))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip5, weight_cmip5, 0.5))+" " \
                             +sprintf(fmt, weighted_mean(trend_cmip5, weight_cmip5))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip5, weight_cmip5, 0.9))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip5, weight_cmip5, 0.95))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip5, weight_cmip5, 0.975))+" " \
                             +sprintf(fmt, max(trend_cmip5))
          write_table(ofile, "a", [/hlist/], "%s")
        end if
        if (isvar("trend_cmip5").and.isvar("trend_cmip6")) then
          trend_cmip = array_append_record(trend_cmip5, trend_cmip6, 0)
          weight_cmip = array_append_record(weight_cmip5, weight_cmip6, 0)
          hlist = "cmip5+6: "+sprintf(fmt, min(trend_cmip))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.025))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.0375))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.05))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.1))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.5))+" " \
                             +sprintf(fmt, weighted_mean(trend_cmip, weight_cmip))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.9))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.95))+" " \
                             +sprintf(fmt, weighted_percentile(trend_cmip, weight_cmip, 0.975))+" " \
                             +sprintf(fmt, max(trend_cmip))
          write_table(ofile, "a", [/hlist/], "%s")
        end if
      end if
      hlist = "-------------------------------------------------------------------"
      write_table(ofile, "a", [/hlist/], "%s")
      hlist = ""
      write_table(ofile, "a", [/hlist/], "%s")

      delete(path_cmip5)
      delete(path_cmip6)
      if (isvar("trend_cmip5")) then
        delete(trend_cmip5)
      end if
      if (isvar("trend_cmip6")) then
        delete(trend_cmip6)
      end if
      if (isvar("trend_cmip")) then
        delete(trend_cmip)
      end if
      if (isvar("weight_cmip5")) then
        delete(weight_cmip5)
      end if
      if (isvar("weight_cmip6")) then
        delete(weight_cmip6)
      end if
      if (isvar("weight_cmip")) then
        delete(weight_cmip)
      end if
      if (isvar("path_obs")) then
        delete(path_obs)
      end if
      if (isvar("trend_obs")) then
        delete(trend_obs)
      end if
    end do

    delete(obs_dataset)
  end do  
  
end
