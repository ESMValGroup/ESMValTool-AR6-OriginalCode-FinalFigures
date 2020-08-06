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
;    20191203 kosaka_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

load "$diag_scripts/../interface_scripts/interface.ncl"

;load "$diag_scripts/shared/statistics.ncl"
;load "$diag_scripts/shared/plot/style.ncl"
;load "$diag_scripts/shared/plot/contour_maps.ncl"
;load "$diag_scripts/shared/plot/contourplot.ncl"

load "$diag_scripts/ar6ch3_hiatus/functions.ncl"

begin

  enter_msg(DIAG_SCRIPT, "")

  scale = 10. ; deg/decade

  input_dirs = diag_script_info@input_files + "/"
  ndirs = dimsizes(input_dirs)
;print (input_dirs)
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
    
      path_cmip5 = input_dirs(n)+"trendGMST_"+blending+"_"+masking+"_hiatus_cmip5.nc"
      if (fileexists(path_cmip5)) then
        f = addfile(path_cmip5, "r")
        trend_cmip5 = f->trend
        trend_cmip5 = trend_cmip5 * scale
      end if
      path_cmip6 = input_dirs(n)+"trendGMST_"+blending+"_"+masking+"_hiatus_cmip6.nc"
      if (fileexists(path_cmip6)) then
        f = addfile(path_cmip6, "r")
        trend_cmip6 = f->trend
        trend_cmip6 = trend_cmip6 * scale
      end if

      if (.not.isvar("obs_dataset")) then
        if (isvar("trend_cmip6")) then
          if (isatt(trend_cmip6, "mask_reference")) then
            obs_dataset = trend_cmip6@mask_reference
          end if
        elseif (isvar("trend_cmip5")) then
          if (isatt(trend_cmip5, "mask_reference")) then
            obs_dataset = trend_cmip5@mask_reference
          end if
        else
          continue
        end if
      end if

      blend_method = ""
      if (blending .eq. "blended") then
        if (isvar("trend_cmip6")) then
          if (isatt(trend_cmip6, "blend_method")) then
            blend_method = trend_cmip6@blend_method
          end if
        elseif (isvar("trend_cmip5")) then
          if (isatt(trend_cmip5, "blend_method")) then
            blend_method = trend_cmip5@blend_method
          end if
        end if
      end if
      
      if (isvar("obs_dataset")) then
        path_obs = input_dirs(n)+"trendGMST_"+blending+"_"+masking+"_hiatus_"+obs_dataset+".nc"
        if (fileexists(path_obs)) then
          f = addfile(path_obs, "r")
          trend_obs = f->trend
          trend_obs = trend_obs * scale
          if (dimsizes(trend_obs).gt.1) then
            tmp = trend_obs(0)
            delete(trend_obs)
            trend_obs = tmp
            delete(tmp)
          end if
        end if
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
        hlist = "blended: method = "+blend_method+", "
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
                             +sprintf(fmt, percentile_val(trend_cmip6, 0.025))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip6, 0.0375))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip6, 0.05))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip6, 0.1))+" " \
                             +sprintf(fmt, dim_median(trend_cmip6))+" " \
                             +sprintf(fmt, avg(trend_cmip6))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip6, 0.9))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip6, 0.95))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip6, 0.975))+" " \
                             +sprintf(fmt, max(trend_cmip6))
          write_table(ofile, "a", [/hlist/], "%s")
        end if
        if (isvar("trend_cmip5")) then
          hlist = "cmip5:   "+sprintf(fmt, min(trend_cmip5))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip5, 0.025))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip5, 0.0375))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip5, 0.05))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip5, 0.1))+" " \
                             +sprintf(fmt, dim_median(trend_cmip5))+" " \
                             +sprintf(fmt, avg(trend_cmip5))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip5, 0.9))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip5, 0.95))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip5, 0.975))+" " \
                             +sprintf(fmt, max(trend_cmip5))
          write_table(ofile, "a", [/hlist/], "%s")
        end if
        if (isvar("trend_cmip5").and.isvar("trend_cmip6")) then
          trend_cmip = array_append_record(trend_cmip5, trend_cmip6, 0)
          hlist = "cmip5+6: "+sprintf(fmt, min(trend_cmip))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip, 0.025))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip, 0.0375))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip, 0.05))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip, 0.1))+" " \
                             +sprintf(fmt, dim_median(trend_cmip))+" " \
                             +sprintf(fmt, avg(trend_cmip))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip, 0.9))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip, 0.95))+" " \
                             +sprintf(fmt, percentile_val(trend_cmip, 0.975))+" " \
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
