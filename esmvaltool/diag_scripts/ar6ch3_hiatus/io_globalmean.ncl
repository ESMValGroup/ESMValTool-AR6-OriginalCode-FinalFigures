; calc_composite
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs: 
;
; History
;    20210203 kosaka_yu: updated for the new recipe structure
;    20200531 kosaka_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

function make_GMList(work_dir, do_post, do_maskGlobalmean_post, do_blend, do_mask, \
                     syear_hiatus, eyear_hiatus, syear_post, eyear_post, datasets)
local work_dir, do_post, do_maskGlobalmean_post, do_blend, do_mask, datasets, GMList, l, \
      head_masked, head_unmasked, syear_hiatus, eyear_hiatus, syear_post, eyear_post
begin
  load "$diag_scripts/ar6ch3_hiatus/config_datasets.ncl"
  
  if (do_post) then
    head_unmasked = "# Dataset            ensemble    "+syear_hiatus+"-"+eyear_hiatus+"  "+syear_post+"-"+eyear_post
    if (do_maskGlobalmean_post) then
      head_masked = head_unmasked
    else
      head_masked = head_unmasked+" (unmasked)"
    end if
  else
    head_unmasked = "# Dataset            ensemble    "+syear_hiatus+"-"+eyear_hiatus
    head_masked   = head_unmasked
  end if
   
  GMList = NewList("fifo")
  ListAppend(GMList, new(1, string))

  do l = 0, dimsizes(datasets)-1
    ListAppend(GMList, new(1, string))
    if (datasets(l).eq."CMIP6") then
      GMList[l] = "CMIP6"
      GMList[l]@ofile_GSATunmasked = work_dir + "trendGSAT_unmasked_CMIP6.txt"
      write_table(GMList[l]@ofile_GSATunmasked, "w", [/head_unmasked/], "%s")
      if (do_mask) then
        GMList[l]@ofile_GSATmasked = work_dir + "trendGSAT_masked_CMIP6.txt"
        write_table(GMList[l]@ofile_GSATmasked, "w", [/head_masked/], "%s")
      end if
      if (do_blend) then
        GMList[l]@ofile_GMSTunmasked = work_dir + "trendGMST_unmasked_CMIP6.txt"
        write_table(GMList[l]@ofile_GMSTunmasked, "w", [/head_unmasked/], "%s")
      end if
      if (do_mask.and.do_blend) then
        GMList[l]@ofile_GMSTmasked = work_dir + "trendGMST_masked_CMIP6.txt"
        write_table(GMList[l]@ofile_GMSTmasked, "w", [/head_masked/], "%s")
      end if
    elseif (datasets(l).eq."CMIP5") then
      GMList[l] = "CMIP5"
      GMList[l]@ofile_GSATunmasked = work_dir + "trendGSAT_unmasked_CMIP5.txt"
      write_table(GMList[l]@ofile_GSATunmasked, "w", [/head_unmasked/], "%s")
      if (do_mask) then
        GMList[l]@ofile_GSATmasked = work_dir + "trendGSAT_masked_CMIP5.txt"
        write_table(GMList[l]@ofile_GSATmasked, "w", [/head_masked/], "%s")
      end if
      if (do_blend) then
        GMList[l]@ofile_GMSTunmasked = work_dir + "trendGMST_unmasked_CMIP5.txt"
        write_table(GMList[l]@ofile_GMSTunmasked, "w", [/head_unmasked/], "%s")
      end if
      if (do_mask.and.do_blend) then
        GMList[l]@ofile_GMSTmasked = work_dir + "trendGMST_masked_CMIP5.txt"
        write_table(GMList[l]@ofile_GMSTmasked, "w", [/head_masked/], "%s")
      end if
    else
      GMList[l] = datasets(l)
      if (any(datasets(l).eq.reanalyses)) then
        GMList[l]@ofile_GSATunmasked = work_dir + "trendGSAT_unmasked_"+datasets(l)+"_OBS.txt"
        write_table(GMList[l]@ofile_GSATunmasked, "w", [/head_unmasked/], "%s")
        if (do_mask) then
          GMList[l]@ofile_GSATmasked = work_dir + "trendGSAT_masked_"+datasets(l)+"_OBS.txt"
          write_table(GMList[l]@ofile_GSATmasked, "w", [/head_masked/], "%s")
        end if
        if (do_blend) then
          GMList[l]@ofile_GMSTunmasked = work_dir + "trendGMST_unmasked_"+datasets(l)+"_OBS.txt"
          write_table(GMList[l]@ofile_GMSTunmasked, "w", [/head_unmasked/], "%s")
        end if
        if (do_mask.and.do_blend) then
          GMList[l]@ofile_GMSTmasked = work_dir + "trendGMST_masked_"+datasets(l)+"_OBS.txt"
          write_table(GMList[l]@ofile_GMSTmasked, "w", [/head_masked/], "%s")
        end if
      else
        GMList[l]@ofile_GSATunmasked = work_dir + "trendGMST_"+datasets(l)+"_OBS.txt"
        write_table(GMList[l]@ofile_GSATunmasked, "w", [/head_unmasked/], "%s")
      end if
    end if
  end do
  return(GMList)
end

procedure write_GMtrend(info, trends, blending, masking, OutList)
local n, ii, key, ofile, ens
begin
  if (any(info@project.eq.(/"CMIP5", "CMIP6"/))) then
    key = "project"
  else
    key = "dataset"
;    ens = "1"
  end if
  do ii = 0, ListCount(OutList)-1
    if (OutList[ii].eq.info@$key$) then
      n = ii
      break
    end if
  end do
  if (OutList[n].ne.info@$key$) then
    error_msg("f", DIAG_SCRIPT, "write_GM", "No matching output found for "+info@$key$)
  end if

  if (.not.blending.and..not.masking) then
    ofile = OutList[n]@ofile_GSATunmasked
  elseif (blending.and..not.masking) then
    ofile = OutList[n]@ofile_GMSTunmasked
  elseif (.not.blending.and.masking) then
    ofile = OutList[n]@ofile_GSATmasked
  elseif (blending.and.masking) then
    ofile = OutList[n]@ofile_GMSTmasked
  end if
  if (any(info@$key$.eq.(/"CMIP5", "CMIP6"/))) then
    ens = tostring(info@ensemble)
  elseif (any(info@$key$.eq.(/"HadCRUT4", "HadCRUT5", "CowtanWay"/))) then
    ens = info@type
  else
    ens = "1"
  end if

  if (dimsizes(trends).eq.1) then
    write_table(ofile, "a", [/info@dataset, ens, trends(0)/], "%-20s %-10s %10.7f")
  else
    write_table(ofile, "a", [/info@dataset, ens, trends(0), trends(1)/], "%-20s %-10s %10.7f %10.7f")
  end if
  
end

procedure addatt_GMtrend(infos, att, blending, masking, OutList)
local n, ii, key, oatt, tmp, tmp2
begin
  if (any(infos[0]@project.eq.(/"CMIP5", "CMIP6"/))) then
    key = "project"
  else
    key = "dataset"
  end if
  do ii = 0, ListCount(OutList)-1
    if (OutList[ii].eq.infos[0]@$key$) then
      n = ii
      break
    end if
  end do
  if (OutList[n].ne.infos[0]@$key$) then
    error_msg("f", DIAG_SCRIPT, "addatt_GM", "No matching output found for "+infos[0]@$key$)
  end if

  if (.not.blending.and..not.masking) then
    oatt = att+"_GSATunmasked"
  elseif (blending.and..not.masking) then
    oatt = att+"_GMSTunmasked"
  elseif (.not.blending.and.masking) then
    oatt = att+"_GSATmasked"
  elseif (blending.and.masking) then
    oatt = att+"_GMSTmasked"
  end if

  do ii = 0, ListCount(infos)-1
    if (isatt(OutList[n], oatt)) then
      tmp = OutList[n]@$oatt$
      if (isstring(tmp)) then
        tmp2 = str_split(tmp, ",")
        if (all(tmp.ne.infos[ii]@$att$)) then
          delete(OutList[n]@$oatt$)
          OutList[n]@$oatt$ = str_join(array_append_record(tmp2, infos[ii]@$att$, 0), ",")
        end if
        delete(tmp2)
      else
        delete(OutList[n]@$oatt$)
        OutList[n]@$oatt$ = array_append_record(tmp, infos[ii]@$att$, 0)
      end if
      delete(tmp)
    else
      OutList[n]@$oatt$ = infos[ii]@$att$
    end if
  end do
end

function read_GMtrend_as_list(paths)
local paths, path, head, data, dims, n, x, y, hiatus_period, post_period, add_post, \
      syear_hiatus, eyear_hiatus, syear_post, eyear_post, ii, jj, kk
begin
  do ii = 0, dimsizes(paths)-1
    path = paths(ii)
    head = readAsciiHead(path, 1)
    hiatus_period = str_get_field(head(0), 4, " ")
    syear_hiatus = stringtoint(str_get_field(hiatus_period, 1, "-"))
    eyear_hiatus = stringtoint(str_get_field(hiatus_period, 2, "-"))
    post_period = str_get_field(head(0), 5, " ")
    if (post_period.ne."") then
      add_post = True
      syear_post = stringtoint(str_get_field(post_period, 1, "-"))
      eyear_post = stringtoint(str_get_field(post_period, 2, "-"))
    else
      add_post = False
    end if
    data = readAsciiTable(path, 1, "string", 1)
    dims = dimsizes(data)
    n = dims(0)
    if (ii.eq.0) then
      x = NewList("fifo")
    end if
    y = new(1, "float", default_fillvalue("float"))
    do jj = 0, n - 1
      ListAppend(x, new(1, "float"))
      kk = ListCount(x)-1
      x[kk] = stringtofloat(str_get_field(data(jj, 0), 3, " "))
      x[kk]@dataset = str_get_field(data(jj, 0), 1, " ")
      x[kk]@ensemble = str_get_field(data(jj, 0), 2, " ")
      x[kk]@hiatus_period = hiatus_period
      x[kk]@syear_hiatus = syear_hiatus
      x[kk]@eyear_hiatus = eyear_hiatus
      if (add_post) then
        x[kk]@post_period = post_period
        x[kk]@syear_post = syear_post
        x[kk]@eyear_post = eyear_post
      end if
      if (str_get_field(data(jj, 0), 4, " ").ne."") then
        y = stringtofloat(str_get_field(data(jj, 0), 4, " "))
      else
        y = y@_FillValue
      end if
      x[kk]@trend_post = y
    end do
    delete([/y, data, dims/])
  end do
  return(x)
end

function list_to_array(x)
local y, n, i
begin
  n = ListCount(x)
  do i = 0, n-1
    if (.not.isnumeric(x[i])) then
      error_msg("f", DIAG_SCRIPT, "list_to_array", "Non-numeric variable in the list")
    end if
  end do
  y = new(n, "float")
  do i = 0, n-1
    y(i) = tofloat(x[i])
  end do
  return(y)
end