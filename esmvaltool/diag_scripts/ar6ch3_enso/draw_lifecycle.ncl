; draw_hadleycell_trend
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs:
;
; History
;    20200511 kosaka_yu: cleaned to meet the code style;
;                        removed bottom two panels that are removed from SOD
;    20200105 kosaka_yu: show shading instead of lines;
;                        use duration based on composite
;    20191012 kosaka_yu: updated for ESMValTool 2.0.0b2
;    20190316 kosa_yu: written.
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

load "$diag_scripts/ar6ch3_enso/modelling_center.ncl"
load "$diag_scripts/ar6ch3_enso/setting_mme.ncl"

procedure CalcMetrics(ts, duration_threshold, duration_threshold_unit)
local nevent, t0, x, t_pos_seg, t_neg_seg, t, tlast, nseg_pos, nseg_neg, \
      i, ipos, ineg, n, nref, duration, t_peak, mag_peak, lag_peak, tmp, \
      date, ens, interval, int, is, idx, \
      seg_bnds, tmpbnds, nonoverlap, nonoverlapping_events, iis, thres
begin
  idx = ts
  ; Event frequency
  nevent = dimsizes(idx(:, 0))
  ts@frequency = tofloat(nevent) / tofloat(idx@esize)

  ; Magnitude at reference month
  t0 = toint(idx@time_ref)
  x = dim_avg_n(idx, 0)  ; El Nino
  if (x(t0).lt.0.) then
    x = -x               ; La Nina
    idx = -idx
  end if
  ts@mag_ref  = x(t0)  ; magnitude at target_month
  delete(x)

  if (duration_threshold_unit.eq."deg") then
    thres = duration_threshold
  elseif (duration_threshold_unit.eq."stdv") then
    thres = duration_threshold * ts@stdv_ref
  else
    error_msg("f", DIAG_SCRIPT, "", \
              "Undefined duration_threshold_unit: "+duration_threshold_unit)
  end if

  t_peak = new(nevent, "integer")
  mag_peak = new(nevent, "float")
  lag_peak = new(nevent, "float")
  duration = new(nevent, "float")
  nonoverlap = new(nevent, "logical")

  do i = 0, nevent-1
    ; Divide into segments
;    if (thres.eq.0.) then
;      sign = sign_matlab(idx(i, :))
;      npos = 0
;      nneg = 0
;      do j = 0, dimsizes(sign)-2
;        if (ismissing(sign(j)).or.ismissing(sign(j+1))) then
;          continue
;        end if
;        if (abs(sign(j+1) - sign(j)).gt.1) then
;          if (sign(j).gt.0) then
;            npos = npos + 1
;          else
;            nneg = nneg + 1
;          end if
;          if (isvar("seg_bnds")) then
;            tmpbnds = seg_bnds
;            delete(seg_bnds)
;            seg_bnds = array_append_record(tmpbnds, tofloat(j)+0.5, 0)
;            delete(tmpbnds)
;          else
;            seg_bnds = tofloat(j)+0.5
;          end if
;        elseif (j.gt.0.and.sign(j).eq.0) then
;          if (abs(sign(j-1)-sign(j+1)).gt.1) then
;            if (sign(j-1).gt.0) then
;              npos = npos + 1
;            else
;              nneg = nneg + 1
;            end if
;            if (isvar("seg_bnds")) then
;              tmpbnds = seg_bnds
;              delete(seg_bnds)
;              seg_bnds = array_append_record(tmpbnds, tofloat(j), 0)
;              delete(tmpbnds)
;            else
;              seg_bnds = tofloat(j)
;            end if
;          end if
;        end if
;      end do
;      if (.not.isvar("seg_bnds")) then
;        continue
;      end if
;
;      j = toint(seg_bnds(dimsizes(seg_bnds)-1)+1.)
;      if (sign(j).gt.0) then
;        npos = npos + 1
;      else
;        nneg = nneg + 1
;      end if
;
;      t_pos_seg = new((/2, npos/), "integer")
;      t_neg_seg = new((/2, nneg/), "integer")
;      t_pos_seg = t_pos_seg@_FillValue
;      t_neg_seg = t_neg_seg@_FillValue
;      ipos = 0
;      ineg = 0
;      do j = 0, dimsizes(seg_bnds)-1
;        if (sign(toint(seg_bnds(j)-0.5)).gt.0) then
;          t_pos_seg(1, ipos) = toint(seg_bnds(j)-0.5)
;          ipos = ipos + 1
;          t_neg_seg(0, ineg) = toint(seg_bnds(j)+1.)
;        else
;          t_neg_seg(1, ineg) = toint(seg_bnds(j)-0.5)
;          ineg = ineg + 1
;          t_pos_seg(0, ipos) = toint(seg_bnds(j)+1.)
;        end if
;      end do
;      delete(seg_bnds)
;      delete(sign)
;    else
    npos = 0
    nneg = 0
    do j = 0, dimsizes(idx(i, :))-2
      if (ismissing(idx(i, j)).or.ismissing(idx(i, j+1))) then
        continue
      end if
      if (idx(i, j).gt.thres .and. idx(i, j+1).le.thres) then
        if (isvar("t_pos_seg")) then
          t_pos_seg(1, npos-1) = j
        else
          npos = 1
          t_pos_seg = new((/2, npos/), "integer")
          t_pos_seg(1, npos-1) = j
        end if
      elseif (idx(i, j).le.thres .and. idx(i, j+1).gt.thres) then
        npos = npos + 1
        if (isvar("t_pos_seg")) then
          tmp = t_pos_seg
          delete(t_pos_seg)
          t_pos_seg = new((/2, npos/), "integer")
          t_pos_seg(:, 0:npos-2) = tmp
          delete(tmp)
          t_pos_seg(0, npos-1) = j+1
        else
          t_pos_seg = new((/2, npos/), "integer")
          t_pos_seg(0, npos-1) = j+1
        end if
      end if
      if (idx(i, j).lt.-thres .and. idx(i, j+1).ge.-thres) then
        if (isvar("t_neg_seg")) then
          t_neg_seg(1, nneg-1) = j
        else
          nneg = 1
          t_neg_seg = new((/2, nneg/), "integer")
          t_neg_seg(1, nneg-1) = j
        end if
      elseif (idx(i, j).ge.-thres .and. idx(i, j+1).lt.-thres) then
        nneg = nneg + 1
        if (isvar("t_neg_seg")) then
          tmp = t_neg_seg
          delete(t_neg_seg)
          t_neg_seg = new((/2, nneg/), "integer")
          t_neg_seg(:, 0:nneg-2) = tmp
          delete(tmp)
          t_neg_seg(0, nneg-1) = j+1
        else
          t_neg_seg = new((/2, nneg/), "integer")
          t_neg_seg(0, nneg-1) = j+1
        end if
      end if
    end do
    if (.not.isvar("t_pos_seg")) then
      npos = 1
      t_pos_seg = new((/2, 1/), "integer")
    end if
    if (.not.isvar("t_neg_seg")) then
      nneg = 1
      t_neg_seg = new((/2, 1/), "integer")
    end if
;    end if
    if (ismissing(t_pos_seg(0, 0))) then
      t_pos_seg(0, 0) \
        = min(where(ismissing(idx(i, :)), 99999, \
                    ispan(0, dimsizes(idx&time)-1, 1)))
    end if
    if (ismissing(t_neg_seg(0, 0))) then
      t_neg_seg(0, 0) \
        = min(where(ismissing(idx(i, :)), 99999, \
                    ispan(0, dimsizes(idx&time)-1, 1)))
    end if
    if (ismissing(t_pos_seg(1, npos-1))) then
      t_pos_seg(1, npos-1) \
        = max(where(ismissing(idx(i, :)), -1, \
                    ispan(0, dimsizes(idx&time)-1, 1)))
    end if
    if (ismissing(t_neg_seg(1, nneg-1))) then
      t_neg_seg(1, nneg-1) \
        = max(where(ismissing(idx(i, :)), -1, \
                    ispan(0, dimsizes(idx&time)-1, 1)))
    end if

;    do j = 0, npos-1
;      print("positive: "+t_pos_seg(0, j)+" "+t_pos_seg(1, j))
;    end do
;    do j = 0, nneg - 1
;      print("negative: "+t_neg_seg(0, j)+" "+t_neg_seg(1, j))
;    end do

  ; Find reference segment
    do n = 0, npos-1
      if (t_pos_seg(0, n).le.t0.and.t0.le.t_pos_seg(1, n)) then
        nref = n
        break
      end if
    end do
  ; Find peak
    t_peak(i) \
      = t_pos_seg(0, nref) \
      + maxind(idx(i, t_pos_seg(0, nref):t_pos_seg(1, nref)))
    mag_peak(i)  = idx(i, t_peak(i))  ; magnitude at peak
    lag_peak(i)  = t_peak(i) - t0  ; magnitude lag relative to reference month
  ; Get duration
    duration(i) = 0.
    if (t_pos_seg(0, nref).le.0 .or. \
        t_pos_seg(1, nref).ge.dimsizes(idx(i, :))-1) then
      duration(i) = duration@_FillValue
    else
      if (thres.eq.0.) then
        duration(i) \
          = t_pos_seg(1, nref) \
          + idx(i, t_pos_seg(1, nref)) \
          / (idx(i, t_pos_seg(1, nref)) - idx(i, t_pos_seg(1, nref) + 1))
        duration(i) = duration(i) \
          - (t_pos_seg(0, nref) - idx(i, t_pos_seg(1, nref)) \
             / (idx(i, t_pos_seg(1, nref)) - idx(i, t_pos_seg(1, nref) - 1)))
      else
        t = t0
        do while (.not.ismissing(idx(i, t)) .and. (idx(i, t).gt.thres))
          duration(i) = duration(i) + 1
          t = t + 1
        end do
        t = t0 - 1
        do while (.not.ismissing(idx(i, t)) .and. (idx(i, t).gt.thres))
          duration(i) = duration(i) + 1
          t = t - 1
        end do
      end if
    end if

    if (isvar("t_pos_seg")) then
      delete(t_pos_seg)
    end if
    if (isvar("t_neg_seg")) then
      delete(t_neg_seg)
    end if
  end do

  ts@lag_peak = lag_peak  ; avg(lag_peak)
  ts@mag_peak = mag_peak  ; avg(mag_peak)
  ts@duration = duration  ; avg(duration)

  ens = get_unique_values(idx@ens)
  do n = 0, dimsizes(ens)-1
    is = ind(idx@ens.eq.ens(n))
    date = new((/dimsizes(is), 2/), "integer")
    do i = 0, dimsizes(is)-1
      if (ismissing(idx@year(is(i))).or.ismissing(lag_peak(is(i)))) then
        continue
      end if
      date(i, 0) = idx@year(is(i))
      date(i, 1) = idx@reference_month + toint(lag_peak(is(i)))
      do while (date(i, 1).lt.1)
        date(i, 1) = date(i, 1) + 12
        date(i, 0) = date(i, 0) - 1
      end do
      do while (date(i, 1).gt.12)
        date(i, 1) = date(i, 1) - 12
        date(i, 0) = date(i, 0) + 1
      end do
    end do
    yyyymm = date(:, 0)*100 + date(:, 1)

    ; identify multi-year events
    nonoverlap_1mem = new(dimsizes(is), "logical")
    nonoverlapping_events = get_unique_values(yyyymm)
    do j = 0, dimsizes(nonoverlapping_events)-1
      iis = ind(nonoverlapping_events(j).eq.yyyymm)
      if (dimsizes(iis).eq.1) then
        i = ind(yyyymm.eq.nonoverlapping_events(j))
        nonoverlap_1mem(i) = True
      else
        do i = 0, dimsizes(iis)-1
          if (yyyymm(iis(i))/100.eq.idx@year(is(iis(i)))) then
            nonoverlap_1mem(iis(i)) = True
          else
            nonoverlap_1mem(iis(i)) = False
          end if
        end do
      end if
      delete(iis)
    end do
    do i = 0, dimsizes(is)-1
      nonoverlap(is(i)) = nonoverlap_1mem(i)
    end do
    delete(nonoverlap_1mem)
    delete(yyyymm)

    delete(date)
    qsort(nonoverlapping_events)
    date = new((/dimsizes(nonoverlapping_events), 2/), "integer")
    date(:, 0) = nonoverlapping_events/100
    date(:, 1) = nonoverlapping_events - date(:, 0)*100

    do i = 0, dimsizes(nonoverlapping_events)-2  ; dimsizes(is)-2
; ----------------------------------------------
;      if (date(i+1, 0)-date(i, 0).le.1) then   ;
;        continue                               ; OK?
;      end if                                   ;
; ----------------------------------------------
      int = (date(i+1, 0)-date(i, 0))*12 + date(i+1, 1) - date(i, 1)
      if (ismissing(int).or.int.eq.0) then
        continue
      end if
      if (.not.isvar("interval")) then
        interval = tofloat(int)
      else
        tmp = interval
        delete(interval)
        interval = array_append_record(tmp, tofloat(int), 0)
        delete(tmp)
      end if
    end do
    delete(nonoverlapping_events)

    delete(date)
    delete(is)
  end do
  ts@nonoverlap = nonoverlap
  if (isvar("interval")) then
    ts@interval_to_subsequent_same = avg(interval)
    delete(interval)
  else
    ts@interval_to_subsequent_same = -1
  end if

end

function duration_composite(x, multiyear_as_single, \
                            duration_threshold, duration_threshold_unit)
local x, ts, d, dims, t, lag, thres
begin
  x!0 = "event"
  dims = dimsizes(x)
  x&event = ispan(0, dims(0)-1, 1)
  delete(dims)
  ts = x(0, :)
  if (any(multiyear_as_single .eq. (/"always", "metric_only"/))) then
    ts = dim_avg_n(where(conform(x, x@nonoverlap, 0), \
                         x, x@_FillValue), 0)
  else
    ts = dim_avg_n(x, 0)
  end if
  t = toint(x@time_ref)
  if (ts(t).lt.0.) then
    ts = -ts
  end if
  if (duration_threshold_unit.eq."deg") then
    thres = duration_threshold
  elseif (duration_threshold_unit.eq."stdv") then
    thres = duration_threshold * x@stdv_ref
  else
    error_msg("f", DIAG_SCRIPT, "", \
              "Undefined duration_threshold_unit: "+duration_threshold_unit)
  end if

  if (ts(t).gt.thres) then
    d = 1.
    lag = -1
    do while (ts(t+lag).gt.thres)
      d = d + 1.
      lag = lag - 1
    end do
    lag = 1
    do while (ts(t+lag).gt.thres)
      d = d + 1.
      lag = lag + 1
    end do
  else
    d = default_fillvalue("float")
  end if
  return(d)
end

begin

  enter_msg(DIAG_SCRIPT, "")

  obs_projects = (/"OBS", "OBS6", "obs", "obs4mips", "ana4mips"/)

  multiyear_as_single = "metric_only"
  ; Regard a multiyear event as a single event for
  ; "always"
  ; "metric_only" (bottom plots only)
  ; "never"
  if (isatt(diag_script_info, "multiyear_as_single")) then
    multiyear_as_single = diag_script_info@multiyear_as_single
  end if

  duration_threshold = 0.25
  if (isatt(diag_script_info, "duration_threshold")) then
    duration_threshold = diag_script_info@duration_threshold
  end if
  duration_threshold_unit = "stdv"
  if (isatt(diag_script_info, "unit_duration_threshold")) then
    duration_threshold_unit = diag_script_info@unit_duration_threshold
  end if
  if (any(duration_threshold_unit.eq.(/"degree", "degrees", "degC", "degK", \
                                       "Degree", "Degrees", "DegC", "DegK", \
                                       "Deg"/))) then
    duration_threshold_unit = "deg"
  elseif (any(duration_threshold_unit.eq.(/"sigma", "std_dev", \
                                           "stddev", "s.d."/))) then
    duration_threshold_unit = "stdv"
  end if

  duration_individual = False
  if (isatt(diag_script_info, "duration_individual")) then
    duration_individual = diag_script_info@duration_individual
  end if

  input_dir = diag_script_info@input_files
  paths_el = systemfunc("ls "+input_dir+"/ElNino_*.nc")
  paths_la = systemfunc("ls "+input_dir+"/LaNina_*.nc")

  do i = 0, dimsizes(paths_el)-1
    f = addfile(paths_el(i), "r")
    tmp = f->ENSOindex
    if (any(tmp@project.eq.obs_projects)) then
      if (.not.isvar("obs_datasets")) then
        obs_datasets  = tmp@dataset
      else
        obs_datasets := array_append_record(obs_datasets, tmp@dataset, 0)
      end if
    else
      if (.not.isvar("datasets")) then
        datasets = tmp@dataset
      else
        datasets := array_append_record(datasets, tmp@dataset, 0)
      end if
    end if
    tsize = dimsizes(tmp)
    delete(tmp)
  end do
  do i = 0, dimsizes(paths_la)-1
    f = addfile(paths_la(i), "r")
    tmp = f->ENSOindex
    if (any(tmp@project.eq.obs_projects)) then
      if (.not.isvar("obs_datasets")) then
        obs_datasets  = tmp@dataset
      else
        obs_datasets := array_append_record(obs_datasets, tmp@dataset, 0)
      end if
    else
      if (.not.isvar("datasets")) then
        datasets = tmp@dataset
      else
        datasets := array_append_record(datasets, tmp@dataset, 0)
      end if
    end if
    tsize = dimsizes(tmp)
    delete(tmp)
  end do

  obs_names = get_unique_values(obs_datasets)
  dataset_names = get_unique_values(datasets)
  institutes = dataset_names
  institutes@_FillValue = default_fillvalue(typeof(institutes))
  do i = 0, dimsizes(dataset_names)-1
    tmp = get_modelling_center(dataset_names(i))
    institutes(i) = tmp(0)
    delete(tmp)
  end do
  institute_names = get_unique_values(institutes)

  do ii = 0, dimsizes(obs_names)-1
    paths = systemfunc("ls "+input_dir+"/ElNino_"+obs_names(ii)+"_*.nc")
    el_ref = new((/dimsizes(paths), tsize/), "float")
    do i = 0, dimsizes(paths)-1
      f = addfile(paths(i), "r")
      el_ref(i, :) = f->ENSOindex
;    if (.not.isvar("ens")) then
;      ens = el_ref(i, :)@ensemble
;    else
;      tmp = ens
;      delete(ens)
;      ens = array_append_record(tmp, el_ref(i, :)@ensemble, 0)
;      delete(tmp)
;    end if
      if (.not.isvar("year")) then
        year = el_ref@year
      else
        tmp = year
        delete(year)
        year = array_append_record(tmp, el_ref@year, 0)
        delete(tmp)
      end if
    end do
    ens = new(dimsizes(paths), "string")
    ens = "no_ens"
    el_ref@ens  = ens
    el_ref@year := year
    CalcMetrics(el_ref, duration_threshold, duration_threshold_unit)
    delete(paths)
    delete(ens)
    delete(year)
    if (.not.isdefined("el_obs")) then
      el_obs = NewList("fifo")
    end if
    ListAppend(el_obs, new(dimsizes(el_ref), typeof(el_ref)))
    i = ListCount(el_obs) - 1
    el_obs[i] = el_ref
    delete(el_ref)

    paths = systemfunc("ls "+input_dir+"/LaNina_"+obs_names(ii)+"_*.nc")
    la_ref = new((/dimsizes(paths), tsize/), "float")
    do i = 0, dimsizes(paths)-1
      f = addfile(paths(i), "r")
      la_ref(i, :) = f->ENSOindex
;    if (.not.isvar("ens")) then
;      ens = la_ref(i, :)@ensemble
;    else
;      tmp = ens
;      delete(ens)
;      ens = array_append_record(tmp, la_ref(i, :)@ensemble, 0)
;      delete(tmp)
;    end if
      if (.not.isvar("year")) then
        year = la_ref@year
      else
        tmp = year
        delete(year)
        year = array_append_record(tmp, la_ref@year, 0)
        delete(tmp)
      end if
    end do
    ens = new(dimsizes(paths), "string")
    ens = "no_ens"
    la_ref@ens  = ens
    la_ref@year := year
    CalcMetrics(la_ref, duration_threshold, duration_threshold_unit)
    delete(paths)
    delete(ens)
    delete(year)
    if (.not.isdefined("la_obs")) then
      la_obs = NewList("fifo")
    end if
    ListAppend(la_obs, new(dimsizes(la_ref), typeof(la_ref)))
    i = ListCount(la_obs) - 1
    la_obs[i] = la_ref
    delete(la_ref)
  end do

  do ii = 0, dimsizes(dataset_names)-1

    paths = systemfunc("ls " + input_dir + "/ElNino_" \
                       + dataset_names(ii) + "_r*_???.nc")
    if (.not.all(ismissing(paths))) then
      tmp = new((/dimsizes(paths), tsize/), "float")
      do i = 0, dimsizes(paths)-1
        f = addfile(paths(i), "r")
        tmp(i, :) = f->ENSOindex
        if (.not.isvar("ens")) then
          ens = tmp@ensemble
        else
          tmpens = ens
          delete(ens)
          ens = array_append_record(tmpens, tmp@ensemble, 0)
          delete(tmpens)
        end if
        if (.not.isvar("year")) then
          year = tmp@year
        else
          tmpyear = year
          delete(year)
          year = array_append_record(tmpyear, tmp@year, 0)
          delete(tmpyear)
        end if
      end do
      tmp@ens := ens
      tmp@year := year
      CalcMetrics(tmp, duration_threshold, duration_threshold_unit)
      delete(ens)
      delete(year)

      if (.not.isdefined("el")) then
        el = NewList("fifo")
      end if
      ListAppend(el, new(dimsizes(tmp), typeof(tmp)))
      i = ListCount(el) - 1
      el[i] = tmp
      delete(tmp)
    end if
    delete(paths)

    paths = systemfunc("ls " + input_dir + "/LaNina_"\
                       + dataset_names(ii) + "_r*_???.nc")
    if (.not.all(ismissing(paths))) then
      tmp = new((/dimsizes(paths), tsize/), "float")
      do i = 0, dimsizes(paths)-1
        f = addfile(paths(i), "r")
        tmp(i, :) = f->ENSOindex
        if (.not.isvar("ens")) then
          ens = tmp@ensemble
        else
          tmpens = ens
          delete(ens)
          ens = array_append_record(tmpens, tmp@ensemble, 0)
          delete(tmpens)
        end if
        if (.not.isvar("year")) then
          year = tmp@year
        else
          tmpyear = year
          delete(year)
          year = array_append_record(tmpyear, tmp@year, 0)
          delete(tmpyear)
        end if
      end do
      tmp@ens := ens
      tmp@year := year
      CalcMetrics(tmp, duration_threshold, duration_threshold_unit)
      delete(ens)
      delete(year)

      if (.not.isdefined("la")) then
        la = NewList("fifo")
      end if
      ListAppend(la, new(dimsizes(tmp), typeof(tmp)))
      i = ListCount(la) - 1
      la[i] = tmp
      delete(tmp)
    end if
    delete(paths)
  end do

;  do ii = 0, ListCount(el)-1
;    if (el[ii]@project.eq."CMIP6") then
;      if (.not.isvar("el_cmip6")) then
;        el_cmip6 = el[ii]
;      else
;        tmp = el_cmip6
;        delete(el_cmip6)
;        el_cmip6 = new((/dimsizes(tmp(:, 0))+dimsizes(el[ii](:, 0)), \
;                       dimsizes(tmp(0, :))/), typeof(tmp))
;        el_cmip6(0:dimsizes(tmp(:, 0))-1, :) = tmp
;        el_cmip6(dimsizes(tmp(:, 0)):,    :) = el[ii]
;        delete(tmp)
;      end if
;    elseif (el[ii]@project.eq."CMIP5") then
;      if (.not.isvar("el_cmip5")) then
;        el_cmip5 = el[ii]
;      else
;        tmp = el_cmip5
;        delete(el_cmip5)
;        el_cmip5 = new((/dimsizes(tmp(:, 0))+dimsizes(el[ii](:, 0)), \
;                       dimsizes(tmp(0, :))/), typeof(tmp))
;        el_cmip5(0:dimsizes(tmp(:, 0))-1, :) = tmp
;        el_cmip5(dimsizes(tmp(:, 0)):,    :) = el[ii]
;        delete(tmp)
;      end if
;    end if
;  end do
;  do ii = 0, ListCount(la)-1
;    if (la[ii]@project.eq."CMIP6") then
;      if (.not.isvar("la_cmip6")) then
;        la_cmip6 = la[ii]
;      else
;        tmp = la_cmip6
;        delete(la_cmip6)
;        la_cmip6 = new((/dimsizes(tmp(:, 0))+dimsizes(la[ii](:, 0)), \
;                       dimsizes(tmp(0, :))/), typeof(tmp))
;        la_cmip6(0:dimsizes(tmp(:, 0))-1, :) = tmp
;        la_cmip6(dimsizes(tmp(:, 0)):,    :) = la[ii]
;        delete(tmp)
;      end if
;    elseif (la[ii]@project.eq."CMIP5") then
;      if (.not.isvar("la_cmip5")) then
;        la_cmip5 = la[ii]
;      else
;        tmp = la_cmip5
;        delete(la_cmip5)
;        la_cmip5 = new((/dimsizes(tmp(:, 0))+dimsizes(la[ii](:, 0)), \
;                       dimsizes(tmp(0, :))/), typeof(tmp))
;        la_cmip5(0:dimsizes(tmp(:, 0))-1, :) = tmp
;        la_cmip5(dimsizes(tmp(:, 0)):,    :) = la[ii]
;        delete(tmp)
;      end if
;    end if
;  end do

  wks = get_wks("dummy_for_wks", DIAG_SCRIPT, \
                "enso_lifecycle")

; ========================= CMIP5, CMIP6 and obs ========================

  ENSOPlots = new(2, "graphic")
  dummy   = new(500, "graphic")
  idummy = 0

  lag = el_obs[0]&time
  lagpoly = array_append_record(lag, lag(::-1), 0)

  rests := True
  rests@gsnDraw           = False
  rests@gsnFrame          = False
  rests@vpWidthF = 0.7
  rests@vpHeightF = 0.5
  rests@xyDashPattern     = 0
  rests@xyMonoLineColor   = False
  rests@gsnYRefLine       = 0.
  rests@trXMinF = -24
  rests@trXMaxF = 47
  rests@trYMinF = -2.5
  rests@trYMaxF = 2.5
  rests@tmXTOn = False
;  rests@tmYROn = False
  rests@tmXBMode          = "Explicit"
  rests@tmXBLabelFontHeightF = 0.02
  rests@tmXBMajorThicknessF = 0.
  ; ======= Need adjustment =======
  rests@tmXBMajorLengthF = 0.009
  rests@tmXBMajorOutwardLengthF = 0.01
  rests@tmXBMinorOutwardLengthF = 0.01
  rests@tmYLMajorLengthF = 0.01
  rests@tmYLMajorOutwardLengthF = 0.01
  ; ===============================
  rests@tmYLLabelFontHeightF = 0.02
  rests@tmXBValues = (/-18, -6, 6, 18, 30, 42, 54, 66, 78/)
  rests@tmXBLabels = (/"yr(-2)", "yr(-1)", "yr(0)", "yr(1)", "yr(2)", \
                       "yr(3)", "yr(4)", "yr(5)", "yr(6)"/)
  rests@tmXBMinorValues   = lag
  rests@tmLabelAutoStride = False

  resp = True
  resp@gsFillOpacityF = 0.1
  resp@gsLineThicknessF = 0.4
  resp@gsLineColor = "transparent"

  txres = True
  txres@txFontHeightF = 0.02
  txres@txJust = "centerleft"

  rests@tiMainFontHeightF = 0.028
  rests@tiMainOffsetYF = -0.008
  rests@xyLineThicknessF = 0.01
  rests@tiMainString = "(a) El Nino"
  ENSOPlots(0) \
    = gsn_csm_xy(wks, (/rests@trXMinF, rests@trXMaxF/), (/0., 0./), rests)
  rests@tiMainString = "(b) La Nina"
  ENSOPlots(1) \
    = gsn_csm_xy(wks, (/rests@trXMinF, rests@trXMaxF/), (/0., 0./), rests)

  rests@xyLineColor = "grey50"
  rests@xyDashPattern = 1
  rests@xyLineThicknessF = 0.1
  do i = 0, 8
    plot = gsn_csm_xy(wks, (/-24.5+12.*i, -24.5+12*i/), \
                      (/rests@trYMinF, rests@trYMaxF/), rests)
    overlay(ENSOPlots(0), plot)
    plot = gsn_csm_xy(wks, (/-24.5+12.*i, -24.5+12.*i/), \
                      (/rests@trYMinF, rests@trYMaxF/), rests)
    overlay(ENSOPlots(1), plot)
  end do

  rests@xyDashPattern = 0
  rests@xyLineThicknessF = 0.25
  res = True
  res@project = "CMIP6"
  color_cmip6 = get_color(res)
  res@project = "CMIP5"
  color_cmip5 = get_color(res)

;  el_cmip5 = el[0](0, :)
;  el_cmip6 = el[0](0, :)
;  el_cmip5 = 0.
;  ncmip5 = 0
;  el_cmip6 = 0.
;  ncmip6 = 0
;  do i = 0, ListCount(el)-1
;    delete(rests@xyLineColor)
;    rests@xyLineColor = get_color(el[i])
;    rests@xyDashPattern =  get_lineindex(el[i])
;    rests@xyLineThicknessF = 0.5
;    rests@xyLineOpacityF = 0.5
;    if (multiyear_as_single .eq. "always") then
;      ts = dim_avg_n(where(conform(el[i]@nonoverlap, el[i], 0), \
;                           el[i], el[i]@_FillValue), 0)
;    else
;      ts = dim_avg_n(el[i], 0)
;    end if
;    plot = gsn_csm_xy(wks, lag, ts, rests)
;    overlay(ENSOPlots(0), plot)
;    if (el[i]@project.eq."CMIP5") then
;      el_cmip5 = el_cmip5 + ts
;      ncmip5 = ncmip5 + 1
;    elseif (el[i]@project.eq."CMIP6") then
;      el_cmip6 = el_cmip6 + ts
;      ncmip6 = ncmip6 + 1
;    end if
;  end do
;
;  el_cmip5 = el_cmip5 / tofloat(ncmip5)
;  el_cmip6 = el_cmip6 / tofloat(ncmip6)

  do i = 0, ListCount(el)-1
    dims = dimsizes(el[i])
    el[i]!0 = "event"
    el[i]&event = ispan(0, dims(0)-1, 1)
    delete(dims)
    if (multiyear_as_single .eq. "always") then
      ts = el[i](0, :)
      ts = dim_avg_n(where(conform(el[i], el[i]@nonoverlap, 0), \
                           el[i], el[i]@_FillValue), 0)
    else
      ts = dim_avg_n_Wrap(el[i], 0)
    end if
    if (el[i]@project.eq."CMIP5") then
      if (isvar("el_cmip5")) then
        dims = dimsizes(el_cmip5)
        dims(0) = dims(0) + 1
        tmp = el_cmip5
        delete(el_cmip5)
        el_cmip5 = new(dims, typeof(tmp))
        el_cmip5(0:dims(0)-2, :) = tmp
        el_cmip5(dims(0)-1, :) = (/ts/)
        delete(tmp)
        delete(dims)
      else
        el_cmip5 = new((/1, dimsizes(ts)/), typeof(ts))
        el_cmip5(0, :) = ts
      end if
    elseif (el[i]@project.eq."CMIP6") then
      if (isvar("el_cmip6")) then
        dims = dimsizes(el_cmip6)
        dims(0) = dims(0) + 1
        tmp = el_cmip6
        delete(el_cmip6)
        el_cmip6 = new(dims, typeof(tmp))
        el_cmip6(0:dims(0)-2, :) = tmp
        el_cmip6(dims(0)-1, :) = (/ts/)
        delete(tmp)
        delete(dims)
      else
        el_cmip6 = new((/1, dimsizes(ts)/), typeof(ts))
        el_cmip6(0, :) = ts
      end if
    end if
  end do
  el_cmip5!0 = "model"
  el_cmip6!0 = "model"
  dims = dimsizes(el_cmip5)
  el_cmip5&model = ispan(0, dims(0)-1, 1)
  dims = dimsizes(el_cmip6)
  el_cmip6&model = ispan(0, dims(0)-1, 1)
  delete(dims)

  resp = True
  resp@gsLineOpacityF = 0.2
  if (isvar("el_cmip5")) then
    resp@gsFillColor :=  get_color(el_cmip5)
    dummy(idummy) = gsn_add_polygon(wks, ENSOPlots(0), \
                                    lagpoly, make_poly(el_cmip5), resp)
    idummy = idummy + 1
  end if
  if (isvar("el_cmip6")) then
    resp@gsFillColor :=  get_color(el_cmip6)
    dummy(idummy) = gsn_add_polygon(wks, ENSOPlots(0), \
                                    lagpoly, make_poly(el_cmip6), resp)
    idummy = idummy + 1
  end if

  el_cmip5_mean = dim_avg_n_Wrap(el_cmip5, 0)
  el_cmip6_mean = dim_avg_n_Wrap(el_cmip6, 0)
  el_cmip5_mean@project = "CMIP5"
  el_cmip6_mean@project = "CMIP6"
  rests@xyDashPattern =  0
  rests@xyLineThicknessF = 3.
  rests@xyLineOpacityF = 1.
  rests@xyLineColor := get_color(el_cmip5_mean)
  plot = gsn_csm_xy(wks, lag, el_cmip5_mean, rests)
  overlay(ENSOPlots(0), plot)
  rests@xyLineColor := get_color(el_cmip6_mean)
  plot = gsn_csm_xy(wks, lag, el_cmip6_mean, rests)
  overlay(ENSOPlots(0), plot)

;  la_cmip5 = la[0](0, :)
;  la_cmip6 = la[0](0, :)
;  la_cmip5 = 0.
;  ncmip5 = 0
;  la_cmip6 = 0.
;  ncmip6 = 0
;  do i = 0, ListCount(la)-1
;    delete(rests@xyLineColor)
;    rests@xyLineColor = get_color(la[i])
;    rests@xyDashPattern =  get_lineindex(la[i])
;    rests@xyLineThicknessF = 0.5
;    rests@xyLineOpacityF = 0.5
;    if (multiyear_as_single .eq. "always") then
;      ts = dim_avg_n(where(conform(la[i]@nonoverlap, la[i], 0), \
;                           la[i], la[i]@_FillValue), 0)
;    else
;      ts = dim_avg_n(la[i], 0)
;    end if
;    plot = gsn_csm_xy(wks, lag, ts, rests)
;    overlay(ENSOPlots(1), plot)
;    if (la[i]@project.eq."CMIP5") then
;      la_cmip5 = la_cmip5 + ts
;      ncmip5 = ncmip5 + 1
;    elseif (la[i]@project.eq."CMIP6") then
;      la_cmip6 = la_cmip6 + ts
;      ncmip6 = ncmip6 + 1
;    end if
;    delete(ts)
;  end do
;  la_cmip5 = la_cmip5 / tofloat(ncmip5)
;  la_cmip6 = la_cmip6 / tofloat(ncmip6)

  do i = 0, ListCount(la)-1
    dims = dimsizes(la[i])
    la[i]!0 = "event"
    la[i]&event = ispan(0, dims(0)-1, 1)
    delete(dims)
    if (multiyear_as_single .eq. "always") then
      ts = la[i](0, :)
      ts = dim_avg_n(where(conform(la[i], la[i]@nonoverlap, 0), \
                           la[i], la[i]@_FillValue), 0)
    else
      ts = dim_avg_n_Wrap(la[i], 0)
    end if
    if (la[i]@project.eq."CMIP5") then
      if (isvar("la_cmip5")) then
        dims = dimsizes(la_cmip5)
        dims(0) = dims(0) + 1
        tmp = la_cmip5
        delete(la_cmip5)
        la_cmip5 = new(dims, typeof(tmp))
        la_cmip5(0:dims(0)-2, :) = tmp
        la_cmip5(dims(0)-1, :) = (/ts/)
        delete(tmp)
        delete(dims)
      else
        la_cmip5 = new((/1, dimsizes(ts)/), typeof(ts))
        la_cmip5(0, :) = ts
      end if
    elseif (la[i]@project.eq."CMIP6") then
      if (isvar("la_cmip6")) then
        dims = dimsizes(la_cmip6)
        dims(0) = dims(0) + 1
        tmp = la_cmip6
        delete(la_cmip6)
        la_cmip6 = new(dims, typeof(tmp))
        la_cmip6(0:dims(0)-2, :) = tmp
        la_cmip6(dims(0)-1, :) = (/ts/)
        delete(tmp)
        delete(dims)
      else
        la_cmip6 = new((/1, dimsizes(ts)/), typeof(ts))
        la_cmip6(0, :) = ts
      end if
    end if
  end do
  la_cmip5!0 = "model"
  la_cmip6!0 = "model"
  dims = dimsizes(la_cmip5)
  la_cmip5&model = ispan(0, dims(0)-1, 1)
  dims = dimsizes(la_cmip6)
  la_cmip6&model = ispan(0, dims(0)-1, 1)
  delete(dims)

  if (isvar("la_cmip5")) then
    resp@gsFillColor :=  get_color(la_cmip5)
    dummy(idummy) = gsn_add_polygon(wks, ENSOPlots(1), \
                                    lagpoly, make_poly(la_cmip5), resp)
    idummy = idummy + 1
  end if
  if (isvar("la_cmip6")) then
    resp@gsFillColor :=  get_color(la_cmip6)
    dummy(idummy) = gsn_add_polygon(wks, ENSOPlots(1), \
                                    lagpoly, make_poly(la_cmip6), resp)
    idummy = idummy + 1
  end if

  la_cmip5_mean = dim_avg_n_Wrap(la_cmip5, 0)
  la_cmip6_mean = dim_avg_n_Wrap(la_cmip6, 0)
  la_cmip5_mean@project = "CMIP5"
  la_cmip6_mean@project = "CMIP6"
  rests@xyDashPattern =  0
  rests@xyLineThicknessF = 3.
  rests@xyLineOpacityF = 1.
  rests@xyLineColor = get_color(la_cmip5_mean)
  plot = gsn_csm_xy(wks, lag, la_cmip5_mean, rests)
  overlay(ENSOPlots(1), plot)
  rests@xyLineColor = get_color(la_cmip6_mean)
  plot = gsn_csm_xy(wks, lag, la_cmip6_mean, rests)
  overlay(ENSOPlots(1), plot)

  do i = 0, ListCount(el_obs)-1
    delete(rests@xyLineColor)
    rests@xyLineColor = get_color(el_obs[i])
    rests@xyDashPattern =  get_lineindex(el_obs[i])
    rests@xyLineThicknessF = 1.5
    rests@xyLineOpacityF = 1.
    if (multiyear_as_single .eq. "always") then
      plot \
        = gsn_csm_xy(wks, lag, \
                     dim_avg_n(where(conform(el_obs[i], \
                                             el_obs[i]@nonoverlap, 0), \
                                     el_obs[i], el_obs[i]@_FillValue), 0), \
                     rests)
    else
      plot = gsn_csm_xy(wks, lag, dim_avg_n(el_obs[i], 0), rests)
    end if
    overlay(ENSOPlots(0), plot)
  end do
  do i = 0, ListCount(la_obs)-1
    delete(rests@xyLineColor)
    rests@xyLineColor = get_color(la_obs[i])
    rests@xyDashPattern =  get_lineindex(la_obs[i])
    rests@xyLineThicknessF = 1.5
    rests@xyLineOpacityF = 1.
    if (multiyear_as_single .eq. "always") then
      plot \
        = gsn_csm_xy(wks, lag, \
                     dim_avg_n(where(conform(la_obs[i], \
                                             la_obs[i]@nonoverlap, 0), \
                                     la_obs[i], la_obs[i]@_FillValue), 0), \
                     rests)
    else
      plot = gsn_csm_xy(wks, lag, dim_avg_n(la_obs[i], 0), rests)
    end if
    overlay(ENSOPlots(1), plot)
  end do

;  if (isvar("el_cmip5")) then
;    rests@xyLineColor = color_cmip5
;    plot = gsn_csm_xy(wks, lag, dim_avg_n(el_cmip5, 0), rests)
;    overlay(ENSOPlots(0), plot)
;  end if
;  if (isvar("la_cmip5")) then
;    rests@xyLineColor = color_cmip5
;    plot = gsn_csm_xy(wks, lag, dim_avg_n(la_cmip5, 0), rests)
;    overlay(ENSOPlots(1), plot)
;  end if
;  if (isvar("el_cmip6")) then
;    rests@xyLineColor = color_cmip6
;    plot = gsn_csm_xy(wks, lag, dim_avg_n(el_cmip6, 0), rests)
;    overlay(ENSOPlots(0), plot)
;  end if
;  if (isvar("la_cmip6")) then
;    rests@xyLineColor = color_cmip6
;    plot = gsn_csm_xy(wks, lag, dim_avg_n(la_cmip6, 0), rests)
;    overlay(ENSOPlots(1), plot)
;  end if

  j = 0
  lres = True
  do i = 0, ListCount(el_obs)-1
    y = rests@trYMaxF-0.2-0.3*j
    lres@gsLineColor = get_color(el_obs[i])
    lres@gsLineDashPattern =  get_lineindex(el_obs[i])
    lres@gsLineThicknessF = 1.5
    lres@gsLineOpacityF = 1.
    dummy(idummy) = gsn_add_polyline(wks, ENSOPlots(0), \
                                     (/-23., -18./), (/y, y/), lres)
    idummy = idummy + 1
    txres@txFontColor := get_color(el_obs[i])
    dummy(idummy) = gsn_add_text(wks, ENSOPlots(0), \
                                 el_obs[i]@dataset, -17., y, txres)
    idummy = idummy + 1
    j = j + 1
  end do
  if (isvar("el_cmip6")) then
    y = rests@trYMaxF-0.2-0.3*j
    resp@gsLineOpacityF = 0.2
    resp@gsFillColor :=  get_color(el_cmip6)
    dummy(idummy) = gsn_add_polygon(wks, ENSOPlots(0), \
                                    (/-23., -18., -18., -23./), \
                                    (/y+0.1, y+0.1, y-0.1, y-0.1/), resp)
    idummy = idummy + 1
    lres@gsLineColor = get_color(el_cmip6)
    lres@gsLineDashPattern =  get_lineindex(el_cmip6)
    lres@gsLineThicknessF = 3.
    lres@gsLineOpacityF = 1.
    dummy(idummy) = gsn_add_polyline(wks, ENSOPlots(0), \
                                     (/-23., -18./), (/y, y/), lres)
    idummy = idummy + 1
    txres@txFontColor := get_color(el_cmip6)
    dummy(idummy) = gsn_add_text(wks, ENSOPlots(0), \
                                 "CMIP6", -17., y, txres)
    idummy = idummy + 1
    j = j + 1
  end if
  if (isvar("el_cmip5")) then
    y = rests@trYMaxF-0.2-0.3*j
    resp@gsLineOpacityF = 0.2
    resp@gsFillColor :=  get_color(el_cmip5)
    dummy(idummy) = gsn_add_polygon(wks, ENSOPlots(0), \
                                    (/-23., -18., -18., -23./), \
                                    (/y+0.1, y+0.1, y-0.1, y-0.1/), resp)
    idummy = idummy + 1
    lres@gsLineColor = get_color(el_cmip5)
    lres@gsLineDashPattern =  get_lineindex(el_cmip5)
    lres@gsLineThicknessF = 3.
    lres@gsLineOpacityF = 1.
    dummy(idummy) = gsn_add_polyline(wks, ENSOPlots(0), \
                                     (/-23., -18./), (/y, y/), lres)
    idummy = idummy + 1
    txres@txFontColor := get_color(el_cmip5)
    dummy(idummy) = gsn_add_text(wks, ENSOPlots(0), \
                                 "CMIP5", -17., y, txres)
    idummy = idummy + 1
    j = j + 1
  end if

  pres = True
  pres@gsnFrame = False
  pres@gsnMaximize = False
;  pres@gsnPanelXWhiteSpacePercent = 0.
;  pres@gsnPanelYWhiteSpacePercent = 8.
  pres@gsnPanelLeft = 0.08
  pres@gsnPanelRight = 0.92
  pres@gsnPanelBottom = 0.63
;  pres@gsnPanelFigureStrings = (/ "(a) El Nino", "(b) La Nina" /)
;  pres@gsnPanelFigureStringsFontHeightF = 0.015
  pres@gsnPanelFigureStringsPerimOn = False
  pres@amJust   = "TopCenter"
  pres@amOrthogonalPosF = -0.62
  pres@amParallelPosF = 0.
  gsn_panel(wks, ENSOPlots, (/1, 2/), pres)

; =========================== Individual Models =============================

  plots = new((/2, 2/), "graphic")
  w = 0.3  ; width of bars for MME

  do ii = 0, ListCount(el)-1
    if (isvar("centers")) then
      tmp = centers
      delete(centers)
      center = get_modelling_center(el[ii]@dataset)
      centers = array_append_record(tmp, center(0), 0)
      delete(tmp)
      delete(center)
    else
      centers = get_modelling_center(el[ii]@dataset)
    end if
  end do
  centers := get_unique_values(centers)
  sqsort(centers)
  ncenters = dimsizes(centers)

  ncols = ncenters
  if (isvar("el_cmip6")) then
    ncols = ncols + 1
    if (isvar("el_cmip5")) then
      icolcmip6 = -2
    else
      icolcmip6 = -1
    end if
  end if
  if (isvar("el_cmip5")) then
    ncols = ncols + 1
    icolcmip5 = -1
  end if

  col = ispan(ncenters-ncols, ncenters-1, 1)
  names = new(ncols, "string")
  names!0 = "col"
  names&col = col
  names({0:}) = (/centers/)
  if (isvar("el_cmip6")) then
    names({icolcmip6}) = "CMIP6"
  end if
  if (isvar("el_cmip5")) then
    names({icolcmip5}) = "CMIP5"
  end if

  res := True
  res@gsnDraw = False
  res@gsnFrame = False
  res@trXMinF = col(0) - 0.5
  res@trXMaxF = col(ncols-1) + 0.5
  res@vpWidthF = 0.7
  res@vpHeightF = 0.3
  res@tmXTOn = False
  res@tmXMajorGrid = True
  res@tmYMajorGrid = True
  res@tmXMajorGridLineDashPattern = 0  ; 11
  res@tmYMajorGridLineDashPattern = 0  ; 11
  res@tmXMajorGridLineColor = "grey"
  res@tmYMajorGridLineColor = "transparent"  ; "grey"
  res@tmXMajorGridThicknessF = 0.005
  res@tmYMajorGridThicknessF = 0.  ; 0.005

  res@tmXBMode = "Explicit"
  res@tmXBLabelFontHeightF = 0.015
  res@tmXBValues = col
  res@tmXBLabels = names
  res@tmXBLabelAngleF = 60.  ; 45.
  res@tmXBLabelJust = "CenterRight"

  mkres = True
  mkres@gsMarkerThicknessF = 2.0
  mkres@gsMarkerSizeF    = 0.01
  lnres = True
  lnres@gsLineThicknessF = 2.

  res@trYMaxF = 28.  ; 38.
  res@trYMinF = 8.  ; 10.
  res@tmYLMode = "Manual"
  res@tmYLTickStartF = 0.
  res@tmYLTickEndF = 132.
  res@tmYLTickSpacingF = 6.
  res@tmYLMinorPerMajor = 2
  res@tmYLLabelFontHeightF = 0.018
  res@tiMainFontHeightF = 0.025
  res@tiMainOffsetYF = -0.008
  if (isvar("y")) then
    delete(y)
  end if
  do ii = 0, ListCount(el_obs)-1
    res@xyLineColor = get_color(el_obs[ii])
    res@xyLineThicknessF = 1.
    res@xyDashPattern = get_lineindex(el_obs[ii])
    if (ii.eq.0) then
      res@tiMainString = "(c) El Nino duration (months)"
    else
      res@tiMainString = ""
    end if
    if (duration_individual) then
      if (any(multiyear_as_single .eq. (/"always", "metric_only"/))) then
        duration = el_obs[ii]@duration
        y = avg(where(el_obs[ii]@nonoverlap, duration, duration@_FillValue))
        delete(duration)
      else
        y = avg(el_obs[ii]@duration)
      end if
    else
      y = duration_composite(el_obs[ii], multiyear_as_single, \
                             duration_threshold, duration_threshold_unit)
    end if
    if (ii.eq.0) then
      plots(0, 0) = gsn_csm_xy(wks, (/res@trXMinF, res@trXMaxF/), \
                               (/y, y/), res)
    else
      plot = gsn_csm_xy(wks, (/res@trXMinF, res@trXMaxF/), (/y, y/), res)
      overlay(plots(0, 0), plot)
    end if
    if (ii.eq.0) then
      res@tiMainString = "(d) La Nina duration (months)"
    else
      res@tiMainString = ""
    end if
    if (duration_individual) then
      if (any(multiyear_as_single .eq. (/"always", "metric_only"/))) then
        duration = la_obs[ii]@duration
        y = avg(where(la_obs[ii]@nonoverlap, duration, duration@_FillValue))
        delete(duration)
      else
        y = avg(la_obs[ii]@duration)
      end if
    else
      y = duration_composite(la_obs[ii], multiyear_as_single, \
                             duration_threshold, duration_threshold_unit)
    end if
    if (ii.eq.0) then
      plots(0, 1) = gsn_csm_xy(wks, (/res@trXMinF, res@trXMaxF/), \
                               (/y, y/), res)
    else
      plot = gsn_csm_xy(wks, (/res@trXMinF, res@trXMaxF/), (/y, y/), res)
      overlay(plots(0, 1), plot)
    end if
  end do

  nmodel_cmip5 = 0
  nmodel_cmip6 = 0
  do ii = 0, ListCount(el)-1
    center = get_modelling_center(el[ii]@dataset)
    i = ind(center(0).eq.centers)
    mkres@gsMarkerColor    = get_color(el[ii])
    mkres@gsMarkerIndex    = get_markerindex(el[ii])
    if (duration_individual) then
      if (any(multiyear_as_single .eq. (/"always", "metric_only"/))) then
        duration = el[ii]@duration
        y = avg(where(el[ii]@nonoverlap, duration, duration@_FillValue))
        delete(duration)
      else
        y = avg(el[ii]@duration)
      end if
    else
      y = duration_composite(el[ii], multiyear_as_single, \
                             duration_threshold, duration_threshold_unit)
    end if
    dummy(idummy) = gsn_add_polymarker(wks, plots(0, 0), i, y, mkres)
    idummy = idummy + 1
    delete(center)
    if (y.gt.0.) then
      if (el[ii]@project.eq."CMIP6") then
        nmodel_cmip6 = nmodel_cmip6 + 1
        if (isvar("el_cmip6_composited")) then
          tmp = el_cmip6_composited
          delete(el_cmip6_composited)
          el_cmip6_composited = new(nmodel_cmip6, typeof(y))
          el_cmip6_composited = array_append_record(tmp, y, 0)
          delete(tmp)
        else
          el_cmip6_composited = y
        end if
      elseif (el[ii]@project.eq."CMIP5") then
        nmodel_cmip5 = nmodel_cmip5 + 1
        if (isvar("el_cmip5_composited")) then
          tmp = el_cmip5_composited
          delete(el_cmip5_composited)
          el_cmip5_composited = new(nmodel_cmip5, typeof(y))
          el_cmip5_composited = array_append_record(tmp, y, 0)
          delete(tmp)
        else
          el_cmip5_composited = y
        end if
      end if
    end if
    delete(y)
  end do
  if (nmodel_cmip6.gt.2) then
    median = dim_median(el_cmip6_composited)
    qsort(el_cmip6_composited)
    x5   = round(0.05*dimsizes(el_cmip6_composited), 3) - 1
    x25  = round(0.25*dimsizes(el_cmip6_composited), 3) - 1
    x75  = round(0.75*dimsizes(el_cmip6_composited), 3) - 1
    x95  = round(0.95*dimsizes(el_cmip6_composited), 3) - 1
    percentile5  = el_cmip6_composited(x5)
    percentile25 = el_cmip6_composited(x25)
    percentile75 = el_cmip6_composited(x75)
    percentile95 = el_cmip6_composited(x95)
;    xmax = max(el_cmip6_composited)
;    xmin = min(el_cmip6_composited)
    median@project = "CMIP6"
    polyres = True
    polyres@gsFillColor = get_color(median)
    polyres@gsFillOpacityF = 1.
    dummy(idummy) \
      = gsn_add_polygon(wks, plots(0, 0), \
                        (/icolcmip6-w, icolcmip6-w, \
                          icolcmip6+w, icolcmip6+w/), \
                        (/percentile25, percentile75, \
                          percentile75, percentile25/), \
                        polyres)
    idummy = idummy + 1
    lnres@gsLineColor := get_color(median)
    dummy(idummy) = gsn_add_polyline(wks, plots(0, 0), \
                                     (/icolcmip6, icolcmip6/), \
                                     (/percentile5, percentile95/), lnres)
    idummy = idummy + 1
    lnres@gsLineColor := "white"
    dummy(idummy) = gsn_add_polyline(wks, plots(0, 0), \
                                     (/icolcmip6-w, icolcmip6+w/), \
                                     (/median, median/), lnres)
    idummy = idummy + 1
    delete(el_cmip6_composited)
  end if
  if (nmodel_cmip5.gt.2) then
    median = dim_median(el_cmip5_composited)
    qsort(el_cmip5_composited)
    x5   = round(0.05*dimsizes(el_cmip5_composited), 3) - 1
    x25  = round(0.25*dimsizes(el_cmip5_composited), 3) - 1
    x75  = round(0.75*dimsizes(el_cmip5_composited), 3) - 1
    x95  = round(0.95*dimsizes(el_cmip5_composited), 3) - 1
    percentile5  = el_cmip5_composited(x5)
    percentile25 = el_cmip5_composited(x25)
    percentile75 = el_cmip5_composited(x75)
    percentile95 = el_cmip5_composited(x95)
    median@project = "CMIP5"
    polyres = True
    polyres@gsFillColor      = get_color(median)
    polyres@gsFillOpacityF = 1.
    dummy(idummy) \
      = gsn_add_polygon(wks, plots(0, 0), \
                        (/icolcmip5-w, icolcmip5-w, \
                          icolcmip5+w, icolcmip5+w/), \
                        (/percentile25, percentile75, \
                          percentile75, percentile25/), polyres)
    idummy = idummy + 1
    lnres@gsLineColor := get_color(median)
    dummy(idummy) = gsn_add_polyline(wks, plots(0, 0), \
                                     (/icolcmip5, icolcmip5/), \
                                     (/percentile5, percentile95/), lnres)
    idummy = idummy + 1
    lnres@gsLineColor := "white"
    dummy(idummy) = gsn_add_polyline(wks, plots(0, 0), \
                                     (/icolcmip5-w, icolcmip5+w/), \
                                     (/median, median/), lnres)
    idummy = idummy + 1
    delete(el_cmip5_composited)
  end if

  nmodel_cmip5 = 0
  nmodel_cmip6 = 0
  do ii = 0, ListCount(la)-1
    center = get_modelling_center(la[ii]@dataset)
    i = ind(center(0).eq.centers)
    mkres@gsMarkerColor    = get_color(la[ii])
    mkres@gsMarkerIndex    = get_markerindex(la[ii])
    if (duration_individual) then
      if (any(multiyear_as_single .eq. (/"always", "metric_only"/))) then
        duration = la[ii]@duration
        y = avg(where(la[ii]@nonoverlap, duration, duration@_FillValue))
        delete(duration)
      else
        y = avg(la[ii]@duration)
      end if
    else
      y = duration_composite(la[ii], multiyear_as_single, \
                             duration_threshold, duration_threshold_unit)
    end if
    dummy(idummy) = gsn_add_polymarker(wks, plots(0, 1), i, y, mkres)
    idummy = idummy + 1
    delete(center)
    if (y.gt.0.) then
      if (la[ii]@project.eq."CMIP6") then
        nmodel_cmip6 = nmodel_cmip6 + 1
        if (isvar("la_cmip6_composited")) then
          tmp = la_cmip6_composited
          delete(la_cmip6_composited)
          la_cmip6_composited = new(nmodel_cmip6, typeof(y))
          la_cmip6_composited = array_append_record(tmp, y, 0)
          delete(tmp)
        else
          la_cmip6_composited = y
        end if
      elseif (la[ii]@project.eq."CMIP5") then
        nmodel_cmip5 = nmodel_cmip5 + 1
        if (isvar("la_cmip5_composited")) then
          tmp = la_cmip5_composited
          delete(la_cmip5_composited)
          la_cmip5_composited = new(nmodel_cmip5, typeof(y))
          la_cmip5_composited = array_append_record(tmp, y, 0)
          delete(tmp)
        else
          la_cmip5_composited = y
        end if
      end if
    end if
    delete(y)
  end do
  if (nmodel_cmip6.gt.2) then
    median = dim_median(la_cmip6_composited)
    qsort(la_cmip6_composited)
    x5   = round(0.05*dimsizes(la_cmip6_composited), 3) - 1
    x25  = round(0.25*dimsizes(la_cmip6_composited), 3) - 1
    x75  = round(0.75*dimsizes(la_cmip6_composited), 3) - 1
    x95  = round(0.95*dimsizes(la_cmip6_composited), 3) - 1
    percentile5  = la_cmip6_composited(x5)
    percentile25 = la_cmip6_composited(x25)
    percentile75 = la_cmip6_composited(x75)
    percentile95 = la_cmip6_composited(x95)
    median@project = "CMIP6"
    polyres = True
    polyres@gsFillColor = get_color(median)
    dummy(idummy) \
      = gsn_add_polygon(wks, plots(0, 1), \
                        (/icolcmip6-w, icolcmip6-w, \
                          icolcmip6+w, icolcmip6+w/), \
                        (/percentile25, percentile75, \
                          percentile75, percentile25/), polyres)
    idummy = idummy + 1
    lnres@gsLineColor := get_color(median)
    dummy(idummy) = gsn_add_polyline(wks, plots(0, 1), \
                                     (/icolcmip6, icolcmip6/), \
                                     (/percentile5, percentile95/), lnres)
    idummy = idummy + 1
    lnres@gsLineColor := "white"
    dummy(idummy) = gsn_add_polyline(wks, plots(0, 1), \
                                     (/icolcmip6-w, icolcmip6+w/), \
                                     (/median, median/), lnres)
    idummy = idummy + 1
    delete(la_cmip6_composited)
  end if
  if (nmodel_cmip5.gt.2) then
    median = dim_median(la_cmip5_composited)
    qsort(la_cmip5_composited)
    x5   = round(0.05*dimsizes(la_cmip5_composited), 3) - 1
    x25  = round(0.25*dimsizes(la_cmip5_composited), 3) - 1
    x75  = round(0.75*dimsizes(la_cmip5_composited), 3) - 1
    x95  = round(0.95*dimsizes(la_cmip5_composited), 3) - 1
    percentile5  = la_cmip5_composited(x5)
    percentile25 = la_cmip5_composited(x25)
    percentile75 = la_cmip5_composited(x75)
    percentile95 = la_cmip5_composited(x95)
    xmax = max(la_cmip5_composited)
    xmin = min(la_cmip5_composited)
    median@project = "CMIP5"
    polyres = True
    polyres@gsFillColor = get_color(median)
    dummy(idummy) \
      = gsn_add_polygon(wks, plots(0, 1), \
                        (/icolcmip5-w, icolcmip5-w, \
                          icolcmip5+w, icolcmip5+w/), \
                        (/percentile25, percentile75, \
                          percentile75, percentile25/), polyres)
    idummy = idummy + 1
    lnres@gsLineColor := get_color(median)
    dummy(idummy) = gsn_add_polyline(wks, plots(0, 1), \
                                     (/icolcmip5, icolcmip5/), \
                                     (/percentile5, percentile95/), lnres)
    idummy = idummy + 1
    lnres@gsLineColor := "white"
    dummy(idummy) = gsn_add_polyline(wks, plots(0, 1), \
                                     (/icolcmip5-w, icolcmip5+w/), \
                                     (/median, median/), lnres)
    idummy = idummy + 1
    delete(la_cmip5_composited)
  end if

; -----------------------------------------------------------

;  res@trYMaxF = 126.
;  res@trYMinF = 18.
;  do ii = 0, ListCount(el_obs)-1
;    res@xyLineColor = get_color(el_obs[ii])
;    res@xyLineThicknessF = 2.
;    res@xyDashPattern = 1
;    if (ii.eq.0) then
;      res@tiMainString = "(e) El Nino-to-El Nino interval (months)"
;    else
;      res@tiMainString = ""
;    end if
;    y = el_obs[ii]@interval_to_subsequent_same
;    plots(1, 0) = gsn_csm_xy(wks, (/res@trXMinF, res@trXMaxF/), (/y, y/), res)
;    if (ii.eq.0) then
;      res@tiMainString = "(f) La Nina-to-La Nina interval (months)"
;    else
;      res@tiMainString = ""
;    end if
;    y = la_obs[ii]@interval_to_subsequent_same
;    plots(1, 1) = gsn_csm_xy(wks, (/res@trXMinF, res@trXMaxF/), (/y, y/), res)
;  end do

;  nmodel_cmip5 = 0
;  nmodel_cmip6 = 0
;  do ii = 0, ListCount(el)-1
;    center = get_modelling_center(el[ii]@dataset)
;    i = ind(center(0).eq.centers)
;    mkres@gsMarkerColor    = get_color(el[ii])
;    mkres@gsMarkerIndex    = get_markerindex(el[ii])
;    y = el[ii]@interval_to_subsequent_same
;    dummy(idummy) = gsn_add_polymarker(wks, plots(1, 0), i, y, mkres)
;    idummy = idummy + 1
;    delete(center)
;    if (y.gt.0.) then
;      if (el[ii]@project.eq."CMIP6") then
;        nmodel_cmip6 = nmodel_cmip6 + 1
;        if (isvar("el_cmip6_composited")) then
;          tmp = el_cmip6_composited
;          delete(el_cmip6_composited)
;          el_cmip6_composited = new(nmodel_cmip6, typeof(y))
;          el_cmip6_composited = array_append_record(tmp, y, 0)
;          delete(tmp)
;        else
;          el_cmip6_composited = y
;        end if
;      elseif (el[ii]@project.eq."CMIP5") then
;        nmodel_cmip5 = nmodel_cmip5 + 1
;        if (isvar("el_cmip5_composited")) then
;          tmp = el_cmip5_composited
;          delete(el_cmip5_composited)
;          el_cmip5_composited = new(nmodel_cmip5, typeof(y))
;          el_cmip5_composited = array_append_record(tmp, y, 0)
;          delete(tmp)
;        else
;          el_cmip5_composited = y
;        end if
;      end if
;    end if
;  end do
;  if (nmodel_cmip6.gt.2) then
;    median = dim_median(el_cmip6_composited)
;    qsort(el_cmip6_composited)
;    x5   = round(0.05*dimsizes(el_cmip6_composited), 3) - 1
;    x25  = round(0.25*dimsizes(el_cmip6_composited), 3) - 1
;    x75  = round(0.75*dimsizes(el_cmip6_composited), 3) - 1
;    x95  = round(0.95*dimsizes(el_cmip6_composited), 3) - 1
;    percentile5  = el_cmip6_composited(x5)
;    percentile25 = el_cmip6_composited(x25)
;    percentile75 = el_cmip6_composited(x75)
;    percentile95 = el_cmip6_composited(x95)
;    xmax = max(el_cmip6_composited)
;    xmin = min(el_cmip6_composited)
;    median@project = "CMIP6"
;    polyres = True
;    polyres@gsFillColor = get_color(median)
;    dummy(idummy) \
;      = gsn_add_polygon(wks, plots(1, 0), \
;                        (/icolcmip6-w, icolcmip6-w, \
;                          icolcmip6+w, icolcmip6+w/), \
;                        (/percentile25, percentile75, \
;                          percentile75, percentile25/), polyres)
;    idummy = idummy + 1
;    lnres@gsLineColor := get_color(median)
;    dummy(idummy) = gsn_add_polyline(wks, plots(1, 0), \
;                                     (/icolcmip6, icolcmip6/), \
;                                     (/percentile5, percentile95/), lnres)
;    idummy = idummy + 1
;    lnres@gsLineColor := "white"
;    dummy(idummy) = gsn_add_polyline(wks, plots(1, 0), \
;                                     (/icolcmip6-w, icolcmip6+w/), \
;                                     (/median, median/), lnres)
;    idummy = idummy + 1
;    delete(el_cmip6_composited)
;  end if
;  if (nmodel_cmip5.gt.2) then
;    median = dim_median(el_cmip5_composited)
;    qsort(el_cmip5_composited)
;    x5   = round(0.05*dimsizes(el_cmip5_composited), 3) - 1
;    x25  = round(0.25*dimsizes(el_cmip5_composited), 3) - 1
;    x75  = round(0.75*dimsizes(el_cmip5_composited), 3) - 1
;    x95  = round(0.95*dimsizes(el_cmip5_composited), 3) - 1
;    percentile5  = el_cmip5_composited(x5)
;    percentile25 = el_cmip5_composited(x25)
;    percentile75 = el_cmip5_composited(x75)
;    percentile95 = el_cmip5_composited(x95)
;    median@project = "CMIP5"
;    polyres = True
;    polyres@gsFillColor = get_color(median)
;    dummy(idummy) \
;      = gsn_add_polygon(wks, plots(1, 0), \
;                        (/icolcmip5-w, icolcmip5-w, \
;                          icolcmip5+w, icolcmip5+w/), \
;                        (/percentile25, percentile75, \
;                          percentile75, percentile25/), polyres)
;    idummy = idummy + 1
;    lnres@gsLineColor := get_color(median)
;    dummy(idummy) = gsn_add_polyline(wks, plots(1, 0), \
;                                     (/icolcmip5, icolcmip5/), \
;                                     (/percentile5, percentile95/), lnres)
;    idummy = idummy + 1
;    lnres@gsLineColor := "white"
;    dummy(idummy) = gsn_add_polyline(wks, plots(1, 0), \
;                                     (/icolcmip5-w, icolcmip5+w/), \
;                                     (/median, median/), lnres)
;    idummy = idummy + 1
;    delete(el_cmip5_composited)
;  end if

;  nmodel_cmip5 = 0
;  nmodel_cmip6 = 0
;  do ii = 0, ListCount(la)-1
;    center = get_modelling_center(la[ii]@dataset)
;    i = ind(center(0).eq.centers)
;    mkres@gsMarkerColor    = get_color(la[ii])
;    mkres@gsMarkerIndex    = get_markerindex(la[ii])
;    y = la[ii]@interval_to_subsequent_same
;    dummy(idummy) = gsn_add_polymarker(wks, plots(1, 1), i, y, mkres)
;    idummy = idummy + 1
;    delete(center)
;    if (y.gt.0) then
;      if (la[ii]@project.eq."CMIP6") then
;        nmodel_cmip6 = nmodel_cmip6 + 1
;        if (isvar("la_cmip6_composited")) then
;          tmp = la_cmip6_composited
;          delete(la_cmip6_composited)
;          la_cmip6_composited = new(nmodel_cmip6, typeof(y))
;          la_cmip6_composited = array_append_record(tmp, y, 0)
;          delete(tmp)
;        else
;          la_cmip6_composited = y
;        end if
;      elseif (la[ii]@project.eq."CMIP5") then
;        nmodel_cmip5 = nmodel_cmip5 + 1
;        if (isvar("la_cmip5_composited")) then
;          tmp = la_cmip5_composited
;          delete(la_cmip5_composited)
;          la_cmip5_composited = new(nmodel_cmip5, typeof(y))
;          la_cmip5_composited = array_append_record(tmp, y, 0)
;          delete(tmp)
;        else
;          la_cmip5_composited = y
;        end if
;      end if
;    end if
;  end do
;  if (nmodel_cmip6.gt.2) then
;    median = dim_median(la_cmip6_composited)
;    qsort(la_cmip6_composited)
;    x5   = round(0.05*dimsizes(la_cmip6_composited), 3) - 1
;    x25  = round(0.25*dimsizes(la_cmip6_composited), 3) - 1
;    x75  = round(0.75*dimsizes(la_cmip6_composited), 3) - 1
;    x95  = round(0.95*dimsizes(la_cmip6_composited), 3) - 1
;    percentile5  = la_cmip6_composited(x5)
;    percentile25 = la_cmip6_composited(x25)
;    percentile75 = la_cmip6_composited(x75)
;    percentile95 = la_cmip6_composited(x95)
;    median@project = "CMIP6"
;    polyres = True
;    polyres@gsFillColor = get_color(median)
;    dummy(idummy) \
;      = gsn_add_polygon(wks, plots(1, 1), \
;                        (/icolcmip6-w, icolcmip6-w, \
;                          icolcmip6+w, icolcmip6+w/), \
;                        (/percentile25, percentile75, \
;                          percentile75, percentile25/), polyres)
;    idummy = idummy + 1
;    lnres@gsLineColor := get_color(median)
;    dummy(idummy) = gsn_add_polyline(wks, plots(1, 1), \
;                                     (/icolcmip6, icolcmip6/), \
;                                     (/percentile5, percentile95/), lnres)
;    idummy = idummy + 1
;    lnres@gsLineColor := "white"
;    dummy(idummy) = gsn_add_polyline(wks, plots(1, 1), \
;                                     (/icolcmip6-w, icolcmip6+w/), \
;                                     (/median, median/), lnres)
;    idummy = idummy + 1
;    delete(la_cmip6_composited)
;  end if
;  if (nmodel_cmip5.gt.2) then
;    median = dim_median(la_cmip5_composited)
;    qsort(la_cmip5_composited)
;    x5   = round(0.05*dimsizes(la_cmip5_composited), 3) - 1
;    x25  = round(0.25*dimsizes(la_cmip5_composited), 3) - 1
;    x75  = round(0.75*dimsizes(la_cmip5_composited), 3) - 1
;    x95  = round(0.95*dimsizes(la_cmip5_composited), 3) - 1
;    percentile5  = la_cmip5_composited(x5)
;    percentile25 = la_cmip5_composited(x25)
;    percentile75 = la_cmip5_composited(x75)
;    percentile95 = la_cmip5_composited(x95)
;    median@project = "CMIP5"
;    polyres = True
;    polyres@gsFillColor = get_color(median)
;    dummy(idummy) \
;      = gsn_add_polygon(wks, plots(1, 1), \
;                        (/icolcmip5-w, icolcmip5-w, \
;                          icolcmip5+w, icolcmip5+w/), \
;                        (/percentile25, percentile75, \
;                          percentile75, percentile25/), polyres)
;    idummy = idummy + 1
;    lnres@gsLineColor := get_color(median)
;    dummy(idummy) = gsn_add_polyline(wks, plots(1, 1), \
;                                     (/icolcmip5, icolcmip5/), \
;                                     (/percentile5, percentile95/), lnres)
;    idummy = idummy + 1
;    lnres@gsLineColor := "white"
;    dummy(idummy) = gsn_add_polyline(wks, plots(1, 1), \
;                                     (/icolcmip5-w, icolcmip5+w/), \
;                                     (/median, median/), lnres)
;    idummy = idummy + 1
;    delete(la_cmip5_composited)
;  end if

  pres := True
  pres@gsnMaximize = False
;  pres@gsnPanelXWhiteSpacePercent = 0.2
;  pres@gsnPanelYWhiteSpacePercent = 8.
  pres@gsnPanelLeft = 0.
  pres@gsnPanelRight = 1.
  pres@gsnPanelTop = 0.63
  pres@gsnPanelFigureStringsFontHeightF = 0.012
  pres@gsnPanelFigureStringsPerimOn = False
  pres@amJust   = "TopCenter"
  pres@amOrthogonalPosF = -0.68
  pres@amParallelPosF = 0.
  gsn_panel(wks, ndtooned(plots), (/2, 2/), pres)

end
