; ****************************************************************************
; Fig_3_41_ohc.ncl
;
; Description:
; Generates for Fig. 3.41 the input OHC file by collecting all CMIP6 historical
; and hist-nat results alongside with the observations
;
; Creator: Lisa Bock (lisa.bock@dlr.de)
; Creation Date: 02 March 2021 
; ****************************************************************************

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"


begin

  start_year = 1850
  end_year = 2014
  ntime = end_year - start_year + 1
  time = new(ntime, integer)
  time = ispan(start_year, end_year, 1)

  model_arr_stat = new((/3, 3, ntime/), double)
  model_arr_stat!0 = "exp"
  model_arr_stat!1 = "stat"
  model_arr_stat!2 = "time"
  model_arr_stat&exp = (/"exp1", "exp2", "ref"/)
  model_arr_stat&time = time
  model_arr_stat&stat = (/"mean", "min", "max"/)


  do iexp = 0, 1

    if (iexp .eq. 0) then
      input_files = systemfunc("ls /work/bd0854/b309059/esmvaltool_output/" + \
                    "Chapter_3/FGD/fig_3_41/OHC/CMIP6_*_historical_*_detrended_total.nc")
    else
      input_files := systemfunc("ls /work/bd0854/b309059/esmvaltool_output/" + \
                    "Chapter_3/FGD/fig_3_41/OHC/CMIP6_*_hist-nat_*_detrended_total.nc")
    end if

    nfiles = dimsizes(input_files)

    dim_data   = (/nfiles, ntime/)
    data  := new(dim_data, double)
    data!0 = "model"
    data!1 = "time"
    data&model = new(nfiles, string, "model")
    data&time = time

    ; Loop over files in the list, read and append data
    do ii = 0, nfiles - 1

      print("Read in " + input_files(ii))

      f = addfile(input_files(ii), "r")

      data_temp = f->ohc

      ; calculate anom
      anom = data_temp(121)
print("anom = " + anom)
      data_temp(:) = data_temp(:) - anom

      data(ii, :) = 10.e-22 * (/data_temp(:)/)

      data&model(ii) = f@source_id

      delete(data_temp)

    end do

print("min = " + min(data(:, :)))
print("max = " + max(data(:, :)))

    ; *************************************
    ; calculate means over ensemble members
    ; *************************************

    datasetnames := data&model
    ndim := dimsizes(datasetnames)
    ensemble_assign := new(ndim, integer)
    ensemble_size := new(ndim, integer)

    enum = 1
    temp = datasetnames(0)
    do gg = 0, ndim - 1
      if (temp.eq.datasetnames(gg)) then ; does the model name match what's in temp?
        ensemble_assign(gg) = enum ; if so, assign it the same number
      else
        enum = enum + 1              ; if not, assign it the next number
        ensemble_assign(gg) = enum
        temp = datasetnames(gg)
      end if
    end do   
    ensemble_assign@model = datasetnames
    delete(temp)
    ;ensemble_assign@models = str_join(datasetnames+"/"+ensembles,",")

    data_em := data(:max(ensemble_assign)-1,:)
    data_em = data_em@_FillValue
    datasetnames_em := datasetnames(:max(ensemble_assign)-1)

    do gg = 1,max(ensemble_assign)         ; calculate ensemble means
      wind := ind(ensemble_assign.eq.gg)
      ensemble_size(wind) = dimsizes(wind)
      if (dimsizes(wind).eq.1) then
        data_em(gg-1,:) = (/ data(wind,:) /)
        datasetnames_em(gg-1) = datasetnames(wind)
      else
        data_em(gg-1,:) = (/ dim_avg_n(data(wind,:),0) /)
        datasetnames_em(gg-1) = datasetnames(wind(0))
      end if
    end do

    ; *************************************
    ; calculate multi-model mean
    ; *************************************

    model_arr_stat(iexp, 0, :) = dim_avg_n(data_em(:,:),0) 

print("min = " + min(model_arr_stat(iexp, 0, :)))
print("max = " + max(model_arr_stat(iexp, 0, :)))
    ; *************************************
    ; calculate 5% and 95% quantiles
    ; *************************************

    q_05 = new(ntime, double)
    q_95 = new(ntime, double)

    do itime = 0, ntime -1
      model_num = num(.not.ismissing(data_em(:, itime)))
      ind_m := ind(.not.ismissing(data(:,itime)))
      ind_sort := dim_pqsort(data(ind_m, itime), 1)
      count = 0.
      do imodel = 0, dimsizes(ind_m) - 1
        count = count + 1./(model_num*ensemble_size(ind_m(ind_sort(imodel))))
        if(count .gt. 0.95) then
          break  
        end if
        if(ismissing(q_05(itime)) .and. count .gt. 0.05) then
          q_05(itime) =  data(ind_m(ind_sort(imodel)), itime)
        end if
        q_95(itime) =  data(ind_m(ind_sort(imodel)), itime)
      end do
    end do

    model_arr_stat(iexp, 1, :) = (/q_05/)
    model_arr_stat(iexp, 2, :) = (/q_95/)

  end do

  ; read obs data

  f_obs = "/work/bd0854/b309059/esmvaltool_output/Chapter_3/FGD/fig_3_41/" + \
          "OHC/AR6_FGD_assessment_timeseries_OHC.csv"

  field_obs = asciiread(f_obs,-1,"string")

  delim = ","

  start_obs = 1971
  end_obs = 2014
  ntime_obs = end_obs - start_obs + 1 
  idx1 = ind(start_obs .eq. model_arr_stat&time)
  idx2 = ind(end_obs .eq. model_arr_stat&time)

  tmp = todouble(str_get_field(field_obs, 8, delim))
  model_arr_stat(2, 0, idx1:idx2) = (/tmp(2: ntime_obs + 1)/)


  ; write netcdf

  ;setfileoption("nc", "Format",  "NetCDF4")
  fn = "/work/bd0854/b309059/esmvaltool_output/Chapter_3/FGD/fig_3_41/ohc_damip.nc"
  system("/bin/rm -f " + fn) ; remove if exists
  ncdf = addfile(fn, "c")

  ;===================================================================
  ;Define the file dimensions, NOTE that both dimensions are unlimited.
  dimNames = getvardims(model_arr_stat)

  do i = 0, dimsizes(dimNames) - 1
    if (typeof(model_arr_stat&$dimNames(i)$).eq."string") then
      ; Convert string array to two dimensional character array
      char_array = stringtochar(model_arr_stat&$dimNames(i)$)
      ncdf->$dimNames(i)$ = (/char_array/)
      delete(char_array)
    else
      ncdf->$dimNames(i)$ = (/model_arr_stat&$dimNames(i)$/)
    end if
  end do
  ncdf->ohc = (/model_arr_stat/)

end
