;=================================================================================================
; create blank array for use when something may be/is wrong.
;
undef("create_empty_array")
function create_empty_array(yS:numeric,yE:numeric,mS:numeric,mE:numeric,opttype:string)
local yS,yE,mS,mE,opttype
begin
  if (ismissing(yS).or.ismissing(yE)) then
     yS = 1
     yE = 50
  end if
  timeT =  yyyymm_time(yS, yE, "integer")
  time = timeT({yS*100+mS:yE*100+mE})
  if (opttype.eq."time_lat_lon") then
     blankarr = new((/dimsizes(time),90,180/),"float",1.e20)
     blankarr!0 = "time"    ; time coordinate variable assigned below
     blankarr&time = time
     blankarr!1 = "lat"
     blankarr&lat = fspan(-89,89,90)
     blankarr!2 = "lon"
     blankarr&lon = fspan(0,358,180)
     blankarr&lat@units = "degrees_north"
     blankarr&lon@units = "degrees_east"
  end if
  if (opttype.eq."time_lev_lat") then
     blankarr = new((/dimsizes(time),41,90/),"float",1.e20)
     blankarr!0 = "time"    ; time coordinate variable assigned below
     blankarr&time = time
     blankarr!1 = "lev"
     blankarr&lev =fspan(0,5000,41)
     blankarr!2 = "lat"
     blankarr&lat = fspan(-89,89,90)
     blankarr&lat@units = "degrees_north"
     blankarr&lev@units = "m"
     blankarr&lev@positive = "down"
  end if
  blankarr@units = ""
  blankarr@is_all_missing = True
  return(blankarr)
  delete([/yS,yE,mS,mE,opttype,blankarr,timeT,time/])
end
;===================================================================================================
; read in atmospheric/land data from selected files
; assign time coordinate variables, check for issues with the array, assign _FillValue (if needed)
; assign dimension names (for ease-of-use), check and modify units
;
; vname settings at top of this script can be modified if a different variable name is
; encountered. For instance, if a TS data file has the TS array named as "sfc_t", one
; could add "sfc_t" to the vname TS coding as follows:
;   if (vn.eq."TS") then
;      vname = (/"TS","ts","sst","sfc_t"/)
;   end if
;
undef("data_read_in")
function data_read_in(zpath:string,vn:string,yearS:integer,yearE:integer)
; path for TS file(s), variable name, start year, and end year are read in.
local zpath,vn,cpath0,ta,tfiles,c,arr,farr,yearS,yearE,mocheck,fils_precc,fils_precl
begin
  if (vn.eq."TS") then
     vname = (/"TS","ts","sst","t_surf","skt"/)
  end if
  if (vn.eq."PSL") then
     vname = (/"PSL","psl","slp","SLP","prmsl","msl","slp_dyn"/)
  end if
  if (vn.eq."TREFHT") then
     vname = (/"TREFHT","tas","temp","air","temperature_anomaly","temperature","t2m","t_ref","T2","tempanomaly"/)
  end if
  if (vn.eq."PRECT") then
     vname = (/"PRECC","PRECL","PRECT","pr","PPT","ppt","p","P","precip","PRECIP","tp","prcp","prate"/)
  end if
  if (vn.eq."SNOWDP") then
     vname = (/"SNOWDP","snd"/)
  end if

  if (ismissing(zpath) ) then
     print("File missing, creating blank array of data. View "+vn+" namelist for details.")
     arr = create_empty_array(yearS,yearE,1,12,"time_lat_lon")
     sydata = yearS     ; assign these variables based on yearS/yearE provided in namelist. Doesn't matter
     eydata = yearE     ; as data array is totally missing..
     smdata = 1
     emdata = 12
  else
     cpath0 = stringtochar(zpath)
     ta = stringtochar("*")
     if (any(cpath0.eq.ta(0)).or.any(cpath0.eq."{")) then   ; check for "*" and "{" denoting multiple files
        tfiles = systemfunc("ls "+zpath+" 2> /dev/null")   ; /dev/null suppresses all standard error output
        if (vn.eq."PRECT") then   ; special section for precip, as might need to do PRECC+PRECL
           b = addfile(tfiles(0),"r")   ; check for PRECC
           if (isfilevar(b,"PRECC").or.isfilevar(b,"PRECL")) then  ; PRECC/PRECL section
              fils_precc = str_match(tfiles,"PRECC")
              fils_precl = str_match(tfiles,"PRECL")
              if (any(ismissing(fils_precc)).or.any(ismissing(fils_precl))) then
                 print("Fatal: Need both PRECC and PRECL file(s), creating blank array")
                 print(fils_precc)
                 print(fils_precl)
                 arr = create_empty_array(yearS,yearE,1,12,"time_lat_lon")
                 sydata = yearS     ; assign these variables based on yearS/yearE provided in namelist. Doesn't matter
                 eydata = yearE     ; as data array is totally missing..
                 smdata = 1
                 emdata = 12
                 break
              end if
              c = addfiles(fils_precc,"r")
              arr = c[:]->PRECC
              c2  = addfiles(fils_precl,"r")
              arr = (/ arr+c2[:]->PRECL  /)
              arr@long_name = "Large-scale (stable) + convective precipitation rate (liq + ice)"
              delete([/c2,fils_precc,fils_precl/])
           else    ; pr, ppt, PPT, PRECT multiple/single file read-in here..
              c = addfiles(tfiles,"r")
              do ii=0,dimsizes(vname)-1
                 if (isfilevar(c[0],vname(ii))) then
                    arr = c[:]->$vname(ii)$
                    break
                 end if
              end do
           end if
           delete(b)
        else
           c = addfiles(tfiles,"r")
           do ii=0,dimsizes(vname)-1
              if (isfilevar(c[0],vname(ii))) then
                 arr = c[:]->$vname(ii)$
                 break
              end if
           end do
        end if

        nfil = dimsizes(tfiles)
        cpathS = stringtochar(tfiles(0))      ; this section will work for PRECC/PRECL, as it will read the first
        cpathE = stringtochar(tfiles(nfil-1)) ; PRECC file and the last PRECL file.
        ncharS = dimsizes(cpathS)
        ncharE = dimsizes(cpathE)
        sydata = stringtointeger(charactertostring(cpathS(ncharS-17:ncharS-14)))
        smdata = stringtointeger(charactertostring(cpathS(ncharS-13:ncharS-12)))
        eydata = stringtointeger(charactertostring(cpathE(ncharE-10:ncharE-7)))
        emdata = stringtointeger(charactertostring(cpathE(ncharE-6:ncharE-5)))
        delete([/cpathS,cpathE,ncharS,ncharE,nfil/])
;        delete(c)
     else
        c = addfile(zpath,"r")
        do i=0,dimsizes(vname)-1
           if (isfilevar(c,vname(i))) then
              arr = c->$vname(i)$
              break
           end if
        end do
        cpath = stringtochar(zpath)
        nchar = dimsizes(cpath)
        sydata = stringtointeger(charactertostring(cpath(nchar-17:nchar-14)))
        smdata = stringtointeger(charactertostring(cpath(nchar-13:nchar-12)))
        eydata = stringtointeger(charactertostring(cpath(nchar-10:nchar-7)))
        emdata = stringtointeger(charactertostring(cpath(nchar-6:nchar-5)))
        delete([/cpath,nchar/])
;        delete(c)
     end if
     delete([/ta,cpath0/])
  end if

  if (isvar("arr").eq.False) then
     print("Variable ("+vn+") not found. Examine input file "+zpath+". Creating empty array and continuing")
     arr = create_empty_array(yearS,yearE,1,12,"time_lat_lon")
  end if

  if (isshort(arr)) then
     arrT = short2flt(arr)
     delete(arr)
     arr = arrT
     delete(arrT)
  end if

  if (.not.isatt(arr,"_FillValue")) then     ; assign _FillValue if one is not present
     if (isatt(arr,"missing_value")) then
        arr@_FillValue = arr@missing_value
     else
        arr@_FillValue = default_fillvalue(typeof(arr))
     end if
  end if

  dimz = dimsizes(arr)
  if (any(dimz.eq.1)) then
     arrT = rm_single_dims(arr)
     delete(arr)
     arr = arrT
     delete(arrT)
  end if
  if (dimsizes(dimz).le.2) then
     print("Possible curvilinear (or unstructured) grid detected. The CVDP cannot analyze curvilinear data. Please regrid to a rectilinear grid for inclusion in CVDP comparisons.")
     print("Input file: "+zpath)
     print("Setting array to all missing")
     delete(arr)
     arr = create_empty_array(yearS,yearE,smdata,emdata,"time_lat_lon")
     sydata = yearS     ; assign these variables based on yearS/yearE provided in namelist. Doesn't matter
     eydata = yearE     ; as data array is totally missing..
     smdata = 1
     emdata = 12
  end if
  delete(dimz)

  arr!0 = "time"
  arr!1 = "lat"
  arr!2 = "lon"

  if (isatt(arr,"valid_range")) then    ; check to make sure data is in valid range. Reset to stay within the valid range if needed.
;     print("Data outside valid_range in "+zpath+", resetting data to be within valid_range")
     arr = where(arr.lt.arr@valid_range(0),arr@valid_range(0),arr)
     arr = where(arr.gt.arr@valid_range(1),arr@valid_range(1),arr)
  end if

  if (any(abs(arr).ge.1.e20)) then   ; check for inf values or values way out of range, reset to _FillValue.
     print("Values greater than 1.e20 or less than -1.e20 detected in "+zpath+", resetting to _FillValue")
     arr = where(abs(arr).ge.1.e20,arr@_FillValue,arr)
  end if

;  if (.not.iscoord(arr,"lat")) then
;     print("The data might have 2D latitudes. The functions.ncl coding is not set up to handle this, exiting from data_read_in.ncl")
;     print("zpath")
;     exit
;  end if
;  if (.not.iscoord(arr,"lon")) then
;     print("The data might have 2D longitudes. The functions.ncl coding is not set up to handle this, exiting from data_read_in.ncl")
;     print("zpath")
;  exit
;  end if

  if (yearS.lt.sydata.or.yearE.gt.eydata) then
     print("Requested "+yearS+"-"+yearE+" time span is outside the input file "+zpath+" time span of "+sydata+"-"+eydata+"")
     print("Setting array to all missing")
     delete(arr)
     arr = create_empty_array(yearS,yearE,smdata,emdata,"time_lat_lon")
     sydata = yearS     ; assign these variables based on yearS/yearE provided in namelist. Doesn't matter
     eydata = yearE     ; as data array is totally missing..
     smdata = 1
     emdata = 12
  else
     timeT =  yyyymm_time(sydata, eydata, "integer")
     time = timeT({sydata*100+smdata:eydata*100+emdata})
     if (iscoord(arr,"time")) then
        delete(arr&time)
     end if
     dimz = dimsizes(arr)
     if (dimz(0).eq.dimsizes(time)) then
        arr&time = time
     else
        print("Possible mismatch detected between time specified in file name and file variables, setting array to missing")
        print("File = "+zpath)
        print("Read from file name: "+min(time)+"-"+max(time))
        delete(arr)
        arr = create_empty_array(yearS,yearE,smdata,emdata,"time_lat_lon")
        sydata = yearS     ; assign these variables based on yearS/yearE provided in namelist. Doesn't matter
        eydata = yearE     ; as data array is totally missing..
        smdata = 1
        emdata = 12
     end if
     delete(dimz)
     delete([/time,timeT/])
  end if
  delete([/sydata,smdata,eydata,emdata/])

;  printVarSummary(arr)
;  printVarSummary(arr({sy*100+1:ey*100+12},:,:))
  if (arr&lat(0).ge.0) then
     farr = arr({yearS*100+1:yearE*100+12},::-1,:)   ; flip the latitudes
  else
     farr = arr({yearS*100+1:yearE*100+12},:,:)
  end if
;  printVarSummary(farr)
  delete(arr)

  mocheck = (/(yearS*100+1)-min(farr&time),(yearE*100+12) - max(farr&time)/)
  if (any(mocheck.ne.0)) then    ; previously: if (mod(dimsizes(farr&time),12).ne.0) then
     if (mocheck(0).ne.0) then
        print("First requested year is incomplete")
     end if
     if (mocheck(1).ne.0) then
        print("Last requested year is incomplete")
     end if
     print("Incomplete data year(s) requested for file "+zpath+", printing out time and creating blank array")
     print("Time requested: "+yearS+"-"+yearE)
     print(farr&time)
     delete(farr)
     farr = create_empty_array(yearS,yearE,1,12,"time_lat_lon")
  end if
  delete(mocheck)

  if (farr&lon(0).lt.0) then
     farr = lonFlip(farr)     ; lon flip
  end if
  if (min(farr&lon).lt.0.or.max(farr&lon).gt.360) then
     print(farr&lon)
     print("path = "+zpath)
     print("Fatal: Longitudes not in expected 0-360E range, creating blank array")
     delete(farr)
     farr = create_empty_array(yearS,yearE,1,12,"time_lat_lon")
  end if

  if (vn.eq."TREFHT".or.vn.eq."TS") then      ; units check
     if (farr@units.eq."K".or.farr@units.eq."Kelvin".or.farr@units.eq."deg_k".or.farr@units.eq."deg_K") then
        if (max(farr).ge.100) then    ; data sets can be anomalies with units of K, so check for range before subtracting
           farr = farr-273.15
        end if
        farr@units = "C"
     end if
     if (farr@units.eq."degrees_C".or.farr@units.eq."degrees C".or.farr@units.eq."degree_C".or.farr@units.eq."degree C") then
        farr@units = "C"
     end if
  end if
  if (vn.eq."PSL") then
     if (farr@units.eq."Pa".or.farr@units.eq."Pascals".or.farr@units.eq."Pascal") then
        farr = farr/100.
        farr@units = "hPa"
     end if
  end if
  if (vn.eq."PRECT") then    ; convert (if necessary) to mm/day
     if (farr@units.eq."m/s".or.farr@units.eq."m s-1") then
        farr = farr*86400000.
     end if
     if (farr@units.eq."kg m-2 s-1".or.farr@units.eq."kg/m2/s".or.farr@units.eq."kg/m^2/s".or.farr@units.eq."kg/(s*m2)".or.farr@units.eq."mm/s") then
        farr = farr*86400.
     end if
     if (farr@units.eq."m".or.farr@units.eq."m/month".or.farr@units.eq."cm".or.farr@units.eq."cm/month".or.farr@units.eq."mm".or.farr@units.eq."mm/month") then
        yr = toint(farr&time)/100
        mo = toint(farr&time - (yr*100))
        days = days_in_month(yr,mo)
        do gg = 0,dimsizes(farr&time)-1
		         farr(gg,:,:) = (/ farr(gg,:,:) / days(gg) /)
	       end do
        if (farr@units.eq."cm".or.farr@units.eq."cm/month") then
           farr = farr*10.   ; convert from cm/day to mm/day
        end if
        if (farr@units.eq."m".or.farr@units.eq."m/month") then
           farr = farr*1000.   ; convert from m/day to mm/day
        end if
     end if
     if (farr@units.eq."m/day".or.farr@units.eq."m day-1") then
        farr = farr*1000.
     end if
     farr@units = "mm/day"
  end if
  if (vn.eq."SNOWDP") then
     if (.not.isatt(farr,"is_all_missing")) then
        if (farr@units.ne."m".and.farr@units.ne."meters") then
           print("Warning: SNOWDP/snd units may not be in meters. listed units = "+farr@units)
        end if
     end if
  end if

  date = farr&time         ; switch time to be CF-conforming
  delete(farr&time)
  yyyy = date/100
  mm = date-(yyyy*100)
  days = (days_in_month(yyyy,mm))/2
  hms  = days
  hms = 0   ; hours, minutes, seconds all the same (=0)
  time = cd_inv_calendar(yyyy,mm,days,hms,hms,hms,"months since "+min(yyyy)+"-01-15 00:00:00",0)
  time@long_name = "Time"
  time@standard_name = "time"
  time@actual_range = (/min(time),max(time)/)
  time!0 = "time"
  time&time = time
  farr&time = time
  delete([/time,yyyy,mm,days,hms,date/])
  return(farr)
end
;=================================================================================================
undef ("eofunc_north2")
function eofunc_north2(eval[*]:numeric, N[1]:integer, prinfo[1]:logical)
;
; North, G.R. et al (1982): Sampling Errors in the Estimation of Empirical Orthogonal Functions.
; Mon. Wea. Rev., 110, 699–706.
; doi: http://dx.doi.org/10.1175/1520-0493(1982)110<0699:SEITEO>2.0.CO;2
;
; Usage after 'eofunc'. Here ntim was used,
;             prinfo = True
;             sig    = eval_north(eof@eval, ntim, prinfo)
;
; Copied directly from v6.3.0 contributed.ncl for use in the package regardless of NCL version.
;
local neval, dlam, low, high, sig, n
begin
  neval   = dimsizes(eval)
  if (neval.eq.1)
      print("eofunc_north: neval=1, no testing can be performed")
      sig = True
      sig@long_name = "EOF separation is not testable N=1"
      sig@N         =  N
      return(sig)
  end if

  dlam    = eval * sqrt(2.0/N)   ; eq 24
  low     = eval-dlam
  high    = eval+dlam

  sig     = new(dimsizes(eval), logical)
  sig = False  ; default is not significantly separated

; first and last eigenvalues are special cales

  if (eval(0).gt.high(1)) then
      sig(0) = True
  end if
  if (eval(neval-1).lt.low(neval-2)) then
      sig(neval-1) = True
  end if

; loop over other eignvalues

  if (N.gt.2) then
      do n=1,neval-2
         if (eval(n).lt.low(n-1) .and. eval(n).gt.high(n+1)) then
             sig(n) = True
         end if
      end do
  end if

  if (prinfo) then
      print(dlam+"   "+low+"   "+eval+"   "+high+"  "+sig)
  end if

  sig@long_name = "EOF separation"
  sig@N         =  N
  return(sig)
end
;=================================================================================================
undef("piControl_list_setup")
function piControl_list_setup(ipath:string,yrlength[1]:integer,N[1]:integer,optA[1]:integer)
; Given a directory, the length of desired segment (yrlength), and the number of segments per control (N),
;  the option as to whether the CMIP5 (optA=0) or CMIP6 (optA=1) repositories
; return a list of paths and start years for each segment (returned as an attribute)
;
; Note: This function will only identify data from the first control member from each center.

local ipath,yrlength,N,models,modelsT,findir,finyr,cntr,tfiles,cpath,nchar,sydata,smdata,eydata,emdata,cpathS,cpathE,ncharS,ncharE,npossible,findirF,optA,ttx
begin
  models = systemfunc("ls "+ipath)     ; ipath directory structure looks like this: /project/data/cmip6/piControl/Amon/ts/
  findir = new(dimsizes(models)*N,string)                                ; or this: /project/data/cmip5/ETH/cmip5/piControl/Amon/ts/
  finyr = new(dimsizes(models)*N,integer)
  cntr = 0
  if (optA.eq.0) then
     ttx = "/"   ; for CMIP5 repository with example structure /project/data/cmip5/ETH/cmip5/piControl/Amon/ts/ACCESS1-3/r1i1p1/
  else
     ttx = "/*/" ; for CMIP6 repository with example structure /project/data/cmip6/piControl/Amon/ts/BCC-CSM2-MR/r1i1p1f1/gn/
  end if

  do gg = 0,dimsizes(models)-1
     modelsT := systemfunc("ls "+ipath+models(gg))
     if (.not.ismissing(modelsT(0))) then
        tfiles := systemfunc("ls "+ipath+models(gg)+"/"+modelsT(0)+ttx)
        if (.not.ismissing(tfiles(0))) then
           if (dimsizes(tfiles).eq.1) then
              cpath := stringtochar(tfiles)
              nchar := dimsizes(cpath)
              sydata = stringtointeger(charactertostring(cpath(nchar-17:nchar-14)))
              smdata = stringtointeger(charactertostring(cpath(nchar-13:nchar-12)))
              eydata = stringtointeger(charactertostring(cpath(nchar-10:nchar-7)))
              emdata = stringtointeger(charactertostring(cpath(nchar-6:nchar-5)))
           else
              cpathS := stringtochar(tfiles(0))
              cpathE := stringtochar(tfiles(dimsizes(tfiles)-1))
              ncharS := dimsizes(cpathS)
              ncharE := dimsizes(cpathE)
              sydata = stringtointeger(charactertostring(cpathS(ncharS-17:ncharS-14)))
              smdata = stringtointeger(charactertostring(cpathS(ncharS-13:ncharS-12)))
              eydata = stringtointeger(charactertostring(cpathE(ncharE-10:ncharE-7)))
              emdata = stringtointeger(charactertostring(cpathE(ncharE-6:ncharE-5)))
           end if
           npossible = (eydata-sydata)/yrlength
           if (npossible.ge.1) then
              if (npossible.eq.1) then
                 findir(cntr) = ipath+models(gg)+"/"+modelsT(0)+ttx
                 finyr(cntr) = sydata
                 cntr = cntr+1
              else
                 do hh = 0,min((/N,npossible/))-1
                    findir(cntr) = ipath+models(gg)+"/"+modelsT(0)+ttx
                    finyr(cntr) = sydata+(hh*yrlength)
                    cntr = cntr+1
                 end do
              end if
           end if
        end if
     end if
  end do
  findirF = findir(:cntr-1)
  findirF@start_year = finyr(:cntr-1)
  findirF@end_year = finyr(:cntr-1)+yrlength-1
  return(findirF)
  delete([/ipath,yrlength,N,findir,findirF,finyr/])
end
;=================================================================================================
undef("boxplot_stat")
function boxplot_stat(bparrT:numeric)
local bparrT,bparr,bpcalcarr,x05,x25,x75,x95,dimBP
begin
  bparr = bparrT(ind(.not.ismissing(bparrT)))
  qsort(bparr)
  dimBP = dimsizes(bparr)
  x05  = round(.05*dimBP,3)-1     ; -1 to account for NCL indexing starting
  x25  = round(.25*dimBP,3)-1     ; -1 to account for NCL indexing starting
  x75  = round(.75*dimBP,3)-1
  x95  = round(.95*dimBP,3)-1

  bpcalcarr = new(5,float)
  bpcalcarr(0) = bparr(x05)   ;min(bparr)
  bpcalcarr(1) = bparr(x25)
  bpcalcarr(2) = dim_median(bparr)
  bpcalcarr(3) = bparr(x75)
  bpcalcarr(4) = bparr(x95)    ;max(bparr)
  return(bpcalcarr)
end

;=================================================================================================
; Adam Phillips
;
undef("boxplot2")
function boxplot2 (wks:graphic,x[*]:numeric,y[*][*]:numeric,boxOpts:logical,plotres:logical,lineres:logical)
;
;    This function creates a blank linLog plot object, on which box plots are created by extensive
;    use of gsn_add_polyline. The user can draw as many or as few box plots as they wish.
;    The plot is not drawn, and the frame is not advanced. May be used with gsn_panel. Missing data
; may be present in the input Y array, but not the input X array.
;    The function options are as follows:
;
;    wks     ----- The workstation
;    x[*]    ----- A one-dimensional array containing the X-axis values of where the box plot(s)
;                  will be drawn.
;    y[*][*] ----- A two-dimensional array, where the rightmost dimension contains the box plot
;               reference pt. data. y(n,0)=bottom_value, y(n,1)=bottom_value_of_box,
;               y(n,2)=mid-value_of_box,y(n,3)=top_value_of_box,y(n,4)=top_value
;    boxOpts ----- 4 options attached as attributes may be attached here.
;               boxWidth ---- Scalar or array that contains the widths of the boxes.
;               boxColors ---- Scalar or array that contains the colors that the boxes will be drawn in.
;                           Ex. boxOpts@boxColors =  (/"green","blue"/)
;                        If the number of colors specified does not equal the number of
;                        boxes drawn, only the first color specified will be used. Also used to set the
;                        boxFill color if boxFill = True.
;                boxFill_Colors ---- Scalar or array that contains the colors that the boxes will be filled in.
;                                    If not set the boxes will not be filled.
;                boxFill_midColor ---- Scalar or array that contains the colors that the center box
;                                      polyline will be drawn in. Only activated when boxFill = True.
;    plotres ----- An optional xy-plot resource list. Will override defaults where applicable.
;    lineres ----- An optional resource list pertaining to the lines used to draw the boxes/lines.
;
;    Example call: plot3 = boxplot(wks,ntime,newyval,opti,res,False)
;               draw(plot3)
;               frame(wks)
;
begin
     dimquery = dimsizes(y)
     numbox = dimquery(0)
     boxWidths = new((/numbox/),float)
     if (numbox.ne.dimsizes(x)) then
        print("boxplot: Fatal: X must be one-dimensional and both X and Y must have the same leftmost dimension")
        exit
     end if
     if (any(ismissing(x))) then
        print("boxplot: Fatal: X array cannot contain missing data, exiting")
        exit
     end if

;    Developing x-axis
     xAxis = new(numbox+2,typeof(x))
     xAxis(1:numbox) = x
     if (numbox.ne.1) then
        dx = x(1)-x(0)
        xAxis(0) = x(0)-dx
        xAxis(numbox+1) = x(numbox-1)+dx
     else
        dx = 1
        xAxis(0) = x-dx
        xAxis(2) = x+dx
     end if

     if (boxOpts) then
        if (isatt(boxOpts,"boxWidth")) then
           if (dimsizes(boxOpts@boxWidth).ne.1.and.dimsizes(boxOpts@boxWidth).ne.numbox) then
              print("boxplot: Number of input box widths must either equal 1 or the number of boxes ("+numbox+"). Using first specified box width only.")
              boxWidths(:) = boxOpts@boxWidth(0)
           else
              boxWidths = boxOpts@boxWidth
           end if
        else
           boxWidths(:) = dx*.3
        end if
     else
        boxWidths(:) = dx*.3
     end if


     labarr = new(numbox+2,"string")         ;Prepare actual X-axis labels...
     labarr(0) = ""
     labarr(numbox+1) = ""
     labarr(1:numbox) = xAxis(1:numbox)

; Whether to maximize plot in frame.
     maximize = get_res_value(plotres,"gsnMaximize",False)

     if (plotres) then
;       print("Plot resources detected, accepting")
        fsatts = getvaratts(plotres)
        do ty = 0,dimsizes(fsatts)-1
           if (fsatts(ty).eq."tmXBLabels") then        ;Special section to test for XBLabels
              if (dimsizes(plotres@tmXBLabels).ne.numbox) then
                     print("boxplot: Fatal:Number of XB Labels does not match number of boxes, exiting")
                     exit
                  else
                     labarr(1:numbox) = plotres@$fsatts(ty)$
                  end if
                  xblab = plotres@tmXBLabels
                  delete(plotres@tmXBLabels)      ;Delete so tmXBLabels is not used when all
           end if                       ;atts are assigned below...
        end do
        delete(fsatts)
     end if

     plot = create "plot" logLinPlotClass wks
        "trYMinF" : min(y)-2
        "trYMaxF" : max(y)+2
        "trXMinF" : min(xAxis)
        "trXMaxF" : max(xAxis)
        "pmTickMarkDisplayMode" : "Always"
        "tmXBMode"   : "Explicit"
        "tmXBValues" : xAxis
        "tmXBLabels" : labarr
        "tmYROn"  : False
        "tmXTOn"  : False
        "tmYRBorderOn" : False
        "tmXTBorderOn" : False
        "pmTitleDisplayMode": "Always"                   ; allow titles
        "tiMainOn"     : True
        "tiMainString"   : ""
     end create
     if (plotres) then
        attsetvalues(plot,plotres)
     end if

     polyres = True                          ;Set up defaults
     polyres@gsLineColor       = "black"                     ; color of lines
     polyres@gsLineThicknessF  = 1.5                          ; thickness of lines
     polyres@gsLineDashPattern = 0
     if (lineres) then
        fsatts = getvaratts(lineres)
        do ty = 0,dimsizes(fsatts)-1
           polyres@$fsatts(ty)$ = lineres@$fsatts(ty)$
        end do
     end if

     polygres = True     ; box fill resources
     polygres@gsLineColor = "white"
     polygres@gsFillColor = "white"
     polygres@gsLineThicknessF  = 0.5

     cflag  = 1
     cfflag = 1
     cmflag = 1
     if (boxOpts) then
        if (isatt(boxOpts,"boxColors")) then
           dum3 = new((/numbox/),graphic)
           boxcolor = boxOpts@boxColors
           if (dimsizes(boxcolor).eq.1.or.dimsizes(boxcolor).ne.numbox) then
              if (dimsizes(boxcolor).ne.numbox) then
                 print("boxplot: warning: Number of input colors must either equal 1 or the number of boxes ("+numbox+"). Using first specified color only.")
              end if
              polyres@gsLineColor = boxcolor(0)
           else
              cflag = 2
           end if
        end if
        if (isatt(boxOpts,"boxFill_Colors")) then
           boxfillcolor = boxOpts@boxFill_Colors
           if (dimsizes(boxfillcolor).eq.1.or.dimsizes(boxfillcolor).ne.numbox) then
              if (dimsizes(boxfillcolor).ne.numbox) then
                 print("boxplot: warning: Number of input box fill colors must either equal 1 or the number of boxes ("+numbox+"). Using first specified color only.")
              end if
              polygres@gsLineColor = boxfillcolor(0)
              polygres@gsFillColor = boxfillcolor(0)
           else
              cfflag = 2
           end if
           if (isatt(boxOpts,"boxFill_midColor")) then
              midlinecolor = boxOpts@boxFill_midColor
              if (dimsizes(midlinecolor).eq.1.or.dimsizes(midlinecolor).ne.numbox) then
                 if (dimsizes(midlinecolor).ne.numbox) then
                    print("boxplot: warning: Number of input box fill colors must either equal 1 or the number of boxes ("+numbox+"). Using first specified color only.")
                 end if
              else
                 cmflag = 2
              end if
           end if
        end if
     end if

     dum = new((/numbox,9/),graphic)
     do gg = 0,numbox-1
        ff = xAxis(gg+1)
        if (cflag.eq.2) then
            polyres@gsLineColor = boxcolor(gg)
        end if
        yy = (/y(gg,4),y(gg,4)/)
        xx = (/(ff-(boxWidths(gg)/5.)),(ff+(boxWidths(gg)/5.))/)
        if (.not.(any(ismissing(xx).or.ismissing(yy)))) then
           dum(gg,0) = gsn_add_polyline(wks,plot,xx,yy,polyres)
        end if

        yy = (/y(gg,3),y(gg,4)/)
        xx = (/ff,ff/)
        if (.not.(any(ismissing(xx).or.ismissing(yy)))) then
           dum(gg,1) = gsn_add_polyline(wks,plot,xx,yy,polyres)
        end if

        if (isatt(boxOpts,"boxFill_Colors")) then
           xx5 = (/(ff-(boxWidths(gg)/2.)),(ff+(boxWidths(gg)/2.)),(ff+(boxWidths(gg)/2.)),(ff-(boxWidths(gg)/2.)),(ff-(boxWidths(gg)/2.))/)
           yy5 = (/y(gg,1),y(gg,1),y(gg,3),y(gg,3),y(gg,1)/)
           if (cfflag.eq.2) then
              polygres@gsLineColor = boxfillcolor(gg)
              polygres@gsFillColor = boxfillcolor(gg)
           end if
           dum3(gg) = gsn_add_polygon(wks,plot,xx5,yy5,polygres)
        end if
        yy = (/y(gg,3),y(gg,3)/)
        xx = (/(ff-(boxWidths(gg)/2.)),(ff+(boxWidths(gg)/2.))/)
        if (.not.(any(ismissing(xx).or.ismissing(yy)))) then
           dum(gg,2) = gsn_add_polyline(wks,plot,xx,yy,polyres)
        end if

        yy = (/y(gg,1),y(gg,3)/)
        xx = (/(ff-(boxWidths(gg)/2.)),(ff-(boxWidths(gg)/2.))/)
        if (.not.(any(ismissing(xx).or.ismissing(yy)))) then
           dum(gg,3) = gsn_add_polyline(wks,plot,xx,yy,polyres)
        end if

        yy = (/y(gg,1),y(gg,3)/)
        xx = (/(ff+(boxWidths(gg)/2.)),(ff+(boxWidths(gg)/2.))/)
        if (.not.(any(ismissing(xx).or.ismissing(yy)))) then
           dum(gg,5) = gsn_add_polyline(wks,plot,xx,yy,polyres)
        end if

        yy = (/y(gg,1),y(gg,1)/)
        xx = (/(ff-(boxWidths(gg)/2.)),(ff+(boxWidths(gg)/2.))/)
        if (.not.(any(ismissing(xx).or.ismissing(yy)))) then
           dum(gg,6) = gsn_add_polyline(wks,plot,xx,yy,polyres)
        end if

        yy = (/y(gg,0),y(gg,1)/)
        xx = (/ff,ff/)
        if (.not.(any(ismissing(xx).or.ismissing(yy)))) then
           dum(gg,7) = gsn_add_polyline(wks,plot,xx,yy,polyres)
        end if

        yy = (/y(gg,0),y(gg,0)/)
        xx = (/(ff-(boxWidths(gg)/5.)),(ff+(boxWidths(gg)/5.))/)
        if (.not.(any(ismissing(xx).or.ismissing(yy)))) then
           dum(gg,8) = gsn_add_polyline(wks,plot,xx,yy,polyres)
        end if

        if (isatt(boxOpts,"boxFill_Colors").and.isatt(boxOpts,"boxFill_midColor")) then
           savecolor = polyres@gsLineColor     ; save color to reset after drawing mid-value to reset the color
           if (cmflag.eq.1) then
              polyres@gsLineColor = midlinecolor(0)
           else
              polyres@gsLineColor = midlinecolor(gg)
           end if
        end if
        yy = (/y(gg,2),y(gg,2)/)
        xx = (/(ff-(boxWidths(gg)/2.)),(ff+(boxWidths(gg)/2.))/)
        if (.not.(any(ismissing(xx).or.ismissing(yy)))) then
           dum(gg,4) = gsn_add_polyline(wks,plot,xx,yy,polyres)
        end if
        if (isatt(boxOpts,"boxFill_Colors").and.isatt(boxOpts,"boxFill_midColor")) then
            polyres@gsLineColor = savecolor
        end if

     end do
     dumname = unique_string("dum")
     plot@$dumname$ = dum
     if (isatt(boxOpts,"boxFill_Colors")) then
        dumname = unique_string("dum")
        plot@$dumname$ = dum3
     end if
     if (isvar("xblab").eq."True") then
        plotres@tmXBLabels = xblab      ;reassign XBLabels for possible future plots
     end if
 if (maximize) then
    mres = True
    mres@gsnDraw  = False
    mres@gsnFrame = False
    maximize_output(wks,mres)
 end if
 return(plot)
end

;=================================================================================================
undef("taylor_diagram")
function taylor_diagram (wks:graphic ,RATIO[*][*]:numeric, CC[*][*]:numeric \
                                     ,rOpts:logical)
;--------------------------------------------------------------------
; This version of taylor_diagram supports "paneling"
; It requires NCL version 4.2.0.a034 because it uses "gsn_create_legend"
;--------------------------------------------------------------------

;
; Generate a Taylor Diagram:
; Generate Multiple Aspects of Model Performance in a Single Diagram
; Taylor, K. E., J. Geophys. Res., 106, D7, 7183-7192, 2001
;
; An example:
; http://www.grida.no/climate/ipcc_tar/wg1/fig8-4.htm
;
; This expects one or more datasets. The left dimension
; is the number of datasets. The rightmost is the number of pts.
;
; Markers are at:
; http://www.ncl.ucar.edu/Document/Graphics/Resources/gs.shtml
;
; By default, the function can handle up to 10 variable comparisons..
; To expand ...  modify the 'Colors' and 'Markers' attributes.
; The user can change / add some default settings.
;
; The defaults that the user can modify:
;
; rOpts                 = True
;                                  ; 'made-up' resources
; rOpts@Colors          =  (/ "blue" , "red", "green", "cyan", "black" \
;                           , "turquoise", "brown", "yellow"/)
; rOpts@Markers         =  (/ 2, 3, 6, 14, 9, 12, 7, 4/) ; Marker Indices
; rOpts@markerTxOffset  = 0.0175   ; offset for text above marker
; rOpts@stnRad          = (/ 1. /) ;  (/ 0.50, 0.75, 1.5 /)
; rOpts@centerDiffRMS   = False    ;  True mean draw additional radii from REF
; rOpts@caseLabelsFontHeightF = 0.05
; rOpts@varLabelsFontHeightF  = 0.013
; rOpts@varLabelsYloc         = 0.65
; rOpts@legendWidth           = 0.015
; rOpts@legendHeight          = 0.030*nCase
; rOpts@taylorDraw            = True
; rOpts@taylorFrame           = True
;
;                                  ; standard NCL resources
; rOpts@tiMainString    = "Taylor" ; not using title makes plot bigger
; rOpts@gsMarkerSizeF   = 0.0085   ; marker size
; rOpts@gsMarkerThicknessF = 1.0
; rOpts@txFontHeightF   = 0.0125   ; text size
; rOpts@tiMainFontHeightF = 0.0225 ; tiMainString size
;
; It returns to the user a graphic object containing the
; Taylor background and plotted x/y pts.
; This graphic object contains a simple Taylor background appropriate
; for standardized data and the markers for the datasets.
; ==================================================================
; This version allows paneling:
;      The 'cumbersome' "dum" variables were added by
;      Adam Phillips to allow paneling via "gsn_add_?".
; ==================================================================
begin
  dimR                  = dimsizes(RATIO)
  nCase                 = dimR(0)    ; # of cases [models]
  nVar                  = dimR(1)    ; # of variables

                                     ; x/y coordinates for plotting
  X    = new ( (/nCase,nVar/) , typeof(RATIO) )
  Y    = new ( (/nCase,nVar/) , typeof(RATIO) )

  do nc=0,nCase-1
     angle      = acos( CC(nc,:) )   ; array operation
     X(nc,:)    = RATIO(nc,:)*cos( angle )
     Y(nc,:)    = RATIO(nc,:)*sin( angle )
  end do

  xyMin                 = 0.
  xyOne                 = 1.00
  xyMax                 = 1.65
  xyMax_Panel           = xyMax+ 0.10            ; paneling purposes

  if (rOpts .and. isatt(rOpts,"txFontHeightF"))  then
      FontHeightF       = rOpts@txFontHeightF    ; user wants to specify size
  else
      FontHeightF       = 0.0175
  end if

; ----------------------------------------------------------------
; Part 1:
; base plot: Based upon request of Mark Stevens
; basic x-y and draw the 1.0 observed and the outer curve at 1.65
; ----------------------------------------------------------------

  rxy                   = True
  rxy@gsnDraw           = False
  rxy@gsnFrame          = False
  rxy@vpHeightF         = 0.65
  rxy@vpWidthF          = 0.65
;  rxy@tmYLBorderOn      = False
;  rxy@tmXBBorderOn      = False

  rxy@tiYAxisString     = "Standardized Deviations (Normalized)"
  rxy@tiYAxisFontHeightF= FontHeightF                        ; default=0.025

  rxy@tmXBMode          = "Explicit"
  rxy@tmXBValues        = (/0.0,0.25,0.50,0.75,1.00,1.25,1.5/)    ; major tm
                                                                  ; default  "OBS" or "REF"
 ;rxy@tmXBLabels        = (/"0.00","0.25","0.50","0.75","REF" ,"1.25","1.50"/)
  rxy@tmXBLabels        = (/"0.00","0.25","0.50","0.75","REF" ,"1.25","1.50"/)
  if (rOpts .and. isatt(rOpts,"OneX") )  then                     ; eg: rOpts@OneX="1.00"
     ;rxy@tmXBLabels        = (/"0.00","0.25","0.50","0.75",rOpts@OneX,"1.25","1.50"/)
      rxy@tmXBLabels        = (/"    ","0.25","0.50","0.75",rOpts@OneX,"1.25","1.50"/)
  end if

  rxy@tmXBMajorLengthF  = 0.015      ; default=0.02 for a vpHeightF=0.6
  rxy@tmXBLabelFontHeightF = FontHeightF
  rxy@tmXBMinorOn       = False
  rxy@trXMaxF           = xyMax_Panel

  rxy@tmYLMode          = "Manual"
  rxy@tmYLMinorOn       = False
  rxy@tmYLMajorLengthF  = rxy@tmXBMajorLengthF
  rxy@tmYLLabelFontHeightF = FontHeightF
  rxy@tmYLMode          = "Explicit"
  rxy@tmYLValues        = (/0.0,0.25,0.50, 0.75, 1.00, 1.25, 1.5/) ; major tm
  rxy@tmYLLabels        = rxy@tmYLValues
  rxy@trYMaxF           = xyMax_Panel

  rxy@tmGridDrawOrder  = "PostDraw"

  rxy@tmYRBorderOn      = False
  rxy@tmYROn            = False      ; Turn off right tick marks.

  rxy@tmXTBorderOn      = False
  rxy@tmXTOn            = False      ; Turn off right tick marks.

  rxy@xyDashPatterns    = (/ 0 /)    ; line characteristics (dash,solid)
  rxy@xyLineThicknesses = (/ 1./)    ; choose line thickness

  rxy@gsnFrame          = False      ; Don't advance the frame.

                                            ; create outer 'correlation axis'
  npts    = 200                        ; arbitrary
  xx      = fspan(xyMin,xyMax,npts)
  yy      = sqrt(xyMax^2 - xx^2    )   ; outer correlation line (xyMax)

  sLabels = (/"0.1","0.2","0.3","0.4","0.5","0.6" \ ; correlation labels
             ,"0.7","0.8","0.9","0.95","0.99"/)  ;,"1.0"     /); also, major tm
  cLabels = stringtofloat(sLabels)
  rad     = 4.*atan(1.0)/180.
  angC    = acos(cLabels)/rad                     ; angles: correlation labels

  if (rOpts .and. isatt(rOpts,"tiMainString")) then
      rxy@tiMainString      = rOpts@tiMainString
     ;rxy@tiMainOffsetYF    = 0.015               ; default  0.0
      if (isatt(rOpts,"tiMainFontHeightF")) then
           rxy@tiMainFontHeightF = rOpts@tiMainFontHeightF
      else
           rxy@tiMainFontHeightF = 0.0225         ; default  0.025
      end if
  end if
;;if (rOpts .and. isatt(rOpts,"gsnCenterString")) then
;;    rxy@gsnCenterString  = rOpts@gsnCenterString      ; only gsn_csm_xy
;;end if

  taylor  = gsn_xy(wks,xx,yy,rxy)                 ; Create and draw XY plot.

  rsrRes  = True
  rsrRes@gsLineOpacityF = 0.25
  rsrRes@gsLineThicknessF  = rxy@xyLineThicknesses(0)  ; line thickness
  rsrRes@gsLineDashPattern = 0                    ; solid line pattern
                                                  ; draw x and y to xyMax
  dum0 = gsn_add_polyline(wks,taylor,(/0.,  0. /),(/0.,xyMax/), rsrRes)
  dum1 = gsn_add_polyline(wks,taylor,(/0.,xyMax/),(/0.,  0. /), rsrRes)

  xx   = fspan(xyMin, xyOne ,npts)                ; draw 1.0 standard radius
  yy   = sqrt(xyOne - xx^2)
  rsrRes@gsLineColor = "LightGray"
  rsrRes@gsLineDashPattern = 1                    ; dashed line pattern
  rsrRes@gsLineThicknessF  = rxy@xyLineThicknesses(0)  ; line thickness
  dum2 = gsn_add_polyline(wks,taylor,xx,yy, rsrRes)
  delete(xx)
  delete(yy)

  if (rOpts .and. isatt(rOpts,"stnRad") ) then
      rsrRes@gsLineThicknessF  = 1   ; rxy@xyLineThicknesses(0)
      nStnRad = dimsizes(rOpts@stnRad)

      dum3  = new(nStnRad,graphic)
      do n=0,nStnRad-1
         rr = rOpts@stnRad(n)
         xx = fspan(xyMin, rr ,npts)
         yy = sqrt(rr^2   - xx^2)
         dum3(n) = gsn_add_polyline(wks,taylor,xx,yy, rsrRes)
      end do
      taylor@$unique_string("dum")$ = dum3

      delete(xx)
      delete(yy)
  end if

  getvalues taylor                                ; get style info from taylor
    "tmYLLabelFont"        : tmYLLabelFont        ; use for correlation axis
    "tmYLLabelFontHeightF" : tmYLLabelFontHeightF
  end getvalues

; ----------------------------------------------------------------
; Part 2:
; Correlation labels
; ----------------------------------------------------------------
  radC    = xyMax                                  ; for correlation labels
  xC      = radC*cos(angC*rad)
  yC      = radC*sin(angC*rad)
; added to get some separation
  xC      = xC + 0.020*cos(rad*angC)
  yC      = yC + 0.060*sin(rad*angC)

  xC(:5) = xC(:5) - 0.01
  yC(:5) = yC(:5) + 0.01

  txRes               = True                      ; text mods desired
  txRes@txFontHeightF = FontHeightF               ; match YL
  txRes@tmYLLabelFont = tmYLLabelFont             ; match YL
  txRes@txAngleF      = -45.
  if (.not.isatt(rOpts,"drawCorLabel") .or. rOpts@drawCorLabel) then
      dum4 = gsn_add_text(wks,taylor,"Correlation",1.30,1.30,txRes)
	 taylor@$unique_string("dum")$ = dum4
  end if
  txRes@txAngleF      = 0.0
  txRes@txFontHeightF = FontHeightF*0.50          ; bit smaller

;;dum0 = gsn_add_text(wks,taylor,"OBSERVED",1.00,0.075,txRes)

  plRes               = True
  plRes@gsLineThicknessF = 2.

  txRes@txJust        = "CenterLeft"              ; Default="CenterCenter".
  txRes@txFontHeightF = FontHeightF               ; match YL
 ;txRes@txBackgroundFillColor = "white"

  tmEnd = 0.975
  radTM = xyMax*tmEnd                             ; radius end: major TM
  xTM   = new( 2 , "float")
  yTM   = new( 2 , "float")

  dum5 = new(dimsizes(sLabels),graphic)
  dum6 = dum5

  do i=0,dimsizes(sLabels)-1                      ; Loop to draw strings
    txRes@txAngleF = angC(i)
    dum5(i) = gsn_add_text(wks, taylor, sLabels(i),xC(i),yC(i),txRes) ; cor label
    xTM(0)   = xyMax*cos(angC(i)*rad)             ; major tickmarks at
    yTM(0)   = xyMax*sin(angC(i)*rad)             ; correlation labels
    xTM(1)   = radTM*cos(angC(i)*rad)
    yTM(1)   = radTM*sin(angC(i)*rad)
    dum6(i) = gsn_add_polyline(wks,taylor,xTM,yTM,plRes)
  end do
                                                  ; minor tm locations
  mTM     = (/0.05,0.15,0.25,0.35,0.45,0.55,0.65 \
             ,0.75,0.85,0.91,0.92,0.93,0.94,0.96,0.97,0.98  /)
  angmTM  = acos(mTM)/rad                         ; angles: correlation labels
  radmTM  = xyMax*(1.-(1.-tmEnd)*0.5)             ; radius end: minor TM

  dum7 = new(dimsizes(mTM),graphic)

  do i=0,dimsizes(mTM)-1                          ; manually add tm
    xTM(0)   = xyMax*cos(angmTM(i)*rad)           ; minor tickmarks
    yTM(0)   = xyMax*sin(angmTM(i)*rad)
    xTM(1)   = radmTM*cos(angmTM(i)*rad)
    yTM(1)   = radmTM*sin(angmTM(i)*rad)
    dum7(i)  = gsn_add_polyline(wks,taylor,xTM,yTM,plRes)
  end do
                                                  ; added for Wanli
  if (rOpts .and. isatt(rOpts,"ccRays") ) then
      angRL = acos(rOpts@ccRays)/rad             ; angles: radial lines

      rlRes = True
      rlRes@gsLineDashPattern= 2  ; line pattern
      rlRes@gsLineThicknessF = 1  ; choose line thickness
      rlRes@gsLineOpacityF = 0.25
      if (isatt(rOpts,"ccRays_color")) then
          rlRes@gsLineColor    =  "LightGray"
      end if

      dum8 = new(dimsizes(angRL),graphic)
      do i=0,dimsizes(angRL)-1
         xRL     = xyMax*cos(angRL(i)*rad)
         yRL     = xyMax*sin(angRL(i)*rad)
         dum8(i) = gsn_add_polyline(wks,taylor,(/0, xRL /),(/0,  yRL  /),rlRes)
      end do
      taylor@$unique_string("dum")$ = dum8
  end if

; ----------------------------------------------------------------
; Part 3:
; Concentric about 1.0 on XB axis
; ----------------------------------------------------------------
  if (rOpts .and. isatt(rOpts,"centerDiffRMS") \
            .and. rOpts@centerDiffRMS) then
      respl                    = True                ; polyline mods desired
      respl@gsLineOpacityF = 0.25
      respl@tfPolyDrawOrder    = "PreDraw"
      respl@xyLineThicknessF   = 1.0                 ; line thickness
      respl@xyLineDashPattern  = 2                   ; short dash lines
      respl@gsLineColor        = "Black"             ; line color
      if (isatt(rOpts,"centerDiffRMS_color")) then
          respl@gsLineColor    =  "LightGray"
      end if
	     respl@tfPolyDrawOrder = "PreDraw"

      dx   = 0.25
      ncon = 4                                       ; 0.75, 0.50, 0.25, 0.0
      npts = 100                                     ; arbitrary
      ang  = fspan(180,360,npts)*rad

      dum9 = new(ncon,graphic)

      do n=1,ncon
         rr  = n*dx            ; radius from 1.0 [OBS] abscissa
         xx  = 1. + rr*cos(ang)
         yy  = fabs( rr*sin(ang) )
         if (n.le.2) then
             dum9(n-1) = gsn_add_polyline(wks,taylor,xx,yy,respl)
         end if
         if (n.eq.3) then
             n3 = floattointeger( 0.77*npts )
             dum9(n-1) = gsn_add_polyline(wks,taylor,xx(0:n3),yy(0:n3),respl)
         end if
         if (n.eq.4) then
             n4 = floattointeger( 0.61*npts )
             dum9(n-1) = gsn_add_polyline(wks,taylor,xx(0:n4),yy(0:n4),respl)
         end if
      end do
      delete(ang)
      delete(xx)
      delete(yy)
      taylor@$unique_string("dum")$ = dum9

  end if
; ---------------------------------------------------------------
; Part 4:
; generic resources that will be applied to all users data points
; of course, these can be changed
; http://www.ncl.ucar.edu/Document/Graphics/Resources/gs.shtml
; ---------------------------------------------------------------
  if (rOpts .and. isatt(rOpts,"Markers")) then
      Markers = rOpts@Markers
  else
      Markers = (/ 4, 6, 8,  0, 9, 12, 7, 2, 11, 16/) ; Marker Indices
  end if

  if (rOpts .and. isatt(rOpts,"Colors")) then
      Colors  = rOpts@Colors
  else
      Colors  = (/ "red", "blue", "green", "cyan", "orange" \
                 , "turquoise", "brown", "yellow", "purple", "black"/)
  end if

  if (rOpts .and. isatt(rOpts,"gsMarkerThicknessF")) then
      gsMarkerThicknessF = rOpts@gsMarkerThicknessF
  else
      gsMarkerThicknessF = 1.0
  end if

  if (rOpts .and. isatt(rOpts,"gsMarkerSizeF")) then
      gsMarkerSizeF      = rOpts@gsMarkerSizeF
  else
      gsMarkerSizeF      = 0.0085                  ; Default: 0.007
  end if

  gsRes = True
  gsRes@gsMarkerThicknessF = gsMarkerThicknessF      ; default=1.0
  gsRes@gsMarkerSizeF      = gsMarkerSizeF           ; Default: 0.007

  ptRes = True                        ; text options for points
  ptRes@txJust             = "BottomCenter"; Default="CenterCenter".
  ptRes@txFontThicknessF   = 1.2      ; default=1.00
  ptRes@txFontHeightF      = 0.0125   ; default=0.05
  if (rOpts .and. isatt(rOpts,"txFontHeightF")) then
      ptRes@txFontHeightF  = rOpts@txFontHeightF
  end if

  markerTxYOffset          = 0.0175   ; default
  if (rOpts .and. isatt(rOpts,"markerTxYOffset")) then
      markerTxYOffset = rOpts@markerTxYOffset             ; user defined offset
  end if

  dum10 = new((nCase*nVar),graphic)
;  dum11 = dum10

  do n=0,nCase-1
     gsRes@gsMarkerIndex   = Markers(n)             ; marker style (+)
     gsRes@gsMarkerColor   = Colors(n)              ; marker color
     ptRes@txFontColor     = gsRes@gsMarkerColor
    do i=0,nVar-1
       dum10(n*nVar+i) = gsn_add_polymarker(wks,taylor,X(n,i),Y(n,i),gsRes)
;       dum11(n*nVar+i) = gsn_add_text(wks,taylor,(i+1),X(n,i),Y(n,i)+markerTxYOffset,ptRes)
    end do
  end do

; ---------------------------------------------------------------
; Part 5: ; add case legend and variable labels
; ---------------------------------------------------------------

  if (rOpts .and. isatt(rOpts,"caseLabels")) then

      if (isatt(rOpts,"caseLabelsFontHeightF")) then
          caseLabelsFontHeightF = rOpts@caseLabelsFontHeightF
      else
          caseLabelsFontHeightF = 0.05
      end if

      lgres                    = True
      lgres@lgMarkerColors     = Colors        ; colors of markers
      lgres@lgMarkerIndexes    = Markers       ; Markers
      lgres@lgMarkerSizeF      = gsMarkerSizeF ; Marker size
      lgres@lgItemType         = "Markers"     ; draw markers only
      lgres@lgLabelFontHeightF = caseLabelsFontHeightF  ; font height of legend case labels

      if (isatt(rOpts,"legendWidth")) then
          lgres@vpWidthF       = rOpts@legendWidth
      else
          lgres@vpWidthF       = 0.15           ; width of legend (NDC)
      end if

      if (isatt(rOpts,"legendHeight")) then
          lgres@vpHeightF      = rOpts@legendHeight
      else
          lgres@vpHeightF      = 0.030*nCase   ; height of legend (NDC)
      end if

      lgres@lgPerimOn          = False         ; turn off perimeter
      nModel                   = dimsizes( rOpts@caseLabels )
      lbid = gsn_create_legend(wks,nModel,rOpts@caseLabels,lgres)

      amres = True
      amres@amParallelPosF     =  0.35
      amres@amOrthogonalPosF   = -0.35
      annoid1 = gsn_add_annotation(taylor,lbid,amres)	; add legend to plot
  end if

  if (rOpts .and. isatt(rOpts,"varLabels")) then
      nVar    = dimsizes(rOpts@varLabels)

      if (isatt(rOpts,"varLabelsFontHeightF")) then
          varLabelsFontHeightF = rOpts@varLabelsFontHeightF
      else
          varLabelsFontHeightF = 0.013
      end if

      txres = True
      txres@txFontHeightF = varLabelsFontHeightF
      txres@txJust = "CenterLeft"              ; justify to the center left

     ;delta_y = 0.02
     delta_y = 0.06
      if (rOpts .and. isatt(rOpts,"varLabelsYloc")) then
          ys  = rOpts@varLabelsYloc            ; user specified
      else
          ys  = max( (/nVar*delta_y , 0.30/) )
      end if


      do i = 1,nVar
         if (i.eq.1) then
             dum12 = new(nVar,graphic)
	 end if

         dum12(i-1) = gsn_add_text(wks,taylor,i+" - "+rOpts@varLabels(i-1), .125,ys,txres)
         ys = ys- delta_y
      end do

      taylor@$unique_string("dum")$ = dum12
  end if

  taylor@$unique_string("dum")$ = dum0   ; x-axis
  taylor@$unique_string("dum")$ = dum1   ; y-axis
  taylor@$unique_string("dum")$ = dum2   ; 1.0 std curve
  taylor@$unique_string("dum")$ = dum5   ; labels [COR]
  taylor@$unique_string("dum")$ = dum6   ; major tm [COR]
  taylor@$unique_string("dum")$ = dum7   ; minor tm
  taylor@$unique_string("dum")$ = dum10  ; markers
;  taylor@$unique_string("dum")$ = dum11  ; text

  if (.not.isatt(rOpts,"taylorDraw") .or. \
     (isatt(rOpts,"taylorDraw") .and. rOpts@taylorDraw)) then
	draw(taylor)
  end if
  if (.not.isatt(rOpts,"taylorFrame") .or. \
     (isatt(rOpts,"taylorFrame") .and. rOpts@taylorFrame)) then
	frame(wks)
  end if
  return(taylor)
end
;==================================================================================================
undef("taylor_diagram3")
function taylor_diagram3 (wks:graphic ,RATIO[*][*]:numeric, CC[*][*]:numeric \
                                     ,rOpts:logical)
;--------------------------------------------------------------------
; This version of taylor_diagram supports "paneling"
; It requires NCL version 4.2.0.a034 because it uses "gsn_create_legend"
;--------------------------------------------------------------------

;
; Generate a Taylor Diagram:
; Generate Multiple Aspects of Model Performance in a Single Diagram
; Taylor, K. E., J. Geophys. Res., 106, D7, 7183-7192, 2001
;
; An example:
; http://www.grida.no/climate/ipcc_tar/wg1/fig8-4.htm
;
; This expects one or more datasets. The left dimension
; is the number of datasets. The rightmost is the number of pts.
;
; Markers are at:
; http://www.ncl.ucar.edu/Document/Graphics/Resources/gs.shtml
;
; By default, the function can handle up to 10 variable comparisons..
; To expand ...  modify the 'Colors' and 'Markers' attributes.
; The user can change / add some default settings.
;
; The defaults that the user can modify:
;
; rOpts                 = True
;                                  ; 'made-up' resources
; rOpts@Colors          =  (/ "blue" , "red", "green", "cyan", "black" \
;                           , "turquoise", "brown", "yellow"/)
; rOpts@Markers         =  (/ 2, 3, 6, 14, 9, 12, 7, 4/) ; Marker Indices
; rOpts@markerTxOffset  = 0.0175   ; offset for text above marker
; rOpts@stnRad          = (/ 1. /) ;  (/ 0.50, 0.75, 1.5 /)
; rOpts@centerDiffRMS   = False    ;  True mean draw additional radii from REF
; rOpts@caseLabelsFontHeightF = 0.05
; rOpts@varLabelsFontHeightF  = 0.013
; rOpts@varLabelsYloc         = 0.65
; rOpts@legendWidth           = 0.015
; rOpts@legendHeight          = 0.030*nCase
; rOpts@taylorDraw            = True
; rOpts@taylorFrame           = True
;
;                                  ; standard NCL resources
; rOpts@tiMainString    = "Taylor" ; not using title makes plot bigger
; rOpts@gsMarkerSizeF   = 0.0085   ; marker size
; rOpts@gsMarkerThicknessF = 1.0
; rOpts@txFontHeightF   = 0.0125   ; text size
; rOpts@tiMainFontHeightF = 0.0225 ; tiMainString size
;
; It returns to the user a graphic object containing the
; Taylor background and plotted x/y pts.
; This graphic object contains a simple Taylor background appropriate
; for standardized data and the markers for the datasets.
; ==================================================================
; This version allows paneling:
;      The 'cumbersome' "dum" variables were added by
;      Adam Phillips to allow paneling via "gsn_add_?".
; ==================================================================
begin
  dimR                  = dimsizes(RATIO)
  nCase                 = dimR(0)    ; # of cases [models]
  nVar                  = dimR(1)    ; # of variables

                                     ; x/y coordinates for plotting
  X    = new ( (/nCase,nVar/) , typeof(RATIO) )
  Y    = new ( (/nCase,nVar/) , typeof(RATIO) )

  do nc=0,nCase-1
     angle      = acos( CC(nc,:) )   ; array operation
     X(nc,:)    = RATIO(nc,:)*cos( angle )
     Y(nc,:)    = RATIO(nc,:)*sin( angle )
  end do

  xyMin                 = 0.
  xyOne                 = 1.00
  xyMax                 = 3.5   ;1.90  ;1.65
  xyMax_Panel           = xyMax+ 0.16            ; paneling purposes

  if (rOpts .and. isatt(rOpts,"txFontHeightF"))  then
      FontHeightF       = rOpts@txFontHeightF    ; user wants to specify size
  else
      FontHeightF       = 0.0175
  end if

; ----------------------------------------------------------------
; Part 1:
; base plot: Based upon request of Mark Stevens
; basic x-y and draw the 1.0 observed and the outer curve at 1.65
; ----------------------------------------------------------------

  rxy                   = True
  rxy@gsnDraw           = False
  rxy@gsnFrame          = False
  rxy@vpHeightF         = 0.65
  rxy@vpWidthF          = 0.65
;  rxy@tmYLBorderOn      = False
;  rxy@tmXBBorderOn      = False

  rxy@tiYAxisString     = "Standardized Deviations (Normalized)"
  rxy@tiYAxisFontHeightF= FontHeightF                        ; default=0.025

  rxy@tmXBMode          = "Explicit"
  rxy@tmXBValues        = (/0.0, 0.50, 1.00, 1.5,2.0,2.5,3.0,3.50/)    ; major tm
                                                                  ; default  "OBS" or "REF"
  rxy@tmXBLabels        = (/"0","0.5","REF","1.5","2.0","2.5","3.0","3.5"/)
  if (rOpts .and. isatt(rOpts,"OneX") )  then                     ; eg: rOpts@OneX="1.00"
     ;rxy@tmXBLabels        = (/"0.00","0.25","0.50","0.75",rOpts@OneX,"1.25","1.50"/)
      rxy@tmXBLabels        = (/"    ","0.25","0.50","0.75",rOpts@OneX,"1.25","1.50","1.75","2.0","2.25","2.5","2.75","3.0","3.25","3.50"/)
  end if

  rxy@tmXBMajorLengthF  = 0.015      ; default=0.02 for a vpHeightF=0.6Ã§Ã§
  rxy@tmXBLabelFontHeightF = FontHeightF
  rxy@tmXBMinorOn       = False
  rxy@trXMaxF           = xyMax_Panel

  rxy@tmYLMode          = "Manual"
  rxy@tmYLMinorOn       = False
  rxy@tmYLMajorLengthF  = rxy@tmXBMajorLengthF
  rxy@tmYLLabelFontHeightF = FontHeightF
  rxy@tmYLMode          = "Explicit"
  rxy@tmYLValues        = (/0.0, 0.50, 1.00, 1.5,2.0,2.5,3.0,3.50/) ; major tm
  rxy@tmYLLabels        = rxy@tmYLValues
  rxy@trYMaxF           = xyMax_Panel

  rxy@tmYRBorderOn      = False
  rxy@tmYROn            = False      ; Turn off right tick marks.

  rxy@tmXTBorderOn      = False
  rxy@tmXTOn            = False      ; Turn off right tick marks.

  rxy@xyDashPatterns    = (/ 0 /)    ; line characteristics (dash,solid)
  rxy@xyLineThicknesses = (/ 1./)    ; choose line thickness

  rxy@tmGridDrawOrder  = "PostDraw"

  rxy@gsnFrame          = False      ; Don't advance the frame.

                                            ; create outer 'correlation axis'
  npts    = 200                        ; arbitrary
  xx      = fspan(xyMin,xyMax,npts)
  yy      = sqrt(xyMax^2 - xx^2    )   ; outer correlation line (xyMax)

  sLabels = (/"    0.0","0.1","0.2","0.3","0.4","0.5","0.6" \ ; correlation labels
             ,"0.7","0.8","0.9","0.95","0.99"/)    ;,"   1.0"     /); also, major tm
  cLabels = stringtofloat(sLabels)
  rad     = 4.*atan(1.0)/180.
  angC    = acos(cLabels)/rad                     ; angles: correlation labels

  if (rOpts .and. isatt(rOpts,"tiMainString")) then
      rxy@tiMainString      = rOpts@tiMainString
     ;rxy@tiMainOffsetYF    = 0.015               ; default  0.0
      if (isatt(rOpts,"tiMainFontHeightF")) then
           rxy@tiMainFontHeightF = rOpts@tiMainFontHeightF
      else
           rxy@tiMainFontHeightF = 0.0225         ; default  0.025
      end if
  end if
;;if (rOpts .and. isatt(rOpts,"gsnCenterString")) then
;;    rxy@gsnCenterString  = rOpts@gsnCenterString      ; only gsn_csm_xy
;;end if

  taylor  = gsn_xy(wks,xx,yy,rxy)                 ; Create and draw XY plot.

  rsrRes  = True
  rsrRes@gsLineOpacityF = 0.25
  rsrRes@gsLineThicknessF  = rxy@xyLineThicknesses(0)  ; line thickness
  rsrRes@gsLineDashPattern = 0                    ; solid line pattern
                                                  ; draw x and y to xyMax
  dum0 = gsn_add_polyline(wks,taylor,(/0.,  0. /),(/0.,xyMax/), rsrRes)
  dum1 = gsn_add_polyline(wks,taylor,(/0.,xyMax/),(/0.,  0. /), rsrRes)

  xx   = fspan(xyMin, xyOne ,npts)                ; draw 1.0 standard radius
  yy   = sqrt(xyOne - xx^2)
  rsrRes@gsLineColor = "LightGray"
  rsrRes@gsLineDashPattern = 1                    ; dashed line pattern
  rsrRes@gsLineThicknessF  = rxy@xyLineThicknesses(0)  ; line thickness
  dum2 = gsn_add_polyline(wks,taylor,xx,yy, rsrRes)
  delete(xx)
  delete(yy)

  if (rOpts .and. isatt(rOpts,"stnRad") ) then
      rsrRes@gsLineThicknessF  = 1   ; rxy@xyLineThicknesses(0)

      nStnRad = dimsizes(rOpts@stnRad)

      dum3  = new(nStnRad,graphic)
      do n=0,nStnRad-1
         rr = rOpts@stnRad(n)
         xx = fspan(xyMin, rr ,npts)
         yy = sqrt(rr^2   - xx^2)
         dum3(n) = gsn_add_polyline(wks,taylor,xx,yy, rsrRes)
      end do
      taylor@$unique_string("dum")$ = dum3

      delete(xx)
      delete(yy)
  end if

  getvalues taylor                                ; get style info from taylor
    "tmYLLabelFont"        : tmYLLabelFont        ; use for correlation axis
    "tmYLLabelFontHeightF" : tmYLLabelFontHeightF
  end getvalues

; ----------------------------------------------------------------
; Part 2:
; Correlation labels
; ----------------------------------------------------------------
  radC    = xyMax                                  ; for correlation labels
  xC      = radC*cos(angC*rad)+.025
  yC      = radC*sin(angC*rad)+.005
; added to get some separation
  xC      = xC + 0.020*cos(rad*angC)
  yC      = yC + 0.120*sin(rad*angC)

  xC(:6) = xC(:6) - 0.05
  yC(:6) = yC(:6) + 0.05

  txRes               = True                      ; text mods desired
  txRes@txFontHeightF = FontHeightF               ; match YL
  txRes@tmYLLabelFont = tmYLLabelFont             ; match YL
  txRes@txAngleF      = -45.
  if (.not.isatt(rOpts,"drawCorLabel") .or. rOpts@drawCorLabel) then
     dum4 = gsn_add_text(wks,taylor,"Correlation",2.8,2.8,txRes)
	    taylor@$unique_string("dum")$ = dum4
  end if
  txRes@txAngleF      = 0.0

;;dum0 = gsn_add_text(wks,taylor,"OBSERVED",1.00,0.075,txRes)

  plRes               = True
  plRes@gsLineThicknessF = 2.

  txRes@txJust        = "CenterLeft"              ; Default="CenterCenter".
  txRes@txFontHeightF = FontHeightF               ; match YL
 ;txRes@txBackgroundFillColor = "white"

  tmEnd = 0.975
  radTM = xyMax*tmEnd                             ; radius end: major TM
  xTM   = new( 2 , "float")
  yTM   = new( 2 , "float")

  dum5 = new(dimsizes(sLabels),graphic)
  dum6 = dum5

  do i=0,dimsizes(sLabels)-1                      ; Loop to draw strings
;    print(""+sLabels(i)+" "+xC(i)+" "+yC(i))
    txRes@txAngleF = angC(i)
    dum5(i) = gsn_add_text(wks, taylor, sLabels(i),xC(i),yC(i),txRes) ; cor label
    xTM(0)   = xyMax*cos(angC(i)*rad)             ; major tickmarks at
    yTM(0)   = xyMax*sin(angC(i)*rad)             ; correlation labels
    xTM(1)   = radTM*cos(angC(i)*rad)
    yTM(1)   = radTM*sin(angC(i)*rad)
    dum6(i) = gsn_add_polyline(wks,taylor,xTM,yTM,plRes)
  end do
                                                  ; minor tm locations
  mTM     = (/0.05,0.15,0.25,0.35,0.45,0.55,0.65 \
             ,0.75,0.85,0.91,0.92,0.93,0.94,0.96,0.97,0.98  /)
  angmTM  = acos(mTM)/rad                         ; angles: correlation labels
  radmTM  = xyMax*(1.-(1.-tmEnd)*0.5)             ; radius end: minor TM

  dum7 = new(dimsizes(mTM),graphic)

  do i=0,dimsizes(mTM)-1                          ; manually add tm
    xTM(0)   = xyMax*cos(angmTM(i)*rad)           ; minor tickmarks
    yTM(0)   = xyMax*sin(angmTM(i)*rad)
    xTM(1)   = radmTM*cos(angmTM(i)*rad)
    yTM(1)   = radmTM*sin(angmTM(i)*rad)
    dum7(i)  = gsn_add_polyline(wks,taylor,xTM,yTM,plRes)
  end do
                                                  ; added for Wanli
  if (rOpts .and. isatt(rOpts,"ccRays") ) then
      angRL = acos(rOpts@ccRays)/rad             ; angles: radial lines

      rlRes = True
      rlRes@gsLineDashPattern= 2  ; line pattern
      rlRes@gsLineThicknessF = 1  ; choose line thickness
      rlRes@gsLineOpacityF = 0.25
      if (isatt(rOpts,"ccRays_color")) then
          rlRes@gsLineColor    =  "LightGray"
      end if

      dum8 = new(dimsizes(angRL),graphic)
      do i=0,dimsizes(angRL)-1
         xRL     = xyMax*cos(angRL(i)*rad)
         yRL     = xyMax*sin(angRL(i)*rad)
         dum8(i) = gsn_add_polyline(wks,taylor,(/0, xRL /),(/0,  yRL  /),rlRes)
      end do
      taylor@$unique_string("dum")$ = dum8
  end if

; ----------------------------------------------------------------
; Part 3:
; Concentric about 1.0 on XB axis
; I think this is correct. Still test mode.
; ----------------------------------------------------------------
  if (rOpts .and. isatt(rOpts,"centerDiffRMS") \
            .and. rOpts@centerDiffRMS) then
      respl                    = True                ; polyline mods desired
      respl@gsLineOpacityF = 0.25
      respl@xyLineThicknessF   = 1.0                 ; line thickness
      respl@xyLineDashPattern  = 2                   ; short dash lines
      respl@gsLineColor        = "Black"             ; line color
      if (isatt(rOpts,"centerDiffRMS_color")) then
          respl@gsLineColor    =  "LightGray"
      end if
	     respl@tfPolyDrawOrder = "PreDraw"
      dx   = 0.5
      ncon = 5                                       ; 0.75, 0.50, 0.25, 0.0
      npts = 100                                     ; arbitrary
      ang  = fspan(180,360,npts)*rad

      dum9 = new(ncon,graphic)
	  ;;;; IF YOU DESIRE RMS LABELS
	  ;rms_labels = (/2.75,2.50,2.25,2.00,1.75,1.5,1.25,1.00,0.75,0.5,0.25,0.0/)

      do n=1,ncon
		 ;respl@gsLineLabelString = rms_labels(n-1)  ;IF YOU DESIRE RMS LABELS
		 ;respl@gsLineLabelFontColor    =  "LightGray"
         rr  = n*dx            ; radius from 1.0 [OBS] abscissa
         xx  = 1. + rr*cos(ang)
         yy  = fabs( rr*sin(ang) )
         ; if (n.le.2) then
         ;              dum9(n-1) = gsn_add_polyline(wks,taylor,xx,yy,respl)
         ;          end if
         ; if (n.eq.3) then
         ;              ;n3 = floattointeger( 0.77*npts )
         ;              ;dum9(n-1) = gsn_add_polyline(wks,taylor,xx(0:n3),yy(0:n3),respl)
         ; 			 dum9(n-1) = gsn_add_polyline(wks,taylor,xx,yy,respl)
         ;          end if
         ;          if (n.eq.4) then
         ;              ;n4 = floattointeger( 0.61*npts )
         ;              ;dum9(n-1) = gsn_add_polyline(wks,taylor,xx(0:n4),yy(0:n4),respl)
         ; 			 dum9(n-1) = gsn_add_polyline(wks,taylor,xx,yy,respl)
         ;          end if
		 ;;;;;; Erik's Changes ;;;; May need to change accordingly
         if (n.le.9) then
                      dum9(n-1) = gsn_add_polyline(wks,taylor,xx,yy,respl)
         end if

         if (n.eq.9) then
             n9 = floattointeger( 0.71*npts )
             dum9(n-1) = gsn_add_polyline(wks,taylor,xx(0:n9),yy(0:n9),respl)
			 ;dum9(n-1) = gsn_add_polyline(wks,taylor,xx,yy,respl)
         end if
         if (n.eq.10) then
             n10 = floattointeger( 0.61*npts )
             dum9(n-1) = gsn_add_polyline(wks,taylor,xx(0:n10),yy(0:n10),respl)
			 ;dum9(n-1) = gsn_add_polyline(wks,taylor,xx,yy,respl)
         end if
         if (n.eq.11) then
             n11 = floattointeger( 0.53*npts )
             dum9(n-1) = gsn_add_polyline(wks,taylor,xx(0:n11),yy(0:n11),respl)
			 ;dum9(n-1) = gsn_add_polyline(wks,taylor,xx,yy,respl)
         end if


      end do
      delete(ang)
      delete(xx)
      delete(yy)
      taylor@$unique_string("dum")$ = dum9

  end if
; ---------------------------------------------------------------
; Part 4:
; generic resources that will be applied to all users data points
; of course, these can be changed
; http://www.ncl.ucar.edu/Document/Graphics/Resources/gs.shtml
; ---------------------------------------------------------------
  if (rOpts .and. isatt(rOpts,"Markers")) then
      Markers = rOpts@Markers
  else
      Markers = (/ 4, 6, 8,  0, 9, 12, 7, 2, 11, 16/) ; Marker Indices
  end if

  if (rOpts .and. isatt(rOpts,"Colors")) then
      Colors  = rOpts@Colors
  else
      Colors  = (/ "red", "blue", "green", "cyan", "orange" \
                 , "turquoise", "brown", "yellow", "purple", "black"/)
  end if

  if (rOpts .and. isatt(rOpts,"gsMarkerThicknessF")) then
      gsMarkerThicknessF = rOpts@gsMarkerThicknessF
  else
      gsMarkerThicknessF = 1.0
  end if

  if (rOpts .and. isatt(rOpts,"gsMarkerSizeF")) then
      gsMarkerSizeF      = rOpts@gsMarkerSizeF
  else
      gsMarkerSizeF      = 0.0085                  ; Default: 0.007
  end if

  gsRes = True
  gsRes@gsMarkerThicknessF = gsMarkerThicknessF      ; default=1.0
  gsRes@gsMarkerSizeF      = gsMarkerSizeF           ; Default: 0.007

  ptRes = True                        ; text options for points
  ptRes@txJust             = "BottomCenter"; Default="CenterCenter".
  ptRes@txFontThicknessF   = 1.2      ; default=1.00
  ptRes@txFontHeightF      = 0.0125   ; default=0.05
  if (rOpts .and. isatt(rOpts,"txFontHeightF")) then
      ptRes@txFontHeightF  = rOpts@txFontHeightF
  end if

  markerTxYOffset          = 0.0175   ; default
  if (rOpts .and. isatt(rOpts,"markerTxYOffset")) then
      markerTxYOffset = rOpts@markerTxYOffset             ; user defined offset
  end if

  dum10 = new((nCase*nVar),graphic)
  ; dum11 = dum10

  do n=0,nCase-1
     gsRes@gsMarkerIndex   = Markers(n)             ; marker style (+)
     gsRes@gsMarkerColor   = Colors(n)              ; marker color
     ptRes@txFontColor     = gsRes@gsMarkerColor
    do i=0,nVar-1
       dum10(n*nVar+i) = gsn_add_polymarker(wks,taylor,X(n,i),Y(n,i),gsRes)
;;       print("X(n,i) = " + X(n,i) + " " + "Y(n,i) = " + Y(n,i))
       ; changes
	   ; dum11(n*nVar+i) = gsn_add_text(wks,taylor,(i+1),X(n,i),Y(n,i)+markerTxYOffset,ptRes)

	   ; if (rOpts .and. isatt(rOpts,"markerLabels")) then
	   ; 	      dum11(n*nVar+i) = gsn_add_text(wks,taylor,rOpts@markerLabels(i),X(n,i),Y(n,i)+markerTxYOffset,ptRes)
	   ; 	   else
	   ; 	      dum11(n*nVar+i) = gsn_add_text(wks,taylor,(i+1),X(n,i),Y(n,i)+markerTxYOffset,ptRes)
	   ; 	   end if
    end do
  end do

; ---------------------------------------------------------------
; Part 5: ; add case legend and variable labels
; ---------------------------------------------------------------

  if (rOpts .and. isatt(rOpts,"caseLabels")) then

      if (isatt(rOpts,"caseLabelsFontHeightF")) then
          caseLabelsFontHeightF = rOpts@caseLabelsFontHeightF
      else
          caseLabelsFontHeightF = 0.05
      end if

      lgres                    = True
      lgres@lgMarkerColors     = Colors        ; colors of markers
      lgres@lgMarkerIndexes    = Markers       ; Markers
      lgres@lgMarkerSizeF      = gsMarkerSizeF ; Marker size
      lgres@lgItemType         = "Markers"     ; draw markers only
      lgres@lgLabelFontHeightF = caseLabelsFontHeightF  ; font height of legend case labels

      if (isatt(rOpts,"legendWidth")) then
          lgres@vpWidthF       = rOpts@legendWidth
      else
          lgres@vpWidthF       = 0.15           ; width of legend (NDC)
      end if

      if (isatt(rOpts,"legendHeight")) then
          lgres@vpHeightF      = rOpts@legendHeight
      else
          lgres@vpHeightF      = 0.030*nCase   ; height of legend (NDC)
      end if

      lgres@lgPerimOn          = False         ; turn off perimeter
      nModel                   = dimsizes( rOpts@caseLabels )
      lbid = gsn_create_legend(wks,nModel,rOpts@caseLabels,lgres)

      amres = True
      amres@amParallelPosF     =  0.35
      amres@amOrthogonalPosF   = -0.35
      annoid1 = gsn_add_annotation(taylor,lbid,amres)	; add legend to plot
  end if

  if (rOpts .and. isatt(rOpts,"varLabels")) then
      nVar    = dimsizes(rOpts@varLabels)

      if (isatt(rOpts,"varLabelsFontHeightF")) then
          varLabelsFontHeightF = rOpts@varLabelsFontHeightF
      else
          varLabelsFontHeightF = 0.013
      end if

      txres = True
      txres@txFontHeightF = varLabelsFontHeightF
      txres@txJust = "CenterLeft"              ; justify to the center left

     ;delta_y = 0.02
      delta_y = 0.06
      if (rOpts .and. isatt(rOpts,"varLabelsYloc")) then
          ys  = rOpts@varLabelsYloc            ; user specified
      else
          ys  = max( (/nVar*delta_y , 0.30/) )
      end if


      do i = 1,nVar
         if (i.eq.1) then
             dum12 = new(nVar,graphic)
	 end if

         dum12(i-1) = gsn_add_text(wks,taylor,i+" - "+rOpts@varLabels(i-1), .125,ys,txres)
         ys = ys- delta_y
      end do

      taylor@$unique_string("dum")$ = dum12
  end if

  taylor@$unique_string("dum")$ = dum0   ; x-axis
  taylor@$unique_string("dum")$ = dum1   ; y-axis
  taylor@$unique_string("dum")$ = dum2   ; 1.0 std curve
  taylor@$unique_string("dum")$ = dum5   ; labels [COR]
  taylor@$unique_string("dum")$ = dum6   ; major tm [COR]
  taylor@$unique_string("dum")$ = dum7   ; minor tm
  taylor@$unique_string("dum")$ = dum10  ; markers
  ; taylor@$unique_string("dum")$ = dum11  ; text

  if (.not.isatt(rOpts,"taylorDraw") .or. \
     (isatt(rOpts,"taylorDraw") .and. rOpts@taylorDraw)) then
	draw(taylor)
  end if
  if (.not.isatt(rOpts,"taylorFrame") .or. \
     (isatt(rOpts,"taylorFrame") .and. rOpts@taylorFrame)) then
	frame(wks)
  end if
  return(taylor)
end
;-----------------------------------------------------------------------------------------
undef("blank_boxplot")
function blank_boxplot(wks2:graphic,pres:logical)
local wks2,pres,pres2,options,xcoordinate,carr,plot
begin
  pres2 = pres
  options = True
  options@boxWidth = .3
  options@boxColors = "white"
  if (isatt(pres2,"trXMinF")) then
     xcoordinate = pres2@trXMinF - 100   ; arbitrary, just move bar off of the plot
  else
     xcoordinate = -100.
  end if
  if (isatt(pres2,"tmXBLabels")) then
     delete(pres2@tmXBLabels)
  end if
  pres2@tmXBLabels = ""
  carr = new((/1,5/),float)
  carr(0,:) = (/-5,-4,-3,-2,-1/)
  plot = boxplot2(wks2,xcoordinate,carr,options,pres2,False)
  return(plot)
end
;-----------------------------------------------------------------------------------------
undef("weight_array_for_mme_calc")
function weight_array_for_mme_calc(enum[*]:integer,warr:numeric)
local enum,warr,farr,w_ind,factor
begin
  if (dimsizes(dimsizes(warr)).eq.1) then
     farr = warr(:max(enum)-1)
     farr = farr@_FillValue
     do gg = 1,max(enum)
        w_ind := ind(enum.eq.gg)
        factor = 1./dimsizes(w_ind)
        if (dimsizes(w_ind).eq.1) then
           farr(gg-1) = (/ warr(w_ind) /)
        else
           farr(gg-1) = (/ dim_sum_n(warr(w_ind)*factor,0) /)
        end if
     end do
  end if
  if (dimsizes(dimsizes(warr)).eq.2) then
     farr = warr(:max(enum)-1,:)
     farr = farr@_FillValue
     do gg = 1,max(enum)
        w_ind := ind(enum.eq.gg)
        factor = 1./dimsizes(w_ind)
        if (dimsizes(w_ind).eq.1) then
           farr(gg-1,:) = (/ warr(w_ind,:) /)
        else
           farr(gg-1,:) = (/ dim_sum_n(warr(w_ind,:)*factor,0) /)
        end if
     end do
  end if
  if (dimsizes(dimsizes(warr)).eq.3) then
     farr = warr(:max(enum)-1,:,:)
     farr = farr@_FillValue
     do gg = 1,max(enum)
        w_ind := ind(enum.eq.gg)
        factor = 1./dimsizes(w_ind)
        if (dimsizes(w_ind).eq.1) then
           farr(gg-1,:,:) = (/ warr(w_ind,:,:) /)
        else
           farr(gg-1,:,:) = (/ dim_sum_n(warr(w_ind,:,:)*factor,0) /)
        end if
     end do
  end if
  return(farr)
end
;=================================================================================================
; create blank array for use when something may be/is wrong.
;
undef("obs_percentile")
procedure obs_percentile(prstring:string,modval[*]:numeric,obsval[*]:numeric)
local prstring,modval,obsval,arrsort,finstring,uiuc,finchar
begin
  finstring = ""
  do rr = 0,dimsizes(obsval)-1
     arrsort = new((/dimsizes(modval)+1/),typeof(modval))
     arrsort(:dimsizes(modval)-1) = (/ modval /)
     arrsort(dimsizes(modval)) = (/ obsval(rr) /)
     qsort(arrsort)
     uiuc := ind(arrsort.eq.obsval(rr))*1.
     finstring = finstring+obsval(rr)+" ("+((uiuc(0)/dimsizes(ind(.not.ismissing(modval))))*100.)+"), "
     delete(arrsort)
  end do
  finchar = tochar(finstring)
  print(prstring+" Observed values (percentiles) = "+tostring(finchar(:dimsizes(finchar)-3)))
end





