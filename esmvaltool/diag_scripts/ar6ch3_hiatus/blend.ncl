; blend.ncl
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs: 
;
; History
;    20190304 kosaka_yu: written.
;    20191201 kosaka_yu: modified to handle various blending methods.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

function blend_tas_tos(tas, tos, sic, sftlf, method)
local tas, tos, sic, sftlf, lf, lftmp, ts, SAT, SST, SIC, SFTLF, stime, etime, method
begin
  SAT = set_yyyymm(tas)
  SST = set_yyyymm(tos)
  SIC = set_yyyymm(sic)
  stime = max((/SAT&time(0), SST&time(0), SIC&time(0)/))
  etime = min((/SAT&time(dimsizes(SAT&time)-1), SST&time(dimsizes(SST&time)-1), SIC&time(dimsizes(SIC&time)-1)/))
  lf = sftlf
  if (any(method.eq.(/"HadCRUT4", "Cowtan2015"/))) then
    lftmp = new(array_append_record(2, dimsizes(lf), 0), typeof(lf))
    lftmp(0, :, :) = lf
    lftmp(1, :, :) = 0.25
    lf = where(lf.gt.1.e-5, dim_max_n(lftmp, 0), 0.)
    delete(lftmp)
  end if  
  SFTLF = conform(SAT({stime:etime}, :, :), lf, (/1, 2/))
  ts = SAT({stime:etime}, :, :)
  if (method.eq."HadCRUT4") then
    ts = SAT({stime:etime}, :, :)*SFTLF \
          + SST({stime:etime}, :, :)*(1. - SFTLF)
  else
    ts = SAT({stime:etime}, :, :)*(SFTLF + (1. - SFTLF)*SIC({stime:etime}, :, :)) \
          + SST({stime:etime}, :, :)*(1. - SFTLF)*(1.-SIC({stime:etime}, :, :))
  end if
  return(ts)
end

function apply_mask_ref(x, ref)
local x, y, ref
begin
  y = x
  y = where(ismissing(ref), x@_FillValue, x)
  return(y)
  delete(y)
end
