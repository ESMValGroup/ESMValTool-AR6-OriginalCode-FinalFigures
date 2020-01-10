; percentiles
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Calculates percentile statistics, following Matlab "prctile" function
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

;function percentile_rank(x, x0)
;local X, i, imax, rank, x, x0
;begin
;
;end

; ==========================================================================
; percentile_val(x, x0): Obtain (x0 x 100)-th percentile value in an array x
;  x:  numeric, a 1-D array
;  x0: float or double, percentile in [0, 1]
; --------------------------------------------------------------------------
function percentile_val(x, x0)
local X, imax, i, val, x, x0
begin
  if (any(ismissing(x))) then
    X = x(ind(.not.ismissing(x)))
  else
    X = x
  end if
  qsort(X)
  imax = tofloat(dimsizes(X))
  if (x0.lt.0.5/imax) then
    val = X(0)
  elseif (x0.gt.(imax - 0.5)/imax) then
    val = X(toint(imax)-1)
  else
    i = x0*imax + 0.5
    val = X(toint(floor(i))) + (X(toint(floor(i))+1) - X(toint(floor(i))))*(i - floor(i))
  end if
  return(val)
end
