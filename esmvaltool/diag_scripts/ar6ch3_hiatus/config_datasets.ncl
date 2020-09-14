; blend_globalmean
; ############################################################################
; Author: Yu Kosaka (RCAST, U. Tokyo, Japan)
; IPCC AR6 Chapter 3
; ############################################################################
; Description
;
;    Outputs: 
;
; History
;    20200531 kosaka_yu: written.
;
; Required diag_script_info attributes (diagnostic specific)
;    none
;
; Optional diag_script_info attributes (diagnostic specific)
;
; ############################################################################

begin
  obs_projects = (/"obs", "OBS", "OBS6", "obs4mips", "ana4mips"/)
  reanalyses = (/"ERA5", "ERA-Interim", "JRA-55", "JRA-25", "CFSR", \
                 "MERRA", "MERRA2", "NCEP", "NCEP2", "ERA-20C", "20CR"/)
end
