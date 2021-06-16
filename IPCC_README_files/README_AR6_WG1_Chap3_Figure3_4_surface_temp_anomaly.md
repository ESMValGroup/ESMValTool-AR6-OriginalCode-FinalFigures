
ANOMALY OF NERA-SURFACE AIR TEMPERATURE
=======================================

Figure number: Figure 3.4
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.4](../images/ar6_wg1_chap3_figure3_4_surface_temp_anomaly.png?raw=true)


Description:
------------
Observed and simulated time series of the anomalies in annual and global mean
near-surface air temperature (GSAT). All anomalies are differences from the 
1850–1900 time-mean of each individual time series. The reference period 1850–
1900 is indicated by grey shading. (a) Single simulations from CMIP6 models 
(thin lines) and the multi-model mean (thick red line). Observational data
(thick black lines) are HadCRUT5, and are blended surface temperature (2 m air
temperature over land and sea surface temperature over the ocean). All models 
have been subsampled using the HadCRUT5 observational data mask. Vertical lines
indicate large historical volcanic eruptions. CMIP6 models which are marked 
with an asterisk are either tuned to reproduce observed warming directly, or 
indirectly by tuning equilibrium climate sensitivity. Inset: GSAT for each 
model over the reference period, not masked to any observations. (b). Multi-
model means of CMIP5 (blue line) and CMIP6 (red line) ensembles and associated 
5 to 95 percentile ranges (shaded regions). Observational data are HadCRUT5, 
Berkeley Earth, NOAAGlobalTemp-Interim and Kadow et al. (2020). Masking was 
done as in (a). CMIP6 historical simulations are extended with SSP2-4.5 
simulations for the period 2015-2020 and CMIP5 simulations are extended with 
RCP4.5 simulations for the period 2006-2020. All available ensemble members were
used (see Section 3.2). The multi-model means and percentiles were calculated 
solely from simulations available for the whole time span (1850-2020).


Author list:
------------
- Bock, L.: DLR, Germany; lisa.bock@dlr.de
- Bellouin, N.: University of Reading, UK
- Eyring, V.: DLR., Germany
- Gillett, N.: Environment and Climate Change Canada


Publication sources:
--------------------
Bock, L., Lauer, A., Schlund, M., Barreiro, M., Bellouin, N., Jones, C., 
Predoi, V., Meehl, G., Roberts, M., and Eyring, V.: Quantifying progress 
across different CMIP phases with the ESMValTool, Journal of Geophysical 
Research: Atmospheres, 125, e2019JD032321. https://doi.org/10.1029/2019JD032321


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter_3


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: fix_cmip6_models_newcore


Recipe & diagnostics:
---------------------
Recipe used: recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_atmosphere.yml

Diagnostics used: 
- diag_scripts/ipcc_ar6/tas_anom.ncl
- diag_scripts/ipcc_ar6/tsline_blend_collect.ncl


Expected image path:
--------------------
- recipe_ipccwg1ar6ch3_atmosphere_YYYYMMDD_HHMMSS/plots/fig_3_4_cmip5/fig-3-4/gsat_Global_CMIP5_historical-rcp45_anom_1850-2020.eps
- recipe_ipccwg1ar6ch3_atmosphere_YYYYMMDD_HHMMSS/plots/fig_3_4_cmip6/fig-3-4/gsat_Global_CMIP6_historical-ssp245_anom_1850-2020.eps
- recipe_ipccwg1ar6ch3_atmosphere_YYYYMMDD_HHMMSS/plots/fig_3_4_collect/collect/gsat_Global_multimodel_anom_1850-2020.eps


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/ar6_newcore_lisa_conda_environment.yml
- pip file: IPCC_environments/ar6_newcore_lisa_pip_environment.txt


Hardware description:
---------------------
What machine was used:  Mistral
