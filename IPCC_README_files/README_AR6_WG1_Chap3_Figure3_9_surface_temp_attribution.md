
ANOMALY OF NERA-SURFACE AIR TEMPERATURE - ATTRIBUTION
=====================================================

Figure number: Figure 3.9
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.9](../images/ar6_wg1_chap3_figure3_9_surface_temp_attribution.png?raw=true)


Description:
------------
Global, land, ocean and continental annual mean near-surface air temperatures
anomalies in CMIP6 models and observations. Timeseries are shown for CMIP6 
historical anthropogenic and natural (brown) natural-only (green), greenhouse 
gas only (grey) and aerosol only (blue) simulations (multi-model means shown as
thick lines, and shaded ranges between the 5th and 95th percentiles) and for 
HadCRUT5 (black). All models have been subsampled using the HadCRUT5 
observational data mask. Temperatures anomalies are shown relative to 1950-2010 
for Antarctica and relative to 1850–1900 for other continents. CMIP6 historical 
simulations are expand by the SSP2-4.5 scenario simulations. All available 
ensemble members were used (see Section 3.2). Regions are defined by 
Iturbide et al. (2020).


Author list:
------------
- Bock, L.: DLR, Germany; lisa.bock@dlr.de
- Gillett, N.: Environment and Climate Change Canada


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapteterr_3


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: fix_cmip6_models_newcore


Recipe & diagnostics:
---------------------
Recipe used: recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_atmosphere_fig_3_9.yml

Diagnostic used: diag_scripts/ipcc_ar6/tas_anom_damip.ncl


Expected image path:
--------------------
- recipe_ipccwg1ar6ch3_atmosphere_fig_3_9_YYYYMMDD_HHMMSS/plots/fig_3_9_tas_anom_damip_1/tsline/tas_anom_damip_global_1850-2020.eps
- recipe_ipccwg1ar6ch3_atmosphere_fig_3_9_YYYYMMDD_HHMMSS/plots/fig_3_9_tas_anom_damip_2/tsline/tas_anom_damip_america_europe_1850-2020.eps
- recipe_ipccwg1ar6ch3_atmosphere_fig_3_9_YYYYMMDD_HHMMSS/plots/fig_3_9_tas_anom_damip_3/tsline/tas_anom_damip_africa_asia_1850-2020.eps
- recipe_ipccwg1ar6ch3_atmosphere_fig_3_9_YYYYMMDD_HHMMSS/plots/fig_3_9_tas_anom_damip_4/tsline/tas_anom_damip_antarctica_1850-2020.eps


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/ar6_newcore_lisa_conda_environment.yml
- pip file: IPCC_environments/ar6_newcore_lisa_pip_environment.txt


Hardware description:
---------------------
What machine was used:  Mistral


Further instructions:
---------------------
Create map with shapefiles by using IPCC_additional_scripts/map.ncl
(-> shapefiles.png).
