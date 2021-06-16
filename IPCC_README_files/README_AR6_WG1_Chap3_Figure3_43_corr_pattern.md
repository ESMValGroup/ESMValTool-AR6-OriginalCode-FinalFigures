
CORRELATION PATTERN
==========================

Figure number: 3.43
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.43](../images/ar6_wg1_chap3_figure3_43_corr_pattern.png?raw=true)


Description:
------------
Centred pattern correlations between models and observations for the annual mean 
climatology over the period 1980–1999. Results are shown for individual CMIP3 
(cyan), CMIP5 (blue) and CMIP6 (red) models (one ensemble member from each model
is used) as short lines, along with the corresponding ensemble averages (long 
lines). Correlations are shown between the models and the primary reference 
observational data set (from left to right: ERA5, GPCP-SG, CERES-EBAF, 
CERES-EBAF, CERES-EBAF, CERES-EBAF, JRA-55, ESACCI-SST, ERA5, ERA5, ERA5, ERA5, 
ERA5, ERA5, AIRS, ERA5). In addition, the correlation between the primary 
reference and additional observational data sets (from left to right: NCEP, GHCN, 
-, -, -, -, ERA5, HadISST, NCEP, NCEP, NCEP, NCEP, NCEP, NCEP, ERA5, NCEP) are 
shown (solid grey circles) if available. To ensure a fair comparison across a 
range of model resolutions, the pattern correlations are computed after 
regridding all datasets to a resolution of 4º in longitude and 5º in latitude. 


Author list:
------------
- Bock, L.: DLR, Germany; lisa.bock@dlr.de
- Eyring, V.: DLR., Germany
- Morgenstern, O.: NIWA, New Zealand


Publication sources:
--------------------
Bock, L., Lauer, A., Schlund, M., Barreiro, M., Bellouin, N., Jones, C., Predoi, V., Meehl, G., Roberts, M., and Eyring, V.: Quantifying progress across different CMIP phases with the ESMValTool, Journal of Geophysical Research: Atmospheres, 125, e2019JD032321. https://doi.org/10.1029/2019JD032321


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter_3


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: fix_cmip6_models_newcore


Recipe & diagnostics:
---------------------
Recipes used: 
- recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_fig_3_43.yml

Diagnostics used: 
- diag_scripts/ipcc_ar6/corr_pattern.ncl
- diag_scripts/ipcc_ar6/corr_pattern_collect.ncl


Expected image path:
--------------------
- recipe_ipccwg1ar6ch3_fig_3_43_YYYYMMDD_HHMMSS/plots/fig_3_43/cor_collect/patterncor.pdf


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/ar6_newcore_lisa_conda_environment.yml
- pip file: IPCC_environments/ar6_newcore_lisa_pip_environment.txt


Hardware description:
---------------------
Machine used: Mistral
