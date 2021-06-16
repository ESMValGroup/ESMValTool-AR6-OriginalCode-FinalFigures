
RELATIVE MODEL PERFORMANCE
==========================

Figure number: 3.42
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.42](../images/ar6_wg1_chap3_figure3_42_perfmetrics.png?raw=true)


Description:
------------
Relative space-time root-mean-square deviation (RMSD) calculated from the 
climatological seasonal cycle of the CMIP simulations (1980-1999) compared to 
observational datasets. (a) CMIP3, CMIP5, and CMIP6 for 16 atmospheric variables
(b) CMIP5 and CMIP6 for 10 land variables and four ocean/sea-ice variables. A 
relative performance measure is displayed, with blue shading indicating better 
and red shading indicating worse performance than the median of all model 
results. A diagonal split of a grid square shows the relative error with respect 
to the reference data set (lower right triangle) and an additional data set 
(upper left triangle). Reference/additional datasets are from top to bottom in 
(a): ERA5/NCEP, GPCP-SG/GHCN, CERES-EBAF/-, CERES-EBAF/-, CERES-EBAF/-, 
CERES-EBAF/-, JRA-55/ERA5, ESACCI-SST/HadISST, ERA5/NCEP, ERA5/NCEP, ERA5/NCEP, 
ERA5/NCEP, ERA5/NCEP, ERA5/NCEP, AIRS/ERA5, ERA5/NCEP and in (b): CERES-EBAF/-, 
CERES-EBAF/-, CERES-EBAF/-, CERES-EBAF/-, LandFlux-EVAL/-, Landschuetzer2016/ 
JMA-TRANSCOM; MTE/FLUXCOM, LAI3g/-, JMA-TRANSCOM, ESACCI-SOILMOISTURE/-, 
HadISST/ATSR, HadISST/-, HadISST/-, ERA-Interim/-. White boxes are used when 
data are not available for a given model and variable.


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
- recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_fig_3_42_a.yml
- recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_fig_3_42_b.yml

Diagnostics used: 
- diag_scripts/perfmetrics/main.ncl
- diag_scripts/perfmetrics/cycle_latlon.ncl
- diag_scripts/perfmetrics/collect.ncl


Expected image path:
--------------------
- recipe_ipccwg1ar6ch3_fig_3_42_a_YYYYMMDD_HHMMSS/plots/collect/RMSD/hus400-global_to_tas-global_RMSD.pdf
- recipe_ipccwg1ar6ch3_fig_3_42_b_YYYYMMDD_HHMMSS/plots/collect/RMSD/hfds-global_to_rlus-global_RMSD.pdf


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/ar6_newcore_lisa_conda_environment.yml
- pip file: IPCC_environments/ar6_newcore_lisa_pip_environment.txt


Hardware description:
---------------------
Machine used: Mistral
