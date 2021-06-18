
BLOCKING
========

Figure number: Figure 3.18
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.18](../images/ar6_wg1_chap3_figure3_18_blocking.png?raw=true)


Description:
------------
Instantaneous Northern-Hemisphere blocking frequency (% of days) in the extended
northern winter season (DJFM) for the years 1979-2000. Results are shown for 
ERA5 reanalysis (black), CMIP5 (blue) and CMIP6 (red) models. Coloured lines 
show multi-model means and shaded ranges show corresponding 5-95% constructed 
with one realization from each model Figure is adapted from Davini and D’Andrea 
(2020), their Figure 12 and following the D’Andrea et al. (1998) definition of 
blocking.


Author list:
------------
- Kazeroni, R.: DLR, Germany; remi.kazeroni@dlr.de
- Davini, A.: CNR-ISAC, Italy
- Bock, L.: DLR, Germany
- Eyring, V.: DLR, Germany
- Morgenstern, O.: NIWA, New Zealand


Publication sources:
--------------------
Davini, P., and D’Andrea, F. (2020). From CMIP3 to CMIP6: Northern hemisphere 
atmospheric blocking simulation in present and future climate. J. Clim. 33, 
10021–10038. doi:10.1175/JCLI-D-19-0862.1.


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter_3


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: fix_cmip6_models_newcore


Recipe & diagnostics:
---------------------
Recipe used: recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_blocking.yml

Diagnostic used: diag_scripts/miles/miles_block_groupby_projects.R


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_miles_block_groupby_extended_DJFM_20210301_151120/plots/miles_diagnostics/miles_block/Multimodel/historical/1979-2000/DJFM/Block/DA98_Multimodel_historical_1979-2000_DJFM.png


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/ar6_newcore_remi_conda_environment.yml
- pip file: e.g. IPCC_environments/ar6_newcore_remi_pip_environment.txt


Hardware description:
---------------------
What machine was used: Mistral