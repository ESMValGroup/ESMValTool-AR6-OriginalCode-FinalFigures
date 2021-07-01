
SPEED-UP Of ZONAL MEAN WIND
===========================

Figure number: 3.19
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.19](../images/ar6_wg1_chap3_figure3_19_zonal_wind_trends.png?raw=true)


Description:
------------
Long-term mean (thin black contour) and linear trend (colour) of zonal mean DJF 
zonal winds over 1985-2014 in the SH. Displayed are (a) ERA5 and (b) CMIP6 
multi-model mean (58 CMIP6 models). Only one ensemble member per model is 
included.


Author list:
------------
- Bock, L.: DLR, Germany; lisa.bock@dlr.de
- Cionni, I.: ENEA, Italy
- Hassler, B.: DLR, Germany
- Eyring, V.: DLR, Germany


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter_3


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: fix_cmip6_models_newcore


Recipe & diagnostics:
---------------------
Recipe used: recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_fig_3_19.yml

Diagnostic used: diag_scripts/ipcc_ar6/zonal_westerly_winds.ncl


Expected image path:
--------------------
- recipe_ipccwg1ar6ch3_fig_3_19_YYYYMMDD_HHMMSS/plots/fig_3_19/clim/zonal_westerly_winds.eps


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/ar6_newcore_lisa_conda_environment.yml
- pip file: IPCC_environments/ar6_newcore_lisa_pip_environment.txt


Hardware description:
---------------------
Machine used: Mistral
