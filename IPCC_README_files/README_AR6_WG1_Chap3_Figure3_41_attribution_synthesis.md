
ATTRIBUTION SYNTHESIS
=====================

Figure number: Figure 3.41
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.41](../images/ar6_wg1_chap3_figure3_41_attribution_synthesis.png?raw=true)


Description:
------------
Summary figure showing simulated and observed changes in key large-scale 
indicators of climate change across the climate system, for continental, ocean 
basin and larger scales. Black lines show observations, brown lines and shading 
show the multi-model mean and 5-95th percentile ranges for CMIP6 historical 
simulations including anthropogenic and natural forcing, and blue lines and 
shading show corresponding ensemble means and 5-95th percentile ranges for CMIP6 
natural-only simulations. Temperature timeseries are as in Figure 3.9, but with 
smoothing using a low pass filter. Precipitation timeseries are as in Figure 
3.15 and ocean heat content as in Figure 3.26.


Author list:
------------
- Bock, L.: DLR, Germany; lisa.bock@dlr.de
- Gillett, N.: Environment and Climate Change Canada


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: [ar6_chapter_3](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter_3)


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: [fix_cmip6_models_newcore](https://github.com/ESMValGroup/ESMValCore-AR6-OriginalCode-FinalFigures/tree/fix_cmip6_models_newcore)


Recipe & diagnostics:
---------------------
Recipes used: 
- [recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_atmosphere_fig_3_9.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3/esmvaltool/recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_atmosphere_fig_3_9.yml)
- [recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_fig_3_41_pr.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3/esmvaltool/recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_fig_3_41_pr.yml)
- [recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_fig_3_41_siconc.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3/esmvaltool/recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_fig_3_41_siconc.yml)

Diagnostic used: [diag_scripts/ipcc_ar6/tas_anom_damip.ncl](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3/esmvaltool/diag_scripts/ipcc_ar6/tas_anom_damip.ncl)


Expected image path:
--------------------
- Fig_3_41.pdf (See the section Additional instructions below.)


Additional datasets:
--------------------
received from Lee De Mora (CA in Chapter 3, ledm@pml.ac.uk):
- CMIP6_*_historical_*_detrended_total.nc
- CMIP6_*_hist-nat_*_detrended_total.nc

received from Chapter 2:
- AR6_FGD_assessment_timeseries_OHC.csv


Software description:
---------------------
- ESMValTool environment file: [IPCC_environments/ar6_newcore_lisa_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/ar6_newcore_lisa_conda_environment.yml)
- pip file: [IPCC_environments/ar6_newcore_lisa_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/ar6_newcore_lisa_pip_environment.txt)


Hardware description:
---------------------
Machine used:  Mistral


Additional instructions:
------------------------

1. Run the three ESMValTool recipes listed above to generate the nc-files: 
   tas_anom_damip_*.nc; precip_anom_*.nc; tsline_collect_siconc_*.nc
2. Run [IPCC_additional_scripts/Fig_3_41_ohc.ncl](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_additional_scripts/Fig_3_41_ohc.ncl) to generate the nc-file: 
   ohc_damip.nc (input are ohc-files listed under addtional datasets)
3. Run [IPCC_additional_scripts/Fig_3_41_collect.ncl](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_additional_scripts/Fig_3_41_collect.ncl) to create the final figure: 
   Fig_3_41.pdf
