
SEA SURFACE TEMPERATURE AND SALINITY - MODEL BIAS
=================================================

Figure number: 3.23
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.23](../images/ar6_wg1_chap3_figure3_23_ocean_model_bias.png?raw=true)


Description:
------------
Multi-model-mean bias of (a) sea surface temperature and (b) near-surface 
salinity, defined as the difference between the CMIP6 multi-model mean and the 
climatology from the World Ocean Atlas 2018. The CMIP6 multi-model mean is 
constructed with one realization of 46 CMIP6 historical experiments for the 
period 1995–2014 and the climatology from the World Ocean Atlas 2018 is an 
average over all available years (1955-2017). Uncertainty is represented using 
the advanced approach: No overlay indicates regions with robust signal, where 
≥66% of models show change greater than variability threshold and ≥80% of all 
models agree on sign of change; diagonal lines indicate regions with no change 
or no robust signal, where <66% of models show a change greater than the 
variability threshold; crossed lines indicate regions with conflicting signal, 
where ≥66% of models show change greater than variability threshold and <80% of 
all models agree on sign of change. For more information on the advanced 
approach, please refer to the Cross-Chapter Box Atlas.1.


Author list:
------------
- Bock, L.: DLR, Germany; lisa.bock@dlr.de
- Barimalala, R.: University of Cape Town, South Africa
- Durack, P.: Lawrence Livermore National Laboratory, USA


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: [ar6_chapter_3](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter_3)


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: [fix_cmip6_models_newcore](https://github.com/ESMValGroup/ESMValCore-AR6-OriginalCode-FinalFigures/tree/fix_cmip6_models_newcore)


Recipe & diagnostics:
---------------------
Recipe used: [recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_ocean.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3/esmvaltool/recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_ocean.yml)

Diagnostics used: 
- [diag_scripts/ipcc_ar5/ch12_calc_IAV_for_stippandhatch.ncl](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3/esmvaltool/diag_scripts/ipcc_ar5/ch12_calc_IAV_for_stippandhatch.ncl)
- [diag_scripts/ipcc_ar6/model_bias.ncl](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3/esmvaltool/diag_scripts/ipcc_ar6/model_bias.ncl)


Expected image path:
--------------------
- recipe_ipccwg1ar6ch3_ocean_YYYYMMDD_HHMMSS/plots/fig_3_22_sos/fig_3_22/model_bias_sos_annualclim_CMIP6.eps
- recipe_ipccwg1ar6ch3_ocean_YYYYMMDD_HHMMSS/plots/fig_3_22_tos/fig_3_22/model_bias_tos_annualclim_CMIP6.eps


Software description:
---------------------
- ESMValTool environment file: [IPCC_environments/ar6_newcore_lisa_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/fix_links/IPCC_environments/ar6_newcore_lisa_conda_environment.yml)
- pip file: [IPCC_environments/ar6_newcore_lisa_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/fix_links/IPCC_environments/ar6_newcore_lisa_pip_environment.txt)


Hardware description:
---------------------
Machine used: Mistral
