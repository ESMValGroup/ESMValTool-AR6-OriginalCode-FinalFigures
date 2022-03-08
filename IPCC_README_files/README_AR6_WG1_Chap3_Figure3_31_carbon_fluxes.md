
Carbon fluxes in emission driven simulations
============

Figure number: Figure 3.31
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.31](../images/ar6_wg1_chap3_figure3_31_carbon_fluxes.png?raw=true)


Description:
------------
Evaluation of historical emission-driven CMIP6 simulations for 1850-2014. Observations (black) are compared to simulations of global mean (a) atmospheric CO2 concentration (ppmv), with observations from the National Oceanic and Atmospheric Administration Earth System Research Laboratory (NOAA ESRL) (Dlugokencky and Tans, 2020), (b) air surface temperature anomaly (°C) with respect to the 1850-1900 mean, with observations from HadCRUT4 (Morice et al., 2012) (c) land carbon uptake (PgC yr-1), (d) ocean carbon uptake (PgC yr-1), both with observations from the Global Carbon Project (GCP) (Friedlingstein et al., 2019) and grey shading indicating the observational uncertainty. Land and ocean carbon uptakes are plotted using a 10-year running mean for better visibility. The ocean uptake is offset to 0 in 1850 to correct for pre-industrial riverine-induced carbon fluxes.


Author list:
------------
- Gier, B.K.: Univ. of Bremen, Germany; gier@uni-bremen.de
- Bellouin, N.: University of Reading, UK
- Bock, L.: DLR, Germany

Publication sources:
--------------------
- Friedlingstein, P., Meinshausen, M., Arora, V. K., Jones, C. D., Anav, A., Liddicoat, S. K., & Knutti, R. (2014). Uncertainties in CMIP5 Climate Projections due to Carbon Cycle Feedbacks, Journal of Climate, 27(2), 511-526. https://doi.org/10.1175/JCLI-D-12-00579.1


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: [ar6_chapter_3_tina
](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter_3_tina)

ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: [gier_fixes](https://github.com/ESMValGroup/ESMValCore-AR6-OriginalCode-FinalFigures/tree/gier_fixes)


Recipe & diagnostics:
---------------------
Recipe(s) used: 
- [recipes/recipe_ipccar6_esmhist_timeseries.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3_tina/esmvaltool/recipes/recipe_ipccar6_esmhist_timeseries.yml)

Diagnostic(s) used:
- [diagnostics/ipcc_ar6/fig_carbonsinks_timeseries_panels.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3_tina/esmvaltool/diag_scripts/ipcc_ar6/fig_carbonsinks_timeseries_panels.py)


Expected image path:
--------------------
- recipe_ipccar6_esmhist_timeseries_YYYYMMDD_HHMMSS/plots/fig_3_30_co2/fig_3_30_co2/fig_ipcca6_3_31.png


Software description:
---------------------
- ESMValTool environment file: [IPCC_environments/env_ipcc_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/fix_links/IPCC_environments/env_ipcc_conda_environment.yml)
- pip file: [IPCC_environments/env_ipcc_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/fix_links/IPCC_environments/env_ipcc_pip_environment.txt)


Hardware description:
---------------------
Machine used: Mistral
