
Relative Amplitude Changes for Carbon Land-Atmosphere Flux
============

Figure number: Figure 3.32
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.32](../images/ar6_wg1_chap3_figure3_32_nbp_amplitude.png?raw=true)


Description:
------------
Relative change in the amplitude of the seasonal cycle of global land carbon uptake in the historical CMIP6 simulations from 1961-2014. Net biosphere production estimates from 19 CMIP6 models (red), the data-led reconstruction JMA-TRANSCOM (Maki et al., 2010; dotted) and atmospheric CO2 seasonal cycle amplitude changes from observations (global as dashed line, Mauna Loa Observatory (MLO) (Dlugokencky et al., 2020) in bold black). Seasonal cycle amplitude is calculated using the curve fit algorithm package (https://www.esrl.noaa.gov/gmd/ccgg/mbl/crvfit/crvfit.html) from the National Oceanic and Atmospheric Administration Earth System Research Laboratory (NOAA ESRL). Relative changes are referenced to the 1961-1970 mean and for short time series adjusted to have the same mean as the model ensemble in the last 10 years. Interannual variation was removed with a 9-year Gaussian smoothing. Shaded areas show the one sigma model spread (grey) for the CMIP6 ensemble and the one sigma standard deviation of the smoothing (red) for the CO2 MLO observations. Inset: average seasonal cycle of ensemble mean net biosphere production and its one sigma model spread for 1961-1970 (orange dashed line, light orange shading) and 2005-2014 (solid green line, green shading).


Author list:
------------
- Gier, B.K.: Univ. of Bremen, Germany; gier@uni-bremen.de
- Bellouin, N.: University of Reading, UK
- Bock, L.: DLR, Germany

Publication sources:
--------------------
- Zhao, F., Zeng, N., Asrar, G., Friedlingstein, P., Ito, A., Jain, A., Kalnay, E., Kato, E., Koven, C. D., Poulter, B., Rafique, R., Sitch, S., Shu, S., Stocker, B., Viovy, N., Wiltshire, A., and Zaehle, S.: Role of CO2, climate and land use in regulating the seasonal amplitude increase of carbon fluxes in terrestrial ecosystems: a multimodel analysis, Biogeosciences, 13, 5121–5137, https://doi.org/10.5194/bg-13-5121-2016, 2016. 


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter_3_tina


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: gier_fixes


Recipe & diagnostics:
---------------------
Recipe(s) used: 
- recipes/recipe_ipcc6_co2_sca.yml

Diagnostic(s) used:
- diagnostics/ipcc_ar6/CO2_SCA_trend.py


Expected image path:
--------------------
- recipe_ipcc6_esmhist_timeseries_YYYYMMDD_HHMMSS/plots/fig_3_31_sca/fig_3_31_sca/fig_ipccar6_3_32.png


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/env_ipcc_conda_environment.yml
- pip file: IPCC_environments/env_ipcc_pip_environment.txt


Hardware description:
---------------------
Machine used: Mistral

