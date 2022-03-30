
TEMPERATURE AND PRECIPITATION EXTREMES
============

Figure number: CCB3.2 Figure 1
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![CCB 3.2 Figure 1](../images/ar6_wg1_chap3_ccb3_2_fig1_temperature_and_precipitations_extremes.png?raw=true)


Description:
------------
The figure has four panels (2x2). Left column shows anomalies of annual maximum daily maximum 
temperature (TXx), the right one shows annual maximum 1-day precipitation (Rx1day). The upper row
shows natural and human forcing simulations, while the lower row shows only natural forcing simulations. 

Comparison of observed and simulated changes in global mean temperature and precipitation extremes. 
Time series of globally averaged 5-year mean anomalies of the annual maximum daily maximum 
temperature (TXx in °C) and annual maximum 1-day precipitation (Rx1day as standardized probability
index in %) during 1953-2017 from the HadEX3 observations and the CMIP5 and CMIP6 multi-model 
ensembles with natural and human forcing (upper) and natural forcing only (lower). For CMIP5, 
historical simulations for 1953-2005 are combined with corresponding RCP4.5 scenario runs for 
2006-2017. For CMIP6, historical simulations for 1953-2014 are combined with SSP2-4.5 scenario 
simulations for 2015-2017. Numbers in brackets represents the number of models used. The time-fixed
observational mask has been applied to model data throughout the whole period. Grid cells with more
than 70% data availability during 1953-2017 plus data for at least 3 years during 2013-2017 are 
used. Coloured lines indicate multi-model means, while shading represents 5th-95th percentile 
ranges, based on all available ensemble members with equal weight given to each model 
(Section 3.2). Anomalies are relative to 1961-1990 means. Figure is updated from 
Seong et al. (2021), their Figure 3 and Paik et al. (2020), their Figure 3.

Author list:
------------
- Elizaveta Malinina: ECCC, Canada, elizaveta.malinina-rieger@canada.ca, githubid: malininae 
- Seung-Ki Min: Pohan University of Science and Technology, Korea
- Ying Sun: China Meteorological Administration, China
- Nathan Gillett: ECCC, Canada
- Krishnan Raghavan: Indian Institute of Tropical Meteorology, India
- Xuebing Zhang, ECCC, Canada

Publication sources:
--------------------
- Determining the Anthropogenic Greenhouse Gas Contribution to the Observed Intensification of 
  Extreme Precipitation, S. Paik et al, Geophysical Research Letters 47.12, e2019GL086875,
  https://doi.org/10.1029/2019GL086875, 2020.
- Anthropogenic Greenhouse Gas and Aerosol Contributions to Extreme Temperature Changes during 
  1951–2015, Seong M.-G. et al, Journal of Climate, 34(3), 857-870, 
  https://doi.org/10.1175/JCLI-D-19-1023.1, 2021.

ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: [ar6_chapter3_cryo_xcb32](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter3_cryo_xcb32/)

ESMValCore Branch:
------------------
**NB!**: Technically a new repository for ESMValCore wasn't created, just the custom variables were
added into the esmvalcore distributed with conda.  

These tables were added to **anaconda3/envs/esmvaltool/lib/python3.8/site-packages/esmvalcore/cmor/tables/custom/**
 
The [txx](../esmvalcore_custom_variables/CMOR_txx.dat) and [rx1day](../esmvalcore_custom_variables/CMOR_rx1day.dat) 
tables are in this repository. 

Recipe & diagnostics:
---------------------
Recipe used: [recipes/recipe_ipcc_ar6_wg1_fgd_xcb_3_2.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter3_cryo_xcb32/esmvaltool/recipes/recipe_ipcc_ar6_wg1_fgd_xcb_3_2.yml)

Diagnostic used: [diag_scripts/extreme_events/extremes_ipcc_ar6_wg1_fgd_xcb_32.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter3_cryo_xcb32/esmvaltool/diag_scripts/extreme_events/extremes_ipcc_ar6_wg1_fgd_xcb_32.py)

Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_ipcc_ar6_wg1_fgd_xcb_3_2_YYYYMMDD_HHMMSS/plots/xcbox32/xcbox32/figure_xcb32.png 

Recipe generations tools: 
-------------------------
N/A

Ancillary figures and datasets:
-------------------------------
The diagnostic runs very-very long (10-ish), the reason for this is calculation of probability 
distribution function for precipitaton extreme probability for each grid cell. For this reason, 
the calculated cdfs are saved into ntcdf files for each model into the intrenal esmvaltool run 
directory into rx1day_cdfs/cdf_$project$_$experiment$_$model$.nc

Additional datasets:
--------------------
The HadEX3 data for TXx and Rx1day were downloaded from https://www.metoffice.gov.uk/hadobs/hadex3/download.html
and cmorized with cmorizers/obs/cmorize_obs_hadex3.py in the above mentioned repository. The data 
is provided under Open Goevrnment Licence http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

Software description:
---------------------
The software from Mistral:
- ESMValTool environment file: [IPCC_environments/mistral_cryo_and_xcb32_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/mistral_cryo_and_xcb32_conda_environment.yml)
- pip file: [IPCC_environments/mistral_cryo_and_xcb32_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/mistral_cryo_and_xcb32_pip_environment.txt)

The software from Liza's computer:
- ESMValTool environment file: [IPCC_environments/liza_dell_computer_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/liza_dell_computer_conda_environment.yml)
- pip file: [IPCC_environments/liza_dell_computer_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/liza_dell_computer_pip_environment.txt)


Hardware description:
---------------------
The data was processed on Mistral, the final version was pre-processed on the 5th of March 2021 
at 00:21:33 UTC. Then the cdfs were calculated the following days. Some minor "cosmetic" edits 
were done on Liza's dell laptop on the 10th of March 2021.
