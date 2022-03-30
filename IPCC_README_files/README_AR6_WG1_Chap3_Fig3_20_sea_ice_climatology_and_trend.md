
SEA ICE CLIMATOLOGY AND TREND
============

Figure number: Figure 3.20
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.20](../images/ar6_wg1_chap3_fig3_20_sea_ice_climatology_and_trend.png?raw=true)

Description:
------------
It is a scatter plot, there are four panels (2x2). The upper row has CMIP5 data, the lower one shows 
CMIP6 data. The left column depicts Arctic sea ice area (SIA) for September, while the right one shows 
Antarctic SIA in February. 

Caption: Mean (x-axis) and trend (y-axis) in Arctic sea ice area (SIA) in September (left) and 
Antarctic SIA in February (right) for 1979-2017 from CMIP5 (upper) and CMIP6 (lower) models. 
All individual models (ensemble means) and the multi-model mean values are compared with the
observations (OSISAF, NASA Team, and Bootstrap). Solid line indicates a linear regression 
slope with corresponding correlation coefficient (r) and p-value provided. Note the 
different scales used on the y-axis for Arctic and Antarctic SIA. Results remain essentially
the same when using sea ice extent (SIE).

Author list:
------------
- Elizaveta Malinina: ECCC, Canada, elizaveta.malinina-rieger@canada.ca, githubid: malininae 
- Seung-Ki Min: Pohan University of Science and Technology, Korea
- Yeon-Hee Kim: Pohan University of Science and Technology, Korea
- Nathan Gillett: ECCC, Canada

ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: [ar6_chapter3_cryo_xcb32](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter3_cryo_xcb32/)

ESMValCore Branch:
------------------
**NB!**: Technically a new repository for ESMValCore wasn't created, just the custom variables were
added into the esmvalcore distributed with conda.  

These tables were added to **anaconda3/envs/esmvaltool/lib/python3.8/site-packages/esmvalcore/cmor/tables/custom/**
 
The [siarean](../esmvalcore_custom_variables/CMOR_siarean.dat) and [siareas](../esmvalcore_custom_variables/CMOR_siareas.dat)
tables needed for running this diagnostic are located in this repository.

Recipe & diagnostics:
---------------------
Recipe used: [recipes/recipe_ipcc_ar6_wg1_fgd_sea_ice_joint.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter3_cryo_xcb32/esmvaltool/recipes/recipe_ipcc_ar6_wg1_fgd_sea_ice_joint.yml)

Diagnostic used: [seaice/seaice/sie_ipcc_ar6_wg1_fgd_3_18.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter3_cryo_xcb32/esmvaltool/diag_scripts/seaice/seaice/sie_ipcc_ar6_wg1_fgd_3_18.py)

Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_ipcc_ar6_wg1_fgd_sea_ice_joint_YYYYMMDD_HHMMSS/plots/ipcc_ar6_wg1_fgd_3_18/ipcc_ar6_wg1_fgd_3_18/fig_3_18_scatter.png 

Recipe generations tools: 
-------------------------
N/A 

Additional datasets:
--------------------
There were three additional datasets used downloaded at https://www.fdr.uni-hamburg.de/record/8559#.YG5C5ehKg2w .
Following files should be used: SeaIceArea__NorthernHemisphere__monthly__UHH__v2019_fv0.01.nc,
SeaIceArea__SouthernHemisphere__monthly__UHH__v2019_fv0.01.nc. The data is distributed under
Creative Commons Attribution 4.0 International. 

These datasets were cmorized by cmorizers/obs/cmorize_obs_uhh_sia.py. Since the original dataset file 
contains three products, the results are stored in cmorize_obs_YYYYMMDD_HHMMSS/Tier2/UHH-SIA/$dataset_name$
The $dataset_name$ folder should be moved to the observational dataset folder under OBS/Tier2/$dataset_name$ 
skipping the UHH-SIA level.

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
at 16:40:18 UTC. Some minor "cosmetic" edits were done on Liza's dell laptop the same day.