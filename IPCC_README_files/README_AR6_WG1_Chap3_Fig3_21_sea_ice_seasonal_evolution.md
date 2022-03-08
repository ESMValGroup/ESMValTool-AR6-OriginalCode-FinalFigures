
SEA ICE SEASONAL EVOLUTION
============

Figure number: Figure 3.21
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.21](../images/ar6_wg1_chap3_fig3_21_sea_ice_seasonal_evolution.png?raw=true)

Description:
------------
This is a mesh plot with six panels (3 rows, 2 columns). The left column shows Arctic Sea Ice Area
(SIA), the right one depicts Antarctic SIA. The upper row shows observations, the middle is a plot
for natural and anthropogenic model simulations, and the lower panel shows model simulations with
natural forcing. 

Caption: Seasonal evolution of observed and simulated Arctic (left) and Antarctic (right) sea ice area
(SIA) over 1979–2017. SIA anomalies relative to the 1979–2000 means from observations 
(OBS from OSISAF, NASA Team, and Bootstrap, top) and historical (ALL, middle) and natural 
only (NAT, bottom) simulations from CMIP5 and CMIP6 multi-models. These anomalies were 
obtained by computing non-overlapping 3-year mean SIA anomalies for March (February for 
Antarctic SIA), June, September, and December separately. CMIP5 historical simulations are 
extended by using RCP4.5 scenario simulations after 2005 while CMIP6 historical simulations 
are extended by using SSP2-4.5 scenario simulations after 2014. CMIP5 NAT simulations end 
in 2012. Numbers in brackets represents the number of models used. The multi-model mean is 
obtained by taking the ensemble mean for each model first and then averaging over models. 
Grey dots indicate multi-model mean anomalies stronger than inter-model spread (beyond ± 1 
standard deviation). Results remain very similar when based on sea ice extent (SIE). 
Units: 106 km2. 

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

Diagnostic used: [seaice/seaice/sie_ipcc_ar6_wg1_fgd_3_19.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter3_cryo_xcb32/esmvaltool/diag_scripts/seaice/seaice/sie_ipcc_ar6_wg1_fgd_3_19.py)

Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
recipe_ipcc_ar6_wg1_fgd_sea_ice_joint_YYYYMMDD_HHMMSS/plots/ipcc_ar6_wg1_fgd_3_19/ipcc_ar6_wg1_fgd_3_19/fig_3_19_timeseries.pdf

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