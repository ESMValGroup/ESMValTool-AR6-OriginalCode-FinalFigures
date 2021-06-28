
FIGURE TITLE
============

Figure number: Figure 3.22
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.22](../images/ar6_wg1_chap3_fig3_22_snow_anomalies.png?raw=true)

Description:
------------
The figure consists of two panels, upper one has information on CMIP5, lower on CMIP6. It is a multi
line plot with 5-95 percentile shadings on the model data.

Caption: Time series of Northern Hemisphere March-April mean snow cover extent (SCE) from 
observations, CMIP5 and CMIP6 simulations. The observations (grey lines) are updated Brown-NOAA 
(Brown and Robinson, 2011), Mudryk et al. (2020), and GLDAS2. CMIP5 (upper) and CMIP6 (lower)
simulations of the response to natural plus anthropogenic forcing are shown in orange, natural
forcing only in green, and the pre-industrial control simulation range is presented in blue. 
5-year mean anomalies are shown for the 1923-2017 period with the x-axis representing the centre
years of each 5-year mean. CMIP5 all forcing simulations are extended by using RCP4.5 scenario 
simulations after 2005 while CMIP6 all forcing simulations are extended by using SSP2-4.5 scenario
simulations after 2014. Shading indicates 5th-95th percentile ranges for CMIP5 and CMIP6 all and 
natural forcings simulations, and solid lines are ensemble means, based on all available ensemble 
members with equal weight given to each model (Section 3.2). The blue vertical bar indicates the
mean 5th-95th percentile range of pre-industrial control simulation anomalies, based on 
non-overlapping segments. The numbers in brackets indicate the number of models used. Anomalies 
are relative to the average over 1971-2000. For models, SCE is restricted to grid cells with land 
fraction ≥ 50%. Greenland is excluded from the total area summation. Figure is modified from 
Paik et al. (2020), their Figure 1.

Author list:
------------
- Elizaveta Malinina: ECCC, Canada, elizaveta.malinina-rieger@canada.ca, githubid: malininae 
- Seung-Ki Min: Pohan University of Science and Technology, Korea
- Olaf Morgenstern: NIWA, New Zeeland
- Seungmok Paik: Pohan University of Science and Technology, Korea
- Nathan Gillett: ECCC, Canada

Publication sources:
--------------------
- Quantifying the Anthropogenic Greenhouse Gas Contribution to the Observed Spring Snow-Cover Decline 
  Using the CMIP6 Multimodel Ensemble, S. Paik and S.-K. Min,  Journal of Climate 33.21: 9261-9269, 
  DOI:564, 2020.

ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter3_cryo_xcb32

ESMValCore Branch:
------------------
**NB!**: Technically a new repository for ESMValCore wasn't created, just the custom variables were
added into the esmvalcore distributed with conda.  

These tables were added to **anaconda3/envs/esmvaltool/lib/python3.8/site-packages/esmvalcore/cmor/tables/custom/**
 
The [scen](../esmvalcore_custom_variables/CMOR_scen.dat) tables needed for running this diagnostic
is located in this repository.

Recipe & diagnostics:
---------------------
Recipe used: recipes/recipe_ipcc_ar6_wg1_fgd_3_20.yml

Diagnostic used: seaice/sie_ipcc_ar6_wg1_fgd_3_20.py

Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_ipcc_ar6_wg1_fgd_3_20_YYYYMMDD_HHMMSS/plots/ipcc_ar6_wg1_fgd_3_20/ipcc_ar6_wg1_fgd_3_20/fig_3_20_timeseries.png

Recipe generations tools: 
-------------------------
N/A

Additional datasets:
--------------------
Following datasets were downloaded and cmorized using cmorizers from above mentioned directory: 
- **BR2011**, source: http://www.the-cryosphere.net/5/219/2011/ tc-5-219-2011-supplement.zip, cmorizer: cmorizers/obs/cmorize_obs_br2011.py
- **NOAA_CDR**, source: https://climate.rutgers.edu/snowcover/table_area.php?ui_set=2, cmorizer: cmorizers/obs/cmorize_obs_noaa_cdr.py
- **Mudryk2020**, source: http://data.ec.gc.ca/data/climate/scientificknowledge/climate-research-publication-based-data/northern-hemisphere-blended-snow-extent-and-snow-mass-time-series/SCE_timeseries.nc , 
cmorizer: cmorizers/obs/cmorize_obs_mudryk2020.py
- **GLDAS_NOAH**, source: https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH10_M.2.0, 
  cmorizer: cmorizers/obs/cmorize_obs_gldas_noah.py

Also, the "official" Greenland shape from general IPCC shape dataset
https://github.com/SantanderMetGroup/ATLAS/blob/devel/reference-regions/IPCC-WGI-reference-regions-v4_shapefile.zip  
was extracted using geopandas.

Software description:
---------------------
The software from Mistral:
- ESMValTool environment file: IPCC_environments/mistral_cryo_and_xcb32_conda_environment.yml
- pip file: IPCC_environments/mistral_cryo_and_xcb32_pip_environment.txt

The software from Liza's computer:
- ESMValTool environment file: IPCC_environments/liza_dell_computer_conda_environment.yml
- pip file: IPCC_environments/liza_dell_computer_pip_environment.txt

Hardware description:
---------------------
The data was processed on Mistral, the final version was pre-processed on the 5th of March 2021 
at 02:34:00 UTC. Some minor "cosmetic" edits were done on Liza's dell laptop the same day.
