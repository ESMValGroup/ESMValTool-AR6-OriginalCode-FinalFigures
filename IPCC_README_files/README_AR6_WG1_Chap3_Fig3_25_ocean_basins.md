
OCEAN BASINS
============

Figure number: Figure 3.25
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.25](../images/ar6_wg1_chap3_fig3_25_ocean_basin.png?raw=true)

Description:
------------
The figure has eight panels, all contour plots. Left columns show potential temperature, right 
shows salinity bias. Each panel has two sub-panels upper one shows values from 0 to 1000 m depth, 
lower one shows the biases from 1000 to 5000 m.

Caption: CMIP6 potential temperature and salinity biases for the global ocean, Atlantic, Pacific
and Indian Oceans. Shown in colour are the time-mean differences between the CMIP6 historical 
multi-model climatological mean and observations, zonally averaged for each basin 
(excluding marginal and regional seas). The observed climatological values are obtained from the
World Ocean Atlas 2018 (WOA18, 1981-2010; Prepared by the Ocean Climate Laboratory,
National Oceanographic Data Center, Silver Spring, MD, USA), and are shown as labelled 
black contours for each of the basins. The simulated annual mean climatologies for 1981 to 2010 
are calculated from available CMIP6 historical simulations, and the WOA18 climatology utilized 
synthesized observed data from 1981 to 2010. A total of 30 available CMIP6 models have 
contributed to the temperature panels (left column) and 28 models to the salinity panels 
(right column). Potential temperature units are °C and salinity units are the Practical 
Salinity Scale 1978 [PSS-78].

Author list:
------------
- Elizaveta Malinina: ECCC, elizaveta.malinina-rieger@canada.ca, github: malininae
- Lee de Mora, Plymouth Marine Laboratory, ledm@pml.ac.uk, github: ledm
- Paul J. Durack, LLNL
- Nathan Gillett, ECCC
- Krishna Achutarao, Indian Institute of Technology Delhi
- Shayne McGregor, Monash University
- Rondrotiana Barimalala, University of Cape Town
- Valeriu Predoi, University of Reading
- Veronika Eyring, DLR

ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: [ar6_chapter3_cryo_xcb32](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter3_cryo_xcb32/)

Recipe & diagnostics:
---------------------
Recipe used: [recipes/recipe_ocean_basin_profile_bias.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter3_cryo_xcb32/esmvaltool/recipes/recipe_ocean_basin_profile_bias.yml)

Diagnostic used: [ocean/diagnostic_basin_profile_bias.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter3_cryo_xcb32/esmvaltool/diag_scripts/ocean/diagnostic_basin_profile_bias.py)

Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_ocean_basin_profile_bias_YYYYMMDD_HHMMSS/plots/diag_transect_thetao/Ocean_basin_profile_bias/fig_basin_profile_bias_CMIP6.

Recipe generations tools: 
-------------------------
N/A

Additional datasets:
--------------------
Except of CMIP6, WOA18 dataset is needed (https://www.ncei.noaa.gov/access/world-ocean-atlas-2018/bin/woa18.pl?parameter=t). 
File: woa18_decav81B0_t00_01.nc. As well as (https://www.ncei.noaa.gov/access/world-ocean-atlas-2018/bin/woa18.pl?parameter=s)
File: woa18_decav81B0_s00_01.nc. It was cmorized with cmorizers/obs/cmorize_obs_woa.py 
from the above mentioned repository.

To run the recipe three shape files are needed: indian_ocean.shp, atlantic.shp, and pacific.shp. 
Those were created from https://github.com/SantanderMetGroup/ATLAS/blob/devel/reference-regions/IPCC-WGI-reference-regions-v4_shapefile.zip 
with [this script](../IPCC_additional_scripts/shape_file_creation.py), which selects the ocean basins shapes,
adds the south part and plots the basins. 

Software description:
---------------------
The software from Mistral:
- ESMValTool environment file: [IPCC_environments/jasmin_ocean_figs_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/development_ar6_chap_3_ocean_environment.yml)
- pip file: [IPCC_environments/jasmin_ocean_figs_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/development_ar6_chap_3_ocean_pip_environment.txt)

The software from Liza's computer:
- ESMValTool environment file: [IPCC_environments/liza_dell_computer_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/liza_dell_computer_conda_environment.yml)
- pip file: [IPCC_environments/liza_dell_computer_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/liza_dell_computer_pip_environment.txt)


Hardware description:
---------------------
The data was processed on Jasmin, the final version was processed on the 26th of February 2021.
Some minor "cosmetic" edits were done on Liza's dell laptop on the 28th of February.

Any further instructions: 
-------------------------
The preprocessor for the figure with this version of core (v2.1) is very-very-very slow because 
of regridding. It is not enough cores or time to process it in one recipe, so it was split into 
two parts for a quicker processing, and afterwards gathered into one folder to produce the final figure. 
