
SST BIAS
============

Figure number: Figure 3.24
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.24](../images/ar6_wg1_chap3_fig3_24_sst_bias.png?raw=true)

Description:
------------
The figure has three panels. 

Caption: Biases in zonal mean and equatorial sea surface temperature (SST) in CMIP5 and CMIP6 models. 
CMIP6 (red), CMIP5 (blue) and HighResMIP (green) multi-model mean (a) zonally-averaged SST bias; 
(b) equatorial SST bias; and (c) equatorial SST compared to observed mean SST (black line) for 
1979-1999. The inter-model 5th and 95th percentiles are depicted by the respective shaded range. 
Model climatologies are derived from the 1979-1999 mean of the historical simulations,
using one simulation per model. The Hadley Centre Sea Ice and Sea Surface Temperature version 1
(HadISST) (Rayner et al., 2003) observational climatology for 1979-1999 is used as the reference 
for the error calculation in (a) and (b); and for observations in (c). 

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
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter3_cryo_xcb32

Recipe & diagnostics:
---------------------
Recipe used: recipes/recipe_ocean_fig_3_19_zonal_sst.yml

Diagnostic used: ocean/diagnostic_fig_3_19_zonal_sst.py

Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_ocean_fig_3_19_zonal_sst_YYYYMMDD_HHMMSS/plots/diag_3_19/diagnostic_3_19/fig_3.19.png

Recipe generations tools: 
-------------------------
N/A


Additional datasets:
--------------------
Except CMIP5, CMIP6 and HighResMIP, WOA18 dataset is needed 
(https://www.ncei.noaa.gov/access/world-ocean-atlas-2018/bin/woa18.pl?parameter=t). File: woa18_decav81B0_t00_01.nc. 
It was cmorized with cmorizers/obs/cmorize_obs_woa.py from the above mentioned repository.


Software description:
---------------------
The software from Jasmin:
- ESMValTool environment file: IPCC_environments/jasmin_ocean_figs_conda_environment.yml
- pip file: IPCC_environments/jasmin_ocean_figs_pip_environment.txt

The software from Liza's computer:
- ESMValTool environment file: IPCC_environments/liza_dell_computer_conda_environment.yml
- pip file: IPCC_environments/liza_dell_computer_pip_environment.txt


Hardware description:
---------------------
The data was processed on Jasmin, the final version was processed on the 7th of March 2021 at 23:32:04 UTC.
Some minor "cosmetic" edits were done on Liza's dell laptop the same day.