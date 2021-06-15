
PRECIPITATION ANOMALY
=====================

Figure number: 3.15
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.15](../images/ar6_wg1_chap3_figure3_15_precip_anomaly.png?raw=true)


Description:
------------
Observed and simulated time series of anomalies in zonal average annual mean 
precipitation.  a), c)-f) Evolution of global and zonal average annual mean 
precipitation (mm day-1) over areas of land where there are observations, 
expressed relative to the base-line period of 1961–1990, simulated by CMIP6 
models (one ensemble member per model) forced with both anthropogenic and 
natural forcings (brown) and natural forcings only (green). Multi-model means 
are shown in thick solid lines and shading shows the 5-95% confidence interval
of the individual model simulations. The data is smoothed using a low pass 
filter. Observations from three different data sets are included: gridded 
values derived from Global Historical Climatology Network (GHCN V2) station
data, updated from Zhang et al. (2007), data from the Global Precipitation 
Climatology Product (GPCP L3 v2.3, Huffman and Bolvin (2013)) and from the 
Climate Research Unit (CRU TS4.02, Harris et al. (2014)). Also plotted are 
boxplots showing interquartile and 5-95% ranges of simulated trends over the 
period for simulations forced with both anthropogenic and natural forcings 
(brown) and natural forcings only (blue). Observed trends for each observational
product are shown as horizontal lines. Panel b) shows annual mean precipitation 
rate (mm day-1) of GHCN V2 for the years 1950-2014 over land areas used to 
compute the plots.


Author list:
------------
- Bock, L.: DLR, Germany; lisa.bock@dlr.de
- Barreiro, M.: Universidad de la República, Uruguay 
- Eyring, V.: DLR., Germany


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter_3


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: fix_cmip6_models_newcore


Recipe & diagnostics:
---------------------
Recipe used: recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_atmosphere.yml

Diagnostic used: diag_scripts/ipcc_ar6/precip_anom.ncl


Expected image path:
--------------------
- recipe_ipccwg1ar6ch3_atmosphere_YYYYMMDD_HHMMSS/plots/fig_3_15_precip_anom/tsline/precip_anom_1950-2014.eps


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/ar6_newcore_lisa_conda_environment.yml
- pip file: IPCC_environments/ar6_newcore_lisa_pip_environment.txt


Hardware description:
---------------------
Machine used: Mistral
