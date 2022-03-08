
TRENDS IN TOTAL COLUMN WATER VAPOR
==================================

Figure number: Figure 3.12
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.12](../images/ar6_wg1_chap3_figure3_12_water_vapor_trends.png?raw=true)


Description:
------------
Column water vapor path trends (%/decade) for the period 1988-2019 averaged over 
the near-global oceans (50°S-50°N). The figure shows satellite data (RSS) and 
ERA5.1 reanalysis, as well as CMIP5 (sky blue) and CMIP6 (brown) historical 
simulations. All available ensemble members were used (see Section 3.2. Fits to 
the model trend probability distributions were performed with kernel density 
estimation. Figure is updated from Santer et al. (2007). 


Author list:
------------
- Santer, S.D.: LLNL, U.S.; santer1@llnl.gov
- Weigel, K.: University of Bremen, Germany
- Kazeroni, R.: DLR, Germany


Publication sources:
--------------------
Santer, B. D., Mears, C., Wentz, F. J., Taylor, K. E., Glecker, P. J., Wigley, 
T. M. L., et al. (2007). Identification of human-induced changes in atmospheric 
moisture content. Proc. Natl. Acad. Sci. 25, https://doi.org/10.1073/pnas.0702872104

Santer, B. D., Po-Chedley, S., Mears, C., Fyfe, J. C., Gillett, N., Fu, Q., 
Painter, J. F., Solomon, S., Steiner, A. K., Wentz, F. J., Zelinka, M. D., & 
Zou, C. (2021). Using Climate Model Simulations to Constrain Observations, 
Journal of Climate, 34(15), 6281-6301, https://doi.org/10.1175/JCLI-D-20-0768.1


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: [ar6_chapter_3](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter_3)


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: [fix_cmip6_models_newcore](https://github.com/ESMValGroup/ESMValCore-AR6-OriginalCode-FinalFigures/tree/fix_cmip6_models_newcore)


Recipe & diagnostics:
---------------------
Recipe used: [recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_santer21jclim.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3/esmvaltool/recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_santer21jclim.yml)

Diagnostic used: [diag_scripts/santer21jclim/santer21jclimfig.py](diag_scripts/santer21jclim/santer21jclimfig.py)


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_ipccwg1ar6ch3_santer21jclim_newextendedv3_YYYYMMDD_HHMMSS/plots/wvtrend/santer20jclim/fig1.png


Ancillary figures and datasets:
-------------------------------
There is one optional, additional figure which can be turned on in the recipe 
displays the pdf for ensemble members of chosen models (instead of all CMIP5 and 
CMIP6 models) by listing model names under the key word "add_model_dist".


Additional datasets:
--------------------
RSS data:
SOURCE = "ftp://ftp.remss.com/vapor/monthly_1deg/"
Remote Sensing Systems.
Monthly Mean Total Precipitable Water Data Set
on a 1 degree grid made from Remote Sensing
Systems Version-7 Microwave Radiometer Data,
Date created: 20201110T185551Z
accessed on 2020-11-19]. Santa Rosa, CA, USA.
Available at www.remss.com

cmorized with [esmvaltool/cmorizers/obs/cmorize_obs_rss.ncl](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3/esmvaltool/cmorizers/obs/cmorize_obs_rss.ncl) (in the ar6_chapter_3
branch, not in the main ESMValTool branch, yet) using the "precipitable_water" 
variable. The "fixed coverage" mask was produced from the same file, using 
"precipitable_water_climatology" and "precipitable_water_anomaly" with a tool 
based on the cmorizer: [esmvaltool/cmorizers/obs/cmorize_obs_rssanom.ncl](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3/esmvaltool/cmorizers/obs/cmorize_obs_rssanom.ncl).


Software description:
---------------------
- ESMValTool environment file: [IPCC_environments/ar6_newcore_remi_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/fix_links/IPCC_environments/ar6_newcore_remi_conda_environment.yml)
- pip file: [IPCC_environments/ar6_newcore_remi_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/fix_links/IPCC_environments/ar6_newcore_remi_pip_environment.txt)


Hardware description:
---------------------
Machine used: Mistral
