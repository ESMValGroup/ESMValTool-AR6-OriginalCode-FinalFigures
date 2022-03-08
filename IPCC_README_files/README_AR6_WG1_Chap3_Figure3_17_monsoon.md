
GLOBAL MONSOON DOMAIN AND INTENSITY
===================================

Figure number: Figure 3.17
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.17](../images/ar6_wg1_chap3_figure3_17_monsoon.png?raw=true)


Description:
------------
Model evaluation of global monsoon domain, intensity, and circulation. (a-b) 
Climatological summer-winter range of precipitation rate, scaled by annual mean 
precipitation rate (shading) and 850 hPa wind velocity (arrows) based on (a) 
GPCP and ERA5 and (b) a multi-model ensemble mean of CMIP6 historical 
simulations for 1979-2014. Enclosed by red lines is the monsoon domain based on 
the definition by Wang and Ding (2008) (c-d) 5-year running mean anomalies of 
(c) global land monsoon precipitation index defined as the percentage anomaly of 
the summertime precipitation rate averaged over the monsoon regions over land, 
relative to its average for 1979-2014 (the period indicated by light grey 
shading) and (d) the tropical monsoon circulation index defined as the vertical 
shear of zonal winds between 850 and 200 hPa levels averaged over 0º-20ºN, from 
120ºW eastward to 120ºE in NH summer (Wang et al., 2013; m s–1) in CMIP5 
historical and RCP4.5 simulations, CMIP6 historical and AMIP simulations. Summer 
and winter are defined for individual hemispheres: May through September for NH 
summer and SH winter, and November through March for NH winter and SH summer. 
The number of models and ensembles are given in the legend. The multi-model 
ensemble mean and percentiles are calculated after weighting individual members 
with the inverse of the ensemble size of the same model, so that individual 
models are equally weighted irrespective of ensemble size.

Author list:
------------
- Kosaka, Y.: University of Tokyo, Japan; ykosaka@atmos.rcast.u-tokyo.ac.jp
- Kazeroni, R.: DLR, Germany


Publication sources:
--------------------
Please list any publications that describe, explain or use this figure. 
- A paper title, A. Author et al, journal of science stuff 9, p51, DOI:564, 2021. 


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: [ar6_chapter_3](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter_3)


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: [fix_cmip6_models_newcore](https://github.com/ESMValGroup/ESMValCore-AR6-OriginalCode-FinalFigures/tree/fix_cmip6_models_newcore)


Recipe & diagnostics:
---------------------
Recipe used: [recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_globalmonsoon.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3/esmvaltool/recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_globalmonsoon.yml)

Diagnostics used:
- [diag_scripts/ar6ch3_monsoon/monsoon_domain_intensity.ncl](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3/esmvaltool/diag_scripts/ar6ch3_monsoon/monsoon_domain_intensity.ncl)
- [diag_scripts/ar6ch3_monsoon/draw_global_monsoon.ncl](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_3/esmvaltool/diag_scripts/ar6ch3_monsoon/draw_global_monsoon.ncl)


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_ipccwg1ar6ch3_globalmonsoon_YYYYMMDD_HHMMSS/plots/draw_global_monsoon_cmip5+6/plot_global_monsoon_runmean5yr/global_monsoon_GPCP-SG+ERA5_movingave5yrs.pdf


Ancillary figures and datasets:
-------------------------------
In addition to the main figure, diagnostics may produce several figures and datasets along the way or several versions of the main figure. Please use this space to highlight anything that may be useful for future iterations:


Additional datasets:
--------------------
What additional datasets were used to produce this figure?
Where are they on the computational machine or in the respository?
Can they be re-created?
What are their access permissions/Licenses?


Software description:
---------------------
- ESMValTool environment file: [IPCC_environments/ar6_newcore_remi_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/fix_links/IPCC_environments/ar6_newcore_remi_conda_environment.yml)
- pip file: [IPCC_environments/ar6_newcore_remi_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/fix_links/IPCC_environments/ar6_newcore_remi_pip_environment.txt)


Hardware description:
---------------------
Machine used: Mistral
