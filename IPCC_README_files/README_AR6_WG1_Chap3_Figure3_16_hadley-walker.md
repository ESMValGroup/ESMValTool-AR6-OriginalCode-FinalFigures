
TROPICAL OVERTUNING CIRCULATION CHANGES
=======================================

Figure number: Figure 3.16
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.16](../images/ar6_wg1_chap3_figure3_16_hadley-walker.png?raw=true)


Description:
------------
Model evaluation and attribution of changes in Hadley cell extent and Walker 
circulation strength. (a-c) Trends in subtropical edge latitude of the Hadley 
cells in (a) the Northern Hemisphere for 1980-2014 annual mean and (b-c) 
Southern Hemisphere for (b) 1980-2014 annual mean and (c) 1980/81-1999/2000 
December-January-February mean. Positive values indicate northward shifts. 
(d-f) Trends in the Pacific Walker circulation strength for (d) 1901-2010, 
(e) 1951-2010 and (f) 1980-2014. Positive values indicate strengthening. 
Based on CMIP5 historical (extended with RCP4.5), CMIP6 historical, AMIP, 
pre-industrial control, and single forcing simulations along with HadSLP2 and 
reanalyses. Pre-industrial control simulations are divided into non-overlapping 
segments of the same length as the other simulations. White boxes and whiskers 
represent mean, interquartile ranges and 5th and 95th percentiles, calculated 
after weighting individual members with the inverse of the ensemble of the same 
model, so that individual models are equally weighted (Section 3.2). The filled 
boxes represent the 5-95% confidence interval on the multi-model mean trends of 
the models with at least 3 ensemble members, with dots indicating the ensemble 
means of individual models. The edge latitude of the Hadley cell is defined 
where the surface zonal wind velocity changes sign from negative to positive, as 
described in the Appendix of Grise et al. (2018). The Pacific Walker circulation 
strength is evaluated as the annual mean difference of sea level pressure 
between [5°S-5ºN, 160ºW-80ºW] and [5ºS-5ºN, 80ºE-160ºE].


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
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter_3


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: fix_cmip6_models_newcore


Recipe & diagnostics:
---------------------
Recipe used: recipes/ipccar6wg1ch3/recipe_ar6ch3_hadley+walker_cmip5+6+damip.yml

Diagnostics used:
- diag_scripts/ar6ch3_hadley_walker/trend_walker_strength.ncl
- diag_scripts/ar6ch3_hadley_walker/draw_pdf.ncl


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_ar6ch3_hadley+walker_cmip5+6+damip_YYYYMMDD_HHMMSS/plots/draw/draw_pdf/hadley_walker_trends.pdf


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
- ESMValTool environment file: IPCC_environments/ar6_newcore_remi_conda_environment.yml
- pip file: IPCC_environments/ar6_newcore_remi_pip_environment.txt


Hardware description:
---------------------
Machine used: Mistral


Any further instructions: 
-------------------------
