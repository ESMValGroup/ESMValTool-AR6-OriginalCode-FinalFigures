
ENSO SEASONALITY
================

Figure number: Figure 3.37
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.37](../images/ar6_wg1_chap3_figure3_37_enso_seasonality.png?raw=true)


Description:
------------
ENSO seasonality in observations (black) and historical simulations from CMIP5 
(blue; extended with RCP4.5) and CMIP6 (red) for 1951-2010. (a) Climatological 
standard deviation of the monthly ENSO index (SST anomaly averaged over the Niño 
3.4 region; °C). Shading and lines represent 5th-95th percentiles and multi-
model ensemble means, respectively. (b) Seasonality metric, which is defined for 
each model and each ensemble member as the ratio of the ENSO index 
climatological standard deviation in November-January (NDJ) to that in March-May 
(MAM). Each dot represents an ensemble member from the model indicated on the 
vertical axis. The boxes and whiskers represent the multi-model ensemble mean, 
interquartile ranges and 5th and 95th percentiles of CMIP5 and CMIP6 
individually. The CMIP5 and CMIP6 multi-model ensemble means and observational 
values are indicated at the top right of the panel. The multi-model ensemble 
means and percentile values are evaluated after weighting individual members 
with the inverse of the ensemble size of the same model, so that individual 
models are equally weighted irrespective of their ensemble sizes. All results 
are based on 5-month running mean SST anomalies with triangular-weights after 
linear detrending. 


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
Recipe used: recipes/ipccar6wg1ch3/recipe_ar6ch3_enso_cmip5+6.yml

Diagnostics used: 
- diag_scripts/ar6ch3_enso/define_ensoindex.ncl
- diag_scripts/ar6ch3_enso/draw_seasonality.ncl


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_ar6ch3_enso_cmip5+6_YYYYMMDD_HHMMSS/plots/enso_index/draw_enso_seasonality/enso_seasonality.pdf


Recipe generations tools: 
-------------------------
Were any tools used to populate the recipe? if so what were they? N/A if not applicable. 


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
What machine was used: Mistral


Any further instructions: 
-------------------------

