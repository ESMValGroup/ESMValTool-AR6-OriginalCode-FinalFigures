
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
- McGregor, S.: Monash University, Australia
- Cassou, C.: CNRS-Cerfacs, France
- Kazeroni, R.: DLR, Germany


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
- recipe_ar6ch3_enso_cmip5+6_YYYYMMDD_HHMMSS/plots/enso_index/draw_enso_seasonality/enso_seasonality.pdf


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/ar6_newcore_remi_conda_environment.yml
- pip file: IPCC_environments/ar6_newcore_remi_pip_environment.txt


Hardware description:
---------------------
Machine used: Mistral
