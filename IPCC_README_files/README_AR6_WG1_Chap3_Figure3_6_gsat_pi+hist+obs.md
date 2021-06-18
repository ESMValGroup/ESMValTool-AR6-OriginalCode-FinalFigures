
VARIABILITY OF GSAT
===================

Figure number: Figure 3.6
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.6](../images/ar6_wg1_chap3_figure3_6_gsat_pi+hist+obs.png?raw=true)


Description:
------------
Simulated internal variability of global surface air temperature (GSAT) versus 
observed changes. (a) Time series of 5-year running mean GSAT anomalies in 45 
CMIP6 pre-industrial control (unforced) simulations. The 10 most variable models 
in terms of 5-year running mean GSAT are coloured according to the legend on 
Figure 3.4. (b) Histograms of GSAT changes in CMIP6 historical simulations 
(extended by SSP2-4.5 simulations) from 1850-1900 to 2010-2019 are shown by pink 
shading in (c), and GSAT changes from the first 51 years average to the last 20 
years average of 170-year overlapping segments of the pre-industrial control 
simulations shown in (a) are shown by blue shading. GMST changes in 
observational datasets for the same period are indicated by black vertical 
lines. (c) Observed GMST anomaly time series relative to the 1850-1900 average. 
Black lines represent the 5-year running means while grey lines show unfiltered 
annual time series.


Author list:
------------
- Kosaka, Y.: University of Tokyo, Japan; ykosaka@atmos.rcast.u-tokyo.ac.jp
- Bellouin, N.: University of Reading, UK
- Cassous, C.: CNRS-Cerfacs, France
- Gillett, N.: Environment and Climate Change Canada
- Bock, L.: DLR, Germany


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
Recipe used: recipes/ipccar6wg1ch3/recipe_ipccwg1ar6ch3_fig_3_6.yml

Diagnostic used: diag_scripts/ipccwg1ar6ch3_gsat_pi/gsat_pi_hist_obs.ncl


Expected image path:
--------------------
- recipe_ipccwg1ar6ch3_fig_3_6_YYYYMMDD_HHMMSS/plots/gsat_pi+hist+obs/gsat_pi_hist_obs/gsat_pi+hist+obs.png


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/ar6_newcore_lisa_conda_environment.yml
- pip file: IPCC_environments/ar6_newcore_lisa_pip_environment.txt


Hardware description:
---------------------
What machine was used:  Mistral
