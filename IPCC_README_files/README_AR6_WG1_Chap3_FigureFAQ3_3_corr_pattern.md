
CORRELATION PATTERN
==========================

Figure number: FAQ 3.3
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure FAQ 3.3](../images/ar6_wg1_chap3_figureFAQ3_3_corr_pattern.png?raw=true)


Description:
------------
Pattern correlations between models and observations of three different 
variables: surface air temperature, precipitation and sea level pressure. 
Results are shown for the three last generations of models, from the Coupled 
Model Intercomparison Project (CMIP): CMIP3 (orange), CMIP5 (blue) and CMIP6 
(purple). Individual model results are shown as short lines, along with the 
corresponding ensemble average (long line). For the correlations the yearly 
averages of the models are compared with the reference observations for the 
period 1980-1999, with 1 representing perfect similarity between the models and 
observations. CMIP3 simulations performed in 2003-2007 were assessed in the 
IPCC Fourth Assessment, CMIP5 simulations performed in 2008-2013 were assessed 
in the IPCC Fifth Assessment, and CMIP6 simulations performed in 2015-2021 are 
assessed for in the IPCC Sixth Assessment.


Author list:
------------
- Bock, L.: DLR, Germany; lisa.bock@dlr.de
- Eyring, V.: DLR., Germany
- Morgenstern, O.: NIWA, New Zealand


Publication sources:
--------------------
Bock, L., Lauer, A., Schlund, M., Barreiro, M., Bellouin, N., Jones, C., Predoi, V., Meehl, G., Roberts, M., and Eyring, V.: Quantifying progress across different CMIP phases with the ESMValTool, Journal of Geophysical Research: Atmospheres, 125, e2019JD032321. https://doi.org/10.1029/2019JD032321


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter_3


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: fix_cmip6_models_newcore


Recipe & diagnostics:
---------------------
Recipes used: 
- recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_fig_faq3_3_ens.yml

Diagnostics used: 
- diag_scripts/ipcc_ar6/corr_pattern.ncl
- diag_scripts/ipcc_ar6/corr_pattern_collect.ncl


Expected image path:
--------------------
- recipe_ipccwg1ar6ch3_fig_faq3_3_ens_YYYYMMDD_HHMMSS/plots/fig_6/cor_collect/patterncor.pdf


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/ar6_newcore_lisa_conda_environment.yml
- pip file: IPCC_environments/ar6_newcore_lisa_pip_environment.txt


Hardware description:
---------------------
Machine used: Mistral
