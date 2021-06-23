
ATTRIBUTION OF NAM AND SAM TRENDS
=================================

Figure number: Figure 3.34
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.34](../images/ar6_wg1_chap3_figure3_34_nam_sam_trends_damip.png?raw=true)


Description:
------------
Attribution of observed seasonal trends in the annular modes to forcings. 
Simulated and observed trends in NAM indices over 1958-2019 (a) and in SAM 
indices over 1979-2019 (b) and over 2000-2019 (c) for boreal winter (December-
February average; DJF) and summer (June-August average; JJA). The indices are 
based on the difference of the normalized zonally averaged monthly mean sea 
level pressure between 35ºN and 65ºN for the NAM and between 40ºS and 65ºS for 
the SAM as defined in Jianping and Wang (2003) and Gong and Wang (1999), 
respectively: the unit is decade–1. Ensemble mean, interquartile ranges and 5th 
and 95th percentiles are represented by empty boxes and whiskers for pre-
industrial control simulations and historical simulations. The number of 
ensemble members and models used for computing the distribution is given in the 
upper-left legend. Grey lines show observed trends from the ERA5 and JRA-55 
reanalyses. Multi-model multi-member ensemble means of the forced component of 
the trends as well as their 5- 95% confidence intervals assessed from 
t-statistics, are represented by filled boxes, based on CMIP6 individual forcing 
simulations from DAMIP ensembles; greenhouse gases in brown, aerosols in light 
blue, stratospheric ozone in purple and natural forcing in green. Models with at 
least 3 ensemble members are used for the filled boxes, with black dots 
representing the ensemble means of individual models. 


Author list:
------------
- Phillips, A.: NCAR, U.S.; asphilli@ucar.edu
- Kosaka, Y.: University of Tokyo, Japan
- Cassous, C.: CNRS-Cerfacs, France
- Karmouche, S.: University of Bremen, Germany
- Bock, L.: DLR, Germany
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
Recipe used: recipes/ipccar6wg1ch3/recipe_ipccwg1ar6ch3_modes.yml

Diagnostics used: 
- diag_scripts/ipccwg1ar6ch3_modes/nam_sam.trends.summary.bar.ncl
- diag_scripts/ipccwg1ar6ch3_modes/nam_sam.damip.alt_def.ncwrite.ncl
- diag_scripts/ipccwg1ar6ch3_modes/nam_sam.hist.alt_def.ncwrite.ncl
- diag_scripts/ipccwg1ar6ch3_modes/nam_sam.obs.alt_def.ncwrite.ncl
- diag_scripts/ipccwg1ar6ch3_modes/nam_sam.piControl.alt_def.ncwrite.ncl


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_name_YYYYMMDD_HHMMSS/plots/nam_sam_da/nam_sam_trends_damip/nam_sam_trends_damip.png


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
- ESMValTool environment file: IPCC_environments/ar6_newcore_lisa_conda_environment.yml
- pip file: IPCC_environments/ar6_newcore_lisa_pip_environment.txt


Hardware description:
---------------------
Machine used: Mistral


Any further instructions: 
-------------------------

