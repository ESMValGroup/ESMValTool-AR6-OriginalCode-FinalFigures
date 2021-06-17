
DETECTION AND ATTRIBUTION ANALYIS OF TROPICAL PRECIPITATION
===========================================================

Figure number: Figure 3.14
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter X

![Figure 3.14](../images/ar6_wg1_chap3_fig3_14_precip_wetdry.png?raw=true)


Description:
------------
Wet (a) and dry (b) region tropical mean (30°S-30°N) annual precipitation 
anomalies. Observed data are shown with black lines (GPCP), ERA5 reanalysis in 
grey, single model simulations results are shown with light blue/red lines 
(CMIP6), and multi-model-mean results are shown with dark blue/red lines 
(CMIP6). Wet and dry region annual anomalies are calculated as the running mean 
over 12 months relative to a 1988-2020 based period. The regions are defined as 
the wettest third and driest third of the surface area, calculated for the 
observations and for each model separately for each season (following Polson and 
Hegerl, 2017). Scaling factors (panels c,d) are calculated for the combination 
of the wet and dry region mean, where the observations, reanalysis and all the 
model simulations are first standardised using the mean standard deviation of 
the pre-industrial control simulations. Two total least squares regression 
methods are used: noise in variables (following Polson and Hegerl, 2017) which 
estimates a best estimate and a 5-95% confidence interval using the 
pre-industrial controls (circle and thick green line) and the pre-industrial 
controls with double the variance (thin green line); and a bootstrap method 
(DelSole et al., 2019) (5-95% confidence interval shown with a purple line and 
best estimate with a purple circle). Panel (c) shows results for GPCP and panel 
(d) for ERA5. Figure is adapted from Schurer et al. (2020).


Author list:
------------
- Schurer, A.: University of Edinburgh, UK; A.Schurer@ed.ac.uk
- Kazeroni, R.: DLR, Germany


Publication sources:
--------------------
Schurer, A. P., Hegerl, G. C., Ballinger, A., and Friedman, A. R. (2020). Human 
influence strengthens the contrast between tropical wet and dry regions. 
Environ. Res. Lett. 15. doi:https://doi.org10.1088/1748-9326/ab83ab.


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: branch_name


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: branch_name


Recipe & diagnostics:
---------------------
Recipe(s) used: e.g. recipes/ipccar6wg1ch3/recipe_eyring21ipcc_3-3-ocean.yml
Please describe this recipe:

Diagnostic(s) used: e.g. diag_scripts/ipcc_ar6/tsline.ncl
Please describe this diagnostic:


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_name_YYYYMMDD_HHMMSS/plots/diagnostic_name/subdirectory/filename.extension


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
Software versions, name of environment file (see **save conda environment** in CONTRIBUTING.md), other software packages,…
- ESMValTool environment file: e.g. IPCC_environments/$NAME_conda_environment.yml
- pip file: e.g. IPCC_environments/$NAME_pip_environment.txt
- Other software used:


Hardware description:
---------------------
What machine was used:  e.g. Mistral or Jasmin
When was this machine used?


Any further instructions: 
-------------------------

