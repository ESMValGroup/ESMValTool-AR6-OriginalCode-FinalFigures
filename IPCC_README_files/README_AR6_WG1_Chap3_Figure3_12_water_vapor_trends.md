
TRENDS IN TOTAL COLUMN WATER VAPOR
==================================

Figure number: Figure 3.12
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.12](../images/ar6_wg1_chap3_figure3_12_water_vapor_trends.png?raw=true)


Description:
------------
Column water vapor path trends (%/decade) for the period 1998-2019 averaged over 
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
moisture content. Proc. Natl. Acad. Sci. 25. Available at: https://doi.org/10.1073/pnas.0702872104.


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter_3


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: fix_cmip6_models_newcore


Recipe & diagnostics:
---------------------
Recipe used: recipes/ipccar6wg1ch3/recipe_ipccwg1ar6ch3_santer20jclim.yml

Diagnostic used: diag_scripts/santer20jclim/santer20jclimfig.py


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_ipccwg1ar6ch3_santer20jclim_newextendedv3_YYYYMMDD_HHMMSS/plots/wvtrend/santer20jclim/fig1.png


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
Machine used: Mistral


Any further instructions: 
-------------------------
