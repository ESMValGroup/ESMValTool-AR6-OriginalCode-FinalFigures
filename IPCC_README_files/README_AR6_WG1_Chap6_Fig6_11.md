Figure 6.11 is created starting from the ESMValTool recipe recipes/ar6ch6/recipe_erf_histSST-piAer_Fig6.11.yml.  It calls 
diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.11_erf_aer_time.py which provides data output used by ipynb/ipcc_ar6wg1_Fig6.11_FGD_submit.ipynb.



TEMPORAL REGIONAL MEAN NET EFFECTIVE RADIATIVE FORCING DUE TO AEROSOLS
============

Figure number: Figure 6.11
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 6

![Figure 6.11](../images/ar6_wg1_chap6_fig6_11_netERF_time_aer.png?raw=true)


Description:
------------
Time evolution of 20-year multi-model mean averages of the annual area-weighted mean regional net Effective Radiative Forcings (ERFs) due to aerosols for each of the 14 major regions in the Atlas, and global mean, using the models and model experiments as in Figure 6.10. 


Author list:
------------
- Kuo, C: Lawrence Berkeley National Laboratory, chaincy@berkeley.edu (lead only) chaincy-ipcc, chaincy-cal 


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: IPCC_AR6_WG1_Ch6 


Recipe & diagnostics:
---------------------
Recipe(s) used:  
[recipes/ar6ch6/recipe_erf_histSST-piAer_Fig6.11.yml] (https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/recipes/ar6ch6/recipe_erf_histSST-piAer_Fig6.11.yml)  
Please describe this recipe:

Diagnostic(s) used:  
[diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.11_erf_aer_time.py] (https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.11_erf_aer_time.py)  
ipcc_ar6wg1_fig6.11_erf_aer_time.py imports:   
* [diag_scripts/ar6ch6/cmapipcc.py] (https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/diag_scripts/ar6ch6/cmapipcc.py) 
 
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

