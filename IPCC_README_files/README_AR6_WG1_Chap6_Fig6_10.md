
NET EFFECTIVE RADIATIVE FORCING: AEROSOLS
============

Figure number: Figure 6.10  
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 6  

![Figure 6.10](../images/ar6_wg1_chap6_fig6_10_netERFmap_SWLW_aer.png?raw=true)


Description:
------------
Multi-model mean Effective radiative forcings (ERFs) due to aerosol changes between 1850 and recent-past (1995-2014).  


Author list:
------------
- Kuo, C: Lawrence Berkeley National Laboratory, chaincy@berkeley.edu (lead only) chaincy-ipcc, chaincy-cal 

ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: [IPCC_AR6_WG1_Ch6] (https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6)


Recipe & diagnostics:
---------------------
Recipe(s) used:   
[recipes/ar6ch6/recipe_erf_histSST-piAer_Fig6.10.yml] (https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/recipes/ar6ch6/recipe_erf_histSST-piAer_Fig6.10.yml)   
Please describe this recipe:  
Collect the upward shortwave and longwave fluxes at the top of the atmosphere, for AerChemMIP (Collins et al,2017,GMD,10(2),585-607, https://doi.org/10.5194/gmd-10-585-2017) experiments histSST and histSST-piAer, over the period 1995-2014.


Diagnostic(s) used:   
[diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.10_erf_piAer.py] (https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.10_erf_piAer.py)   
The above diagnostic imports:  
[diag_scripts/ar6ch6/cmapipcc.py] (https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/diag_scripts/ar6ch6/cmapipcc.py)  
[diag_scripts/ar6ch6/ch6_fns.py] (https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/diag_scripts/ar6ch6/ch6_fns.py)    
[diag_scripts/ar6ch6/chem_div_disc.txt] (https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/diag_scripts/ar6ch6/chem_div_disc.txt)  

Please describe this diagnostic:  
Calculates the spatial distribution of the net ERF with area-weighted global mean ERF, calculated from the difference in upward shortwave and longwave fluxes at the top of the atmosphere, between AerChemMIP experiments histSST and histSST-piAer, averaged over 1995-2014.

Figure 6.10 is created by running the ESMValTool recipe recipes/ar6ch6/recipe_erf_histSST-piAer_Fig6.10.yml  
panel a) will be run through recipes/ar6ch6/recipe_erf_histSST-piAer_Fig6.10.yml via diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.10_erf_piAer.py  
panel b) is created through ipynb/ipcc_ar6wg1_Fig6.10b_FGD_submit.ipynb via netcdf output from diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.10_erf_piAer.py  

Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:  
- recipe_erf_histSST-piAer_Fig6.10_YYYYMMDD_HHMMSS/plots/diffexpts/ar6fig6_erf/fig6_erf_piAer.png  


Recipe generations tools: 
-------------------------
Were any tools used to populate the recipe? if so what were they? N/A if not applicable.   
N/A


Ancillary figures and datasets:
-------------------------------
In addition to the main figure, diagnostics may produce several figures and datasets along the way or several versions of the main figure. Please use this space to highlight anything that may be useful for future iterations:  
Panel b) of Figure 6.10 is created by the file  
[ipynb/ipcc_ar6wg1_Fig6.10b_FGD_submit.ipynb] (https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/ipynb/ipcc_ar6wg1_Fig6.10b_FGD_submit.ipynb)  
It reads in netcdf output from diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.10_erf_piAer.py, which are output into the recipe path:  
- recipe_erf_histSST-piAer_Fig6.10_YYYYMMDD_HHMMSS/plots/diffexpts/ar6fig6_erf/erf_hatch_LW.nc
- recipe_erf_histSST-piAer_Fig6.10_YYYYMMDD_HHMMSS/plots/diffexpts/ar6fig6_erf/erf_hatch_SW.nc


Additional datasets:
--------------------
What additional datasets were used to produce this figure?   
The AR6 WG1 Atlas reference regions are used to produce Figure 6.10b.    
Where are they on the computational machine or in the respository?   
On the IPCC_AR6_WG1_Ch6 branch of ESMValTool-AR6-OriginalCode-FinalFigures:   
/ESMValTool-AR6-OriginalCode-FinalFigures/ipynb/data/regionmask/AR6_WGI_referenceRegions/AR6_WGI_referenceRegions.shp  
/ESMValTool-AR6-OriginalCode-FinalFigures/ipynb/data/regionmask/AR6_WGI_referenceRegions/AR6_WGI_referenceRegions.dbf  
/ESMValTool-AR6-OriginalCode-FinalFigures/ipynb/data/regionmask/AR6_WGI_referenceRegions/AR6_WGI_referenceRegions.prj  
/ESMValTool-AR6-OriginalCode-FinalFigures/ipynb/data/regionmask/AR6_WGI_referenceRegions/AR6_WGI_referenceRegions.rda  
/ESMValTool-AR6-OriginalCode-FinalFigures/ipynb/data/regionmask/AR6_WGI_referenceRegions/AR6_WGI_referenceRegions.shx  

What are their access permissions/Licenses?  
The license details can be found at https://github.com/IPCC-WG1/Atlas/blob/main/LICENSE.md    

Software description:
---------------------
- ESMValTool environment file:   
/ESMValTool-AR6-OriginalCode-FinalFigures/IPCC_environments/ar6wg1_chap6_figs_conda_environment.yml  
- Other software used:  
A Jupyter notebook (ipcc_ar6wg1_Fig6.10b_FGD_submit.ipynb) is used to create Figure 6.10 panel b. See section "Ancillary figures and datasets" and "Additional datasets", above, for information.  

Hardware description:
---------------------
What machine was used: 
Mistral  
 
When was this machine used?  
Last used July 2021 to produce figures from ESMValTool  

