
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
- Kuo, C: Lawrence Berkeley National Laboratory, USA; chaincy@berkeley.edu, chaincy.ipccwg1@gmail.com (lead only); githubid:chaincy-ipcc, githubid:chaincy-cal 


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: IPCC_AR6_WG1_Ch6 


Recipe & diagnostics:
---------------------
Recipe(s) used:  
[recipes/ar6ch6/recipe_erf_histSST-piAer_Fig6.11.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/recipes/ar6ch6/recipe_erf_histSST-piAer_Fig6.11.yml)  
Please describe this recipe:  
Time series of the upward top-of-the-atmosphere shortwave flux (rsut) and upward top-of-the-atmosphere flux (rlut) at each grid. The diagnostic scripts will calculate the annual mean effective radiative forcing (ERF) due to aerosols differencing shortwave and longwave fluxes between histSST-piAer and histSST AerChemMIP experiments.  



Diagnostic(s) used:  
[diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.11_erf_aer_time.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.11_erf_aer_time.py)  
ipcc_ar6wg1_fig6.11_erf_aer_time.py imports:   
* [diag_scripts/ar6ch6/cmapipcc.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/diag_scripts/ar6ch6/cmapipcc.py) 
 
Please describe this diagnostic:
Script [diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.11_erf_aer_time.py]  calculates the gridded annual mean ERF of all models for IPCC AR6 WG1 Figure 6.11,  The final Figure 6.11 is created from [ipynb/ipcc_ar6wg1_Fig6.11_FGD_submit.ipynb](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/ipynb/ipcc_ar6wg1_Fig6.11_FGD_submit.ipynb).


Expected data output path:
--------------------
This is the path of the time series data relative to the automatically generated ESMValTool output location:
- recipe_erf_histSST-piAer_Fig6.11_YYYMMDD_HHMMSS/plots/diffexpts/ar6fig6_erf_time/LW_diff_timemap.nc   
- recipe_erf_histSST-piAer_Fig6.11_YYYMMDD_HHMMSS/plots/diffexpts/ar6fig6_erf_time/SW_diff_timemap.nc   

Recipe generations tools: 
-------------------------
Were any tools used to populate the recipe? if so what were they? N/A if not applicable.   
N/A

Ancillary figures and datasets:
-------------------------------
In addition to the main figure, diagnostics may produce several figures and datasets along the way or several versions of the main figure. Please use this space to highlight anything that may be useful for future iterations:

The final Figure 6.11 is created by the following Jupyter notebook:  
[ipynb/ipcc_ar6wg1_Fig6.11_FGD_submit.ipynb](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/ipynb/ipcc_ar6wg1_Fig6.11_FGD_submit.ipynb)  
The Jupyter notebook reads in netcdf output from diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.11_erf_aer_time.py, which are output into the recipe path:  
- recipe_erf_histSST-piAer_Fig6.11_YYYMMDD_HHMMSS/plots/diffexpts/ar6fig6_erf_time/LW_diff_timemap.nc   
- recipe_erf_histSST-piAer_Fig6.11_YYYMMDD_HHMMSS/plots/diffexpts/ar6fig6_erf_time/SW_diff_timemap.nc   

Additional datasets:
--------------------
What additional datasets were used to produce this figure?   
The AR6 WG1 Atlas reference regions are used to produce Figure 6.11.    
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
A Jupyter notebook (ipcc_ar6wg1_Fig6.11_FGD_submit.ipynb) is used to create Figure 6.11. See section "Ancillary figures and datasets" and "Additional datasets", above, for information.  

Hardware description:
---------------------
What machine was used: 
Mistral  
 
When was this machine used?  
Last used July 2021 to produce figures from ESMValTool  
