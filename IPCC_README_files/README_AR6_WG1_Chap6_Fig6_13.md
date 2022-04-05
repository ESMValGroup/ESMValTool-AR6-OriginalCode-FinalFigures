Surface Air Temperature Response due to Aerosols
============

Figure number: Figure 6.13
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 6

![Figure 6.13](../images/ar6_wg1_chap6_fig6_13_tas_coupledOnly_aer.png?raw=true)


Description:
------------
Multi-model mean surface air temperature response due to aerosol changes between 1850 and recent-past (1995-2014) calculated as the difference between CMIP6 ‘historical’ and AerChemMIP ‘hist-piAer’ experiments, where a) is the spatial pattern of the annual mean surface air temperature response, and b) is the mean zonally averaged response. Model means are derived from years 1995-2014. Uncertainty is represented using the advanced approach: No overlay indicates regions with robust signal, where ≥66% of models show change greater than variability threshold and ≥80% of all models agree on sign of change; diagonal lines indicate regions with no change or no robust signal, where <66% of models show a change greater than the variability threshold; crossed lines indicate regions with conflicting signal, where ≥66% of models show change greater than variability threshold and <80% of all models agree on sign of change. For more information on the advanced approach, please refer to the Cross-Chapter Box Atlas.1. AerChemMIP models MIROC6, MRI-ESM2-0, NorESM2-LM, GFDL-ESM4, GISS-E2-1-G, UKESM1-0-LL are used in the analysis.

Author list:
------------
- Kuo, C: Lawrence Berkeley National Laboratory, USA; chaincy@berkeley.edu, chaincy.ipccwg1@gmail.com (lead only); githubid: chaincy-ipcc, githubid: chaincy-cal 

ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: [IPCC_AR6_WG1_Ch6](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6_ESMValTool)


Recipe & diagnostics:
---------------------
Recipe used:   
[recipes/ar6ch6/recipe_ckuo_ipcc_6_13_tas_hist-piAer.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/IPCC_AR6_WG1_Ch6_ESMValTool/esmvaltool/recipes/ar6ch6/recipe_ckuo_ipcc_6_13_tas_hist-piAer.yml)   

Recipe description:  
This recipe creates the plots of the mean change in surface air temperature due to aerosols, as a geomap plot and a zonal mean change plot, over the time period 1995-2014.  IPCC AR6 WG1  Figure 6.13 shows the surface air temperature change due to aerosols as the difference in the CMIP6 model variable 'tas' for coupled atmosphere-ocean models from the CMIP6 'historical' experiment and AerChemMIP experiments 'hist-piAer'.   

Diagnostics used:   
[diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.13_tas_piAer_coupledOnly.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/IPCC_AR6_WG1_Ch6_ESMValTool/esmvaltool/diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.13_tas_piAer_coupledOnly.py)   
[diag_scripts/ar6ch6/cmapipcc.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/IPCC_AR6_WG1_Ch6_ESMValTool/esmvaltool/diag_scripts/ar6ch6/cmapipcc.py)  
[diag_scripts/ar6ch6/ch6_fns.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/IPCC_AR6_WG1_Ch6_ESMValTool/esmvaltool/diag_scripts/ar6ch6/ch6_fns.py)

Diagnostics description:      
The mean change in surface air temperature ('tas') due to aerosols, between the coupled atmospheric-ocean models from CMIP6 'historical' experiments and AerChemMIP 'hist-piAer' experiments are calculated in grids over the years 1995-2014.   

Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_tas_hist-piAer_Fig6.13_YYYYMMDD_HHMMSS/plots/diffexpts/ar6fig6_piAer/fig6_fastslow_piAer_signagree_tas.png

Recipe generations tools: 
-------------------------
N/A   

Software description:
---------------------
- ESMValTool environment file: [IPCC_environments/ar6wg1_chap6_figs_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/main/IPCC_environments/ar6wg1_chap6_figs_conda_environment.yml)

Hardware description:
---------------------
Machine used: Mistral

When was this machine used?
Last used July 2021 to produce figures from ESMValTool   
