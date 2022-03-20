# IPCC AR6 WG1 Chapter 6 figures, created using ESMValTool 

Detailed README files for Figures 6.10, 6.11, and 6.13 can be found in [IPCC_README_files] (https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/IPCC_AR6_WG1_Ch6/IPCC_README_files)  

Figure 6.10 is created by running the ESMValTool recipe recipes/ar6ch6/recipe_erf_histSST-piAer_Fig6.10.yml
panel a) will be run through recipes/ar6ch6/recipe_erf_histSST-piAer_Fig6.10.yml via diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.10_erf_piAer.py
panel b) is created through ipynb/ipcc_ar6wg1_Fig6.10b_FGD_submit.ipynb via netcdf output from diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.10_erf_piAer.py

Figure 6.11 is created starting from the ESMValTool recipe recipes/ar6ch6/recipe_erf_histSST-piAer_Fig6.11.yml.  It calls 
diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.11_erf_aer_time.py which provides data output used by ipynb/ipcc_ar6wg1_Fig6.11_FGD_submit.ipynb.

Figure 6.13 is created by running ESMValTool recipe recipes/ar6ch6/recipe_tas_hist-piAer_Fig6.13.yml, which calls 
diag_scripts/ar6ch6/ipcc_ar6wg1_fig6.13_tas_piAer_coupledOnly.py.

The conda environment for running ESMValTool recipes has been exported to environment.yml via the command:
% conda env export > environment.yml

The conda environment for running Jupyter notebooks ipynb/ipcc*.ipynb has been exported to ipynb/ipcc_ipynb_environment.yml via the command:
% conda env export > ipcc_ipynb_environment.yml

