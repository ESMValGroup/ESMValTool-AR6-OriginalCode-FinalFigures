Observed and simulated tropical mean temperature trends through the atmosphere
============

Figure number: Figure 3.10
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.10](../images/ar6_wg1_chap3_fig3_10_vertical_temp.png?raw=true)


Description:
------------
Vertical profiles of temperature trends in the tropics (20°S-20°N) for three periods: (a) 1979-2014 (b) 1979-1997 (ozone depletion era) (c) 1998-2014 (ozone recovery era). The black lines show trends in the RICH 1.7 (long dashed) and RAOBCORE 1.7 (dashed) radiosonde datasets (Haimberger et al. 2012), and in the ERA5/5.1 reanalysis (solid). Grey envelopes show the relative uncertainty in the RICH dataset based on 32 RICH-obs members of a previous version, 1.5.1, which used version 1.7.3 of the RICH software but with the parameters of version 1.5.1. ERA5 was used as reference for calculating the adjustments between 2010 and 2019, and ERA-Interim was used for the year before that. The grey envelopes have been scaled around the RICH 1.7 best estimates, based on the assumption that the relative uncertainty remains the same between the two versions. Red lines show trends in CMIP6 historical simulations by 52 couple climate models. Blue lines show trends in 39 CMIP6 models that used prescribed, rather than simulated, sea surface temperatures (SSTs). (Adapted from Figure 1 of Mitchell et al. (2020).)


Author list:
------------
- Lo, Eunice: University of Bristol, UK, eunice.lo@bristol.ac.uk, githubid: yteunicelo 
- Mitchell, Dann: University of Bristol, UK
- Seviour, William: University of Exeter, UK
- Haimberger, Leopold: University of Vienna, Austria
- Polvani, Lorenzo: Columbia University, USA


Publication sources:
--------------------
- The vertical profile of recent tropical temperature trends: Persistent model biases in the context of internal variability, D. Mitchell et al, Environmental Research Letters 15, DOI:10.1088/1748-9326/ab9af7, 2020. 


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chap_3_fig_10


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: N/A


Recipe & diagnostics:
---------------------
Recipe used: recipes/recipe_verticaltemp.yml - Recipe to extract and process data for plotting vertical profiles of atmospheric temperatures

Diagnostic used: diag_scripts/vertical_temp_profiles/vertical_temp_trends.py - Diagnostic that calculates trends of atmospheric temperatures at different pressure levels


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_verticaltemp_20210225_100312/plots/atmos_trends/fig_3_10/png/vertical_temp_profiles_20S-20N_1979_2014_rich_raobcore_1.7_rio_range_1.5.1_recentred_all_5-95_20210225.png


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/ar6_chap_3_fig_3_10_conda_environment.yml
- pip file: IPCC_environments/ar6_chap_3_fig_3_10_pip_environment.txt


Hardware description:
---------------------
Machine used: Jasmin on 2021-02-25
