HISTORIC PRECIPITATION CHANGE IN THE SAHELIAN WEST AFRICAN MONSOON
==================================================================

Figure number: Figure 10.11
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 10

![Figure 10.11](../images/ar6_wg1_chap10_figure10_11_Sahel.png?raw=true)


Description:
------------
Attribution of historic precipitation change in the Sahelian West African monsoon during June to September. (a) Time series of CRU TS precipitation anomalies (mm day -1 , baseline 1955–1984) in the Sahel box (10°N–20°N, 20°W–30°E) indicated in panel (b) applying the same low-pass filter as that used in Figure 10.10. The two periods used for difference diagnostics are shown in grey columns. (b) Precipitation change (mm day -1 ) in CRU TS data for 1980–1990 minus 1950–1960 periods. (c) Precipitation difference (mm day -1 ) between 1.5x and 0.2x historical aerosol emissions scaling factors averaged over 1955–1984 and five ensemble members of HadGEM3 experiments after Shonk et al. (2020). (d) Sahel precipitation anomaly time series (mm day -1 , baseline 1955–1984) in CMIP6 for 49 historical simulations with all forcings (red), and thirteen for each of greenhouse gas-only forcing (light blue) and aerosol-only forcing (grey), with a thirteen-point weighted running mean applied (a variant on the binomial filter with weights [1-6-19-42-71-96-106-96-71-42-19-6-1]). The CMIP6 subsample of all forcings matching the individual forcing simulations is also shown (pink). (e) Precipitation linear trend (% per decade) for (left) decline (1955–1984) and (right) recovery periods (1985–2014) for ensemble means and individual CMIP6 historical experiments (including single-forcing) as in panel (d) plus 34 CMIP5 models (dark blue). Box-and-whisker plots show the trend distribution of the three coupled and the d4PDF atmosphere-only SMILEs used throughout Chapter 10 and follow the methodology used in Figure 10.6. The two black crosses represent observational estimates from GPCC and CRU TS. Trends are estimated using ordinary least-squares regression. Further details on data sources and processing are available in the chapter data table (Table 10.SM.11).


Author list:
------------
- Jury, M.W.: BSC, Spain; martin.w.jury@gmail.com; githubid: mwjury
- Turner, A.: University of Reading, UK


Publication sources:
--------------------
N/A


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: [ar6_chapter_10](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter_10)


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: [working_cordex_2.2](https://github.com/ESMValGroup/ESMValCore-AR6-OriginalCode-FinalFigures/tree/working_cordex_2.2)


Recipe & diagnostics:
---------------------
Recipe used: [recipes/ar6_wgi_ch10/recipe_Sahel.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_10/esmvaltool/recipes/ar6_wgi_ch10/recipe_Sahel.yml)

Diagnostic used: [diag_scripts/ar6_wgi_ch10/diagnostic_IPCC_AR6_CH10.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_10/esmvaltool/diag_scripts/ar6_wgi_ch10/diagnostic_IPCC_AR6_CH10.py)


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_Sahel_YYYYMMDD_HHMMSS/plots/ch_10/fig_11/Fig_11.png


Recipe generations tools:
-------------------------
N/A


Ancillary figures and datasets:
-------------------------------
N/A


Additional datasets:
--------------------
'external' data has been included in ESMValTool totalling 4.3MB
- [esmvaltool/diag_scripts/ar6_wgi_ch10/CH10_additional_data/ATurner_Aerosols/*.nc](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter_10/esmvaltool/diag_scripts/ar6_wgi_ch10/CH10_additional_data)


Software description:
---------------------
- ESMValTool environment file: [IPCC_environments/ar6_chapter_10_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/ar6_chap_3_fig_3_10_conda_environment.yml)
- pip file: [IPCC_environments/ar6_chapter_10_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/ar6_chapter_10_pip_environment.txt)


Hardware description:
---------------------
Internal Wegener Center (University of Graz, Austria) machine wegc203128.

** The documentation was created by Chapter 10 Chapter Scientist Martin W. Jury (email: martin.w.jury@gmail.com, githubid: mwjury). Please, contact Martin in case any questions in documentation arise.