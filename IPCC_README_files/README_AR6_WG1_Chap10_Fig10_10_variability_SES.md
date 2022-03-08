OBSERVED AND PROJECTED CHANGES IN AUSTRAL SUMMER PRECIPITATION
==============================================================

Figure number: Figure 10.10
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 10

![Figure 10.10](../images/ar6_wg1_chap10_figure10_10_variability_SES.png?raw=true)


Description:
------------
Observed and projected changes in austral summer (December to February) mean precipitation in GPCC, CRU TS and 100 members of the MPI-ESM. (a) 55-year trends (2015‒2070) from the ensemble members with the lowest (left) and highest (right) trend (% per decade, baseline 1995–2014). (b) Time series (%, baseline 1995–2014) for different spatial scales (from top to bottom: global averages; S.E. South America; grid boxes close to São Paulo and Buenos Aires) with a five-point weighted running mean applied (a variant on the binomial filter with weights [1-3-4-3-1]). The brown (green) lines correspond to the ensemble member with weakest (strongest) 55-year trend and the grey lines to all remaining ensemble members. Box-and-whisker plots show the distribution of 55-year linear trends across all ensemble members, and follow the methodology used in Figure 10.6. Trends are estimated using ordinary least squares. Further details on data sources and processing are available in the chapter data table (Table 10.SM.11).


Author list:
------------
- Jury, M.W.: BSC, Spain; martin.w.jury@gmail.com; githubid: mwjury
- Maraun, D.: UniGraz, Austria


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
Recipe used: [recipes/ar6_wgi_ch10/recipe_Douglas_SES_DJF.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_10/esmvaltool/recipes/ar6_wgi_ch10/recipe_Douglas_SES_DJF.yml)

Diagnostic used: [diag_scripts/ar6_wgi_ch10/diagnostic_IPCC_AR6_CH10.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_10/esmvaltool/diag_scripts/ar6_wgi_ch10/diagnostic_IPCC_AR6_CH10.py)


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_Douglas_SES_DJF_YYYYMMDD_HHMMSS/plots/ch_10/fig_10/Fig_10.png


Recipe generations tools:
-------------------------
N/A


Ancillary figures and datasets:
-------------------------------
N/A


Additional datasets:
--------------------
'external' data (IPCC Atlas region SES shape) has been included in ESMValTool totalling ~10KB
- [esmvaltool/diag_scripts/ar6_wgi_ch10/CH10_additional_data/SOUTH-AMERICA_Land_S.E.South-America_SES.*
](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter_10/esmvaltool/diag_scripts/ar6_wgi_ch10/CH10_additional_data)

Software description:
---------------------
- ESMValTool environment file: [IPCC_environments/ar6_chapter_10_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/ar6_chap_3_fig_3_10_conda_environment.yml)
- pip file: [IPCC_environments/ar6_chapter_10_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/ar6_chapter_10_pip_environment.txt)


Hardware description:
---------------------
Internal Wegener Center (University of Graz, Austria) machine wegc203128.

** The documentation was created by Chapter 10 Chapter Scientist Martin W. Jury (email: martin.w.jury@gmail.com, githubid: mwjury). Please, contact Martin in case any questions in documentation arise.
