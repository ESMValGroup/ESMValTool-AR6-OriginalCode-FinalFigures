ASPECTS OF THE MEDITERRANEAN SUMMER WARMING
===========================================

Figure number: Figure 10.20
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 10

![Figure 10.20](../images/ar6_wg1_chap10_figure10_20_Mediterranean_past.png?raw=true)


Description:
------------
Aspects of Mediterranean summer warming. (a) Mechanisms and feedbacks involved in enhanced Mediterranean summer warming. (b) Locations of observing stations in E-OBS and (Donat et al., 2014). (c) Differences in temperature observational data sets (NOAA Global Temp, Berkeley Earth, CRUTEM4 and GISTEMP) with respect to E-OBS for the land points between the Mediterranean Sea and 46°N and west of 30°E. (d) Observed summer (June to August) surface air temperature linear trends (°C decade -1 ) over the 1960‒2014 period from Berkeley Earth. (e) Time series of area averaged Mediterranean (25°N‒50°N, 10°W‒40°E) land point summer temperature anomalies (°C, baseline 1995–2014). Dark blue, brown and turquoise lines show low-pass filtered temperature of Berkeley Earth, CRU TS and HadCRUT5, respectively. Orange, light blue and green lines show low-pass filtered ensemble means of HighResMIP (4 members), CORDEX EUR-44 (20 members) and CORDEX EUR-11 (37 members). Blue and red lines and shadings show low-pass filtered ensemble means and standard deviations of CMIP5 (41 members) and CMIP6 (36 members). The filter is the same as the one used in Figure 10.10. (f) Distribution of 1960‒2014 Mediterranean summer temperature linear trends (°C decade -1 ) for observations (black crosses), CORDEX EUR-11 (green circles), CORDEX EUR-44 (light blue circles), HighResMIP (orange circles), CMIP6 (red circles), CMIP5 (blue circles) and selected SMILEs (grey box-and-whisker plots, MIROC6, CSIRO-Mk3-6-0, MPI-ESM and d4PDF). Ensemble means are also shown. CMIP6 models showing a very high ECS (Box. 4.1) have been marked with a black cross. All trends are estimated using ordinary least-squares and box-and-whisker plots follow the methodology used in Figure 10.6. (g) Ensemble mean differences with respect to the Berkeley Earth linear trend for 1960‒2014 (°C decade -1 ) of CMIP5, CMIP6, HighResMIP, CORDEX EUR-44 and CORDEX EUR-11. Further details on data sources and processing are available in the chapter data table (Table 10.SM.11).


Author list:
------------
- Jury, M.W.: BSC, Spain; martin.w.jury@gmail.com; githubid: mwjury
- Haarsma, R.: KNMI, Netherlands
- Dosio, A.: JRC, Italy
- Doblas-Reyes, F.J.: BSC, Spain
- Terray, L.: CERFACS/CNRS, France


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
Recipe used: [recipes/ar6_wgi_ch10/recipe_Mediterranean.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_10/esmvaltool/recipes/ar6_wgi_ch10/recipe_Mediterranean.yml)

Diagnostic used: [diag_scripts/ar6_wgi_ch10/diagnostic_IPCC_AR6_CH10.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_10/esmvaltool/diag_scripts/ar6_wgi_ch10/diagnostic_IPCC_AR6_CH10.py)


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_Mediterranean_YYYYMMDD_HHMMSS/plots/ch_10/fig_20_and_21/Fig_20.png


Recipe generations tools:
-------------------------
N/A


Ancillary figures and datasets:
-------------------------------
N/A


Additional datasets:
--------------------
'external' data has been included in ESMValTool totalling ~288kB
observational gridpoint differences (12KB):
- [esmvaltool/diag_scripts/ar6_wgi_ch10/CH10_additional_data/GvdSchrier_pdfs/diff_trend_eobs-*.csv](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter_10/esmvaltool/diag_scripts/ar6_wgi_ch10/CH10_additional_data)
obs station locations (276K):
- [esmvaltool/diag_scripts/ar6_wgi_ch10/CH10_additional_data/Mediterranean_station_info/](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter_10/esmvaltool/diag_scripts/ar6_wgi_ch10/CH10_additional_data)


Software description:
---------------------
- ESMValTool environment file: [IPCC_environments/ar6_chapter_10_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/ar6_chapter_10_conda_environment.yml)
- pip file: [IPCC_environments/ar6_chapter_10_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/ar6_chapter_10_pip_environment.txt)


Hardware description:
---------------------
Internal Wegener Center (University of Graz, Austria) machine wegc203128.

** The documentation was created by Chapter 10 Chapter Scientist Martin W. Jury (email: martin.w.jury@gmail.com, githubid: mwjury). Please, contact Martin in case any questions in documentation arise.
