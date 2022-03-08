URBAN WARMING
=============

Figure number: Chapter Box 10.3 Figure 1
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 10

![Figure 10 CB 3.1](../images/ar6_wg1_chap10_figureCBox10_3_1_Urban_warming.png?raw=true)


Description:
------------
Urban warming compared to global GHG-induced warming. (a) Change in the annual mean surface air temperature over the period 1950‒2018 based on the local linear trend retrieved from CRU TS (°C per 68 years). This background warming is compared to the local warming that has been reported during 1950‒2018 in the literature from historical urbanization. The relative share of the total warming as percentage between the urban warming and the surrounding warming is plotted in a circle for each city. This map has been compiled from a review study (Hamdi et al., 2020). (b) Low-pass filtered time series of the annual mean temperature (°C) observed in the urban station of Tokyo (red line) and the rural reference station in Choshi (blue line) in Japan. The filter is the same as the one used in Figure 10.10. (c) Uncertainties in the relative share of urban warming with respect to the total warming (%) related to the use of different global observational datasets: CRU TS (brown circles), Berkeley Earth (dark blue downward triangle), HadCRUT5 (cyan upward triangle), Cowtan Way (orange plus) and GISTEMP (purple squares). Further details on data sources and processing are available in the chapter data table (Table 10.SM.11).


Author list:
------------
- Jury, M.W.: BSC, Spain; martin.w.jury@gmail.com; githubid: mwjury
- Hamdi, R.: Royal Meteorological Institute of Belgium, Belgium


Publication sources:
--------------------
Hamdi, R., Kusaka, H., Doan, Q.-V., Cai, P., He, H., Luo, G., Kuang, W., Caluwaerts, S., Duchêne, F., Van Schaeybroek, B., & Termonia, P. (2020). The State-of-the-Art of Urban Climate Change Modeling and Observations. Earth Systems and Environment, 4(4), 631–646. https://doi.org/10.1007/s41748-020-00193-3


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: [ar6_chapter_10](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter_10)


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: [working_cordex_2.2](https://github.com/ESMValGroup/ESMValCore-AR6-OriginalCode-FinalFigures/tree/working_cordex_2.2)


Recipe & diagnostics:
---------------------
Recipe used: [recipes/ar6_wgi_ch10/recipe_UrbanBox.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_10/esmvaltool/recipes/ar6_wgi_ch10/recipe_UrbanBox.yml)

Diagnostic used: [diag_scripts/ar6_wgi_ch10/diagnostic_IPCC_AR6_CH10.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chapter_10/esmvaltool/diag_scripts/ar6_wgi_ch10/diagnostic_IPCC_AR6_CH10.py)


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_UrbanBox_YYYYMMDD_HHMMSS/plots/ch_10/fig_CB-3.1/Fig_CB-3.1_CRU.png


Recipe generations tools:
-------------------------
N/A


Ancillary figures and datasets:
-------------------------------
N/A


Additional datasets:
--------------------
'external' data has been included in ESMValTool totalling ~362KB
- [esmvaltool/diag_scripts/ar6_wgi_ch10/CH10_additional_data/Urban_Box_data](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chapter_10/esmvaltool/diag_scripts/ar6_wgi_ch10/CH10_additional_data)


Software description:
---------------------
- ESMValTool environment file: [IPCC_environments/ar6_chapter_10_conda_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/ar6_chap_3_fig_3_10_conda_environment.yml)
- pip file: [IPCC_environments/ar6_chapter_10_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/ar6_chapter_10_pip_environment.txt)


Hardware description:
---------------------
Internal Wegener Center (University of Graz, Austria) machine wegc203128.

** The documentation was created by Chapter 10 Chapter Scientist Martin W. Jury (email: martin.w.jury@gmail.com, githubid: mwjury). Please, contact Martin in case any questions in documentation arise.