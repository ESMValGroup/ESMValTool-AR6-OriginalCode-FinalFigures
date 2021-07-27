PROJECTED CHANGES IN PRECIPITATION OVER THE ALPS
================================================

Figure number: Figure 10.9
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 10

![Figure 10.9](../images/ar6_wg1_chap10_figure10_9_precip_alps.png?raw=true)


Description:
------------
Projected changes in summer (June to August) precipitation (in percent with respect to the mean precipitation) over the Alps between  the periods 2070‒2099 and 1975‒2004. (a) Mean of four GCMs regridded to a common 1.32°x1.32° grid resolution; (b) mean of six RCMs driven with these GCMs. The grey isolines show elevation at 200 m intervals of the underlying model data. Further details on data sources and processing are available in the chapter data table (Table 10.SM.11). Adapted from Giorgi et al. (2016).


Author list:
------------
- Jury, M.W.: BSC, Spain; martin.w.jury@gmail.com; githubid: mwjury
- Dosio, A.: JRC, Italy
- Coppola, E.: ICTP, Italy


Publication sources:
--------------------
Giorgi, F., Torma, C., Coppola, E., Ban, N., Schär, C., & Somot, S. (2016). Enhanced summer convective rainfall at Alpine high elevations in response to climate warming. Nature Geoscience, 9(8), 584–589. https://doi.org/10.1038/ngeo2761


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter_10


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: working_cordex_2.2


Recipe & diagnostics:
---------------------
Recipe used: recipes/ar6_wgi_ch10/recipe_CoppolaAlps.yml

Diagnostic used: diag_scripts/ar6_wgi_ch10/diagnostic_IPCC_AR6_CH10.py


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_CoppolaAlps_YYYYMMDD_HHMMSS/plots/ch_10/fig_10_9/Fig_9.png


Recipe generations tools:
-------------------------
N/A


Ancillary figures and datasets:
-------------------------------
N/A


Additional datasets:
--------------------
'external' data has been included in ESMValTool totalling 40KB:
- esmvaltool/diag_scripts/ar6_wgi_ch10/CH10_additional_data/ECoppola_Alps/GCM.nc
- esmvaltool/diag_scripts/ar6_wgi_ch10/CH10_additional_data/ECoppola_Alps/RCM.nc


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/ar6_chapter_10_conda_environment.yml
- pip file: IPCC_environments/ar6_chapter_10_pip_environment.txt


Hardware description:
---------------------
Internal Wegener Center (University of Graz, Austria) machine wegc203128.

** The documentation was created by Chapter 10 Chapter Scientist Martin W. Jury (email: martin.w.jury@gmail.com, githubid: mwjury). Please, contact Martin in case any questions in documentation arise.