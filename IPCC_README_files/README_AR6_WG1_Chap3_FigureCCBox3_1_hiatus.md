
15-YEAR TRENDS OF SURFACE GLOBAL WARMING
========================================

Figure number: Figure CC-Box 3.1
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure CCBox 3.1](../images/ar6_wg1_chap3_figure_ccbox3_1.png?raw=true)


Description:
------------
15-year trends of surface global warming for 1998-2012 and 2012-2026. (a, b) 
GSAT and GMST trends for 1998-2012 (a) and 2012-2026 (b). Histograms are based 
on GSAT in historical simulations of CMIP6 (red shading, extended by SSP2-4.5) 
and CMIP5 (grey shading; extended by RCP4.5). Filled and open diamonds at the 
top represent multi-model ensemble means of GSAT and GMST trends, respectively. 
Hatching shows histograms of HadCRUT5.0.1.0. Triangles at the top of (a) 
represent GMST trends of Berkeley Earth, GISTEMP, Kadow et al. (2020) and 
NOAAGlobalTemp-Interim, and the GSAT trend of ERA5. Selected CMIP6 members whose 
1998-2012 trends are lower than the HadCRUT5.0.1.0 mean trend are indicated by 
purple shading (a) and (b). In (a), model GMST and GSAT, and ERA5 GSAT are 
masked to match HadCRUT data coverage. (c-d) Trend maps of annual near-surface 
temperature for 1998-2012 based on HadCRUT5.0.1.0 mean (c) and composited 
surface air temperature trends of subsampled CMIP6 simulations (d) that are 
included in purple shading area in (a). In (c), cross marks indicate trends that 
are not significant at the 10% level based on t-tests with serial correlation 
taken into account. Ensemble size used for each of the histograms and the trend 
composite is indicated at the top right of each of panels (a,b,d). Model 
ensemble members are weighted with the inverse of the ensemble size of the same 
model, so that individual models are equally weighted.


Author list:
------------
- Kosaka, Y.: University of Tokyo, Japan; ykosaka@atmos.rcast.u-tokyo.ac.jp
- Kazeroni, R.: DLR, Germany


Publication sources:
--------------------
Please list any publications that describe, explain or use this figure. 
- A paper title, A. Author et al, journal of science stuff 9, p51, DOI:564, 2021. 


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chap_3_hiatus


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: branch_name


Recipe & diagnostics:
---------------------
Recipe used: recipes/ipccar6wg1ch3/recipe_ipccwg1ar6ch3_fig_ccb3_1.yml

Diagnostics used: 
- diag_scripts/ar6ch3_hiatus/trend.ncl
- diag_scripts/ar6ch3_hiatus/composite.ncl
- diag_scripts/ar6ch3_hiatus/draw.ncl
- diag_scripts/ar6ch3_hiatus/summary_percentiles.ncl


Expected image path:
--------------------
- recipe_ipccwg1ar6ch3_fig_ccb3_1_YYYYMMDD_HHMMSS/plots/trend_hiatus_posthiatus_5x5/plot_refHadCRUT5_HadCRUT5mean/hiatus_and_posthiatus_pdfs.pdf


Recipe generations tools: 
-------------------------
Were any tools used to populate the recipe? if so what were they? N/A if not applicable. 
N/A


Ancillary figures and datasets:
-------------------------------
In addition to Cross-Chapter Box 3.1 Figure 1, This recipe will create 
tables summarizing percentiles values of 1998-2012 GMST trends in CMIP5 and CMIP6 
calculated based on the resolution, anomaly definition, and masking of individual 
observational datasets.


Additional datasets:
--------------------
What additional datasets were used to produce this figure?
Where are they on the computational machine or in the respository?
Can they be re-created?
What are their access permissions/Licenses?


Software description:
---------------------
Software versions, name of environment file (see **save conda environment** in CONTRIBUTING.md), other software packages,…
- ESMValTool environment file: IPCC_environments/esmvaltool_ar6_yu_conda_environment.yml
- pip file: IPCC_environments/esmvaltool_ar6_yu_pip_environment.txt


Hardware description:
---------------------
What machine was used:  avocado.rcast.u-tokyo.ac.jp


Any further instructions: 
-------------------------

