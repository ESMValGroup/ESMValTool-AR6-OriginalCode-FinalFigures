
NAM, NAO and SAM
================

Figure number: Figure 3.33
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.33](../images/ar6_wg1_chap3_figure3_33_nam_nao_sam.png?raw=true)


Description:
------------
Model evaluation of NAM, NAO and SAM in boreal winter. Regression of Mean Sea 
Level Pressure (MSLP) anomalies (in hPa) onto the normalized principal component 
(PC) of the leading mode of variability obtained from empirical orthogonal 
decomposition of the boreal winter (Dec.-Feb) MSLP poleward of 20ºN for the 
observed Northern Annular Mode (NAM, a), over 20ºN-80°N, 90°W-40°E for the North 
Atlantic Oscillation as shown by the black sector (NAO, b), and poleward of 20ºS 
for the Southern Annular Mode (SAM, c) for the JRA-55 reanalysis. Cross marks 
indicate regions where the anomalies are not significant at the 10% level based 
on t-test. The period used to calculate the NAO/NAM is 1958-2014 but 1979-2014 
for the SAM. (d-f) Same but for the multi-model multi-member ensemble (MME) mean 
from CMIP6 historical simulations. Models are weighted in compositing to account 
for differences in their respective ensemble size. Hatching stands for regions 
where less than 80% of the runs agree with the MME sign. (g-i) Taylor diagram 
summarizing the representation of the modes in models and observations following 
Lee et al. (2019) for CMIP5 (light blue) and CMIP6 (red) historical runs. The 
reference pattern is taken from JRA-55 (a-c). The ratio of standard deviation 
(radial distance), spatial correlation (radial angle) and resulting root-mean-
squared-errors (solid isolines) are given for individual ensemble members 
(crosses) and for other observational products (ERA5 and NOAA 20CRv3, black 
dots). Coloured dots stand for weighted MME statistics for CMIP5 (blue) and 
CMIP6 (light red) as well as for AMIP simulations from CMIP6 (orange). (j-l) 
Histograms of the trends built from all individual ensemble members and all the 
models (brown bars). Vertical lines in black show all the observational 
estimates. The orange, light-red, and light blue lines indicate the weighted MME 
of CMIP6 AMIP, CMIP6 and CMIP5 historical simulations, respectively.


Author list:
------------
- Phillips, A.: NCAR, U.S.; asphilli@ucar.edu
- Kosaka, Y.: University of Tokyo, Japan
- Cassous, C.: CNRS-Cerfacs, France
- Karmouche, S.: University of Bremen, Germany
- Bock, L.: DLR, Germany
- Kazeroni, R.: DLR, Germany


Publication sources:
--------------------
Please list any publications that describe, explain or use this figure. 
- A paper title, A. Author et al, journal of science stuff 9, p51, DOI:564, 2021. 


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter_3


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: fix_cmip6_models_newcore


Recipe & diagnostics:
---------------------
Recipe used: recipes/ipccar6wg1ch3/recipe_ipccwg1ar6ch3_modes.yml

Diagnostic used: diag_scripts/ipccwg1ar6ch3_modes/nam_nao_sam.ncl


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_ipccwg1ar6ch3_modes_YYYYMMDD_HHMMSS/plots/nam_nao_sam/nam_nao_sam/nam_nao_sam.png


Recipe generations tools: 
-------------------------
Were any tools used to populate the recipe? if so what were they? N/A if not applicable. 


Ancillary figures and datasets:
-------------------------------
In addition to the main figure, diagnostics may produce several figures and datasets along the way or several versions of the main figure. Please use this space to highlight anything that may be useful for future iterations:


Additional datasets:
--------------------
What additional datasets were used to produce this figure?
Where are they on the computational machine or in the respository?
Can they be re-created?
What are their access permissions/Licenses?


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/ar6_newcore_lisa_conda_environment.yml
- pip file: IPCC_environments/ar6_newcore_lisa_pip_environment.txt


Hardware description:
---------------------
What machine was used: Mistral


Any further instructions: 
-------------------------

