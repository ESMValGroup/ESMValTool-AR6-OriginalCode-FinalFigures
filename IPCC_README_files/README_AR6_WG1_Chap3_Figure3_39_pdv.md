
PDV
===

Figure number: Figure 3.39
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.39](../images/ar6_wg1_chap3_figure3_39_pdv.png?raw=true)


Description:
------------
Model evaluation of the PDV. (a, b) SST anomalies (ºC) regressed onto the 
Tripole Index (TPI; Henley et al., 2015) for 1900-2014 in (a) ERSSTv5 and (b) 
CMIP6 MME composite obtained by weighting ensemble members by the inverse of the 
model ensemble size. A 10-year low-pass filter was applied beforehand. Cross 
marks in (a) represent regions where the anomalies are not significant at the 
10% level based on t-test. Hatching in (b) indicates regions where less than 80% 
of the runs agree with the MME in sign. (c) A Taylor diagram summarizing the 
representation of the PDV pattern in CMIP5 (each a cross in light blue, and the 
weighted MME as a dot in dark blue), CMIP6 (each ensemble member is shown as a 
cross in red, weighted MME as a dot in orange) and observations over [40ºS-60ºN, 
110ºE-70ºW]. The reference pattern is taken from ERSSTv5 and black dots indicate 
other observational products (HadISSTv1 and COBE-SST2). (d) Autocorrelation of 
unfiltered annual TPI at lag 1 year and 10-year low-pass filtered TPI at lag 10 
years for observations over 1900-2014 (horizontal lines) and 115-year chunks of 
pre-industrial control simulations (open boxes) and individual historical 
simulations over 1900-2014 (filled boxes) from CMIP5 (blue) and CMIP6 (red). (e) 
As in (d), but standard deviation of the unfiltered and filtered TPI (ºC). Boxes 
and whiskers show weighted MME mean, interquartile ranges and 5th and 95th 
percentiles. (f) Time series of the 10-year low-pass filtered TPI (ºC) in 
ERSSTv5, HadISSTv1 and COBE-SST2 observational estimates (black) and CMIP5 and 
CMIP6 historical simulations. The thick red and light blue lines are the 
weighted MME mean for the historical simulations in CMIP5 and CMIP6, 
respectively, and the envelopes represent the 5th-95th percentile range across 
ensemble members. The 5-95% confidence interval for the CMIP6 MME is given in 
thin dashed lines.


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

Diagnostic used: diag_scripts/ipccwg1ar6ch3_modes/pdv.ncl


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_ipccwg1ar6ch3_modes_YYYYMMDD_HHMMSS/plots/pdv-amv/pdv/pdv.pdf


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
- ESMValTool environment file: IPCC_environments/ar6_newcore_remi_conda_environment.yml
- pip file: IPCC_environments/ar6_newcore_remi_pip_environment.txt


Hardware description:
---------------------
Machine used: Mistral


Any further instructions: 
-------------------------

