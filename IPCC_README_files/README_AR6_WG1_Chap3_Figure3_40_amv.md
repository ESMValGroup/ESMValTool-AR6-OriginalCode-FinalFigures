
AMV
===

Figure number: Figure 3.40
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.40](../images/ar6_wg1_chap3_figure3_40_amv.png?raw=true)


Description:
------------
Model evaluation of the Atlantic Multi-decadal Variability (AMV). (a, b) Sea 
surface temperature (SST) anomalies (ºC) regressed onto the AMV index defined 
as the 10-year low-pass filtered North Atlantic (0º-60°N, 80°W-0°E) area-weighted 
SST* anomalies over 1900-2014 in (a) ERSSTv5 and (b) the CMIP6 multi-model 
ensemble (MME) composite obtained by weighting ensemble members by the 
inverse of each model’s ensemble size. The asterisk denotes that the global 
mean SST anomaly has been removed at each time step of the computation. Cross 
marks in (a) represent regions where the anomalies are not significant at the 
10% level based on a t-test. Diagonal lines in (b) show regions where less than 
80% of the runs agree in sign. (c) A Taylor diagram summarizing the representation 
of the AMV pattern in CMIP5 (each member is shown as a cross in light blue, and 
the weighted multi-model mean is shown as a dot in dark blue), CMIP6 (each member 
is shown as a cross in red, and the weighted multi-model mean is shown as a dot 
in orange) and observations over [0º-60°N, 80°W-0°E]. The reference pattern is 
taken from ERSSTv5 and black dots indicate other observational products (HadISSTv1 
and COBE-SST2). (d) Autocorrelation of unfiltered annual AMV index at lag 1 year 
and 10-year low-pass filtered AMV index at lag 10 years for observations over 
1900-2014 (horizontal lines) and 115-year chunks of pre-industrial control 
simulations (open boxes) and individual historical simulations over 1900-2014 
(filled boxes) from CMIP5 (blue) and CMIP6 (red). (e) As in (d), but showing 
standard deviation of the unfiltered and filtered AMV indices (ºC). Boxes and 
whiskers show the weighted multi-model mean, interquartile ranges and 5th and 
95th percentiles. (f) Time series of the AMV index (ºC) in ERSSTv5, HadISSTv1 
and COBE-SST2 observational estimates (black) and CMIP5 and CMIP6 historical 
simulations. The thick red and light blue lines are the weighted multi-model mean 
for the historical simulations in CMIP5 and CMIP6, respectively, and the envelopes 
represent the 5th-95th percentile range obtained from all ensemble members. The 
5-95% confidence interval for the CMIP6 MME is shown by the thin dashed line. 


Author list:
------------
- Phillips, A.: NCAR, USA; asphilli@ucar.edu
- Kosaka, Y.: University of Tokyo, Japan
- Cassou, C.: CNRS-Cerfacs, France
- Karmouche, S.: University of Bremen, Germany
- Bock, L.: DLR, Germany
- Kazeroni, R.: DLR, Germany


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter_3


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: fix_cmip6_models_newcore


Recipe & diagnostics:
---------------------
Recipe used: recipes/ipccwg1ar6ch3/recipe_ipccwg1ar6ch3_modes.yml

Diagnostic used: diag_scripts/ipccwg1ar6ch3_modes/amv.ncl


Expected image path:
--------------------
- recipe_ipccwg1ar6ch3_modes_YYYYMMDD_HHMMSS/plots/pdv-amv/amv/amv.pdf


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/esmvaltool_ar6_yu_conda_environment.yml
- pip file: IPCC_environments/esmvaltool_ar6_yu_pip_environment.txt


Hardware description:
---------------------
Machine used: avocado.atmos.rcast.u-tokyo.ac.jp
