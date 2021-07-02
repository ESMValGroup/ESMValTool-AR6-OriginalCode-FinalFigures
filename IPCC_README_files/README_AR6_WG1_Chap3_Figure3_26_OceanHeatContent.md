
AR6 WG1 Chap3 Figure 3.26 Ocean Heat Content
============================================

Figure number: Figure 3.26
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter X

![AR6 WG1 Chap3 Figure 3.26 Ocean Heat Content](../images/ar6_wg1_chap3_figure3_26_oceanheatcontent.png?raw=true)


Description:
------------
Please describe the figure.

THis is a time series of ocean heat content, with four panes.
On the left, the figure shows the change (relative t o 1971)
in heat content of the total global Ocean.
The red area is the 5-95 percentile range and the grey area is the observational
record.

On the right hand side, the three panes show the depth ranges 0-700m, 700m-2000m,
and below 2000m.

The heat content was calculated using the TEOS-10 gsw (v3.4.0) python toolkit: https://teos-10.github.io/GSW-Python/

In all cases, the model temperature and salinity were de-drifted against the
pi-control. From there, we ensured that the pressure was calculated correctly,
the cell volume was available, we use absolute salinity & conservative temperature.

The multi model mean is caluated such that each modelled ensemble has the
same weight. ie One-model, one vote.

From there the calculation of te "heat content" was:

a) The global ocean heat content is interpreted to be calculated as the volume integral
   of the product of in situ density, ρ , and potential enthalpy, h0 (with reference sea pressure of 0 dbar).
b) The in situ density is calculated using gsw_rho(SA,CT,p).  
    Here the actual pressure at the target depth is used (i.e., the mass of the water).
c) The *surface referenced* enthalpy should be calculated using gsw_enthalpy(SA,CT,0).
   Note that here the 0 dbar pressure is critical to arrive at the surface value,
   which is the value of enthalpy and absolute salinity that is available for exchange with the atmosphere.
d) The product of the in situ density times the surface-referenced enthalpy is
   the relevant energy quantity: gsw_rho(SA,CT,p)*gsw_enthalpy(SA,CT,0)
e) For the anomalous energy, we calculate:
   gsw_rho(SA,CT,p)*gsw_enthalpy(SA,CT,0)-gsw_rho(<SA>,<CT>,p)*gsw_enthalpy(<SA>,<CT>,0).
f) integrate the surface-referenced enthalpy times rho,
   i.e., the previous line gives the 3D integrand, and then we want to integrate it from the bottom up.  


Author list:
------------
- Lee de Mora, Plymouth Marine Laboratory, ledm@pml.ac.uk, github: ledm
- Paul J. Durack, durack1@llnl.gov
- Nathan Gillett, nathan.gillett@canada.ca
- Krishna Achutarao, krishna.achutarao@gmail.com
- Shayne McGregor, shayne.mcgregor@monash.edu
- Rondrotiana Barimalala, rondrotiana.barimalala@uct.ac.za
- Elizaveta Malinina-Rieger, elizaveta.malinina-rieger@canada.ca
- Valeriu Predoi, valeriu.predoi@ncas.ac.uk
- Veronika Eyring, Veronika.Eyring@dlr.de


Publication sources:
--------------------
Please list any publications that describe, explain or use this figure.
- Durack, P., Gleckler, P., Landerer, F. et al. Quantifying underestimates of long-term upper-ocean warming. Nature Clim Change 4, 999–1005 (2014). https://doi.org/10.1038/nclimate2389



ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: development_ar6_chap_3_ocean


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: optimize_mem_annual_statistic_plus_amoc


Recipe & diagnostics:
---------------------
Recipe(s) used: e.g. recipes/recipe_ocean_heat_content_TSV_all.yml
Please describe this recipe:

Diagnostic(s) used: e.g. ocean/diagnostic_chap3_ocean_heat_content.py
Please describe this diagnostic:

The OHC, Halo and SSS trends plots are all produced using tyhe same recipe and
diagnostic. This is because they all require the same process to de-drift.

Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- plots/diag_ohc/diagnostic/multimodel_ohc/ multimodel_ohc_range_10-90_large_full_1995.0-2014.0.png



Recipe generation tools:
-------------------------
Were any tools used to populate the recipe? if so what were they? N/A if not applicable.

Two scripts are included to populate this recipe:

- check_TSV.py
- recipe_filler.py

Check_TSV is a tool to generate the dataset list in the recipe_ocean_heat_content_TSV_all.yml recipe.

This tool is relatively complex, as it needs to find all possible cases
where the following six datasets exist for a given model & ensemnle member:
- historical temperature (thetao)
- historical salinity (so)
- piControl temperature (thetao)
- piControl salinity (so)
- volcello: both historical and piControl for models where volume varies with time.
- volcello: piControl only for models where volume is fixed in time.

The tool checks that the data for all these 5 or 6 datasets must be available
for the entire time range.
In addition, the tool checks where the historical was branched from the piControl
and adds the relevant picontrol years.

The recipe filler is an earlier and more general version of the check_TSV.py tool.
It can be used to add whatever data is available into a recipe. I believe
that a version of it was added to the ESMValTool master by Valeriu.



Ancillary figures and datasets:
-------------------------------
In addition to the main figure, diagnostics may produce several figures and datasets
along the way or several versions of the main figure. Please use this space to highlight anything that may be useful for future iterations:

The OHC diagnostic produces the OHC, SSS trends and Halosteric SLR figures.
This code is particularly complex and several ancillary figures are produced along the way
for each model and each ensemble member.

These figures include the following directories related to the de-derifting process:
  - piControl:
    - maps showing the raw temperature and salinity data at the surface at the final time step of the PI control run.
  - piTrend:
    - histograms showing the distributiuon of the de-drifting linear regression (slope & intersect)
  - slope:
    - maps showing the slope over the surface for the  entire PI control
  - intersect:
    - maps showing the intersect overthe surface for the entire PI control
  - trend_intact:
    - maps showing the raw temperature and salininty data at the surface at the final time step of historical and hist-nat run
  - detrended:
    - maps showing the dedrifted temperature and salininty data at the surface at the final time step of historical and hist-nat run.
  - detrended_quad:
    - 4 pane figure showing the surface map for the historical detrended, trend-intact, the difference and the quoitent.
  - vw_timeseries:
    - time series figure showing the volume Weighted mean for the detrended and trend intact.
  - detrending_ts:
    - time series figure showing the global volume weighted mean (or total) temperature, salininity or OHC for the historical and piControl.
  - multi_model_mean:
    - shows maps of the multi-model mean surface temperature and salinity at various points in time and specific time ranges.


THe following directories contain figures related to the Ocean Heat Content calculation:
  - detrending_ts:
      - time series figure showing the global volume weighted mean (or total) temperature, salininity or OHC for the historical and piControl.
  - ohc_summary:
    - Single model ensemble version of the final figure, showing each volume range.
  - OHC_full_instact and OHC_full_detrended:
    - map showing the full water column OHC for each ensemble member at various points in time.
  - ohc_ts:
    - single model time series figure showing the time development of the detrended OHC.
  - dynheight_ohcAtlantic & dynheight_ohcPacific:
    - map showing the dynamic height in the Atlantic and pacific regions (useful to heightlift the regional maps.)
  - multimodel_ohc:
    - The full ocean heat content figure.



Additional datasets:
--------------------
What additional datasets were used to produce this figure?
Where are they on the computational machine or in the respository?
Can they be re-created?
What are their access permissions/Licenses?


The observational data for this figure is take from the file:
- 210204_0908_DM-AR6FGDAssessmentTimeseriesOHC-v1.csv
All columns are used in the final fiugre.

These files were downloaded directly from Paul Durack
via the invite-only google drive page: https://drive.google.com/drive/folders/1VO2FehHCz1zJu8tLvp1dNPF2IURJudJN


Software description:
---------------------
Software versions, name of environment file (see **save conda environment** in CONTRIBUTING.md), other software packages,…
- ESMValTool environment file: e.g. IPCC_environments/development_ar6_chap_3_ocean_environment.yml
- pip file: e.g. IPCC_environments/development_ar6_chap_3_ocean_pip_environment.txt
- Other software used:


Hardware description:
---------------------
What machine was used: Jasmin
When was this machine used? December 2020 to March 2021


Any further instructions:
-------------------------

While this code was written for the IPCC report, there are several limitations
and potential sources of error. In this section, we document some potential problems.

This code uses shelve files, which are sometimes not portable between different
versions of python.

We can not guarentee that the auxiliary data will remain available indefinately.

If the hatching is turned on in the Halosateric SLR figure, and the multi_model_agrement_with_* figures
do not exist, then the code will try to create a new figure while another is unfinished.
This will break matplotlib.

The dedrifting is calculated using the entire picontrol instead of limiting it
to the period specific in the historical run. Other analyses have used shorter
periods for their dedrifting range. This method was chosen due to the time constraints.

Other analyses have used polymetric dedrifting, to remove even more of the
picontrol trend from the hisotircal run, where as we used a much linear regression
straight line fit.

The DAMIP experiment has the flaw that the Omon wasn't required to
contribue the cell volume. This means that the hist-nat datasets do not include
any time-variying cell volume data. To maximise the data available, we assume that
the hist-nat data can use the mean along of the time axis of the pre-industrial control
data.

We have interchangably used the terms de-drifting and de-trending, but the
correct term for the process that we've applied is de-drifting. When something
is marked as de-trended, it is actually dedrifted.
