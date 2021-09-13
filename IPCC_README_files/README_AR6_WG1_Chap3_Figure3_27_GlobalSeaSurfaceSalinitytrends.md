AR6 WG1 Chapter 3 Figure 3.27 Global Sea Surface Salinity Trends
================================================================

Figure number: Figure 3.27
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.27:  Global Sea Surface Salinity Trends](../images/ar6_wg1_chap3_fig3_27_GlobalSeaSurfaceSalinityTrends.png?raw=true)


Description:
------------

This is a two pane figure where both panes show:
 - the trend in the near surface salinity (colour scale)
 - the mean near surface salinity over the time period (black contours).

The top pane shows the observational data and the bottom pane shows the multi-model
CMIP6 mean.

The model sea surface salinity was de-drifted against the pi-control.
The multi model mean is caluated such that each modelled ensemble has the
same weight. ie One-model, one vote.


Author list:
------------
- Lee de Mora, Plymouth Marine Laboratory, ledm@pml.ac.uk
- Paul J. Durack, Lawrence Livermore National Laboratory,  durack1@llnl.gov
- Nathan Gillett, University of Victoria
- Krishna Achutarao, Indian Institute of Technology, Delhi
- Shayne McGregor, Monash University, Melbourne
- Rondrotiana Barimalala, University of Cape Town
- Elizaveta Malinina-Rieger, Environment and Climate Change Canada
- Valeriu Predoi, University of Reading
- Veronika Eyring, German Aerospace Center (DLR)


Publication sources:
--------------------
Please list any publications that describe, explain or use this figure.
- Durack, Paul J., Susan E. Wijffels, 2010: Fifty-Year Trends in Global Ocean
  Salinities and Their Relationship to Broad-Scale Warming. J. Climate, 23, 4342–4362.


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: development_ar6_chap_3_ocean


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: optimize_mem_annual_statistic_plus_amoc


Recipe & diagnostics:
---------------------
Recipe(s) used:
- recipes/recipe_ocean_heat_content_TSV_all.yml

Diagnostic(s) used:
- ocean/diagnostic_chap3_ocean_heat_content.py


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- plots/diag_ohc/diagnostic/sea_surface_salinity_plot/alinity_trends_only_1950-2014_DW1950_decadal.png


Recipe generations tools:
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
- volcello: both historical anbd piControl for models where volume varies with time.
- volcello: piControl only for models where volume is fixed in time.

The tool checks that the data for all these 5 or 6 datasets must be available
for the entire time range.
In addition, the tool checks where tyhe historical was branched from the piControl
and adds the relevant picontrol years.

The recipe filler is an earlier and more general version of the check_TSV.py tool.
It can be used to add whatever data is available into a recipe. I believe
that a version of it was added to the ESMValTool master by Valeriu.


Ancillary figures and datasets:
-------------------------------
In addition to the main figure, diagnostics may produce several figures and datasets along the way or several versions of the main figure. Please use this space to highlight anything that may be useful for future iterations:


The OHC diagnostic produces the OHC, SSS trends and Halosteric SLR figures.
This code is particularly complex and several ancillairy figures are produced along the way
for each model and each ensemble member.

These figures include the following directories related to the de-derifting process and the sea surface salinity trends figure:
  - piControl:
    - maps showing the raw temperature and salinity data at the surface at the final time step of the PI control run.
  - piTrend:
    - histograms showing the distributiuon of the de-drifting linear regression (slope & intersect)
  - slope:
    - maps showing the slope over the surface for the  entire PI control
  - intersect:
    - maps showing the intersect overthe surface for the entire PI control
  - trend_intact:
    - maps showing the raw temperature and salinity data at the surface at the final time step of historical and hist-nat run
  - detrended:
    - maps showing the dedrifted temperature and salinity data at the surface at the final time step of historical and hist-nat run.
  - detrended_quad:
    - 4 pane figure showing the surface map for the historical detrended, trend-intact, the difference and the quoitent.
  - vw_timeseries:
    - time series figure showing the volume Weighted mean for the detrended and trend intact.
  - detrending_ts:
    - time series figure showing the global volume weighted mean (or total) temperature, salinity or OHC for the historical and piControl.
  - multi_model_mean:
    - shows maps of the multi-model mean surface temperature and salinity at various points in time and specific time ranges.
  - sea_surface_salinity_plot directory:
    - The full sea surface salinity trends figure.


Additional datasets:
--------------------
What additional datasets were used to produce this figure?
Where are they on the computational machine or in the respository?
Can they be re-created?
What are their access permissions/Licenses?

The observational data from here is taken from: the files:
- DurackandWijffels_GlobalOceanChanges_19500101-20191231__210122-205355_beta.nc'
- DurackandWijffels_GlobalOceanChanges_19700101-20191231__210122-205448_beta.nc
depending on which time range you are looking at.
The field of interest are salinity_mean (shown as black contours) and salinity_change (shown in colourscale).
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
