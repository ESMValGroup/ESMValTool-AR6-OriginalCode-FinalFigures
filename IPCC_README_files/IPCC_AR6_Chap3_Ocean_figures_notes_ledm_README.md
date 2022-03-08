
IPCC AR6 Chapter 3 Ocean plots: AMOC, OHC, Halo SLR and SSS trends
==================================================================

ESMValTool description for Intergovernmental Panel on Climate change, Sixth
Assessment Report, Chapter 3 figures:
- Atlantic Meridional Overturning Current ([Figure 3.30](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/images/ar6_wg1_chap3_fig3_30_amoc.png))
- Ocean Heat Content ([Figure 3.26](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/images/ar6_wg1_chap3_figure3_26_oceanheatcontent.png))
- Halosteric Sea Level Rise ([Figure 3.28](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/images/ar6_wg1_chap3_fig3_28_halostericsealevel.png))
- Global Sea Surface Salinity trends ([Figure 3.27](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/images/ar6_wg1_chap3_fig3_27_GlobalSeaSurfaceSalinityTrends.png))

Figures authorship team:
=======================

- Lee de Mora, Plymouth Marine Laboratory, UK; ledm@pml.ac.uk
- Paul J. Durack, Lawrence Livermore National Laboratory, USA; durack1@llnl.gov
- Nathan Gillett, University of Victoria, Canada
- Krishna Achutarao, Indian Institute of Technology, Delhi, India
- Shayne McGregor, Monash University, Melbourne, Australia
- Rondrotiana Barimalala, University of Cape Town, South Africa
- Elizaveta Malinina-Rieger, Environment and Climate Change Canada
- Valeriu Predoi, University of Reading, UK
- Veronika Eyring, DLR, Germany



Table 1:
========


| Name                                    | Fig. | Recipe and Diagnostic                         | Final plot path and  Final Plot name                 |
| --------------------------------------- | ---- | --------------------------------------------- | ---------------------------------------------------- |
| Atlantic Meridional Overturning Current | 3.30 | [recipe_ocean_amoc_streamfunction_profiles.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chap_3_ocean_figures/esmvaltool/recipes/recipe_ocean_amoc_streamfunction_profiles.yml) | diag_timeseries_amoc_hist/AMOC_timeseries            |
|                                         |      | [ocean/diagnostic_amoc_profiles.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chap_3_ocean_figures/esmvaltool/diag_scripts/ocean/diagnostic_amoc_profiles.py)             | fig_3.24                                             |   
| Ocean Heat Content                      | 3.26 | [recipe_ocean_heat_content_TSV_all.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chap_3_ocean_figures/esmvaltool/recipes/recipe_ocean_heat_content_TSV_all.yml)         | plots/diag_ohc/diagnostic/multimodel_ohc             |   
|                                         |      | [ocean/diagnostic_chap3_ocean_heat_content.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chap_3_ocean_figures/esmvaltool/diag_scripts/ocean/diagnostic_chap3_ocean_heat_content.py)  | multimodel_ohc_range_10-90_large_full_1995.0-2014.0  |
| Halosteric Sea Level Rise               | 3.28 | [recipe_ocean_heat_content_TSV_all.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chap_3_ocean_figures/esmvaltool/recipes/recipe_ocean_heat_content_TSV_all.yml)         | plots/diag_ohc/diagnostic/halosteric_multipane/      |
|                                         |      | [ocean/diagnostic_chap3_ocean_heat_content.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chap_3_ocean_figures/esmvaltool/diag_scripts/ocean/diagnostic_chap3_ocean_heat_content.py)  | halosteric_multipane_historical_1950-2015            |
| Global Sea Surface Saliinty trends      | 3.27 | [recipe_ocean_heat_content_TSV_all.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chap_3_ocean_figures/esmvaltool/recipes/recipe_ocean_heat_content_TSV_all.yml)         | plots/diag_ohc/diagnostic/sea_surface_salinity_plot/ |
|                                         |      | [ocean/diagnostic_chap3_ocean_heat_content.py](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/ar6_chap_3_ocean_figures/esmvaltool/diag_scripts/ocean/diagnostic_chap3_ocean_heat_content.py)  | salinity_trends_only_1950-2014_DW1950_decadal        |

Table 1:  all recipes, diagnostics, and paths described in this document.

Notes on paths:

The OHC, Halo and SSS trends plots are all produced using the same recipe and
diagnostic. This is because they all require the same process to de-drift.

The recipes are in the location:
- [ESMValTool_AR6/esmvaltool/recipes](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chap_3_ocean_figures/esmvaltool/recipes)

The diagnostics are in the location:
- [ESMValTool_AR6/esmvaltool/diagnostics/ocean](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chap_3_ocean_figures/esmvaltool/diag_scripts/ocean)

The final plots directory will live in the output directory, generated at run time by ESMValTool.

The final plot name will be appended by the relevant extension, provided by the ESMValTool config-user.yml settings file.
Typically, it will be either .pfd or .png.

The details on the code versions is appended to the end of this file.


Auxiliary Data
==============

Some auxiliary data is required to perform this analysis. In general,
this is either observational data, model data processed elsewhere or
shapefiles used to define specific regions for analysis.

AMOC auxiliary data:
--------------------
The RAPID array dataset (MOC vertical profiles in NetCDF format, moc_vertical.nc) is required.
This data can be found at:  https://www.rapid.ac.uk/rapidmoc/rapid_data/datadl.php
The full doi for this data set is: 10.5285/5acfd143-1104-7b58-e053-6c86abc0d94b

The CMIP6 amoc trends file (Figure_AR6_DAMIP_AMOC_26N_1000m.json) is also required.
This data was produced by Matt Menary outside of ESMValTool
and was downloaded from: https://github.com/mattofficeuk/AR6/tree/master/JSON_data


Halosteric Sea Level auxiliary data:
------------------------------------

The file names are:
 - 210201_EN4.2.1.g10_annual_steric_1950-2019_5-5350m.nc
 - 210201_Ishii17_v7.3_annual_steric_1955-2019_0-3000m.nc
 - 210127_DurackandWijffels_V1.0_70yr_steric_1950-2019_0-2000db_210122-205355_beta.nc
These are the observational datasets that were added to panes a, b, and to panes
c,d and e. The variables steric_height_halo_anom_depthInterp and steric_height_thermo_anom_depthInterp are used.
These files were downloaded directly from Paul Durack
via the invite-only google drive page: https://drive.google.com/drive/folders/1VO2FehHCz1zJu8tLvp1dNPF2IURJudJN

In addition, shapefiles are required to calculate the regional boundaries:
- Pacific.shp
- Atlantic.shp
These regions should be standarized through AR6, and were emailed to me by chapter author Lisa Bock.


Sea surface salinity auxiliary data:
------------------------------------

The observational data from here is taken from: the files:
- DurackandWijffels_GlobalOceanChanges_19500101-20191231__210122-205355_beta.nc'
- DurackandWijffels_GlobalOceanChanges_19700101-20191231__210122-205448_beta.nc
depending on which time range you are looking at.
The field of interest are salinity_mean (shown as black contours) and salinity_change (shown in colourscale).
These files were downloaded directly from Paul Durack
via the invite-only google drive page: https://drive.google.com/drive/folders/1VO2FehHCz1zJu8tLvp1dNPF2IURJudJN


Ocean Heat Content auxiliary data:
----------------------------------

The observational data for this figure is take from the file:
- 210204_0908_DM-AR6FGDAssessmentTimeseriesOHC-v1.csv
All columns are used in the final figure.

These files were downloaded directly from Paul Durack
via the invite-only google drive page: https://drive.google.com/drive/folders/1VO2FehHCz1zJu8tLvp1dNPF2IURJudJN


Auxiliary tools
===============
- check_TSV.py
- recipe_filler.py

These tools are not part of ESMValTool, but are available upon request.
Check_TSV is a tool to generate the dataset list in the recipe_ocean_heat_content_TSV_all.yml recipe.

This tool is relatively complex, as it needs to find all possible cases
where the following six datasets exist for a given model & ensemble member:
- historical temperature (thetao)
- historical salinity (so)
- piControl temperature (thetao)
- piControl salinity (so)
- volcello: both historical anbd piControl for models where volume varies with time.
- volcello: piControl only for models where volume is fixed in time.

The tool checks that the data for all these 5 or 6 datasets must be available
for the entire time range.
In addition, the tool checks where the historical was branched from the piControl
and adds the relevant picontrol years.

The recipe filler is an earlier and more general version of the check_TSV.py tool.
It can be used to add whatever data is available into a recipe. I believe
that a version of it was added to the ESMValTool master by Valeriu Predoi.


Auxiliary figures
=================

In addition to the final figure, the AMOC diagnostic can produce a single figure plot for each pane.

The OHC diagnostic produces the OHC, SSS trends and Halosteric SLR figures.
This code is particularly complex and several ancillary figures are produced along the way
for each model and each ensemble member.

These figures include the following directories related to the de-drifting process and the sea surface salinity trends figure:
  - piControl:
    - maps showing the raw temperature and salinity data at the surface at the final time step of the PI control run.
  - piTrend:
    - histograms showing the distribution of the de-drifting linear regression (slope & intersect)
  - slope:
    - maps showing the slope over the surface for the  entire PI control
  - intersect:
    - maps showing the intersect over the surface for the entire PI control
  - trend_intact:
    - maps showing the raw temperature and salinity data at the surface at the final time step of historical and hist-nat run
  - detrended:
    - maps showing the dedrifted temperature and salinity data at the surface at the final time step of historical and hist-nat run.
  - detrended_quad:
    - 4 pane figure showing the surface map for the historical detrended, trend-intact, the difference and the quotient.
  - vw_timeseries:
    - time series figure showing the volume Weighted mean for the detrended and trend intact.
  - detrending_ts:
    - time series figure showing the global volume weighted mean (or total) temperature, salinity or OHC for the historical and piControl.
  - multi_model_mean:
    - shows maps of the multi-model mean surface temperature and salinity at various points in time and specific time ranges.
  - sea_surface_salinity_plot directory:
    - The full sea surface salinity trends figure.

The following figure directories contain figures for the Dynamic Height calculation:
  - dyn_height_timeseries:
    - Shows a timeseries of the mean dynamic height.
  - slr_height_*_detrended:
    - Surface height map for various time points.
  - SLR_Regional_trend_scatter:
    - scatter plots for the regional thermostericd and halosteric data. Like panes a and b of the halosteric SLR figure.
  - SLR_timeseries_all:
    - time series plots show the time development of each of the total, thermo and halo SLR mean for the global, atlantic and pacific regions.
  - multi_model_agrement_with_*:
    - map showing where the CMIP data agrees with the observations.   
  - halosteric_multipane:
    - The full Halosteric sea level figure.

The following directories contain figures related to the Ocean Heat Content calculation:
  - detrending_ts:
      - time series figure showing the global volume weighted mean (or total) temperature, salinity or OHC for the historical and piControl.
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




Caveats, Bugs and limitations of the current methods:
====================================================

While this code was written for the IPCC report, there are several limitations
and potential sources of error. In this section, we document some potential problems.

This code uses shelve files, which are sometimes not portable between different
versions of python.

We cannot guarantee that the auxiliary data will remain available indefinitely.

If the hatching is turned on in the Halosateric SLR figure, and the multi_model_agrement_with_* figures
do not exist, then the code will try to create a new figure while another is unfinished.
This will break matplotlib.

The dedrifting is calculated using the entire picontrol instead of limiting it
to the period specific in the historical run. Other analyses have used shorter
periods for their dedrifting range. This method was chosen due to the time constraints.

Other analyses have used polymetric dedrifting, to remove even more of the
picontrol trend from the historical run, where as we used a much linear regression
straight line fit.

The DAMIP experiment has the flaw that the Omon wasn't required to
contribute the cell volume. This means that the hist-nat datasets do not include
any time-varying cell volume data. To maximize the data available, we assume that
the hist-nat data can use the mean along of the time axis of the pre-industrial control
data.

We have interchangeably used the terms de-drifting and de-trending, but the
correct term for the process that we've applied is de-drifting. When something
is marked as de-trended, it is actually dedrifted.


Code version details:
=====================

The following branches of ESMValTool-AR6 and ESMValCore were used to produce these figures.

| Code           | Branch                                  | Commit date                   | Commit hash                              |
| -------------- | --------------------------------------- | ----------------------------- | ---------------------------------------- |
| ESMValTool-AR6 | [ar6_chap_3_ocean_figures](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/tree/ar6_chap_3_ocean_figures)                | Mon Mar 8 10:22:43 2021 +0000 | 561349aceb46aedb8b555ab7bab25e029fcddfad |
| ESMValCore     | [optimize_mem_annual_statistic_plus_amoc](https://github.com/ESMValGroup/ESMValCore-AR6-OriginalCode-FinalFigures/tree/optimize_mem_annual_statistic_plus_amoc) | Mon Mar 8 11:46:54 2021 +0000 | 5b744f78a72c2dbbc03141eb39a2b5555dd06220 |


Software description:
---------------------
- ESMValTool environment file: e.g. [IPCC_environments/development_ar6_chap_3_ocean_environment.yml](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/development_ar6_chap_3_ocean_environment.yml)
- pip file: e.g. [IPCC_environments/development_ar6_chap_3_ocean_pip_environment.txt](https://github.com/ESMValGroup/ESMValTool-AR6-OriginalCode-FinalFigures/blob/main/IPCC_environments/development_ar6_chap_3_ocean_pip_environment.txt)


Hardware description:
---------------------
Machine used: Jasmin

