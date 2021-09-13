AR6 WG1 Chapter 3, figure 3.30: Atlantic Meridional Overturning Circulation (AMOC)
=================================================================================

Figure number: Figure 3.30
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![AR6 WG1 Chap3 Figure 3.30 Atlantic Meridional Overturning Circulation (AMOC)](../images/ar6_wg1_chap3_fig3_30_amoc.png?raw=true)


Description:
------------
This is a six pane figure the summarises the behaviour of the
Atlantic Meridional Overturning Circulation in CMIP5, CMIP6 and in the
observational record.

In panes a), b) and c). CMIP5 and CMIP6 are shown in blue and red, respectively,
and the observations are shown in grey bands.

Pane a) shows the depth profile of the AMOC in model and observations.
Pane b) shows the distribution of 8 year trends in the AMOC for the CMIP5 & CMIP6
ensemble means, and for all CMIP6  models that contrinuted toi the mean.
Similarly, pane c) shows the distribution of interannual AMOC changes for the CMIP5 & CMIP6
ensemble means, and for all CMIP6  models that contrinuted toi the mean.

Panes d), e), and f) show the AMOC behaviour of various DAMIP for three different
time periods.


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


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: development_ar6_chap_3_ocean


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: optimize_mem_annual_statistic_plus_amoc


Recipe & diagnostics:
---------------------
Recipe(s) used: recipes/recipe_ocean_amoc_streamfunction_profiles.yml

Diagnostic(s) used: diagocean/diagnostic_amoc_profiles.py


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- plots/diag_timeseries_amoc_hist/AMOC_timeseries/fig_3.24.png

Please note that this filename uses the older incorrect figure number (3.24).


Recipe generations tools:
-------------------------
Were any tools used to populate the recipe? if so what were they? N/A if not applicable.

We used the recipe_filler.py script to populate the dataset list in the recipe,
based on the available data on jasmin/badc in January 2021.


Ancillary figures and datasets:
-------------------------------
In addition to the main figure, diagnostics may produce several figures and datasets
along the way or several versions of the main figure.
Please use this space to highlight anything that may be useful for future iterations:


In addition to the final figure, the AMOC diagnostic can produce a single figure plot for each pane.


Additional datasets:
--------------------
What additional datasets were used to produce this figure?
Where are they on the computational machine or in the respository?
Can they be re-created?
What are their access permissions/Licenses?

The RAPID array dataset (MOC vertical profiles in NetCDF format, moc_vertical.nc) is required.
This data can be found at:  https://www.rapid.ac.uk/rapidmoc/rapid_data/datadl.php
The full doi for this data set is: 10.5285/5acfd143-1104-7b58-e053-6c86abc0d94b

The CMIP6 amoc trends file (Figure_AR6_DAMIP_AMOC_26N_1000m.json) is also required.
This data was produced by Matt Menary outside of ESMValTool
and was downloaded from: https://github.com/mattofficeuk/AR6/tree/master/JSON_data



Software description:
---------------------
Software versions, name of environment file (see **save conda environment** in CONTRIBUTING.md), other software packages,…
- ESMValTool environment file: e.g. IPCC_environments/$NAME_conda_environment.yml
- pip file: e.g. IPCC_environments/$NAME_pip_environment.txt
- Other software used:


Hardware description:
---------------------
What machine was used: Jasmin

When was this machine used:


Any further instructions:
-------------------------
