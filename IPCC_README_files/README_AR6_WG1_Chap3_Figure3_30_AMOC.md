
AR6 WG1 Chapter 3, figure .30: Atlantic Meridional Overturning Circulation (AMOC)
=================================================================================

Figure number: Figure 3.30
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![AR6 WG1 Chap3 Figure 3.30](../images/ar6_wg1_chap3_fig3_30_amoc.png?raw=true)


Description:
------------
This


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
- A paper title, A. Author et al, journal of science stuff 9, p51, DOI:564, 2021.


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: development_ar6_chap_3_ocean


ESMValCore Branch:
------------------
- ESMValCore-AR6-OriginalCode-FinalFigures: branch_name


Recipe & diagnostics:
---------------------
Recipe(s) used: e.g. recipes/recipe_ocean_amoc_streamfunction_profiles.yml
Please describe this recipe:

Diagnostic(s) used: e.g. diagocean/diagnostic_amoc_profiles.py
Please describe this diagnostic:


Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- plots/diagnostic_name/diag_timeseries_amoc_hist/AMOC_timeseries/fig_3.24.png

Please note that this filename uses the older incorrect figure number (3.24).


Recipe generations tools:
-------------------------
Were any tools used to populate the recipe? if so what were they? N/A if not applicable.

We used the recipe_filler.py script to populate the dataset list in the recipe,
based on the available data on jasmin/badc in January 2021.


Ancillary figures and datasets:
-------------------------------
In addition to the main figure, diagnostics may produce several figures and datasets along the way or several versions of the main figure. Please use this space to highlight anything that may be useful for future iterations:


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
