
DRIVERS OF OBSERVED WARMING
============

Figure number: Figure 3.8
From the IPCC Working Group I Contribution to the Sixth Assessment Report: Chapter 3

![Figure 3.8](../images/ar6_wg1_chap3_fig3_8_drivers_of_observed_warming.png?raw=true)


Description:
------------
Assessed contributions to observed warming, and supporting lines of evidence. Shaded bands show assessed likely ranges of temperature change in GSAT, 2010-2019 relative to 1850-1900, attributable to net human influence, well-mixed greenhouse gases, other human forcings (aerosols, ozone, and land-use change), natural forcings, and internal variability, and the 5-95% range of observed warming. Bars show 5-95% ranges based on (left to right) Haustein et al. (2017), Gillett et al. (2021) and Ribes et al. (2021), and crosses show the associated best estimates. No 5-95% ranges were provided for the Haustein et al. (2017) greenhouse gas or other human forcings contributions. The Ribes et al. (2021) results were updated using a revised natural forcing time series, and the Haustein et al. (2017) results were updated using HadCRUT5. The Chapter 7 best estimates and ranges are derived using assessed forcing time series and a two-layer energy balance model as described in Section 7.3.5.3. Coloured symbols show the simulated responses to the forcings concerned in each of the models indicated. 


Author list:
------------
- Gillett, N: ECCC, Canada, nathan.gillett@canada.ca, npgillett
- Kirchmeier-Young, M: ECCC, Canada
- Cowtan, K: University of York, UK


ESMValTool Branch:
------------------
- ESMValTool-AR6-OriginalCode-FinalFigures: ar6_chapter_3_nathan


Recipe & diagnostics:
---------------------
Recipe used: recipes/recipe_gsat_attribute.yml

Diagnostic used: ipcc_ar6/fig3_8.py

Expected image path:
--------------------
This is the path of the image relative to the automatically generated ESMValTool output location:
- recipe_gsat_attribute_YYYYMMDD_HHMMSS/plots/gillett20_figs/fig3_8/fig3_8.png


Recipe generations tools: 
-------------------------
N/A


Ancillary figures and datasets:
-------------------------------
In addition to the main figure, the diagnostic was used to output the csv file which was afterwards was used by TSU to create FAQ 3.1 Fig 1 and SPM Fig 1 panel (b).

Additional datasets:
--------------------
HadCRUT4 HadCRUT.4.6.0.0.median.nc file should be located in esmvaltool auxiliary directory:
it can be downloaded from https://www.metoffice.gov.uk/hadobs/hadcrut4/data/current/download.html 

In the same auxiliary directory, an AR6_GSAT.csv file with GSAT timeseries from Chapter 2 should be present.    

Another non-esmvaltool preprocessed dataset in esmvaltool auxiliary directory is CNRM-CM6-1-5x5-sftlf.nc, a CNRM-CM6-1 sftlf file regridded with cdo onto 5*5 degrees greed.


Software description:
---------------------
- ESMValTool environment file: IPCC_environments/esmvaltool_ar6_attribution_conda_environment.yml
- pip file: IPCC_environments/esmvaltool_ar6_attribution_pip_environment.txt
- Other software used: cdo==1.5.3


Hardware description:
---------------------
Internal ECCC-CCCma machine lxwrk3.

** The documentation was created by Chapter 3 Chapter Scientist Elizaveta Malinina (email: elizaveta.malinina-rieger@canada.ca, githubid: malininae). Please, contact Elizaveta in case any questions in documentation arise.
