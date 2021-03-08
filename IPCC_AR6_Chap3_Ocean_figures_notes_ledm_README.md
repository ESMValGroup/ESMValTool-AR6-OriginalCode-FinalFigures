
IPCC AR6 Chapter 3 Ocean plots: AMOC, OHC, Halo SLR and SSS trends
==================================================================

ESMValTool decription for Intergovernmentalk Panel on Cliamte change, Sixth
Assessmenent Report, Chapter 3 figures:
- Atlantic Meridional Overturning Current  (figure 3.XX )
- Ocean Heat Content (figure 3.XX )
- Halosteric Sea Level Rise (figure 3.XX )
- Global Sea Surface Saliinty trends  (figure 3.XX )

Figures authorship team:
=======================

- Lee de Mora, ledm@pml.ac.uk
- Paul J. Durack, durack1@llnl.gov
- Nathan Gillett, nathan.gillett@canada.ca
- Krishna Achutarao, krishna.achutarao@gmail.com
- Shayne McGregor, shayne.mcgregor@monash.edu
- Rondrotiana Barimalala, rondrotiana.barimalala@uct.ac.za
- Elizaveta Malinina-Rieger, elizaveta.malinina-rieger@canada.ca
- Valeriu Predoi, valeriu.predoi@ncas.ac.uk
- Veronika Eyring, Veronika.Eyring@dlr.de


Table 1:
========

| --------------------------------------- | ---- | --------------------------------------------- | ---------------------------------------------------- |
| Name                                    | Fig. | Recipe and Diagnostic                         | Final plot path and  Final Plot name                 |
| --------------------------------------- | ---- | --------------------------------------------- | ---------------------------------------------------- |
| Atlantic Meridional Overturning Current | 3.XX | recipe_ocean_amoc_streamfunction_profiles.yml | diag_timeseries_amoc_hist/AMOC_timeseries            |
|                                         |      | ocean/diagnostic_amoc_profiles.py             | fig_3.24                                             |      
| --------------------------------------- | ---- | --------------------------------------------- | ---------------------------------------------------- |
| Ocean Heat Content                      | 3.XX | recipe_ocean_heat_content_TSV_all.yml         | plots/diag_ohc/diagnostic/multimodel_ohc             |   
|                                         |      | ocean/diagnostic_chap3_ocean_heat_content.py  | multimodel_ohc_range_10-90_large_full_1995.0-2014.0  |
| --------------------------------------- | ---- | --------------------------------------------- | ---------------------------------------------------- |
| Halosteric Sea Level Rise               | 3.XX | recipe_ocean_heat_content_TSV_all.yml         | plots/diag_ohc/diagnostic/halosteric_multipane/      |
|                                         |      | ocean/diagnostic_chap3_ocean_heat_content.py  | halosteric_multipane_historical_1950-2015            |
| --------------------------------------- | ---- | --------------------------------------------- | ---------------------------------------------------- |
| Global Sea Surface Saliinty trends      | 3.XX | recipe_ocean_heat_content_TSV_all.yml         | plots/diag_ohc/diagnostic/sea_surface_salinity_plot/ |
|                                         |      | ocean/diagnostic_chap3_ocean_heat_content.py  | salinity_trends_only_1950-2014_DW1950_decadal        |
| --------------------------------------- | ---- | --------------------------------------------- | ---------------------------------------------------- |
Table 1:  all recipes, diagnostics, and paths described in this document.

Notes on paths:

The OHC, Halo and SSS trends plots are all produced using tyhe same recipe and
diagnostic. This is because they all require the same process to de-dedrift.

The recipes are in the location:
- ESMValTool_AR6/esmvaltool/recipes

THe diagnostics are in the location:
- ESMValTool_AR6/esmvaltool/diagnostics/ocean

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
These regions should be standarised throught AR6, and were emailed to me by chapter author Liza Bock.


Sea surface salininty auxiliary data:
------------------------------------

The observational data from here is taken from: the files:
- DurackandWijffels_GlobalOceanChanges_19500101-20191231__210122-205355_beta.nc'
- DurackandWijffels_GlobalOceanChanges_19700101-20191231__210122-205448_beta.nc
depending on which time range you are looking at.
THe field of interest are salinity_mean (shown as black contours) and salinity_change (shown in colourscale).
These files were downloaded directly from Paul Durack
via the invite-only google drive page: https://drive.google.com/drive/folders/1VO2FehHCz1zJu8tLvp1dNPF2IURJudJN


Ocean Heat Content auxiliary data:
----------------------------------

The observational data for this figure is take from the file:
- 210204_0908_DM-AR6FGDAssessmentTimeseriesOHC-v1.csv
All columns are used in the final fiugre.

These files were downloaded directly from Paul Durack
via the invite-only google drive page: https://drive.google.com/drive/folders/1VO2FehHCz1zJu8tLvp1dNPF2IURJudJN


Auxiliary tools
===============
- check_TSV.py
- recipe_filler.py

These tools are not part of ESMValTool, but are available upon request.
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


Auxiliary figures
=================

In addition to the final figure, the AMOC diagnostic can produce a single figure plot for each pane.

The OHC diagnostic produces the OHC, SSS trends and Halosteric SLR figures.
This code is particularly complex and several ancillairy figures are produced along the way
for each model and each ensemble member.

These figures include the following directories related to the de-derifting process and the sea surface salininity trends figure:
  - piControl:
    - maps showing the raw temperature and salininty data at the surface at the final time step of the PI control run.
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
  - sea_surface_salinity_plot directory:
    - The full sea surface salininty trends figure.

The following figure directories contain figures for the Dynamic Height calculation:
  - dyn_height_timeseries:
    - Shows a timeseries of the mean dynamic height.
  - slr_height_*_detrended:
    - Surface height map for various time points.
  - SLR_Regional_trend_scatter:
    - scatter plots for the regional thermostericd and halosteric data. Like panes a and b of the halosteric SLR figure.
  - SLR_timeseries_all:
    - time series plots shows the time development of each of the total, thermo and halo SLR mean for the global, atlantic and pacific regions.
  - multi_model_agrement_with_*:
    - map showing where the CMIP data agrees with the observations.   
  - halosteric_multipane:
    - The full Halosteric sea level figure.

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




Caveats, Bugs and limitations of the current methods:
====================================================

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







Code version details:
=====================

The following branches of ESMValTool-AR6 and ESMValCore were used to produce these figures.

| Code           | Branch                                  | Commit date                   | Commit hash                              | Tag |
| ESMValTool-AR6 | ar6_chap_3_ocean_figures                | Mon Mar 8 10:22:43 2021 +0000 | 561349aceb46aedb8b555ab7bab25e029fcddfad |     |
| ESMValCore     | optimize_mem_annual_statistic_plus_amoc | Mon Mar 8 11:46:54 2021 +0000 | 5b744f78a72c2dbbc03141eb39a2b5555dd06220 |     |


Conda log
---------
Please find below the full list of packages in the environment used.

# packages in environment at /home/users/ldemora/anaconda3_20190821/envs/ar6:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                      1_llvm    conda-forge
_r-mutex                  1.0.1               anacondar_1    conda-forge
alabaster                 0.7.12                   pypi_0    pypi
antlr-python-runtime      4.7.2                 py38_1001    conda-forge
astroid                   2.4.1                    pypi_0    pypi
attrs                     19.3.0                     py_0    conda-forge
autodocsumm               0.1.13                   pypi_0    pypi
babel                     2.8.0                    pypi_0    pypi
binutils-meta             1.0.4                         0    conda-forge
binutils_impl_linux-64    2.34                 h53a641e_5    conda-forge
binutils_linux-64         2.34                hc952b39_20    conda-forge
bokeh                     2.1.1            py38h32f6830_0    conda-forge
boost-cpp                 1.72.0               h8e57a91_0    conda-forge
brotlipy                  0.7.0           py38h1e0a361_1000    conda-forge
bwidget                   1.9.14                        0    conda-forge
bzip2                     1.0.8                h516909a_2    conda-forge
c-compiler                1.0.4                h516909a_0    conda-forge
ca-certificates           2020.6.20            hecda079_0    conda-forge
cairo                     1.16.0            hcf35c78_1003    conda-forge
cartopy                   0.17.0          py38h9cf8511_1015    conda-forge
cdo                       1.5.3                    pypi_0    pypi
cdsapi                    0.3.0                    pypi_0    pypi
certifi                   2020.6.20        py38h32f6830_0    conda-forge
cf-units                  2.1.4            py38h8790de6_0    conda-forge
cffi                      1.14.0           py38hd463f26_0    conda-forge
cfitsio                   3.470                h3eac812_5    conda-forge
cftime                    1.2.0            py38h8790de6_1    conda-forge
chardet                   3.0.4           py38h32f6830_1006    conda-forge
click                     7.1.2              pyh9f0ad1d_0    conda-forge
click-plugins             1.1.1                      py_0    conda-forge
cligj                     0.5.0                    pypi_0    pypi
cloudpickle               1.5.0                      py_0    conda-forge
cmocean                   2.0                      pypi_0    pypi
codespell                 1.17.1                   pypi_0    pypi
colorama                  0.4.3                    pypi_0    pypi
compilers                 1.0.4                         0    conda-forge
coverage                  5.2                      pypi_0    pypi
cryptography              2.9.2            py38h766eaa4_0    conda-forge
curl                      7.71.1               he644dc0_0    conda-forge
cxx-compiler              1.0.4                hc9558a2_0    conda-forge
cycler                    0.10.0                     py_2    conda-forge
cython                    0.29.20          py38h950e882_0    conda-forge
cytoolz                   0.10.1           py38h516909a_0    conda-forge
dask                      2.20.0                     py_0    conda-forge
dask-core                 2.20.0                     py_0    conda-forge
decorator                 4.4.2                      py_0    conda-forge
distributed               2.20.0           py38h32f6830_0    conda-forge
docutils                  0.16                     pypi_0    pypi
dodgy                     0.2.1                    pypi_0    pypi
easytest                  0.1.5                    pypi_0    pypi
eccodes                   2.17.0               h59f7be3_1    conda-forge
ecmwf-api-client          1.5.4                    pypi_0    pypi
eofs                      1.4.0                    pypi_0    pypi
esmf                      8.0.0           mpi_mpich_h9a42a66_106    conda-forge
esmpy                     8.0.0           mpi_mpich_py38ha9b28fa_101    conda-forge
esmvalcore                2.0.0                     dev_0    <develop>
esmvaltool                2.0.0b4                   dev_0    <develop>
expat                     2.2.9                he1b5a44_2    conda-forge
fftw                      3.3.8           nompi_h7f3a6c3_1111    conda-forge
fiona                     1.8.13.post1             pypi_0    pypi
fire                      0.3.1                    pypi_0    pypi
flake8                    3.8.3                    pypi_0    pypi
flake8-polyfill           1.0.2                    pypi_0    pypi
font-ttf-dejavu-sans-mono 2.37                 hab24e00_0    conda-forge
font-ttf-inconsolata      2.001                hab24e00_0    conda-forge
font-ttf-source-code-pro  2.030                hab24e00_0    conda-forge
font-ttf-ubuntu           0.83                 hab24e00_0    conda-forge
fontconfig                2.13.1            h86ecdb6_1001    conda-forge
fonts-conda-forge         1                             0    conda-forge
fortran-compiler          1.0.4                he991be0_0    conda-forge
freetype                  2.10.2               he06d7ca_0    conda-forge
freexl                    1.0.5             h14c3975_1002    conda-forge
fribidi                   1.0.9                h516909a_0    conda-forge
fsspec                    0.7.4                      py_0    conda-forge
gcc_impl_linux-64         7.5.0                hd420e75_6    conda-forge
gcc_linux-64              7.5.0               h09487f9_20    conda-forge
gdal                      3.0.4            py38h172510d_6    conda-forge
gdk-pixbuf                2.38.2               h3f25603_4    conda-forge
geos                      3.8.1                he1b5a44_0    conda-forge
geotiff                   1.5.1               h05acad5_10    conda-forge
gettext                   0.19.8.1          hc5be6a0_1002    conda-forge
gfortran_impl_linux-64    7.5.0                hdf63c60_6    conda-forge
gfortran_linux-64         7.5.0               h09487f9_20    conda-forge
ghostscript               9.22              hf484d3e_1001    conda-forge
giflib                    5.2.1                h516909a_2    conda-forge
glib                      2.65.0               h6f030ca_0    conda-forge
gobject-introspection     1.64.1           py38h03d966d_1    conda-forge
graphite2                 1.3.13            he1b5a44_1001    conda-forge
graphviz                  2.42.3               h0511662_0    conda-forge
gsl                       2.6                  h294904e_0    conda-forge
gsw                       3.4.0                    pypi_0    pypi
gxx_impl_linux-64         7.5.0                hdf63c60_6    conda-forge
gxx_linux-64              7.5.0               h09487f9_20    conda-forge
harfbuzz                  2.4.0                h9f30f68_3    conda-forge
hdf4                      4.2.13            hf30be14_1003    conda-forge
hdf5                      1.10.5          mpi_mpich_ha7d0aea_1004    conda-forge
hdfeos2                   2.20              h64bfcee_1000    conda-forge
hdfeos5                   5.1.16               h8b6279f_5    conda-forge
heapdict                  1.0.1                      py_0    conda-forge
html5lib                  1.1                pyh9f0ad1d_0    conda-forge
icu                       64.2                 he1b5a44_1    conda-forge
idna                      2.10               pyh9f0ad1d_0    conda-forge
imagemagick               7.0.10_23       pl526h201ca68_0    conda-forge
imagesize                 1.2.0                    pypi_0    pypi
iris                      2.4.0                    py38_0    conda-forge
isodate                   0.6.0                    pypi_0    pypi
isort                     5.0.4                    pypi_0    pypi
jasper                    1.900.1           h07fcdf6_1006    conda-forge
jbig                      2.1               h516909a_2002    conda-forge
jinja2                    2.11.2             pyh9f0ad1d_0    conda-forge
joblib                    0.16.0                     py_0    conda-forge
jpeg                      9d                   h516909a_0    conda-forge
json-c                    0.13.1            hbfbb72e_1002    conda-forge
kealib                    1.4.13               hec59c27_0    conda-forge
keepalive                 0.5                        py_1    conda-forge
kiwisolver                1.2.0            py38hbf85e49_0    conda-forge
krb5                      1.17.1               hfafb76e_1    conda-forge
lazy-object-proxy         1.4.3                    pypi_0    pypi
ld_impl_linux-64          2.34                 h53a641e_5    conda-forge
libaec                    1.0.4                he1b5a44_1    conda-forge
libblas                   3.8.0               17_openblas    conda-forge
libcblas                  3.8.0               17_openblas    conda-forge
libcroco                  0.6.13               h8d621e5_1    conda-forge
libcurl                   7.71.1               hcdd3856_0    conda-forge
libdap4                   3.20.6               h1d1bd15_0    conda-forge
libedit                   3.1.20191231         h46ee950_0    conda-forge
libffi                    3.2.1             he1b5a44_1007    conda-forge
libgcc-ng                 9.2.0                h24d8f2e_2    conda-forge
libgdal                   3.0.4                h3dfc09a_6    conda-forge
libgfortran-ng            7.5.0                hdf63c60_6    conda-forge
libgomp                   9.2.0                h24d8f2e_2    conda-forge
libiconv                  1.15              h516909a_1006    conda-forge
libkml                    1.3.0             hb574062_1011    conda-forge
liblapack                 3.8.0               17_openblas    conda-forge
libllvm8                  8.0.1                hc9558a2_0    conda-forge
libnetcdf                 4.7.4           mpi_mpich_h755db7c_1    conda-forge
libopenblas               0.3.10               h5ec1e0e_0    conda-forge
libpng                    1.6.37               hed695b0_1    conda-forge
libpq                     12.2                 h5513abc_1    conda-forge
librsvg                   2.49.3               h33a7fed_0    conda-forge
libspatialite             4.3.0a            h2482549_1038    conda-forge
libssh2                   1.9.0                hab1572f_2    conda-forge
libstdcxx-ng              9.2.0                hdf63c60_2    conda-forge
libtiff                   4.1.0                hc7e4089_6    conda-forge
libtool                   2.4.6             h14c3975_1002    conda-forge
libunwind                 1.3.1             hf484d3e_1000    conda-forge
libuuid                   2.32.1            h14c3975_1000    conda-forge
libwebp                   1.1.0                h56121f0_4    conda-forge
libwebp-base              1.1.0                h516909a_3    conda-forge
libxcb                    1.13              h14c3975_1002    conda-forge
libxml2                   2.9.10               hee79883_0    conda-forge
libxslt                   1.1.33               h31b3aaa_0    conda-forge
llvm-openmp               10.0.0               hc9558a2_0    conda-forge
llvmlite                  0.33.0                   pypi_0    pypi
locket                    0.2.0                      py_2    conda-forge
lxml                      4.5.1            py38hbb43d70_0    conda-forge
lz4-c                     1.9.2                he1b5a44_1    conda-forge
make                      4.3                  h516909a_0    conda-forge
markupsafe                1.1.1            py38h1e0a361_1    conda-forge
matplotlib-base           3.2.2            py38h2af1d28_0    conda-forge
mccabe                    0.6.1                    pypi_0    pypi
mock                      4.0.2                    pypi_0    pypi
more-itertools            8.4.0                    pypi_0    pypi
mpi                       1.0                       mpich    conda-forge
mpi4py                    3.0.3            py38h4a80816_1    conda-forge
mpich                     3.3.2                hc856adb_0    conda-forge
msgpack-python            1.0.0            py38hbf85e49_1    conda-forge
munch                     2.5.0                      py_0    conda-forge
nc-time-axis              1.2.0                      py_1    conda-forge
ncl                       6.6.2               hfe5c2fd_21    conda-forge
nco                       4.9.2           mpi_mpich_h9a76d41_102    conda-forge
ncurses                   6.1               hf484d3e_1002    conda-forge
netcdf-fortran            4.5.2           mpi_mpich_h6a79edc_4    conda-forge
netcdf4                   1.5.3           mpi_mpich_py38h894258e_3    conda-forge
networkx                  2.4                        py_1    conda-forge
nose                      1.3.7                    pypi_0    pypi
numba                     0.50.1                   pypi_0    pypi
numpy                     1.18.5           py38h8854b6b_0    conda-forge
olefile                   0.46                       py_0    conda-forge
openjpeg                  2.3.1                h981e76c_3    conda-forge
openssl                   1.1.1g               h516909a_0    conda-forge
ossuuid                   1.6.2             hf484d3e_1000    conda-forge
owslib                    0.20.0                     py_0    conda-forge
packaging                 20.4               pyh9f0ad1d_0    conda-forge
pandas                    1.0.5            py38hcb8c335_0    conda-forge
pango                     1.42.4               h7062337_4    conda-forge
partd                     1.1.0                      py_0    conda-forge
pathspec                  0.8.0                    pypi_0    pypi
pcre                      8.44                 he1b5a44_0    conda-forge
pcre2                     10.35                h2f06484_0    conda-forge
pep8-naming               0.10.0                   pypi_0    pypi
perl                      5.26.2            h516909a_1006    conda-forge
pillow                    7.2.0            py38h9776b28_0    conda-forge
pip                       20.1.1                     py_1    conda-forge
pixman                    0.38.0            h516909a_1003    conda-forge
pkg-config                0.29.2            h516909a_1006    conda-forge
pluggy                    0.13.1                   pypi_0    pypi
poppler                   0.67.0               h14e79db_8    conda-forge
poppler-data              0.4.9                         1    conda-forge
postgresql                12.2                 h8573dbc_1    conda-forge
proj                      7.0.0                h966b41f_4    conda-forge
prospector                1.3.0                    pypi_0    pypi
prov                      1.5.3                    pypi_0    pypi
psutil                    5.7.0            py38h1e0a361_1    conda-forge
pthread-stubs             0.4               h14c3975_1001    conda-forge
py                        1.9.0                    pypi_0    pypi
pycodestyle               2.6.0                    pypi_0    pypi
pycparser                 2.20               pyh9f0ad1d_2    conda-forge
pydocstyle                5.0.2                    pypi_0    pypi
pydot                     1.4.1           py38h32f6830_1002    conda-forge
pyepsg                    0.4.0                      py_0    conda-forge
pyflakes                  2.2.0                    pypi_0    pypi
pygments                  2.6.1                    pypi_0    pypi
pykdtree                  1.3.1           py38h8790de6_1003    conda-forge
pyke                      1.1.1           py38h32f6830_1002    conda-forge
pylint                    2.5.2                    pypi_0    pypi
pylint-celery             0.3                      pypi_0    pypi
pylint-django             2.0.15                   pypi_0    pypi
pylint-flask              0.6                      pypi_0    pypi
pylint-plugin-utils       0.6                      pypi_0    pypi
pyopenssl                 19.1.0                     py_1    conda-forge
pyparsing                 2.4.7              pyh9f0ad1d_0    conda-forge
pyproj                    2.6.1.post1      py38h7521cb9_0    conda-forge
pyroma                    2.6                      pypi_0    pypi
pyshp                     2.1.0                      py_0    conda-forge
pysocks                   1.7.1            py38h32f6830_1    conda-forge
pytest                    5.4.3                    pypi_0    pypi
pytest-cov                2.10.0                   pypi_0    pypi
pytest-env                0.6.2                    pypi_0    pypi
pytest-flake8             1.0.6                    pypi_0    pypi
pytest-html               2.1.1                    pypi_0    pypi
pytest-metadata           1.10.0                   pypi_0    pypi
pytest-mock               3.1.1                    pypi_0    pypi
python                    3.8.3           cpython_he5300dc_0    conda-forge
python-dateutil           2.8.1                      py_0    conda-forge
python-stratify           0.1.1           py38h8790de6_1002    conda-forge
python_abi                3.8                      1_cp38    conda-forge
pytz                      2020.1             pyh9f0ad1d_0    conda-forge
pyyaml                    5.3.1            py38h1e0a361_0    conda-forge
r-base                    4.0.2                h95c6c4b_0    conda-forge
r-curl                    4.3               r40hcdcec82_1    conda-forge
r-udunits2                0.13            r40hcdcec82_1004    conda-forge
rdflib                    5.0.0            py38h32f6830_2    conda-forge
readline                  8.0                  hf8c457e_0    conda-forge
requests                  2.24.0             pyh9f0ad1d_0    conda-forge
requirements-detector     0.7                      pypi_0    pypi
scikit-learn              0.23.1           py38h3a94b23_0    conda-forge
scipy                     1.5.0            py38h18bccfc_0    conda-forge
seaborn                   0.10.1                   pypi_0    pypi
seawater                  3.3.4                    pypi_0    pypi
sed                       4.7               h1bed415_1000    conda-forge
setoptconf                0.2.0                    pypi_0    pypi
setuptools                49.1.0           py38h32f6830_0    conda-forge
shapely                   1.7.0            py38hd168ffb_3    conda-forge
six                       1.15.0             pyh9f0ad1d_0    conda-forge
snowballstemmer           2.0.0                    pypi_0    pypi
sortedcontainers          2.2.2              pyh9f0ad1d_0    conda-forge
sparqlwrapper             1.8.5           py38h32f6830_1003    conda-forge
sphinx                    3.1.2                    pypi_0    pypi
sphinx-rtd-theme          0.5.0                    pypi_0    pypi
sphinxcontrib-applehelp   1.0.2                    pypi_0    pypi
sphinxcontrib-devhelp     1.0.2                    pypi_0    pypi
sphinxcontrib-htmlhelp    1.0.3                    pypi_0    pypi
sphinxcontrib-jsmath      1.0.1                    pypi_0    pypi
sphinxcontrib-qthelp      1.0.3                    pypi_0    pypi
sphinxcontrib-serializinghtml 1.1.4                    pypi_0    pypi
sqlite                    3.32.3               hcee41ef_0    conda-forge
tbb                       2020.1               hc9558a2_0    conda-forge
tblib                     1.6.0                      py_0    conda-forge
tempest-remap             2.0.3           mpi_mpich_hf005093_8    conda-forge
termcolor                 1.1.0                    pypi_0    pypi
threadpoolctl             2.1.0              pyh5ca1d4c_0    conda-forge
tiledb                    1.7.7                h8efa9f0_3    conda-forge
tk                        8.6.10               hed695b0_0    conda-forge
tktable                   2.10                 h555a92e_3    conda-forge
toml                      0.10.1                   pypi_0    pypi
toolz                     0.10.0                     py_0    conda-forge
tornado                   6.0.4            py38h1e0a361_1    conda-forge
tqdm                      4.47.0                   pypi_0    pypi
typing_extensions         3.7.4.2                    py_0    conda-forge
tzcode                    2020a                h516909a_0    conda-forge
udunits2                  2.2.27.6          h4e0c4b3_1001    conda-forge
urllib3                   1.25.9                     py_0    conda-forge
vmprof                    0.4.15                   pypi_0    pypi
wcwidth                   0.2.5                    pypi_0    pypi
webencodings              0.5.1                      py_1    conda-forge
wheel                     0.34.2                     py_1    conda-forge
wrapt                     1.12.1                   pypi_0    pypi
xarray                    0.16.0                   pypi_0    pypi
xerces-c                  3.2.2             h8412b87_1004    conda-forge
xesmf                     0.3.0                    pypi_0    pypi
xlsxwriter                1.2.9                    pypi_0    pypi
xorg-imake                1.0.7                         0    conda-forge
xorg-kbproto              1.0.7             h14c3975_1002    conda-forge
xorg-libice               1.0.10               h516909a_0    conda-forge
xorg-libsm                1.2.3             h84519dc_1000    conda-forge
xorg-libx11               1.6.9                h516909a_0    conda-forge
xorg-libxau               1.0.9                h14c3975_0    conda-forge
xorg-libxaw               1.0.13            h14c3975_1002    conda-forge
xorg-libxdmcp             1.1.3                h516909a_0    conda-forge
xorg-libxext              1.3.4                h516909a_0    conda-forge
xorg-libxmu               1.1.3                h516909a_0    conda-forge
xorg-libxpm               3.5.13               h516909a_0    conda-forge
xorg-libxrender           0.9.10            h516909a_1002    conda-forge
xorg-libxt                1.1.5             h516909a_1003    conda-forge
xorg-makedepend           1.0.6                he1b5a44_1    conda-forge
xorg-renderproto          0.11.1            h14c3975_1002    conda-forge
xorg-xextproto            7.3.0             h14c3975_1002    conda-forge
xorg-xproto               7.0.31            h14c3975_1007    conda-forge
xz                        5.2.5                h516909a_0    conda-forge
yamale                    2.2.0              pyh9f0ad1d_0    conda-forge
yaml                      0.2.5                h516909a_0    conda-forge
yamllint                  1.23.0                   pypi_0    pypi
yapf                      0.30.0                   pypi_0    pypi
zict                      2.0.0                      py_0    conda-forge
zlib                      1.2.11            h516909a_1006    conda-forge
zstd                      1.4.4                h6597ccf_3    conda-forge

Conda env export
----------------

Please find below the contents of the automatically generated environment.yml
file, listing the full list of packages in the environment used.



name: ar6
channels:
  - defaults
  - conda-forge
dependencies:
  - _libgcc_mutex=0.1=conda_forge
  - _openmp_mutex=4.5=1_llvm
  - _r-mutex=1.0.1=anacondar_1
  - antlr-python-runtime=4.7.2=py38_1001
  - attrs=19.3.0=py_0
  - binutils-meta=1.0.4=0
  - binutils_impl_linux-64=2.34=h53a641e_5
  - binutils_linux-64=2.34=hc952b39_20
  - bokeh=2.1.1=py38h32f6830_0
  - boost-cpp=1.72.0=h8e57a91_0
  - brotlipy=0.7.0=py38h1e0a361_1000
  - bwidget=1.9.14=0
  - bzip2=1.0.8=h516909a_2
  - c-compiler=1.0.4=h516909a_0
  - ca-certificates=2020.6.20=hecda079_0
  - cairo=1.16.0=hcf35c78_1003
  - cartopy=0.17.0=py38h9cf8511_1015
  - certifi=2020.6.20=py38h32f6830_0
  - cf-units=2.1.4=py38h8790de6_0
  - cffi=1.14.0=py38hd463f26_0
  - cfitsio=3.470=h3eac812_5
  - cftime=1.2.0=py38h8790de6_1
  - chardet=3.0.4=py38h32f6830_1006
  - click=7.1.2=pyh9f0ad1d_0
  - click-plugins=1.1.1=py_0
  - cloudpickle=1.5.0=py_0
  - compilers=1.0.4=0
  - cryptography=2.9.2=py38h766eaa4_0
  - curl=7.71.1=he644dc0_0
  - cxx-compiler=1.0.4=hc9558a2_0
  - cycler=0.10.0=py_2
  - cython=0.29.20=py38h950e882_0
  - cytoolz=0.10.1=py38h516909a_0
  - dask=2.20.0=py_0
  - dask-core=2.20.0=py_0
  - decorator=4.4.2=py_0
  - distributed=2.20.0=py38h32f6830_0
  - eccodes=2.17.0=h59f7be3_1
  - esmf=8.0.0=mpi_mpich_h9a42a66_106
  - esmpy=8.0.0=mpi_mpich_py38ha9b28fa_101
  - expat=2.2.9=he1b5a44_2
  - fftw=3.3.8=nompi_h7f3a6c3_1111
  - font-ttf-dejavu-sans-mono=2.37=hab24e00_0
  - font-ttf-inconsolata=2.001=hab24e00_0
  - font-ttf-source-code-pro=2.030=hab24e00_0
  - font-ttf-ubuntu=0.83=hab24e00_0
  - fontconfig=2.13.1=h86ecdb6_1001
  - fonts-conda-forge=1=0
  - fortran-compiler=1.0.4=he991be0_0
  - freetype=2.10.2=he06d7ca_0
  - freexl=1.0.5=h14c3975_1002
  - fribidi=1.0.9=h516909a_0
  - fsspec=0.7.4=py_0
  - gcc_impl_linux-64=7.5.0=hd420e75_6
  - gcc_linux-64=7.5.0=h09487f9_20
  - gdal=3.0.4=py38h172510d_6
  - gdk-pixbuf=2.38.2=h3f25603_4
  - geos=3.8.1=he1b5a44_0
  - geotiff=1.5.1=h05acad5_10
  - gettext=0.19.8.1=hc5be6a0_1002
  - gfortran_impl_linux-64=7.5.0=hdf63c60_6
  - gfortran_linux-64=7.5.0=h09487f9_20
  - ghostscript=9.22=hf484d3e_1001
  - giflib=5.2.1=h516909a_2
  - glib=2.65.0=h6f030ca_0
  - gobject-introspection=1.64.1=py38h03d966d_1
  - graphite2=1.3.13=he1b5a44_1001
  - graphviz=2.42.3=h0511662_0
  - gsl=2.6=h294904e_0
  - gxx_impl_linux-64=7.5.0=hdf63c60_6
  - gxx_linux-64=7.5.0=h09487f9_20
  - harfbuzz=2.4.0=h9f30f68_3
  - hdf4=4.2.13=hf30be14_1003
  - hdf5=1.10.5=mpi_mpich_ha7d0aea_1004
  - hdfeos2=2.20=h64bfcee_1000
  - hdfeos5=5.1.16=h8b6279f_5
  - heapdict=1.0.1=py_0
  - html5lib=1.1=pyh9f0ad1d_0
  - icu=64.2=he1b5a44_1
  - idna=2.10=pyh9f0ad1d_0
  - imagemagick=7.0.10_23=pl526h201ca68_0
  - iris=2.4.0=py38_0
  - jasper=1.900.1=h07fcdf6_1006
  - jbig=2.1=h516909a_2002
  - jinja2=2.11.2=pyh9f0ad1d_0
  - joblib=0.16.0=py_0
  - jpeg=9d=h516909a_0
  - json-c=0.13.1=hbfbb72e_1002
  - kealib=1.4.13=hec59c27_0
  - keepalive=0.5=py_1
  - kiwisolver=1.2.0=py38hbf85e49_0
   - krb5=1.17.1=hfafb76e_1
   - ld_impl_linux-64=2.34=h53a641e_5
   - libaec=1.0.4=he1b5a44_1
   - libblas=3.8.0=17_openblas
   - libcblas=3.8.0=17_openblas
   - libcroco=0.6.13=h8d621e5_1
   - libcurl=7.71.1=hcdd3856_0
   - libdap4=3.20.6=h1d1bd15_0
   - libedit=3.1.20191231=h46ee950_0
   - libffi=3.2.1=he1b5a44_1007
   - libgcc-ng=9.2.0=h24d8f2e_2
   - libgdal=3.0.4=h3dfc09a_6
   - libgfortran-ng=7.5.0=hdf63c60_6
   - libgomp=9.2.0=h24d8f2e_2
   - libiconv=1.15=h516909a_1006
   - libkml=1.3.0=hb574062_1011
   - liblapack=3.8.0=17_openblas
   - libllvm8=8.0.1=hc9558a2_0
   - libnetcdf=4.7.4=mpi_mpich_h755db7c_1
   - libopenblas=0.3.10=h5ec1e0e_0
   - libpng=1.6.37=hed695b0_1
   - libpq=12.2=h5513abc_1
   - librsvg=2.49.3=h33a7fed_0
   - libspatialite=4.3.0a=h2482549_1038
   - libssh2=1.9.0=hab1572f_2
   - libstdcxx-ng=9.2.0=hdf63c60_2
   - libtiff=4.1.0=hc7e4089_6
   - libtool=2.4.6=h14c3975_1002
   - libunwind=1.3.1=hf484d3e_1000
   - libuuid=2.32.1=h14c3975_1000
   - libwebp=1.1.0=h56121f0_4
   - libwebp-base=1.1.0=h516909a_3
   - libxcb=1.13=h14c3975_1002
   - libxml2=2.9.10=hee79883_0
   - libxslt=1.1.33=h31b3aaa_0
   - llvm-openmp=10.0.0=hc9558a2_0
   - locket=0.2.0=py_2
   - lxml=4.5.1=py38hbb43d70_0
   - lz4-c=1.9.2=he1b5a44_1
   - make=4.3=h516909a_0
   - markupsafe=1.1.1=py38h1e0a361_1
   - matplotlib-base=3.2.2=py38h2af1d28_0
   - mpi=1.0=mpich
   - mpi4py=3.0.3=py38h4a80816_1
   - mpich=3.3.2=hc856adb_0
   - msgpack-python=1.0.0=py38hbf85e49_1
   - munch=2.5.0=py_0
   - nc-time-axis=1.2.0=py_1
   - ncl=6.6.2=hfe5c2fd_21
   - nco=4.9.2=mpi_mpich_h9a76d41_102
   - ncurses=6.1=hf484d3e_1002
   - netcdf-fortran=4.5.2=mpi_mpich_h6a79edc_4
   - netcdf4=1.5.3=mpi_mpich_py38h894258e_3
   - networkx=2.4=py_1
   - numpy=1.18.5=py38h8854b6b_0
   - olefile=0.46=py_0
   - openjpeg=2.3.1=h981e76c_3
   - openssl=1.1.1g=h516909a_0
   - ossuuid=1.6.2=hf484d3e_1000
   - owslib=0.20.0=py_0
   - packaging=20.4=pyh9f0ad1d_0
   - pandas=1.0.5=py38hcb8c335_0
   - pango=1.42.4=h7062337_4
   - partd=1.1.0=py_0
   - pcre=8.44=he1b5a44_0
   - pcre2=10.35=h2f06484_0
   - perl=5.26.2=h516909a_1006
   - pillow=7.2.0=py38h9776b28_0
   - pip=20.1.1=py_1
   - pixman=0.38.0=h516909a_1003
   - pkg-config=0.29.2=h516909a_1006
   - poppler=0.67.0=h14e79db_8
   - poppler-data=0.4.9=1
   - postgresql=12.2=h8573dbc_1
   - proj=7.0.0=h966b41f_4
   - psutil=5.7.0=py38h1e0a361_1
   - pthread-stubs=0.4=h14c3975_1001
   - pycparser=2.20=pyh9f0ad1d_2
   - pydot=1.4.1=py38h32f6830_1002
   - pyepsg=0.4.0=py_0
   - pykdtree=1.3.1=py38h8790de6_1003
   - pyke=1.1.1=py38h32f6830_1002
   - pyopenssl=19.1.0=py_1
   - pyparsing=2.4.7=pyh9f0ad1d_0
   - pyproj=2.6.1.post1=py38h7521cb9_0
   - pyshp=2.1.0=py_0
   - pysocks=1.7.1=py38h32f6830_1
   - python=3.8.3=cpython_he5300dc_0
   - python-dateutil=2.8.1=py_0
   - python-stratify=0.1.1=py38h8790de6_1002
   - python_abi=3.8=1_cp38
   - pytz=2020.1=pyh9f0ad1d_0
   - pyyaml=5.3.1=py38h1e0a361_0
   - r-base=4.0.2=h95c6c4b_0
     - r-curl=4.3=r40hcdcec82_1
     - r-udunits2=0.13=r40hcdcec82_1004
     - rdflib=5.0.0=py38h32f6830_2
     - readline=8.0=hf8c457e_0
     - requests=2.24.0=pyh9f0ad1d_0
     - scikit-learn=0.23.1=py38h3a94b23_0
     - scipy=1.5.0=py38h18bccfc_0
     - sed=4.7=h1bed415_1000
     - setuptools=49.1.0=py38h32f6830_0
     - shapely=1.7.0=py38hd168ffb_3
     - six=1.15.0=pyh9f0ad1d_0
     - sortedcontainers=2.2.2=pyh9f0ad1d_0
     - sparqlwrapper=1.8.5=py38h32f6830_1003
     - sqlite=3.32.3=hcee41ef_0
     - tbb=2020.1=hc9558a2_0
     - tblib=1.6.0=py_0
     - tempest-remap=2.0.3=mpi_mpich_hf005093_8
     - threadpoolctl=2.1.0=pyh5ca1d4c_0
     - tiledb=1.7.7=h8efa9f0_3
     - tk=8.6.10=hed695b0_0
     - tktable=2.10=h555a92e_3
     - toolz=0.10.0=py_0
     - tornado=6.0.4=py38h1e0a361_1
     - typing_extensions=3.7.4.2=py_0
     - tzcode=2020a=h516909a_0
     - udunits2=2.2.27.6=h4e0c4b3_1001
     - urllib3=1.25.9=py_0
     - webencodings=0.5.1=py_1
     - wheel=0.34.2=py_1
     - xerces-c=3.2.2=h8412b87_1004
     - xorg-imake=1.0.7=0
     - xorg-kbproto=1.0.7=h14c3975_1002
     - xorg-libice=1.0.10=h516909a_0
     - xorg-libsm=1.2.3=h84519dc_1000
     - xorg-libx11=1.6.9=h516909a_0
     - xorg-libxau=1.0.9=h14c3975_0
     - xorg-libxaw=1.0.13=h14c3975_1002
     - xorg-libxdmcp=1.1.3=h516909a_0
     - xorg-libxext=1.3.4=h516909a_0
     - xorg-libxmu=1.1.3=h516909a_0
     - xorg-libxpm=3.5.13=h516909a_0
     - xorg-libxrender=0.9.10=h516909a_1002
     - xorg-libxt=1.1.5=h516909a_1003
     - xorg-makedepend=1.0.6=he1b5a44_1
     - xorg-renderproto=0.11.1=h14c3975_1002
     - xorg-xextproto=7.3.0=h14c3975_1002
     - xorg-xproto=7.0.31=h14c3975_1007
     - xz=5.2.5=h516909a_0
     - yamale=2.2.0=pyh9f0ad1d_0
     - yaml=0.2.5=h516909a_0
     - zict=2.0.0=py_0
     - zlib=1.2.11=h516909a_1006
     - zstd=1.4.4=h6597ccf_3
     - pip:
       - alabaster==0.7.12
       - astroid==2.4.1
       - autodocsumm==0.1.13
       - babel==2.8.0
       - cdo==1.5.3
       - cdsapi==0.3.0
       - cligj==0.5.0
       - cmocean==2.0
       - codespell==1.17.1
       - colorama==0.4.3
       - coverage==5.2
       - docutils==0.16
       - dodgy==0.2.1
       - easytest==0.1.5
       - ecmwf-api-client==1.5.4
       - eofs==1.4.0
       - fiona==1.8.13.post1
       - fire==0.3.1
       - flake8==3.8.3
       - flake8-polyfill==1.0.2
       - gsw==3.4.0
       - imagesize==1.2.0
       - isodate==0.6.0
       - isort==5.0.4
       - lazy-object-proxy==1.4.3
       - llvmlite==0.33.0
       - mccabe==0.6.1
       - mock==4.0.2
       - more-itertools==8.4.0
       - nose==1.3.7
       - numba==0.50.1
       - pathspec==0.8.0
       - pep8-naming==0.10.0
       - pluggy==0.13.1
       - prospector==1.3.0
       - prov==1.5.3
       - py==1.9.0
       - pycodestyle==2.6.0
       - pydocstyle==5.0.2
       - pyflakes==2.2.0
       - pygments==2.6.1
       - pylint==2.5.2
       - pylint-celery==0.3
       - pylint-django==2.0.15
       - pylint-flask==0.6
       - pylint-plugin-utils==0.6
       - pyroma==2.6
       - pytest==5.4.3
       - pytest-cov==2.10.0
       - pytest-env==0.6.2
       - pytest-flake8==1.0.6
       - pytest-html==2.1.1
       - pytest-metadata==1.10.0
       - pytest-mock==3.1.1
       - requirements-detector==0.7
       - seaborn==0.10.1
       - seawater==3.3.4
       - setoptconf==0.2.0
       - snowballstemmer==2.0.0
       - sphinx==3.1.2
       - sphinx-rtd-theme==0.5.0
       - sphinxcontrib-applehelp==1.0.2
       - sphinxcontrib-devhelp==1.0.2
       - sphinxcontrib-htmlhelp==1.0.3
       - sphinxcontrib-jsmath==1.0.1
       - sphinxcontrib-qthelp==1.0.3
       - sphinxcontrib-serializinghtml==1.1.4
       - termcolor==1.1.0
       - toml==0.10.1
       - tqdm==4.47.0
       - vmprof==0.4.15
       - wcwidth==0.2.5
       - wrapt==1.12.1
       - xarray==0.16.0
       - xesmf==0.3.0
       - xlsxwriter==1.2.9
       - yamllint==1.23.0
       - yapf==0.30.0
   prefix: /home/users/ldemora/anaconda3_20190821/envs/ar6
