# ESMValTool

[![Documentation Status](https://readthedocs.org/projects/esmvaltool/badge/?version=latest)](https://esmvaltool.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3401363.svg)](https://doi.org/10.5281/zenodo.3401363)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ESMValGroup?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![CircleCI](https://circleci.com/gh/ESMValGroup/ESMValTool.svg?style=svg)](https://circleci.com/gh/ESMValGroup/ESMValTool)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/79bf6932c2e844eea15d0fb1ed7e415c)](https://www.codacy.com/gh/ESMValGroup/ESMValTool?utm_source=github.com&utm_medium=referral&utm_content=ESMValGroup/ESMValTool&utm_campaign=Badge_Coverage)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/79bf6932c2e844eea15d0fb1ed7e415c)](https://www.codacy.com/gh/ESMValGroup/ESMValTool?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ESMValGroup/ESMValTool&amp;utm_campaign=Badge_Grade)
[![Docker Build Status](https://img.shields.io/docker/build/esmvalgroup/esmvaltool.svg)](https://hub.docker.com/r/esmvalgroup/esmvaltool/)
[![Anaconda-Server Badge](https://anaconda.org/esmvalgroup/esmvaltool/badges/installer/conda.svg)](https://conda.anaconda.org/esmvalgroup)

ESMValTool: A community diagnostic and performance metrics tool for routine evaluation of Earth system models in CMIP

# Input data

This is a branch for IPCC AR6 WGI Chapter 3 cryosphere figures, ocean basin and sst figures as well as cross-chapter box 3.2 figure 1. 

| number in the FGD  |                    recipe                 |                    diagnostic                      |

|         3.20       | recipe_ipcc_ar6_wg1_fgd_sea_ice_joint.yml | seaice/sie_ipcc_ar6_wg1_fgd_3_18.py                |

|         3.21       | recipe_ipcc_ar6_wg1_fgd_sea_ice_joint.yml | seaice/sie_ipcc_ar6_wg1_fgd_3_19.py                |

|         3.22       | recipe_ipcc_ar6_wg1_fgd_3_20.yml          | seaice/sie_ipcc_ar6_wg1_fgd_3_20.py                |

|         3.24       | recipe_ocean_fig_3_19_zonal_sst.yml       | ocean/diagnostic_fig_3_19_zonal_sst.py             |

|         3.25       | recipe_ocean_basin_profile_bias.yml       | ocean/diagnostic_basin_profile_bias.py             |

|      XCB 3.2.1     | recipe_ipcc_ar6_wg1_fgd_xcb_3_2.yml       | extreme_events/extremes_ipcc_ar6_wg1_fgd_xcb_32.py |

Also several cmorizer were created, cmorizer_obs_woa.py was updated. Created esmvaltool/diag_scripts/shared/plot/styles_python/matplotlib/ipcc_ar6_fgd.mplstyle and updated esmvaltool/diag_scripts/shared/plot/styles_python/cmip6.yml .

However, in esmvalcore several cmor tables should be added 
# cmor/tables/custom/CMOR_scen.dat
SOURCE: CMIP5 

!============

variable_entry:    scen

!============

modeling_realm:    land

!----------------------------------

! Variable attributes:

!----------------------------------

standard_name:

units:             1e6 km2

cell_methods:      area: time: mean

cell_measures:

long_name:         Snow Cover Extent in all Northern-Hemisphere grid cells

comment:           sce is the area of land with snow on the ground

!----------------------------------

! Additional variable information:

!----------------------------------

dimensions:        time

type:              real

!----------------------------------

!

# cmor/tables/custom/CMOR_siareas.dat

!============

variable_entry:    siareas

!============

modeling_realm:    seaIce ocean

!----------------------------------

! Variable attributes:

!----------------------------------

standard_name:

units:             1e6 km2

cell_methods:      area: time: mean

cell_measures:

long_name:         Sea Ice Area in the Southern Hemisphere

comment:           total area covered by sea ice in the Southern Hemisphere

!----------------------------------

! Additional variable information:

!----------------------------------

dimensions:        time

out_name:          siareas

type:              real

!----------------------------------

!

# cmor/tables/custom/CMOR_siarean.dat

!============

variable_entry:    siarean

!============

modeling_realm:    seaIce ocean

!----------------------------------

! Variable attributes:

!----------------------------------

standard_name:

units:             1e6 km2

cell_methods:      area: time: mean

cell_measures:

long_name:         Sea Ice Area in the Northern Hemisphere

comment:           total area covered by sea ice in the Northern Hemisphere

!----------------------------------

! Additional variable information:

!----------------------------------

dimensions:        time

out_name:          siarean

type:              real

!----------------------------------

!

# cmor/tables/custom/CMOR_rx1day.dat

SOURCE: CMIP5

!============

variable_entry:    rx1day

!============

modeling_realm:    atmos

!----------------------------------

! Variable attributes:

!----------------------------------

standard_name:

units:             kg m-2

cell_methods:      time: mean

cell_measures:     area: areacella

long_name:         Annual/monthly maximum 1-day precipitation

comment:           ETCCDI (Extreme climate change index) annual/monthly maximum 1-day precipitation

!----------------------------------

! Additional variable information:

!----------------------------------

dimensions:        longitude latitude time

type:              real

!----------------------------------

!

# cmor/tables/custom/CMOR_txx.dat

SOURCE: CMIP5

!============

variable_entry:    txx

!============

modeling_realm:    atmos

!----------------------------------

! Variable attributes:

!----------------------------------

standard_name:

units:             K

cell_methods:      time: mean

cell_measures:     area: areacella

long_name:         Annual/monthly maximum value of daily maximum temperature

comment:           ETCCDI (Extreme climate change index) annual/monthly maximum value of daily maximum temperature

!----------------------------------

! Additional variable information:

!----------------------------------

dimensions:        longitude latitude time

type:              real

!----------------------------------

!



Updates follow. 
