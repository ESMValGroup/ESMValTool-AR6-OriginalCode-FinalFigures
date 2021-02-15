#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic script to plot figure 3.38 of IPCC AR6 chapter 3.

Description
-----------
Calculate and Plot the global annual mean values of CO2, as well as land-
and ocean carbon sinks and compare them to Le Quere estimates.

Author
------
Bettina Gier (Uni Bremen, Germany)

Project
-------
Eval4CMIP

Configuration options in recips
-------------------------------


ecs_filename : str, optional
    Name of the netcdf in which the ECS data is saved (default: ecs.nc).
output_name : str, optional
    Name of the output netcdf file (default: fig09-42a.*.
save : dict, optional
    Keyword arguments for the fig.saveplot() function.
axes_functions : dict, optional
    Keyword arguments for the plot appearance functions.

###############################################################################

"""

import logging
import os

import iris
from iris import Constraint

from esmvaltool.diag_scripts.shared import (extract_variables, plot,
                                            run_diagnostic,
                                            variables_available)

logger = logging.getLogger(os.path.basename(__file__))



def main(cfg):


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
