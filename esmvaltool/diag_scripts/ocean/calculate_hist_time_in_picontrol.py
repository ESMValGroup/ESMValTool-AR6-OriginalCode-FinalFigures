"""
Calculate hist time in picontrol
=======================

Takes a file from a historical dataset, and
calculates the year in the PI control run.

Author: Lee de Mora (PML)
        ledm@pml.ac.uk
"""

import logging
import os

import iris
import matplotlib.pyplot as plt
import numpy as np
import cftime

from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import run_diagnostic

from esmvalcore.preprocessor._time import climate_statistics


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    """
    Load the config file and some metadata, then pass them the plot making
    tools.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.

    """
    for index, metadata_filename in enumerate(cfg['input_files']):
        logger.info('metadata filename:\t%s', metadata_filename)

        metadatas = diagtools.get_input_files(cfg, index=index)
    #     if 'anomaly' in cfg.keys():
    #         anomaly_period = cfg['anomaly']
    #     else:
    #         anomaly_period = None
    #
    #     #######
    #     # Multi model time series
    #     # Subtract anomaly
    #     if anomaly_period:
    #         multi_model_time_series(
    #             cfg,
    #             metadatas,
    #             anomaly_period = anomaly_period
    #         )
    #
    #         multi_model_time_series(
    #            cfg,
    #             metadatas,
    #             anomaly_period = None
    #         )
    #     else:
    #         multi_model_time_series(
    #             cfg,
    #             metadatas,
    #         )
    #
        for filename in sorted(metadatas):
            metadata = metadatas[filename]
            dataset = metadata['dataset']
            print(dataset, metadata['exp'], metadata['ensemble'],metadata['project'], metadata['mip'])

            cube = iris.load(filename)
            times = cube.coord('time')
            units = times.units.name
            calendar = times.units.calendar

            # We can assume a shared calendar!

            child_branch_time = c.attributes['branch_time_in_child']

            parent_branch_yr = num2date(c.attributes['branch_time_in_parent'], units=c.attributes['parent_time_units'], calendar=calendar ).year # A

            child_branch_yr = num2date(c.attributes['branch_time_in_child'], units=units, calendar=calendar ).year # A

            diff = child_branch_yr - parent_branch_yr
            print('difference:', diff )
            historical_range = [1860, 2014]

            print(dataset, ':\t', historical_range, 'is', historical_range + diff)



    logger.info('Success')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
