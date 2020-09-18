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
    # for index, metadata_filename in enumerate(cfg['input_files']):
    #     logger.info('metadata filename:\t%s', metadata_filename)
    #
    #     metadatas = diagtools.get_input_files(cfg, index=index)
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
    #     for filename in sorted(metadatas):
    #         continue
    #         if metadatas[filename]['frequency'] != 'fx':
    #             logger.info('-----------------')
    #             logger.info(
    #                 'model filenames:\t%s',
    #                 filename,
    #             )
    #
    #             ######
    #             # Time series of individual model
    #             make_time_series_plots(cfg, metadatas[filename], filename)
    logger.info('Success')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
