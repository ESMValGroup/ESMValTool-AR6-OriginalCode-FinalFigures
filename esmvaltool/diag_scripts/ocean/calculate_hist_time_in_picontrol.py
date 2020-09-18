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

from netCDF4 import num2date

from esmvaltool.diag_scripts.shared import run_diagnostic
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    """
    Load the config file and some metadata, then print the linked pi years.

    The recipe can use the config argument, time_range
 

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.

    """
    for index, metadata_filename in enumerate(cfg['input_files']):
        metadatas = diagtools.get_input_files(cfg, index=index)

        for filename in sorted(metadatas):
            metadata = metadatas[filename]
            dataset = metadata['dataset']
            print('-----------\n', dataset, metadata['exp'], 
                  metadata['ensemble'],metadata['project'], metadata['mip'])

            cube = iris.load_cube(filename)
            times = cube.coord('time')
            units = times.units.name
            calendar = times.units.calendar

            # We can assume a shared calendar!

            parent_branch_yr = num2date(cube.attributes['branch_time_in_parent'], 
                                        units=cube.attributes['parent_time_units'], 
                                        calendar=calendar ).year 

            child_branch_yr = num2date(cube.attributes['branch_time_in_child'], 
                                       units=units, calendar=calendar ).year # A

            diff = child_branch_yr - parent_branch_yr
            #print('difference:', diff )
            historical_range = [1860, 2014]
            print('Origin is:', 
                cube.attributes['parent_activity_id'],
                cube.attributes['parent_experiment_id'],
                cube.attributes['parent_mip_era'],
                cube.attributes['parent_source_id'],
                cube.attributes['parent_variant_label'],
                )
            print(dataset, ':\t', historical_range, 'is', [h-diff for h in historical_range])

    logger.info('Success')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
