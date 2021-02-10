"""ESMValTool CMORizer for GPCP-SG data.

Tier
    Tier 1

Source
    https://psl.noaa.gov/data/gridded/data.gpcp.html

Last access
    20210209

Download and processing instructions
    Download the following file:
        precip.mon.mean.nc

"""

import logging
import os

import iris
from cf_units import Unit

from . import utilities as utils

logger = logging.getLogger(__name__)


def _get_filepath(in_dir, basename):
    """Find correct name of file (extend basename with timestamp)."""
    all_files = [
        f for f in os.listdir(in_dir)
        if os.path.isfile(os.path.join(in_dir, f))
    ]
    for filename in all_files:
        if filename.endswith(basename):
            return os.path.join(in_dir, filename)
    raise OSError(
        f"Cannot find input file ending with '{basename}' in '{in_dir}'")


def _extract_variable(raw_var, cmor_info, attrs, filepath, out_dir):
    """Extract variable."""
    var = cmor_info.short_name
    cube = iris.load_cube(filepath, utils.var_name_constraint(raw_var))
    new_cube = cube[1] # containing the data
    new_cube.coord('time').bounds = None
    new_cube.coord('time').bounds = cube[0].data
    new_cube.coord('latitude').bounds = None
    new_cube.coord('latitude').bounds = cube[2].data
    new_cube.coord('longitude').bounds = None
    new_cube.coord('longitude').bounds = cube[3].data
    utils.save_variable(cube,
                        var,
                        out_dir,
                        attrs,
                        unlimited_dimensions=['time'])


def cmorization(in_dir, out_dir, cfg, _):
    """Cmorization func call."""
    glob_attrs = cfg['attributes']
    cmor_table = cfg['cmor_table']
    filepath = _get_filepath(in_dir, cfg['filename'])
    logger.info("Found input file '%s'", filepath)

    # Run the cmorization
    for (var, var_info) in cfg['variables'].items():
        logger.info("CMORizing variable '%s'", var)
        glob_attrs['mip'] = var_info['mip']
        cmor_info = cmor_table.get_variable(var_info['mip'], var)
        raw_var = var_info.get('raw', var)
        _extract_variable(raw_var, cmor_info, glob_attrs, filepath, out_dir)
