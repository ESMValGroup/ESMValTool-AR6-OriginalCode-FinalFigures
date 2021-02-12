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
    cube = iris.load(filepath)
    for i in range(4):
        if cube[i].var_name == 'lat_bnds':
            lat_bnds = cube[i].data
        elif cube[i].var_name == 'lon_bnds':
            lon_bnds = cube[i].data
        elif cube[i].var_name == 'time_bnds':
            time_bnds = cube[i].data
        else:
            new_cube = cube[i]

    # Fix metadata
    utils.fix_var_metadata(new_cube, cmor_info)
    utils.set_global_atts(new_cube, attrs)

    # Fix units
    new_cube.units = 'kg m-2 s-1'

    # Convert data from precipitation rate to precipitation flux
    new_cube.data = new_cube.core_data() / 86400.0

    # Fix bounds
    new_cube.coord('time').bounds = None
    new_cube.coord('time').bounds = time_bnds
    new_cube.coord('latitude').bounds = None
    new_cube.coord('latitude').bounds = lat_bnds
    new_cube.coord('longitude').bounds = None
    new_cube.coord('longitude').bounds = lon_bnds

    # Save variable
    utils.save_variable(new_cube,
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
