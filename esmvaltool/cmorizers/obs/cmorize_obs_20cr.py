"""ESMValTool CMORizer for NOAA_CDR data.

Tier
    Tier 2: other freely-available dataset.

Source
    # https://climate.rutgers.edu/snowcover/table_area.php?ui_set=2

Last access
    20201106

Download and processing instructions
    Choose monthly data and save them as txt
    The same name as in the links

"""

import logging
import os

import iris
from cf_units import Unit
import cftime
import numpy as np

from . import utilities as utils

logger = logging.getLogger(__name__)


def _apply_land_mask(cube, mask):
    new_mask = mask[0, :, :]
    cb_mask = ~ np.asarray(new_mask.data[np.newaxis, :, :], dtype=np.bool)
    cube.data.mask = cb_mask


def _fix_time_coord(cube):
    """Convert the time to the gregorian calendar. """
    time_coord = cube.coord('time')
    time_coord.guess_bounds()
    new_unit = Unit('days since 1850-01-01 00:00:00', calendar='gregorian')
    new_time_points = cftime.num2pydate(time_coord.points,
                                        time_coord.units.origin,
                                        time_coord.units.calendar)
    time_points = cftime.date2num(new_time_points, new_unit.origin,
                                  calendar=new_unit.calendar)
    new_time_bounds = cftime.num2pydate(time_coord.bounds,
                                        time_coord.units.origin,
                                        time_coord.units.calendar)
    time_bounds = cftime.date2num(new_time_bounds, new_unit.origin,
                                  calendar=new_unit.calendar)

    cube.coord('time').points = time_points
    cube.coord('time').bounds = time_bounds
    cube.coord('time').units = new_unit


def _extract_variable(var, var_info, cmor_info, attrs, filepath, mask_filepath,
                      out_dir):
    """Extract variable."""
    raw_var = var_info.get('raw', var)
    var = cmor_info.short_name
    cube = iris.load_cube(filepath, utils.var_name_constraint(raw_var))
    # Fix units
    cube.units = var_info.get('raw_units', var)
    cube.convert_units(cmor_info.units)

    mask = iris.load_cube(mask_filepath)
    _apply_land_mask(cube, mask)

    _fix_time_coord(cube)
    utils.fix_var_metadata(cube, cmor_info)
    utils.convert_timeunits(cube, 1950)
    utils.fix_coords(cube)
    utils.set_global_atts(cube, attrs)
    utils.save_variable(cube,
                        var,
                        out_dir,
                        attrs,
                        unlimited_dimensions=['time'])


def cmorization(in_dir, out_dir, cfg, _):
    """Cmorization function call."""
    glob_attrs = cfg['attributes']
    cmor_table = cfg['cmor_table']
    filename = cfg['filename']

    # Run the cmorization
    for (var, var_info) in cfg['variables'].items():
        filepath = os.path.join(in_dir,
                                filename.format(raw_variable=var_info['raw']))
        if os.path.isfile(filepath):
            logger.info("Found input file '%s'", filepath)
        mask_filepath = os.path.join(in_dir, cfg['filename_land_mask'])
        if os.path.isfile(mask_filepath):
            logger.info("Found input file '%s'", mask_filepath)
        logger.info("CMORizing variable '%s'", var)
        glob_attrs['mip'] = var_info['mip']
        cmor_info = cmor_table.get_variable(var_info['mip'], var)
        _extract_variable(var, var_info, cmor_info, glob_attrs, filepath,
                          mask_filepath, out_dir)
