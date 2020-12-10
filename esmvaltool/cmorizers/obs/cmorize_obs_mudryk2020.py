"""ESMValTool CMORizer for Mudryk et al. (2020) data.

Tier
    Tier 2: other freely-available dataset.

Source
    http://data.ec.gc.ca/data/climate/scientificknowledge/climate-research-publication-based-data/northern-hemisphere-blended-snow-extent-and-snow-mass-time-series/

Last access
    20201111

Download and processing instructions
    Just download the data following the link

"""
import logging
import os

import iris
from cf_units import Unit
import cftime
import datetime
import numpy as np

from . import utilities as utils

logger = logging.getLogger(__name__)


def _fix_time_coord(cube):
    """Convert the time to the gregorian calendar. """
    time_coord = cube.coord('time')
    years = np.arange(
        cftime.num2pydate(time_coord.points[0], time_coord.units.origin,
                          calendar=time_coord.units.calendar).year,
        cftime.num2pydate(time_coord.points[-1], time_coord.units.origin,
                          calendar=time_coord.units.calendar).year + 1)
    months = np.arange(1, 13)

    upd_times_points = []
    upd_times_bounds = []

    for year in years:
        for month in months:
            upd_times_points.append(datetime.datetime(year, month, 15))
            if month == 12:
                upd_times_bounds.append([datetime.datetime(year, month, 1),
                                         datetime.datetime(year, month, 31)])
            else:
                upd_times_bounds.append([datetime.datetime(year, month, 1),
                                         datetime.datetime(year, month + 1,
                                                           1)
                                         - datetime.timedelta(days=1)])

    new_unit = Unit('days since 1850-01-01 00:00:00', calendar='gregorian')
    # time_points = cftime.date2num(upd_times_points, new_unit.origin, calendar=new_unit.calendar)
    # time_bounds = cftime.date2num(upd_times_bounds, new_unit.origin, calendar=new_unit.calendar)

    # time_dim = iris.coords.DimCoord(time_points, bounds = time_bounds, standard_name = 'time', var_name = 'time',
    #                                 long_name = 'time', units = new_unit)
    time_dim = iris.coords.DimCoord(new_unit.date2num(upd_times_points),
                                    bounds=new_unit.date2num(upd_times_bounds),
                                    standard_name='time', var_name='time',
                                    long_name='time', units=new_unit)

    return (time_dim)


def _extract_variable(var, var_info, cmor_info, attrs, filepath, out_dir):
    """Extract variable."""
    raw_var = var_info.get('raw', var)
    var = cmor_info.short_name
    cube = iris.load_cube(filepath, utils.var_name_constraint(raw_var))
    # Fix units
    cube.units = var_info.get('raw_units', var)
    cube.convert_units(cmor_info.units)

    time_dim = _fix_time_coord(cube)
    cube = iris.cube.Cube(cube.data, standard_name=cube.standard_name,
                          long_name='snow cover extent',
                          var_name=cube.var_name, units=cube.units,
                          attributes=cube.attributes,
                          cell_methods=cube.cell_methods,
                          dim_coords_and_dims=[(time_dim, 0)])

    utils.fix_var_metadata(cube, cmor_info)
    utils.convert_timeunits(cube, 1950)
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
        # else:
        #     add here error message
        logger.info("CMORizing variable '%s'", var)
        glob_attrs['mip'] = var_info['mip']
        cmor_info = cmor_table.get_variable(var_info['mip'], var)
        _extract_variable(var, var_info, cmor_info, glob_attrs, filepath,
                          out_dir)
