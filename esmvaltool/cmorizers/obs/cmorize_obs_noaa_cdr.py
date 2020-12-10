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
from datetime import datetime, timedelta

import iris
import numpy as np
from cf_units import Unit
import zipfile

from . import utilities as utils

logger = logging.getLogger(__name__)


def _get_coords(points, bounds):
    """Extract coordinates."""
    time_units = Unit('days since 1850-01-01 00:00:00')

    time_dim = iris.coords.DimCoord(time_units.date2num(points),
                                    var_name='time',
                                    standard_name='time',
                                    long_name='time',
                                    bounds=time_units.date2num(bounds),
                                    units=time_units)
    return [(time_dim, 0)]


def _extract_variable(short_name, var_info, data_dic, cfg, out_dir):
    # Fix units

    months = np.arange(1, 13)

    if np.all(data_dic['eurasia'][:, :-1] != data_dic['north_am'][:, :-1]):
        logger.info("Fix the data: years and months")

    years = np.arange(int(data_dic['eurasia'][0, 0]),
                      int(data_dic['eurasia'][-1, 0]) + 1)

    data = np.ma.masked_all((len(years) * len(months)))

    time_points = []
    time_bounds = []

    for y, year in enumerate(years):
        for m, month in enumerate(months):
            if len(np.where((data_dic['eurasia'][:, 0] == year) & (
                data_dic['eurasia'][:, 1] == month))[0]) > 0:
                idx = np.where((data_dic['eurasia'][:, 0] == year) & (
                        data_dic['eurasia'][:, 1] == month))[0][0]
                data[y * 12 + m] = data_dic['eurasia'][idx, 2] + \
                                   data_dic['north_am'][idx, 2]
            time_points.append(datetime(year=year, month=month, day=15))
            first_bound = datetime(year=year, month=month, day=1)
            if month == 12:
                last_bound = datetime(year=year + 1, month=1, day=1)
            else:
                last_bound = datetime(year=year, month=month + 1, day=1)
            time_bounds.append(
                (first_bound, last_bound - timedelta(seconds=1)))

    sce_units = Unit(var_info['raw_units'])

    dims = _get_coords(np.asarray(time_points), np.asarray(time_bounds))

    final_cube = iris.cube.Cube(data, dim_coords_and_dims=dims,
                                units=sce_units)

    cmor_info = cfg['cmor_table'].get_variable(var_info['mip'], short_name)
    final_cube.convert_units(cmor_info.units)
    utils.convert_timeunits(final_cube, 1950)

    # Fix metadata
    attrs = cfg['attributes']
    attrs.update(var_info)

    utils.fix_var_metadata(final_cube, cmor_info)
    utils.set_global_atts(final_cube, attrs)

    # Save variable
    utils.save_variable(final_cube,
                        short_name,
                        out_dir,
                        attrs,
                        unlimited_dimensions=['time'])


def cmorization(in_dir, out_dir, cfg, _):
    """Cmorization func call."""

    for file_key in cfg['filenames'].keys():
        filepath = os.path.join(in_dir, cfg['filenames'][file_key])
        if os.path.isfile(filepath):
            logger.info("Found input file '%s'", filepath)

    # loading the data
    data_dic = {}
    data_dic['eurasia'] = np.loadtxt(
        os.path.join(in_dir, cfg['filenames']['eurasia']))
    if cfg['custom']['greenland_masked']:
        data_dic['north_am'] = np.loadtxt(
            os.path.join(in_dir, cfg['filenames']['na_no_greenland']))
    else:
        data_dic['north_am'] = np.loadtxt(
            os.path.join(in_dir, cfg['filenames']['na_with_greenland']))

    # Run the cmorization
    for (var, var_info) in cfg['variables'].items():
        logger.info("CMORizing variable '%s'", var)
        _extract_variable(var, var_info, data_dic, cfg, out_dir)
