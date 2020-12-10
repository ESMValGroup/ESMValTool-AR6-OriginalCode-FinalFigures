"""ESMValTool CMORizer for BR2011 data.

Tier
    Tier 2: other freely-available dataset.

Source
    https://tc.copernicus.org/articles/5/219/2011/tc-5-219-2011-supplement.zip

Last access
    20201105
"""

import logging
import os
from datetime import datetime, timedelta

import iris
import numpy as np
import pandas as pd
from cf_units import Unit
import zipfile

from . import utilities as utils

logger = logging.getLogger(__name__)


def _clean(cont_file):
    """Remove unzipped input files."""
    if os.path.isfile(cont_file):
        os.remove(cont_file)
        logger.info("Removed tmp file %s", cont_file)


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


def _extract_variable_data(month, cfg, data_table):
    """Extract variable."""

    if month.lower() == 'march':
        n_month = 3
    elif month.lower() == 'april':
        n_month = 4

    index = cfg['excel_info']['column_n']

    data = data_table.iloc[:, index]
    years = np.asarray(data.index)
    sce = data.values

    return (n_month, years, sce)


def _finalize_cube(short_name, var_info, years, data_dic, cfg, out_dir):
    # Fix units

    months = np.arange(1, 13)

    data = np.ma.masked_all((len(years) * len(months)))

    time_points = []
    time_bounds = []

    for y, year in enumerate(years):
        for m, month in enumerate(months):
            if month in data_dic.keys():
                if np.isfinite(data_dic[month][y]):
                    data[y * 12 + m] = data_dic[month][y]
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


def _unzip(filepath, file_to_extract, out_dir):
    """Unzip `*.zip` file."""
    logger.info("Starting extraction of %s to %s", filepath, out_dir)
    zipf = zipfile.ZipFile(filepath, 'r')
    content_file = zipf.extract(file_to_extract, out_dir)
    logger.info("Succefully extracted %s to %s", file_to_extract, out_dir)

    return content_file


def cmorization(in_dir, out_dir, cfg, _):
    """Cmorization func call."""
    filepath = os.path.join(in_dir, cfg['filename'])
    if os.path.isfile(filepath):
        logger.info("Found input file '%s'", filepath)

    file_to_extract = cfg['content_filename']

    cont_filepath = _unzip(filepath, file_to_extract, out_dir)

    header = cfg['excel_info']['header']
    footer = cfg['excel_info']['skip_footer']

    # Run the cmorization
    for (var, var_info) in cfg['variables'].items():
        logger.info("CMORizing variable '%s'", var)
        all_months_dic = {}
        for month in cfg['excel_info']['months'].keys():
            # we skip the info with contact
            columns = cfg['excel_info']['months'][month]['columns']
            data_table = pd.read_excel(cont_filepath, sheet_name='SCE',
                                       header=header, skipfooter=footer,
                                       index_col=0,
                                       usecols=columns)
            n_month, years, month_arr = _extract_variable_data(month, cfg,
                                                               data_table)
            all_months_dic[n_month] = month_arr
        _finalize_cube(var, var_info, years, all_months_dic, cfg, out_dir)

    _clean(cont_filepath)
