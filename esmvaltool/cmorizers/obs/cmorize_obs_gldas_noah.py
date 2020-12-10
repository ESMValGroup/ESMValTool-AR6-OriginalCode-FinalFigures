"""ESMValTool CMORizer for GLDAS2.0 NOAH data.

Tier
    Tier 3: restricted datasets (i.e., dataset which requires a registration
 to be retrieved or provided upon request to the respective contact or PI).

Source
    https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH10_M.2.0/

Last access
    20201110

Download and processing instructions
    - For download instructions register for Earthdata,
    order files and follow the instructions.

"""
import logging
import numpy as np
import os
import datetime

import cf_units
import iris

from . import utilities as utils

logger = logging.getLogger(__name__)


def _fix_time_monthly(cube):
    """Fix time by setting it to 15th of month."""
    # Read dataset time unit and calendar from file
    dataset_time_unit = str(cube.coord('time').units)
    dataset_time_calender = cube.coord('time').units.calendar
    # Convert datetime
    time_as_datetime = cf_units.num2date(cube.coord('time').core_points(),
                                         dataset_time_unit,
                                         dataset_time_calender)
    newtime = []
    for timepoint in time_as_datetime:
        midpoint = datetime(timepoint.year, timepoint.month, 15)
        newtime.append(midpoint)

    newtime = cf_units.date2num(newtime,
                                dataset_time_unit,
                                dataset_time_calender)
    # Put them on the file
    cube.coord('time').points = newtime
    cube.coord('time').bounds = None
    return cube


def _load_cubes(in_files, var_info, start_year, end_year):
    var = var_info['raw']

    sample_cube = iris.load_cube(in_files[0], var)

    data = np.ma.masked_all(
        (len(in_files), sample_cube.shape[1], sample_cube.shape[2]))
    data.fill_value = -9999.0

    for n, file in enumerate(in_files):
        logger.info("Loading file '%s'", file)
        cube = iris.load_cube(file, var)
        data[n, :, :] = cube.data[0, :, :]

    months = np.arange(1, 13)
    years = np.arange(start_year, end_year)

    new_unit = cf_units.Unit('days since 1850-01-01 00:00:00',
                             calendar='gregorian')

    time_points = []
    time_bounds = []

    for year in years:
        for month in months:
            time_points.append(datetime.datetime(year, month, 15))
            if month == 12:
                time_bounds.append([datetime.datetime(year, month, 1),
                                    datetime.datetime(year, month, 31)])
            else:
                time_bounds.append([datetime.datetime(year, month, 1),
                                    (datetime.datetime(year, month + 1,
                                                       1) - datetime.timedelta(
                                        days=1))])

    time_dim = iris.coords.DimCoord(
        cf_units.date2num(time_points, new_unit.origin,
                          calendar=new_unit.calendar),
        var_name='time', standard_name='time', long_name='time',
        bounds=cf_units.date2num(time_bounds, new_unit.origin,
                                 calendar=new_unit.calendar),
        units=new_unit)
    lon_dim = cube.coord('longitude')
    lat_dim = cube.coord('latitude')
    dims = [(time_dim, 0), (lat_dim, 1), (lon_dim, 2)]

    joint_cube = iris.cube.Cube(data, dim_coords_and_dims=dims,
                                units=var_info['raw_units'])

    return joint_cube


def _extract_variable(var, var_info, cmor_info, glob_attrs, start_year,
                      end_year, filenames, out_dir):
    cube = _load_cubes(filenames, var_info, start_year, end_year)

    # Fix units
    cube.units = var_info.get('raw_units', var)
    cube.convert_units(cmor_info.units)

    utils.fix_var_metadata(cube, cmor_info)
    utils.convert_timeunits(cube, 1950)
    utils.fix_coords(cube)
    utils.set_global_atts(cube, glob_attrs)
    utils.save_variable(cube,
                        var,
                        out_dir,
                        glob_attrs,
                        unlimited_dimensions=['time'])

    logger.info("Finished CMORizing %s", var)


def cmorization(in_dir, out_dir, cfg, _):
    """Run CMORizer for GLDAS_NOAH."""

    glob_attrs = cfg['attributes']
    cmor_table = cfg['cmor_table']

    start_year = cfg['custom']['start_year']
    end_year = cfg['custom']['end_year'] + 1

    months = np.arange(1, 13)

    for (var, var_info) in cfg['variables'].items():
        filename = var_info['file']
        reso = cfg['custom']['space_resolution_label']
        filenames = []
        for year in range(start_year, end_year):
            for month in months:
                filepath = os.path.join(in_dir, filename.format(
                    space_resolution_label=reso,
                    year=year, month=str(month).zfill(2)))
                if os.path.isfile(filepath):
                    logger.info("Found input file '%s'", filepath)
                    filenames.append(filepath)
                else:
                    logger.info("There is no input file '%s', download it.",
                                filepath)
        # Now get list of files
        glob_attrs['mip'] = var_info['mip']
        cmor_info = cmor_table.get_variable(var_info['mip'], var)
        logger.info("CMORizing variable '%s' from input files", var)
        _extract_variable(var, var_info, cmor_info, glob_attrs, start_year,
                          end_year, filenames, out_dir)
