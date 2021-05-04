"""ESMValTool CMORizer for WOA data.

Tier
   Tier 2: other freely-available dataset.

Source
   https://data.nodc.noaa.gov/woa/

Last access
   20190131

Download and processing instructions
   Download the following files for WOA13:
     WOA13/DATAv2/temperature/netcdf/decav81B0/1.00/woa13_decav81B0_t00_01.nc
     WOA13/DATAv2/salinity/netcdf/decav81B0/1.00/woa13_decav81B0_s00_01.nc
     WOA13/DATAv2/oxygen/netcdf/all/1.00/woa13_all_o00_01.nc
     WOA13/DATAv2/nitrate/netcdf/all/1.00/woa13_all_n00_01.nc
     WOA13/DATAv2/phosphate/netcdf/all/1.00/woa13_all_p00_01.nc
     WOA13/DATAv2/silicate/netcdf/all/1.00/woa13_all_i00_01.nc
   Download the following files for WOA18:
     WOA18/DATA/temperature/netcdf/decav/1.00/woa18_decav_t00_01.nc
     WOA18/DATA/salinity/netcdf/decav/1.00/woa18_decav_s00_01.nc


Modification history
   20210105-malinina_elizaveta: adapting to WOA18
   20130328-lovato_tomas: cmorizer revision
   20190131-predoi_valeriu: adapted to v2.
   20190131-demora_lee: written.

"""
import cftime
import cf_units
from datetime import datetime
import logging
import os

import iris

from .utilities import (constant_metadata, convert_timeunits, fix_coords,
                        fix_var_metadata, save_variable, set_global_atts)

logger = logging.getLogger(__name__)


def _fix_data(cube, var):
    """Specific data fixes for different variables."""
    logger.info("Fixing data ...")
    with constant_metadata(cube):
        mll_to_mol = ['po4', 'si', 'no3']
        if var in mll_to_mol:
            cube /= 1000.  # Convert from ml/l to mol/m^3
        elif var == 'thetao':
            cube += 273.15  # Convert to Kelvin
        elif var == 'o2':
            cube *= 44.661 / 1000.  # Convert from ml/l to mol/m^3
    return cube


def _fix_time_coord(cube):

    time_start = datetime.fromisoformat(cube.attributes['time_coverage_start'])
    time_end = datetime.fromisoformat(cube.attributes['time_coverage_end'])

    origin = 'days since 1850-01-01 00:00:00'
    calendar = 'gregorian'
    bounds = cftime.date2num([time_start, time_end], origin, calendar= calendar)
    time = bounds.mean() - 1 # this is because the time_coverage_ends
    # on 12/31 at 00, so otherwise it kicks the average to the 2nd of july

    cube.coord('time').points = time
    cube.coord('time').bounds = bounds
    cube.coord('time').units = cf_units.Unit(origin, calendar)

    return cube


def extract_variable(var_info, raw_info, out_dir, attrs, year, use_time_attr=False):
    """Extract to all vars."""
    var = var_info.short_name
    cubes = iris.load(raw_info['file'])
    rawvar = raw_info['name']

    for cube in cubes:
        if cube.var_name == rawvar:
            fix_var_metadata(cube, var_info)
            if use_time_attr==True:
                _fix_time_coord(cube)
            else:
                convert_timeunits(cube, year)
            fix_coords(cube)
            _fix_data(cube, var)
            set_global_atts(cube, attrs)
            save_variable(
                cube, var, out_dir, attrs, unlimited_dimensions=['time'])


def cmorization(in_dir, out_dir, cfg, _):
    """Cmorization func call."""
    cmor_table = cfg['cmor_table']
    glob_attrs = cfg['attributes']

    # run the cmorization
    for var, vals in cfg['variables'].items():
        inpfile = os.path.join(in_dir, vals['file'])
        logger.info("CMORizing var %s from file %s", var, inpfile)
        use_time_attrs = cfg['custom']['use_time_attrs']
        yr = cfg['custom']['years']
        var_info = cmor_table.get_variable(vals['mip'], var)
        raw_info = {'name': vals['raw'], 'file': inpfile}
        glob_attrs['mip'] = vals['mip']
        extract_variable(var_info, raw_info, out_dir, glob_attrs, yr, use_time_attrs)
