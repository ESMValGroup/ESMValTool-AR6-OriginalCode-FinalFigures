"""ESMValTool CMORizer for HadEX3 data.

Tier
    Tier 2: other freely-available dataset.

Source
    https://www.metoffice.gov.uk/hadobs/hadex3/download.html

Last access
    20201009

Download and processing instructions
    Download the following files:
        Annual.nc.gz
    No registration is required for downloading the data.

When using the dataset in a paper, the following are possible citations to use.

Dunn, R. J. H., et al. (2020), Development of an updated global land in-situ-based
dataset of temperature and precipitation extremes: HadEX3 JGR-A

Dunn, R. J. H., Donat, M. G., Alexander, L. V., al. (2014), Investigating uncertainties
in global gridded datasets of climate extremes, Climate of the Past, 10, 2171-2199

Donat, M. G., Alexander, L. V., .... Dunn, R. J. H., et al. (2013), Updated analyses
of temperature and precipitation extreme indices since the beginning of the twentieth century:
The HadEX2 dataset, J. Geophys. Res. Atmos., 118, 2098-2118
"""

import logging
import os

import iris
from cf_units import Unit
import cftime

from . import utilities as utils

logger = logging.getLogger(__name__)


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


def _extract_variable(var, var_info, cmor_info, attrs, filepath, out_dir):
    """Extract variable."""
    raw_var = var_info.get('raw', var)
    var = cmor_info.short_name
    cube = iris.load_cube(filepath, utils.var_name_constraint(raw_var))
    # Fix units
    cube.units = var_info.get('raw_units', var)
    cube.convert_units(cmor_info.units)

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
        # else:
        #     add here error message
        logger.info("CMORizing variable '%s'", var)
        glob_attrs['mip'] = var_info['mip']
        cmor_info = cmor_table.get_variable(var_info['mip'], var)
        _extract_variable(var, var_info, cmor_info, glob_attrs, filepath,
                          out_dir)
