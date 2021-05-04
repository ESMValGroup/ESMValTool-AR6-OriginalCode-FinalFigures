"""ESMValTool CMORizer for OSISAF data from Dirk Notz.

Tier
    Tier 3: Restricted dataset

Source
    Private communication

Download and processing instructions
    Request the dataset from Dirk Notz.

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

def _reshape_cube(cube):

    time_dim = cube.coord('time')
    time_dim.guess_bounds()

    # data = cube.data.reshape(cube.data.shape[0])
    data = cube.data
    data.mask = np.isnan(data.data)

    new_cube = iris.cube.Cube(data, standard_name=cube.standard_name, long_name='sea ice area',
                              var_name=cube.var_name, units=cube.units, attributes='',
                              cell_methods=cube.cell_methods,
                              dim_coords_and_dims=[(time_dim, 0)])

    return(new_cube)

def _extract_variable(var_info, cmor_info, attrs, filepath, cube_name,
                      title_did, out_dir):

    """Extract variable."""
    var = cmor_info.short_name
    cube = iris.load_cube(filepath, cube_name)
    # Fix units
    cube.units = var_info.get('raw_units', var)
    cube.convert_units(cmor_info.units)

    attrs['dataset_id'] = attrs['dataset_base_id'] + '-' + title_did

    out_sub_dir = os.path.join(out_dir, attrs['dataset_id'])

    if not os.path.isdir(out_sub_dir):
        os.makedirs(out_sub_dir)
        logger.info("Subdirectory was created : %s \n Output will be written"
                    " there.", out_sub_dir)

    cube = _reshape_cube(cube)
    utils.fix_var_metadata(cube, cmor_info)
    utils.convert_timeunits(cube, 1950)
    utils.set_global_atts(cube, attrs)
    utils.save_variable(cube,
                        var,
                        out_sub_dir,
                        attrs,
                        unlimited_dimensions=['time'])


def cmorization(in_dir, out_dir, cfg, _):
    """Cmorization function call."""
    glob_attrs = cfg['attributes']
    cmor_table = cfg['cmor_table']
    datasets = cfg['custom']

    # Run the cmorization
    for (var, var_info) in cfg['variables'].items():
        filename = var_info['filename']
        filepath = os.path.join(in_dir, filename)
        if os.path.isfile(filepath):
            logger.info("Found input file '%s'", filepath)
        else:
            logger.error("No files '%s' found in '%s' ", filename, in_dir)
        glob_attrs['mip'] = var_info['mip']
        cmor_info = cmor_table.get_variable(var_info['mip'], var)
        for dataset in datasets.keys():
            cube_name = datasets[dataset]['var_name']
            title_did = datasets[dataset]['title_did']
            logger.info("CMORizing variable '%s' in dataset '%s'", var,
                        title_did)
            _extract_variable(var_info, cmor_info, glob_attrs, filepath,
                              cube_name, title_did, out_dir)
