"""
Diagnostic tools
================

This module contains several python tools used elsewhere by the ocean
diagnostics package.

This tool is part of the ocean diagnostic tools package in the ESMValTool.

Author: Lee de Mora (PML)
    ledm@pml.ac.uk
"""
import logging
import os
import sys
import iris

import numpy as np
import cftime
import matplotlib
import matplotlib.pyplot as plt
import yaml

from esmvaltool.diag_scripts.shared._base import _get_input_data_files

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def get_obs_projects():
    """
    Return a list of strings with the names of observations projects.

    Please keep this list up to date, or replace it with something more
    sensible.

    Returns
    ---------
    list
        Returns a list of strings of the various types of observational data.
    """
    obs_projects = [
        'obs4mips',
    ]
    return obs_projects


def folder(name):
    """
    Make a directory out of a string or list or strings.

    Take a string or a list of strings, convert it to a directory style,
    then make the folder and the string.
    Returns folder string and final character is always os.sep. ('/')

    Arguments
    ---------
    name: list or string
        A list of nested directories, or a path to a directory.

    Returns
    ---------
    str
        Returns a string of a full (potentially new) path of the directory.
    """
    sep = os.sep
    if isinstance(name, list):
        name = os.sep.join(name)
    if name[-1] != sep:
        name = name + sep
    if os.path.exists(name) is False:
        os.makedirs(name)
        logger.info('Making new directory:\t%s', str(name))
    return name


def get_input_files(cfg, index=''):
    """
    Load input configuration file as a Dictionairy.

    Get a dictionary with input files from the metadata.yml files.
    This is a wrappper for the _get_input_data_files function from
    diag_scripts.shared._base.

    Arguments
    ---------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.
    index: int
        the index of the file in the cfg file.

    Returns
    ---------
    dict
        A dictionairy of the input files and their linked details.
    """
    if isinstance(index, int):
        metadata_file = cfg['input_files'][index]
        with open(metadata_file) as input_file:
            metadata = yaml.safe_load(input_file)
        return metadata
    return _get_input_data_files(cfg)


def bgc_units(cube, name):
    """
    Convert the cubes into some friendlier units.

    This is because many CMIP standard units are not the standard units
    used by the BGC community (ie, Celsius is prefered over Kelvin, etc.)

    Parameters
    ----------
    cube: iris.cube.Cube
        the opened dataset as a cube.
    name: str
        The string describing the data field.

    Returns
    -------
    iris.cube.Cube
        the cube with the new units.
    """
    new_units = ''
    if name in ['tos', 'thetao']:
        new_units = 'celsius'

    if name in ['no3', ]:
        new_units = 'mmol m-3'

    if name in ['chl', ]:
        new_units = 'mg m-3'

    if name in ['zostoga', ]:
        new_units = 'mm'
    if name in ['intpp', ]:
        new_units = 'mol m-2 d-1'

    if name in ['fgco2', ]:
        new_units = 'g m-2 d-1'

    if name in ['spco2', 'dpco2', ]:
        new_units = 'uatm'

    if name in ['mfo', 'amoc', 'msftmyz', 'msftyz', 'msftmz']:
        # sverdrup are 1000000 m3.s-1, but mfo is kg s-1.
        new_units = 'Tg s-1'

    if new_units != '':
        logger.info(' '.join(
            ["Changing units from",
             str(cube.units), 'to', new_units]))
        cube.convert_units(new_units)

    return cube


def match_model_to_key(
        model_type,
        cfg_dict,
        input_files_dict,
):
    """
    Match up model or observations dataset dictionairies from config file.

    This function checks that the control_model, exper_model and
    observational_dataset dictionairies from the recipe are matched with the
    input file dictionairy in the cfg metadata.

    Arguments
    ---------
    model_type: str
        The string model_type to match (only used in debugging).
    cfg_dict: dict
        the config dictionairy item for this model type, parsed directly from
        the diagnostics/ scripts, part of the recipe.
    input_files_dict: dict
        The input file dictionairy, loaded directly from the get_input_files()
         function, in diagnostics_tools.py.

    Returns
    ---------
    dict
        A dictionairy of the input files and their linked details.
    """
    for input_file, intput_dict in input_files_dict.items():
        intersect_keys = intput_dict.keys() & cfg_dict.keys()
        match = True
        for key in intersect_keys:
            if intput_dict[key] == cfg_dict[key]:
                continue
            match = False
        if match:
            return input_file
    logger.warning("Unable to match model: %s", model_type)
    return ''


def cube_time_to_float(cube):
    """
    Convert from time coordinate into decimal time.

    Takes an iris time coordinate and returns a list of floats.
    Parameters
    ----------
    cube: iris.cube.Cube
        the opened dataset as a cube.

    Returns
    -------
    list
        List of floats showing the time coordinate in decimal time.

    """
    times = cube.coord('time')
    datetime = guess_calendar_datetime(cube)

    dtimes = times.units.num2date(times.points)
    floattimes = []
    for dtime in dtimes:
        # TODO: it would be better to have a calendar dependent value
        # for daysperyear, as this is not accurate for 360 day calendars.
        if times.units.calendar == '365_day':
            daysperyear = 365.
        elif times.units.calendar == '360_day':
            daysperyear = 360.
        elif times.units.calendar in ['gregorian', 'proleptic_gregorian', 'julian']:
            daysperyear = 365.25
        else:
            print("Calendar Not recoginised:", str(times.units.calendar))
            assert 0

        try:
            dayofyr = dtime.dayofyr
        except AttributeError:
            time = datetime(dtime.year, dtime.month, dtime.day)
            time0 = datetime(dtime.year, 1, 1, 0, 0)
            dayofyr = (time - time0).days

        floattime = dtime.year + dayofyr / daysperyear + dtime.hour / (
            24. * daysperyear)
        if dtime.hour:
            floattime += dtime.hour / (24. * daysperyear)
        if dtime.minute:
            floattime += dtime.minute / (24. * 60. * daysperyear)
        floattimes.append(floattime)
    return floattimes


def load_calendar_datetime(calendar):
    """
    Load a cftime.datetime calendar..

    Parameters
    ----------
    calendar: str
        the caendar name.

    Returns
    -------
    cftime.datetime
        A datetime creator function from cftime, based on the requeted calendar.
    """
    calendar = calendar.lower()

    if calendar in ['360_day', ]:
        datetime = cftime.Datetime360Day
    elif calendar in ['365_day', 'noleap']:
        datetime = cftime.DatetimeNoLeap
    elif calendar in ['julian', ]:
        datetime = cftime.DatetimeJulian
    elif calendar in ['gregorian', ]:
        datetime = cftime.DatetimeGregorian
    elif calendar in ['proleptic_gregorian', ]:
        datetime = cftime.DatetimeProlepticGregorian
    else:
        logger.warning('Calendar set to Gregorian, instead of %s',
                       calendar)
        datetime = cftime.DatetimeGregorian
    return datetime

def guess_calendar_datetime(cube):
    """
    Guess the cftime.datetime form to create datetimes.

    Parameters
    ----------
    cube: iris.cube.Cube
        the opened dataset as a cube.

    Returns
    -------
    cftime.datetime
        A datetime creator function from cftime, based on the cube's calendar.
    """
    time_coord = cube.coord('time')

    if time_coord.units.calendar in ['360_day', ]:
        datetime = cftime.Datetime360Day
    elif time_coord.units.calendar in ['365_day', 'noleap']:
        datetime = cftime.DatetimeNoLeap
    elif time_coord.units.calendar in ['julian', ]:
        datetime = cftime.DatetimeJulian
    elif time_coord.units.calendar in ['gregorian', ]:
        datetime = cftime.DatetimeGregorian
    elif time_coord.units.calendar in ['proleptic_gregorian', ]:
        datetime = cftime.DatetimeProlepticGregorian
    else:
        logger.warning('Calendar set to Gregorian, instead of %s',
                       time_coord.units.calendar)
        datetime = cftime.DatetimeGregorian
    return datetime


def get_decade(coord, value):
    """
    Determine the decade.

    Called by iris.coord_categorisation.add_categorised_coord.
    """
    date = coord.units.num2date(value)
    return date.year - date.year % 10


def decadal_average(cube):
    """
    Calculate the decadal_average.

    Parameters
    ----------
    cube: iris.cube.Cube
        The input cube

    Returns
    -------
    iris.cube
    """
    iris.coord_categorisation.add_categorised_coord(cube, 'decade', 'time',
                                                    get_decade)
    return cube.aggregated_by('decade', iris.analysis.MEAN)


def load_thresholds(cfg, metadata):
    """
    Load the thresholds for contour plots from the config files.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.
    metadata: dict
        the metadata dictionairy

    Returns
    -------
    list:
        List of thresholds
    """
    thresholds = set()

    if 'threshold' in cfg:
        thresholds.update(float(cfg['threshold']))

    if 'threshold' in metadata:
        thresholds.update(float(metadata['threshold']))

    if 'thresholds' in cfg:
        thresholds.update([float(thres) for thres in cfg['thresholds']])

    if 'thresholds' in metadata:
        thresholds.update([float(thres) for thres in metadata['thresholds']])

    return sorted(list(thresholds))


def get_colour_from_cmap(number, total, cmap='jet'):
    """
    Get a colour `number` of `total` from a cmap.

    This function is used when several lines are created evenly along a
    colour map.

    Parameters
    ----------
    number: int, float
        The
    total: int

    cmap: string,  plt.cm
        A colour map, either by name (string) or from matplotlib
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if number > total:
        raise ValueError('The cannot be larger than the total length '
                         'of the list ie: {} > {}'.format(number, total))

    if total > 1:
        colour = cmap(float(number) / float(total - 1.))
    else:
        colour = cmap(0.)
    return colour


def add_legend_outside_right(plot_details, ax1, column_width=0.1, loc='right',
        fontsize = 'small', order = [], nrows = 25, ncols=4):
    """
    Add a legend outside the plot, to the right.

    plot_details is a 2 level dict,
    where the first level is some key (which is hidden)
    and the 2nd level contains the keys:
    'c': color
    'lw': line width
    'label': label for the legend.
    ax1 is the axis where the plot was drawn.

    Parameters
    ----------
    plot_details: dict
        A dictionary of the plot details (color, linestyle, linewidth, label)
    ax1: matplotlib.pyplot.axes
        The pyplot axes to add the
    column_width: float
        The width of the legend column. This is used to adjust for longer words
        in the legends
    loc: string
       Location of the legend. Options are "right" and "below".

    Returns
    -------
    cftime.datetime
        A datetime creator function from cftime, based on the cube's calendar.

    """
    # ####
    # Create dummy axes:
    legend_size = len(plot_details) + 1
    box = ax1.get_position()
    if loc.lower() == 'right':
        #nrows = 25
        ncols = int(legend_size / nrows) + 1
        ax1.set_position([
            box.x0, box.y0, box.width * (1. - column_width * ncols), box.height
        ])

    if loc.lower() == 'below':
        #ncols = 4
        nrows = int(legend_size / ncols) + 1
        ax1.set_position([
            box.x0, box.y0 + (nrows * column_width), box.width,
            box.height - (nrows * column_width)
        ])

    # Add emply plots to dummy axis.
    if order == []:
        order = sorted(plot_details.keys())

    for index in order:
        colour = plot_details[index]['c']

        linewidth = plot_details[index].get('lw', 1)

        linestyle = plot_details[index].get('ls', '-')

        label = plot_details[index].get('label', str(index))

        plt.plot([], [], c=colour, lw=linewidth, ls=linestyle, label=label)

    if loc.lower() == 'right':
        legd = ax1.legend(
            loc='center left',
            ncol=ncols,
            prop={'size': 10},
            bbox_to_anchor=(1., 0.5),
            fontsize=fontsize)
    if loc.lower() == 'below':
        legd = ax1.legend(
            loc='upper center',
            ncol=ncols,
            prop={'size': 10},
            bbox_to_anchor=(0.5, -2. * column_width),
            fontsize=fontsize)
    legd.draw_frame(False)
    legd.get_frame().set_alpha(0.)


def get_image_format(cfg, default='png'):
    """
    Load the image format from the global config file.

    Current tested options are svg, png.

    The cfg is the opened global config.
    The default format is used if no specific format is requested.
    The default is set in the user config.yml
    Individual diagnostics can set their own format which will
    supercede the main config.yml.

    Arguments
    ---------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.

    Returns
    ---------
    str
        The image format extention.
    """
    image_extention = default

    # Load format from config.yml and set it as default
    if 'output_file_type' in cfg:
        image_extention = cfg['output_file_type']

    # Load format from config.yml and set it as default
    if 'image_format' in cfg:
        image_extention = cfg['image_format']

    matplotlib_image_formats = plt.gcf().canvas.get_supported_filetypes()
    if image_extention not in matplotlib_image_formats:
        logger.warning(' '.join([
            'Image format ', image_extention, 'not in matplot:',
            ', '.join(matplotlib_image_formats)
        ]))

    image_extention = '.' + image_extention
    image_extention = image_extention.replace('..', '.')
    return image_extention


def get_image_path(
        cfg,
        metadata,
        prefix='diag',
        suffix='image',
        metadata_id_list='default',
):
    """
    Produce a path to the final location of the image.

    The cfg is the opened global config,
    metadata is the metadata dictionairy (for the individual dataset file)

    Arguments
    ---------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.
    metadata: dict
        The metadata dictionairy for a specific model.
    prefix: str
        A string to prepend to the image basename.
    suffix: str
        A string to append to the image basename
    metadata_id_list: list
        A list of strings to add to the file path. It loads these from the cfg.

    Returns
    ---------
    str
        The ultimate image path

    """
    #####
    if metadata_id_list == 'default':
        metadata_id_list = [
            'project',
            'dataset',
            'mip',
            'exp',
            'ensemble',
            'field',
            'short_name',
            'preprocessor',
            'diagnostic',
            'start_year',
            'end_year',
        ]

    path = folder(cfg['plot_dir'])
    if prefix:
        path += prefix + '_'
    # Check that the keys are in the dict.
    intersection = [va for va in metadata_id_list if va in metadata]
    path += '_'.join([str(metadata[b]) for b in intersection])
    if suffix:
        path += '_' + suffix

    image_extention = get_image_format(cfg)

    if path.find(image_extention) == -1:
        path += image_extention

    path = path.replace(' ', '_')

    logger.info("Image path will be: %s", path)
    return path


def make_cube_layer_dict(cube):
    """
    Take a cube and return a dictionairy layer:cube

    Each item in the dict is a layer with a separate cube for each layer.
    ie: cubes[depth] = cube from specific layer

    Cubes with no depth component are returned as dict, where the dict key
    is a blank empty string, and the value is the cube.

    Parameters
    ----------
    cube: iris.cube.Cube
        the opened dataset as a cube.

    Returns
    ---------
    dict
        A dictionairy of layer name : layer cube.
    """
    #####
    # Check layering:
    coords = cube.coords()
    layers = []
    for coord in coords:
        if coord.standard_name in ['depth', 'region']:
            layers.append(coord)

    cubes = {}
    if layers == []:
        cubes[''] = cube
        return cubes

    # if len(layers) > 1:
    #     # This field has a strange number of layer dimensions.
    #     # depth and regions?
    #     print(cube)
    #     raise ValueError('This cube has both `depth` & `region` coordinates:'
    #                      ' %s', layers)

    # iris stores coords as a list with one entry:
    layer_dim = layers[0]
    if len(layer_dim.points) in [
            1,
    ]:
        cubes[''] = cube
        return cubes

    if layer_dim.standard_name == 'depth':
        coord_dim = cube.coord_dims('depth')[0]
        for layer_index, layer in enumerate(layer_dim.points):
            slices = [slice(None) for index in cube.shape]
            slices[coord_dim] = layer_index
            cubes[layer] = cube[tuple(slices)]

    if layer_dim.standard_name == 'region':
        coord_dim = cube.coord_dims('region')[0]
        for layer_index, layer in enumerate(layer_dim.points):
            slices = [slice(None) for index in cube.shape]
            slices[coord_dim] = layer_index
            layer = layer.replace('_', ' ').title()
            cubes[layer] = cube[tuple(slices)]
    return cubes


def get_cube_range(cubes):
    """
    Determinue the minimum and maximum values of a list of cubes.

    Parameters
    ----------
    cubes: list of iris.cube.Cube
        A list of cubes.

    Returns
    ----------
    list:
        A list of two values: the overall minumum and maximum values of the
        list of cubes.

    """
    mins = []
    maxs = []
    for cube in cubes:
        mins.append(cube.data.min())
        maxs.append(cube.data.max())
    return [np.min(mins), np.max(maxs), ]


def get_cube_range_diff(cubes):
    """
    Determinue the largest deviation from zero in an list of cubes.

    Parameters
    ----------
    cubes: list of iris.cube.Cube
        A list of cubes.

    Returns
    ----------
    list:
        A list of two values: the maximum deviation from zero and its opposite.
    """
    ranges = []
    for cube in cubes:
        ranges.append(np.abs(cube.data.min()))
        ranges.append(np.abs(cube.data.max()))
    return [-1. * np.max(ranges), np.max(ranges)]


def get_array_range(arrays):
    """
    Determinue the minimum and maximum values of a list of arrays..

    Parameters
    ----------
    arrays: list of numpy.array
        A list of numpy.array.

    Returns
    ----------
    list:
        A list of two values, the overall minumum and maximum values of the
        list of cubes.
    """
    mins = []
    maxs = []
    for arr in arrays:
        mins.append(arr.min())
        maxs.append(arr.max())
    logger.info('get_array_range: %s, %s', np.min(mins), np.max(maxs))
    return [np.min(mins), np.max(maxs), ]


misc_div_colors_list = [
    (8, 29, 88),
    (9, 30, 91),
    (11, 31, 95),
    (13, 33, 99),
    (15, 34, 103),
    (17, 36, 106),
    (18, 37, 110),
    (20, 39, 114),
    (22, 40, 118),
    (24, 42, 121),
    (26, 43, 125),
    (28, 44, 129),
    (29, 46, 133),
    (31, 47, 137),
    (33, 49, 140),
    (35, 50, 144),
    (36, 52, 148),
    (36, 54, 149),
    (36, 57, 150),
    (36, 60, 151),
    (36, 62, 153),
    (36, 65, 154),
    (35, 68, 155),
    (35, 70, 156),
    (35, 73, 158),
    (35, 76, 159),
    (35, 78, 160),
    (34, 81, 161),
    (34, 83, 163),
    (34, 86, 164),
    (34, 89, 165),
    (34, 91, 166),
    (33, 94, 168),
    (33, 97, 169),
    (33, 101, 171),
    (32, 104, 172),
    (32, 107, 174),
    (32, 110, 175),
    (32, 113, 177),
    (31, 117, 178),
    (31, 120, 180),
    (31, 123, 181),
    (30, 126, 183),
    (30, 129, 184),
    (30, 133, 186),
    (29, 136, 187),
    (29, 139, 189),
    (29, 142, 190),
    (29, 145, 192),
    (31, 147, 192),
    (34, 150, 192),
    (36, 152, 192),
    (38, 154, 193),
    (40, 157, 193),
    (43, 159, 193),
    (45, 161, 193),
    (47, 164, 194),
    (50, 166, 194),
    (52, 168, 194),
    (54, 171, 194),
    (56, 173, 195),
    (59, 175, 195),
    (61, 178, 195),
    (63, 180, 195),
    (66, 182, 195),
    (70, 183, 195),
    (74, 185, 194),
    (78, 186, 194),
    (82, 188, 193),
    (85, 189, 192),
    (89, 191, 192),
    (93, 192, 191),
    (97, 194, 191),
    (101, 195, 190),
    (105, 197, 190),
    (109, 198, 189),
    (113, 199, 188),
    (117, 201, 188),
    (121, 202, 187),
    (124, 204, 187),
    (129, 205, 186),
    (133, 207, 186),
    (138, 209, 185),
    (142, 211, 185),
    (147, 212, 185),
    (151, 214, 184),
    (156, 216, 184),
    (160, 218, 183),
    (165, 219, 183),
    (169, 221, 182),
    (174, 223, 182),
    (178, 225, 181),
    (183, 226, 181),
    (187, 228, 181),
    (192, 230, 180),
    (197, 232, 180),
    (200, 233, 179),
    (202, 234, 179),
    (205, 235, 179),
    (207, 236, 179),
    (209, 237, 179),
    (212, 238, 178),
    (214, 239, 178),
    (217, 240, 178),
    (219, 241, 178),
    (221, 242, 178),
    (224, 242, 178),
    (226, 243, 177),
    (229, 244, 177),
    (231, 245, 177),
    (233, 246, 177),
    (236, 247, 177),
    (237, 248, 178),
    (238, 248, 181),
    (240, 249, 183),
    (241, 249, 186),
    (242, 250, 188),
    (243, 250, 191),
    (244, 250, 193),
    (245, 251, 196),
    (246, 251, 198),
    (247, 252, 201),
    (249, 252, 203),
    (250, 253, 206),
    (251, 253, 208),
    (252, 254, 211),
    (253, 254, 213),
    (254, 254, 216),
    (255, 254, 203),
    (255, 253, 200),
    (255, 252, 197),
    (255, 251, 195),
    (255, 250, 192),
    (255, 249, 189),
    (255, 247, 186),
    (255, 246, 183),
    (255, 245, 181),
    (255, 244, 178),
    (255, 243, 175),
    (255, 242, 172),
    (255, 241, 170),
    (255, 240, 167),
    (255, 238, 164),
    (255, 237, 161),
    (254, 236, 159),
    (254, 235, 156),
    (254, 234, 153),
    (254, 232, 151),
    (254, 231, 148),
    (254, 230, 145),
    (254, 229, 143),
    (254, 227, 140),
    (254, 226, 137),
    (254, 225, 135),
    (254, 223, 132),
    (254, 222, 130),
    (254, 221, 127),
    (254, 220, 124),
    (254, 218, 122),
    (254, 217, 119),
    (254, 215, 116),
    (254, 213, 114),
    (254, 211, 111),
    (254, 208, 108),
    (254, 206, 106),
    (254, 203, 103),
    (254, 201, 101),
    (254, 198, 98),
    (254, 196, 95),
    (254, 193, 93),
    (254, 191, 90),
    (254, 188, 87),
    (254, 186, 85),
    (254, 184, 82),
    (254, 181, 79),
    (254, 179, 77),
    (253, 176, 75),
    (253, 174, 74),
    (253, 172, 73),
    (253, 169, 72),
    (253, 167, 71),
    (253, 165, 70),
    (253, 162, 69),
    (253, 160, 68),
    (253, 158, 67),
    (253, 155, 66),
    (253, 153, 65),
    (253, 151, 64),
    (253, 148, 63),
    (253, 146, 62),
    (253, 144, 61),
    (253, 141, 60),
    (252, 138, 59),
    (252, 134, 58),
    (252, 130, 57),
    (252, 126, 55),
    (252, 122, 54),
    (252, 118, 53),
    (252, 114, 52),
    (252, 110, 51),
    (252, 106, 50),
    (252, 102, 49),
    (252, 98, 47),
    (252, 94, 46),
    (252, 90, 45),
    (252, 87, 44),
    (252, 83, 43),
    (252, 79, 42),
    (250, 75, 41),
    (249, 72, 40),
    (247, 69, 39),
    (246, 65, 38),
    (244, 62, 37),
    (243, 59, 36),
    (241, 56, 36),
    (239, 52, 35),
    (238, 49, 34),
    (236, 46, 33),
    (235, 42, 32),
    (233, 39, 31),
    (232, 36, 30),
    (230, 33, 29),
    (228, 29, 29),
    (227, 26, 28),
    (225, 24, 28),
    (222, 23, 29),
    (220, 21, 29),
    (217, 19, 30),
    (215, 18, 31),
    (213, 16, 31),
    (210, 14, 32),
    (208, 13, 32),
    (205, 11, 33),
    (203, 9, 34),
    (201, 8, 34),
    (198, 6, 35),
    (196, 5, 36),
    (194, 3, 36),
    (191, 1, 37),
    (189, 0, 37),
    (185, 0, 38),
    (181, 0, 38),
    (177, 0, 38),
    (174, 0, 38),
    (170, 0, 38),
    (166, 0, 38),
    (162, 0, 38),
    (158, 0, 38),
    (154, 0, 38),
    (151, 0, 38),
    (147, 0, 38),
    (143, 0, 38),
    (139, 0, 38),
    (135, 0, 38),
    (131, 0, 38),
    (128, 0, 38)]
misc_div_colors_list = np.array(misc_div_colors_list)/256.
misc_div = matplotlib.colors.ListedColormap(misc_div_colors_list, name='misc_div', N=None)

misc_seq_colors_list = [
    [25, 51, 178], [26, 51, 178], [26, 52, 177], [27, 53, 177], [27, 53, 177],
    [28, 54, 176], [28, 55, 176], [28, 56, 176], [29, 56, 175], [29, 57, 175],
    [29, 58, 175], [30, 58, 174], [30, 59, 174], [31, 60, 174], [31, 60, 173],
    [31, 61, 173], [32, 62, 172], [32, 62, 172], [32, 63, 172], [33, 64, 171],
    [33, 65, 171], [33, 65, 171], [34, 66, 170], [34, 67, 170], [34, 67, 170],
    [35, 68, 169], [35, 69, 169], [35, 69, 169], [36, 70, 168], [36, 71, 168],
    [36, 71, 168], [37, 72, 167], [37, 72, 167], [37, 73, 167], [38, 74, 166],
    [38, 74, 166], [38, 75, 166], [39, 76, 165], [39, 76, 165], [39, 77, 165],
    [40, 78, 164], [40, 78, 164], [40, 79, 164], [40, 80, 163], [41, 80, 163],
    [41, 81, 163], [41, 82, 162], [42, 82, 162], [42, 83, 162], [42, 84, 161],
    [43, 84, 161], [43, 85, 161], [43, 86, 160], [44, 86, 160], [44, 87, 160],
    [44, 88, 159], [45, 88, 159], [45, 89, 159], [45, 90, 158], [46, 90, 158],
    [46, 91, 158], [46, 91, 157], [47, 92, 157], [47, 93, 157], [48, 93, 156],
    [48, 94, 156], [48, 95, 156], [49, 95, 155], [49, 96, 155], [49, 97, 154],
    [50, 97, 154], [50, 98, 154], [51, 98, 153], [51, 99, 153], [52, 100, 152],
    [52, 100, 152], [52, 101, 152], [53, 101, 151], [53, 102, 151],
    [54, 103, 150], [54, 103, 150], [55, 104, 149], [55, 104, 149],
    [56, 105, 148], [56, 106, 148], [57, 106, 147], [57, 107, 147],
    [58, 107, 146], [58, 108, 146], [59, 109, 145], [59, 109, 145],
    [60, 110, 144], [60, 110, 144], [61, 111, 143], [61, 111, 143],
    [62, 112, 142], [63, 112, 141], [63, 113, 141], [64, 114, 140],
    [64, 114, 140], [65, 115, 139], [65, 115, 139], [66, 116, 138],
    [66, 117, 138], [67, 117, 137], [68, 118, 137], [68, 118, 136],
    [69, 119, 136], [69, 120, 135], [70, 120, 135], [71, 121, 134],
    [71, 121, 134], [72, 122, 133], [72, 123, 133], [73, 123, 132],
    [74, 124, 132], [74, 125, 131], [75, 125, 131], [76, 126, 130],
    [77, 127, 130], [77, 128, 129], [78, 128, 129], [79, 129, 129],
    [79, 130, 128], [80, 131, 128], [81, 131, 127], [82, 132, 127],
    [83, 133, 127], [83, 134, 126], [84, 135, 126], [85, 135, 126],
    [86, 136, 125], [87, 137, 125], [88, 138, 125], [88, 139, 125],
    [89, 140, 124], [90, 141, 124], [91, 141, 124], [92, 142, 123],
    [93, 143, 123], [94, 144, 123], [95, 145, 123], [96, 146, 122],
    [96, 147, 122], [97, 148, 122], [98, 149, 122], [99, 150, 121],
    [100, 151, 121], [101, 152, 121], [102, 153, 121], [103, 153, 121],
    [104, 154, 120], [105, 155, 120], [106, 156, 120], [107, 157, 120],
    [108, 158, 119], [109, 159, 119], [110, 160, 119], [111, 161, 119],
    [112, 162, 118], [113, 163, 118], [114, 164, 118], [115, 165, 118],
    [115, 166, 117], [116, 167, 117], [117, 168, 117], [118, 169, 117],
    [119, 170, 116], [120, 171, 116], [121, 172, 116], [122, 173, 116],
    [123, 174, 115], [124, 175, 115], [125, 176, 115], [126, 177, 115],
    [127, 178, 114], [128, 179, 114], [130, 180, 114], [131, 181, 114],
    [132, 182, 113], [133, 183, 113], [134, 184, 113], [135, 185, 113],
    [136, 186, 112], [137, 187, 112], [138, 189, 112], [139, 190, 112],
    [140, 191, 111], [141, 192, 111], [142, 193, 111], [143, 194, 111],
    [144, 195, 110], [145, 196, 110], [146, 197, 110], [147, 198, 109],
    [148, 199, 109], [150, 200, 109], [151, 201, 109], [152, 202, 108],
    [153, 204, 108], [154, 205, 108], [155, 206, 108], [156, 207, 107],
    [158, 208, 107], [159, 209, 107], [160, 210, 107], [161, 211, 106],
    [163, 212, 106], [164, 213, 106], [165, 215, 106], [167, 216, 105],
    [168, 217, 105], [170, 218, 105], [171, 219, 105], [173, 220, 104],
    [174, 221, 104], [176, 222, 104], [177, 223, 104], [179, 224, 104],
    [181, 225, 103], [183, 226, 103], [184, 227, 103], [186, 228, 103],
    [188, 229, 103], [190, 230, 103], [192, 231, 103], [194, 232, 102],
    [196, 233, 102], [198, 234, 102], [200, 235, 102], [202, 236, 102],
    [204, 237, 102], [206, 237, 102], [208, 238, 102], [210, 239, 102],
    [212, 240, 102], [214, 241, 102], [217, 241, 102], [219, 242, 102],
    [221, 243, 102], [223, 244, 102], [225, 245, 102], [227, 245, 102],
    [229, 246, 102], [231, 247, 102], [233, 247, 102], [236, 248, 102],
    [238, 249, 102], [240, 250, 102], [242, 250, 102], [244, 251, 102],
    [246, 252, 102], [248, 252, 102], [250, 253, 102], [252, 254, 102],
    [255, 254, 102]]

misc_seq_colors_list = np.array(misc_seq_colors_list)/256.
misc_seq = matplotlib.colors.ListedColormap(misc_seq_colors_list, name='misc_seq', N=None)


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False, method='standard'):
    """
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.

    taken from https://stackoverflow.com/a/29677616
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    if method == 'standard':
        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    elif method == 'cdf':
        weighted_quantiles = np.cumsum(sample_weight)
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)
