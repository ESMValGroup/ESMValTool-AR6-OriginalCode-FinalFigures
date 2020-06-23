"""
Transects diagnostics figure 3.17
=================================

Diagnostic to produce images of a transect. These plost show either latitude or
longitude against depth, and the cube value is used as the colour scale.

Note that this diagnostic assumes that the preprocessors do the bulk of the
hard work, and that the cube received by this diagnostic (via the settings.yml
and metadata.yml files) has no time component, and one of the latitude or
longitude coordinates has been reduced to a single value.

An approproate preprocessor for a 3D+time field would be::

  preprocessors:
    prep_transect:
      time_average:
      extract_slice: # Atlantic Meridional Transect
        latitude: [-50.,50.]
        longitude: 332.

This tool is part of the ocean diagnostic tools package in the ESMValTool.

Author: Lee de Mora (PML)
        ledm@pml.ac.uk

"""
import logging
import os
import sys
from itertools import product

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np

from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import run_diagnostic

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# TODO: Fix colour scale. Why can't it do the total range?
# TODO: Add further models and colour ranges
# TODO: Write figure description
# TODO: Add x and y axis labels.
# TODO: Add documentation and PEP8 compliance.

def titlify(title):
    """
    Check whether a title is too long then add it to current figure.

    Parameters
    ----------
    title: str
        The title for the figure.
    """
    cutoff = 40
    if len(title) > cutoff:
        # Find good mid point
        titles = title.split(' ')
        length = 0
        for itr, word in enumerate(titles):
            length += len(word)
            if length > cutoff:
                titles[itr] += '\n'
                length = 0.
        title = ' '.join(titles)
    return title


def determine_transect_str(cube, region=''):
    """
    Determine the Transect String.

    Takes a guess at a string to describe the transect.

    Parameters
    ----------
    cube: iris.cube.Cube
        Input cube to use to determine the transect name.

    """
    if region:
        return region

    options = ['latitude', 'longitude']
    cube_dims = [c.standard_name for c in cube.coords()]
    for option in options:
        if option not in cube_dims:
            continue
        coord = cube.coord(option)

        if len(coord.points) > 1:
            continue
        value = coord.points.mean()
        value = round(value, 2)
        if option == 'latitude':
            return str(value) + ' N'
        if option == 'longitude':
            if value > 180.:
                return str(value - 360.) + ' W'
            return str(value) + ' E'
    return ''


def make_depth_safe(cube):
    """
    Make the depth coordinate safe.

    If the depth coordinate has a value of zero or above, we replace the
    zero with the average point of the first depth layer.

    Parameters
    ----------
    cube: iris.cube.Cube
        Input cube to make the depth coordinate safe

    Returns
    ----------
    iris.cube.Cube:
        Output cube with a safe depth coordinate

    """
    depth = cube.coord('depth')

    # it's fine
    if depth.points.min() * depth.points.max() > 0.:
        return cube

    if depth.attributes['positive'] != 'down':
        raise Exception('The depth field is not set up correctly')

    depth_points = []
    bad_points = depth.points <= 0.
    print('depth points:', depth.points)
    for itr, point in enumerate(depth.points):
        if bad_points[itr]:
            depth_points.append(depth.bounds[itr, :].mean())
        else:
            depth_points.append(point)

    cube.coord('depth').points = depth_points
    return cube


def make_cube_region_dict(cube):
    """
    Take a cube and return a dictionairy region: cube.

    Each item in the dict is a layer with a separate cube for each layer.
    ie: cubes[region] = cube from specific region

    Cubes with no region component are returns as:
    cubes[''] = cube with no region component.

    This is based on the method diagnostics_tools.make_cube_layer_dict,
    however, it wouldn't make sense to look for depth layers here.

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
        if coord.standard_name in ['region', ]:
            layers.append(coord)

    cubes = {}
    if layers == []:
        cubes[''] = cube
        return cubes

    # iris stores coords as a list with one entry:
    layer_dim = layers[0]
    if len(layer_dim.points) in [1, ]:
        cubes[''] = cube
        return cubes

    if layer_dim.standard_name == 'region':
        coord_dim = cube.coord_dims('region')[0]
        for layer_index, layer in enumerate(layer_dim.points):
            slices = [slice(None) for index in cube.shape]
            slices[coord_dim] = layer_index
            layer = layer.replace('_', ' ').title()
            cubes[layer] = cube[tuple(slices)]
    return cubes


def make_mean_of_cube_list(cube_list, long_name):
    """
    Takes the mean of a list of cubes (not an iris.cube.CubeList).

    Assumes all the cubes are the same shape.
    """
    removes = []
    cube_names = sorted(cube_list.keys())
    for cube_name in cube_names:
        cube = cube_list[cube_name]
        if str(cube.coord(axis='Z').units) in['centimeters', 'cm']:
            print('This stupid model uses centimeters in the z dimension:', cube_name) 
            removes.append(cube_name)
            continue
        cube = standardize_depth_coord(cube)
    for cube_name in removes:
        del cube_list[cube_name]
    
    # Remake list after remving fails
    cube_names = sorted(cube_list.keys())

    cube_mean = cube_list[cube_names[0]]
    #if long_name == 'Sea Water Potential Temperature':
    #        units_str = 'kelvin'
    #if long_name == 'Sea Water Salinity':
    #        units_str = '1.'
    #print('\n\ntaking mean:\n',cube_mean.metadata[4]['source_id'], cube_mean.coord(axis='Z'))

    #print ('make_mean_of_cube_list: cube -1', cube_mean.data.mean())
    length = 1
    models_includes = {'pass' : [], 'fail':[]}
    for i, cube_name in enumerate(cube_names[1:]):
        cube = cube_list[cube_name]
        #cube.units = units_str
        #cube_mean.units = units_str
        mean = cube.data.mean()
        print(i, cube_name, cube.units, mean)
        if np.ma.is_masked(mean):
            models_includes['fail'].append(cube_name)
            continue


        if long_name == 'Sea Water Potential Temperature':
            if mean < -200.:
                cube.data += 273.15

        if long_name == 'Sea Water Salinity':
            if 0.02 < mean < 0.04:
                cube.data = cube.data * 1000.
            if abs(mean) < 1E-10:
                models_includes['fail'].append(cube_name)
                continue
 
        try:
            cube_mean+=cube
            length+=1
            models_includes['pass'].append(cube_name)
        except: 
            models_includes['fail'].append(cube_name)
        
    print("\n---------\nNote that the mean only includes ", length, "models, out of a possible", len(cube_list))
    print("Passed Models", models_includes['pass'])
    print("Failed Models", models_includes['fail'], '\n------------')
    
    cube_mean = cube_mean/ float(length)
    #cube_mean.units = units_str
    #print ('make_mean_of_cube_list: cube_mean', cube_mean.data.mean())
    # assert 0
    return cube_mean


def standardize_depth_coord(cube):
    """
    Need to stanardize the depth coordinate to substract them.
    """
    cube.coord(axis='Z').long_name='Vertical T levels'
    cube.coord(axis='Z').attributes={'description':
        'generic ocean model vertical coordinate (nondimensional or'
        ' dimensional)',
        'positive': 'down'}
    cube.coord(axis='Z').standard_name='depth'
    try:
        model_name = cube.metadata[4]['source']
    except:
        model_name = ''
    print('standardize_depth_coord: ', model_name,  cube.coord(axis='Z'))
    return cube


def make_multimodelmean_transects(
        cfg,
        short_name = 'thetao',
        key = 'multimodel_mean'
):
    """
    Make a simple plot of the transect for an indivudual model.

    This tool loads the cube from the file, checks that the units are
    sensible BGC units, checks for layers, adjusts the titles accordingly,
    determines the ultimate file name and format, then saves the image.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.

    """
    metadatas = diagtools.get_input_files(cfg,)
    obs_key = 'observational_dataset'
    obs_filename = ''
    obs_metadata = {}
    obs_filename = diagtools.match_model_to_key(obs_key,
                                      cfg[obs_key],
                                      metadatas,)
    print('obs_filename', obs_filename)

    # Load cube and set up units
    cubes = {}
    for filename in sorted(metadatas):
        if filename == obs_filename:
            continue
        metadata = metadatas[filename]
        if metadata['short_name'] != short_name: continue
        # if metadata['dataset'] == 'MRI-ESM1': 
        #    continue
        cube = iris.load_cube(filename)
        cube = diagtools.bgc_units(cube, metadata['short_name'])
        cubes[metadata['dataset']] = cube
        print('loading:', metadata['dataset'], cube.data.mean(), cube.units )
    
    if len(cubes) == 0:
        return
    short_name = metadatas[filename]['short_name']
    # Take the multimodel mean.
#    if key == 'multimodel_mean':
#        cube = make_mean_of_cube_list(cubes, metadata['long_name'])
    if key == 'CMIP5': 
        assert 0
    elif key == 'CMIP6':
        assert 0
    elif key in cubes.keys():
        cubes = {key: cubes[key]}
    if len(cubes) == 1:
        print('length:', len(cubes), key, metadata['long_name'])

    cube = make_mean_of_cube_list(cubes, metadata['long_name'])

    print('model cube mean:' , cube.data.mean(), cube.units)

    cube = make_depth_safe(cube)
    #cubes = make_cube_region_dict(cube)

    obs_cube = iris.load_cube(obs_filename)
    obs_cube = diagtools.bgc_units(obs_cube, metadata['short_name'])
    obs_cube = standardize_depth_coord(obs_cube)
    print('pre: obs cube mean:' ,  obs_cube.data.mean(), obs_cube.units)

    if metadata['long_name'] == 'Sea Water Potential Temperature':
        contours = [0, 5, 10, 15, 20, 25, 30, 35]
        diff_contours = [-3., -2, -1, 1, 2, 3]
        fmt = '%1.0f'
        diff_fmt = '%1.0f'
        cube.units = 'celsius'
        obs_cube.units = 'celsius'

    if metadata['long_name'] == 'Sea Water Salinity':
        contours = [32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36 ]
        diff_contours = [ -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1 ]
        fmt = '%1.1f'
        diff_fmt = '%1.2f'
        cube.units = '1.'
        obs_cube.units = '1.'
        obs_cube.data = obs_cube.data * 1000.
    print('post: obs cube mean:' ,  obs_cube.data.mean(), obs_cube.units)


    # Add title to plot
    title = ' '.join(
        [metadata['long_name'], ])

    title = titlify(title)

    # Saving cubes:
    output_cube = diagtools.folder(cfg['work_dir']) + key + '_'+short_name+'.nc'
    logger.info('Saving cubes to %s', output_cube)
    iris.save(cube, output_cube)

    output_cube = diagtools.folder(cfg['work_dir']) + 'obs_'+short_name+'.nc'
    logger.info('Saving cubes to %s', output_cube)
    iris.save(obs_cube, output_cube)


    cube = cube - obs_cube
    cmap = 'seismic'
    colour_range = diagtools.get_cube_range_diff([cube,])

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    iris.plot.contourf(cube, 15, cmap=cmap, )#vmin=colour_range[0], vmax=colour_range[1] )
    plt.ylim(1000, 0)
    plt.clim(colour_range)
    plt.title(title)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    CS = iris.plot.contour(obs_cube, contours, colors='k' )
    ax1.clabel(CS, CS.levels, inline=True, fontsize=10, fmt = fmt)

    CS_diff = iris.plot.contour(cube, diff_contours, colors='white' )
    ax1.clabel(CS_diff, CS_diff.levels, inline=True, fontsize=10, fmt = diff_fmt)

    ax2 = fig.add_subplot(212)
    iris.plot.contourf(cube, 14, cmap=cmap, )
    plt.ylim(5000, 1000)
    plt.clim(colour_range)

    locs, labels = plt.yticks()
    ax2.set_yticks([2000, 3000, 4000, 5000,])
    plt.colorbar(orientation='horizontal')
    plt.clim(colour_range)

    CS = iris.plot.contour(obs_cube, contours, colors='black')
    ax2.clabel(CS, CS.levels, inline=True, fontsize=10, fmt = fmt)

    CS_diff = iris.plot.contour(cube, diff_contours, colors='white')
    ax2.clabel(CS_diff, CS_diff.levels, inline=True, fontsize=10, fmt = diff_fmt)

    fig.subplots_adjust(hspace=0.01)

    # Load image format extention
    image_extention = diagtools.get_image_format(cfg)

    # Determine image filename:
    path = diagtools.folder(cfg['plot_dir'])+'fig_basin_profile_bias_'+short_name +'_'+key+image_extention

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path)

    plt.close()













def make_transects_plots(
        cfg,
        metadata,
        filename,
        obs_filename,
):
    """
    Make a simple plot of the transect for an indivudual model.

    This tool loads the cube from the file, checks that the units are
    sensible BGC units, checks for layers, adjusts the titles accordingly,
    determines the ultimate file name and format, then saves the image.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.
    metadata: dict
        The metadata dictionairy for a specific model.
    filename: str
        The preprocessed model file.
    obs_filename: str
        Observartional filename

    """
    # Load cube and set up units
    cube = iris.load_cube(filename)
    cube = diagtools.bgc_units(cube, metadata['short_name'])

    cube = make_depth_safe(cube)
    cubes = make_cube_region_dict(cube)

    print(obs_filename)
    obs_cube = iris.load_cube(obs_filename)
    obs_cube = diagtools.bgc_units(obs_cube, metadata['short_name'])
    obs_cube = standardize_depth_coord(obs_cube)
 
    if metadata['long_name'] == 'Sea Water Potential Temperature':
        contours = [0,5,  10, 15, 20, 25,30,35]
        diff_contours = [-3., -2, -1, 1, 2, 3]
    if metadata['long_name'] == 'Sea Water Salinity':
        contours = [30, 31, 32, 33, 34, 35, 36 ]
        diff_contours = [ -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1 ]
        obs_cube.data = obs_cube.data * 1000.
    for region, cube in cubes.items():
        # Add title to plot
        title = ' '.join(
            [metadata['dataset'], metadata['long_name']])

        title = titlify(title)

        cube = cube - obs_cube

        cmap = 'seismic'
        colour_range = diagtools.get_cube_range_diff([cube,])

        #f, (ax1, ax2,) = plt.subplots(2, sharex=True, sharey=False)
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        iris.plot.contourf(cube, 15, cmap=cmap, )#vmin=colour_range[0], vmax=colour_range[1] )
        plt.ylim(1000, 0)
        plt.clim(colour_range)
        plt.title(title)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        CS = iris.plot.contour(obs_cube, contours, colors='k' )
        ax1.clabel(CS, CS.levels, inline=True, fontsize=10)

        CS_diff = iris.plot.contour(cube, diff_contours, colors='white' )
        ax1.clabel(CS_diff, CS_diff.levels, inline=True, fontsize=10)

        ax2 = fig.add_subplot(212)
        iris.plot.contourf(cube, 14, cmap=cmap, )#vmin=colour_range[0], vmax=colour_range[1] )
        plt.ylim(5000, 1000)
        plt.clim(colour_range)

        locs, labels = plt.yticks()
        ax2.set_yticks([2000, 3000, 4000, 5000,])
        plt.colorbar(orientation='horizontal')
        plt.clim(colour_range)

        CS = iris.plot.contour(obs_cube, contours, colors='k' )
        ax2.clabel(CS, CS.levels, inline=True, fontsize=10)

        CS_diff = iris.plot.contour(cube, diff_contours, colors='white' )
        ax2.clabel(CS_diff, CS_diff.levels, inline=True, fontsize=10)

        fig.subplots_adjust(hspace=0.01)

        #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

        # Load image format extention
        image_extention = diagtools.get_image_format(cfg)

        # Determine image filename:
        path = diagtools.get_image_path(
            cfg,
            metadata,
            suffix=region + 'transect' + image_extention,
        )

        # Saving files:
        if cfg['write_plots']:
            logger.info('Saving plots to %s', path)
            plt.savefig(path)

        plt.close()


def add_sea_floor(cube):
    """
    Add a simple sea floor line from the cube mask.

    Parameters
    ----------
    cube: iris.cube.Cube
        Input cube to use to produce the sea floor.

    """
    land_cube = cube.copy()
    land_cube.data = np.ma.array(land_cube.data)
    mask = 1. * land_cube.data.mask
    if mask.shape == ():
        mask = np.zeros_like(land_cube.data)
    land_cube.data = np.ma.masked_where(mask == 0, mask)
    land_cube.data.mask = mask
    qplt.contour(land_cube, 2, cmap='Greys_r', rasterized=True)

def main(cfg):
    """
    Load the config file and some metadata, then pass them the plot making
    tools.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.

    """
    #####
    metadatas = diagtools.get_input_files(cfg,)
    model_names = {}
    for filename in sorted(metadatas):
        metadata = metadatas[filename]
        model_names[metadata['dataset']] = True

    for short_name in ['thetao', 'so']:
       make_multimodelmean_transects(cfg, short_name = short_name)
       for model_name in model_names:
           make_multimodelmean_transects(cfg, short_name = short_name, key = model_name)


    # for index, metadata_filename in enumerate(cfg['input_files']):
    #     logger.info(
    #         'metadata filename:\t%s',
    #         metadata_filename,
    #     )
    #     metadatas = diagtools.get_input_files(cfg, index=index)
    #     obs_key = 'observational_dataset'
    #     obs_filename = ''
    #     obs_metadata = {}
    #     obs_filename = diagtools.match_model_to_key(obs_key,
    #                                       cfg[obs_key],
    #                                     metadatas)
    #
    #     for filename in sorted(metadatas):
    #         logger.info('-----------------')
    #         logger.info(
    #             'model filenames:\t%s',
    #             filename,
    #         )
    #
    #         ######
    #         # Time series of individual model
    #         make_transects_plots(cfg, metadatas[filename], filename, obs_filename)
    #
    #
    # logger.info('Success')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
