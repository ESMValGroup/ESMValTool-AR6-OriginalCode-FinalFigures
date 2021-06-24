"""
Transects diagnostics figure 3.25 in Chapter 3 of IPCC AR6 WGI
=================================

Diagnostic to produce images of a transect. These plots show either latitude or
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

Revised and corrected (15.01.2021): Elizaveta Malinina (CCCma)
                        elizaveta.malinina-rieger@canada.ca
"""
import logging
import os
import sys

import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np

from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import run_diagnostic
import esmvaltool.diag_scripts.shared.plot as eplot

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


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


def match_model_to_key(
        model_type,
        cfg_dict,
        input_files_dict,
        variable_group,
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
    matched_keys = {}
    print('variable_group', variable_group)
    for input_file, intput_dict in input_files_dict.items():
        print(input_file)
        intersect_keys = intput_dict.keys() & cfg_dict.keys()
        if input_file.find(variable_group) == -1:
            continue
        print('maybe:',input_file)
        match = True

        for key in intersect_keys:
            if intput_dict[key] == cfg_dict[key]:
                matched_keys[key] = [intput_dict[key], cfg_dict[key]]
                continue
            match = False
        if match:
            print("Found obs file:", input_file, intput_dict, '\n',matched_keys)
            return input_file
    logger.warning("Unable to match model: %s", model_type)
    assert 0
    return ''





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
    print('make_mean_of_cube_list: cube_mean', cube_mean.data.mean())
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
        cfg, fig, grid,
        short_name = 'thetao',
        key = 'all',
        variable_group='global_thetao',
        litera = ''
):
    """
    Make a simple plot of the transect for an indivudual model.

    This tool loads the cube from the file, checks that the units are
    sensible BGC units, checks for layers, adjusts the titles accordingly,
    determines the ultimate file name and format, then saves the image.

    key can be a model name, a project name, 'all' or 'all-ensembles'.
    when key is all, it first takes the model ensemble mean. ie each model gets one vote.
    when key  all-ensembles, each ensemble member gets one vote.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.
    fig: pyplot.figure
        the figure to plot on
    grid: matplotlib.gridspec instance
        a gridspec to plot on


    """
    metadatas = diagtools.get_input_files(cfg,)
    obs_key = 'observational_dataset'
    obs_dataset = ''
    print('=======\nStarting', short_name, variable_group, key)
    obs_filename = match_model_to_key(obs_key,
                                      cfg[obs_key],
                                      metadatas,
                                      variable_group,
                                      )

    print('obs_filename', short_name,variable_group, key, obs_filename)

    # Load cube and set up units
    cubes = {}
    long_name = ''
    for filename in sorted(metadatas):
        print('---------\n',filename)
        metadata = metadatas[filename]
        print(metadata['project'], metadata['short_name'], metadata['variable_group'])
        print('looking for', short_name, variable_group, key )
        if filename == obs_filename:
            obs_dataset = metadata['dataset']
            continue
        if metadata['project'] == 'OBS':
            continue
        if metadata['short_name'] != short_name:
            continue
        if metadata['variable_group'] != variable_group:
            continue

        project = metadata['project']
        dataset = metadata['dataset']
        ensemble = metadata['ensemble']
        experiment = metadata['exp']

        long_name = metadata['long_name']

        cube = iris.load_cube(filename)
        cube = diagtools.bgc_units(cube, short_name)

        index_tuple = (dataset, project, ensemble, experiment)
        print('loading', index_tuple)
        cubes[index_tuple] = cube.copy()
        if filename.find(dataset) == -1: assert 0

    print('all cubes found:', len(cubes), 'cubes')

    if obs_filename.find(variable_group) == -1:
        assert 0

    obs_cube = iris.load_cube(obs_filename)
    obs_cube = diagtools.bgc_units(obs_cube, short_name)
    obs_cube = standardize_depth_coord(obs_cube)
    print(key, cubes.keys())

    if key == obs_dataset:
        cubes = {key: obs_cube}
    elif key in 'all':
        datasets = {index[0]:True for index in cubes.keys()}
        new_cubes = {}
        for dataset in datasets.keys():
            new_list = {index:cube for index, cube in cubes.items() if dataset in index}
            new_cubes[dataset] = make_mean_of_cube_list(new_list, long_name)
        cubes = new_cubes.copy()

    elif key in ['', 'all-ensembles']:
        pass
    else:
        cubes = {index: cube for index, cube in cubes.items() if key in index}

    if len(cubes) == 0:
       return

    print('key:', key, 'found:', len(cubes), 'cubes')

    if key in cubes.keys() and len(cubes) != 1:
        print('WTF?', key, len(cubes), cubes)
        assert 0

    print_cubes = True
    if print_cubes:
        print('key:',key)
        print('cubes.keys():',cubes.keys())
        print('\n\n\nThe following cubes ', len(cubes),'should all be ',key, ':\n', cubes)
        for index, cube in cubes.items(): print (index,':\n', cube)

    cube = make_mean_of_cube_list(cubes, long_name)

    cube = make_depth_safe(cube)

    rgb_location = os.path.dirname(eplot.__file__)

    if long_name == 'Sea Water Potential Temperature':
        contours = [0, 5, 10, 15, 20, 25, 30, 35]
        diff_contours = [-3, -2, -1, 1, 2, 3]
        fmt = '%1.0f'
        diff_fmt = '%1.0f'
        diff_clip = [-3, 3]
        diff_step = 0.25
        cube.units = 'celsius'
        obs_cube.units = 'celsius'
        unit = r'$^o$C'
        title = 'Potential Temperature'
        rgb_file = np.loadtxt( os.path.join(rgb_location, 'rgb', 'ipcc-ar6_temperature_div.rgb'),
        skiprows=3)

    if long_name == 'Sea Water Salinity':
        contours = [32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36 ]
        diff_contours = [ -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1 ]
        fmt = '%1.1f'
        diff_fmt = '%1.2f'
        diff_step = 0.125
        #Liza's addition
        diff_clip = [-1, 1]
        cube.units = '1.'
        obs_cube.units = '1.'
        obs_cube.data = obs_cube.data * 1000.
        unit = r'PSS-78'
        title = 'Salinity'
        rgb_file = np.loadtxt(
            os.path.join(rgb_location, 'rgb', 'ipcc-ar6_misc_div.rgb'),
            skiprows=3)
    print('post: obs cube mean:' ,  obs_cube.data.mean(), obs_cube.units)

    # Add title to plot
    ocean = ' '.join([variable_group.replace('so_', '').replace('thetao_', '').title(), ])
    title = titlify(title)

    # Saving cubes:
    output_cube = diagtools.folder(cfg['work_dir']) + key + '_'+short_name+'.nc'
    logger.info('Saving cubes to %s', output_cube)
    iris.save(cube, output_cube)

    output_cube = diagtools.folder(cfg['work_dir']) + 'obs_'+short_name+'.nc'
    logger.info('Saving cubes to %s', output_cube)
    iris.save(obs_cube, output_cube)

    if cube == obs_cube:
        assert 0
    cube = cube - obs_cube.data
    cube.data = np.ma.clip(cube.data, diff_clip[0], diff_clip[1])

    cmap = mcolors.LinearSegmentedColormap.from_list('colormap', rgb_file/255)
    colour_range = diagtools.get_cube_range_diff([cube,])

    ax1 = fig.add_subplot(grid[0, :])

    iris.plot.contourf(cube, 15, cmap=cmap,  vmin=diff_clip[0],
                       vmax=diff_clip[1],
                       levels=np.arange(diff_clip[0],diff_clip[1]+diff_step, diff_step),
                       axes= ax1, extend='both')
    ax1.set_ylim(1000, 0)
    ax1.set_yticks([1000, 800, 600, 400, 200, 0])
    ax1.text(-87,100,litera)
    plt.clim(colour_range)
    ax1.set_title(title)

    CS = iris.plot.contour(obs_cube, contours, linewidths=0.7,  colors='k', axes= ax1)
    ax1.clabel(CS, CS.levels, inline=True, fontsize=10, fmt = fmt)

    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis='x', which='both', bottom=False)

    ax2 = fig.add_subplot(grid[1,:])

    iris.plot.contourf(cube, 14, cmap=cmap,  vmin=diff_clip[0], vmax=diff_clip[1], axes= ax2, extend='both' )
    ax2.set_ylim(5000, 1000)
    ax2.set_xticks(np.arange(-90,91,30))
    ax2.set_xlabel(r'Latitude ($^o$N)')
    plt.clim(colour_range)

    ax2.set_yticks([2000, 3000, 4000, 5000,])
    plt.clim(colour_range)

    CS = iris.plot.contour(obs_cube, contours, linewidths=0.7,  colors='black', axes= ax2)
    ax2.clabel(CS, CS.levels, inline=True, fontsize=10, fmt = fmt)

    if litera == '(a)':
        ax2.arrow(-28, 1700, -6, -650, length_includes_head=True,
                  head_width=2, head_length=50)
        ax2.text(-25, 2300, 'WOA18\nclimatology', fontsize=10)
    elif litera == '(b)':
        ax2.arrow(-20, 1800, -4, -600, length_includes_head=True,
                  head_width=2, head_length=50)
        ax2.text(-18, 2300, 'WOA18\nclimatology', fontsize=10)

    plt.text(0.9, 0.1, ocean,fontsize=11, horizontalalignment='center',
        verticalalignment='center', transform = ax2.transAxes)

    return cmap, title, diff_clip, diff_step, unit

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
    model_names = {'CMIP6':True}
    short_names = {}
    for filename in sorted(metadatas):
        metadata = metadatas[filename]
        short_names[metadata['short_name']] = True

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    variable_group_suffixes = ['_global', '_atlantic', '_pacific', '_indian']

    n_rows = len(variable_group_suffixes)
    n_cols = len(short_names)

    for model_name in model_names:

        fig = plt.figure()
        fig.set_size_inches(9., 12.)
        outer = gridspec.GridSpec(n_rows, n_cols, figure=fig, left=0.1,
                                  right=0.98, top=0.94, bottom=0.11,
                                  wspace=0.1, hspace=0.1)

        lits = np.asarray([['(a)','(c)', '(e)','(g)'], ['(b)','(d)','(f)','(h)']])

        for nrow, var_group in enumerate(variable_group_suffixes):
            for ncol, short_name in enumerate(list(short_names.keys())[::-1]):
                variable_group = short_name+var_group
                inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                         subplot_spec=outer[
                                                             nrow, ncol],
                                                         wspace=0.0,
                                                         hspace=0.04)
                cmap, l_name, lims, step, unit = make_multimodelmean_transects(
                                                 cfg, fig, inner,
                                                 short_name=short_name,
                                                 key=model_name,
                                                 variable_group=variable_group,
                                                 litera=lits[ncol, nrow])
                if nrow == n_rows-1:
                    cax = fig.add_axes([0.11 + ncol*0.49, 0.05, 0.39, 0.01])
                    norm = mcolors.BoundaryNorm(np.arange(lims[0], lims[1]+step, step),
                                                cmap.N, extend='both')
                    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                                      cax=cax, orientation = 'horizontal',
                                      label=l_name+' difference ('+ unit +
                                            ')\n'+model_name+ ' - WOA18',
                                      extend='both')
                    cb.set_ticklabels(cb.get_ticks())
                bnds = outer[nrow,0].get_position(fig)
                crds = bnds.bounds
                x = crds[0] / 6
                y = crds[1] + crds[3] / 3
                fig.text(x, y, 'depth (m)', rotation=90, fontsize='small')

        fig.suptitle('Potential temperature and salinity bias for ocean basins (1981-2010)')

        axs = fig.get_axes()
        axs.pop(-1)
        axs.pop(-3)
        [a.set_title('') for a in axs[4:]]
        [a.set_xticklabels('') for a in axs[:-4]]
        [a.set_xlabel('') for a in axs[:-4]]
        [a.set_yticklabels('') for a in axs[2:len(axs):4]]
        [a.set_yticklabels('') for a in axs[3:len(axs):4]]

        # Load image format extention
        image_extention = diagtools.get_image_format(cfg)

        # Determine image filename:
        path = diagtools.folder(cfg['plot_dir']) + 'fig_basin_profile_bias_'\
               + model_name + image_extention

        path_png = diagtools.folder(cfg['plot_dir']) + 'fig_basin_profile_bias_'\
               +model_name + '.png'

        # Saving files:
        if cfg['write_plots']:
            logger.info('Saving plots to %s', path)
            plt.savefig(path)
            plt.savefig(path_png, dpi=250)

        plt.close()

    logger.info('Success')

if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
