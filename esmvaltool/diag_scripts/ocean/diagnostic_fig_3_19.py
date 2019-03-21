"""
Transects diagnostics
=====================

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

dpi = 100

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
    plt.title(title)



def make_single_zonal_mean_plots(
        cfg,
        metadata,
        filename,
        obs_metadata={},
        obs_filename='',
):
    """
    Make a zonal mean error plot for an individual model.

    The optional observational dataset must be added.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.
    metadata: dict
        The metadata dictionairy for a specific model.
    filename: str
        The preprocessed model file.
    obs_metadata: dict
        The metadata dictionairy for the observational dataset.
    obs_filename: str
        The preprocessed observational dataset file.

    """
  # Load cube and set up units
    cube = iris.load_cube(filename)
    cube = diagtools.bgc_units(cube, metadata['short_name'])
    short_name = metadata['short_name']

    # Is this data is a multi-model dataset?
    multi_model = metadata['dataset'].find('MultiModel') > -1

   # Add observational data.
    if obs_filename:
        obs_cube = iris.load_cube(obs_filename)
        obs_cube = diagtools.bgc_units(obs_cube, metadata['short_name'])

        obs_key = obs_metadata['dataset']

    new_cube = cube - obs_cube

    # Zonal_mean_error
    if new_cube.data.shape == new_cube.coord('latitude').points.shape:
            plt.plot(new_cube.coord('latitude').points, new_cube.data )
            plt.ylabel('Latitude ('+r'$^\circ$'+'N)')
            key_word = 'Zonal mean SST error'

    # Equatorial_mean_error
    if new_cube.data.shape == new_cube.coord('longitude').points.shape:
            plt.plot(new_cube.coord('longitude').points, new_cube.data )
            plt.ylabel('Longitude ('+r'$^\circ$'+'E)')
            key_word = 'Equatorial SST error'

    plt.axhline(0., linestyle=':', linewidth=0.2, color='k')
    plt.ylabel('SST error ('+r'$^\circ$'+'C)')

    # Add title to plot
    if multi_model:
        title = ' '.join([key_word, 'multimodel mean', ])
    else:
        title = ' '.join([key_word, metadata['dataset'], ])
    plt.title(title)

   # Add Legend outside right.
    # diagtools.add_legend_outside_right(plot_details, plt.gca())

    # Load image format extention
    image_extention = diagtools.get_image_format(cfg)

    # Determine image filename:
    if multi_model:
        path = diagtools.folder(
            cfg['plot_dir']) + os.path.basename(filename).replace(
                '.nc', key_word.replace(' ','') + image_extention)
    else:
        path = diagtools.get_image_path(
            cfg,
            metadata,
            suffix= key_word.replace(' ','')+short_name + image_extention,
        )

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path)

    plt.close()


def match_model_to_key(
        model_type,
        cfg_dict,
        input_files_dict,
        variable_groups
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
        if intput_dict['variable_group'] not in variable_groups: continue
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


def plot_zonal_cube(cube, plot_details):
    # Zonal_mean_error
    if cube.data.shape == cube.coord('latitude').points.shape:
        plt.plot(cube.coord('latitude').points, cube.data,
             c = plot_details['c'],
             lw = plot_details['lw'],
             ls = plot_details['ls'],
             )
        xlabel = 'Latitude ('+r'$^\circ$'+'N)'
        key_word = 'Zonal mean SST error'

    # Equatorial_mean_error
    if cube.data.shape == cube.coord('longitude').points.shape:
        plt.plot(cube.coord('longitude').points, cube.data,
             c = plot_details['c'],
             lw = plot_details['lw'],
             ls = plot_details['ls'],
             )
        xlabel = 'Longitude ('+r'$^\circ$'+'E)'
        key_word = 'Equatorial SST error'

    return key_word, xlabel


def fill_between_two_cubes(cube1, cube2, color):
    # Zonal_mean_error
    if cube1.data.shape == cube1.coord('latitude').points.shape:
        plt.fill_between(cube1.coord('latitude').points,
                        cube1.data,
                        cube2.data,
                        color=color,
                        alpha = 0.4
             )

    if cube1.data.shape == cube1.coord('longitude').points.shape:
        plt.fill_between(cube1.coord('longitude').points,
                        cube1.data,
                        cube2.data,
                        color=color,
                        alpha = 0.4
             )



def make_mean_of_cube_list(cube_list):
    """
    Takes the mean of a list of cubes (not an iris.cube.CubeList).

    Assumes all the cubes are the same shape.
    """
    cube_mean = cube_list[0]
    for cube in cube_list[1:]:
        cube_mean+=cube
    cube_mean = cube_mean/ float(len(cube_list))
    cube_mean.units = 'celsius'
    return cube_mean


def make_std_of_cube_list(cube_list):
    """
    Makes the standard deviation of a list of cubes (not an iris.cube.CubeList).

    assumes all the cubes are the same shape and 1D
    """
    cube_std = cube_list[0].copy()
    out_data = np.zeros_like(cube_std.data)
    out_mask = np.zeros_like(cube_std.data)
    out_dict = {}

    for cube in cube_list:
        for (d,),  dat in np.ndenumerate(cube.data):
            if dat == 1e+20:
                out_mask[d] += 1
            try:        out_dict[d].append(dat)
            except:     out_dict[d] = [dat, ]

    for d, dat in out_dict.items():
        out_data[d] = np.std(dat)

    #out_data = np.ma.masked_where(out_mask, out_data)
    out_data = np.ma.masked_where(cube_std.data.mask, out_data)
    cube_std.data = out_data
    return cube_std


def load_obs(cfg, groups):
    """
    Load the observations.
    """
    obs_key = 'observational_dataset'
    obs_filename = ''
    obs_metadata = {}
    metadatas = diagtools.get_input_files(cfg)
    if obs_key in cfg:
        obs_filename = match_model_to_key(obs_key, #using local copy
                                                    cfg[obs_key],
                                                    metadatas,
                                                    groups)
        obs_metadata = metadatas[obs_filename]
        obs_cube = iris.load_cube(obs_filename)
        obs_cube = diagtools.bgc_units(obs_cube, obs_metadata['short_name'])
        obs_key = obs_metadata['dataset']
    return obs_cube, obs_key, obs_filename



def make_multimodle_zonal_mean_plots(
        cfg,
        pane = 'a',
        save = True,
        shortname = 'thetao',
):
    """
    Make a zonal mean error plot for an individual model.

    The optional observational dataset must be added.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.

    """
    metadatas = diagtools.get_input_files(cfg)
    plot_details = {}
    cmap = plt.cm.get_cmap('jet')
    if pane in ['a', 'c']:
        groups = ['thetao_zonal', 'tos_zonal', ]
    if pane in ['b', 'd']:
        groups = ['thetao_equator', 'tos_equator', ]

    #####
    # Load obs data and details
    obs_cube, obs_key, obs_filename = load_obs(cfg, groups)

    plot_details = {}
    cmap = plt.cm.get_cmap('jet')

    #####
    # calculate the number of models
    number_models = {}
    projects = {}
    for i, filename in enumerate(sorted(metadatas)):
        metadata = metadatas[filename]
        if filename == obs_filename: continue
        if metadata['variable_group'] not in groups: continue
        number_models[metadata['dataset']] = True
        projects[metadata['project']] = True
    model_numbers = {model:i for i, model in enumerate(sorted(number_models))}
    print (number_models, model_numbers)
    number_models = len(number_models)


    #####
    # List of cubes to make means/stds.
    project_cubes = {project:{} for project in projects}

    for i, filename in enumerate(sorted(metadatas)):
        if filename == obs_filename: continue

        metadata = metadatas[filename]
        short_name = metadata['short_name']
        dataset = metadata['dataset']
        project = metadata['project']

        if metadata['variable_group'] not in groups:
            continue

        cube = iris.load_cube(filename)
        cube = diagtools.bgc_units(cube, short_name)

        if number_models == 1:
            color = 'black'
        else:
            value = float(model_numbers[dataset] ) / (number_models - 1.)
            color = cmap(value)

        # Is this data is a multi-model dataset?
        if dataset.find('MultiModel') > -1: continue

        print('dataset:', dataset, cube.shape)
        new_cube = cube - obs_cube
        if new_cube.data.mean() < -200. :
                print ("Warning: this model is borken:", dataset, 'mean:', new_cube.data.mean())
                continue
        if pane in ['a', 'b']:
            plot_details[dataset] = {'c': color, 'ls': '-', 'lw': 1,
                                     'label': dataset}
            key_word, xlabel = plot_zonal_cube(new_cube, plot_details[dataset])

        ####
        # Calculate the project lines
        project_cubes[project][dataset] = cube
    # Plot the project means.
    for project in projects:
        #if project in ['OBS', 'obs4mip']: continue
        for ds, cube in project_cubes[project].items():
                print(ds, '\t', cube.data.mean() )
    #assert 0
    # Plot the project means.
    for project in projects:
        if project in ['OBS', 'obs4mip']: continue

        ####
        # Calculate error
        errorcubeslist = [cube - obs_cube for cube in project_cubes[project].values()]
        project_mean_error = make_mean_of_cube_list(errorcubeslist)

        if project == 'CMIP5':
                mip_color  = 'red'
        elif project == 'CMIP3':
                mip_color  = 'dodgerblue'
        elif project == 'CMIP6':
                mip_color  = 'green'
        else:  assert 0
        plot_details[project] = {'c': mip_color, 'ls': '-', 'lw': 2,
                                             'label': project}
        if pane in 'abc':
                key_word, xlabel = plot_zonal_cube(project_mean_error, plot_details[project])
                ylabel = 'SST error ('+r'$^\circ$'+'C)'
        if pane in 'c':
                cube_std = make_std_of_cube_list(errorcubeslist)
                fill_between_two_cubes(project_mean_error - cube_std, project_mean_error + cube_std, mip_color)

        if pane in 'd':
                cubeslist = [cube  for cube in project_cubes[project].values()]
                project_mean = make_mean_of_cube_list(cubeslist)
                key_word, xlabel = plot_zonal_cube(project_mean, plot_details[project])

                cube_std = make_std_of_cube_list(cubeslist)
                fill_between_two_cubes(project_mean - cube_std, project_mean + cube_std, mip_color)

                plot_details[obs_key] = {'c': 'black', 'ls': '-', 'lw': 2,
                                         'label': obs_key}
                key_word, xlabel = plot_zonal_cube(obs_cube, plot_details[obs_key])
                ylabel = 'SST ('+r'$^\circ$'+'C)'


    #####
    # title and axis lables
    if pane in 'abc':
        plt.axhline(0., linestyle=':', linewidth=0.2, color='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title = ' '.join(['(', pane, ')', key_word ])
    plt.title(title)

    if not save:
        return plt.gca(), plot_details

    # Add Legend outside right.
    diagtools.add_legend_outside_right(plot_details, plt.gca())

    # Load image format extention
    image_extention = diagtools.get_image_format(cfg)

    # Determine image filename:
    path = diagtools.get_image_path(
        cfg,
        metadata,
        suffix= key_word.replace(' ','') + pane + short_name+ image_extention,
        )

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path, dpi=dpi)

    plt.close()




def main(cfg):
    """
    Load the config file and some metadata, then pass them the plot making
    tools.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.

    """
    dpi = 100

    #####
    # individual panes
    for pane in [ 'c', 'd', 'a', 'b',]:
        make_multimodle_zonal_mean_plots(cfg, pane=pane, save = True)

    #####
    # Altogether
    fig = plt.figure()
    fig.set_size_inches(10., 9.)
    plot_details = {}
    axes = []
    for pane,sbpt in zip([ 'a', 'b', 'c', 'd', ], [221,222,223,224]):
        axes.append(plt.subplot(sbpt))
        ax, pt_dets = make_multimodle_zonal_mean_plots(cfg, pane=pane, save = False)
        plot_details.update(pt_dets)

    plt.subplots_adjust(right=0.80, hspace=0.3, wspace=0.3)
    cax = plt.axes([0.8, 0.1, 0.075, 0.8])
    plt.axis('off')

    # ####
    # Make legend order
    legend_order = []
    obs = ['HadISST', 'WOA', 'CORA']
    projects = ['CMIP6', 'CMIP5', 'CMIP3',] #'WOA', 'OBS']
    for ob in obs:
        if ob in plot_details.keys(): legend_order.append(ob)
    for proj in projects:
        if proj in plot_details.keys(): legend_order.append(proj)
    for linename in sorted(plot_details.keys()):
        if linename in legend_order: continue
        legend_order.append(linename)

    # ####
    # Make dummy axes
    for index in legend_order:
        colour = plot_details[index]['c']
        linewidth = plot_details[index].get('lw', 1)
        linestyle = plot_details[index].get('ls', '-')
        label = plot_details[index].get('label', str(index))
        plt.plot([], [], c=colour, lw=linewidth, ls=linestyle, label=label)



    # handles, labels = plt.gca().get_legend_handles_labels()
    # print (handles, labels, legend_order)
    # assert 0
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    # Make legend
    legd = cax.legend(
        # handles,
        # labels,
        loc='center left',
        ncol=1,
        prop={'size': 10},)
        #bbox_to_anchor=(1., 0.5))
    #legd.draw_frame(False)
    #legd.get_frame().set_alpha(0.)

    # Load image format extention and path
    image_extention = diagtools.get_image_format(cfg)
    path = cfg['plot_dir'] + 'fig_3.19'+image_extention

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path)

    plt.close()

    logger.info('Success')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
