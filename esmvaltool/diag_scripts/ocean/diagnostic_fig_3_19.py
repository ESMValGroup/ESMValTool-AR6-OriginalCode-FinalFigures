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
            suffix= key_word.replace(' ','') + image_extention,
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
        variable_group
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
        if intput_dict['variable_group'] != variable_group: continue
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


def make_mean_cubes(args):
    """
    Takes the mean of several cubes.
    """

    return sum(args) / float(len(args))
    #n = float(len(cubes))
    #return reduce(iris.analysis.maths.add, cubes) / n


def make_multimodle_zonal_mean_plots(
        cfg,
        pane = 'a',
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
    metadatas = diagtools.get_input_files(cfg)
    plot_details = {}
    cmap = plt.cm.get_cmap('jet')
    if pane in ['a', 'c']:
        group = 'thetao_zonal'
    if pane in ['b', 'd']:
        group = 'thetao_equator'

    #####
    # Load obs data and details
    obs_key = 'observational_dataset'
    obs_filename = ''
    obs_metadata = {}
    if obs_key in cfg:
        obs_filename = match_model_to_key(obs_key, #using local copy
                                                    cfg[obs_key],
                                                    metadatas,
                                                    group)
        obs_metadata = metadatas[obs_filename]
        obs_cube = iris.load_cube(obs_filename)
        obs_cube = diagtools.bgc_units(obs_cube, obs_metadata['short_name'])
        obs_key = obs_metadata['dataset']

    plot_details = {}
    cmap = plt.cm.get_cmap('jet')

    #####
    # calculate the number of models
    number_models = {}
    projects = {}
    for i, filename in enumerate(sorted(metadatas)):
        metadata = metadatas[filename]
        if filename == obs_filename: continue
        if group != metadata['variable_group']: continue
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

        if group != metadata['variable_group']:
            continue

        cube = iris.load_cube(filename)
        cube = diagtools.bgc_units(cube, short_name)

        if number_models == 1:
            color = 'black'
        else:
            value = float(model_numbers[dataset] ) / (number_models - 1.)
            color = cmap(value)
            print('colors:', i, model_numbers[dataset], number_models, '\tvalue:', value )

        print('dataset:', dataset, '\tcolor:', color)

        plot_details[dataset] = {'c': color, 'ls': '-', 'lw': 1,
                                 'label': dataset}

        # Is this data is a multi-model dataset?
        multi_model = dataset.find('MultiModel') > -1

        if multi_model:
            continue
            #Doing this by hand!
            plot_details[dataset] = {'c': 'red', 'ls': '-', 'lw': 2,
                                                 'label': dataset}

        new_cube = cube - obs_cube
        if pane in ['a', 'b']:
            key_word, xlabel = plot_zonal_cube(new_cube, plot_details[dataset])

        ####
        # Calculate the project lines
        project_cubes[project][dataset] = cube

    # Plot the project means.
    for project in projects:
        if project in ['OBS', 'obs4mip']: continue

        ####
        # Calculate error
        errorcubeslist = [cube - obs_cube for cube in project_cubes[project].values()]
        project_mean_error = errorcubeslist[0]
        for cube in errorcubeslist[1:]: project_mean_error+=cube
        project_mean_error = project_mean_error/ float(len(errorcubeslist))


        #project_cubes[project] / float(project_counts[project])
        if project == 'CMIP5':
                mip_color  = 'red'
        if project == 'CMIP3':
                mip_color  = 'dodgerblue'
        if project == 'CMIP6':
                mip_color  = 'green'
        label = project + ' mean'
        plot_details[dataset] = {'c': mip_color, 'ls': '-', 'lw': 2,
                                             'label': label}
        if pane in 'abc':
                key_word, xlabel = plot_zonal_cube(project_mean_error, plot_details[dataset])
        if pane in 'd':
                cubeslist = [cube  for cube in project_cubes[project].values()]
                project_mean = cubeslist[0]
                for cube in cubeslist[1:]:
                    project_mean+=cube
                project_mean = project_mean/ float(len(cubeslist))
                key_word, xlabel = plot_zonal_cube(project_mean, plot_details[dataset])

    #####
    # title and axis lables
    if pane in 'abc':
        plt.axhline(0., linestyle=':', linewidth=0.2, color='k')
    plt.xlabel(xlabel)
    plt.ylabel('SST error ('+r'$^\circ$'+'C)')
    title = ' '.join(['(', pane, ')', key_word, metadata['dataset'], ])
    plt.title(title)

   # Add Legend outside right.
    diagtools.add_legend_outside_right(plot_details, plt.gca())

    # Load image format extention
    image_extention = diagtools.get_image_format(cfg)

    # Determine image filename:

    path = diagtools.get_image_path(
        cfg,
        metadata,
        suffix= key_word.replace(' ','') + pane + image_extention,
        )

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path)

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
    for pane in ['a', 'b', 'c', 'd']:
            make_multimodle_zonal_mean_plots(cfg, pane=pane)
    #####
    for index, metadata_filename in enumerate(cfg['input_files']):
        continue
        metadatas = diagtools.get_input_files(cfg, index=index)

        logger.info(
            'metadata filename:\t%s',
            metadata_filename,
        )
        obs_key = 'observational_dataset'
        obs_filename = ''
        obs_metadata = {}
        if obs_key in cfg:
            obs_filename = diagtools.match_model_to_key(obs_key,
                                                        cfg[obs_key],
                                                        metadatas)
            obs_metadata = metadatas[obs_filename]

        for filename in sorted(metadatas):

            logger.info('-----------------')
            logger.info(
                'model filenames:\t%s',
                filename,
            )
            ######
            # Transects of individual model
            # make_transects_plots(cfg, metadatas[filename], filename)

            ######
            # fig 3.19 for of individual model
            if obs_filename and filename != obs_filename:
                make_single_zonal_mean_plots(cfg,
                                  metadatas[filename],
                                  filename,
                                  obs_metadata=obs_metadata,
                                  obs_filename=obs_filename)




    logger.info('Success')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
