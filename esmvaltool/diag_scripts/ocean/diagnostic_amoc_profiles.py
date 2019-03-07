"""
AMOC Profile diagnostics.
=========================

Diagnostic to produce figure of the profile over time from a cube.
These plost show cube value (ie temperature) on the x-axis, and depth/height
on the y axis. The colour scale is the time series.

Note that this diagnostic assumes that the preprocessors do the bulk of the
hard work, and that the cube received by this diagnostic (via the settings.yml
and metadata.yml files) has a time component, and depth component, but no
latitude or longitude coordinates.

An approproate preprocessor for a 3D+time field would be::

  preprocessors:
    prep_profile:
      extract_volume:
        long1: 0.
        long2:  20.
        lat1:  -30.
        lat2:  30.
        z_min: 0.
        z_max: 3000.
      average_region:
        coord1: longitude
        coord2: latitude

In order to add an observational dataset to the profile plot, the following
arguments are needed in the diagnostic script::

  diagnostics:
    diagnostic_name:
      variables:
        ...
      additional_datasets:
      - {observational dataset description}
      scripts:
        script_name:
          script: ocean/diagnostic_profiles.py
          observational_dataset: {observational dataset description}

This tool is part of the ocean diagnostic tools package in the ESMValTool.

Author: Lee de Mora (PML)
        ledm@pml.ac.uk
"""
import logging
import os
import sys

import numpy as np
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats import linregress


from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import run_diagnostic
from esmvaltool.preprocessor import time_average
# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def calculate_trend(cube, window = '8 years'):
    """
    Calculate a moving average.

    The window is a string which is a number and a measuremet of time.
    For instance, the following are acceptable window strings:

    * ``5 days``
    * ``12 years``
    * ``1 month``
    * ``5 yr``

    Also note the the value used is the total width of the window.
    For instance, if the window provided was '10 years', the the moving
    average returned would be the average of all values within 5 years
    of the central value.

    In the case of edge conditions, at the start an end of the data, they
    only include the average of the data available. Ie the first value
    in the moving average of a ``10 year`` window will only include the average
    of the five subsequent years.

    Parameters
    ----------
    cube: iris.cube.Cube
        Input cube
    window: str
        A description of the window to use for the

    Returns
    ----------
    iris.cube.Cube:
        A cube with the movinage average set as the data points.

    """
    window = window.split()
    window_len = int(window[0]) / 2.
    win_units = str(window[1])

    if win_units not in [
            'days', 'day', 'dy', 'months', 'month', 'mn', 'years', 'yrs',
            'year', 'yr'
    ]:
        raise ValueError("Moving average window units not recognised: " +
                         "{}".format(win_units))

    times = cube.coord('time').units.num2date(cube.coord('time').points)
    float_times = diagtools.cube_time_to_float(cube)

    datetime = diagtools.guess_calendar_datetime(cube)

    output = []

    times = np.array([
        datetime(time_itr.year, time_itr.month, time_itr.day, time_itr.hour,
                 time_itr.minute) for time_itr in times
    ])

    for time_itr in times:
        if win_units in ['years', 'yrs', 'year', 'yr']:
            tmin = datetime(time_itr.year - window_len, time_itr.month,
                            time_itr.day, time_itr.hour, time_itr.minute)
            tmax = datetime(time_itr.year + window_len, time_itr.month,
                            time_itr.day, time_itr.hour, time_itr.minute)

        if win_units in ['months', 'month', 'mn']:
            tmin = datetime(time_itr.year, time_itr.month - window_len,
                            time_itr.day, time_itr.hour, time_itr.minute)
            tmax = datetime(time_itr.year, time_itr.month + window_len,
                            time_itr.day, time_itr.hour, time_itr.minute)

        if win_units in ['days', 'day', 'dy']:
            tmin = datetime(time_itr.year, time_itr.month,
                            time_itr.day - window_len, time_itr.hour,
                            time_itr.minute)
            tmax = datetime(time_itr.year, time_itr.month,
                            time_itr.day + window_len, time_itr.hour,
                            time_itr.minute)

        arr = np.ma.masked_where((times < tmin) + (times > tmax), cube.data)
        time_arr = np.ma.masked_where(arr.mask, float_times)

        # print(time_itr, linregress(time_arr.compressed(), arr.compressed()))
        output.append(linregress(time_arr, arr)[0])
    return np.array(output)


def calculate_interannual(cube):
    """
    Calculate the interannnual variability.
    """
    cube.aggregated_by('year', iris.analysis.MEAN)
    data = cube.data
    arr = []
    for d,dat  in enumerate(data[:-1]):
        arr.append(data[d+1] - dat)
    return np.array(arr)


def get_26North(cube):
    """
    Extract 26.5 North.
    """
    latitude = cube.coord('latitude').points
    closest_lat = np.min(np.abs(latitude - 26.5))
    cube = cube.extract(iris.Constraint(latitude=closest_lat))
    print('get_26North:',cube.data.shape)

    return cube

def get_max_amoc(cube):
    """
    Extract maximum.
    """
    cube = cube.collapsed('depth', iris.analysis.MAX)
    print('get_max_amoc:',cube.data.shape)

    return cube


def load_cube(filename, metadata):
    """Load cube and set up units"""
    cube = iris.load_cube(filename)
    print('load_cube',cube.data.shape, cube.coords)
    cube = diagtools.bgc_units(cube, metadata['short_name'])
    print('load_cube', cube.data.shape)
    cube = get_26North(cube)
    return cube

def count_models(metadatas, obs_filename):
    """calculate the number of models."""
    number_models = {}
    projects = {}
    for i, filename in enumerate(sorted(metadatas)):
        metadata = metadatas[filename]
        if filename == obs_filename: continue
        number_models[metadata['dataset']] = True
        projects[metadata['project']] = True
    model_numbers = {model:i for i, model in enumerate(sorted(number_models))}
    print (number_models, model_numbers)
    number_models = len(number_models)
    return model_numbers, number_models, projects


def make_pane_a(
        cfg,
        fig=None,
        ax=None
):
    """
    Make a profile plot for an individual model.

    The optional observational dataset can also be added.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.
    pane: string
        Which pane to produce. Either b or c.
    fig: Pyplot.figure()
        The pyplot figure
    ax: pyplot.axes
        The pyplot axes.

    Returns
    ----------
    fig: Pyplot.figure() - optional
        The pyplot figure (only returned if also provided)
    ax: pyplot.axes - optional
        The pyplot axes.  (only returned if also provided)
    """
    savefig = False
    if fig in [None,] and ax in [None,]:
        fig = plt.figure()
        fig.set_size_inches(10., 9.)
        ax = plt.subplot(111)
        savefig = True

    metadatas = diagtools.get_input_files(cfg)

    obs_key = 'observational_dataset'
    obs_filename = ''
    obs_metadata = {}
    if obs_key in cfg:
        obs_filename = diagtools.match_model_to_key(obs_key,
                                                    cfg[obs_key],
                                                    metadatas)
        obs_metadata = metadatas[obs_filename]

    cubes = {}
    for filename in sorted(metadatas.keys()):
        dataset = metadatas[filename]['dataset']
        cube = load_cube(filename, metadatas[filename])
        cubes[dataset] = time_average(cube)
    cmap = plt.cm.get_cmap('jet')

    #####
    # calculate the number of models
    model_numbers, number_models, projects= count_models(metadatas, obs_filename)

    plot_details = {}
    for filename in sorted(metadatas.keys()):
        dataset =  metadatas[filename]['dataset']
        value = float(model_numbers[dataset] ) / (number_models - 1.)

        max_index = np.argmax(cubes[dataset].data)

        label = ' '.join([metadatas[filename]['dataset'],
                          ':',
                          '('+str(round(cubes[dataset].data[max_index] , 1)),
                          str(cubes[dataset].units)+',',
                          str(int(cubes[dataset].coord('depth').points[max_index])),
                          str(cubes[dataset].coord('depth').units)+')'
                          ])
        if filename == obs_filename:
            plot_details[obs_key] = {'c': 'black', 'ls': '-', 'lw': 2,
                                     'label': label}
        else:
            plot_details[dataset] = {'c': cmap(value),
                                     'ls': '-',
                                     'lw': 1,
                                     'label': label}
        qplt.plot(cubes[dataset], cubes[dataset].coord('depth'),
             color = plot_details[dataset]['c'],
             linewidth = plot_details[dataset]['lw'],
             linestyle = plot_details[dataset]['ls'],
             label = label
             )
        # Add a marker at the maximum
        plt.plot(cubes[dataset].data[max_index],
                 cubes[dataset].coord('depth').points[max_index],
                 c =  plot_details[dataset]['c'],
                 marker = 'd',
                 markersize = '10',
                 )

    # Add observational data.
    # if obs_filename:
    #     obs_cube = iris.load_cube(obs_filename)
    #     obs_cube = diagtools.bgc_units(obs_cube, metadata['short_name'])
    #     # obs_cube = obs_cube.collapsed('time', iris.analysis.MEAN)
    #
    #     obs_key = obs_metadata['dataset']
    #     qplt.plot(obs_cube, obs_cube.coord('depth'), c='black')
    #
    #     plot_details[obs_key] = {'c': 'black', 'ls': '-', 'lw': 1,
    #                              'label': obs_key}

    # Add title to plot
    # title = ' '.join([
    #     metadata['dataset'],
    #     metadata['long_name'],
    # ])
    # plt.title(title)
    plt.title('(a) AMOC streamfunction profiles at 26.5N')

    # Add Legend outside right.
    # diagtools.add_legend_outside_right(plot_details, plt.gca())
    leg = plt.legend(loc='lower right', prop={'size':6})
    leg.draw_frame(False)
    leg.get_frame().set_alpha(0.)

    if not savefig:
        return fig, ax
    # Load image format extention and path
    image_extention = diagtools.get_image_format(cfg)
    path = cfg['plot_dir'] + 'fig_3.24a'+image_extention

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path)

    plt.close()


def make_pane_bc(
        cfg,
        pane = 'b',
        fig=None,
        ax=None
):
    """
    Make a box and whiskers plot for panes b and c.

    If a figure and axes are not provided, if will save the pane as it's own
    image, otherwise it returns the fig and ax.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.
    pane: string
        Which pane to produce. Either b or c.
    fig: Pyplot.figure()
        The pyplot figure
    ax: pyplot.axes
        The pyplot axes.

    Returns
    ----------
    fig: Pyplot.figure() - optional
        The pyplot figure (only returned if also provided)
    ax: pyplot.axes - optional
        The pyplot axes.  (only returned if also provided)
    """
    savefig = False
    if fig in [None,] and ax in [None,]:
        fig = plt.figure()
        fig.set_size_inches(10., 9.)
        ax = plt.subplot(1, 1, 1)
        savefig = True

    metadatas = diagtools.get_input_files(cfg)

    obs_key = 'observational_dataset'
    obs_filename = ''
    obs_metadata = {}
    if obs_key in cfg:
        obs_filename = diagtools.match_model_to_key(obs_key,
                                                    cfg[obs_key],
                                                    metadatas)
        obs_metadata = metadatas[obs_filename]

    trends = {}
    for filename in sorted(metadatas.keys()):
        dataset = metadatas[filename]['dataset']
        cube = load_cube(filename, metadatas[filename])
        print (cube.data.shape)
        if pane == 'b':
            cube.aggregated_by('year', iris.analysis.MEAN)
            cube = get_max_amoc(cube)
            trends[dataset] = calculate_trend(cube)
        if pane == 'c':
            cube.aggregated_by('year', iris.analysis.MEAN)
            cube = get_max_amoc(cube)
            trends[dataset] = calculate_interannual(cube)

    #####
    # calculate the number of models
    model_numbers, number_models, projects= count_models(metadatas, obs_filename)

    box_data = [trends[dataset] for dataset in sorted(trends)]
    box = ax.boxplot(box_data,
                     0,
                     sym = 'k.',
                     whis = [1, 99],
                     showmeans= False,
                     meanline = False,
                     showfliers = True,
                     labels = sorted(trends.keys()))
    plt.xticks(rotation=45)
    plt.setp(box['fliers'], markersize=1.0)


    # Add observational data.

    # if obs_filename:
    #     obs_cube = iris.load_cube(obs_filename)
    #     obs_cube = diagtools.bgc_units(obs_cube, metadata['short_name'])
    #     # obs_cube = obs_cube.collapsed('time', iris.analysis.MEAN)
    #
    #     obs_key = obs_metadata['dataset']
    #     qplt.plot(obs_cube, obs_cube.coord('depth'), c='black')
    #
    #     plot_details[obs_key] = {'c': 'black', 'ls': '-', 'lw': 1,
    #                              'label': obs_key}
    if savefig:
        plt.subplots_adjust(bottom=0.25)

    # pane specific stuff
    if pane == 'b':
        plt.title('(b) Distribution of 8 year AMOC trends in CMIP5')
        plt.axhline(-0.55, c='k', lw=8, alpha=0.1, zorder = 0) # Wrong numbers!
        if not savefig:
            plt.setp( ax.get_xticklabels(), visible=False)

    if pane == 'c':
        plt.title('(c) Distribution of interannual AMOC changes in CMIP5')
        plt.axhline(-4.4, c='k', lw=8, alpha=0.1, zorder = 0) # wrong numbers!


    # title = ' '.join([
    #     metadata['dataset'],
    #     metadata['long_name'],
    # ])
    # plt.title(title)

    # Add Legend outside right.
    # diagtools.add_legend_outside_right(plot_details, plt.gca())
    # fig.set_size_inches(10., 9.)
    # leg = plt.legend(loc='lower right', prop={'size':10})
    # leg.draw_frame(False)
    # leg.get_frame().set_alpha(0.)

    if not savefig:
        return fig, ax

    # Load image format extention and path
    image_extention = diagtools.get_image_format(cfg)
    path = cfg['plot_dir'] + 'fig_3.24'+pane+image_extention

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path)

    plt.close()


def  make_figure(cfg):
    """
    Make the entire figure.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.

    """
    fig = plt.figure()
    fig.set_size_inches(w=11,h=7)
    gs1 = gridspec.GridSpec(2,5)

    # fig.subplots_adjust(wspace=0.25, hspace=0.1)

    axa = plt.subplot2grid((2,5), (0,0), colspan=2, rowspan=2)
    fig, axa = make_pane_a(cfg, fig=fig, ax=axa)

    axb = plt.subplot2grid((2,5), (0,2), colspan=3, rowspan=1)
    fig, axb = make_pane_bc(cfg, pane='b', fig=fig, ax=axb)

    axc = plt.subplot2grid((2,5), (1,2), colspan=3, rowspan=1)
    fig, axc = make_pane_bc(cfg, pane='c', fig=fig, ax=axc)

    plt.subplots_adjust(bottom=0.2, wspace=0.4, hspace=0.2)

    # Load image format extention and path
    image_extention = diagtools.get_image_format(cfg)
    path = cfg['plot_dir'] + 'fig_3.24'+image_extention

    # Watermakr
    fig.text(0.95, 0.05, 'Draft',
             fontsize=50, color='gray',
             ha='right', va='bottom', alpha=0.5)

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path)

    plt.close()



def main(cfg):
    """
    Run the diagnostics profile tool.

    Load the config file, find an observational dataset filename,
    pass loaded into the plot making tool.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.

    """
    make_figure(cfg)
    return

    # individual plots:
    make_pane_bc(cfg, pane='c')
    make_pane_bc(cfg, pane='b')
    make_pane_a(cfg)



    # for index, metadata_filename in enumerate(cfg['input_files']):
        # make_pane_a(cfg)
        # continue
        #
        # logger.info('metadata filename:\t%s', metadata_filename)
        #
        # metadatas = diagtools.get_input_files(cfg, index=index)
        #
        # obs_key = 'observational_dataset'
        # obs_filename = ''
        # obs_metadata = {}
        # if obs_key in cfg:
        #     obs_filename = diagtools.match_model_to_key(obs_key,
        #                                                 cfg[obs_key],
        #                                                 metadatas)
        #     obs_metadata = metadatas[obs_filename]
        #
        # for filename in sorted(metadatas.keys()):
        #
        #     if filename == obs_filename:
        #         continue
        #
        #     logger.info('-----------------')
        #     logger.info(
        #         'model filenames:\t%s',
        #         filename,
        #     )
        #
        #     ######
        #     # Time series of individual model
        #     make_single_profiles_plots(cfg, metadatas[filename], filename,
        #                         obs_metadata=obs_metadata,
        #                         obs_filename=obs_filename)

    logger.info('Success')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
