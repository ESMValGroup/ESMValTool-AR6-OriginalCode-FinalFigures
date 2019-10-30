"""
Fig 3.20 diagnostics
=======================

Diagnostic to produce figures of the time development of a field from
cubes. These plost show time on the x-axis and cube value (ie temperature) on
the y-axis.

Two types of plots are produced: individual model timeseries plots and
multi model time series plots. The inidivual plots show the results from a
single cube, even if this is a mutli-model mean made by the _multimodel.py
preproccessor. The multi model time series plots show several models
on the same axes, where each model is represented by a different line colour.

Note that this diagnostic assumes that the preprocessors do the bulk of the
hard work, and that the cube received by this diagnostic (via the settings.yml
and metadata.yml files) has a time component, no depth component, and no
latitude or longitude coordinates.

Author: Lee de Mora (PML)
        ledm@pml.ac.uk
"""

import logging
import os

import iris
import matplotlib.pyplot as plt
import numpy as np
import cf_units
import datetime

from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import run_diagnostic

from esmvalcore.preprocessor._time import extract_time


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))


def timeplot(cube, **kwargs):
    """
    Create a time series plot from the cube.

    Note that this function simple does the plotting, it does not save the
    image or do any of the complex work. This function also takes and of the
    key word arguments accepted by the matplotlib.pyplot.plot function.
    These arguments are typically, color, linewidth, linestyle, etc...

    If there's only one datapoint in the cube, it is plotted as a
    horizontal line.

    Parameters
    ----------
    cube: iris.cube.Cube
        Input cube

    """
    cubedata = np.ma.array(cube.data)
    if len(cubedata.compressed()) == 1:
        plt.axhline(cubedata.compressed(), **kwargs)
        return

    times = diagtools.cube_time_to_float(cube)
    plt.plot(times, cubedata, **kwargs)


def moving_average(cube, window):
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
        output.append(arr.mean())
    cube.data = np.array(output)
    return cube


def annual_average(cube,):
    """
    Adding annual average
    """
    coord_names = [coord[0].long_name for coord in cube.coords() ]
    if 'year' not in coord_names:
        iris.coord_categorisation.add_year(cube, 'time')
    cube.aggregated_by('year', iris.analysis.MEAN)
    return cube


def add_aux_times(cube):
    """
    Check for presence and add aux times if absent.
    """
    coord_names = [coord[0].long_name for coord in cube.coords() ]

    if 'day_of_month' not in coord_names:
        iris.coord_categorisation.add_day_of_month(cube, 'time')

    if 'month_number' not in coord_names:
        iris.coord_categorisation.add_month_number(cube, 'time')

    if 'day_of_year' not in coord_names:
        iris.coord_categorisation.add_day_of_year(cube, 'time')
    return cube


def zero_around(cube, year_initial=1971., year_final=1971.):
    """
    Zero around the time range provided.

    """
    new_cube = extract_time(cube, year_initial, 1, 1, year_final, 12, 31)
    mean = new_cube.data.mean()
    cube.data = cube.data - mean
    return cube


def make_new_time_array(times, new_datetime):
    """
    Make an array of these times in a new calendar.
    """
    # Enforce the same time.
    new_times = [
            new_datetime(time_itr.year,
                         time_itr.month,
                         15,
                         hour=0,
                         minute=0,
                         second=0,
                         ) for time_itr in times ]
    return np.array(new_times)


def recalendar(cube, calendar, units = 'days since 1950-1-1 00:00:00'):
    """
    Convert the 1D cube's time coordinate to a new standardised calendar.
    """

    #load times
    times = cube.coord('time').units.num2date(cube.coord('time').points).copy()

    # fix units; leave calendars
    cube.coord('time').convert_units(
        cf_units.Unit(units, calendar=cube.coord('time').units.calendar))

    # fix calendars
    cube.coord('time').units = cf_units.Unit(cube.coord('time').units.origin,
                                             calendar=calendar.lower())

    # make new array of correct datetimes.
    new_datetime = diagtools.load_calendar_datetime(calendar)
    new_times = make_new_time_array(times, new_datetime)

    # Make a list of floats for the time.
    time_c = [cube.coord('time').units.date2num(time) for time in new_times]
    cube.coord('time').points = np.array(time_c)

    # Remove bounds
    cube.coord('time').bounds = None

    # remove all aux coordinates
    for auxcoord in cube.aux_coords:
        cube.remove_coord(auxcoord)

    return cube


def make_mean_of_cube_list(cube_list):
    """
    Takes the mean of a list of cubes (not an iris.cube.CubeList).

    Assumes all the cubes are the same shape.
    """
    # Fix empty times
    full_times = {}
    times = []
    for cube in cube_list:
        # make time coords uniform:
        cube.coord('time').long_name='Time axis'
        cube.coord('time').attributes={'time_origin': '1950-01-01 00:00:00'}
        times.append(cube.coord('time').points)

        for time in cube.coord('time').points:
            print(cube.name, time, cube.coord('time').units)
            try:
                full_times[time] += 1
            except:
                full_times[time] = 1

    for t, v in sorted(full_times.items()):
        if v != len(cube_list):
            print('FAIL', t, v, '!=', len(cube_list),'\nfull times:',  full_times)
            assert 0

    cube_mean=cube_list[0]
    #try: iris.coord_categorisation.add_year(cube_mean, 'time')
    #except: pass
    #try: iris.coord_categorisation.add_month(cube_mean, 'time')
    #except: pass

    cube_mean.remove_coord('year')
    #cube.remove_coord('Year')
    try: model_name = cube_mean.metadata[4]['source_id']
    except: model_name = ''
    print(model_name,  cube_mean.coord('time'))

    for i, cube in enumerate(cube_list[1:]):
        #try: iris.coord_categorisation.add_year(cube, 'time')
        #except: pass
        #try: iris.coord_categorisation.add_month(cube, 'time')
        #except: pass
        cube.remove_coord('year')
        #cube.remove_coord('Year')
        try: model_name = cube_mean.metadata[4]['source_id']
        except: model_name = ''
        print(i, model_name, cube.coord('time'))
        cube_mean+=cube
        #print(cube_mean.coord('time'), cube.coord('time'))
    cube_mean = cube_mean/ float(len(cube_list))
    return cube_mean


def make_fig_3_20(
        cfg,
        metadatas,
        variable_group
):
    """
    Make a time series plot showing several preprocesssed datasets.

    This tool loads several cubes from the files, checks that the units are
    sensible BGC units, checks for layers, adjusts the titles accordingly,
    determines the ultimate file name and format, then saves the image.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.
    metadatas: dict

    """
    metadatas = diagtools.get_input_files(cfg)

    ####
    # Load the data for each layer as a separate cube
    model_cubes = {}
    piControl_cubes = {}
    for filename in sorted(metadatas):
        if metadatas[filename]['variable_group'] != variable_group:
            continue
        cube = iris.load_cube(filename)
        cube = diagtools.bgc_units(cube, metadatas[filename]['short_name'])
        
        if metadatas[filename]['exp'] == 'historical':
            model_cubes[filename] = cube
        if metadatas[filename]['exp'] == 'piControl':
            piControl_cubes[filename] = cube

    # Load image format extention
    image_extention = diagtools.get_image_format(cfg)

    title = ''
    z_units = ''
    plot_details = {}

    #####
    # Load obs data and details
    obs_cube = ''
    obs_key = ''
    obs_filename = ''

    #####
    # calculate the projects
    projects = {}
    for i, filename in enumerate(sorted(metadatas)):
        if metadatas[filename]['variable_group'] != variable_group:
            continue
        metadata = metadatas[filename]
        if filename == obs_filename: continue
        projects[metadata['project']] = True

    #####
    # List of cubes to make means/stds.
    project_cubes = {project:[] for project in projects}

    # Plot each file in the group
    project_colours={'CMIP3': 'blue', 'CMIP5':'black', 'CMIP6':'green'}

    for index, filename in enumerate(sorted(metadatas)):
        if metadatas[filename]['variable_group'] != variable_group:
            continue
        cube = model_cubes[filename]
        metadata = metadatas[filename]

        project = metadata['project']

        # Is this data is a multi-model dataset?
        if metadata['dataset'].find('MultiModel') > -1:
            continue

        # Reduce by average of 1986 - 2005.
        cube = zero_around(cube)

        # Change to a standard calendar.
        cube = recalendar(cube, 'Gregorian')

        project_cubes[project].append(cube)

        # Take a moving average, if needed.
        #if 'moving_average' in cfg:

        coord_names = [coord[0].long_name for coord in cube.coords() ]
        if 'year' not in coord_names:
            iris.coord_categorisation.add_year(cube, 'time')

        # cube = moving_average(cube, '10 years')

        #if 'annual_average' in cfg:
        # cube = annual_average(cube)

        print(project, metadata['dataset'], cube.data.min())

        # Make plots for single models
        timeplot(
                    cube,
                    c=project_colours[project],
                    ls='-',
                    lw=0.5,
                    alpha=0.5,
                )

    for project in projects:
        cube = make_mean_of_cube_list(project_cubes[project])

        plot_details[project] = {
            'c': project_colours[project],
            'ls': '-',
            'lw': 2.,
            'label': project
        }
        timeplot(
                    cube,
                    color=plot_details[project]['c'],
                    # label=plot_details[project]['label'],
                    ls=plot_details[project]['ls'],
                    lw=plot_details[project]['lw'],
                )

    # Draw horizontal line at zero
    plt.axhline(0., c='k', ls='--', lw=0.5)

    # Draw vertical area indicating the time period used to make a mean.
    # plt.axvspan(1986., 2006., alpha=0.1, color='k')

    # Add title, legend to plots
    plt.xlabel('Year')
    #plt.ylabel('Thermal Expansion (mm)')

    # Resize and add legend outside thew axes.
    plt.gcf().set_size_inches(8., 4.)
    diagtools.add_legend_outside_right(
         plot_details, plt.gca(), column_width=0.1)

    # Saving image:
    path = diagtools.folder(cfg['plot_dir']) + 'fig_3_20_' + variable_group + image_extention
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
    for index, metadata_filename in enumerate(cfg['input_files']):
        logger.info('metadata filename:\t%s', metadata_filename)

        metadatas = diagtools.get_input_files(cfg, index=index)

        #######
        # Multi model time series
        for variable_group in ['ohcga', 'ohc700']:
            make_fig_3_20(
                cfg,
                metadatas,
                variable_group
            )
    logger.info('Success')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
