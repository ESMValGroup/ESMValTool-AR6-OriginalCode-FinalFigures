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
import json

from scipy.stats import linregress
import cf_units

from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import run_diagnostic
from esmvalcore.preprocessor import climate_statistics
from esmvalcore.preprocessor._regrid import extract_levels

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def calculate_trend(cube, window = '8 years', tails=False, intersect_wanted=False):
    """
    Calculate a trend inside a window.

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

    When tails is True, the start and end of the data, they
    only include the average of the data available. Ie the first value
    in the moving average of a ``10 year`` window will only include the average
    of the five subsequent years.
    When tails is False, these tails are ignored.

    Parameters
    ----------
    cube: iris.cube.Cube
        Input cube
    window: str
        A description of the window to use for the
    tails: bool
        Boolean flag to switch off tails.

    Returns
    ----------
    iris.cube.Cube:
        A cube with the movinage average set as the data points.

    """
    assert 0
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

    slopes = []
    intercepts = []

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
        print (time_itr, len(arr))

        # No Tails
        if not tails:
            print(time_itr, [tmin, tmax], 'Length:', len(arr.compressed()), len(times), window_len*2 + 1)
            if len(arr.compressed()) != window_len*2 + 1:
                print("Wrong size")
                continue

#        print(time_itr, len(arr), len(times), window_len*2 + 1)
        time_arr = np.ma.masked_where(arr.mask, float_times)
        #if debug:
        #    print(time_itr, times, time_arr.compressed(),  arr.compressed())

        # print(time_itr, linregress(time_arr.compressed(), arr.compressed()))
        lnregs = linregress(time_arr.compressed(), arr.compressed())
        slopes.append(lnregs[0])
        intercepts.append(lnregs[1])
        #print(slopes)
    if intersect_wanted:
        return np.array(slopes), np.array(intercepts)
    else:
        return np.array(slopes)


def calculate_interannual(cube,):
    """
    Calculate the interannnual variability.
    """
    #if time_res == 'annual':
    #        cube = cube.aggregated_by('year', iris.analysis.MEAN)

    data = cube.data
    return np.array(data[1:] - data[:-1])


def calculate_midpoint(arr,):
    """
    Calculate the midpoint - usually for time axis
    """
    arr = np.ma.array(arr)
    return np.array(arr[1:] + arr[:-1])/2.


def calculate_basic_trend(cube, ): #window = '8 years'):
    """
    Calculate the 8 year window trend.

    The other function may be too complicated.
    this one keeps it simler.

        xx=(1:8);
        x=[xx*0+1;xx];
        for i=1:length(time2)-7
        b=regress(amoc_anave(i:i+7)',x');
        amoc_trend_slope(i)=b(2);
        end
    """
    # Assume annual data
    annual_data = cube.data
    times = diagtools.cube_time_to_float(cube)

    slopes, intercepts, new_times = [], [], []
    for itr in range(len(cube.data) -7):
        eight_years_data = annual_data[itr:itr+8]
        eight_years_times = times[itr:itr+8]
        print(itr, len(eight_years_data))
        if len(eight_years_data) == 8:
            assert ("Not the correct number of years: "+str(len(eight_years_data)) )
        lnregs = linregress(eight_years_times, eight_years_data)
        slopes.append(lnregs[0])
        intercepts.append(lnregs[1])
        new_times.append(np.mean(eight_years_times))

    return np.array(new_times), np.array(slopes), np.array(intercepts)


def calculate_basic_trend_arr(years, data): #window = '8 years'):
    """
    Calculate the 8 year window trend.

        xx=(1:8);
        x=[xx*0+1;xx];
        for i=1:length(time2)-7
        b=regress(amoc_anave(i:i+7)',x');
        amoc_trend_slope(i)=b(2);
        end
    """
    # Assume annual data
    slopes, intercepts, new_times = [], [], []
    years = np.ma.array([float(y) for y in years])
    for itr in range(len(data) -7):

        eight_years_data = list(data[itr:itr+8].compressed())
        eight_years_times = list(years[itr:itr+8].compressed())
        #print(itr, len(eight_years_data))
        #for var in [eight_years_times, eight_years_data]:
        #    print(var, type(var))
        if len(eight_years_data) != 8:
            #print("Not the correct number of data: "+str(len(eight_years_data)) )
            continue
        if len(eight_years_data) != len(eight_years_times):
            #print("Not the correct number of data/years: "+str(len(eight_years_data))
            #    +' vs '+str(len(eight_years_times)))
            continue

        lnregs = linregress(eight_years_times, eight_years_data)
        slopes.append(lnregs[0])
        intercepts.append(lnregs[1])
        new_times.append(np.mean(eight_years_times))

    return np.array(new_times), np.array(slopes), np.array(intercepts)


def annual_mean_from_april(cube, ):
    """
    Calculate the annual mean from April-March.

    Data from January, February and March will be marked
    into the previous year.
    Args:
    * cube (:class:`iris.cube.Cube`):
        The cube containing 'coord'. The new coord will be added into
        it.
    """
    coord = cube.coord('time')
    # Define the adjustments to be made to the year.
    month_year_adjusts = [None, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    name='year_from_april'

    # Define a categorisation function.
    def _year_from_april(coord, value):
        dt = coord.units.num2date(value)
        year = dt.year
        year += month_year_adjusts[dt.month]
        return year

    # Apply the categorisation.
    iris.coord_categorisation.add_categorised_coord(cube, name, coord, _year_from_april)

    print(cube.coord('time'), cube.coord(name))
    cube = cube.aggregated_by([name, ], iris.analysis.MEAN)
    return cube


def get_26North(cube):
    """
    Extract 26.5 North. (RAPID array)
    """
    coord_names = [cu.standard_name for cu in cube.coords()]
    print(coord_names)
    if 'latitude' in coord_names:
        latitude = cube.coord('latitude').points
        closest_lat = np.argmin(np.abs(latitude - 26.5))
        cube = cube.extract(iris.Constraint(latitude=latitude[closest_lat]))

    elif 'grid_latitude' in coord_names:
        latitude = cube.coord('grid_latitude').points
        closest_lat = np.argmin(np.abs(latitude - 26.5))
        cube = cube.extract(iris.Constraint(grid_latitude=latitude[closest_lat]))
    else: assert 0
    # print(closest_lat, latitude[closest_lat])
    # cube = cube.extract(iris.Constraint(latitude=latitude[closest_lat]))
    #print('get_26North: pre', cube.data.shape, latitude, latitude[closest_lat])
    #cube = cube.extract(iris.Constraint(latitude=latitude[closest_lat]))
    print('get_26North: post', cube.data.shape)
    return cube


def get_max_amoc(cube):
    """
    Extract maximum AMOC in the profile.
    """
    cube = cube.collapsed('depth', iris.analysis.MAX)
    print('get_max_amoc:',cube.data.shape)
    return cube


def load_cube(filename, metadata):
    """
    Load cube and set up correct units, and find 26.5 N

    """
    cube = iris.load_cube(filename)
    print('load_cube',cube.data.shape, cube.units)
    cube = diagtools.bgc_units(cube, metadata['short_name'])
    print('load_cube', cube.data.shape, cube.units)
    depth = cube.coord('depth')
    if str(depth.units) == 'centimeters':
        print('Bad depth units:',depth.units)
        depth.units = cf_units.Unit('m')
        depth.points = depth.points/100.
        if depth.bounds[0,1] < depth.points[0]:
            depth.bounds = depth.bounds/100.
        print('fixed units:', cube.coord('depth'))
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



def make_mean_of_cube_list(cube_list):
    """
    Takes the mean of a list of cubes (not an iris.cube.CubeList).

    Assumes all the cubes are the same shape.
    """
    # Fix empty times
    full_times = {}
    times = []
    print(cube_list)
    levels = []
    for cube in cube_list:
        # make time coords uniform:
        cube.coord('time').long_name='Time axis'
        cube.coord('time').attributes={'time_origin': '1950-01-01 00:00:00'}
        times.append(cube.coord('time').points)

        for time in cube.coord('time').points:
            print(time, cube.coord('time').units,':', cube.coord('time').units.num2date(time))
            try:
                full_times[time] += 1
            except:
                full_times[time] = 1
        level_points = cube.coord(axis='Z').points
        if len(level_points) > len(levels):
            levels = level_points

    for i, cube in enumerate(cube_list):
        print('regridding vertical' ) #cube.coord(axis='Z'))
        cube = extract_levels(cube, levels, 'linear_horizontal_extrapolate_vertical')
        print('post_regrid:',cube.data.max())
        cube.data = np.ma.masked_where(cube.data > 100000., cube.data)
        print('post_regrid (masked):',cube.data.max())
        cube_list[i] = cube
        #rint(depth in cube.coord('depth').points)

    #for t, v in sorted(full_times.items()):
    #    if v != len(cube_list):
    #        print('FAIL', t, v, '!=', len(cube_list),'\nfull times:',  full_times)
    #        assert 0

    cube_mean=cube_list[0].data
    #try: iris.coord_categorisation.add_year(cube_mean, 'time')
    #except: pass
    #try: iris.coord_categorisation.add_month(cube_mean, 'time')
    #except: pass

    #cube_mean.remove_coord('year')
    #cube.remove_coord('Year')
    #try: model_name = cube_mean.metadata[4]['source_id']
    #except: model_name = ''
    #print(model_name,  cube_mean.coord('time'))

    for i, cube in enumerate(cube_list[1:]):
        #try: iris.coord_categorisation.add_year(cube, 'time')
        #except: pass
        #try: iris.coord_categorisation.add_month(cube, 'time')
        #except: pass
        #cube.remove_coord('year')
        #cube.remove_coord('Year')
        #try: model_name = cube_mean.metadata[4]['source_id']
        #except: model_name = ''
        #print(i, model_name, cube.coord('time'))
        cube_mean+=cube.data
        #print(cube_mean.coord('time'), cube.coord('time'))
    cube_mean = cube_mean/ float(len(cube_list))
    cube.data = cube_mean
    return cube


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
    projects = {}
    for filename in sorted(metadatas.keys()):
        project =  metadatas[filename]['project']
        projects[project] = []

    for filename in sorted(metadatas.keys()):
        dataset = metadatas[filename]['dataset']
        short_name = metadatas[filename]['short_name']
        project =  metadatas[filename]['project']
        if short_name == 'amoc':
            continue
        print(dataset, short_name)
        cube = load_cube(filename, metadatas[filename])
        cube = get_26North(cube)
         
        print('post get_26North:', cube.shape, cube.data.max(), cube.units)
        if len(cube.coords('time')) and len(cube.coord('time').points) >1 :
            cubes[dataset] = climate_statistics(cube, operator='mean',
                                                period='full')
        else:
            cubes[dataset] = cube
        projects[project].append(cube)

    cmap = plt.cm.get_cmap('jet')

    #####
    # calculate the number of models
    model_numbers, number_models, projects_numbers= count_models(metadatas, obs_filename)

    plot_details = {}
    labeldone = {'CMIP5': False, 'CMIP6':False}
    for filename in sorted(metadatas.keys()):
        dataset =  metadatas[filename]['dataset']
        project =  metadatas[filename]['project']
        short_name = metadatas[filename]['short_name']
        if short_name == 'amoc':
            continue
        if number_models>1:
            value = float(model_numbers[dataset] ) / (number_models - 1.)
        else:
            value = 0.5

        max_index = np.argmax(cubes[dataset].data)
        #print(dataset, short_name, max_index, cubes[dataset].data[max_index] )
        #print(cubes[dataset])
        label = ' '.join([metadatas[filename]['dataset'],
                          ':',
                          '('+str(round(cubes[dataset].data[max_index] , 1)),
                          str(cubes[dataset].units)+',',
                          str(int(cubes[dataset].coord('depth').points[max_index])),
                          str(cubes[dataset].coord('depth').units)+')'
                          ])

        print(label)
        colour = cmap(value)
        if project == 'CMIP5':
            colour = 'dodgerblue'
            label = 'Individual CMIP5 models'
        elif project == 'CMIP6':
            colour = 'red'
            label = 'Individual CMIP6 models'

        if filename == obs_filename:
            plot_details[dataset] = {'c': 'black', 'ls': '-', 'lw': 3,
                                     'label': label, 'zorder':10}
        else:
            plot_details[dataset] = {'c': colour,
                                     'ls': '-',
                                     'lw': 1,
                                     #'label': label,
                                      'zorder': 1 }
        if not labeldone[project]:
            qplt.plot(cubes[dataset], cubes[dataset].coord('depth'),
                color = plot_details[dataset]['c'],
                linewidth = plot_details[dataset]['lw'],
                linestyle = plot_details[dataset]['ls'],
                label = label,
                zorder=plot_details[dataset]['zorder']
                )
            labeldone[project] = True
        else:
             qplt.plot(cubes[dataset], cubes[dataset].coord('depth'),
                 color = plot_details[dataset]['c'],
                 linewidth = plot_details[dataset]['lw'],
                 linestyle = plot_details[dataset]['ls'],
                 # label = label,
                 zorder=plot_details[dataset]['zorder']
                 )

        # Add a marker at the maximum
        plt.plot(cubes[dataset].data[max_index],
                 cubes[dataset].coord('depth').points[max_index],
                 c =  plot_details[dataset]['c'],
                 marker = 'd',
                 markersize = '10',
                 zorder=plot_details[dataset]['zorder']
                 )

    for project in projects:
        colour = cmap(value)
        if project == 'CMIP5':
            colour = 'darkblue'
        if project == 'CMIP6':
            colour = 'darkred'

        cube = make_mean_of_cube_list(projects[project])

        max_index = np.argmax(cube.data)

        label = ' '.join([project,
                          ':',
                          '('+str(round(cube.data[max_index] , 1)),
                          str(cube.units)+',',
                          str(int(cube.coord('depth').points[max_index])),
                          str(cube.coord('depth').units)+')'
                          ])

        plot_details[project] = {
            'c': colour,
            'ls': '-',
            'lw': 3.,
            'label': label,
            'zorder': 10,
        }

        qplt.plot(cube, cube.coord('depth'),
             color = plot_details[project]['c'],
             linewidth = plot_details[project]['lw'],
             linestyle = plot_details[project]['ls'],
             label = label,
             zorder=plot_details[project]['zorder']
             )

        plt.plot(cube.data[max_index],
                 cube.coord('depth').points[max_index],
                 c = colour,
                 marker = 'd',
                 markersize = '10',
                 zorder = plot_details[project]['zorder']
                 )

    add_obs = True
    if add_obs:
        # RAPID data from: https://www.rapid.ac.uk/rapidmoc/rapid_data/datadl.php
        # Downloaded 15/3/2019
        # The full doi for this data set is: 10.5285/5acfd143-1104-7b58-e053-6c86abc0d94b
        # moc_vertical.nc: MOC vertical profiles in NetCDF format
        obs_filename = cfg['auxiliary_data_dir']+"/moc_vertical.nc"
        if not os.path.exists(obs_filename):
            raise OSError("Observational data file missing. Please Download moc_vertical.nc data from https://www.rapid.ac.uk/rapidmoc/rapid_data/datadl.php and put it in the auxiliary_data_dir directory: "+str(obs_filename))
        obs_dataset = "RAPID"
        obs_cube = iris.load_cube(obs_filename)
        #cube = get_26North(cube)
        obs_cube = obs_cube.collapsed('time', iris.analysis.MEAN)
        max_index = np.argmax(obs_cube.data)
        print(obs_cube, max_index)
        label = ' '.join([obs_dataset,
                          ':',
                          '('+str(round(obs_cube.data[max_index] , 1)),
                          str(obs_cube.units)+',',
                          str(int(obs_cube.coord('depth').points[max_index])),
                          str(obs_cube.coord('depth').units)+')'
                          ])

        plot_details[obs_dataset] = {'c': 'black',
                                 'ls': '-',
                                 'lw': 3,
                                 'label': label}

        qplt.plot(obs_cube, obs_cube.coord('depth'),
            color = plot_details[obs_dataset]['c'],
            linewidth = plot_details[obs_dataset]['lw'],
            linestyle = plot_details[obs_dataset]['ls'],
            label = label
            )

        # Add a marker at the maximum
        plt.plot(obs_cube.data[max_index],
                 obs_cube.coord('depth').points[max_index],
                 c =  plot_details[obs_dataset]['c'],
                 marker = 'd',
                 markersize = '10',
                 )

    # Add title to plot
    # title = ' '.join([
    #     metadata['dataset'],
    #     metadata['long_name'],
    # ])
    # plt.title(title)
    plt.title('(a) AMOC streamfunction profiles')# at 26.5N')

    # Add Legend outside right.
    # diagtools.add_legend_outside_right(plot_details, plt.gca())
    leg = plt.legend(loc='lower right', prop={'size':6})
    leg.draw_frame(False)
    leg.get_frame().set_alpha(0.)

    if not savefig:
        return fig, ax

    # Load image format extention and path
    image_extention = diagtools.get_image_format(cfg)
    path = cfg['plot_dir'] + '/fig_3.24a'+image_extention

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path)

    plt.close()


def make_pane_bc(
        cfg,
        pane = 'b',
        fig=None,
        ax=None,
        timeseries = False,
        time_res="April-March",
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

    projects = {}
    for filename in sorted(metadatas.keys()):
        project =  metadatas[filename]['project']
        projects[project] = []

    #####
    # Load the CMIP data and calculate the trend or interannual variability
    trends = {}
    for filename in sorted(metadatas.keys()):
        dataset = metadatas[filename]['dataset']
        short_name = metadatas[filename]['short_name']
        project =  metadatas[filename]['project']
        if short_name != 'amoc':
            continue
        cube = load_cube(filename, metadatas[filename])
        print (cube.data.shape)
        if time_res=='monthly':
            cube = cube.aggregated_by(['month','year'], iris.analysis.MEAN)
        elif time_res=='annual':
            cube = cube.aggregated_by(['year',], iris.analysis.MEAN)
        elif time_res=="April-March":
            cube = annual_mean_from_april(cube)
        else: assert 0
        if pane == 'b':
            #cube = get_max_amoc(cube)
            #cube = cube.aggregated_by('year', iris.analysis.MEAN)
            new_times, slopes, intercepts = calculate_basic_trend(cube)
            trends[dataset] = slopes
            #trends[dataset] = calculate_trend(cube)
        if pane == 'c':
            #cube = get_max_amoc(cube)
            #cube = cube.aggregated_by('year', iris.analysis.MEAN)
            trends[dataset] = calculate_interannual(cube)
        projects[project].append(dataset)

    box_order = []

    #####
    # Add observational data.
    add_obs = True
    if add_obs:
        # RAPID data from: https://www.rapid.ac.uk/rapidmoc/rapid_data/datadl.php
        # Downloaded 15/3/2019
        # The full doi for this data set is: 10.5285/5acfd143-1104-7b58-e053-6c86abc0d94b
        # moc_transports.nc: MOC vertical profiles in NetCDF format
        obs_filename = cfg['auxiliary_data_dir']+"/moc_transports.nc"
        obs_dataset = "RAPID"
        variable_constraint = iris.Constraint(cube_func=(lambda c: c.var_name == 'moc_mar_hc10'))
        obs_cube = iris.load(obs_filename, constraints=variable_constraint)[0]
        iris.coord_categorisation.add_month(obs_cube, 'time', name='month')
        iris.coord_categorisation.add_year(obs_cube, 'time', name='year')
        #obs_cube = obs_cube.aggregated_by(['month','year'], iris.analysis.MEAN)
        if time_res=="April-March":
            obs_cube = annual_mean_from_april(obs_cube)
        elif time_res=='monthly':
            obs_cube = obs_cube.aggregated_by(['month','year'], iris.analysis.MEAN)
        elif time_res=='annual':
            obs_cube = obs_cube.aggregated_by(['year',], iris.analysis.MEAN)
        else: assert 0
        if pane == 'b':
            #obs_cube = get_max_amoc(obs_cube)
            new_times, slopes, intercepts = calculate_basic_trend(obs_cube)
            trends[obs_dataset] = slopes
        if pane == 'c':
            #obs_cube = get_max_amoc(obs_cube)
            trends[obs_dataset] = calculate_interannual(obs_cube)

    #####
    # calculate the number of models
    model_numbers, number_models, projects_numbers= count_models(metadatas, obs_filename)

    ####
    # Add project datasets
    for project in sorted(projects.keys()):
        datasets = projects[project]
        if len(datasets) == 0:
            continue
        box_order.append(project)
        trends[project] = []
        for dataset in datasets:
            print(project, dataset)
            print('trends[dataset]:',project, dataset, len(trends[dataset]))
            trends[project].extend(list(trends[dataset]))
            #except: trends[project] = trends[dataset]
            print('trends[project]:',project,len(trends[project]))

        if project == 'CMIP6':
            box_order.extend(sorted(datasets))
    #assert 0

    box_order.append(obs_dataset)

    if timeseries:
        # Draw the trend/variability as a time series
        cmap = plt.cm.get_cmap('jet')
        for dataset in sorted(trends):
            print(dataset, trends[dataset])
            try:
                value = float(model_numbers[dataset] ) / (number_models - 1.)
                color = cmap(value)
                lw = 1.
            except:
                color = 'black'
                lw = 3
            plt.plot(trends[dataset], c = color, lw=lw, label = dataset)
    else:
        # Draw the trend/variability as a box and whisker diagram.
        box_data = [trends[dataset] for dataset in box_order]
        box = ax.boxplot(box_data,
                         0,
                         sym = 'k.',
                         whis = [1, 99],
                         showmeans= False,
                         meanline = False,
                         showfliers = True,
                         labels = box_order) #sorted(trends.keys()))
        # Boxes indicate 25th to 75th percentiles, whiskers indicate 1st and 99th percentiles, and dots indicate outliers.
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.setp(box['fliers'], markersize=1.0)

    if savefig:
        plt.subplots_adjust(bottom=0.25)

    # pane specific stuff
    if pane == 'b':
        plt.title('(b) Distribution of 8 year AMOC trends')
        plt.axhline(-0.53, c='k', lw=8, alpha=0.1, zorder = 0) # Wrong numbers!
        plt.ylabel('Sv yr'+r'$^{-1}$')
        if not savefig:
            plt.setp( ax.get_xticklabels(), visible=False)

    if pane == 'c':
        plt.title('(c) Distribution of interannual AMOC changes')
        plt.axhline(-4.4, c='k', lw=8, alpha=0.1, zorder = 0) # wrong numbers!
        plt.ylabel('Sv')

    # If putting all the panes in one figure, return them now.
    if not savefig:
        return fig, ax
   # Save the pane as its own image.

    plt.axhline(0., ls='--', color='k', lw=0.5)
    if timeseries:
        plt.legend()

    # Load image format extention and path
    image_extention = diagtools.get_image_format(cfg)
    if timeseries:
        path = cfg['plot_dir'] + '/fig_3.24_'+pane+'_timeseries'+image_extention
    else:
        path = cfg['plot_dir'] + '/fig_3.24_'+pane+image_extention

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path)

    plt.close()


def make_amoc_trends(
    cfg,
    savefig = True,
    panes = [],
    fig=None,
    axes=None,
    ):
    #cfg = {'auxiliary_data_dir': '/users/modellers/ledm/workspace/ESMValTool_AR6/run/auxiliary_data'}
    preprocesed_filename = cfg['auxiliary_data_dir']+"/Figure_AR6_DAMIP_AMOC_26N_1000m.json"
    #
    data_str = open(preprocesed_filename, 'r').read()
    data = json.loads(data_str)

    # with open(preprocesed_filename, 'r') as handle:
    #     json_load = json.load(handle)
    #
    # amoc_c5_ts = np.ma.asarray(json_load["amoc_c5_ts"])  # Note the use of numpy masked arrays (np.ma)
    # amoc_c6_ts = np.ma.asarray(json_load["amoc_c6_ts"])
    # cmip5_models = json_load["cmip5_models"]
    # cmip6_models = json_load["cmip6_models"]
    # year = np.asarray(json_load["year"])

    years = np.ma.array(data['year'])
    models = {i:model for i,model in enumerate(data['damip6_models'])}
    #experiments = {'historical':0, 'hist-aer':1, 'hist-GHG':2, 'hist-nat':3, 'hist-stratO3':4}
    experiments = {'HIST':0, 'AER':1, 'GHG':2, 'NAT':3, 'hist-stratO3':4}

    latitudes = {'26.5N':0, '35N':1}
    ensembles = [i for i in range(10)]

    time_series = np.ma.array(data['amoc_damip6_ts'])
    time_series = np.ma.masked_where(time_series==None, time_series)
    # (7, 5, 10, 2, 251)
    #model = 0
    #exp = 1
    #ensemble = 2
    #lat = 3
    #time = 4

    time_ranges = {'1850-2014': (1849,2014), '1940-1985': (1940,1985),  '1985-2014': (1985,2014), }

    # pane a: 1850-2014 8 year trends
    # trends = {exp:{} for exp in experiments}

    latitude = '26.5N'
    lat = latitudes['26.5N']

    trends = {}
    for exp in experiments:
        trends[exp] = {}
        for time_range, range_values in time_ranges.items():
            trends[exp][time_range] = []

    for time_range, range_values in time_ranges.items(): #['1850-2014', '1940-1985', '1985-2014']:
        arr_min_index = np.argmin(np.abs(np.array(years) - range_values[0])) -1
        if arr_min_index<0: arr_min_index=0
        arr_max_index = np.argmin(np.abs(np.array(years) - range_values[1])) +1
        years2014 = years[arr_min_index:arr_max_index]
        sliice = slice(arr_min_index, arr_max_index, )
        print(time_range, arr_max_index, arr_min_index, sliice, years[arr_min_index] , '->',years[arr_max_index])
        #continue
        for experiment in ['GHG', 'NAT', 'AER','HIST']:
            exp = experiments[experiment]
            for mod, model in models.items():
                for ens in ensembles:
                    dat = time_series[mod,exp,ens,lat, sliice]
                    #print( dat, type(dat))
                    if not len(dat.compressed()):
                        continue
                        #dat = np.ma.masked_where((years > 2015) + dat.mask, dat)
                    #if len(dat.compressed())!= len(years2014.compressed()):
                #        print(len(dat.compressed()), '!=', len(years2014.compressed()))
                #        assert 0
                    new_times, slopes, intercepts = calculate_basic_trend_arr(years2014, dat)
                    #print(model,experiment,'ensemble:', ens, ', mean slope:', slopes.mean())
                    trends[experiment][time_range].extend( slopes)

                #kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)

    if savefig:
        fig = plt.figure()
        fig.set_size_inches(9., 4.)
        time_ranges_subplot = {'1850-2014': 131, '1940-1985': 132,  '1985-2014': 133, }
        time_ranges_panes = {'1850-2014': '(a)', '1940-1985': '(b)',  '1985-2014': '(c)', }

    else:
        time_ranges_subplot = {'1850-2014': axes[0], '1940-1985': axes[1],  '1985-2014': axes[2], }
        time_ranges_panes = {'1850-2014': '(d)', '1940-1985': '(e)',  '1985-2014': '(f)', }
        # time_ranges_subplot = {'1850-2014': axes[0], '1940-1985': axes[1],  '1985-2014': axes[2], }

    for time_range, subplot in time_ranges_subplot.items(): #['1850-2014', '1940-1985', '1985-2014']:
        # fig = plt.figure()
        # for experiment in ['GHG', 'NAT', 'AER', 'HIST']:
        #     plt.hist(trends[experiment][time_range], bins=50,histtype='stepfilled', normed=True, alpha=0.5, label = experiment)
        # plt.title(' '.join(['AMOC', latitude, time_range]))
        # plt.legend()
        # plt.savefig('tmp'+time_range+'.png')
        # plt.close()

        if savefig:
            ax = plt.subplot(subplot)
            if subplot not in [131, 311]:
                plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax = subplot
        print(time_range, subplot)
        ax.set_ylim([-0.55,0.5])

        box_order =  ['GHG', 'NAT', 'AER', 'HIST']
        box_data = [trends[experiment][time_range] for experiment in box_order]
        ax.axhline(0., ls='--', color='k', lw=0.5)
        box = ax.boxplot(box_data,
                         0,
                         sym = 'k.',
                         whis = [ 5, 95],
                         showmeans= False,
                         meanline = False,
                         showfliers = False,
                         labels = box_order) #sorted(trends.keys()))
        # Boxes indicate 25th to 75th percentiles, whiskers indicate 1st and 99th percentiles, and dots indicate outliers.
        # plt.xticks(rotation=30, ha="right", fontsize=8)
        # plt.setp(box['fliers'], markersize=1.0)
        print(time_range, time_ranges_panes[time_range])
        ax.set_title(' '.join([time_ranges_panes[time_range], time_range]))

    if not savefig:
        return fig, axes

    # Load image format extention and path
    image_extention = diagtools.get_image_format(cfg)
    path = cfg['plot_dir'] + '/fig_3.24_def_amoc_trends'+image_extention

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path)
    plt.close()


def  make_figure(cfg, debug=False, timeseries=False):
    """
    Make the entire figure.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.

    """
    fig = plt.figure()
    fig.set_size_inches(w=11,h=7)

    # axa = plt.subplot2grid((2,5), (0,0), colspan=2, rowspan=2)
    # fig, axa = make_pane_a(cfg, fig=fig, ax=axa)
    # axb = plt.subplot2grid((2,5), (0,2), colspan=3, rowspan=1)
    # fig, axb = make_pane_bc(cfg, pane='b', fig=fig, ax=axb, timeseries=timeseries)
    # axc = plt.subplot2grid((2,5), (1,2), colspan=3, rowspan=1)
    # fig, axc = make_pane_bc(cfg, pane='c', fig=fig, ax=axc, timeseries=timeseries)
    # plt.subplots_adjust(bottom=0.2, wspace=0.4, hspace=0.2)

    #plt.subplots_adjust(bottom=0.2, wspace=0.4, hspace=0.2)
    axd = plt.subplot2grid((3,3), (2,0), colspan=1, rowspan=1)
    axe = plt.subplot2grid((3,3), (2,1), colspan=1, rowspan=1)
    axf = plt.subplot2grid((3,3), (2,2), colspan=1, rowspan=1)
    fig, axes = make_amoc_trends(
        cfg,
        savefig = False,
        panes = ['d', 'e', 'f'],
        fig=fig,
        axes=[axd, axe, axf],
        )

    # (rows, columns)
    axa = plt.subplot2grid((3,3), (0,0), colspan=1, rowspan=2)
    fig, axa = make_pane_a(cfg, fig=fig, ax=axa)

    axb = plt.subplot2grid((3,3), (0,1), colspan=2, rowspan=1)
    fig, axb = make_pane_bc(cfg, pane='b', fig=fig, ax=axb, timeseries=timeseries)

    axc = plt.subplot2grid((3,3), (1,1), colspan=2, rowspan=1)
    fig, axc = make_pane_bc(cfg, pane='c', fig=fig, ax=axc, timeseries=timeseries)
    #fig, axc = make_pane_bc(cfg, pane='c', fig=fig, ax=axc, timeseries=timeseries)
    
    #plt.subplots_adjust(bottom=0.15, wspace=0.2, hspace=0.4)
    plt.tight_layout()
    # Load image format extention and path
    image_extention = diagtools.get_image_format(cfg)
    if timeseries:
        path = cfg['plot_dir'] + '/fig_3.24_timeseries'+image_extention
    else:
        path = cfg['plot_dir'] + '/fig_3.24'+image_extention

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
    # individual plots:
    # make_timeseriespane_bc(cfg, pane='b')
    # make_timeseriespane_bc(cfg, pane='c')

    make_figure(cfg, timeseries= False)
    make_amoc_trends(cfg, savefig=True)

    make_pane_a(cfg)
    #make_pane_a(cfg)

    make_pane_bc(cfg, pane='b', timeseries=False)
    make_pane_bc(cfg, pane='c', timeseries=False)
    #make_pane_bc(cfg, pane='c', timeseries=True)
    #make_pane_bc(cfg, pane='b', timeseries=True)

    # overall plots:
    # make_figure(cfg, timeseries= True)
    #make_figure(cfg, timeseries= False)

    logger.info('Success')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
