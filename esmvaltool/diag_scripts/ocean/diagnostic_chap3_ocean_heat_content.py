"""
Fig 3.20 diagnostics
=======================

Diagnostic to produce figures of the time development of the Ocean
heat content at various depth ranges.

This one derives the OHC in place. This is quicker than deriving it in
ESMValTool.

There are four recipes that call this diagnostic:
Currently called :
- recipe_ocean_fig_3_20_ohcgt_no_derivation.yml
- recipe_ocean_fig_3_20_ohc700_no_derivation_cmip6_testing.yml
- recipe_ocean_fig_3_20_ohc7002000_no_derivation_cmip6_testing.yml
- recipe_ocean_fig_3_20_ohc2000_no_derivation_cmip6_testing.yml


Author: Lee de Mora (PML)
        ledm@pml.ac.uk
"""

import logging
import os

import iris
import matplotlib.pyplot as plt
import numpy as np
import itertools
import cf_units
import datetime
from scipy.stats import linregress
from scipy.io import loadmat
import netCDF4 

#from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

from glob import glob
from dask import array as da
from shelve import open as shopen

from matplotlib.colors import LogNorm

from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import run_diagnostic

from esmvalcore.preprocessor._time import extract_time
try:
    import gsw
except: 
    print('Unable to load gsw.\n You need to install it in your conda environmetn with:\npip install gsw')



# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))


def derive_ohc_ga(cube, volume):
    """
    derive the Ocean heat content from temperature and volunme
    """
    # Only work for thetaoga

    if volume.ndim == 0:
        volume = np.tile(volume.data, cube.data.shape[0])

    elif volume.ndim == 1:
        volume = volume.data

    elif volume.ndim == 3:
        print('calculating volume sum (3D):')
        volume = np.ma.sum(volume.data)
        print(volume, volume.shape, cube.data.shape)
        volume = np.tile(volume, cube.data.shape)

    elif volume.ndim == 4:
        print('calculating volume sum (4D):', volume)
        vols = []
        for t in np.arange(volume.shape[0]):
            print('4D volune calc:', t,)
            #vol = np.ma.sum(volume.data[t])
            vol = da.sum(volume.data[t])
            vols.append(vol)
            print(vol)
        volume = np.array(vols)

    else:
        print(cube.data.shape , 'does not match', volume.data.shape)
        assert 0

    const = 4.09169e+6
    cube.data = cube.data * volume * const
    # if time_coord_present:
    #     for coord, dim in dim_coords:
    #         cube.add_dim_coord(coord, dim)
    #     for coord, dims in aux_coords:
    #         cube.add_aux_coord(coord, dims)
    return cube


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
    print('Adding', kwargs, 'to plot.')
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


def detrend(cfg, metadata, cube, pi_cube, method = 'linear regression'):
    """
    Detrend the historical cube using the pi_control cube.
    """
    if cube.data.shape != pi_cube.data.shape:
        print(cube.data.shape, pi_cube.data.shape)
        assert 0

    decimal_time = diagtools.cube_time_to_float(cube)
    if method == 'linear regression':
        linreg = linregress( np.arange(len(decimal_time)), pi_cube.data)
        line = [ (t * linreg.slope) + linreg.intercept for t in np.arange(len(decimal_time))]

    fig = plt.figure()
    plt.plot(decimal_time, cube.data, label = 'historical')
    plt.plot(decimal_time, pi_cube.data, label = 'PI control')
    plt.plot(decimal_time, line, label = 'PI control '+method)

    detrended = cube.data - np.array(line)
    cube.data = detrended
    plt.plot(decimal_time, detrended, label = 'Detrended historical')

    plt.axhline(0., c = 'k', ls=':' )
    plt.legend()
    dataset = metadata['dataset']
    plt.title(dataset +' detrending ('+method+')')

    image_extention = diagtools.get_image_format(cfg)
    path = diagtools.folder(cfg['plot_dir']) + 'detrending_' + dataset + image_extention
    logger.info('Saving detrending plots to %s', path)
    plt.savefig(path)
    plt.close()
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
        variable_group,
        plot_projects = 'All'
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

    #TODO add pi control detrending using relevant years (https://github.com/ESMValGroup/ESMValCore/issues/342)
    #TODO Need to add observational data
    #TODO Put both panes on the same figure
    #TODO colour scales.
    if variable_group == 'ohcgt':
        variable_groups = ['thetaoga_Ofx', 'thetaoga_Omon' ]#'ohc_omon', 'ohcgt_Omon', 'ohcgt_Ofx', 'ohcgt']
        volume_groups = [ 'volcello_Ofx' ,'volcello_Omon']#'ohc_omon', 'ohcgt_Omon', 'ohcgt_Ofx', 'ohcgt']
    if variable_group == 'ohc700':
            variable_groups = ['thetao700_Ofx', 'thetao700_fx', 'thetao700_Omon',
                               'thetao700_CMIP6_Ofx', 'thetao700_CMIP6_Omon',
                               'thetao700_CMIP5_fx']
            volume_groups = ['volcello700_Ofx' ,'volcello700_Omon',
                             'volcello700_CMIP6_Ofx', 'volcello700_CMIP6_Omon',
                             'volcello700_CMIP5_fx']
    if variable_group == 'ohc7002000':
            variable_groups = ['thetao7002000_Ofx', 'thetao7002000_fx', 'thetao7002000_Omon',
                               'thetao7002000_CMIP6_Ofx', 'thetao7002000_CMIP6_Omon',
                               'thetao7002000_CMIP5_fx']
            volume_groups = ['volcello7002000_Ofx' ,'volcello7002000_Omon',
                             'volcello7002000_CMIP6_Ofx', 'volcello7002000_CMIP6_Omon',
                             'volcello7002000_CMIP5_fx']
    if variable_group == 'ohc2000':
            variable_groups = ['thetao2000_Ofx', 'thetao2000_fx', 'thetao2000_Omon',
                               'thetao2000_CMIP6_Ofx', 'thetao2000_CMIP6_Omon',
                               'thetao2000_CMIP5_fx']
            volume_groups = ['volcello2000_Ofx' ,'volcello2000_Omon',
                             'volcello2000_CMIP6_Ofx', 'volcello2000_CMIP6_Omon',
                             'volcello2000_CMIP5_fx']
    linestyles = ['-', ':', '--', '-.', '-', ':', '--', '-.','-', ':', '--', '-.','-', ':', '--', '-.', '-', ':', '--', '-.','-', ':', '--', '-.',]
    linethicknesses = [0.5,0.5,0.5,0.5, 1.,1.,1.,1., 1.5,1.5,1.5,1.5, 2.,2.,2.,2., 2.5,2.5,2.5,2.5,2.5,2.5,]

    ####
    # Load the data for each layer as a separate cube
    hist_cubes = {}
    piControl_cubes = {}
    hist_vol_cubes = {}
    piControl_vol_cubes = {}
    data_loaded = False
    for filename in sorted(metadatas):
        dataset = metadatas[filename]['dataset']
        if metadatas[filename]['variable_group'] in variable_groups:
            data_loaded = True
            cube = iris.load_cube(filename)
            cube = diagtools.bgc_units(cube, metadatas[filename]['short_name'])

            if metadatas[filename]['exp'] == 'historical':
                hist_cubes[filename] = cube
            if metadatas[filename]['exp'] == 'piControl':
                piControl_cubes[dataset] = cube

        if metadatas[filename]['variable_group'] in volume_groups:
            data_loaded = True
            cube = iris.load_cube(filename)
            cube = diagtools.bgc_units(cube, metadatas[filename]['short_name'])

            # Volume has a time component.
            if metadatas[filename]['mip'] in ['Omon', 'Oyr',]:
                if metadatas[filename]['exp'] == 'historical':
                    hist_vol_cubes[dataset] = cube
                if metadatas[filename]['exp'] == 'piControl':
                    piControl_vol_cubes[dataset] = cube

            # volume has no time component
            if metadatas[filename]['mip'] in ['Ofx', 'fx',]:
                if metadatas[filename]['exp'] == 'historical':
                    hist_vol_cubes[dataset] = cube
                    piControl_vol_cubes[dataset] = cube
                if metadatas[filename]['exp'] == 'piControl':
                    hist_vol_cubes[dataset] = cube
                    piControl_vol_cubes[dataset] = cube


    if not data_loaded:
        return
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
    matfile = cfg['auxiliary_data_dir'] + '/OHC/AR6_GOHC_GThSL_timeseries_2019-11-26.mat'
    matdata = loadmat(matfile)

    #####
    # calculate the projects
    projects = {}
    for i, filename in enumerate(sorted(metadatas)):
        if metadatas[filename]['variable_group'] not in variable_groups:
            continue
        if filename == obs_filename: continue
        projects[metadatas[filename]['project']] = True

    #####
    # List of cubes to make means/stds.
    project_cubes = {project:[] for project in projects}

    # Plot each file in the group
    project_colours={'CMIP3': 'blue', 'CMIP5':'purple', 'CMIP6':'green', 'obs': 'black'}

    linecount=-1
    for index, filename in enumerate(sorted(metadatas)):
        metadata = metadatas[filename]
        dataset = metadata['dataset']
        project = metadata['project']
        if metadatas[filename]['variable_group'] not in variable_groups:
            continue

        #print(index, dataset,project,metadatas[filename]['variable_group'])

        if metadatas[filename]['exp'] == 'piControl':
            print(metadatas[filename]['exp'], '==', 'piControl')
            continue
        if plot_projects == 'all':
            pass
        elif plot_projects != project:
            print(plot_projects, '!=', project)
            continue
        cube = hist_cubes[filename]
        pi_cube = piControl_cubes[dataset]
        if dataset not in hist_vol_cubes:
            print(dataset, 'not in', hist_vol_cubes.keys())
            print('ie, can not find volume cube for', dataset)
            continue
        vol_cube =  hist_vol_cubes[dataset]
        if dataset in piControl_vol_cubes:
            pi_vol_cube =  piControl_vol_cubes[dataset]
        else:
            pi_vol_cube = vol_cube

        print('deriving:', project, dataset, 'historical')
        cube = derive_ohc_ga(cube, vol_cube)
        print('deriving:', project, dataset, 'piControl')
        pi_cube = derive_ohc_ga(pi_cube, pi_vol_cube)

        # Is this data is a multi-model dataset?
        if metadata['dataset'].find('MultiModel') > -1:
            continue

        # do the various operations.
        cube = zero_around(cube, year_initial=1971., year_final=1971.)
        pi_time = pi_cube.coord('time')
        pi_year = pi_time.units.num2date(pi_time.points)[0].year + 11.
        pi_cube = zero_around(pi_cube, year_initial=pi_year-5., year_final=pi_year+5.)
        cube = detrend(cfg, metadata, cube, pi_cube)

        # Change to a standard calendar.
        cube = recalendar(cube, 'Gregorian')

        project_cubes[project].append(cube)


        coord_names = [coord[0].long_name for coord in cube.coords() ]
        if 'year' not in coord_names:
            iris.coord_categorisation.add_year(cube, 'time')

        # Make plots for single models
        if plot_projects == 'all':
            timeplot(
                    cube,
                    c=project_colours[project],
                    ls='-',
                    lw=0.5,
                    alpha=0.5,
                )
        else:
                print('plotting:', dataset,project, linecount, project_colours[project],linestyles[linecount],linethicknesses[linecount])
                linecount+=1
                plot_details[dataset] = {
                    'c': project_colours[project],
                    'ls': linestyles[linecount],
                    'lw': linethicknesses[linecount],
                    'label': dataset
                }
                timeplot(
                        cube,
                        c=project_colours[project],
                        ls=linestyles[linecount],
                        lw=linethicknesses[linecount],
                        alpha=0.5,
                )

        # save cube:
        output_cube = diagtools.folder(cfg['work_dir']) + '_'.join(['OHC','zeroed','detrended','recalendared',project, dataset])+'.nc'
        logger.info('Saving cubes to %s', output_cube)
        iris.save(cube, output_cube)

    for project in projects:
        if plot_projects == 'all':
            pass
        elif plot_projects != project:
            continue
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
        output_cube = diagtools.folder(cfg['work_dir']) + '_'.join(['OHC','zeroed','detrended','recalendared',project, 'mean'])+'.nc'
        logger.info('Saving project cubes to %s', output_cube)
        iris.save(cube, output_cube)


    # Add observations
    add_obs = False
    if add_obs:
        # Data sent via email!
        matfile = cfg['auxiliary_data_dir'] + '/OHC/AR6_GOHC_GThSL_timeseries_2019-11-26.mat'
        matdata = loadmat(matfile)
        # depths = matdata['dep']
        depths = ['0-300 m', '0-700 m','700-2000 m','>2000 m','Full-depth']
        obs_years = matdata['time_yr'][0] + 0.5
        obs_years = np.ma.masked_where(obs_years < 1960., obs_years)
        hc_data = matdata['hc_global']

        def strip_name(array): return str(array[0][0]).strip(' ')
        def zetta_to_joules(dat): return dat * 1.E21

        hc_global = {}
        for z, depth in enumerate(depths):
            hc_global[depth] = {}
            for ii, array in enumerate(matdata['hc_yr_fname']):
                name = strip_name(array)
                series = hc_data[ii,z,:]
                series = np.ma.masked_invalid(series)
                series = zero_around_dat(obs_years, series)
                series = zetta_to_joules(series)
                series = np.ma.masked_where(obs_years.mask, series)
                hc_global[depth][name] = series

        if variable_group == 'ohcgt':
            obs_series = hc_global['Full-depth']['Domingues+Ishii+Purkey (Full)']
        if variable_group == 'ohc700':
            obs_series = hc_global['0-700 m']['Domingues+Ishii+Purkey (Full)']

            #print(obs_series, hc_global.keys(), hc_global['0-700 m'])
            #assert 0
        project = 'obs'
        plot_details[project] = {
            'c': project_colours[project],
            'ls': '-',
            'lw': 2.,
            'label': 'Observations',
            }
        #print(obs_series)
        plt.plot(obs_years,
                 obs_series,
                 c = plot_details['obs']['c'],
                 lw = plot_details['obs']['lw'],
                 ls = plot_details['obs']['ls'],
                 )
    # Add observations
    if plot_projects == 'all':
        add_all_obs = True
    elif plot_projects == 'obs':
        add_all_obs = True
    else:
        add_all_obs = False

    if add_all_obs:
        matfile = cfg['auxiliary_data_dir'] + '/OHC/AR6_GOHC_GThSL_timeseries_2019-11-26.mat'
        matdata = loadmat(matfile)
        # depths = matdata['dep']
        depths = ['0-300 m', '0-700 m','700-2000 m','>2000 m','Full-depth']
        obs_years = matdata['time_yr'][0] + 0.5
        obs_years = np.ma.masked_where(obs_years < 1960., obs_years)
        hc_data = matdata['hc_global']

        def strip_name(array): return str(array[0][0]).strip(' ')
        def zetta_to_joules(dat): return dat * 1.E21

        hc_global = {}
        for z, depth in enumerate(depths):
            hc_global[depth] = {}
            for ii, array in enumerate(matdata['hc_yr_fname']):
                name = strip_name(array)
                series = hc_data[ii,z,:]
                series = np.ma.masked_invalid(series)
                series = zero_around_dat(obs_years, series)
                series = zetta_to_joules(series)
                series = np.ma.masked_where(obs_years.mask, series)
                hc_global[depth][name] = series

        if variable_group == 'ohcgt':
            obs_series = hc_global['Full-depth']
        elif variable_group == 'ohc700':
            obs_series = hc_global['0-700 m']
        elif variable_group == 'ohc7002000':
            obs_series = hc_global['700-2000 m']
        elif variable_group == 'ohc2000':
            obs_series = hc_global['>2000 m']
        else:
            print('Unable to determine depth:', variable_group)
            assert 0

        for i, name in enumerate(sorted(obs_series.keys())):
            if np.isnan(obs_series[name].max()): continue
            if len(obs_series[name].compressed()) == 0: continue
            project = 'obs'
            if plot_projects == 'all':
                plot_details[project] = {
                    'c': project_colours[project],
                    'ls': '-',
                    'lw': 2.,
                    'label': 'Observations',
                    }
                plt.plot(obs_years,
                         obs_series[name],
                         c = plot_details['obs']['c'],
                         lw = 0.5,
                         ls = plot_details['obs']['ls'],
                         )
            else:
                plot_details[name] = {
                    'c': project_colours[project],
                    'ls': linestyles[i],
                    'lw': linethicknesses[i],
                    'label': name,
                    }
                plt.plot(obs_years,
                         obs_series[name],
                         c = plot_details[name]['c'],
                         lw = plot_details[name]['lw'],
                         ls = plot_details[name]['ls'],
                         )

    # Draw horizontal line at zero
    plt.axhline(0., c='k', ls='--', lw=0.5)

    # Draw vertical area indicating the time period used to make a mean.
    # plt.axvspan(1986., 2006., alpha=0.1, color='k')

    # Add title, legend to plots
    plt.xlabel('Year')
    if variable_group == 'ohcgt':
        plt.ylabel('Change in Global Total Heat Content, J')
    if variable_group == 'ohc700':
        plt.ylabel('Change in Heat Content in top 700m, J')
    if variable_group == 'ohc7002000':
        plt.ylabel('Change in Heat Content in 700m-2000m, J')
    if variable_group == 'ohc2000':
        plt.ylabel('Change in Heat Content below 2000m, J')
    # Resize and add legend outside thew axes.
    plt.gcf().set_size_inches(8., 4.)
    diagtools.add_legend_outside_right(
         plot_details, plt.gca(), column_width=0.12)

    # Saving image:
    path = diagtools.folder(cfg['plot_dir']) + 'fig_3_20_' + variable_group + '_'+plot_projects + image_extention
    logger.info('Saving plots to %s', path)
    plt.savefig(path)
    plt.close()


#####
# Above here is old code.
def zero_around(cube, year_initial=1971., year_final=1971.):
    """
    Zero around the time range provided.

    """
    new_cube = extract_time(cube, year_initial, 1, 1, year_final, 12, 31)
    mean = new_cube.data.mean()
    cube.data = cube.data - mean
    return cube

def zero_around_dat(times, data, year=1971.):
    """
    Zero around the time range provided.
    """
    index = np.argmin(np.abs(np.array(times) - year))
    return data - np.ma.mean(data[index-1:index+2])


def top_left_text(ax, text):
    """
    Adds text to the top left of ax.
    """
    plt.text(0.05, 0.9, text,
             horizontalalignment='left',
             verticalalignment='center',
             transform=ax.transAxes)


def load_convert(fn):
     cube = iris.load_cube(fn)
     cube.convert_units('ZJ')
     times = diagtools.cube_time_to_float(cube)
     data = zero_around_dat(times, cube.data, year=1971.)
     return times, data


def single_pane(fig, ax, fn, color='red', xlim=False, no_ticks=False):
    times, data = load_convert(fn)
    ax.plot(times, data, c=color)
    if xlim:
        ax.set_xlim(xlim)
    if no_ticks:
        ax.set_xticklabels([])
    return fig, ax  


def multimodel_2_25(cfg, metadatas, ocean_heat_content_timeseries):
    """
    Multimodel version of the 2.25 plot.
    """

    depth_ranges = ['total', '0-700m', '700-2000m', '0-2000m', '2000m_plus']
    RHS_xlim = [1995, 2017]

    projects = list({index[0]:True for index in ocean_heat_content_timeseries.keys()}.keys())
    datasets = list({index[1]:True for index in ocean_heat_content_timeseries.keys()}.keys())
    ensembles = list({index[3]:True for index in ocean_heat_content_timeseries.keys()}.keys())

    datasets = sorted(datasets)
    color_dict = {da:c for da, c in zip(datasets, ['r' ,'b'])}
    color_dict['Observations'] = 'black'
    # ocean_heat_content_timeseries keys:
    # (project, dataset, 'piControl', pi_ensemble, 'ohc', 'intact', depth_range)

    depth_dict = { 321: 'total', 
                   322: '0-2000m', 
                   323: '0-700m',
                   324: '0-700m', 
                   326:  '700-2000m',
                   (6, 2, 9):  '700-2000m', 
                   (6, 2, 11): '2000m_plus'}
    xlims_dict = { 321: False,
                   322: RHS_xlim,
                   323: False,
                   324: RHS_xlim,
                   326: RHS_xlim,
                   (6, 2, 9):  False,
                   (6, 2, 11): False}
    no_ticks= {321: True,
               322: True,
               323: True,
               324: True,
               326: False,
               (6, 2, 9):  True,
               (6, 2, 11): False}

    plot_details={}
    fig = plt.figure()
    fig.set_size_inches(10, 7)
    axes= {}
    for subplot in depth_dict.keys():
        if isinstance(subplot, int):
            axes[subplot] =  plt.subplot(subplot)
        else:
            axes[subplot] =  plt.subplot(subplot[0], subplot[1], subplot[2]) 
        if subplot== 323:
            
            plt.ylabel('OHC (ZJ)', fontsize=16)

    for project, dataset, ensemble in itertools.product(projects, datasets, ensembles): 

        total_key = (project, dataset, 'historical', ensemble, 'ohc', 'detrended', 'total')
        fn = ocean_heat_content_timeseries.get(total_key, False)
        if not fn: 
            continue

        for subplot, ax in axes.items():
            key =  (project, dataset, 'historical', ensemble, 'ohc', 'detrended', depth_dict[subplot])
            fn = ocean_heat_content_timeseries[key]
            fig, ax = single_pane(fig, ax, fn,  
                                  color=color_dict[dataset], 
                                  xlim=xlims_dict[subplot],
                                  no_ticks=no_ticks[subplot])

    for subplot, ax in axes.items():
        ax.axhline(0., c='k', ls=':')

    top_left_text(axes[321], 'Full-depth')
    top_left_text(axes[322], '0-2000m')
    top_left_text(axes[323], '0-700m')
    top_left_text(axes[324], '0-700m')
    top_left_text(axes[326], '700m - 2000m')
    top_left_text(axes[(6, 2, 9)], '700m - 2000m')
    top_left_text(axes[(6, 2, 11)], '> 2000m')

    plt.suptitle('Global Ocean Heat Content')

    add_all_obs = True
    if add_all_obs:
        matfile = cfg['auxiliary_data_dir'] + '/OHC/AR6_GOHC_GThSL_timeseries_2019-11-26.mat'
        matdata = loadmat(matfile)
        depths = ['0-300 m', '0-700 m','700-2000 m','>2000 m','Full-depth']
        obs_years = matdata['time_yr'][0] + 0.5
        obs_years = np.ma.masked_where(obs_years < 1960., obs_years)
        hc_data = matdata['hc_global']
        datasets.append('Observations')

        def strip_name(array): return str(array[0][0]).strip(' ')
        def zetta_to_joules(dat): return dat * 1.E21

        hc_global = {}
        for z, depth in enumerate(depths):
            hc_global[depth] = {}
            for ii, array in enumerate(matdata['hc_yr_fname']):
                name = strip_name(array)
                series = hc_data[ii,z,:]
                series = np.ma.masked_invalid(series)
                series = zero_around_dat(obs_years, series)
                series = zetta_to_joules(series)
                series = np.ma.masked_where(obs_years.mask, series)
                hc_global[depth][name] = series

        for subplot, depth_key in depth_dict.items():
            if depth_key == 'total':
                obs_series = hc_global['Full-depth'] 
            elif depth_key == '0-700m':
                obs_series = hc_global['0-700 m']
            elif depth_key == '700-2000m':
                obs_series = hc_global['700-2000 m']
            elif depth_key == '2000m_plus':
                obs_series = hc_global['>2000 m']
            else:
                obs_series = {}

            for i, name in enumerate(sorted(obs_series.keys())):
                if np.isnan(obs_series[name].max()): continue
                if len(obs_series[name].compressed()) == 0: continue
                axes[subplot].plot(obs_years,
                           obs_series[name]*1E-21,
                           c = 'black',
                           lw = 0.5,
                           )

    if len(datasets) <=5:
        axleg = plt.axes([0.0, 0.00, 0.9, 0.10]) 
    else:
        axleg = plt.axes([0.0, 0.00, 0.9, 0.15])
    axleg.axis('off')

    # Add emply plots to dummy axis.
    for dataset in datasets:
        axleg.plot([], [], c=color_dict[dataset], lw=2, ls='-', label=dataset)

    legd = axleg.legend(
            loc='upper center',
            ncol=5,
            prop={'size': 10},
            bbox_to_anchor=(0.5, 0.5,),
            fontsize=12)
    legd.draw_frame(False)
    legd.get_frame().set_alpha(0.)

    fig_dir = diagtools.folder([cfg['plot_dir'], 'multimodel_ohc'])
    image_extention = diagtools.get_image_format(cfg)
    fig_fn = fig_dir + '_'.join(['multimodel_ohc',
                                 ])+image_extention

    plt.savefig(fig_fn)
    print('multimodel_ohc: saving',fig_fn)
    plt.close()


def fig_like_2_25(cfg, metadatas, ocean_heat_content_timeseries, dataset, ensemble, project, exp):
    """
    Produce a 6 pane figure showing the time series.
    """
    depth_ranges = ['total', '0-700m', '700-2000m', '0-2000m', '2000m_plus']

    RHS_xlim = [1995, 2017]
    fns = {depth_range: ocean_heat_content_timeseries[(project, dataset, exp, ensemble, 'ohc', 'detrended',depth_range)] for depth_range in depth_ranges}

    cubes = {depth_range:iris.load_cube(fn) for depth_range, fn in fns.items()}

    zettaJoules = True
    if zettaJoules:
         for dr, cube in cubes.items():
             cube.convert_units('ZJ')

    zero_around_1971=True
    if zero_around_1971:
        if exp.find('hist')>-1:
            for dr, cube in cubes.items():
                times = diagtools.cube_time_to_float(cube)
                cube.data = zero_around_dat(times, cube.data, year=1971.)

    fig = plt.figure()
    fig.set_size_inches(10, 7)

    times = diagtools.cube_time_to_float(cubes['total'])
    ax1 = plt.subplot(321)
    plt.plot(times, cubes['total'].data)
    top_left_text(ax1, 'Full-depth')
    ax1.set_xticklabels([])

    ax2 = plt.subplot(322)
    plt.plot(times, cubes['0-2000m'].data)
    ax2.set_xlim(RHS_xlim)
    top_left_text(ax2, '0-2000m')
    ax2.set_xticklabels([])

    ax3 = plt.subplot(323)
    plt.plot(times, cubes['0-700m'].data)
    top_left_text(ax3, '0-700m')
    ax3.set_xticklabels([])

    if zettaJoules:
        plt.ylabel('OHC (ZJ)', fontsize=16)
    else:
        plt.ylabel('OHC '+str(cube.units), fontsize=16)    

    ax4 = plt.subplot(324)
    plt.plot(times, cubes['0-700m'].data)
    ax4.set_xlim(RHS_xlim)
    top_left_text(ax4, '0-700m')
    ax4.set_xticklabels([])

    ax9 = plt.subplot(6, 2, 9)
    plt.plot(times, cubes['700-2000m'].data)
    top_left_text(ax9, '700m - 2000m')
    ax9.set_xticklabels([])

    ax11 = plt.subplot(6, 2, 11)
    plt.plot(times, cubes['2000m_plus'].data)
    top_left_text(ax11, '> 2000m')

    ax6 = plt.subplot(326)
    plt.plot(times, cubes['700-2000m'].data)
    ax6.set_xlim(RHS_xlim)
    top_left_text(ax6, '700m - 2000m')

    plt.suptitle(' '.join(['Global Ocean Heat Content:', project, dataset, exp, ensemble]))
    if zero_around_1971:
        for ax in [ax1, ax2, ax3, ax4, ax6, ax9, ax11]:
            ax.axhline(0., c='k', ls=':')

    fig_dir = diagtools.folder([cfg['plot_dir'], 'ohc_summary'])
    image_extention = diagtools.get_image_format(cfg)
    fig_fn = fig_dir + '_'.join([project, exp, dataset, ensemble, 'ohc_summary',
                                 ])+image_extention

    plt.savefig(fig_fn)
    print('detrending_fig: saving',fig_fn)
    plt.close()

    


def detrending_fig(cfg, 
        metadatas, 
        detrended_hist, 
        trend_intact_hist, 
        detrended_piC, 
        trend_intact_piC,
        depth_range,
        key):
    """
    Make figure showing detrending process as a time series.
    """
    print('detrending_fig: detrended_hist',detrended_hist)
    print('detrending_fig: trend_intact_hist', trend_intact_hist)
    print('detrending_fig: detrended_piC',detrended_piC)
    print('detrending_fig: trend_intact_piC', trend_intact_piC)

    if not trend_intact_piC: 
        skip_intact_piC = True
    else: skip_intact_piC = False 
    short_name = metadatas[detrended_hist]['short_name']
    dataset = metadatas[detrended_hist]['dataset']
    ensemble = metadatas[detrended_hist]['ensemble']
    project = metadatas[detrended_hist]['project']
    exp = metadatas[detrended_hist]['exp']


    cube_d_h = iris.load_cube(detrended_hist)
    cube_i_h = iris.load_cube(trend_intact_hist)
    cube_d_p = iris.load_cube(detrended_piC)
    if not  skip_intact_piC:
        cube_i_p = iris.load_cube(trend_intact_piC)

    times = diagtools.cube_time_to_float(cube_d_h)

    print('detrending_fig:', key, cube_d_h.data.max(), cube_i_h.data.max())
    print('detrending_fig: times', key, times)
    d_h_data = zero_around_dat(times, cube_d_h.data, year=1971.)
    i_h_data = zero_around_dat(times, cube_i_h.data, year=1971.)
    d_p_data = zero_around_dat(times, cube_d_p.data, year=1971.)
    if not  skip_intact_piC:
        i_p_data = zero_around_dat(times, cube_i_p.data, year=1971.)

    #print('detrending_fig:', key, d_h_data.max(), i_h_data.max(), i_p_data.max())
    print('detrending_fig: d h:', key, d_h_data.max())
    print('detrending_fig: i h:', key, i_h_data.max())
    print('detrending_fig: d p:', key, d_p_data.max())
    print('detrending_fig: i p:', key, i_p_data.max())


    plt.plot(times, d_h_data, color = 'red', label = 'Detrended Historical')
    plt.plot(times, i_h_data, color = 'blue', label = 'Historical')
    plt.plot(times, d_p_data, color = 'orange', label = 'Detrended PI Control')
    if not  skip_intact_piC:
        plt.plot(times, i_p_data, color = 'green', label = 'PI Control')

    plt.axhline(0., c = 'k', ls=':' )
    title = ' '.join([key, dataset, exp, ensemble, depth_range])
    plt.title(title)
    plt.legend()

    fig_dir = diagtools.folder([cfg['plot_dir'], 'detrending_ts'])
    image_extention = diagtools.get_image_format(cfg)
    fig_fn = fig_dir + '_'.join([project, exp, dataset, ensemble, key, 'detrending_ts',
                                   depth_range])+image_extention

    plt.savefig(fig_fn)
    print('detrending_fig: saving',fig_fn)
    plt.close()



def add_map_subplot(subplot, cube, nspace, title='',
                    cmap='viridis', extend='neither', log=False):
    """
    Add a map subplot to the current pyplot figure.
    Parameters
    ----------
    subplot: int
        The matplotlib.pyplot subplot number. (ie 221)
    cube: iris.cube.Cube
        the iris cube to be plotted.
    nspace: numpy.array
        An array of the ticks of the colour part.
    title: str
        A string to set as the subplot title.
    cmap: str
        A string to describe the matplotlib colour map.
    extend: str
        Contourf-coloring of values outside the levels range
    log: bool
        Flag to plot the colour scale linearly (False) or
        logarithmically (True)
    """
    plt.subplot(subplot)
    logger.info('add_map_subplot: %s', subplot)
    print('add_map_subplot: %s', subplot, title, (cmap, extend, log))
    if log:
        qplot = iris.quickplot.contourf(
            cube,
            nspace,
            linewidth=0,
            cmap=plt.cm.get_cmap(cmap),
            norm=LogNorm(),
            zmin=nspace.min(),
            zmax=nspace.max())
        qplot.colorbar.set_ticks([0.1, 1., 10.])
    else:
        qplot = iris.plot.contourf(
            cube,
            nspace,
            linewidth=0,
            cmap=plt.cm.get_cmap(cmap),
            extend=extend,
            zmin=nspace.min(),
            zmax=nspace.max())
        cbar = plt.colorbar(orientation='horizontal')
        cbar.set_ticks(
            [nspace.min(), (nspace.max() + nspace.min()) / 2.,
             nspace.max()])

    try: plt.gca().coastlines()
    except: pass
    plt.title(title)


def single_pane_map_plot(
        cfg,
        metadata,
        cube,
        key='',
        ):
    """
    Make a single pane map figure.
    """
    short_name = metadata['short_name']
    dataset = metadata['dataset']
    ensemble = metadata['ensemble']
    project = metadata['project']
    exp = metadata['exp']

    try: 
        times = cube.coord('time').units.num2date(cube.coord('time').points)
        year = str(times[0].year)
    except: year = ''

    unique_id = [dataset, exp, ensemble, short_name, year, key]

    # Determine image filename

    path = diagtools.folder([cfg['plot_dir'], key]) + '_'.join(unique_id)
    path = path.replace(' ', '') + diagtools.get_image_format(cfg)

    if os.path.exists(path): 
        return

    nspace = np.linspace(
        cube.data.min(), cube.data.max(), 20, endpoint=True)
    title = ' '.join(unique_id)
    print('single_pane_map_plot:', unique_id, nspace, [cube.data.min(), cube.data.max()], cube.data.shape)
    add_map_subplot(111, cube, nspace, title=title,)
    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path, dpi=200)
    plt.close()


def make_difference_plots(
        cfg,
        metadata,
        detrended_cube,
        hist_cube):
    """
    Make a figure showing four maps and the other shows a scatter plot.
    The four pane image is a latitude vs longitude figures showing:
    * Top left: model
    * Top right: observations
    * Bottom left: model minus observations
    * Bottom right: model over observations
    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.
    metadata: dict
        the input files dictionairy
    """

    short_name = metadata['short_name']
    dataset = metadata['dataset']
    ensemble = metadata['ensemble']
    project = metadata['project']

    long_name = ' '.join([ short_name, dataset, ]) 
    units = str(hist_cube.units)

    # Load image format extention
    image_extention = diagtools.get_image_format(cfg)

    fig = plt.figure()
    fig.set_size_inches(9, 6)

    # Create the cubes
    cube221 = detrended_cube[-1,0]
    cube222 = hist_cube[-1, 0]
    cube223 = cube221 - cube222
    cube224 =  cube221/cube222

    # create the z axis for plots 2, 3, 4.
    extend = 'neither'
    zrange12 = diagtools.get_cube_range([cube221, cube222])
    #if 'maps_range' in metadata[input_file]:
    #    zrange12 = metadata[input_file]['maps_range']
    #    extend = 'both'
    zrange3 = diagtools.get_cube_range_diff([cube223])
    #if 'diff_range' in metadata[input_file]:
    #    zrange3 = metadata[input_file]['diff_range']
    #    extend = 'both'

    cube224.data = np.ma.clip(cube224.data, 0.1, 10.)

    print('plotting:', long_name, 'zrange12:',zrange12)
    n_points = 12
    linspace12 = np.linspace(
        zrange12[0], zrange12[1], n_points, endpoint=True)
    linspace3 = np.linspace(
        zrange3[0], zrange3[1], n_points, endpoint=True)
    logspace4 = np.logspace(-1., 1., 12, endpoint=True)

    # Add the sub plots to the figure.
    add_map_subplot(
        221, cube221, linspace12, cmap='viridis', title='Detrended',
        extend=extend)
    add_map_subplot(
        222, cube222, linspace12, cmap='viridis',
        title='Historical',
        extend=extend)
    add_map_subplot(
        223,
        cube223,
        linspace3,
        cmap='bwr',
        title='Difference',
        extend=extend)
    if np.min(zrange12) > 0.:
        add_map_subplot(
            224,
            cube224,
            logspace4,
            cmap='bwr',
            title='Quotient',
            log=True)

    # Add overall title
    fig.suptitle(long_name + ' [' + units + ']', fontsize=14)

    # Determine image filename    
    fn_list = ['Detrended', project, dataset, ensemble, short_name, 'quad_maps']
    path = diagtools.folder(cfg['plot_dir']) + '_'.join(fn_list)
    path = path.replace(' ', '') + image_extention

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path, dpi=200)

    plt.close()


def maenumerate(marr):
    """
    masked array version of ndenumerate.
    """
    mask = ~marr.mask.ravel()
    for i, m in itertools.izip(np.ndenumerate(marr), mask):
        if m: yield i


def calc_ohc_ts(cfg, metadatas, ohc_fn, depth_range, trend):
    """
    Calculate Ocean Heat Content time series.
    """
    exp = metadatas[ohc_fn]['exp']
    dataset = metadatas[ohc_fn]['dataset']
    ensemble = metadatas[ohc_fn]['ensemble']
    project = metadatas[ohc_fn]['project']

    work_dir = diagtools.folder([cfg['work_dir'], 'ohc_ts'])
    output_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'ohc_ts',
                                     trend, depth_range])+'.nc'
  
    fig_dir = diagtools.folder([cfg['plot_dir'], 'ohc_ts'])
    image_extention = diagtools.get_image_format(cfg)
    fig_fn = fig_dir + '_'.join([project, dataset, exp, ensemble, 'ohc_ts', 
                                 trend, depth_range])+image_extention

    if os.path.exists(output_fn):
        return output_fn

    print('calc_ohc_ts: calculating:', depth_range, trend, dataset, exp, ensemble)

    # depth_ranges = ['total', '0-700m', '700-2000m', '0-2000m', '2000m_plus']
    cube = iris.load_cube(ohc_fn)
    times = diagtools.cube_time_to_float(cube)
    print('calc_ohc_ts', exp, dataset, ensemble, project, depth_range, trend)
    if depth_range != 'total':
    
        if depth_range ==  '0-700m':
            zmax = 700.
            zmin = 0.
        elif depth_range ==  '700-2000m':
            zmax = 2000.
            zmin = 700.
        elif depth_range ==  '0-2000m':
            zmax = 2000.
            zmin = 0.
        elif depth_range ==  '2000m_plus':
            zmax = 10000.
            zmin = 2000.
        else:
            print('Depth range',depth_range, 'not recognised')
            assert 0
        z_constraint = iris.Constraint(
            coord_values={
                cube.coord(axis='Z'): lambda cell: zmin < cell.point < zmax})
        cube = cube.extract(z_constraint)

    cube = cube.collapsed([cube.coord(axis='z'),
                           'longitude', 'latitude'],
                          iris.analysis.SUM)

    iris.save(cube, output_fn)

    title = ' '.join([dataset, exp, ensemble, trend, 'OHC', depth_range])
    fig = plt.figure()
    timeplot(cube)
    plt.title(title)
    plt.savefig(fig_fn)
    plt.close()

    return output_fn


def derive_ohc(cube, volume):
    """
    derive the Ocean heat content from temperature and volunme
    """
    # Only work for thetao
    # needs a time axis.
    times = diagtools.cube_time_to_float(cube)
    const = 4.09169e+6

    if volume.ndim == 3:
        print('calculating volume sum (3D):')
        for t, time in enumerate(times):
            cube.data[t] = cube.data[t] * volume.data * const 
    elif volume.ndim == 4:
        for t, time in enumerate(times):
            cube.data[t] = cube.data[t] * volume[t].data * const
    else:
        print('Volume and temperature do not match')
        assert 0
    cube.units = cf_units.Unit('J')
    cube.name = 'Ocean heat Content'
    cube.short_name = 'ohc'
    cube.var_name = 'ohc'
    print(cube.units)

    return cube




def calc_ohc_basic(cfg, metadatas, thetao_fn, volcello_fn, trend=''):
    """
    Calculate OHC form files using basic fudge factor..
    """
    exp = metadatas[thetao_fn]['exp']
    dataset = metadatas[thetao_fn]['dataset']
    ensemble = metadatas[thetao_fn]['ensemble']
    project = metadatas[thetao_fn]['project']

    work_dir = diagtools.folder([cfg['work_dir'], 'OHC'])
    output_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'ocean_heat', trend])+'.nc'

    if os.path.exists(output_fn):
        return output_fn

    times = diagtools.cube_time_to_float(cube)
    const = 4.09169e+6

    if volume.ndim == 3:
        print('calculating volume sum (3D):')
        for t, time in enumerate(times):
            cube.data[t] = cube.data[t] * volume.data * const
    elif volume.ndim == 4:
        for t, time in enumerate(times):
            cube.data[t] = cube.data[t] * volume[t].data * const
    else:
        print('Volume and temperature do not match')

    cube = iris.load_cube(thetao_fn)
    vol_cube = iris.load_cube(volcello_fn)
    ohc_cube = derive_ohc(cube, vol_cube)
    iris.save(ohc_cube, output_fn)

    for t in [0, -1]:
        single_pane_map_plot(
                cfg,
                metadatas[thetao_fn],
                ohc_cube[t, 0],
                key='OHC'
                )


def calc_slr_full(cfg, metadatas,
        hist_thetao_fn, hist_so_fn,
        pi_thetao_fn, pi_so_fn,
        trend='intact'
        ):
    """
    First we calculate the climatological average (mean along the time dimension) of the Pre-industrial control thetao (temperature) and so (salinity) datasets. These will be called thetao_bar and so_bar. Also, we apply the gsw.conversions to them to conservative temperature (ctemp_bar) and absolute salinity (asal_bar).

    Then, using the Gibbs Seawater library, we iterate of the 4D dataset.

    for each point in time and space in a 4D dataset:
        pressure = gsw.conversions.p_from_z(depth, lat) # dbar

        # note that so and thetao have already been detrended here.
        ctemp = gsw.conversions.CT_from_t(so, thetao, pressure)
        asal = gsw.conversions.SA_from_SP(so, pressure, lon, lat)

        # Calculate specific volumes:     
        svan_clim = gsw.specvol_anom_standard(asal_bar, ctemp_bar, pressure)
        svan_total = gsw.specvol_anom_standard(asal, ctemp, pressure) - svan_clim
        svan_thermosteric = gsw.specvol_anom_standard(asal_bar, ctemp, pressure) - svan_clim
        svan_halosteric = svan_total - svan_thermosteric
    """

    # PI control:
    exp = metadatas[hist_thetao_fn]['exp']
    dataset = metadatas[hist_thetao_fn]['dataset']
    ensemble = metadatas[hist_thetao_fn]['ensemble']
    project = metadatas[hist_thetao_fn]['project']

    work_dir = diagtools.folder([cfg['work_dir'], 'SLR'])
    output_fns = {}
    output_fns['total'] = work_dir + '_'.join([project, dataset, exp, ensemble, 'total_SLR', trend])+'.nc'
    output_fns['thermo'] = work_dir + '_'.join([project, dataset, exp, ensemble, 'thermo_SLR', trend])+'.nc'
    output_fns['halo'] = work_dir + '_'.join([project, dataset, exp, ensemble, 'halo_SLR', trend])+'.nc'

    if False not in [os.path.exists(fn) for fn in output_fns.values()]:
        for key, fn in output_fns.items():
          cube1 = iris.load_cube(fn)
          for t in [0, -1]:
              single_pane_map_plot(
                      cfg,
                      metadatas[hist_thetao_fn],
                      cube1[t, 0],
                      key='SLR_'+key+'_'+trend
                      )
        return output_fns['total'], output_fns['thermo'], output_fns['halo'] 

    # load hist netcdfs
    print('load hist netcdfs')
    so_cube = iris.load_cube(hist_so_fn)
    thetao_cube = iris.load_cube(hist_thetao_fn)

    # load dimensions
    lats = thetao_cube.coord('latitude').points
    lons = thetao_cube.coord('longitude').points
    if lats.ndim == 1:
        lon, lat = np.meshgrid(lons,lats)
    elif lats.ndim == 2:
        lat = lats
        lon = lons

    depths = -1.*np.abs(thetao_cube.coord('depth').points) # depth is negative here.
    times = diagtools.cube_time_to_float(thetao_cube) # decidmal time.

    # Calculate climatology   
    print('Calculate clim')
    psal_bar = iris.load_cube(pi_so_fn)
    psal_bar = psal_bar.collapsed('time', iris.analysis.MEAN)
    print('clim: so:', psal_bar.shape)
    temp_bar = iris.load_cube(pi_thetao_fn)
    temp_bar = temp_bar.collapsed('time', iris.analysis.MEAN)
    print('clim: thetao:', temp_bar.shape)

    depths_bar = -1.*np.abs(temp_bar.coord('depth').points)

    #output
    print('copying output cube')
    slr_total = thetao_cube.data.copy()
    slr_thermo = thetao_cube.data.copy()

    # Calculate SLR in 2D 
    for z in np.arange(thetao_cube.data.shape[1]):
        print('Calculate SLR in 2D:', z) 
        if depths.ndim == 1:
            depth = np.zeros_like(lat) + depths[z]
        elif depths.ndim == 3:
            depth = depths[z]

        if depths.ndim != 4:
            pressure = gsw.conversions.p_from_z(depth, lat) # dbar

        # clim:
        if depths_bar.ndim == 1:
            depth_bar = np.zeros_like(lat) + depths_bar[z]
        elif depths_bar.ndim == 3:
            depth_bar = depths_bar[z]
        else: assert 0
        print('starting clim calc')
        pressure_bar = gsw.conversions.p_from_z(depth_bar, lat) # dbar
        print('pressure_bar', pressure_bar.shape, lat.shape, depth_bar.shape)
        psal_z_bar = psal_bar.data[z]
        print('psal_z_bar', psal_z_bar.shape)
        ctemp_bar = gsw.conversions.CT_from_t(psal_z_bar, temp_bar.data[z], pressure_bar)    
        print('ctemp_bar:',ctemp_bar.shape, psal_z_bar.shape)
        asal_bar = gsw.conversions.SA_from_SP(psal_z_bar, pressure_bar, lon, lat)
        print('asal_bar', asal_bar.shape)
        svan_clim = gsw.specvol_anom_standard(asal_bar, ctemp_bar, pressure_bar)
        print('svan_clim:', svan_clim.shape)

        # not sure about this part
        svan_clim = svan_clim * pressure_bar
        print(z,'svan_clim max:', svan_clim.max())
        for t, time in enumerate(times):
            print(t, z, 'performing 2D SLR')
            if depths.ndim == 4:
                depth = depths[t, z]
                pressure = gsw.conversions.p_from_z(depth, lat)# dbar

            psal = so_cube[t, z].data
            temp = thetao_cube[t, z].data
            ctemp = gsw.conversions.CT_from_t(psal, temp, pressure)
            asal = gsw.conversions.SA_from_SP(psal, pressure, lon, lat)

            # Calculate specific volumes: m3 kg-1
            svan_total = gsw.specvol_anom_standard(asal, ctemp, pressure)*pressure - svan_clim
            svan_thermosteric = gsw.specvol_anom_standard(asal_bar, ctemp, pressure)*pressure - svan_clim
            #svan_total = gsw.specvol_anom_standard(asal, ctemp, pressure) - svan_clim
            #svan_thermosteric = gsw.specvol_anom_standard(asal_bar, ctemp, pressure) - svan_clim
            """
            pressure is in dbar (1 decibar is 10000 Pa is 10000 kg m-1 s-2), 
            so these calculations produce units of m^3 kg^-1 *dbar = 1e5 m^2 s-2.

From there, we divide the dynamic heights by 9.81m s_2 (acceleration due to gravity) then multiply by 1000 to convert to mm.  ( I guess this is where Paul’s factor of 100 comes from?)
            """
            
#           svan_halosteric = svan_total - svan_thermosteric
            slr_total[t, z] = svan_total * 10000. # dynamic heights (m2 s-2)
            slr_thermo[t, z] = svan_thermosteric * 10000. # dynamic heights (m2 s-2)

    slr_total = slr_total.sum(axis=1) * 1000 / 9.81 # units: mm
    slr_thermo = slr_thermo.sum(axis=1)* 1000 / 9.81 # units: mm
    slr_halo = slr_total - slr_thermo     

    cube0 = thetao_cube[:,0,:,:].copy()
    cube0.data = slr_total
    cube0.units = cf_units.Unit('mm')
    cube0.name = 'Total Sea Level Rise'
    cube0.short_name = 'slr_total'
    cube0.var_name = 'slr_total'
    iris.save(cube0, output_fns['total'])

    cube1 = thetao_cube[:,0,:,:].copy()
    cube1.data = slr_thermo
    cube1.units = cf_units.Unit('mm')
    cube1.name = 'Halosteric Sea Level Rise'
    cube1.short_name = 'slr_thermo'
    cube1.var_name = 'slr_thermo'
    iris.save(cube1, output_fns['thermo'])

    cube2 = thetao_cube[:,0,:,:].copy()
    cube2.data = slr_halo
    cube2.units = cf_units.Unit('mm')
    cube2.name = 'Halosteric Sea Level Rise'
    cube2.short_name = 'slr_halo'
    cube2.var_name = 'slr_halo'
    iris.save(cube2, output_fns['halo'])

    for t in [0, -1]:
        for key, cube in zip(('total', 'thermo', 'halo'),
                             (cube0, cube1, cube2)):
            single_pane_map_plot(
                cfg,
                metadatas[hist_thetao_fn],
                cube[t],
                key='SLR_'+key+'_'+trend
                )
    return output_fns['total'], output_fns['thermo'], output_fns['halo']


def calc_ohc_full(cfg, metadatas, thetao_fn, so_fn, volcello_fn, trend='intact'):
    """
    Calculate OHC form files using full method..
    """
    exp = metadatas[thetao_fn]['exp']
    dataset = metadatas[thetao_fn]['dataset']
    ensemble = metadatas[thetao_fn]['ensemble']
    project = metadatas[thetao_fn]['project']


    work_dir = diagtools.folder([cfg['work_dir'], 'OHC'])
    output_ohc_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'ocean_heat', trend])+'.nc'

    print('\n-------\ncalc_ohc_full', (exp, dataset, ensemble, project, trend), thetao_fn)
    print('output_ohc_fn:', output_ohc_fn)

    if os.path.exists(output_ohc_fn):
        cube1 = iris.load_cube(output_ohc_fn)
        for t in [0, -1]:
            single_pane_map_plot(
                    cfg,
                    metadatas[thetao_fn],
                    cube1[t, 0],
                    key='OHC_full_'+trend
                    )

        return output_ohc_fn 

    thetao_cube = iris.load_cube(thetao_fn)
    so_cube = iris.load_cube(so_fn)
    vol_cube = iris.load_cube(volcello_fn)

    for fn in [thetao_fn, so_fn, volcello_fn, output_ohc_fn]:
        if fn.find(exp) == -1: 
            print('ERROR:', exp, 'not in', fn)
            assert 0

    lats = thetao_cube.coord('latitude').points
    lons = thetao_cube.coord('longitude').points
    depths = -1.*np.abs(thetao_cube.coord('depth').points) # depth is negative here.
    times = diagtools.cube_time_to_float(thetao_cube) # decidmal time.
    ohc_data = thetao_cube.data.copy()
    
    count = 0
    # layer by layer calculation.
    if lats.ndim == 1:
        #lat, lon = np.meshgrid(lats,lons)
        lon, lat = np.meshgrid(lons,lats)

    elif lats.ndim == 2:
        lat = lats
        lon = lons

    # 2D!
    for z in np.arange(thetao_cube.data.shape[1]):
        if  depths.ndim == 1:
            depth = np.zeros_like(lat) + depths[z]
        elif  depths.ndim == 3:
            depth = depths[z]

        if vol_cube.ndim == 3:
            vol = vol_cube.data[z]

        for t, time in enumerate(times):
            print (t,z, 'performing 2D calc')
            if depths.ndim == 4:
                depth = depths[t,z]
            if vol_cube.ndim == 4:
                vol = vol_cube.data[t, z]

            psal = so_cube.data[t,z]
            temp = thetao_cube.data[t,z]
            pressure = gsw.conversions.p_from_z(depth, lat) # dbar
            ctemp = gsw.conversions.CT_from_t(psal, temp, pressure)
            asal = gsw.conversions.SA_from_SP(psal, pressure, lon, lat)
            rho = gsw.density.rho(asal, ctemp, pressure) #  kg/ m3
            energy = gsw.energy.internal_energy(asal, ctemp, pressure) # J/kg
            #enthalpy = gsw.energy.enthalpy(asal, ctemp, pressure) # J/kg

            cell_energy = energy * rho * vol
            ohc_data[t,z] = cell_energy

    cube = thetao_cube.copy()

#   Paul:
#   So to calculate H (heat content, J m-2), you need rho (density, kg m^3)
#   and cp (specific heat capacity of seawater, J) in addition to vanilla CMIPx output.
#
#   So the ingredients to a good OHC are:
#   -   Salinity (so)
#   -   Temperature (thetao)
#   -   Pressure (depth, from axis)
#   -   Specific heat capacity of cell (TEOS-10 library: S, T, P inputs)
#   -   Density of cell (TEOS-10 library; S, T, P inputs
#   H = rho x cp x grid cell temperature. 

#   In practice:
#   conserv_temp = gsw.conversions.CT_from_t(so_cube.data, thetao_cube.data, pressure.data) 
#   pressure = pressure = gsw.conversions.p_from_z(depth, latitude) # dbar
#   abs_sal = gsw.conversions.SA_from_SP(so, p, lon, lat)
#   rho = gsw.density.rho(so_cube.data, conserv_temp.data, pressure.data) #  kg/ m
#   energy = gsw.energy.internal_energy(so_cube.data, conserv_temp.data, pressure.data)
#   or?
#   energy =  gsw.energy.enthalpy(so_cube.data, conserv_temp.data, pressure.data) J/kg
#   total energy per cubic meter = density * energy (internal or enthalpy) J/cell
#   then intergrate it over several axes.

    cube = thetao_cube.copy()
    cube.data = ohc_data 
    cube.units = cf_units.Unit('J')
    cube.name = 'Ocean heat Content ' + trend
    cube.short_name = 'ohc'
    cube.var_name = 'ohc'
    iris.save(cube, output_ohc_fn)

    for t in [0, -1]:
        single_pane_map_plot(
                cfg,
                metadatas[thetao_fn],
                cube[t, 0],
                key='OHC_full_'+trend
                )
    return output_ohc_fn 

def mpi_detrend(iter_pack, cubedata, decimal_time, slopes, intercepts):
    index, _ = iter_pack
    data = cubedata[:, index[0], index[1], index[2]]
    if np.ma.is_masked(data.max()):
        return [], index

    line = [(t * slopes[index]) + intercepts[index] for t in np.arange(len(decimal_time))]
    return index, np.array(line)


def detrend_from_PI(cfg, metadatas, filename, trend_shelve):
    """
    Use the shelve calculoated in calc_pi_trend to detrend 
    the historical data.
    """
    exp = metadatas[filename]['exp']
    short_name = metadatas[filename]['short_name']
    dataset = metadatas[filename]['dataset']
    ensemble = metadatas[filename]['ensemble']
    project = metadatas[filename]['project']

    work_dir = diagtools.folder([cfg['work_dir'], 'detrended'])
    output_fn = work_dir + '_'.join([project, dataset, exp, ensemble, short_name, 'detrended'])+'.nc'

    if os.path.exists(output_fn):
        print('Detrended already exists:', output_fn)
        cube = iris.load_cube(filename)
        detrended = iris.load_cube(output_fn) 
        make_difference_plots(
            cfg,
            metadatas[filename],
            detrended,
            cube,
            )
        return output_fn
    print ('loading from', trend_shelve)

    if not glob(trend_shelve+'*'):
        print('Trend shelve doesn\'t exist', trend_shelve)
        assert 0

    sh = shopen(trend_shelve)
    slopes = sh['slopes']
    intercepts = sh['intercepts']
    sh.close()

    cube = iris.load_cube(filename)
    for t in [0, -1]:
        single_pane_map_plot(
            cfg,
            metadatas[filename],
            cube[t,0],
            key='trend_intact'
            )

    decimal_time = diagtools.cube_time_to_float(cube)
    dummy = cube.data.copy()
    dummy = np.ma.masked_where(dummy>10E10, dummy)
    count = 0


    parrallel = True
    if not parrallel:    
        for index, arr in np.ndenumerate(dummy[0]): 
            if np.ma.is_masked(arr): continue
            data = cube.data[:, index[0], index[1], index[2]]
            if np.ma.is_masked(data.max()): continue

            if not count%250000:
                print(count, index, 'detrending')

            line = [(t * slopes[index]) + intercepts[index] for t in np.arange(len(decimal_time))]
            dummy[:, index[0], index[1], index[2]] = np.array(line)
            count+=1
    else:
        # parrlel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            print('ProcessPoolExecutor: executing detrending mpi')
            # iter_pack, cubedata, decimal_time, slopes, intercepts
            ndenum = np.ndenumerate(dummy)

            for dtline, index in executor.map(mpi_detrend, 
                                              ndenum,
                                              itertools.repeat(cube.data),
                                              itertools.repeat(decimal_time),
                                              itertools.repeat(slopes),
                                              itertools.repeat(intercepts),
                                              chunksize=100000):
                if dtline:
                    if count%250000 == 0:
                        print(count, 'detrend')
                    dummy[:, index[0], index[1], index[2]] = dtline
                    count+=1


    detrended = cube.copy()
    detrended.data = detrended.data - np.ma.array(dummy)
    iris.save(detrended, output_fn)

    for t in [0, -1]:
        single_pane_map_plot(
                cfg,
                metadatas[filename],
                detrended[t,0],
                key='detrended'
                )

    make_difference_plots(
        cfg,
        metadatas[filename],
        detrended,
        cube,
        )

    return output_fn

def guess_PI_ensemble(dicts, keys, ens_pos = None):
    """
    Take a punt as the pi ensemble member.
    """
    keys.append('piControl')
    for index, value in dicts.items():
        intersection = set(index) & set(keys)
        
        if len(intersection) == len(keys):
            print("guess_PI_ensemble: full match", index, keys, index[ens_pos] )
            return index[ens_pos]
    print('Did Not find pi control ensemble:', keys, ens_pos)
    assert 0


def mpi_fit(iter_pack, cubedata, time_itr, tmin): #, mpi_data):
            index, _ = iter_pack
            data = cubedata[:, index[0], index[1], index[2]]
            if np.ma.is_masked(data.max()):
                return [], index
            data = data - np.ma.mean(data[tmin-1:tmin+2])
            linreg = linregress(time_itr, data)
            return linreg, index



def calc_pi_trend(cfg, metadatas, filename, method='linear regression', overwrite=False):
    """
    Calculate the trend in the 
    """
    exp = metadatas[filename]['exp']
    short_name = metadatas[filename]['short_name']
    dataset = metadatas[filename]['dataset']
    ensemble = metadatas[filename]['ensemble']
    project = metadatas[filename]['project']

    work_dir = diagtools.folder([cfg['work_dir'], 'pi_trend'])
    output_fn = work_dir + '_'.join([project, dataset, exp, ensemble, short_name, 'pitrend'])+'.nc'
    output_shelve = output_fn.replace('.nc', '.shelve')

    # Check if overwriting the file
    if overwrite and os.path.exists(output_fn):
        os.remove(output_fn)

    # Check if it already exists:
    if os.path.exists(output_fn):
        return output_shelve

    cube = iris.load_cube(filename)
    decimal_time = diagtools.cube_time_to_float(cube)
   
    if method != 'linear regression':
        assert 0

    if glob(output_shelve+'*'):
        print ('loading from', output_shelve)
        sh = shopen(output_shelve)
        slopes = sh.get('slopes', {})
        intercepts = sh.get('intercepts', {})
        count = sh.get('count', 0)
        sh.close()
    else:
        print('Starting picontrol calculation from fresh')
        slopes = {} 
        intercepts = {}
        count = 0

    for t in [0, -1]:
        single_pane_map_plot(
            cfg, 
            metadatas[filename], 
            cube[t, 0], 
            key ='piControl')

    dummy = cube[0].data
    #dummy = np.ma.masked_where(dummy>10E10, dummy)
    if count == len(dummy.compressed()):
        return output_shelve
    times = cube.coord('time')
    pi_year = times.units.num2date(times.points)[0].year + 11. # (1971 equivalent)

    parrallel_calc = True
    if not parrallel_calc:
        decimal_aranged = np.arange(len(decimal_time))

        print('Calculating linear regression for:', [project, dataset, exp, ensemble, short_name, ])
        for index, arr in np.ndenumerate(dummy): #[~cube.data[0].mask]):
            if np.ma.is_masked(arr): continue
            if slopes.get(index, False): continue

            # Load time series for individual point
            data = cube.data[:, index[0], index[1], index[2]]
            if np.ma.is_masked(data.max()): continue

            # Zero PI control around year 11 (1971 in hist)
            data = zero_around_dat(decimal_time, data, year=pi_year)

            # Calculate Linear Regression
            linreg = linregress( decimal_aranged, data)
            if linreg.slope > 1E10 or linreg.intercept>1E10:
                print('linear regression failed:', linreg)
                print('slope:', linreg.slope,'intercept', linreg.intercept)
                print('from time:', np.arange(len(decimal_time)))
                print('from data:', cube.data[:, index[0], index[1], index[2]])
                assert 0

            # Store results of Linear Regression
            slopes[index] = linreg.slope
            intercepts[index] = linreg.intercept
            count+=1
            if count%250000 == 0:
                print('Saving shelve: ', count, index, linreg.slope, linreg.intercept)
                sh = shopen(output_shelve)
                sh['slopes'] = slopes
                sh['intercepts'] = intercepts
                sh['count'] = count
                sh.close()

        if count == 0: 
            print('linear regression failed', count)
            assert 0
    else:
        print('Performing parrallel calculation:')
        time_arange = np.arange(len(times.points))
        ndenum = np.ndenumerate(dummy)

        tmin = np.argmin(np.abs(np.array(decimal_time) - pi_year))
        print('ProcessPoolExecutor: starting')
#        executor = ProcessPoolExecutor(max_workers=1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            print('ProcessPoolExecutor: executing')
            for linreg, index in executor.map(mpi_fit,
                                              ndenum,
                                              itertools.repeat(cube.data),
                                              itertools.repeat(time_arange),
                                              itertools.repeat(tmin), 
                                              chunksize=100000):
#                                              itertools.repeat(y_dat)):
#                                             chunksize=10000):
#               print(linreg, index, count)
                if linreg:
                    if count%250000 == 0:
                        print(count, 'linreg:', linreg[0],linreg[1])
                    slopes[index] = linreg.slope
                    intercepts[index] = linreg.intercept
                    count+=1

    print('Saving final shelve: ', count, index )#linreg.slope, linreg.intercept)
    sh = shopen(output_shelve)
    sh['slopes'] = slopes
    sh['intercepts'] = intercepts
    sh['count'] = count
    sh.close()

    plot_histo = True
    if plot_histo:	 
        fig = plt.figure()
        fig.add_subplot(211)
        plt.hist(list(slopes.values()), bins=21, color='red', )
        plt.title('Slopes')

        fig.add_subplot(212)
        plt.hist(list(intercepts.values()), bins=21, color='blue')
        plt.title('Intercepts')

        path = diagtools.folder([cfg['plot_dir'], 'pi_trend'])
        path += '_'.join([project, dataset, exp, ensemble, short_name, 'pitrend'])+'.png'
        print('Saving figure:', path)
        plt.savefig(path)
        plt.close()

    # Create NetCDF for slopes
    print('Creating netcdf for slopes:', output_fn)
    nc = netCDF4.Dataset(output_fn, mode='w')
    nc.dataset = dataset
    nc.exp = exp
    nc.ensemble = ensemble
    nc.project = project

    toexclude = ['thetao', 'volcello', 'so', 'areacello']
    src = netCDF4.Dataset(filename, 'r')
    # copy global attributes all at once via dictionary
    # dst.setncatts(src.__dict__)
    # copy dimensions
    for name, dimension in src.dimensions.items():
        print('copying dimension', name)
        nc.createDimension(
            name, (len(dimension) if not dimension.isunlimited() else None))

    var_dims = (u'lev', u'lat', u'lon') # default dims
    for name, variable in src.variables.items():
        if name in toexclude:
            var_dims = variable.dimensions[1:]
        if name not in toexclude:
            print('copying variables', name)
            x = nc.createVariable(name, variable.datatype, variable.dimensions)
            nc[name][:] = src[name][:]
            # copy variable attributes all at once via dictionary
            nc[name].setncatts(src[name].__dict__)

    # variables:
    # only works if regridded:
    nc_slopes = nc.createVariable('slopes', np.float64, var_dims)
    nc_slopes.units = ''
    nc_intercepts = nc.createVariable('intercepts', np.float64, var_dims)
    nc_intercepts.units = ''

    # Slopes:
    slopes_arr = np.zeros_like(dummy.data)
    for index, slope in slopes.items():
        slopes_arr[index[0],index[1],index[2]] = slope
    slopes_arr = np.ma.masked_where(slopes_arr==0.,slopes_arr)
    nc_slopes[:,:,:] = slopes_arr

    # intercepts:
    intercepts_arr = np.zeros_like(dummy.data)
    for index, intercept in intercepts.items():
        intercepts_arr[index[0],index[1],index[2]] = intercept
    intercepts_arr = np.ma.masked_where(intercepts_arr==0.,intercepts_arr)
    nc_intercepts[:,:,:] = intercepts_arr
    nc.close()

    cubes = iris.load_raw(output_fn)
    single_pane_map_plot(
        cfg,
        metadatas[filename],
        cubes.extract(iris.Constraint(name='slopes'))[0][0],
        key ='slope')
    single_pane_map_plot(
        cfg,
        metadatas[filename],
        cubes.extract(iris.Constraint(name='intercepts'))[0][0],
        key ='intercept')

    return output_shelve 
#   return output_fn


def main(cfg):
    """
    Load the config file and some metadata, then pass them the plot making
    tools.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.

    """
    detrending_method = 'Full'
    metadatas = diagtools.get_input_files(cfg)
    projects = {}
    datasets = {}
    ensembles = {}
    experiments = {}
    variable_group = {}
    short_names = {}
    file_dict = {}

    trend_shelves = {}
    detrended_ncs = {}
    ocean_heat_content = {}
    specvol_anomalies = {}
    ocean_heat_content_timeseries = {}

    print('\n\n\ncreation loop')
    for filename in sorted(metadatas):
        print('creation loop', filename,':', metadatas[filename])
        exp = metadatas[filename]['exp'] 
        variable_group = metadatas[filename]['variable_group']
        short_name = metadatas[filename]['short_name']
        dataset = metadatas[filename]['dataset']
        ensemble = metadatas[filename]['ensemble']
        project = metadatas[filename]['project']
        print((project, dataset, exp, ensemble, short_name))
        file_dict[(project, dataset, exp, ensemble, short_name)] = filename
   
    print('\nCalculating trend')
    # Calculated trend.
    for (project, dataset, exp, ensemble, short_name), filename in file_dict.items():
        print('iterating', project, dataset, exp, ensemble, short_name, filename)
        if exp != 'piControl': 
            continue
        if short_name in ['volcello', 'areacello']: 
            continue
        trend_shelves[(project, dataset, exp, ensemble, short_name)] = calc_pi_trend(cfg, metadatas, filename)

    print('\nDetrend from PI.')
    for (project, dataset, exp, ensemble, short_name), filename in file_dict.items():
        print('detrending:', project, dataset, exp, ensemble, short_name, filename)
        if short_name in ['volcello', 'areacello']:
              continue
        pi_ensemble = guess_PI_ensemble(trend_shelves, [project, dataset, short_name], ens_pos = 3)
        trend_shelve = trend_shelves[(project, dataset, 'piControl', pi_ensemble, short_name)]
        detrended_fn = detrend_from_PI(cfg, metadatas, filename, trend_shelve)
        detrended_ncs[(project, dataset, exp, ensemble, short_name)] = detrended_fn
        metadatas[detrended_fn] = metadatas[filename].copy()


    print('\n-------------\nCalculate Sea Level Rise')

    for (project, dataset, exp, ensemble, short_name), detrended_fn in detrended_ncs.items():
        if short_name != 'thetao':
            continue
        if exp ==  'piControl': # no need to calculate this.
            continue
        hist_thetao_fn = detrended_fn
        hist_so_fn =  detrended_ncs[(project, dataset, exp, ensemble, 'so')]
        pi_ensemble = guess_PI_ensemble(detrended_ncs, [project, dataset, 'thetao',], ens_pos = 3)
        pi_thetao_fn =  detrended_ncs[(project, dataset, 'piControl', pi_ensemble, 'thetao')]
        pi_so_fn = detrended_ncs[(project, dataset, 'piControl', pi_ensemble, 'so')]

        slr_total_fn, slr_thermo_fn, slr_halo_fn = calc_slr_full(cfg,
            metadatas,
            hist_thetao_fn,
            hist_so_fn,
            pi_thetao_fn, 
            pi_so_fn,
            trend='detrended'
            )
        print(slr_total_fn, slr_thermo_fn, slr_halo_fn)
    assert 0
          
    print('\nCalculate ocean heat content - trend intact')
    for (project, dataset, exp, ensemble, short_name), fn in file_dict.items():
        if short_name != 'thetao': 
            continue
        #if exp ==  'piControl': # no need to calculate this.  
        #    continue
        # volcello is in same ensemble, same exp.
        if (project, dataset, exp, ensemble, 'volcello') in file_dict:
            volcello_fn = file_dict[(project, dataset, exp, ensemble, 'volcello')]
        else:
            print('ocean heat content calculation: no volcello', (project, dataset, exp, ensemble, short_name))
            assert 0

        for index in file_dict.keys():
            if project not in index: continue
            if dataset not in index: continue
            if exp not in index: continue
            if ensemble not in index: continue
            print('trend intact calculation:', dataset, ':', index)

        if detrending_method == 'Basic':
            ohc_fn = calc_ohc_basic(cfg, metadatas, fn, volcello_fn, trend='intact')
#            specvol_fn = ''
        elif detrending_method == 'Full':
            so_fn = file_dict[(project, dataset, exp, ensemble, 'so')]
            ohc_fn = calc_ohc_full(cfg, metadatas, fn, so_fn, volcello_fn, trend='intact')

        if ohc_fn.find(exp) == -1:
            print('ERROR - ohc_fn',(project, dataset, exp, ensemble, short_name), ohc_fn )
            assert 0

        ocean_heat_content[(project, dataset, exp, ensemble, 'ohc','intact')] = ohc_fn
#        specvol_anomalies[(project, dataset, exp, ensemble, 'specvol_anom','intact')] = specvol_fn

        metadatas[ohc_fn] = metadatas[fn].copy()
#        metadatas[specvol_fn] = metadatas[fn].copy()

    print('\nCalculate ocean heat content - detrended')
    for (project, dataset, exp, ensemble, short_name), detrended_fn in detrended_ncs.items():
        if short_name != 'thetao':
            continue
        #if exp ==  'piControl': # no need to calculate this.
        #    continue

        # volcello is in same ensemble, same exp.
        if (project, dataset, exp, ensemble, 'volcello') in file_dict:
            volcello_fn = file_dict[(project, dataset, exp, ensemble, 'volcello')]
        else:
            print('ocean heat content calculation: no volcello', (project, dataset, exp, ensemble, short_name))
            assert 0 

        for index in detrended_ncs.keys():
            if dataset not in index: continue
            if exp not in index: continue
            if ensemble not in index: continue
            print(dataset, ':', index)

        if detrending_method == 'Basic':
            ohc_fn = calc_ohc_basic(cfg, metadatas, detrended_fn, volcello_fn, trend='detrended')
        elif detrending_method == 'Full':
            print('detrending_method:', detrending_method, project, dataset, exp, ensemble)
            so_fn = detrended_ncs[(project, dataset, exp, ensemble, 'so')]
            print('detrending_method:', detrending_method, so_fn)
            ohc_fn = calc_ohc_full(cfg, metadatas, detrended_fn, so_fn, volcello_fn, trend='detrended')
        else: 
            assert 0

        if ohc_fn.find(exp) == -1:
            print('\n\nERROR - ohc_fn',(project, dataset, exp, ensemble, short_name), ohc_fn )
            print('detrended_fn', detrended_fn)
            print('so_fn', so_fn)
            print('volcello_fn', volcello_fn)
            print('metadatas:',metadatas[detrended_fn],'\n\n')
            assert 0

        ocean_heat_content[(project, dataset, exp, ensemble, 'ohc', 'detrended')] = ohc_fn 
#        specvol_anomalies[(project, dataset, exp, ensemble, 'specvol_anom','detrended')] = specvol_fn

        metadatas[ohc_fn] = metadatas[detrended_fn].copy()
#        metadatas[specvol_fn] = metadatas[detrended_fn].copy()


    print('\n---------------------\nCalculate OHC time series')
    depth_ranges = ['total', '0-700m', '700-2000m', '0-2000m', '2000m_plus']
    for depth_range in depth_ranges:
        for (project, dataset, exp, ensemble, short_name, trend), ohc_fn in ocean_heat_content.items():
            ohc_ts_fn = calc_ohc_ts(cfg, metadatas, ohc_fn, depth_range, trend)
            ocean_heat_content_timeseries[(project, dataset, exp, ensemble, short_name, trend, depth_range)] = ohc_ts_fn
            print('OHC:', (project, dataset, exp, ensemble, short_name, trend, depth_range))
            if ohc_fn.find(exp) == -1:
                print('ERROR - ohc_fn',project, dataset, exp, ensemble, short_name, trend, ':', ohc_fn)
                assert 0
            if ohc_ts_fn.find(exp) == -1: 
                print('ERROR - ohc_ts_fn',project, dataset, exp, ensemble, short_name, trend, ':', ohc_ts_fn)
                assert 0
            metadatas[ohc_ts_fn] = metadatas[ohc_fn]

    projects = {index[0]:True for index in ocean_heat_content_timeseries.keys()}
    datasets = {index[1]:True for index in ocean_heat_content_timeseries.keys()}
    exps = {index[2]:True for index in ocean_heat_content_timeseries.keys()}
    ensembles = {index[3]:True for index in ocean_heat_content_timeseries.keys()}

    def print_dict(dic,name=''):
        print('\n---------------\nprinting', name)
        for key in sorted(dic.keys()): 
            print(name, key, dic[key])

    # OHC detrending TS:
    print('\n----------------------\nplotting detrending figure. ')
    for dataset, ensemble, project, depth_range  in itertools.product(datasets.keys(), ensembles.keys(), projects.keys(), depth_ranges):   
        for index in ocean_heat_content_timeseries.keys():
            if dataset not in index: continue
            if depth_range not in index: continue
            #if ensemble not in index: continue
            if project not in index: continue
            print('detrending diagrma:', dataset, depth_range, ':', index)
        
        detrended_hist = ocean_heat_content_timeseries.get((project, dataset, 'historical', ensemble, 'ohc', 'detrended', depth_range), False)
        if detrended_hist == False:
            continue

        pi_ensemble = guess_PI_ensemble(ocean_heat_content_timeseries, [project, dataset, 'ohc', 'detrended'], ens_pos = 3)

        trend_intact_hist = ocean_heat_content_timeseries[(project, dataset, 'historical', ensemble, 'ohc', 'intact', depth_range)]
        detrended_piC = ocean_heat_content_timeseries[(project, dataset, 'piControl', pi_ensemble, 'ohc', 'detrended', depth_range)]
        trend_intact_piC = ocean_heat_content_timeseries.get((project, dataset, 'piControl', pi_ensemble, 'ohc', 'intact', depth_range), '')
        if detrended_hist.find('hist')==-1:
            print('Wrong: detrended_hist', detrended_hist)
            print_dict(ocean_heat_content_timeseries, 'ocean_heat_content_timeseries')
            assert 0 
        if trend_intact_hist.find('hist')==-1:
            assert 0
        if detrended_piC.find('piControl')==-1:
            assert 0
        detrending_fig(cfg, metadatas, detrended_hist, trend_intact_hist, detrended_piC, trend_intact_piC, depth_range, 'OHC')

    # Multi model time series plotx for each time series
    # Figure based on 2.25.
    # 
    for dataset, ensemble, project, exp in itertools.product(datasets.keys(), ensembles.keys(), projects.keys(), exps.keys()):
        try: 
            ocean_heat_content_timeseries[(project, dataset, exp, ensemble, 'ohc', 'detrended', 'total')]
        except: continue
        fig_like_2_25(cfg, metadatas, ocean_heat_content_timeseries, dataset, ensemble, project, exp)
    
    multimodel_2_25(cfg, metadatas, ocean_heat_content_timeseries)

    logger.info('Success')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
