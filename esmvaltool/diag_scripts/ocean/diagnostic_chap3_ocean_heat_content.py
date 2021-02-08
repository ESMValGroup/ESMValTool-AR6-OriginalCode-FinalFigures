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
plt.style.use("./ipcc_ar6_fgd.mplstyle")
import matplotlib

import numpy as np
import itertools
import cf_units
import datetime
from scipy.stats import linregress
from scipy.io import loadmat
import netCDF4
from collections import Counter


#from concurrent.futures import ProcessPoolExecutor
import concurrent.futures


from glob import glob
from dask import array as da
from shelve import open as shopen

import csv

from matplotlib.colors import LogNorm
from matplotlib.offsetbox import AnchoredText
import cartopy.crs as ccrs

from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import run_diagnostic

from esmvalcore.preprocessor._time import extract_time
from esmvalcore.preprocessor._regrid import regrid, extract_levels
from esmvalcore.preprocessor._area import extract_shape
from esmvalcore.preprocessor._volume import volume_statistics

try:
    import gsw
except:
    print('Unable to load gsw.\n You need to install it in your conda environmetn with:\npip install gsw')

#         box_colours =  {'GHG': [178,178,178], 'NAT':[0,79,0], 'AER':[0,52,102], 'HIST':[196,121,0]}

CMIP5_blue = '#2551cc'
CMIP6_red = '#cc2323'
histnat_green= '#004F00' # 0,79,0
historical_beige = '#c47900' #

model_type = {
    'EOS80': [], # The default
    'TEOS-10': ['GFDL', 'ACCESS-CM2', 'ACCESS-ESM1-5', ],
    'in situ': [], # Hopefully none!
    'potential temperature': [],
    }


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))

# def add_aux_times(cube):
#     """
#     Check for presence and add aux times if absent.
#     """
#     coord_names = [coord[0].long_name for coord in cube.coords() ]
#
#     if 'day_of_month' not in coord_names:
#         iris.coord_categorisation.add_day_of_month(cube, 'time')
#
#     if 'month_number' not in coord_names:
#         iris.coord_categorisation.add_month_number(cube, 'time')
#
#     if 'day_of_year' not in coord_names:
#         iris.coord_categorisation.add_day_of_year(cube, 'time')
#     return cube


# def detrend(cfg, metadata, cube, pi_cube, method = 'linear regression'):
#     """
#     Detrend the historical cube using the pi_control cube.
#     """
#     if cube.data.shape != pi_cube.data.shape:
#         print(cube.data.shape, pi_cube.data.shape)
#         assert 0
#
#     decimal_time = diagtools.cube_time_to_float(cube)
#     if method == 'linear regression':
#         linreg = linregress( np.arange(len(decimal_time)), pi_cube.data)
#         line = [ (t * linreg.slope) + linreg.intercept for t in np.arange(len(decimal_time))]
#
#     fig = plt.figure()
#     plt.plot(decimal_time, cube.data, label = 'historical')
#     plt.plot(decimal_time, pi_cube.data, label = 'PI control')
#     plt.plot(decimal_time, line, label = 'PI control '+method)
#
#     detrended = cube.data - np.array(line)
#     cube.data = detrended
#     plt.plot(decimal_time, detrended, label = 'Detrended historical')
#
#     plt.axhline(0., c = 'k', ls=':' )
#     plt.legend()
#     dataset = metadata['dataset']
#     plt.title(dataset +' detrending ('+method+')')
#
#     image_extention = diagtools.get_image_format(cfg)
#     path = diagtools.folder(cfg['plot_dir']) + 'detrending_' + dataset + image_extention
#     logger.info('Saving detrending plots to %s', path)
#     plt.savefig(path)
#     plt.close()
#     return cube


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

def regrid_to_1x1(cube, scheme = 'linear'):
    """
    regrid a cube to a common 1x1 grid.
    """
    # regrid to a common grid:
    return regrid(cube, '1x1', scheme)






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


def make_mean_of_cube_list_notime(cube_list):
    """
    Takes the mean of a list of cubes (not an iris.cube.CubeList).

    Assumes all the cubes are the same shape.
    """
    # Fix empty times
    cube_mean=cube_list[0]
    #try: iris.coord_categorisation.add_year(cube_mean, 'time')
    #except: pass
    #try: iris.coord_categorisation.add_month(cube_mean, 'time')
    #except: pass

    try: cube_mean.remove_coord('year')
    except: pass
    #cube.remove_coord('Year')
    try: model_name = cube_mean.metadata[4]['source_id']
    except: model_name = ''

    #cube_mean = fix_depth(cube_mean)

    for i, cube in enumerate(cube_list[1:]):
        #try: iris.coord_categorisation.add_year(cube, 'time')
        #except: pass
        #try: iris.coord_categorisation.add_month(cube, 'time')
        #except: pass
        #cube = fix_depth(cube)

        try: cube.remove_coord('year')
        except: pass
        #cube.remove_coord('Year')
        try: model_name = cube_mean.metadata[4]['source_id']
        except: model_name = ''
        print(i, model_name)
        cube_mean+=cube
        #print(cube_mean.coord('time'), cube.coord('time'))
    cube_mean = cube_mean/ float(len(cube_list))
    return cube_mean




#####
# Above here is old code.
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


def zero_around(cube, year_initial=1971., year_final=1971.):
    """
    Zero around the time range provided.

    """
    new_cube = extract_time(cube, year_initial, 1, 1, year_final, 12, 31)
    mean = new_cube.data.mean()
    cube.data = cube.data - mean
    return cube


def zero_around_dat(times, data, year):
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
     data = zero_around_dat(times, cube.data, 1971.)
     return times, data


def single_timeseries(fn, path, keys):
    #times, data = load_convert(fn)
    fig = plt.figure()
    ax = plt.subplot(111)
    cube = iris.load_cube(fn)
    times = diagtools.cube_time_to_float(cube)
    data = cube.data
    ax.plot(times, data,) #color)
    plt.title(' '.join(keys))
    print('Saving single_timeseries:', path)
    plt.savefig(path)
    plt.close()

#def single_pane(fig, ax, fn, color='red', xlim=False, no_ticks=False):
#    times, data = load_convert(fn)
#    ax.plot(times, data, c=color)
#    if xlim:
#        ax.set_xlim(xlim)
#    if no_ticks:
#        ax.set_xticklabels([])
#    return fig, ax


def multimodel_2_25(cfg, metadatas, ocean_heat_content_timeseries,
        plot_style='viridis',
        plot_type='7_panes',
        show_UKESM=False):
    """
    Multimodel version of the 2.25 plot.
    produced when do_OHC is true

    """
    depth_ranges = ['total', '0-700m', '700-2000m', '0-2000m', '2000m_plus']

    projects = list({index[0]:True for index in ocean_heat_content_timeseries.keys()}.keys())
    datasets = list({index[1]:True for index in ocean_heat_content_timeseries.keys()}.keys())
    ensembles = list({index[3]:True for index in ocean_heat_content_timeseries.keys()}.keys())

    datasets = sorted(datasets)
    #color_dict = {da:c for da, c in zip(datasets, ['r' ,'b'])}

    if plot_style=='viridis':
        color_dict = {dataset:c for dataset, c in zip(datasets, plt.cm.viridis(np.linspace(0,1,len(datasets))))}
        if show_UKESM:
            for dataset in color_dict.keys():
                if dataset.lower().find('ukesm')>-1:
                    color_dict[dataset] = 'purple'

    if plot_style=='all_one':
        color_dict = {dataset: CMIP6_red for dataset in datasets}

        if show_UKESM:
            for dataset in color_dict.keys():
                if dataset.lower().find('ukesm')>-1:
                    color_dict[dataset] = 'purple'

    if plot_style in ['mono', '5-95']:
        color_dict={}
    color_dict['Observations'] = 'black'

    # ocean_heat_content_timeseries keys:
    # (project, dataset, 'piControl', pi_ensemble, 'ohc', 'intact', depth_range)
    plot_details={}
    axes= {}
    axes_text_y={}
    fig = plt.figure()

    if  plot_type=='7_panes':
        LHS_xlim = [1860, 2017]
        RHS_xlim = [1960, 2017]

        depth_dict = { 321: 'total',
                   322: '0-2000m',
                   323: '0-700m',
                   324: '0-700m',
                   326:  '700-2000m',
                   (6, 2, 9):  '700-2000m',
                   (6, 2, 11): '2000m_plus'}
        xlims_dict = { 321: LHS_xlim,
                   322: RHS_xlim,
                   323: LHS_xlim,
                   324: RHS_xlim,
                   326: RHS_xlim,
                   (6, 2, 9):  LHS_xlim,
                   (6, 2, 11): LHS_xlim,}
        no_ticks= {321: True,
               322: True,
               323: True,
               324: True,
               326: False,
               (6, 2, 9):  True,
               (6, 2, 11): False}
        axes_texts = {
            321: 'Full-depth',
            322: '0-2000m',
            323: '0-700m',
            324: '0-700m',
            326: '700m - 2000m',
            (6, 2, 9): '700m - 2000m',
            (6, 2, 11): '> 2000m',
        }

        fig.set_size_inches(10, 7)

    if  plot_type=='4_panes':
        LHS_xlim = [1860, 2019]
        depth_dict = { 411: 'total',
                   412: '0-700m',
                   413: '700-2000m',
                   414: '2000m_plus'  }
        xlims_dict = { 411: LHS_xlim,
                   412: LHS_xlim,
                   413: LHS_xlim,
                   414: LHS_xlim }
        no_ticks= {411: True,
               412: True,
               413: True,
               414: False,}
        axes_texts ={411: 'Full-depth',
               412: '0-700m',
               413: '700m - 2000m',
               414:  '> 2000m',}
        fig.set_size_inches(6 , 5)

    if  plot_type=='large_full':
        LHS_xlim = [1860, 2019]
        depth_dict = { 'full': 'total',
                   '0-700': '0-700m',
                   '7-20': '700-2000m',
                   '2plus': '2000m_plus'  }
        xlims_dict = { 'full': LHS_xlim,
                   '0-700': LHS_xlim,
                   '7-20': LHS_xlim,
                   '2plus': LHS_xlim }
        no_ticks= {'full': False,
               '0-700': True,
               '7-20': True,
               '2plus': False,}
        axes_texts ={'full': 'Full depth',
               '0-700': '0m - 700m',
               '7-20': '700m - 2000m',
               '2plus':  '> 2000m',}
        axes_text_y['full'] = 0.95

        fig.set_size_inches(8 , 5)
        gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1.5, 1.], wspace=0.16)
        gs1 = gs[1].subgridspec(3, 1, hspace=0.06 ) # maps
        #scatters
        axes['full'] = fig.add_subplot(gs[0,0]) # LHS
        axes['0-700'] = fig.add_subplot(gs1[0]) # RHS top
        axes['7-20'] = fig.add_subplot(gs1[1]) # LHS
        axes['2plus'] = fig.add_subplot(gs1[2]) # LHS



    for subplot in depth_dict.keys():
        if isinstance(subplot, int):
            axes[subplot] =  plt.subplot(subplot)
        elif isinstance(subplot, tuple):
            axes[subplot] =  plt.subplot(subplot[0], subplot[1], subplot[2])
#        if subplot in [413, 323]:
#            plt.ylabel('OHC (ZJ)', fontsize=16)
    # Y axis label
    fig.text(0.03, 0.5, 'OHC (ZJ)', va='center', rotation='vertical')
    # fig.text(0.5, 0.04, , ha='center')

    if plot_style=='mono': # just plot between the most extremem values.
        fill_betweens = {subplot:{} for subplot in axes.keys()}

        for project, dataset, ensemble in itertools.product(projects, datasets, ensembles):
            total_key = (project, dataset, 'historical', ensemble, 'ohc', 'detrended', 'total')
            fn = ocean_heat_content_timeseries.get(total_key, False)
            if not fn:
                continue

            for subplot, ax in axes.items():
                key =  (project, dataset, 'historical', ensemble, 'ohc', 'detrended', depth_dict[subplot])
                print(subplot, depth_dict[subplot])
                fn = ocean_heat_content_timeseries[key]

                times, data = load_convert(fn)
                if '2000m_plus' ==  depth_dict[subplot]:
                    print(key, times, data)
                for t, d in zip(times, data):
                    t = int(t) + 0.5
                    if np.ma.is_masked(d): continue
                    if np.isnan(d): continue
                    if np.isinf(d): continue

                    try: fill_betweens[subplot][t].append(d)
                    except: fill_betweens[subplot][t]= [d, ]
                    if show_UKESM and dataset.lower().find('ukesm')>-1:
                        axes[subplot].plot(times, data, c='purple', alpha=1.0, lw=1.5, zorder=2)
        for subplot in fill_betweens.keys():
            times = sorted(fill_betweens[subplot].keys())
            mins = [np.min(fill_betweens[subplot][t]) for t in times]
            maxs = [np.max(fill_betweens[subplot][t]) for t in times]
            axes[subplot].fill_between(times, mins, maxs, color='grey', alpha=0.5)

    elif plot_style=='5-95': # plot between 5-95 percentiles, weighted such that each model gets an even vote.
        fill_betweens = {subplot:{} for subplot in axes.keys()}
        weights = {subplot:{} for subplot in axes.keys()}

        for project, dataset, ensemble in itertools.product(projects, datasets, ensembles):
            total_key = (project, dataset, 'historical', ensemble, 'ohc', 'detrended', 'total')
            fn = ocean_heat_content_timeseries.get(total_key, False)
            if not fn:
                continue

            for subplot, ax in axes.items():
                key =  (project, dataset, 'historical', ensemble, 'ohc', 'detrended', depth_dict[subplot])
                print(subplot, depth_dict[subplot])
                fn = ocean_heat_content_timeseries[key]

                times, data = load_convert(fn)
                for t, d in zip(times, data):
                    t = int(t) + 0.5
                    if np.ma.is_masked(d): continue
                    if np.isnan(d): continue
                    if np.isinf(d): continue

                    if t in fill_betweens[subplot]:
                        fill_betweens[subplot][t].append(d)
                        weights[subplot][t].append(dataset)
                    else:
                        fill_betweens[subplot][t] = [d, ]
                        weights[subplot][t] = [dataset, ]

                    if show_UKESM and dataset.lower().find('ukesm')>-1:
                        axes[subplot].plot(times, data, c='purple', alpha=0.7, lw=0.7, zorder=2)

        for subplot in fill_betweens.keys():
            times = sorted(fill_betweens[subplot].keys())
            t_weights = [] # list of datasets.
            pc5s = []
            pc95s = []
            pc50s = []
            for t in times:
                counts = Counter(weights[subplot][t])
                for dset in weights[subplot][t]:
                    t_weights.append(1./float(counts[dset]))
                [pc5, pc50, pc95] = diagtools.weighted_quantile(fill_betweens[subplot][t], [0.05, 0.5, 0.95], sample_weight=t_weights)
                pc5s.append(pc5)
                pc50s.append(pc50)
                pc95s.append(pc95)
                print(pc5, pc50, pc95)
            print('5-95:', pc5s, pc95s)
            axes[subplot].fill_between(times, pc5s, pc95s, color=CMIP6_red, alpha=0.35, edgecolor=None)
            axes[subplot].plot(times, pc50s, c=CMIP6_red, lw=1.5, zorder=2)

            if xlims_dict[subplot]:
                axes[subplot].set_xlim(xlims_dict[subplot])
            if no_ticks[subplot]:
                axes[subplot].set_xticklabels([])

    else:
        for project, dataset, ensemble in itertools.product(projects, datasets, ensembles):
            total_key = (project, dataset, 'historical', ensemble, 'ohc', 'detrended', 'total')
            fn = ocean_heat_content_timeseries.get(total_key, False)
            if not fn:
                continue

            for subplot, ax in axes.items():
                key =  (project, dataset, 'historical', ensemble, 'ohc', 'detrended', depth_dict[subplot])
                fn = ocean_heat_content_timeseries[key]

                times, data = load_convert(fn)
                if show_UKESM and dataset.lower().find('ukesm')>-1:
                    ax.plot(times, data, c=color_dict[dataset], alpha=1.0, lw=1.5, zorder=2)
                else:
                    ax.plot(times, data, c=color_dict[dataset], alpha=0.5)
                if xlims_dict[subplot]:
                    ax.set_xlim(xlims_dict[subplot])
                if no_ticks[subplot]:
                    ax.set_xticklabels([])

    for subplot, ax in axes.items():
        ax.axhline(0., c='k', ls=':')

    for axes_key, text in axes_texts.items():
        y_loc = axes_text_y.get(axes_key, 0.9)
        plt.text(0.05, y_loc, text,
             horizontalalignment='left',
             verticalalignment='center',
             transform=axes[axes_key].transAxes)

    plt.suptitle('Global Ocean Heat Content')

    add_all_obs = False
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
                series = zero_around_dat(obs_years, series, 1971)
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
                           lw = 1.5, #0.5,
                           zorder=2,
                           )

    all_obs_sigma = True
    if all_obs_sigma:
        fn = cfg['auxiliary_data_dir'] + '/OHC/210204_0908_DM-AR6FGDAssessmentTimeseriesOHC-v1.csv'
        header = ['Year', 'Central Estimate 0-700m', '0-700m Uncertainty (1-sigma)', 'Central Estimate 700-2000m', '700-2000m Uncertainty (1-sigma)', 'Central Estimate >2000m', '>2000m Uncertainty (1-sigma)', 'Central Estimate Full-depth', 'Full-depth Uncertainty (1-sigma)']

#        central_colums = {'Full-depth':7, '0-700 m':1, '700-2000 m':3, '>2000 m':5}
#        sigma_columns = {'Full-depth':8, '0-700 m':2, '700-2000 m':4, '>2000 m':6}

        central_colums = {'total':7, '0-700m':1, '700-2000m':3, '2000m_plus':5}
        sigma_columns = {'total':8, '0-700m':2, '700-2000m':4, '2000m_plus':6}

        for subplot, depth_key in depth_dict.items():
            times = []
            lower_sigma = []
            centers = []
            upper_sigma = []

            with open(fn, ) as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for r, row in enumerate(reader):
                    if r < 2: continue
                    time = float(row[0])
                    center = float(row[central_colums[depth_key]])
                    sigma = float(row[sigma_columns[depth_key]])

                    times.append(time)
                    lower_sigma.append(center - sigma)
                    centers.append(center)
                    upper_sigma.append(center + sigma)
            print(depth_key, times)

            axes[subplot].plot(np.array(times),
                           np.array(centers),
                           lw = 1.5, #0.5,
                           zorder=2,
                           color='black',  
                           )
            axes[subplot].fill_between(np.array(times), np.array(lower_sigma), np.array(upper_sigma), color='black', alpha=0.35, edgecolor=None)

    legend_fs = 12
    if plot_style in ['all_one', 'mono', '5-95']:leg_size=2
    else:
        leg_size = len(datasets)

    if leg_size <=5:
        axleg = plt.axes([0.05, 0.02, 0.9, 0.10])
    elif leg_size <=10:
        axleg = plt.axes([0.05, 0.02, 0.9, 0.15])
    elif leg_size <=15:
        axleg = plt.axes([0.05, 0.02, 0.9, 0.20])
        Legend_fs = 8
    else:
        axleg = plt.axes([0.05, 0.02, 0.9, 0.25])
        legend_fs = 8

    axleg.axis('off')
    if plot_style in ['mono', ]:
        axleg.plot([], [], c='grey', lw=5, ls='-', alpha=0.5, label='CMIP6')
        axleg.plot([], [], c='black', lw=1, ls='-', label='Observations')
        if show_UKESM:
            axleg.plot([], [], c='purple', lw=1, ls='-', label='UKESM')
    if plot_style in ['5-95', ]:
        axleg.plot([], [], c=CMIP6_red, lw=5, ls='-', alpha=0.85, label='CMIP6')
        if all_obs_sigma:
            axleg.plot([], [], c='black', lw=1.5, marker='s', markerfacecolor=(0.,0.,0.,0.35), markeredgewidth=0., markersize = 11, ls='-', label='Observations '+r'$\pm 1 \sigma$')
        else:
            axleg.plot([], [], c='black', lw=1, ls='-', label='Observations')

        if show_UKESM:
            axleg.plot([], [], c='purple', lw=1, ls='-', label='UKESM')

    if plot_style == 'all_one':
        axleg.plot([], [], c=CMIP6_red, lw=3, ls='-', label='CMIP6')
        axleg.plot([], [], c='black', lw=3, ls='-', label='Observations')
        if show_UKESM:
            axleg.plot([], [], c='purple', lw=1, ls='-', label='UKESM')

    if  plot_type=='large_full':
        axes['full'].set_yticks([-400., -300., -200., -100., 0., 100., 200., 300., 400.])

    if plot_style == 'viridis':
        # Add emply plots to dummy axis.
        for dataset in datasets:
            axleg.plot([], [], c=color_dict[dataset], lw=2, ls='-', label=dataset)

    legd = axleg.legend(
            loc='upper center',
            #loc='lower center',
            ncol=5,
            numpoints=1,
            handlelength=0.85,
            prop={'size': legend_fs},
            bbox_to_anchor=(0.5, 0.5,),
            fontsize=legend_fs)

    legd.draw_frame(False)
    legd.get_frame().set_alpha(0.)
    #plt.tight_layout()

    fig_dir = diagtools.folder([cfg['plot_dir'], 'multimodel_ohc'])
    image_extention = diagtools.get_image_format(cfg)

    if show_UKESM:
        fig_fn = fig_dir + '_'.join(['multimodel_ohc_range', plot_style, plot_type,'UKESM',
                                     ])+image_extention
    else:
        fig_fn = fig_dir + '_'.join(['multimodel_ohc_range', plot_style, plot_type,
                                     ])+image_extention

    plt.savefig(fig_fn)
    print('multimodel_ohc: saving',fig_fn)
    plt.close()


def fig_like_2_25(cfg, metadatas, ocean_heat_content_timeseries, dataset, ensemble, project, exp):
    """
    Produce a 6 pane figure showing the time series.
    """
    depth_ranges = ['total', '0-700m', '700-2000m', '0-2000m', '2000m_plus']

    fig_dir = diagtools.folder([cfg['plot_dir'], 'ohc_summary'])
    image_extention = diagtools.get_image_format(cfg)
    fig_fn = fig_dir + '_'.join([project, exp, dataset, ensemble, 'ohc_summary',
                                 ])+image_extention

    if os.path.exists(fig_fn): return

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

    plt.savefig(fig_fn)
    print('detrending_fig: saving',fig_fn)
    plt.close()



def shift_pi_time(hist_cube, pi_cube):
    times = hist_cube.coord('time')
    units = times.units.name
    calendar = times.units.calendar
    #num2date = times.units.num2date
    parent_branch_yr = netCDF4.num2date(hist_cube.attributes['branch_time_in_parent'],
                                units=hist_cube.attributes['parent_time_units'],
                                calendar=calendar ).year

    child_branch_yr = netCDF4.num2date(hist_cube.attributes['branch_time_in_child'],
                               units=units, calendar=calendar ).year

    diff = child_branch_yr - parent_branch_yr

    pi_dec_times = diagtools.cube_time_to_float(pi_cube)
    pi_dec_times = np.array([t + diff for t in pi_dec_times])
    return pi_dec_times
#   print('date in parent:\t', historical_range, 'is', [h-diff for h in historical_range])

#   dates = num2date(times.points, units, calendar=calendar)
#   file_time_range = [dates[0].year, dates[-1].year]
#   print('date range in file:\t', file_time_range, 'is', [h-diff for h in file_time_range] , 'in parent')



def detrending_fig(cfg,
        metadatas,
        detrended_hist,
        trend_intact_hist,
        detrended_piC,
        trend_intact_piC,
        depth_range,
        key,
        year = 1971,
        draw_zero = True,
        native_time = False,
        ):
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

    fig_dir = diagtools.folder([cfg['plot_dir'], 'detrending_ts', key])
    image_extention = diagtools.get_image_format(cfg)
    if not year: year=''
    fig_fn = fig_dir + '_'.join([project, exp, dataset, ensemble, key, 'detrending_ts', str(year),
                                   depth_range])+image_extention

    if os.path.exists(fig_fn):
        return

    cube_d_h = iris.load_cube(detrended_hist)
    cube_i_h = iris.load_cube(trend_intact_hist)
    cube_d_p = iris.load_cube(detrended_piC)
    if not  skip_intact_piC:
        cube_i_p = iris.load_cube(trend_intact_piC)

    times = {}
    if native_time:
        times['dh'] = diagtools.cube_time_to_float(cube_d_h)
        times['ih'] = diagtools.cube_time_to_float(cube_i_h)
        times['dp'] = shift_pi_time(cube_d_h, cube_d_p)
        times['ip'] = shift_pi_time(cube_i_h, cube_i_p)
    else:

        times['dh'] = diagtools.cube_time_to_float(cube_d_h)
        times['ih'] = times['dh']
        times['dp'] = times['dh']
        times['ip'] = times['dh']

#   print('detrending_fig:', key, cube_d_h.data.max(), cube_i_h.data.max())
#    print('detrending_fig: times', key, times)
    if year:
        d_h_data = zero_around_dat(times['dh'], cube_d_h.data, year=year )
        i_h_data = zero_around_dat(times['ih'], cube_i_h.data, year=year )
        d_p_data = zero_around_dat(times['dp'], cube_d_p.data, year=year )
        if not  skip_intact_piC:
            i_p_data = zero_around_dat(times['ip'], cube_i_p.data, year=year )
    else:
        d_h_data = cube_d_h.data
        i_h_data = cube_i_h.data
        d_p_data = cube_d_p.data
        if not  skip_intact_piC:
            i_p_data = cube_i_p.data

    #print('detrending_fig:', key, d_h_data.max(), i_h_data.max(), i_p_data.max())
    print('detrending_fig: d h:', key, d_h_data.max())
    print('detrending_fig: i h:', key, i_h_data.max())
    print('detrending_fig: d p:', key, d_p_data.max())
    print('detrending_fig: i p:', key, i_p_data.max())

    plt.plot(times['dh'], d_h_data, color = 'red', label = 'Detrended Historical')
    plt.plot(times['ih'], i_h_data, color = 'blue', label = 'Historical')
    plt.plot(times['dp'], d_p_data, color = 'orange', label = 'Detrended PI Control')
    if not  skip_intact_piC:
        plt.plot(times['ip'], i_p_data, color = 'green', label = 'PI Control')

    if draw_zero:
        plt.axhline(0., c = 'k', ls=':' )
    title = ' '.join([key, dataset, exp, ensemble, depth_range])
    plt.title(title)
    plt.legend()

    fig_dir = diagtools.folder([cfg['plot_dir'], 'detrending_ts', key])
    image_extention = diagtools.get_image_format(cfg)
    if not year: year=''
    fig_fn = fig_dir + '_'.join([project, exp, dataset, ensemble, key, 'detrending_ts', str(year),
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
    print('add_map_subplot: ', subplot, title, (cmap, extend, log))
    plt.title(title)
    cmax = cube.data.max()
    if np.ma.is_masked(cmax): return
    if not np.isfinite(cmax): return
    if np.min(nspace) == np.max(nspace): return

    if log:
        qplot = iris.quickplot.contourf(
            cube,
            nspace,
            linewidth=0,
            cmap=plt.cm.get_cmap(cmap),
            norm=LogNorm(),
            zmin=nspace.min(),
            zmax=nspace.max())
        if subplot !=111:
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
        if subplot!=111:
            cbar.set_ticks(
                [nspace.min(), (nspace.max() + nspace.min()) / 2.,
                 nspace.max()])

    try: plt.gca().coastlines()
    except: pass


def single_pane_map_plot(
        cfg,
        metadata,
        cube,
        key='',
        overwrite = False,
        sym_zero = False,
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
    filename = '_'.join(unique_id).replace('/', '_')
    path = diagtools.folder([cfg['plot_dir'], key]) + filename
    path = path.replace(' ', '') + diagtools.get_image_format(cfg)

    if not overwrite and os.path.exists(path):
        return

    if sym_zero:
        cmap=diagtools.misc_div
        max_val = np.max(np.abs([cube.data.min(), cube.data.max()]))
        nspace = np.linspace(
            -max_val, max_val, 22, endpoint=True)

    else:
        cmap='viridis'
        nspace = np.linspace(
            cube.data.min(), cube.data.max(), 30, endpoint=True)
    title = ' '.join(unique_id)
    print('single_pane_map_plot:', unique_id, nspace, [cube.data.min(), cube.data.max()], cube.data.shape)
    add_map_subplot(111, cube, nspace, title=title,cmap=cmap)
    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path, dpi=200)
    plt.close()

def make_difference_plots(
        cfg,
        metadata,
        detrended_cube,
        hist_cube,
        label1='',
        label2='',
        key='',
        ):
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
    exp = metadata['exp']
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
    zrange3 = diagtools.get_cube_range_diff([cube223])
    cube224.data = np.ma.clip(cube224.data, 0.1, 10.)

    print('plotting:', long_name, 'zrange12:',zrange12)
    n_points = 12
    linspace12 = np.linspace(
        zrange12[0], zrange12[1], n_points, endpoint=True)
    linspace3 = np.linspace(
        zrange3[0], zrange3[1], n_points, endpoint=True)
    logspace4 = np.logspace(-1., 1., 12, endpoint=True)

    # Add the sub plots to the figure.
    if label1=='': label1 = 'Detrended '+exp
    if label2=='': label2 = 'Trend intact '+exp
    if key=='': key = 'Detrended'
    add_map_subplot(
        221, cube221, linspace12, cmap='viridis',
        title=label1,
        extend=extend)
    add_map_subplot(
        222, cube222, linspace12, cmap='viridis',
        title=label2,
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
    fn_list = [key, project, dataset, ensemble, exp, short_name, 'quad_maps']
    path = diagtools.folder([cfg['plot_dir'], key+'_quad']) + '_'.join(fn_list)
    path = path.replace(' ', '') + image_extention

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path, dpi=200)
    plt.close()


def SLR_sanity_check(
        cfg,
        metadata,
        steric_fn,
        thermo_fn,
        halo_fn,
        ):
    """
    Make a figure showing four maps, sanity checking the SLR (halo = steric - thermo)
    The four pane image is a latitude vs longitude figures showing:
    * Top left: halosteric
    * Top right:  steric - thermosteric
    * Bottom left: difference
    * Bottom right: quotient
    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.
    metadata: dict
        the input files dictionairy
    """
    short_name = "SLR"
    dataset = metadata['dataset']
    ensemble = metadata['ensemble']
    exp = metadata['exp']
    project = metadata['project']

    long_name = ' '.join([ short_name, dataset, ])

    # Load image format extention
    image_extention = diagtools.get_image_format(cfg)

    fig = plt.figure()
    fig.set_size_inches(9, 6)

    # Create the cubes
    steric = iris.load_cube(steric_fn)
    thermo =  iris.load_cube(thermo_fn)
    halo = iris.load_cube(halo_fn)
    for cube in [steric, thermo, halo]:
        cube.data = np.ma.masked_invalid(cube.data)

    time_str = str(int(diagtools.cube_time_to_float(steric)[-1]))

    cube221 = halo[-1]
    cube222 = steric[-1] - thermo[-1]
    cube223 = cube221 - cube222
    cube224 =  (cube221 - cube222)/(cube221 + cube222)

    # create the z axis for plots 2, 3, 4.
    extend = 'neither'
    zrange12 = diagtools.get_cube_range([cube221, cube222])
    zrange3 = diagtools.get_cube_range_diff([cube223])
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
        221, cube221, linspace12, cmap='viridis', title='Halosteric',
        extend=extend)
    add_map_subplot(
        222, cube222, linspace12, cmap='viridis',
        title='Steric - Thermosteric',
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
            title='Difference/Sum',
            log=True)

    # Add overall title
    units = str(cube221.units)
    fig.suptitle(long_name + ' [' + units + '] '+ time_str, fontsize=14)

    # Determine image filename
    fn_list = ['Detrended', project, dataset, ensemble, exp, short_name, 'quad_maps']
    path = diagtools.folder([cfg['plot_dir'], 'SLR_sanity_check']) + '_'.join(fn_list)
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
    cube = fix_depth(cube)
    zunits = str(cube.coord(axis='Z').units)
    if zunits.lower() not in ['m', 'meter', 'meters']:
        print('calc_ohc_ts', exp, dataset, ensemble, project, depth_range, trend)
        print('Unknown units in z direction:', zunits)
#       if str(thetao_bar.coord(axis='z').units).lower() in ['cm', 'centimeters']:
#           depths_bar = depths_bar /100.
#           print('changed depth units')
#           thicknesses_bar = thicknesses_bar/100.
        assert 0

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


def step_4and5(
        dataset,
        lon,
        lat,
        pressure,
        sal,
        temp,
        debug = False):
    """
    Calculates correct conservative temperature and absolute salinity.
    """
    if debug: print('step_4and5: pre. ',dataset,'\nsal:', sal,'\ntemp:',temp)
    if isinstance(sal, iris.cube.Cube):
        sal = sal.data
    if isinstance(temp, iris.cube.Cube):
        temp = temp.data

    mask1 = sal.mask
    mask2 = temp.mask
#    mask_sum = np.int(mask1)+np.int(mask2)
#    print('step_4and5, mask_sum:', mask_sum)
#    if len(np.where(mask_sum == 1))>0: assert 0

    # 4 i) In all cases, we interpret the model salinity as equal to Preformed Salinity
    if dataset in model_type['TEOS-10']:
        if debug:
            print('applying:SA_from_Sstar')
        sal = gsw.SA_from_Sstar(sal, pressure, lon, lat)
    else: #if dataset in model_type['EOS80']:
        #print('assuming that ', dataset, 'is EOS-80')
        if debug:
            print('applying:SA_from_Sstar * 35.16504 / 35.')
        sal = gsw.SA_from_Sstar(sal* 35.16504 / 35., pressure, lon, lat)

    # 5) Conversions of temperature to Conservative Temperature
    # 5 a) If the model is using TEOS-10, and the variable you have is thetao (potential temperature),
    #      then you should convert thetao (potential temperature) to CT=bigthetao (Conservative Temperature)
    if dataset in model_type['TEOS-10'] and dataset in model_type['potential temperature']:
        if debug:
            print('applying gsw.CT_from_pt')
        temp = gsw.CT_from_pt(sal, temp)

    # 5 b) If the model is not using TEOS-10, then you should interpret thetao as equal to bigthetao.  CT=thetao.  DO NOT RECALCULATE.
    # Do nothing.

    # 5 c) If you have in situ temperature (T), then you should convert to Conservative Temperature
    if dataset in model_type['in situ']:
        if debug:
            print('applying gsw.CT_from_t')
        temp = gsw.CT_from_t(sal, temp, pressure)

    sal = np.ma.masked_where(mask1 + sal.mask, sal)
    temp = np.ma.masked_where(mask2 + temp.mask, temp)

    return sal, temp


def calc_dyn_height_clim(cfg,
        metadatas,
        hist_thetao_fn,
        hist_so_fn,
        clim_type='fullhistorical',
        trend='detrended',
        method='dyn_height'):
    """
    Calculate the climatoligical dyncamic height.
    """
    # Load relevant metadata
    exp = metadatas[hist_thetao_fn]['exp']
    dataset = metadatas[hist_thetao_fn]['dataset']
    ensemble = metadatas[hist_thetao_fn]['ensemble']
    project = metadatas[hist_thetao_fn]['project']

    # Generate output paths for SLR netcdfs
    if method == 'dyn_height':
        work_dir = diagtools.folder([cfg['work_dir'], 'dyn_height_clim'])
        clim_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'dyn_height_clim', trend, clim_type]) + '.nc'
    elif method == 'Landerer':
        work_dir = diagtools.folder([cfg['work_dir'], 'Landerer_clim'])
        clim_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'Landerer_clim', trend, clim_type]) + '.nc'
    else: assert 0

    if os.path.exists(clim_fn):
        print ('calc_dyn_height_clim: exits already', clim_fn)
        return clim_fn
    # Load historical temperature and salinity netcdfs
    print('load hist netcdfs')
    psal_bar = iris.load_cube(hist_so_fn)
    thetao_bar = iris.load_cube(hist_thetao_fn)

    # Calculate climatologies for historical period Temperature and salinity.
    # Note that _bar suffix indicates climatology data.
    print('Calculate clim', dataset, exp,  clim_type)
    if clim_type == '1971-2018':
        # 1) vs. 1971-2018 (in comparison to cross-chapter Box 9.1)
        psal_bar = extract_time(psal_bar, 1971, 1, 1, 2018, 12, 31)
        thetao_bar = extract_time(thetao_bar, 1971, 1, 1, 2018, 12, 31)
    elif clim_type == '2005-2018':
        # 2) vs. 2005-2018 (in comparison to Argo Period)
        psal_bar = extract_time(psal_bar, 2005, 1, 1, 2018, 12, 31)
        thetao_bar = extract_time(thetao_bar, 2005, 1, 1, 2018, 12, 31)
    elif clim_type == '1850-1900':
        # 3) vs. 1850-1900 (traditional "pre-industrial")
        psal_bar = extract_time(psal_bar, 1850, 1, 1, 1900, 12, 31)
        thetao_bar = extract_time(thetao_bar, 1850, 1, 1, 1900, 12, 31)
    elif clim_type == '1995-2014':
        # 4) vs. 1995-2014 (AR5 standard)
        psal_bar = extract_time(psal_bar, 1995, 1, 1, 2014, 12, 31)
        thetao_bar = extract_time(thetao_bar, 1995, 1, 1, 2014, 12, 31)
    elif clim_type == '1985-2014':
        # 5) vs. 1985-2014 (Convenient 30 yr from AR5)
        psal_bar = extract_time(psal_bar, 1985, 1, 1, 2014, 12, 31)
        thetao_bar = extract_time(thetao_bar, 1985, 1, 1, 2014, 12, 31)
    elif clim_type == '2004-2018':
        # 6) vs. 2004-2018 (in comparison to RAPID Period)
        psal_bar = extract_time(psal_bar, 2004, 1, 1, 2018, 12, 31)
        thetao_bar = extract_time(thetao_bar, 2004, 1, 1, 2018, 12, 31)
    elif clim_type == 'fullhistorical':
        # 7) Full historical
        pass
    elif clim_type == 'piControl':
        # 8) vs. detrended PI-control (same window as full historical)
        if exp != 'piControl':
            assert 0
    else:
        assert 0
    # Calculaste average along the time dimension
    psal_bar = psal_bar.collapsed('time', iris.analysis.MEAN)
    thetao_bar = thetao_bar.collapsed('time', iris.analysis.MEAN)
    single_pane_map_plot(
        cfg,
        metadatas[hist_so_fn],
        psal_bar[0],
        key=method+'_clim/'+clim_type+'_surface_so',
        sym_zero=False,
        )

    single_pane_map_plot(
        cfg,
        metadatas[hist_thetao_fn],
        thetao_bar[0],
        key=method+'_clim/'+clim_type+'_surface_thetao',
        sym_zero=False,
        )

    # load latitude longitude dimensions
    lats = thetao_bar.coord('latitude').points
    lons = thetao_bar.coord('longitude').points

    # Make sure latitude and longitude are 2D
    if lats.ndim == 1:
        lon, lat = np.meshgrid(lons, lats)
    elif lats.ndim == 2:
        lat = lats
        lon = lons

    # Load depth & pressure data, ensuring depth is negative
    depths_bar = -1.*np.abs(thetao_bar.coord(axis='z').points)
    thicknesses_bar =np.abs (thetao_bar.coord(axis='z').bounds[...,1] - thetao_bar.coord(axis='z').bounds[...,0])
    if str(thetao_bar.coord(axis='z').units).lower() in ['cm', 'centimeters']:
        depths_bar = depths_bar /100.
        print('changed depth units')
        thicknesses_bar = thicknesses_bar/100.

    # Create output cubes to receive SLR data.
    print('Creating output arrays')
    if method == 'dyn_height':
        dyn_height_clim = np.zeros(thetao_bar[0].shape) # 2d + time
    else:
        dyn_height_clim = np.zeros(thetao_bar.shape)
    count = 0
    gravity = 9.7963 # m /s^2

    print('Starting clim SLR calculation from fresh')
    slr_clim = thetao_bar[0].copy() # 2D
    if depths_bar.ndim == 1:
       depth_bar = depths_bar
       thickness_bar = thicknesses_bar
    psal_bar_data = psal_bar.data
    thetao_bar_data =  thetao_bar.data
    # Iterate over each y line:
    for (y, x), la  in np.ndenumerate(lat):

    #    for y, z in np.arange(len(lat[:,0])):
        sal_bar = psal_bar_data[:, y, x]
        temp_bar = thetao_bar_data[:, y, x]

        if np.ma.is_masked(sal_bar.max()):
             continue
        la = lat[y, x]
        lo = lon[y, x]
        print('Calculate clim dyn height:', dataset, clim_type, (y,x,)) #, 'latitude:', la, lo)

        # load clim depth
        if depths_bar.ndim == 3:
            depth_bar = depths_bar[:, y, x]
            thickness_bar = thicknesses_bar[:, y, x]

        if depth_bar.shape != sal_bar.shape:
            assert 0

        # Mask below 2000m
        temp_bar =  np.ma.masked_where(temp_bar.mask + (depth_bar<-2000.), temp_bar)
        sal_bar = np.ma.masked_where(sal_bar.mask + (depth_bar<-2000.), sal_bar)

        # calculate pressure
        pressure_bar = gsw.conversions.p_from_z(depth_bar, la)
        pressure_bar = np.ma.masked_where(temp_bar.mask, pressure_bar)

        # calculate corerct conservative temperature and absolute salinity
        sal_bar, temp_bar = step_4and5(dataset, lo, la, pressure_bar, sal_bar, temp_bar)

        # copy fixed data back into original cube.
        psal_bar_data[:, y, x] = psal_bar_data[:, y, x] - sal_bar
        thetao_bar_data[:, y, x] = thetao_bar_data[:, y, x] - temp_bar

        # Calculate climatoligical pressure
        # pressure_bar = np.array([gsw.conversions.p_from_z(depth_bar[:, y, :], la[y]) for y in np.arange(len(la))]) # dbar
        #print(y, lola[y], la.shape, lo.shape, lat[:,0].shape)
        #pressure_bar = gsw.conversions.p_from_z(depth_bar, la)
        #psal_bar, temp_bar = step_4and5(dataset, lon, lat, pressure_bar, psal_bar, temp_bar)

        # Calculuate climatological Dynamic height anomaly
        # max_dp is the maximum difference between layers, set to a very high value
        # to avoid expensive interpolation.

        # Convert dynamic height into mm.
        if method == 'dyn_height':
            dyn_height_clim[y, x] = gsw.geo_strf_dyn_height(sal_bar, temp_bar, pressure_bar).sum() * 1000. / gravity
        elif method == 'Landerer':
            dyn_height_clim[:, y, x] = gsw.rho(sal_bar, temp_bar, pressure_bar)
            # assert 0 # Need to calculate this with and without clim thickness.

    # POlot fixed temperatrure and salinity.
    single_pane_map_plot(
          cfg,
          metadatas[hist_so_fn],
          psal_bar[0],
          key='Dyn_height_clim/'+method+clim_type+'_surface_sal_fix',
          sym_zero=False,
          )
    single_pane_map_plot(
          cfg,
          metadatas[hist_thetao_fn],
          thetao_bar[0],
          key='Dyn_height_clim/'+method+clim_type+'_surface_temp_fix',
          sym_zero=False,
          )

    # Save climatological SLR cube as a netcdf.
    print("saving output cube:", clim_fn)
    if method == 'dyn_height':
        cube0 = thetao_bar[0, :, :].copy()
    elif method == 'Landerer':
        cube0 = thetao_bar[:, :, :].copy()
    cube0.data = dyn_height_clim #p.ma.masked_where(slr_clim==0., slr_clim)
    cube0.units = cf_units.Unit('mm')
    cube0.name = 'Climatological ('+ clim_type+') dynamic height'
    cube0.long_name = 'Climatological ('+ clim_type+') dynamic height '
    cube0.short_name = 'dynh_clim'
    cube0.var_name = 'dynh_clim'
    cube0.standard_name = 'steric_change_in_mean_sea_level'
    iris.save(cube0, clim_fn)

    if method == 'dyn_height':
        single_pane_map_plot(
              cfg,
              metadatas[hist_thetao_fn],
              cube0,
              key='Dyn_height_clim/'+method+clim_type+'_dynamic_height',
              sym_zero=False,
              )
    if method == 'Landerer':
        single_pane_map_plot(
              cfg,
              metadatas[hist_thetao_fn],
              cube0[0],
              key='Dyn_height_clim/'+method+clim_type+'_dynamic_height',
              sym_zero=False,
              )
    return clim_fn










def calc_dyn_height(
        cfg,
        metadatas,
        hist_thetao_fn,
        hist_so_fn,
        trend='detrended'):
    """ Calculates the steric, thermo and haloseeteric dynamic height.
    """

    # Load relevant metadata
    exp = metadatas[hist_thetao_fn]['exp']
    dataset = metadatas[hist_thetao_fn]['dataset']
    ensemble = metadatas[hist_thetao_fn]['ensemble']
    project = metadatas[hist_thetao_fn]['project']

    # Generate output paths for SLR netcdfs
    if method == 'dyn_height':
        work_dir = diagtools.folder([cfg['work_dir'], 'dyn_height'])
        total_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'total_dyn_height', trend])+'.nc'
        thermo_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'thermo_dyn_height', trend])+'.nc'
        halo_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'halo_dyn_height', trend])+'.nc'
    elif method == 'Landerer':
        work_dir = diagtools.folder([cfg['work_dir'], 'Landerer'])
        total_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'total_dyn_height', trend])+'.nc'
        thermo_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'thermo_dyn_height', trend])+'.nc'
        halo_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'halo_dyn_height', trend])+'.nc'

    # Check whether output paths exists already.
    # If they exist, then make some basic figures and return paths.
    if False not in [os.path.exists(fn) for fn in [total_fn, thermo_fn, halo_fn]]:
        for fn, key in zip([total_fn, thermo_fn, halo_fn], ['total', 'thermo', 'halo']):
            cube1 = iris.load_cube(fn)
            for t in [0, -1, 'mean']:
                if t == 'mean':
                    dat = cube1.copy().collapsed('time', iris.analysis.MEAN)
                else:
                    dat = cube1[t]
                single_pane_map_plot(
                      cfg,
                      metadatas[hist_thetao_fn],
                      dat,
                      key=method+'_'+key+'_'+trend,
                      sym_zero=True,
                      )
        return total_fn, thermo_fn, halo_fn

    # Load historical temperature and salinity netcdfs
    so_cube = iris.load_cube(hist_so_fn)
    thetao_cube = iris.load_cube(hist_thetao_fn)
    so_cube_data = so_cube.data
    thetao_cube_data = thetao_cube.data

    # load latitude longitude dimensions
    lats = thetao_cube.coord('latitude').points
    lons = thetao_cube.coord('longitude').points

    # Make sure latitude and longitude are 2D
    if lats.ndim == 1:
        lon, lat = np.meshgrid(lons, lats)
    elif lats.ndim == 2:
        lat = lats
        lon = lons

    # Load depth data, ensuring depth is negative
    depths = -1.*np.abs(thetao_cube.coord(axis='z').points.copy())
    thicknesses =np.abs (thetao_bar.coord(axis='z').bounds[...,1] - thetao_bar.coord(axis='z').bounds[...,0])

    # Load time array as decidimal time.
    times = diagtools.cube_time_to_float(thetao_cube)

    # Calculate climatologies for historical period Temperature and salinity.
    # Note that _bar suffix indicates climatology data.
    psal_bar = so_cube.copy()
    thetao_bar = thetao_cube.copy()

    psal_bar = psal_bar.collapsed('time', iris.analysis.MEAN)
    psal_bar_data = psal_bar.data
    thetao_bar = thetao_bar.collapsed('time', iris.analysis.MEAN)
    thetao_bar_data = thetao_bar.data
    depths_bar = -1.*np.abs(thetao_bar.coord(axis='z').points.copy())

    if str(thetao_cube.coord(axis='z').units).lower() in ['cm', 'centimeters']:
        depths = depths /100.
        depths_bar = depths_bar/100,

    # Create output cubes to receive SLR data.
    dyn_total = np.zeros(thetao_cube[:, 0, : , :].shape) #2d + time
    dyn_thermo = dyn_total.copy()
    dyn_halo = dyn_total.copy()

    count = 0
    gravity = 9.7963 # m /s^2

    # Now perform the SLR calculation for each point in time for each lat line:
    if depths.ndim == 1:
        depth = depths
    #    depth = np.tile(depths, (len(lat[0]), 1)).T

    for (y, x), la in np.ndenumerate(lat):
        print('Calculate SLR in 1D:', dataset, (y,x), 'of', lat.shape)
        sal_bar = psal_bar_data[:, y, x]
        if np.ma.is_masked(sal_bar.max()):
             continue
        temp_bar = thetao_bar_data[:, y, x]

        # Load historical coords
        # la = lat[y, x]
        lo = lon[y, x]

        # Load depth dataset
        if depths.ndim == 3:
            depth = depths[:, y, x]
        if depths.ndim != 4:
            pressure = gsw.conversions.p_from_z(depth, la) # dbar

        # Sort out climatological data
        if depths_bar.ndim == 1:
            depth_bar = depths_bar #hs_bar, (len(la), 1)).T
            #print('tiling depth:', depths_bar.shape, sal_bar.shape, depth_bar.shape)
            if depth_bar.shape != sal_bar.shape: assert 0
        elif depths_bar.ndim == 3:
            depth_bar = depths_bar[:, y, x]
        else:
            assert 0

        # Mask below 2000m
        temp_bar =  np.ma.masked_where(temp_bar.mask + (depth_bar<-2000.), temp_bar)
        sal_bar = np.ma.masked_where(sal_bar.mask + (depth_bar<-2000.), sal_bar)

        pressure_bar = gsw.conversions.p_from_z(depth_bar, la) # dbar
        pressure_bar = np.ma.masked_where(sal_bar.mask, pressure_bar)
        sal_bar, temp_bar = step_4and5(dataset, lo, la, pressure_bar, sal_bar, temp_bar)

        # Calculate SLR for each point in time in the historical dataset.
        for t, time in enumerate(times):
            #print('----\n', [t,'of', len(times)], [y,'of',len(lat[:,0])], time)

            if depths.ndim == 4:
                depth = depths[t, :, y, x]
                pressure = gsw.conversions.p_from_z(depth, la) # dbar
            if depth.shape != sal_bar.shape:
                print(depth.shape, sal_bar.shape, depths.shape)
                assert 0

            # load salinity and temperature data
            sal = so_cube_data[t, :, y, x]
            temp = thetao_cube_data[t, :, y, x]
            temp =  np.ma.masked_where(temp.mask + (depth<-2000.), temp)
            sal = np.ma.masked_where(sal.mask + (depth<-2000.), sal)

            pressure = np.ma.masked_where(temp.mask, pressure)

            # Confirm that we use absolute salininty & conservative temperature
            sal, temp = step_4and5(dataset, lo, la, pressure, sal, temp)

            # Calculate Dynamic height anomaly
            #print(dataset, 'About to cal gsdh_total')
            gsdh_total = gsw.geo_strf_dyn_height(sal, temp, pressure, ).sum() * 1000. / gravity
            #print(dataset, 'About to cal gsdh_thermo')
            gsdh_thermo = gsw.geo_strf_dyn_height(sal_bar, temp, pressure).sum() * 1000. / gravity
            #print(dataset, 'About to cal gsdh_halo')
            gsdh_halo = gsw.geo_strf_dyn_height(sal, temp_bar, pressure).sum() * 1000. / gravity

            # Put in the output array:
            dyn_total[t, y, x] = gsdh_total
            dyn_thermo[t, y, x] = gsdh_thermo
            dyn_halo[t, y, x] = gsdh_halo

            if x == t == 0:
                print(dataset, 'total', gsdh_total.min(),  gsdh_total.max())
                print(dataset, 'dyn_thermo', gsdh_thermo.min(), gsdh_thermo.max())
                print(dataset, 'halo', gsdh_halo.min(), gsdh_halo.max())

            count += 1
            if t == 0:
                print(count, (t, y, x), 'performing 2D SLR')

    dyn_total  = np.ma.masked_where(thetao_cube[:, 0, :, :].data.mask, dyn_total)
    dyn_thermo = np.ma.masked_where(thetao_cube[:, 0, :, :].data.mask, dyn_thermo)
    dyn_halo   = np.ma.masked_where(thetao_cube[:, 0, :, :].data.mask, dyn_halo )

    dyn_total = np.ma.masked_invalid(dyn_total)
    dyn_thermo = np.ma.masked_invalid(dyn_thermo)
    dyn_halo = np.ma.masked_invalid(dyn_halo)

    # Save SLR data as a cube and then a netcdf..
    cube0 = thetao_cube[:, 0, :, :].copy()
    cube0.data = dyn_total
    cube0.units = cf_units.Unit('mm')
    cube0.name = 'Total Steric Dynamic Height'
    cube0.long_name = 'Total Steric Dynamic Height'
    cube0.short_name = 'dyn_total'
    cube0.var_name = 'dyn_total'
    cube0.standard_name = 'steric_change_in_mean_sea_level'
    iris.save(cube0, total_fn)

    cube1 = thetao_cube[:, 0, :, :].copy()
    cube1.data = dyn_thermo
    cube1.units = cf_units.Unit('mm')
    cube1.name = 'Thermosteric Dynamic Height'
    cube1.long_name = 'Thermosteric Dynamic Height'
    cube1.short_name = 'dyn_thermo'
    cube1.var_name = 'dyn_thermo'
    cube1.standard_name = 'thermosteric_change_in_mean_sea_level'
    iris.save(cube1, thermo_fn)

    cube2 = thetao_cube[:, 0, :, :].copy()
    cube2.data = dyn_halo
    cube2.units = cf_units.Unit('mm')
    cube2.name = 'Halosteric Dynamic Height'
    cube2.long_name = 'Halosteric Dynamic Height'
    cube2.short_name = 'dyn_halo'
    cube2.var_name = 'dyn_halo'
    cube2.standard_name = 'halosteric_change_in_mean_sea_level'
    iris.save(cube2, halo_fn)

    for cube, key in zip([cube0, cube1, cube2], ['total', 'thermo', 'halo']):
        for t in [0, -1, 'mean']:
            if t == 'mean':
                dat = cube.copy().collapsed('time', iris.analysis.MEAN)
            else:
                dat = cube[t]
            single_pane_map_plot(
                cfg,
                metadatas[hist_thetao_fn],
                dat,
                key='dyn_height_'+key+'_'+trend,
                sym_zero=True,
                )

    return total_fn, thermo_fn, halo_fn



def check_units(cfg, metadata, files = [], keys = []):
    return
    print('--------\nChecking units:')
    print(keys)
    for fn in files:
        cube = iris.load_cube(fn)
        print('--------\nloaded:', fn)
        print(keys)
        print(cube.standard_name,cube.units, cube[0,0].data.min(), cube[0,0].data.max()) #,  cube.data.min(), cube.data.max(),)

        coords = cube.coords()
        for coord in coords:
            print(coord.var_name, coord.points.min(), coord.points.max(), coord.units)



# def calc_dyn_height_full(cfg,
#         metadatas,
#         hist_thetao_fn,
#         hist_so_fn,
#         picontrol_thetao_fn,
#         picontrol_so_fn,
#         trend='detrended',
#         method='dyn_height'
#         ):
#     """
#     calc_dyn_height_full: Calculates the Sea Level Rise
#     """
#     # Load relevant metadata
#     exp = metadatas[hist_thetao_fn]['exp']
#     dataset = metadatas[hist_thetao_fn]['dataset']
#     ensemble = metadatas[hist_thetao_fn]['ensemble']
#     project = metadatas[hist_thetao_fn]['project']
#
#     clim_types = ['1971-2018',  '2005-2018', '1850-1900' , '1995-2014',
#                       '1985-2014', '2004-2018', 'fullhistorical', 'piControl']
#     # So, need to figure out the reference period for SSP stuff.
#
#     clim_files = {}
#     for clim_type in clim_types:
#         if clim_type == 'piControl':
#             clim_fn = calc_dyn_height_clim(
#                 cfg,
#                 metadatas,
#                 picontrol_thetao_fn,
#                 picontrol_so_fn,
#                 clim_type=clim_type,
#                 trend=trend,
#                 method=method)
#         else:
#             clim_fn = calc_dyn_height_clim(
#                 cfg,
#                 metadatas,
#                 hist_thetao_fn,
#                 hist_so_fn,
#                 clim_type=clim_type,
#                 trend=trend,
#                 method=method)
#         clim_files[clim_type] = clim_fn
#
#     steric_fn, thermo_fn, halo_fn = calc_dyn_height(
#         cfg,
#         metadatas,
#         hist_thetao_fn,
#         hist_so_fn,
#         trend=trend,
#         method=method)
#
#     clim_files['total'] = steric_fn
#     clim_files['thermo'] = thermo_fn
#     clim_files['halo'] = halo_fn
#
#     return clim_files


def calc_landerer_slr(
        cfg,
        metadatas,
        thetao_fn,
        so_fn,
        picontrol_thetao_fn,
        picontrol_so_fn,
        hist_thetao_fn=None,
        hist_so_fn=None,
        trend='detrended',
        ):
    """
    calc_landerer_slr: Calculates the Sea Level Rise using the Landerer method.
    """
    # Load relevant metadata
    exp = metadatas[thetao_fn]['exp']
    dataset = metadatas[thetao_fn]['dataset']
    ensemble = metadatas[thetao_fn]['ensemble']
    project = metadatas[thetao_fn]['project']

    method = 'Landerer'
    clim_type = '1850-1900'
    if exp.find('ssp')>-1:
        # Use the historical dataset for ssp future scenarios.
        clim_fn = calc_dyn_height_clim(
            cfg,
            metadatas,
            hist_thetao_fn,
            hist_so_fn,
            clim_type=clim_type,
            trend=trend,
            method=method)
    else:
        clim_fn = calc_dyn_height_clim(
            cfg,
            metadatas,
            thetao_fn,
            so_fn,
            clim_type=clim_type,
            trend=trend,
            method=method)


    # clim_file = calc_dyn_height_clim(
    #             cfg,
    #             metadatas,
    #             picontrol_thetao_fn,
    #             picontrol_so_fn,
    #             clim_type=clim_type,
    #             trend=trend,
    #             method=method)
    work_dir = diagtools.folder([cfg['work_dir'], 'Landerer'])
    total_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'total_slr', trend])+'.nc'
    thermo_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'thermo_slr', trend])+'.nc'
    halo_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'halo_slr', trend])+'.nc'

    slr_fn_dict= {}
    slr_fn_dict['total'] = total_fn
    slr_fn_dict['thermo'] = thermo_fn
    slr_fn_dict['halo'] = halo_fn

    # Check whether output paths exists already.
    # If they exist, then make some basic figures and return paths.
    if False not in [os.path.exists(fn) for fn in [total_fn, thermo_fn, halo_fn]]:
        return slr_fn_dict

        for fn, key in zip([total_fn, thermo_fn, halo_fn], ['total', 'thermo', 'halo']):
            cube1 = iris.load_cube(fn)
            for t in [0, -1, 'mean']:
                if t == 'mean':
                    dat = cube1.copy().collapsed('time', iris.analysis.MEAN)
                else:
                    dat = cube1[t]
                single_pane_map_plot(
                      cfg,
                      metadatas[thetao_fn],
                      dat,
                      key=method+'_'+key+'_'+trend,
                      sym_zero=True,
                      )
        return slr_fn_dict
                      

    # Load main temperature and salinity netcdfs
    so_cube = iris.load_cube(so_fn)
    thetao_cube = iris.load_cube(thetao_fn)
    so_cube_data = so_cube.data
    thetao_cube_data = thetao_cube.data

    # load latitude longitude dimensions
    lats = thetao_cube.coord('latitude').points
    lons = thetao_cube.coord('longitude').points

    # Make sure latitude and longitude are 2D
    if lats.ndim == 1:
        lon, lat = np.meshgrid(lons, lats)
    elif lats.ndim == 2:
        lat = lats
        lon = lons

    # Load depth data, ensuring depth is negative
    depths = -1.*np.abs(thetao_cube.coord(axis='z').points.copy())
    # thicknesses =np.abs (thetao_bar.coord(axis='z').bounds[...,1] - thetao_bar.coord(axis='z').bounds[...,0])
    thicknesses =np.abs (thetao_cube.coord(axis='z').bounds[...,1] - thetao_cube.coord(axis='z').bounds[...,0])

    # Load time array as decidimal time.
    times = diagtools.cube_time_to_float(thetao_cube)

    # Calculate climatologies for historical period Temperature and salinity.
    # Note that _bar suffix indicates climatology data.
    # Acrtually, this is clim_rho.
    rho_clim_cube = iris.load_cube(clim_fn)
    rho_clim_data = rho_clim_cube.data

    if exp.find('ssp')>-1:
        # Use historical files for this calculation.
        psal_bar =  iris.load_cube(hist_so_fn)
        thetao_bar = iris.load_cube(hist_thetao_fn)
    else:
        psal_bar = so_cube.copy()
        thetao_bar = thetao_cube.copy()

    if clim_type == '1850-1900':
        psal_bar = extract_time(psal_bar, 1850, 1, 1, 1900, 12, 31)
        thetao_bar = extract_time(thetao_bar, 1850, 1, 1, 1900, 12, 31)
        psal_bar = psal_bar.collapsed('time', iris.analysis.MEAN)
        thetao_bar = thetao_bar.collapsed('time', iris.analysis.MEAN)
    else:
        assert 0
    psal_bar_data = psal_bar.data
    thetao_bar_data = thetao_bar.data
    depths_bar = -1.*np.abs(thetao_bar.coord(axis='z').points.copy())
    print(project, dataset, exp, ensemble, 'slr', trend, 'depths_bar:', depths_bar)
    #thickneses_bar = np.abs(thetao_bar.coord(axis='z').bounds[...,1] - thetao_bar.coord(axis='z').bounds[...,0])

    if str(thetao_cube.coord(axis='z').units).lower() in ['cm', 'centimeters']:
        depths = depths /100.
        depths_bar = depths_bar/100

    # Create output cubes to receive SLR data.
    slr_total = np.zeros(thetao_cube[:, 0, : , :].shape) #2d + time
    slr_thermo = slr_total.copy()
    slr_halo = slr_total.copy()

    count = 0
    gravity = 9.7963 # m /s^2

    # Now perform the SLR calculation for each point in time for each lat line:
    if depths.ndim == 1:
        depth = depths
        thickness = thicknesses
    if depths_bar.ndim == 1:
        depth_bar = depths_bar
        #thickness_bar = thickneses_bar
    #    depth = np.tile(depths, (len(lat[0]), 1)).T

    for (y, x), la in np.ndenumerate(lat):
        print('Calculate SLR in 1D:', dataset, (y,x), 'of', lat.shape)
        sal_bar = psal_bar_data[:, y, x]
        if np.ma.is_masked(sal_bar.max()):
             continue
        temp_bar = thetao_bar_data[:, y, x]
        rho_bar = rho_clim_data[:, y, x]

        # Load historical coords
        # la = lat[y, x]
        lo = lon[y, x]

        # Load depth dataset
        if depths.ndim == 3:
            depth = depths[:, y, x]
            thickness = thicknesses[:, y, x]
        if depths.ndim != 4:
            pressure = gsw.conversions.p_from_z(depth, la) # dbar

        # Sort out climatological data
        if depths_bar.ndim == 3:
            depth_bar = depths_bar[:, y, x]
            #thickness_bar = thicknesses_bar[:, y, x]

        # Mask below 2000m
        temp_bar =  np.ma.masked_where(temp_bar.mask + (depth_bar<-2000.), temp_bar)
        sal_bar = np.ma.masked_where(sal_bar.mask + (depth_bar<-2000.), sal_bar)
        pressure_bar = gsw.conversions.p_from_z(depth_bar, la) # dbar
        pressure_bar = np.ma.masked_where(sal_bar.mask, pressure_bar)
        sal_bar, temp_bar = step_4and5(dataset, lo, la, pressure_bar, sal_bar, temp_bar)

        # Calculate SLR for each point in time in the historical dataset.
        for t, time in enumerate(times):

            if depths.ndim == 4:
                depth = depths[t, :, y, x]
                thickness = thicknesses[t, :, y, x]
                pressure = gsw.conversions.p_from_z(depth, la) # dbar
            if depth.shape != sal_bar.shape:
                print(depth.shape, sal_bar.shape, depths.shape)
                assert 0

            # load salinity and temperature data
            sal = so_cube_data[t, :, y, x]
            temp = thetao_cube_data[t, :, y, x]
            temp =  np.ma.masked_where(temp.mask + (depth<-2000.), temp)
            sal = np.ma.masked_where(sal.mask + (depth<-2000.), sal)

            pressure = np.ma.masked_where(temp.mask, pressure)

            # Confirm that we use absolute salininty & conservative temperature
            sal, temp = step_4and5(dataset, lo, la, pressure, sal, temp)

            #Calculate SLR, and convert into mm
            total = thickness*((rho_bar - gsw.rho(sal, temp, pressure) )/rho_bar)
            halo = thickness*((rho_bar - gsw.rho(sal, temp_bar, pressure) )/rho_bar)
            thermo = thickness*((rho_bar - gsw.rho(sal_bar, temp, pressure) )/rho_bar)

            # Put in the output array:
            slr_total[t, y, x] = total.sum() * 1000.
            slr_thermo[t, y, x] = thermo.sum() * 1000.
            slr_halo[t, y, x] = halo.sum() * 1000.

            count += 1
            if t == 0:
                print(count, (t, y, x), 'performing 2D SLR')

    slr_total  = np.ma.masked_where(thetao_cube[:, 0, :, :].data.mask, slr_total)
    slr_thermo = np.ma.masked_where(thetao_cube[:, 0, :, :].data.mask, slr_thermo)
    slr_halo   = np.ma.masked_where(thetao_cube[:, 0, :, :].data.mask, slr_halo )

    slr_total = np.ma.masked_invalid(slr_total)
    slr_thermo = np.ma.masked_invalid(slr_thermo)
    slr_halo = np.ma.masked_invalid(slr_halo)

    # Save SLR data as a cube and then a netcdf..
    cube0 = thetao_cube[:, 0, :, :].copy()
    cube0.data = slr_total
    cube0.units = cf_units.Unit('mm')
    cube0.name = 'Total Steric Dynamic Height'
    cube0.long_name = 'Total Steric Dynamic Height'
    cube0.short_name = 'slr_total'
    cube0.var_name = 'slr_total'
    cube0.standard_name = 'steric_change_in_mean_sea_level'
    iris.save(cube0, total_fn)

    cube1 = thetao_cube[:, 0, :, :].copy()
    cube1.data = slr_thermo
    cube1.units = cf_units.Unit('mm')
    cube1.name = 'Thermosteric Dynamic Height'
    cube1.long_name = 'Thermosteric Dynamic Height'
    cube1.short_name = 'slr_thermo'
    cube1.var_name = 'slr_thermo'
    cube1.standard_name = 'thermosteric_change_in_mean_sea_level'
    iris.save(cube1, thermo_fn)

    cube2 = thetao_cube[:, 0, :, :].copy()
    cube2.data = slr_halo
    cube2.units = cf_units.Unit('mm')
    cube2.name = 'Halosteric Dynamic Height'
    cube2.long_name = 'Halosteric Dynamic Height'
    cube2.short_name = 'slr_halo'
    cube2.var_name = 'slr_halo'
    cube2.standard_name = 'halosteric_change_in_mean_sea_level'
    iris.save(cube2, halo_fn)

    for cube, key in zip([cube0, cube1, cube2], ['total', 'thermo', 'halo']):
        for t in [0, -1, 'mean']:
            if t == 'mean':
                dat = cube.copy().collapsed('time', iris.analysis.MEAN)
            else:
                dat = cube[t]
            single_pane_map_plot(
                cfg,
                metadatas[thetao_fn],
                dat,
                key='slr_height_'+key+'_'+trend,
                sym_zero=True,
                )

    return slr_fn_dict


# def plot_dyn_height_ts(cfg, metadata, dyn_averages,  trend, region):
#     """
#     Make some time series plots.
#     """
#     exp = metadata['exp']
#     dataset = metadata['dataset']
#     ensemble = metadata['ensemble']
#     project = metadata['project']
#
#     clim_types = ['1971-2018',  '2005-2018', '1850-1900' , '1995-2014',
#                   '1985-2014', '2004-2018', 'fullhistorical', 'piControl']
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.title(' '.join([project, dataset, exp, ensemble, trend, region, ]))
#
#     for dyn_type, dyn_fn in dyn_averages.items():
#         dyn_cube = iris.load_cube(dyn_fn)
#         if dyn_cube.data.ndim == 0:
#             plt.axhline(dyn_cube.data, c = 'k', ls=':' )
#         else:
#             times = diagtools.cube_time_to_float(dyn_cube)
#             plt.plot(times, dyn_cube.data, label = dyn_type)
#
#     plt.xlabel('Year')
#     plt.ylabel('Nonanomalous Dynamic Height, mm')
#     plt.legend()
#
#     path = diagtools.folder([cfg['plot_dir'], 'dyn_height_timeseries'])
#     path += '_'.join([project, dataset, exp, ensemble, trend, 'all_timeseries'])+diagtools.get_image_format(cfg)
#     print('Saving figure:', path)
#     plt.savefig(path)
#     plt.close()


# def plot_slr_full_ts(cfg, metadata, dyn_averages, trend, region):
#     """
#     Make SLR time series plots for individual models.
#
#     8 pane picture, with
#     """
#     exp = metadata['exp']
#     dataset = metadata['dataset']
#     ensemble = metadata['ensemble']
#     project = metadata['project']
#
#     steric_types = ['total', 'thermo', 'halo']
#     cubes = {}
#     print('---------\n', project, dataset, exp, ensemble, region)
#     for dyn_type, fn in dyn_averages.items():
#
#         cubes[dyn_type] = iris.load_cube(fn)
#         print(dyn_type, ':',cubes[dyn_type].data.shape, 'mean:', cubes[dyn_type].data.mean())#s.path.basename fn)
#     panes = {'1971-2018':424, '2005-2018':428, '1850-1900':423 , '1995-2014':426,
#              '1985-2014':425, '2004-2018':427, 'fullhistorical':422, 'piControl':421}
#     yranges = {
#        '1971-2018': [1971, 2018 + 1],
#        '2005-2018': [2005, 2018 + 1],
#        '1850-1900': [1850, 1900 + 1],
#        '1995-2014': [1995, 2014 + 1],
#        '1985-2014': [1985, 2014 + 1],
#        '2004-2018': [2004, 2018 + 1],
#        'fullhistorical': [1850, 2015 + 1],
#        'piControl': [1850, 2015 + 1],
#        }
#
#     fig = plt.figure()
#     fig.set_size_inches(12, 8)
#     for clim_type, sbp in panes.items():
#         ax = fig.add_subplot(sbp)
#         for steric_type in steric_types:
#             times = diagtools.cube_time_to_float(cubes[steric_type])
#             data = - cubes[steric_type].data + cubes[clim_type].data
#             plt.plot(times, data, label = steric_type.title())
#         plt.axhline(0., c = 'k', ls='-', lw=0.5)
#         plt.axhline(0.5, c = 'k', ls=':', lw=0.5)
#         plt.axhline(-0.5, c = 'k', ls=':', lw=0.5)
#
#         if clim_type == 'piControl':
#             ax.axvspan(yranges[clim_type][0], yranges[clim_type][1], alpha=0.35, color='red')
#         else:
#             ax.axvspan(yranges[clim_type][0], yranges[clim_type][1], alpha=0.35, color='black')
#
#         ax.text(.5,.82, clim_type,
#             horizontalalignment='center',
#             transform=ax.transAxes)
#
#     fig.add_subplot(111, frame_on=False)
#     plt.tick_params(labelcolor="none", bottom=False, left=False)
#     plt.xlabel('Year')
#     plt.ylabel('Steric anomaly, mm')
#     plt.suptitle(' '.join([project, dataset, exp, ensemble, trend, 'SLR']))
#     path = diagtools.folder([cfg['plot_dir'], 'SLR_timeseries'])
#     path += '_'.join([project, dataset, exp, ensemble, trend, 'slr_timeseries'])+diagtools.get_image_format(cfg)
#     print('Saving figure:', path)
#     plt.savefig(path)
#     plt.close()




def plot_slr_full_ts_all(cfg, metadatas, dyn_fns, plot_region, clim_range = '1850-1900', method='dyn_height'):
    """
    Make SLR time series plots for multiple   models.

    4 pane picture, with
    """
    datasets = {}
    trends = {}
    dyn_types = {}
    for (project, dataset, exp, ensemble, dyn_type, region, trend), fn in dyn_fns.items():
         datasets[dataset ] = True
         trends[trend] = True
         dyn_types[dyn_type] = True
         print(project, dataset, exp, ensemble, dyn_type, trend, fn)
    datasets = datasets.keys()
    trends = trends.keys()
    dyn_types = dyn_types.keys()

#   exp = metadata['exp']
#   dataset = metadata['dataset']
#   ensemble = metadata['ensemble']
#   project = metadata['project']

    #steric_types = ['total', 'thermo', 'halo']
    panes = {'total': 411, 'thermo':412, 'halo':413} #sanity_check': 414}
    #panes = {'1971-2018':424, '2005-2018':428, '1850-1900':423 , '1995-2014':426,
    #         '1985-2014':425, '2004-2018':427, 'fullhistorical':422, 'piControl':421}
    for trend_plot in trends:
        fig = plt.figure()
        fig.set_size_inches(12, 8)
        timesdict = {}
        datadict = {}

        for pane_type, sbp in panes.items():
            ax = fig.add_subplot(sbp)
            for (project, dataset, exp, ensemble, dyn_type, region, trend), fn in dyn_fns.items():
                if plot_region != region: continue
                if trend_plot!= trend: continue

                if pane_type+'_ts' != dyn_type:
                    continue

                cube = iris.load_cube(fn)
                times = diagtools.cube_time_to_float(cube)
                all_keys = (project, dataset, exp, ensemble, clim_range+'_ts', plot_region, trend)

                if np.ma.is_masked(cube.data.max()):
                    print('Data is all masked:',cube.data, clim_value)
                    assert 0

                print(all_keys, 'file:', fn)
                if method == 'dyn_height':
                    clim_fn = dyn_fns[all_keys]  #(project, dataset, exp, ensemble, clim_range+'_ts', trend)]
                    clim_value = iris.load_cube(clim_fn).data
                    if np.ma.is_masked(clim_value.max()):
                        print('Clim data is all masked:',cube.data, clim_value)
                        assert 0
                    data = -1 *(cube.data - clim_value )
                elif method=='Landerer':
                   data = cube.data

                plt.plot(times, data, label = dataset)
                timesdict[(project, dataset, exp, ensemble, dyn_type, region, trend)] = times
                datadict[(project, dataset, exp, ensemble, dyn_type, region, trend)] = np.ma.array(data)

            plt.legend()
            ax.text(.5,.82, pane_type.title(),
                horizontalalignment='center',
                transform=ax.transAxes)

        # Sanity check pane.
        ax = fig.add_subplot(414)
        for (project, dataset, exp, ensemble, dyn_type, region, trend), data  in  datadict.items():
             print("Sanity Check:", (project, dataset, exp, ensemble, dyn_type, region, trend))
             if plot_region != region: continue
             if trend_plot!= trend: continue
             if dyn_type != 'total_ts': continue

             times = timesdict[(project, dataset, exp, ensemble, dyn_type, region, trend)]
             data = data - datadict[(project, dataset, exp, ensemble, 'halo_ts', region, trend)]
             data = data - datadict[(project, dataset, exp, ensemble, 'thermo_ts', region, trend)]

             plt.plot(times, data, label = dataset)

        plt.legend()
        ax.text(.5,.82, 'Total - thermo - halo',
             horizontalalignment='center',
             transform=ax.transAxes)


        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.xlabel('Year')
        plt.ylabel('Steric anomaly, mm')
        plt.suptitle(' '.join(['Sea level rise', trend_plot, plot_region]))

        path = diagtools.folder([cfg['plot_dir'], 'SLR_timeseries_all'])
        path += '_'.join(['slr_timeseries_all', trend_plot, plot_region, method])+diagtools.get_image_format(cfg)
        print('Saving figure:', path)
        plt.savefig(path)
        plt.close()



def plot_slr_regional_scatter(cfg, metadatas, dyn_fns,
        plot_exp = 'historical',
        #plot_clim = '1850-1900_ts',
        plot_dyn = 'halo_ts',
        plot_trend = 'detrended',
        method='dyn_height',
        time_range = [1970, 2020],
        fig=None,
        subplot = 111,
        show_UKESM=False,
        show_legend=True,
    ):
    """
    Add a interannual trend plot for each model and for the
    """
    # First add individual models:
    trends = {}
    datasets = {}
    ensembles = {}
    exps = {}

    hist_datasets = {}
    histnat_datasets = {}
    for (project, dataset, exp, ensemble, dyn_type, region, trend), fn in dyn_fns.items():
        if dyn_type != plot_dyn: continue
        if trend != plot_trend: continue
        if region == 'Global': continue # only want pacfic/atlantic
#       if exp != plot_exp: continue
        datasets[dataset] = True
        exps[exp] = True
        ensembles[ensemble] = True
        if exp == 'historical':
            hist_datasets[dataset] = True
        if exp == 'hist-nat':
            histnat_datasets[dataset] = True
        cube = iris.load_cube(fn)
        times = diagtools.cube_time_to_float(cube)
        if method == 'dyn_height': # anomaly is
            assert 0
        if method == 'Landerer': # anomaly is already calculated.
            data = cube.data
        print('------\nplot_slr_regional_scatter',project, dataset, exp, ensemble, dyn_type, region, trend)
        times = np.ma.masked_outside(times, time_range[0], time_range[1])
        data = np.ma.masked_where(times.mask, data)
        linreg = linregress(times.compressed(), data.compressed())
        trends[(dataset, exp, ensemble, region)] = linreg.slope
    if isinstance(subplot, int) and subplot==111:
        fig = plt.figure()
    if isinstance(subplot, int):
        ax = fig.add_subplot(subplot)
    else:
        ax = subplot
        plt.sca(ax) # set current axes

    count = 0
    # colours = {'observations':'black', 'historical': CMIP6_red, 'hist-nat': histnat_green}
    colours = {'observations':'black', 'historical': historical_beige, 'hist-nat': histnat_green}

    labels = []
    max_value = 0.

    for dataset, exp, ensemble in itertools.product(datasets, exps, ensembles):
        pac =  trends.get((dataset, exp, ensemble, 'Pacific'), None)
        alt =  trends.get((dataset, exp, ensemble, 'Atlantic'), None)
        print('Scatter:', dataset, exp, ensemble, 'pac:', pac, 'alt:', alt)
        if None in [pac, alt]: continue
        col=  colours[exp]
        if exp == 'historical':

            label = 'historical (n = '+str(int(len(hist_datasets.keys())))+')'
        elif exp == 'hist-nat':
            label = 'hist-nat (n = '+str(int(len(histnat_datasets.keys())))+')'
        else:
            label = exp
        max_value = np.max([max_value, abs(pac), abs(alt)])
        if show_UKESM and dataset.lower().find('ukesm')>-1:
            col='purple'
            label = 'UKESM'
        if label not in labels:
            plt.scatter([], [], c=col, marker='s', label = label)
        plt.scatter(pac, alt, c=col, marker='s', alpha=0.5)
        labels.append(label)
        count +=1
    if not count:
         plt.close()
         return

    add_means=True
    if add_means==True:
        pac_means = {}
        alt_means = {}
        for dataset, exp, ensemble in itertools.product(datasets, exps, ensembles):
            pac =  trends.get((dataset, exp, ensemble, 'Pacific'), None)
            alt =  trends.get((dataset, exp, ensemble, 'Atlantic'), None)
            if None in [pac, alt]:
                continue
            if (dataset, exp) in pac_means.keys():
                pac_means[(dataset, exp)].append(pac)
                alt_means[(dataset, exp)].append(alt)
            else:
                pac_means[(dataset, exp)] = [pac, ]
                alt_means[(dataset, exp)] = [alt, ]
        for exp0 in exps:
            exps_pac = []
            exps_alt = []
            # Add each model:
            for (dataset, exp1), values in pac_means.items():
                if not len(values): continue
                if exp1 != exp0: continue
                exps_pac.append(np.mean(values))
                exps_alt.append(np.mean(alt_means[(dataset, exp1)]))
            col=  colours[exp0]
#            label = 'Ensemblex_mean'
#            if label not in labels:
#                plt.scatter(np.mean(exps_pac), np.mean(exps_alt), c=col, s=20, marker='D', label = label)
#            else:
            plt.scatter(np.mean(exps_pac), np.mean(exps_alt), facecolor=col,edgecolor='k', s=70, marker='D')
        plt.scatter([], [], edgecolor='k', facecolor='none',  s=35, marker='D', label='Mean 1950-2014')

    add_obs = True
    if add_obs:
        obs_labels= {
            '210201_EN4.2.1.g10_annual_steric_1950-2019_5-5350m.nc': 'EN4 1950-2019',
            '210201_Ishii17_v7.3_annual_steric_1955-2019_0-3000m.nc':'Ishii 1955-2019',
            '210127_DurackandWijffels_V1.0_70yr_steric_1950-2019_0-2000db_210122-205355_beta.nc':'D&W 1950-2019',
            }
        obs_markers= {
            '210201_EN4.2.1.g10_annual_steric_1950-2019_5-5350m.nc': '^',
            '210201_Ishii17_v7.3_annual_steric_1955-2019_0-3000m.nc':'o',
            '210127_DurackandWijffels_V1.0_70yr_steric_1950-2019_0-2000db_210122-205355_beta.nc':'s',
            }
        for obs_type in [
                         '210127_DurackandWijffels_V1.0_70yr_steric_1950-2019_0-2000db_210122-205355_beta.nc',
                         '210201_EN4.2.1.g10_annual_steric_1950-2019_5-5350m.nc',
                         '210201_Ishii17_v7.3_annual_steric_1955-2019_0-3000m.nc',
#                         '141013_DurackandWijffels10_V1.0_50yr_steric_1950-2000_0-2000db.nc',
#                         '141013a_DurackandWijffels10_V1.0_30yr_steric_1970-2000_0-2000db.nc',
#                         '151103_Ishii09_v6.13_annual_steric_1950-2010_0-3000m.nc',
                         ]:
            aux_file = cfg['auxiliary_data_dir']+'/DurackFiles/' + obs_type
            #label = 'Observations'
            label = obs_labels[obs_type]
            marker = obs_markers[obs_type]
            obs_cubes = iris.load_raw(aux_file)
            print(obs_type, ':', obs_cubes)
            if plot_dyn in ['halo_ts']:
                cube = obs_cubes.extract(iris.Constraint(name='steric_height_halo_anom_depthInterp'))[0]
            if plot_dyn in ['thermo_ts']:
                cube = obs_cubes.extract(iris.Constraint(name='steric_height_thermo_anom_depthInterp'))[0]

            # extract integral of surface to 2000m:
            cube = cube[17, :, :]
            obs_dat = {}

            for region in ['Pacific', 'Atlantic']:
                shapefiles = 'includeSO'
                if shapefiles in [None, 'old', 'v4']:
                    shapefile =  cfg['auxiliary_data_dir']+'/shapefiles/IPCC_WGI/IPCC-WGI-reference-'+region+'-v4.shp'
                elif shapefiles == 'includeSO':
                    shapefile =  cfg['auxiliary_data_dir']+'/shapefiles/includeSO/'+region.lower()+'.shp'


                region_cube = extract_shape(
                        cube.copy(),
                        shapefile,
                        )
                areas = iris.analysis.cartography.area_weights(region_cube)

                mean = region_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN, weights=areas)

                obs_dat[region] = mean.data
            print(obs_type, obs_type, ':', obs_dat['Pacific'], obs_dat['Atlantic'])
            if label not in labels:
                plt.scatter(obs_dat['Pacific'], obs_dat['Atlantic'], c='black', marker=marker, label = label)
                labels.append(label)
            else:
                plt.scatter(obs_dat['Pacific'], obs_dat['Atlantic'], c='black', marker=marker)

    if show_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(labels) ==6:
            
            order = [3,4,5,2,0, 1]
        else:
            order = np.arange(len(labels))

        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
             loc = 'lower right', framealpha=0., prop={'size': 7}, markerfirst=False )
        #plt.legend(loc = 'lower right', framealpha=0., prop={'size': 7}, markerfirst=False )

    ax.set_aspect("equal")

    plt.axhline(0, c='k', ls='--')
    plt.axvline(0, c='k', ls='--')

    limits='custom'
    if limits=='custom' and plot_dyn == 'halo_ts':
        ax.set_xlim([-0.5, 1.])
        ax.set_ylim([-1., 0.5])
    elif limits=='custom' and plot_dyn == 'thermo_ts':
        ax.set_xlim([-1, 2.5])
        ax.set_ylim([-1., 2.5])
    else:
        max_value = int(max_value)+1.
        ax.set_xlim([-max_value, max_value])
        ax.set_ylim([-max_value, max_value])

    if plot_dyn == 'halo_ts':
        plt.xlabel('Pacific Halosteric Trend (mm yr'+r'$^{-1}$'+')')
        plt.ylabel('Atlantic Halosteric Trend (mm yr'+r'$^{-1}$'+')')
    if plot_dyn == 'thermo_ts':
        plt.xlabel('Pacific Thermosteric Trend (mm yr'+r'$^{-1}$'+')')
        plt.ylabel('Atlantic Thermosteric Trend (mm yr'+r'$^{-1}$'+')')

    # Saving files:
    if isinstance(subplot, int) and subplot==111:
        imgf = diagtools.get_image_format(cfg)
        path = diagtools.folder([cfg['plot_dir'], 'SLR_Regional_trend_scatter'])
        path += '_'.join([plot_exp, plot_dyn, plot_trend,
                          str(int(time_range[0]))+'-'+ str(int(time_range[1])),
                          'SLR_Regional_trend_scatter'])+imgf
        if show_UKESM:
            path = path.replace(imgf, '_UKESM'+imgf)

        if cfg['write_plots']:
            logger.info('Saving plots to %s', path)
            plt.savefig(path, dpi=200)
        plt.close()
    else:
        return fig, ax




def calc_halo_multimodel_mean(cfg, metadatas, dyn_fns,
        plot_trend = 'detrended',
        plot_dyn = 'halo',
        plot_exp = 'historical',
        plot_region = 'Global',
        time_range=[1950, 2000],
        method = 'dyn_height',
    ):
    """
    Make a multimodel mean halosteric plot.
    """

    time_range_str = '-'.join([str(t) for t in time_range])
    unique_id = [plot_dyn, plot_exp, method, plot_region, 'mean', time_range_str]
    multimodel_mean_fn = diagtools.folder([cfg['work_dir'], 'multimodel_halosteric_map'])
    multimodel_mean_fn += '_'.join(unique_id)+'.nc'

    if os.path.exists(multimodel_mean_fn):
        mean_cube = iris.load_cube(multimodel_mean_fn)
    else:
        cube_list = {}
        datasets = {}
        for (project, dataset, exp, ensemble, dyn_type, region, trend), fn in dyn_fns.items():
            datasets[dataset] = True

        # Calculatge the individual model mean
        for dataset in datasets.keys():
            cube_list[dataset] = {}

            for (project, dataset_itr, exp, ensemble, dyn_type, region, trend), fn in dyn_fns.items():
                print( (project, dataset_itr, exp, ensemble, dyn_type, region, trend))
                if dataset != dataset_itr: continue
                if trend != plot_trend: continue
                if exp != plot_exp: continue
                if dyn_type != plot_dyn: continue
                if region != plot_region: continue

                trend_fn = diagtools.folder([cfg['work_dir'], 'multimodel_halosteric_map'])
                trend_fn += '_'.join([project, dataset_itr, exp, ensemble, dyn_type, region, trend, time_range_str])+'.nc'
                if os.path.exists(trend_fn):
                    cube_list[dataset][exp] = iris.load_cube(trend_fn)

                cube_list[dataset][exp] = iris.load_cube(fn)

                # extract time
                cube_list[dataset][exp] = extract_time(cube_list[dataset][exp], time_range[0], 1, 1, time_range[1],12,31)
                # need to calculate the linear regression here:
                times = diagtools.cube_time_to_float(cube_list[dataset][exp])
                time_arange = np.arange(len(times))
                slopes = {}
                count = 0
                cube_data = cube_list[dataset][exp][0].data
                ndenum = np.ndenumerate(cube_list[dataset][exp][0].data)

                with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                    print('ProcessPoolExecutor: executing')
                    for linreg, index in executor.map(mpi_fit_2D,
                                                      ndenum,
                                                      itertools.repeat(cube_list[dataset][exp].data),
                                                      itertools.repeat(time_arange),
                                                      chunksize=100000):
                        if linreg:
                            if linreg.slope > 1E10 or linreg.intercept>1E10:
                                print('linear regression failed:', linreg)
                                print('slope:', linreg.slope,'intercept', linreg.intercept)
                                print('from time:', np.arange(len(decimal_time)))
                                print('from data:', cube.data[:, index[0], index[1], index[2]])
                                assert 0

                            if count%250000 == 0:
                                print(count, 'linreg:', linreg[0],linreg[1])
                            slopes[index] = linreg.slope
                            count+=1
                #cube_list[dataset][exp] = cube_list[dataset][exp][0]
                cube_list[dataset][exp] = cube_list[dataset][exp].collapsed('time', iris.analysis.MEAN)

                for index, slope in slopes.items():
                    #print(index, slope)
                    cube_data[index[0], index[1]] = slope
                cube_list[dataset][exp].data = cube_data

                print('saving trend file:', trend_fn)
                iris.save(cube_list[dataset][exp], trend_fn)
                if np.ma.is_masked(cube_list[dataset][exp].data.max()):
                    print('Data is all masked:',(project, dataset_itr, exp, ensemble, dyn_type, region, trend))
                    assert 0

            # Calculated all trends, now take single model ensemble mean:
            cube_list[dataset] = [c for exp, c in cube_list[dataset].items()]
            cube_list[dataset] = make_mean_of_cube_list_notime(cube_list[dataset])

            # regrid to a common grid:
            print('Regridding', dataset, trend, exp, region, cube_list[dataset].shape)
            cube_list[dataset] = regrid_to_1x1( cube_list[dataset])

        # Take mean of several cubes
        cube_list = [c for exp, c in cube_list.items()]
        mean_cube = make_mean_of_cube_list_notime(cube_list)

        # Save cube:
        iris.save(mean_cube, multimodel_mean_fn)

    return multimodel_mean_fn


def make_multimodel_halosteric_salinity_trend(cfg, metadatas,
        multimodel_mean_fn,
        plot_trend = 'detrended',
        plot_dyn = 'halo',
        plot_exp = 'historical',
        plot_region = 'Global',
        plot_range = [-2., 2],
        nbins=20,
        time_range=[1950, 2000],
        method = 'Landerer',
        subplot=111,
        fig = None,
    ):
    """
    Make a multimodel mean halosteric plot.

    To make a stand-alone plot, provide no figure and subplot = 111.
    """
    time_range_str = '-'.join([str(t) for t in time_range])
    mean_cube = iris.load_cube(multimodel_mean_fn)

    # Make the plot.
    # Determine image filename
    unique_id = [plot_dyn, plot_exp, method, plot_region, 'mean', time_range_str]

    cmap=diagtools.misc_div
    nspace = np.linspace(plot_range[0], plot_range[1], nbins, endpoint=True)
    #cmap = plt.cm.get_cmap('coolwarm')
    if isinstance(subplot, int) and subplot==111:
        fig = plt.figure()
        title = ' '.join(['CMIP6 Multimodel mean', time_range_str ])

    central_longitude=-160.+3.5
    square = False
    if square:
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
    else:
       proj = ccrs.Robinson(central_longitude=central_longitude)
       mean_cube = mean_cube.intersection(longitude=(central_longitude-180., central_longitude+180.), latitude=(-73., 73.))

    if isinstance(subplot, int):
        ax = fig.add_subplot(subplot, projection=proj)
    else:
        ax=subplot
        plt.sca(ax) # set current axes


    if square:
        extent = [central_longitude-180., central_longitude+180., -73, 73]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    clip = True
    if clip:
        mean_cube.data = np.ma.clip(mean_cube.data, plot_range[0], plot_range[1])

    qplot = iris.plot.contourf(
        mean_cube,
        nspace,
        linewidth=0,
        cmap=cmap,
        extend='neither',
        zmin=plot_range[0],
        zmax=plot_range[1])

    if isinstance(subplot, int) and subplot==111:
        plt.title(title)
        cbar = plt.colorbar(orientation='horizontal')

    try: plt.gca().coastlines()
    except: pass

    ax = add_map_text(ax, 'CMIP6')

    # Saving files:
    if isinstance(subplot, int) and subplot==111:
        if cfg['write_plots']:
            filename = '_'.join(unique_id).replace('/', '_')
            path = diagtools.folder([cfg['plot_dir'], 'multimodel_halosteric_map']) + filename
            path = path.replace(' ', '') + diagtools.get_image_format(cfg)
            logger.info('Saving plots to %s', path)
            plt.savefig(path, dpi=200)
        plt.close()
    else:
        return fig, ax, qplot


def add_map_text(ax, text, spaces='       '):
    """
    Add a small text to a map.
    """
    #ax.text(0., 0., text, fontsize=10)
    #artisttext = AnchoredText(text+'       ',
    #                    loc=4, prop={'size': 12}, frameon=False)
    artisttext = AnchoredText(spaces+text, #+'       ',
                        loc='upper left', prop={'size': 10}, frameon=False)
    ax.add_artist(artisttext)
    return ax


def plot_halo_multipane(
        cfg,
        metadatas,
        slr_fns,
        plot_exp = 'historical',
        #plot_dyn = plot_dyn,
        method= 'Landerer',
        time_range=[1950, 2000],
        show_UKESM=False,
    ):
    """
    Make the halosteric multi pane figure needed for IPCC WG1 chapter 3, fig 3.27

    if do_SLR is true!
    """
    # Create figure
    fig = plt.figure()
    fig.set_size_inches(10, 7)

    axes = {}
    # Obs pane 1
#   #axes[321] = plt.subplot(321)
#   if 1950 in time_range:
 #      obs_files = ['DurackandWijffels10_V1.0_50yr', ]#'DurackandWijffels10_V1.0_30yr']
  # if 1970 in time_range:
   #    obs_files = ['DurackandWijffels10_V1.0_50yr', ]#'DurackandWijffels10_V1.0_30yr']
    #f 1860 in time_range:
     #  obs_files = ['DurackandWijffels10_V1.0_50yr', 'DurackandWijffels10_V1.0_30yr']

    #obs_files = ['DurackandWijffels10_V1.0_50yr', 'DurackandWijffels10_V1.0_30yr', 'Ishii09_v6.13_annual_steric_1950-2010']
    #obs_files = ['Ishii09_v6.13_annual_steric_1950-2010', 'DurackandWijffels_GlobalOceanChanges_1950-2020_210111_10_18_04_beta']
    #obs_files = ['DurackandWijffels10_V1.0_50yr', 'Ishii09_v6.13_annual_steric_1950-2010']
    #obs_files = ['Ishii09_v6.13_annual_steric_1950-2010','DurackandWijffels_GlobalOceanChanges_19500101-20191231__210122-205355_beta.nc']
    # obs_files = ['210127_DurackandWijffels',  'Ishii09_v6.13_annual_steric_1950-2010']
    obs_files = ['210127_DurackandWijffels_V1.0_70yr_steric_1950-2019_0-2000db_210122-205355_beta.nc',
                 '210201_EN4.2.1.g10_annual_steric_1950-2019',
                 '210201_Ishii17_v7.3_annual_steric_1955-2019_0-3000m.nc', ]


    reverse = True
    if reverse and len(obs_files) == 2:
        gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.06)
        gs0 = gs[0].subgridspec(2, 1, hspace=0.35) # scatters
        gs1 = gs[1].subgridspec(3, 1, hspace=0.06 ) # maps
        #scatters
        scatter1=fig.add_subplot(gs0[0,0])
        scatter2=fig.add_subplot(gs0[1,0])

        central_longitude=-160.+3.5
        proj = ccrs.Robinson(central_longitude=central_longitude)
        ax0 = fig.add_subplot(gs1[0, 0], projection=proj)
        ax1 = fig.add_subplot(gs1[1, 0], projection=proj)
        ax2 = fig.add_subplot(gs1[2, 0], projection=proj)
        subplots = [ax0, ax1]
        cmip_subplots = [ax2,]
        cbar_axes = [ax0, ax1, ax2]
        fig.set_size_inches(10, 7)

    if reverse and len(obs_files) == 3:
        fig.set_size_inches(9,  7)

        gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 1.00], wspace=0.)
        gs0 = gs[0].subgridspec(2, 1, hspace=0.35) # scatters
        gs1 = gs[1].subgridspec(4, 1, hspace=0.06 ) # maps
        #scatters
        scatter1=fig.add_subplot(gs0[0,0])
        scatter2=fig.add_subplot(gs0[1,0])

        central_longitude=-160.+3.5
        proj = ccrs.Robinson(central_longitude=central_longitude)
        ax0 = fig.add_subplot(gs1[0, 0], projection=proj)
        ax1 = fig.add_subplot(gs1[1, 0], projection=proj)
        ax2 = fig.add_subplot(gs1[2, 0], projection=proj)
        ax3 = fig.add_subplot(gs1[3, 0], projection=proj)

        subplots = [ax0, ax1, ax2]
        cmip_subplots = [ax3,]
        cbar_axes = [ax0, ax1, ax2, ax3]

#            gs = matplotlib.gridspec.GridSpec(6, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.450, wspace =0.0600)
#            central_longitude=-120.
#            proj = ccrs.Robinson(central_longitude=central_longitude)
#            ax0 = fig.add_subplot(gs[0:2, 1], projection=proj)
#            ax1 = fig.add_subplot(gs[2:4, 1], projection=proj)
#            ax2 = fig.add_subplot(gs[4:6, 1], projection=proj)
#            subplots = [ax0, ax1]
#            cmip_subplots = [ax2,]
#            scatter1=fig.add_subplot(gs[:3, 0])
#            scatter2=fig.add_subplot(gs[3:, 0])
        # else:
        #     gs = matplotlib.gridspec.GridSpec(6, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.10, wspace =0.25)
        #     central_longitude=-160.+3.5
        #     proj = ccrs.Robinson(central_longitude=central_longitude)
        #     ax0 = fig.add_subplot(gs[0:2, 0], projection=proj)
        #     ax1 = fig.add_subplot(gs[2:4, 0], projection=proj)
        #     ax2 = fig.add_subplot(gs[4:6, 0], projection=proj)
        #     subplots = [ax0, ax1]
        #     cmip_subplots = [ax2,]
        #     scatter1=fig.add_subplot(gs[:3, 1])
        #     scatter2=fig.add_subplot(gs[3:, 1])

    #plot_range=[-1.65, 1.65]
    plot_range=[-1.625, 1.625]
    nbins=14
    cbar_ticks = np.arange(-1.5, 1.75, 0.25)
#    plot_range=[-2.05, 2.05]

    for sbp, obs_file in zip(subplots, obs_files ):
        fig, axes[sbp] = plot_halo_obs_mean(
            cfg,
            metadatas,
            plot_dyn = 'halo',
            subplot=sbp,
            depth_range='2000m',
            plot_range=plot_range,
            nbins=nbins,
            obs_file=obs_file,
            fig=fig,
#            ax=sbp,
        )

    # model pane ( C3)
    # Load data
    multimodel_mean_fn = calc_halo_multimodel_mean(
        cfg, metadatas, slr_fns,
        plot_trend = 'detrended',
        plot_dyn = 'halo',
        plot_exp = plot_exp,
        plot_region = 'Global',
        time_range = time_range,
    )
    for pane in cmip_subplots:
        # make plot: (c3)
        fig, axes[pane], qplot = make_multimodel_halosteric_salinity_trend(
            cfg,
            metadatas,
            multimodel_mean_fn,
            plot_trend = 'detrended',
            plot_dyn =  'halo',
            plot_exp = plot_exp,
            plot_region = 'Global',
            time_range = time_range,
            plot_range=plot_range,
            nbins=nbins,
            method = method,
            fig=fig,
            subplot = pane,
        )
    #qplot.keys()
#    cmap='coolwarm'
#    nspace = np.linspace(-2., 2, 15, endpoint=True)
#    mapable = matplotlib.cm.ScalarMappable(norm=nspace,cmap=cmap)
#    if reverse:
#        fig.colorbar(qplot, ax=cbar_axes, location='right',label='Trend, mm yr'+r'$^{-1}$', ticks=cbar_ticks)
#    else:
#        fig.colorbar(qplot, ax=cbar_axes, location='left',label='Trend, mm yr'+r'$^{-1}$', ticks=cbar_ticks)

    #rhs:
    # Halosteric trend scatter:
    fig, axes[scatter1] = plot_slr_regional_scatter(cfg, metadatas, slr_fns,
            plot_exp = plot_exp,
            plot_dyn = 'halo_ts',
            method=method,
            time_range=time_range,
            fig=fig,
            subplot = scatter1,
            show_UKESM=show_UKESM,
            show_legend=False
        )
    # Thermosteric trend scatter
    fig, axes[scatter2] = plot_slr_regional_scatter(cfg, metadatas, slr_fns,
            plot_exp = plot_exp,
            plot_dyn = 'thermo_ts',
            method=method,
            time_range=time_range,
            fig=fig,
            subplot = scatter2,
            show_UKESM=show_UKESM,
            show_legend=True,
        )

#    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
#    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
#    cmap='coolwarm'
#    nspace = np.linspace(-2., 2, 15, endpoint=True)
#    cbar = plt.colorbar(cax=cax,cmap=cmap, )
#    cbar.set_clim(-2.0, 2.0)

    #plt.tight_layout()

    time_range_str = '-'.join([str(t) for t in time_range])

    fig.suptitle('Halosteric and thermosteric sea level trends')

    if reverse:
        fig.colorbar(qplot, ax=cbar_axes, location='right',label='Trend, mm yr'+r'$^{-1}$', ticks=cbar_ticks)
    else:
        fig.colorbar(qplot, ax=cbar_axes, location='left',label='Trend, mm yr'+r'$^{-1}$', ticks=cbar_ticks)

    #plt.tight_layout()

    # Determine image filename
    filename = '_'.join(['halosteric_multipane', plot_exp, time_range_str ]).replace('/', '_')
    if show_UKESM:
        filename+='_UKESM'
    path = diagtools.folder([cfg['plot_dir'], 'halosteric_multipane']) + filename
    path = path.replace(' ', '') + diagtools.get_image_format(cfg)

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path, dpi=200)
    plt.close()


def plot_halo_obs_mean(
        cfg, metadatas,
        plot_dyn = 'halo',
        subplot=111,
        depth_range='2000m',
        plot_range=[-2., 2],
        nbins=20,
        obs_file='DurackandWijffels10_V1.0_50yr',
#        ax = None,
        fig = None
        ):
    """
    make the observational halosteric observation plot.
    """
    # Load the observational data.
    if obs_file=='DurackandWijffels10_V1.0_50yr':
        aux_file = cfg['auxiliary_data_dir']+'/DurackFiles/141013_DurackandWijffels10_V1.0_50yr_steric_1950-2000_0-2000db.nc'
#       legend_txt = 'D&W 1950-2000'
        legend_txt = 'D&W'

    if obs_file=='DurackandWijffels10_V1.0_30yr':
        aux_file = cfg['auxiliary_data_dir']+'/DurackFiles/141013a_DurackandWijffels10_V1.0_30yr_steric_1970-2000_0-2000db.nc'
        legend_txt = 'D&W 1970-2000'

    if obs_file=='Ishii09_v6.13_annual_steric_1950-2010':
        aux_file = cfg['auxiliary_data_dir']+'/DurackFiles/151103_Ishii09_v6.13_annual_steric_1950-2010_0-3000m.nc'
        #legend_txt = 'Ishii 1950-2010'
        legend_txt = 'Ishii' #1950-2010'

    if obs_file in ['210127_DurackandWijffels', '210127_DurackandWijffels_V1.0_70yr_steric_1950-2019_0-2000db_210122-205355_beta.nc']:
        aux_file = cfg['auxiliary_data_dir']+'/DurackFiles/210127_DurackandWijffels_V1.0_70yr_steric_1950-2019_0-2000db_210122-205355_beta.nc'
        legend_txt = 'D&W'

    if obs_file == '210201_EN4.2.1.g10_annual_steric_1950-2019':
        aux_file = cfg['auxiliary_data_dir']+'/DurackFiles/210201_EN4.2.1.g10_annual_steric_1950-2019_5-5350m.nc'
        legend_txt = 'EN4'

    if obs_file == '210201_EN4.2.1.g10_annual_steric_1970-2019':
        aux_file = cfg['auxiliary_data_dir']+'/DurackFiles/210201_EN4.2.1.g10_annual_steric_1970-2019_5-5350m.nc'
        legend_txt = 'EN4'

    if obs_file == '210201_Ishii17_v7.3_annual_steric_1955-2019_0-3000m.nc':
        aux_file = cfg['auxiliary_data_dir']+'/DurackFiles/210201_Ishii17_v7.3_annual_steric_1955-2019_0-3000m.nc'
        legend_txt = 'Ishii'


    print('opening:', aux_file)
    obs_cubes = iris.load_raw(aux_file)
    if plot_dyn == 'halo':
        cube = obs_cubes.extract(iris.Constraint(name='steric_height_halo_anom_depthInterp'))[0]
    if plot_dyn in ['thermo']:
        cube = obs_cubes.extract(iris.Constraint(name='steric_height_thermo_anom_depthInterp'))[0]

    # extract integral of surface to 2000m:
    if depth_range=='2000m':
        cube = cube[17, :, :]
    #(and FYI 14 = 700m and 12 = 300m)

    cmap=diagtools.misc_div
    # cmap = plt.cm.get_cmap(cmap)
    nspace = np.linspace(plot_range[0], plot_range[1], nbins, endpoint=True)

    if isinstance(subplot, int) and subplot==111:
        fig = plt.figure()
        title = ' '.join(['Observational mean', legend_txt])

    central_longitude=-160.+3.5
    square = False
    if square:
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
    else:
       proj = ccrs.Robinson(central_longitude=central_longitude)
       cube = cube.intersection(longitude=(central_longitude-180., central_longitude+180.), latitude=(-73., 73.))

    if isinstance(subplot, int):
        ax = fig.add_subplot(subplot, projection=proj)
    else:
        ax = subplot
        plt.sca(ax) # set current axes

    if square:
        extent = [central_longitude-180., central_longitude+180., -73, 73]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax = add_map_text(ax, legend_txt)

    clip = True
    if clip:
       cube.data = np.ma.clip(cube.data, plot_range[0], plot_range[1])

    qplot = iris.plot.contourf(
        cube,
        nspace,
        linewidth=0,
        cmap=cmap,
        extend='neither',
        zmin=nspace.min(),
        zmax=nspace.max())

    if subplot==111:
        plt.title(title)
        cbar = plt.colorbar(orientation='horizontal')

    try: plt.gca().coastlines()
    except: pass

    # Saving files:
    if subplot==111:
        if cfg['write_plots']:
            filename = '_'.join(obs_file).replace('/', '_')
            keys = [obs_file, 'observational', plot_dyn, depth_range, 'map']
            path = diagtools.folder([cfg['plot_dir'], 'observational_steric_maps']) + '_'.join(keys)
            path = path.replace(' ', '') + diagtools.get_image_format(cfg)
            logger.info('Saving plots to %s', path)
            plt.savefig(path, dpi=200)
        plt.close()
    else:
        return fig, ax


def plot_slr_full3d_ts(cfg, metadata, dyn_files, area_fn, trend):
    """
    8 pane picture, with
    """
    exp = metadata['exp']
    dataset = metadata['dataset']
    ensemble = metadata['ensemble']
    project = metadata['project']

    steric_types = ['total', 'thermo', 'halo']
    cubes = {}
    print('---------\n', project, dataset, exp, ensemble)
    for dyn_type, fn in dyn_files.items():
        cubes[dyn_type] = iris.load_cube(fn)
        cubes[dyn_type].data = np.ma.masked_invalid(cubes[dyn_type].data)

        print(dyn_type, ':',cubes[dyn_type].data.shape, 'mean:', cubes[dyn_type].data.mean())#s.path.basename fn)
    panes = {'1971-2018':424, '2005-2018':428, '1850-1900':423 , '1995-2014':426,
             '1985-2014':425, '2004-2018':427, 'fullhistorical':422, 'piControl':421}

    yranges = {
       '1971-2018': [1971, 2018 + 1],
       '2005-2018': [2005, 2018 + 1],
       '1850-1900': [1850, 1900 + 1],
       '1995-2014': [1995, 2014 + 1],
       '1985-2014': [1985, 2014 + 1],
       '2004-2018': [2004, 2018 + 1],
       'fullhistorical': [1850, 2015 + 1],
       'piControl': [1850, 2015 + 1],
       }
    area_cube = iris.load_cube(area_fn)
    area_cube.data = np.ma.masked_invalid(area_cube.data)

    fig = plt.figure()
    fig.set_size_inches(12, 8)
    for clim_type, sbp in panes.items():
        ax = fig.add_subplot(sbp)
        for steric_type in steric_types:
            steric_cube = cubes[steric_type].copy()
            times = diagtools.cube_time_to_float(steric_cube)

            clim_tile = da.tile(cubes[clim_type].core_data(), [steric_cube.shape[0], 1, 1])
            grid_areas = da.tile(area_cube.core_data(), [steric_cube.shape[0], 1, 1])
            print('calculating:', dataset, steric_type, clim_type)
            print(steric_cube.data.shape, clim_tile.shape, grid_areas.shape)
            steric_cube.data = steric_cube.data - clim_tile
            steric_cube = steric_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN, weights=grid_areas)

            plt.plot(times, -1.*steric_cube.data, label = steric_type.title())
        plt.axhline(0., c = 'k', ls='-', lw=0.5)
        plt.axhline(0.5, c = 'k', ls=':', lw=0.5)
        plt.axhline(-0.5, c = 'k', ls=':', lw=0.5)

        if clim_type == 'piControl':
            ax.axvspan(yranges[clim_type][0], yranges[clim_type][1], alpha=0.35, color='red')
        else:
            ax.axvspan(yranges[clim_type][0], yranges[clim_type][1], alpha=0.35, color='black')

        ax.text(.5,.82, clim_type,
            horizontalalignment='center',
            transform=ax.transAxes)

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel('Year')
    plt.ylabel('Steric anomaly, mm')
    plt.suptitle(' '.join([project, dataset, exp, ensemble, trend, 'SLR']))
    path = diagtools.folder([cfg['plot_dir'], 'SLR_timeseries_3d'])
    path += '_'.join([project, dataset, exp, ensemble, trend, 'slr_timeseries_3d'])+diagtools.get_image_format(cfg)
    print('Saving figure:', path)
    plt.savefig(path)
    plt.close()




def calc_dyn_timeseries(cfg, metadatas, dyn_fn, areacella_fn,
        project, dataset, exp, ensemble, slr_type, region,
        trend,
        method = 'dyn_height',
        shapefiles = 'includeSO'):
    """
    Calculate dynamic time series.
    """
    if method == 'dyn_height':
        work_dir = diagtools.folder([cfg['work_dir'], 'dyn_height_ts'])
    elif method =='Landerer':
        work_dir = diagtools.folder([cfg['work_dir'], 'Landerer_slr_ts'])

    if shapefiles is None or region in ['Global', ]:
        slr_ts_fn = work_dir + '_'.join([project, dataset, exp, ensemble, slr_type, region, trend, 'timeseries', ])+'.nc'
    else:
        slr_ts_fn = work_dir + '_'.join([project, dataset, exp, ensemble, slr_type, region, trend, 'timeseries', shapefiles])+'.nc'

    if os.path.exists(slr_ts_fn):
        return slr_ts_fn

    # Load data and calculate average.
    dyn_cube = iris.load_cube(dyn_fn)
    area_cube = iris.load_cube(areacella_fn)
    dyn_cube.data = np.ma.masked_invalid(dyn_cube.data)
    area_cube.data = np.ma.masked_invalid(area_cube.data)

    # coord_names = [c.var_name for c in dyn_cube.coords()]
    # print('calc_dyn_timeseries:', dyn_fn, '\ncoord_names:', coord_names)
    if region in ['Pacific', 'Atlantic']:
        if shapefiles is None:
            shapefile =  cfg['auxiliary_data_dir']+'/shapefiles/IPCC_WGI/IPCC-WGI-reference-'+region+'-v4.shp'
        elif shapefiles == 'includeSO':
            shapefile =  cfg['auxiliary_data_dir']+'/shapefiles/includeSO/'+region.lower()+'.shp'

        dyn_cube = extract_shape(
            dyn_cube,
            shapefile,
            )
        area_cube = extract_shape(
            area_cube,
            shapefile,
            )
#        print(dyn_cube)
#        single_pane_map_plot(
#                cfg,
#                metadatas[dyn_fn],
#                dyn_cube[0, ],
#                key='dynheight_ohc'+region
#                )
       #assert 0

    elif region == 'Global':
        pass
    else:
        print('calc_dyn_timeseries, region not recognised:', region)
        assert 0

    ndim = dyn_cube.data.ndim
    if ndim == 3:
        grid_areas = da.tile(area_cube.core_data(), [dyn_cube.shape[0], 1, 1])
    else:
        grid_areas = area_cube.data
    print(dataset, grid_areas.shape, area_cube.shape, dyn_cube.shape)

    dyn_cube = dyn_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN, weights=grid_areas)

    # Save NetCDF
    iris.save(dyn_cube, slr_ts_fn)

    print('calc_slr_timeseries:', dataset, exp, ensemble, dyn_cube.data, region)
    # Make plot
    if ndim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        times = diagtools.cube_time_to_float(dyn_cube)
        plt.plot(times, dyn_cube.data, color='red')
        plt.title(' '.join([project, dataset, exp, ensemble, slr_type, trend, region, str(shapefiles),]))

        plt.xlabel('Year')
        plt.ylabel('Change in Sea Level, mm')
        #plt.axhline(0., c = 'k', ls=':' )

        path = diagtools.folder([cfg['plot_dir'], 'dyn_height_timeseries', method, region])
        path += '_'.join([project, dataset, exp, ensemble, slr_type, 'timeseries', region, str(shapefiles)])+diagtools.get_image_format(cfg)
        print('Saving figure:', path)
        plt.savefig(path)
        plt.close()

    return slr_ts_fn



def calc_slr_timeseries(cfg, slr_fn, areacella_fn, project, dataset, exp, ensemble, region, slr_type):
    """
    Calculate SLR time series.
    """
    work_dir = diagtools.folder([cfg['work_dir'], 'SLR'])
    slr_ts_fn = work_dir + '_'.join([project, dataset, exp, ensemble, region, slr_type, 'timeseries'])+'.nc'
    if os.path.exists(slr_ts_fn):
        return slr_ts_fn

    # Load data and calculate average.
    slr_cube = iris.load_cube(slr_fn)
    area_cube = iris.load_cube(areacella_fn)

    slr_cube.data = np.ma.masked_invalid(slr_cube.data)
    area_cube.data = np.ma.masked_invalid(area_cube.data)

    grid_areas = da.tile(area_cube.core_data(), [slr_cube.shape[0], 1, 1])
    slr_cube = slr_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN, weights=grid_areas)

    # Save NetCDF
    iris.save(slr_cube, slr_ts_fn)

    print('calc_slr_timeseries:', dataset, exp, ensemble, slr_cube.data)
    # Make plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    times = diagtools.cube_time_to_float(slr_cube)
    plt.plot(times, slr_cube.data, color='red', )
    plt.title(' '.join([project, dataset, exp, ensemble, slr_type, 'timeseries']))

    plt.xlabel('Year')
    plt.ylabel('Change in Sea Level, mm')
    plt.axhline(0., c = 'k', ls=':' )

    path = diagtools.folder([cfg['plot_dir'], 'SLR_timeseries'])
    path += '_'.join([project, dataset, exp, ensemble, slr_type, region, 'timeseries'])+diagtools.get_image_format(cfg)
    print('Saving figure:', path)
    plt.savefig(path)
    plt.close()

    return slr_ts_fn


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
        return output_ohc_fn
        cube1 = iris.load_cube(output_ohc_fn)
        for t in [0, -1]:
            single_pane_map_plot(
                    cfg,
                    metadatas[thetao_fn],
                    cube1[t, 0],
                    key='OHC_full_'+trend
                    )

        return output_ohc_fn
    else:
       print('these files should exist now')
#       return None
#       assert 0

    # Load netcdf files
    thetao_cube = iris.load_cube(thetao_fn)
    so_cube = iris.load_cube(so_fn)
    vol_cube = iris.load_cube(volcello_fn)

    # Fix depth cubes.
    thetao_cube = fix_depth(thetao_cube)
    so_cube = fix_depth(so_cube)
    vol_cube = fix_depth(vol_cube)

    # Load coordinates
    lats = thetao_cube.coord('latitude').points
    lons = thetao_cube.coord('longitude').points
    depths = -1.*np.abs(thetao_cube.coord(axis='z').points) # depth is negative here.
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

    if vol_cube.ndim not in [3, 4]:
        print("Looks like the volume has a weird shape:",vol_cube.ndim, vol_cube.data.shape, vol_cube)
        assert 0
    if vol_cube.ndim == 4 and vol_cube.shape != thetao_cube.shape:
        print("Volume file is not the same shape as temperature:")
        print('Volune file:',volcello_fn)
        print('Temperature file:', thetao_fn)
        print('Salinity file:', so_fn)
        print(dataset, thetao_cube.shape, 'so:', so_cube.shape, vol_cube.shape, depths.shape)
        vol_cube = vol_cube.collapsed([vol_cube.coord('time'),], iris.analysis.MEAN)

    # 2D!
    print(dataset, thetao_cube.shape, 'so:', so_cube.shape, vol_cube.shape, depths.shape)
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

            sal = so_cube.data[t,z]
            temp = thetao_cube.data[t,z]

            # 4.) Calculate the pressure at each gridpoint: p=gsw_p_from_z(z,lat).
            pressure = gsw.conversions.p_from_z(depth, lat) # dbar


            # Confirm that we use absolute salininty & conservative temperature
            sal, temp = step_4and5(dataset, lon, lat, pressure, sal, temp)

            # 6) Calculation of "heat content"
            # 6a) The global ocean heat content is interpreted to be calculated as the volume integral of the product of in situ density, ρ , and potential enthalpy, h0 (with reference sea pressure of 0 dbar).
            # 6b) The in situ density is calculated using gsw_rho(SA,CT,p).  Here the actual pressure at the target depth is used (i.e., the mass of the water).
            # 6c) The *surface referenced* enthalpy should be calculated using gsw_enthalpy(SA,CT,0).  Note that here the 0 dbar pressure is critical to arrive at the surface value, which is the value of enthalpy and absolute salinity that is available for exchange with the atmosphere.
            # 6d) The product of the in situ density times the surface-referenced enthalpy is the relevant energy quantity: gsw_rho(SA,CT,p)*gsw_enthalpy(SA,CT,0)
            # 6e) For the anomalous energy, which is what we want, we calculate gsw_rho(SA,CT,p)*gsw_enthalpy(SA,CT,0)-gsw_rho(<SA>,<CT>,p)*gsw_enthalpy(<SA>,<CT>,0).
            # 6f) integrate the surface-referenced enthalpy times rho, i.e., the previous line gives the 3D integrand, and then we want to integrate it from the bottom up.  (Brodie & I would like the vertical integral done first, so we can make maps, Chps 2, 3 probably only want the global integral?)

            # 6b) The in situ density is calculated using gsw_rho(SA,CT,p).  Here the actual pressure at the target depth is used (i.e., the mass of the water).
            rho = gsw.density.rho(sal, temp, pressure) #  kg/ m3

            # 6c) The *surface referenced* enthalpy should be calculated using gsw_enthalpy(SA,CT,0).  Note that here the 0 dbar pressure is critical to arrive at the surface value, which is the value of enthalpy and absolute salinity that is available for exchange with the atmosphere.
            enthalpy = gsw.energy.enthalpy(sal, temp, 0.)

            # 6d) The product of the in situ density times the surface-referenced enthalpy is the relevant energy quantity: gsw_rho(SA,CT,p)*gsw_enthalpy(SA,CT,0)
            cell_energy = enthalpy * rho *vol
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
    print('Saving OHC file:', output_ohc_fn)
    iris.save(cube, output_ohc_fn)

    for t in [0, -1]:
        single_pane_map_plot(
                cfg,
                metadatas[thetao_fn],
                cube[t, 0],
                key='OHC_full_'+trend
                )
    return output_ohc_fn


def volume_integrated_plot(cfg, metadata, ohc_fn, area_fn):
    """
    Make a plot of the final year in the ohc data.
    """
    short_name = metadata['short_name']
    dataset = metadata['dataset']
    ensemble = metadata['ensemble']
    project = metadata['project']
    exp = metadata['exp']


    # Determine image filename
    unique_id = [dataset, exp, ensemble, short_name] #ear ]
    path = diagtools.folder([cfg['plot_dir'], 'volume_integrated_plot']) + '_'.join(unique_id)
    path = path.replace(' ', '') + diagtools.get_image_format(cfg)
    if os.path.exists(path): return

    cube = iris.load_cube(ohc_fn)
    area = iris.load_cube(area_fn)

    t1 = cube[-10:].copy()
    t1 = t1.collapsed([t1.coord('time'),], iris.analysis.MEAN)
    t2 = cube[:10].copy()
    t2 = t2.collapsed([t2.coord('time'),], iris.analysis.MEAN)

    cube = t1 - t2
    cube = cube.collapsed([cube.coord(axis='z'),],
                          iris.analysis.SUM)
    cube.data = cube.data / area.data

    cmap='viridis'
    nspace = np.linspace(
         cube.data.min(), cube.data.max(), 20, endpoint=True)

    title = ' '.join(unique_id)
    print('volume_integrated_plot', unique_id, nspace, [cube.data.min(), cube.data.max()], cube.data.shape)
    add_map_subplot(111, cube, nspace, title=title,cmap=cmap)
    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path, dpi=200)
    plt.close()


def SLR_multimodel_plot(cfg, metadatas, slr_fns, plot_dataset = 'all'):
    """
    SLR_multimodel_plot: multimodel SLR time series plot
    """

    path = diagtools.folder([cfg['plot_dir'], 'SLR_multimodel_SLR']) + 'SLR_multimodel_SLR_'+plot_dataset
    path = path.replace(' ', '') + diagtools.get_image_format(cfg)

    datasets = list({index[1]:True for index in slr_fns.keys()}.keys())
    datasets = sorted(datasets)
    color_dict = {dataset:c for dataset, c in zip(datasets, plt.cm.viridis(np.linspace(0,1,len(datasets))))}

    fig = plt.figure()
    fig.set_size_inches(10, 7)
    axes= {}
    subplots = {'slr_total_ts':311, 'slr_thermo_ts':312, 'slr_halo_ts':313}
    subplot_ylabel = {311: 'Total, mm', 312: 'Thermosteric, mm', 313: 'Halosteric, mm'}

    for sbp_slr, subplot in subplots.items():
        plt.subplot(subplot)
        plt.ylabel(subplot_ylabel[subplot])
        print('plotting slr ts:', sbp_slr, subplot, plot_dataset, len(slr_fns))
        #assert 0

        for (project, dataset, exp, ensemble, slr_type, trend), slr_fn in slr_fns.items():
            print(sbp_slr, slr_type, (project, dataset, exp, ensemble, slr_type, trend))
            if sbp_slr != slr_type:
                continue

            print('plot?', project, dataset, exp, ensemble, slr_type, sbp_slr, times, data)

            if plot_dataset == 'all':
                pass
            elif plot_dataset != dataset:
                continue

            if trend == 'detrended':
                ls = '-'
            if trend == 'intact':
                ls = ':'

            cube = iris.load_cube(slr_fn)
            times = diagtools.cube_time_to_float(cube)
            data = np.ma.masked_invalid(cube.data)

            print('plot:', project, dataset, exp, ensemble, slr_type, sbp_slr, times, data)
            plt.plot(times, data, c = color_dict[dataset], ls=ls)

    if plot_dataset == 'all':
        plt.suptitle('Steric Sea Level Rise')
    else:
        plt.suptitle('Steric Sea Level Rise - '+plot_dataset)

    # Legend section
    if len(datasets) <=5:
        axleg = plt.axes([0.0, 0.00, 0.9, 0.10])
    else:
        axleg = plt.axes([0.0, 0.00, 0.9, 0.15])
    axleg.axis('off')
    for dataset in datasets:
        if plot_dataset == 'all':
            pass
        elif plot_dataset != dataset:
            continue

        axleg.plot([], [], c=color_dict[dataset], lw=2, ls='-', label=dataset)

    legd = axleg.legend(
            loc='upper center',
            ncol=5,
            prop={'size': 10},
            bbox_to_anchor=(0.5, 0.5,),
            fontsize=12)
    legd.draw_frame(False)
    legd.get_frame().set_alpha(0.)

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path, dpi=200)
    plt.close()
    if plot_dataset == 'all': assert 0


def SLR_map_plot(cfg, metadata, dyn_fn, clim_fn, time_range, method='dyn_height', keys = ['a', 'b']):
    """
     Make a plot of the SLR final decade.
    """
    cube = iris.load_cube(dyn_fn)
    if method =='dyn_height':
        clim_cube = iris.load_cube(clim_fn)

        try:
            cube = extract_time(cube, time_range[0], 1, 1, time_range[1], 12, 31)
            cube = cube.collapsed([cube.coord('time'),], iris.analysis.MEAN)
            time_str = str(int(time_range[0])) + ' - ' + str(int(time_range[1]))
            keys.append(time_str)
            clim_data = da.tile(clim_cube.core_data(), [cube.shape[0], 1, 1])
        except:
            time_str = keys[4]
            print('No time here:',dyn_fn)
            clim_data = clim_cube.data
        cube.data = -1.*(cube.data - clim_data)
    elif method =='Landerer':
        cube = extract_time(cube, time_range[0], 1, 1, time_range[1], 12, 31)
        cube = cube.collapsed([cube.coord('time'),], iris.analysis.MEAN)
        time_str = str(int(time_range[0])) + ' - ' + str(int(time_range[1]))
        keys.append(time_str)

    # Determine image filename
    path = diagtools.folder([cfg['plot_dir'], 'SLR_map_plots']) + '_'.join(keys)
    path = path.replace(' ', '') + diagtools.get_image_format(cfg)

    cmap=diagtools.misc_div
    max_val = np.max(np.abs([cube.data.min(), cube.data.max()]))
    nspace = np.linspace(
        -max_val, max_val, 22, endpoint=True)

    title = ' '.join(keys)

    print('SLR_map_plot', keys, nspace, [cube.data.min(), cube.data.max()], cube.data.shape)

    add_map_subplot(111, cube, nspace, title=title,cmap=cmap)
    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path, dpi=200)
    plt.close()


def mpi_detrend(iter_pack, cubedata, decimal_time, slopes, intercepts):
    index, _ = iter_pack
    data = cubedata[:, index[0], index[1], index[2]]
    if np.ma.is_masked(data.max()):
        return [], index

    line = [(t * slopes[index]) + intercepts[index] for t in np.arange(len(decimal_time))]
    return index, np.array(line)


def calculate_volume_weighted_mean(cfg, metadata, detrended_fn, volcello_fn, trend = 'detrended'): #max_depth=10000. ):
    """
    Calculate the volume weighted mean.
    """
    exp = metadata['exp']
    short_name = metadata['short_name']
    dataset = metadata['dataset']
    ensemble = metadata['ensemble']
    project = metadata['project']

    work_dir = diagtools.folder([cfg['work_dir'], 'vw_timeseries'])
    keys = [project, dataset, exp, ensemble, short_name, trend, 'volume', 'weighted', 'mean']
    output_fn = work_dir + '_'.join(keys)+'.nc'

    img_path = diagtools.folder([cfg['plot_dir'], 'vw_timeseries', short_name,])
    img_keys = '_'.join(keys)
    img_path = img_path + img_keys + diagtools.get_image_format(cfg)

    if os.path.exists(output_fn):
        if not os.path.exists(img_path):
            single_timeseries(output_fn, img_path, keys)
        return output_fn

    print('calculate_volume_weighted_mean', output_fn)
    cube = iris.load_cube(detrended_fn)
    vm_cube =  volume_statistics(
        cube,
        'mean',
        fx_variables={'volcello': detrended_fn})

    iris.save(vm_cube, output_fn)
    single_timeseries(output_fn, img_path, keys)
    return output_fn



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
        return output_fn
        cube = iris.load_cube(filename)
        detrended = iris.load_cube(output_fn)
        make_difference_plots(
            cfg,
            metadatas[filename],
            detrended,
            cube,
            )
        return output_fn
    print ('detrend_from_PI: loading from', trend_shelve)

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


    parrallel = False
    if not parrallel:
        for index, arr in np.ndenumerate(dummy[0]):
            if np.ma.is_masked(arr): continue
            data = cube.data[:, index[0], index[1], index[2]]
            slope = slopes.get(index, False)
            intercept = intercepts.get(index, False)
            if count%100000 == 0:
                print('dedrifting:', index, data.max(),  slope, intercept)
            if np.ma.is_masked(data.max()): continue

            if not count%250000:
                print(count, index, 'detrending')

            # Unable to detrend places where there's not a fit
            #if slopes.get(index, False) == False:
            #    continue
            #print(count, index, 'detrending', slopes.get(index, False), intercepts.get(index, False) )
            if not slopes.get(index, False):
                print('Failure:', filename, index)
            # ne = [(t * slopes[index]) + intercepts[index] for t in np.arange(len(decimal_time))]
            line = [(t * slope) + intercept for t in np.arange(len(decimal_time))]

            dummy[:, index[0], index[1], index[2]] = np.array(line)
            count+=1
    else:
        # parrlel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            print('ProcessPoolExecutor: executing detrending mpi')
            # iter_pack, cubedata, decimal_time, slopes, intercepts
            ndenum = np.ndenumerate(dummy)

            for dtline, index in executor.map(mpi_detrend,
                                              ndenum,
                                              itertools.repeat(cube.data),
                                              itertools.repeat(decimal_time),
                                              itertools.repeat(slopes),
                                              itertools.repeat(intercepts),
                                              ): #chunksize=1000000):
                if dtline:
                    if count % 25000 == 0:
                        print(count, 'detrend')
                    dummy[:, index[0], index[1], index[2]] = dtline
                    count += 1

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

def fix_depth(cube):
    z = cube.coord(axis='Z')
    z.var_name = 'depth'
    z.long_name='ocean depth coordinate'
    z.standard_name='depth'
    z.attributes={'positive': 'down'}

    if str(z.units).lower() in ['cm', 'centimeters']:
        z.units = cf_units.Unit('m')

        if np.abs(z.points.max()) > 200000:
            z.points = z.points/100.

        try:
            if np.abs(z.bounds.max()) > 200000:
                z.bounds = z.bounds/100.
        except: pass

    print(cube, z)
    return cube


def calculate_multi_model_mean(cfg, metadatas, detrended_ncs,
    master_project = '',
    master_short_name = '',
    master_exp = '',
    trend = 'detrended',
    time_range = 1950):
    """
    Calculated the multimodel means

    check whethger the output exists already
    if so, do nothing

    for each model:
        for each model ensemble:
            first extract the requires yeard,
            take the mean over the time axis
        Take the multi ensemble member mean
        regrid to 1x1

    Once that is done, take the multi model mean.
    save the netcdf
    make a plot.
    """
    if isinstance(time_range, list):
        time_range_str = '-'.join(str(t) for t in time_range)
    else:
        time_range_str = str(time_range)
    work_dir = diagtools.folder([cfg['work_dir'], 'multi_model_mean'])
    out_fn  = work_dir + '_'.join(['multi_model_mean', master_project, master_short_name, master_exp, trend, time_range_str])+'.nc'
    print('Preparing:', out_fn)
    if os.path.exists(out_fn):
        return out_fn

    # detrended_ncs.keys (project, dataset, exp, ensemble, short_name): fn
    datasets = {index[1]:True for index in detrended_ncs.keys()}

    multimodel_cubes = []
    full_depth = False
    included_datasets = ''
    for master_dataset in datasets:
        if master_dataset in ['FGOALS-f3-L','FGOALS-g3', ]:
            continue
        cubes = []
        work_dir = diagtools.folder([cfg['work_dir'], 'regridded_means'])
        single_fn = work_dir + '_'.join([master_dataset, master_project, master_short_name, master_exp, trend, time_range_str])+'.nc'

        if os.path.exists(single_fn):
            mean_cube =  iris.load_cube(single_fn)
            multimodel_cubes.append(mean_cube.copy())
            for (project, dataset, exp, ensemble, short_name), fn in detrended_ncs.items():
                if master_dataset != dataset: continue
                if master_project != project: continue
                if master_exp != exp: continue
                if master_short_name != short_name: continue
                included_datasets += ' '.join([dataset, ensemble, exp,])
            included_datasets+='\n'
            continue

        cubes = []
        for (project, dataset, exp, ensemble, short_name), fn in detrended_ncs.items():
            if master_dataset != dataset: continue
            if master_project != project: continue
            if master_exp != exp: continue
            if master_short_name != short_name: continue

            cube =  iris.load_cube(fn)
            if isinstance(time_range, list):
               cube = extract_time(cube, time_range[0], 1, 1, time_range[1], 12, 31)
               cube = cube.collapsed('time', iris.analysis.MEAN)
            else:
               cube = extract_time(cube, int(time_range), 1, 1, int(time_range), 12, 31)

            if not full_depth:
                cube = fix_depth(cube)
                cube = extract_levels(cube, [0., ], "nearest_horizontal_extrapolate_vertical")
            cubes.append(cube.copy())
            included_datasets += ' '.join([dataset, ensemble, exp,])
        included_datasets+='\n'

        if len(cubes) ==0:
            print('did not find any cubes', master_dataset, master_project, master_exp, master_short_name)
            assert 0
        print('Making a multi ensemble member mean:',master_dataset)
        mean_cube = make_mean_of_cube_list_notime(cubes)

        # regrid:
        print('regridding:',master_dataset)
        mean_cube = regrid_to_1x1(mean_cube)

        # vertical regrid:
        if full_depth:
            mean_cube = fix_depth(mean_cube)
            levels = [1, 10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450,
                      500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1250,
                      1500, 1750, 2000, 2400, 2800, 3200, 3600, 4000, 4400,
                      4800, 5200]
            print('vertical regridding:',master_dataset)
            mean_cube = extract_levels(mean_cube, levels, "nearest_horizontal_extrapolate_vertical")

        # Save single model mean.
        iris.save(mean_cube, single_fn)

        multimodel_cubes.append(mean_cube)

    print('Making the multi model mean of regridded cubes:')
    mean_cube = make_mean_of_cube_list_notime(multimodel_cubes)
    mean_cube.attributes['included_datasets'] = included_datasets

    print('Writing cube to file:', out_fn)
    iris.save(mean_cube, out_fn)
    metadata = {}
    metadata['short_name'] = master_short_name
    metadata['dataset'] = 'multi'
    metadata['ensemble'] = 'multi'
    metadata['project'] = master_project
    metadata['exp'] = master_exp

    single_pane_map_plot(
        cfg,
        metadata,
        mean_cube,
        key='multimodel_mean/'+'_'.join([master_project, master_short_name, master_exp, trend, time_range_str]),
        sym_zero=False,
    )
    return out_fn


def sea_surface_salinity_plot(
        cfg,
        fn,
        master_short_name,
        time_range,
        fig=None,
        ax=None,
        subplot=111,
        fig_type = 'mean',
        ref_file = None,
        ref_file_2 = None,
        obs_key='DW1970',
        calc_trend = True
    ):
    """
    Make a multi-pane plot of the Sea Surface Salininty.
    """
    if isinstance(time_range, list):
        time_range_str = '-'.join(str(t) for t in time_range)
    else:
        time_range_str = str(time_range)
    central_longitude=-160.+3.5

    if fn:
        mean_cube = iris.load_cube(fn)
        mean_cube = mean_cube.intersection(longitude=(central_longitude-180., central_longitude+180.), latitude=(-73., 73.))

    if ref_file not in [None, (None,)]:
        print('loading ref file:', fig_type, obs_key, ref_file)
        ref_cube = iris.load_cube(ref_file)
        ref_cube = ref_cube.intersection(longitude=(central_longitude-180., central_longitude+180.), latitude=(-73., 73.))

    if ref_file_2 not in [None, (None,)]:
        print('loading ref file:', fig_type, obs_key, ref_file_2)
        ref_cube_2 = iris.load_cube(ref_file_2)
        ref_cube_2 = ref_cube_2.intersection(longitude=(central_longitude-180., central_longitude+180.), latitude=(-73., 73.))


    thresholds =[32., 32.5, 33., 33.5, 34., 34.5, 35., 35.5, 36., 36.5, 37., 37.5, ]
    levels =[32., 33., 34., 35., 36., 37.]
    linestyles = ['-' for thres in thresholds]
    colours = ['k' for thres in thresholds]
    linewidths = [1., 0.5, 1., 0.5, 1., 0.5, 1., 0.5, 1., 0.5, 1., 0.5,]

    if calc_trend:
        thresholds_white = np.arange(-2, 2.25, 0.25)
        change_ticks = [-0.2, -0.1, 0., 0.1, 0.2]
        linestyles_white = ['-' for thres in thresholds_white]
        colours_white = ['w' for thres in thresholds_white]
        linewidths_white = [0.5 for thres in thresholds_white]

    if obs_key=='DW1950':
        obs_file = cfg['auxiliary_data_dir']+'/DurackFiles/DurackandWijffels_GlobalOceanChanges_19500101-20191231__210122-205355_beta.nc'
        obs_time_str = '1950-2019'
        obs_denom = 2020.-1950.

    if obs_key=='DW1970':
        obs_file = cfg['auxiliary_data_dir']+'/DurackFiles/DurackandWijffels_GlobalOceanChanges_19700101-20191231__210122-205448_beta.nc'
        obs_denom = 2020.-1970.
        obs_time_str = '1970-2019'


    obs_cubes = iris.load_raw(obs_file)
    obs_change_cube = obs_cubes.extract(iris.Constraint(name='salinity_change'))[0]
    #bs_change_cube = regrid_to_1x1(obs_change_cube[0]) # surface
    obs_change_cube = obs_change_cube.intersection(longitude=(central_longitude-180., central_longitude+180.), latitude=(-73., 73.))[0]
    obs_change_cube.coord('longitude').attributes['circular'] = True

    obs_mean_cube = obs_cubes.extract(iris.Constraint(name='salinity_mean'))[0]
    #bs_mean_cube = regrid_to_1x1(obs_mean_cube[0]) # surface
    obs_mean_cube = obs_mean_cube.intersection(longitude=(central_longitude-180., central_longitude+180.), latitude=(-73., 73.))[0]
    obs_mean_cube.coord('longitude').attributes['circular'] = True

    if obs_change_cube.data.ndim == 3:
        obs_change_cube = obs_change_cube[0]
    if obs_mean_cube.data.ndim == 3:
        obs_mean_cube = obs_mean_cube[0]

    # Set black contour cube:
    black_con_cube = obs_mean_cube
    if ref_file_2 not in [None, (None,)]:
        black_con_cube = ref_cube_2

    if calc_trend:
        obs_change_cube.data = (obs_change_cube.data/obs_denom)*1000.
        print('calc_trend:', obs_change_cube.data.min(), obs_change_cube.data.max())

    #change_nspace_50 =  np.linspace(-11.5, 11.5, 24)
    #change_nspace_70 =  np.linspace(-13, 13 , 16, endpoint=True)
    change_nspace_50 =  np.linspace(-11., 11., 12)
    change_nspace_70 =  np.linspace(-11., 11., 12)

    if fig_type=='obs_change': # pane a (221, 211, also white contours)
        cube = obs_change_cube
        #nspace = np.linspace(-0.2, 0.2 , 22, endpoint=True)
        if obs_key=='DW1950':
            nspace = change_nspace_50
        if obs_key=='DW1970':
            nspace = change_nspace_70
        print(obs_key, nspace, cube.data.min(), cube.data.max())
        #assert 0
        cmap=diagtools.misc_div
        #abel= obs_key+' trend ('+obs_time_str+')'
        label = "Durack & Wijffels (1950-2019)"

    elif fig_type=='obs_mean': # pane b (222, also contours)
        cube = obs_mean_cube
        nspace = np.linspace(32., 38, 22, endpoint=True)
        cmap=diagtools.misc_seq
        label= obs_key+' mean ('+obs_time_str+')'

    elif fig_type=='model-obs': # pane c (212,223)
        cube = mean_cube
        cube.data = mean_cube.data - obs_mean_cube.data
        nspace = np.linspace(-2., 2., 22, endpoint=True)
        cmap=diagtools.misc_div
        label= 'CMIP6 ('+time_range_str+') - '+obs_key +' ('+obs_time_str+')'

    elif fig_type=='model-mean': # pane 4 (224)
        cube = mean_cube
        nspace = np.linspace(32., 38, 22, endpoint=True)
        cmap=diagtools.misc_seq
        label= 'CMIP6 ('+time_range_str+') mean'

    elif fig_type=='model_change': # pane 4 (224)
        cube = mean_cube
        cube.data = mean_cube.data - ref_cube.data
        denom = 1000./(time_range[1]-time_range[0] +1.)
        cube.data = cube.data * denom
        if obs_key=='DW1950':
            nspace = change_nspace_50
        if obs_key=='DW1970':
            nspace = change_nspace_70
        cmap=diagtools.misc_div
        #label= 'CMIP6 trend ('+time_range_str+')'
        label = "CMIP6 historical (1950-2014)" 
    else:
        print("Fig type not recognised", fig_type)
        assert 0

    proj = ccrs.Robinson(central_longitude=central_longitude)

    if fig is None:
        fig = plt.figure()
        fig.set_size_inches(10, 7)

    if isinstance(subplot, int) and subplot==111:
        ax = fig.add_subplot(subplot, projection=proj)
    else:
        # ax=subplot
        plt.sca(ax)

    #if fig_type=='mean':
    #    cmap=diagtools.misc_seq
    #    nspace = np.linspace(
    #        cube.data.min(),
    #        cube.data.max(), 22, endpoint=True)

    #if fig_type=='trend':
    #    cmap=diagtools.misc_div
    #    plot_max = np.max([cube.data.max(), np.abs(cube.data.min())])
    #    nspace = np.linspace(-plot_max, plot_max, 22, endpoint=True)
    clip = True
    if clip:
        cube.data = np.ma.clip(cube.data, nspace.min(), nspace.max())

    print(fig_type, subplot, cube.data.shape, )
    qplot = iris.plot.contourf(
        cube,
        nspace,
        linewidth=0,
        cmap=cmap,
        zmin=np.min(nspace),
        zmax=np.max(nspace))

    try: plt.gca().coastlines()
    except: pass

    black_contours = True
    if black_contours:
        black_con = iris.plot.contour(black_con_cube,
                 thresholds,
                 colors=colours,
                 linewidths=linewidths,
                 linestyles=linestyles,
                 rasterized=True,
                 extend='both',
                 )
        ax.clabel(black_con, levels, inline=True, fontsize=8, fmt = '%1.0f')
    white_contours = False
    if white_contours:
        white_con = iris.plot.contour(obs_change_cube,
                 thresholds_white,
                 colors=colours_white,
                 linewidths=linewidths_white,
                 linestyles=linestyles_white,
                 rasterized=True,
                 extend='both',
                 )
        #ax.clabel(white_con, change_ticks, inline=True, fontsize=8, fmt = '%1.0f')

    #ax = add_map_text(ax, label)
    ax.set_title(label)

    # Saving files:
    if isinstance(subplot, int) and subplot==111:
        plt.title('Surface Salinity ' +fig_type.title()+' '+time_range_str)
        plt.colorbar()
        if cfg['write_plots']:
            unique_id = [master_short_name, fig_type, time_range_str ]
            filename = '_'.join(unique_id).replace('/', '_')
            path = diagtools.folder([cfg['plot_dir'], 'sea_surface_salinity_plot']) + filename
            path = path.replace(' ', '') + diagtools.get_image_format(cfg)
            logger.info('Saving plots to %s', path)
            plt.savefig(path, dpi=200)
        plt.close()
    else:
        return fig, ax, qplot

def sea_surface_salinity_multipane(
    cfg,
    ss_files,
    plot_type = '2_pane',
    start_year = 1970,
    end_year = 2014,
    obs_key='DW1970'
    ):
    """
    Plot the multipane sea surface salininty plot.
    do_ss if True
    """
    time_range_str = '-'.join([str(t) for t in [start_year, end_year]])

    central_longitude=-160.+3.5
    proj = ccrs.Robinson(central_longitude=central_longitude)
    fig = plt.figure()

    if plot_type == '2_pane':
        subplots = [211, 212]
        fig_types = ['obs_change', 'model-obs']
        fig.set_size_inches(10, 7)

        # keys = ['
    if plot_type == '4_pane':
        subplots = [221, 222, 223, 224]
        fig_types = ['obs_change', 'obs_mean', 'model-obs', 'model-mean']
        fig.set_size_inches(10, 6)

    if plot_type == 'CMIP_only':
        subplots = [211, 212]
        fig_types = ['model-mean', 'model-obs']
        fig.set_size_inches(10, 7)

    if plot_type == 'trends_only':
        subplots = [211, 212]
        fig_types = ['obs_change', 'model_change']
        fig.set_size_inches(10, 7)

    axes = {}
    qplots = {}
    for subplot, fig_type in zip(subplots, fig_types):
        axes[subplot] = fig.add_subplot(subplot, projection=proj)
        ref_file = None
        ref_file_2 = None
        if fig_type in ['obs_change', 'obs_mean']:
            fn = None
        if fig_type in ['model-obs', 'model-mean']:
            fn = ss_files[(start_year, end_year)]

        if fig_type in ['model_change',]:
            fn = ss_files[end_year]
            ref_file = ss_files[start_year]
            ref_file_2 = ss_files[(start_year, end_year)]

            if ref_file is None:
                print('cant find:', ref_file)
                assert 0

        fig, axes[subplot], qplots[subplot] = sea_surface_salinity_plot(
                cfg,
                fn,
                'so',
                [start_year, end_year],
                fig_type = fig_type,
                fig=fig,
                ax=axes[subplot],
                subplot=subplot,
                obs_key=obs_key,
                ref_file=ref_file,
                ref_file_2 = ref_file_2,
           )

    if plot_type == '2_pane':
        #fig.colorbar(qplots[211], ax=[axes[211], axes[212]], location='right',label='Change in PSU') #, ticks=cbar_ticks)
        fig.colorbar(qplots[211], ax=[axes[211], ], location='right',label='Change in PSU/yr *1e'+r'$^{3}$') #, ticks=cbar_ticks)
        fig.colorbar(qplots[212], ax=[axes[212], ], location='right',label='Error in PSU') #, ticks=cbar_ticks)

        fig.suptitle('Sea Surface Salinity change '+obs_key)
    if plot_type == '4_pane':
        #fig.colorbar(qplots[221], ax=[axes[221], axes[223]], location='bottom',label='Change in PSU') #, ticks=cbar_ticks)
        fig.colorbar(qplots[221], ax=[axes[221], ], location='right',label='Change in PSU *1000.') #, ticks=cbar_ticks)
        fig.colorbar(qplots[223], ax=[axes[223], ], location='right',label='Error in PSU') #, ticks=cbar_ticks)

        fig.colorbar(qplots[222], ax=[axes[222], axes[224]], location='right',label='PSU') #, ticks=cbar_ticks)
        fig.suptitle('Sea Surface Salinity change '+obs_key)

    if plot_type == 'CMIP_only':
        fig.colorbar(qplots[211], ax=[axes[211], ], location='right',label='PSU') #, ticks=cbar_ticks)
        fig.colorbar(qplots[212], ax=[axes[212], ], location='right',label='Error in PSU') #, ticks=cbar_ticks)

    if plot_type == 'trends_only':
        #fig.suptitle('Near-Surface Salinity trends') #s_key)
        cbar_ticks = np.linspace(-10., 10., 11, )
        cbar = fig.colorbar(qplots[211], ax=[axes[211],axes[212] ], location='right',label='Near-Surface Salinity trends, mPSS-78/yr', ticks=cbar_ticks)

    unique_id = ['salinity', plot_type, time_range_str, obs_key]
    filename = '_'.join(unique_id).replace('/', '_')
    path = diagtools.folder([cfg['plot_dir'], 'sea_surface_salinity_plot']) + filename
    path = path.replace(' ', '') + diagtools.get_image_format(cfg)
    logger.info('Saving plots to %s', path)
    plt.savefig(path, dpi=200)
    plt.close()


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


def guess_areacello_fn(dicts, keys):
    """
    Take a punt at the areacello filename.
    """
    for index, value in dicts.items():
        intersection = set(index) & set(keys)

        if len(intersection) == len(keys):
            print("guess_areacello_fn: full match", index, keys, value)
            return value
    print('Did Not find Areacello:', keys)

    if 'FGOALS-g3' in keys:
        # No area file for fgoals, but it's the same in FGOALS-f3-L
        for i,a in enumerate(keys):
            if a == 'FGOALS-g3':
                keys[i] = 'FGOALS-f3-L'
        return guess_areacello_fn(dicts, keys)
    #return guess_volcello_fn(dicts, t_index, vol_key = 'areacello')
    assert 0

def guess_volcello_fn_old(dicts, keys, optional = []):
    """
    Take a punt at the volcello filename.
    # When it's hist-* we want historical
    # When itr's piControl, we want piControl.
    # When the model is a Ofx model, we want the Ofx version.
    """
    print("guess_volcello_fn: looking for:", keys, optional)
    for index, value in dicts.items():
        both = (keys+optional)

        intersection = set(index) & set(both)
        print('iterating: index:',index, 'both:',both, 'intersection:', intersection, 'file:',value)

        if sorted([i for i in intersection]) == sorted([i for i in both]):
            print("guess_volcello_fn: full match", index, keys, optional, value)
            return value

    for index, value in dicts.items():
        for option in optional:
            both = (keys+[option, ])
            intersection = set(index) & set(both)
            if len(intersection) == len(both):
                print("guess_volcello_fn: partial match", index, keys, option, value)
                return value

    for index, value in dicts.items():
        intersection = set(index) & set(keys)
        if len(intersection) == len(keys):
            print("guess_volcello_fn: minimal match", index, keys, value)
            return value
    print('Did Not find Volcello:', keys ) #ional)
    assert 0

def guess_volcello_fn(dicts, t_index, vol_key = 'volcello'):
    """
    Take a punt at the volcello filename.
    # When it's hist-* we want historical
    # When itr's piControl, we want piControl.
    # When the model is a Ofx model, we want the Ofx version.
    """
    (t_project, t_dataset, t_exp, t_ensemble, t_short_name) = t_index

    key_list = [[t_project, t_dataset, t_exp, t_ensemble, vol_key   ],] # everything.
    enss = ['r1i1p1f1', 'r1i1p1f2', 'r1i1p2f1', 'r1i1p1f3']
    if t_exp == 'hist_nat':
        for pi_ens in enss:
            key_list.append([t_project, t_dataset, 'historical', pi_ens, vol_key   ]) # historical volcello

    for pi_ens in enss:
        key_list.append([t_project, t_dataset, 'piControl', pi_ens, vol_key   ]) # piControl volcello

    for pi_ens in enss:
        key_list.append([t_project, t_dataset, 'historical', pi_ens, vol_key   ]) # historical volcello


    for keys in key_list:
        print("guess_volcello_fn: looking for:", keys)

        for index, value in dicts.items():
            intersection = set(index) & set(keys)
            print('iterating: index:',index, 'keys:',keys, 'intersection:', intersection, 'file:',value)
            if sorted([i for i in intersection]) == sorted([i for i in keys]):
                print("guess_volcello_fn: full match", index, value)
                return value


    # for index, value in dicts.items():
    #     for option in optional:
    #         both = (keys+[option, ])
    #         intersection = set(index) & set(both)
    #         if len(intersection) == len(both):
    #             print("guess_volcello_fn: partial match", index, keys, option, value)
    #             return value
    #
    # for index, value in dicts.items():
    #     intersection = set(index) & set(keys)
    #     if len(intersection) == len(keys):
    #         print("guess_volcello_fn: minimal match", index, keys, value)
    #         return value
    print('Did Not find Volcello:', keys)
    assert 0




def guess_vol_exp(exp):
    # there are no hist-* volume files.
    if exp.find('hist')>-1: return 'historical'
    return exp


def mpi_fit(iter_pack, cubedata, time_itr):#, tmin):
    index, _ = iter_pack
    data = cubedata[:, index[0], index[1], index[2]]
    if np.ma.is_masked(data.max()):
        return [], index

    if data.mask.sum() ==  len(data):
        # No masked values
        data = data - np.ma.mean(data) #[tmin-1:tmin+2])
        linreg = linregress(time_itr, data)
        return linreg, index

    times = np.ma.array(time_itr)
    times = np.ma.masked_where(data.mask, times).compressed()
    data = data.compressed()
    data = data - np.ma.mean(data) #in-1:tmin+2])

    linreg = linregress(times, data)
    return linreg, index


def mpi_fit_2D(iter_pack, cubedata, time_itr):#, tmin):
    index, _ = iter_pack
    data = cubedata[:, index[0], index[1]]
    if np.ma.is_masked(data.max()):
        return [], index

    if data.mask.sum() ==  len(data):
        # No masked values
        data = data - np.ma.mean(data) #[tmin-1:tmin+2])
        linreg = linregress(time_itr, data)
        return linreg, index

    times = np.ma.array(time_itr)
    times = np.ma.masked_where(data.mask, times).compressed()
    data = data.compressed()
    data = data - np.ma.mean(data) #in-1:tmin+2])

    linreg = linregress(times, data)
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
        print ('calc_pi_trend: loading from', output_shelve)
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

    pi_year = np.array([t.units.num2date(times.points)[0].year for t in times]).mean() # mid point

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
        NDenum = np.ndenumerate(dummy)

#        tmin = np.argmin(np.abs(np.array(decimal_time) - pi_year))
        print('ProcessPoolExecutor: starting')
#        executor = ProcessPoolExecutor(max_workers=1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            print('ProcessPoolExecutor: executing')
            for linreg, index in executor.map(mpi_fit,
                                              ndenum,
                                              itertools.repeat(cube.data),
                                              itertools.repeat(time_arange),
#                                              itertools.repeat(tmin),
                                              chunksize=100000):
#                                              itertools.repeat(y_dat)):
#                                             chunksize=10000):
#               print(linreg, index, count)
                if linreg:
                    if linreg.slope > 1E10 or linreg.intercept>1E10:
                        print('linear regression failed:', linreg)
                        print('slope:', linreg.slope,'intercept', linreg.intercept)
                        print('from time:', np.arange(len(decimal_time)))
                        print('from data:', cube.data[:, index[0], index[1], index[2]])
                        assert 0

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
        plt.hist(list(slopes.values()), bins=15, color='red', )
        plt.title('Slopes')

        fig.add_subplot(212)
        plt.hist(list(intercepts.values()), bins=15, color='blue')
        plt.title('Intercepts')

        path = diagtools.folder([cfg['plot_dir'], 'pi_trend'])
        path += '_'.join([project, dataset, exp, ensemble, short_name, 'pitrend'])+diagtools.get_image_format(cfg)
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

    do_SS =  True
    do_SLR = False # True  #False
    do_OHC = False #True  #True
    bad_models = ['NorESM2-LM', 'NorESM2-MM',
                  'FGOALS-f3-L', 'FGOALS-g3',
                  #'CESM2-FV2', 'CESM2-WACCM-FV2', 'CESM2-WACCM', 'CESM2'
                 ]
    # NorESM2 doesn't include depth data. z axis is werid.
    # FGOALS are missing longitude and lattitide boundaries (but have irregular grids)
    # CESM2-FV2 still has depth in cm.


    print('\n\n\ncreation loop')
    for filename in sorted(metadatas):
        exp = metadatas[filename]['exp']
        variable_group = metadatas[filename]['variable_group']
        short_name = metadatas[filename]['short_name']
        dataset = metadatas[filename]['dataset']
        ensemble = metadatas[filename]['ensemble']
        project = metadatas[filename]['project']
        print((project, dataset, exp, ensemble, short_name))
        if dataset in bad_models:
            continue
        file_dict[(project, dataset, exp, ensemble, short_name)] = filename

    print('\nCalculating trend')
    # Calculated trend.
    for (project, dataset, exp, ensemble, short_name), filename in file_dict.items():
        print('iterating', project, dataset, exp, ensemble, short_name, filename)
        if exp != 'piControl':
            continue
        if short_name in ['volcello', 'areacello', 'thkcello' ]:
            continue
        trend_shelves[(project, dataset, exp, ensemble, short_name)] = calc_pi_trend(cfg, metadatas, filename)

    print('\nDetrend from PI.')
    for (project, dataset, exp, ensemble, short_name), filename in file_dict.items():
        print('detrending:', project, dataset, exp, ensemble, short_name, filename)
        if short_name in ['volcello', 'areacello', 'thkcello']:
              continue
        pi_ensemble = guess_PI_ensemble(trend_shelves, [project, dataset, short_name], ens_pos = 3)
        trend_shelve = trend_shelves[(project, dataset, 'piControl', pi_ensemble, short_name)]
        detrended_fn = detrend_from_PI(cfg, metadatas, filename, trend_shelve)
        detrended_ncs[(project, dataset, exp, ensemble, short_name)] = detrended_fn
        metadatas[detrended_fn] = metadatas[filename].copy()

    # Make a plot for the
    if do_SS:
        short_names = ['so', ] # 'thetao']
        time_ranges =  [1950, 1970, 2014, [1950, 2014], [1970, 2014]]
        ss_files = {}
        for master_short_name, time_range in itertools.product(short_names, time_ranges):
            fn = calculate_multi_model_mean(cfg, metadatas, detrended_ncs,
                master_project = 'CMIP6',
                master_short_name = master_short_name,
                master_exp = 'historical',
                time_range = time_range)
            if isinstance(time_range, list):
                ss_files[tuple(time_range)] = fn
            else:
                ss_files[time_range] = fn

            #sea_surface_salinity_plot(
            #    cfg,
            #    fn,
            #    master_short_name,
            #    time_range,
            #    fig_type = 'mean',
            #)
        #sea_surface_salinity_plot(
        #    cfg,
        #    ss_files[2014],
        #    master_short_name,
        #    [2014, 1950],
        #    fig_type = 'trend',
        #    ref_file=ss_files[1950]
        #)
        plot_types = ['trends_only',] # '2_pane', '4_pane', 'CMIP_only']
        for plot_type,start_year in itertools.product(plot_types, [1950, 1970]):
            if start_year == 1950:
                obs_key='DW1950'
            if start_year == 1970:
                obs_key='DW1970'
            sea_surface_salinity_multipane(
                cfg,
                ss_files,
                plot_type = plot_type,
                start_year = start_year,
                end_year = 2014,
                obs_key=obs_key,
            )
    if do_SLR == do_OHC == False:
        return
    print('Make time series plots')
    volume_weighted_means={}
    for (project, dataset, exp, ensemble, short_name), filename in file_dict.items():
        if short_name in ['volcello', 'areacello', 'thkcello']:
            continue
        # vol_exp = guess_vol_exp(exp)
        # volcello_fn = guess_volcello_fn(file_dict, [project, dataset, 'volcello', vol_exp], optional=[ensemble,])
        volcello_fn = guess_volcello_fn(file_dict,  (project, dataset, exp, ensemble, short_name))

        vwts = calculate_volume_weighted_mean(cfg, metadatas[filename], filename, volcello_fn, trend = 'intact')
        volume_weighted_means[(project, dataset, exp, ensemble, short_name, 'intact')] = vwts
        metadatas[vwts] = metadatas[filename]

    for (project, dataset, exp, ensemble, short_name), detrended_fn  in detrended_ncs.items():
        if short_name in ['volcello', 'areacello', 'thkcello']:
            continue
        # vol_exp = guess_vol_exp(exp)
        # volcello_fn = guess_volcello_fn(file_dict, [project, dataset, 'volcello', vol_exp], optional=[ensemble,])
        volcello_fn = guess_volcello_fn(file_dict,  (project, dataset, exp, ensemble, short_name))

        vwts = calculate_volume_weighted_mean(cfg, metadatas[detrended_fn], detrended_fn, volcello_fn, trend = 'detrended')
        volume_weighted_means[(project, dataset, exp, ensemble, short_name, 'detrended')] = vwts
        metadatas[vwts] = metadatas[detrended_fn]

    for (project, dataset, exp, ensemble, short_name, trend), fn in volume_weighted_means.items():
        print('iterating VWM files:',(project, dataset, exp, ensemble, short_name, trend), fn)
        if short_name in ['volcello', 'areacello', 'thkcello']:
            continue
        if exp in 'piControl': continue
        if trend == 'intact': continue
        print('Listing VWM plots:',(project, dataset, exp, ensemble, short_name, trend))
        hist_detrended = fn
        hist_intact = volume_weighted_means[(project, dataset, exp, ensemble, short_name, 'intact')]

        pi_ensemble = guess_PI_ensemble(trend_shelves, [project, dataset, short_name], ens_pos = 3)
        pi_detrended = volume_weighted_means[(project, dataset, 'piControl', pi_ensemble, short_name, 'detrended')]
        pi_intact = volume_weighted_means[(project, dataset, 'piControl', pi_ensemble, short_name, 'intact')]

        detrending_fig(cfg,
            metadatas,
            hist_detrended,
            hist_intact,
            pi_detrended,
            pi_intact,
            '',
            short_name,
            year = None,
            draw_zero=False,
            native_time=True,
            )

    print('\n-------------\nCalculate Sea Level Rise')
    dyn_fns = {}
    slr_fns = {}

    method = 'Landerer'
    # bad_models = ['NorESM2-LM','CESM2-FV2',]
    trends = ['detrended', ] #'intact']
    for (project, dataset, exp, ensemble, short_name)  in sorted(detrended_ncs.keys()):
        for trend in trends:
            # if dataset != 'ACCESS-CM2': continue
            if dataset in bad_models: continue

            if not do_SLR: continue
            if short_name != 'thetao':
                continue
            if exp ==  'piControl': # no need to calculate this.
                continue

            pi_ensemble = guess_PI_ensemble(trend_shelves, [project, dataset, short_name], ens_pos = 3)

            hist_thetao_fn = None
            hist_so_fn = None

            if trend == 'detrended':
                thetao_fn = detrended_ncs[(project, dataset, exp, ensemble, short_name)]
                so_fn =  detrended_ncs[(project, dataset, exp, ensemble, 'so')]
                picontrol_thetao_fn = detrended_ncs[(project, dataset, 'piControl', pi_ensemble, short_name)]
                picontrol_so_fn =  detrended_ncs[(project, dataset, 'piControl', pi_ensemble, 'so')]
                # A future scenario needs both hist and pi control for references.
                if exp.find('ssp')>-1:
                    hist_thetao_fn = detrended_ncs[(project, dataset, 'historical', ensemble, short_name)]
                    hist_so_fn =  detrended_ncs[(project, dataset, 'historical', ensemble, 'so')]

            if trend ==  'intact':
                thetao_fn = file_dict[(project, dataset, exp, ensemble, 'thetao')]
                so_fn = file_dict[(project, dataset, exp, ensemble, 'so')]
                picontrol_thetao_fn = file_dict[(project, dataset, 'piControl', pi_ensemble, short_name)]
                picontrol_so_fn =  file_dict[(project, dataset, 'piControl', pi_ensemble, 'so')]
                if exp.find('ssp')>-1:
                    hist_thetao_fn = file_dict[(project, dataset, 'historical', ensemble, short_name)]
                    hist_so_fn =  file_dict[(project, dataset, 'historical', ensemble, 'so')]

            check_units(cfg, metadatas[thetao_fn],
                files = [thetao_fn, so_fn, picontrol_thetao_fn, picontrol_so_fn],
                keys = [project, dataset, exp, ensemble, short_name])

            #method = 'dyn_height'
            if method == 'dyn_height':
                assert 0
                # # Not ready to calculate using this method and ssp data.
                # dyn_height_fns = calc_dyn_height_full(
                #     cfg,
                #     metadatas,
                #     hist_thetao_fn,
                #     hist_so_fn,
                #     picontrol_thetao_fn,
                #     picontrol_so_fn,
                #     trend=trend,
                #     method=method,
                #     )
                # areacella_fn = guess_areacello_fn(file_dict, [project, dataset, 'areacello'])
                # regions = ['Global', 'Atlantic', 'Pacific']
                # for region in regions:
                #     dyn_averages={}
                #     for dyn_type, dyn_fn in dyn_height_fns.items():
                #         dyn_ts_fn = calc_dyn_timeseries(cfg, dyn_fn, areacella_fn, project, dataset, exp, ensemble, dyn_type, region, trend, method = method)
                #         dyn_fns[(project, dataset, exp, ensemble, dyn_type, region, trend)] = dyn_fn
                #         dyn_fns[(project, dataset, exp, ensemble, dyn_type + '_ts', region, trend)] = dyn_ts_fn
                #         metadatas[dyn_ts_fn] = metadatas[hist_thetao_fn]
                #         metadatas[dyn_fn] = metadatas[hist_thetao_fn]
                #         dyn_averages[dyn_type] = dyn_ts_fn
            elif method == 'Landerer':
                slr_fns_dict = calc_landerer_slr(
                    cfg,
                    metadatas,
                    thetao_fn,
                    so_fn,
                    picontrol_thetao_fn,
                    picontrol_so_fn,
                    hist_thetao_fn=hist_thetao_fn,
                    hist_so_fn=hist_so_fn,
                    trend=trend,
                    )
                areacella_fn = guess_areacello_fn(file_dict, [project, dataset, 'areacello'])
                regions = ['Global', 'Atlantic', 'Pacific']
                for region in regions:
                    slr_averages={}
                    for slr_type, slr_fn in slr_fns_dict.items():
                        slr_ts_fn = calc_dyn_timeseries(cfg, metadatas, slr_fn, areacella_fn, project, dataset, exp, ensemble, slr_type, region, trend, method = method)
                        slr_fns[(project, dataset, exp, ensemble, slr_type, region, trend)] = slr_fn
                        slr_fns[(project, dataset, exp, ensemble, slr_type + '_ts', region, trend)] = slr_ts_fn
                        metadatas[slr_ts_fn] = metadatas[thetao_fn]
                        metadatas[slr_fn] = metadatas[thetao_fn]
                        slr_averages[slr_type] = slr_ts_fn
                # Calculate spatial average/time series


            # if method == 'dyn_height':
            #     plot_dyn_height_ts(cfg, metadatas[dyn_ts_fn], dyn_averages, trend, region)
            #     plot_slr_full_ts(cfg, metadatas[dyn_ts_fn], dyn_averages, trend, region)

    mmm_slr = {}
    if do_SLR and method == 'Landerer':
        # this is the multiple pane halosteric plot.
        do_plot_halo_multipane = True
        if do_plot_halo_multipane:
            plot_exp = 'historical'
            time_ranges = [[1950, 2015], ] #[1970, 2015], ] #[1950, 2000], [1970, 2000], [1970, 2015], [1950, 2015], [1860, 2015]]
            for time_range,ukesm in itertools.product(time_ranges, [False, ]):
                plot_halo_multipane(
                    cfg,
                    metadatas,
                    slr_fns,
                    plot_exp = plot_exp,
                    method= 'Landerer',
                    time_range=time_range,
                    show_UKESM=ukesm,
                )
        # below here is inthe individual plots:
        plot_scatter = False
        if plot_scatter:
            plot_dyn = 'halo'
            plot_exp = 'historical'
            #plot_clim = '1850-1900_ts'
            time_ranges = [[1950, 2000], [1970, 2015], [1950, 2015], [1860, 2015]]
            plot_region = 'Global'

            for plot_dyn in ['halo_ts', 'thermo_ts']:
                for time_range in time_ranges:
                    plot_slr_regional_scatter(cfg, metadatas, slr_fns,
                        plot_exp = plot_exp,
                        #plot_clim = plot_clim,
                        plot_dyn = plot_dyn,
                        method=method,
                        time_range=time_range,
                    )
        plot_multimodel_slr_ts = False
        if plot_multimodel_slr_ts:
            regions = ['Global', 'Atlantic', 'Pacific']
            for region in regions:
                plot_slr_full_ts_all(cfg, metadatas, slr_fns, region, method = method )

        plot_slr_maps = False
        if plot_slr_maps:
            # Plot SLR maps:
            time_ranges=[[1950, 2000], [1970, 2015], [1950, 2015]]
            for time_range in time_ranges:
                for (project, dataset, exp, ensemble, dyn_type, region, trend), dyn_fn in dyn_fns.items():
                    # Can't make a map plot for a time series.
                    if dyn_type.find('_ts') > -1: continue
                    if dyn_type not in ['halo', 'thermo', 'total']: continue
                    if region != 'Global': continue
                    SLR_map_plot(cfg, metadatas[dyn_fn], dyn_fn, clim_fn, time_range, keys =(project, dataset, exp, ensemble, dyn_type, region, trend, method), method=method)

        plot_multimodel_mean = False
        if plot_multimodel_mean:
            plot_dyn = 'halo'
            plot_exp = 'historical'
            #plot_clim = '1850-1900'
            time_ranges=[[1950, 2000], [1970, 2015], [1950, 2015]]
            plot_region = 'Global'
            for trend, time_range in itertools.product(trends, time_ranges, ):
                multimodel_mean_fn = calc_halo_multimodel_mean(cfg, metadatas, slr_fns,
                    plot_trend = trend,
                    plot_dyn = plot_dyn,
                    plot_exp = plot_exp,
                    plot_region = plot_region,
                    time_range = time_range,
                    method = method,
                )
                make_multimodel_halosteric_salinity_trend(cfg, metadatas,
                    multimodel_mean_fn,
                    plot_trend = trend,
                    plot_dyn = plot_dyn,
                    plot_exp = plot_exp,
                    plot_region = plot_region,
                    time_range = time_range,
                    method = method,
                )

        plot_halosteric_obs = False
        if plot_halosteric_obs:
            obs_keys = ['DurackandWijffels10_V1.0_50yr', 'DurackandWijffels10_V1.0_30yr',
                        'Ishii09_v6.13_annual_steric_1950-2010',
                        'DurackandWijffels_GlobalOceanChanges_19500101-20191231__210122-205355_beta.nc']
            for obs_file in obs_keys:
                plot_halo_obs_mean(
                    cfg,
                    metadatas,
                    plot_dyn = 'halo',
                    subplot=111,
                    depth_range='2000m',
                    obs_file=obs_file,
                )

    if do_SLR and method == 'dyn_height':
        assert 0

    # END of SLR calculation
    if not do_OHC: return

    print('\nCalculate ocean heat content - trend intact')
    for (project, dataset, exp, ensemble, short_name), fn in file_dict.items():
        if short_name != 'thetao':
            continue
        #if exp ==  'piControl': # no need to calculate this.
        #    continue
        # volcello is in same ensemble, same exp.
        #vol_exp = guess_vol_exp(exp)
        #volcello_fn = guess_volcello_fn(file_dict, [project, dataset, 'volcello', vol_exp], optional=[ensemble,])
        volcello_fn = guess_volcello_fn(file_dict,  (project, dataset, exp, ensemble, short_name))

        if not volcello_fn:
            #(project, dataset, exp, ensemble, 'volcello') in file_dict:
            #volcello_fn = guess_volcello_fn(file_dict, [project, dataset, 'volcello'],optional=[ensemble, exp])
            #volcello_fn = file_dict[(project, dataset, exp, ensemble, 'volcello')]
        #else:
            print('ocean heat content calculation: no volcello', (project, dataset, exp, ensemble, short_name))
            assert 0

        for index in file_dict.keys():
            if project not in index: continue
            if dataset not in index: continue
            if exp not in index: continue
            if ensemble not in index: continue
            print('trend intact calculation:', dataset, ':', index)

        if detrending_method == 'Basic':
            assert 0
        elif detrending_method == 'Full':
            so_fn = file_dict[(project, dataset, exp, ensemble, 'so')]
            ohc_fn = calc_ohc_full(cfg, metadatas, fn, so_fn, volcello_fn, trend='intact')
            if ohc_fn is None: continue


        if ohc_fn.find(exp) == -1:
            print('ERROR - ohc_fn',(project, dataset, exp, ensemble, short_name), ohc_fn )
            assert 0

        ocean_heat_content[(project, dataset, exp, ensemble, 'ohc','intact')] = ohc_fn

        metadatas[ohc_fn] = metadatas[fn].copy()

    print('\nCalculate ocean heat content - detrended')
    for (project, dataset, exp, ensemble, short_name), detrended_fn in detrended_ncs.items():
        # Only calculate once for each dataset, exp, ensemble.
        if short_name != 'thetao':
            continue
        # vol_exp = guess_vol_exp(exp)
        # volcello_fn = guess_volcello_fn(file_dict, [project, dataset, 'volcello', vol_exp], optional=[ensemble,]) # old
        volcello_fn = guess_volcello_fn(file_dict,  (project, dataset, exp, ensemble, short_name))
        if not volcello_fn:
            print('ocean heat content calculation: no volcello', (project, dataset, exp, ensemble, short_name))
            assert 0

        for index in detrended_ncs.keys():
            if dataset not in index: continue
            if exp not in index: continue
            if ensemble not in index: continue
            print(dataset, ':', index)

        if detrending_method == 'Basic':
            assert 0
        elif detrending_method == 'Full':
            print('detrending_method:', detrending_method, project, dataset, exp, ensemble)
            so_fn = detrended_ncs[(project, dataset, exp, ensemble, 'so')]
            print('detrending_method:', detrending_method, so_fn)
            ohc_fn = calc_ohc_full(cfg, metadatas, detrended_fn, so_fn, volcello_fn, trend='detrended')
            print('Finished calc_ohc_full calc',ohc_fn)
            if ohc_fn is None: continue
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

        metadatas[ohc_fn] = metadatas[detrended_fn].copy()

    do_volume_integrated_plot = False
    for (project, dataset, exp, ensemble, keya, keyb), ohc_fn in ocean_heat_content.items():
        if not do_volume_integrated_plot: continue
        areacello_fn = guess_areacello_fn(file_dict, [project, dataset, 'areacello'])
        volume_integrated_plot(cfg, metadatas[ohc_fn], ohc_fn, areacello_fn)

    print('\n---------------------\nCalculate OHC time series')
    depth_ranges = ['total', '0-700m', '700-2000m', '0-2000m', '2000m_plus']
    for depth_range in depth_ranges:
        for (project, dataset, exp, ensemble, short_name, trend), ohc_fn in ocean_heat_content.items():
            if exp not in ['historical', 'piControl']: continue
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
    do_detrending_fig = False
    for dataset, ensemble, project, depth_range  in itertools.product(datasets.keys(), ensembles.keys(), projects.keys(), depth_ranges):
        if not do_detrending_fig: continue
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
        detrending_fig(cfg, metadatas, detrended_hist, trend_intact_hist, detrended_piC, trend_intact_piC, depth_range, 'OHC', native_time=True)

    # Multi model time series plotx for each time series
    # Figure based on 2.25.
    #
    do_fig_like225 = False
    for dataset, ensemble, project, exp in itertools.product(datasets.keys(), ensembles.keys(), projects.keys(), exps.keys()):
        if not do_fig_like225: continue
        try:
            ocean_heat_content_timeseries[(project, dataset, exp, ensemble, 'ohc', 'detrended', 'total')]
        except: continue
        fig_like_2_25(cfg, metadatas, ocean_heat_content_timeseries, dataset, ensemble, project, exp)

    multimodel_2_25(cfg, metadatas, ocean_heat_content_timeseries, plot_type='large_full', plot_style='5-95', show_UKESM=False)
    #multimodel_2_25(cfg, metadatas, ocean_heat_content_timeseries, plot_type='4_panes', plot_style='5-95', show_UKESM=False)

    for plot_style, plot_type ,ukesm in itertools.product(['viridis', 'mono','all_one', '5-95'],['7_panes', '4_panes', 'large_full'], [True, False]):
        continue
        multimodel_2_25(cfg, metadatas, ocean_heat_content_timeseries, plot_type =plot_type , plot_style=plot_style, show_UKESM=ukesm)


    logger.info('Success')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
