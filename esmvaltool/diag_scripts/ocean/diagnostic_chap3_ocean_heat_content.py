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
    #color_dict = {da:c for da, c in zip(datasets, ['r' ,'b'])}
    color_dict = {dataset:c for dataset, c in zip(datasets, plt.cm.viridis(np.linspace(0,1,len(datasets))))}

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


def single_pane_map_plot(
        cfg,
        metadata,
        cube,
        key='',
        overwrite = True,
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
        cmap='coolwarm'
        max_val = np.max(np.abs([cube.data.min(), cube.data.max()]))
        nspace = np.linspace(
            -max_val, max_val, 21, endpoint=True)

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
        trend='detrended',):
    """
    Calculate the climatoligical dyncamic height.
    """
    # Load relevant metadata
    exp = metadatas[hist_thetao_fn]['exp']
    dataset = metadatas[hist_thetao_fn]['dataset']
    ensemble = metadatas[hist_thetao_fn]['ensemble']
    project = metadatas[hist_thetao_fn]['project']

    # Generate output paths for SLR netcdfs
    work_dir = diagtools.folder([cfg['work_dir'], 'dyn_height_clim'])
    clim_fn = work_dir + '_'.join([project, dataset, exp, ensemble, 'dyn_height_clim', trend, clim_type]) + '.nc'

    if os.path.exists(clim_fn):
        print ('loading from', clim_fn)
        return clim_fn

    # Load historical temperature and salinity netcdfs
    print('load hist netcdfs')
    psal_bar = iris.load_cube(hist_so_fn)
    thetao_bar = iris.load_cube(hist_thetao_fn)

    # Calculate climatologies for historical period Temperature and salinity.
    # Note that _bar suffix indicates climatology data.
    print('Calculate clim', dataset, clim_type)
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
    elif clim_type == '1971-2018':
        # 4) vs. 1995-2014 (AR5 standard)
        psal_bar = extract_time(psal_bar, 1995, 1, 1, 2014, 12, 31)
        thetao_bar = extract_time(thetao_bar, 1995, 1, 1, 2014, 12, 31)
    elif clim_type == '1971-2018':
        # 5) vs. 1985-2014 (Convenient 30 yr from AR5)
        psal_bar = extract_time(psal_bar, 1985, 1, 1, 2014, 12, 31)
        thetao_bar = extract_time(thetao_bar, 1985, 1, 1, 2014, 12, 31)
    elif clim_type == '1971-2018':
        # 6) vs. 2004-2018 (in comparison to RAPID Period)
        psal_bar = extract_time(psal_bar, 2004, 1, 1, 2018, 12, 31)
        thetao_bar = extract_time(thetao_bar, 2004, 1, 1, 2018, 12, 31)
    elif clim_type == 'fullhistorical':
        # 7) Full historical
        pass
    elif clim_type == 'piControl':
        # 8) vs. detrended PI-control (same window as full historical)
#        return None
        if exp != 'piControl':
            assert 0

    # Calculaste average along the time dimension
    psal_bar = psal_bar.collapsed('time', iris.analysis.MEAN)
    thetao_bar = thetao_bar.collapsed('time', iris.analysis.MEAN)

    single_pane_map_plot(
          cfg,
          metadatas[hist_so_fn],
          psal_bar[0],
          key='Dyn_height_clim/'+clim_type+'_surface_so',
          sym_zero=False,
          )

    single_pane_map_plot(
          cfg,
          metadatas[hist_thetao_fn],
          thetao_bar[0],
          key='Dyn_height_clim/'+clim_type+'_surface_thetao',
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

    # Create output cubes to receive SLR data.
    print('Creating output arrays')
    dyn_height_clim = np.zeros(thetao_bar[0].shape) # 2d + time
    count = 0
    gravity = 9.7963 # m /s^2

    print('Starting clim SLR calculation from fresh')
    slr_clim = thetao_bar[0].copy() # 2D
    if depths_bar.ndim == 1:
       depth_bar = np.tile(depths_bar, (len(lat[0]), 1)).T

    # Iterate over each y line:
    for y in np.arange(len(lat[:,0])):
        sal_bar = psal_bar.data[:, y, :]
        temp_bar = thetao_bar.data[:, y, :]

        if np.ma.is_masked(sal_bar.max()):
             continue
        la = lat[y, :]
        lo = lon[y, :]
        print('Calculate clim dyn height:', dataset, clim_type, (y,)) #, 'latitude:', la, lo)

        # load clim depth
        if depths_bar.ndim == 3:
            depth_bar = depths_bar[:, y, :]

        if depth_bar.shape != sal_bar.shape:
            assert 0

        # calculate pressure
        pressure_bar = gsw.conversions.p_from_z(depth_bar, la)
        pressure_bar = np.ma.masked_where(temp_bar.mask, pressure_bar)

        # calculate corerct conservative temperature and absolute salinity
        sal_bar, temp_bar = step_4and5(dataset, lo, la, pressure_bar, sal_bar, temp_bar)

        # copy fixed data back into original cube.
        psal_bar.data[:, y, :] = sal_bar
        thetao_bar.data[:, y, :] = temp_bar

        # Calculate climatoligical pressure
        # pressure_bar = np.array([gsw.conversions.p_from_z(depth_bar[:, y, :], la[y]) for y in np.arange(len(la))]) # dbar
        #print(y, lola[y], la.shape, lo.shape, lat[:,0].shape)
        #pressure_bar = gsw.conversions.p_from_z(depth_bar, la)
        #psal_bar, temp_bar = step_4and5(dataset, lon, lat, pressure_bar, psal_bar, temp_bar)

        # Calculuate climatological Dynamic height anomaly
        # max_dp is the maximum difference between layers, set to a very high value
        # to avoid expensive interpolation.

        # Convert dynamic height into mm.
        dyn_height_clim[y, :] = gsw.geo_strf_dyn_height(sal_bar, temp_bar, pressure_bar)[0] * 1000. / gravity

    # POlot fixed temperatrure and salinity.
    single_pane_map_plot(
          cfg,
          metadatas[hist_so_fn],
          psal_bar[0],
          key='Dyn_height_clim/'+clim_type+'_surface_asal',
          sym_zero=False,
          )
    single_pane_map_plot(
          cfg,
          metadatas[hist_thetao_fn],
          thetao_bar[0],
          key='Dyn_height_clim/'+clim_type+'_surface_ctemp',
          sym_zero=False,
          )

#   make_difference_plots(
#       cfg,
#       metadatas[hist_thetao_fn],
#       psal_bar[0],
#       hist_cube,
#       label1='',
#       label2='',
#       key='',
#       )


    # Save climatological SLR cube as a netcdf.
    print("saving output cube:", clim_fn)
    cube0 = thetao_bar[0, :, :].copy()
    cube0.data = dyn_height_clim #p.ma.masked_where(slr_clim==0., slr_clim)
    cube0.units = cf_units.Unit('mm')
    cube0.name = 'Climatological ('+ clim_type+') dynamic height'
    cube0.long_name = 'Climatological ('+ clim_type+') dynamic height '
    cube0.short_name = 'dynh_clim'
    cube0.var_name = 'dynh_clim'
    cube0.standard_name = 'steric_change_in_mean_sea_level'
    iris.save(cube0, clim_fn)

    single_pane_map_plot(
          cfg,
          metadatas[hist_thetao_fn],
          cube0,
          key='Dyn_height_clim/'+clim_type+'_dynamic_heightsurface_ctemp',
          sym_zero=False,
          )
    return clim_fn










def calc_dyn_height(
        cfg,
        metadatas,
        hist_thetao_fn,
        hist_so_fn,
        trend=trend):
    """ Calculates the steric, thermo and haloseeteric dynamic height.
    """

    # Load relevant metadata
    exp = metadatas[hist_thetao_fn]['exp']
    dataset = metadatas[hist_thetao_fn]['dataset']
    ensemble = metadatas[hist_thetao_fn]['ensemble']
    project = metadatas[hist_thetao_fn]['project']

    # Generate output paths for SLR netcdfs
    work_dir = diagtools.folder([cfg['work_dir'], 'SLR'])
    output_fns = {}
    # output_fns['total'] = work_dir + '_'.join([project, dataset, exp, ensemble, 'total_SLR', trend])+'.nc'
    # output_fns['thermo'] = work_dir + '_'.join([project, dataset, exp, ensemble, 'thermo_SLR', trend])+'.nc'
    # output_fns['halo'] = work_dir + '_'.join([project, dataset, exp, ensemble, 'halo_SLR', trend])+'.nc'
    # output_fns['clim'] = work_dir + '_'.join([project, dataset, exp, ensemble, 'clim_SLR', trend])+'.nc'

    # Load historical temperature and salinity netcdfs
    so_cube = iris.load_cube(hist_so_fn)
    thetao_cube = iris.load_cube(hist_thetao_fn)

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
    depths = -1.*np.abs(thetao_cube.coord(axis='z').points)

    # Load time array as decidimal time.
    times = diagtools.cube_time_to_float(thetao_cube)

    # Calculate climatologies for historical period Temperature and salinity.
    # Note that _bar suffix indicates climatology data.
    psal_bar = so_cube.copy()
    thetao_bar = thetao_cube.copy()

    psal_bar = psal_bar.collapsed('time', iris.analysis.MEAN)
    thetao_bar = thetao_bar.collapsed('time', iris.analysis.MEAN)
    depths_bar = -1.*np.abs(thetao_bar.coord(axis='z').points)

    # Create output cubes to receive SLR data.
    slr_total = np.zeros(thetao_cube[:, 0].shape) #2d + time
    slr_thermo = slr_total.copy()
    slr_halo = slr_total.copy()

    count = 0
    gravity = 9.7963 # m /s^2

    # Now perform the SLR calculation for each point in time for each lat line:
    if depths.ndim == 1:
        depth = np.tile(depths, (len(lat[0]), 1)).T

    for y in np.arange(len(lat[:,0])):
        print('Calculate SLR in 1D:', dataset, y, 'of', len(la))
        sal_bar = psal_bar.data[:, y, :]
        if np.ma.is_masked(sal_bar.max()):
             continue
        temp_bar = thetao_bar.data[:, y, :]

        la = lat[y, :]
        lo = lon[y, :]

        # Load depth dataset
        if depths.ndim == 3:
            depth = depths[:, y, :]
        if depth.shape != sal_bar.shape: assert 0

        if depths.ndim != 4:
            pressure = gsw.conversions.p_from_z(depth, la) # dbar

        # Load climatological depth dataset
        # load clim depth
        if depths_bar.ndim == 1:
            depth_bar = np.tile(depths_bar, (len(la), 1)).T
            print('tiling depth:', depths_bar.shape, sal_bar.shape, depth_bar.shape)
            if depth_bar.shape != sal_bar.shape: assert 0
        elif depths_bar.ndim == 3:
            depth_bar = depths_bar[:, y, :]
        else:
            assert 0

        pressure_bar = gsw.conversions.p_from_z(depth_bar, la) # dbar
        pressure_bar = np.ma.masked_where(sal_bar.mask, pressure_bar)
        sal_bar, temp_bar = step_4and5(dataset, lo, la, pressure_bar, sal_bar, temp_bar)

        # Calculate SLR for each point in time in the historical dataset.
        for t, time in enumerate(times):
            if depths.ndim == 4:
                depth = depths[t, :, y, :]
                pressure = gsw.conversions.p_from_z(depth, la) # dbar

            # load salinity and temperature data
            sal = so_cube.data[t, :, y, :]
            temp = thetao_cube.data[t, :, y, :]
            pressure = np.ma.masked_where(temp.mask, pressure)

           # print('looping 1A:',t,y, time, sal, temp, pressure)
            # Confirm that we use absolute salininty & conservative temperature
            sal, temp = step_4and5(dataset, lo, la, pressure, sal, temp)
            #print('looping 1B:',t,y, time, sal, temp, pressure)

            # Calculate Dynamic height anomaly
            seafloorpres = pressure.max(axis=0)
            if p_ref == 'SeaFloor':
                gsdh_total = gsw.geo_strf_dyn_height(sal, temp, pressure, p_ref = seafloorpres)[0]
                gsdh_thermo = gsw.geo_strf_dyn_height(sal_bar, temp, pressure, p_ref = seafloorpres)[0]
            elif p_ref == '2000m':
                ref_pressure = gsw.conversions.p_from_z(-2000., la)
                ref_pressure = np.array([np.min([r, f])for r, f in zip(ref_pressure, seafloorpres)])
                gsdh_total = gsw.geo_strf_dyn_height(sal, temp, pressure, p_ref = ref_pressure)[0]
                gsdh_thermo = gsw.geo_strf_dyn_height(sal_bar, temp, pressure, p_ref = ref_pressure)[0]
            else:
                gsdh_total = gsw.geo_strf_dyn_height(sal, temp, pressure, )[0] * 1000. / gravity
                gsdh_thermo = gsw.geo_strf_dyn_height(sal_bar, temp, pressure)[0] * 1000. / gravity
                gsdh_halo = gsw.geo_strf_dyn_height(sal, temp_bar, pressure)[0] * 1000. / gravity

            # Convert to mm and calculate anomaly relative to clim data
            #gsdh_total = gsdh_total - slr_clim[y, :]
            #gsdh_thermo = gsdh_thermo - slr_clim[y, :]
            #gsdh_halo = gsdh_halo - slr_clim[y, :]

            # Put in the output array:
            slr_total[t, y, :] = gsdh_total
            slr_thermo[t, y, :] = gsdh_thermo
            slr_halo[t, y, :] = gsdh_halo

            print('----\n', [t,'of', len(times)], [y,'of',len(lat[:,0])], time)
            print(dataset, 'total', gsdh_total.min(),  gsdh_total.max())
            print(dataset, 'slr_thermo', gsdh_thermo.min(), gsdh_thermo.max())
            print(dataset, 'halo', gsdh_halo.min(), gsdh_halo.max())

            count += 1
            if t == 0:
                print(count, (t, y), 'performing 2D SLR')

    slr_total  = np.ma.masked_where(thetao_cube[:, 0, :, :].data.mask, slr_total)
    slr_thermo = np.ma.masked_where(thetao_cube[:, 0, :, :].data.mask, slr_thermo)
    slr_halo   = np.ma.masked_where(thetao_cube[:, 0, :, :].data.mask, slr_halo )

    # Save SLR data as a cube and then a netcdf..
    cube0 = thetao_cube[:, 0, :, :].copy()
    cube0.data = slr_total
    cube0.units = cf_units.Unit('mm')
    cube0.name = 'Total Steric Dynamic Height'
    cube0.long_name = 'Total Steric Dynamic Height'
    cube0.short_name = 'dyn_total'
    cube0.var_name = 'dyn_total'
    cube0.standard_name = 'steric_change_in_mean_sea_level'
    iris.save(cube0, output_fns['total'])

    cube1 = thetao_cube[:, 0, :, :].copy()
    cube1.data = slr_thermo
    cube1.units = cf_units.Unit('mm')
    cube1.name = 'Thermosteric Dynamic Height'
    cube1.long_name = 'Thermosteric Dynamic Height'
    cube1.short_name = 'dyn_thermo'
    cube1.var_name = 'dyn_thermo'
    cube1.standard_name = 'thermosteric_change_in_mean_sea_level'
    iris.save(cube1, output_fns['thermo'])

    cube2 = thetao_cube[:, 0, :, :].copy()
    cube2.data = slr_halo
    cube2.units = cf_units.Unit('mm')
    cube2.name = 'Halosteric Dynamic Height'
    cube2.long_name = 'Halosteric Dynamic Height'
    cube2.short_name = 'dyn_halo'
    cube2.var_name = 'dyn_halo'
    cube2.standard_name = 'halosteric_change_in_mean_sea_level'
    iris.save(cube2, output_fns['halo'])

    return output_fns['total'], output_fns['thermo'], output_fns['halo']









def calc_slr_full(cfg,
        metadatas,
        hist_thetao_fn,
        hist_so_fn,
        picontrol_thetao_fn,
        picontrol_so_fn,
        trend='detrended',
        ):
    """
    calc_slr_full: Calculates the Sea Level Rise
    """
    # Load relevant metadata
    exp = metadatas[hist_thetao_fn]['exp']
    dataset = metadatas[hist_thetao_fn]['dataset']
    ensemble = metadatas[hist_thetao_fn]['ensemble']
    project = metadatas[hist_thetao_fn]['project']

    # Generate output paths for SLR netcdfs
    work_dir = diagtools.folder([cfg['work_dir'], 'SLR'])
    output_fns = {}
    output_fns['total'] = work_dir + '_'.join([project, dataset, exp, ensemble, 'total_SLR', trend])+'.nc'
    output_fns['thermo'] = work_dir + '_'.join([project, dataset, exp, ensemble, 'thermo_SLR', trend])+'.nc'
    output_fns['halo'] = work_dir + '_'.join([project, dataset, exp, ensemble, 'halo_SLR', trend])+'.nc'
    output_fns['clim'] = work_dir + '_'.join([project, dataset, exp, ensemble, 'clim_SLR', trend])+'.nc'

    # Check whether output paths exists already.
    # If they exist, then make some basic figures and return paths.
#    if False not in [os.path.exists(fn) for fn in output_fns.values()]:
#        for key, fn in output_fns.items():
#            cube1 = iris.load_cube(fn)
#            for t in [0, -1, 'mean']:
#                if fn.find('clim') > -1:
#                    dat = cube1
#                    if t != 0: continue
#                else:
#                    if t == 'mean':
#                        dat = cube1.copy().collapsed('time', iris.analysis.MEAN)
#                    else:
#                        dat = cube1[t]
#                single_pane_map_plot(
#                      cfg,
#                      metadatas[hist_thetao_fn],
#                      dat,
#                      key='SLR_'+key+'_'+trend,
#                      sym_zero=True,
#                      )
#            print('files exist:', key, fn)
#        return output_fns['total'], output_fns['thermo'], output_fns['halo']
#    else:
#        print('outputs not found:', output_fns)
#        for key, fn in output_fns.items():
#            if not os.path.exists(fn):
#                print('file does not exist:', key, fn)

    clim_types = ['1971-2018',  '2005-2018', '1850-1900' , '1995-2014',
                  '1985-2014', '2004-2018', 'fullhistorical', 'piControl']

    clim_files = {}
    for clim_type in clim_types:
        if clim_type == 'piControl':
            clim_fn = calc_dyn_height_clim(
                cfg,
                metadatas,
                picontrol_thetao_fn,
                picontrol_so_fn,
                clim_type=clim_type,
                trend=trend)
        else:
            clim_fn = calc_dyn_height_clim(
                cfg,
                metadatas,
                hist_thetao_fn,
                hist_so_fn,
                clim_type=clim_type,
                trend=trend)
        clim_files[clim_type] = clim_fn

    steric_fn, thermo_fn, halo_fn = calc_dyn_height(
        cfg,
        metadatas,
        hist_thetao_fn,
        hist_so_fn,
        trend=trend)
    assert 0


    # Load historical temperature and salinity netcdfs
    so_cube = iris.load_cube(hist_so_fn)
    thetao_cube = iris.load_cube(hist_thetao_fn)

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
    depths = -1.*np.abs(thetao_cube.coord(axis='z').points)

    # Load time array as decidimal time.
    times = diagtools.cube_time_to_float(thetao_cube)

    # Calculate climatologies for historical period Temperature and salinity.
    # Note that _bar suffix indicates climatology data.
    print('Calculate clim')
    psal_bar = so_cube.copy()
    thetao_bar = thetao_cube.copy()
#    if np.array(times).min() < 1880:
#        psal_bar = extract_time(so_cube.copy(), 1850, 1, 1, 1880, 12, 31)
#        thetao_bar = extract_time(thetao_cube.copy(), 1850, 1, 1, 1880, 12, 31)
#    else:
#        psal_bar = extract_time(so_cube.copy(), 1950, 1, 1, 1980, 12, 31)
#        thetao_bar = extract_time(thetao_cube.copy(), 1950, 1, 1, 1980, 12, 31)

    psal_bar = psal_bar.collapsed('time', iris.analysis.MEAN)
    print('clim: so:', psal_bar.shape)
    thetao_bar = thetao_bar.collapsed('time', iris.analysis.MEAN)
    print('clim: thetao:', thetao_bar.shape)
    depths_bar = -1.*np.abs(thetao_bar.coord(axis='z').points)

    # Create output cubes to receive SLR data.
    print('Creating output arrays')
    slr_total = np.zeros(thetao_cube[:,0].shape) #2d + time
    print(slr_total.shape)
    slr_thermo = slr_total.copy()
    slr_halo = slr_total.copy()

    count = 0

    gravity = 9.7963 # m /s^2
    # Performd SLR calculation for Climatology cube
    # If cube already exists, then just load the clim SLR data.
    p_ref = 'default' #'2000m' #'default' #'SeaFloor' # 'default'

    if os.path.exists(output_fns['clim']):
        print ('loading from', output_fns['clim'])
        clim_cube = iris.load_cube(output_fns['clim'])
        slr_clim = clim_cube.data
    else:
        print('Starting clim SLR calculation from fresh')
        slr_clim = slr_total[0].copy() # 2D

        if depths_bar.ndim == 1:
           depth_bar = np.tile(depths_bar, (len(lat[0]), 1)).T

        # Iterate over each y line:
        for y in np.arange(len(lat[:,0])):
            sal_bar = psal_bar.data[:, y, :]
            temp_bar = thetao_bar.data[:, y, :]
            if np.ma.is_masked(sal_bar.max()):
                 continue
            la = lat[y, :]
            lo = lon[y, :]
            print('Calculate SLR in 1D:', (y,)) #, 'latitude:', la, lo)

            # load clim depth
            if depths_bar.ndim == 3:
                depth_bar = depths_bar[:, y, :]
            if depth_bar.shape != sal_bar.shape: assert 0

            # Calculate climatoligical pressure
            # pressure_bar = np.array([gsw.conversions.p_from_z(depth_bar[:, y, :], la[y]) for y in np.arange(len(la))]) # dbar
            #print(y, lola[y], la.shape, lo.shape, lat[:,0].shape)
            pressure_bar = gsw.conversions.p_from_z(depth_bar, la)
            pressure_bar = np.ma.masked_where(temp_bar.mask, pressure_bar)


            sal_bar, temp_bar = step_4and5(dataset, lo, la, pressure_bar, sal_bar, temp_bar)

            # Calculuate climatological Dynamic height anomaly
            # max_dp is the maximum difference between layers, set to a very high value
            # to avoid expensive interpolation.
            if  p_ref == 'SeaFloor':
                seafloorpres = pressure_bar.max(axis=0)
                gsdh_clim = gsw.geo_strf_dyn_height(sal_bar, temp_bar, pressure_bar, p_ref = seafloorpres)[0]
            elif p_ref == '2000m':
                seafloorpres = pressure_bar.max(axis=0)
                ref_pressure = gsw.conversions.p_from_z(-2000., la)
                ref_pressure = np.array([np.min([r, f])for r, f in zip(ref_pressure, seafloorpres)])
                print(sal_bar.shape, temp_bar.shape, pressure_bar.shape, ref_pressure.shape)
                gsdh_clim = gsw.geo_strf_dyn_height(sal_bar, temp_bar, pressure_bar, p_ref = ref_pressure)[0]
            else:
                gsdh_clim = gsw.geo_strf_dyn_height(sal_bar, temp_bar, pressure_bar)[0]

        #     print('sal_bar:',sal_bar, '\ntemp_bar:', temp_bar, '\npressure_bar:', pressure_bar)
        #     for pref in [0, 10, 2000, pressure_bar.max()]:
        #         print('reference pressure:', pref, gsw.geo_strf_dyn_height(sal_bar, temp_bar, pressure_bar, p_ref = pref)* 1000. / gravity)
        #     assert 0
            # Convert dynamic height into mm.
            slr_clim[y, :] = gsdh_clim * 1000. / gravity
            print('clim:',y, gsdh_clim * 1000. / gravity, 'mm')

        # Save climatological SLR cube as a netcdf.
        cube0 = thetao_cube[0, 0, :, :].copy()
        cube0.data = slr_clim #p.ma.masked_where(slr_clim==0., slr_clim)
        cube0.units = cf_units.Unit('mm')
        cube0.name = 'Climatological Sea Level Rise'
        cube0.long_name = 'Climatological Sea Level Rise'
        cube0.short_name = 'slr_clim'
        cube0.var_name = 'slr_clim'
        cube0.standard_name = 'steric_change_in_mean_sea_level'
        iris.save(cube0, output_fns['clim'])

    # Now perform the SLR calculation for each point in time for each lat line:
    if depths.ndim == 1:
        depth = np.tile(depths, (len(lat[0]), 1)).T

    for y in np.arange(len(lat[:,0])):
        sal_bar = psal_bar.data[:, y, :]
        temp_bar = thetao_bar.data[:, y, :]
        if np.ma.is_masked(sal_bar.max()):
             continue
        la = lat[y, :]
        lo = lon[y, :]
        print('Calculate SLR in 1D:', y, 'of', len(la) )

        # Load depth dataset
        if depths.ndim == 3:
            depth = depths[:, y, :]
        if depth.shape != sal_bar.shape: assert 0

        if depths.ndim != 4:
            pressure = gsw.conversions.p_from_z(depth, la) # dbar

        # Load climatological depth dataset
        # load clim depth
        if depths_bar.ndim == 1:
            depth_bar = np.tile(depths_bar, (len(la), 1)).T
            print('tiling depth:', depths_bar.shape, sal_bar.shape, depth_bar.shape)
            if depth_bar.shape != sal_bar.shape: assert 0
        elif depths_bar.ndim == 3:
            depth_bar = depths_bar[:, y, :]
        else:
            assert 0

        pressure_bar = gsw.conversions.p_from_z(depth_bar, la) # dbar
        pressure_bar = np.ma.masked_where(sal_bar.mask, pressure_bar)

        # Calculate SLR for each point in time in the historical dataset.
        for t, time in enumerate(times):
            if depths.ndim == 4:
                depth = depths[t, :, y, :]
                pressure = gsw.conversions.p_from_z(depth, la) # dbar

            # load salinity and temperature data
            sal = so_cube.data[t, :, y, :]
            temp = thetao_cube.data[t, :, y, :]
            pressure = np.ma.masked_where(temp.mask, pressure)

           # print('looping 1A:',t,y, time, sal, temp, pressure)
            # Confirm that we use absolute salininty & conservative temperature
            sal, temp = step_4and5(dataset, lo, la, pressure, sal, temp)
            #print('looping 1B:',t,y, time, sal, temp, pressure)

            # Calculate Dynamic height anomaly
            seafloorpres = pressure.max(axis=0)
            if p_ref == 'SeaFloor':
                gsdh_total = gsw.geo_strf_dyn_height(sal, temp, pressure, p_ref = seafloorpres)[0]
                gsdh_thermo = gsw.geo_strf_dyn_height(sal_bar, temp, pressure, p_ref = seafloorpres)[0]
            elif p_ref == '2000m':
                ref_pressure = gsw.conversions.p_from_z(-2000., la)
                ref_pressure = np.array([np.min([r, f])for r, f in zip(ref_pressure, seafloorpres)])
                gsdh_total = gsw.geo_strf_dyn_height(sal, temp, pressure, p_ref = ref_pressure)[0]
                gsdh_thermo = gsw.geo_strf_dyn_height(sal_bar, temp, pressure, p_ref = ref_pressure)[0]
            else:
                gsdh_total = gsw.geo_strf_dyn_height(sal, temp, pressure, )[0] * 1000. / gravity
                gsdh_thermo = gsw.geo_strf_dyn_height(sal_bar, temp, pressure)[0] * 1000. / gravity
                gsdh_halo = gsw.geo_strf_dyn_height(sal, temp_bar, pressure)[0] * 1000. / gravity

            # Convert to mm and calculate anomaly relative to clim data
            gsdh_total = gsdh_total - slr_clim[y, :]
            gsdh_thermo = gsdh_thermo - slr_clim[y, :]
            gsdh_halo = gsdh_halo - slr_clim[y, :]

            # Put in the output array:
            slr_total[t, y, :] = gsdh_total
            slr_thermo[t, y, :] = gsdh_thermo
            slr_halo[t, y, :] = gsdh_halo

            print('----\n', [t,'of', len(times)], [y,'of',len(lat[:,0])], time)
            print(dataset, 'total', gsdh_total.min(),  gsdh_total.max())
            print(dataset, 'slr_thermo', gsdh_thermo.min(), gsdh_thermo.max())
            print(dataset, 'halo', gsdh_halo.min(), gsdh_halo.max())

            count += 1
            if t == 0:
                print(count, (t, y), 'performing 2D SLR')

    slr_total  = np.ma.masked_where(thetao_cube[:, 0, :, :].data.mask, slr_total)
    slr_thermo = np.ma.masked_where(thetao_cube[:, 0, :, :].data.mask, slr_thermo)
    slr_halo   = np.ma.masked_where(thetao_cube[:, 0, :, :].data.mask, slr_halo )

    # Save SLR data as a cube and then a netcdf..
    cube0 = thetao_cube[:, 0, :, :].copy()
    cube0.data = slr_total
    cube0.units = cf_units.Unit('mm')
    cube0.name = 'Total Sea Level Rise'
    cube0.long_name = 'Total Sea Level Rise'
    cube0.short_name = 'slr_total'
    cube0.var_name = 'slr_total'
    cube0.standard_name = 'steric_change_in_mean_sea_level'
    iris.save(cube0, output_fns['total'])

    cube1 = thetao_cube[:, 0, :, :].copy()
    cube1.data = slr_thermo
    cube1.units = cf_units.Unit('mm')
    cube1.name = 'Thermosteric Sea Level Rise'
    cube1.long_name = 'Thermosteric Sea Level Rise'
    cube1.short_name = 'slr_thermo'
    cube1.var_name = 'slr_thermo'
    cube1.standard_name = 'thermosteric_change_in_mean_sea_level'
    iris.save(cube1, output_fns['thermo'])

    cube2 = thetao_cube[:, 0, :, :].copy()
    cube2.data = slr_halo
    cube2.units = cf_units.Unit('mm')
    cube2.name = 'Halosteric Sea Level Rise'
    cube2.long_name = 'Halosteric Sea Level Rise'
    cube2.short_name = 'slr_halo'
    cube2.var_name = 'slr_halo'
    cube2.standard_name = 'halosteric_change_in_mean_sea_level'
    iris.save(cube2, output_fns['halo'])

    return output_fns['total'], output_fns['thermo'], output_fns['halo']


def calc_slr_timeseries(cfg, slr_fn, areacella_fn, project, dataset, exp, ensemble, slr_type):
    """
    Calculate SLR time series.
    """
    work_dir = diagtools.folder([cfg['work_dir'], 'SLR'])
    slr_ts_fn = work_dir + '_'.join([project, dataset, exp, ensemble, slr_type, 'timeseries'])+'.nc'
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
    path += '_'.join([project, dataset, exp, ensemble, slr_type, 'timeseries'])+'.png'
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
        cube1 = iris.load_cube(output_ohc_fn)
        for t in [0, -1]:
            single_pane_map_plot(
                    cfg,
                    metadatas[thetao_fn],
                    cube1[t, 0],
                    key='OHC_full_'+trend
                    )

        return output_ohc_fn

    # Load netcdf files
    thetao_cube = iris.load_cube(thetao_fn)
    so_cube = iris.load_cube(so_fn)
    vol_cube = iris.load_cube(volcello_fn)

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

    # Determine image filename
    unique_id = [dataset, exp, ensemble, short_name] #ear ]
    path = diagtools.folder([cfg['plot_dir'], 'volume_integrated_plot']) + '_'.join(unique_id)
    path = path.replace(' ', '') + diagtools.get_image_format(cfg)

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

        for (project, dataset, exp, ensemble, slr_type, trend), slr_fn in slr_fns.items():
            if sbp_slr != slr_type: continue
            if plot_dataset == 'all': pass
            elif dataset != plot_dataset: continue

            if trend == 'detrended': ls = '-'
            if trend == 'intact': ls = ':'

            cube = iris.load_cube(slr_fn)
            times = diagtools.cube_time_to_float(cube)
            data = np.ma.masked_invalid(cube.data)
            print(project, dataset, exp, ensemble, slr_type, sbp_slr, times, data)
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



def SLR_map_plot(cfg, metadata, slr_fn, slr_type):
    """
     Make a plot of the SLR final decade.
    """
    short_name = metadata['short_name']
    dataset = metadata['dataset']
    ensemble = metadata['ensemble']
    project = metadata['project']
    exp = metadata['exp']

    cube = iris.load_cube(slr_fn)

    t1 = cube[-10:].copy()
    t1 = t1.collapsed([t1.coord('time'),], iris.analysis.MEAN)
    times = diagtools.cube_time_to_float(cube)
    time_str = str(int(times[-10])) + ' - ' + str(int(times[-1]))

    cube = t1
    unique_id = [dataset, exp, ensemble, slr_type, time_str]

    # Determine image filename
    path = diagtools.folder([cfg['plot_dir'], 'SLR_map_plots']) + '_'.join(unique_id)
    path = path.replace(' ', '') + diagtools.get_image_format(cfg)

    cmap='coolwarm'
    max_val = np.max(np.abs([cube.data.min(), cube.data.max()]))
    nspace = np.linspace(
        -max_val, max_val, 21, endpoint=True)

    title = ' '.join(unique_id)

    print('SLR_map_plot', unique_id, nspace, [cube.data.min(), cube.data.max()], cube.data.shape)

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


    parrallel = False
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
    assert 0

def guess_volcello_fn(dicts, keys, optional = []):
    """
    Take a punt at the volcello filename.
    """
    for index, value in dicts.items():
        both = (keys+optional)
        intersection = set(index) & set(both)
        if len(intersection) == len(both):
            print("guess_volcello_fn: full match", index, keys, optional, value)
            return value

        intersection = set(index) & set(keys)
        if len(intersection) == len(keys):
            print("guess_volcello_fn: partial match", index, keys, value)
            return value
    print('Did Not find Volcello:', keys, optional)
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
    assert 0
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
        #print('creation loop', filename,':', metadatas[filename])
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
    slr_fns = {}
    do_SLR = True
    trends = ['detrended', 'intact']
    for trend in trends:
        for (project, dataset, exp, ensemble, short_name)  in sorted(detrended_ncs.keys()):
            if not do_SLR: continue
            if short_name != 'thetao':
                continue
            if exp ==  'piControl': # no need to calculate this.
                continue

            pi_ensemble = guess_PI_ensemble(trend_shelves, [project, dataset, short_name], ens_pos = 3)

            if trend == 'detrended':
                hist_thetao_fn = detrended_ncs[(project, dataset, exp, ensemble, short_name)]
                hist_so_fn =  detrended_ncs[(project, dataset, exp, ensemble, 'so')]
                picontrol_thetao_fn = detrended_ncs[(project, dataset, 'piControl', pi_ensemble, short_name)]
                picontrol_so_fn =  detrended_ncs[(project, dataset, 'piControl', pi_ensemble, 'so')]

            if trend ==  'intact':
                hist_thetao_fn = file_dict[(project, dataset, exp, ensemble, 'thetao')]
                hist_so_fn = file_dict[(project, dataset, exp, ensemble, 'so')]
                picontrol_thetao_fn = file_dict[(project, dataset, 'piControl', pi_ensemble, short_name)]
                picontrol_so_fn =  file_dict[(project, dataset, 'piControl', pi_ensemble, 'so')]


            #slr_total_fn, slr_thermo_fn, slr_halo_fn =
            calc_slr_full(cfg,
                metadatas,
                hist_thetao_fn,
                hist_so_fn,
                picontrol_thetao_fn,
                picontrol_so_fn,
                trend=trend
                )
            continue

            print(slr_total_fn, slr_thermo_fn, slr_halo_fn)
            slr_fns[(project, dataset, exp, ensemble, 'slr_total', trend)] = slr_total_fn
            slr_fns[(project, dataset, exp, ensemble, 'slr_thermo', trend)] = slr_thermo_fn
            slr_fns[(project, dataset, exp, ensemble, 'slr_halo', trend)] = slr_halo_fn

            metadatas[slr_total_fn] = metadatas[hist_thetao_fn]
            metadatas[slr_thermo_fn] = metadatas[hist_thetao_fn]
            metadatas[slr_halo_fn] = metadatas[hist_thetao_fn]

            SLR_sanity_check(
                cfg,
                metadatas[slr_total_fn],
                slr_total_fn,
                slr_thermo_fn,
                slr_halo_fn)

            # Calculate time series?
            areacella_fn = guess_areacello_fn(file_dict, [project, dataset, 'areacello'])
            for slr_fn, slr_type in zip([slr_total_fn, slr_thermo_fn, slr_halo_fn], ['slr_total', 'slr_thermo', 'slr_halo']):
                slr_ts_fn = calc_slr_timeseries(cfg, slr_fn, areacella_fn, project, dataset, exp, ensemble, slr_type)
                slr_fns[(project, dataset, exp, ensemble, slr_type+'_ts', trend)] = slr_ts_fn
                metadatas[slr_ts_fn] = metadatas[detrended_fn]
            # break

    # Plot SLR maps:
    for (project, dataset, exp, ensemble, slr_type, trend), slr_fn in slr_fns.items():
        # Can't make a map plot for a time series.
        if slr_type.find('_ts') > -1: continue
        SLR_map_plot(cfg, metadatas[slr_fn], slr_fn, slr_type)

    SLR_multimodel_plot(cfg, metadatas, slr_fns)
    datasets = list({index[1]:True for index in slr_fns.keys()}.keys())
    for dataset in datasets:
        SLR_multimodel_plot(cfg, metadatas, slr_fns, plot_dataset = dataset)
    assert 0

    print('\nCalculate ocean heat content - trend intact')
    for (project, dataset, exp, ensemble, short_name), fn in file_dict.items():
        if short_name != 'thetao':
            continue
        #if exp ==  'piControl': # no need to calculate this.
        #    continue
        # volcello is in same ensemble, same exp.
        volcello_fn = guess_volcello_fn(file_dict, [project, dataset, 'volcello'],optional=[ensemble, exp])
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

        volcello_fn = guess_volcello_fn(file_dict, [project, dataset, 'volcello'],optional=[ensemble, exp])
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

    for (project, dataset, exp, ensemble, keya, keyb), ohc_fn in ocean_heat_content.items():
        areacello_fn = guess_areacello_fn(file_dict, [project, dataset, 'areacello'])
        volume_integrated_plot(cfg, metadatas[ohc_fn], ohc_fn, areacello_fn)

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
