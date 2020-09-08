#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic script to plot figure 9.42a of IPCC AR5 chapter 9.

Description
-----------
Calculate and plot trends in CO2 Seasonal cycle amplitude

Author
------
Bettina Gier (Univ. of Bremen, Germany)

Project
-------
Eval4CMIP

Configuration options in recipe
-------------------------------
save : dict, optional
    Keyword arguments for the `fig.saveplot()` function.

"""

import logging
import os

import iris
from iris import Constraint
import iris.quickplot
import matplotlib.pyplot as plt
import matplotlib.dates as mda
# import iris.plot as iplt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from esmvaltool.diag_scripts.shared import (
    ProvenanceLogger, extract_variables, get_diagnostic_filename,
    get_plot_filename, group_metadata, io, plot, run_diagnostic,
    variables_available)
import esmvaltool.diag_scripts.shared.ESRL_ccgcrv.ccgfilt as ccgfilt
from esmvaltool.diag_scripts.shared.ESRL_ccgcrv.ccgdates import (
    decimalDate, decimalDateFromDatetime, calendarDate, datetimeFromDecimalDate)
from matplotlib import rcParams
import numpy as np
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

logger = logging.getLogger(os.path.basename(__file__))

def time_slice(cube, start_year, end_year, start_month = 1, start_day = 1, end_month = 12,
               end_day = 31):
    """Slice cube on time.
    Parameters
    ----------
        cube: iris.cube.Cube
            input cube.
        start_year: int
            start year
        start_month: int
            start month
        start_day: int
            start day
        end_year: int
            end year
        end_month: int
            end month
        end_day: int
            end day
    Returns
    -------
    iris.cube.Cube
        Sliced cube.
    """
    import datetime
    time_units = cube.coord('time').units
    if time_units.calendar == '360_day':
        if start_day > 30:
            start_day = 30
        if end_day > 30:
            end_day = 30
    start_date = datetime.datetime(
        int(start_year), int(start_month), int(start_day))
    end_date = datetime.datetime(int(end_year), int(end_month), int(end_day))

    t_1 = time_units.date2num(start_date)
    t_2 = time_units.date2num(end_date)
    constraint = iris.Constraint(
            time=lambda t: (t_1 < time_units.date2num(t.point) < t_2))

    cube_slice = cube.extract(constraint)

    # Issue when time dimension was removed when only one point as selected.
    if cube_slice.ndim != cube.ndim:
        time_1 = cube.coord('time')
        time_2 = cube_slice.coord('time')
        if time_1 == time_2:
            logger.debug('No change needed to time.')
            return cube

    return cube_slice

def get_provenance_record():
    """Create a provenance record describing the diagnostic data and plot."""
    record = {
        'caption':
        ('Changes to the amplitude of the seasonal cycle of '
         'atmospheric CO2 at Mauna Loa and global NBP. Observations and estimates '
         'from global land models.(Zhao et al., 2016).'),
        'statistics': ['mean', 'stddev', 'trend'],
        'domains': ['global'],
        'plot_types': ['times', 'seas'],
        'authors': ['gier_bettina'],
        'references': ['zhao2016bg'],
        'realms': ['atmos'],
        'themes': ['phys']
    }
    return record

def plot_data(timeranges, yrs, amps, t_cycle, avg_cycle, dataset_groups, filename):
    min_year = min([timeranges[dataset][0] for dataset in timeranges])
    max_year = max([timeranges[dataset][1] for dataset in timeranges])

    """Plot data"""
    linestyles = {"CO2-MLO": "-",  "CO2-GLOB": "dashed", "JMA-TRANSCOM": "dotted"}
    colors = {"Multi-Model Mean": (204/255., 35/255., 35/255.), "CO2-MLO": (0, 0, 0),
              "CO2-GLOB": (0, 0, 0), "JMA-TRANSCOM": (0, 0, 0),
              #"CO2-GLOB": (36/255., 147/255., 126/255.), "JMA-TRANSCOM": (54/255., 156/255., 232/255.,),
              "first10": (237/255., 128/255., 55/255.), "last10": (36/255., 147/255., 126/255.)}
    CMIP6_colors = [(0, 0, 0), (112/255., 160/255., 205/255.), (178/255., 178/255., 178/255.), (196/255., 121/255., 0), (0/255., 52/255., 102/255.), (0, 79/255., 0)]
    CMIP6_shading = [(128/255., 128/255., 128/255.), (91/255., 174/255., 178/255.), (191/255., 191/255., 191/255.), (204/255., 174/255., 113/255.), (67/255., 147/255., 195/255.), (223/255., 237/255., 195/255.)]
    fig, ax = plt.subplots()#plt.figure(figsize=(6, 4))

    # Compute MMM Shading
    mmm_amps = {}
    for dataset in dataset_groups['mmm_datasets']:
        mmm_amps[dataset] = amps[dataset]
    mmm_time, mmm_mean, mmm_std = compute_mmm_std(timeranges, mmm_amps, "yearly")
    plt.plot(mmm_time, mmm_mean, label = "Multi-Model Mean", color=colors["Multi-Model Mean"], linewidth = 3)
    plt.fill_between(mmm_time, mmm_mean-mmm_std, mmm_mean + mmm_std, color=colors["Multi-Model Mean"], alpha=.2)

    # Non-mmm lines
    for dataset in yrs:
        if not dataset in dataset_groups['mmm_datasets']:
            # Possibly need to readjust if start_year >> mmm_start_year to get relative amplitude in line
            if timeranges[dataset][0] > min_year:
                # Make mean of last 10 years of dataset similar to mmm_mean
                mean_dataset_10 = np.mean(amps[dataset][-10:])
                if timeranges[dataset][1] < max_year:
                    mmm_mean_10 = np.mean(mmm_mean[-10+timeranges[dataset][1]-\
                                                        max_year:timeranges[dataset][1]-max_year])
                else:
                    mmm_mean_10 = np.mean(mmm_mean[-10:])
            else:
                mmm_mean_10 = 0.
                mean_dataset_10 = 0.
            if dataset in dataset_groups['ref_model']:
                plt.plot(yrs[dataset], amps[dataset] - mean_dataset_10 + mmm_mean_10, label = dataset,
                         color = colors[dataset], linewidth = 3)
                plt.fill_between(yrs[dataset], amps[dataset] - amps['ref_std'] - mean_dataset_10 + mmm_mean_10, \
                                     amps[dataset] + amps['ref_std'] - mean_dataset_10 + mmm_mean_10, \
                                     color = colors[dataset], alpha=.2) #CMIP6_shading.pop(0), alpha=.3)
            else:
                plt.plot(yrs[dataset], amps[dataset] - mean_dataset_10 + mmm_mean_10,
                         label = dataset, color = colors[dataset], linewidth = 2, linestyle=linestyles[dataset])
                #CMIP6_shading.pop(0)
    # Prettify
    plt.ylabel('Relative change in amplitude')
    plt.legend(fontsize=10, loc = 2)
    plt.xlim(min_year, max_year-1)
    plt.ylim(0.78)

    # INSET
    axins = inset_axes(ax, width='30%', height='30%', loc=4, borderpad=2)
    #CMIP6_colors = [(0, 0, 0), (112/255., 160/255., 205/255.), (196/255., 121/255., 0), (178/255., 178/255., 178/255.), (0/255., 52/255., 102/255.), (0, 79/255., 0)]
    CMIP6_shading = [(128/255., 128/255., 128/255.), (91/255., 174/255., 178/255.), (204/255., 174/255., 113/255.), (191/255., 191/255., 191/255.), (67/255., 147/255., 195/255.), (223/255., 237/255., 195/255.)]
    # Initial
    iyrs = [min_year, min_year+9]
    cycle_data = {}
    for dataset in avg_cycle:
        int_ind = (iyrs[0] - timeranges[dataset][0]) * 12
        fin_ind = int_ind + (iyrs[1] - timeranges[dataset][0]) * 12 + 11
        if int_ind>=0 and fin_ind<len(avg_cycle[dataset]):
            cycle_data[dataset] = compute_avg_sca(t_cycle[dataset][int_ind:fin_ind], \
                                                      avg_cycle[dataset][int_ind:fin_ind])
            if dataset in dataset_groups['plot_sca']:
                axins.plot(range(1, 13), cycle_data[dataset], linestyle = "--", color = colors[dataset])#color = CMIP6_colors.pop(1))
    # Compute MMM cycle
    mmm_cycle_data = {}
    for dataset in dataset_groups['mmm_datasets']:
        mmm_cycle_data[dataset] = cycle_data[dataset]
    _, mmm_meanc, mmm_stdc =  compute_mmm_std(timeranges, mmm_cycle_data, "cycle")
    axins.plot(range(1, 13), mmm_meanc, "k--", label = "Multi-Model Mean", color=colors["first10"])
    axins.fill_between(range(1, 13), mmm_meanc-mmm_stdc, mmm_meanc + mmm_stdc, color = colors["first10"], alpha=.2)

    # Final
    #fyrs = [1999, 2008]
    fyrs = [max_year-9, max_year]
    cycle_data = {}
    for dataset in avg_cycle:
        int_ind = (fyrs[0] - timeranges[dataset][0]) * 12
        fin_ind = len(avg_cycle[dataset]) - (timeranges[dataset][1] - fyrs[1]) * 12
        # int_ind doesn't seem quite right yet..
        if int_ind>0 and fin_ind<len(avg_cycle[dataset]):
            cycle_data[dataset] = compute_avg_sca(t_cycle[dataset][int_ind:fin_ind], \
                                                      avg_cycle[dataset][int_ind:fin_ind])
            if dataset in dataset_groups['plot_sca']:
                axins.plot(range(1, 13), cycle_data[dataset])
        elif int_ind>0 and fin_ind==len(avg_cycle[dataset]):
            cycle_data[dataset] = compute_avg_sca(t_cycle[dataset][int_ind:], \
                                                      avg_cycle[dataset][int_ind:])
            if dataset in dataset_groups['plot_sca']:
                axins.plot(range(1, 13), cycle_data[dataset])
    # Compute MMM cycle
    mmm_cycle_data = {}
    for dataset in dataset_groups['mmm_datasets']:
        mmm_cycle_data[dataset] = cycle_data[dataset]
    _, mmm_meanf, mmm_stdf =  compute_mmm_std(timeranges, mmm_cycle_data, "cycle")
    axins.plot(range(1, 13), mmm_meanf, "k", label = "Multi-Model Mean", color = colors["last10"])
    axins.fill_between(range(1, 13), mmm_meanf-mmm_stdf, mmm_meanf + mmm_stdf, color = colors["last10"], alpha=.2)#color=(128/255., 128/255., 128/255.),alpha=.3)


    xlab = range(2, 13, 2)
    xlabels = ['Feb', 'Apr', 'Jun', 'Aug', 'Oct', 'Dec']
    plt.xticks(xlab, xlabels)
    plt.ylabel(r'NBP PgC yr$^{-1}$')
    #plt.ylabel(r'F$_{TA}$ PgC yr$^{-1}$')
    axins.patch.set_alpha(0.0)

    fig.savefig(filename)
    plt.close(fig)

def compute_mmm_std(timeranges, data, freq):
    """Compute Multi-Model Mean and its std"""
    min_year = min([timeranges[dataset][0] for dataset in timeranges])
    #min_year2 = min([dataset['start_year'] for dataset in cfg['input_data'].values()])
    max_year = max([timeranges[dataset][1] for dataset in timeranges])
    if freq == "cycle":
        timelength = 12
        mmm_time = range(1, 13)
    elif freq == "monthly":
        timelength = (max_year - min_year + 1) * 12
        mmm_time = np.arange(min_year + 0.5/12, max_year + 1, 1./12)
    elif freq == "yearly":
        timelength = max_year - min_year #SCA is missing the last year
        mmm_time = np.arange(min_year, max_year, 1)
    else:
        logger.info("Frequency not supported!")
    mmm_mean = np.zeros(timelength)
    mmm_std = np.zeros(timelength)
    for ii in range(timelength):
        dataset_values = []
        for dataset in data:
            if freq == "cycle":
                dataset_values.append(data[dataset][ii])
            else:
                if timeranges[dataset][0] <= mmm_time[ii] <= timeranges[dataset][1]:
                    if freq == "monthly":
                        dataset_values.append(data[dataset][ii - (min_year - timeranges[dataset][0])*12])
                    elif freq == "yearly":
                        dataset_values.append(data[dataset][ii - (min_year - timeranges[dataset][0])])
        mmm_std[ii] = np.std(dataset_values)
        mmm_mean[ii] = np.mean(dataset_values)
    return mmm_time, mmm_mean, mmm_std

def compute_avg_sca(t_data, cycle_data):
    """Compute average Seasonal Cycle"""
    time_data = [datetimeFromDecimalDate(dt) for dt in t_data]
    monthly = {}
    for itim, time in enumerate(time_data):
        if time.month in monthly:
            monthly[time.month].append(cycle_data[itim])
        else:
            monthly[time.month] = [cycle_data[itim]]
    avg_sca = np.zeros(12)
    for month in monthly:
        avg_sca[month-1] = np.mean(monthly[month])
    return avg_sca

# From http://stackoverflow.com/a/14314054/3293881 by @Jaime
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# From http://stackoverflow.com/a/40085052/3293881
def strided_app(a, L, S=1 ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

# https://stackoverflow.com/questions/43284304/how-to-compute-volatility-standard-deviation-in-rolling-window-in-pandas
def rolling_meansqdiff_numpy(a, w):
    A = strided_app(a, w)
    B = moving_average(a,w)
    subs = A-B[:,None]
    sums = np.einsum('ij,ij->i',subs,subs)
    return (sums/w)**0.5

def main(cfg):
    """Run the diagnostic."""
    #logger.info(cfg['input_data'].keys())
    data_sca = {}
    rel_trend_sca = {}
    dataset_yrs = {}
    sca_x0 = {}
    timeranges = {}
    dataset_groups = {}
    dataset_groups['mmm_datasets'] = []
    dataset_groups['plot_sca'] = []
    dataset_groups['ref_model'] = cfg['ref_model']

    if 'co2data' in cfg:
        for dataset in cfg['co2data']:
            #logger.info(dataset)
            cfg['input_data'][dataset] = {'filename': cfg['co2data'][dataset][0],
                                          'dataset': dataset, 'short_name': 'co2',
                                          'start_year': cfg['co2data'][dataset][1],
                                          'end_year': cfg['co2data'][dataset][2],
                                          'project': 'OBS'}

    input_data = cfg['input_data'].values()
    #logger.info(input_data)
    for data in input_data:
        # Deal with co2 read-in differently!
        name = data['dataset']
        logger.info("Processing %s", name)
        cube_data = iris.load_cube(data['filename'])
        # FTA is -NBP
        #if data['short_name'] == 'nbp':
        #    cube_data = cube_data * (-1)

        if data['short_name'] == 'co2':
            cube_data = time_slice(cube_data, start_year = data['start_year'], end_year = data['end_year'])
            # Need to select the right years

        timeranges[name] = [data['start_year'], data['end_year']]
        #logger.info(data)
        x_data = cube_data.coord('time')

        # Turn time coordinate into Year.frac for the NOAA ccgfilt routine
        t_data = x_data.units.num2date(x_data.points)
        t_data = [decimalDateFromDatetime(dt) for dt in t_data]

        # Calculate seasonal cycle amplitudes
        t_data = np.asarray(t_data)
        idx = np.where((np.isnan(cube_data.data)==False) & (np.isinf(cube_data.data)==False))
        filt = ccgfilt.ccgFilter(t_data[idx[0]], cube_data.data[idx])
        sca = filt.getAmplitudes()
        dataset_yrs[name], amps, _, _, _, _ = list(zip(*sca))

        # Subtract means, compute relative change in amplitude
        # 9 year gaussian smoothing
        # window length: w = 2 * int(truncate*sigma + 0.5) + 1
        gaus_amps = gaussian_filter(amps, 1.2, truncate=3.0)
        sca_mean_smooth = np.mean(gaus_amps[:11])
        # relative trends
        rel_trend_sca[name] = gaus_amps/sca_mean_smooth

        # Compute std of smoothing - only need this for ref_model!
        if name in dataset_groups['ref_model']:
            sca_mean = np.mean(amps[:11])
            rel_trend_unsmoothed = amps/sca_mean
            resid = np.array(rel_trend_unsmoothed) - np.array(rel_trend_sca[name])
            rolling_std = rolling_meansqdiff_numpy(resid, 9)
            # Extending first and last std outwards
            rel_trend_sca['ref_std'] = np.concatenate(([rolling_std[0]]*4, rolling_std, \
                                                          [rolling_std[-1]]*4), axis=None)
        #logger.info(rolling_std)

        # Get smooth Seasonal Cycle
        sca_x0[name] = filt.xinterp
        harmonics = filt.getHarmonicValue(sca_x0[name])
        # unit covnersion: -> PgC/yr *12 for months, * Earth surface
        data_sca[name] = (harmonics + filt.smooth - filt.trend) \
            * 3600. * 24. * 365 / 1.e12 * 5100644720000

        # Determine if model or OBS
        if 'CMIP' in data['project']:
            dataset_groups['mmm_datasets'].append(name)
        else:
            if data['short_name'] == 'nbp':
                dataset_groups['plot_sca'].append(name)
    # Plot data
    min_year = min([timeranges[dataset][0] for dataset in timeranges])
    max_year = max([timeranges[dataset][1] for dataset in timeranges])
    plot_path = get_plot_filename('SCA_trend_xy_' + str(min_year) + '_' + str(max_year), cfg)
    plot_data(timeranges, dataset_yrs, rel_trend_sca, sca_x0, data_sca, dataset_groups, plot_path)

    # Write netcdf file TODO! Currently just dummy
    netcdf_path = get_diagnostic_filename('SCA_trend_xy_' + str(min_year) + '_' + str(max_year), cfg)
    #save_scalar_data(rel_trend_sca, netcdf_path, var_attrs)
    #netcdf_path = write_data(cfg, hist_cubes, pi_cubes, ecs_cube)

    # Provenance
    provenance_record = get_provenance_record()
    if plot_path is not None:
        provenance_record['plot_file'] = plot_path
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(netcdf_path, provenance_record)

if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
