#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot based on Santer et al. (2020).

###############################################################################
santer20jclim/santer20jclimfig.py
Author: Katja Weigel (IUP, Uni Bremen, Germany)
EVal4CMIP ans 4C project
###############################################################################

Description
-----------
    Total column water vapour trends following Santer et al. (2020).

Configuration options
---------------------
filer: optional, filter all data sets (netCDF file with 0 and 1 for used grid).
       The data must be interpolated to the same lat/lon grid as the filter.
       The filter must cover at least the time period used for the data.

###############################################################################

"""

import logging
import os
from collections import OrderedDict
from pprint import pformat
from cf_units import Unit
import iris
import iris.coord_categorisation as cat
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# from scipy.stats import norm
from esmvaltool.diag_scripts.shared import (ProvenanceLogger,
                                            plot, get_diagnostic_filename,
                                            get_plot_filename,
                                            group_metadata,
                                            select_metadata, run_diagnostic,
                                            variables_available)

logger = logging.getLogger(os.path.basename(__file__))


def _apply_filter(cfg, cube):
    """Apply filter from RSS Anomalies to all data and calculates mean."""
    if 'filter' in cfg:
        filt = iris.load(cfg['filter'])[0]
        timefil = filt.coord('time')
        latfil = filt.coord('latitude')
        lonfil = filt.coord('longitude')
        timecu = cube.coord('time')

        start = timecu.units.num2date(timecu.points[0])
        end = timecu.units.num2date(timecu.points[-1])
        stime = timefil.nearest_neighbour_index(timefil.units.date2num(start))
        etime = timefil.nearest_neighbour_index(timefil.units.date2num(end))
        filt = filt[stime:etime + 1, :, :]

        startlat = cube.coord('latitude').points[0]
        endlat = cube.coord('latitude').points[-1]
        slat = latfil.nearest_neighbour_index(startlat)
        elat = latfil.nearest_neighbour_index(endlat)
        filt = filt[:, slat:elat + 1, :]

        startlon = cube.coord('longitude').points[0]
        endlon = cube.coord('longitude').points[-1]
        slon = lonfil.nearest_neighbour_index(startlon)
        elon = lonfil.nearest_neighbour_index(endlon)
        filt = filt[:, :, slon:elon + 1]

        cube.data = cube.data * filt.data

    coords = ('longitude', 'latitude')
    cube_grid_areas = iris.analysis.cartography.area_weights(cube)
    cube = (cube.collapsed(coords, iris.analysis.MEAN,
                           weights=cube_grid_areas))

    return cube


def _calculate_anomalies(cube):
    """Remove annual cycle from time series."""
    c_n = cube.aggregated_by('month_number', iris.analysis.MEAN)
    month_number = cube.coord('month_number').points
    startind = np.where(month_number == 1)
    endind = np.where(month_number == 12)
    data_in = cube.data
    data_new = np.full(data_in.shape, 0.5)
    for iii in range((startind[0])[0], (endind[0])[-1], 12):
        data_new[iii:iii + 12] = ((cube.data[iii:iii + 12] - c_n.data) /
                                  c_n.data) * 100.0 * 120.0

    cube.data = data_new
    cube.units = Unit('percent')

    return cube


def _check_full_data(dataset_path, cube):
    """Check if cube covers time series from start year to end year."""
    # cstart_month = cube.coord('month_number').points[0]
    # cend_month = cube.coord('month_number').points[1]
    cstart_year = cube.coord('year').points[0]
    cend_year = cube.coord('year').points[-1]
    start_year = int(dataset_path['start_year'])
    end_year = int(dataset_path['end_year'])

    check = 0
    if start_year == cstart_year and end_year == cend_year:
        check = 1
        print("Full time series:")
        print(start_year, cstart_year, end_year, cend_year)
    else:
        print("FAILED:")
        print(start_year, cstart_year, end_year, cend_year)

    return check


def _get_sel_files(cfg, dataname, dim=2):
    """Get filenames from cfg for single models or multi-model mean."""
    selection = []
    if dim == 2:
        for hlp in select_metadata(cfg['input_data'].values(),
                                   dataset=dataname):
            selection.append(hlp['filename'])
    else:
        for hlp in cfg['input_data'].values():
            selection.append(hlp['filename'])

    return selection


def _get_sel_files_var(cfg, varnames):
    """Get filenames from cfg for all model mean and differen variables."""
    selection = []

    for var in varnames:
        for hlp in select_metadata(cfg['input_data'].values(), short_name=var):
            selection.append(hlp['filename'])

    return selection


def _plot_extratrends(cfg, extratrends, trends):
    """Plot trends for ensembles."""
    xhist = np.linspace(0, 4, 41)
    artrend = {}
    kde1 = {}
    fig, axx = plt.subplots(figsize=(8, 6))
    maxh = 2.5

    for extramodel in cfg['add_model_dist']:
        teststr = list(extratrends[extramodel].keys())[0]
        if teststr in trends['cmip6'].keys():
            style = plot.get_dataset_style(extramodel, style_file='cmip6')
            facec = style['facecolor']
            edgec = style['color']
            colc = style['color']
        elif teststr in trends['cmip5'].keys():
            style = plot.get_dataset_style(extramodel, style_file='cmip5')
            facec = style['facecolor']
            edgec = style['color']
            colc = style['color']
        else:
            facec = (0, 0, 1, 0.1)
            edgec = (0, 0, 1, 0.6)
            colc = (0, 0, 1, 1.0)

        artrend[extramodel] = np.fromiter(extratrends[extramodel].values(),
                                          dtype=float)
        kde1[extramodel] = stats.gaussian_kde(artrend[extramodel],
                                              bw_method="scott")
        axx.hist(artrend[extramodel], bins=xhist, density=True,
                 edgecolor=edgec,
                 facecolor=facec)
        axx.plot(xhist, kde1[extramodel](xhist),
                 color=colc,
                 linewidth=3,
                 label=extramodel)

    if trends['obs']:
        for iii, obsname in enumerate(trends['obs'].keys()):
            obscoli = float(iii)
            if iii > 4:
                obscoli = float(iii) - 4.5
            if iii > 8:
                obscoli = float(iii) - 8.75
            if iii > 12:
                obscoli = float(iii) - 12.25
            axx.vlines(trends['obs'][obsname], 0, maxh,
                       colors=(1.0 - 0.25 * obscoli, 0.25 * obscoli,
                               0.5 + obscoli * 0.1),
                       linewidth=3,
                       label=obsname)

    axx.legend(loc=0)
    axx.set_ylim([0, 2.5])
    axx.set_ylabel('Probability density')
    axx.set_xlabel('Trend in Water Vapor Path [%/dec]')
    fig.tight_layout()
    fig.savefig(get_plot_filename('extrabar', cfg), dpi=300)

    plt.close()


def _plot_trends(cfg, trends):
    """Plot trends."""
    xhist = np.linspace(0, 4, 41)
    # CMIP5
    if trends['cmip5']:
        artrend_c5 = np.fromiter(trends['cmip5'].values(), dtype=float)
        weights_c5 = np.fromiter(trends['cmip5weights'].values(), dtype=float)
        # yyy_c5, xxx_c5 = np.histogram(artrend_c5, bins=xhist)
        kde1_c5 = stats.gaussian_kde(artrend_c5, weights=weights_c5,
                                     bw_method="scott")
    # CMIP6
    if trends['cmip6']:
        artrend_c6 = np.fromiter(trends['cmip6'].values(), dtype=float)
        weights_c6 = np.fromiter(trends['cmip6weights'].values(), dtype=float)
        # yyy_c6, xxx_c6 = np.histogram(artrend_c6, bins=xhist)
        kde1_c6 = stats.gaussian_kde(artrend_c6, weights=weights_c6,
                                     bw_method="scott")

    fig, axx = plt.subplots(figsize=(8, 6))

    maxh = 2.5
    # CMIP5
    if trends['cmip5']:
        axx.hist(artrend_c5, bins=xhist, density=True,
                 weights=weights_c5,
                 edgecolor=(0, 0, 1, 0.6),
                 facecolor=(0, 0, 1, 0.1))
        axx.plot(xhist, kde1_c5(xhist),
                 color=(0, 0, 1, 1),
                 linewidth=3,
                 label="CMIP5")
        # maxh = np.max(kde1_c5(xhist))

    # CMIP6
    if trends['cmip6']:
        axx.hist(artrend_c6, bins=xhist, density=True,
                 weights=weights_c6,
                 edgecolor=(0.8, 0.4, 0.1, 0.6),
                 facecolor=(0.8, 0.4, 0.1, 0.1))
        axx.plot(xhist, kde1_c6(xhist),
                 color=(0.8, 0.4, 0.1, 1),
                 linewidth=3,
                 label="CMIP6")
        # maxh = np.max(kde1_c6(xhist))

    if trends['obs']:
        for iii, obsname in enumerate(trends['obs'].keys()):
            obscoli = float(iii)
            if iii > 4:
                obscoli = float(iii) - 4.5
            if iii > 8:
                obscoli = float(iii) - 8.75
            if iii > 12:
                obscoli = float(iii) - 12.25
            axx.vlines(trends['obs'][obsname], 0, maxh,
                       colors=(1.0 - 0.25 * obscoli, 0.25 * obscoli,
                               0.5 + obscoli * 0.1),
                       linewidth=3,
                       label=obsname)

    axx.legend(loc=0)
    axx.set_ylim([0, 2.5])
    axx.set_ylabel('Probability density')
    axx.set_xlabel('Trend in Water Vapor Path [%/dec]')
    fig.tight_layout()
    fig.savefig(get_plot_filename('bar', cfg), dpi=300)

    plt.close()


def get_provenance_record(ancestor_files, caption, statistics,
                          domains, plot_type='geo'):
    """Get Provenance record."""
    record = {
        'caption': caption,
        'statistics': statistics,
        'domains': domains,
        'plot_type': plot_type,
        'themes': ['atmDyn', 'monsoon', 'EC'],
        'authors': [
            'weigel_katja',
        ],
        'references': [
            'li17natcc',
        ],
        'ancestors': ancestor_files,
    }
    return record


def get_latlon_index(coords, lim1, lim2):
    """Get index for given 1D vector, e.g. lats or lons between 2 limits."""
    index = (np.where(
        np.absolute(coords - (lim2 + lim1) / 2.0) <= (lim2 - lim1) / 2.0))[0]
    return index


def cube_to_save_ploted(var, lats, lons, names):
    """Create cube to prepare plotted data for saving to netCDF."""
    new_cube = iris.cube.Cube(var, var_name=names['var_name'],
                              long_name=names['long_name'],
                              units=names['units'])
    new_cube.add_dim_coord(iris.coords.DimCoord(lats,
                                                var_name='lat',
                                                long_name='latitude',
                                                units='degrees_north'), 0)
    new_cube.add_dim_coord(iris.coords.DimCoord(lons,
                                                var_name='lon',
                                                long_name='longitude',
                                                units='degrees_east'), 1)

    return new_cube


def cube_to_save_scatter(var1, var2, names):
    """Create cubes to prepare scatter plot data for saving to netCDF."""
    cubes = iris.cube.CubeList([iris.cube.Cube(var1,
                                               var_name=names['var_name1'],
                                               long_name=names['long_name1'],
                                               units=names['units1'])])
    cubes.append(iris.cube.Cube(var2, var_name=names['var_name2'],
                                long_name=names['long_name2'],
                                units=names['units2']))

    return cubes


def plot_ts_and_trend(cfg, cube_anom, dataset):
    """Plot time series and calculate trends."""
    reg_var = stats.linregress(np.linspace(0, len(cube_anom.data) - 1,
                                           len(cube_anom.data)),
                               cube_anom.data)
    y_reg = reg_var.slope * np.linspace(0, len(cube_anom.data) - 1,
                                        len(cube_anom.data)) + \
        reg_var.intercept

    fig, axx = plt.subplots(figsize=(8, 6))
    axx.plot(np.linspace(0, len(cube_anom.data) - 1, len(cube_anom.data)),
             y_reg, color='k',
             label='trend = ' + str(reg_var.slope) + ' %/dec')
    axx.plot(np.linspace(0, len(cube_anom.data) - 1, len(cube_anom.data)),
             cube_anom.data,
             color='b',
             marker='x')

    axx.legend(loc=0, framealpha=1)
    fig.tight_layout()

    fig.savefig(get_plot_filename('ts_' + dataset, cfg), dpi=300)

    plt.close()

    return reg_var.slope


def plot_reg_li(cfg, data_ar, future_exp):
    """Plot scatter plot and regression."""
    # data_ar {"datasets": datasets, "ar_diff_rain": ar_diff_rain,
    #          "ar_diff_ua": ar_diff_ua, "ar_diff_va": ar_diff_va,
    #          "ar_hist_rain": ar_hist_rain, "mism_diff_rain": mism_diff_rain,
    #          "mwp_hist_rain": mwp_hist_rain}
    reg = stats.linregress(data_ar["mwp_hist_rain"], data_ar["mism_diff_rain"])
    y_reg = reg.slope * np.linspace(5.5, 8.5, 2) + reg.intercept

    fig, axx = plt.subplots(figsize=(7, 7))

    axx.plot(np.linspace(5.5, 8.8, 2), y_reg, color='k')

    for iii, model in enumerate(data_ar["datasets"]):
        proj = (select_metadata(cfg['input_data'].values(),
                                dataset=model))[0]['project']
        style = plot.get_dataset_style(model, style_file=proj.lower())
        axx.plot(
            data_ar["mwp_hist_rain"][iii],
            data_ar["mism_diff_rain"][iii],
            marker=style['mark'],
            color=style['color'],
            markerfacecolor=style['facecolor'],
            linestyle='none',
            markersize=10,
            markeredgewidth=2.0,
            label=model)

    axx.set_xlim([5.5, 8.8])
    axx.set_ylim([-0.01, 0.55])
    axx.text(8.1, 0.01, 'r = {:.2f}'.format(reg.rvalue))
    axx.set_xticks(np.linspace(6, 8, 3))
    axx.set_yticks(np.linspace(0.0, 0.5, 6))
    axx.vlines(6, 0, 0.5, colors='r', linestyle='solid')
    axx.set_xlabel('Western Pacific precip.')
    axx.set_ylabel('ISM rainfall change')
    axx.legend(ncol=2, loc=0, framealpha=1)

    fig.tight_layout()
    fig.savefig(get_plot_filename('li17natcc_fig2a', cfg), dpi=300)
    plt.close()

    caption = ' Scatter plot of the simulated tropical western Pacific ' + \
        'precipitation (mm d−1 ) versus projected average ISM ' + \
        '(Indian Summer Monsoon) rainfall changes under the ' + future_exp + \
        ' scenario. The red line denotes the observed present-day ' + \
        'western Pacific precipitation and the inter-model ' + \
        'correlation (r) is shown.'

    provenance_record = get_provenance_record(_get_sel_files_var(cfg,
                                                                 ['pr', 'ts']),
                                              caption, ['corr'], ['reg'],
                                              plot_type='scatter')

    diagnostic_file = get_diagnostic_filename('li17natcc_fig2a', cfg)

    logger.info("Saving analysis results to %s", diagnostic_file)

    iris.save(cube_to_save_scatter(data_ar["mwp_hist_rain"],
                                   data_ar["mism_diff_rain"],
                                   {'var_name1': 'm_pr',
                                    'long_name1': 'Mean Precipitation',
                                    'units1': 'mm d-1',
                                    'var_name2': 'd_pr',
                                    'long_name2': 'Precipitation Change',
                                    'units2': 'mm d-1'}),

              target=diagnostic_file)

    logger.info("Recording provenance of %s:\n%s", diagnostic_file,
                pformat(provenance_record))
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(diagnostic_file, provenance_record)


def plot_reg_li2(cfg, datasets, mdiff_ism, mdiff_ism_cor, hist_ism):
    """Plot scatter plot and regression."""
    fig, axx = plt.subplots(figsize=(7, 7))

    axx.plot(np.linspace(-2, 21, 2), 0.5 * np.linspace(-2, 21, 2), color='k')

    axx.plot(
        np.mean((mdiff_ism / hist_ism) * 100.0),
        np.mean((mdiff_ism_cor / hist_ism) * 100.0),
        color='k',
        marker='v',
        linestyle='none',
        markersize=12,
        markeredgewidth=3.0,
        label='multi-model mean')

    for iii, model in enumerate(datasets):

        proj = (select_metadata(cfg['input_data'].values(),
                                dataset=model))[0]['project']
        style = plot.get_dataset_style(model, style_file=proj.lower())
        axx.plot(
            mdiff_ism[iii] / hist_ism[iii] * 100.0,
            mdiff_ism_cor[iii] / hist_ism[iii] * 100.0,
            marker=style['mark'],
            color=style['color'],
            markerfacecolor=style['facecolor'],
            linestyle='none',
            markersize=10,
            markeredgewidth=2.0,
            label=model)

    axx.errorbar(
        np.mean((mdiff_ism / hist_ism) * 100.0),
        np.mean((mdiff_ism_cor / hist_ism) * 100.0),
        xerr=np.std((mdiff_ism / hist_ism) * 100.0),
        yerr=np.std((mdiff_ism_cor / hist_ism) * 100.0),
        color='k',
        marker='v',
        linestyle='-',
        markersize=10,
        markeredgewidth=3.0,
        capsize=10)

    axx.set_xlim([-2, 21])
    axx.set_ylim([-2, 21])
    axx.text(
        15,
        7.1,
        'y = {:.1f}x'.format(0.5),
        rotation=np.rad2deg(np.arctan(0.5)),
        horizontalalignment='center',
        verticalalignment='center')
    axx.set_xticks(np.linspace(0, 20, 5))
    axx.set_yticks(np.linspace(0, 20, 5))
    axx.vlines(0, -2, 21, colors='k', linestyle='solid')
    axx.hlines(0, -2, 21, colors='k', linestyle='solid')
    axx.set_xlabel('Uncorrected ISM rainfall change ratio')
    axx.set_ylabel('Corrected ISM rainfall change ratio (% per °C)')
    axx.legend(ncol=2, loc=2, framealpha=1)

    fig.tight_layout()
    fig.savefig(get_plot_filename('li17natcc_fig2b', cfg), dpi=300)
    plt.close()

    caption = ' Scatter plot of the uncorrected versus corrected average ' + \
        'ISM (Indian Summer Monsoon) rainfall change ratios (% per degree ' + \
        'Celsius of global SST warming). The error bars for the ' + \
        'Multi-model mean indicate the standard deviation spread among ' + \
        'models and the 2:1 line (y = 0.5x) is used to illustrate the ' + \
        'Multi-model mean reduction in projected rainfall increase.'

    provenance_record = get_provenance_record(_get_sel_files_var(cfg,
                                                                 ['pr', 'ts']),
                                              caption, ['corr'], ['reg'],
                                              plot_type='scatter')

    diagnostic_file = get_diagnostic_filename('li17natcc_fig2b', cfg)

    logger.info("Saving analysis results to %s", diagnostic_file)

    iris.save(cube_to_save_scatter(np.mean((mdiff_ism / hist_ism) * 100.0),
                                   np.mean((mdiff_ism_cor / hist_ism) * 100.0),
                                   {'var_name1': 'rd_pr',
                                    'long_name1': 'Relative Precipitation ' +
                                                  'Change',
                                    'units1': 'percent K-1',
                                    'var_name2': 'corr_pr',
                                    'long_name2': 'Precipitation Correction',
                                    'units2': 'percent K-1'}),

              target=diagnostic_file)

    logger.info("Recording provenance of %s:\n%s", diagnostic_file,
                pformat(provenance_record))
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(diagnostic_file, provenance_record)


###############################################################################
# Setup diagnostic
###############################################################################


def get_reg_2d_li(mism_diff_rain, ar_hist_rain, lats, lons):
    """Linear regression of 1D and 2D array, returns 2D array of p and r."""
    reg2d = np.zeros((len(lats), len(lons), 4))
    for iii in range(len(lats)):
        for jjj in range(len(lons)):
            reg = stats.linregress(mism_diff_rain, ar_hist_rain[iii, jjj, :])
            reg2d[iii, jjj, 0] = reg.rvalue
            reg2d[iii, jjj, 1] = reg.pvalue
            reg2d[iii, jjj, 2] = reg.slope
            reg2d[iii, jjj, 3] = reg.intercept

    return reg2d


def main(cfg):
    """Run the diagnostic."""
    ###########################################################################
    # Read recipe data
    ###########################################################################

    # Dataset data containers
    input_data = (cfg['input_data'].values())

    if not variables_available(cfg, ['prw']):
        logger.warning("This diagnostic was written and tested only for prw.")

    ###########################################################################
    # Read data
    ###########################################################################

    # Create iris cube for each dataset and save annual means
    trends = {}
    trends['cmip5'] = OrderedDict()
    trends['cmip6'] = OrderedDict()
    trends['cmip5weights'] = OrderedDict()
    trends['cmip6weights'] = OrderedDict()
    trends['obs'] = {}
    f_c5 = open(cfg['work_dir'] + "cmip5_trends.txt", "a")
    f_c5.write("Model Alias Trend Weight\n")
    f_c6 = open(cfg['work_dir'] + "cmip6_trends.txt", "a")
    f_c6.write("Model Alias Trend Weight\n")

    if 'add_model_dist' in cfg:
        extratrends = {}
        for extramodel in cfg['add_model_dist']:
            # wstr = extramodel + 'weights'
            extratrends[extramodel] = OrderedDict()
            # extratrends[wstr] = OrderedDict()

    number_of_subdata = OrderedDict()
    available_dataset = list(group_metadata(input_data, 'dataset'))
    for dataset in available_dataset:
        meta = select_metadata(input_data, dataset=dataset)
        number_of_subdata[dataset] = float(len(meta))
        for dataset_path in meta:
            cube = iris.load(dataset_path['filename'])[0]
            cat.add_year(cube, 'time', name='year')
            if not _check_full_data(dataset_path, cube):
                number_of_subdata[dataset] = number_of_subdata[dataset] - 1

    for dataset_path in input_data:
        project = dataset_path['project']
        cube_load = iris.load(dataset_path['filename'])[0]
        cube = _apply_filter(cfg, cube_load)
        cat.add_month_number(cube, 'time', name='month_number')
        cat.add_year(cube, 'time', name='year')
        alias = dataset_path['alias']
        dataset = dataset_path['dataset']
        # selection = select_metadata(input_data, dataset=dataset)
        # number_of_subdata = float(len(select_metadata(input_data,
        #                                               dataset=dataset)))
        if not _check_full_data(dataset_path, cube):
            continue
        cube_anom = _calculate_anomalies(cube)

        if project == 'CMIP6':
            trend = plot_ts_and_trend(cfg, cube_anom, alias)
            trends['cmip6'][alias] = trend
            trends['cmip6weights'][alias] = 1. / number_of_subdata[dataset]
            print(dataset, alias, trend, 1. / number_of_subdata[dataset])
            f_c6.write(dataset + ' ' +
                       alias + ' ' +
                       str(round(trend, 4)) + ' ' +
                       str(round(1. / number_of_subdata[dataset], 4)) + '\n')
        elif project == 'CMIP5':
            trend = plot_ts_and_trend(cfg, cube_anom, alias)
            trends['cmip5'][alias] = trend
            trends['cmip5weights'][alias] = 1. / number_of_subdata[dataset]
            f_c5.write(dataset + ' ' +
                       alias + ' ' +
                       str(round(trend, 4)) + ' ' +
                       str(round(1. / number_of_subdata[dataset], 4)) + '\n')
        else:
            trend = plot_ts_and_trend(cfg, cube_anom, dataset)
            trends['obs'][dataset] = trend

        if 'add_model_dist' in cfg:
            if dataset in cfg['add_model_dist']:
                # wstr = dataset + 'weights'
                extratrends[dataset][alias] = trend
                # extratrends[wstr][alias] = 1./float(len(selection))

    f_c5.close()
    f_c6.close()
    _plot_trends(cfg, trends)
    if 'add_model_dist' in cfg:
        if extratrends:
            _plot_extratrends(cfg, extratrends, trends)

    ###########################################################################
    # Process data
    ###########################################################################


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
