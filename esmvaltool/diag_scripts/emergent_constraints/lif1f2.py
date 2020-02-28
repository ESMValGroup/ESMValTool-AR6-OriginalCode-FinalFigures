#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calculates emergent constraint on Indian summer monsoon.

###############################################################################
testkw/lif1.py
Author: Katja Weigel (IUP, Uni Bremen, Germany)
EVal4CMIP project
###############################################################################

Description
-----------
    Calculates emergent constraint on Indian summer monsoon
    following Li et al. (2017).

Configuration options
---------------------
    output_name     : Name of the output files.

###############################################################################

"""

import logging
import os
from pprint import pformat
import iris
import iris.coord_categorisation as cat
import numpy as np
from scipy import stats
import cartopy.crs as cart
import matplotlib.pyplot as plt
import esmvaltool.diag_scripts.shared as e
import esmvaltool.diag_scripts.shared.names as n
from esmvaltool.diag_scripts.shared._base import (
    ProvenanceLogger, get_diagnostic_filename, get_plot_filename,
    select_metadata)

logger = logging.getLogger(os.path.basename(__file__))
TEMP_NAME = 'ts'


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


def plot_rain_and_wind(cfg, dataname, data, lats, lons):
    """Plot contour map."""
    if not cfg[n.WRITE_PLOTS]:
        return

    plotdata = {}
    if data[0].ndim == 3:
        plotdata['pr'] = np.mean(data[0], axis=2)
        plotdata['ua'] = np.mean(data[1], axis=2)
        plotdata['va'] = np.mean(data[2], axis=2)
    else:
        plotdata['pr'] = data[0]
        plotdata['ua'] = data[1]
        plotdata['va'] = data[2]

    # Plot data
    # create figure and axes instances
    fig, axx = plt.subplots(figsize=(8, 5))
    axx = plt.axes(projection=cart.PlateCarree())
    axx.set_extent([45, 120, -15, 30], cart.PlateCarree())

    # draw filled contours
    cnplot = plt.contourf(
        lons,
        lats,
        plotdata['pr'],
        np.linspace(-0.75, 0.75, 11),
        transform=cart.PlateCarree(),
        cmap='BrBG',
        # cmap='RdBu_r',
        extend='both')

    if data[0].ndim == 3:
        plt.contour(
            lons,
            lats,
            np.std(data[0], axis=2), [0.2, 0.4, 0.6],
            transform=cart.PlateCarree(),
            colors='w')
    axx.coastlines()
    axx.quiver(
        lons[::2],
        lats[::2],
        plotdata['ua'][::2, ::2],
        plotdata['va'][::2, ::2],
        scale=5.0,
        color='indigo',
        transform=cart.PlateCarree())

    # add colorbar
    cbar = fig.colorbar(cnplot, ax=axx, shrink=0.8)
    # add colorbar title string
    cbar.set_label(r'Rainfall change, mm d$^{-1}$')

    axx.set_xlabel('Longitude')
    axx.set_ylabel('Latitude')
    axx.set_title(dataname + ' changes')
    axx.set_xticks([40, 65, 90, 115])
    axx.set_xticklabels(['40°E', '65°E', '90°E', '115°E'])
    axx.set_yticks([-10, 0, 10, 20, 30])
    axx.set_yticklabels(['10°S', '0°', '10°N', '20°N', '30°N'])

    fig.tight_layout()
    fig.savefig(get_plot_filename(dataname + '_rain_wind_change', cfg),
                dpi=300)
    plt.close()

    caption = dataname + ': Changes in precipitation (colour shade, ' + \
        r'mm d$^{-1}$)' + \
        r'and 850-hPa wind (m s$^{-1}$ scaled with 0.5) ' + \
        'during the Indian summer monsoon season (May to September)' + \
        'from 1980–2009 to 2070–2099 projected ' + \
        'under the RCP 8.5 scenario. All climatology changes are' + \
        'normalized by the corresponding global mean SST increase ' + \
        'for each model.'

    if data[0].ndim == 3:
        caption = caption + ' The white contours display the inter-model ' + \
            'standard deviations of precipitation changes.'

    provenance_record = get_provenance_record(_get_sel_files(cfg, dataname,
                                                             dim=data[0].ndim),
                                              caption, ['diff'], ['reg'])

    diagnostic_file = get_diagnostic_filename(dataname + '_rain_wind_change',
                                              cfg)

    logger.info("Saving analysis results to %s", diagnostic_file)

    cubelist = iris.cube.CubeList([cube_to_save_ploted(plotdata['pr'], lats,
                                                       lons,
                                                       {'var_name': 'd_pr',
                                                        'long_name': 'Prec' +
                                                                     'ipita' +
                                                                     'tion ' +
                                                                     'Change',
                                                        'units': 'mm d-1'})])

    if data[0].ndim == 3:
        cubelist.append(cube_to_save_ploted(np.std(data[0], axis=2), lats,
                                            lons, {'var_name': 'std_pr',
                                                   'long_name': 'Standard ' +
                                                                'Deviation ' +
                                                                'of the Prec' +
                                                                'ipitation ',
                                                   'units': 'mm d-1'}))

    cubelist.append(cube_to_save_ploted(plotdata['ua'][::2, ::2], lats[::2],
                                        lons[::2],
                                        {'var_name': 'd_ua',
                                         'long_name': 'Eastward Wind Change',
                                         'units': 'm s-1'}))

    cubelist.append(cube_to_save_ploted(plotdata['va'][::2, ::2], lats[::2],
                                        lons[::2],
                                        {'var_name': 'd_va',
                                         'long_name': 'Northward Wind Change',
                                         'units': 'm s-1'}))

    iris.save(cubelist, target=diagnostic_file)

    logger.info("Recording provenance of %s:\n%s", diagnostic_file,
                pformat(provenance_record))
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(diagnostic_file, provenance_record)


def plot_rain(cfg, titlestr, data, lats, lons):
    """Plot contour map."""
    if not cfg[n.WRITE_PLOTS]:
        return

    # Plot data
    # create figure and axes instances
    fig, axx = plt.subplots(figsize=(7, 5))
    axx = plt.axes(projection=cart.PlateCarree())
    axx.set_extent([45, 120, -15, 35], cart.PlateCarree())

    # draw filled contours
    cnplot = plt.contourf(
        lons,
        lats,
        data,
        np.linspace(-0.75, 0.75, 11),
        transform=cart.PlateCarree(),
        cmap='BrBG',
        # cmap='RdBu_r',
        extend='both')

    axx.coastlines()
    # ISM (60◦ –95◦ E, 10◦ –30◦ N)
    axx.plot(
        [60, 95, 95, 60, 60], [10, 10, 30, 30, 10],
        transform=cart.PlateCarree(),
        color='k')

    # add colorbar
    cbar = fig.colorbar(cnplot, ax=axx, shrink=0.8)
    # add colorbar title string
    cbar.set_label(r'Rainfall change, mm d$^{-1}$')

    axx.set_xlabel('Longitude')
    axx.set_ylabel('Latitude')
    axx.set_title(titlestr)
    axx.set_xticks([40, 65, 90, 115])
    axx.set_xticklabels(['40°E', '65°E', '90°E', '115°E'])
    axx.set_yticks([-10, 0, 10, 20, 30])
    axx.set_yticklabels(['10°S', '0°', '10°N', '20°N', '30°N'])

    fig.tight_layout()
    if titlestr == 'Multi-model mean rainfall change due to model error':
        figname = 'fig2c'
    else:
        figname = 'fig2d'

    fig.savefig(get_plot_filename(figname, cfg), dpi=300)
    plt.close()


def plot_2dcorrelation_li(cfg, reg2d, lats, lons):
    """Plot contour map."""
    # Set mask for pvalue > 0.005
    mask = reg2d[:, :, 1] > 0.05
    zzz = np.ma.array(reg2d[:, :, 0], mask=mask)

    # Plot data
    # create figure and axes instances
    fig, axx = plt.subplots(figsize=(8, 4))
    axx = plt.axes(projection=cart.PlateCarree(central_longitude=180))
    axx.set_extent(
        [-150, 60, -15, 30], cart.PlateCarree(central_longitude=180))

    # draw filled contours
    cnplot = plt.contourf(
        lons,
        lats,
        zzz, [-0.7, -0.6, -0.5, -0.4, 0.4, 0.5, 0.6, 0.7],
        transform=cart.PlateCarree(),
        cmap='BrBG',
        # cmap='RdBu_r',
        extend='both',
        corner_mask=False)

    axx.coastlines()
    # Western pacific (140◦ E–170◦ W, 12◦ S–12◦ N)
    axx.plot(
        [-40, 10, 10, -40, -40], [-12, -12, 12, 12, -12],
        transform=cart.PlateCarree(central_longitude=180),
        color='k')
    # Indian Ocean (SEIO; 70◦ –100◦ E, 8◦ S–2◦ N)
    axx.plot(
        [70, 100, 100, 70, 70], [-8, -8, 2, 2, -8],
        transform=cart.PlateCarree(),
        color='k')

    # add colorbar
    cbar = fig.colorbar(cnplot, ax=axx, shrink=0.6, orientation='horizontal')
    # add colorbar title string
    cbar.set_label('Correlation')

    axx.set_xlabel('Longitude')
    axx.set_ylabel('Latitude')
    axx.set_title('Inter-model relationship between ISM rainfall changes' +
                  'and mean precip.')
    axx.set_xticks(np.linspace(-150, 60, 8))
    axx.set_xticklabels(
        ['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W'])
    axx.set_yticks([-15, 0, 15, 30])
    axx.set_yticklabels(['15°S', '0°', '15°N', '30°N'])

    fig.tight_layout()
    fig.savefig(get_plot_filename('fig1b_rain_correlation', cfg), dpi=300)
    plt.close()


def plot_reg_li(cfg, data_ar):
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
        style = e.plot.get_dataset_style(model)
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
    axx.text(8.1, 0.01, 'r = {:.2f}'.format(reg.rvalue))
    axx.set_xticks(np.linspace(6, 8, 3))
    axx.set_yticks(np.linspace(0.0, 0.5, 6))
    axx.vlines(6, 0, 0.5, colors='r', linestyle='solid')
    axx.set_xlabel('Western Pacific precip.')
    axx.set_ylabel('ISM rainfall change')
    axx.legend(ncol=2, loc=0, framealpha=1)

    fig.tight_layout()
    fig.savefig(get_plot_filename('fig2a', cfg), dpi=300)
    plt.close()


def plot_reg_li2(cfg, datasets, mdiff_ism, mdiff_ism_cor, hist_ism):
    """Plot scatter plot and regression."""
    y_reg = 0.5 * np.linspace(-2, 21, 2)

    fig, axx = plt.subplots(figsize=(7, 7))

    axx.plot(np.linspace(-2, 21, 2), y_reg, color='k')

    # axx.plot(np.mean(mdiff_ism) / np.mean(hist_ism) * 100.0,
    #         np.mean(mdiff_ism_cor) / np.mean(hist_ism) * 100.0,
    #         color='k', marker='v', linestyle='none',
    #             markersize=12, markeredgewidth=3.0,
    #             label='MME')
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
        style = e.plot.get_dataset_style(model)
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
    fig.savefig(get_plot_filename('fig2b', cfg), dpi=300)
    plt.close()

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


def substract_li(cfg, data, lats, lons):
    """Difference between historical and future fields."""
    pathlist = data.get_path_list(short_name='pr', exp='historical')

    ar_diff_rain = np.zeros((len(lats), len(lons), len(pathlist)))
    mism_diff_rain = np.zeros(len(pathlist))
    mwp_hist_rain = np.zeros(len(pathlist))
    ar_hist_rain = np.zeros((len(lats), len(lons), len(pathlist)))
    ar_diff_ua = np.zeros((len(lats), len(lons), len(pathlist)))
    ar_diff_va = np.zeros((len(lats), len(lons), len(pathlist)))
    datasets = []
    future_exp = 'rcp85'
    for iii, dataset_path in enumerate(pathlist):

        # Substract historical experiment from rcp85 experiment
        datasets.append(data.get_info(n.DATASET, dataset_path))
        ar_diff_rain[:, :, iii] = (data.get_data(short_name='pr',
                                                 exp=future_exp,
                                                 dataset=datasets[iii]) -
                                   data.get_data(short_name='pr',
                                                 exp='historical',
                                                 dataset=datasets[iii])) / \
            (data.get_data(short_name=TEMP_NAME,
                           exp=future_exp, dataset=datasets[iii]) -
             data.get_data(short_name=TEMP_NAME,
                           exp='historical', dataset=datasets[iii]))
        # ISM (60◦ –95◦ E, 10◦ –30◦ N)
        mism_diff_rain[iii] = \
            np.mean((ar_diff_rain[:,
                                  get_latlon_index(lons, 60, 95),
                                  iii])[get_latlon_index(lats, 10, 30), :])
        ar_hist_rain[:, :, iii] = data.get_data(
            short_name='pr', exp='historical', dataset=datasets[iii])
        # Western pacific (140◦ E–170◦ W, 12◦ S–12◦ N)
        mwp_hist_rain[iii] = \
            np.mean((ar_hist_rain[:,
                                  get_latlon_index(lons, 140, 170),
                                  iii])[get_latlon_index(lats, -12, 12), :])

        ar_diff_ua[:, :, iii] = (data.get_data(short_name='ua',
                                               exp=future_exp,
                                               dataset=datasets[iii]) -
                                 data.get_data(short_name='ua',
                                               exp='historical',
                                               dataset=datasets[iii])) / \
            (data.get_data(short_name=TEMP_NAME,
                           exp=future_exp, dataset=datasets[iii]) -
             data.get_data(short_name=TEMP_NAME,
                           exp='historical', dataset=datasets[iii]))

        ar_diff_va[:, :, iii] = (data.get_data(short_name='va',
                                               exp=future_exp,
                                               dataset=datasets[iii]) -
                                 data.get_data(short_name='va',
                                               exp='historical',
                                               dataset=datasets[iii])) / \
            (data.get_data(short_name=TEMP_NAME,
                           exp=future_exp, dataset=datasets[iii]) -
             data.get_data(short_name=TEMP_NAME,
                           exp='historical', dataset=datasets[iii]))

        plot_rain_and_wind(cfg, datasets[iii], [
            ar_diff_rain[:, :, iii], ar_diff_ua[:, :, iii],
            ar_diff_va[:, :, iii]
        ], lats, lons)

    return {
        "datasets": datasets,
        "ar_diff_rain": ar_diff_rain,
        "ar_diff_ua": ar_diff_ua,
        "ar_diff_va": ar_diff_va,
        "ar_hist_rain": ar_hist_rain,
        "mism_diff_rain": mism_diff_rain,
        "mwp_hist_rain": mwp_hist_rain
    }


def correct_li(data, lats, lons, reg):
    """Correction of mean western pacific rain to measured value (6 mm d−1)."""
    # Prec bias for each model
    mwp_hist_cor = data["mwp_hist_rain"] - 6.0

    proj_err = np.zeros((len(lats), len(lons), len(data["datasets"])))
    ar_diff_cor = np.zeros((len(lats), len(lons), len(data["datasets"])))
    mism_hist_rain = np.zeros((len(data["datasets"])))
    mism_diff_cor = np.zeros((len(data["datasets"])))

    for iii in range(0, len(data["datasets"])):

        # Errors of climate projection
        proj_err[:, :, iii] = mwp_hist_cor[iii] * reg[:, :, 2]

        # Correction for prec difference
        ar_diff_cor[:, :, iii] = data["ar_diff_rain"][:, :, iii] - \
            proj_err[:, :, iii]
        mism_hist_rain[iii] = \
            np.mean((data["ar_hist_rain"][:,
                                          get_latlon_index(lons, 60, 95),
                                          iii])[get_latlon_index(lats,
                                                                 10, 30),
                                                :])
        mism_diff_cor[iii] = \
            np.mean((ar_diff_cor[:,
                                 get_latlon_index(lons, 60, 95),
                                 iii])[get_latlon_index(lats, 10, 30), :])

    return {
        "datasets": data["datasets"],
        "ar_diff_cor": ar_diff_cor,
        "proj_err": proj_err,
        "mism_diff_cor": mism_diff_cor,
        "mism_hist_rain": mism_hist_rain,
        "mwp_hist_cor": mwp_hist_cor
    }


def main(cfg):
    """Run the diagnostic."""
    ###########################################################################
    # Read recipe data
    ###########################################################################

    # Dataset data containers
    data = e.Datasets(cfg)
    logging.debug("Found datasets in recipe:\n%s", data)

    # Variables
    var = e.Variables(cfg)
    logging.debug("Found variables in recipe:\n%s", var)

    # Check for tas and rlnst
    if not var.vars_available('pr', 'ua', 'va', TEMP_NAME):
        raise ValueError("This diagnostic needs 'pr', 'ua', " +
                         " 'va', and a temperature variable")

    ###########################################################################
    # Read data
    ###########################################################################

    # Create iris cube for each dataset and save annual means
    for dataset_path in data:
        cube = iris.load(dataset_path)[0]
        cat.add_month_number(cube, 'time', name='month_number')
        # MJJAS mean (monsoon season)
        cube = cube[np.where(
            np.absolute(cube.coord('month_number').points - 7) <= 2)]
        cube = cube.collapsed('time', iris.analysis.MEAN)

        short_name = data.get_info(n.SHORT_NAME, dataset_path)
        if short_name == 'pr':
            # convert from kg m-2 s-1 to mm d-1
            # cube.convert_units('mm d-1') doesn't work.
            cube.data = cube.data * (60.0 * 60.0 * 24.0)
            cube.units = 'mm d-1'
            # Possible because all data must be interpolated to the same grid.
            if 'lats' not in locals():
                lats = cube.coord('latitude').points
                lons = cube.coord('longitude').points

        data.set_data(cube.data, dataset_path)
    ###########################################################################
    # Process data
    ###########################################################################

    data_ar = substract_li(cfg, data, lats, lons)

    # data_ar {"datasets": datasets, "ar_diff_rain": ar_diff_rain,
    #          "ar_diff_ua": ar_diff_ua, "ar_diff_va": ar_diff_va,
    #          "ar_hist_rain": ar_hist_rain, "mism_diff_rain": mism_diff_rain,
    #          "mwp_hist_rain": mwp_hist_rain}
    # plot_rain_and_wind(cfg, 'MME', data_ar[1:4], lats, lons)

    plot_rain_and_wind(cfg, 'Multi-model_mean', [
        data_ar["ar_diff_rain"], data_ar["ar_diff_ua"], data_ar["ar_diff_va"]
    ], lats, lons)

    # Regression between mean ISM rain difference and historical rain
    reg2d = get_reg_2d_li(data_ar["mism_diff_rain"], data_ar["ar_hist_rain"],
                          lats, lons)

    plot_2dcorrelation_li(cfg, reg2d, lats, lons)

    plot_reg_li(cfg, data_ar)

    # Regression between mean WP rain and rain difference for each location
    reg2d_wp = get_reg_2d_li(data_ar["mwp_hist_rain"], data_ar["ar_diff_rain"],
                             lats, lons)

    data_ar2 = correct_li(data_ar, lats, lons, reg2d_wp)
    # return {"datasets": data["datasets"], "ar_diff_cor": ar_diff_cor,
    #         "proj_err": proj_err, "mism_diff_cor": mism_diff_cor,
    #         "mism_hist_rain": mism_hist_rain, "mwp_hist_cor": mwp_hist_cor}

    plot_reg_li2(cfg, data_ar["datasets"], data_ar["mism_diff_rain"],
                 data_ar2["mism_diff_cor"], data_ar2["mism_hist_rain"])

    plot_rain(cfg, 'Multi-model mean rainfall change due to model error',
              np.mean(data_ar2["proj_err"], axis=2), lats, lons)
    plot_rain(cfg, 'Corrected multi-model mean rainfall change',
              np.mean(data_ar2["ar_diff_cor"], axis=2), lats, lons)


if __name__ == '__main__':
    with e.run_diagnostic() as config:
        main(config)
