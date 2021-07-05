#!/pf/b/b380860/miniconda3/envs/evt21/bin/python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic script to plot:
- multi-ensembles timeseries figures
- trend histograms
of IPCC AR6 chapter 10.

Author
------
Martin Jury (BSC, Spain)

Project
-------
ipcc_ar6




Description
-----------
Derive multimodel multi-ensembles timeseries.

Inputs from recipe
------------------
Annual (subseasonal, submonthly) data from preprocessor

Configuration options in recipe
-------------------------------
timeseries:
    # anomalies
    anomalies: bool
        if data is to be treated as anomalies
    relative: bool
        if anomalies to be done relative
    period_norm:
        start_year: 1995
        end_year: 2014
            the period used for the anomalies

    # provenance information to maintain
    domain: str

    # multimodel arguments
    span: str
            overlap or full; if overlap stas are computed on common time-span;
            if full stats are computed on full time spans.

    # timeseries treatment
    window: str or int
            if low pass filtering is to be performed on the plotted timeseries
            implemented string options are 5weighted and 13weighted (see
            https://archive.ipcc.ch/publications_and_data/ar4/wg1/en/ch3sappendix-3-a.html)
            or simple unweighted int

    order: list
            holding information on the plotting order of the single timeseries

    # input data
    timeseries_xyz:
        ensembles: list
            # list of diagnostic variable group, i.e. preprocessed ensembles,
            # e.g. [tas_cmip6]
        labeling: python formatted string, list of python formatted string
            # e.g. '{project} {exp} {N}'
            #     ['{activity} {exp}', '{project} {exp}']
            # if the initial labeling faild due to a KeyError or TypeError
            # subsequentl list entries are tried
            # formatting strings can be metadata describing preprocessed data
            # but have to be unifrom throughout the ensemble (except for
            # dataset) or can me {metric}. Also the number of ensemble members
            # can be added ({N}).
        metrics: list of metrics to be calculated
            # e.g. [mean, mean_pm_std],
            # possible metrics are:
                - mean      np.ma.mean
                - median    np.ma.median
                - std       np.ma.std
                - min       np.ma.min
                - max       np.ma.max
                - envelop   [np.ma.min, np.ma.max]
                - trendline_[metric] metric=[mean, median, std, min, max]
                # special
                - single        returns the single cubes datasets and updates
                                dataset info
                - mean_pm_std   [np.ma.mean + np.ma.std,
                                 np.ma.mean - np.ma.std]
                - minmaxtrend   [the models that shows the min&max trend]
                                user may set 'period_trend' in timeseries_xyz
                                dict
                - minmaxtrend-lines  like minmaxtrend but showing the trendline
        period:
            start_year: 1960
            end_year: 2100
                # period of the timeseries to plot
        indicate_bars: bool (False)
        # Example (Multiple timeseries can be provided):
        #     timeseries_obs:
        #         period:
        #         start_year: 1960
        #         end_year: 2019
        #         ensembles: [tas_obs]
        #         labeling: '{dataset}'
        #         metrics: [single]
        #     timeseries_mod_1:
        #         ensembles: [tas_cmip5, tas_cmip6]
        #         labeling: '{project} {exp} {metric}'
        #         metrics: [mean, mean_pm_std]
        #         period:
        #         start_year: 1960
        #         end_year: 2100
    # plotting options
    plotting_information: (each optional)
        # legend options [todo: treat as keyword arguments]
        add_legend: (bool) true
        ncol: int
        loc: int

        # ax keywords [todo: treat as keyword arguments]
        period_plot: dict
            start_year: 1960
            end_year: 2100
            # period to be shown in plot, if xlims is given, xlims is used
        xlims: tuple
            # period to be shown in plot
        ylims: tuple
        xlabel: str
        ylabel: str
        zero_line: bool
        title: str
        grid_lines: bool

        # splitting the x-axis
        # This works only correctly if the ylims and xlims/period plot are set.
        # It splits the x-axis at the indicated year (split_year) and
        # introduces a secondary (right spline) y axis.
        split_x_axis: true
        split_year: int
            # the x-axis will be split in the following way
            xlims1 = [pi['xlims'][0], pi['split_year'] - 0.5]
            xlims2 = [pi['split_year'] + 0.5, pi['xlims'][1]]
        split_ylims: [-1., 8.]
            # the ylims for the seconded axis instance

        # Special plotting objects
        indicate_period:
                # plots a lying I |-| over the given period and maybe names it and
                # maybe adds a shading. Possible to add multiple (non overlapping)
                # indication periods
            start_year: int
            end_year: int
            name: str
            add_shading: bool


Description
-----------
Derive trend histogram and trend symbols.

Inputs from recipe
------------------
Annual data from preprocessor

Configuration options in recipe
-------------------------------
histogram:
    # provenance information to maintain
    domain: str
        # e.g. 'Mediterranean'

    # trend calculation
    trend_base: str
        # possible options are [decade, year, all]
    period_trend:
        start_year: 1960
        end_year: 2014
        # the period used for the trend calculation

    # anomalies
    relative: bool
        # if trends should be calculated on relative values
    period_norm:
        start_year: 1995
        end_year: 2014
        # the period used as basis for the relative values

    # input data
    add_mean_trends: bool
        # if mean trends should be added as extra line of symbols
    # input data can have arbitrary names, but must include the word
    # 'histogram'
    histogram_histogram:
        metric: trend
            # str so far only option
        ensembles: list
            # list of diagnostic variable group, i.e. preprocessed ensembles,
            # e.g. [tas_cmip6]
        labeling: python formatted string, list of python formatted string
            # e.g. '{project} {exp} {N}'
            #     ['{activity} {exp}', '{project} {exp}']
            # if the initial labeling faild due to a KeyError or TypeError
            # subsequentl list entries are tried
            # formatting strings can be metadata describing preprocessed data
            # but have to be unifrom throughout the ensemble (except for
            # dataset) or can me {metric}. Also the number of ensemble members
            # can be added ({N}).
        plot_type: str
            # must be either histogram, symbols, violin, or boxplot
    histogram_symbols: same as histogram_histogram
        metric: trend
        ensembles: [tas_cmip5, tas_obs]
        labeling: '{project}'
        plot_type: symbols
    # Example
    #     add_mean_trends: True
    #     histogram_histogram:
    #         metric: trend
    #         ensembles: [tas_cmip6]
    #         labeling: '{project}'
    #     histogram_symbols:
    #         metric: trend
    #         ensembles: [tas_cmip5, tas_obs]
    #         labeling: '{project}'

    # plotting options
    plotting_information: (each optional)
        # legend options [todo: treat as keyword arguments]
        add_legend: true
        ncol: int
        loc: int

        # ax keywords [todo: treat as keyword arguments]
        xlims: tuple
        ylims: tuple
        title: str
        xlabel: str
            # default set to 'Trend ({units} {trend_base}$^{{-1}}$)'
        ylabel: str
            # default set to '(% of simulations)'

        # histogram information, major tick marks will be set to the bins
        bins_spacing: 0.05


Description
-----------
Derive RWL as a function of GWL.

Inputs from recipe
------------------
Annual (subseasonal, submonthly) data from preprocessor

Configuration options in recipe
-------------------------------
gwlrwl:
    interp_grid:
        start: flt
        stop: flt
        intervall: flt
        used to intep RWL(GWL) to RWL(interp_grid)
        e.g.:
        interp_grid:
            start: -1.
            stop: 10.
            intervall: 0.05

    # anomalies
    anomalies: bool
        if data is to be treated as anomalies
    relative: bool
        if anomalies to be done relative
    period_norm:
        start_year: 1850
        end_year: 1900
            the period used for the anomalies

    # provenance information to maintain
    domain: str

    # multimodel arguments
    multimodel_threshhold: flt
        threshhold used for the mulitmodel statistics (default: 0.8)

    metrics: list of metrics to be calculated
        # e.g. [mean, mean_pm_std],
        # possible metrics are:
            - mean      np.ma.mean
            - median    np.ma.median
            - std       np.ma.std
            - min       np.ma.min
            - max       np.ma.max
            - envelop   [np.ma.min, np.ma.max]
            # special
            - mean_pm_std   [np.ma.mean + np.ma.std,
                                np.ma.mean - np.ma.std]
    labeling: python formatted string, list of python formatted string
        # e.g. '{project} {exp} {N}'
        #     ['{activity} {exp}', '{project} {exp}']
        # if the initial labeling faild due to a KeyError or TypeError
        # subsequentl list entries are tried
        # formatting strings can be metadata describing preprocessed data
        # but have to be unifrom throughout the ensemble (except for
        # dataset) or can me {metric}. Also the number of ensemble members
        # can be added ({N}).

    # plotting options
    plotting_information: (each optional)
        # legend options [todo: treat as keyword arguments]
        add_legend: (bool) true
        ncol: int
        loc: int

        # ax keywords [todo: treat as keyword arguments]
        xlims: tuple
        ylims: tuple
        xlabel: str, if none cube coord will be used
        ylabel: str, if none cube units
        45degree_line: bool
        zero_line_x: bool
        zero_line_y: bool
        title: str
        grid_lines: bool




Description
-----------
Derive several kinds of mapplots

Inputs from recipe
------------------
- time_mean data from preprocessor (for metric types: mean, bias)
- Annual data from preprocessor (for metric types: trends: [trend, trend-bias,
                                 trend-min-median-max, trend-min-median-max],
                                 but also for mean if relative treatment is
                                 wanted)


Configuration options in recipe
-------------------------------
mapplot:
    # provenance information to maintain
    domain: str
        # e.g. 'Mediterranean'

    # Basic information
    type: str
        # type of input data, can be [single, MultiModelMean,
        #                             single-MultiModelMean]
        # single-MultiModelMean may apply to trend-min-mean-max metric
        # both kind of MultiModelMean need output from the preprocessor
        # multi_model_statistics
    metric: str
        # type of metric, can be [mean, trend, bias, trend-bias,
        #                         trend-min-median-max, trend-min-mean-max,
        #                         trend-difference-minN-maxN]
        # for trend-min-median-max, trend-min-mean-max,
        # trend-difference-minN-maxN the regional definition
        # of region_metric is used to calculate the trends (if no
        # region_metric is the definition of region is used).

    trend_base: str
        # possible options are [decade, year, all]

    masking: bool (default False)
        # if input data has been regridded an additional masking to the
        # reference_dataset cube mask can be performed
        # i.e. newmask = oldmask | refmask
    reference_dataset: str (optional)
        # dataset name to be used as reference e.g. BerkeleyEarth
    exclude_obs: bool
        # Only applies if type != MultiModelMean.
        # If the obs is to be excluded from the metric calculations.
        # The default is False for metrics [mean, trend, bias, trend-bias]
        # and True for metrics [trend-min-median-max, trend-min-mean-max]

    anomalies: bool (optional, default is False)
    relative: bool (optional, default is False)
    period_plot: dict (optional)
        # Needs annual data, only applied to trend metrics and mean if
        # relative selected.
        # e.g.
        # period_plot:
        #   start_year: 1991
        #   end_year: 2014
    period_norm: dict (optional)
        # The period describing the anomalies / relative basis.
        # e.g.
        # period_norm:
        #   start_year: 1960
        #   end_year: 1990

    ensembles: list
        # list of diagnostic variable group, i.e. preprocessed ensembles,
        # e.g. [tas_cmip6], currently only lists of N=1 are implemented
    region: yaml dic (optional)
        # the region to appear on the plot (this may be different from the
        # preprocessor region, in order not to have space wo data in the plot)
        # e.g.
        # region:
        #   start_longitude: -15.
        #   end_longitude: 45.
        #   start_latitude: 20.
        #   end_latitude: 55.
    region_metric: yaml dic (optional)
        # same form as region, but used for the trend calculation of
        # trend-min-median-max and trend-min-mean-max

    centerlononzero: bool (optional)
        # transform data/plot to be centered on 0° longitude
    box: yaml dic (optional)
        # Draws a black rectangular box on the figure
        # e.g. :
        # box:
        #   start_longitude: 9.
        #   end_longitude: 18.
        #   start_latitude: 46.
        #   end_latitude: 50.
        # can be provided mutiple times, e.g. box_A, box_north, ...

    wind_overlay: bool (optional)
        # overlays mapplot with wind arrows
    wind_diagnostics: list of str
        # holding the u and v component of the wind
        # e.g. [mapplot_ua_cmip6_trend_bias, mapplot_va_cmip6_trend_bias]
    normalise_wind: bool
        # Normalise the data for uniform arrow size

    # plotting options (each option)
    plotting_information:
        projection: str (default LambertConformal)
            # can be PlateCarree, Mercator, Miller, LambertConformal
        plot_title: bool (default True)
        title_format: python formatted string
            # e.g. '{project}'
            # formatting strings can be metadata describing preprocessed data
            # but have to be unifrom throughout the ensemble (except for
            # dataset). If metric is trend-min-median-max or trend-min-mean-max
            # a respective string is added to the title
        coastlines: bool (default True)
            # if coastlines should be drawn
        plot_box: bool (default True)
            # if the regions indicated by 'box' in the name should be drawn

        # color management and color bar
        extend: str (default 'both')
            # can be [both, min, max, neither]
            # if the upper/lower colorbar should be a triangle
            # please note that data outside the specified color range appears
            # as nan
        add_colorbar: bool (default True)
            # If the colorbar should be added to the bottom of the plot
            # todo: add more possitioning options, but difficult due to use
            # of seperate axis
        cmap_str: str
            # used to load IPCC cmap, missing colors are set to grey
            # any valid IPCC cmap str, e.g. temp_seq_disc, temp_div, prec_div,
            # for all tas/pr metric combinations sensefull defaults are
            # provided
        Ncolors: int
            # N of colors to use (not for disc cmaps has to be < 21)
        minmax: tuple
            # min max of colorbar
            # If not set this will be roughly the cubes data limits
        truncate_colormap: tuple
            # may truncate the color values, subsequently this will update the
            # minmax tuple once the cmap is truncated
        cbar_label_format: python formatted string (default '{units}')
            # can hold cube units (=units) or trend_base
            # for trends a sensefull user setting would be:
            # '{units} {trend_base}$^{{-1}}$'
        mintozero: bool (default True)
            # if no minmax is given sets the min to 0
        maxtozero: bool (default False)
            # if no minmax is given sets the max to 0


    # # trend calculation
    # trend_base: str
    #     # possible options are [decade, year, all]
    # period_trend:
    #     start_year: 1960
    #     end_year: 2014
    #     # the period used for the trend calculation

    # # normalisation
    # relative: bool
    #     # if trends should be calculated on relative values
    # period_norm:
    #     start_year: 1995
    #     end_year: 2014
    #     # the period used as basis for the relative values

    # # input data
    # add_mean_trends: bool
    #     # if mean trends should be added as extra line of symbols
    # histogram_histogram:
    #     metric: trend
    #         # str so far only option
    #     ensembles: list
    #         # list of diagnostic variable group, i.e. preprocessed ensembles,
    #         # e.g. [tas_cmip6]
    #     labeling:  python formatted string
    #         # e.g. '{project}'
    #         # formatting strings can be metadata describing preprocessed data
    #         # but should have to be unifrom throughout the ensemble.
    # histogram_symbols: same as histogram_histogram
    #     metric: trend
    #     ensembles: [tas_cmip5, tas_obs]
    #     labeling: '{project}'
    # # Example
    # #     add_mean_trends: True
    # #     histogram_histogram:
    # #         metric: trend
    # #         ensembles: [tas_cmip6]
    # #         labeling: '{project}'
    # #     histogram_symbols:
    # #         metric: trend
    # #         ensembles: [tas_cmip5, tas_obs]
    # #         labeling: '{project}'

    # # plotting options
    # plotting_information: (each optional)
    #     # legend options [todo: treat as keyword arguments]
    #     add_legend: true
    #     ncol: int
    #     loc: int

    #     # ax keywords [todo: treat as keyword arguments]
    #     xlims: tuple
    #     ylims: tuple
    #     title: str
    #     xlabel: str
    #         # default set to 'Trend ({units} {trend_base}$^{{-1}}$)'
    #     ylabel: str
    #         # default set to '(% of simulations)'

    #     # histogram information, major tick marks will be set to the bins
    #     bins_spacing: 0.05


Description
-----------
Combine two mapplots diagnostics named mapplotcombination

Inputs from diagnostics
------------------------
- processed mapplot diagnostics

Configuration options in recipe
-------------------------------
mapplotcombination:
    Same as mapplot but additional:
    operator: str
        Currently supported are '-', '+'
    root_diagnostics: list of str
        Holding information on [mapplotA, mapplotB]
        mapplotA, mapplotB have to be mapplot diagnostics
        calculations are mapplotA operator mapplotB




matplotlibrc: str
    # path to mplstyle file
savepdf: bool
    # if output plot also as pdf





Todo:
    - add subtitles and subtitles formating



"""
from esmvalcore.preprocessor import area_statistics, extract_region, extract_time, convert_units
import logging
import os
import copy
import math as m
import re
from collections import OrderedDict
from string import ascii_lowercase

from cf_units import Unit, CALENDAR_STANDARD
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import iris
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pylab as pylab
import matplotlib.colors as mcolors
import matplotlib.path as mpath
from matplotlib.patches import Polygon

from esmvalcore.preprocessor import (extract_region, extract_time, extract_season,
                                     climate_statistics, seasonal_statistics,
                                     area_statistics, area_statistics_mask,
                                     annual_statistics, extract_shape,
                                     DEFAULT_ORDER, regrid, linear_trend,
                                     multi_model_statistics
                                     )
from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            io, get_plot_filename,
                                            ProvenanceLogger,
                                            get_diagnostic_filename
                                            )
import esmvaltool.diag_scripts.shared.names as n
from esmvaltool.diag_scripts.shared.plot import get_path_to_mpl_style


root_path = os.path.dirname(os.path.realpath(__file__))
exernal_path = os.path.join(root_path, "CH10_additional_data")
path_to_ctabels = os.path.join(root_path, "colormaps")
matplotlibrc_path = os.path.join(root_path, "ar6_wgi_ch10.mplstyle")

zero_line = {'color': 'k', 'linewidth': 1., 'ls': '--'}

logger = logging.getLogger(os.path.basename(__file__))

import IPython
from traitlets.config import get_config
c = get_config()
c.InteractiveShellEmbed.colors = "Linux"
# IPython.embed(config=c)

def ar6_wg1_ch10_figures(cfg):
    """Create Multi-ensemble histogram trend plots"""
    # get matplotlib style sheet
    plt.style.use(get_path_to_mpl_style(matplotlibrc_path))

    # Get a description of the preprocessed data that we will use as input.
    input_data = cfg['input_data'].values()
    ensembles = group_metadata(input_data, 'variable_group')

    # Do obsmasking if there is any to be done
    diagnostics_obsmask = select_diagnostics(cfg, ['obsmasking'])
    for diag_name, diagnostic in diagnostics_obsmask.items():
        obs_masking(cfg, diag_name, diagnostic, ensembles)

    # Get information on relevant diagnostics
    diagnostics = select_diagnostics(cfg, ['timeseries', 'histogram',
                                           'mapplot', 'gwlrwl', 'boxplot',
                                           'placeholder', 'mapplotcombination',
                                           'dependent', 'external'])

    new_diagnostics = {}
    remove_diagnostics_keys = []

    # Prepare diagnostics data
    for diag_name, diagnostic in diagnostics.items():
        if diag_name.split('_')[0] == 'mapplot':
            logger.info(f"Processing {diag_name}")
            if diagnostic['type'] == 'multimodel':
                diagnostic['type'] = 'MultiModelMean'
            prepare_mapplot_data(diag_name, diagnostic, ensembles)
            if isinstance(diagnostic['cube'], list):
                subdiags = split_map_diagnostics(diag_name, diagnostic)
                for k, v in subdiags.items():
                    new_diagnostics.update({k: v})
                remove_diagnostics_keys.append(diag_name)
            ##################
            ## some strange behavior where CMIP5 gets overwritten with CMIP6
            diagnostics[diag_name] = copy.deepcopy(diagnostic)
            ###################
        elif diag_name.split('_')[0] == 'histogram':
            logger.info(f"Processing {diag_name}")
            diagnostics[diag_name] = prepare_histogram_data(diag_name,
                                                            diagnostic,
                                                            ensembles)

        elif diag_name.split('_')[0] == 'timeseries':
            logger.info(f"Processing {diag_name}")
            prepare_timeseries_data(diag_name, diagnostic, ensembles)

        elif diag_name.split('_')[0] == 'gwlrwl':
            logger.info(f"Processing {diag_name}")
            prepare_gwlrwl_data(diag_name, diagnostic, ensembles)

        elif diag_name.split('_')[0] == 'boxplot':
            logger.info(f"Processing {diag_name}")
            prepare_boxplot_data(diag_name, diagnostic, ensembles)

        # all the external data
        elif diag_name.split('_')[0] == 'external':
            if diag_name == 'external_mapplot_APHRODITE_stationdensity':
                new_diagnostic = prepare_aphrodite_station_density(diag_name,
                                                                   diagnostic)
            elif 'external_mapplot_SMURPHS_' in diag_name:
                new_diagnostic = prepare_SMURPHS_mapplot(diag_name, diagnostic)
            elif diag_name == 'external_timeseries_UrbanBox_Japan':
                new_diagnostic = prepare_timeseries_Urban_Japan(diag_name,
                                                                diagnostic)
            elif diag_name == 'external_pdfs_GvdSchrier_obsdiff':
                new_diagnostic = prepare_GvdSchrier_pdfs(diag_name, diagnostic)
            elif diag_name == 'external_mapplot_obs_stations_Mediterranean':
                new_diagnostic = get_station_info_Mediterranean(diag_name,
                                                                diagnostic)
            elif diag_name in ['external_mapplot_GCM', 'external_mapplot_RCM',
                               'external_mapplot_GCM_orog',
                               'external_mapplot_RCM_orog']:
                new_diagnostic = get_ECoppola_data(diag_name, diagnostic,
                                                   ensembles)
            else:
                logger.error(f'Diagnostic {diag_name} not implemented.')
                raise NotImplementedError

            new_diagnostics.update(new_diagnostic)
            remove_diagnostics_keys.append(diag_name)

        elif diag_name.split('_')[0] == 'placeholder':
            continue
        elif diag_name.split('_')[0] == 'mapplotcombination':
            continue
        elif diag_name.split('_')[0] == 'dependent':
            continue
        else:
            logger.error(f"Diagnostic type for {diag_name} not implemented.")
            raise NotImplementedError


    # add the mapplot subdiagnostics if there are any
    for k, v in new_diagnostics.items():
        diagnostics.update({k: v})
    # remove the diagnostics that have been transformed to subdiagnostics
    for k in remove_diagnostics_keys:
        diagnostics.pop(k)

    ########
    # Do the mapplotcombination diagnostics
    remove_diagnostics_keys = []
    new_diagnostics = {}

    for diag_name, diagnostic in diagnostics.items():
        if diag_name.split('_')[0] == 'mapplotcombination':
            logger.info(f"Processing {diag_name}")
            new_diagnostics.update(prepare_combination_data(diag_name,
                                                            diagnostic,
                                                            diagnostics))
            remove_diagnostics_keys.append(diag_name)
        else:
            continue
    for k, v in new_diagnostics.items():
        diagnostics.update({k: v})
    # remove the diagnostics that have been transformed to subdiagnostics
    for k in remove_diagnostics_keys:
        diagnostics.pop(k)

    ########
    # Do diagnostics depending on other diagnostics
    remove_diagnostics_keys = []
    new_diagnostics = {}

    for diag_name, diagnostic in diagnostics.items():
        if diag_name.split('_')[0] == 'dependent':
            logger.info(f"Processing {diag_name}")
            if 'UrbanBox' in diag_name and 'boxplot' in diag_name:
                new_diagnostics.update(prepare_boxplot_UrbanBox(diag_name,
                                                                diagnostic,
                                                                diagnostics))
                remove_diagnostics_keys.append(diag_name)

    # add the combination subdiagnostics if there are any
    for k, v in new_diagnostics.items():
        diagnostics.update({k: v})
    # remove the diagnostics that have been transformed to subdiagnostics
    for k in remove_diagnostics_keys:
        diagnostics.pop(k)

    ########
    # Do preprocessing changes to the diagnostics
    post_preprocessing_changes = select_diagnostics(cfg, ['postpreprocessing'])

    if post_preprocessing_changes:
        for diag_name, changes in post_preprocessing_changes['postpreprocessing_changes'].items():
            for k, v in changes.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, dict):
                            logger.error('dic depth not defined')
                        else:
                            diagnostics[diag_name][k][k2] = v2
                elif k in diagnostics[diag_name].keys():
                    diagnostics[diag_name][k] = v
                elif k in diagnostics[diag_name]['plotting_information'].keys():
                    diagnostics[diag_name]['plotting_information'][k] = v
                else:
                    logger.warning(f"{k} not in {diag_name}")

    #####################
    # Numerical values needed for chapter text
    ####################
    # SESA indices
    # for cube,label in zip(diagnostics['histogram_SESA']['histogram_symbols']['trends'][1],
    #                       diagnostics['histogram_SESA']['histogram_symbols']['aliases'][1]):
    #       print(f"{label}: {cube.data}")
    # per_trend = diagnostics['histogram_SESA']['period_trend']
    # for obs_ds in ensembles[diagnostics['histogram_SESA']['histogram_symbols']['ensembles'][1]]:
    #     fname = obs_ds['filename']
    #     cube = iris.load_cube(fname)
    #     cube =  extract_time(cube, **per_trend)
    #     cube = climate_statistics(cube)
    #     print(f"{obs_ds['dataset']}: {cube.data}")
    ####################
    # NAM indices
    # for cube,label in zip(diagnostics['histogram_pr']['histogram_symbols']['trends'][1],
    #                       diagnostics['histogram_pr']['histogram_symbols']['aliases'][1]):
    #       print(f"{label}: {cube.data / 31 * 32}")
    ####################
    # Mediterranean indices
    # df = diagnostics['timeseries_tas_future_indices']['boxes_1']['data']
    # print(f"{df['label'][0]} {np.min(list(df['data'][0]))} - {np.max(list(df['data'][0]))}")
    # df = diagnostics['timeseries_tas_future_indices']['boxes_2']['data']
    # print(f"{df['label'][0]} {np.min(list(df['data'][0]))} - {np.max(list(df['data'][0]))}")
    # df = diagnostics['timeseries_tas_future_indices']['boxes_3']['data'].iloc[0]
    # print(f"{df['label']} {np.min(list(df['data']))} - {np.max(list(df['data']))}")
    # df = diagnostics['timeseries_tas_future_indices']['boxes_3']['data'].iloc[1]
    # print(f"{df['label']} {np.min(list(df['data']))} - {np.max(list(df['data']))}")
    # CMIP5 3.4553927363284895 - 8.218222088083879
    # CMIP6 historical-SSP5-8.5 (N=36) 3.58467598194004 - 8.74031794193589
    # CORDEX EUR-44 3.78408135975125 - 6.640432353472263
    # CORDEX EUR-11 3.839502235246378 - 6.575832031251952
    ####################
    # Indian Monsoon out of range models timeseries boxplot
    # ymax = diagnostics['timeseries_pr']['plotting_information']['ylims'][1]
    # df = diagnostics['timeseries_pr']['boxes_1']['data']
    # models = diagnostics['timeseries_pr']['boxes_1'][f"datasets_{diagnostics['timeseries_pr']['boxes_1']['ensembles'][0]}"]
    # print(f"{np.array(models)[np.array(df['data'].loc[0]) > ymax]}")
    # ['GISS-E2-R-CC' 'GISS-E2-R' 'IPSL-CM5B-LR']
    # df = diagnostics['timeseries_pr']['boxes_2']['data']
    # models = diagnostics['timeseries_pr']['boxes_2'][f"datasets_{diagnostics['timeseries_pr']['boxes_2']['ensembles'][0]}"]
    # print(f"{np.array(models)[np.array(df['data'].loc[0]) > ymax]}")
    # ['CanESM5-CanOE' 'CanESM5' 'GISS-E2-1-G']
    ####################
    # HKH trends
    # data_ind = diagnostics['histogram_tas']['histogram_symbols']['ensembles'].index('obsmasking_tas_areamean')
    # trend_cubes = diagnostics['histogram_tas']['histogram_symbols']['trends'][data_ind]
    # trends = np.array([cube.data for cube in trend_cubes])
    # print(f"OBS: {trends.min()} - {trends.max()}")
    # data_ind = diagnostics['histogram_tas']['histogram_symbols']['ensembles'].index('tas_cmip6_areamean')
    # trend_cubes = diagnostics['histogram_tas']['histogram_symbols']['trends'][data_ind]
    # trends = np.array([cube.data for cube in trend_cubes])
    # print(f"CMIP6: {trends.min()} - {np.median(trends)} - {trends.max()}")

    ####################
    # Do the plotting
    for diag_name, diagnostic in diagnostics.items():
        if diag_name.split('_')[0] == 'placeholder':
            continue

        if diag_name.split('_')[0] == 'mapplot':
            if cfg['write_plots']:
                ax_grid = gridspec.GridSpec(20+1, 10)

                fig = plt.figure(
                    figsize=diagnostic['plotting_information']['figsize'])
                ax = fill_mapplot_ax(fig, ax_grid[:20,:],
                                     diag_name, diagnostic, diagnostics)
                if diagnostic['plotting_information']['add_colorbar']:
                    cbar = fill_colorbar_ax(fig, ax_grid[20:,:],
                                            diag_name, diagnostic)
                if diagnostic['plotting_information']['add_legend']:
                    add_marker_legend_by_handles_labels(fig, ax_grid[:,-1:])
                plot_paths = _save_fig(cfg, diag_name)

        elif diag_name.split('_')[0] == 'histogram':
            if cfg['write_plots']:
                fig, ax = plt.subplots(
                    figsize=diagnostic['plotting_information']['figsize'])
                ax = fill_histogram_ax(ax, diag_name, diagnostic)
                plot_paths = _save_fig(cfg, diag_name)

        elif diag_name.split('_')[0] == 'timeseries':
            if cfg['write_plots']:
                fig, ax = plt.subplots(
                    figsize=diagnostic['plotting_information']['figsize'])
                ax = fill_timeseries_ax(ax, diag_name, diagnostic)
                plot_paths = _save_fig(cfg, diag_name)

        elif diag_name.split('_')[0] == 'gwlrwl':
            if cfg['write_plots']:
                fig, ax = plt.subplots(
                    figsize=diagnostic['plotting_information']['figsize'])
                ax = fill_gwlrwl_ax(ax, diag_name, diagnostic)
                plot_paths = _save_fig(cfg, diag_name)

        elif diag_name.split('_')[0] == 'boxplot':
            if cfg['write_plots']:
                fig, ax = plt.subplots(
                    figsize=diagnostic['plotting_information']['figsize'])
                ax = fill_boxplot_ax(ax, diag_name, diagnostic)
                plot_paths = _save_fig(cfg, diag_name)

        elif diag_name.split('_')[0] == 'pdfs':
            if cfg['write_plots']:
                fig, ax = plt.subplots(
                    figsize=diagnostic['plotting_information']['figsize'])
                ax = fill_pdf_ax(ax, diag_name, diagnostic)
                plot_paths = _save_fig(cfg, diag_name)

        else:
            logger.error(f"Diagnostic type for {diag_name} not implemented.")
            raise NotImplementedError

        # Save output
        if cfg['write_netcdf']:
            nc_paths = write_data(cfg, diag_name, diagnostic)

        # Provenance
        if not diag_name.split('_')[0] in ['pdfs']:
            ancestor_files, projects, obsnames = get_ancestors(diag_name,
                                                            diagnostic,
                                                            ensembles)
            simplified_diag = get_diagnostic_for_provenance(diag_name,
                                                            diagnostic)
            provenance_record = get_provenance_record(simplified_diag,
                                                    projects,
                                                    ancestor_files,
                                                    obsnames = obsnames,
                                                    nc_paths = nc_paths,
                                                    plot_paths = plot_paths)

            for plot_path in plot_paths:
                with ProvenanceLogger(cfg) as provenance_logger:
                    provenance_logger.log(plot_path, provenance_record)
        else:
            pass

    ####################
    # Do the plotting of the combination figures
    if select_combi_diagnostics(cfg):
        combinations = select_combi_diagnostics(cfg)
        for combo_name, diagnostic in combinations.items():
            prepare_combination_figure(combo_name, diagnostic)

            plot_paths = plot_combination_figure(combo_name, diagnostic,
                                                 diagnostics, cfg)
            make_combination_provenance(combo_name, diagnostic, diagnostics,
                                        cfg, plot_paths, ensembles)


###############################################################################
# "External" Data
###############################################################################

def ECoppola_multi_model_statistics(cubes):
    data = []
    for cube in cubes:
        data.append(cube.data)
    data = np.array(data)
    data = np.mean(data, axis=0)

    common_attributes = copy.deepcopy(cubes[0].attributes)
    for cube in cubes[1:]:
        for k, v in cube.attributes.items():
            if k in common_attributes.keys():
                if v != common_attributes[k]:
                    common_attributes.pop(k)

    cube = copy.deepcopy(cubes[0])
    cube.data = data
    cube.attributes = common_attributes

    return cube


def get_ECoppola_data(diag_name, diagnostic, ensembles):
    """Load the E.Coppola's data over the Alps or orog data of models."""
    Ns = {'GCM': 4, 'RCM': 6}
    model_type = diag_name.split('_')[2]

    logger.info(f"Checking integrity of diagnostic settings: {diag_name}")
    verify_mapplot_diagnostic_settings(diagnostic)

    if 'orog' in diag_name:
        regrid_scheme = OrderedDict({"method": "cdo_remapcon",
                                     "tmp_dir": "/tmp/EVT_tmp"})
        grid_cube = iris.load(os.path.join(exernal_path,
                                           diagnostic['target_grid']))[0]

        ens_cubes, ens_datasets, ens_aliases = [], [], []
        for dic in ensembles[diagnostic['ensemble']]:
            f = dic['filename']
            cube = iris.load_cube(f)
            rcube = regrid(cube, grid_cube, regrid_scheme)
            if dic['dataset'] == 'EC-EARTH': # data should be m but is dm
                rcube.data = rcube.data / 10.
            ens_cubes.append(rcube)
            ens_datasets.append(dic['dataset'])
            ens_aliases.append(dic['alias'])
        mmm_cube = ECoppola_multi_model_statistics(ens_cubes)
        mmm_cube.attributes.update({'datasets': '; '.join(ens_datasets),
                                    'aliases': '; '.join(ens_aliases)})
        diagnostic.update({'cube': mmm_cube})
        diagnostic.update({'dataset': f'ECoppola {model_type} orog data'})
        diagnostic.update({'N': Ns[model_type]})
        diagnostic.update({'ensembles': ['external']})
        diagnostic['plotting_information'].update({'label': model_type})
    else:
        cubes = iris.load(os.path.join(exernal_path, diagnostic['filename']))
        cube = cubes[[c.var_name
                      for c in cubes].index(model_type.lower())]
        cube.units = Unit("%")

        diagnostic.update({'cube': cube})
        diagnostic.update({'dataset': f'ECoppola {model_type} data'})
        diagnostic.update({'N': Ns[model_type]})
        diagnostic.update({'ensembles': ['external']})
        diagnostic['plotting_information'].update({'label': model_type})

    new_diag_name = '_'.join(diag_name.split('_')[1:])
    return({new_diag_name: diagnostic})


def prepare_GvdSchrier_pdfs(diag_name, diagnostic):
    """Load the GvdSchrier data on obs differences over the Mediterranean."""
    ds_name_map = {'giss': 'GISTEMP',
                   'ncdc': 'NOAA Global Temp',
                   'berkeley': 'BerkeleyEarth',
                   'crutem4': 'CRUTEM4'}

    diff = np.arange(-80,81,1)*0.001
    diff = ["{:.3f}".format(x) for x in diff]

    dfs = []
    for region in diagnostic['regions']:
        df = pd.DataFrame(diff, columns=['diff'])
        df['region'] = region
        for ds in diagnostic['dss']:
            ds_name = ds_name_map[ds]
            # load
            file_name = diagnostic['filespath'].format(
                **{'ds': ds, 'region': region})
            df_tmp = pd.read_csv(os.path.join(exernal_path,
                                              file_name),
                                 sep="|", dtype={'diff': str})
            # weighted share
            df_tmp = df_tmp.rename(columns={ds: ds_name})
            df_tmp[ds_name+'_weights'] = df_tmp[ds_name] / \
                df_tmp[ds_name].sum() * 100.
            df = df.merge(df_tmp, left_on='diff', right_on='diff', how='outer')
        df = df.fillna(0)
        df['diff'] = df['diff'].astype(float) * diagnostic['factor']
        dfs.append(df)

    diagnostic.update({'data': dfs,
                       'datasets': list(ds_name_map.values())})

    new_diag_name = '_'.join(diag_name.split('_')[1:])
    return({new_diag_name: diagnostic})


def get_station_info_Mediterranean(diag_name, diagnostic):
    """Load E-OBS and Donat et al. 2014 station location information."""
    import pandas as pd

    logger.info(f"Checking integrity of diagnostic settings: {diag_name}")
    verify_mapplot_diagnostic_settings(diagnostic)

    logger.info(f"Retrieving station information over the Mediterranean")
    dfs = {k: pd.read_csv(os.path.join(exernal_path, f), sep="|")
           for k, f in diagnostic['filenames'].items()}

    for df in dfs.values():
        df.columns = [k.strip() for k in df.keys()]

    diagnostic.update({'markers': dfs})

    new_diag_name = '_'.join(diag_name.split('_')[1:])
    return({new_diag_name: diagnostic})


def prepare_boxplot_UrbanBox(diag_name, diagnostic, diagnostics,
                             data_type=['city']):
    """Combine the information from different obs ds over cities."""
    logger.info(f"Checking integrity of diagnostic settings: {diag_name}")
    verify_boxplot_diagnostic_settings(diagnostic)

    source_diagnostics = diagnostic['source_diagnostics']
    source_ensembles = []

    df_base = diagnostics[source_diagnostics[0]]['data_markers']
    cities = list(df_base['Specific country or region'])
    cities_types = list(df_base['type'])
    x, y = list(df_base['x']), list(df_base['y'])
    data_cities = list(df_base['Temperature'])
    df = pd.DataFrame(zip(cities, cities_types, x, y, data_cities),
                      columns=['city', 'type', 'x', 'y', 'data_cities'])

    for source_diagnostic in source_diagnostics:
        ds = diagnostics[source_diagnostic]['dataset']
        df_base = diagnostics[source_diagnostic]['data_markers']
        cities = list(df_base['Specific country or region'])
        tas_obs = list(df_base['tas obs'])
        tas_sum = list(df_base['tas sum'])
        rel_urban = list(df_base['rel urban'] * 100)
        rel_obs = list(df_base['rel obs'] * 100)
        df_tmp = pd.DataFrame(zip(cities, tas_obs, tas_sum, rel_urban,
                                  rel_obs),
                              columns=['city', ds+' data_obs', ds+' data_sum',
                                       ds+' rel_urban', ds+' rel_obs'])
        df = pd.merge(df, df_tmp, on='city')

        source_ensembles.extend(diagnostics[source_diagnostic]['ensembles'])

    df = df[df['type'].isin(data_type)]

    # sort after longitude
    df = df.sort_values('x')
    df = df.reset_index(drop = True)
    diagnostic.update({'data': df,
                       'ensembles': source_ensembles})

    # add some plotting information
    pi = diagnostic['plotting_information']
    pi.update({'xlims': (-0.5, len(df)-0.5)})
    pi_def = {'remove_xticks': True,
              'second_yaxis': True,
              'second_xaxisspline': True,
              'plot_xtickcities': True}
    for kd, default in  pi_def.items():
        if kd not in pi:
            pi.update({kd: default})

    new_diag_name = '_'.join(diag_name.split('_')[1:])
    return({new_diag_name: diagnostic})


def prepare_timeseries_Urban_Japan(diag_name, diagnostic):
    """Get UrbanBox timeseries data."""
    urban_path = os.path.join(exernal_path, "Urban_Box_data")
    ts_file = 'Tokyo_Choshi_annual.csv'
    diagnostic.update({'filename': os.path.join(urban_path, ts_file)})

    # check ts diagnostic setting
    verify_timeseries_diagnostic_settings(diagnostic)
    diagnostic['plotting_information']['period_plot'] = \
        _check_period_information(
            diagnostic['plotting_information']['period_plot'], "plot period")
    # load data
    df = pd.read_csv(os.path.join(urban_path, ts_file))

    # make a dummy cube to be filled with data
    df['year'].values+0.5
    n_years = len(df['year'].values)+1
    start_year = df['year'].values[0]
    n_days = int((n_years)*365.25 + 50)
    time_coord = iris.coords.DimCoord(np.arange(n_days, dtype=float),
                                      var_name='time',
                                      standard_name='time',
                                      long_name='time',
                                      units=Unit(
                                        'days since {}-01-01 00:00:00'.format(
                                            start_year),
                                        calendar=CALENDAR_STANDARD),
                                      )
    dummycube = iris.cube.Cube(np.zeros(n_days, np.int64),
                               standard_name="air_temperature",
                               long_name="Near-Surface Air Temperature",
                               var_name="tas",
                               units=Unit("degC"),
                               dim_coords_and_dims=[(time_coord, 0)])
    dummycube = annual_statistics(dummycube)
    dummycube = extract_time(dummycube, df['year'].values[0], 1, 1,
                             df['year'].values[-1], 12, 31)

    # get the cubes
    cubes = []
    for city in ['Tokyo (Urban)','Choshi (Rural)']:
        cube = dummycube.copy()
        cube.data = df[city].values
        cubes.append(cube)

    # make anomalies
    if diagnostic['anomalies']:
        cubes = _anomalies(cubes, diagnostic)

    # apply possible filters and extract plotting period
    cubes = finalize_timeseries_data(cubes, [],
                                     diagnostic['plotting_information']['period_plot'],
                                     diagnostic['window'])

    # create the timeseries dicts
    for city, cube in zip(['Tokyo (Urban)','Choshi (Rural)'], cubes):
        tsk = f"timeseries {city}"
        ts = {'data': [[[cube]]],
              'labels': [[city]],
              'ensembles': ['external'],
              'metrics': ['mean'],
              'indicate_bars': False}
        diagnostic.update({tsk: ts})

    new_diag_name = '_'.join(diag_name.split('external_')[1:])
    return {new_diag_name: diagnostic}


def prepare_aphrodite_station_density(diag_name, diagnostic):
    """Get APHRODITE station density from file."""
    logger.info(f"Checking integrity of diagnostic settings: {diag_name}")
    verify_mapplot_diagnostic_settings(diagnostic)

    logger.info(f"Adding external dataset {diag_name}")

    cubes = iris.load(diagnostic['filename'])
    cube = [cube for cube in cubes if cube.var_name == 'rstn'][0]
    if not cube.coord('time').has_bounds():
        cube.coord('time').guess_bounds()
    cube = extract_season(cube, diagnostic['season'])
    cube = seasonal_statistics(cube, seasons=[diagnostic['season']])
    cube = cube[0,:,:]
    for coord in ['latitude', 'longitude']:
        cube.coord(coord).guess_bounds()

    diagnostic.update({'cube': cube})
    diagnostic.update({'dataset': 'APHRO_MA_050deg_V1101'})
    diagnostic.update({'ensembles': ['external']})

    new_diag_name = '_'.join(diag_name.split('external_')[1:])
    return {new_diag_name: diagnostic}


def prepare_SMURPHS_mapplot(diag_name, diagnostic):
    """Load the SMURPHS pr data from file."""
    from pathlib import Path

    logger.info(f"Checking integrity of diagnostic settings: {diag_name}")
    verify_mapplot_diagnostic_settings(diagnostic)

    path = os.path.join(exernal_path, diagnostic['filename'])
    cubes = []
    mo_runids, histories = '', ''
    for file_path in sorted(Path(
        os.path.dirname(path)).glob(os.path.basename(path))):
        logger.info(f"loading {str(file_path)}")
        cube = iris.load_cube(str(file_path))

        # some metadata stuff
        for coord in ['latitude', 'longitude']:
            if not cube.coord(coord).has_bounds():
                cube.coord(coord).guess_bounds()
            if cube.coord(coord).var_name != coord[:3]:
                cube.coord(coord).var_name = coord[:3]

        # do "preprocessor" steps
        if 'region' in diagnostic:
            region = diagnostic['region']
            cube = extract_region(cube, region['start_longitude'],
                                  region['end_longitude'],
                                  region['start_latitude'],
                                  region['end_latitude'])
        # centering on zero
        if 'centerlononzero' in diagnostic.keys():
            if diagnostic['centerlononzero']:
                cube = center_lon_on_zero(cube)

        if 'convert_units' in diagnostic.keys():
            cube = convert_units(cube, diagnostic['convert_units']['units'])

        cubes.append(cube)
        mo_runid = cube.attributes['mo_runid']
        history = cube.attributes['history']
        mo_runids = ''.join([mo_runids, f'{mo_runid}; '])
        histories = ''.join([f'{mo_runid}: {history}; '])
            # {histories.append(cube.attributes['history'])}'
    mo_runids = mo_runids[:-2]
    histories = histories[:-2]

    cube = multi_model_statistics(cubes, 'full',
                                  [diagnostic['metric']])[diagnostic['metric']]

    # do "preprocessor" steps (that have to be done after the mmm stat)
    if 'period_plot' in diagnostic:
        period = diagnostic['period_plot']
        period = _check_period_information(period, '')
        cube = extract_time(cube, **period)
    cube = climate_statistics(cube)

    cube.attributes = cubes[0].attributes
    for k in ['mo_runid', 'history']: cube.attributes.pop(k)
    cube.attributes.update({'mo_runids': mo_runids,
                            'histories': histories,
                            'source_path': diagnostic['filename']})

    diagnostic.update({'cube': cube})
    diagnostic.update({'dataset': 'SMURPHS'})
    diagnostic['plotting_information'].update({'label': 'SMURPHS'})
    if diagnostic['plotting_information']['indicate_N']:
        diagnostic.update({'N': len(cubes)})
    diagnostic.update({'ensembles': ['external']})

    new_diag_name = '_'.join(diag_name.split('_')[1:])
    return({new_diag_name: diagnostic})


def prepare_IITM_timeseries(diagnostic, tsk, ts):
    """Loads Andy Turners IITM data.

    source:
    ftp://www.tropmet.res.in/pub/data/rain/iitm-regionrf.txt
    """
    check_timeseries_definitions(tsk, ts)

    # load the data
    df = pd.read_csv(os.path.join(exernal_path, ts['filename']),
                     sep=',', header=3)
    target_column = ['YEAR', 'JJAS']
    df = df[target_column]

    df = df[df['YEAR'].between(ts['period']['start_year'],
                               ts['period']['end_year'],
                               inclusive=True)]
    df = df.assign(JJAS = df.JJAS.mul(ts['factor']))

    df = df.reset_index(drop = True)

    # build the cube
    n_years = len(df)
    start_year = df['YEAR'].values[0]
    n_days = int((n_years)*365.25 + 50)

    time_coord = iris.coords.DimCoord(np.arange(n_days, dtype=float),
                                      var_name='time',
                                      standard_name='time',
                                      long_name='time',
                                      units=Unit(
                                        'days since {}-01-01 00:00:00'.format(
                                            start_year),
                                        calendar=CALENDAR_STANDARD),
                                      )
    cube = iris.cube.Cube(np.zeros(n_days, np.int64),
                          standard_name="precipitation_flux",
                          long_name="Precipitation",
                          var_name="pr",
                          units=Unit("mm day-1"),
                          dim_coords_and_dims=[(time_coord, 0)])
    cube = extract_season(cube, season='JJAS')
    cube = seasonal_statistics(cube, seasons=['JJAS'])
    iris.coord_categorisation.add_year(cube, 'time')

    # make it per day
    data = df['JJAS']
    data = np.divide(data, np.diff(cube.coord('time').bounds).flatten())
    cube.data = data

    # get possilby anomalies and possilby relative
    cube = _anomalies([cube], diagnostic)[0]
    cube = finalize_timeseries_data([cube], '',
                                    ts['period'],
                                    diagnostic['window'])[0]

    ts.update({'data': [[[cube]]],
               'labels': [['IITM']]})

    return {tsk.split('external')[1]: ts}


##############################################################################
# All connected to combination diagnostics
##############################################################################

def prepare_combination_data(diag_name, diagnostic, diagnostics):
    """Simple arithmetics between two diagnostics."""
    import operator
    ops = { "+": operator.add, "-": operator.sub } # etc.

    keys_to_maintain = ['trends_over_n', 'anomalies', 'relative']
    pi_keys_to_maintain = ['label']

    op = ops[diagnostic['operator']]
    cube_1 = diagnostics[diagnostic['root_diagnostics'][0]]['cube']
    cube_2 = diagnostics[diagnostic['root_diagnostics'][1]]['cube']

    cube = cube_1.copy()
    cube.data = op(cube_1,cube_2).data
    if not np.all(cube.data.mask == (cube_1.data.mask | cube_2.data.mask)):
        raise ValueError

    # combine cube attributes
    attr_keys = set(cube_1.attributes.keys()).intersection(
                    set(cube_2.attributes.keys()))
    comb_attr = {'operation': f"nc_A {diagnostic['operator']} nc_B"}
    for k in attr_keys:
        attr_1 = cube_1.attributes[k]
        attr_2 = cube_2.attributes[k]
        if attr_1 == attr_2:
            comb_attr.update({k: attr_1})
        else: comb_attr.update({k: f"nc_A: {attr_1}, nc_B: {attr_2}"})
    cube.attributes = comb_attr
    diagnostic['cube'] = cube

    new_diag_name = diag_name.replace('mapplotcombination', 'mapplot')

    for k in keys_to_maintain:
        if k in diagnostics[diagnostic['root_diagnostics'][0]].keys() and \
            k in diagnostics[diagnostic['root_diagnostics'][1]].keys():
            v_1 = diagnostics[diagnostic['root_diagnostics'][0]][k]
            v_2 = diagnostics[diagnostic['root_diagnostics'][1]][k]
            if v_1 == v_2:
                diagnostic.update({k: v_1})
            else:
                logger.warning(f"Difference in key {k} for root_diagnostics"\
                            f"Values are {v_1} and {v_2}. Taking {v_1}")
                diagnostic.update({k: v_1})

    for k in pi_keys_to_maintain:
        v_1 = diagnostics[diagnostic['root_diagnostics'][0]][
            'plotting_information'][k]
        v_2 = diagnostics[diagnostic['root_diagnostics'][1]][
            'plotting_information'][k]
        if v_1 == v_2:
            diagnostic['plotting_information'].update({k: v_1})
        else:
            logger.warning(f"Difference in key {k} for root_diagnostics"\
                           f"Values are {v_1} and {v_2}. Taking {v_1}")
            diagnostic['plotting_information'].update({k: v_1})

    # add some defaults
    logger.info(f"Checking integrity of diagnostic settings: {diag_name}")
    verify_mapplot_diagnostic_settings(diagnostic)

    return({new_diag_name: diagnostic})


###############################################################################
# All connected to diagnostics_combination_figure plotting
###############################################################################
def make_combination_provenance(combo_name, combo_diagnostic, diagnostics,
                                cfg, plot_paths, ensembles):
    """Do the provenance information."""

    provenance_records = {}
    for diag_name, pos in combo_diagnostic['diagnostics_position'].items():
        if not np.any([dtype in diag_name for dtype in
            ['timeseries', 'histogram', 'mapplot', 'gwlrwl', 'boxplot']]):
            continue
        logger.info(f"{diag_name}")

        diagnostic = diagnostics[diag_name]

        if cfg['write_netcdf']:
            nc_paths = write_data(cfg, diag_name, diagnostic)

        ancestor_files, projects, obsnames = get_ancestors(diag_name,
                                                           diagnostic,
                                                           ensembles)
        simplified_diag = get_diagnostic_for_provenance(diag_name,
                                                        diagnostic)
        provenance_record = get_provenance_record(simplified_diag,
                                                  projects,
                                                  ancestor_files,
                                                  obsnames = obsnames,
                                                  nc_paths = nc_paths,
                                                  plot_paths = plot_paths)
        try:
            provenance_record.update({'figure_panel': pos['title_leading']})
        except KeyError:
            pass
        provenance_records.update({diag_name: provenance_record})

    for plot_path in plot_paths:
        with ProvenanceLogger(cfg) as provenance_logger:
            provenance_logger.log(plot_path, provenance_records)


def plot_combination_figure(combo_name, diagnostic, diagnostics, cfg):
    """ Plot the final figure. """
    title_kwag_default = {'size': 'xx-large',
                          'weight': 'normal',
                          'horizontalalignment': 'left',
                          'verticalalignment': 'bottom'}
    subtitle_kwag_default = {'size': 'xx-large',
                             'weight': 'normal',
                             'horizontalalignment': 'left',
                             'verticalalignment': 'bottom'}

    combination_diags = {}
    for diag_name in diagnostic['diagnostics_position'].keys():
        if 'colorbar' in diag_name: continue
        combination_diags.update({diag_name: copy.deepcopy(
                                                diagnostics[diag_name])})

    fig = plt.figure(figsize=diagnostic['figsize'])
    ax_grid_root = gridspec.GridSpec(diagnostic['gridspec'][0],
                                     diagnostic['gridspec'][1])

    for diag_name, pos in diagnostic['diagnostics_position'].items():
        # plot title
        if 'colorbar' not in diag_name:
            pi = combination_diags[diag_name]['plotting_information']
            if diagnostic['add_title']:
                if 'add_title' in pos:
                    if pos['add_title'] == False:
                        if pi['title']:
                            pi['title'] = ''
                        pass
                    else:
                        add_combi_title(pi, pos,
                                        title_kwag_default, ax_grid_root, fig)
                else:
                    add_combi_title(pi, pos,
                                    title_kwag_default, ax_grid_root, fig)

            # plot subtitle
            if diagnostic['add_subtitle']:
                if 'add_subtitle' in pos:
                    if pos['add_subtitle'] == False:
                        if pi['subtitle']:
                            pi['subtitle'] = ''
                        pass
                    else:
                        add_combi_subtitle(pi, pos, subtitle_kwag_default,
                                           ax_grid_root, fig)
                else:
                    add_combi_subtitle(pi, pos, subtitle_kwag_default,
                                       ax_grid_root, fig)

        # Do the diagnostics plots
        # init grid
        ax_grid = get_ax_grid(ax_grid_root, pos['x_pos'], pos['y_pos'])

        if 'colorbar' in diag_name:
            mapplot_diag_name = pos['mapplot_diagnostic']
            cbar, ax = fill_colorbar_ax(fig, ax_grid,
                                        mapplot_diag_name,
                                        combination_diags[mapplot_diag_name])
        elif 'mapplot' in diag_name:
            ax = fill_mapplot_ax(fig, ax_grid,
                                 diag_name, combination_diags[diag_name],
                                 diagnostics)
            if combination_diags[diag_name]['plotting_information']['add_legend']:
                ax_grid = get_ax_grid(ax_grid_root, pos['x_pos'],
                                      [pos['y_pos'][1]-1, pos['y_pos'][1]])
                logger.info("{} {}".format(pos['x_pos'], [pos['y_pos'][1]-1, pos['y_pos'][1]]))
                add_marker_legend_by_handles_labels(fig, ax_grid)
        elif 'histogram' in diag_name:
            ax = fig.add_subplot(ax_grid)
            ax = fill_histogram_ax(ax, diag_name, combination_diags[diag_name])
        elif 'timeseries' in diag_name:
            ax = fig.add_subplot(ax_grid)
            ax = fill_timeseries_ax(ax, diag_name,
                                    combination_diags[diag_name])
        elif 'boxplot' in diag_name:
            ax = fig.add_subplot(ax_grid)
            ax = fill_boxplot_ax(ax, diag_name,
                                 combination_diags[diag_name])
        elif 'gwlrwl' in diag_name:
            ax = fig.add_subplot(ax_grid)
            ax = fill_gwlrwl_ax(ax, diag_name, combination_diags[diag_name])
        elif 'pdfs' in diag_name:
            ax = fig.add_subplot(ax_grid)
            ax = fill_pdf_ax(ax, diag_name, combination_diags[diag_name])
        elif 'placeholder' in diag_name:
            pass

    if 'figure_name' in diagnostic:
        return _save_fig(cfg, diagnostic['figure_name'])
    else:
        return _save_fig(cfg, combo_name)


def add_combi_title(pi, pos, title_kwag_default, ax_grid_root, fig):
    """Add titles of a panel."""
    if pi['title'] or 'title' in pos:
        if 'title' in pos:
            title = pos['title']
        elif pi['title']:
            title = pi['title']
        else:
            return

        if pi['title']:
            pi['title'] = ''
            pi['plot_title'] = False

        title_kwag = title_kwag_default.copy()
        if 'titlekwags' in pos:
            for k, v in pos['titlekwags'].items():
                title_kwag.update({k: v})
        if 'title_leading' in pos:
            title = f"{pos['title_leading']} {title}"
        else:
            title = f"{title}"
        pos.update({'title': title})
        ax_grid = get_ax_grid(ax_grid_root, pos['x_pos_title'],
                                pos['y_pos_title'])
        ax = fig.add_subplot(ax_grid)
        ax.axis('off')
        ax.text(0.5, 0.5, title, **title_kwag)


def add_combi_subtitle(pi, pos, subtitle_kwag_default, ax_grid_root, fig):
    """Add subtitle of a panel."""
    if 'subtitle' in pi or 'subtitle' in pos:
        if 'subtitle' in pos:
            subtitle = pos['subtitle']
        elif 'subtitle' in pi.keys():
            if pi['subtitle']:
                subtitle = pi['subtitle']
            else:
                return
        else:
            return

        if 'subtitle' in pi:
            if pi['subtitle']:
                pi['subtitle'] = ''
                pi['plot_title'] = False

        subtitle_kwag = subtitle_kwag_default.copy()
        if 'subtitlekwags' in pos:
            for k, v in pos['subtitlekwags'].items():
                subtitle_kwag.update({k: v})

        pos.update({'subtitle': subtitle})
        ax_grid = get_ax_grid(ax_grid_root, pos['x_pos_subtitle'],
                                pos['y_pos_subtitle'])
        ax = fig.add_subplot(ax_grid)
        ax.axis('off')
        ax.text(0.5, 0.5, subtitle, **subtitle_kwag)


def get_ax_grid(ax_grid_root, x_pos, y_pos):
    """ Returns ax_grid based on x/y pos.

    x_pos: int/list
    """
    if isinstance(y_pos, list) and isinstance(x_pos, list):
        ax_grid = ax_grid_root[y_pos[0]: y_pos[1], x_pos[0]: x_pos[1]]
    elif isinstance(y_pos, int) and isinstance(x_pos, list):
        ax_grid = ax_grid_root[y_pos, x_pos[0]: x_pos[1]]
    elif isinstance(y_pos, list) and isinstance(x_pos, int):
        ax_grid = ax_grid_root[y_pos[0]: y_pos[1], x_pos]
    elif isinstance(y_pos, int) and isinstance(x_pos, int):
        ax_grid = ax_grid_root[y_pos, x_pos]

    return ax_grid


def add_subplot_letter(ax, text, x=-0.1, y=1.05, size='xx-large',
                       weight='bold', horizontalalignment='left',
                       verticalalignment='bottom', **kwargs):
    ax.text(x, y, text, transform=ax.transAxes,
            size=size, weight=weight, horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment, **kwargs)


###############################################################################
# All connected to diagnostics_combination_figure preperation
###############################################################################
def prepare_combination_figure(combo_name, diagnostic):
    """Prepare combination diagnostic."""
    logger.info(f"Checking integrity of diagnostic settings: {combo_name}")
    verify_combination_figure_settings(diagnostic)

    logger.info(f"Deriving figure grid of: {combo_name}")
    get_combination_figure_grid(combo_name, diagnostic)

    logger.info(f"Building leading title of: {combo_name}")
    get_title_leading(combo_name, diagnostic)


def get_title_leading(combo_name, diagnostic):
    """Get tile leading."""
    if diagnostic['title_leading']:
        title_format = diagnostic['title_leading']

        for pos_name, pos in diagnostic['diagnostics_position'].items():
            if not 'position' in pos: continue
            title_label_dic = {}
            if 'ascii_lowercase' in title_format:
                title_label_dic.update({'ascii_lowercase':
                                        ascii_lowercase[pos['position']-1]})
            title_leading = title_format.format(**title_label_dic)
            pos.update({'title_leading': title_leading})
    else:
        for pos_name, pos in diagnostic['diagnostics_position'].items():
            pos.update({'title_leading': ''})


def verify_combination_figure_settings(diagnostic):
    """ Verify settings."""
    defaults = {'title_leading': '({ascii_lowercase})',
                'figsize': [20, 40],
                'add_title': True,
                'add_subtitle': True}
    valids = {}
    valid_types = {'diagnostics_position': dict,
                   'figsize': list,
                   'gridspec': list,
                   'title_leading': str,
                   'add_title': bool,
                   'add_subtitle': bool}

    verify_diagnostic_settings(diagnostic, defaults, valids, valid_types)


def get_combination_figure_grid(combo_name, diagnostic):
    """ Gets the x_grid, y_grid, y_subtitle, y_title for every
    diagnostics_position entry."""

    for pos_name, pos in diagnostic['diagnostics_position'].items():
        if not 'x_pos' in pos:
            if not 'x_ind' in pos:
                logger.error(f"Neither x_pos nor x_ind given for {combo_name}-"
                             f"{pos_name}")
                raise KeyError
            else:
                x_pos = [diagnostic['x_grid'][ind] for ind in pos['x_ind']]
                pos.update({'x_pos': x_pos})

        if not 'y_pos' in pos:
            if not 'y_ind' in pos:
                logger.error(f"Neither y_pos nor y_ind given for {combo_name}-"
                             f"{pos_name}")
                raise KeyError
            else:
                y_pos = [diagnostic['y_grid'][pos['y_ind']][0],
                         diagnostic['y_grid'][pos['y_ind']][0] +
                         diagnostic['y_grid'][pos['y_ind']][1]]
                pos.update({'y_pos': y_pos})

        if diagnostic['add_title']:
            if not 'y_pos_title' in pos:
                if not 'y_ind' in pos:
                    logger.warning(f"Neither y_pos_title nor y_ind given for "
                                   f"{combo_name}-{pos_name}")
                else:
                    y_pos_title = diagnostic['y_grid'][pos['y_ind']][2]
                    pos.update({'y_pos_title': y_pos_title})
            if not 'x_pos_title' in pos:
                pos.update({'x_pos_title': pos['x_pos'][0]})
        if diagnostic['add_subtitle']:
            if not 'y_pos_subtitle' in pos:
                if not 'y_ind' in pos:
                    logger.warning(f"Neither y_pos_subtitle nor y_ind given "
                                   f"for {combo_name}-{pos_name}")
                else:
                    y_pos_subtitle = diagnostic['y_grid'][pos['y_ind']][3]
                    pos.update({'y_pos_subtitle': y_pos_subtitle})
            if not 'x_pos_subtitle' in pos:
                pos.update({'x_pos_subtitle': pos['x_pos_title']})

    return


###############################################################################
# All connected to obs prerpocessor masking
###############################################################################
def obs_masking(cfg, diag_name, diagnostic, ensembles):
    """ Preprocessor depending masking of obs input data.

    In order to ensure a correct usage of CRU data in updated versions of
    our figures and based on the approaches sighted in D & A literature,
    we plan to deal with this the following:
    1) For annual values, not missing if stn is not 0 for at least 10
    months. For 3-month season, not missing if stn is not 0 for at least 2
    months
    2) For trends on grid point basis: For each 10-yr sub-period (of the
    trend period), we require to have at least 8 years not missing;
    otherwise the trend will be missing for this grid point.
    3) For area average trends: Having at least 70% of the area not
    missing, otherwise it is missing.
    4) For climatologies, there have to be at least 80% data.
    """
    order = DEFAULT_ORDER
    order = ['extract_time', 'extract_region', 'extract_shape',
             'seasonal_statistics', 'climate_statistics', 'annual_statistics',
             'area_statistics', 'area_statistics_mask']

    mask_ths = {'area_statistics': 0.8,
                'climate_statistics': 0.8,
                'annual_statistics': 0.8,
                'seasonal_statistics': {3: 2, # mask greater equal, so 2 or 3, allowing 1 missing
                                        4: 2, # mask greater equal, so 2 or 3 or 4, allowing 1 missing
                                        12: 4}, # 4 # mask greater equal 4, allowing 3 missing
                }

    if 'custom_threshholds' in diagnostic:
        for k, v in diagnostic['custom_threshholds'].items():
            mask_ths[k] = v

    logger.info(f"Running obsmasking preprocessor {diag_name}")

    ensemble_name = '_'.join(diag_name.split('_')[1:])
    variable_group = diagnostic['variable_group']
    prepr_tasks = copy.deepcopy(diagnostic)
    prepr_tasks.pop('variable_group')
    if 'custom_threshholds' in prepr_tasks.keys():
        prepr_tasks.pop('custom_threshholds')

    ens = ensembles[variable_group]
    datasets = diagnostic['datasets']

    if len(ens) >= len(datasets):
        current_ens = []
        for ds in datasets:
            for member in ens:
                if member['dataset'] == ds:
                    current_ens.append(member)
        if len(current_ens) != len(datasets):
            logger.error("There is inconcistency linking desired " \
                "datasets to data beeing processed by the preprocessor")
            raise ValueError
        else:
            prepr_tasks.pop('datasets')
    else:
        logger.error("More datasets wanted than processed by the " \
            "preprocessor")
        raise ValueError


    # check if steps are valid
    for step in prepr_tasks.keys():
        if step not in order:
            logger.error(f"{step} not found in ordering")
            raise NotImplementedError

    # put the tasks in order
    prepr_tasks_n = {}
    for k in order:
        if k in prepr_tasks.keys():
            prepr_tasks_n.update({k: prepr_tasks[k]})
    prepr_tasks = prepr_tasks_n

    # loop over datasets and do the masking and preprocessor steps
    processed_datasets = []
    for ds_name, dataset in zip(datasets, current_ens):
        # load the data
        logger.info(f"Running obsmasking on dataset {ds_name}")
        f = dataset['filename']
        cube = iris.load_cube(f)
        # run the preprocessor functions
        for step, settings in prepr_tasks.items():
            logger.info(f"Running obsmasking preprocessor step {step}")

            # area mean
            if step in ('area_statistics', 'area_statistics_mask'):
                if not 'extract_shape' in prepr_tasks:
                    masking_threshhold = mask_ths['area_statistics']
                    st_names = [coord.standard_name
                                for coord in cube.coords(dim_coords=True)]
                    if 'time' in st_names:
                        time_mask = np.zeros(cube.shape)
                        for t_ind, spatial in enumerate(cube.data):
                            if (spatial.mask.sum() / m.prod(spatial.shape)) > \
                                1 - masking_threshhold:
                                time_mask[t_ind] = 1
                        time_mask = time_mask.astype('bool')
                        cube.data = np.ma.masked_where(
                            time_mask == 1., cube.data)
                    else:
                        logger.error(f"{step} data /wo time not implemented")
                        raise NotImplementedError

            # seasonal_statistics
            elif step == 'seasonal_statistics':
                if 'seasons' in settings:
                    seasons = tuple([sea.upper()
                                     for sea in settings['seasons']])
                else:
                    seasons=('DJF', 'MAM', 'JJA', 'SON')
                if not cube.coords('clim_season'):
                    iris.coord_categorisation.add_season(cube,
                                                        'time',
                                                        name='clim_season',
                                                        seasons=seasons)
                if not cube.coords('season_year'):
                    iris.coord_categorisation.add_season_year(cube,
                                                            'time',
                                                            name='season_year',
                                                            seasons=seasons)
                combined_mask = []
                sea_len = len(list(set(cube.coord('clim_season').points))[0])
                masking_threshhold = mask_ths['seasonal_statistics'][sea_len]
                uniques = sorted(list(set(zip(
                    cube.coord('clim_season').points,
                    cube.coord('season_year').points))))
                for clim_season, season_year in uniques:
                    #extract the season
                    const = iris.Constraint(
                        clim_season=clim_season,
                        season_year=season_year)
                    sea_cube = cube.extract(const)
                    # mask not full seasons
                    if sea_cube.coord('time').shape[0] == 1:
                        sea_mask = np.ones(sea_cube.shape).astype('bool')
                        combined_mask.append(sea_mask)
                    # build mask
                    else:
                        sea_mask = sea_cube.data.mask.sum(axis=0)
                        sea_mask = np.ma.masked_greater_equal(sea_mask,
                                                            masking_threshhold)
                        for i in range(0, sea_cube.shape[0]):
                            if not sea_mask.mask.shape:
                                resultant_mask = np.ones(sea_cube.data.shape[1:]) *\
                                                [sea_mask.mask]
                                combined_mask.append(resultant_mask.astype('bool'))
                            else:
                                combined_mask.append(sea_mask.mask)
                combined_mask = np.array(combined_mask)
                cube.data = np.ma.masked_where(combined_mask==1., cube.data)

            # climate_statistics
            elif step == 'climate_statistics':
                masking_threshhold = mask_ths['climate_statistics']
                clim_mask_sum = cube.data.mask.sum(axis=0)
                clim_mask = np.zeros(cube.shape[1:])
                clim_mask = np.ma.masked_where(
                    (clim_mask_sum / cube.shape[0]) > \
                    (1 - masking_threshhold), clim_mask.data)
                clim_mask = np.array([clim_mask.mask] * cube.shape[0])
                cube.data = np.ma.masked_where(clim_mask == 1., cube.data)

            # annual_statistics
            elif step == 'annual_statistics':
                combined_mask = []
                masking_threshhold = mask_ths['annual_statistics']

                if not cube.coords('year'):
                    iris.coord_categorisation.add_year(cube, 'time')
                uniques = sorted(list(set(
                    cube.coord('year').points)))
                for year in uniques:
                    #extract the season
                    const = iris.Constraint(year=year)
                    year_cube = cube.extract(const)
                    # build mask
                    mask_cube = year_cube.data.mask.sum(axis=0)
                    mask_cube = np.ma.masked_where(
                        (mask_cube / year_cube.shape[0]) > \
                        (1 - masking_threshhold), mask_cube.data)
                    for i in range(0, year_cube.shape[0]):
                        if not mask_cube.mask.shape:
                            resultant_mask = np.ones(year_cube.data.shape[1:]) *\
                                            [mask_cube.mask]
                            combined_mask.append(resultant_mask.astype('bool'))
                        else:
                            combined_mask.append(mask_cube.mask)
                combined_mask = np.array(combined_mask)
                cube.data = np.ma.masked_where(combined_mask == 1., cube.data)

            elif step == 'extract_shape':
                logger.warning(f"{step} masking not implemented yet")
            elif step in ['extract_region', 'extract_time']:
                pass
            else:
                logger.error(f"{step} not implemented")
                raise NotImplementedError

            function = globals()[step]
            cube = function(cube, **settings)

        name = os.path.splitext(os.path.basename(f))[0] + \
            f'_{ensemble_name}_{ds_name}'
        if cfg[n.WRITE_NETCDF]:
            filepath = os.path.join(cfg[n.WORK_DIR], name + '.nc')
            logger.debug("Saving obs masking results to %s", filepath)
            iris.save(cube, target=filepath)

        processed_dataset = copy.deepcopy(dataset)
        processed_dataset['filename'] = filepath

        processed_datasets.append(processed_dataset)

    ensembles.update({diag_name: processed_datasets})


###############################################################################
# All connected to pdf plotting
###############################################################################
def fill_pdf_ax(ax, diag_name, diagnostic):
    """
    Check diagnostic pdf plotting information.

    Loops over singe pdf df data and plots them. Finish up plot.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type gwlrwl diagnostic.
    """
    logger.info("Checking integrity of diagnostic plotting "\
                "information: {}".format(diag_name))
    prepare_pdf_plot(diagnostic)
    pi = diagnostic['plotting_information']

    logger.info("Doing plot for diagnostic: {}".format(diag_name))
    if 'line' in diagnostic['type']:
        df = diagnostic['data'][0]
        for ds in diagnostic['datasets']:
            ax.plot(df['diff'].values, df[ds+'_weights'].values,
                    color=label_colors(ds), label=ds, lw=1)
    else:
        raise NotImplementedError

    finish_pdf_ax(ax, pi)
    return ax


def finish_pdf_ax(ax, pi):
    """Finish up the scatter plot.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    pi: :obj:`dict`
        Plotting information for the plot.
    """
    if pi['xlims']:
        ax.set_xlim(pi['xlims'])
    if pi['ylims']:
        ax.set_ylim(pi['ylims'])

    if pi['xlabel']:
        ax.set_xlabel(pi['xlabel'])
    if pi['ylabel']:
        ax.set_ylabel(format_units(pi['ylabel']))

    if pi['title']:
        ax.set_title(pi['title'])

    if pi['zero_line_y']:
        ax.plot(ax.get_xlim(), [0, 0], zorder=1, **zero_line)
    if pi['zero_line_x']:
        ax.plot([0, 0], ax.get_ylim(), zorder=1, **zero_line)
    if pi['grid_lines']:
        ax.grid(axis='y', color=pi['dgrey'], linewidth=.5, zorder=1)

    if pi['force_ytick_integers']:
        ax.yaxis.get_major_locator().set_params(integer=True)

    if pi['add_legend']:
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        by_label = translate_obs_naming(by_label)
        ax.legend(by_label.values(), by_label.keys(),
                  ncol=pi['ncol'], loc=pi['loc'])


def prepare_pdf_plot(diagnostic):
    """
    Ensure valid plotting_information.

    Adds defaults if nothing is specified. Possibly adds y_label (=cube.units).

    Parameters
    ----------
    diagnostic: :obj:`dict`
        Of type gwlrwl diagnostic.

    Updates
    ----------
    diagnostic['plotting_information'] :obj:`dict`
    """
    defaults = {'add_legend': True,
                'xlabel': None,
                'ylabel': None,
                'ncol': 2,
                'loc': 3,
                'zero_line_x': False,
                'zero_line_y': False,
                'xlims': None,
                'ylims': None,
                'dgrey': [0.4,0.4,0.4],
                'mgrey': [0.65,0.65,0.65],
                'lgrey': [0.9,0.9,0.9],
                'grid_lines': False,
                'title': '',
                'force_ytick_integers': False,
                }

    pi = diagnostic['plotting_information']

    for kd, default in  defaults.items():
        if kd not in pi:
            pi.update({kd: default})

    if pi['ylabel'] == None:
        logger.warning("No ylabel defined, using none")

    if pi['xlabel'] == None:
        logger.warning("No xlabel defined, using none")


###############################################################################
# All connected to gwlrwl plotting
###############################################################################
def fill_gwlrwl_ax(ax, diag_name, diagnostic):
    """
    Check diagnostic gwlrwl plotting information.

    Loops over singe gwlrwl data and plots them. Finish up plot.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type gwlrwl diagnostic.
    """
    logger.info("Checking integrity of diagnostic plotting "\
                "information: {}".format(diag_name))
    prepare_gwlrwl_plot(diagnostic)
    pi = diagnostic['plotting_information']

    logger.info("Doing plot for diagnostic: {}".format(diag_name))

    if len(diagnostic['data']) == 1:
        for labels, metrics, ens_cubes in zip(diagnostic['labels'],
                diagnostic['metrics'], diagnostic['data']):
            for stat_cubes, label in zip(ens_cubes, labels):
                for metric, cubes in zip(metrics, stat_cubes):
                    if metric in ['mean', 'median', 'std', 'min', 'max']:
                        ax.plot(cubes.coord('air_temperature').points,
                                cubes.data, color=label_colors(label),
                                label=label, zorder=10)
                    elif metric in ['mean_pm_std', 'envelop']:
                        ax.fill_between(cubes[0].coord('air_temperature').points,
                                        cubes[0].data, cubes[1].data,
                                        color=label_colors(label),
                                        alpha=pi[metric+'_alpha'],
                                        zorder=5)
    elif len(diagnostic['data']) > 4:
        logger.error("Only up to 4 linestyles implemented")
        raise NotImplementedError
    else:
        linestyles = ['-', '--', ':', ]
        for labels, metrics, ens_cubes, ls in zip(diagnostic['labels'],
                diagnostic['metrics'], diagnostic['data'], linestyles):
            for stat_cubes, label in zip(ens_cubes, labels):
                for metric, cubes in zip(metrics, stat_cubes):
                    if metric in ['mean', 'median', 'std', 'min', 'max']:
                        ax.plot(cubes.coord('air_temperature').points,
                                cubes.data, color=label_colors(label),
                                label=label, zorder=10, ls=ls)
                    elif metric in ['mean_pm_std', 'envelop']:
                        ax.fill_between(cubes[0].coord('air_temperature').points,
                                        cubes[0].data, cubes[1].data,
                                        color=label_colors(label),
                                        alpha=pi[metric+'_alpha'],
                                        zorder=5, ls=ls)

    finish_gwlrwl_ax(ax, pi)
    return ax


def finish_gwlrwl_ax(ax, pi):
    """Finish up the gwlrwl plot.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    pi: :obj:`dict`
        Plotting information for the plot.
    """
    if pi['xlims']:
        ax.set_xlim(pi['xlims'])
    if pi['ylims']:
        ax.set_ylim(pi['ylims'])

    if pi['xlabel']:
        ax.set_xlabel(pi['xlabel'])
    if pi['ylabel']:
        ax.set_ylabel(format_units(pi['ylabel']))

    if pi['title']:
        ax.set_title(pi['title'])

    if pi['zero_line_y']:
        ax.plot(ax.get_xlim(), [0, 0], zorder=1, **zero_line)
    if pi['zero_line_x']:
        ax.plot([0, 0], ax.get_ylim(), zorder=1, **zero_line)
    if pi['45degree_line']:
        min45 = min([ax.get_xlim()[0], ax.get_ylim()[0]])
        max45 = max([ax.get_xlim()[1], ax.get_ylim()[1]])
        ax.plot([min45, max45], [min45, max45], zorder=1, **zero_line)
    if pi['grid_lines']:
        ax.grid(axis='y', color=pi['dgrey'], linewidth=.5, zorder=1)
    if pi['add_legend']:
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        by_label = translate_obs_naming(by_label)
        ax.legend(by_label.values(), by_label.keys(),
                  ncol=pi['ncol'], loc=pi['loc'], numpoints =1)


def prepare_gwlrwl_plot(diagnostic):
    """
    Ensure valid plotting_information.

    Adds defaults if nothing is specified. Possibly adds y_label (=cube.units).

    Parameters
    ----------
    diagnostic: :obj:`dict`
        Of type gwlrwl diagnostic.

    Updates
    ----------
    diagnostic['plotting_information'] :obj:`dict`
    """
    unit_format_def = '({})'
    defaults = {'add_legend': True,
                'xlabel': None,
                'ylabel': None,
                'ncol': 1,
                'loc': 2,
                'zero_line_x': False,
                'zero_line_y': False,
                '45degree_line': True,
                'xlims': None,
                'ylims': None,
                'mean_pm_std_alpha': 0.2,
                'minmax_ls': [':','--'],
                'dgrey': [0.4,0.4,0.4],
                'mgrey': [0.65,0.65,0.65],
                'lgrey': [0.9,0.9,0.9],
                'grid_lines': False,
                'title': ''
                }

    pi = diagnostic['plotting_information']

    for kd, default in  defaults.items():
        if kd not in pi:
            pi.update({kd: default})

    if pi['ylabel'] == None:
        sample_cube = diagnostic['data'][0][0][0]
        unit = str(sample_cube.coords(dim_coords=True)[0].units)
        if unit == 'degC':
            unit = '°C'
        if '-1' in unit:
            unit.replace('-1','$^{-1}$')
        pi['ylabel'] = unit_format_def.format(unit)
        logger.warning("No ylabel defined, using cube coord units: "\
                       "{}".format(pi['ylabel']))

    if pi['xlabel'] == None:
        sample_cube = diagnostic['data'][0][0][0]
        unit = str(sample_cube.units)
        if unit == 'degC':
            unit = '°C'
        if '-1' in unit:
            unit.replace('-1','$^{-1}$')
        pi['xlabel'] = unit_format_def.format(unit)
        logger.warning("No xlabel defined, using cube units: "\
                       "{}".format(pi['xlabel']))


###############################################################################
# All connected to plotting the boxplot data
###############################################################################
def fill_boxplot_ax(ax, diag_name, diagnostic):
    """Fills the boxplot ax.

    - Check diagnostic boxplot plotting information.
    - Prepare data to be plotted and some plot specs.
    - Plot the boxplots.
    - Finish up plot.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type boxplot diagnostic.
    """
    logger.info("Checking integrity of diagnostic plotting "
                "information: {}".format(diag_name))
    prepare_boxplot_plot(diagnostic)

    logger.info("Doing plot for diagnostic: {}".format(diag_name))
    plot_boxplot_ax(ax, diagnostic)

    finish_boxplot_ax(ax, diagnostic)

    return ax


def plot_boxplot_ax(ax, diagnostic):
    """Plot the boxplot plot.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    diagnostic: :obj:`dict`
        Of type histogram diagnostic.
    """
    pi = diagnostic['plotting_information']
    df = diagnostic['data']

    if diagnostic['type'] in ['bp_single-values', 'bp_single-trends']:
        if len(df) == 1:
            box = ax.boxplot(df['data'], widths=[0.5]*len(df),
                             labels=df['label'], zorder=5)
        else:
            if len(df) > 50:
                flierprops = dict(marker='o', markersize=3)
                box = ax.boxplot(df['data'], widths=[0.6]*len(df),
                                 labels=df['label'], zorder=5,
                                 flierprops=flierprops)
            else:
                box = ax.boxplot(df['data'], widths=[0.7]*len(df),
                                 labels=df['label'], zorder=5)
        color_boxes(ax, box, df)
    elif diagnostic['type'] == 'marker':
        for idx in df.index:
            for y_dat in diagnostic['y_data']:
                if y_dat == 'data_cities':
                    label = df.loc[idx, 'type']
                    ax.scatter(idx, df.loc[idx, 'data_cities'],
                               label=label,
                               color=label_colors(label),
                               marker=label_markers(label))
                if y_dat in ['data_obs', 'data_sum', 'rel_obs', 'rel_urban']:
                    keys, labels = [], []
                    for k in df.keys():
                        if y_dat in k:
                            keys.append(k)
                            labels.append(k.split(' ')[0])
                    for key, label in zip(keys, labels):
                        ax.scatter(idx, df.loc[idx, key],
                                   label=label,
                                   edgecolor=label_colors(label),
                                   marker=label_markers(label),
                                   facecolors='none')


def color_boxes(ax, box, df):
    """Color all boxplot patches."""
    for itemkey in ['whiskers','caps']:
        for patch, idx in zip(box[itemkey], np.repeat(df.index, 2)):
            patch.set_color(label_colors(df.loc[idx, 'label']))
            patch.set_zorder(2)
    for patch, idx in zip(box['boxes'], df.index):
        color = label_colors(df.loc[idx, 'label'])
        patch.set_color(color)
        boxX = list(patch.get_xdata())
        boxY = list(patch.get_ydata())
        boxCoords = list(zip(boxX, boxY))
        boxPolygon = Polygon(boxCoords, facecolor=color)
        patch = ax.add_patch(boxPolygon)
        patch.set_zorder(2)
    for patch, idx in zip(box['fliers'], df.index):
        patch.set_markeredgecolor(label_colors(df.loc[idx, 'label']))
        patch.set_zorder(2)
    for itemkey in ['medians']:
        for patch,idx in zip(box[itemkey], df.index):
            patch.set_color(boxplot_colors_bw(df.loc[idx, 'label']))
            patch.set_linewidth(2.0)
            patch.set_zorder(4)


def finish_boxplot_ax(ax, diagnostic):
    """Finish up the boxplot plot.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    pi: :obj:`dict`
        Plotting information for the plot.
    """
    pi = diagnostic['plotting_information']
    df = diagnostic['data']

    if pi['remove_xticks']:
        ax.tick_params(axis='x', which='both', bottom=False)
    if pi['second_xaxisspline']:
        ax.spines['top'].set_visible(True)
    if pi['second_yaxis']:
        ax.spines['right'].set_visible(True)
        ax.tick_params(axis='y', which='both', right=True, labelright=True)

    if pi['xlims']:
        ax.set_xlim(pi['xlims'][0]-0.5,
                    pi['xlims'][1]+0.5)
    if pi['ylims']:
        ax.set_ylim(pi['ylims'])

    if pi['xlabel']:
        ax.set_xlabel(pi['xlabel'])
    if pi['ylabel']:
        ax.set_ylabel(format_units(pi['ylabel']))

    if pi['force_ytick_integers']:
        ax.yaxis.get_major_locator().set_params(integer=True)

    if pi['plot_xticklabels']:
        xtickNames = plt.setp(ax, xticklabels=df['label'].values)
        plt.setp(xtickNames, rotation=90)
    elif pi['plot_xtickcities']:
        xtick_labels = list(df['city'])
        ax.set_xticks(np.arange(0, len(xtick_labels)))
        xtickNames = ax.set_xticklabels(xtick_labels)
        plt.setp(xtickNames, rotation=90)
        ax.tick_params(axis='x', which='major', bottom=True)
    else:
        ax.xaxis.set_ticklabels([])

    if pi['zero_line']:
        ax.plot(ax.get_xlim(), [0, 0], zorder=0, **zero_line)
    if pi['100_line']:
        ax.plot(ax.get_xlim(), [100, 100], zorder=0, **zero_line)
    if pi['grid_lines']:
        ax.yaxis.grid(True, color=pi['dgrey'], linewidth=.5, zorder=0)
    if pi['add_legend']:
        if 'label' in df.keys():
            labelpatches = []
            for label in df['label']:
                labelpatches.append((mlines.Line2D([], [],
                                    color=label_colors(label),
                                    label=label, marker="s", fillstyle='full',
                                    linestyle="-")))
            by_label = OrderedDict(zip(df['label'], labelpatches))
            by_label = translate_obs_naming(by_label)
            ax.legend(by_label.values(), by_label.keys(),
                    facecolor='w', edgecolor='w', framealpha=1.0,
                    frameon=True, ncol=pi['ncol'], loc=pi['loc'])
        else:
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            by_label = translate_obs_naming(by_label)
            ax.legend(by_label.values(), by_label.keys(),
                    ncol=pi['ncol'], loc=pi['loc'])


def prepare_boxplot_plot(diagnostic):
    """Ensure valid plotting_information and add defaults.

    Parameters
    ----------
    diagnostic: :obj:`dict`
        Of type boxplot diagnostic.

    Updates
    ----------
    diagnostic['plotting_information'] :obj:`dict`
    """
    defaults = {'add_legend': True,
                'xlabel': None,
                'ylabel': '',
                'ncol': 2,
                'loc': 2,
                'zero_line': False,
                '100_line': False,
                'grid_lines': False,
                'xlims': None,
                'ylims': None,
                'dgrey': [0.4,0.4,0.4],
                'mgrey': [0.65,0.65,0.65],
                'lgrey': [0.9,0.9,0.9],
                'title': '',
                'plot_xticklabels': False,
                'plot_xtickcities': False,
                'remove_xticks': True,
                'second_yaxis': True,
                'second_xaxisspline': True,
                'force_ytick_integers': False,
                }

    pi = diagnostic['plotting_information']
    for kd, default in  defaults.items():
        if kd not in pi:
            pi.update({kd: default})



###############################################################################
# All connected to preparing the boxplot data
###############################################################################
def prepare_boxplot_data(diag_name, diagnostic, ensembles):
    """Updates diagnostic: :obj:`dict` with prepared data.

    Parameters
    ----------
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type gwlrwl diagnostic.
    ensembles: :obj:`dict` of :obj:`list` of :obj:`dict`
        Holding ensemble sorted cfg['input_data']
    """
    logger.info(f"Checking integrity of diagnostic settings: {diag_name}")
    verify_boxplot_diagnostic_settings(diagnostic)

    logger.info(f"Deriving data for diagnostic: {diag_name}")

    pi = diagnostic['plotting_information']

    columns = ['alias', 'label', 'source', 'data']
    df = pd.DataFrame(columns=columns)

    if diagnostic['type'] == 'bp_single-values':
        for ens in diagnostic['ensembles']:
            logger.info(f"\t adding: {ens}")

            if ens in diagnostic['ordering']:
                for alias in diagnostic['ordering'][ens]:
                    for dic in ensembles[ens]:
                        if dic['alias'] == alias:
                            # get the data
                            f = dic['filename']
                            cube = iris.load_cube(f)

                            # get the label
                            if isinstance(pi['labeling'], str):
                                label = derive_labels_boxplot(dic,
                                                              pi['labeling'])
                            else:
                                for label_format in pi['labeling']:
                                    try:
                                        label = derive_labels_boxplot(
                                            dic, label_format)
                                        break
                                    except KeyError: continue
                                    except TypeError: continue

                            # this is bad! but happens
                            if label == 'CMIP':
                                label = 'CMIP6'
                            elif label in ['OBS', 'obs4mips']:
                                label = 'OBS+REAN'
                            df_tmp = pd.DataFrame([[dic['alias'], label, 'EVT',
                                                    cube.data]],
                                                  columns=columns)
                            df = df.append(df_tmp)

            else:
                for dic in ensembles[ens]:
                    # get the data
                    f = dic['filename']
                    cube = iris.load_cube(f)

                    # get the label
                    if isinstance(pi['labeling'], str):
                        label = derive_labels_boxplot(dic, pi['labeling'])
                    else:
                        for label_format in pi['labeling']:
                            try:
                                label = derive_labels_boxplot(dic, label_format)
                                break
                            except KeyError: continue
                            except TypeError: continue

                    # this is bad! but happens
                    if label == 'CMIP':
                        label = 'CMIP6'
                    elif label in ['OBS', 'obs4mips']:
                        label = 'OBS+REAN'
                    df_tmp = pd.DataFrame([[dic['alias'], label, 'EVT', cube.data]],
                                        columns=columns)
                    df = df.append(df_tmp)

    elif diagnostic['type'] == 'bp_single-trends':
        diagnostic['period_norm'] = _check_period_information(
                diagnostic['period_norm'], "relative treatment")
        for per_trend in diagnostic['periods_boxes'].values():
            per_trend = _check_period_information(per_trend, "box period")
            for ens in diagnostic['ensembles']:
                logger.info(f"\t adding: {ens}")
                data = []
                for dic in ensembles[ens]:
                    # get the data
                    f = dic['filename']
                    cube = iris.load_cube(f)
                    cube, trend_over_n = calculate_trend(cube,
                                diagnostic['relative'],
                                diagnostic['trend_base'],
                                period_trend=per_trend,
                                period_norm=diagnostic['period_norm'])
                    data.append(cube.data)
                # get the label
                if isinstance(pi['labeling'], str):
                    label = derive_labels_boxplot(dic, pi['labeling'])
                else:
                    for label_format in pi['labeling']:
                        try:
                            label = derive_labels_boxplot(dic, label_format)
                            break
                        except KeyError: continue
                        except TypeError: continue
                if 'labeling' in per_trend:
                    if label:
                        label = per_trend['labeling'] + ' ' + label
                    else:
                        label = per_trend['labeling']
                df_tmp = pd.DataFrame([[dic['alias'], label, 'EVT', data]],
                                      columns=columns)
                df = df.append(df_tmp)

    df = df.reset_index(drop = True)
    diagnostic.update({'data' : df})

    if pi['ylabel_format'] == None:
        pi['ylabel_format'] = "({units})"
        logger.warning("No ylabel_format defined, using cube units")
    if pi['ylabel_format'] == "({units})":
        unit = str(cube.units)
        ylab_dic = {'units': unit}
        pi.update({'ylabel': pi['ylabel_format'].format(**ylab_dic)})
    else:
        ylab_dic = {}
        if 'units' in pi['ylabel_format']:
            ylab_dic.update({'units':  str(cube.units)})
        if 'trend_base' in pi['ylabel_format']:
            ylab_dic.update({'trend_base': diagnostic['trend_base']})
        pi.update({'ylabel': pi['ylabel_format'].format(**ylab_dic)})



def derive_labels_boxplot(dic, label_format):
    """Extract label from datasets according to ts['labeling'] formatting.

    Parameters
    ----------
    dic: :obj:`dict`
        holding single cfg['input_data']
    label_format: formatter string

    Returns
    -------
    label: str
    """
    if not label_format:
        return ''
    else:
        keys = re.findall('\{(.*?)\}', label_format)
        labeling = {k: set() for k in keys}
        for k in keys:
            labeling[k].add(dic[k])
        for k,v in labeling.items():
            labeling[k] = list(v)[0]
        return label_format.format(**labeling)


def verify_boxplot_diagnostic_settings(diagnostic):
    """ Verify boxplot diagnostic definitions.

    - add defaults if not present
    - check valids
    - check valid types
    """

    defaults = {'type': 'single-values',
                'ordering': []}
    valids = {}
    valid_types = {}

    verify_diagnostic_settings(diagnostic, defaults, valids, valid_types)

    # check some plotting information settings
    pi_defaults = {'figsize': (10, 8),
                   'ylabel': "({units})",
                  }

    for k, v in pi_defaults.items():
        if k not in diagnostic['plotting_information']:
            diagnostic['plotting_information'][k] = v
            logger.warning(f"No information for {k} given. "
                           f"Using {v}")


###############################################################################
# All connected to preparing the gwlrwl data
###############################################################################
def prepare_gwlrwl_data(diag_name, diagnostic, ensembles):
    """
    Updates diagnostic: :obj:`dict` with prepared data.

    Parameters
    ----------
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type gwlrwl diagnostic.
    ensembles: :obj:`dict` of :obj:`list` of :obj:`dict`
        Holding ensemble sorted cfg['input_data']
    """
    logger.info(f"Checking integrity of diagnostic settings: {diag_name}")
    verify_gwlrwl_diagnostic_settings(diagnostic)

    logger.info(f"Deriving data for diagnostic: {diag_name}")
    ds_unique_keys = {'ensemblestat': ['dataset', 'diagnostic', #'activity',
                                       'end_year', 'exp'],
                      'single': ['alias', 'dataset', 'diagnostic',
                                 'end_year', 'ensemble', 'exp']}

    final_cubes, final_labels = [], []
    for gwl_rwl_ens, metrics in zip(diagnostic['ensembles'],
                                    diagnostic['metrics']):
        ens_cubes, ens_labels = [], []
        for gwl_ens, rwl_ens in gwl_rwl_ens:
            # load data
            gwl_cubes, rwl_cubes, N = [], [], 0
            for dic in ensembles[gwl_ens]:
                f = dic['filename']
                cube = iris.load_cube(f)
                gwl_cubes.append(cube)
                N += 1

                if 'EnsembleMean' in dic['filename']:
                    ds_keys = ds_unique_keys['ensemblestat']
                else:
                    ds_keys = ds_unique_keys['single']

                for rdic in ensembles[rwl_ens]:
                    if np.all([dic[k] == rdic[k] for k in ds_keys]):
                        f = rdic['filename']
                        cube = iris.load_cube(f)
                        rwl_cubes.append(cube)
            if len(gwl_cubes) != len(rwl_cubes):
                logger.error("Ensembles of unqual length detected".format(metric))
                raise ValueError

            # Make the anomalies
            if diagnostic['anomalies']:
                gwl_cubes = _anomalies(gwl_cubes, diagnostic)
                rwl_cubes = _anomalies(rwl_cubes, diagnostic)

            rwl2gwl_cubes = []
            for gcube, rcube in zip(gwl_cubes, rwl_cubes):
                rwl2gwl_cubes.append(interpolate_gwlrwl_cube(gcube, rcube,
                                                             diagnostic))

            # calculate statistics
            stat_cubes = []
            for metric in metrics:
                if metric in ['mean', 'median', 'min', 'max', 'std']:
                    stat_cube = compute_rwl_statistic(rwl2gwl_cubes, metric,
                            threshhold=diagnostic['multimodel_threshhold'])
                    stat_cubes.append(stat_cube)
                elif metric == 'mean_pm_std':
                    mean_cube = compute_rwl_statistic(rwl2gwl_cubes, 'mean',
                            threshhold=diagnostic['multimodel_threshhold'])
                    std_cube = compute_rwl_statistic(rwl2gwl_cubes, 'std',
                            threshhold=diagnostic['multimodel_threshhold'])
                    stat_cube = [mean_cube + std_cube, mean_cube - std_cube]
                    for cube in stat_cube:
                        cube.var_name = rwl2gwl_cubes[0].var_name
                        cube.units = rwl2gwl_cubes[0].units
                        cube.standard_name = rwl2gwl_cubes[0].standard_name
                    stat_cubes.append(stat_cube)
                elif metric == 'envelop':
                    min_cube = compute_rwl_statistic(rwl2gwl_cubes, 'min',
                            threshhold=diagnostic['multimodel_threshhold'])
                    max_cube = compute_rwl_statistic(rwl2gwl_cubes, 'max',
                            threshhold=diagnostic['multimodel_threshhold'])
                    stat_cubes.append([min_cube, max_cube])
                else:
                    logger.error("Metric {} not implemented".format(metric))
                    raise NotImplementedError

            # derive the labels
            if isinstance(diagnostic['labeling'], str):
                label = derive_labels_timeseries(diagnostic, gwl_ens,
                            ensembles[gwl_ens], '', diagnostic['labeling'], N)
            label = format_label(label)

            ens_labels.append(label)
            ens_cubes.append(stat_cubes)
        final_cubes.append(ens_cubes)
        final_labels.append(ens_labels)

    diagnostic['data'] = final_cubes
    diagnostic['labels'] = final_labels


def compute_rwl_statistic(cubes, statistic_name, threshhold=0.8):
    """Compute multimodel statistic."""
    from functools import partial

    data = np.ma.array([cube.data for cube in cubes])

    if statistic_name == 'median':
        statistic_function = np.ma.median
    elif statistic_name == 'mean':
        statistic_function = np.ma.mean
    elif statistic_name == 'std':
        statistic_function = np.ma.std
    elif statistic_name == 'max':
        statistic_function = np.ma.max
    elif statistic_name == 'min':
        statistic_function = np.ma.min
    elif re.match(r"^(p\d{1,2})(\.\d*)?$", statistic_name):
        quantile = float(statistic_name[1:]) / 100
        statistic_function = partial(_quantile, quantile=quantile)
    else:
        raise NotImplementedError

    if len(data.shape) == 2:
        statistic = statistic_function(data, axis=0)

    def _get_overlap(data, threshhold=0.8):
        """Derive the mask for data based on overlap threshhold, i.e. there is
        valid data for at least threshhold x Ndata."""
        threshhold = data.shape[0] * (1 - threshhold)

        mask = np.sum(data.mask, axis=0)
        newmask = np.zeros(mask.shape, dtype=bool)
        newmask[mask > threshhold] = True

        return newmask

    statistic.mask = _get_overlap(data, threshhold=threshhold)

    return_cube = cubes[0].copy()
    return_cube.attributes = None
    return_cube.data = statistic

    return return_cube


def interpolate_gwlrwl_cube(gcube, rcube, diagnostic):
    """Interpolate on common axis.

    Parameters
    ----------
    gcube: iris.cube.Cube
    rcube: iris.cube.Cube

    Returns
    -------
    cube: iris.cube.Cube with interpolated rcube on interp grid as coordinate
    """
    # set up interpolation grid
    num = int((diagnostic['interp_grid']['stop'] - \
               diagnostic['interp_grid']['start']) / \
               diagnostic['interp_grid']['intervall'] + 1)
    interp_grid = np.linspace(diagnostic['interp_grid']['start'],
                              diagnostic['interp_grid']['stop'],
                              num)

    # do the interpolation
    data = np.interp(interp_grid, gcube.data, rcube.data,
                     left=np.nan, right=np.nan)
    data = np.ma.masked_invalid(data)

    # build the new cube
    tas_coord = iris.coords.DimCoord(interp_grid,
                                     var_name='tas',
                                     standard_name='air_temperature',
                                     units=gcube.units
    )
    cube = iris.cube.Cube(data,
                          var_name='tas',
                          standard_name='air_temperature',
                          dim_coords_and_dims=[(tas_coord, 0)],
                          units=gcube.units
    )
    cube.attributes = rcube.attributes
    for coord in rcube.coords(dim_coords=False):
        try:
            cube.add_aux_coord(coord)
        except ValueError:
            cube.add_aux_coord(coord.collapsed())

    return cube


def verify_gwlrwl_diagnostic_settings(diagnostic):
    """ Verify gwlrwl diagnostic definitions.

    - add defaults if not present
    - check valids
    - check valid types
    """
    defaults = {'anomalies': False, # get information on anomalies
                'relative': False, # get information on relative treatment
                'multimodel_threshhold': 0.8,
                'interp_grid': {'start': -1.,
                                'stop': 10.,
                                'intervall': 0.05},
    }
    valids = {}
    valid_types = {'anomalies': bool,
                   'relative': bool,
                   'multimodel_threshhold': float}

    verify_diagnostic_settings(diagnostic, defaults, valids, valid_types)

    if diagnostic['anomalies']:
        if not 'period_norm' in diagnostic:
            logger.warning("Anomalies selected, but no period given. "\
                           "using entire period")
            diagnostic.update({'period_norm': None})
        else:
            diagnostic['period_norm'] = _check_period_information(
                diagnostic['period_norm'], "anomalies")

    # check some plotting information settings
    pi_defaults = {'figsize': (10, 10)}

    for k, v in pi_defaults.items():
        if k not in diagnostic['plotting_information']:
            diagnostic['plotting_information'][k] = v
            logger.warning(f"No information for {k} given. "
                           f"Using {v}")


###############################################################################
# All connected to mapplot plotting
###############################################################################
def add_marker_legend_by_handles_labels(fig, ax_grid):
    """Retrive unique handles and labels and make legend."""
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    for label, handle in by_label.items():
        handle.set_linewidth(0.)
    ax = fig.add_subplot(ax_grid)
    ax.axis('off')
    ax.legend(by_label.values(), by_label.keys(), ncol=2, loc=3,
              bbox_to_anchor=(0.0, -1.0), markerscale=3)


def fill_colorbar_ax(fig, ax_grid, diag_name, diagnostic):
    """Plots the mapplot colorbar ax.

    - Initialise the colorbar axis.
    - Plot the colorbar.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
    ax_grid: matplotlib.gridspec.GridSpec
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type mapplot diagnostic.

    Returns
    -------
    ax: matplotlib.axes
        Matplotlib ax instance.
    """

    cbar_params = {'ytick.labelsize': 'x-large',
                   'ytick.major.size': 0.,
                   'ytick.minor.size': 0.,
                   'xtick.labelsize': 'x-large',
                   'xtick.major.size': 0.,
                   'xtick.minor.size': 0.,
                   'axes.labelsize': 'x-large',
                   'font.family':'sans-serif',
                   'font.sans-serif':'Arial',
                   'font.size':9.0,
                    }

    pi = diagnostic['plotting_information']

    with plt.rc_context(cbar_params):
        ax = fig.add_subplot(ax_grid)
        m = matplotlib.cm.ScalarMappable(norm=pi['norm'], cmap=pi['cmap'])
        m.set_array(diagnostic['cube'].data)

        ticks = pi['ticks']
        if ticks[0] > 0: lower_bound = ticks[0] * 1000 * -1
        elif ticks[0] == 0: lower_bound = -1000
        else: lower_bound = ticks[0] * 1000
        if ticks[-1] > 0: upper_bound = ticks[-1] * 1000
        elif ticks[-1] == 0: upper_bound = 1000
        else: upper_bound = ticks[-1] * 1000 * -1

        if pi['extend'] == 'both':
            boundaries = [lower_bound] + list(ticks) + [upper_bound]
        elif pi['extend'] == 'min':
            boundaries = [lower_bound] + list(ticks) + [pi['norm'].vmax]
        elif pi['extend'] == 'max':
            boundaries = [pi['norm'].vmin] + list(ticks) + [upper_bound]
        elif pi['extend'] == 'neither':
            boundaries = [pi['norm'].vmin] + list(ticks) + [pi['norm'].vmax]

        cbar = fig.colorbar(m, cax=ax, shrink=0.5,  #pad=0.08,
                            orientation = 'horizontal',
                            extend = pi['extend'],
                            boundaries=boundaries,
                            spacing = 'uniform', extendfrac = 'auto',
                            aspect=20, fraction=0.046, pad=0.04,
                            drawedges=True)
        if pi['cbar_label']:
            cbar.ax.set_xlabel(format_units(pi['cbar_label']))
        cbar.ax.tick_params(labelsize='x-large')

    return(cbar, ax)


def fill_mapplot_ax(fig, ax_grid, diag_name, diagnostic, diagnostics):
    """Fills the mapplot ax.

    - Prepare data to be plotted (probably removed).
    - Check diagnostic mapplot plotting information.
    - Plot the mapplot.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
    ax_grid: matplotlib.gridspec.GridSpec
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type histogram diagnostic.

    Returns
    -------
    ax: matplotlib.axes
        Matplotlib ax instance.
    """
    logger.info("Checking integrity of diagnostic plotting "
                "information: {}".format(diag_name))
    prepare_mapplot_plot(diagnostic)

    logger.info("Doing plot for diagnostic: {}".format(diag_name))
    ax = plot_mapplot_ax(fig, ax_grid, diagnostic, diagnostics)

    return ax


def plot_mapplot_ax(fig, ax_grid, diagnostic, diagnostics):
    """Plots the mapplot.

    - Initialise the mapplot axis.
    - Set plot extends etc.
    - Plot the data.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
    ax_grid: matplotlib.gridspec.GridSpec
    diagnostic: :obj:`dict`
        Of type mapplot diagnostic.

    Returns
    -------
    ax: matplotlib.axes
        Matplotlib ax instance.

    Todo
    ----
    Add latlonlabels (or not).
            latlonlabels: bool (default False)
            # if lat lon axes should have label (not implemented yet)
    """
    import iris.plot as iplt
    from cartopy import feature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    dgrey = [0.4, 0.4, 0.4]
    mgrey = [0.65, 0.65, 0.65]
    lgrey = [0.9, 0.9, 0.9]
    box_format = {'lw': 1,
                    'color': 'k'}

    pi = diagnostic['plotting_information']

    ax = fig.add_subplot(ax_grid, projection=pi['projection_crs'])

    if pi['coastlines']:
        ax.coastlines(resolution=pi['res_coastlines'], linestyle='-', linewidths=1.)
    if pi['lakes']:
        lakes = feature.NaturalEarthFeature('physical', 'lakes',
                                            pi['res_lakes'],
                                            linestyle='-', linewidths=1.,
                                            edgecolor='k',
                                            facecolor=pi['missing_grey'])
        ax.add_feature(lakes)

    if pi['plot_box']:
        box_keys = [k for k in diagnostic.keys() if 'box' in k]
        for bk in box_keys:
            lons_box, lats_box = region_to_square(diagnostic[bk])
            if pi['projection'] == 'LambertConformal':
                lons_box,lats_box = interpolate_along_longitude(lons_box,
                                                                lats_box)
            ax.plot(lons_box, lats_box, transform=ccrs.PlateCarree(),
                    **box_format)

    if pi['latlonlabels']: # currently only implemented for 'Mercator', 'Miller'
        if pi['projection'] in ['Mercator', 'Miller']:
            xticks = np.arange(np.floor(pi['plot_extend'][0]),
                               np.ceil(pi['plot_extend'][1]), 2)
            yticks = np.arange(np.floor(pi['plot_extend'][2]),
                               np.ceil(pi['plot_extend'][3]), 2)
            ax.set_xticks(xticks, crs=ccrs.PlateCarree())
            ax.set_yticks(yticks, crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.tick_params(which='minor', length=0)


    if pi['projection'] == 'LambertConformal':
        proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
        rect_in_target = proj_to_data.transform_path(pi['path_extend'])
        ax.set_boundary(rect_in_target, use_as_clip_path=True)
    if 'region' in diagnostic:
        ax.set_extent(pi['plot_extend'], crs=ccrs.PlateCarree())

    if 'cube' in diagnostic.keys():
        # plot the data
        if pi['projection'] == 'LambertConformal':
            iplt.pcolormesh(diagnostic['cube'],
                            cmap=pi['cmap'], norm=pi['norm'])
        else:
            iplt.pcolormesh(diagnostic['cube'],
                            cmap=pi['cmap'], norm=pi['norm'])

    # plot wind overlay data
    if diagnostic['wind_overlay']:
        uwind = diagnostics[diagnostic['wind_diagnostics'][0]]['cube']
        vwind = diagnostics[diagnostic['wind_diagnostics'][1]]['cube']
        if not uwind.coord('longitude') == vwind.coord('longitude') or \
            not uwind.coord('latitude') == vwind.coord('latitude'):
            logger.error("Wind components need to have the same spatial")
            logger.error("coordinates")
            raise ValueError
        x = uwind.coord('longitude').points
        y = uwind.coord('latitude').points
        u = uwind.data
        v = vwind.data

        if 'normalise_wind' in diagnostic:
            if diagnostic['normalise_wind']:
                # Normalise the data for uniform arrow size
                u_norm = u / np.sqrt(u ** 2.0 + v ** 2.0)
                v_norm = v / np.sqrt(u ** 2.0 + v ** 2.0)
                ax.quiver(x, y, u_norm, v_norm, pivot='middle', transform=ccrs.PlateCarree())
            else:
                ax.quiver(x, y, u, v, pivot='middle', transform=ccrs.PlateCarree())
        else:
            ax.quiver(x, y, u, v, pivot='middle', transform=ccrs.PlateCarree())

    if diagnostic['orog_overlay']:
        orog_diagnostic = diagnostics[diagnostic['orog_diagnostic']]
        pi_orog = orog_diagnostic['plotting_information']
        contours = np.linspace(pi_orog['minmax'][0], pi_orog['minmax'][1],
                               pi_orog['Ncolors']+1)
        iplt.contour(orog_diagnostic['cube'], contours,
                     colors=[dgrey] * len(contours),
                     linewidths=1.)

    if diagnostic['pattern_overlay']:
        if pi['overlay_pattern'] == 'stippling':
            stipple_format = {'facecolor': 'k',
                              'edgecolor': 'k',
                              'marker': 'o',
                              's': 1,
                              }
            ol_cube = diagnostic['overlay_cube']
            latlon_inds = np.where(ol_cube.data == 1)
            lats = ol_cube.coord('latitude').points[latlon_inds[0]]
            lons = ol_cube.coord('longitude').points[latlon_inds[1]]
            ax.scatter(lons, lats, label=diagnostic['pattern_metric'],
                       transform=ccrs.PlateCarree(), **stipple_format)

    if 'plot_markers' in diagnostic:
        add_mapplot_markers(ax, diagnostic)

    if 'add_shapes' in diagnostic:
        if diagnostic['add_shapes']:
            add_shapes(ax, diagnostic['shape_paths'])

    if pi['plot_title']:
        ax.set_title(pi['title'])
    if pi['indicate_N']:
        ax.annotate(str(diagnostic['N']), (0.9, 0.9),
                    xycoords='axes fraction',
                    fontsize=pylab.rcParams['legend.fontsize'])

    return ax


def add_shapes(ax, shape_paths):
    """Add shapes to the map."""
    import cartopy.io.shapereader as shpreader

    dgrey = [0.4, 0.4, 0.4]
    mgrey = [0.65, 0.65, 0.65]
    lgrey = [0.9, 0.9, 0.9]

    shapeformat = {'lw': 1,
                   'edgecolor': 'k',
                   'facecolor': 'none'}

    for shape_path in shape_paths:
        country_records = shpreader.Reader(os.path.join(exernal_path,
                                                        shape_path)).records()
        for country_record in country_records:
            try:
                ax.add_geometries(country_record.geometry, ccrs.PlateCarree(),
                                  **shapeformat)
            except TypeError:
                ax.add_geometries([country_record.geometry],
                                  ccrs.PlateCarree(),
                                  **shapeformat)


def add_mapplot_markers(ax, diagnostic):
    """Adds markers to a mapplot."""
    dgrey = [0.4, 0.4, 0.4]
    mgrey = [0.65, 0.65, 0.65]
    lgrey = [0.9, 0.9, 0.9]

    if diagnostic['plot_markers_style'] == 'pies': # the urban figure
        pos_map = {'Kolkata area': {'horizontalalignment':'right', 'verticalalignment':'center', 'latos':0, 'lonos':-5.8},
                   'Ho Chi Minh area': {'horizontalalignment':'center', 'verticalalignment':'top', 'latos':-5.8, 'lonos':0},
                   'Brussels area': {'horizontalalignment':'right', 'verticalalignment':'bottom', 'latos':4.2, 'lonos':-4.2},
                   'Tehran area': {'horizontalalignment':'center', 'verticalalignment':'bottom', 'latos':5.8, 'lonos':0},
                   'Khartoum area': {'horizontalalignment':'center', 'verticalalignment':'top', 'latos':-5.8, 'lonos':7.},
                   'Athens area': {'horizontalalignment':'right', 'verticalalignment':'center', 'latos':0., 'lonos':-5.8},
                   'Moscow area': {'horizontalalignment':'left', 'verticalalignment':'bottom', 'latos':4.2, 'lonos':4.2},
                   'New York': {'horizontalalignment':'left', 'verticalalignment':'top', 'latos':-4.2, 'lonos':4.2},
                   'Los Angeles': {'horizontalalignment':'right', 'verticalalignment':'top', 'latos':-4.2, 'lonos':-4.2},
                   'Mexico City': {'horizontalalignment':'right', 'verticalalignment':'top', 'latos':-4.2, 'lonos':-4.2},
                   'Lagos': {'horizontalalignment':'right', 'verticalalignment':'top', 'latos':-4.2, 'lonos':-4.2},
                   'Johannesburg': {'horizontalalignment':'left', 'verticalalignment':'top', 'latos':-4.2, 'lonos':4.2},
                   'Sao Paulo': {'horizontalalignment':'left', 'verticalalignment':'top', 'latos':-4.2, 'lonos':4.2},
                   'Buenos Aires': {'horizontalalignment':'left', 'verticalalignment':'top', 'latos':-4.2, 'lonos':4.2},
                   'Sydney': {'horizontalalignment':'center', 'verticalalignment':'top', 'latos':-5.8, 'lonos':0},
                   'Cairo': {'horizontalalignment':'right', 'verticalalignment':'top', 'latos':-4.2, 'lonos':-4.2},
                   'Hong Kong': {'horizontalalignment':'left', 'verticalalignment':'top', 'latos':-4.2, 'lonos':4.2},
                   'China': {'horizontalalignment':'center', 'verticalalignment':'bottom', 'latos':6.8, 'lonos':0},
                   'Thailand': {'horizontalalignment':'right', 'verticalalignment':'center', 'latos':-2, 'lonos':-6.8},
                   'Japan': {'horizontalalignment':'center', 'verticalalignment':'top', 'latos':-6.8, 'lonos':0}}
        marker_size = {'city': 650,
                       'country' : 1000}
        text_kwags = {'fontsize': 'large',
                      'weight': 'bold'}

        df = diagnostic['data_markers']

        for idx in df.index:
            x = df.loc[idx,'x']
            y = df.loc[idx,'y']

            city = np.round(df.loc[idx, 'Temperature'], 2)
            surrounding = np.round(df.loc[idx, 'tas obs'], 2)
            total = np.round(df.loc[idx,'tas sum'], 2)
            ratios = [df.loc[idx,'rel obs'],
                      df.loc[idx,'rel urban']]
            size = marker_size[df.loc[idx,'type']]

            draw_pie_marker(ax, ratios, x, y, size=size)

            city_text = df.loc[idx,'Specific country or region']
            text = city_text + f" +{city}°C"

            halign = pos_map[city_text]['horizontalalignment']
            valign = pos_map[city_text]['verticalalignment']
            lonos = pos_map[city_text]['lonos']
            latos = pos_map[city_text]['latos']

            ax.text(x+lonos, y+latos, text,
                    horizontalalignment=halign, verticalalignment=valign,
                    bbox=dict(facecolor=lgrey, edgecolor='none',
                              boxstyle='round'),
                    **text_kwags)

    elif diagnostic['plot_markers_style'] == 'filled': # the urban figure
        pi = diagnostic['plotting_information']
        for k, df in diagnostic['markers'].items():
            ax.plot(df['LON'], df['LAT'], label=k,
                    transform=ccrs.PlateCarree(), **pi['marker_format'][k])

    elif diagnostic['plot_markers_style'] == 'custom_list': # the urban figure
        pi = diagnostic['plotting_information']
        for k, v in diagnostic['markers'].items():
            ax.plot(v['lon'], v['lat'], label=k,
                    transform=ccrs.PlateCarree(), **pi['marker_format'][k])

    else:
        logger.error(f"markers style {diagnostic['plot_markers_style']}" \
                     " not implemented")
        raise NotImplementedError


def draw_pie_marker(ax, ratios, X=0, Y=0, size = 1000):
    """Draws a pie marker.

    From https://www.geophysique.be/2010/11/15/matplotlib-basemap-tutorial-05-adding-some-pie-charts/
    """
    dgrey = [0.4, 0.4, 0.4]
    mgrey = [0.65, 0.65, 0.65]
    lgrey = [0.9, 0.9, 0.9]
    colors = [lgrey, dgrey]

    N = len(ratios)
    xy = []
    start = 0.
    for ratio in ratios:
        xs = [0] + np.cos(np.linspace(2 * m.pi * start,
                                     2 * m.pi * (start + ratio), 30)).tolist()
        ys = [0] + np.sin(np.linspace(2 * m.pi * start,
                                     2 * m.pi * (start + ratio), 30)).tolist()
        xy_path = mpath.Path([[x, y] for x, y in zip(xs, ys)])
        xy.append(xy_path)
        start += ratio

    for i, xyi in enumerate(xy):
        ax.scatter([X], [Y], marker=xyi, s=size, facecolor=colors[i],
                   edgecolor='none', zorder=10)

    xs = [1] + np.cos(np.linspace(2 * m.pi * 0,
                                  2 * m.pi * 1, 30)).tolist()
    ys = [0] + np.sin(np.linspace(2 * m.pi * 0,
                                  2 * m.pi * 1, 30)).tolist()

    xy_path_outline = mpath.Path([[x, y] for x, y in zip(xs, ys)])
    ax.scatter([X], [Y], marker=xy_path_outline, s=size, facecolor='none',
                edgecolor='k', zorder=10)


def center_lon_on_zero(cube):
    """Centers a cube on the zero longitude.

    Parameters
    ----------
    cube: iris.cube.Cube

    Returns
    -------
    cube: iris.cube.Cube
    """
    from esmvaltool.cmorizers.obs.utilities import _roll_cube_data

    logger.info("Centering longitude on zero...")

    if any(cube.coord('longitude').points > 180.):
        lon_coord = cube.coord('longitude').copy()

        lonsabove180 = lon_coord.points[lon_coord.points > 180.] - 360.
        lonsbelow180 = lon_coord.points[lon_coord.points <= 180.]
        lons = np.hstack((lonsabove180, lonsbelow180))

        lonbounds_above180 = lon_coord.bounds[lon_coord.points > 180.] - 360.
        lonbounds_below180 = lon_coord.bounds[lon_coord.points <= 180.]
        lon_bounds = np.vstack((lonbounds_above180, lonbounds_below180))

        cube.coord('longitude').points = lons
        cube.coord('longitude').bounds = lon_bounds
        cube.coord('longitude').circular = False

        loncoordindex = [cc.var_name for cc in cube.coords()].index('lon')
        _roll_cube_data(cube, len(lonsabove180), loncoordindex)

    if np.any(cube.coord('longitude').bounds > 180):
        lon_inds_above180 = [
            bind
            for bind, bounds in enumerate(cube.coord('longitude').bounds)
            if np.any(bounds > 180)
        ]
        # oh boy is this ugly
        if lon_inds_above180:
            if len(cube.shape) == 2 and \
                cube.coords(dimensions=[1])[0].var_name == 'lon':
                if lon_inds_above180[0]+1 == \
                    len(cube.coord('longitude').points):
                    logger.warning('Removing lons above 180deg')
                    cube = cube[:, :-1]
                else:
                    raise NotImplementedError
            elif len(cube.shape) == 3 and \
                cube.coords(dimensions=[2])[0].var_name == 'lon':
                if lon_inds_above180[0]+1 == \
                    len(cube.coord('longitude').points):
                    logger.warning('Removing lons above 180deg')
                    cube = cube[:, :, :-1]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        lon_inds_belowmin180 = [
            bind
            for bind, bounds in enumerate(cube.coord('longitude').bounds)
            if np.any(bounds < -180)
        ]
        if lon_inds_belowmin180:
            raise NotImplementedError

    return(cube)


def truncate_colormap(cmap, minmax, trunc_minmax):
    """Truncates a color map.

    Parameters
    ----------
    cmap: matplotlib.colors.LinearSegmentedColormap
        As specified in the cmap_str.
    minmax: tuple
        Of old minmax values.
    trunc_minmax: tuple
        Of minmax values ot truncate minmax values to.

    Returns
    -------
    new_cmap: matplotlib.colors.LinearSegmentedColormap
        Truncated color map.
    """
    N = cmap.N
    span = minmax[1]-minmax[0]

    if minmax[0] > trunc_minmax[0]:
        logger.error("Cannot truncate to new cmap with lower minimum than " \
                     "origin")
    elif minmax[0] == trunc_minmax[0]:
        cmin = 0.
    else:
        cmin = (trunc_minmax[0]-minmax[0])/span

    if minmax[1] < trunc_minmax[1]:
        logger.error("Cannot truncate to new cmap with higher maximum than " \
                     "origin")
    elif minmax[1] == trunc_minmax[1]:
        cmax = 1.
    else:
        cmax = 1+(trunc_minmax[1]-minmax[1])/span

    spacing = (minmax[1] - minmax[0]) / N
    newN = int((trunc_minmax[1] - trunc_minmax[0]) / spacing)

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=cmin, b=cmax),
        cmap(np.linspace(cmin, cmax, newN)), N=newN)

    return new_cmap


def prepare_mapplot_plot(diagnostic):
    """Updates diagnostic: (:obj:`dict`) with mapplot plotting information.

    - Ensure valid plotting_information.
    - Adds defaults if nothing is specified
    - Possibly adds trend_base to plotting_information
    - Retrieve the settings for the color map
    - Add cartopy projection related stuff

    Parameters
    ----------
    diagnostic: :obj:`dict`
        Of type mapplot diagnostic.

    Updates
    -------
    diagnostic['plotting_information'] :obj:`dict`
    """
    defaults = {'projection': 'LambertConformal', # can be PlateCarree, Mercator, Miller, LambertConformal
                'projection_crs': None,
                'plot_title': True,
                'coastlines': True,
                'lakes': False,
                'res_coastlines': '50m',
                'res_lakes': '50m',
                'plot_box': True,
                # color management and color bar
                'latlonlabels': False,
                'add_colorbar': True,
                'add_legend': False,
                'cbar_label_format': '{units}',
                'cbar_label': None,
                'Ncolors': 12,
                'mintozero': True,
                'maxtozero': False,
                'minmax': None,
                'truncate_colormap': None,
                'indicate_N': False,
                'cmap_str': None,
                'cmap': None,
                'norm': None,
                'ticks': None,
                'extend': 'both',
                'title': None,
                'overlay_pattern': 'stippling',
                'dgrey': [0.4, 0.4, 0.4],
                'mgrey': [0.65, 0.65, 0.65],
                'missing_grey': [0.75, 0.75, 0.75],
                'lgrey': [0.9, 0.9, 0.9],
                }

    pi = diagnostic['plotting_information']

    for kd, default in defaults.items():
        if kd not in pi:
            pi.update({kd: default})

    if 'trend_base' in pi['cbar_label_format']:
        pi.update({'trend_base': diagnostic['trend_base']})
        pi.update({'trends_over_n': diagnostic['trends_over_n']})

    # color and color bar information
    # check if this has been done before
    if not pi['norm']:
        if 'cube' in diagnostic.keys():
            get_cmap_setting(pi, diagnostic['cube'], diagnostic['metric'])

    # init the correct projection
    add_cartopy_crs(diagnostic)

    if pi['plot_title'] and pi['title'] == None:
        if 'label' in pi.keys():
            pi['title'] = pi['label']


def get_cmap_setting(pi, cube, metric):
    """Retrieve the settings for the color map of mapplots diagnostics.

    Parameters
    ----------
    pi(diagnostic['plotting_information']: :obj:`dict`
        Of type mapplot diagnostic 'plotting_information'.
    cube: iris.cube.Cube
    metric: str
        A valid mapplots metric

    Returns (updated pi dict items)
    -------
    cmap_str: str
        If not specified allready a parameter dependent default is used.
    minmax: tuple
        If not specified allready a sensible minmax is set.
    cmap: matplotlib.colors.LinearSegmentedColormap
        As specified in the cmap_str.
        May be truncated if pi 'truncate_colormap' is specified
    norm: matplotlib.colors.Normalize
        Normalizing data into the ``[0.0, 1.0]`` interval.
    ticks: array
        Holding colorbar ticks
    cbar_label:
        As defined in cbar_label_format.
    """
    # defined cmap
    default_cmaps = {'tas':{'mean': 'temp_seq_disc',
                            'bias': 'temp_div_disc',
                            'trend': 'temp_div_disc',
                            'trend-bias': 'temp_div_disc',
                            'trend-min-median-max': 'temp_div_disc',
                            'trend-min-mean-max': 'temp_div_disc',},
                     'pr':{'mean': 'prec_seq_disc',
                           'bias': 'prec_div_disc',
                           'trend': 'prec_div_disc',
                           'trend-bias': 'prec_div_disc',
                           'trend-min-median-max': 'prec_div_disc',
                           'trend-min-mean-max': 'prec_div_disc',},
                            }
    if pi['cmap_str'] == None:
        var = cube.var_name
        try:
            pi.update({'cmap_str': default_cmaps[var][metric]})
            logger.info(f"Using default color map {pi['cmap_str']}")
        except KeyError:
            logger.error(f"No default color map defined for {var} {metric}")
            raise NotImplementedError

    # automatically retrieve minmax if undefined
    if pi['minmax'] == None:
        if pi['mintozero']:
            data_limits = (0, cube.data.max())
        elif pi['maxtozero']:
            data_limits = (cube.data.min(), 0)
        else:
            data_limits = (cube.data.min(), cube.data.max())
        data_range = np.diff(data_limits)

        if data_range > 10:
            factor = 10.
        elif data_range > 1:
            factor = 1.
        elif data_range > .1:
            factor = .1
        elif data_range > .01:
            factor = .01
        else:
            factor = .001

        minmax = (np.floor(data_limits[0] / factor) * factor,
                  np.ceil(data_limits[1] / factor) * factor)
        pi['minmax'] = minmax

    # load the cmap
    cmap = load_IPCCAR6_colors(pi['cmap_str'], pi['Ncolors'])

    # truncate if need be
    if pi['truncate_colormap'] != None:
        cmap = truncate_colormap(cmap, pi['minmax'],
                                 pi['truncate_colormap'])
        pi['minmax'] = pi['truncate_colormap']

    # set handling of missing values
    cmap.set_bad(pi['missing_grey'])

    # finally set the color map
    pi['cmap'] = cmap

    # and build the normalization for the data and ticks for color bar
    norm = mcolors.Normalize(vmin=pi['minmax'][0],
                             vmax=pi['minmax'][1])
    ticksspacing = (norm.vmax - norm.vmin) / cmap.N
    ticks = np.arange(norm.vmin + ticksspacing,
                      norm.vmax - 0.000001, ticksspacing)
    pi['norm'] = norm
    pi['ticks'] = ticks

    # color bar label
    if not pi['cbar_label']:
        cbar_label_dic = {}
        if 'units' in pi['cbar_label_format']:
            unit = str(cube.units)
            unit = format_units(unit)
            cbar_label_dic.update({'units': unit})
        if 'trend_base' in pi['cbar_label_format']:
            if pi['trend_base'] in ['year', 'decade']:
                cbar_label_dic.update({'trend_base': pi['trend_base']})
            elif pi['trend_base'] == 'all':
                cbar_label_dic.update({'trend_base': str(pi['trends_over_n']) +
                                                     'years'})
            else:
                pass
        pi['cbar_label'] = pi['cbar_label_format'].format(**cbar_label_dic)


def add_cartopy_crs(diagnostic):
    """Derive cartopy crs for projection.

    Parameters
    ----------
    diagnostic: :obj:`dict`
        Of type mapplot diagnostic.

    Returns (updated diagnostic dict items)
    -------
    diagnostic['plotting_information']['projection_crs'] cartopy.crs

    Optional: (in case of defined region)
    diagnostic['plotting_information']['plot_extend']
        Of outer bounds

    Optional: (in case of LambertConformal)
    diagnostic['plotting_information']['path_extend'] matplotlib.path.Path
        Of shape of the plotted region to be cut out
    """
    pi = diagnostic['plotting_information']

    proj_str = diagnostic['plotting_information']['projection']

    square_proj_dict = {'PlateCarree':ccrs.PlateCarree(),
                        'Mercator':ccrs.Mercator(),
                        'Miller':ccrs.Miller()}
    square_proj_dict_center180 = {'PlateCarree':ccrs.PlateCarree(central_longitude=180),
                                  'Mercator':ccrs.Mercator(central_longitude=180),
                                  'Miller':ccrs.Miller(central_longitude=180)}
    if proj_str in square_proj_dict:
        if diagnostic['centerlononzero']:
            pi['projection_crs'] = square_proj_dict[proj_str]
        else:
            pi['projection_crs'] = square_proj_dict_center180[proj_str]
        if 'region' in diagnostic:
            pi.update({'plot_extend': [diagnostic['region']['start_longitude'],
                                       diagnostic['region']['end_longitude'],
                                       diagnostic['region']['start_latitude'],
                                       diagnostic['region']['end_latitude']]})
        return

    elif proj_str == 'LambertConformal':
        if 'region' not in diagnostic:
            logger.error(f"Projection {proj_str} only possible if region is "
                        "specified.")
            raise NotImplementedError
        else:
            region = diagnostic['region']

        central_longitude = np.mean([region['start_longitude'],
                                     region['end_longitude']])
        central_latitude = np.mean([region['start_latitude'],
                                    region['end_latitude']])
        if central_latitude < 0.: # southern Hemisphere
            proj = ccrs.LambertConformal(central_longitude=central_longitude,
                                         central_latitude=central_latitude,
                                         cutoff=+30,
                                         standard_parallels=(-33, -45))
        else:
            proj = ccrs.LambertConformal(central_longitude=central_longitude,
                                         central_latitude=central_latitude)
        pi['projection_crs'] =  proj

        # plotextend has to be a little larger so everything is on there
        pi.update({'plot_extend': [region['start_longitude'] - 1.,
                                   region['end_longitude'] + 1.,
                                   region['start_latitude'] - 1.,
                                   region['end_latitude'] + 1.]})

        # path to cut out is exact though
        lons, lats = region_to_square(region)
        path_ext = [[lon,lat] for lon,lat in zip(lons,lats)]
        path_ext = mpath.Path(path_ext).interpolated(20)
        pi.update({'path_extend': path_ext})
        return

    else:
        logger.error(f"Projection {proj_str} not implemented.")
        raise NotImplementedError

# _helper ################################################################
def region_to_square(region):
    """Make outer box out of region definitions."""
    lats = [region['start_latitude'],
            region['start_latitude'],
            region['end_latitude'],
            region['end_latitude'],
            region['start_latitude']]
    lons = [region['start_longitude'],
            region['end_longitude'],
            region['end_longitude'],
            region['start_longitude'],
            region['start_longitude']]
    return (lons, lats)


def interpolate_along_longitude(lons, lats, lonres=0.1):
    """Interpolate points along longitude list.

    Parameters
    ----------
    lons: list
        of longitudes
    lats: list
        of latitudes

    Keywords
    --------
    lonres: float
        of new longitude spacing, default is 0.1

    Returns
    -------
    newlons: list
        of longitudes
    newlats: list
        of latitudes
    """
    newlons, newlats = [], []
    if np.any(abs(np.diff(lons)) > 0.):
        inds = np.where(abs(np.diff(lons)) > 0.)[0]
        for ind in range(0,len(lons)):
            if ind in inds:
                n_lon_interp = int(round(abs(
                    lons[ind+1]-lons[ind]) / lonres + 1, 2))
                newlons.extend(
                    np.linspace(lons[ind], lons[ind+1], n_lon_interp))
                newlats.extend([lats[ind]] * n_lon_interp)
            else:
                newlons.append(lons[ind])
                newlats.append(lats[ind])
        return(newlons, newlats)
    else:
        logger.error(f"No longitude points to interpolate {lons[0:10]}")
        raise ValueError


def extract_lon_lat_box_from_cube(cube):
    """ Returns the outer longitudes latitudes of a cube. """
    return(np.min(cube.coords('longitude')[0].bounds),
           np.max(cube.coords('longitude')[0].bounds),
           np.min(cube.coords('latitude')[0].bounds),
           np.max(cube.coords('latitude')[0].bounds))


###############################################################################
# All connected to preparing the mapplot data
###############################################################################
def split_map_diagnostics(diag_name, diagnostic):
    """ Splits mapplot diagnostics of metric = 'single'.

    Parameters
    ----------
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type histogram diagnostic.

    Returns
    -------
    subdiags: :obj:`dict` of :obj:`dict`
        dictionary of splitted diagnostic dictionaries
    """
    subdiags = {}
    N = diagnostic['N']
    for label, cube in zip(diagnostic['plotting_information']['label'],
                           diagnostic['cube']):
        subdiag_name = diag_name + '_' + label
        subdiag = copy.deepcopy(diagnostic)
        subdiag['plotting_information']['label'] = label
        subdiag['cube'] = cube
        subdiag['N'] = N
        subdiags.update({subdiag_name: subdiag})
    return subdiags


def prepare_mapplot_data(diag_name, diagnostic, ensembles):
    """Updates diagnostic: (:obj:`dict`) with prepared data.

    Parameters
    ----------
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type histogram diagnostic.
    ensembles: :obj:`dict` of :obj:`list` of :obj:`dict`
        Holding ensemble sorted cfg['input_data']

    Returns (updated diagnostic dict items)
    -------
    'cube': iris.cube.Cube / list of iris.cube.Cube
    'label': str / list of str
    'datasets': str / list of str

    Optional:
    'ref_cube': iris.cube.Cube
    """
    logger.info(f"Checking integrity of diagnostic settings: {diag_name}")
    verify_mapplot_diagnostic_settings(diagnostic)

    logger.info(f"Deriving data for diagnostic: {diag_name}")

    cubes, labels, datasets = [], [], []
    if 'bias' in diagnostic['metric']: ref_cubes = []
    mean_cube = None
    if diagnostic['plotting_information']['indicate_N']: Ns = []

    for ens in diagnostic['ensembles']:
        ens_cubes = []
        ens_datasets = []
        # load the data
        if diagnostic['type'] == 'single':
            for dic in ensembles[ens]:
                if dic['dataset'] == 'MultiModelMean': continue
                if diagnostic['exclude_obs'] and \
                    'obs' in dic['project'].lower(): continue
                f = dic['filename']
                cube = iris.load_cube(f)
                ens_cubes.append(cube)
                ens_datasets.append(dic['dataset'])
        elif diagnostic['type'] == 'MultiModelMean':
            for dic in ensembles[ens]:
                if dic['dataset'] == diagnostic['type']:
                    f = dic['filename']
                    cube = iris.load_cube(f)
                    if not cube.coord('time').has_bounds():
                        cube.coord('time').guess_bounds()
                    ens_cubes.append(cube)
                    ens_datasets.append(dic['dataset'])
            if diagnostic['pattern_overlay']:
                pattern_cubes = []
                for dic in ensembles[ens]:
                    if dic['dataset'] != diagnostic['type'] and \
                        'obs' not in dic['project'].lower():
                        f = dic['filename']
                        cube = iris.load_cube(f)
                        if not cube.coord('time').has_bounds():
                            cube.coord('time').guess_bounds()
                        pattern_cubes.append(cube)
        elif diagnostic['type'] == 'single-MultiModelMean':
            for dic in ensembles[ens]:
                if diagnostic['exclude_obs'] and \
                    'obs' in dic['project'].lower(): continue
                f = dic['filename']
                cube = iris.load_cube(f)
                if dic['dataset'] == 'MultiModelMean':
                    mean_cube = cube
                    mean_cube_dataset = dic['dataset']
                else:
                    ens_cubes.append(cube)
                    ens_datasets.append(dic['dataset'])

        # identify models with certain trend properties
        if diagnostic['metric'] in \
            ['trend-min-mean-max', 'trend-min-median-max']:
            # regional extraction
            if 'region_metric' in diagnostic:
                region = diagnostic['region_metric']
            else: region = None
            if 'shapefile' in diagnostic:
                shapefile = os.path.join(exernal_path, diagnostic['shapefile'])
            else: shapefile = None

            if shapefile and region:
                logger.error('Both shapefile and region for trend '\
                             'calculation present')
                raise KeyError

            if 'period_plot' in diagnostic: period = diagnostic['period_plot']
            else: period = None

            if diagnostic['metric'] == 'trend-min-mean-max':
                minmax_inds = get_trendminmax_indices(ens_cubes, period=period,
                                                    region=region,
                                                    shapefile=shapefile)
                ens_cubes = [ens_cubes[minmax_inds[0]],
                             mean_cube,
                             ens_cubes[minmax_inds[1]]]
                ens_datasets = [ens_datasets[minmax_inds[0]],
                                mean_cube_dataset,
                                ens_datasets[minmax_inds[1]]]
            else: #'trend-min-median-max'
                minmedmax_inds = get_trendminmedianmax_indices(ens_cubes,
                                    period=period, region=region,
                                    shapefile=shapefile)
                ens_cubes = [ens_cubes[minmedmax_inds[0]],
                            ens_cubes[minmedmax_inds[1]],
                            ens_cubes[minmedmax_inds[2]]]
                ens_datasets = [ens_datasets[minmedmax_inds[0]],
                                ens_datasets[minmedmax_inds[1]],
                                ens_datasets[minmedmax_inds[2]]]
        elif 'trend-difference' in diagnostic['metric']:
            if 'region_metric' in diagnostic:
                region = diagnostic['region_metric']
            else: region = None
            if 'period_plot' in diagnostic: period = diagnostic['period_plot']
            else: period = None

            # get the single components of the metric
            metric_parts = []
            for metr in diagnostic['metric'].split('-')[2:]:
                N = int(re.search(r'\d+', metr).group())
                mtype = metr.split(str(N))[0]
                metric_parts.append([mtype, N])
            metric_parts = get_trenddifference_indices(ens_cubes,
                                metric_parts, period=period, region=region)
            ens_cubes_new, ens_datasets_new = [], []
            for metr, N, inds in metric_parts:
                datasets_new = []
                for ind in inds:
                    ens_cubes_new.append(ens_cubes[ind])
                    datasets_new.append(ens_datasets[ind])
                ens_datasets_new.append(' '.join(datasets_new))
            ens_cubes = ens_cubes_new
            ens_datasets = ' -- '.join(ens_datasets_new)


        # centering on zero
        if diagnostic['centerlononzero']:
            ens_cubes = [center_lon_on_zero(cube) for cube in ens_cubes]
            if diagnostic['pattern_overlay']:
                pattern_cubes = [center_lon_on_zero(cube)
                                 for cube in pattern_cubes]

        # check if ref cube is needed
        if 'bias' in diagnostic['metric'] or diagnostic['masking'] == True:
            ref_cube = None
            if 'reference_ensemble' in diagnostic.keys():
                for dic in ensembles[diagnostic['reference_ensemble']]:
                    if dic['dataset'] == diagnostic['reference_dataset']:
                        f = dic['filename']
                        ref_cube = iris.load_cube(f)
            else:
                for dic in ensembles[ens]:
                    if dic['dataset'] == diagnostic['reference_dataset']:
                        f = dic['filename']
                        ref_cube = iris.load_cube(f)

            if ref_cube is None:
                logger.error("Masking selected but no reference cube found"
                             "for reference_dataset"
                             f"{diagnostic['reference_dataset']}")
                raise KeyError
            # centering on zero
            if diagnostic['centerlononzero']:
                ref_cube = center_lon_on_zero(ref_cube)

            # ensure the cubes have the same regional data
            if (ens_cubes[0].coord('latitude').shape !=
                ref_cube.coord('latitude').shape) or \
                (ens_cubes[0].coord('longitude').shape !=
                ref_cube.coord('longitude').shape):
                logger.warning("Masking cube shape is different, "
                                "cropping to mask cube")
                sta_lon, end_lon, sta_lat, end_lat = \
                    extract_lon_lat_box_from_cube(ref_cube)
                ens_cubes = [extract_region(cube, sta_lon, end_lon, sta_lat,
                                end_lat) for cube in ens_cubes]
                # centering on zero, again, because extract recion changes is
                if diagnostic['centerlononzero']:
                    ens_cubes = [center_lon_on_zero(cube)
                                    for cube in ens_cubes]
                if diagnostic['pattern_overlay']:
                    pattern_cubes = [extract_region(cube, sta_lon, end_lon,
                                     sta_lat, end_lat)
                                     for cube in pattern_cubes]
                    if diagnostic['centerlononzero']:
                        pattern_cubes = [center_lon_on_zero(cube)
                                        for cube in pattern_cubes]

            if 'trend' in diagnostic['metric']:
                if ens_cubes[0][0].shape != ref_cube[0].shape:
                    logger.error("Shape difference of cube and reference cube")
                    raise TypeError
                if diagnostic['pattern_overlay']:
                    if pattern_cubes[0][0].shape != ref_cube[0].shape:
                        logger.error("Shape difference of cube and reference cube")
                        raise TypeError
            else:
                if 'period_plot' in diagnostic.keys() and \
                    diagnostic['period_plot'] != None:
                    # attempt to do a time mean cube
                    per = diagnostic['period_plot']
                    ens_cubes = [extract_time(cube, per['start_year'],
                        per['start_month'], per['start_day'],
                        per['end_year'], per['end_month'],
                        per['end_day']) for cube in ens_cubes]
                    ref_cube = extract_time(ref_cube,
                        per['start_year'], per['start_month'],
                        per['start_day'], per['end_year'],
                        per['end_month'], per['end_day'])
                if ens_cubes[0].shape != ref_cube.shape:
                    logger.error("Shape difference of cube and ")
                    logger.error("reference cube")
                    raise TypeError
                if len(ens_cubes[0].shape) == 3 and \
                    len(ref_cube.shape) == 3:
                    ref_cube = climate_statistics(ref_cube)
                    ens_cubes = [climate_statistics(cube) \
                        for cube in ens_cubes]

            for cube in ens_cubes:
                if 'trend' in diagnostic['metric']:
                    coord = ref_cube.coord('time')
                    axis = ref_cube.coord_dims(coord)[0]
                    ref_mask = np.all(ref_cube.data.mask, axis=axis)
                    ref_mask = np.array(cube.shape[0] * [ref_mask])
                    cube.data.mask = ref_mask | cube.data.mask
                else:
                    cube.data.mask = ref_cube.data.mask | cube.data.mask
            if diagnostic['pattern_overlay']:
                for cube in pattern_cubes:
                    if 'trend' in diagnostic['metric']:
                        coord = ref_cube.coord('time')
                        axis = ref_cube.coord_dims(coord)[0]
                        ref_mask = np.all(ref_cube.data.mask, axis=axis)
                        ref_mask = np.array(cube.shape[0] * [ref_mask])
                        cube.data.mask = ref_mask | cube.data.mask
                    else:
                        cube.data.mask = ref_cube.data.mask | cube.data.mask

        # get possible trend cubes
        if 'trend' in diagnostic['metric']:
            trends_cubes, trends_over_n = [], []
            for cube in ens_cubes:
                cube_trend, trend_over_n = calculate_trend(cube,
                                     diagnostic['relative'],
                                     diagnostic['trend_base'],
                                     period_trend=diagnostic['period_plot'],
                                     period_norm=diagnostic['period_norm'])
                trends_over_n.append(trend_over_n)
                trends_cubes.append(cube_trend)
            ens_cubes = trends_cubes

            if diagnostic['pattern_overlay']:
                trends_cubes, trends_over_n = [], []
                for cube in pattern_cubes:
                    cube_trend, trend_over_n = calculate_trend(cube,
                                        diagnostic['relative'],
                                        diagnostic['trend_base'],
                                        period_trend=diagnostic['period_plot'],
                                        period_norm=diagnostic['period_norm'])
                    trends_over_n.append(trend_over_n)
                    trends_cubes.append(cube_trend)
                pattern_cubes = trends_cubes

            if 'trend-difference' in diagnostic['metric']:
                metric_cubes = [ens_cubes[0:metric_parts[0][1]],
                                ens_cubes[metric_parts[0][1]:]]
                metric_cubes_mean = []
                for mcubes in metric_cubes:
                    data = []
                    for mcube in mcubes:
                        data.append(mcube.data)
                    data = np.ma.array(data)
                    # mask gp for which there is less than 80% data
                    threshhold = data.shape[0] * (1 - 0.8)
                    mask = np.sum(data.mask, axis=0)
                    newmask = np.zeros(mask.shape, dtype=bool)
                    newmask[mask > threshhold] = True

                    data = np.mean(data, axis=0)
                    data.mask = newmask
                    mcube = copy.deepcopy(mcube)
                    mcube.data = data
                    metric_cubes_mean.append(mcube)
                ens_cubes = [metric_cubes_mean[0] - metric_cubes_mean[1]]
            if 'bias' in diagnostic['metric']:
                ref_cube, trend_over_n = calculate_trend(ref_cube,
                                        diagnostic['relative'],
                                        diagnostic['trend_base'],
                                        period_trend=diagnostic['period_plot'],
                                        period_norm=diagnostic['period_norm'])
                trends_over_n.append(trend_over_n)

            if not len(set(trends_over_n)) == 1:
                logger.error("Trends have been calculated over a different "
                            "numer of years")
                raise ValueError
            else:
                trends_over_n = list(set(trends_over_n))[0]
                diagnostic.update({'trends_over_n': trends_over_n})

        # calculate the bias if wanted
        if 'bias' in diagnostic['metric']:
            if 'trend' in diagnostic['metric']:
                ens_cubes = [absolute_bias(cube, ref_cube)
                             for cube in ens_cubes]
            else:
                if not diagnostic['relative']:
                    ens_cubes = [absolute_bias(cube, ref_cube)
                                 for cube in ens_cubes]
                else:
                    ens_cubes = [relative_bias(cube, ref_cube)
                                 for cube in ens_cubes]
            ref_cubes.append(ref_cube)

        # calculate overlays
        if diagnostic['pattern_overlay']:
            if not np.all([pattern_cubes[0].shape == pcube.shape
                           for pcube in pattern_cubes[1:]]):
                logger.error("Difference in between pattern cubes found")
            if not pattern_cubes[0].shape == ref_cube.shape:
                logger.error("Difference between pattern cubes and ref cube "\
                             "found")
            if not pattern_cubes[0].shape == ens_cubes[0].shape:
                logger.error("Difference between pattern cubes and data cube "\
                             "found")

            if diagnostic['pattern_metric'] == "obs_trend_within_model_spread":
                tmp_data = []
                for cube in pattern_cubes:
                    tmp_data.append(cube.data)
                tmp_data = np.ma.array(tmp_data)
                minmax_data = [np.ma.min(tmp_data, axis=0),
                               np.ma.max(tmp_data, axis=0)]
                minmax_data[0].mask = minmax_data[0].mask | \
                                      ens_cubes[0].data.mask
                minmax_data[1].mask = minmax_data[1].mask | \
                                      ens_cubes[0].data.mask
                ol_data = (minmax_data[0] < ref_cube.data) & \
                          (minmax_data[1] > ref_cube.data)
                ol_data = ~ol_data
                ol_data = ol_data.astype(np.int64)

            ol_cube = copy.deepcopy(pattern_cubes[0])
            ol_cube.data = ol_data
            ol_cube.var_name = diagnostic['pattern_metric']
            ol_cube.long_name = diagnostic['pattern_metric']
            ol_cube.standard_name = None
            ol_cube.units = Unit('')

        if len(ens_cubes[0].shape) > 2:
            # anomalies
            if diagnostic['anomalies']:
                if diagnostic['period_norm'] != None:
                    norm_cubes = [get_base_cube(cube,
                                        period_norm=diagnostic['period_norm'])
                                    for cube in ens_cubes]
                else:
                    norm_cubes = [get_base_cube(cube) for cube in ens_cubes]

                if diagnostic['relative']:
                    for cube, norm_cube in zip(ens_cubes, norm_cubes):
                        cube = relative_anomalies(cube, norm_cube)
                else:
                    for cube, norm_cube in zip(ens_cubes, norm_cubes):
                        cube = absolute_anomalies(cube, norm_cube)

            # cutting out period_plot
            if diagnostic['period_plot'] != None:
                ens_cubes = [get_base_cube(cube,
                                    period_norm=diagnostic['period_plot'])
                                for cube in ens_cubes]

        # get the lable
        if 'plot_title' in diagnostic['plotting_information']:
            if diagnostic['metric'] in ['trend-min-mean-max',
                                        'trend-min-median-max']:
                # force this label format
                label_format = '{project}'
                label = derive_labels_mapplots(ens, ensembles[ens],
                                        label_format, diagnostic)
                if 'median' in diagnostic['metric']:
                    label = ['_'.join([label, submetric]) for submetric in
                                    ['min', 'MultiModelMedian', 'max']]
                else:
                    label = ['_'.join([label, submetric]) for submetric in
                                ['min', 'MultiModelMean', 'max']]
            else:
                if 'title_format' not in diagnostic['plotting_information']:
                    if diagnostic['type'] == 'single':
                        if 'trend-difference' in diagnostic['metric']:
                            label_format = '{project}'
                        else:
                            label_format = '{dataset}'
                    elif diagnostic['type'] == 'MultiModelMean':
                        label_format = '{project}'
                    else:
                        logger.error("Default title format for "
                                    f"diagnostic['type'] not defined.")
                else:
                    label_format = \
                            diagnostic['plotting_information']['title_format']
                label = derive_labels_mapplots(ens, ensembles[ens],
                                        label_format, diagnostic)

        # get the models N
        if diagnostic['plotting_information']['indicate_N']:
            N = derive_N_mapplots(diagnostic, ensembles[ens])

        # cube shape checks
        shape_len = 2
        valid_coords = ['latitude', 'longitude',
                        'grid_latitude', 'grid_longitude']
        for cube in ens_cubes:
            cube_coord_names = [coord.standard_name for coord in
                                cube.coords(dim_coords=True)]
            if len(cube.shape) != shape_len:
                logger.error(f"Need a cube with shape length of {shape_len}"
                              "Got cube with shape length of "
                             f"{len(cube.shape)}")
                raise TypeError
            if not all(coord in valid_coords for coord in cube_coord_names):
                logger.error(f"Need a cube with valid coordinates "
                             f"{valid_coords}"
                             f"for plotting a map. Got {cube_coord_names}")
                raise TypeError

        if len(ens_datasets) != len(ens_cubes) and \
            not 'trend-difference' in diagnostic['metric']:
            logger.error("Lenghts of datasets and cubes does not match")
            raise TypeError

        cubes.extend(ens_cubes)
        datasets.extend(ens_datasets)
        if isinstance(label, str): labels.append(label)
        else: labels.extend(label)
        if diagnostic['plotting_information']['indicate_N']:
            if isinstance(N, int): Ns.append(N)
            else: Ns.extend(N)

    if len(cubes) == 1:
        diagnostic.update({'cube': cubes[0]})
        diagnostic.update({'dataset': datasets[0]})
        diagnostic['plotting_information'].update({'label': labels[0]})
        if 'bias' in diagnostic['metric']:
            diagnostic.update({'ref_cube': ref_cubes[0]})
        if diagnostic['plotting_information']['indicate_N']:
            diagnostic.update({'N': Ns[0]})
        if diagnostic['pattern_overlay']:
            diagnostic.update({'overlay_cube': ol_cube})
    else:
        diagnostic.update({'cube': cubes})
        diagnostic.update({'dataset': datasets})
        diagnostic['plotting_information'].update({'label': labels})
        if 'bias' in diagnostic['metric']:
            diagnostic.update({'ref_cube': ref_cubes})
        if diagnostic['plotting_information']['indicate_N']:
            diagnostic.update({'N': Ns})

    if 'plot_markers' in diagnostic:
        logger.info("Deriving additional symbol data for diagnostic: "\
                    f" {diag_name}")
        derive_marker_data(diag_name, diagnostic)


def derive_marker_data(diag_name, diagnostic):
    """Get the correct derive data function."""
    if diagnostic['plot_markers'] == 'external_UrbanBox':
        derive_urbanbox_data(diagnostic)


def derive_urbanbox_data(diagnostic, file_path=exernal_path):
    """Derive data for Urban Warming markers."""
    logger.info("Deriving data for the Urban warming box")

    cube = diagnostic['cube']
    cube.units = Unit('degC')
    gps_3x3, gps_5x5, gps_7x7 = [], [], []

    urban_path = os.path.join(exernal_path, "Urban_Box_data")
    cities_file = 'cities.csv'
    countries_file = 'countries.csv'

    shape_path = os.path.join(urban_path,
                              "country_shapes_naturalearth")

    # Load the cities
    df = pd.read_csv(os.path.join(urban_path, cities_file))
    logger.info(f"Loaded {os.path.join(urban_path, cities_file)}")

    # remove some cities
    for city in ['Puerto rico', 'Porto area', 'Seoul area', 'Barrow Alaska']:
        df = df[df['Specific country or region'] != city]

    # add some colums
    for key in ['tas obs', 'tas sum']:
        df[key] = ''
    df['type'] = 'city'

    # add the temperature of the surroundings
    logger.info("Deriving surrounding warming data")
    for idx in df.index:
        logger.info(f"Adding {df.loc[idx,'Specific country or region']}")

        lon, lat = df.loc[idx,'x'], df.loc[idx,'y']
        try:
            latind = [aind
                      for aind,bnd in enumerate(cube.coord('latitude').bounds)
                      if bnd[0] < lat <= bnd[1]][0]
        except IndexError:
            latind = [aind
                      for aind,bnd in enumerate(cube.coord('latitude').bounds)
                      if bnd[0] >= lat > bnd[1]][0]
        lonind = [oind
                  for oind,bnd in enumerate(cube.coord('longitude').bounds)
                  if bnd[0] < lon <= bnd[1]][0]

        if cube.data.mask[latind, lonind]:
            if np.all(cube.data.mask[latind-3:latind+4, lonind-3:lonind+4]):
                logger.error(f"Data needs 9x9 surrounding grid points")
                raise NotImplemented
            if np.all(cube.data.mask[latind-2:latind+3, lonind-2:lonind+3]):
                logger.info(f'Using 7x7 surrounding grid points')
                tas = cube.data[latind-3:latind+4, lonind-3:lonind+4].mean()
                gps_7x7.append(df.loc[idx,'Specific country or region'])
            elif np.all(cube.data.mask[latind-1:latind+2, lonind-1:lonind+2]):
                logger.info(f'Using 5x5 surrounding grid points')
                tas = cube.data[latind-2:latind+3, lonind-2:lonind+3].mean()
                gps_5x5.append(df.loc[idx,'Specific country or region'])
            else:
                logger.info(f'Using 3x3 surrounding grid points')
                tas = cube.data[latind-1:latind+2, lonind-1:lonind+2].mean()
                gps_3x3.append(df.loc[idx,'Specific country or region'])
        else:
            tas = cube.data[latind, lonind]

        df.loc[idx,'tas obs'] = tas
        df.loc[idx,'tas sum'] = tas + df.loc[idx,'Temperature']

    # Load the countries
    df2 = pd.read_csv(os.path.join(urban_path, countries_file))
    logger.info(f"Loaded {os.path.join(urban_path, countries_file)}")

    # keep China, Japan and Thailand
    target_countries = ['China', 'Japan', 'Thailand']
    df2 = df2[df2['Specific country or region'].isin(target_countries)]

    # add some colums
    for key in ['tas obs', 'tas sum']:
        df2[key] = ''
    df2['type'] = 'country'

    if 'add_shapes' in diagnostic:
        if diagnostic['add_shapes']:
            diagnostic.update({'shape_paths': []})

    for idx in df2.index:
        logger.info(f"Adding {df2.loc[idx,'Specific country or region']}")

        country = df2.loc[idx, 'Specific country or region']
        country_shape_path = os.path.join(shape_path, country, country + '.shp')
        results = extract_shape(cube, country_shape_path)
        results = area_statistics(results, 'mean')

        df2.loc[idx,'tas obs'] = results.data
        df2.loc[idx,'tas sum'] = results.data + df2.loc[idx,'Temperature']

        if 'add_shapes' in diagnostic:
            if diagnostic['add_shapes']:
                diagnostic['shape_paths'].append(country_shape_path)

    df = df.append(df2)
    df = df.sort_values(by=['tas sum'])
    df = df.reset_index(drop=True)

    df['rel obs'] = df['tas obs'] / df['tas sum']
    df['rel urban'] = df['Temperature'] / df['tas sum']

    diagnostic.update({"data_markers": df})


def derive_labels_mapplots(ens, ensemble, label_format, diagnostic):
    """Extract labels from datasets according to label_format formatting.

    Parameters
    ----------
    ens: str
    ensemble: :obj:`list` of :obj:`dict`
        Holding cfg['input_data']
    label_format: format str
    input_type: str
        mapplot type setting

    Returns
    -------
    label: str or list of str
    """
    keys = re.findall(r'\{(.*?)\}', label_format)

    labeling = {k: set() for k in keys}
    for dataset in ensemble:
        for k in keys:
            labeling[k].add(dataset[k])

    if diagnostic['type'] == 'MultiModelMean' or diagnostic['exclude_obs']:
        if 'project' in labeling:
            if 'OBS' in labeling['project']:
                labeling['project'].remove('OBS')

    if not np.all([len(value) == 1 for value in labeling.values()]):
        logger.warning(f"Unable to build clear labeling for {ens} with "
                       f"scheme {label_format}. Building label on dataset "
                       "level")
        labels = [label_format.format(**dataset) for dataset in ensemble]
        labels = [format_label(label) for label in labels]
        return labels
    else:
        for k, v in labeling.items():
            labeling[k] = list(v)[0]
        label = label_format.format(**labeling)
        label = format_label(label)
        return label


def derive_N_mapplots(diagnostic, ensemble):
    """
    Get N of datasets.

    Parameters
    ----------
    diagnostic: :obj:`dict`
        Of type mapplot diagnostic.
    ensemble: :obj:`list` of :obj:`dict`
        Holding cfg['input_data']

    Returns
    -------
    N: int
    """
    input_type = diagnostic['type']

    if input_type == 'single':
        N = []
        for dic in ensemble:
            if dic['dataset'] == 'MultiModelMean': continue
            if diagnostic['exclude_obs'] and \
                'obs' in dic['project'].lower(): continue
            N.append(1)
    else:
        N = 0
        for dic in ensemble:
            if dic['dataset'] == 'MultiModelMean': continue
            if 'obs' in dic['project'].lower(): continue
            N += 1
    return N


def verify_mapplot_diagnostic_settings(diagnostic):
    """ Verifies mapplot diagnostic definitions. """
    defaults = {'centerlononzero': False,
                'relative': False,
                'anomalies': False,
                'period_plot': None,
                'period_norm': None,
                'masking': False,
                'metric': 'mean',
                'wind_overlay': False,
                'orog_overlay': False,
                'pattern_overlay': False,
                'N': None,
                }

    valids = {'type': ['single', 'MultiModelMean', 'single-MultiModelMean'],
    }
    valid_types = {}

    verify_diagnostic_settings(diagnostic, defaults, valids, valid_types)

    if diagnostic['metric'] in ['trend-min-median-max', 'trend-min-mean-max'] \
        or 'trend-difference' in diagnostic['metric']:
        verify_diagnostic_settings(diagnostic, {'exclude_obs': True},
                                   valids, valid_types)
    else:
        verify_diagnostic_settings(diagnostic, {'exclude_obs': False},
                                   valids, valid_types)

    if diagnostic['relative']:
        diagnostic['anomalies'] = True

    if diagnostic['metric'] == 'trend':
        verify_diagnostic_settings(diagnostic, {},
                                   {'trend_base': ['all', 'decade', 'year']},
                                   {})

    if diagnostic['period_plot'] != None:
        diagnostic['period_plot'] = _check_period_information(
            diagnostic['period_plot'], "plot period")

    if diagnostic['period_norm'] != None:
        diagnostic['period_norm'] = _check_period_information(
            diagnostic['period_norm'], "norm period")

    if diagnostic['masking'] == True:
        if not 'reference_dataset' in diagnostic:
            logger.error("Masking selected but no 'reference_dataset' defined")
            raise KeyError

    if 'bias' in diagnostic['metric']:
        if not 'reference_dataset' in diagnostic:
            logger.error("Masking selected but no 'reference_dataset' defined")
            raise KeyError
        if not diagnostic['masking']:
            diagnostic['masking'] = True

    if 'region' in diagnostic:
        verify_diagnostic_region_setting('region', diagnostic['region'])
    box_dicts = {k: v for k, v in diagnostic.items() if 'box' in k}
    for k,v in box_dicts.items():
        verify_diagnostic_region_setting(k, v)

    if diagnostic['metric'] in ['trend-min-median-max', 'trend-min-mean-max'] \
        or 'trend-difference' in diagnostic['metric']:
        if not 'region_metric' in diagnostic:
            if 'shapefile' in diagnostic:
                pass
            elif 'box' in diagnostic:
                logger.warning("No region for calculation of metric "
                               f"{diagnostic['metric']} defined. Using plot "
                               f"box region {diagnostic['box']}.")
                diagnostic['region_metric'] = diagnostic['box']
            elif 'region' in diagnostic:
                logger.warning("No region for calculation of metric "
                               f"{diagnostic['metric']} defined. Using plot "
                               f"region {diagnostic['region']}.")
                diagnostic['region_metric'] = diagnostic['region']
            else:
                logger.warning("No region for calculation of metric "
                               f"{diagnostic['metric']} defined. Using data "
                               f"region.")
        else:
            verify_diagnostic_region_setting(
                'region_metric', diagnostic['region_metric'])

    # check some plotting information settings
    pi_defaults = {'indicate_N': False,
                   'figsize': (16, 12)}

    for k, v in pi_defaults.items():
        if k not in diagnostic['plotting_information']:
            diagnostic['plotting_information'][k] = v
            logger.warning(f"No information for {k} given. "
                           f"Using {v}")


def verify_diagnostic_region_setting(key, setting):
    """ Verify diagnostic region definitions.

    Parameters
    ----------
    key: str
        of region dict
    setting: :obj:`dict`
        diagnostic setting
    """
    valid_type = dict
    valid_keys = ['end_latitude', 'end_longitude',
                  'start_latitude', 'start_longitude']

    if not isinstance(setting, dict):
        logger.error(f"{type(setting)} no valid type for {key}.")
        raise NotImplementedError
    if not all(elem in setting.keys() for elem in valid_keys):
        logger.error(f"{key} misses arguments. "
                        f"Valid arguments are {valid_keys}.")
        raise KeyError


###############################################################################
# All connected to histogram plotting
###############################################################################
def fill_histogram_ax(ax, diag_name, diagnostic):
    """Fills the histogram ax.

    - Check diagnostic histogram plotting information.
    - Prepare data to be plotted and some plot specs.
    - Plot the histogram and marker.
    - Finish up plot.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type histogram diagnostic.
    """
    logger.info("Checking integrity of diagnostic plotting "
                "information: {}".format(diag_name))
    prepare_histogram_plotting_data(diagnostic)
    prepare_histogram_plot(diagnostic)

    logger.info("Doing plot for diagnostic: {}".format(diag_name))

    plot_histogram_ax(ax, diagnostic)

    finish_histogram_ax(ax, diagnostic)

    return ax


def plot_histogram_ax(ax, diagnostic):
    """Plot the histogram.

    - the histogram plot
    - the marker above the histogram
    - the mean marker above the other marker

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    diagnostic: :obj:`dict`
        Of type histogram diagnostic.
    """
    hist_dicts = {k: v for k, v in diagnostic.items() if 'histogram' in k}
    pi = diagnostic['plotting_information']

    # plot the histogram
    if pi['histogram']:
        hist_diag = [v for v in hist_dicts.values() if v['plot_type'] == 'histogram'][0]
        histcolors = [label_colors(label) \
                      for label in hist_diag['labels']]
        plt.hist(hist_diag['data'],
                 pi['bins'],
                 weights=pi['weights'],
                 label=hist_diag['labels'],
                 color=histcolors)
        if pi['diag_order']:
            pi.update({'yposs': pi['ylims'][1] + \
                np.diff(pi['ylims']) / 10 * (np.arange(pi['nrows']) + 1.5)})
    else:
        pi.update({'yposs': np.arange(pi['nrows']) + 1})

    if pi['ylims']:
        ax.set_ylim(pi['ylims'])
    else:
        pi.update({'ylims': ax.get_ylim()})

    # Do this now so we can resize the plot accordingly later and addapt the
    # ylims for the symbols
    if pi['xlims']:
        ax.set_xlim(pi['xlims'])

    # Add the symbols
    if pi['hist_order_det']:
        xpossx, sym_labels, ind = [], [], 0
        ymax = pi['ylims'][1]

        # for histk in pi['diag_order']:
        for histk, label in pi['hist_order_det']:
            if histk != 'histogram_means':
                data_index = diagnostic[histk]['labels'].index(label)
                trends = diagnostic[histk]['data'][data_index]
                ypos = pi['yposs'][ind]
                ind += 1
                if diagnostic[histk]['plot_type'] == 'symbols':
                    for trend in trends:
                        plt.plot(trend, ypos, marker=label_markers(label),
                                color=label_colors(label),
                                label=label, clip_on=False)
                        print(label, trend, ypos, label_colors(label))
                    # ad an x to single models?
                    if 'mark_single_models' in diagnostic[histk].keys():
                        ensemble = diagnostic[histk]['ensembles'][data_index]
                        if ensemble in \
                            diagnostic[histk]['mark_single_models'].keys():
                            aliases = diagnostic[histk]['mark_single_models'][ensemble]
                            ens_aliases = diagnostic[histk]['aliases'][data_index]
                            mark_trends = [trends[ens_aliases.index(al)]
                                           for al in aliases]
                            for trend in mark_trends:
                                plt.plot(trend, ypos, marker='x', color='k',
                                         clip_on=False)
                elif diagnostic[histk]['plot_type'] == 'violin':
                    ymax = ypos
                    violin = ax.violinplot([np.array(trends)], [ypos],
                                vert=False,
                                widths=[np.diff(pi['yposs'])[0] / 1.5],
                                showmeans=True)
                    color = label_colors(label)
                    for patch in violin['bodies']:
                        patch.set_facecolor(color)
                        patch.set_edgecolor('black')
                        patch.set_alpha(1)
                    for pkey in ['cmeans', 'cmaxes', 'cmins', 'cbars']:
                        violin[pkey].set_color('black')
                elif diagnostic[histk]['plot_type'] == 'boxplot':
                    ymax = ypos
                    box = ax.boxplot([np.array(trends)],
                                positions = [ypos],
                                vert=False,
                                widths=[np.diff(pi['yposs'])[0] / 1.5])
                    color = label_colors(label)
                    patch = box['boxes'][0]
                    boxX = list(patch.get_xdata())
                    boxY = list(patch.get_ydata())
                    boxCoords = list(zip(boxX, boxY))
                    boxPolygon = Polygon(boxCoords, facecolor=color)
                    ax.add_patch(boxPolygon)
                    patch = box['medians'][0]
                    patch.set_color('k')
                if trends:
                    xpossx.append(np.max(trends))
                sym_labels.append(label)
            else:
                ypos = pi['yposs'][ind]
                ind += 1
                for trends, label in zip(diagnostic[histk]['data'],
                                         diagnostic[histk]['labels']):
                    plt.plot(trends[0], ypos, marker=label_markers(label),
                             color=label_colors(label),
                             label=label, clip_on=False)
                    print(label, trends[0], ypos, label_colors(label))
                xpossx.append(np.max(diagnostic[histk]['data']))
                sym_labels.append(pi['mean_text_label'])

        pi.update({'xposs': xpossx,
                   'sym_labels': sym_labels})
        pi['ylims'] = (pi['ylims'][0], ymax + np.diff(pi['yposs'])[0] / 1.5)


def finish_histogram_ax(ax, diagnostic):
    """Finish up the histogram plot.

    - add set keywords
    - adapt ratio of plot according to added marker rows
    - add top axis
    - set ticks to bins
    - add legend

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    diagnostic: :obj:`dict`
        Of type histogram diagnostic.
    """

    pi = diagnostic['plotting_information']

    if pi['xlabel']:
        ax.set_xlabel(format_units(pi['xlabel']))
    if pi['title']:
        ax.set_title(pi['title'], y= (pi['yposs'][-1] / pi['ylims'][1]) * 1.06)

    # adapt the axis ratio acording to new data
    xratio = 1.
    ax.set_ylim(pi['ylims'])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1 /
                     xratio, box.height * pi['ylims'][1] / pi['yposs'][-1]])

    # add the top axis
    if pi['histogram']:
        ax.spines['top'].set_visible(True)
        ax.tick_params(axis='x', which='both', top=True)
        if pi['ylabel']:
            ax.set_ylabel(pi['ylabel'])
    else:
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', which='both', left=False)
        ax.set_yticklabels([])

    if pi['add_legend']:
        if pi['histogram']:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            if pi['diag_order']:
                for histk in pi['diag_order']:
                    if histk != 'histogram_means':
                        for ens in diagnostic[histk]['labels']:
                            try:
                                by_label.pop(ens)
                            except KeyError:
                                pass
            for k in by_label.keys():
                by_label[k].set_linewidth(0)
            by_label = translate_obs_naming(by_label)
            ax.legend(by_label.values(), by_label.keys(), ncol=pi['ncol'],
                    loc=pi['loc'])

        if pi['hist_order_det']:
            xofs = np.diff(ax.get_xlim())/20.
            xposs_x = np.max(pi['xposs'])
            if xposs_x < ax.get_xlim()[1]:
                new_xposs_x = xposs_x + 2. * xofs
                if xposs_x < new_xposs_x:
                    xposs_x = new_xposs_x
            texts = []
            for y, xs, label in zip(pi['yposs'], pi['xposs'],
                                    pi['sym_labels']):
                texts.append(ax.annotate(label, (xposs_x, y),
                             horizontalalignment='left',
                             fontsize=pylab.rcParams['legend.fontsize'],
                             verticalalignment='center',
                             annotation_clip=False))
                ax.plot([xs + 0.75 * xofs, xposs_x - 0.25 * xofs],
                        [y, y], ls='--', color='k', lw=1., clip_on=False)


def prepare_histogram_plotting_data(diagnostic):
    """Puts cube data in arrays.

    Parameters
    ----------
    diagnostic: :obj:`dict`
        Of type histogram diagnostic.

    Updates
    -------
    diagnostic['plotting_information'] :obj:`dict`
    diagnostic ensemble data :obj:`list`
    """
    hist_keys = [k for k in diagnostic.keys() if 'histogram' in k]
    for histk in hist_keys:
        trends = []
        for ens_cubes in diagnostic[histk]['trends']:
            ens_trends = [cube.data for cube in ens_cubes]
            trends.append(ens_trends)
        diagnostic[histk].update({'data': trends})


def prepare_histogram_plot(diagnostic):
    """Updates diagnostic: (:obj:`dict`) with histogram plotting information.

    - Ensure valid plotting_information.
    - Adds defaults if nothing is specified
    - Possibly adds x_label 'Trend ({units} {trend_base}$^{{-1}}$)'
    Complements information on:
    - xlims
    - bins
    - weights

    Parameters
    ----------
    diagnostic: :obj:`dict`
        Of type histogram diagnostic.

    Updates
    -------
    diagnostic['plotting_information'] :obj:`dict`
    """
    from collections import Counter

    xlabel_format_def = 'Trend ({units} {trend_base}$^{{-1}}$)'
    defaults = {'add_legend': True,
                'xlabel': None,
                'ylabel': '(% of simulations)',
                'ncol': 1,
                'loc': 1,
                'xlims': None,
                'ylims': None,
                'bins_spacing': None,
                'dgrey': [0.4, 0.4, 0.4],
                'mgrey': [0.65, 0.65, 0.65],
                'lgrey': [0.9, 0.9, 0.9],
                'add_shading': True,
                'title': '',
                'mean_text_label': 'Ensemble means',
                'histogram': False,
                'diag_order': None,
                }

    pi = diagnostic['plotting_information']

    for kd, default in defaults.items():
        if kd not in pi:
            pi.update({kd: default})

    # if xlabel is None extract information from a cube
    if pi['xlabel'] == None:
        if diagnostic['trend_base'] in ['year', 'decade']:
            xlab_dic = {'trend_base': diagnostic['trend_base']}
        elif diagnostic['trend_base'] == 'all':
            xlab_dic = {'trend_base': str(diagnostic['trends_over_n']) +
                                                     'years'}
        else:
            pass
        histkeys = [k for k in diagnostic.keys() if 'histogram' in k]
        try:
            unit = str(diagnostic[histkeys[0]]['trends'][0][0].units)
        except IndexError:
            unit = str(diagnostic[histkeys[0]]['trends'][1][0].units)
        if unit == 'degC':
            unit = '°C'
        xlab_dic.update({'units': unit})
        pi['xlabel'] = xlabel_format_def.format(**xlab_dic)
        logger.warning("No xlabel defined, using cube units: "
                       "{}".format(pi['xlabel']))

    # get xlims
    if pi['xlims'] is None:
        all_trends = []
        for histk in ['histogram_histogram', 'histogram_symbols']:
            if histk not in diagnostic:
                continue
            for ens_data in diagnostic[histk]['data']:
                for data in ens_data:
                    all_trends.append(data)
        data_limits = (np.min(all_trends), np.max(all_trends))
        data_range = np.diff(data_limits)

        if data_range > 10:
            factor = 10.
        elif data_range > 1:
            factor = 1.
        elif data_range > .1:
            factor = .1
        elif data_range > .01:
            factor = .01
        else:
            factor = .001

        xlims = (np.floor(data_limits[0] / factor) * factor,
                 np.ceil(data_limits[1] / factor) * factor)
        pi['xlims'] = xlims

    # get bins
    hist_dicts = {k: v for k, v in diagnostic.items() if 'histogram' in k}
    hist_types = {k: v['plot_type'] for k, v in hist_dicts.items()}

    # if 'histogram_histogram' in diagnostic:
    if 'histogram' in hist_types.values():
        hist_diag = [v for v in hist_dicts.values() if v['plot_type'] == 'histogram'][0]
        if pi['bins_spacing'] is None:
            data_range = np.round(np.diff(pi['xlims']), 4)
            pi['bins_spacing'] = data_range[0] / 10

        nbins = int(np.round(np.round(np.diff(pi['xlims']), 4) /
                             pi['bins_spacing']))
        bins = np.linspace(np.round(pi['xlims'][0], 4),
                           np.round(pi['xlims'][1], 4), nbins + 1)
        bins = [np.round(bin, 4) for bin in bins]
        pi.update({'bins': bins})

        xws = []
        for ens in hist_diag['trends']:
            x_w = np.empty(len(ens))
            x_w.fill(1 / len(ens) * 100.)
            xws.append(x_w)
        pi.update({'weights': xws})
        pi.update({'histogram': True})

        if not pi['ylims']:
            ylim_up = []
            for data, weights in zip(hist_diag['data'], pi['weights']):
                digitized = np.digitize(np.array(data), pi['bins'])
                ylim_up.append(np.max(list(Counter(digitized).values())) * \
                               weights[0])
            pi['ylims'] = [0, np.ceil(np.max(ylim_up))]

    remain_diags = {k: v for k, v in hist_dicts.items() if v['plot_type'] != 'histogram'}
    if remain_diags:
        # get the number of rows
        nrows = 0
        for histk, hist in remain_diags.items():
            if histk == 'histogram_means':
                nrows += 1
            else:
                nrows += len(hist['labels'])
        pi.update({'nrows': nrows})

        # ordering
        default_order = ['violin', 'boxplot', 'symbols']
        remain_types = {k: v['plot_type'] for k, v in remain_diags.items() if k != 'histogram_means'}

        hist_order_det = []
        for plot_type in default_order:
            for k, v in remain_types.items():
                if v == plot_type:
                    labels = remain_diags[k]['labels']
                    for label in labels:
                        hist_order_det.append((k, label))

        if diagnostic['add_mean_trends']:
            if diagnostic['add_mean_trends_before_obs']:
                hist_order_det.insert(-1, ('histogram_means', ''))
            else:
                hist_order_det.append(('histogram_means', ''))

        pi.update({'hist_order_det': hist_order_det})


###############################################################################
# All connected to preparing the histogram data
###############################################################################
def prepare_histogram_data(diag_name, diagnostic, ensembles):
    """Updates diagnostic: ():obj:`dict`) with prepared data.

    Parameters
    ----------
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type histogram diagnostic.
    ensembles: :obj:`dict` of :obj:`list` of :obj:`dict`
        Holding ensemble sorted cfg['input_data']
    """
    diagnostic = copy.deepcopy(diagnostic)

    logger.info(f"Checking integrity of diagnostic settings: {diag_name}")
    verify_histogram_diagnostic_settings(diagnostic)

    logger.info(f"Deriving data for diagnostic: {diag_name}")

    # check if same number of time present in all cubes
    trends_over_n = []

    hist_dicts = {k: v for k, v in diagnostic.items() if 'histogram' in k}
    for histk, hist in hist_dicts.items():
        ens_trends, labels, model_aliases = [], [], []
        for ens in hist['ensembles']:
            if ens == 'dummy':
                labels.append('')
                model_aliases.append('')
                ens_trends.append([])
                continue
            trends, model_alias = [], []
            for dic in ensembles[ens]:
                f = dic['filename']
                model_alias.append(dic['alias'])
                cube = iris.load_cube(f)

                cube_trend, trend_over_n = calculate_trend(cube,
                                diagnostic['relative'],
                                diagnostic['trend_base'],
                                period_trend=diagnostic['period_trend'],
                                period_norm=diagnostic['period_norm'])

                trends.append(cube_trend)
                trends_over_n.append(trend_over_n)
            ens_trends.append(trends)

            if isinstance(hist['labeling'], str):
                label = derive_labels_histogram(ens, ensembles[ens],
                                                hist['labeling'])
            else:
                for labeling_format in hist['labeling']:
                    try:
                        label = derive_labels_histogram(ens, ensembles[ens],
                                                        labeling_format)
                        break
                    except KeyError: continue
                    except TypeError: continue

            label = format_label(label)
            labels.append(label)
            model_aliases.append(model_alias)
        hist.update({'trends': copy.deepcopy(ens_trends)})
        hist.update({'labels': copy.deepcopy(labels)})
        hist.update({'aliases': copy.deepcopy(model_aliases)})

    if not len(set(trends_over_n)) == 1:
        logger.warning("Trends have been calculated over a different "
                       "numer of years: {[n for n in trends_over_n]}")
        if diagnostic['trend_base'] == 'all':
            raise ValueError
    else:
        trends_over_n = list(set(trends_over_n))[0]
        diagnostic.update({'trends_over_n': trends_over_n})

    if diagnostic['add_mean_trends']:
        mean_trends, mean_ensembles, labels, metrics = [], [], [], []
        for histk, hist in hist_dicts.items():
            for eind, ens in enumerate(hist['ensembles']):
                if 'obs' in ens.lower():
                    continue
                if ens == 'dummy':
                    continue
                logger.info(f"getting mean trends of {ens}")
                ens_trends = hist['trends'][eind]
                mean_trends.append([get_mean_trend_cube(ens_trends)])
                mean_ensembles.append(ens)
                labels.append(hist['labels'][eind])
            metrics.append(hist['metric'])
        metrics = list(set(metrics))
        if len(metrics) == 1: metrics = 'mean ' + metrics[0]
        mean_dict = {'trends': mean_trends,
                     'ensembles': mean_ensembles,
                     'labels': labels,
                     'metric': metrics,
                     'plot_type': 'symbols'}
        diagnostic.update({'histogram_means': mean_dict})

    return diagnostic


def derive_labels_histogram(ens, ensemble, label_format):
    """Extract label from datasets based on histogram['labeling'] formatting.

    Parameters
    ----------
    ens: str
    ensemble: :obj:`list` of :obj:`dict`
        Holding cfg['input_data']
    label_format: format str

    Returns
    -------
    label: str or list of str
    """
    keys = re.findall('\{(.*?)\}', label_format)
    if 'N' in keys:
        N = 0
        keys.remove('N')
    else: N = None

    labeling = {k: set() for k in keys}
    for dataset in ensemble:
        if N != None: N += 1
        for k in keys:
            labeling[k].add(dataset[k])

    if not np.all([len(value) == 1 for value in labeling.values()]):
        logger.warning(f"Unable to build clear labeling for {ens} with "
                       f"scheme {label_format}. Falling back to {ens}")
        return ens
    else:
        for k,v in labeling.items():
            labeling[k] = list(v)[0]
        if N: labeling.update({'N': N})
        return label_format.format(**labeling)


def get_mean_trend_cube(cubes):
    """ Doing a mean trend cube out of single linear trend cubes."""
    from functools import reduce

    # ensure time consistency (very sloppy and breaks due to different
    # calendar types) todo has to be reworked see also trendtodo
    # just for completeness, though the trends are calculated in per timestep
    # trend, so this shouldn't be an issue
    all_times = []
    for cube in cubes:
        all_times.append(np.diff(cube.coord('year').bounds).flatten()[0])
    all_times = list(set(all_times))
    if len(all_times) != 1:
        logger.warning("Trend cubes have differences in their time coordinate")
        max_diff = [abs(all_times[0] - tt) for tt in all_times[1:]]
        if max(max_diff) > 1:
            logger.warning(f"max time coordinate difference is {max_diff}")

    mean_cube = cubes[0].copy()
    for coord in ['latitude', 'longitude']:
        mean_cube.remove_coord(coord)
    mean_cube.attributes = {}

    data = [cube.data for cube in cubes]
    data = np.mean(data)
    mean_cube.data = data

    return mean_cube


def verify_histogram_diagnostic_settings(diagnostic):
    """Verify histogram diagnostic definitions."""
    defaults = {'trend_base': 'all',
                'relative': False,
                'period_norm': None,
                'add_mean_trends': True,
                'add_mean_trends_before_obs': True}
    valids = {'trend_base': ['all', 'decade', 'year']}
    valid_types = {'relative': bool,
                   'add_mean_trends': bool,
                   }

    verify_diagnostic_settings(diagnostic, defaults, valids, valid_types)

    if not 'period_trend' in diagnostic:
        logger.warning("No period given for histogram trends. "
                       "Using entire period")
        diagnostic.update({'period_trend': None})
    else:
        diagnostic['period_trend'] = _check_period_information(
            diagnostic['period_trend'], "histogram")

    if diagnostic['relative']:
        if not 'period_norm' in diagnostic:
            logger.warning("Relatve treatment of trends selected, but no"
                           " period given. Using entire period")
            diagnostic.update({'period_norm': None})
        else:
            diagnostic['period_norm'] = _check_period_information(
                diagnostic['period_norm'], "relative treatment")

    # check some plotting information settings
    pi_defaults = {'figsize': (12, 12)}

    for k, v in pi_defaults.items():
        if k not in diagnostic['plotting_information']:
            diagnostic['plotting_information'][k] = v
            logger.warning(f"No information for {k} given. "
                           f"Using {v}")


###############################################################################
# All connected to preparing diagnostic information
###############################################################################
def select_diagnostics(metadata, subkeys):
    """Select metadata based on partial strings.

    Parameters
    ----------
    metadata : :obj:`list` of :obj:`dict`
        A list of metadata describing preprocessed data.
    subkey : str
        The substring that the key should include.

    Returns
    -------
    :obj:`dict` of :obj:`dict`
        A dictionary containing the requested groups.
    """
    if isinstance(subkeys, str):
        subkeys = [subkeys]
    elif isinstance(subkeys, list):
        pass
    else:
        logger.error("Subkey has to be of type str or list."
                     f"Got {type(subkeys)}.")
        raise KeyError

    groups = {}
    for k,ensemble_dic in metadata.items():
        for subkey in subkeys:
            if subkey == k.split('_')[0]:
                groups.update({k: ensemble_dic})

    return groups


def select_combi_diagnostics(metadata):
    """Select metadata based on partial strings.

    Parameters
    ----------
    metadata : :obj:`list` of :obj:`dict`
        A list of metadata describing preprocessed data.

    Returns
    -------
    :obj:`dict` of :obj:`dict`
        A dictionary containing the requested groups.
    """

    groups = {}
    for k,ensemble_dic in metadata.items():
        if 'diagnostics_combination_figure' in k:
            groups.update({k: ensemble_dic})

    return groups

###############################################################################
# All connected to provenance
###############################################################################
def write_data(cfg, diag_name, diagnostic):
    """Write netcdf file."""

    nc_paths = []

    if 'timeseries' in diag_name:
        bname_format = '{}_{}_{}_{}' # diag_name, tsk, ens, metric
        bname_format_raw = '{}_{}_{}_{}-raw' # diag_name, tsk, ens, metric
        metric_save_names = {'envelop': ['min', 'max'],
                        'mean_pm_std': ['mean+std', 'mean-std'],
                        'minmaxtrend': ['min-trend', 'max-trend']}
        ts_dicts = {k: v for k, v in diagnostic.items() if 'timeseries' in k}
        for tsk, ts in ts_dicts.items():
            logger.info("Writing data for timeseries: {}".format(tsk))

            for ens, ens_data in zip(ts['ensembles'], ts['data']):
                if ens == 'external': # continue
                    if len(ts['ensembles']) == 1:
                        cube = ens_data[0][0]
                        ds = ts['labels'][ts['ensembles'].index('external')][0]
                        basename = bname_format.format(diag_name, tsk, ens,
                                                    metric) + '_' + ds
                        path = get_diagnostic_filename(basename, cfg)
                        io.iris_save(cube, path)
                        nc_paths.append(path)
                else:
                    for metric, metric_data in zip(ts['metrics'], ens_data):
                        if metric == 'single':
                            if f'datasets_{ens}' in ts.keys():
                                if len(set(ts[f'datasets_{ens}'])) != \
                                    len(ts[f'datasets_{ens}']):
                                    for ds, cube in zip(ts[f'datasets_{ens}'], metric_data):
                                        if 'realization' in cube.attributes.keys():
                                            basename = bname_format.format(diag_name, tsk, ens,
                                                                        metric) + '_' + ds + '_' + str(cube.attributes['realization'])#
                                            path = get_diagnostic_filename(basename, cfg)
                                            io.iris_save(cube, path)
                                            nc_paths.append(path)
                                        else:
                                            raise KeyError
                                else:
                                    for ds, cube in zip(ts[f'datasets_{ens}'], metric_data):
                                        basename = bname_format.format(diag_name, tsk, ens,
                                                                    metric) + '_' + ds
                                        path = get_diagnostic_filename(basename, cfg)
                                        io.iris_save(cube, path)
                                        nc_paths.append(path)
                            else:
                                for ds, cube in zip(ts['datasets'], metric_data):
                                    basename = bname_format.format(diag_name, tsk, ens,
                                                                metric) + '_' + ds
                                    path = get_diagnostic_filename(basename, cfg)
                                    io.iris_save(cube, path)
                                    nc_paths.append(path)
                            if ts['indicate_bars']:
                                if np.array(ts['data_raw']).shape == (1,1,1):
                                    cube = ts['data_raw'][0][0][0]
                                    basename = bname_format_raw.format(diag_name, tsk, ens,
                                                                metric) + '_' + ts[f'datasets_{ens}'][0]
                                    path = get_diagnostic_filename(basename, cfg)
                                    io.iris_save(cube, path)
                                    nc_paths.append(path)

                        elif len(metric_data) == 1:
                            cube = metric_data[0]
                            basename = bname_format.format(diag_name, tsk, ens,
                                                        metric)
                            path = get_diagnostic_filename(basename, cfg)
                            io.iris_save(cube, path)
                            nc_paths.append(path)
                        elif metric in metric_save_names.keys():
                            for (metr, cube) in zip(metric_save_names[metric],
                                                    metric_data):
                                basename = bname_format.format(diag_name, tsk, ens,
                                                            metr)
                                path = get_diagnostic_filename(basename, cfg)
                                io.iris_save(cube, path)
                                nc_paths.append(path)
                        else:
                            logger.error(f"Metric {metric} not implemented")
        bx_dicts = {k: v for k, v in diagnostic.items() if 'boxes' in k}
        if bx_dicts:
            bname_format = '{}_{}.csv'
            for bxk, bx in bx_dicts.items():
                logger.info("Writing data for boxplot in timeseries: {}".format(bxk))
                basename = bname_format.format(diag_name, bxk)
                path = get_diagnostic_filename(basename, cfg)[:-3]
                df = bx['data']
                df.to_csv(path, index=False)
    elif 'histogram' in diag_name:
        bname_format = '{}_{}_{}' # diag_name, ens, ds_name
        model_format_cmip6 = '{institution_id}_{source_id}_'\
                             '{parent_variant_label}'
        model_format_miroc6 = '{institution_id}_{source_id}_'\
                              '{variant_label}'
        model_format_cmip5 = '{institute_id}_{model_id}_'\
                             '{parent_experiment_rip}'
        model_format_mpige = '{institute_id}_{model_id}_'\
                             'r{realization}'
        model_format_csiro = '{institute_id}_{model_id}_'\
                             'r{realization}'
        model_format_cordex = '{institute_id}_{model_id}_{driving_model_id}_'\
                              '{experiment_id}'
        obs_format = '{}_{}' #title,version
        mean_format = '{}_{}_mean' # diag_name, ens, ds_name
        hist_dicts = {k: v for k, v in diagnostic.items() if 'histogram' in k}
        for histk, hist in hist_dicts.items():
            for eind, ens in enumerate(hist['ensembles']):
                if histk == 'histogram_means':
                    cube = hist['trends'][eind]
                    basename = mean_format.format(diag_name, ens)
                    path = get_diagnostic_filename(basename, cfg)
                    io.iris_save(cube, path)
                    nc_paths.append(path)
                else:
                    for cube in hist['trends'][eind]:
                        if 'mip_era' in cube.attributes:
                            if 'CMIP6' in cube.attributes['mip_era']:
                                runs = cube.attributes['parent_variant_label']
                                if len(runs) > 20:
                                    cube.attributes['parent_variant_label'] = \
                                        ';'.join(list(set(runs.split(';'))))
                                inst = cube.attributes['institution_id']
                                if len(inst) > 20:
                                    cube.attributes['institution_id'] = \
                                        ';'.join(list(set(inst.split(';'))))
                                if 'miroc6' in ens.lower():
                                    ds_name = model_format_miroc6.format(
                                                **cube.attributes)
                                else:
                                    ds_name = model_format_cmip6.format(
                                                **cube.attributes)
                        elif 'project_id' in cube.attributes:
                            if 'obs' in cube.attributes['project_id'].lower():
                                ds_name = obs_format.format(
                                    cube.attributes['title'].split(' ')[0],
                                    cube.attributes['version'])
                            elif 'create-ip' in cube.attributes['project_id'].lower():
                                ds_name = obs_format.format(
                                    cube.attributes['title'].split(' ')[0],
                                    cube.attributes['model_id'])
                            elif 'cmip' in cube.attributes['project_id'].lower():
                                try:
                                    runs = cube.attributes['parent_experiment_rip']
                                    if len(runs) > 20:
                                        cube.attributes['parent_experiment_rip'] = \
                                            ';'.join(list(set(runs.split(';'))))
                                    if '/' in \
                                        cube.attributes['parent_experiment_rip']:
                                        cube.attributes['parent_experiment_rip'] = \
                                            cube.attributes['parent_experiment_rip'].replace('/', '-')
                                    # mpige
                                    if cube.attributes.get('references') == \
                                        'Maher et.al. 2018':
                                        ds_name = model_format_mpige.format(
                                            **cube.attributes)
                                    elif cube.attributes.get('model_id') == \
                                        'CSIRO-Mk3-6-0' and \
                                        'realization' in cube.attributes.keys():
                                        ds_name = model_format_csiro.format(
                                            **cube.attributes)
                                    else:
                                        ds_name = model_format_cmip5.format(
                                            **cube.attributes)
                                except KeyError:
                                    if cube.attributes['institute_id'] == 'NIMR-KMA' and  \
                                        cube.attributes['model_id'] == 'HadGEM2-AO':
                                        cube.attributes.update({'parent_experiment_rip': 'r1i1p1'})
                                        ds_name = model_format_cmip5.format(
                                            **cube.attributes)
                                    elif cube.attributes['institute_id'] == 'INM' and  \
                                        cube.attributes['model_id'] == 'inmcm4':
                                        cube.attributes.update({'parent_experiment_rip': 'r1i1p1'})
                                        ds_name = model_format_cmip5.format(
                                            **cube.attributes)
                                    else:
                                        logger.error(f"Unable to save cube {cube.attributes}")
                                        raise KeyError
                            elif 'cordex' in cube.attributes['project_id'].lower():
                                exp = cube.attributes['experiment_id']
                                if len(exp) > 20:
                                    cube.attributes['experiment_id'] = \
                                        ';'.join(list(set(exp.split(';'))))
                                    if '/' in cube.attributes['experiment_id']:
                                        cube.attributes['experiment_id'] = \
                                            cube.attributes['experiment_id'].replace('/', '-')
                                ds_name = model_format_cordex.format(
                                        **cube.attributes)
                        elif 'd4pdf' in cube.attributes['title']:
                            ds_name = f"d4pdf_member{cube.attributes['title'][-3:]}"
                        ds_name = ds_name.replace(';', '-')
                        basename = bname_format.format(diag_name, ens, ds_name)
                        path = get_diagnostic_filename(basename, cfg)
                        io.iris_save(cube, path)
                        nc_paths.append(path)
    elif 'mapplot' in diag_name:
        if diagnostic['type'] == 'MultiModelMean':
            label = diagnostic['ensembles'][0]
            bname_format = '{}_{}_{}_{}'  # diag_name, ens/label, type, metric
        else:
            label = '' # included in diag_name
            bname_format = '{}_{}{}_{}'  # diag_name, ens/label, type, metric
        if 'cube' in diagnostic.keys():
            cube = diagnostic['cube']
            basename = bname_format.format(diag_name, label,
                                            diagnostic['type'],
                                            diagnostic['metric'])
            path = get_diagnostic_filename(basename, cfg)
            io.iris_save(cube, path)
            nc_paths.append(path)
    elif 'boxplot' in diag_name:
        basename = f'{diag_name}.csv'
        path = get_diagnostic_filename(basename, cfg)[:-3]
        df = diagnostic['data']
        df.to_csv(path, index=False)
        nc_paths.append(path)
    elif 'gwlrwl' in diag_name:
        bname_format = '{}.csv'
        basename = bname_format.format(diag_name)
        path = get_diagnostic_filename(basename, cfg)[:-3]

        columns = ['diag_name', 'label', 'metric', 'coord_name', 'coord', 'data']
        dfs = []

        for labels, metrics, ens_cubes in zip(diagnostic['labels'],
                diagnostic['metrics'], diagnostic['data']):
            for stat_cubes, label in zip(ens_cubes, labels):
                for metric, cubes in zip(metrics, stat_cubes):
                    if metric in ['mean', 'median', 'std', 'min', 'max']:
                        # IPython.embed(config=c)
                        coord_name = cubes.dim_coords[0].standard_name
                        coord = cubes.coord(coord_name).points
                        data = copy.deepcopy(cubes.data.data)
                        data[cubes.data.mask == True] = np.nan
                        df_tmp = pd.DataFrame([[diag_name, label.replace(' ', '_'),
                                               metric, coord_name, coord, data]],
                                              columns=columns)
                        dfs.append(df_tmp)
                    elif metric == 'mean_pm_std':
                        for metric_add, cube in zip(['mean_p_std', 'mean_m_std'],
                                                    cubes):
                            coord_name = cube.dim_coords[0].standard_name
                            coord = cube.coord(coord_name).points
                            data = copy.deepcopy(cube.data.data)
                            data[cube.data.mask == True] = np.nan
                            df_tmp = pd.DataFrame([[diag_name, label.replace(' ', '_'),
                                                metric_add, coord_name, coord, data]],
                                                columns=columns)
                            dfs.append(df_tmp)
                    else:
                        raise NotImplementedError
        df = pd.concat(dfs)
        df.to_csv(path, index=False)
        nc_paths.append(path)
    return nc_paths


def get_provenance_record(simplified_diag, projects, ancestor_files,
                          obsnames=None, nc_paths=None, plot_paths=None):
    """Create a provenance record describing the diagnostic data and plot."""
    if obsnames:
        obsnames = 'and observations: ' + obsnames

    record = {
        'caption':
        ('{} of {} ensembles {}.'.format(simplified_diag['type'], projects,
                                         obsnames)),
        'authors': ['jury_martin'],
        'references': ['IPCC AR6 CH10'],
        'realms': ['atmos'],
        'themes': ['phys'],
        'ancestors':
        ancestor_files,
    }
    for k, v in simplified_diag.items():
        record.update({k: v})
    if nc_paths:
        record.update({'nc_paths': nc_paths})
    if plot_paths:
        record.update({'plot_paths': plot_paths})

    return record


def get_diagnostic_for_provenance(diag_name, diagnostic):
    """ Get simplified form of diagnostic information.

    General info from keys, specified info from subkeys."""
    if 'timeseries' in diag_name:
        keys = ['anomalies', 'period_norm', 'relative']
        optionals = None
        key = 'timeseries'
        sub_keys = ['ensembles', 'metrics', 'period']
    elif 'histogram' in diag_name:
        keys = ['period_trend', 'trend_base', 'relative', 'period_norm']
        optionals = None
        key = 'histogram'
        sub_keys = ['ensembles', 'metric']
    elif 'mapplot' in diag_name:
        keys = ['type', 'metric']
        optionals = ['region', 'period_norm', 'period_plot', 'period_trend',
                     'region_metric', 'anomalies', 'relative']
        key = 'mapplot'
        sub_keys = []
    elif 'gwlrwl' in diag_name:
        keys = ['anomalies', 'period_norm', 'relative', 'ensembles', 'metrics']
        optionals = None
        key = 'gwlrwl'
        sub_keys = []
    elif 'boxplot' in diag_name:
        keys = ['ensembles']
        optionals = None
        key = 'boxplot'
        sub_keys = []
    else:
        return

    simple_diag = {}
    simple_diag.update({'type': key})
    for k in keys:
        try:
            simple_diag.update({k: diagnostic[k]})
        except KeyError:
            pass

    if optionals:
        for k in optionals:
            try:
                simple_diag.update({k: diagnostic[k]})
            except KeyError:
                pass

    for sub_diag_key, sub_diag in diagnostic.items():
        if key in sub_diag_key:
            try:
                simple_diag.update({
                     sub_diag_key: {key: sub_diag[key] for key in sub_keys}})
            except KeyError:
                pass

    return simple_diag


def get_ancestors(diag_name, diagnostic, ensembles):
    """Returns ancestor_files, projects, obsnames for IPCCAR6CH10 diagnostics.

    Parameters
    ----------
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type timeseries diagnostic.
    ensembles: :obj:`dict` of :obj:`list` of :obj:`dict`
        Holding ensemble sorted cfg['input_data']

    Returns
    -------
    ancestor_files: list
        of used input filenames
    poject: str
        joined projects by ', '
    obsnames: str
        joined obsnames by ', '
    """
    if 'timeseries' in diag_name:
        key = 'timeseries'
        dens = [v['ensembles'] for k, v in diagnostic.items() if key in k]
        dens = [item for sublist in dens for item in sublist]
        dens = list(set(dens))
    elif 'histogram' in diag_name:
        keys = ['histogram_histogram', 'histogram_symbols']
        dens = [v['ensembles'] for k, v in diagnostic.items() if k in keys]
        dens = [item for sublist in dens for item in sublist]
    elif 'mapplot' in diag_name:
        dens = [ens for ens in diagnostic['ensembles']]
    elif 'gwlrwl' in diag_name:
        dens = [i for sl in diagnostic['ensembles'] for it in sl for i in it]
    elif 'boxplot' in diag_name:
        dens = [ens for ens in diagnostic['ensembles']]
    else:
        return

    ancestor_files = []
    for ens in dens:
        if ens == 'external':
            try:
                ancestor_files.append(diagnostic['filename'])
            except KeyError:
                pass
        elif ens != 'dummy':
            ancestor_files.extend([d['filename'] for d in ensembles[ens]])

    projects = []
    for ens in dens:
        if ens not in ['dummy', 'external']:
            projects.extend([d['project'] for d in ensembles[ens]])
    projects = [p for p in projects if 'obs' not in p.lower()]
    projects = list(set(projects))
    projects = ', '.join(projects)

    dens = [d for d in dens if 'obs' in d.lower()]
    obs_format = '{} {}'
    obs = []
    for ens in dens:
        for d in ensembles[ens]:
            obs.append(obs_format.format(d['dataset'], d['version']))
    obs = ', '.join(obs)

    return ancestor_files, projects, obs


###############################################################################
# All connected to timeseries plotting
###############################################################################
def fill_timeseries_ax(ax, diag_name, diagnostic):
    """Check diagnostic timeseries plotting information.

    Loops over singe timeseries dicts and plots them.
    Finish up plot.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type timeseries diagnostic.
    """
    logger.info("Checking integrity of diagnostic plotting "\
                "information: {}".format(diag_name))
    prepare_timeseries_plot(diagnostic)
    pi = diagnostic['plotting_information']

    logger.info("Doing plot for diagnostic: {}".format(diag_name))
    ts_dicts = {k: v for k, v in diagnostic.items() if 'timeseries' in k}
    if 'order' in diagnostic.keys():
        ts_dicts = {k: ts_dicts[k] for k in diagnostic['order']}

    if pi['split_x_axis']:
        ax2 = ax.twinx()
        ax.set_zorder(10)
        ax.patch.set_visible(False)

        xlims1 = [pi['xlims'][0], pi['split_year'] - 0.5]
        xlims2 = [pi['split_year'] + 0.5, pi['xlims'][1]]
        pi.update({'xlims1': xlims1,
                   'xlims2': xlims2})
        ts_tmps = []
        # timeseries
        for ts in ts_dicts.values():
            for axx, xlims in zip([ax, ax2], [xlims1, xlims2]):
                if ts['period']['end_year'] < xlims[0]:
                    continue
                if ts['period']['start_year'] > xlims[1]:
                    continue
                ts_tmp = split_timeseries_data(ts, xlims)
                plot_timeseries_ax(axx, pi, ts_tmp)
                ts_tmps.append(ts_tmp)
        # boxes
        boxes_dicts = {k: v for k, v in diagnostic.items() if 'boxes' in k }
        if boxes_dicts:
            ax3 = ax2.twiny()

            # merge the available data frames
            dfs = [bx['data'] for bxk, bx in boxes_dicts.items()]
            df = pd.concat(dfs)
            df = df.reset_index(drop = True)

            # get the ratios
            width_pb = (pi['xlims2'][1] - pi['xlims1'][0]) / 30
            ax3_xticks = [pi['xlims2'][1] + width_pb + ii*width_pb
                          for ii in range(len(df))]

            pi.update({'split_xlims': [xlims2[-1],
                                       ax3_xticks[-1] + width_pb*0.85]})

            box = ax3.boxplot(df['data'], widths=[width_pb * 0.7]*len(df),
                              positions=ax3_xticks,
                              labels=df['label'], zorder=5)
            color_boxes(ax3, box, df)
            pi.update({'ax2_labels': df['label']})
            finish_timeseries_ax_splitxy(ax, ax2, ax3, pi)
        else:
            finish_timeseries_ax_splitx(ax, ax2, pi)

    else:
        if ts_dicts:
            for ts in ts_dicts.values():
                plot_timeseries_ax(ax, pi, ts)
        boxes_dicts = {k: v for k, v in diagnostic.items() if 'boxes' in k }
        if boxes_dicts:
            # merge the available data frames
            dfs = [bx['data'] for bxk, bx in boxes_dicts.items()]
            df = pd.concat(dfs)
            df = df.reset_index(drop = True)

            if ts_dicts:
                ax2 = ax.twiny()
                # get the ratios
                if not pi['xlims']:
                    pi['xlims'] = ax.get_xlim()
                width_pb = np.diff(pi['xlims'])[0] / 30
                ax2_xticks = pi['xlims'][1] + width_pb + \
                            [ii*width_pb for ii in range(len(df))]

                pi.update({'split_xlims': [pi['xlims'][-1],
                                        ax2_xticks[-1] + width_pb*0.85]})
            else:
                ax2 = ax
                width_pb = np.float64(1.)
                ax2_xticks = np.float64(0.5) + [ii*width_pb for ii in range(len(df))]

            box = ax2.boxplot(df['data'], widths=[width_pb * 0.7]*len(df),
                              positions=ax2_xticks,
                              labels=df['label'], zorder=5)
            color_boxes(ax2, box, df)

            pi.update({'ax2_labels': df['label']})
            finish_timeseries_ax_splity(ax, ax2, pi)
        else:
            finish_timeseries_ax(ax, pi)

    return ax


def split_timeseries_data(ts, xlims):
    """Subselects data based on provided xlims.

    Used to split data if x_axis split is selected."""
    ts_tmp = ts.copy()

    new_data = []
    for cubess in ts_tmp['data']:
        new_cubess = []
        for cubes in cubess:
            new_cube = []
            for cube in cubes:
                try:
                    new_cube.append(extract_time(cube, xlims[0], 1, 1, xlims[1], 12, 31))
                except ValueError:
                    pass
            new_cubess.append(new_cube)
        new_data.append(new_cubess)
    ts_tmp['data'] = new_data

    return ts_tmp


def prepare_timeseries_plot(diagnostic):
    """Ensure valid plotting_information.

    Adds defaults if nothing is specified, possibly adds y_label (=cube.units).

    Parameters
    ----------
    diagnostic: :obj:`dict`
        Of type timeseries diagnostic.

    Updates
    ----------
    diagnostic['plotting_information'] :obj:`dict`
    """
    ylabel_format_def = '({})'
    defaults = {'add_legend': True,
                'auto_indper_ylims': True,
                'xlabel': '',
                'ylabel': None,
                'ncol': 2,
                'loc': 2,
                'zero_line': True,
                'ylims': None,
                'mean_pm_std_alpha': 0.2,
                'minmax_ls': ['-','-'],
                'split_x_axis': False,
                'dgrey': [0.4,0.4,0.4],
                'mgrey': [0.65,0.65,0.65],
                'lgrey': [0.9,0.9,0.9],
                'grid_lines': False,
                'add_shading': True,
                'title': '',
                'xlims': None,
                'force_ytick_integers': False,
                }

    pi = diagnostic['plotting_information']

    # check if period plot or xlims is present
    if 'period_plot' in pi and 'xlims' in pi:
        logger.warning("Both period_plot and xlims defined, using xlims.")
        del pi['period_plot']
    elif 'period_plot' in pi and not 'xlims' in pi:
        pi['xlims'] = (pi['period_plot']['start_year']-0.5,
                       pi['period_plot']['end_year']+0.5)

    for kd, default in  defaults.items():
        if kd not in pi:
            pi.update({kd: default})

    # if ylabel is None extract information from a cube
    if pi['ylabel'] == None:
        tsk = [k for k in diagnostic.keys() if 'timeseries' in k]
        if tsk:
            tsk = tsk[0]
            unit = str(diagnostic[tsk]['data'][0][0][0].units)
            if unit == 'degC':
                unit = '°C'
            if '-1' in unit:
                unit.replace('-1','$^{-1}$')
            pi['ylabel'] = ylabel_format_def.format(unit)
            logger.warning("No ylabel defined, using cube units: "\
                        "{}".format(pi['ylabel']))


def plot_timeseries_ax(ax, pi, ts):
    """Plot data in a single timeseries dict.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    pi: :obj:`dict`
        Plotting information for the plot.
    ts: :obj:`dict`
        Of type timeseries dict.
    """
    for labels, cubess, ens in zip(ts['labels'], ts['data'], ts['ensembles']):
        for metric, label, cubes in zip(ts['metrics'], labels, cubess):
            if metric in ['mean', 'median', 'std', 'min', 'max'] or \
                'trendline' in metric:
                ax.plot(cubes[0].coord('year').points, cubes[0].data,
                         color=label_colors(label), label=label, zorder=10)
            elif metric == 'single':
                if np.any(['datasets' in k for k in ts.keys()]):
                    label = ts[f'datasets_{ens}']
                for lab, cube in zip(label, cubes):
                    if isinstance(label, str):
                        ax.plot(cube.coord('year').points, cube.data,
                                 color=label_colors(label), label=label,
                                 zorder=10)
                    else:
                        ax.plot(cube.coord('year').points, cube.data,
                                 color=label_colors(lab), label=lab,
                                 zorder=10)
            elif metric in ['minmaxtrend', 'minmaxtrend-lines']:
                if 'precipitation' in cubes[0].standard_name:
                    minmax_add = ['Driest','Wettest']
                elif 'temperature' in cubes[0].standard_name:
                    minmax_add = ['Coldest','Warmest']
                else:
                    minmax_add = ['MIN','MAX']
                for add, cube, ls in zip(minmax_add, cubes,
                                         pi['minmax_ls']):
                    ax.plot(cube.coord('year').points, cube.data,
                                color=label_colors(label+' '+add),
                                label=label+' '+add, ls=ls, zorder=15)
            elif metric in ['mean_pm_std', 'envelop']:
                ax.fill_between(cubes[0].coord('year').points,
                                cubes[0].data, cubes[1].data,
                                color=label_colors(label),
                                alpha=pi[metric+'_alpha'],
                                label=label,
                                zorder=5)
            else:
                logger.error("Plotting of metric {} not implemented "\
                                "just yet ..".format(metric))
                raise NotImplementedError

    if ts['indicate_bars']:
        if np.array(ts['data_raw']).shape == (1,1,1):
            cube = ts['data_raw'][0][0][0]
            # plot the bars
            plt.bar(cube.coord('year').points, cube.data, width=1.,
                    color=pi['mgrey'], zorder=5)
        else:
            logger.warning("Indicating bars only possible for "\
                            "timeseries holding a single cube. Got "\
                            "{}".format(np.array(ts['data_raw']).shape))


def finish_timeseries_ax_splitxy(ax, ax2, ax3, pi):
    """Finish up the timeseries splitxy plot.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    ax2: matplotlib.axes
        Matplotlib ax instance.
    pi: :obj:`dict`
        Plotting information for the plot.
    """
    ax3.tick_params(axis='x', which='both', top=False, labeltop=False)
    ax3.spines['left'].set_visible(True)
    ax3.spines['left'].set_position(('data', pi['xlims'][1]))
    ax3.spines['right'].set_visible(False)

    ax2.spines['right'].set_visible(True)
    ax2.tick_params(axis='y', which='both', right=True, labelright=True)
    ax2.spines['right'].set_position(('data', pi['split_xlims'][1]+0.5))

    if pi['xlims']:
        ticks = ax.get_xticks()
        ticks = [tick for tick in ticks if tick <= pi['xlims'][1]]
        ax.set_xticks(ticks)

        ticks = ax.get_xticks(minor=True)
        ticks = [tick for tick in ticks if tick <= pi['xlims'][1]]
        ax.set_xticks(ticks, minor=True)

        ax.set_xlim(pi['xlims'][0]-0.5,
                    pi['split_xlims'][1]+0.5)
        ax3.set_xlim(pi['xlims'][0]-0.5,
                     pi['split_xlims'][1]+0.5)
    if pi['ylims']:
        ax.set_ylim(pi['ylims'])
        ax2.set_ylim(pi['split_ylims'])
        ax3.set_ylim(pi['split_ylims'])
    else:
        ax3.set_ylim(ax.get_ylim())

    if pi['force_ytick_integers']:
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax2.yaxis.get_major_locator().set_params(integer=True)
        ax3.yaxis.get_major_locator().set_params(integer=True)

    if pi['xlabel']:
        ax.set_xlabel(pi['xlabel'])
    if pi['ylabel']:
        ax.set_ylabel(format_units(pi['ylabel']))
        ax2.set_ylabel(format_units(pi['ylabel']))

    if pi['title']:
        ax.set_title(pi['title'])

    if pi['zero_line']:
        ax.plot(pi['xlims1'], [0, 0], zorder=1, **zero_line)
        ax2.plot(pi['xlims2'], [0, 0], zorder=6, **zero_line)
    if pi['grid_lines']:
        for axx in [ax, ax2]:
            axx.grid(axis='y', color=pi['dgrey'], linewidth=.5, zorder=6)

    if pi['add_legend']:
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles.extend(handles2)
        labels.extend(labels2)
        if 'ax2_labels' in pi:
            labelpatches = []
            for label in pi['ax2_labels']:
                labelpatches.append((mlines.Line2D([], [],
                                    color=label_colors(label),
                                    label=label, marker="s", fillstyle='full',
                                    linestyle="-")))
            handles.extend(labelpatches)
            labels.extend(pi['ax2_labels'])
        by_label = OrderedDict(zip(labels, handles))
        by_label = translate_obs_naming(by_label)
        ax.legend(by_label.values(), by_label.keys(),
                  ncol=pi['ncol'], loc=pi['loc'])

    ind_pers = {k: v for k, v in pi.items() if 'indicate_period' in k}
    if any(ind_pers):
        plot_indication_periods(ind_pers, ax, pi, ax2=ax2)

    permid = pi['split_year'] - 0.5
    ax2.plot([permid, permid], ax2.get_ylim(), color='k', lw=1., zorder=2)


def finish_timeseries_ax_splity(ax, ax2, pi):
    """Finish up the timeseries splity plot.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    ax2: matplotlib.axes
        Matplotlib ax instance.
    pi: :obj:`dict`
        Plotting information for the plot.
    """
    if ax2 != ax:
        ax2.tick_params(axis='x', which='both', top=False, labeltop=False)
        ax2.spines['left'].set_visible(True)
        ax2.spines['left'].set_position(('data', pi['xlims'][1]))
        ax2.spines['right'].set_visible(False)

        ax.spines['right'].set_visible(True)
        ax.tick_params(axis='y', which='both', right=True, labelright=True)
        ax.spines['right'].set_position(('data', pi['split_xlims'][1]+0.5))
    else:
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', which='both', length=0,
                       top=False, labeltop=False)
        ax.tick_params(axis='y', which='both', right=True, labelright=True,
                       left=False, labelleft=False)

        ax.set_xticklabels([])
        ax.yaxis.set_label_position("right")

    if pi['xlims']:
        ticks = ax.get_xticks()
        ticks = [tick for tick in ticks if tick <= pi['xlims'][1]]
        ax.set_xticks(ticks)

        ticks = ax.get_xticks(minor=True)
        ticks = [tick for tick in ticks if tick <= pi['xlims'][1]]
        ax.set_xticks(ticks, minor=True)

        ax.set_xlim(pi['xlims'][0]-0.5,
                    pi['split_xlims'][1]+0.5)
        ax2.set_xlim(pi['xlims'][0]-0.5,
                     pi['split_xlims'][1]+0.5)
    if pi['ylims']:
        ax.set_ylim(pi['ylims'])
        ax2.set_ylim(pi['ylims'])
    else:
        ax2.set_ylim(ax.get_ylim())

    if pi['force_ytick_integers']:
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax2.yaxis.get_major_locator().set_params(integer=True)

    if pi['xlabel']:
        ax.set_xlabel(pi['xlabel'])
    if pi['ylabel']:
        ax.set_ylabel(format_units(pi['ylabel']))

    if pi['title']:
        ax.set_title(pi['title'])

    if pi['zero_line']:
        if pi['xlims']: xlims = pi['xlims']
        else: xlims = ax.get_xlim()
        ax.plot(xlims, [0, 0], zorder=1, **zero_line)
    if pi['grid_lines']:
        ax.grid(axis='y', color=pi['dgrey'], linewidth=.5, zorder=1)

    if pi['add_legend']:
        handles, labels = ax.get_legend_handles_labels()
        if 'ax2_labels' in pi:
            labelpatches = []
            for label in pi['ax2_labels']:
                labelpatches.append((mlines.Line2D([], [],
                                    color=label_colors(label),
                                    label=label, marker="s", fillstyle='full',
                                    linestyle="-")))
            handles.extend(labelpatches)
            labels.extend(pi['ax2_labels'])
        by_label = OrderedDict(zip(labels, handles))
        by_label = translate_obs_naming(by_label)
        ax.legend(by_label.values(), by_label.keys(),
                  ncol=pi['ncol'], loc=pi['loc'])

    ind_pers = {k: v for k, v in pi.items() if 'indicate_period' in k}
    if any(ind_pers):
        plot_indication_periods(ind_pers, ax, pi)


def finish_timeseries_ax_splitx(ax, ax2, pi):
    """Finish up the timeseries splitx plot.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    ax2: matplotlib.axes
        Matplotlib ax instance.
    pi: :obj:`dict`
        Plotting information for the plot.
    """
    if pi['xlims']:
        ax.set_xlim(pi['xlims'][0]-0.5,
                    pi['xlims'][1]+0.5)
    if pi['ylims']:
        ax.set_ylim(pi['ylims'])
        ax2.set_ylim(pi['split_ylims'])

    if pi['force_ytick_integers']:
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax2.yaxis.get_major_locator().set_params(integer=True)

    if pi['xlabel']:
        ax.set_xlabel(pi['xlabel'])
    if pi['ylabel']:
        ax.set_ylabel(format_units(pi['ylabel']))
        ax2.set_ylabel(format_units(pi['ylabel']))

    if pi['title']:
        ax.set_title(pi['title'])

    permid = pi['split_year'] - 0.5
    ax2.yaxis.set_label_position("right")
    ax2.spines['right'].set_visible(True)
    if pi['zero_line']:
        ax.plot(pi['xlims1'], [0, 0], zorder=6, **zero_line)
        ax2.plot(pi['xlims2'], [0, 0], zorder=6, **zero_line)

    if pi['grid_lines']:
        for axx in [ax, ax2]:
            axx.grid(axis='y', color=pi['dgrey'], linewidth=.5, zorder=6)
    if pi['add_legend']:
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles.extend(handles2)
        labels.extend(labels2)
        by_label = OrderedDict(zip(labels, handles))
        by_label = translate_obs_naming(by_label)
        ax.legend(by_label.values(), by_label.keys(),
                  ncol=pi['ncol'], loc=pi['loc'])

    ind_pers = {k: v for k, v in pi.items() if 'indicate_period' in k}
    if any(ind_pers):
        plot_indication_periods(ind_pers, ax, pi, ax2 = ax2)

    ax2.plot([permid, permid], ax2.get_ylim(), color='k', lw=1., zorder=2)



def finish_timeseries_ax(ax, pi):
    """Finish up the timeseries plot.

    Parameters
    ----------
    ax: matplotlib.axes
        Matplotlib ax instance.
    pi: :obj:`dict`
        Plotting information for the plot.
    """
    if pi['xlims']:
        ax.set_xlim(pi['xlims'][0]-0.5,
                    pi['xlims'][1]+0.5)
    if pi['ylims']:
        ax.set_ylim(pi['ylims'])

    if pi['force_ytick_integers']:
        ax.yaxis.get_major_locator().set_params(integer=True)

    if pi['xlabel']:
        ax.set_xlabel(pi['xlabel'])
    if pi['ylabel']:
        ax.set_ylabel(format_units(pi['ylabel']))

    if pi['title']:
        ax.set_title(pi['title'])

    if pi['zero_line']:
        ax.plot(ax.get_xlim(), [0, 0], zorder=1, **zero_line)
    if pi['grid_lines']:
        ax.grid(axis='y', color=pi['dgrey'], linewidth=.5, zorder=1)
    if pi['add_legend']:
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        by_label = translate_obs_naming(by_label)
        ax.legend(by_label.values(), by_label.keys(),
                    ncol=pi['ncol'], loc=pi['loc'])

    ind_pers = {k: v for k, v in pi.items() if 'indicate_period' in k}
    if any(ind_pers):
        plot_indication_periods(ind_pers, ax, pi)


def plot_indication_periods(ind_pers, ax, pi, ax2=None):
    """Draw (|---|) indicated periods as given in ind_pers.

    Parameters
    ----------
    ind_pers: :obj:`dict`
        Holing information on start_year and end_year, and optional name
    ax: matplotlib.axes
        Matplotlib ax instance.
    pi: :obj:`dict`
        Plotting information for the plot.
    """
    color = pi['dgrey']

    ylims = ax.get_ylim()
    if pi['auto_indper_ylims']:
        ylims = (ylims[1] - (ylims[1] - ylims[0]) * 1.12, ylims[1])
        ax.set_ylim(ylims)
        if pi['split_x_axis']:
            ylims = ax2.get_ylim()
            ylims = (ylims[1] - (ylims[1] - ylims[0]) * 1.12, ylims[1])
            ax2.set_ylim(ylims)
            axx = ax2
        else:
            axx = ax
    else:
        if pi['split_x_axis']:
            axx = ax2
        else:
            axx = ax

    # adapt ylims to have space
    yincr = (ylims[1] - ylims[0]) / 50.
    y = (ylims[1] - ylims[0]) / 10. + ylims[0]
    ys = [ylims[0], ylims[0], ylims[1], ylims[1], ylims[0]]

    for ind_per in ind_pers.values():
        xn, xx = ind_per['start_year'] - 0.5, ind_per['end_year'] # + 0.5
        xm = (xn + xx) / 2
        if 'add_shading' in pi:
            if pi['add_shading']:
                xs = [xn, xx, xx, xn, xn]
                poly = []
                for xp,yp in zip(xs,ys):
                    poly.append((xp,yp))
                poly = Polygon(poly, facecolor=pi['lgrey'],
                               edgecolor=pi['lgrey'], zorder=1)
                axx.add_patch(poly)

        # plot the singe lines of |---|
        axx.plot([xn, xx], [y, y], c=color, linewidth=2, zorder=10,
                 clip_on=False)
        axx.plot([xn, xn], [y-yincr, y+yincr], c=color, linewidth=2, zorder=10,
                    clip_on=False)
        axx.plot([xx, xx], [y-yincr, y+yincr], c=color, linewidth=2, zorder=10,
                    clip_on=False)
        if 'name' in ind_per:
            text = ind_per['name']
            axx.text(xm, y-yincr, text, color=color, size='x-large',
                    ha='center', va='top', wrap=False, zorder=10)


###############################################################################
# All connected to coloring, formatting of labels
###############################################################################
def translate_obs_naming(labels):
    """Updates bad obs labels."""
    old_2_new = {'CRU': 'CRU TS',
                 'BerkeleyEarth': 'Berkeley Earth',
                 'NCDC': 'NCEI'}

    nlabels = OrderedDict()
    for k,v in labels.items():
        if k in old_2_new.keys():
            nlabels.update({old_2_new[k]: v})
        else:
            nlabels.update({k: v})

    return nlabels


def label_colors(label):
    """Get color for label. Largely predefined colors for the IPCC-AR6.

    Parameters
    ----------
    label: str
        Dataset label

    Returns
    -------
    color: str (color) or numpy.ndarray
    """
    dgrey = [0.4, 0.4, 0.4]
    mgrey = [0.65, 0.65, 0.65]
    lgrey = [0.9, 0.9, 0.9]

    colors = {# observations
              'OBS': 'k',
              'CRU': ('red-yellow_cat', 2),
              'CRU_unmasked': mgrey,
              'E-OBS': ('spectrum_cat', 5),
              'APHRO-MA': ('gree-blue_cat', 3),
              'JRA-55': [0.067, 0.467, 0.2], # ('gree-blue_cat', 5),
              # observations TAS
              'GISS': ('dark_cat', 7),
              'GISTEMP': ('dark_cat', 7),
              'NCDC': ('dark_cat', 5),
              'CRUTEM4': ('dark_cat', 6),
              'BerkeleyEarth': ('dark_cat', 4),
              'HadCRUT4': ('ssp_cat_1', 0),
              'HadCRUT5': ('ssp_cat_1', 0),
              'CowtanWay': ('dark_cat', 5),
              # observations PR
              'GPCC': ('dark_cat', 4),
              'GHCN': ('dark_cat', 6),
              'REGEN': ('dark_cat', 6),
              'IITM': ('ssp_cat_2', 5),

              # ensembles
              'CMIP5 historical-RCP8.5': ('cmip_cat', 1),
              'CMIP5 RCP8.5': ('cmip_cat', 1),
              'CMIP6 GHG': ('chem_cat', 3),
              'CMIP6 AER': ('chem_cat', 4),
              'DAMIP hist-GHG': ('chem_cat', 3),
              'DAMIP hist-aer': ('chem_cat', 4),
              'CMIP6 hist-GHG': ('chem_cat', 3),
              'CMIP6 hist-aer': ('chem_cat', 4),
              'HighResMIP hist-1950': ('spectrum_cat', 5),
              'HighResMIP highres-future': ('spectrum_cat', 5),
              'HighResMIP hist-1950-highres-future': ('spectrum_cat', 5),
              'HighResMIP highres-future-hist-1950': ('spectrum_cat', 5),
              'CORDEX': ('spectrum_cat', 2),
              'CORDEX historical-RCP8.5': ('spectrum_cat', 2),
              'CORDEX EUR-44': ('spectrum_cat', 2),
              'CORDEX EUR-44 historical-RCP8.5': ('spectrum_cat', 2),
              'CORDEX EUR-11': ('gree-blue_cat', 1),
              'CORDEX EUR-11 historical-RCP8.5': ('gree-blue_cat', 1),

              # SMILEs
              'MPIGE': dgrey,
              'MPI-ESM': dgrey,
              'MIROC6': dgrey,
              'CanESM2': dgrey,
              'CSIRO-Mk3-6-0': dgrey,
              'CESM1-CAM5': dgrey,
              'GFDL-ESM2M': dgrey,
              'GFDL-CM3': dgrey,
              'EC-EARTH': dgrey,
              'MRI-AGCM3.2H': dgrey,
              'd4PDF': dgrey,
              'd4PDF-GE': dgrey,
              'MPI-ESM historical-RCP8.5': dgrey,
              'MIROC6 historical-SSP5-8.5': dgrey,
              'CanESM2 historical_rcp85': dgrey,
              'CSIRO-Mk3-6-0 historical_rcp85': dgrey,
              'CESM1-CAM5 historical_rcp85': dgrey,
              'GFDL-ESM2M historical_rcp85': dgrey,
              'GFDL-CM3 historical_rcp85': dgrey,
              'EC-EARTH historical_rcp85': dgrey,

              # Scnearios
              'SSP1-2.6': ('ssp_cat_2', 1),
              'SSP2-4.5': ('ssp_cat_2', 2),
              'SSP3-7.0': ('ssp_cat_2', 3),
              'SSP5-8.5': ('ssp_cat_2', 8),
              'historical-SSP1-2.6': ('ssp_cat_2', 1),
              'historical-SSP2-4.5': ('ssp_cat_2', 2),
              'historical-SSP3-7.0': ('ssp_cat_2', 3),
              'historical-SSP5-8.5': ('ssp_cat_2', 8),
              'CMIP6 historical': ('cmip_cat', 0),
              'CMIP6 historical*': ('bright_cat', 0),
              'CMIP6 SSP1-2.6': ('ssp_cat_2', 1),
              'CMIP6 SSP2-4.5': ('ssp_cat_2', 2),
              'CMIP6 SSP3-7.0': ('ssp_cat_2', 3),
              'CMIP6 SSP5-8.5': ('ssp_cat_2', 8),
              'CMIP6 historical-SSP1-2.6': ('ssp_cat_2', 1),
              'CMIP6 historical-SSP2-4.5': ('ssp_cat_2', 2),
              'CMIP6 historical-SSP3-7.0': ('ssp_cat_2', 3),
              'CMIP6 historical-SSP5-8.5': ('ssp_cat_2', 8),

              'RCP2.6': ('rcp_cat', 3),
              'RCP4.5': ('rcp_cat', 2),
              'RCP6.0': ('rcp_cat', 1),
              'RCP8.5': ('rcp_cat', 0),
              'historical-RCP2.6': ('rcp_cat', 3),
              'historical-RCP4.5': ('rcp_cat', 2),
              'historical-RCP6.0': ('rcp_cat', 1),
              'historical-RCP8.5': ('rcp_cat', 0),
              'CMIP5 RCP2.6': ('rcp_cat', 3),
              'CMIP5 RCP4.5': ('rcp_cat', 2),
              'CMIP5 RCP6.0': ('rcp_cat', 1),
              'CMIP5 RCP8.5': ('rcp_cat', 0),
              'CMIP5 historical-RCP2.6': ('rcp_cat', 3),
              'CMIP5 historical-RCP4.5': ('rcp_cat', 2),
              'CMIP5 historical-RCP6.0': ('rcp_cat', 1),
              'CMIP5 historical-RCP8.5': ('rcp_cat', 0),

              # internal var figure
              'Trend': mgrey,

              # histogram
              'OBS': 'k',
              'OBS+REAN': 'k',
              'CMIP5': ('cmip_cat', 1),
              'CMIP6': ('cmip_cat', 0),
              'HighResMIP': ('spectrum_cat', 5),
              'EUR-44': ('spectrum_cat', 2),
              'EUR-11': ('gree-blue_cat', 1),

              # Urban Box
              'Choshi (Rural)': ('cmip_cat', 1),
              'Tokyo (Urban)': ('cmip_cat', 0),
              'city': dgrey,
              'country': dgrey,

              # timely
              'near term': ('rcp_cat', 2),
              'mid term': ('rcp_cat', 1),
              'long term': ('rcp_cat', 0),
              'near': ('rcp_cat', 2),
              'mid': ('rcp_cat', 1),
              'long': ('rcp_cat', 0),
    }

    if '(N=' in label:
        label = label.split(' (N=')[0]

    try:
        # special settings for particular Figures
        if 'SESA' in config['recipe'] and label in ['CRU']:
            sesa_colors = {'CRU': 'k'}
            return sesa_colors[label]
        elif 'NAM' in config['recipe'] and label in ['CRU']:
            NAM_colors = {'CRU': 'k'}
            return NAM_colors[label]
        elif 'Sahel' in config['recipe'] and \
            label in ['CRU', 'CMIP6 historical', 'CMIP5 historical-RCP8.5']:
            sahel_colors = {'CRU': 'k',
                            'CMIP6 historical': load_IPCCAR6_colors('cmip_cat')[0],
                            'CMIP5 historical-RCP8.5': load_IPCCAR6_colors('cmip_cat')[1],
                            'CMIP5 historical-RCP8.5': load_IPCCAR6_colors('cmip_cat')[1],
                           }
            return sahel_colors[label]
        elif 'IndianMonsoon' in config['recipe'] and \
            label in ['CMIP5 historical-RCP8.5']:
            india_colors = {'CMIP5 historical-RCP8.5': load_IPCCAR6_colors('cmip_cat')[1],
                            }
            return india_colors[label]
        elif 'SES' in config['recipe'] and label in ['MPI-ESM']:
            douglas_colors = {'MPI-ESM': mgrey,}
            return douglas_colors[label]
        else:
            color = colors[label]
            if isinstance(colors[label], tuple):
                color_table, color_idx = colors[label]
                return load_IPCCAR6_colors(color_table)[color_idx]
            else:
                return colors[label]
    except KeyError:
        if 'MIN' in label:
            return load_IPCCAR6_colors('temp_div_disc', Ncolors=20)(0.9)
        elif 'MAX' in label:
            return load_IPCCAR6_colors('temp_div_disc', Ncolors=20)(0.1)
        elif 'Wettest' in label:
            return load_IPCCAR6_colors('prec_div_disc', Ncolors=20)(0.9)
        elif 'Driest' in label:
            return load_IPCCAR6_colors('prec_div_disc', Ncolors=20)(0.1)
        elif 'Warmest' in label:
            return load_IPCCAR6_colors('temp_div_disc', Ncolors=20)(0.9)
        elif 'Coldest' in label:
            return load_IPCCAR6_colors('temp_div_disc', Ncolors=20)(0.1)
        else:
            color = list(np.random.choice(range(256), size=3))
            color = [c/255. for c in color]
            return color


def boxplot_colors_bw(label):
    """Get colors for plot_boxplot_ax."""
    colors = {'OBS': 'w',
              'OBS+REAN': 'w',
              'CMIP5': 'w',
              'CMIP6': 'k',
              'HighResMIP': 'k',
              'EUR-44': 'k',
              'EUR-11': 'w',
              'CMIP6 historical-SSP1-2.6': 'w',
              'CMIP6 historical-SSP5-8.5': 'w',
             }
    if '(N=' in label:
        label = label.split(' (N=')[0]
    if label in colors.keys():
        return colors[label]
    else:
        return 'k'


def label_markers(dataset):
    """Get marker for label.

    Parameters
    ----------
    label: str
        Dataset label

    Returns
    -------
    color: str (marker)
    """
    marker = OrderedDict([
        ('hist-GHG', 'v'),
        ('hist-aer', '^'),
        ('MPI-GE', 's'),
        ('d4PDF', 's'),
        ('MPI-ESM', 's'),
        ('CanESM2', 's'),
        ('CSIRO-Mk3-6-0', 's'),
        ('CESM1-CAM5', 's'),
        ('GFDL-ESM2M', 's'),
        ('GFDL-CM3', 's'),
        ('EC-EARTH', 's'),
        ('MIROC6', 's'),
        ('OBS', 'x'),
        ('HighResMIP', 'o'),
        ('CMIP5', 'o'),
        ('CMIP6', 'o'),
        ('CORDEX', 'o'),
        ('CRU', 'o'),
        ('BerkeleyEarth', 'v'),
        ('HadCRUT4', '^'),
        ('HadCRUT5', '^'),
        ('CowtanWay', 'P'),
        ('GISTEMP', 's'),
        ('city', 'p'),
        ('country', 'h'),
        ])

    for k, v in marker.items():
        if k == dataset:
            return v

    for k, v in marker.items():
        if k in dataset:
            return v

    logger.error(f"No marker for {dataset} defined.")
    raise KeyError



def load_IPCCAR6_colors(color_table, Ncolors=21, reverse=False):
    """Load IPCC color tables as defined in colormaps dir csvs

    source: https://github.com/IPCC-WG1/colormaps

    Parameters
    ----------
    color_table: str
        Of color_table.txt (a csv file), as listed in colors_dic.values()
        if color_table should be reversed, add '_r' to this string

    Keywords
    --------
    Ncolors: int (default = 21)
    reverse: bool
    path_to_ctabels: str
        filepath to directory

    Returns
    -------
    cm:
        - for colortable in 'categorical_colors_rgb_0-255'
          numpy.ndarray
            of shape (*, 3) of RBG between (0,1.)
        - for colortable in 'continuous_colormaps_rgb_0-1' or
            'discrete_colormaps_rgb_0-255'
          matplotlib.colors.LinearSegmentedColormap

    discrete_colormaps are of maximum Ncolor=21
    categorical_colors are returned as np array
    """
    colors_dic = {'categorical_colors_rgb_0-255':['bright_cat', 'chem_cat',
                        'cmip_cat', 'contrast_cat', 'dark_cat',
                        'gree-blue_cat', 'rcp_cat', 'red-yellow_cat',
                        'spectrum_cat', 'ssp_cat_1', 'ssp_cat_2'],
                  'continuous_colormaps_rgb_0-1':['chem_div', 'chem_seq',
                        'cryo_div', 'cryo_seq', 'misc_div', 'misc_seq_1',
                        'misc_seq_2', 'misc_seq_3', 'prec_div', 'prec_seq',
                        'slev_div', 'slev_seq', 'temp_div', 'temp_seq',
                        'wind_div', 'wind_seq'],
                  'discrete_colormaps_rgb_0-255':['chem_div_disc',
                        'chem_seq_disc', 'cryo_div_disc', 'cryo_seq_disc',
                        'misc_div_disc', 'misc_seq_1_disc', 'misc_seq_2_disc',
                        'misc_seq_3_disc', 'prec_div_disc', 'prec_seq_disc',
                        'slev_div_disc', 'slev_seq_disc', 'temp_div_disc',
                        'temp_seq_disc', 'wind_div_disc', 'wind_seq_disc']}

    if color_table.split('_')[-1] == 'r':
        reverse = True
        color_table = color_table.split('_r')[0]

    if not any([color_table in vs for k,vs in colors_dic.items()]):
        logger.error("Color table {} not found".format(color_table))
        raise NotImplementedError

    for k,vs in colors_dic.items():
        if color_table in vs:
            subdir = k
            break

    lf = os.path.join(path_to_ctabels, subdir, color_table + '.txt')
    logger.info("Loading color table information from {}.".format(lf))

    if subdir == 'categorical_colors_rgb_0-255':
        rgb_in_txt = np.loadtxt(lf)
        cm = rgb_in_txt/255.
        return(cm)
    elif subdir == 'continuous_colormaps_rgb_0-1':
        rgb_in_txt = np.loadtxt(lf)
        if reverse:
            rgb_in_txt = rgb_in_txt[::-1]
        cm = mcolors.LinearSegmentedColormap.from_list(color_table,
                                                       rgb_in_txt, N=Ncolors)
        return(cm)
    elif subdir == 'discrete_colormaps_rgb_0-255':
        if Ncolors > 21:
            logger.warning("{} only available for maximum of 21 colors. "\
                           "Setting Number of colors to 21".format(subdir))
            Ncolors = 21
        df = pd.read_csv(lf)

        str_key_table = '_'.join(color_table.split('_')[:-1])

        for idx in df.index:
            strcell = str(df.iloc[idx].values[0])
            if str_key_table in strcell and \
                int(strcell.split('_')[-1]) == Ncolors :
                col_data = df.iloc[idx+1:idx+Ncolors+1].values
                col_data = [[int(da) for da in dat[0].split(' ')] for dat in col_data]
                rgb_in_txt = np.array(col_data)
                break
        rgb_in_txt = rgb_in_txt/255.
        if reverse:
            rgb_in_txt = rgb_in_txt[::-1]
        cm = mcolors.LinearSegmentedColormap.from_list(
             color_table+'_'+str(Ncolors), rgb_in_txt, N=Ncolors)
        return(cm)


###############################################################################
# All connected to preparing of multiple diagnostics
###############################################################################
def _save_fig(cfg, basename, dpi=1200):
    """Save matplotlib figure."""
    path = get_plot_filename(basename, cfg)
    plt.savefig(path,
                bbox_inches='tight')
    path = [path]
    if cfg.get('savepdf'):
        pdf_path = path[0].replace('.'+cfg['output_file_type'], '.pdf')
        plt.savefig(pdf_path, format='pdf', dpi=dpi,
                    bbox_inches='tight')
        path.append(pdf_path)
    if cfg.get('saveeps'):
        eps_path = path[0].replace('.'+cfg['output_file_type'], '.eps')
        plt.savefig(eps_path, format='eps', dpi=dpi,
                    bbox_inches='tight')
        path.append(eps_path)
    if cfg.get('savesvg'):
        svg_path = path[0].replace('.'+cfg['output_file_type'], '.svg')
        plt.savefig(svg_path, format='svg', dpi=dpi,
                    bbox_inches='tight')
        path.append(svg_path)
    logger.info("Wrote %s", path)
    plt.close()
    return path


def _check_period_information(period, info):
    """Check a dictionary for correct period information.

    Amend missing month and day informations.

    Parameters
    ----------
    period: :obj:`dict`
        Containing (partial) period information
    info: str
        just for logging

    Returns
    ----------
    period: dict
    """
    if not 'start_year' in period:
        logger.error("No start_year given")
        raise KeyError
    if not 'end_year' in period:
        logger.error("No end year given")
        raise KeyError
    if not 'start_month' in period:
        period.update({'start_month': 1})
    if not 'start_day' in period:
        period.update({'start_day': 1})
    if not 'end_month' in period:
        period.update({'end_month': 12})
    if not 'end_day' in period:
        period.update({'end_day': 31})
    logger.info("Using {} period: "\
                "{start_year}-{start_month}-{start_day} -- "\
                "{end_year}-{end_month}-{end_day}"\
                "".format(info, **period))

    return period


def verify_diagnostic_settings(diagnostic, defaults, valids, valid_types):
    """Verify diagnostic definitions.

    Parameters
    ----------
    diagnostic: :obj:`dict`
        diagnostic dict
    defaults
        - add defaults if not present
    valids
        - check valids
    valid_types
        - check valid types
    """
    for k, v in defaults.items():
        if k not in diagnostic:
            diagnostic[k] = v
            logger.warning(f"No information for {k} given. "
                           f"Using {v}")
    for k, v in valids.items():
        if diagnostic[k] not in v:
            if k in defaults:
                logger.warning(f"{diagnostic[k]} no valid argument for {k}."
                               f"Falling back to default value {defaults[k]}")
                diagnostic[k] = defaults[k]
            else:
                logger.error(f"{diagnostic[k]} no valid argument for {k}."
                             f"No default given.")
                raise KeyError
    for k, v in valid_types.items():
        if not isinstance(diagnostic[k], v):
            if k in defaults:
                logger.warning(f"{type(diagnostic[k])} no valid type for {k}."
                               f"Falling back to default value {defaults[k]}")
                diagnostic[k] = defaults[k]
            else:
                logger.error(f"{type(diagnostic[k])} no valid type for {k}."
                             f"No default given.")
                raise KeyError


def relative_bias(cube, cube_norm):
    """Prepare relative anomalies cube.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube
    cube_norm: iris.cube.Cube
        to calculate anomalies of input cube by

    Returns
    -------
    cube: iris.cube.Cube
        anomalies cube
    """
    cube.data = (cube.data - cube_norm.data) / cube_norm.data * 100
    cube.units = Unit('%')
    cube.long_name = cube.long_name + " relative bias"
    return cube


def absolute_bias(cube, cube_norm):
    """Prepare absolute anomalies cube.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube
    cube_norm: iris.cube.Cube
        to calculate anomalies of input cube by

    Returns
    -------
    cube_norm: iris.cube.Cube
        anomalies cube
    """
    cube.data = cube.data - cube_norm.data
    cube.long_name = cube.long_name + " bias"
    return cube


def relative_anomalies(cube, cube_norm):
    """Prepare relative anomalies cube.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube
    cube_norm: iris.cube.Cube
        to calculate anomalies of input cube by

    Returns
    -------
    cube_norm: iris.cube.Cube
        anomalies cube
    """
    cube.data = (cube.data - cube_norm.data) / cube_norm.data * 100
    cube.units = Unit('%')
    cube.long_name = cube.long_name + " relative anomaly"
    return cube


def absolute_anomalies(cube, cube_norm):
    """Prepare absolute anomalies cube.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube
    cube_norm: iris.cube.Cube
        to calculate anomalies of input cube by

    Returns
    -------
    cube_norm: iris.cube.Cube
        anomalies cube
    """
    cube.data = cube.data - cube_norm.data
    cube.long_name = cube.long_name + " anomaly"
    return cube


def get_base_cube(cube, period_norm=None):
    """Prepares time mean cube used for anomalies data.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube
    period_norm: dict
        Holding period information

    Returns
    -------
    cube_norm: iris.cube.Cube
        time mean cube

    todo:
    add warning exception for: UserWarning: Collapsing a non-contiguous coord
    """
    if period_norm:
        cube_norm =  extract_time(cube, **period_norm)
    else:
        cube_norm = cube.copy()
    cube_norm = climate_statistics(cube_norm)

    return cube_norm


def format_label(label):
    """Make label conform with IPCC standards.

    Parameters
    ----------
    label: str

    Returns
    -------
    label: str
    """
    if isinstance(label, list):
        return label

    if 'rcp' in label:
        rcp = re.findall('rcp\d\d', label)[0]
        em_scen = rcp.split('rcp')[1]
        em_scen = em_scen[0] + '.' + em_scen[1]
        label = label.replace(rcp, 'RCP' + em_scen)
    elif 'ssp' in label:
        ssp = re.findall('ssp\d\d\d', label)[0]
        em_scen = ssp.split('ssp')[1]
        em_scen = em_scen[0] + '-' + em_scen[1] + '.' + em_scen[2]
        label = label.replace(ssp, 'SSP' + em_scen)

    translate_dic = {'MPIGE': 'MPI-GE',
                     'MRI-AGCM3.2H': 'd4PDF'}
    for k, v in translate_dic.items():
        if k in label:
            label = label.replace(k, v)

    return label


def format_units(unit):
    """Formats units."""

    repl_map = {'degC': '°C',
                'K ': '°C ',
                'kg m-2': 'mm',
                'month-1': 'month$^{{-1}}$',
                'day-1': 'day$^{{-1}}$',
                'd-1': 'day$^{{-1}}$'}

    for k, v in repl_map.items():
        if k in unit:
            unit = unit.replace(k, v)

    return unit


def get_trendminmax_indices(cubes, period=None, region=None, shapefile=None):
    """Extract the cubes that have min and max trend from the cubes. """
    trends = []
    # calculate the trends over the period in question
    for cube in cubes:
        if period: # extract the period
            cube =  extract_time(cube, **period)
        if shapefile:
            cube = extract_shape(cube, shapefile)
        elif region: # extract the region
            cube = extract_region(cube,
                                  region['start_longitude'],
                                  region['end_longitude'],
                                  region['start_latitude'],
                                  region['end_latitude'])
        if len(cube.shape) > 2:
            cube = area_statistics(cube, 'mean')
        # we are only dealing with yearly data, so we will only use that as the base
        if not cube.coords('year'):
            iris.coord_categorisation.add_year(cube, 'time')
        # simple check if its 'yearly' data
        if not cube.coord('year').shape[0] == cube.shape[0]:
            logger.error('Data for trend calculation has to be yearly')
            raise ValueError
        cube =  linear_trend(cube, coordinate='year')
        trends.append(cube.data)
    minmax_inds = [np.where(trends == np.min(trends))[0][0],
                   np.where(trends == np.max(trends))[0][0]]
    return minmax_inds


def get_trendminmedianmax_indices(cubes, period=None, region=None,
                                  shapefile=None):
    """ Extract the cubes that have min and max trend from the cubes. """
    trends = []
    # calcuate the trends over the period in question
    for cube in cubes:
        if period: # extract the period
            cube =  extract_time(cube, **period)
        if shapefile:
            cube = extract_shape(cube, shapefile)
        elif region: # extract the region
            cube = extract_region(cube,
                                  region['start_longitude'],
                                  region['end_longitude'],
                                  region['start_latitude'],
                                  region['end_latitude'])
        else:
            logger.error('Both shapefile and region for trend '\
                            'calculation present')
            raise KeyError

        if len(cube.shape) > 2:
            cube = area_statistics(cube, 'mean')
        # we are only dealing with yearly data, so we will only use that as the base
        if not cube.coords('year'):
            iris.coord_categorisation.add_year(cube, 'time')
        # simple check if its 'yearly' data
        if not cube.coord('year').shape[0] == cube.shape[0]:
            logger.error('Data for trend calculation has to be yearly')
            raise ValueError
        cube =  linear_trend(cube, coordinate='year')
        trends.append(cube.data)
    median_ind = np.where(trends == sorted(trends)[
                    int(np.floor(len(cubes)/2))])[0][0]
    minmedmax_inds = [np.where(trends == np.min(trends))[0][0],
                      median_ind,
                      np.where(trends == np.max(trends))[0][0]]
    return minmedmax_inds


def get_trenddifference_indices(cubes, metrics, period=None, region=None):
    """ Extract the N cubes that have min and max trends from the cubes. """
    trends = []
    # calcuate the trends over the period in question
    for cube in cubes:
        if period: # extract the period
            cube =  extract_time(cube, **period)
        if region: # extract the region
            cube = extract_region(cube,
                                  region['start_longitude'],
                                  region['end_longitude'],
                                  region['start_latitude'],
                                  region['end_latitude'])
        if len(cube.shape) > 2:
            cube = area_statistics(cube, 'mean')
        # we are only dealing with yearly data, so we will only use that as the base
        if not cube.coords('year'):
            iris.coord_categorisation.add_year(cube, 'time')
        # simple check if its 'yearly' data
        if not cube.coord('year').shape[0] == cube.shape[0]:
            logger.error('Data for trend calculation has to be yearly')
            raise ValueError
        cube =  linear_trend(cube, coordinate='year')
        trends.append(cube.data)
    trend_inds = np.argsort(trends)
    for ind, (metric, N) in enumerate(metrics):
        if metric == 'max':
            metrics[ind].append(list(trend_inds[-N:]))
        elif metric == 'min':
            metrics[ind].append(list(trend_inds[:N]))
    return metrics


###############################################################################
# All connected to preparing the timeseries data
###############################################################################
def prepare_timeseries_data(diag_name, diagnostic, ensembles):
    """Updates diagnostic: :obj:`dict` with prepared data.

    Parameters
    ----------
    diag_name: str
        The Name of the diagnostic.
    diagnostic: :obj:`dict`
        Of type timeseries diagnostic.
    ensembles: :obj:`dict` of :obj:`list` of :obj:`dict`
        Holding ensemble sorted cfg['input_data']
    """
    logger.info(f"Checking integrity of diagnostic settings: {diag_name}")
    verify_timeseries_diagnostic_settings(diagnostic)

    logger.info(f"Deriving data for diagnostic: {diag_name}")

    # loop over specified single timeseries and boxplots
    ts_dicts = {k: v for k, v in diagnostic.items()
                if 'timeseries' in k.split("_") or 'boxes' in k.split("_")}
    for tsk, ts in ts_dicts.items():
        logger.info(f"Deriving data for timeseries: {tsk}")
        check_timeseries_definitions(tsk, ts)

        for ens in ts['ensembles']:
            logger.info(f"\t adding: {ens}")
            # single loaded cubes
            ens_cubes = []
            labels = []

            # processed cubes for single metrics (can be more than one)
            final_cubes = []
            if ts['indicate_bars']:
                final_cubes_raw = []

            # load the data
            N = 0
            if 'datasets' in ts.keys():
                included_ds = []
                for ds in ts['datasets']:
                    for dic in ensembles[ens]:
                        if dic['dataset'] == ds:
                            f = dic['filename']
                            cube = iris.load_cube(f)
                            ens_cubes.append(cube)
                            included_ds.append(ds)
                            N += 1
                if not ens_cubes:
                    logger.error("Did not find any of the datasets {} in "\
                        "the ensemble {}".format(ts['datasets'], ens))
                    raise KeyError
                if included_ds != ts['datasets']:
                    ids = set(included_ds)
                    missing_ds = [ds for ds in ts['datasets'] if ds not in ids]
                    included_ds = [ds for ds in ts['datasets'] if ds in ids]
                    if missing_ds:
                        logger.warning("Did not find the datasets {} in "\
                            "the ensemble {}. Continuing only with "\
                            "{}".format(missing_ds, ens, included_ds))
                    ts['datasets'] = included_ds
            else:
                for dic in ensembles[ens]:
                    f = dic['filename']
                    cube = iris.load_cube(f)
                    ens_cubes.append(cube)
                    N += 1

            # get possilby anomalies and possilby relative
            ens_cubes = _anomalies(ens_cubes, diagnostic)

            for metric in ts['metrics']:
                tmp_ens = copy.deepcopy(ens_cubes)
                cubes = derive_timeseries_by_metric(tmp_ens,
                                                    metric, ens,
                                                    ensembles[ens],
                                                    diagnostic['span'], ts)
                # if bar data has to be maintained do so
                if ts['indicate_bars']:
                    cubes_raw = cubes.copy()
                    for cube in cubes_raw:
                        if not cube.coords('year'):
                            iris.coord_categorisation.add_year(cube, 'time')

                # apply possible filters and extract plotting period
                if 'boxes' in tsk:
                    cubes = finalize_timeseries_data(cubes, metric,
                                                    ts['period'],
                                                    '')
                else:
                    cubes = finalize_timeseries_data(cubes, metric,
                                                     ts['period'],
                                                     diagnostic['window'])

                final_cubes.append(cubes)
                if ts['indicate_bars']:
                    final_cubes_raw.append(cubes_raw)

                if isinstance(ts['labeling'], str):
                    label = derive_labels_timeseries(ts, ens, ensembles[ens],
                                                     metric, ts['labeling'], N)
                else:
                    for labeling_format in ts['labeling']:
                        try:
                            label = derive_labels_timeseries(ts, ens,
                                        ensembles[ens], metric,
                                        labeling_format, N)
                            break
                        except KeyError: continue
                        except TypeError: continue
                label = format_label(label)
                labels.append(label)

            ts['data'].append(final_cubes)
            if ts['indicate_bars']:
                ts['data_raw'].append(final_cubes_raw)
            ts['labels'].append(labels)

    # get external timeseries data
    exts_dicts = {k: v for k, v in diagnostic.items()
                  if 'externaltimeseries' in k.split("_")}
    for tsk, ts in exts_dicts.items():
        logger.info(f"Deriving data for timeseries: {tsk}")
        if 'external' in tsk and 'iitm' in tsk:
            diagnostic.update(prepare_IITM_timeseries(diagnostic, tsk, ts))
            diagnostic.pop(tsk)

    # loop over specified single boxplots
    boxes_dicts = {k: v for k, v in diagnostic.items() if 'boxes' in k }
    if boxes_dicts:
        columns = ['ens', 'metric', 'label', 'data']
        for bxk, bx in boxes_dicts.items():
            df = pd.DataFrame(columns=columns)

            logger.info(f"Deriving data for boxes: {bxk}")
            ens_c_cubes = []
            new_labels = []
            for ens, ens_cubes, labels in zip(bx['ensembles'], bx['data'],
                                              bx['labels']):
                metric_c_cubes = []
                for metric, metric_cubes, label in zip(bx['metrics'],
                                                       ens_cubes,
                                                       labels):
                    c_cubes = []
                    for cube in metric_cubes:
                        c_cubes.append(derive_timeseries_box_data(
                            cube, bx, diagnostic))
                    data = [c_cube.data for c_cube in c_cubes]
                    df_tmp = pd.DataFrame([[ens, metric, label, data]],
                                          columns = columns)
                    df = df.append(df_tmp)

                    metric_c_cubes.append(c_cubes)
                ens_c_cubes.append(metric_c_cubes)
            bx['data_cubes'] = ens_c_cubes
            bx['data'] = df




def derive_timeseries_box_data(cube, box, diagnostic):
    """
    Extracts cc information from cube.
    """
    btype = box['box_type']
    if btype == 'change':
        base_cube =  extract_time(cube, **diagnostic['period_norm'])
        future_cube =  extract_time(cube, **box['period_box'])
        base_cube = climate_statistics(base_cube)
        future_cube = climate_statistics(future_cube)
        change_cube = future_cube - base_cube
        # keep the metadata of the cube
        change_cube.standard_name = cube.standard_name
        change_cube.var_name = cube.var_name
        change_cube.long_name = cube.long_name
        change_cube.attributes = cube.attributes

    elif btype == 'trend':
        # future_cube =  extract_time(cube, **box['period_box'])
        change_cube, trend_over_n = calculate_trend(cube, False, box['trend_base'],
                          period_trend=box['period_box'])
    else:
        logger.error(f"boxes type: {btype} not implemented")
        raise NotImplementedError

    return change_cube


def derive_labels_timeseries(ts, ens, ensemble, metric, label_format, N):
    """Extract label from datasets according to ts['labeling'] formatting.

    Parameters
    ----------
    ts: :obj:`dict`
        Of type timeseries dict.
    ens: str
    ensemble: :obj:`list` of :obj:`dict`
        Holding cfg['input_data']
    metric: str
    label_format: formatter string

    Returns
    -------
    label: str or list of str
    """
    if not label_format:
        return ens
    else:
        keys = re.findall('\{(.*?)\}', label_format)
        if 'metric' in keys:
            keys.remove('metric')
            add_metric = True
        else:
            add_metric = False
        if 'N' in keys:
            keys.remove('N')
        else: N = None

        labeling = {k: set() for k in keys}
        for dataset in ensemble:
            for k in keys:
                labeling[k].add(dataset[k])

        # remove again
        if 'project' in labeling.keys():
            if 'OBS' in labeling['project']:
                labeling['project'].remove('OBS')

        if 'datasets' in ts and 'dataset' in keys:
            labeling['dataset'] = ts['datasets']

        if not np.all([len(value) == 1 for value in labeling.values()]):
            if not len(labeling.keys()) == 1 and \
                not 'dataset' in labeling.keys():
                logger.warning("Unable to build clear labeling for {} with "\
                               "scheme {}. Falling back to "\
                               "{}".format(ens, label_format, ens))
                return ens
            else:
                return list(labeling['dataset'])
        else:
            for k,v in labeling.items():
                labeling[k] = list(v)[0]
            if N: labeling.update({'N': N})
            if add_metric:
                labeling.update({'metric': metric})
            return label_format.format(**labeling)


def verify_timeseries_diagnostic_settings(diagnostic):
    """ Verify timeseries diagnostic definitions.
    - add defaults if not present
    - check valids
    - check valid types
    """
    defaults = {'span': 'full', # treatment of multimodel span
                'window': '5weighted', # running means
                'anomalies': False, # get information on anomalies
                'relative': False, # get information on relative treatment
                }
    valids = {'span': ['overlap', 'full']}
    valid_types = {'anomalies': bool,
                   'relative': bool}
    verify_diagnostic_settings(diagnostic, defaults, valids, valid_types)

    # clarify which period to use for anomalies base
    if diagnostic['anomalies']:
        if not 'period_norm' in diagnostic:
            logger.warning("Anomalies selected, but no period given. "\
                           "using entire period")
            diagnostic.update({'period_norm': None})
        else:
            diagnostic['period_norm'] = _check_period_information(
                diagnostic['period_norm'], "anomalies")

    # check some plotting information settings
    pi_defaults = {'figsize': (16, 10)}

    for k, v in pi_defaults.items():
        if k not in diagnostic['plotting_information']:
            diagnostic['plotting_information'][k] = v
            logger.warning(f"No information for {k} given. "
                           f"Using {v}")


def check_timeseries_definitions(tsk, ts):
    """Check if all necessary information for the single ts is present."""
    # get information on the period to be shown
    if 'period' in ts:
        if 'start_year' in ts['period'] and \
            'end_year' in ts['period']:
            ts['period'] = _check_period_information(ts['period'], tsk)
    else:
        ts['period'] = None
    if 'period_box' in ts:
        if 'start_year' in ts['period_box'] and \
            'end_year' in ts['period_box']:
            ts['period_box'] = _check_period_information(ts['period_box'], tsk)

    # get information for the preprocessing of the timeseries
    if not 'metrics' in ts:
        logger.error("No Information on desired metrics given for {}."\
                     "".format(tsk))
        raise KeyError

    # check if data for bars has to be maintained
    if 'indicate_bars' in ts and \
        ('single' in ts['metrics'] or 'mean' in ts['metrics']):
        if ts['indicate_bars']:
            ts.update({'data_raw': []})
    else:
        ts.update({'indicate_bars': False})

    # init the data list for the ensemble statistics
    ts.update({'data': []})
    # init the labels list
    ts.update({'labels': []})


def finalize_timeseries_data(cubes, metric, period_ts, window):
    """Add year, running mean, extract period"""
    # this is important for plotting
    for cube in cubes:
        if not cube.coords('year'):
            iris.coord_categorisation.add_year(cube, 'time')

    # running means
    if not 'line' in metric:
    # if not 'trendline' in metric:
        cubes = [running_mean(cube, window=window) for cube in cubes]

    # extract the given timeseries period
    if period_ts:
        for c_ind, cube in enumerate(cubes):
            cubes[c_ind] = extract_time(cube, **period_ts)

    return cubes


def _anomalies(cubes, diagnostic):
    """Gets anomalies and possibly makes cube relative.

    Parameters
    ----------
    cubes: list
        Of iris.cube.Cube
    diagnostic: :obj:`dict`
        Of type timeseries diagnostic. Required keys: anomalies, period_norm,
            relative

    Returns
    -------
    cubes: list
        Of possibly anomalies / made relative cubes
    """
    if diagnostic['anomalies']:
        norm_cubes = [get_base_cube(cube,
                                    period_norm=diagnostic['period_norm'])
                      for cube in cubes]
        anom_cubes = []
        if diagnostic['relative']:
            for cube, norm_cube in zip(cubes, norm_cubes):
                anom_cubes.append(relative_anomalies(cube, norm_cube))
        else:
            for cube, norm_cube in zip(cubes, norm_cubes):
                anom_cubes.append(absolute_anomalies(cube, norm_cube))
    return anom_cubes


def derive_timeseries_by_metric(ens_cubes, metric, ens, ensemble, span, ts):
    """
    Derives ensemble t  meseries.
    Possible metrics are:
        - mean      np.ma.mean
        - median    np.ma.median
        - std       np.ma.std
        - min       np.ma.min
        - max       np.ma.max
        - envelop   [np.ma.min, np.ma.max]
        - trendline_[metric] metric=[mean, median, std, min, max]

        # special
        - single        returns the single cubes datasets and updates dataset
                        info
        - mean_pm_std   [np.ma.mean + np.ma.std, np.ma.mean - np.ma.std]
        - minmaxtrend   [the models that shows the min&max trend]
        - minmaxtrend-lines  the trendline of models with the min&max trend
    Parameters
    ----------
    ens_cubes: list
        Of ensemble cubes
    metric: str
        Ensemble metric to derive
    ensemble: :obj:`list` of :obj:`dict`
        Holding cfg['input_data']
    span: str
        overlap or full; if overlap stas are computed on common time-span;
        if full stats are computed on full time spans.
    ts: :obj:`dict`
        Holind single timeseries information

    Returns
    -------
    cubes: list
        Of derived cubes
                # possible metrics include

    """
    period_ts = ts['period']

    if metric in ['mean', 'median', 'min', 'max', 'std']:
        cube = multi_model_statistics(ens_cubes, span,
                                      [metric])[metric]
        cubes = [cube]
    elif metric == 'envelop':
        if len(ens_cubes) < 2:
            logger.error("Metric {} needs at least two cube".format(metric))
            raise NotImplementedError
        else:
            # get the overlap over the whole meriod
            cube_min = multi_model_statistics(ens_cubes, span,
                                                ['min'])['min']
            cube_max = multi_model_statistics(ens_cubes, span,
                                                ['max'])['max']
            cubes = [cube_min, cube_max]
    elif metric == 'mean_pm_std':
        if len(ens_cubes) < 2:
            logger.error(
                "Metric {} needs at least two cube".format(metric))
            raise NotImplementedError
        else:
            cube_mean = multi_model_statistics(ens_cubes, span,
                                                ['mean'])['mean']
            cube_std = multi_model_statistics(ens_cubes, span,
                                                ['std'])['std']
            cubes = [cube_mean.copy(), cube_mean.copy()]
            cubes[0].data = cubes[0].data + cube_std.data
            cubes[1].data = cubes[1].data - cube_std.data
    elif 'trendline' in metric:
        submetric = metric.split('_')[1]
        # extract the given timeseries period
        if period_ts:
            for ec_ind, cube in enumerate(ens_cubes):
                ens_cubes[ec_ind] = extract_time(cube, **period_ts)
        if len(ens_cubes) == 1:
            cube = ens_cubes[0]
        else:
            cube = multi_model_statistics(ens_cubes, span,
                                          [submetric])[submetric]
        cube.data -= detrend(cube).data
        cubes = [cube]
    elif metric in ['minmaxtrend', 'minmaxtrend-lines']:
        if 'period_trend' in ts:
            period_trend = ts['period_trend']
            period_trend = _check_period_information(period_trend,
                                                     "minmaxtrend")
            minmax_inds = get_trendminmax_indices(ens_cubes,
                                                  period=period_trend)
        else:
            minmax_inds = get_trendminmax_indices(ens_cubes, period=period_ts)

        if metric == 'minmaxtrend':
            cubes = [ens_cubes[minmax_inds[0]], ens_cubes[minmax_inds[1]]]
        else:
            cubes = [ens_cubes[minmax_inds[0]], ens_cubes[minmax_inds[1]]]
            if 'period_trend' in ts:
                period_trend = ts['period_trend']
            else:
                period_trend = period_ts
            cubes = [extract_time(cube, **period_trend) for cube in cubes]
            cubes[0].data -= detrend(cubes[0]).data
            cubes[1].data -= detrend(cubes[1]).data
    elif metric == 'single':
        if 'datasets' not in ts:
            datasets = [dic['dataset'] for dic in ensemble]
            ts.update({f'datasets_{ens}': datasets})
            cubes = ens_cubes
        else:
            sub_cubes = []
            for ds in ts['datasets']:
                for cube, dic in zip(ens_cubes, ensemble):
                    if dic['dataset'] == ds:
                        sub_cubes.append(cube)
            cubes = sub_cubes
    else:
        logger.error("Metric {} not implemented".format(metric))
        raise NotImplementedError

    return cubes

###############################################################################
# Statistical functions workable also for multidim cubes
# detrend and multi_model_statistics are adapted from core
###############################################################################
def calculate_trend(cube, relative, trend_base, period_trend=None,
                    period_norm=None):
    """Calculates the trend over a cube.

    Parameters
    ----------
    cube: iris.cube.Cube
    relative: bool
    trend_base: str
        possible values: 'year', 'decade', 'all'
    period_trend: dict
    period_norm: dict

    Returns
    -------
    cube: iris.cube.Cube
        of derived trend
    trend_over_n: int
    """
    from iris.time import PartialDateTime

    if relative:
        cube_norm = get_base_cube(cube, period_norm=period_norm)

    if period_trend:
        cube = extract_time(cube,
                            period_trend['start_year'],
                            period_trend['start_month'],
                            period_trend['start_day'],
                            period_trend['end_year'],
                            period_trend['end_month'],
                            period_trend['end_day'])

    if relative:
        cube = relative_anomalies(cube, cube_norm)

    if cube.data.mask.any():
        logger.warning('Masked data for trend calculation found.')
        logger.info('Checking time invariance.')
        # check for data not masked always or never
        summask = cube.data.mask.sum(axis=0)
        if not(np.any(~((summask == 0) | (summask == cube.shape[0])))):
            logger.info('Time invariance confirmed.')
        else:
            logger.info('Time invariance found.')

            combined_mask = np.zeros(cube.shape[1:])
            if config['recipe'] == "recipe_NAM.yml":
                masking_threshhold = 0.6
            elif config['recipe'] == "recipe_Sahel.yml":
                masking_threshhold = 0.7
            else:
                masking_threshhold = 0.8
            # # # # NAM
            # masking_threshhold = 0.6
            # # # # # Sahel
            # masking_threshhold = 0.7

            if not cube.coords('year'):
                iris.coord_categorisation.add_year(cube, 'time')
            uniques = sorted(list(set(cube.coord('year').points)))

            starts = uniques[::10]
            ends = uniques[9::10]
            if len(ends) < len(starts):
                starts = starts[:-1]
                ends.append(uniques[-1])
                starts.append(uniques[-1]-9)
            lens = [start-end for end,start in zip(starts,ends)]
            if not np.all(np.array(lens) == 9):
                starts = starts[:-1]
                starts.append(ends[-1]-9)

            for start, end in zip(starts, ends):
                t_1 = PartialDateTime(year=int(start), month=1, day=1)
                t_2 = PartialDateTime(year=int(end), month=12, day=31)
                constraint = iris.Constraint(
                    time=lambda t: t_1 <= t.point < t_2)
                decade_cube = cube.extract(constraint)
                mask_data = decade_cube.data.mask.sum(axis=0)
                mask_data = np.ma.masked_where(
                        (mask_data / decade_cube.shape[0]) > \
                        (1 - masking_threshhold), mask_data.data)
                combined_mask += mask_data.mask

            combined_mask = combined_mask.astype('bool')
            combined_mask = np.array([combined_mask] * cube.shape[0])
            cube.data = np.ma.masked_where(combined_mask==1., cube.data)

    # we are only dealing with yearly data, so we will only use that as the base
    if not cube.coords('year'):
        iris.coord_categorisation.add_year(cube, 'time')

    # simple check if its 'yearly' data
    if not cube.coord('year').shape[0] == cube.shape[0]:
        logger.error('Data for trend calculation has to be yearly')
        raise ValueError

    trend_over_n = cube.shape[0] - 1
    cube =  linear_trend(cube, coordinate='year')

    if trend_base == 'year':
        pass
    elif trend_base in ['decade', 'all']:
        standard_name = cube.standard_name
        var_name = cube.var_name
        long_name = cube.long_name
        attr = cube.attributes
        if trend_base == 'decade':
            cube.data = cube.data * 10
        else:
            cube.data = cube.data * trend_over_n
        cube.standard_name = standard_name
        cube.var_name = var_name
        cube.long_name = long_name
        cube.attributes = attr
    else:
        logger.error("Trend_base {} not implemented.".format(trend_base))
        raise NotImplementedError

    return(cube, trend_over_n)


def running_mean(cube, window):
    """Basically the iris cube.rolling_window function iris.analysis.MEAN.

    - Catches warnings
    - two filters of the AR4 are implemented
    https://archive.ipcc.ch/publications_and_data/ar4/wg1/en/ch3sappendix-3-a.html
    - additionaly ordinary unweighted running means
    """
    from esmvalcore.preprocessor._time import low_pass_weights
    from warnings import catch_warnings, filterwarnings

    try:
        window = int(window)
    except ValueError:
        pass

    if window == '5weighted':
        weights = np.array([1,3,4,3,1])
    elif window == '13weighted':
        weights = np.array([1,6,19,42,71,96,106,96,71,42,19,6,1])
    elif isinstance(window, int):
        if window == 1:
            return(cube)
        weights = np.array([1] * window)
    elif window == '':
        return(cube)
    elif 'Lanczos' in window:
        weights = int(window.split('Lanczos')[0])
        weights = low_pass_weights(weights, 1. / weights)
    else:
        logger.error('No filter for window = {} implemented'.format(window))
        raise NotImplementedError

    with catch_warnings():
        filterwarnings(
            action='ignore',
            message='The bounds of coordinate .* were ignored ' \
                    'in the rolling.*',
            category=UserWarning,
            module='iris',
            )
        cube = cube.rolling_window('time', iris.analysis.MEAN, len(weights),
                                   weights=weights)

    return(cube)


def detrend(cube, dimension='time', method='linear'):
    """Addapted from preprocessor to be able to handle missing data

    Detrend data along a given dimension.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube
    dimension: str
        Dimension to detrend
    method: str
        Method to detrend. Available: linear, constant. See documentation of
        'scipy.signal.detrend' for details

    Returns
    -------
    iris.cube.Cube
        Detrended cube
    """
    import dask.array as da
    import scipy.signal

    coord = cube.coord(dimension)
    axis = cube.coord_dims(coord)[0]
    detrended = da.apply_along_axis(
        scipy.signal.detrend,
        axis=axis,
        arr=cube.lazy_data(),
        type=method,
        shape=(cube.shape[axis],)
    )
    detr_cube = cube.copy(detrended)

    mask = np.any(cube.data.mask, axis=axis)
    mask = np.array([mask]*cube.shape[axis])

    detr_cube.data = np.ma.array(detr_cube.data, mask=mask)

    return detr_cube


if __name__ == '__main__':
    with run_diagnostic() as config:
        ar6_wg1_ch10_figures(config)
