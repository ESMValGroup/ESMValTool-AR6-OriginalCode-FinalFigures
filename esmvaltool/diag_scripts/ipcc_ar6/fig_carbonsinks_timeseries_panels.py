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
    variables_available, select_metadata)
import esmvaltool.diag_scripts.shared.ESRL_ccgcrv.ccgfilt as ccgfilt
from esmvaltool.diag_scripts.shared.ESRL_ccgcrv.ccgdates import (
    decimalDate, decimalDateFromDatetime, calendarDate, datetimeFromDecimalDate)
from matplotlib import rcParams
import numpy as np

logger = logging.getLogger(os.path.basename(__file__))

def main(cfg):
    """Run the diagnostic."""
    #cmip6_models = select_metadata(cfg['input_data'].values(), project="CMIP6")
    #datasetnames = list(group_metadata(cmip6_models, 'dataset').keys())

    n_cycle_models = ["ACCESS-ESM1-5", "BNU-ESM", "CESM1-BGC", "NorESM1-ME", "UKESM1-0-LL",
                      "NorESM2-LM", "EC-Earth3-Veg", "CESM2", "CESM2-WACCM",
                      "SAM0-UNICON", "MIROC-ES2L", "MPI-ESM1-2-LR"]
    legend_items = {}

    tas_data = select_metadata(cfg['input_data'].values(), short_name="tas")
    nbp_data = select_metadata(cfg['input_data'].values(), short_name="nbp")
    fgco2_data = select_metadata(cfg['input_data'].values(), short_name="fgco2")
    co2_data = select_metadata(cfg['input_data'].values(), short_name="co2s")
    #logger.info(input_data)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    for data in co2_data:
        # Deal with co2 read-in differently!
        name = data['dataset']
        logger.info("Processing %s", name)
        cube = iris.load_cube(data['filename'])
        cube.convert_units("ppmv")
        iris.coord_categorisation.add_year(cube, 'time')
        cube = cube.aggregated_by('year', iris.analysis.MEAN)
        if name=="ESRL":
            cube = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN)
            ax1.plot(cube.coord("year").points, cube.data, color="black",
                     label = "OBS", linewidth = 2.0)
        else:
            style = plot.get_dataset_style(name, 'cmip6_ipcc')
            if name in n_cycle_models:
                linestyle = "--"
            else:
                linestyle = "-"
            legend_items[name] = {'color': style['color'],
                                  'linestyle': linestyle,
                                  'linewidth': style['thick']}

            ax1.plot(cube.coord("year").points, cube.data, color = style['color'],
                     linestyle = legend_items[name]['linestyle'], label = name,
                     linewidth = legend_items[name]['linewidth'])

    for data in tas_data:
        # Deal with co2 read-in differently!
        name = data['dataset']
        logger.info("Processing %s", name)
        cube = iris.load_cube(data['filename'])
        cube.convert_units('celsius')
        iris.coord_categorisation.add_year(cube, 'time')
        cube = cube.aggregated_by('year', iris.analysis.MEAN)
        cube.data -= np.mean(
            cube.extract(
                iris.Constraint(year=lambda cell: 1850 <= cell <= 1901)).data)
        if name=="HadCRUT4":
            ax2.plot(cube.coord("year").points, cube.data, color="black",
                     label = "OBS", linewidth = 2.0, zorder= 100)
        else:
            ax2.plot(cube.coord("year").points, cube.data,
                     color = legend_items[name]['color'],
                     linestyle = legend_items[name]['linestyle'], label = name,
                     linewidth = legend_items[name]['linewidth'])

    for data in nbp_data:
        # Deal with co2 read-in differently!
        name = data['dataset']
        logger.info("Processing %s", name)
        cube = iris.load_cube(data['filename'])
        cube.convert_units('Pg m-2 yr-1')
        iris.coord_categorisation.add_year(cube, 'time')
        cube = cube.aggregated_by('year', iris.analysis.MEAN)
        cube = cube.rolling_window('year', iris.analysis.MEAN, 10)
        if name=="GCP":
            cube = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN)
            cube.data = cube.data * 148300000000000. #multiply by area
            ax3.plot(cube.coord("year").points, cube.data, color="black",
                     label = "OBS", linewidth = 2.0, zorder= 100)
            ax3.fill_between(cube.coord("year").points, cube.data - 0.6, cube.data + 0.6,
                             color = "black", alpha = 0.2, zorder = 101)
        else:
            ax3.plot(cube.coord("year").points, cube.data,
                     color = legend_items[name]['color'],
                     linestyle = legend_items[name]['linestyle'], label = name,
                     linewidth = legend_items[name]['linewidth'])

    for data in fgco2_data:
        # Deal with co2 read-in differently!
        name = data['dataset']
        logger.info("Processing %s", name)
        cube = iris.load_cube(data['filename'])
        cube.convert_units('Pg m-2 yr-1')
        iris.coord_categorisation.add_year(cube, 'time')
        cube = cube.aggregated_by('year', iris.analysis.MEAN)
        cube = cube.rolling_window('year', iris.analysis.MEAN, 10)
        if name=="GCP":
            cube = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN)
            cube.data = cube.data * 360000000000000. #multiply by area
            ax4.plot(cube.coord("year").points, cube.data, color="black",
                     label = "OBS", linewidth = 2.0, zorder= 100)
            ax4.fill_between(cube.coord("year").points, cube.data - 0.5, cube.data + 0.5,
                             color = "black", alpha = 0.2, zorder= 101)
        else:
            ax4.plot(cube.coord("year").points, cube.data,
                     color = legend_items[name]['color'],
                     linestyle = legend_items[name]['linestyle'], label = name,
                     linewidth = legend_items[name]['linewidth'])

    ax1.set_xlabel("Year")
    ax1.set_ylabel(r"Atmospheric CO$_2$ [ppmv]")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Temperature anomaly [°C]")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Net Land C Flux [PgC yr$^{-1}$]")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Net Ocean C Flux [PgC yr$^{-1}$]")

    lines = []
    labels = []

    lines, labels = ax1.get_legend_handles_labels()

    fig.legend(lines, labels,
               loc='upper left', bbox_to_anchor=(1, 1))
    plot_path = get_plot_filename('carbonsinks_timeseries', cfg)

    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    # Write netcdf file TODO! Currently just dummy
    #netcdf_path = get_diagnostic_filename('SCA_trend_xy_' + str(min_year) + '_' + str(max_year), cfg)
    #save_scalar_data(rel_trend_sca, netcdf_path, var_attrs)
    #netcdf_path = write_data(cfg, hist_cubes, pi_cubes, ecs_cube)

    # Provenance
    #provenance_record = get_provenance_record()
    #if plot_path is not None:
    #    provenance_record['plot_file'] = plot_path
    #with ProvenanceLogger(cfg) as provenance_logger:
    #    provenance_logger.log(netcdf_path, provenance_record)

if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
