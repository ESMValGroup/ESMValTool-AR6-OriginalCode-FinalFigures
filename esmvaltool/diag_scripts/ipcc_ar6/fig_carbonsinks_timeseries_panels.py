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
from esmvaltool.diag_scripts.shared.iris_helpers import unify_1d_cubes
from matplotlib import rcParams
import numpy as np

logger = logging.getLogger(os.path.basename(__file__))

def get_provenance_record():
    """Create a provenance record describing the diagnostic data and plot."""
    record = {
        'caption':
        ('Timeseries for carbon sinks, temperature anomaly and carbon content'
         ' of the atmosphere in the historical period.'),
        'statistics': ['mean', 'anomaly'],
        'domains': ['global'],
        'plot_types': ['times'],
        'authors': ['gier_bettina'],
        'references': ['acknow_project'],
    }
    return record

def write_data(data, plot_path, cfg):
    print(data)
    vars = extract_variables(cfg)
    vars['tasa']['standard_name'] = 'air_temperature_anomaly'
    input_data = list(cfg['input_data'].values())
    path = get_diagnostic_filename('fig_3_30_carbonsinks_timeseries', cfg)

    # Make cubelist
    var_cubes = []
    for var in data:
        print(vars[var])
        print(data[var])
        datasets = list(data[var].keys())
        var_cubelist = iris.cube.CubeList(list(data[var].values()))
        var_cubelist =  unify_1d_cubes(var_cubelist, 'year')
        data_var = [c.data for c in var_cubelist]
        dataset_coord = iris.coords.AuxCoord(datasets, long_name = 'dataset')
        coord = var_cubelist[0].coord('year')
        vars[var]['var_name'] = vars[var].pop('short_name')
        var_cubes.append(iris.cube.Cube(np.ma.array(data_var),
                              aux_coords_and_dims=[(dataset_coord, 0),
                                                   (coord, 1)],
                              **vars[var]))

    io.iris_save(iris.cube.CubeList(var_cubes), path)

    provenance_record = get_provenance_record()
    provenance_record['plot_file'] = plot_path
    provenance_record['ancestors'] = list(group_metadata(input_data,
                                                    'filename').keys())
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(path, provenance_record)

def main(cfg):
    """Run the diagnostic."""

    n_cycle_models = ["ACCESS-ESM1-5", "BNU-ESM", "CESM1-BGC",
                      "NorESM1-ME", "UKESM1-0-LL",
                      "NorESM2-LM", "EC-Earth3-Veg", "EC-Earth3-CC",
                      "CESM2", "CESM2-WACCM",
                      "SAM0-UNICON", "MIROC-ES2L", "MPI-ESM1-2-LR"]
    legend_items = {}

    tas_data = select_metadata(cfg['input_data'].values(), short_name="tas")
    tas_data += select_metadata(cfg['input_data'].values(),
                                    short_name="tasa")
    nbp_data = select_metadata(cfg['input_data'].values(), short_name="nbp")
    fgco2_data = select_metadata(cfg['input_data'].values(),
                                 short_name="fgco2")
    co2_data = select_metadata(cfg['input_data'].values(), short_name="co2s")
    plot_data = {'co2s': {}, 'tasa': {}, 'fgco2': {}, 'nbp': {}}

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
            cube = cube.collapsed(['longitude', 'latitude'],
                                  iris.analysis.MEAN)
            plot_data['co2s'][name] = cube
            ax1.plot(cube.coord("year").points, cube.data, color="black",
                     label = "OBS", linewidth = 1.5)
        else:
            print(cube.data[-1])
            style = plot.get_dataset_style(name, 'cmip6_ipcc')
            legend_items[name] = {'color': style['color'],
                                  'linewidth': style['thick']}

            plot_data['co2s'][name] = cube
            ax1.plot(cube.coord("year").points, cube.data,
                     color = style['color'],
                     label = name, linestyle = "-",
                     linewidth = legend_items[name]['linewidth'])

    n = 0
    hist = 0
    esmhist = 0

    for data in tas_data:
        # Deal with co2 read-in differently!
        name = data['dataset']
        logger.info("Processing %s", name)
        cube = iris.load_cube(data['filename'])
        iris.coord_categorisation.add_year(cube, 'time')
        cube = cube.aggregated_by('year', iris.analysis.MEAN)
        if  name!="HadCRUT5" and data['exp'] == 'historical':
            plot_data['tasa'][name+"_historical"] = cube
            linestyle = "--"
            if hist == 0:
                historical = cube.data
                hist = 1
            else:
                historical += cube.data
            n += 1
        elif name!="HadCRUT5" and data['exp'] == 'esm-hist':
            plot_data['tasa'][name] = cube
            linestyle = "-"
            if esmhist == 0:
                esmhistorical = cube.data
                esmhist = 1
            else:
                esmhistorical += cube.data
        if name=="HadCRUT5":
            plot_data['tasa'][name] = cube
            time = cube.coord("year").points
            ax2.plot(cube.coord("year").points, cube.data, color="black",
                     label = "OBS", linewidth = 1.5, zorder= 100)
            ax2.axhline(color="grey", linestyle = "--")
        else:
            ax2.plot(cube.coord("year").points, cube.data,
                     color = legend_items[name]['color'],
                     linestyle = linestyle, label = name,
                     linewidth = legend_items[name]['linewidth'])

    # Compute tas MMMs and plot:
    ax2.plot(time, esmhistorical/n, color=legend_items["MultiModelMean"]['color'],
             linewidth = legend_items["MultiModelMean"]['linewidth'], linestyle="-")
    ax2.plot(time, historical/n, color=legend_items["MultiModelMean"]['color'],
             linewidth = legend_items["MultiModelMean"]['linewidth'], linestyle="--")

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
            plot_data['nbp'][name] = cube
            ax3.plot(cube.coord("year").points, cube.data, color="black",
                     label = "OBS", linewidth = 1.5, zorder= 100)
            ax3.fill_between(cube.coord("year").points, cube.data - 0.6, cube.data + 0.6,
                             color = "black", alpha = 0.2, zorder = 101)

            ax3.axhline(color="grey", linestyle = "--")
        else:
            plot_data['nbp'][name] = cube
            ax3.plot(cube.coord("year").points, cube.data,
                     color = legend_items[name]['color'],
                     linestyle = "-", label = name,
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
            cube = cube.collapsed(['longitude', 'latitude'],
                                  iris.analysis.MEAN)
            cube.data = cube.data * 360000000000000. #multiply by area
            plot_data['fgco2'][name] = cube
            ax4.plot(cube.coord("year").points, cube.data, color="black",
                     label = "OBS", linewidth = 1.5, zorder= 100)
            ax4.fill_between(cube.coord("year").points, cube.data - 0.5,
                             cube.data + 0.5,
                             color = "black", alpha = 0.2, zorder= 101)

            ax4.axhline(color="grey", linestyle = "--")
        else:
            cube_ini = cube.extract(iris.Constraint(year=1850))
            cube = cube - cube_ini
            plot_data['fgco2'][name] = cube
            ax4.plot(cube.coord("year").points, cube.data,
                     color = legend_items[name]['color'],
                     linestyle = "-", label = name,
                     linewidth = legend_items[name]['linewidth'])

    #tick_params(labelright=True)
    ax1.set_xlim(1850, 2014)
    #ax1.set_ylim(320, 420)
    ax2.set_xlim(1850, 2014)
    ax3.set_xlim(1850, 2014)
    ax4.set_xlim(1850, 2014)
    #ax4.set_ylim(0.5, 3.2)
    ax1.set_xlabel("Year")
    ax1.set_ylabel(r"Atmospheric CO$_2$ [ppmv]")
    ax1.yaxis.set_ticks_position('both')
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Temperature anomaly [°C]")
    ax2.yaxis.set_ticks_position('both')
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Net Land C Flux [PgC yr$^{-1}$]")
    ax3.yaxis.set_ticks_position('both')
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Net Ocean C Flux [PgC yr$^{-1}$]")
    ax4.yaxis.set_ticks_position('both')

    ax1.text(0.00, 1.15, 'a', transform=ax1.transAxes,
             fontsize=14, va='top', ha='right')
    ax2.text(0.00, 1.15, 'b', transform=ax2.transAxes,
             fontsize=14, va='top', ha='right')
    ax3.text(0.00, 1.15, 'c', transform=ax3.transAxes,
             fontsize=14, va='top', ha='right')
    ax4.text(0.00, 1.15, 'd', transform=ax4.transAxes,
             fontsize=14, va='top', ha='right')

    lines = []
    labels = []

    lines, labels = ax1.get_legend_handles_labels()

    fig.legend(lines, labels,
               loc='upper left', bbox_to_anchor=(1, 0.92))
    plot_path = get_plot_filename('fig_3_31_carbonsinks_timeseries', cfg)
    fig.suptitle("Carbon sinks in CMIP6 emission driven simulations")
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches='tight', dpi = 300)
    plt.close(fig)

    write_data(plot_data, plot_path, cfg)

if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
