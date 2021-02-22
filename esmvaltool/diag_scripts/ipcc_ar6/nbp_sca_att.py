#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic script to plot figure 9.42a of IPCC AR5 chapter 9.

Description
-----------
Attribute parts of SCA of NBP

Author
------
Bettina Gier (University of Bremen, Germany)

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
from scipy.ndimage import gaussian_filter
from scipy import stats
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
import pandas as pd
logger = logging.getLogger(os.path.basename(__file__))

def plot_barstacked(data, filename):
    my_data = pd.DataFrame(data=data)
    my_data = my_data.T

    all_mean = [data[k]["All"][0] for k in data.keys()]
    all_std = [data[k]["All"][1] for k in data.keys()]

    # my_data = pd.DataFrame(data = data)
    CMIP6_colors = [(0, 0, 0), (112 / 255., 160 / 255., 205 / 255.), (178 / 255., 178 / 255., 178 / 255.)]

    fig, ax = plt.subplots()
    my_data.plot.barh(ax=ax, legend=True, figsize=(10, 7), stacked=True, y=["CO2", "Climate", "Land Use"],
                      color=CMIP6_colors)
    ax.errorbar(all_mean, np.arange(len(data)), xerr=all_std, color="red", fmt='o', markersize=10,
                elinewidth=2,  # width of error bar line
                # ecolor='k',    # color of error bar
                capsize=5,  # cap length for error bar
                capthick=2  # cap thickness for error bar)
                )

    plt.xlabel(r'NBP seasonal amplitude trend [% yr$^{-1}$]', fontsize=17)
    plt.axvline(x=0, color = "black", linestyle = "--")
    plt.tick_params(axis='x', labelsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.legend(loc="best", prop={'size': 17})
    plt.savefig(filename,bbox_inches='tight')
    plt.close(fig)

def get_provenance_record():
    """Create a provenance record describing the diagnostic data and plot."""
    record = {
        'caption':
        ('Attribution of NBP SCA to CO2, Climate, Land-uSe.(Zhao et al., 2016).'),
        'statistics': ['mean', 'stddev', 'trend'],
        'domains': ['global'],
        'plot_types': ['bar'],
        'authors': ['gier_bettina'],
        'references': ['zhao2016bg'],
        'realms': ['atmos'],
        'themes': ['phys']
    }
    return record

def main(cfg):
    """Run the diagnostic."""
    #logger.info(cfg['input_data'].keys())
    trend_sca = {}
    data_dict = {}

    input_data = cfg['input_data'].values()
    #logger.info(input_data)
    for data in input_data:
        name = data['dataset']
        logger.info("Processing %s", name)
        cube_data = iris.load_cube(data['filename'])

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
        dataset_yrs, amps, _, _, _, _ = list(zip(*sca))

        # Subtract means, compute relative change in amplitude
        # 9 year gaussian smoothing
        # window length: w = 2 * int(truncate*sigma + 0.5) + 1
        gaus_amps = gaussian_filter(amps, 1.2, truncate=3.0)
        sca_mean_smooth = np.mean(gaus_amps[:11])
        # relative trends
        rel_trend_sca = gaus_amps/sca_mean_smooth

        # Do linear regression on relative trend
        sca_linreg = stats.linregress(dataset_yrs, rel_trend_sca)
        if name in trend_sca:
            trend_sca[name][data.get("exp")] = [sca_linreg.slope*100, sca_linreg.stderr*100]
        else:
            trend_sca[name] = {data.get("exp"): [sca_linreg.slope*100, sca_linreg.stderr*100]}

    for dataset in trend_sca:
        data_dict[dataset] = {"All":  trend_sca[dataset]["historical"]}

        data_dict[dataset]["Land Use"] = trend_sca[dataset]["historical"][0] - \
                                         trend_sca[dataset]["hist-noLu"][0]

        data_dict[dataset]["Climate"] = trend_sca[dataset]["historical"][0] - \
                                        trend_sca[dataset]["hist-bgc"][0]

        data_dict[dataset]["CO2"] = trend_sca[dataset]["hist-bgc"][0] + \
                                    trend_sca[dataset]["hist-noLu"][0] - \
                                    trend_sca[dataset]["historical"][0]

    # Plot data
    plot_path = get_plot_filename('SCA_attr_barplot', cfg)
    plot_barstacked(data_dict, plot_path)

    # Write netcdf file TODO! Currently just dummy
    netcdf_path = get_diagnostic_filename('SCA_attr', cfg)
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
