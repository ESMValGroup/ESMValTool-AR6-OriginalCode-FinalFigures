"""
Model vs Observations maps Diagnostic.
======================================

Diagnostic to produce comparison of model and data.
The first kind of image shows four maps and the other shows a scatter plot.

The four pane image is a latitude vs longitude figures showing:

* Top left: model
* Top right: observations
* Bottom left: model minus observations
* Bottom right: model over observations


The scatter plots plot the matched model coordinate on the x axis, and the
observational dataset on the y coordinate, then performs a linear
regression of those data and plots the line of best fit on the plot.
The parameters of the fit are also shown on the figure.

Note that this diagnostic assumes that the preprocessors do the bulk of the
hard work, and that the cube received by this diagnostic (via the settings.yml
and metadata.yml files) has no time component, a small number of depth layers,
and a latitude and longitude coordinates.

An approproate preprocessor for a 3D + time field would be::

  preprocessors:
    prep_map:
      extract_levels:
        levels:  [100., ]
        scheme: linear_extrap
      time_average:
      regrid:
        target_grid: 1x1
        scheme: linear

This tool is part of the ocean diagnostic tools package in the ESMValTool,
and was based on the plots produced by the Ocean Assess/Marine Assess toolkit.

Author: Lee de Mora (PML)
        ledm@pml.ac.uk
"""
import logging
import os
import sys
import math

from matplotlib import pyplot
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import iris
import iris.quickplot as qplt
import numpy as np
from scipy.stats import linregress
import cartopy

from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.shared import run_diagnostic

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def add_map_subplot(subplot, cube, nspace, title='', cmap='', log=False, cbar=True):
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
    log: bool
        Flag to plot the colour scale linearly (False) or
        logarithmically (True)
    """
    plt.subplot(subplot)
    logger.info('add_map_subplot: %s', subplot)
    if log:
        qplot = qplt.contourf(
            cube,
            nspace,
            linewidth=0,
            cmap=plt.cm.get_cmap(cmap),
            norm=LogNorm(),
            zmin=nspace.min(),
            zmax=nspace.max(),
            rasterized=True)
        if cbar: qplot.colorbar.set_ticks([0.1, 1., 10.])
    else:
        qplot = iris.plot.contourf(
            cube,
            nspace,
            linewidth=0,
            cmap=plt.cm.get_cmap(cmap),
            zmin=nspace.min(),
            zmax=nspace.max(),
            rasterized=True)
        if cbar:
            cbar = pyplot.colorbar(orientation='horizontal')
            cbar.set_ticks(
                [nspace.min(), (nspace.max() + nspace.min()) / 2.,
                 nspace.max()])

    plt.gca().add_feature(cartopy.feature.LAND, facecolor=[0.8, 0.8, 0.8])
    plt.gca().coastlines()
    #plt.title(title)



def main(cfg):
    """
    Load the config file, and send it to the plot maker.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.

    """
    #print(cfg)
    settings_file = cfg['input_files']
    metadatas = diagtools.get_input_files(cfg)
    metadata_files = metadatas.keys()



    model_metadatas = {}
    for fn in metadata_files:
        print('\nmetadata_file:', fn)
        print(metadatas[fn])

        if metadatas[fn]['dataset'] == 'CORA':
            continue # Obs
        if metadatas[fn]['start_year'] == metadatas[fn]['end_year'] == 1950:
            model_metadatas['1950'] = fn
        if metadatas[fn]['start_year'] == metadatas[fn]['end_year'] == 2000:
            model_metadatas['2000'] = fn

        if metadatas[fn]['start_year'] == metadatas[fn]['end_year'] == 2050:
            model_metadatas['2050'] = fn
        if metadatas[fn]['start_year'] == metadatas[fn]['end_year'] == 2090:
            model_metadatas['2099'] = fn

        if metadatas[fn]['start_year'] == 1950 and metadatas[fn]['end_year'] == 1955:
            model_metadatas['1950-1955'] = fn

        if metadatas[fn]['start_year'] == 2085 and metadatas[fn]['end_year'] == 2089:
            model_metadatas['2095-2099'] = fn

    # Load image format extention
    image_extention = diagtools.get_image_format(cfg)

    print(model_metadatas)
    model_cubes = {}
    for name, model_fn in model_metadatas.items():
        cube = iris.load_cube(model_fn)
        model_cubes[name] = cube

    thresholds =[33., 33.5, 34., 34.5, 35., 35.5]
    linestyles = ['-' for thres in thresholds]
    colours = ['k' for thres in thresholds]
    linewidths = [1., 0.5, 1., 0.5, 1., 0.5,]




    fig = plt.figure()
    fig.set_size_inches(10, 6)

    # Create the cubes
    # cube221 = cubes['model'][layer]
    # cube222 = cubes['obs'][layer]
    cube223 = model_cubes['2000'] - model_cubes['1950']
    cube223_contours = model_cubes['1950-1955']
    cube224 = model_cubes['2099'] - model_cubes['2050']
    cube224_contours = model_cubes['2095-2099']


    # create the z axis for plots 2, 3, 4.
    # zrange12 = diagtools.get_cube_range([cube221, cube222])
    # cube224.data = np.ma.clip(cube224.data, 0.1, 10.)

    n_points = 16
    # linspace12 = np.linspace(
    #     zrange12[0], zrange12[1], n_points, endpoint=True)
    linspace_all = np.linspace(-0.201, 0.201, n_points, endpoint=True)

    # Add the sub plots to the figure.
    # add_map_subplot(221, cube221, linspace12, cmap='viridis', title=model)
    # add_map_subplot(
    #     222, cube222, linspace12, cmap='viridis', title=' '.join([
    #         obs,
    #     ]))


    # add_map_subplot(
    #     223,
    #     cube223,
    #     linspace_all,
    #     cmap='bwr',
    #     title='',
    #     cbar = False)
    sp221 = plt.subplot(221)
    qplot = iris.plot.contourf(
        cube223,
        linspace_all,
        linewidth=0,
        cmap=plt.cm.get_cmap('bone'),
        zmin=linspace_all.min(),
        zmax=linspace_all.max(),
        rasterized=True)
    plt.gca().add_feature(cartopy.feature.LAND, facecolor=[0.8, 0.8, 0.8])
    plt.gca().coastlines()

    sp222 = plt.subplot(222)
    qplot = iris.plot.contourf(
        cube223,
        linspace_all,
        linewidth=0,
        cmap=plt.cm.get_cmap('bone'),
        zmin=linspace_all.min(),
        zmax=linspace_all.max(),
        rasterized=True)
    plt.gca().add_feature(cartopy.feature.LAND, facecolor=[0.8, 0.8, 0.8])
    plt.gca().coastlines()

    print(linspace_all)
    sp223 = plt.subplot(223)
    qplot = iris.plot.contourf(
        cube223,
        linspace_all,
        linewidth=0,
        cmap=plt.cm.get_cmap('bwr'),
        zmin=linspace_all.min(),
        zmax=linspace_all.max(),
        rasterized=True,
        extend='neither',
        )
    print('cube223:',cube223.data.min(), cube223.data.max())
    print('cube224:',cube224.data.min(), cube224.data.max())

    # TODO: problem here is that the contourf plot just ignores the linspace and zmin/zmax ranges.
    
    pyplot.colorbar()

    iris.plot.contour(cube223_contours,
                 thresholds,
                 colors=colours,
                 linewidths=linewidths,
                 linestyles=linestyles,
                 rasterized=True,
                 extend='both',
                 )
    plt.gca().add_feature(cartopy.feature.LAND, facecolor=[0.8, 0.8, 0.8])
    plt.gca().coastlines()

    sp224 = plt.subplot(224)
    mesh41 = qplot = iris.plot.contourf(
        cube224,
        linspace_all,
        linewidth=0,
        cmap=plt.cm.get_cmap('bwr'),
        zmin=linspace_all.min(),
        zmax=linspace_all.max(),
        extend='both',
        rasterized=True)
    pyplot.colorbar()

    mesh42 = iris.plot.contour(cube224_contours,
                 thresholds,
                 colors=colours,
                 linewidths=linewidths,
                 linestyles=linestyles,
                 rasterized=True,
                 extend='both',
                 )
    plt.gca().add_feature(cartopy.feature.LAND, facecolor=[0.8, 0.8, 0.8])
    plt.gca().coastlines()

    # fig.suptitle(long_name, fontsize=14)
    # Add overall title

    # fig.subplots_adjust(hspace=0.01, wspace=0.01)
    #
    # fig.subplots_adjust(right=0.9)
    # cb_ax = fig.add_axes([0.93, 0.02, 0.02, 0.9])
    # cbar = fig.colorbar(mesh41, cax=cb_ax)
    # plt.colorbar() #img, cax=cbar_ax)


    # Determine image filename:
    path = diagtools.folder(cfg['plot_dir']) + 'fig_3_21' + image_extention

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        plt.savefig(path , )#bbox_inches = 'tight',)

    plt.close()


    logger.info('Success')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
