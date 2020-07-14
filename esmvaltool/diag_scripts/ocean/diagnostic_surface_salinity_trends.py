"""
Figure surface_salinity_trends (3.26 - previously 3.21)
======================================

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
import cartopy.crs as ccrs

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
    assert 0
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


def load_cube_221():
    # the 1950-2000 observational change.
    path = "/data/sthenno1/scratch/ledm/Observations/CSIRO/"
    path += "DurackandWijffels_GlobalOceanSurfaceChanges_1950-2000.nc"
    # Downloaded from https://www.cmar.csiro.au/cgi-bin/oceanchange?file=DurackandWijffels_GlobalOceanSurfaceChanges_1950-2000.nc
    # 2019-04-08
    cubes = iris.load(path)
    cube_out = ''
    for cube in cubes:
        if cube.standard_name == 'change_over_time_in_sea_water_practical_salinity':
            cube_out = cube
    cube_out = cube_out.collapsed('time', iris.analysis.MEAN)
    cube_out = cube_out.extract(iris.Constraint(sea_water_pressure=0.))
    return cube_out


def load_cube_222():
    # the 1950-2000 observational climatolocal mean.
    path = "/data/sthenno1/scratch/ledm/Observations/CSIRO/"
    path += "DurackandWijffels_GlobalOceanSurfaceChanges_1950-2000.nc"
    # Downloaded from https://www.cmar.csiro.au/cgi-bin/oceanchange?file=DurackandWijffels_GlobalOceanSurfaceChanges_1950-2000.nc
    # 2019-04-08
    cubes = iris.load(path)
    cube_out = ''
    for cube in cubes:
        if cube.standard_name == 'sea_water_practical_salinity':
            cube_out = cube
    cube_out = cube_out.collapsed('time', iris.analysis.MEAN)
    cube_out = cube_out.extract(iris.Constraint(sea_water_pressure=0.))
    return cube_out

def load_cube_trends():
    # the 1950-2000 observational climatolocal mean.
    path = "/data/sthenno1/scratch/ledm/Observations/CSIRO/"
    path += "DurackandWijffels_GlobalOceanSurfaceChanges_1950-2000.nc"
    # Downloaded from https://www.cmar.csiro.au/cgi-bin/oceanchange?file=DurackandWijffels_GlobalOceanSurfaceChanges_1950-2000.nc
    # 2019-04-08

    cubes = iris.load(path)
    cube_out = ''
    for cube in cubes:
        if cube.long_name == 'Salinity change error 1950-2000':
            cube_out = cube
    cube_out = cube_out.collapsed('time', iris.analysis.MEAN)
    cube_out = cube_out.extract(iris.Constraint(sea_water_pressure=0.))
    return cube_out

def add_coasts():
    plt.gca().add_feature(cartopy.feature.LAND, facecolor=[0.8, 0.8, 0.8])
    plt.gca().coastlines()


def make_mean_of_cube_list(cube_list):
    """
    Takes the mean of a list of cubes (not an iris.cube.CubeList).

    Assumes all the cubes are the same shape.
    """
    cube_mean = cube_list[0]
    for cube in cube_list[1:]:
        cube_mean+=cube
    cube_mean = cube_mean/ float(len(cube_list))
    cube_mean.units = '1.0'
    return cube_mean


def prepare_model_metadata(cfg):
    metadatas = diagtools.get_input_files(cfg)
    metadata_files = metadatas.keys()
    model_metadatas = {}
    model_cubes = {}

    ##########################
    # make list of filenames for each years.
    for fn in metadata_files:
        print('\nmetadata_file:', fn)
        print(metadatas[fn])
        for year in [1950, 2000, 2050, 2099]:
            if metadatas[fn]['start_year'] == metadatas[fn]['end_year'] == year:
                try:
                    model_metadatas[str(year)].append(fn)
                except:
                    model_metadatas[str(year)] = [fn,]

        if metadatas[fn]['start_year'] == 1950 and metadatas[fn]['end_year'] == 2000:
            try:
                model_metadatas['1950-2000'].append(fn)
            except:
                model_metadatas['1950-2000'] = [fn,]

        if metadatas[fn]['start_year'] == 2050 and metadatas[fn]['end_year'] == 2099:
            try:
                model_metadatas['2050-2099'].append(fn)
            except:
                model_metadatas['2050-2099'] = [fn,]

    ##########################
    # convert list of fn to multi model mean.
    for name, model_fns in model_metadatas.items():
        cubes = []
        for model_fn in model_fns:
            cube = iris.load_cube(model_fn)
            cubes.append(cube)
        model_cubes[name] = make_mean_of_cube_list(cubes)

    return model_cubes


def main(cfg):
    """
    Load the config file, and send it to the plot maker.

    Parameters
    ----------
    cfg: dict
        the opened global config dictionairy, passed by ESMValTool.

    """
    metadatas = diagtools.get_input_files(cfg)

    # Load image format extention
    image_extention = diagtools.get_image_format(cfg)

    model_cubes = prepare_model_metadata(cfg)


    thresholds =[33., 33.5, 34., 34.5, 35., 35.5, 36., 36.5, 37.]
    levels =[33., 34., 35., 36., 37.]
    linestyles = ['-' for thres in thresholds]
    colours = ['k' for thres in thresholds]
    linewidths = [1., 0.5, 1., 0.5, 1., 0.5, 1., 0.5, 1.,]

    thresholds_white = np.arange(-2, 2.25, 0.25)
    change_ticks = [-0.2, -0.1, 0., 0.1, 0.2]
    linestyles_white = ['-' for thres in thresholds_white]
    colours_white = ['w' for thres in thresholds_white]
    linewidths_white = [0.5 for thres in thresholds_white]

    fig = plt.figure()
    fig.set_size_inches(10, 6)

    # Create the cubes
    cube221 = load_cube_221()
    cube222 = load_cube_222()
    cube221_contours = cube222
    cube222_contours = cube222
    cube221_contours_white = cube221 # load_cube_trends()
    cube222_contours_white = cube221

    cube223 = model_cubes['2000'] - model_cubes['1950']
    cube224 = model_cubes['2099'] - model_cubes['2050']
    cube223_contours = model_cubes['1950-2000']
    cube224_contours = model_cubes['2050-2099']
    cube223_contours_white = cube223
    cube224_contours_white = cube224

    n_points = 9
    lims = [34., 37.]
    linspace_sal = np.linspace(lims[0], lims[1], n_points*3, endpoint=True)
    lims = [-0.2, 0.2]
    linspace_change = np.linspace(lims[0], lims[1], n_points, endpoint=True)

    #############################################
    # 221
    sp221 = plt.subplot(221, projection=ccrs.PlateCarree(central_longitude=180.0))
    pyplot.title('A: Obs. change')
    qplot = iris.plot.contourf(
        cube221,
        linspace_change,
        linewidth=0,
        cmap=plt.cm.get_cmap('RdYlBu_r'),
        extend='both',
        rasterized=True)
    add_coasts()
    cbar = pyplot.colorbar(orientation='horizontal', ticks=change_ticks)

    c221 = iris.plot.contour(cube221_contours,
                 thresholds,
                 colors=colours,
                 linewidths=linewidths,
                 linestyles=linestyles,
                 rasterized=True,
                 extend='both',
                 )
    sp221.clabel(c221, levels, inline=True, fontsize=8, fmt = '%1.0f')

    c221w = iris.plot.contour(cube221_contours_white,
                 thresholds_white,
                 colors=colours_white,
                 linewidths=linewidths_white,
                 linestyles=linestyles_white,
                 rasterized=True,
                 extend='both',
                 )

    #############################################
    # 222
    sp222 = plt.subplot(222, projection=ccrs.PlateCarree(central_longitude=180.0))
    pyplot.title('B: Obs. mean')
    qplot = iris.plot.contourf(
        cube222,
        linspace_sal,
        linewidth=0,
        cmap=plt.cm.get_cmap('RdYlBu_r'),
        extend='both',
        rasterized=True)
    cbar = pyplot.colorbar(orientation='horizontal', ticks=levels)
    add_coasts()

    c222 = iris.plot.contour(cube222_contours,
                 thresholds,
                 colors=colours,
                 linewidths=linewidths,
                 linestyles=linestyles,
                 rasterized=True,
                 extend='both',
                 )
    sp222.clabel(c222, levels, inline=True, fontsize=8, fmt = '%1.0f')
    # c222w = iris.plot.contour(cube222_contours_white,
    #              thresholds_white,
    #              colors=colours_white,
    #              linewidths=linewidths_white,
    #              linestyles=linestyles_white,
    #              rasterized=True,
    #              extend='both',
    #              )

    #############################################
    # 223
    sp223 = plt.subplot(223, projection=ccrs.PlateCarree(central_longitude=180.0))
    pyplot.title('C: CMIP5 hist.')
    qplot = iris.plot.contourf(
        cube223,
        linspace_change,
        linewidth=0,
        cmap=plt.cm.get_cmap('RdYlBu_r'),
        rasterized=True,
        extend='both',
        )
    cbar = pyplot.colorbar(orientation='horizontal', ticks=change_ticks)
    add_coasts()

    c223 = iris.plot.contour(cube223_contours,
                 thresholds,
                 colors=colours,
                 linewidths=linewidths,
                 linestyles=linestyles,
                 rasterized=True,
                 extend='both',
                 )
    sp223.clabel(c223, levels, inline=True, fontsize=8, fmt = '%1.0f')
    c223w = iris.plot.contour(cube223_contours_white,
                 thresholds_white,
                 colors=colours_white,
                 linewidths=linewidths_white,
                 linestyles=linestyles_white,
                 rasterized=True,
                 extend='both',
                 )

    #############################################
    # 224
    sp224 = plt.subplot(224, projection=ccrs.PlateCarree(central_longitude=180.0))
    pyplot.title('D: CMIP5 RCPs')

    mesh41 = qplot = iris.plot.contourf(
        cube224,
        linspace_change,
        linewidth=0,
        cmap=plt.cm.get_cmap('RdYlBu_r'),
        extend='both',
        rasterized=True)
    cbar = pyplot.colorbar(orientation='horizontal', ticks=change_ticks)
    add_coasts()
    c224 = iris.plot.contour(cube224_contours,
                 thresholds,
                 colors=colours,
                 linewidths=linewidths,
                 linestyles=linestyles,
                 rasterized=True,
                 extend='both',
                 )

    sp224.clabel(c224, levels, inline=True, fontsize=8, fmt = '%1.0f')
    c224w = iris.plot.contour(cube224_contours_white,
                 thresholds_white,
                 colors=colours_white,
                 linewidths=linewidths_white,
                 linestyles=linestyles_white,
                 rasterized=True,
                 extend='both',
                 )

    #############################################
    # Determine image filename:
    path = diagtools.folder(cfg['plot_dir']) + 'surface_salinity_trends' + image_extention

    # Saving files:
    if cfg['write_plots']:
        logger.info('Saving plots to %s', path)
        fig.savefig(path , )#bbox_inches = 'tight',)
        fig.savefig(path.replace('.pdf', '.png') , dpi=250 )

    plt.close()

    logger.info('Success')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
