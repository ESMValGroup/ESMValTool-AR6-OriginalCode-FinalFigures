import iris
from iris.experimental.equalise_cubes import equalise_attributes
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats, special
import sys

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def select_months(cubelist,month):

    month_constr = iris.Constraint(time=lambda cell: cell.point.month == month)

    cropped_cubelist = iris.cube.CubeList()

    for cube in cubelist:
        cropped_cube=cube.extract(month_constr)
        cropped_cubelist.append(cropped_cube)

    return (cropped_cubelist)

def select_latitudes(cubelist, start_lat=-90, end_lat=90):

    # possibly add here a warning about start_ and end_lat

    lat_constr=iris.Constraint(latitude=lambda cell: start_lat < cell <= end_lat)

    cropped_cubelist = iris.cube.CubeList()

    for n, cube in enumerate(cubelist):
        cropped_cube=cube.extract(lat_constr)
        cropped_cubelist.append(cropped_cube)

    return (cropped_cubelist)


def load_cubelist(filenames):

    cubelist = iris.cube.CubeList()

    for n, filename in enumerate(filenames):
        cube=iris.load_cube(filename)
        cubelist.append(cube)

    return (cubelist)


def calculate_siextent(cubelist, threshold=15):

    # calculates siextent for the hemisphere
    # creates a cubelist with only one dimension: 'time'

    for n, cube in enumerate(cubelist):

        if cube is None:
            continue

        area=iris.analysis.cartography.area_weights(cube,normalize=False)
        time = cube.coord('time')

        mask = (cube.data.mask) | (cube.data.data<=threshold)

        area=np.ma.array(area,mask=mask)

        # not beautiful but it works: here we create a new cube, where data is a sum of area covered by
        # sea ice in the whole space that was provided. The only coordinate is time, taken from the original
        # cube. Might be corrected in the future.
        area=(area.sum(axis=(1, 2))/(1000**2))/1000000
        #for now passing paren cube attributes, clean before merging!!!
        new_cube = iris.cube.Cube(area, standard_name='sea_ice_extent',long_name='sea ice extent', var_name='siextent', units= "10^6km2", attributes=cube.attributes, dim_coords_and_dims=[(time, 0)])

        if n==0:
            conv_cubelist=iris.cube.CubeList([new_cube])
        else:
            conv_cubelist.append(new_cube)

    return (conv_cubelist)

def calculate_siarea(cubelist):

    sia_cubelist = iris.cube.CubeList()
    for cube in cubelist:
        area = iris.analysis.cartography.area_weights(cube, normalize=False)
        time = cube.coord('time')

        siconc = cube.data / 100 # since the data is in %, it has to be converted into fraction
        area = np.ma.array(area, mask=cube.data.mask)

        sia_arr = siconc * area

        sia = (sia_arr.sum(axis=(1, 2)) / (1000 ** 2)) / 1000000 # iris provides area in m, converting it to 10^6km2
        # for now passing parent cube attributes, clean before merging!!!
        sia_cube = iris.cube.Cube(sia, standard_name='sea_ice_area', long_name='sea ice area', var_name='siarea',
                                  units="1e6 km2", attributes=cube.attributes, dim_coords_and_dims=[(time, 0)])
        sia_cubelist.append(sia_cube)

    return (sia_cubelist)

def calculate_siparam(cubelist, siext=True):
    # function which determines if sea ice extent or sea ice are should be calculated

    if siext:
        cubelist=calculate_siextent(cubelist)
    else:
        cubelist=calculate_siarea(cubelist)

    return (cubelist)

def n_year_mean(cubelist, n):

    # the idea behind it is that we pass the cubelist with the same time coords

    n_aver_cubelist = iris.cube.CubeList()

    dcoord = create_coords(cubelist, n)

    for cube in cubelist:
        n_t = len(cube.coord('time').points)
        if n_t%n!=0:
            # add here a warning that the last is an average of n_t%n==0 years
            logger.info('The n of years is not divisible by %s last %s years were not taken into account',
                        str(n), str(n_t%n))
        if len(cube.data.shape) == 1:
            data = [np.average(cube.data[n*i:n*i + n]) for i in range(0, int(n_t / n))]
            n_aver_cube = iris.cube.Cube(np.asarray(data), long_name=cube.long_name + ', ' + str(n) + 'y mean',
                                         var_name=cube.var_name, units=cube.units,
                                         attributes=cube.attributes, dim_coords_and_dims=[(dcoord, 0)])
        elif len(cube.data.shape) == 2:
            data = np.asarray([np.average(cube.data[n * i:n * i + n, :], axis=0) for i in range(0, int(n_t / n))])
            n_aver_cube =iris.cube.Cube(data, long_name=cube.long_name + ', ' + str(n) + 'y mean',
                                                 var_name=cube.var_name, units=cube.units, attributes=cube.attributes,
                                                 dim_coords_and_dims=[(dcoord,0), (cube.coords()[1],1)])
        elif len(cube.data.shape) == 3:
            data = np.asarray([np.average(cube.data[n * i:n * i + n, :, :], axis=0) for i in range(0, int(n_t / n))])
            n_aver_cube = iris.cube.Cube(data, long_name=cube.long_name + ', ' + str(n) + 'y mean',
                                         var_name=cube.var_name, units=cube.units, attributes=cube.attributes,
                                         dim_coords_and_dims=[(dcoord, 0), (cube.coords()[1], 1), (cube.coords()[2],2)])

        n_aver_cubelist.append(n_aver_cube)

    return (n_aver_cubelist)

def create_coords(cubelist, year_n):
    # dirty trick, we try to unify time to merge  the cubelist in the end

    cb = cubelist [0]

    n_t = len(cb.coord('time').points)
    coord = [np.average(cb.coord('time').points[year_n*i:year_n*i + year_n]) for i in range(0, int(n_t / year_n))]
    bnds = [[cb.coord('time').bounds[year_n*i][0], cb.coord('time').bounds[year_n*i + (year_n - 1)][1]] for i in
            range(0, int(n_t / year_n))]
    if n_t%year_n != 0:
        # raise warning
        logger.info('The n of years is not divisible by %s', str(year_n))
        # coord.append(np.average(cb.coord('time').points[int(n_t / year_n):-1]))
        # bnds.append([cb.coord('time').bounds[int(n_t / year_n) * year_n][0], cb.coord('time').bounds[-1][1]])

    dcoord = iris.coords.DimCoord(np.asarray(coord), bounds=np.asarray(bnds),
                                  standard_name=cb.coord('time').standard_name,
                                  units=cb.coord('time').units, long_name=cb.coord('time').long_name,
                                  var_name=cb.coord('time').var_name)

    return (dcoord)


def substract_ref_period(cubelist, ref_period):

    constr = iris.Constraint(time=lambda cell: ref_period[0] <= cell.point.year <= ref_period[1])

    upd_cubelist = iris.cube.CubeList()

    for cube in cubelist:
        mean = cube.extract(constr).collapsed('time', iris.analysis.MEAN)
        upd_cube = cube - mean
        upd_cube.attributes = cube.attributes
        upd_cube.long_name = cube.long_name + ' anomaly'
        upd_cube.var_name = cube.var_name + '_ano'
        upd_cubelist.append(upd_cube)

    return(upd_cubelist)

def figure_handling(cfg, name = 'plot', img_ext=None):

    if cfg['write_plots']:

        if img_ext == None:
            img_ext = diagtools.get_image_format(cfg)

        path=os.path.join(cfg['plot_dir'], name + img_ext)

        logger.info('Saving plots to %s', path)
        plt.savefig(path)

    else:

        plt.show()

    return
