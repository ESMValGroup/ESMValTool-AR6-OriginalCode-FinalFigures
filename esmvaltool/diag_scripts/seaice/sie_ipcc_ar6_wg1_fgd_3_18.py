import iris
from iris.experimental.equalise_cubes import equalise_attributes
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import group_metadata, run_diagnostic
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.ocean import diagnostic_seaice as diagseaice

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# #here will be the functions I need

def filter_datasets(data_dict,proj,exp_list=['historical']):

    proj_dict = {}

    for exp in exp_list:
        exp_dict= {}
        for filename in sorted(data_dict):
            if (data_dict[filename]['project'] == proj) & (data_dict[filename]['exp'] == exp):
                exp_dict[filename] = data_dict[filename]
        proj_dict[exp] = exp_dict

    # maybe add here an error message if the len(rcps and hist is not the same )
    return (proj_dict)

def concatenate_cmip_data(main_data_path_list, add_data_path_list, dim='time'):

    for n, (path_main,path_add) in enumerate(zip(main_data_path_list,add_data_path_list)):

        cube_joint=iris.cube.CubeList([iris.load_cube(path_main),iris.load_cube(path_add)])
        equalise_attributes(cube_joint)
        new_cube=cube_joint.concatenate_cube()

        if n==0:
            new_cube_list=iris.cube.CubeList([new_cube])
        else:
            new_cube_list.append(new_cube)

    return (new_cube_list)

def select_months(cubelist,month):

    month_constr = iris.Constraint(time=lambda cell: cell.point.month == month)

    for n,cube in enumerate(cubelist):
        cropped_cube=cube.extract(month_constr)
        if n==0:
            cropped_cubelist=iris.cube.CubeList([cropped_cube])
        else:
            cropped_cubelist.append(cropped_cube)

    return (cropped_cubelist)

def select_latitudes(cubelist, start_lat=-90, end_lat=90):

    # possibly add here a warning about start_ and end_lat

    lat_constr=iris.Constraint(latitude=lambda cell: start_lat < cell <= end_lat)

    for n, cube in enumerate(cubelist):
        cropped_cube=cube.extract(lat_constr)
        if n==0:
            cropped_cubelist=iris.cube.CubeList([cropped_cube])
        else:
            cropped_cubelist.append(cropped_cube)

    return (cropped_cubelist)

def load_cubelist(filenames):

    for n, filename in enumerate(filenames):
        cube=iris.load_cube(filename)
        if n==0:
            cubelist=iris.cube.CubeList([cube])
        else:
            cubelist.append(cube)

    return (cubelist)

def prepare_cmip_for_3_18(data_dict, project, month, start_lat, end_lat, concatinate=False, concat_list=None):

    # add here a warning message about start lat being smaller than end lat

    if concatinate==True:
        cmip_dict = filter_datasets(data_dict, project, concat_list)
        for i in range(0,len(cmip_dict.keys())-1):
            key1 = list(cmip_dict.keys())[i]
            key2 = list(cmip_dict.keys())[i+1]
            cmip_cubelist = concatenate_cmip_data(list(cmip_dict[key1].keys()),list(cmip_dict[key2].keys()))
    else:
        cmip_dict = filter_datasets(data_dict, project)
        key=list(cmip_dict.keys())[0]
        cmip_cubelist = load_cubelist(list(cmip_dict[key].keys()))

    cmip_cubelist = select_months(cmip_cubelist, month)
    cmip_cubelist = select_latitudes(cmip_cubelist,start_lat,end_lat)

    return (cmip_cubelist)

def calculate_siextent(cubelist, threshold=15):

    # calculates siextent for the hemisphere
    # creates a cubelist with only one dimension: 'time'

    for n, cube in enumerate(cubelist):

        area=iris.analysis.cartography.area_weights(cube,normalize=False)
        time = cube.coord('time')

        mask = (cube.data.mask) | (cube.data.data<=threshold)

        area=np.ma.array(area,mask=mask)

        # not beautiful but it works: here we create a new cube, where data is a sum of area covered by
        # sea ice in the whole space that was provided. The only coordinate is time, taken from the original
        # cube. Might be corrected in the future.
        area=area.sum(axis=(1, 2))/(1000**2)
        #for now passing paren cube attributes, clean before merging!!!
        new_cube = iris.cube.Cube(area, standard_name='sea_ice_extent',long_name='sea ice extent', var_name='siextent', units= "km2", attributes=cube.attributes, dim_coords_and_dims=[(time, 0)])

        if n==0:
            conv_cubelist=iris.cube.CubeList([new_cube])
        else:
            conv_cubelist.append(new_cube)

    return (conv_cubelist)

def calculate_siarea(cubelist):

    return (cubelist)

def calculate_siparam(cubelist, siext=True):

    # function which determines if sea ice extent or sea ice are should be calculated

    if siext:
        cubelist=calculate_siextent(cubelist)
    else:
        cubelist=calculate_siarea(cubelist)

    return (cubelist)
#

# def calculate_trend():
#
#     return
#
# def make_panel():
#
#     return

# def make_plot():
#     somewhere here the make_panel will be called
#
#     return


def main(cfg):

    metadatas = diagtools.get_input_files(cfg)

    # here we call prepapre_cmip_data

    cmip5_NH = prepare_cmip_for_3_18(metadatas, 'CMIP5', cfg['month_latitude_NH'][0], cfg['month_latitude_NH'][1],
                                     90, cfg['concatinate_cmip5'], concat_list=cfg['cmip5_exps_concatinate'])

    cmip5_SH = prepare_cmip_for_3_18(metadatas, 'CMIP5', cfg['month_latitude_SH'][0], -90,
                                     cfg['month_latitude_SH'][1],cfg['concatinate_cmip5'], concat_list=cfg['cmip5_exps_concatinate'])

    cmip6_NH =prepare_cmip_for_3_18(metadatas, 'CMIP6',  cfg['month_latitude_NH'][0], cfg['month_latitude_NH'][1], 90)

    cmip6_SH =prepare_cmip_for_3_18(metadatas, 'CMIP6',  cfg['month_latitude_SH'][0], -90, cfg['month_latitude_SH'][1])

    # check again with original code the lat limits!!!

    cmip_data_list=[cmip5_NH, cmip5_SH, cmip6_NH, cmip6_SH]

    cmip_data_convert=[]

    for cmip_data in cmip_data_list:
        cmip_convert = calculate_siparam(cmip_data, cfg ['seaiceextent'])
        cmip_data_convert.append(cmip_convert)

    # here we calculate trends

    # here we calculate the climatology

    # here we plot the figure


    logger.info('Success')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
