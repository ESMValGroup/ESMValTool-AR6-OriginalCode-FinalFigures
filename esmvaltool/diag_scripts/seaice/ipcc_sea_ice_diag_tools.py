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

def create_attr_dict(project, month, start_lat, end_lat, concatinated, exp_list, siext=True):

    attr_dict={}
    attr_dict['project'] = project
    attr_dict['month'] = month
    attr_dict['start_lat'] = start_lat
    attr_dict['end_lat'] = end_lat
    attr_dict['experiment'] = exp_list[0]
    if siext:
        attr_dict['variable']='SIE'
    else:
        attr_dict['variable']='SIA'
    if concatinated:
        attr_dict['concatinated_from']: exp_list

    return (attr_dict)

def filter_datasets(data_dict,proj,exp_list):

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

    new_cube_list = iris.cube.CubeList()

    for (path_main,path_add) in zip(main_data_path_list,add_data_path_list):

        cube_joint=iris.cube.CubeList([iris.load_cube(path_main),iris.load_cube(path_add)])
        equalise_attributes(cube_joint)
        new_cube=cube_joint.concatenate_cube()
        new_cube_list.append(new_cube)

    return (new_cube_list)

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

def prepare_cmip_for_3_18(data_dict, project, month, start_lat, end_lat, concatinate=False, exp_list=['historical']):

    # add here a warning message about start lat being smaller than end lat
    attr_dict= create_attr_dict(project, month, start_lat, end_lat, concatinated=concatinate, exp_list=exp_list)

    cmip_dict = filter_datasets(data_dict, project, exp_list)
    if concatinate==True:
        for i in range(0,len(cmip_dict.keys())-1):
            key1 = list(cmip_dict.keys())[i]
            key2 = list(cmip_dict.keys())[i+1]
            cmip_cubelist = concatenate_cmip_data(list(cmip_dict[key1].keys()),list(cmip_dict[key2].keys()))
    else:
        key=list(cmip_dict.keys())[0]
        cmip_cubelist = load_cubelist(list(cmip_dict[key].keys()))

    cmip_cubelist = select_months(cmip_cubelist, month)
    cmip_cubelist = select_latitudes(cmip_cubelist,start_lat,end_lat)

    to_be_returned= [cmip_cubelist, attr_dict]

    return (to_be_returned)

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

    return (cubelist)

def calculate_siparam(cubelist, siext=True):
    # function which determines if sea ice extent or sea ice are should be calculated

    if siext:
        cubelist=calculate_siextent(cubelist)
    else:
        cubelist=calculate_siarea(cubelist)

    return (cubelist)

def figure_handling(cfg):

    if cfg['write_plots']:

        img_ext = diagtools.get_image_format(cfg)

        path=os.path.join(cfg['plot_dir'], 'fig_3_18_scatter' + img_ext)

        logger.info('Saving plots to %s', path)
        plt.savefig(path)

    else:

        plt.show()

    plt.close()

    return
