import iris
from iris.experimental.equalise_cubes import equalise_attributes
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats, special
import sys

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import group_metadata, run_diagnostic
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.ocean import diagnostic_seaice as diagseaice

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# #here will be the functions I need

def create_attr_dict(project, month, start_lat, end_lat, concatinated=False, concat_list=None):

    attr_dict={}
    attr_dict['project'] = project
    attr_dict['month'] = month
    attr_dict['start_lat'] = start_lat
    attr_dict['end_lat'] = end_lat
    if concatinated:
        attr_dict['concatinated_from']: concat_list

    return (attr_dict)

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
    attr_dict= create_attr_dict(project, month, start_lat, end_lat, concatinated=concatinate, concat_list=concat_list)

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

    to_be_returned= [cmip_cubelist, attr_dict]

    return (to_be_returned)

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

def calculate_entry_stat(cubelist):

    list_dic= []

    for cube in cubelist:
        x = np.arange(0, len(cube.coord('time').points))  # here we calculate the trend, orig code logic
        reg = stats.linregress(x, cube.data)
        clim= np.average(cube.data)
        stat_params = { 'mean': clim, 'lin_trend_slope': reg.slope}
        # do here a slope check, and if everything is masked do slope np.nan, not 0
        stat_params.update(cube.attributes)
        list_dic.append( stat_params)

    return (list_dic)

def update_dict(list_dict):

    upd_dic=list_dict[1]

    data_dict={}

    for entry in list_dict[0]:
        if upd_dic['project']=='CMIP5':
            key = entry['model_id']
        else:
            key = entry ['source_id']
        if key in data_dict.keys():
            data_dict[key]['mean']=np.concatenate((data_dict[key]['mean'],[entry['mean']]))
            data_dict[key]['lin_trend_slope']= np.average([data_dict[key]['lin_trend_slope'],entry['lin_trend_slope']])
        else:
            data_dict[key]={'mean': [entry['mean']], 'lin_trend_slope': [entry['lin_trend_slope']]}

    for data_key in data_dict.keys():
        data_dict[data_key]['mean'] = np.average(data_dict[data_key]['mean'])
        data_dict[data_key]['lin_trend_slope'] = np.average(data_dict[data_key]['lin_trend_slope']) * 10 # was done in the original code

    upd_dic.update({'data': data_dict})

    return(upd_dic)

def model_stats(inp_dict):

    means = np.asarray([inp_dict['data'][key]['mean'] for key in inp_dict['data'].keys()])
    slopes= np.asarray([inp_dict['data'][key]['lin_trend_slope'] for key in inp_dict['data'].keys()])

    # may be remove it in the future. Basically, checks if we're not comparing numbers with nans
    mask = np.isfinite(means) &np.isfinite(slopes)

    reg = stats.linregress(means [mask], slopes [mask])

    # calculating p value

    tval = reg.slope/reg.stderr
    df= len(means[mask]) - 2
    pval=special.betainc(df/2,0.5,df/(df+tval**2)) # this particular calculation was adopted from orig code, discuss

    inp_dict['slope_models'] = reg.slope
    inp_dict['mme_mean'] = np.average(means[mask])
    inp_dict['mme_slope'] = np.average(slopes[mask])
    inp_dict['intercept'] = reg.intercept
    inp_dict['p_val'] = pval

    return (inp_dict)

def make_panel(data_dict,nrow,ncol,idx):

    if data_dict['start_lat'] > 0:
        # put this above
        # also add in data dict var and verbous month
        data_dict['hemisphere'] = 'NH'
    else:
        data_dict['hemisphere'] = 'SH'

    ax = plt.subplot(nrow,ncol,idx)
    #redo
    title = data_dict['hemisphere'] + '_'+str(data_dict['month']) +'_' +data_dict['project']
    ax.set_title(title)

    return

def make_plot(data_dict):
    # somewhere here the make_panel will be called

    n_panels=len(data_dict.keys())
    # determine those
    nrow=2
    ncol=2

    fig = plt.figure()

    for n, key in enumerate(data_dict.keys()):
        make_panel(data_dict[key],nrow,ncol,n+1)

    return


def main(cfg):

    metadatas = diagtools.get_input_files(cfg)

    cmip_data_dict={}

    cmip_data_dict['cmip5_NH'] = prepare_cmip_for_3_18(metadatas, 'CMIP5', cfg['month_latitude_NH'][0], cfg['month_latitude_NH'][1],
                                     90, cfg['concatinate_cmip5'], concat_list=cfg['cmip5_exps_concatinate'])
    cmip_data_dict['cmip5_SH'] = prepare_cmip_for_3_18(metadatas, 'CMIP5', cfg['month_latitude_SH'][0], -90,
                                     cfg['month_latitude_SH'][1],cfg['concatinate_cmip5'], concat_list=cfg['cmip5_exps_concatinate'])
    cmip_data_dict['cmip6_NH'] =prepare_cmip_for_3_18(metadatas, 'CMIP6',  cfg['month_latitude_NH'][0], cfg['month_latitude_NH'][1], 90)
    cmip_data_dict['cmip6_SH'] =prepare_cmip_for_3_18(metadatas, 'CMIP6',  cfg['month_latitude_SH'][0], -90, cfg['month_latitude_SH'][1])
    # check again with original code the lat limits!!!

    for cmip_data in list(cmip_data_dict.keys()):
        cmip_data_dict[cmip_data][0] = calculate_siparam(cmip_data_dict[cmip_data][0], cfg['seaiceextent'])
        cmip_data_dict[cmip_data][0] = calculate_entry_stat(cmip_data_dict[cmip_data][0])
        cmip_data_dict[cmip_data] = update_dict(cmip_data_dict[cmip_data]) # this reforms dictionary with model as a key
        cmip_data_dict[cmip_data] = model_stats(cmip_data_dict[cmip_data])  # here the statistics for a panel are calculated

    make_plot(cmip_data_dict)

    # check why cmip5 is curced

    logger.info('Success')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
