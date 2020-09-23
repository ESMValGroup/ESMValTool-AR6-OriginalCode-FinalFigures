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
from esmvaltool.diag_scripts.seaice import ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# #here will be the functions I need

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
            if upd_dic['project']=='CMIP5':
                data_dict[key]={'mean': [entry['mean']], 'lin_trend_slope': [entry['lin_trend_slope']], 'institute' : entry['institute_id']}
            else:
                data_dict[key] = {'mean': [entry['mean']], 'lin_trend_slope': [entry['lin_trend_slope']],
                                  'institute': entry['institution_id']}

    for data_key in data_dict.keys():
        data_dict[data_key]['mean'] = np.average(data_dict[data_key]['mean'])
        data_dict[data_key]['lin_trend_slope'] = np.average(data_dict[data_key]['lin_trend_slope']) * 10 # was done in the original code

    verb_month_dict={1:'JAN', 2:'FEB', 3:'MAR', 4:'APR', 5: 'MAY', 6:'JUN',
                     7:'JUL', 8:'AUG', 9: 'SEP', 10:'OCT', 11:'NOV', 12:'DEC'}

    upd_dic['verb_month'] = verb_month_dict[upd_dic['month']]

    if upd_dic['start_lat'] > 0:
        upd_dic['hemisphere'] = 'NH'
    else:
        upd_dic['hemisphere'] = 'SH'

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

    tmp_cbar=plt.cm.jet

    ax = plt.subplot(nrow,ncol,idx)
    title = data_dict['hemisphere'] + ' ' + data_dict['variable'] + ' ' +data_dict['verb_month'] + ' ' + 'Clim & Trend' + ' (' +data_dict['project']+')'
    ax.set_title(title)
    cmap_step=int(256/len(data_dict['data'].keys()))
    xs=[]
    for n, mod in enumerate(list(data_dict['data'].keys())):
        ax.scatter(data_dict['data'][mod]['mean'],data_dict['data'][mod]['lin_trend_slope'],label=mod,c=tmp_cbar(n*cmap_step),marker='s')
        xs.append(data_dict['data'][mod]['mean'])
    xs=np.asarray(xs)
    ax.scatter(data_dict['mme_mean'],data_dict['mme_slope'],s=60,marker='o',c='r',label='MME')
    if data_dict['p_val']<0.05:
        ax.plot(xs,xs*data_dict['slope_models']+data_dict['intercept'],c='k')
    ax.set_ylabel('Trend(10^6km^2/decade)')
    ax.set_xlabel('Clim.(10^6km^2)')
    if idx%2==0:
        ax.legend(loc=6, bbox_to_anchor=(1.0,0.5), fontsize=8, frameon=False, ncol=2)

    return

def make_plot(data_dict):

    n_panels=len(data_dict.keys())

    projects = [data_dict[key]['project'] for key in data_dict.keys()]

    nrow = len(set(projects))
    ncol = n_panels/2

    fig = plt.figure()
    fig.set_size_inches(10., 9.)

    for n, key in enumerate(data_dict.keys()):
        make_panel(data_dict[key],nrow,ncol,n+1)

    fig.subplots_adjust(left=0.085, right=0.75, top=0.96, bottom=0.06, wspace=0.28, hspace= 0.2)

    return

def main(cfg):

    metadatas = diagtools.get_input_files(cfg)

    cmip_data_dict={}

    cmip_data_dict['cmip5_NH'] = ipcc_sea_ice_diag.prepare_cmip_for_3_18(metadatas, 'CMIP5', cfg['month_latitude_NH'][0], cfg['month_latitude_NH'][1],
                                     90, cfg['concatinate_cmip5'], exp_list=cfg['cmip5_exps_concatinate'])
    cmip_data_dict['cmip5_SH'] = ipcc_sea_ice_diag.prepare_cmip_for_3_18(metadatas, 'CMIP5', cfg['month_latitude_SH'][0], -90,
                                     cfg['month_latitude_SH'][1],cfg['concatinate_cmip5'], exp_list=cfg['cmip5_exps_concatinate'])
    cmip_data_dict['cmip6_NH'] =ipcc_sea_ice_diag.prepare_cmip_for_3_18(metadatas, 'CMIP6',  cfg['month_latitude_NH'][0], cfg['month_latitude_NH'][1], 90)
    cmip_data_dict['cmip6_SH'] =ipcc_sea_ice_diag.prepare_cmip_for_3_18(metadatas, 'CMIP6',  cfg['month_latitude_SH'][0], -90, cfg['month_latitude_SH'][1])
    # check again with original code the lat limits!!!

    for cmip_data in list(cmip_data_dict.keys()):
        cmip_data_dict[cmip_data][0] = ipcc_sea_ice_diag.calculate_siparam(cmip_data_dict[cmip_data][0], cfg['seaiceextent'])
        cmip_data_dict[cmip_data][0] = calculate_entry_stat(cmip_data_dict[cmip_data][0])
        cmip_data_dict[cmip_data] = update_dict(cmip_data_dict[cmip_data]) # this reforms dictionary with model as a key
        cmip_data_dict[cmip_data] = model_stats(cmip_data_dict[cmip_data])  # here the statistics for a panel are calculated

    make_plot(cmip_data_dict)

    ipcc_sea_ice_diag.figure_handling(cfg, name = 'fig_3_18_scatter')

    # check why cmip5 is crocked

    logger.info('Success')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
