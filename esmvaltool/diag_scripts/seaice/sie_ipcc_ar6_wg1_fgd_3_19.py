import iris
from iris.experimental.equalise_cubes import equalise_attributes
import iris.plot as iplt
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import group_metadata, run_diagnostic
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.seaice import ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def model_wrap(cubelist, proj):

    models=[]

    for cube in cubelist:
        if proj == 'CMIP5':
            models.append(cube.attributes['model_id'])
        else:
            models.append(cube.attributes['source_id'])

    model_cubelist = iris.cube.CubeList()

    for nm, model in enumerate(set(models)):
        sngl_mod_cubelist = iris.cube.CubeList()
        for cube in cubelist:
            # check if you could do it a better way, may be add function
            if proj == 'CMIP5':
                if cube.attributes['model_id'] == model:
                    aux_coord = iris.coords.AuxCoord(len(sngl_mod_cubelist)+1, long_name= 'realization')
                    cube.add_aux_coord(aux_coord)
                    sngl_mod_cubelist.append(cube)
            else:
                if cube.attributes['source_id'] == model:
                    aux_coord = iris.coords.AuxCoord(len(sngl_mod_cubelist)+1, long_name= 'realization')
                    cube.add_aux_coord(aux_coord)
                    sngl_mod_cubelist.append(cube)

        if len(sngl_mod_cubelist) == 1:
            mod_cube = sngl_mod_cubelist[0]
        else:
            equalise_attributes(sngl_mod_cubelist)
            mod_cube = sngl_mod_cubelist.merge_cube()
            mod_cube = mod_cube.collapsed('realization', iris.analysis.MEAN)

        # dirty trick to merge cubes in the future
        mod_cube.remove_coord('realization')
        mod_cube.cell_methods = None

        mod_cube.add_aux_coord(iris.coords.AuxCoord(nm + 1, long_name='mod_n'))

        model_cubelist.append(mod_cube)

    return(model_cubelist)

def update_dict(list_dict):

    upd_dic=list_dict[1]

    verb_month_dict={1:'JAN', 2:'FEB', 3:'MAR', 4:'APR', 5: 'MAY', 6:'JUN',
                     7:'JUL', 8:'AUG', 9: 'SEP', 10:'OCT', 11:'NOV', 12:'DEC'}

    upd_dic['verb_month'] = verb_month_dict[upd_dic['month']]

    if upd_dic['start_lat'] > 0:
        upd_dic['hemisphere'] = 'NH'
    else:
        upd_dic['hemisphere'] = 'SH'

    model_cubelist = model_wrap(list_dict[0], upd_dic['project'])

    upd_dic.update({'data': model_cubelist})

    return(upd_dic)

def substract_ref_period(cubelist, ref_period):

    constr = iris.Constraint(time=lambda cell: ref_period[0] <= cell.point.year <= ref_period[1])

    upd_cubelist = iris.cube.CubeList()

    for cube in cubelist:
        mean = cube.extract(constr).collapsed('time', iris.analysis.MEAN)
        upd_cube = cube - mean
        upd_cube.attributes = cube.attributes
        upd_cubelist.append(upd_cube)

    return(upd_cubelist)

def create_coords(cubelist, year_n):
    # dirty trick, we try to unify time to merge  the cubelist in the end

    cb = cubelist [0]

    for cube in cubelist:
        if cube.coord('time').units.calendar == 'gregorian':
            cb = cube
            break

    n_t = len(cb.coord('time').points)
    coord = [np.average(cb.coord('time').points[year_n*i:year_n*i + year_n]) for i in range(0, int(n_t / year_n))]
    bnds = [[cb.coord('time').bounds[year_n*i][0], cb.coord('time').bounds[year_n*i + (year_n - 1)][1]] for i in
            range(0, int(n_t / year_n))]
    if n_t%year_n != 0:
        # raise warning
        print('The n of years is not divisible by 3')
        # coord.append(np.average(cb.coord('time').points[int(n_t / year_n):-1]))
        # bnds.append([cb.coord('time').bounds[int(n_t / year_n) * year_n][0], cb.coord('time').bounds[-1][1]])

    dcoord = iris.coords.DimCoord(np.asarray(coord), bounds=np.asarray(bnds),
                                  standard_name=cb.coord('time').standard_name,
                                  units=cb.coord('time').units, long_name=cb.coord('time').long_name,
                                  var_name=cb.coord('time').var_name)

    return (dcoord)

def three_year_mean(cubelist):

    threey_cubelist = iris.cube.CubeList()

    dcoord = create_coords(cubelist, 3)

    for cube in cubelist:
        n_t = len(cube.coord('time').points)
        data = [np.average(cube.data[3*i:3*i + 3]) for i in range(0, int(n_t / 3))]
        if n_t%3!=0:
            # add here a warning that the last is an average of n_t%3==0 years
            print('The n of years is not divisible by 3, last years were not taken into account')
            # data.append(np.average(cube.data[3*int(n_t/3):-1]))
        threey_cube = iris.cube.Cube(np.asarray(data), long_name='sie anomaly, 3y mean', var_name='sie_ano', units=cube.units,
                                     attributes=cube.attributes, dim_coords_and_dims=[(dcoord,0)])
        threey_cubelist.append(threey_cube)

    return (threey_cubelist)

def stat_calc(inp_dict):

    add_dict = {}

    data_cubelist = inp_dict['data']
    del inp_dict['data']  # can't do pop, because data_cubelist is list

    # we can't do iris.analysis.MEAN because adding an aux coord realizes the data
    # and then np.nans screw things over
    mod_cube_arr = np.zeros((len(data_cubelist),len(data_cubelist[0].coord('time').points)))

    for n, cube in enumerate(data_cubelist):
        mod_cube_arr[n,:] = data_cubelist[n].data

    mme_arr = np.nanmean(mod_cube_arr, axis = 0)
    std_arr = np.nanstd(mod_cube_arr, axis = 0)

    # this is to make as time coords only year and not a month
    # needed to further merge cubes
    yearly_stamp_point = []
    yearly_stamp_bounds = []

    for cell in cube.coord('time').cells():
        yearly_stamp_point.append(np.datetime64(str(cell.point.year)+'-06-30'))
        # another dirty trick: since the bounds for december are 1of january next year,
        # we just add +2 to the start year
        yearly_stamp_bounds.append((np.datetime64(str(cell.bound[0].year)+'-01-01'),np.datetime64(str(cell.bound[0].year+2)+'-12-31')))

    yearly_stamp_point = np.asarray(yearly_stamp_point)
    yearly_stamp_bounds = np.asarray(yearly_stamp_bounds)

    # maybe clean it later as np.datimedelta [D]
    points_days_since = yearly_stamp_point - np.datetime64('1850-01-01')
    bounds_days_since = yearly_stamp_bounds - np.datetime64('1850-01-01')

    time_coord = iris.coords.DimCoord(np.asarray(points_days_since,  dtype = np.int32), bounds=np.asarray(bounds_days_since, dtype = np.int32), standard_name='time',
                                      long_name='time', var_name='time', units=cube.coord('time').units)

    mme = iris.cube.Cube(mme_arr, long_name= 'multi-model mean of '+inp_dict['variable']+' anomaly', var_name='mme_sie_ano',
                         units=data_cubelist[0].units, dim_coords_and_dims=[(time_coord,0)])

    std = iris.cube.Cube(std_arr, long_name= 'multi-model std of '+inp_dict['variable']+' anomaly', var_name='std_sie_ano',
                         units=data_cubelist[0].units, dim_coords_and_dims=[(time_coord,0)])

    pl_sct_arr = np.zeros(mme_arr.shape)
    pl_sct_arr[:] = np.nan
    pl_sct_arr[np.abs(mme_arr)>std_arr] = 1

    pl_sct = iris.cube.Cube(pl_sct_arr, long_name= 'significance of '+inp_dict['variable']+' anomaly', var_name='mme_signif',
                         units=data_cubelist[0].units, dim_coords_and_dims=[(time_coord,0)])

    add_dict['mme'] = mme
    add_dict['std'] = std
    add_dict['n_models'] = len(data_cubelist)
    add_dict['mme_sign'] = pl_sct

    inp_dict.update(add_dict)

    return (inp_dict)

def merge_stats_cube(inp_dict):

    upd_dict = {}

    merged_mme_cubelist = iris.cube.CubeList()
    merged_std_cubelist = iris.cube.CubeList()
    merged_sign_cubelist = iris.cube.CubeList()

    vb_month = []

    for month in inp_dict.keys():
        # inp_dict[month]['mme'].add_aux_coord(iris.coords.AuxCoord(inp_dict[month]['verb_month'], long_name='month'))
        inp_dict[month]['mme'].add_aux_coord(iris.coords.AuxCoord(month, long_name='month',var_name='month'))
        merged_mme_cubelist.append(inp_dict[month]['mme'])
        # inp_dict[month]['std'].add_aux_coord(iris.coords.AuxCoord(inp_dict[month]['verb_month'], long_name='month'))
        inp_dict[month]['std'].add_aux_coord(iris.coords.AuxCoord(month, long_name='month',var_name='month'))
        merged_std_cubelist.append(inp_dict[month]['std'])
        # inp_dict[month]['mme_sign'].add_aux_coord(iris.coords.AuxCoord(inp_dict[month]['verb_month'], long_name='month'))
        inp_dict[month]['mme_sign'].add_aux_coord(iris.coords.AuxCoord(month, long_name='month',var_name='month'))
        merged_sign_cubelist.append(inp_dict[month]['mme_sign'])

        vb_month.append(inp_dict[month]['verb_month'])

    upd_dict['mme'] = merged_mme_cubelist.merge_cube()
    upd_dict['std'] = merged_std_cubelist.merge_cube()
    upd_dict['mme_sign'] = merged_sign_cubelist.merge_cube()

    for key in ['mme','mme_sign','std','month','verb_month']:
        del inp_dict[month][key]

    upd_dict['verb_month'] = vb_month
    upd_dict.update(inp_dict[month])

    return (upd_dict)

def make_panel(data_dict, fig, inner, title):

    tmp_cmap = plt.cm.coolwarm_r

    for n,proj in enumerate(data_dict.keys()):
        ax = plt.Subplot(fig, inner [n])
        pmesh = iplt.pcolormesh(data_dict[proj]['mme'], vmin = -1.2, vmax = 1.2, cmap = tmp_cmap, axes = ax)
        ax.scatter((data_dict[proj]['mme_sign'].coord('time').points * data_dict[proj]['mme_sign'].data) ,
                   data_dict[proj]['mme_sign'].coord('month').points[:, None] * data_dict[proj]['mme_sign'].data, s = 60, c= 'silver')
        ylims = ax.get_ylim()
        ax.set_ylim(ylims[::-1])
        ax.set_yticks(data_dict[proj]['mme'].coord('month').points[::-1])
        ax.set_yticklabels(data_dict[proj]['verb_month'][::-1])
        ax.set_xlim((np.datetime64('1979-01-01')-np.datetime64('1850-01-01')).astype(np.int32),
                    (np.datetime64('2018-01-01')-np.datetime64('1850-01-01')).astype(np.int32))
        yrs = np.arange(1980, 2018, 6)
        tks_dates = (np.asarray( [np.datetime64(str(yr)+'-07-01') for yr in yrs ]) - np.datetime64('1850-01-01')).astype(np.int32)
        ax.set_xticks(tks_dates)
        ax.set_xticklabels([str(yr) for yr in yrs])
        if n ==0 :
            ax.set_title(title)
            ax.set_xticks([])
        else:
            ax.set_xticks(tks_dates)
            ax.set_xticklabels([str(yr) for yr in yrs])
        ax.set_ylabel(proj+ ' ['+str(data_dict[proj]['n_models'])+']', rotation=90)
        fig.add_subplot(ax)

    return (pmesh)

def make_plot(data_dict):

    ncols = len(data_dict.keys())
    nrows = np.asarray([len(data_dict[hemisph].keys()) for hemisph in data_dict.keys()]).max()

    fig = plt.figure()
    fig.set_size_inches(10., 9.)
    outer = gridspec.GridSpec(nrows, ncols)

    for ncol, hemisph in enumerate(data_dict.keys()):
        for nrow, exp in enumerate(data_dict[hemisph].keys()):
            nsubpan = len(data_dict[hemisph][exp].keys())
            inner = gridspec.GridSpecFromSubplotSpec(nsubpan, 1,
                                                     subplot_spec = outer[nrow, ncol], wspace = 0.0, hspace = 0.0)
            if nrow ==0:
                if hemisph == 'NH':
                    title = 'Arctic SIE'
                else:
                    title = 'Antarctic SIE'
            else:
                title = ''
            pmesh = make_panel(data_dict[hemisph][exp], fig, inner, title= title)

    cax = fig.add_axes([0.3,0.05,0.4,0.01])
    cbar = fig.colorbar(pmesh, cax=cax, orientation='horizontal')
    cbar.ax.set_xlabel('10^6 km^2')
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)#, wspace=0.28, hspace=0.2)
    # make a joint colorbar


    return

def main(cfg):

    metadatas = diagtools.get_input_files(cfg)

    cmip_data_dict = {'NH':{}, 'SH': {}}

    for hemisph in cmip_data_dict.keys():
        cmip_data_dict[hemisph]['historical'] = {'CMIP5':{}, 'CMIP6':{}}
        mon_lat_str = 'month_latitude_' + hemisph
        if np.abs(cfg[mon_lat_str][1]) == cfg[mon_lat_str][1]:
            start_lat = cfg[mon_lat_str][1]
            end_lat = 90
        else:
            start_lat = -90
            end_lat = cfg[mon_lat_str][1]

        for experiment in cmip_data_dict[hemisph].keys():
            for entry in cmip_data_dict[hemisph][experiment].keys():
                for month in cfg[mon_lat_str][0]:
                    cmip_data_dict[hemisph][experiment][entry][month] = ipcc_sea_ice_diag.prepare_cmip_for_3_18(metadatas, entry,
                                                                             month, start_lat, end_lat, cfg['concatinate_' + entry.lower()],
                                                                             exp_list=cfg[entry.lower() +'_exps_concatinate'])
                    cmip_data_dict[hemisph][experiment][entry][month][0] = ipcc_sea_ice_diag.calculate_siparam(cmip_data_dict[hemisph][experiment][entry][month][0],
                                                                               cfg['seaiceextent'])
                    cmip_data_dict[hemisph][experiment][entry][month][0] = substract_ref_period(cmip_data_dict[hemisph][experiment][entry][month][0], cfg['ref_period'])
                    cmip_data_dict[hemisph][experiment][entry][month][0] = three_year_mean(cmip_data_dict[hemisph][experiment][entry][month][0])
                    cmip_data_dict[hemisph][experiment][entry][month] = update_dict(cmip_data_dict[hemisph][experiment][entry][month])
                    cmip_data_dict[hemisph][experiment][entry][month] = stat_calc(cmip_data_dict[hemisph][experiment][entry][month])
                cmip_data_dict[hemisph][experiment][entry] = merge_stats_cube(cmip_data_dict[hemisph][experiment][entry])

    make_plot(cmip_data_dict)

    ipcc_sea_ice_diag.figure_handling(cfg, name = 'fig_3_19_timeseries')
    # check why cmip5 is crocked

    logger.info('Success')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
