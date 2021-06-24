# This is a script to create a figure 3.20 in Chapter 3 IPCC WGI AR6
# Authors: Elizaveta Malinina (elizaveta.malinina-rieger@canada.ca)
#          Seung-Ki Min, Olaf Morgenstern, Seungmok Paik, Nathan Gillett
# The figure is modified from Fig. 1 from Paik et al. (2020), 10.1175/JCLI-D-20-0002.1

import cf_units
import datetime
import esmvalcore
import iris
from iris.experimental.equalise_cubes import equalise_attributes
import iris.plot as iplt
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
import sys

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, Datasets
from esmvaltool.diag_scripts.seaice import ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag
from esmvalcore.preprocessor import extract_month
import esmvaltool.diag_scripts.shared.plot as eplot


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def concatenate_observations(cfg, datasets):

    # Here the datasets defined in the recipe are concatenated together
    # the new datacube is saved into the work directory

    if cfg['merge_scen_observations']:
        datasets_to_conc= cfg['obs_dataset_merging']
        cubelist_to_concatenate = iris.cube.CubeList()
        merging_fs=''
        f_names = []
        for dataset_to_conc in datasets_to_conc:
            f_name = datasets.get_info('filename', dataset=dataset_to_conc)
            f_names.append(f_name)
            data_cube = iris.load_cube(f_name)
            cubelist_to_concatenate.append(data_cube)
            merging_fs +=dataset_to_conc + '_'
        equalise_attributes(cubelist_to_concatenate)
        conc_cube = cubelist_to_concatenate.concatenate_cube()
        result_f_name = os.path.join(cfg['work_dir'], 'OBS' + merging_fs+ 'data_cube.nc')
        iris.save(conc_cube, result_f_name)
        dtsts_info = datasets.get_dataset_info(f_names[0])
        dtsts_info['filename'] = result_f_name
        dtsts_info['end_year'] = datasets.get_info('end_year',f_names[-1])
        dtsts_info['alias'] = 'OBS'+merging_fs[:-1]
        dtsts_info['dataset'] = merging_fs[:-1]
        datasets.add_dataset(result_f_name, dataset_info=dtsts_info)
        datasets.set_data(data = {}, path=result_f_name, dataset_info=dtsts_info)

    return


def update_dataset_info(datasets):

    projects = set(datasets.get_info_list('project'))

    if 'OBS' in projects:
        obs_datasets = datasets.get_info_list('filename', project='OBS')
        for obs_dataset in obs_datasets:
            obs_dataset_info = datasets.get_dataset_info(obs_dataset)
            updated_info = obs_dataset_info
            updated_info['exp'] = 'OBS'
            datasets.set_data(data={}, path=obs_dataset, dataset_info=updated_info)

    return


def mask_greenland(cubelist, var_name, shape_file):

    # this is a function to mask out greenland from the model data

    # some of the observations are already as snow cover extent
    # that's why we need to mask only the datasets that are grid-resolved
    if 'snw' in var_name:

        masked_cubelist = iris.cube.CubeList()

        for cube in cubelist:
            # here greenland shape is extracted from a cube
            masked_cube = esmvalcore.preprocessor.extract_shape(cube, shape_file, crop=False)
            # here a the mask is reversed to mask greenland out
            mask = ~ masked_cube.data.mask
            cb_mask = cube.data.mask
            # here the mask is applied, the end mask is combination of original
            # mask and greenland mask
            cube.data.mask = cb_mask | mask
            masked_cubelist.append(cube)
    else:
        masked_cubelist = cubelist

    return (masked_cubelist)


def select_months(cubelist, months_list):
    # selecting months, since there are several months needed for calculation
    # usual approach doesn't work and a separate function is needed

    res_cubelist = iris.cube.CubeList()

    for cube in cubelist:
        month_cubelist = iris.cube.CubeList()
        for month in months_list:
            m_cube = extract_month(cube, month)
            month_cubelist.append(m_cube)
        res_cubelist.append(month_cubelist)

    return(res_cubelist)


def calculate_sce(cubelist, var_name, threshold=5):
    # function to calculate snow cover extent

    sce_cubelist = iris.cube.CubeList()

    for n, sub_cubelist in enumerate(cubelist):

        sub_sce_cubelist = iris.cube.CubeList()

        # this is not a beautiful solution, however it works in this setting
        # in order to concatenate the cubelist into one cube to apply stats later
        # we assign to the cube the time coords, which rely onto the first cube
        # we assign 01-july of the years in coords of the  first cube
        tim = sub_cubelist[0].coord('time')
        orig = tim.units.origin
        calendar = tim.units.calendar
        #  dest orig and calendar are done so in the end we can merge cubes
        dest_orig = 'days since 1850-1-1 00:00:00'
        dest_calendar = 'gregorian'
        cf_tim = cf_units.num2date(tim.points, orig, calendar)
        year = np.asarray([pnt.year for pnt in cf_tim])
        pnt_dts = np.asarray([datetime.datetime(yr, 7, 1) for yr in year])
        bds_dts = np.asarray([[datetime.datetime(yr, 1, 1), datetime.datetime(yr, 12, 31)] for yr in year])
        tim_coor = iris.coords.DimCoord(cf_units.date2num(pnt_dts, dest_orig, dest_calendar), standard_name='time',
                                        long_name='time', var_name='time', units=cf_units.Unit(dest_orig, dest_calendar),
                                        bounds=cf_units.date2num(bds_dts, dest_orig, dest_calendar))
        for cube in sub_cubelist:
            if 'scen' in var_name:
                month = cf_units.num2date(cube.coord('time').points[0], orig, calendar).month
                sce_cube = iris.cube.Cube(cube.data, long_name='snow cover extent', var_name='sce',
                                          units="10^6km2", attributes=cube.attributes,
                                          dim_coords_and_dims=[(tim_coor, 0)])
                sce_cube.add_aux_coord(iris.coords.AuxCoord(month, long_name='month', var_name='month'))
            else:
                area = iris.analysis.cartography.area_weights(cube, normalize=False)
                mask = (cube.data.mask) | (cube.data.data <= threshold)
                area = np.ma.array(area, mask=mask)
                area = area.sum(axis=(1, 2))/ 1e12
                month = cf_units.num2date(cube.coord('time').points[0], orig, calendar).month
                # passing parent cube attributes, to save the info on a model
                sce_cube = iris.cube.Cube(area, long_name='snow cover extent', var_name='sce',
                                    units="10^6km2", attributes=cube.attributes, dim_coords_and_dims=[(tim_coor, 0)])
                sce_cube.add_aux_coord(iris.coords.AuxCoord(month, long_name='month', var_name='month'))
            sub_sce_cubelist.append(sce_cube)
        sce_cube = sub_sce_cubelist.merge_cube()
        sce_cube.add_aux_coord(iris.coords.AuxCoord(n, long_name='rec_number', var_name='r_num'))
        sce_cubelist.append(sce_cube)

    return (sce_cubelist)


def monthly_av(cubelist):
    # averaging over the months

    aver_cubelist = iris.cube.CubeList()

    for cube in cubelist:
        aver_cube = cube.collapsed('month', iris.analysis.MEAN)
        aver_cubelist.append(aver_cube)

    return (aver_cubelist)


def shuffle_period(cubelist, period):
    # this is a function to re-shuffle piControl data, piControl can have more
    # than period time points, so it need shuffled to the n period sized array

    shuffled_cubelist = iris.cube.CubeList()

    # creating data for time coordinates, calculated for period
    pnts = [datetime.datetime(yr, 7,1) for yr in np.arange(period[0], period[1]+1)]
    bnds = [[datetime.datetime(yr, 1,1),datetime.datetime(yr, 12, 31)] for yr in np.arange(period[0], period[1]+1)]

    nyears = period[1] - period[0] + 1

    dest_orig = 'days since 1850-1-1 00:00:00'
    dest_calendar = 'gregorian'

    for cube in cubelist:
        time_coord = iris.coords.DimCoord(cf_units.date2num(pnts, dest_orig, dest_calendar), standard_name='time',
                                          long_name='time', var_name='time', units=cf_units.Unit(dest_orig, calendar=dest_calendar),
                                          bounds=cf_units.date2num(bnds, dest_orig, dest_calendar))
        ctl_yrs = len(cube.coord('time').points)
        n_rows = ctl_yrs//nyears
        if ctl_yrs%nyears !=0:
            logger.info('The years in the ctl experiment are not divisible by %s last %s years were not considered',
                        str(nyears), str(ctl_yrs%nyears))
        data = cube.data
        sub_shuf_data = np.ma.zeros((nyears, n_rows))
        n_coord = iris.coords.DimCoord(np.arange(n_rows), long_name='n', var_name='n')
        for row in range(n_rows):
            sub_shuf_data[:, row] = data[row*nyears:(row*nyears + nyears)]
        # this is a shuffled cube for a model
        shuffled_cube =iris.cube.Cube(sub_shuf_data, long_name='shuffled snow cover extent', var_name='sce',
                            units="10^6km2", attributes=cube.attributes, dim_coords_and_dims=[(time_coord, 0), (n_coord, 1)])
        shuffled_cubelist.append(shuffled_cube)

    return (shuffled_cubelist)


def cubelist_averaging(cubelist, exp, model_name):

    # averaging cubelist, this function is a bit of a rest from the previous
    # logic of a program. It saves the data in order to calculate stats later

    # here we look at the number of experiments in a cubelist
    if len(cubelist)>1:
        # if there 2+ members weights are calculated usually
        if exp == 'piControl':
            len_arr = []
            for cube in cubelist:
                if len(cube.shape) > 1:
                    len_arr.append(cube.shape[1])
                else:
                    len_arr.append(1)
            if len(set(len_arr)) == 1:
                for n, cube in enumerate(cubelist):
                    cube.add_aux_coord(iris.coords.AuxCoord(n, long_name='ave_axis', var_name='ave_axis', bounds = [n-1, n+1]))
                equalise_attributes(cubelist)
                merged_cube = cubelist.merge_cube()
                # calculating weights to calculate stats
                wght_aux_coord = iris.coords.AuxCoord(np.ones(len(cubelist))/len(cubelist),
                    long_name='ensamble_weights',
                    var_name='ens_wght')
                merged_cube.add_aux_coord(wght_aux_coord, data_dims=1)
                averaged_cube = merged_cube.collapsed('ave_axis', iris.analysis.MEAN)
            else:
                arr_list = []
                int_wghts = []
                for n, cube in enumerate(cubelist):
                    arr_list.append(cube.data)
                    int_wghts.append(np.ones(cube.shape[1]) * 1/cube.shape[1])
                arr = np.hstack(arr_list)
                int_w_arr = np.concatenate(int_wghts)
                # calculating weights to calculate stats
                wght_aux_coord = iris.coords.AuxCoord(int_w_arr/len(cubelist),
                                                      long_name='ensamble_weights',
                                                      var_name='ens_wght')
                ave_n = iris.coords.DimCoord(np.arange(0, arr.shape[1]), long_name='ave_axis', var_name='ave_axis')
                merged_cube = iris.cube.Cube(arr, long_name=cube.long_name, var_name=cube.var_name,
                            units="10^6km2", dim_coords_and_dims=[(ave_n, 1), (cube.coord('time'), 0)], aux_coords_and_dims=[(wght_aux_coord, 1)])
                av_arr = np.average(arr, axis=1)
                averaged_cube = iris.cube.Cube(av_arr, long_name=cube.long_name, var_name=cube.var_name,
                            units="10^6km2", dim_coords_and_dims=[(cube.coord('time'), 0)])
                averaged_cube.add_aux_coord(iris.coords.AuxCoord(0, long_name='ave_axis', var_name='ave_axis', bounds = [-1, 1]))
        else:
            for n,cube in enumerate(cubelist):
                cube.add_aux_coord(iris.coords.AuxCoord(n, long_name='ave_axis', var_name='ave_axis'))
            equalise_attributes(cubelist)
            merged_cube = cubelist.merge_cube()
            # calculating weights to calculate stats
            wght_aux_coord = iris.coords.AuxCoord(
                np.ones(len(cubelist)) / len(cubelist),
                long_name='ensamble_weights',
                var_name='ens_wght')
            merged_cube.add_aux_coord(wght_aux_coord, data_dims=0)
            merged_cube.transpose()
            averaged_cube = merged_cube.collapsed('ave_axis', iris.analysis.MEAN)
    else:
        # otherwise save just the only element of a cubelist
        # and give it a weight of 1
        averaged_cube = cubelist[0]
        if exp == 'piControl':
            averaged_cube.coord('n').var_name = 'ave_axis'
            averaged_cube.coord('n').long_name = 'ave_axis'
            merged_cube = averaged_cube
            n_ave_points = averaged_cube.coord('ave_axis').points.__len__()
            wght_aux_coord = iris.coords.AuxCoord(np.ones(n_ave_points)/n_ave_points,
                long_name='ensamble_weights',
                var_name='ens_wght')
            if merged_cube.coords()[0].var_name == 'time':
                merged_cube.add_aux_coord(wght_aux_coord, data_dims=1)
            else:
                merged_cube.transpose()
                merged_cube.add_aux_coord(wght_aux_coord, data_dims=1)
        else:
            ave_axis_dim = iris.coords.DimCoord([1], long_name='ave_axis',
                                                var_name='ave_axis', bounds = [-1, 1])
            merged_cube = iris.cube.Cube(averaged_cube.data.reshape((averaged_cube.coord('time').points.__len__(),1)),
                                        long_name=averaged_cube.long_name,
                                        var_name=averaged_cube.var_name,
                                        units="10^6km2",
                                        dim_coords_and_dims=[(averaged_cube.coord('time'), 0), (ave_axis_dim, 1)])
            wght_aux_coord = iris.coords.AuxCoord(
                np.ones(len(cubelist)) / len(cubelist),
                long_name='ensamble_weights',
                var_name='ens_wght')
            merged_cube.add_aux_coord(wght_aux_coord, data_dims=1)
            averaged_cube.add_aux_coord(iris.coords.AuxCoord(0, long_name='ave_axis', var_name='ave_axis', bounds=[-1, 1]))
            averaged_cube.attributes['dataset'] = model_name

    return(averaged_cube, merged_cube)


def calc_weighted_percentiles(mod_dict):
    # here we calculate weighted percentiles; the idea is that we look at
    # the multi-model ensemble in total, a model independent on the number of
    # ensembles has to have a weight of 1. If more than 1 realization is in a
    # model, weight is calculated 1/number of realization for a model

    ave_ax_lens = [mod_dict[mod].coord('ave_axis').points.__len__()
                   for mod in mod_dict.keys()]
    # determining max of realization numbers over all models
    ens_dim_len = np.asarray(ave_ax_lens).max()
    t_dim_lens = [mod_dict[mod].coord('time').points.__len__()
                   for mod in mod_dict.keys()]
    t_dim_len = t_dim_lens[0]
    mod_dim_len = len(mod_dict.keys())

    # we create a arrays for end weights and values to create an end cube
    all_mod_wghts = np.zeros((t_dim_len, ens_dim_len, mod_dim_len))
    all_mod_values = np.ma.masked_all((t_dim_len, ens_dim_len, mod_dim_len))

    for n, mod in enumerate(mod_dict.keys()):
        mod_cube = mod_dict[mod]
        ind_ens_n = mod_cube.coord('ave_axis').points.__len__()
        wghts = mod_cube.aux_coords[0].points
        # if the number of realizations in a model is less than max,
        # other weights and values are set to 0; iris will deal with it
        all_mod_wghts[:, 0:ind_ens_n, n] = wghts[np.newaxis, :] / mod_dim_len
        all_mod_values[:, 0:ind_ens_n, n] = mod_cube.data

    # iris doesn't like nans, so nans need to be masked
    mod_cube.data = np.ma.masked_where(np.isnan(mod_cube.data), mod_cube.data)

    max_ens_coord = iris.coords.DimCoord(np.arange(0, ens_dim_len),
                                         long_name='ave_axis',
                                         var_name='ave_axis')
    mod_coord = iris.coords.DimCoord(np.arange(0, mod_dim_len),
                                     long_name='mod_axis',
                                     var_name='mod_axis')
    all_mods_cube = iris.cube.Cube(all_mod_values,
                                   long_name='all_model_sce_data',
                                   var_name='all_sces', units=mod_cube.units,
                                   dim_coords_and_dims=[
                                       (mod_cube.coord('time'), 0),
                                       (max_ens_coord, 1), (mod_coord, 2)])

    # calculating weighted percentiles over the whole ensemble
    percentiles_cube = all_mods_cube.collapsed(['ave_axis', 'mod_axis'],
                                          iris.analysis.WPERCENTILE,
                                          percent=[5, 95],
                                          weights=all_mod_wghts)

    return (percentiles_cube)


def calc_stats(cubelist, mod_dict, exp):

    #  function calculates statistics for an experiment

    output_dic = {}

    if exp == 'piControl':
        # piConrol only needs these values
        output_dic['5_95_percentiles'] = calc_weighted_percentiles(mod_dict)
        output_dic['number_models'] = len(cubelist)
    elif exp == 'OBS':
        # observations do not need the statistics at all
        for cube in cubelist:
            dataset = cube.attributes['dataset']
            # removing aux coord and cell method to make it easier to plot
            cube.remove_coord('ave_axis')
            cube.cell_methods = None
            output_dic [dataset] = cube
    else:
        # calculating weighted percentiles for the whole experiment
        output_dic['5_95_percentiles'] = calc_weighted_percentiles(mod_dict)
        for n, cube in enumerate(cubelist):
            # removing aux coords, needs for easier calculations and plotting
            cube.remove_coord('ave_axis')
            try:
                cube.remove_coord('ensamble_weights')
            except:
                pass
            cube.add_aux_coord(iris.coords.AuxCoord(n, long_name='coll_axis', var_name='coll_axis', bounds = [n - 1, n+1 ] ))
            cube.cell_methods = None
        equalise_attributes(cubelist)
        exp_cube = cubelist.merge_cube()
        # need to mask out nans, iris doesn't calculate stats with them
        exp_cube.data = np.ma.masked_where(np.isnan(exp_cube.data),
                                           exp_cube.data)
        output_dic['mean'] = exp_cube.collapsed('coll_axis', iris.analysis.MEAN)
        output_dic['min'] = exp_cube.collapsed('coll_axis', iris.analysis.MIN)
        output_dic['max'] = exp_cube.collapsed('coll_axis', iris.analysis.MAX)
        output_dic['number_models'] = len(cubelist)

    return(output_dic)

def make_panel(data_dic, obs, proj, nrows, idx):

    ax = plt.subplot(nrows, 1, idx)
    ax.plot([np.datetime64('1919'), np.datetime64('2023')], [0, 0], c='grey',
            linestyle='dashed', linewidth=0.75, zorder=0)

    prop_dic = {'BR2011_NOAA_CDR':{'color': 'grey', 'l_style': 'solid'},
                'Mudryk2020': {'color': 'grey', 'l_style': 'dashed'},
                'GLDAS_NOAH': {'color': 'grey', 'l_style': 'dotted'}}

    for obs_name in obs.keys():
        iplt.plot(obs[obs_name], linestyle=prop_dic[obs_name]['l_style'],
                  c=prop_dic[obs_name]['color'])

    for n, exp in enumerate(data_dic.keys()):
        if 'historical-' in exp:
            label = 'all forcings ('+ str(data_dic[exp]['number_models'])+')\n     shading:\n5-95 percentiles'
            iplt.plot(data_dic[exp]['mean'], label=label, c=(196/255,121/255,0), linestyle='solid', axes=ax,
                      zorder = len(obs.keys())+1)
            times = np.asarray([np.datetime64('1850-01-01') + np.timedelta64(int(p), 'D')
                                for p in data_dic[exp]['mean'].coord('time').points])
            ax.fill_between(times, data_dic[exp]['5_95_percentiles'][1].data,
                            data_dic[exp]['5_95_percentiles'][0].data,
                            color=(204/255, 174/255, 113/255), alpha=0.3,
                            linewidth=0)
        elif 'nat' in exp.lower():
            label = 'natural forcing (' + str(data_dic[exp]['number_models'])+')\n     shading:\n5-95 percentiles'
            times = np.asarray(
                [np.datetime64('1850-01-01') + np.timedelta64(int(p), 'D')
                 for p in data_dic[exp]['mean'].coord('time').points])
            iplt.plot(data_dic[exp]['mean'], label=label, c=(0, 79/255, 0),
                      linestyle='solid', axes=ax, zorder=len(obs.keys())+2)
            ax.fill_between(times, data_dic[exp]['5_95_percentiles'][1].data,
                            data_dic[exp]['5_95_percentiles'][0].data, color=(0, 79/255, 0),
                            alpha=0.2,linewidth=0)
        elif 'piControl' in exp:
            label = ' pre-industrial\n  control (' + \
                    str(data_dic[exp]['number_models'])+\
                    '):\n5-95 percentiles\n       range'
            ax.text(np.datetime64('2010-01-01'), 1.2, label,
                    color=(25/255, 51/255, 178/255))
            ax.errorbar(
                [np.datetime64('2019-04-01'), np.datetime64('2019-04-01')],
                [data_dic[exp]['5_95_percentiles'][0].data.mean(),
                 data_dic[exp]['5_95_percentiles'][1].data.mean()],
                xerr=(np.timedelta64(250, 'D')), linewidth=1.5,
                color=(25/255, 51/255, 178/255), zorder=len(obs.keys())+3)
    ax.set_ylim(-4, 4)
    ax.set_yticks(np.arange(-4, 5, 2))
    ax.set_xlim(np.datetime64('1920'),np.datetime64('2020'))
    xticks = np.arange(np.datetime64('1920'), np.datetime64('2021'),
                       np.timedelta64(20, 'Y'))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.plot([np.datetime64('1900'), np.datetime64('1900')], [0, 0], c='grey',
            label='Observations')
    ax.arrow(5750, 1.9, -1200, -0.67, length_includes_head=True, color='grey',
             linestyle='dashed')
    ax.text(np.datetime64('1925'), 2.3, 'Brown-NOAA', c='grey')
    ax.text(np.datetime64('1952'), 2.6, 'GLDAS2', c='grey')
    ax.text(np.datetime64('1986'), 1.9, '   Mudryk\net al. (2020)', c='grey')
    ax.text(np.datetime64('1922'), 3.3, proj, fontsize='x-large')
    ax.set_ylabel(r'SCE anomaly (10$^6$ km$^2$)')
    leg = ax.legend(loc=3, frameon=False, handlelength=0, ncol=3,
                    handletextpad=0.2)

    for nt, txt in enumerate(leg.get_texts()):
        col = leg.legendHandles[nt].get_color()
        txt.set_color(col)

    leg._legend_box.align = 'left'

    return


def make_plot(data_dict, cfg):

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))

    plt.style.use(st_file)

    fig = plt.figure()
    fig.set_size_inches(6.75, 9.)
    fig.set_dpi(300)

    obs = data_dict['OBS']['OBS']
    del data_dict['OBS']

    nrows = len(data_dict.keys())

    for n, key in enumerate(sorted(data_dict.keys())):
        make_panel(data_dict[key], obs, key, nrows, n+1)


    fig.suptitle('Northern Hemisphere Snow Cover Extent (SCE) anomalies')
    fig.subplots_adjust(left=0.1, right=0.9, top=0.93, bottom=0.06, wspace=0.28, hspace=0.2)

    return

def main(cfg):

    dtsts = Datasets(cfg)

    # updating the Dataset class to get experiment to observations
    update_dataset_info(dtsts)

    projects = set(dtsts.get_info_list('project'))

    # observations(BR2011&NOAA_CDR) need to be concatenated to create a row
    concatenate_observations(cfg, dtsts)

    # this is the dictionary which will be used for plotting
    plotting_dic = {}

    for project in projects:
        plotting_dic[project] = {}
        proj_dtsts = dtsts.get_dataset_info_list(project=project)
        exps = set([dtst['exp'] for dtst in proj_dtsts])
        for exp in exps:
            models = set(dtsts.get_info_list('dataset', exp=exp, project=project))
            ens_cubelist = iris.cube.CubeList()
            ens_all_dict = {}
            for model in models:
                # since the Datasets class doesn't allow to remove datasets
                # and the merged dataset is in the models list, we skip the
                # single datasets which were used for the merged one
                if model in cfg['obs_dataset_merging']:
                    continue
                mod_fnames = dtsts.get_info_list('filename', dataset=model, exp=exp, project=project)
                var_name = set(dtsts.get_info_list('short_name', dataset=model, exp=exp, project=project))
                mod_cubelist = iris.load(mod_fnames)
                # to mask out greenland, that's why we load the shapefile
                if cfg['maskout_greenland']:
                    # shape file should be located in auxiliary directory
                    greenland_shp = os.path.join(cfg['auxiliary_data_dir'], cfg['greenland_shape_file'])
                    mod_cubelist = mask_greenland(mod_cubelist, var_name, greenland_shp)
                # here the months are selected, there are several months so
                # usual month extraction doesn't work
                mod_cubelist = select_months(mod_cubelist, cfg['months'])
                # calculating snow cover extent
                mod_sce_cubelist = calculate_sce(mod_cubelist, var_name=var_name)
                # averaging over months
                mod_sce_cubelist = monthly_av(mod_sce_cubelist)
                if exp == 'piControl':
                    # piControl has different periods, so the period need to be
                    # shuffled to the main period, which comes from the recipe
                    mod_sce_cubelist = shuffle_period(mod_sce_cubelist, cfg['main_period'])
                # averaging over 5-years
                mod_sce_cubelist = ipcc_sea_ice_diag.n_year_mean(mod_sce_cubelist, n=cfg['years_for_average'])
                # subtracting the reference period, or calculating anomalies
                mod_sce_cubelist = ipcc_sea_ice_diag.subtract_ref_period(mod_sce_cubelist, cfg['ref_period'])
                logger.info("proj %s, exp %s, model %s", project, exp, model)
                # averaging the data for models, needed to calculate the stats
                ens_cube, merged_cube = cubelist_averaging(mod_sce_cubelist, exp, model)
                ens_all_dict[model] = merged_cube
                ens_cubelist.append(ens_cube)
            # calculating statistics for an experiment
            plotting_dic[project][exp] = calc_stats(ens_cubelist, ens_all_dict, exp)

    make_plot(plotting_dic, cfg)
    ipcc_sea_ice_diag.figure_handling(cfg, name='fig_3_20_timeseries')
    ipcc_sea_ice_diag.figure_handling(cfg, name='fig_3_20_timeseries', img_ext='.png')

    logger.info('Success')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
