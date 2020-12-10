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
from esmvaltool.diag_scripts.seaice import \
    ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag
from esmvalcore.preprocessor import extract_month

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def concatenate_observations(cfg, datasets):
    # clean before merging

    if cfg['merge_scen_observations']:
        datasets_to_conc = cfg['obs_dataset_merging']
        cubelist_to_concatenate = iris.cube.CubeList()
        merging_fs = ''
        f_names = []
        for dataset_to_conc in datasets_to_conc:
            f_name = datasets.get_info('filename', dataset=dataset_to_conc)
            f_names.append(f_name)
            data_cube = iris.load_cube(f_name)
            cubelist_to_concatenate.append(data_cube)
            merging_fs += dataset_to_conc + '_'
        equalise_attributes(cubelist_to_concatenate)
        conc_cube = cubelist_to_concatenate.concatenate_cube()
        result_f_name = os.path.join(cfg['work_dir'],
                                     'OBS' + merging_fs + 'data_cube.nc')
        iris.save(conc_cube, result_f_name)
        dtsts_info = datasets.get_dataset_info(f_names[0])
        dtsts_info['filename'] = result_f_name
        dtsts_info['end_year'] = datasets.get_info('end_year', f_names[-1])
        dtsts_info['alias'] = 'OBS' + merging_fs[:-1]
        dtsts_info['dataset'] = merging_fs[:-1]
        datasets.add_dataset(result_f_name, dataset_info=dtsts_info)
        datasets.set_data(data={}, path=result_f_name, dataset_info=dtsts_info)

    return


def update_dataset_info(datasets):
    projects = set(datasets.get_info_list('project'))

    if 'OBS' in projects:
        obs_datasets = datasets.get_info_list('filename', project='OBS')
        for obs_dataset in obs_datasets:
            obs_dataset_info = datasets.get_dataset_info(obs_dataset)
            updated_info = obs_dataset_info
            updated_info['exp'] = 'OBS'
            datasets.set_data(data={}, path=obs_dataset,
                              dataset_info=updated_info)

    return


def mask_greenland(cubelist, var_name, shape_file):
    if 'snw' in var_name:

        masked_cubelist = iris.cube.CubeList()

        for cube in cubelist:
            masked_cube = esmvalcore.preprocessor.extract_shape(cube,
                                                                shape_file,
                                                                crop=False)
            if cube.data.mask.all() == False:
                cb_mask = np.zeros(cube.shape, dtype=bool)
            else:
                cb_mask = cube.data.mask
            cube.mask = cb_mask | masked_cube.data.mask
            masked_cubelist.append(cube)
    else:
        masked_cubelist = cubelist

    return (masked_cubelist)


def select_months(cubelist, months_list):
    res_cubelist = iris.cube.CubeList()

    for cube in cubelist:
        month_cubelist = iris.cube.CubeList()
        for month in months_list:
            m_cube = extract_month(cube, month)
            month_cubelist.append(m_cube)
        res_cubelist.append(month_cubelist)

    return (res_cubelist)


def calculate_sce(cubelist, var_name, threshold=5):
    sce_cubelist = iris.cube.CubeList()

    for n, sub_cubelist in enumerate(cubelist):

        sub_sce_cubelist = iris.cube.CubeList()

        # this is not a beautiful solution, however it works in this setting
        # in order to concatenate the cubelist into one cube to apply stats
        # later
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
        bds_dts = np.asarray(
            [[datetime.datetime(yr, 1, 1), datetime.datetime(yr, 12, 31)] for
             yr in year])
        tim_coor = iris.coords.DimCoord(
            cf_units.date2num(pnt_dts, dest_orig, dest_calendar),
            standard_name='time',
            long_name='time', var_name='time',
            units=cf_units.Unit(dest_orig, dest_calendar),
            bounds=cf_units.date2num(bds_dts, dest_orig, dest_calendar))
        for cube in sub_cubelist:
            if 'scen' in var_name:
                month = cf_units.num2date(cube.coord('time').points[0], orig,
                                          calendar).month
                sce_cube = iris.cube.Cube(cube.data,
                                          long_name='snow cover extent',
                                          var_name='sce',
                                          units="10^6km2",
                                          attributes=cube.attributes,
                                          dim_coords_and_dims=[(tim_coor, 0)])
                sce_cube.add_aux_coord(
                    iris.coords.AuxCoord(month, long_name='month',
                                         var_name='month'))
            else:
                area = iris.analysis.cartography.area_weights(cube,
                                                              normalize=False)
                mask = (cube.data.mask) | (cube.data.data <= threshold)
                area = np.ma.array(area, mask=mask)
                area = area.sum(axis=(1, 2)) / 1e12
                month = cf_units.num2date(cube.coord('time').points[0], orig,
                                          calendar).month
                # for now passing paren cube attributes, clean before merging!!
                sce_cube = iris.cube.Cube(area, long_name='snow cover extent',
                                          var_name='sce',
                                          units="10^6km2",
                                          attributes=cube.attributes,
                                          dim_coords_and_dims=[(tim_coor, 0)])
                sce_cube.add_aux_coord(
                    iris.coords.AuxCoord(month, long_name='month',
                                         var_name='month'))
            sub_sce_cubelist.append(sce_cube)
        sce_cube = sub_sce_cubelist.merge_cube()
        sce_cube.add_aux_coord(
            iris.coords.AuxCoord(n, long_name='rec_number', var_name='r_num'))
        sce_cubelist.append(sce_cube)

    return (sce_cubelist)


def monthly_av(cubelist):
    aver_cubelist = iris.cube.CubeList()

    for cube in cubelist:
        aver_cube = cube.collapsed('month', iris.analysis.MEAN)
        aver_cubelist.append(aver_cube)

    return (aver_cubelist)


def shuffle_period(cubelist, period):
    shuffled_cubelist = iris.cube.CubeList()

    pnts = [datetime.datetime(yr, 7, 1) for yr in
            np.arange(period[0], period[1] + 1)]
    bnds = [[datetime.datetime(yr, 1, 1), datetime.datetime(yr, 12, 31)] for yr
            in np.arange(period[0], period[1] + 1)]

    nyears = period[1] - period[0] + 1

    dest_orig = 'days since 1850-1-1 00:00:00'
    dest_calendar = 'gregorian'

    for cube in cubelist:
        time_coord = iris.coords.DimCoord(
            cf_units.date2num(pnts, dest_orig, dest_calendar),
            standard_name='time',
            long_name='time', var_name='time',
            units=cf_units.Unit(dest_orig, calendar=dest_calendar),
            bounds=cf_units.date2num(bnds, dest_orig, dest_calendar))
        ctl_yrs = len(cube.coord('time').points)
        n_rows = ctl_yrs // nyears
        if ctl_yrs % nyears != 0:
            logger.info(
                'The years in the ctl experiment are not divisible by %s last '
                '%s years were not considered',
                str(nyears), str(ctl_yrs % nyears))
        data = cube.data
        sub_shuf_data = np.ma.zeros((nyears, n_rows))
        n_coord = iris.coords.DimCoord(np.arange(n_rows), long_name='n',
                                       var_name='n')
        for row in range(n_rows):
            sub_shuf_data[:, row] = data[row * nyears:(row * nyears + nyears)]
        shuffled_cube = iris.cube.Cube(sub_shuf_data,
                                       long_name='shuffled snow cover extent',
                                       var_name='sce',
                                       units="10^6km2",
                                       attributes=cube.attributes,
                                       dim_coords_and_dims=[(time_coord, 0),
                                                            (n_coord, 1)])
        shuffled_cubelist.append(shuffled_cube)

    return (shuffled_cubelist)


def cubelist_averaging(cubelist, exp, model_name):
    if len(cubelist) > 1:
        if exp == 'piControl':
            len_arr = []
            for cube in cubelist:
                if len(cube.shape) > 1:
                    len_arr.append(cube.shape[1])
                else:
                    len_arr.append(1)
            if len(set(len_arr)) == 1:
                for n, cube in enumerate(cubelist):
                    cube.add_aux_coord(
                        iris.coords.AuxCoord(n, long_name='ave_axis',
                                             var_name='ave_axis',
                                             bounds=[n - 1, n + 1]))
                equalise_attributes(cubelist)
                merged_cube = cubelist.merge_cube()
                averaged_cube = merged_cube.collapsed('ave_axis',
                                                      iris.analysis.MEAN)
            else:
                dim_len = np.array(len_arr).max()
                arr = np.ma.zeros(
                    (len(cube.coord('time').points), dim_len, len(cubelist)))
                for n, cube in enumerate(cubelist):
                    if len_arr[n] < dim_len:
                        arr[:, 0:len_arr[n], n] = cube.data
                        arr[:, len_arr[n]: dim_len, n] = np.ma.masked_all(
                            (len(cube.coord('time').points),
                             dim_len - len_arr[n]))
                    else:
                        arr[:, :, n] = cube.data
                        coord_n = cube.coord('n')
                av_arr = np.average(arr, axis=2)
                averaged_cube = iris.cube.Cube(av_arr,
                                               long_name=cube.long_name,
                                               var_name=cube.var_name,
                                               units="10^6km2",
                                               dim_coords_and_dims=[
                                                   (coord_n, 1),
                                                   (cube.coord('time'), 0)])
                averaged_cube.add_aux_coord(
                    iris.coords.AuxCoord(0, long_name='ave_axis',
                                         var_name='ave_axis', bounds=[-1, 1]))
        else:
            for n, cube in enumerate(cubelist):
                cube.add_aux_coord(
                    iris.coords.AuxCoord(n, long_name='ave_axis',
                                         var_name='ave_axis'))
            equalise_attributes(cubelist)
            merged_cube = cubelist.merge_cube()
            averaged_cube = merged_cube.collapsed('ave_axis',
                                                  iris.analysis.MEAN)
    else:
        averaged_cube = cubelist[0]
        averaged_cube.add_aux_coord(
            iris.coords.AuxCoord(0, long_name='ave_axis', var_name='ave_axis',
                                 bounds=[-1, 1]))
        averaged_cube.attributes['dataset'] = model_name

    return (averaged_cube)


def calc_stats(cubelist, exp):
    output_dic = {}

    if exp == 'piControl':
        n_sizes = np.asarray(
            [len(cube.coord('n').points) for cube in cubelist])
        n_len = n_sizes.max()
        t_len = len(cubelist[0].coord('time').points)
        joint_arr = np.ma.zeros((t_len, n_len, len(cubelist)))
        for n, cube in enumerate(cubelist):
            n_lim = len(cube.coord('n').points)
            joint_arr[:, 0:n_lim, n] = cube.data
            if n_lim < n_len:
                joint_arr[:, 0:n_lim, n] = cube.data
                joint_arr[:, n_lim:n_len, 1] = np.ma.masked_all(
                    (t_len, n_len - n_lim))
        max_arr = joint_arr.max(axis=(1, 2))
        min_arr = joint_arr.min(axis=(1, 2))
        # to compute percentiles for 3d array only nan percentiles can be used
        # masked features become nans
        joint_arr[joint_arr.mask] = np.nan
        percent = np.nanpercentile(joint_arr.data, [5, 95], axis=(1, 2))
        per_coord = iris.coords.DimCoord([5, 95], long_name='percentile',
                                         var_name='percentile')
        output_dic['min'] = iris.cube.Cube(min_arr, long_name='min_sce',
                                           var_name='min_sce',
                                           units="10^6km2",
                                           dim_coords_and_dims=[
                                               (cube.coord('time'), 0)])
        output_dic['max'] = iris.cube.Cube(max_arr, long_name='min_sce',
                                           var_name='min_sce',
                                           units="10^6km2",
                                           dim_coords_and_dims=[
                                               (cube.coord('time'), 0)])
        output_dic['5_95_percentiles'] = iris.cube.Cube(percent,
                                                        long_name='perc_sce',
                                                        var_name='perc_sce',
                                                        units="10^6km2",
                                                        dim_coords_and_dims=[
                                                            (per_coord, 0), (
                                                            cube.coord('time'),
                                                            1)])
        output_dic['number_models'] = len(cubelist)
    elif exp == 'OBS':
        for cube in cubelist:
            dataset = cube.attributes['dataset']
            cube.remove_coord('ave_axis')
            cube.cell_methods = None
            output_dic[dataset] = cube
    else:
        for n, cube in enumerate(cubelist):
            cube.remove_coord('ave_axis')
            cube.add_aux_coord(
                iris.coords.AuxCoord(n, long_name='coll_axis',
                                     var_name='coll_axis',
                                     bounds=[n - 1, n + 1]))
            cube.cell_methods = None
        equalise_attributes(cubelist)
        exp_cube = cubelist.merge_cube()
        output_dic['mean'] = exp_cube.collapsed('coll_axis',
                                                iris.analysis.MEAN)
        output_dic['min'] = exp_cube.collapsed('coll_axis', iris.analysis.MIN)
        output_dic['max'] = exp_cube.collapsed('coll_axis', iris.analysis.MAX)
        output_dic['5_95_percentiles'] = exp_cube.collapsed('coll_axis',
                                                      iris.analysis.PERCENTILE,
                                                      percent=[5, 95])
        output_dic['number_models'] = len(cubelist)

    return (output_dic)


def make_panel(data_dic, obs, proj, nrows, idx):
    ax = plt.subplot(nrows, 1, idx)
    ax.plot([np.datetime64('1919'), np.datetime64('2021')], [0, 0], c='k',
            linestyle='solid', linewidth=1)

    prop_dic = {'BR2011_NOAA_CDR': {'color': 'tab:brown', 'l_style': 'solid',
                                    'label': 'Brown-NOAA'},
                'Mudryk2020': {'color': 'tab:brown', 'l_style': 'dashed',
                               'label': 'Mudryk et al. (2020)'},
                '20CR': {'color': 'k', 'l_style': 'solid', 'label': '20CR2'},
                'GLDAS_NOAH': {'color': 'k', 'l_style': 'dashed',
                               'label': 'GLDAS2'}}

    for obs_name in obs.keys():
        iplt.plot(obs[obs_name], linestyle=prop_dic[obs_name]['l_style'],
                  c=prop_dic[obs_name]['color'],
                  label=prop_dic[obs_name]['label'])

    for n, exp in enumerate(sorted(data_dic.keys())):
        if 'historical-' in exp:
            label = 'ALL [' + str(data_dic[exp]['number_models']) + ']'
            iplt.plot(data_dic[exp]['mean'], label=label, c='tab:red',
                      linestyle='solid', axes=ax,
                      zorder=len(obs.keys()) + 1)
            times = np.asarray(
                [np.datetime64('1850-01-01') + np.timedelta64(int(p), 'D')
                 for p in data_dic[exp]['mean'].coord('time').points])
            ax.fill_between(times, data_dic[exp]['5_95_percentiles'][1].data,
                            data_dic[exp]['5_95_percentiles'][0].data,
                            color='tab:red', alpha=0.3, linewidth=0)
        elif 'nat' in exp.lower():
            label = 'NAT [' + str(data_dic[exp]['number_models']) + ']'
            iplt.plot(data_dic[exp]['mean'], label=label, c='tab:blue',
                      linestyle='solid', axes=ax,
                      zorder=len(obs.keys()) + 2)
            iplt.plot(data_dic[exp]['min'], c='tab:blue', linestyle='dashed',
                      axes=ax, dashes=(2, 1))
            iplt.plot(data_dic[exp]['max'], c='tab:blue', linestyle='dashed',
                      axes=ax, dashes=(2, 1))
        elif 'piControl' in exp:
            label = 'CTL [' + str(data_dic[exp]['number_models']) + ']'
            ax.plot([1, 1], c='silver', linestyle='solid', label=label,
                    zorder=len(obs.keys()) + 3)
            iplt.plot(data_dic[exp]['5_95_percentiles'][0], c='silver',
                      linestyle='dashed', axes=ax, dashes=(2, 1))
            iplt.plot(data_dic[exp]['5_95_percentiles'][1], c='silver',
                      linestyle='dashed', axes=ax, dashes=(2, 1))

    ax.set_ylim(-4, 4)
    ax.set_yticks(np.arange(-4, 5, 2))
    ax.set_xlim(np.datetime64('1920'), np.datetime64('2020'))
    xticks = np.arange(np.datetime64('1920'), np.datetime64('2021'),
                       np.timedelta64(20, 'Y'))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.text(np.datetime64('1922'), 3.3, proj, fontsize='x-large')
    ax.set_ylabel(r'SCE anomaly (10$^6$ km$^2$)')
    ax.legend(loc=3, frameon=False, ncol=2)

    return


def make_plot(data_dict):
    fig = plt.figure()
    fig.set_size_inches(7., 9.)

    obs = data_dict['OBS']['OBS']
    del data_dict['OBS']

    nrows = len(data_dict.keys())

    for n, key in enumerate(data_dict.keys()):
        make_panel(data_dict[key], obs, key, nrows, n + 1)

    fig.suptitle('Northern Hemisphere Snow Cover Extent (SCE) anomiles')
    fig.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.06,
                        wspace=0.28, hspace=0.2)

    return


def main(cfg):
    dtsts = Datasets(cfg)

    update_dataset_info(dtsts)

    projects = set(dtsts.get_info_list('project'))

    # try to improve it!!!!
    concatenate_observations(cfg, dtsts)

    plotting_dic = {}

    for project in projects:
        plotting_dic[project] = {}
        proj_dtsts = dtsts.get_dataset_info_list(project=project)
        exps = set([dtst['exp'] for dtst in proj_dtsts])
        for exp in exps:
            models = set(
                dtsts.get_info_list('dataset', exp=exp, project=project))
            ens_cubelist = iris.cube.CubeList()
            for model in models:
                if model in cfg['obs_dataset_merging']:
                    continue
                mod_fnames = dtsts.get_info_list('filename', dataset=model,
                                                 exp=exp, project=project)
                var_name = set(
                    dtsts.get_info_list('short_name', dataset=model, exp=exp,
                                        project=project))
                mod_cubelist = iris.load(mod_fnames)
                if cfg['maskout_greenland']:
                    greenland_shp = os.path.join(cfg['auxiliary_data_dir'],
                                                 cfg['greenland_shape_file'])
                    mod_cubelist = mask_greenland(mod_cubelist, var_name,
                                                  greenland_shp)
                mod_cubelist = select_months(mod_cubelist, cfg['months'])
                mod_sce_cubelist = calculate_sce(mod_cubelist,
                                                 var_name=var_name)
                mod_sce_cubelist = monthly_av(mod_sce_cubelist)
                if exp == 'piControl':
                    mod_sce_cubelist = shuffle_period(mod_sce_cubelist,
                                                      cfg['main_period'])
                mod_sce_cubelist = ipcc_sea_ice_diag.n_year_mean(
                    mod_sce_cubelist, n=cfg['years_for_average'])
                mod_sce_cubelist = ipcc_sea_ice_diag.substract_ref_period(
                    mod_sce_cubelist, cfg['ref_period'])
                logger.info("proj %s, exp %s, model %s", project, exp, model)
                ens_cube = cubelist_averaging(mod_sce_cubelist, exp, model)
                ens_cubelist.append(ens_cube)
            plotting_dic[project][exp] = calc_stats(ens_cubelist, exp)

    make_plot(plotting_dic)
    ipcc_sea_ice_diag.figure_handling(cfg, name='fig_3_20_timeseries')

    logger.info('Success')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
