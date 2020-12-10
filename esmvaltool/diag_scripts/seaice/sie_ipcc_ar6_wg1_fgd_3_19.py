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
from esmvaltool.diag_scripts.shared import run_diagnostic, Datasets
from esmvaltool.diag_scripts.seaice import \
    ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def detect_exp(datasets, panel_id, project):
    experiments = list(set(datasets.get_info_list('exp', project=project)))

    if project == 'CMIP5':
        nat_exp = 'historicalNat'
    else:
        nat_exp = 'hist-nat'

    if 'nat' in panel_id.lower():
        exp_name = nat_exp
    else:
        experiments.pop(experiments.index(nat_exp))
        exp_name = experiments[0]

    return (exp_name)


def ensemble_average(cubelist):
    if len(cubelist) == 1:
        averaged_cube = cubelist[0]
    else:
        for n, cube in enumerate(cubelist):
            aux_coord = iris.coords.AuxCoord(n, long_name='realization')
            cube.add_aux_coord(aux_coord)

        equalise_attributes(cubelist)
        merged_cubelist = cubelist.merge_cube()
        averaged_cube = merged_cubelist.collapsed('realization',
                                                  iris.analysis.MEAN)

        # dirty trick to merge cubes in the future
        averaged_cube.remove_coord('realization')
        averaged_cube.cell_methods = None

    return (averaged_cube)


def mme_stats(data_cubelist):
    # we can't do iris.analysis.MEAN because adding an aux coord realizes
    # the data
    # and then np.nans screw things over, also the averaging is easier because
    # of time stamps

    n_models = len(data_cubelist)

    mod_cube_arr = np.ma.masked_all(
        (n_models, len(data_cubelist[0].coord('time').points)))

    var = data_cubelist[0].long_name

    for n, cube in enumerate(data_cubelist):
        mod_cube_arr[n, :] = data_cubelist[n].data

    mme_arr = np.ma.mean(mod_cube_arr, axis=0)
    std_arr = np.ma.std(mod_cube_arr, axis=0)

    # this is to make as time coords only year and not a month
    # needed to further merge cubes
    yearly_stamp_point = []
    yearly_stamp_bounds = []

    for cell in cube.coord('time').cells():
        yearly_stamp_point.append(
            np.datetime64(str(cell.point.year) + '-06-30'))
        # another dirty trick: since the bounds for december are 1of january
        # next year, we just add +2 to the start year
        yearly_stamp_bounds.append(
            (np.datetime64(str(cell.bound[0].year) + '-01-01'),
             np.datetime64(str(cell.bound[0].year + 2) + '-12-31')))

    yearly_stamp_point = np.asarray(yearly_stamp_point)
    yearly_stamp_bounds = np.asarray(yearly_stamp_bounds)

    # maybe clean it later as np.datimedelta [D]
    points_days_since = yearly_stamp_point - np.datetime64('1850-01-01')
    bounds_days_since = yearly_stamp_bounds - np.datetime64('1850-01-01')

    time_coord = iris.coords.DimCoord(
        np.asarray(points_days_since, dtype=np.int32),
        bounds=np.asarray(bounds_days_since, dtype=np.int32),
        standard_name='time',
        long_name='time', var_name='time', units=cube.coord('time').units)

    mme = iris.cube.Cube(mme_arr,
                         long_name='multi-model mean of ' + var + ' anomaly',
                         var_name='mme_sie_ano',
                         units=data_cubelist[0].units,
                         dim_coords_and_dims=[(time_coord, 0)])

    std = iris.cube.Cube(std_arr,
                         long_name='multi-model std of ' + var + ' anomaly',
                         var_name='std_sie_ano',
                         units=data_cubelist[0].units,
                         dim_coords_and_dims=[(time_coord, 0)])

    pl_sct_arr = np.zeros(mme_arr.shape)
    pl_sct_arr[:] = np.nan
    pl_sct_arr[np.abs(mme_arr) > std_arr] = 1

    pl_sct = iris.cube.Cube(pl_sct_arr,
                            long_name='significance of ' + var + ' anomaly',
                            var_name='mme_signif',
                            units=data_cubelist[0].units,
                            dim_coords_and_dims=[(time_coord, 0)])

    stat_dic = {'mme_mean': mme, 'mme_std': std, 'mme_significance': pl_sct,
                'n_models': n_models}

    return (stat_dic)


def reform_data_dic(data_dic):
    verb_month_dict = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY',
                       6: 'JUN',  7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT',
                       11: 'NOV',  12: 'DEC'}

    reformed_dic = {}

    for panel_id in data_dic.keys():
        reformed_dic[panel_id] = {}
        for proj_name in data_dic[panel_id].keys():
            mean_merge_cubelist = iris.cube.CubeList()
            sign_merge_cubelist = iris.cube.CubeList()
            vb_month = []
            for month in data_dic[panel_id][proj_name].keys():
                data_dic[panel_id][proj_name][month]['mme_mean'].add_aux_coord(
                    iris.coords.AuxCoord(month,
                                         long_name='month',
                                         var_name='month'))
                mean_merge_cubelist.append(
                    data_dic[panel_id][proj_name][month]['mme_mean'])
                data_dic[panel_id][proj_name][month][
                    'mme_significance'].add_aux_coord(
                    iris.coords.AuxCoord(month,
                                         long_name='month',
                                         var_name='month'))
                sign_merge_cubelist.append(
                    data_dic[panel_id][proj_name][month]['mme_significance'])
                vb_month.append(verb_month_dict[month])
                n_models = data_dic[panel_id][proj_name][month]['n_models']
            reformed_dic[panel_id][proj_name] = {
                'means': mean_merge_cubelist.merge_cube(),
                'significance': sign_merge_cubelist.merge_cube(),
                'n_models': n_models,
                'verb_months': vb_month}

    return (reformed_dic)


def make_panel(data_dict, fig, inner, title='', panel_id='', put_ylabel=False):
    tmp_cmap = plt.cm.coolwarm_r

    tot_subpans = len(data_dict.keys())

    for n_p, proj in enumerate(data_dict.keys()):
        ax = plt.Subplot(fig, inner[n_p])
        pmesh = iplt.pcolormesh(data_dict[proj]['means'], vmin=-1.4, vmax=1.4,
                                cmap=tmp_cmap, axes=ax)
        if panel_id != 'OBS':
            ax.scatter((data_dict[proj]['significance'].coord('time').points *
                        data_dict[proj]['significance'].data),
                       data_dict[proj]['significance'].coord('month').points[:,
                       None] *
                       data_dict[proj]['significance'].data, s=60, c='silver')
        ylims = ax.get_ylim()
        ax.set_ylim(ylims[::-1])
        ax.set_yticks(data_dict[proj]['means'].coord('month').points[::-1])
        ax.set_yticklabels(data_dict[proj]['verb_months'][::-1])
        ax.set_xlim(
            (np.datetime64('1979-01-01') - np.datetime64('1850-01-01')).astype(
                np.int32),
            (np.datetime64('2018-01-01') - np.datetime64('1850-01-01')).astype(
                np.int32))
        yrs = np.arange(1980, 2018, 6)
        tks_dates = (
            np.asarray([np.datetime64(str(yr) + '-07-01') for yr in
                        yrs]) - np.datetime64('1850-01-01')).astype(
            np.int32)
        ax.set_xticks(tks_dates)
        ax.set_xticklabels([str(yr) for yr in yrs])
        if (n_p == 0) & (title != ''):
            ax.set_title(title)

        if n_p == tot_subpans - 1:
            ax.set_xticks(tks_dates)
            ax.set_xticklabels([str(yr) for yr in yrs])
        else:
            ax.set_xticks([])

        if put_ylabel:
            if data_dict[proj]['n_models'] > 1:
                ax.set_ylabel(
                    proj + ' [' + str(data_dict[proj]['n_models']) + ']',
                    rotation=0, labelpad=32)
            else:
                ax.set_ylabel(proj, rotation=0, labelpad=32)
        else:
            ax.set_ylabel('')
        fig.add_subplot(ax)

    return (ax, pmesh)


def make_plot(data_dict):
    ncols = len(data_dict.keys())
    nrows = np.asarray(
        [len(data_dict[hemisph].keys()) for hemisph in data_dict.keys()]).max()

    fig = plt.figure()
    fig.set_size_inches(10., 11.)
    outer = gridspec.GridSpec(nrows, ncols)

    # this should go into the cfg
    panel_order = ['OBS', 'ALL', 'NAT']

    for ncol, hemisph in enumerate(data_dict.keys()):
        h_ratios = []
        for nrow, panel_id in enumerate(panel_order):
            nsubpan = len(data_dict[hemisph][panel_id].keys())
            inner = gridspec.GridSpecFromSubplotSpec(nsubpan, 1,
                                                     subplot_spec=outer[
                                                         nrow, ncol],
                                                     wspace=0.0, hspace=0.0)
            if nrow == 0:
                if hemisph == 'NH':
                    title = 'Arctic SIA'
                else:
                    title = 'Antarctic SIA'
            else:
                title = ''
            h_ratios.append(len(data_dict[hemisph][panel_id]))
            if ncol == 0:
                put_ylabel = True
            else:
                put_ylabel = False

            ax, pmesh = make_panel(data_dict[hemisph][panel_id], fig, inner,
                                   title=title,
                                   panel_id=panel_id, put_ylabel=put_ylabel)

    outer.set_height_ratios(h_ratios)
    cax = fig.add_axes([0.35, 0.06, 0.4, 0.01])
    cbar = fig.colorbar(pmesh, cax=cax, orientation='horizontal')
    cbar.ax.set_xlabel(r'10$^6$ km$^2$' + '\n SIA anomaly')
    fig.subplots_adjust(left=0.185, right=0.98, top=0.9, bottom=0.11,
                        wspace=0.23)  # , hspace=0.2)
    fig.suptitle(
        'Timeseries of 3-year mean of observed and '
        'simulated \n sea ice are (SIA) anomalies',
        fontsize='x-large', x=0.55)

    for n_id, panel_id in enumerate(panel_order):
        bnds = outer[n_id, 0].get_position(fig)
        crds = bnds.bounds
        x = crds[0] / 10
        y = crds[1] + crds[3] / 2
        fig.text(x, y, panel_id, fontweight='semibold')

    return


def main(cfg):
    dtsts = Datasets(cfg)

    data_dict = {'NH': {}, 'SH': {}}
    panel_ids = ['ALL', 'NAT']
    mod_projects = ['CMIP5', 'CMIP6']

    obs_names = set(dtsts.get_info_list('dataset', project='OBS'))

    for hemisph in data_dict.keys():
        for panel_id in panel_ids:
            data_dict[hemisph][panel_id] = {}
            for mod_project in mod_projects:
                data_dict[hemisph][panel_id][mod_project] = {}
        data_dict[hemisph]['OBS'] = {}
        for obs_name in obs_names:
            data_dict[hemisph]['OBS'][obs_name] = {}
        # here an actual processing begins
        months = cfg['month_latitude_' + hemisph][0]
        border_lat = cfg['month_latitude_' + hemisph][1]
        for month in months:
            for panel_id in panel_ids:
                for mod_project in mod_projects:
                    data_dict[hemisph][panel_id][mod_project][month] = {}
                    exp = detect_exp(dtsts, panel_id, mod_project)
                    models = set(dtsts.get_info_list('dataset', exp=exp,
                                                     project=mod_project))
                    models_cubelist = iris.cube.CubeList()
                    for model in models:
                        ens_fnames = dtsts.get_info_list('filename',
                                                         dataset=model,
                                                         exp=exp,
                                                         project=mod_project)
                        ens_cubelist = iris.load(ens_fnames)
                        ens_cubelist = ipcc_sea_ice_diag.select_months(
                            ens_cubelist, month)
                        if hemisph == 'NH':
                            ens_cubelist = ipcc_sea_ice_diag.select_latitudes(
                                ens_cubelist, border_lat, 90)
                        else:
                            ens_cubelist = ipcc_sea_ice_diag.select_latitudes(
                                ens_cubelist, -90, border_lat)
                        sea_ice_cubelist = ipcc_sea_ice_diag.calculate_siparam(
                            ens_cubelist, cfg['seaiceextent'])
                        ano_cubelist = ipcc_sea_ice_diag.substract_ref_period(
                            sea_ice_cubelist, cfg['ref_period'])
                        mean_ano_cubelist = ipcc_sea_ice_diag.n_year_mean(
                            ano_cubelist, 3)
                        mod_ano_cube = ensemble_average(mean_ano_cubelist)
                        models_cubelist.append(mod_ano_cube)
                    mme_dict = mme_stats(models_cubelist)
                    data_dict[hemisph][panel_id][mod_project][month] = mme_dict
            #   here observations added, they are already in sia, so they don't
            #   need the same preprocessing as models.
            sia_var = 'siarea' + hemisph[0].lower()
            obses = dtsts.get_info_list('dataset', project='OBS',
                                        short_name=sia_var)
            for obs in obses:
                obs_fname = dtsts.get_info('filename', dataset=obs,
                                           project='OBS', short_name=sia_var)
                obs_cubelist = iris.load(obs_fname)
                obs_cubelist = ipcc_sea_ice_diag.select_months(obs_cubelist,
                                                               month)
                obs_cubelist = ipcc_sea_ice_diag.substract_ref_period(
                    obs_cubelist, cfg['ref_period'])
                obs_cubelist = ipcc_sea_ice_diag.n_year_mean(obs_cubelist, 3)
                obs_cb = ensemble_average(obs_cubelist)
                data_dict[hemisph]['OBS'][obs][month] = mme_stats([obs_cb])

        data_dict[hemisph] = reform_data_dic(data_dict[hemisph])

    make_plot(data_dict)

    ipcc_sea_ice_diag.figure_handling(cfg, name='fig_3_19_timeseries')

    logger.info('Success')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
