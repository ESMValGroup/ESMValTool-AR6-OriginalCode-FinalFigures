import cf_units
import cftime
import datetime
import iris
from iris.experimental.equalise_cubes import equalise_attributes
import iris.plot as iplt
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import genextreme as gev

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, Datasets, Variables
from esmvaltool.diag_scripts.seaice import \
    ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag
from esmvalcore.preprocessor import regrid

# # This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))


# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def rename_variables(vars, datasets):
    # this function renames the variables in the model to run the script more
    # effectively tasmax is txx and pr is rx1day

    for var in vars.short_names():
        if var == 'tasmax':
            etccdi = 'txx'
        elif var == 'pr':
            etccdi = 'rx1day'
        else:
            continue
        for dtst in datasets.get_dataset_info_list(short_name=var):
            dtst_info = datasets.get_dataset_info(dtst['filename'])
            updated_info = dtst_info
            updated_info['short_name'] = vars.short_name(etccdi)
            updated_info['long_name'] = vars.long_name(etccdi)
            updated_info['standard_name'] = vars.long_name(etccdi)
            datasets.set_data(data={}, path=dtst['filename'],
                              dataset_info=updated_info)

    return


def sort_experiment_info(datasets, variable, project, exp_key):
    # as variable we mean short_name

    exps = set(
        datasets.get_info_list('exp', short_name=variable, project=project))

    for exp in exps:
        if exp_key.lower() == 'nat':
            if 'nat' in exp.lower():
                exp_name = exp
        elif exp_key.lower() == 'all':
            if (exp == 'historical') | ('historical-' in exp):
                exp_name = exp

    return (exp_name)


def obtain_datasets(all_datasets, project, short_name, exp_key):
    if exp_key == 'OBS':
        exp = exp_key
        exp_dtsts = all_datasets.get_dataset_info_list(project=project,
                                                       short_name=short_name)
        dataset_names = list(set([dtst['dataset'] for dtst in exp_dtsts]))
    else:
        exp = sort_experiment_info(all_datasets, short_name, project, exp_key)
        exp_dtsts = all_datasets.get_dataset_info_list(project=project,
                                                       short_name=short_name,
                                                       exp=exp)
        dataset_names = list(set([dtst['dataset'] for dtst in exp_dtsts]))

    return (dataset_names, exp)


def create_obs_mask(obs_filename, var):
    obs_data = iris.load_cube(obs_filename)

    n_lats = len(obs_data.coord('latitude').points)
    n_lons = len(obs_data.coord('longitude').points)

    obs_mask = np.ones((n_lats, n_lons), dtype=bool)

    n_years = len(obs_data.coord('time').points)
    t_param = round((n_years * 0.7) + 0.5)

    last_years_n = n_years - 5
    year_param = 3

    for lat in range(n_lats):
        for lon in range(n_lons):
            if ((np.ma.count(obs_data.data[:, lat, lon]) >= t_param) & (
                np.ma.count(obs_data.data[last_years_n:, lat, lon])
                                                         >= year_param)):
                obs_mask[lat, lon] = False

    return (obs_mask)


def reform_exp_keys(project, exp_keys):
    # this function is made to pass the correct keys into the next loop
    if project == 'OBS':
        exp_keys = ['OBS']
    else:
        if 'OBS' in exp_keys:
            exp_keys.remove('OBS')

    return (exp_keys)


def obtain_filepaths(all_datasets, dataset, project, short_name, exp):
    if exp == 'OBS':
        filepaths = all_datasets.get_dataset_info_list(dataset=dataset,
                                                       project=project,
                                                       short_name=short_name)
    else:
        filepaths = all_datasets.get_dataset_info_list(dataset=dataset,
                                                       project=project,
                                                       short_name=short_name,
                                                       exp=exp)

    return (filepaths)


def covert_to_flx(cubelist):
    converted_cblst = iris.cube.CubeList()

    for cube in cubelist:
        cube = cube * 86400
        # convert precipitation_flux / (kg m-2 s-1) to mm 60 s*60 m*24 h
        cube.standard_name = 'precipitation_amount'
        cube.var_name = 'precip'
        cube.units = 'mm'
        cube.long_name = 'Precipitation'
        converted_cblst.append(cube)

    return (converted_cblst)


def fit_cdf(cubelist, end_year):
    cdf_cubelist = iris.cube.CubeList()
    constr = iris.Constraint(time=lambda cell: cell.point.year <= end_year)

    for cube in cubelist:
        short_cube = cube.extract(constr)
        ntim_sh = len(short_cube.coord('time').points)
        ntim = len(cube.coord('time').points)
        nlat = len(cube.coord('latitude').points)
        nlon = len(cube.coord('longitude').points)
        cdf = np.ma.masked_all((ntim, nlat, nlon))
        for lat in range(0, nlat):
            for lon in range(0, nlon):
                cube_point = cube.data[:, lat, lon]
                short_cube_point = short_cube.data[:, lat, lon]
                if not (short_cube_point.mask.all() | (
                    np.sum(short_cube_point.data[
                       ~short_cube_point.mask] <= 1e-04) > 0.35 * ntim_sh)):
                    gev_par = gev.fit(short_cube_point[~short_cube_point.mask])
                    cdf[:, lat, lon] = gev.cdf(cube_point, gev_par[0],
                                               loc=gev_par[1],
                                               scale=gev_par[
                                                   2]) * 100
                    # *100 to translate it to %
        cdf.mask[cdf == 100] = True
        cdf_cube = iris.cube.Cube(cdf, long_name='Probability index',
                                  var_name='cdf', units='%',
                                  attributes=cube.attributes,
                                dim_coords_and_dims=cube._dim_coords_and_dims)
        cdf_cubelist.append(cdf_cube)

    return (cdf_cubelist)


def dataset_regriding(cubelist, exp, obs_filename):
    if exp == 'OBS':
        regridded_cubelist = cubelist
    else:
        obs_cube = iris.load_cube(obs_filename)
        regridded_cubelist = iris.cube.CubeList()
        for cube in cubelist:
            regridded_cube = regrid(cube, obs_cube, 'linear')
            regridded_cubelist.append(regridded_cube)

    return (regridded_cubelist)


def apply_obs_mask(cubelist, mask):
    for cube in cubelist:
        if cube.shape[1:] == mask.shape:
            cube.data.mask = cube.data.mask | mask
        else:
            logger.info('The regridding did not work correctly')

    return (cubelist)


def create_coords(cubelist, year_n):
    # dirty trick, we try to unify time to merge  the cubelist in the end

    cb = cubelist[0]

    n_t = len(cb.coord('time').points)

    tim = cb.coord('time')
    orig = tim.units.origin
    calendar = tim.units.calendar
    #  dest orig and calendar are converted, so in the end we can merge cubes
    dest_orig = 'days since 1850-1-1 00:00:00'
    dest_calendar = 'gregorian'
    cf_tim = cf_units.num2date(tim.points, orig, calendar)
    year = np.asarray([pnt.year for pnt in cf_tim])
    pnt_dts = np.asarray([datetime.datetime(yr, 7, 1) for yr in year])
    bds_dts = np.asarray(
        [[datetime.datetime(yr, 1, 1), datetime.datetime(yr, 12, 31)] for yr in
         year])
    tim_coor = iris.coords.DimCoord(
        cf_units.date2num(pnt_dts, dest_orig, dest_calendar),
        standard_name='time',
        long_name='time', var_name='time',
        units=cf_units.Unit(dest_orig, dest_calendar),
        bounds=cf_units.date2num(bds_dts, dest_orig, dest_calendar))

    coord = [np.average(tim_coor.points[year_n * i:year_n * i + year_n]) for i
             in range(0, int(n_t / year_n))]
    bnds = [[tim_coor.bounds[year_n * i][0],
             tim_coor.bounds[year_n * i + (year_n - 1)][1]] for i in
            range(0, int(n_t / year_n))]
    if n_t % year_n != 0:
        logger.info('The n of years is not divisible by %s ', str(year_n))

    dcoord = iris.coords.DimCoord(np.asarray(coord), bounds=np.asarray(bnds),
                                  standard_name='time',
                                  units=cf_units.Unit(dest_orig,
                                                      dest_calendar),
                                  long_name='time', var_name='time')

    return (dcoord)


def n_year_mean(cubelist, n):
    # the idea behind it is that we pass the cubelist with the same time coords

    n_aver_cubelist = iris.cube.CubeList()

    dcoord = create_coords(cubelist, n)

    for cube in cubelist:
        n_t = len(cube.coord('time').points)
        if n_t % n != 0:
            # add here a warning that the last is an average of n_t%n==0 years
            print('The n of years is not divisible by ' + str(
                n) + ' last ' + str(
                n_t % n) + ' years were not taken into account')
        data = np.asarray(
            [np.average(cube.data[n * i:n * i + n], axis=0) for i in
             range(0, int(n_t / n))])
        n_aver_cube = iris.cube.Cube(data,
                                     long_name=cube.long_name + ', ' + str(
                                         n) + 'y mean',
                                     var_name=cube.var_name, units=cube.units,
                                     attributes=cube.attributes,
                                     dim_coords_and_dims=[(dcoord, 0)])

        n_aver_cubelist.append(n_aver_cube)

    return (n_aver_cubelist)


def area_wght_averaging(cubelist):
    area_wght_cblst = iris.cube.CubeList()

    for cube in cubelist:
        weights = iris.analysis.cartography.area_weights(cube, normalize=True)
        area_wght_cube = cube.collapsed(('latitude', 'longitude'),
                                        iris.analysis.MEAN, weights=weights)
        area_wght_cblst.append(area_wght_cube)

    return (area_wght_cblst)


def ens_averaging(cubelist):
    if len(cubelist) > 1:

        for n, cube in enumerate(cubelist):
            cube.add_aux_coord(iris.coords.AuxCoord(n, long_name='n_order',
                                                    var_name='n_order'))
        equalise_attributes(cubelist)
        ens_cube = cubelist.merge_cube()
        aver_ens_cube = ens_cube.collapsed('n_order', iris.analysis.MEAN)
        aver_ens_cube.remove_coord('n_order')
    else:
        aver_ens_cube = cubelist[0]

    return (aver_ens_cube)


def calculate_stats(cubelist):
    stats_dic = {}

    for n, cube in enumerate(cubelist):
        cube.add_aux_coord(
            iris.coords.AuxCoord(n, long_name='n_order', var_name='n_order'))
        cube.cell_methods = ''

    equalise_attributes(cubelist)
    all_mods_cube = cubelist.merge_cube()

    stats_dic['all_model_data'] = all_mods_cube
    stats_dic['mean'] = all_mods_cube.collapsed('n_order', iris.analysis.MEAN)
    stats_dic['min'] = all_mods_cube.collapsed('n_order', iris.analysis.MIN)
    stats_dic['max'] = all_mods_cube.collapsed('n_order', iris.analysis.MAX)
    stats_dic['n_models'] = len(cubelist)

    return (stats_dic)


def make_panel(data_dic, variable, exp_key, nrows, ncols, idx, obs_cube):
    ax = plt.subplot(nrows, ncols, idx)

    ax.set_xlim(datetime.datetime(1948, 1, 1), datetime.datetime(2022, 1, 1))
    ax.plot([datetime.datetime(1948, 1, 1), datetime.datetime(2022, 1, 1)],
            [0, 0], c='k', alpha=0.5,
            linestyle='dashed')

    colors = {}

    if variable == 'rx1day':
        ax.set_ylabel('Rx1day (%)')
        ax.set_ylim(-5.5, 10.5)
        ax.set_yticks(np.arange(-5, 11, 5))
        colors['CMIP5'] = (40 / 255, 130 / 255, 189 / 255)
        colors['CMIP6'] = (116 / 255, 196 / 255, 118 / 255)
        colors['ribbon'] = (224 / 255, 236 / 255, 244 / 255)
        colors['lines'] = (146 / 255, 236 / 255, 148 / 255)
    elif variable == 'txx':
        ax.set_ylabel('TXx ($^o$C)')
        ax.set_ylim(-1.2, 2.4)
        ax.set_yticks(np.arange(-1, 3, 1))
        colors['CMIP5'] = (180 / 255, 90 / 255, 0 / 255)
        colors['CMIP6'] = (222 / 255, 45 / 255, 38 / 255)
        colors['ribbon'] = (254 / 255, 229 / 255, 217 / 255)
        colors['lines'] = (255 / 255, 191 / 255, 128 / 255)

    for proj in data_dic.keys():
        if proj == 'CMIP6':
            for mod in range(data_dic[proj]['all_model_data'].shape[0]):
                iplt.plot(data_dic[proj]['all_model_data'][mod],
                          c=colors['lines'], linestyle='solid', axes=ax,
                          linewidth=1)
        elif proj == 'CMIP5':
            # this is done because iris has limitations on plotting functions
            # and there is no better solution on the internet
            tim = data_dic[proj]['mean'].coord('time')
            conv_time = cftime.num2pydate(tim.points, tim.units.origin,
                                          tim.units.calendar)
            ax.fill_between(conv_time, data_dic[proj]['min'].data,
                            data_dic[proj]['max'].data, color=colors['ribbon'],
                            alpha=0.99, linewidth=0)
        iplt.plot(data_dic[proj]['mean'],
                  label=proj + ' (' + str(data_dic[proj]['n_models']) + ')',
                  c=colors[proj],
                  linestyle='solid', axes=ax, linewidth=2)

    iplt.plot(obs_cube, label='Observation', c='k', linestyle='solid', axes=ax,
              linewidth=2)

    if exp_key == 'ALL':
        label = 'Natural and Human Forcing'
    elif exp_key == 'NAT':
        label = 'Natural Forcing'

    leg = ax.legend(loc=2, frameon=False, ncol=1, title=label,
                    fontsize='medium', title_fontsize='large')

    leg._legend_box.align = 'left'

    ax.set_xlabel('Year')

    return


def make_figure(data_dic):
    ncols = len(data_dic.keys())
    # unorthodox way of calculating non observations
    nrows = np.max(
        [len([exp for exp in data_dic[var].keys() if exp != 'OBS']) for var in
         data_dic.keys()])

    fig = plt.figure()
    fig.set_size_inches(12., 9.)

    for n, vrbl in enumerate(sorted(data_dic.keys())[::-1]):
        obs = data_dic[vrbl].pop('OBS')
        for k, exp_key in enumerate(sorted(data_dic[vrbl].keys())):
            make_panel(data_dic[vrbl][exp_key], vrbl, exp_key, ncols, nrows,
                       (n + 1) + (k + 1) * k, obs_cube=obs['OBS'])

    fig.suptitle(
        'Climate Extremes Indices:'
        ' TXx (yearly maximum value of daily maximum temperature) and \n'
        ' Rx1day (yearly maximum 1-day precipitation)', fontsize='x-large')

    return


def main(cfg):
    vrbls = Variables(cfg)
    all_dtsts = Datasets(cfg)

    rename_variables(vrbls, all_dtsts)
    vrbls_list = list(set(all_dtsts.get_info_list('short_name')))

    plotting_dic = {}

    for vrbl in vrbls_list:
        plotting_dic[vrbl] = {}
        raw_exp_keys = ['ALL', 'NAT', 'OBS']
        for exp_key in raw_exp_keys:
            plotting_dic[vrbl][exp_key] = {}
        projects = set(all_dtsts.get_info_list('project', short_name=vrbl))
        obs_filename = \
        all_dtsts.get_dataset_info(project='OBS', short_name=vrbl)['filename']
        obs_mask = create_obs_mask(obs_filename, vrbl)
        for project in sorted(projects):
            exp_keys = reform_exp_keys(project, raw_exp_keys)
            for exp_key in sorted(exp_keys):
                dtsts, exp = obtain_datasets(all_dtsts, project=project,
                                             short_name=vrbl, exp_key=exp_key)
                ens_cubelist = iris.cube.CubeList()
                for dtst in dtsts:
                    flpths = obtain_filepaths(all_dtsts, dataset=dtst,
                                              project=project, short_name=vrbl,
                                              exp=exp)
                    mod_cblst = ipcc_sea_ice_diag.load_cubelist(
                        [dtst['filename'] for dtst in flpths])
                    if vrbl == 'rx1day':
                        if exp != 'OBS':
                            mod_cblst = covert_to_flx(mod_cblst)
                        mod_cblst = fit_cdf(mod_cblst,
                                            cfg['year_gev_end'])
                    mod_cblst = dataset_regriding(mod_cblst, exp, obs_filename)
                    mod_cblst = apply_obs_mask(mod_cblst, obs_mask)
                    ano_mod_cblst = ipcc_sea_ice_diag.substract_ref_period(
                        mod_cblst, cfg['ref_period'])
                    wght_mod_cblst = area_wght_averaging(ano_mod_cblst)
                    aver_mod_cblst = n_year_mean(wght_mod_cblst, 5)
                    mod_cube = ens_averaging(aver_mod_cblst)
                    ens_cubelist.append(mod_cube)
                if exp == 'OBS':
                    plotting_dic[vrbl][exp_key][project] = ens_cubelist[0]
                else:
                    plotting_dic[vrbl][exp_key][project] = calculate_stats(
                        ens_cubelist)

    make_figure(plotting_dic)

    ipcc_sea_ice_diag.figure_handling(cfg, name='figure_xcb32')

    logger.info('Success')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
