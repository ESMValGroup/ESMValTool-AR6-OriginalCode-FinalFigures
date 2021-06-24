# This is a script to create a cross-chapter box 3.2 Fig 1 in Ch 3 IPCC WGI AR6
# Authors: Elizaveta Malinina (elizaveta.malinina-rieger@canada.ca)
#          Seung-Ki Min, Ying Sun, Nathan Gillett & Chapter 11 team
# The Fig. is updated from Fig. 3 Seong et al. (2021) 10.1175/JCLI-D-19-1023.1
# and Fig. 3 from Paik et al. (2020) 10.1175/JCLI-D-20-0002.1

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
from esmvaltool.diag_scripts.seaice import ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag
from esmvalcore.preprocessor import regrid
import esmvaltool.diag_scripts.shared.plot as eplot

# # This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def rename_variables(vars, datasets):

    # function renames variables in the model to run script more effectively
    # tasmax is txx and pr is rx1day

    for var in vars.short_names():
        if var == 'tasmax':
            etccdi = 'txx'
        elif var =='pr':
            etccdi = 'rx1day'
        else:
            continue
        for dtst in datasets.get_dataset_info_list(short_name=var):
            dtst_info = datasets.get_dataset_info(dtst['filename'])
            updated_info = dtst_info
            updated_info ['short_name'] = vars.short_name(etccdi)
            updated_info ['long_name'] = vars.long_name(etccdi)
            updated_info['standard_name'] =vars.long_name(etccdi)
            datasets.set_data(data={}, path=dtst['filename'], dataset_info=updated_info)

    return


def sort_experiment_info(datasets, variable,project, exp_key):
    # as variable we mean short_name

    exps = set(datasets.get_info_list('exp', short_name = variable, project = project))

    for exp in exps:
        if exp_key.lower() == 'nat':
            if 'nat' in exp.lower():
                exp_name = exp
        elif exp_key.lower() == 'all':
            if (exp == 'historical') | ('historical-' in exp):
                exp_name = exp

    return (exp_name)


def obtain_datasets(all_datasets, project, short_name, exp_key):

    # here the datasets are obtained, depending if the data are models or obs

    if exp_key == 'OBS':
        exp = exp_key
        exp_dtsts = all_datasets.get_dataset_info_list(project=project, short_name=short_name)
        dataset_names = list(set([dtst['dataset'] for dtst in exp_dtsts]))
    else:
        exp = sort_experiment_info(all_datasets, short_name, project, exp_key)
        exp_dtsts = all_datasets.get_dataset_info_list(project=project, short_name=short_name, exp=exp)
        dataset_names = list(set([dtst['dataset'] for dtst in exp_dtsts]))

    return (dataset_names, exp)


def create_obs_mask(obs_filename):

    # the observational mask is created from the observational data file

    obs_data = iris.load_cube(obs_filename)

    n_lats = len(obs_data.coord('latitude').points)
    n_lons = len(obs_data.coord('longitude').points)

    obs_mask = np.ones((n_lats, n_lons), dtype=bool)

    n_years = len(obs_data.coord('time').points)
    # grid cells with more than 70% data availability during all years
    # plus data for at least year_param years during last last_years_n are used
    t_param = round((n_years*0.7)+0.5)

    last_years_n = n_years - 5
    year_param = 3

    for lat in range(n_lats):
        for lon in range(n_lons):
            # the mask as boolean array is created
            if ((np.ma.count(obs_data.data[:, lat, lon])>=t_param)&(np.ma.count(obs_data.data[last_years_n:, lat, lon])
                                                                    >= year_param)):
                obs_mask[lat, lon] = False

    return (obs_mask)


def reform_exp_keys(project, exp_keys):
    # function is made in order to pass the correct keys into the next loop
    if project =='OBS':
        exp_keys = ['OBS']
    else:
        if 'OBS' in exp_keys:
            exp_keys.remove('OBS')

    return(exp_keys)


def obtain_filepaths(all_datasets, dataset, project, short_name, exp):

    # here the filepaths depending if it's observations or models are obtained
    if exp == 'OBS':
        filepaths = all_datasets.get_dataset_info_list(dataset=dataset,
                                                       project=project,
                                                       short_name=short_name)
    else:
        filepaths = all_datasets.get_dataset_info_list(dataset=dataset,
                                                       project=project,
                                                       short_name=short_name,
                                                       exp=exp)

    return(filepaths)


def covert_to_flx(cubelist):

    # the data in observations is in mm, but in model it's flux (kg m-2 s-1),
    # so it needs conversion to mm

    converted_cblst = iris.cube.CubeList()

    for cube in cubelist:
        cube = cube * 86400  # convert (kg m-2 s-1) to mm 60 s*60 m*24 h
        # renaming the cube attributes
        cube.standard_name = 'precipitation_amount'
        cube.var_name = 'precip'
        cube.units = 'mm'
        cube.long_name = 'Precipitation'
        converted_cblst.append(cube)

    return (converted_cblst)


def fit_cdf(cubelist, end_year, mod_name, exp, r_dir, proj):

    # here the extreme distribution function is fitted for the rx1day data
    # since the process is very slow, the data is saved to the run directory

    # defining the directory where the extreme data is stored
    cdf_dir = os.path.join(r_dir, 'rx1day_cdfs')
    # if the directory doesn't exist, create it
    if not os.path.isdir(cdf_dir):
        os.mkdir(cdf_dir)

    # here the content of the directory is called
    cont = np.asarray(os.listdir(cdf_dir))

    # checking if a data for the model, experiment, and project already exists
    # if yes, just load the dataset, if not calculate and save
    if np.any([(mod_name in fname) & (proj in fname) & (exp in fname) for fname in cont]):
        fname = cont[[(mod_name in fname) & (proj in fname) & (exp in fname) for fname in cont]][0]
        cdf_cubelist = iris.load(os.path.join(cdf_dir, fname))
    else:
        cdf_cubelist = iris.cube.CubeList()
        # taking only the data until the end_year which is provided in recipe
        constr = iris.Constraint(time=lambda cell: cell.point.year <= end_year)

        for nc, cube in enumerate(cubelist):
            short_cube = cube.extract(constr)
            ntim_sh = len(short_cube.coord('time').points)
            ntim = len(cube.coord('time').points)
            nlat = len(cube.coord('latitude').points)
            nlon = len(cube.coord('longitude').points)
            # creating an array for cdf
            cdf = np.ma.masked_all((ntim, nlat, nlon))
            for lat in range(0, nlat):
                for lon in range(0, nlon):
                    # getting data for all years for a specific point on the
                    # grid in the cube with all data
                    cube_point = cube.data[:, lat, lon]
                    # getting data for all years for a specific point on the
                    # grid in the short cube (which end in the end year)
                    short_cube_point = short_cube.data[:, lat, lon]
                    # checking the masks
                    if not (short_cube_point.mask.all()|(np.sum(short_cube_point.data[~short_cube_point.mask]<=1e-04)>0.35*ntim_sh)):
                        # getting the extreme distribution function parameters
                        gev_par = gev.fit(short_cube_point[~short_cube_point.mask])
                        # cdf is calculated from params and multiplied by 100
                        # to translate it to %
                        cdf[:,lat,lon] = gev.cdf(cube_point, gev_par[0],
                                        loc=gev_par[1], scale=gev_par[2]) * 100
            #  if cdf value is 100% it's masked out
            cdf.mask[cdf==100]=True
            # saving the cdfs to a cube
            cdf_cube=iris.cube.Cube(cdf, long_name='Probability index',
                                    var_name='cdf', units='%',
                                    attributes=cube.attributes,
                                    dim_coords_and_dims=cube._dim_coords_and_dims)
            cdf_cubelist.append(cdf_cube)
        # saving the cdfs for all ensembles in a model to a netcdf file
        iris.save(cdf_cubelist, os.path.join(cdf_dir,
                                         'cdf_' + proj + '_' + exp +
                                         '_' + mod_name + '.nc'))

    return (cdf_cubelist)


def dataset_regriding(cubelist, exp, obs_filename):

    # re-gridding the model dataset to the observational grid
    # if the project is observations, not need to re-grid, only models

    if exp =='OBS':
        regridded_cubelist = cubelist
    else:
        obs_cube = iris.load_cube(obs_filename)
        regridded_cubelist = iris.cube.CubeList()
        for cube in cubelist:
            regridded_cube = regrid(cube, obs_cube, 'linear')
            regridded_cubelist.append(regridded_cube)

    return (regridded_cubelist)


def apply_obs_mask(cubelist, mask):
    # applying observational mask which was creating earlier depending on the
    # data availability in the obs file

    for cube in cubelist:
        if cube.shape[1:] == mask.shape:
            cube.data.mask = cube.data.mask | mask
        else:
            logger.info('The regridding did not work correctly')

    return(cubelist)


def create_coords(cubelist, year_n):
    # dirty trick, we try to unify time to merge the cubelist in the end

    cb = cubelist [0]

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
    bds_dts = np.asarray([[datetime.datetime(yr, 1, 1), datetime.datetime(yr, 12, 31)] for yr in year])
    tim_coor = iris.coords.DimCoord(cf_units.date2num(pnt_dts, dest_orig, dest_calendar), standard_name='time',
                                    long_name='time', var_name='time', units=cf_units.Unit(dest_orig, dest_calendar),
                                    bounds=cf_units.date2num(bds_dts, dest_orig, dest_calendar))

    coord = [np.average(tim_coor.points[year_n*i:year_n*i + year_n]) for i in range(0, int(n_t / year_n))]
    bnds = [[tim_coor.bounds[year_n*i][0], tim_coor.bounds[year_n*i + (year_n - 1)][1]] for i in
            range(0, int(n_t / year_n))]
    if n_t%year_n != 0:
        # raise warning
        logger.info('The n of years is not divisible by '+str(year_n))

    # creating an iris coordinate
    dcoord = iris.coords.DimCoord(np.asarray(coord), bounds=np.asarray(bnds), standard_name='time',
                                  units=cf_units.Unit(dest_orig, dest_calendar), long_name='time', var_name='time')

    return (dcoord)


def n_year_mean(cubelist, n):

    # this function calculates n_year average for the provided cubelist
    # the idea behind it is that we pass the cubelist with the same time coords

    n_aver_cubelist = iris.cube.CubeList()

    # cubelist might have different calendars, this function just chooses one
    dcoord = create_coords(cubelist, n)

    for cube in cubelist:
        n_t = len(cube.coord('time').points)
        if n_t%n!=0:
            # a warning that the last is an average of n_t%n==0 years
            logger.info('The n of years is not divisible by '+str(n)+' last '+str(n_t%n)+' years were not taken into account')
        # averaging the data for n years and saving it to a cube
        data = np.asarray([np.average(cube.data[n * i:n * i + n], axis=0) for i in range(0, int(n_t / n))])
        n_aver_cube = iris.cube.Cube(data, long_name=cube.long_name + ', ' + str(n) + 'y mean',
                                     var_name=cube.var_name, units=cube.units, attributes=cube.attributes,
                                     dim_coords_and_dims=[(dcoord, 0)])

        n_aver_cubelist.append(n_aver_cube)

    return (n_aver_cubelist)


def area_wght_averaging(cubelist):

    # calculating an area averaged mean for each cube in a cubelist

    area_wght_cblst = iris.cube.CubeList()

    for cube in cubelist:
        weights = iris.analysis.cartography.area_weights(cube, normalize=True)
        area_wght_cube = cube.collapsed(('latitude', 'longitude'), iris.analysis.MEAN, weights=weights)
        area_wght_cblst.append(area_wght_cube)

    return (area_wght_cblst)


def ens_averaging(cubelist):

    # averaging the data for the whole ensemble for a model

    if len(cubelist) > 1:
        for n, cube in enumerate(cubelist):
            cube.add_aux_coord(iris.coords.AuxCoord(n, long_name='n_order', var_name='n_order'))
            # some precip cubes have names like 'precip_ano1' and are renamed
            if 'precip' in cube.var_name:
                cube.var_name = 'precip_ano'
        equalise_attributes(cubelist)
        ens_cube = cubelist.merge_cube()
        aver_ens_cube = ens_cube.collapsed('n_order', iris.analysis.MEAN)
        aver_ens_cube.remove_coord('n_order')
    else:
        aver_ens_cube = cubelist[0]

    return (aver_ens_cube)


def calculate_stats(cubelist):

    stats_dic = {}

    n_models = cubelist.__len__()
    max_n_ens = np.asarray([mod_cblist.__len__() for mod_cblist in cubelist]).max()
    n_time = cubelist[0][0].coord('time').points.__len__()

    all_weights = np.zeros((n_time, max_n_ens, n_models))
    all_mod_values = np.ma.masked_all((n_time, max_n_ens, n_models))

    for nm, mod_cblst in enumerate(cubelist):
        n_ensembles = mod_cblst.__len__()
        all_weights[:, 0:n_ensembles, nm] = np.ones((n_time, n_ensembles)) / (
                n_ensembles * n_models)
        for nens, ens_cb in enumerate(mod_cblst):
            all_mod_values[:, nens, nm] = ens_cb.data

    max_ens_coord = iris.coords.DimCoord(np.arange(0, max_n_ens),
                                         long_name='ens_axis',
                                         var_name='ens_axis')
    mod_coord = iris.coords.DimCoord(np.arange(0, n_models),
                                     long_name='mod_axis',
                                     var_name='mod_axis')

    all_vals_cube = iris.cube.Cube(all_mod_values,
                                   long_name='all_model_'+ ens_cb.var_name+'_data',
                                   var_name='all_'+ ens_cb.var_name,
                                   units=ens_cb.units,
                                   dim_coords_and_dims=[
                                       (ens_cb.coord('time'), 0),
                                       (max_ens_coord, 1), (mod_coord, 2)])

    stats_dic['5_95_perc'] = all_vals_cube.collapsed(['ens_axis', 'mod_axis'],
                                          iris.analysis.WPERCENTILE,
                                          percent=[5, 95],
                                          weights=all_weights)
    stats_dic['mean'] = all_vals_cube.collapsed(['ens_axis', 'mod_axis'],
                                          iris.analysis.MEAN,
                                          weights=all_weights)

    stats_dic['n_models'] = n_models

    return(stats_dic)


def make_panel(data_dic, variable, exp_key, nrows, ncols, idx, ref_period, obs_cube):

    ax = plt.subplot(nrows, ncols, idx)

    ax.set_xlim(datetime.datetime(1948, 1, 1), datetime.datetime(2022, 1, 1))
    ax.plot([datetime.datetime(1948, 1, 1), datetime.datetime(2022, 1, 1)],
            [0, 0], c='silver', linewidth=0.7, linestyle='dashed')
    ax.fill_betweenx([15, -15], [datetime.datetime(ref_period[0], 1, 1),
                                 datetime.datetime(ref_period[0], 12, 31)],
                     [datetime.datetime(ref_period[1], 1, 1),
                      datetime.datetime(ref_period[1], 12, 31)], alpha=0.1,
                     color='lightgrey', linewidth=0)

    colors = {}

    colors['CMIP5'] = (37 / 255, 81 / 255, 204 / 255)
    colors['CMIP6'] = (204 / 255, 35 / 255, 35 / 255)
    colors['ribbon'] = (37 / 255, 81 / 255, 204 / 255)
    colors['lines'] = (204 / 255, 35 / 255, 35 / 255)

    if variable == 'rx1day':
        ax.set_ylabel('Rx1day (%)')
        if idx < 3:
            ax.set_title('Annual maximum 1-day precipitation \n (Rx1day)')
        ax.set_ylim(-5.5, 10.5)
        ax.set_yticks(np.arange(-5,11,5))
        txt_y = 3.12
        txt_x = datetime.datetime(1968, 7, 1)
        ax.text(datetime.datetime(ref_period[0]+4, 7, 1), -4.5, 'Reference period', color='grey')

    elif variable == 'txx':
        ax.set_ylabel('TXx ($^o$C)')
        if idx < 3:
            ax.set_title('Annual maximum daily \n maximum temperature (TXx)')
        ax.set_ylim(-1.2, 2.4)
        ax.set_yticks(np.arange(-1,3,1))
        txt_y = 0.6
        txt_x = datetime.datetime(1950, 1, 1)
        ax.text(datetime.datetime(ref_period[0] + 4, 7, 1), -1,
                'Reference period', color='grey')

    for proj in data_dic.keys():
        tim = data_dic[proj]['mean'].coord('time')
        conv_time = cftime.num2pydate(tim.points, tim.units.origin, tim.units.calendar)
        ax.fill_between(conv_time, data_dic[proj]['5_95_perc'][0].data,
                        data_dic[proj]['5_95_perc'][1].data, color=colors[proj],
                        alpha=0.2, linewidth=0)
        iplt.plot(data_dic[proj]['mean'], label=proj +' ('+
                                            str(data_dic[proj]['n_models'])+
                                            ')', c=colors[proj],
                  linestyle='solid', axes=ax)

    iplt.plot(obs_cube, c='k', linestyle='solid', axes=ax)
    ax.text(txt_x, txt_y, 'Observations:\n   '+obs_cube.attributes['title'].split()[0])


    if exp_key == 'ALL':
        label = 'Natural and Human Forcing'
    elif exp_key == 'NAT':
        label = 'Natural Forcing'

    leg = ax.legend(loc=2, frameon=False, ncol=1, title=label, handlelength=0,
                    fontsize='medium', title_fontsize='large', handletextpad=0.2)

    for nt, txt in enumerate(leg.get_texts()):
        col = leg.legendHandles[nt].get_color()
        txt.set_color(col)

    leg._legend_box.align = 'left'

    ax.set_xlabel('Year')

    return


def make_figure(data_dic, cfg):

    ncols = len(data_dic.keys())
    # unorthodox way of calculating non observations
    nrows = np.max([len([exp for exp in data_dic[var].keys() if exp != 'OBS']) for var in data_dic.keys()])

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))

    plt.style.use(st_file)

    fig = plt.figure()
    fig.set_size_inches(12., 8.)

    for n, vrbl in enumerate(sorted(data_dic.keys())[::-1]):
        obs = data_dic[vrbl].pop('OBS')
        for k, exp_key in enumerate(sorted(data_dic[vrbl].keys())):
            make_panel(data_dic[vrbl][exp_key], vrbl, exp_key, ncols, nrows,
                       (n+1)+(k+1)*k, cfg['ref_period'], obs_cube=obs['OBS'])

    fig.suptitle('Climate Extremes Indices', fontsize='x-large')
    fig.set_dpi(250)

    return


def main(cfg):

    vrbls = Variables(cfg)
    all_dtsts = Datasets(cfg)

    # model datasets are already rx1day and txx, but need renaming
    rename_variables(vrbls, all_dtsts)
    vrbls_list = list(set(all_dtsts.get_info_list('short_name')))

    plotting_dic = {}

    for vrbl in vrbls_list:
        plotting_dic[vrbl] = {}
        raw_exp_keys = ['ALL', 'NAT', 'OBS']
        for exp_key in raw_exp_keys:
            plotting_dic[vrbl][exp_key] = {}
        projects = set(all_dtsts.get_info_list('project', short_name=vrbl))
        obs_filename = all_dtsts.get_dataset_info(project='OBS',
                                                  short_name=vrbl)['filename']
        # creating observational mask from obs_filename
        obs_mask = create_obs_mask(obs_filename)
        for project in sorted(projects):
            # this function reforms raw_exp_keys
            exp_keys = reform_exp_keys(project, raw_exp_keys)
            for exp_key in sorted(exp_keys):
                # the datasets are obtained depending on the project
                dtsts, exp = obtain_datasets(all_dtsts, project=project,
                                             short_name=vrbl, exp_key=exp_key)
                ens_cubelist = iris.cube.CubeList()
                for dtst in dtsts:
                    # the filepaths depending on the projects are obtained
                    flpths = obtain_filepaths(all_dtsts, dataset=dtst,
                                              project=project, short_name=vrbl,
                                              exp=exp)
                    mod_cblst = iris.load([dtst['filename'] for dtst in flpths])
                    if vrbl == 'rx1day':
                        if exp != 'OBS':
                            # pr data in models is flux, to be converted to mm
                            mod_cblst = covert_to_flx(mod_cblst)
                        r_dir = cfg['run_dir']
                        # here the extreme distribution function is fitted
                        # it is very-very slow slow process
                        mod_cblst = fit_cdf(mod_cblst, cfg['year_gev_end'],
                                            mod_name=dtst, exp=exp,
                                            r_dir=r_dir, proj=project)
                    # the dataset is re-gridded onto the observational grid
                    mod_cblst = dataset_regriding(mod_cblst, exp, obs_filename)
                    # here the observational dataset is applied
                    mod_cblst = apply_obs_mask(mod_cblst, obs_mask)
                    # here the anomalies are subtracted
                    ano_mod_cblst = ipcc_sea_ice_diag.subtract_ref_period(mod_cblst, cfg['ref_period'])
                    # here the area weighted average is calculated
                    wght_mod_cblst = area_wght_averaging(ano_mod_cblst)
                    # an 5-year mean is calculated
                    aver_mod_cblst = n_year_mean(wght_mod_cblst,5)
                    # mod_cube = ens_averaging(aver_mod_cblst)
                    ens_cubelist.append(aver_mod_cblst)
                # here depending if it's observations or models either just the
                # data is submitted to the dictionary, or statistics for models
                # are calculated
                if exp == 'OBS':
                    plotting_dic[vrbl][exp_key][project] = ens_cubelist[0][0]
                else:
                    plotting_dic[vrbl][exp_key][project] = calculate_stats(ens_cubelist)

    make_figure(plotting_dic, cfg)

    ipcc_sea_ice_diag.figure_handling(cfg, name='figure_xcb32')
    ipcc_sea_ice_diag.figure_handling(cfg, name='figure_xcb32',
                                      img_ext='.png')

    logger.info('Success')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
