import cf_units
import datetime
import iris
from iris.experimental.equalise_cubes import equalise_attributes
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats, special
import sys

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import group_metadata, run_diagnostic, Datasets
from esmvaltool.diag_scripts.ocean import diagnostic_tools as diagtools
from esmvaltool.diag_scripts.seaice import ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag
from esmvalcore.preprocessor import extract_month


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def select_months(cubelist, months_list):

    res_cubelist = iris.cube.CubeList()

    for cube in cubelist:
        month_cubelist = iris.cube.CubeList()
        for month in months_list:
            m_cube = extract_month(cube, month)
            month_cubelist.append(m_cube)
        res_cubelist.append(month_cubelist)

    return(res_cubelist)

def calculate_sce(cubelist, threshold = 5):

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
        cf_tim = cf_units.num2date(tim.points, orig, calendar)
        year = np.asarray([pnt.year for pnt in cf_tim])
        pnt_dts = np.asarray([datetime.datetime(yr, 7, 1) for yr in year])
        bds_dts = np.asarray([[datetime.datetime(yr, 1, 1), datetime.datetime(yr, 12, 31)] for yr in year])
        tim_coor = iris.coords.DimCoord(cf_units.date2num(pnt_dts, orig, calendar), standard_name='time',
                                        long_name='time', var_name='time', units=tim.units,
                                        bounds=cf_units.date2num(bds_dts, orig, calendar))
        for cube in sub_cubelist:

            area = iris.analysis.cartography.area_weights(cube, normalize=False)
            mask = (cube.data.mask) | (cube.data.data <= threshold)
            area = np.ma.array(area, mask=mask)
            area = area.sum(axis=(1, 2))/ 1e12
            month = cf_units.num2date(cube.coord('time').points[0], orig, calendar).month
            # for now passing paren cube attributes, clean before merging!!!
            sce_cube = iris.cube.Cube(area, long_name='snow cover extent', var_name='sce',
                                units="10^6km2", attributes=cube.attributes, dim_coords_and_dims=[(tim_coor, 0)])
            sce_cube.add_aux_coord(iris.coords.AuxCoord(month, long_name='month', var_name='month'))
            sub_sce_cubelist.append(sce_cube)
        sce_cube = sub_sce_cubelist.merge_cube()
        sce_cube.add_aux_coord(iris.coords.AuxCoord(n, long_name='rec_number', var_name='r_num'))
        sce_cubelist.append(sce_cube)

    return (sce_cubelist)

def monthly_av(cubelist):

    aver_cubelist = iris.cube.CubeList()

    for cube in cubelist:
        aver_cube = cube.collapsed('month', iris.analysis.MEAN)
        aver_cubelist.append(aver_cube)

    return (aver_cubelist)


def shuffle_period(cubelist, period, aver_n):

    shuffled_cubelist = iris.cube.CubeList()

    nyears = period[1] - period[0] + 1

    for cube in cubelist:
        ctl_yrs = len(cube.coord('time').points)
        n_rows = ctl_yrs//nyears
        if ctl_yrs%nyears !=0:
            print('The years in the ctl experiment are not divisible by '+str(nyears)+'last '+str(ctl_yrs%nyears)+
                  ' years were not considered')
        shuffled_cubelist.append()

    return (shuffled_cubelist)


def subtract_ref_period(cubelist, period):

    ano_cubelist = iris.cube.CubeList()

    for cube in cubelist:
        ano_cubelist.append()

    return(ano_cubelist)

# do ensamble average

# calculate stats, mean, 5% percentile, 95% percentile, min, max

def make_panel():

    return

def make_plot():

    return

def main(cfg):

    dtsts = Datasets(cfg)

    projects = ['CMIP5', 'CMIP6']
    # projects might be called through set(keys too)

    for project in projects:
        proj_dtsts = dtsts.get_dataset_info_list(project=project)
        exps = set([dtst['exp'] for dtst in proj_dtsts])
        for exp in exps:
            exp_dtsts = dtsts.get_dataset_info_list(exp=exp, project=project)
            models = set([dtst['dataset'] for dtst in exp_dtsts])
            for model in models:
                model_dtsts = dtsts.get_dataset_info_list(dataset= model, exp= exp, project=project)
                mod_fnames = [dtst['filename'] for dtst in model_dtsts]
                mod_cubelist = ipcc_sea_ice_diag.load_cubelist(mod_fnames)
                mod_cubelist = select_months(mod_cubelist, cfg['months'])
                mod_sce_cubelist = calculate_sce(mod_cubelist)
                mod_sce_cubelist = monthly_av(mod_sce_cubelist)
                if exp == 'piControl':
                    mod_sce_cubelist = shuffle_period(mod_sce_cubelist, cfg['main_period'], cfg['years_for_average'])
                else:
                    mod_sce_cubelist = ipcc_sea_ice_diag.n_year_mean(mod_sce_cubelist,n = cfg ['years_for_average'])
                    # mod_sce_cubelist = subtract_ref_period(mod_sce_cubelist, cfg['ref_period'])


                print('Liza')


    # ipcc_sea_ice_diag.figure_handling(cfg, name='fig_3_19_timeseries')

    logger.info('Success')

if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
