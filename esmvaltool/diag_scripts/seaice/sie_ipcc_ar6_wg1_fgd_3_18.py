# This is a script to create a figure 3.20 in Chapter 3 IPCC WGI AR6
# Authors: Elizaveta Malinina (elizaveta.malinina-rieger@canada.ca)
#          Seung-Ki Min, Yeon-Hee Kim, Nathan Gillett

import iris
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats, special
import sys

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, Datasets
from esmvaltool.diag_scripts.seaice import ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag
import esmvaltool.diag_scripts.shared.plot as eplot

# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def calculate_entry_stat(cubelist):

    list_dic = []

    for cube in cubelist:
        if np.all(cube.data.mask) | (np.all(cube.data == 0)):
            mean = np.nan
            slope = np.nan
        else:
            x = np.arange(0, len(cube.coord('time').points))  # here we calculate the trend, orig code logic
            reg = stats.linregress(x, cube.data)
            mean = np.average(cube.data)
            slope = reg.slope
        # list of dictionaries. Not beautiful, but works.
        # It is not needed to know which realisation the data belongs to.
        list_dic.append({'mean': mean, 'lin_reg_slope': slope})

    return (list_dic)


def ens_average(ens_list_dic):

    if len(ens_list_dic) == 1:
        mod_mean = ens_list_dic[0]['mean']
        mod_dec_slope = ens_list_dic[0]['lin_reg_slope'] * 10
    else:
        means_arr = np.asarray([entry['mean'] for entry in ens_list_dic])
        slopes_arr = np.asarray([entry['lin_reg_slope'] for entry in ens_list_dic])
        mod_mean = np.nanmean(means_arr)
        mod_dec_slope = np.nanmean(slopes_arr) * 10

    return (mod_mean, mod_dec_slope)


def model_stats(inp_dict):

    means = np.asarray([inp_dict[key]['mean'] for key in inp_dict.keys()])
    slopes = np.asarray([inp_dict[key]['dec_slope'] for key in inp_dict.keys()])

    # checks if we're not comparing numbers with nans
    mask = np.isfinite(means) & np.isfinite(slopes)

    reg = stats.linregress(means[mask], slopes[mask])

    # calculating p value

    tval = reg.slope / reg.stderr
    df = len(means[mask]) - 2
    pval = special.betainc(df / 2, 0.5, df / (df + tval ** 2))  # this particular calculation was adopted from orig code

    inp_dict['stat_params'] = {}
    inp_dict['stat_params']['slope_models'] = reg.slope
    inp_dict['stat_params']['mme_mean'] = np.average(means[mask])
    inp_dict['stat_params']['mme_slope'] = np.average(slopes[mask])
    inp_dict['stat_params']['intercept'] = reg.intercept
    inp_dict['stat_params']['p_val'] = pval
    inp_dict['stat_params']['corr_coef'] = stats.pearsonr(means[mask], slopes[mask])[0]

    return (inp_dict)


def make_panel(data_dict, nrow, ncol, idx, obs_dic, verb_month, hemisph, proj):

    obs_cbar = plt.cm.Greys_r

    if hemisph == 'NH':
        region = 'Arctic'
    else:
        region = 'Antarctic'

    title = region + ' SIA in ' + verb_month

    ax = plt.subplot(nrow, ncol, idx)

    ax.set_title(title)
    stat_params = data_dict.pop('stat_params')
    obs_cmap_step = int(200 / len(obs_dic.keys()))
    xs = []

    obs_scat = list()
    for n_o, obs in enumerate(obs_dic.keys()):
        obs_scat_p = ax.scatter(obs_dic[obs]['mean'], obs_dic[obs]['dec_slope'],
                                label='OBS: '+obs.split('-')[1], s=200, marker="*", zorder=5,
                                c=obs_cbar(n_o * obs_cmap_step))
        obs_scat.append(obs_scat_p)

    mme_sty = eplot.get_dataset_style('MultiModelMean', proj.lower() +'.yml')
    mme_scat = ax.scatter(stat_params['mme_mean'], stat_params['mme_slope'], s=200,
               marker=mme_sty['mark'], c=mme_sty['color'],
               label='Multi-Model Mean', zorder=4)

    mod_obs = list()
    for n, model in enumerate(sorted(data_dict.keys())):
        sty = eplot.get_dataset_style(model, proj.lower() +'.yml')
        mod_obs_p = ax.scatter(data_dict[model]['mean'], data_dict[model]['dec_slope'],
                   label=model, edgecolor=sty['color'], facecolor='none',
                   linewidths=2, marker=sty['mark'], s=50, zorder=2)
        mod_obs.append(mod_obs_p)
        xs.append(data_dict[model]['mean'])

    xs = np.asarray(xs)
    lin, = ax.plot(xs, xs * stat_params['slope_models'] + stat_params['intercept'], c='k', zorder=1)

    ax.set_ylabel(r'Trend(10$^6$ km$^2$/ decade)')
    ax.set_xlabel(r'Mean (10$^6$ km$^2$)')

    if hemisph == 'NH':
        ax.set_ylim(-1.5, 0)
        ax.set_yticks(np.arange(-1.5, 0.1, 0.5))
        y_text = -1.45
    else:
        ax.set_ylim(-0.9, 0.3)
        ax.set_yticks(np.arange(-0.9, 0.4, 0.3))
        y_text = -0.85

    ax.set_xlim(-0.5, 10)

    ax.text(6, y_text, 'r=' + str(np.around(stat_params['corr_coef'], 2)) +
            ' (p=' + str(np.around(stat_params['p_val'], 2)) + ')')

    if idx % 2 == 0:
        ax.legend(loc=6, bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False, ncol=2)

    return


def make_plot(data_dict, cfg):

    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)

    ncols = len(data_dict.keys())

    verb_month_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                       5: 'May', 6: 'June', 7: 'July', 8: 'August',
                       9: 'September', 10: 'October', 11: 'November',
                       12: 'December'}

    fig = plt.figure()
    fig.set_size_inches(12., 9.)
    fig.set_dpi(300)

    for n_h, hemisph in enumerate(data_dict.keys()):
        verb_month = verb_month_dict[data_dict[hemisph].pop('month')]
        obs_dic = data_dict[hemisph].pop('OBS')
        nrows = len(data_dict[hemisph].keys())
        for n_p, proj in enumerate(data_dict[hemisph].keys()):
            make_panel(data_dict[hemisph][proj], nrows, ncols, (n_h + 1) + (n_p + 1) * n_p, obs_dic, verb_month,
                       hemisph, proj)

    fig.suptitle('Mean (x-axis) sea ice area (SIA) and its trend (y-axis)',
                 fontsize='x-large', x=0.38)
    fig.subplots_adjust(left=0.08, right=0.7, top=0.88, bottom=0.06, wspace=0.3, hspace=0.4)

    for np, proj in enumerate(data_dict[hemisph].keys()):
        fig.text(0.37, 0.92-0.47*np, proj, fontsize = 'x-large')

    return


def main(cfg):

    dtsts = Datasets(cfg)

    # here we create the dictionary which afterwards will be plotted
    data_dict = {'NH': {'CMIP5': {}, 'CMIP6': {}}, 'SH': {'CMIP5': {}, 'CMIP6': {}}}

    for hemisph in data_dict.keys():
        # here we choose the month and the part of hemisphere
        month = cfg['month_latitude_' + hemisph][0]
        border_lat = cfg['month_latitude_' + hemisph][1]
        for project in data_dict[hemisph].keys():
            models = set(dtsts.get_info_list('dataset', project=project))
            for model in models:
                ens_fnames = dtsts.get_info_list('filename', dataset=model, project=project)
                ens_cubelist = iris.load(ens_fnames)
                ens_cubelist = ipcc_sea_ice_diag.select_months(ens_cubelist, month)
                if hemisph == 'NH':
                    ens_cubelist = ipcc_sea_ice_diag.select_latitudes(ens_cubelist, border_lat, 90)
                else:
                    ens_cubelist = ipcc_sea_ice_diag.select_latitudes(ens_cubelist, -90, border_lat)
                # calculating sea ice area or extent, depending on seaiceextent
                sea_ice_cubelist = ipcc_sea_ice_diag.calculate_siparam(ens_cubelist, cfg['seaiceextent'])
                #  calculating climatology and regression slope
                ens_stats = calculate_entry_stat(sea_ice_cubelist)
                # calculating average over ensembles in the model
                mod_mean, mod_dec_slope = ens_average(ens_stats)
                # creating a dictionary for a model with mean and decadal slope
                data_dict[hemisph][project][model] = {'mean': mod_mean}
                data_dict[hemisph][project][model]['dec_slope'] = mod_dec_slope
            #  calculating the statistics for a multi-model ensemble
            data_dict[hemisph][project] = model_stats(data_dict[hemisph][project])
        # here observations are added, they are already provided as sia. no need to process them the same way as models
        sia_var = 'siarea' + hemisph[0].lower()
        obses = dtsts.get_info_list('dataset', project='OBS', short_name=sia_var)
        data_dict[hemisph]['OBS'] = {}
        for obs in obses:
            obs_fname = dtsts.get_info('filename', dataset=obs, project='OBS', short_name=sia_var)
            obs_cubelist = iris.load(obs_fname)
            obs_cubelist = ipcc_sea_ice_diag.select_months(obs_cubelist, month)
            obs_stats = calculate_entry_stat(obs_cubelist)
            data_dict[hemisph]['OBS'][obs] = {'mean': obs_stats[0]['mean']}
            data_dict[hemisph]['OBS'][obs]['dec_slope'] = obs_stats[0]['lin_reg_slope'] * 10  # decadal
        data_dict[hemisph]['month'] = month

    make_plot(data_dict, cfg)

    ipcc_sea_ice_diag.figure_handling(cfg, name='fig_3_18_scatter')
    ipcc_sea_ice_diag.figure_handling(cfg, name='fig_3_18_scatter', img_ext='.png')

    logger.info('Success')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
