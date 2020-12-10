import iris
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats, special
import sys

# import internal esmvaltool modules here
from esmvaltool.diag_scripts.shared import run_diagnostic, Datasets
from esmvaltool.diag_scripts.seaice import \
    ipcc_sea_ice_diag_tools as ipcc_sea_ice_diag

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
            x = np.arange(0, len(cube.coord(
                'time').points))  # here calculated the trend, orig code logic
            reg = stats.linregress(x, cube.data)
            mean = np.average(cube.data)
            slope = reg.slope
        # list of dictionaries. Not beautiful, but works. It is not needed
        # which realisation the data belongs to.
        list_dic.append({'mean': mean, 'lin_reg_slope': slope})

    return (list_dic)


def ens_average(ens_list_dic):
    if len(ens_list_dic) == 1:
        mod_mean = ens_list_dic[0]['mean']
        mod_dec_slope = ens_list_dic[0]['lin_reg_slope'] * 10
    else:
        means_arr = np.asarray([entry['mean'] for entry in ens_list_dic])
        slopes_arr = np.asarray(
            [entry['lin_reg_slope'] for entry in ens_list_dic])
        mod_mean = np.nanmean(means_arr)
        mod_dec_slope = np.nanmean(slopes_arr) * 10

    return (mod_mean, mod_dec_slope)


def model_stats(inp_dict):
    means = np.asarray([inp_dict[key]['mean'] for key in inp_dict.keys()])
    slopes = np.asarray(
        [inp_dict[key]['dec_slope'] for key in inp_dict.keys()])

    #  checks if we're not comparing numbers with nans
    mask = np.isfinite(means) & np.isfinite(slopes)

    reg = stats.linregress(means[mask], slopes[mask])

    # calculating p value

    tval = reg.slope / reg.stderr
    df = len(means[mask]) - 2
    pval = special.betainc(df / 2, 0.5, df / (
            df + tval ** 2))
    # this particular calculation was adopted from orig code

    inp_dict['stat_params'] = {}
    inp_dict['stat_params']['slope_models'] = reg.slope
    inp_dict['stat_params']['mme_mean'] = np.average(means[mask])
    inp_dict['stat_params']['mme_slope'] = np.average(slopes[mask])
    inp_dict['stat_params']['intercept'] = reg.intercept
    inp_dict['stat_params']['p_val'] = pval
    inp_dict['stat_params']['corr_coef'] = \
    stats.pearsonr(means[mask], slopes[mask])[0]

    return (inp_dict)


def make_panel(data_dict, nrow, ncol, idx, obs_dic, verb_month, hemisph, proj):
    tmp_cbar = plt.cm.terrain
    obs_cbar = plt.cm.Greys_r

    if hemisph == 'NH':
        region = 'Arctic'
    else:
        region = 'Antarctic'

    title = region + ' Sea Ice Area in ' + verb_month + ' (' + proj + ')'

    ax = plt.subplot(nrow, ncol, idx)

    ax.set_title(title)
    stat_params = data_dict.pop('stat_params')
    cmap_step = int(256 / len(data_dict.keys()))
    obs_cmap_step = int(200 / len(obs_dic.keys()))
    xs = []

    for n_o, obs in enumerate(obs_dic.keys()):
        ax.scatter(obs_dic[obs]['mean'], obs_dic[obs]['dec_slope'], label=obs,
                   s=100, marker="*",
                   c=obs_cbar(n_o * obs_cmap_step))

    ax.scatter(stat_params['mme_mean'], stat_params['mme_slope'], s=80,
               marker='o', c='r', label='MME')

    for n, model in enumerate(data_dict.keys()):
        ax.scatter(data_dict[model]['mean'], data_dict[model]['dec_slope'],
                   label=model, c=tmp_cbar(n * cmap_step), marker='s', s=50)
        xs.append(data_dict[model]['mean'])

    xs = np.asarray(xs)
    ax.plot(xs, xs * stat_params['slope_models'] + stat_params['intercept'],
            c='k')

    ax.set_ylabel(r'Trend(10$^6$ km$^2$/ decade)')
    ax.set_xlabel(r'Clim. (10$^6$ km$^2$)')

    if hemisph == 'NH':
        ax.set_ylim(-1.5, 0)
        ax.set_yticks(np.arange(-1.5, 0.1, 0.5))
        ax.set_xlim(2, 12)
        y_text = -1.45
    else:
        ax.set_ylim(-0.9, 0.3)
        ax.set_yticks(np.arange(-0.9, 0.4, 0.3))
        ax.set_xlim(0, 12)
        y_text = -0.85

    ax.text(7.5, y_text, 'r=' + str(np.around(stat_params['corr_coef'], 2)) +
            ' (p=' + str(np.around(stat_params['p_val'], 2)) + ')')

    if idx % 2 == 0:
        ax.legend(loc=6, bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False,
                  ncol=2)

    return


def make_plot(data_dict):
    ncols = len(data_dict.keys())

    verb_month_dict = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY',
                       6: 'JUN',
                       7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV',
                       12: 'DEC'}

    fig = plt.figure()
    fig.set_size_inches(12., 9.)

    for n_h, hemisph in enumerate(data_dict.keys()):
        verb_month = verb_month_dict[data_dict[hemisph].pop('month')]
        obs_dic = data_dict[hemisph].pop('OBS')
        nrows = len(data_dict[hemisph].keys())
        for n_p, proj in enumerate(data_dict[hemisph].keys()):
            make_panel(data_dict[hemisph][proj], nrows, ncols,
                       (n_h + 1) + (n_p + 1) * n_p, obs_dic, verb_month,
                       hemisph, proj)

    fig.suptitle(
        'Climatology (x-axis) and trend (y-axis) in \n sea ice are (SIA)',
        fontsize='x-large', x=0.38)
    fig.subplots_adjust(left=0.08, right=0.7, top=0.9, bottom=0.06, wspace=0.3,
                        hspace=0.28)

    return


def main(cfg):
    dtsts = Datasets(cfg)

    data_dict = {'NH': {'CMIP5': {}, 'CMIP6': {}},
                 'SH': {'CMIP5': {}, 'CMIP6': {}}}

    for hemisph in data_dict.keys():
        month = cfg['month_latitude_' + hemisph][0]
        border_lat = cfg['month_latitude_' + hemisph][1]
        for project in data_dict[hemisph].keys():
            models = set(dtsts.get_info_list('dataset', project=project))
            for model in models:
                ens_fnames = dtsts.get_info_list('filename', dataset=model,
                                                 project=project)
                ens_cubelist = iris.load(ens_fnames)
                ens_cubelist = ipcc_sea_ice_diag.select_months(ens_cubelist,
                                                               month)
                if hemisph == 'NH':
                    ens_cubelist = ipcc_sea_ice_diag.select_latitudes(
                        ens_cubelist, border_lat, 90)
                else:
                    ens_cubelist = ipcc_sea_ice_diag.select_latitudes(
                        ens_cubelist, -90, border_lat)
                sea_ice_cubelist = ipcc_sea_ice_diag.calculate_siparam(
                    ens_cubelist, cfg['seaiceextent'])
                ens_stats = calculate_entry_stat(sea_ice_cubelist)
                mod_mean, mod_dec_slope = ens_average(ens_stats)
                data_dict[hemisph][project][model] = {'mean': mod_mean}
                data_dict[hemisph][project][model]['dec_slope'] = mod_dec_slope
            data_dict[hemisph][project] = model_stats(
                data_dict[hemisph][project])
        # here observations are added, they are already provided as sia.
        # no need to process them the same way as models
        sia_var = 'siarea' + hemisph[0].lower()
        obses = dtsts.get_info_list('dataset', project='OBS',
                                    short_name=sia_var)
        data_dict[hemisph]['OBS'] = {}
        for obs in obses:
            obs_fname = dtsts.get_info('filename', dataset=obs, project='OBS',
                                       short_name=sia_var)
            obs_cubelist = iris.load(obs_fname)
            obs_cubelist = ipcc_sea_ice_diag.select_months(obs_cubelist, month)
            obs_stats = calculate_entry_stat(obs_cubelist)
            data_dict[hemisph]['OBS'][obs] = {'mean': obs_stats[0]['mean']}
            data_dict[hemisph]['OBS'][obs]['dec_slope'] = obs_stats[0][
                                            'lin_reg_slope'] * 10  # decadal
        data_dict[hemisph]['month'] = month

    make_plot(data_dict)

    ipcc_sea_ice_diag.figure_handling(cfg, name='fig_3_18_scatter')

    logger.info('Success')


if __name__ == '__main__':
    # always use run_diagnostic() to get the config (the preprocessor
    # nested dictionary holding all the needed information)
    with run_diagnostic() as config:
        # list here the functions that need to run
        main(config)
