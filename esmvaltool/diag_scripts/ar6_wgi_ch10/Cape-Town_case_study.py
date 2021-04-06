import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.gridspec as gridspec
from string import ascii_lowercase
import logging

logger = logging.getLogger(os.path.basename(__file__))
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

# some globals
root_path = os.path.dirname(os.path.realpath(__file__))
exernal_path = os.path.join(root_path, "CH10_additional_data/Cape_Town")
matplotlibrc_path = os.path.join(root_path, "ar6_wgi_ch10.mplstyle")
path_to_ctabels = os.path.join(root_path, "colormaps")
plt.style.use(matplotlibrc_path)

plot_path = "/data/reloclim/normal/IPCC/ffinal/Cape-town"

cfg = {'output_file_type': 'png',
       'savepdf': True,
       'savesvg': True,
       'saveeps': True,
}
ax_title_format = '({})'
title_kwag_default = {'size': 'xx-large',
                      'weight': 'normal',
                      'horizontalalignment': 'left',
                      'verticalalignment': 'bottom'}
zero_line = {'color': 'k', 'linewidth': 1., 'ls': '--'}

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
months = np.array(months)

dgrey = [0.4, 0.4, 0.4]
mgrey = [0.65, 0.65, 0.65]
lgrey = [0.9, 0.9, 0.9]



#######################
# Addapted from main ch10 diagnostic
def get_ax_grid(ax_grid_root, x_pos, y_pos):
    """ Returns ax_grid based on x/y pos.

    x_pos: int/list
    """
    if type(y_pos) == list and type(x_pos) == list:
        ax_grid = ax_grid_root[y_pos[0]: y_pos[1], x_pos[0]: x_pos[1]]
    elif type(y_pos) == int and type(x_pos) == list:
        ax_grid = ax_grid_root[y_pos, x_pos[0]: x_pos[1]]
    elif type(y_pos) == list and type(x_pos) == int:
        ax_grid = ax_grid_root[y_pos[0]: y_pos[1], x_pos]
    elif type(y_pos) == int and type(x_pos) == int:
        ax_grid = ax_grid_root[y_pos, x_pos]

    return ax_grid


def _save_fig(cfg, pngpath, dpi=1200):
    """Save matplotlib figure."""

    plt.savefig(pngpath,
                bbox_inches='tight')
    path = [pngpath]
    if cfg.get('savepdf'):
        pdf_path = path[0].replace('.'+cfg['output_file_type'], '.pdf')
        plt.savefig(pdf_path, format='pdf', dpi=dpi,
                    bbox_inches='tight')
        path.append(pdf_path)
    if cfg.get('saveeps'):
        eps_path = path[0].replace('.'+cfg['output_file_type'], '.eps')
        plt.savefig(eps_path, format='eps', dpi=dpi,
                    bbox_inches='tight')
        path.append(eps_path)
    if cfg.get('savesvg'):
        svg_path = path[0].replace('.'+cfg['output_file_type'], '.svg')
        plt.savefig(svg_path, format='svg', dpi=dpi,
                    bbox_inches='tight')
        path.append(svg_path)
    logger.info("Wrote %s", path)
    plt.close()
    return path


def load_IPCCAR6_colors(color_table, Ncolors=21, reverse=False,
                        path_to_ctabels = path_to_ctabels):
    """ Load IPCC color tables as defined in colormaps dir csvs
    (https://github.com/IPCC-WG1/colormaps)

    Parameters
    ----------
    color_table: str
        Of color_table.txt (a csv file), as listed in colors_dic.values()
        if color_table should be reversed, add '_r' to this string

    Keywords
    --------
    Ncolors: int (default = 21)
    reverse: bool
    path_to_ctabels: str
        filepath to directory

    Returns
    -------
    cm:
        - for colortable in 'categorical_colors_rgb_0-255'
          numpy.ndarray
            of shape (*, 3) of RBG between (0,1.)
        - for colortable in 'continuous_colormaps_rgb_0-1' or
            'discrete_colormaps_rgb_0-255'
          matplotlib.colors.LinearSegmentedColormap

    discrete_colormaps are of maximum Ncolor=21
    categorical_colors are returned as np array
    """
    import matplotlib.colors as mcolors
    import pandas as pd

    colors_dic = {'categorical_colors_rgb_0-255':['bright_cat', 'chem_cat',
                        'cmip_cat', 'contrast_cat', 'dark_cat',
                        'gree-blue_cat', 'rcp_cat', 'red-yellow_cat',
                        'spectrum_cat', 'ssp_cat_1', 'ssp_cat_2'],
                  'continuous_colormaps_rgb_0-1':['chem_div', 'chem_seq',
                        'cryo_div', 'cryo_seq', 'misc_div', 'misc_seq_1',
                        'misc_seq_2', 'misc_seq_3', 'prec_div', 'prec_seq',
                        'slev_div', 'slev_seq', 'temp_div', 'temp_seq',
                        'wind_div', 'wind_seq'],
                  'discrete_colormaps_rgb_0-255':['chem_div_disc',
                        'chem_seq_disc', 'cryo_div_disc', 'cryo_seq_disc',
                        'misc_div_disc', 'misc_seq_1_disc', 'misc_seq_2_disc',
                        'misc_seq_3_disc', 'prec_div_disc', 'prec_seq_disc',
                        'slev_div_disc', 'slev_seq_disc', 'temp_div_disc',
                        'temp_seq_disc', 'wind_div_disc', 'wind_seq_disc']}

    if color_table.split('_')[-1] == 'r':
        reverse = True
        color_table = color_table.split('_r')[0]

    if not any([color_table in vs for k,vs in colors_dic.items()]):
        logger.error("Color table {} not found".format(color_table))
        raise NotImplementedError

    for k,vs in colors_dic.items():
        if color_table in vs:
            subdir = k
            break

    lf = os.path.join(path_to_ctabels, subdir, color_table + '.txt')
    logger.info("Loading color table information from {}.".format(lf))

    if subdir == 'categorical_colors_rgb_0-255':
        rgb_in_txt = np.loadtxt(lf)
        cm = rgb_in_txt/255.
        return(cm)
    elif subdir == 'continuous_colormaps_rgb_0-1':
        rgb_in_txt = np.loadtxt(lf)
        if reverse:
            rgb_in_txt = rgb_in_txt[::-1]
        cm = mcolors.LinearSegmentedColormap.from_list(color_table,
                                                       rgb_in_txt, N=Ncolors)
        return(cm)
    elif subdir == 'discrete_colormaps_rgb_0-255':
        if Ncolors > 21:
            # logger.warning("{} only available for maximum of 21 colors. "\
            #                "Setting Number of colors to 21".format(subdir))
            Ncolors = 21
        df = pd.read_csv(lf)

        str_key_table = '_'.join(color_table.split('_')[:-1])

        for idx in df.index:
            strcell = str(df.iloc[idx].values[0])
            if str_key_table in strcell and \
                int(strcell.split('_')[-1]) == Ncolors :
                col_data = df.iloc[idx+1:idx+Ncolors+1].values
                col_data = [[int(da) for da in dat[0].split(' ')] for dat in col_data]
                rgb_in_txt = np.array(col_data)
                break
        rgb_in_txt = rgb_in_txt/255.
        if reverse:
            rgb_in_txt = rgb_in_txt[::-1]
        cm = mcolors.LinearSegmentedColormap.from_list(
             color_table+'_'+str(Ncolors), rgb_in_txt, N=Ncolors)
        return(cm)


lcolors = {
           '2015':load_IPCCAR6_colors('ssp_cat_1')[3],
           '2016':load_IPCCAR6_colors('ssp_cat_1')[2],
           '2017':load_IPCCAR6_colors('ssp_cat_1')[4],
           '1933-2014':mgrey,

           'station-based':'k',
           'stations':'k',

           'NCEP/NCAR': load_IPCCAR6_colors('ssp_cat_2')[5],
           'ERA-20C':load_IPCCAR6_colors('ssp_cat_2')[8],
           '20CR':load_IPCCAR6_colors('ssp_cat_2')[2],
           'CRU TS':load_IPCCAR6_colors('gree-blue_cat')[3],
           'GPCC':load_IPCCAR6_colors('gree-blue_cat')[4],

           'CMIP5 historical-RCP85':load_IPCCAR6_colors('cmip_cat')[1],
           'CMIP5':load_IPCCAR6_colors('cmip_cat')[1],
           'CMIP6 historical-SSP585':load_IPCCAR6_colors('cmip_cat')[0],
           'CMIP6':load_IPCCAR6_colors('cmip_cat')[0],
           '8km CCAM': load_IPCCAR6_colors('ssp_cat_1')[4],
           'CORDEX': load_IPCCAR6_colors('ssp_cat_1')[0],
           'MIROC6': load_IPCCAR6_colors('ssp_cat_1')[3],
}

# 2015: orange,
# 2016: red,
# 2017: purple,
# 1933-2014: grey,
# station-based/stations: black,
# NCEP/NCAR light: blue,
# ERA20C dark: red,
# 20CR: yellow,
# CRU TS: olive,
# GPCC: green,
# CMIP5: blue,
# CMIP6: red,
# 8km CCAM: purple,
# CORDEX: cyan,
# MIROC6: orange,



rename_dic = {# C1
              'SAMCMIP5_0.95': 'CMIP5',
              'SAMCMIP5_0.05': 'CMIP5',
              'SAMCMIP5': 'CMIP5',
              'SAMCMIP6': 'CMIP6',
              'SAMCMIP6_0.95': 'CMIP6',
              'SAMCMIP6_0.05': 'CMIP6',
              'SAMMIROC6': 'MIROC6',
              'SAMMIROC6_0.05': 'MIROC6',
              'SAMMIROC6_0.95': 'MIROC6',
              'STATIONS': 'station-based',
              'NCEP': 'NCEP/NCAR',
              'ERA20c': 'ERA-20C',
              'ERA20C': 'ERA-20C',
              'NOAA': '20CR',
              # C2
              'CMIP5': 'CMIP5',
              'CMIP5_0.95': 'CMIP5',
              'CMIP5_0.05': 'CMIP5',
              'CMIP6': 'CMIP6',
              'CMIP6_0.95': 'CMIP6',
              'CMIP6_0.05': 'CMIP6',
              'CORDEX': 'CORDEX',
              'CORDEX_0.95': 'CORDEX',
              'CORDEX_0.05': 'CORDEX',
              'MIROC6': 'MIROC6',
              'MIROC6_0.05': 'MIROC6',
              'MIROC6_0.95': 'MIROC6',
              'CSIR': '8km CCAM',
              'CSIR_0.95': '8km CCAM',
              'CSIR_0.05': '8km CCAM',
              'STATIONS': 'station-based',
              'GPCC_V2018': 'GPCC',
              'GPCCV2018': 'GPCC',
              'CRU_4.03': 'CRU TS',
              'CRU 4.03': 'CRU TS'
    }

panels_dic = {'a': {'title': 'Annual precipitation accumulation',
                    'input': 'panelA.csv',
                    'figsize': [8, 6],
                    'ylabel': '(mm)',
                    'ylims': (0.,699.),
                    'xlabel': '',
                    'xlims': (-0.5, 11.5),
                    'legend_loc':2,
                    },
              'b': {'title': 'Annual precipitation cycle',
                    'input': 'panelB_v2.csv',
                    'figsize': [8, 6],
                    'ylabel': '(mm month$^{-1}$)', #Precipitation
                    'ylims': (0.,120.),
                    'xlabel': '',
                    'xlims': (-0.5, 11.5),
                    'legend_loc':2
                    },
              'c': {'title': 'Southern Annular Mode (SAM) and precipitation anomalies',
                    'input': ['panelC_SAM_v2.csv', 'panelC_rainfall_v2.csv'],
                    'figsize': [16, 8],
                    'ylims': (-49,62),
                    'ylabel_1': 'SAM index ()',
                    'ylims_1': (-2.5,2.5),
                    'legend_loc_1': 2,
                    'legend_col_1': 3,
                    'ylabel_2': 'Precipitation (%)',
                    'ylims_2': (-49.,16),
                    'legend_loc_2': 3,
                    'legend_col_2': 3,
                    'xlabel': '',
                    'xlims': (1925, 2090),
                    'legend_loc':2,
                    },
              'd': {'title': 'SAM and precipitation trends',
                    'input': [['panelD_SAM_1979-2017_v2.csv',
                               'panelD_SAM_1933-2017_v2.csv',
                               'panelD_SAM_2018-2100_v2.csv'],
                              ['panelD_rainfall_1979-2017_v2.csv',
                               'panelD_rainfall_1933-2017_v2.csv',
                               'panelD_rainfall_2018-2100_v2.csv',
                               ]],
                    'figsize': [16, 8],
                    'ylims': (-6.5, 10.1),
                    'ylabel_1': 'SAM index trend (1 year$^{-1}$)',
                    'ylims_1': (-.06,.08),
                    'legend_loc_1': 9,
                    'legend_col_1': 3,
                    'ylabel_2': 'Precipitation trend (mm year$^{-1}$)',
                    'ylims_2': (-7.,4.25),
                    'legend_loc_2': 8,
                    'legend_col_2': 2,
                    'xlabels': ['1979-2017', '1933-2017', '2018-2100'],
                    'xlims': (1925, 2090),
                    'legend_loc':2,
                    },
            }

combi_fig =  {'figure_name': 'Fig_18',
              'title_leading': '({ascii_lowercase})',
              'figsize': [20, 40],
              'gridspec': [200, 60],
              'diagnostics_position': {
                  'a': {'position': 1,
                        'x_pos': [4, 30],
                        'y_pos': [2, 20],
                        'y_pos_title': 0,
                        'x_pos_title': 0,
                        },
                  'b': {'position': 2,
                        'x_pos': [34, 60],
                        'y_pos': [2, 20],
                        'y_pos_title': 0,
                        'x_pos_title': 30,
                        },
                  'c': {'position': 3,
                        'x_pos': [4, 60],
                        'y_pos': [26, 54],
                        'y_pos_title': 24,
                        'x_pos_title': 0,
                        },
                  'd': {'position': 4,
                        'x_pos': [4, 60],
                        'y_pos': [61, 90],
                        'y_pos_title': 58,
                        'x_pos_title': 0,
                        },
              }
          }


def Cape_Town():

    # load data
    for panel, diagnostic in panels_dic.items():
        logger.info("Loading panel %s data")
        data = []
        if isinstance(diagnostic['input'], str):
            in_file = os.path.join(exernal_path, diagnostic['input'])
            df = pd.read_csv(in_file)
            if panel == 'a' and 'Unnamed: 0' in df.columns:
                df['year'] = [val[:4] for val in df['Unnamed: 0'].values]
                df['year'] = df['year'].astype(int)
            data = df
        else:
            for inpu in diagnostic['input']:
                if isinstance(inpu, str):
                    in_file = os.path.join(exernal_path, inpu)
                    df = pd.read_csv(in_file)
                    if 'Unnamed: 0' in df.columns:
                        df['year'] = [val[:4]
                                      for val in df['Unnamed: 0'].values]
                        df['year'] = df['year'].astype(int)
                    data.append(df)
                else:
                    sub_data = []
                    for inp in inpu:
                        in_file = os.path.join(exernal_path, inp)
                        df = pd.read_csv(in_file)
                        sub_data.append(df)
                    data.append(sub_data)
        diagnostic.update({'data': data})

    ##############
    # make single panel plots
    panel = 'a'
    diagnostic = panels_dic[panel]

    png_name = f"{panel}.{cfg['output_file_type']}"
    file_path = os.path.join(plot_path, png_name)
    logger.info("Plotting panel %s results to %s", panel, file_path)

    fig = plt.figure(figsize=diagnostic['figsize'], tight_layout=True)
    ax = fig.add_subplot(111)
    ax = fill_ax_pr_accumulation(ax, diagnostic)
    _save_fig(cfg, file_path)


    ###
    panel = 'b'
    diagnostic = panels_dic[panel]

    png_name = f"{panel}.{cfg['output_file_type']}"
    file_path = os.path.join(plot_path, png_name)
    logger.info("Plotting panel %s results to %s", panel, file_path)

    fig = plt.figure(figsize=diagnostic['figsize'], tight_layout=True)
    ax = fig.add_subplot(111)
    ax = fill_ax_pr_cycle(ax, diagnostic)
    _save_fig(cfg, file_path)


    ###
    panel = 'c'
    diagnostic = panels_dic[panel]

    png_name = f"{panel}.{cfg['output_file_type']}"
    file_path = os.path.join(plot_path, png_name)
    logger.info("Plotting panel %s results to %s", panel, file_path)

    fig = plt.figure(figsize=diagnostic['figsize'], tight_layout=True)
    ax = fig.add_subplot(111)
    ax = fill_ax_pr_timeseries(ax, diagnostic)
    _save_fig(cfg, file_path)


    ###
    panel = 'd'
    diagnostic = panels_dic[panel]

    png_name = f"{panel}.{cfg['output_file_type']}"
    file_path = os.path.join(plot_path, png_name)
    logger.info("Plotting panel %s results to %s", panel, file_path)

    fig = plt.figure(figsize=diagnostic['figsize'], tight_layout=True)
    ax = fig.add_subplot(111)
    ax = fill_ax_pr_trends(ax, diagnostic)
    _save_fig(cfg, file_path)


    ##############
    # make final plots
    png_name = f"{combi_fig['figure_name']}.{cfg['output_file_type']}"
    file_path = os.path.join(plot_path, png_name)
    logger.info("Plotting final plot to %s", file_path)

    fig = plt.figure(figsize=combi_fig['figsize'])
    ax_grid_root = gridspec.GridSpec(combi_fig['gridspec'][0],
                                     combi_fig['gridspec'][1])
    for diag_name, pos in combi_fig['diagnostics_position'].items():
        diagnostic = panels_dic[diag_name]

        title_prestr = ax_title_format.format(
            ascii_lowercase[pos['position'] - 1])
        title = f"{title_prestr} {diagnostic['title']}"
        ax_grid = get_ax_grid(ax_grid_root, pos['x_pos_title'],
                              pos['y_pos_title'])
        ax = fig.add_subplot(ax_grid)
        ax.axis('off')
        ax.text(0.5, 0.5, title, **title_kwag_default)

        ax_grid = get_ax_grid(ax_grid_root, pos['x_pos'], pos['y_pos'])
        ax = fig.add_subplot(ax_grid)
        if diag_name == 'a':
            ax = fill_ax_pr_accumulation(ax, diagnostic)
        elif diag_name == 'b':
            ax = fill_ax_pr_cycle(ax, diagnostic)
        elif diag_name == 'c':
            ax = fill_ax_pr_timeseries(ax, diagnostic)
        elif diag_name == 'd':
            ax = fill_ax_pr_trends(ax, diagnostic)

    _save_fig(cfg, file_path)


###########################
# plotting functions
def fill_ax_pr_accumulation(ax, plot_dic):
    df = plot_dic['data']
    for index, row in df.iterrows():
        if row['year'] in [2015, 2016, 2017]:
            str_year = str(row['year'])
            ax.plot(row[months].values, color=lcolors[str_year], lw=2,
                    label=str_year)
        else:
            ax.plot(row[months].values, color=lcolors['1933-2014'],
                    label='1933-2014')

    ax.set_xlabel(plot_dic['xlabel'])
    ax.set_xlim(plot_dic['xlims'])
    ax.set_ylabel(plot_dic['ylabel'])
    ax.set_ylim(plot_dic['ylims'])
    # ax.set_title(plot_dic['title'])

    ax.tick_params(axis='x', which=u'minor', length=0)
    # ax.set_xticks(np.arange(0,len(months))[::2])
    # ax.set_xticklabels(months[np.arange(0,len(months))[::2]])
    ax.set_xticks(np.arange(0,len(months)))
    ax.set_xticklabels(months[np.arange(0,len(months))])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    # for k in by_label.keys():
    #     by_label[k].set_linewidth(0)
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc=plot_dic['legend_loc'])

    return(ax)


def fill_ax_pr_cycle(ax, plot_dic, bar_keys = ['2015','2016','2017'],
                     width = 0.3):
    df = plot_dic['data']
    x = np.arange(len(df))

    for yind,year in enumerate(bar_keys):
        rects = ax.bar(x - width + width*yind, df[year], width, label=year,
                       color=lcolors[year])
    ax.plot(df['1933-2014 climatology'].values, color='k', lw=2,
            label='1933-2014 climatology')
    ax.plot(df['2015-2017 mean'].values, color=dgrey, lw=2,
            label='2015-2017 mean')

    ax.set_xlabel(plot_dic['xlabel'])
    ax.set_xlim(plot_dic['xlims'])
    ax.set_ylabel(plot_dic['ylabel'])
    ax.set_ylim(plot_dic['ylims'])
    # ax.set_title(plot_dic['title'])

    ax.tick_params(axis='x', which=u'minor', length=pylab.rcParams['xtick.major.size'])
    ax.tick_params(axis='x', which='major', width=0.)
    ax.set_xticks(np.arange(-0.5,len(df)), minor=True)
    # ax.set_xticks(np.arange(0,len(months))[::2])
    # ax.set_xticklabels(months[np.arange(0,len(months))[::2]])
    ax.set_xticks(np.arange(0,len(months)))
    ax.set_xticklabels(months)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    # for k in by_label.keys():
    #     by_label[k].set_linewidth(0)
    ax.legend(by_label.values(), by_label.keys(), ncol=1, loc=plot_dic['legend_loc'])

    return(ax)


def fill_ax_pr_timeseries(ax, plot_dic):
    df1 = plot_dic['data'][0]
    df2 = plot_dic['data'][1]

    ax2_ofs = 40
    ax2_factor = 10

    ax2 = ax.twinx()

    for k in ['SAMCMIP5', 'SAMCMIP6', 'SAMMIROC6']:
        ks = [k+'_0.05', k+'_0.95']
        if k in rename_dic.keys():
            lab = rename_dic[k]
        else: lab=k
        ax2.fill_between(df1['year'].values,
                df1[ks[0]].values * ax2_factor + ax2_ofs,
                df1[ks[1]].values * ax2_factor + ax2_ofs,
                color=lcolors[lab], alpha = 0.2, zorder=2, ls='-', label=lab) # , label=ens
        ax2.plot(df1['year'].values,
                 df1[ks[0]].values * ax2_factor + ax2_ofs,
                color=lcolors[lab], zorder=1, ls='-', lw=0.5, alpha = 0.6) # , label=ens
        ax2.plot(df1['year'].values,
                df1[ks[1]].values * ax2_factor + ax2_ofs,
                color=lcolors[lab], zorder=1, ls='-', lw=0.5, alpha = 0.6) # , label=ens
    for k in ['STATIONS', 'NCEP', 'ERA20c', 'NOAA']:
        if k in rename_dic.keys():
            lab = rename_dic[k]
        else: lab=k
        ax2.plot(df1['year'].values,
                df1[k].values * ax2_factor + ax2_ofs,
                color=lcolors[lab], label=lab) #, lw=lw)

    ax2.yaxis.set_label_position("left")
    ax2.spines['left'].set_visible(True)
    ax2.tick_params(axis='y', which='both', labelleft=True, labelright=False, left=True, right=False)
    # ax2.yaxis.set_label_position("right")
    # ax2.spines['right'].set_visible(True)

    ax2_yticks = np.arange(-2.0,2.1,0.5)
    ax2_ytick_locs = ax2_yticks * ax2_factor + ax2_ofs
    ax2.set_yticks(ax2_ytick_locs)
    ax2.set_yticklabels(ax2_yticks)

    ax2.tick_params(axis='y', which=u'minor', length=0, pad=35)
    ax2.set_yticks([ax2_ytick_locs.mean()+0.01], minor=True)
    ax2.set_yticklabels([plot_dic['ylabel_1']], minor=True, rotation=90, verticalalignment='center')

    handles, labels = ax2.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), ncol=plot_dic['legend_col_1'], loc=plot_dic['legend_loc_1'])

    for k in ['CMIP5', 'CMIP6', 'CORDEX', 'CSIR', 'MIROC6']:
        ks = [k+'_0.05', k+'_0.95']
        if k in rename_dic.keys():
            lab = rename_dic[k]
        else: lab=k
        ax.fill_between(df2['year'].values,
                df2[ks[0]].values,
                df2[ks[1]].values,
                color=lcolors[lab], alpha = 0.2, zorder=2, label=lab) # , label=ens
        ax.plot(df2['year'].values,
                df2[ks[0]].values,
                color=lcolors[lab], zorder=1, ls='-', lw=0.5, alpha = 0.6)
        ax.plot(df2['year'].values,
                df2[ks[1]].values,
                color=lcolors[lab], zorder=1, ls='-', lw=0.5, alpha = 0.6)

    for k in ['STATIONS', 'GPCC_V2018', 'CRU_4.03']:
        if k in rename_dic.keys():
            lab = rename_dic[k]
        else: lab = k
        ax.plot(df2['year'].values,
                df2[k].values,
                color=lcolors[lab], label=lab) #, lw=lw)

    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), ncol=plot_dic['legend_col_2'], loc=plot_dic['legend_loc_2'])

    ax_yticks = np.arange(-40, 11, 10)
    ax.set_yticks(ax_yticks)
    ax.set_yticklabels(ax_yticks)
    ax.tick_params(axis='y', which=u'minor', length=0, pad=35)
    ax.set_yticks([ax_yticks.mean()+0.01], minor=True)
    ax.set_yticklabels([plot_dic['ylabel_2']], minor=True, rotation=90, verticalalignment='center')

    ax.set_ylim(plot_dic['ylims'])
    ax2.set_ylim(plot_dic['ylims'])

    ax.hlines([plot_dic['ylims_2'][1]], ax.get_xlim()[0], ax.get_xlim()[1], colors=['k'],
            linestyles=['-'], linewidth=pylab.rcParams['axes.linewidth'], zorder=3)

    ax.hlines([[0]*2, [0 * ax2_factor + ax2_ofs]*2],
            ax.get_xlim()[0], ax.get_xlim()[1],
            linestyles=['-'], zorder=1, **zero_line)

    # ax.set_title(plot_dic['title'])
    return(ax)


def fill_ax_pr_trends(ax, plot_dic):
    dfs1 = plot_dic['data'][0]
    dfs2 = plot_dic['data'][1]

    ax2 = ax.twinx()
    ax2_ofs = 6.1
    ax2_factor = 50

    spacing_mod = 1
    spacing_obs = 2
    spacing_mod2obs = 3
    spacing = 5
    x_ranges = [] # gives the x postion of the single entries
    for df in dfs1:
        columns = list(df.keys())
        columns.remove('Unnamed: 0')
        modcolumns = [col
                      for col in columns
                      if 'CMIP' in col or 'MIROC' in col]
        try:
            xs = np.arange(0, len(modcolumns), spacing_mod) + x_ranges[-1][-1] + spacing
        except IndexError:
            xs = np.arange(0, len(modcolumns), spacing_mod)
        xs_obs = np.arange(0, (len(columns) - len(modcolumns)) * spacing_obs, spacing_obs) + xs[-1] + spacing_mod2obs
        xs = np.concatenate((xs, xs_obs), axis=0)
        x_ranges.append(xs)

        for k,x in zip(columns, xs):
            if 'CMIP6' in k: lab = 'CMIP6'
            elif 'CMIP5' in k: lab = 'CMIP5'
            elif 'MIROC6' in k: lab = 'MIROC6'
            elif k in rename_dic.keys():
                lab = rename_dic[k]
            else: lab = k
            ax2.plot([x ,x],
                    [df.loc[1, k] * ax2_factor + ax2_ofs, df.loc[2, k] * ax2_factor + ax2_ofs],
                    c=lcolors[lab], linewidth=1., label=lab) #clip_on=False,
            ax2.plot(x, df.loc[0, k] * ax2_factor + ax2_ofs, c=lcolors[lab], linewidth=2.,
                    marker="o", label=lab, markersize=3., fillstyle='full') #, markeredgewidth=1)

    ax2.yaxis.set_label_position("left")
    ax2.spines['left'].set_visible(True)
    ax2.tick_params(axis='y', which='both', labelleft=True, labelright=False, left=True, right=False)
    # ax2.yaxis.set_label_position("right")
    # ax2.spines['right'].set_visible(True)

    # ax2_yticks = np.arange(-.04,.07,0.02)
    ax2_yticks = np.arange(-2, 9, 2) / 100.
    ax2_ytick_locs = ax2_yticks * ax2_factor + ax2_ofs
    ax2.set_yticks(ax2_ytick_locs)
    ax2.set_yticklabels(ax2_yticks)

    ax2.tick_params(axis='y', which=u'minor', length=0, pad=35)
    ax2.set_yticks([ax2_ytick_locs.mean()+0.01], minor=True)
    ax2.set_yticklabels([plot_dic['ylabel_1']], minor=True, rotation=90, verticalalignment='center')

    handles, labels = ax2.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), ncol=plot_dic['legend_col_1'], loc=plot_dic['legend_loc_1'])


    for df, xs in zip(dfs2, x_ranges):
        columns = list(df.keys())
        columns.remove('Unnamed: 0')

        for k,x in zip(columns, xs):
            if 'CMIP6' in k: lab = 'CMIP6'
            elif 'CMIP5' in k: lab = 'CMIP5'
            elif 'MIROC6' in k: lab = 'MIROC6'
            elif k in rename_dic.keys():
                lab = rename_dic[k]
            else: lab=k
            ax.plot([x ,x],
                    [df.loc[1, k], df.loc[2, k]],
                    c=lcolors[lab], linewidth=1., label=lab) #clip_on=False,
            ax.plot(x, df.loc[0, k], c=lcolors[lab], linewidth=2.,
                    marker="o", label=lab, markersize=3., fillstyle='full') #, markeredgewidth=1)


    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), ncol=plot_dic['legend_col_2'], loc=plot_dic['legend_loc_2'])

    ax_yticks = np.arange(-6, 5, 1)
    ax.set_yticks(ax_yticks)
    ax.set_yticklabels(ax_yticks)
    ax.tick_params(axis='y', which=u'minor', length=0, pad=35)
    ax.set_yticks([ax_yticks.mean()+0.01], minor=True)
    ax.set_yticklabels([plot_dic['ylabel_2']], minor=True, rotation=90, verticalalignment='center')

    ax.set_xlim(x_ranges[0][0]-spacing_obs, x_ranges[-1][-1]+spacing_obs)
    ax.set_ylim(plot_dic['ylims'])
    ax2.set_ylim(plot_dic['ylims'])

    # make heading
    ax.tick_params(axis='x', which=u'both', length=0)
    ax.tick_params(axis='x', labeltop=True, labelbottom=False, pad=15) # , left=True, right=False)
    x_means = [np.mean(xs) for xs in x_ranges]
    ax.set_xticks(x_means)
    ax.set_xticklabels(plot_dic['xlabels'], verticalalignment='center')


    ax.hlines([plot_dic['ylims_2'][1]], ax.get_xlim()[0], ax.get_xlim()[1], colors=['k'],
            linestyles=['-'], linewidth=pylab.rcParams['axes.linewidth'], zorder=3)

    ax.hlines([[0]*2, [0 * ax2_factor + ax2_ofs]*2],
            ax.get_xlim()[0], ax.get_xlim()[1],
            linestyles=['-'], zorder=1, **zero_line)

    xv_lines = [(x_ranges[0][-1] + x_ranges[1][1]) / 2,
                (x_ranges[1][-1] + x_ranges[2][1]) / 2]
    ax.vlines(xv_lines,
            ax.get_ylim()[0], ax.get_ylim()[1], colors='k',
            linestyles=['-'], linewidth=pylab.rcParams['axes.linewidth'], zorder=2)
    # ax.set_title(plot_dic['title'])

    return(ax)


#######################
# ESMValTool helper functions


if __name__ == '__main__':
    Cape_Town()
