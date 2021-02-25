from __future__ import division
import logging
import os
import numpy as np
import numpy.ma as ma
import iris
import iris.coord_categorisation as icc
import statsmodels.api as sm
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from esmvaltool.diag_scripts.shared import (
    run_diagnostic, 
    select_metadata, 
    group_metadata,
    save_data,
    save_figure,
)


''' Atmospheric temperature trends at pressure levels '''
''' from CMIP6 coupled and atmos-only models, and radiosonde and reanalysis data '''
''' Eunice Lo '''


def get_provenance_record(caption, ancestor_files):

    ''' create a provenance record describing the diagnostic data and plot '''

    record = {
        'caption': caption,
        'statistics': ['trend'],
        'domains': ['trop'],
        'plot_types': ['vert'],
        'authors': [
            'lo_eunice',
        ],
        'references': [
            'mitchell20erl',
        ],
        'projects': ['ipcc_ar6'],
        'ancestors': ancestor_files,
    }
    return record


def annual_mean(mthly_cubelist):

    ''' turn monthly data into annual averages '''
    ''' discarding years with more than 3 months missing '''

    annual_cubelist = iris.cube.CubeList()

    for c in mthly_cubelist:
        icc.add_year(c, "time", name="year")
        annual_cubelist.append(c.aggregated_by("year", iris.analysis.MEAN, mdtol=0.25))
 
    return annual_cubelist


def area_mean(in_cubelist):

    ''' area-weighted average of input data '''

    out_cubelist = iris.cube.CubeList()

    for c in in_cubelist:
        # area weights
        if not c.coord("latitude").has_bounds():
            c.coord("latitude").guess_bounds()
        if not c.coord("longitude").has_bounds():
            c.coord("longitude").guess_bounds()
        #awghs = iris.analysis.cartography.area_weights(c)
        awghs = iris.analysis.cartography.cosine_latitude_weights(c) 
        out_cubelist.append(c.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=awghs))

    return out_cubelist


def ta_trends(in_cubelist, nlevs, yrx):

    ''' find trends in vertical temperatures '''
    ''' with first order polynomial fit '''
    ''' in deg C/decade '''

    nmodels = len(in_cubelist)
    trends = np.zeros((nmodels, nlevs))

    for m in range(nmodels):
        for p in range(nlevs):
            tay = in_cubelist[m][:,p].data
            # nan as trend if all values are masked
            if tay.mask.all():
                trends[m,p] = np.nan
            else: 
                trends[m,p] = np.polyfit(yrx, tay, 1)[0]

    # units: deg C per decade
    return trends*10.


def remove_nans(in_2darray):

    ''' remove nans from 2D array '''
    ''' and return a list of filtered arrays '''
    ''' for boxplots to show '''

    mask = ~np.isnan(in_2darray)
    out_list = [d[m] for d, m in zip(in_2darray.T, mask.T)]    

    return out_list


def plot_trends(cfg, provenance_record, \
                obs_trends_all, hist_trends_all, amip_trends_all, \
                obs_trends_preoz, hist_trends_preoz, amip_trends_preoz, \
                obs_trends_postoz, hist_trends_postoz, amip_trends_postoz, \
                plevs, syr, myr, eyr):

    ''' plot vertical temperature trends '''

    if not cfg['write_plots']:
        return None

    # already defined esmvaltool output path
    local_path = cfg['plot_dir']

    # plot settings
    font = {'size'   : 16}
    plt.rc('font', **font)
    plt.rcParams["font.family"] = "Arial"

    # colours, dark categorical palette in guidelines_SOD_Figures-3.pdf
    # #dd542e = RGB 221 84 46; #2134db = RGB 33 52 219 
    colours = {"coupled":"#dd542e", "atmos":"#2134db"}

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(18,10))
    
    # whole period
    ax1.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
    # rich-obs v1.5.1 ensemble range centred at rich-mean v1.7
    xlerr1 = obs_trends_all["rens"][1] - obs_trends_all["rens"][0]
    xherr1 = obs_trends_all["rens"][2] - obs_trends_all["rens"][1]
    ax1.fill_betweenx(y=plevs, x1=obs_trends_all["rich"]-xlerr1, x2=obs_trends_all["rich"]+xherr1, color='lightgrey')
    # rich-mean & raobcore v1.7
    ax1.plot(obs_trends_all["rich"], plevs, color="black", linestyle="--", linewidth="4")  
    ax1.plot(obs_trends_all["raob"], plevs, color="black", linestyle=":", linewidth="4")   
    # era5/5.1
    ax1.plot(obs_trends_all["era5"], plevs, color="black", linestyle="-", linewidth="4")
    # coupled
    ha_means = np.nanmean(hist_trends_all, axis=0)
    ha_perts = np.nanpercentile(hist_trends_all, [5, 95], axis=0)
    ax1.errorbar(ha_means, plevs, xerr=[ha_means-ha_perts[0], ha_perts[1]-ha_means], marker="s", \
                 markersize=10, color=colours["coupled"], elinewidth=3, linestyle="none")
    # atmos-only
    aa_means = np.nanmean(amip_trends_all, axis=0)
    aa_perts = np.nanpercentile(amip_trends_all, [5, 95], axis=0)
    ax1.errorbar(aa_means, plevs+(plevs*0.08), xerr=[aa_means-aa_perts[0], aa_perts[1]-aa_means], marker="s", \
                 markersize=10, color=colours["atmos"], elinewidth=3, linestyle="none")
    # overall
    ax1.set_title("a) "+str(syr)+"-"+str(eyr))
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.tick_params("both", length=10, width=1, which="major")
    ax1.tick_params("both", length=0, width=0, which="minor")
    ax1.set_ylim((1000, 19))
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.yaxis.set_major_locator(ticker.FixedLocator(plevs))
    ax1.set_ylabel("Pressure (hPa)")
    ax1.set_xlim((-0.9, 0.9))
    ax1.xaxis.set_major_locator(ticker.FixedLocator([-0.6, -0.3, 0., 0.3, 0.6]))
    ax1.set_xlabel("Trend ($^{\circ}$C/decade)")
    # legend
    legend_elements = [Patch(facecolor=colours["coupled"], edgecolor=colours["coupled"], label="Coupled Ocean"), \
                       Patch(facecolor=colours["atmos"], edgecolor=colours["atmos"], label="Prescribed SSTs")]
    ax1.legend(handles=legend_elements, loc="lower left", fontsize=14, frameon=False)

    # pre-ozone
    ax2.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
    # rich-obs v1.5.1 ensemble range centred at rich-mean v1.7
    xlerr2 = obs_trends_preoz["rens"][1] - obs_trends_preoz["rens"][0]
    xherr2 = obs_trends_preoz["rens"][2] - obs_trends_preoz["rens"][1]
    ax2.fill_betweenx(y=plevs, x1=obs_trends_preoz["rich"]-xlerr2, x2=obs_trends_preoz["rich"]+xherr2, \
                      color='lightgrey', label="RICH-obs v1.5.1 range")
    # rich-mean & raobcore v1.7
    ax2.plot(obs_trends_preoz["rich"], plevs, color="black", linestyle="--", linewidth="4", label="RICH-obs v1.7 mean")
    ax2.plot(obs_trends_preoz["raob"], plevs, color="black", linestyle=":", linewidth="4", label="RAOBCORE v1.7")
    # era5/5.1
    ax2.plot(obs_trends_preoz["era5"], plevs, color="black", linestyle="-", linewidth="4", label="ERA5/5.1")
    # coupled
    hpr_means = np.nanmean(hist_trends_preoz, axis=0)
    hpr_perts = np.nanpercentile(hist_trends_preoz, [5, 95], axis=0)
    ax2.errorbar(hpr_means, plevs, xerr=[hpr_means-hpr_perts[0], hpr_perts[1]-hpr_means], marker="s", \
                 markersize=10, color=colours["coupled"], elinewidth=3, linestyle="none")
    # atmos-only
    apre_means = np.nanmean(amip_trends_preoz, axis=0)
    apre_perts = np.nanpercentile(amip_trends_preoz, [5, 95], axis=0)
    ax2.errorbar(apre_means, plevs+(plevs*0.08), xerr=[apre_means-apre_perts[0], apre_perts[1]-apre_means], marker="s", \
                 markersize=10, color=colours["atmos"], elinewidth=3, linestyle="none")
    # overall
    ax2.set_title("b) "+str(syr)+"-"+str(myr))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params("both", length=10, width=1, which="major")
    ax2.tick_params("both", length=0, width=0, which="minor")
    ax2.set_ylim((1000, 19))
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.yaxis.set_major_locator(ticker.FixedLocator(plevs))
    ax2.set_xlim((-1.5, 0.9))
    ax2.xaxis.set_major_locator(ticker.FixedLocator([-0.9, -0.6, -0.3, 0., 0.3, 0.6]))
    ax2.set_xlabel("Trend ($^{\circ}$C/decade)")
    # legend to be added
    handles, labels = ax2.get_legend_handles_labels()
    handles = [handles[3], handles[0], handles[1], handles[2]]
    labels = [labels[3], labels[0], labels[1], labels[2]]
    ax2.legend(handles, labels, loc="lower left", fontsize=14, frameon=False)

    # post-ozone
    ax3.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
    # rich-obs v1.5.1 ensemble range centred at rich-mean v1.7
    xlerr3 = obs_trends_postoz["rens"][1] - obs_trends_postoz["rens"][0]
    xherr3 = obs_trends_postoz["rens"][2] - obs_trends_postoz["rens"][1]
    ax3.fill_betweenx(y=plevs, x1=obs_trends_postoz["rich"]-xlerr3, x2=obs_trends_postoz["rich"]+xherr3, color='lightgrey')
    # rich-mean & raobcore v1.7
    ax3.plot(obs_trends_postoz["rich"], plevs, color="black", linestyle="--", linewidth="4")
    ax3.plot(obs_trends_postoz["raob"], plevs, color="black", linestyle=":", linewidth="4")
    # era5/5.1
    ax3.plot(obs_trends_postoz["era5"], plevs, color="black", linestyle="-", linewidth="4")
    # coupled
    hpo_means = np.nanmean(hist_trends_postoz, axis=0)
    hpo_perts = np.nanpercentile(hist_trends_postoz, [5, 95], axis=0)
    ax3.errorbar(hpo_means, plevs, xerr=[hpo_means-hpo_perts[0], hpo_perts[1]-hpo_means], marker="s", \
                 markersize=10, color=colours["coupled"], elinewidth=3, linestyle="none") 
    # atmos-only
    apo_means = np.nanmean(amip_trends_postoz, axis=0)
    apo_perts = np.nanpercentile(amip_trends_postoz, [5, 95], axis=0)
    ax3.errorbar(apo_means, plevs+(plevs*0.08), xerr=[apo_means-apo_perts[0], apo_perts[1]-apo_means], marker="s", \
                 markersize=10, color=colours["atmos"], elinewidth=3, linestyle="none") 
    # overall
    ax3.set_title("c) "+str(myr+1)+"-"+str(eyr))
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.tick_params("both", length=10, width=1, which="major")
    ax3.tick_params("both", length=0, width=0, which="minor")
    ax3.set_ylim((1000, 19))
    ax3.set_yscale("log")
    ax3.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3.yaxis.set_major_locator(ticker.FixedLocator(plevs))
    ax3.set_xlim((-1.2, 1.2))
    ax3.xaxis.set_major_locator(ticker.FixedLocator([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]))
    ax3.set_xlabel("Trend ($^{\circ}$C/decade)")

    # save plot
    plt.tight_layout()
    today = date.today().strftime("%Y%m%d")
    fstem = "vertical_temp_profiles_20S-20N_"+str(syr)+"_"+str(eyr)+"_rich_raobcore_1.7_rio_range_1.5.1_recentred_all_5-95_"+today
    save_figure(fstem, provenance_record, cfg)
    #plt.savefig(os.path.join(local_path, fstem+".svg"), format="svg")
    plt.close()

    # save data
    out_data = np.zeros((3, len(plevs), 12))
    out_data[0,:,0] = obs_trends_all["raob"]
    out_data[0,:,1] = obs_trends_all["rich"]  
    out_data[0,:,2] = obs_trends_all["rens"][0]
    out_data[0,:,3] = obs_trends_all["rens"][1]     # lower limit
    out_data[0,:,4] = obs_trends_all["rens"][2]     # upper limit
    out_data[0,:,5] = obs_trends_all["era5"]
    out_data[0,:,6] = ha_means
    out_data[0,:,7] = ha_perts[0]                   # lower limit
    out_data[0,:,8] = ha_perts[1]                   # upper limit
    out_data[0,:,9] = aa_means
    out_data[0,:,10] = aa_perts[0]                  # lower limit
    out_data[0,:,11] = aa_perts[1]                  # upper limit 
    out_data[1,:,0] = obs_trends_preoz["raob"]
    out_data[1,:,1] = obs_trends_preoz["rich"]
    out_data[1,:,2] = obs_trends_preoz["rens"][0]
    out_data[1,:,3] = obs_trends_preoz["rens"][1]   # lower limit
    out_data[1,:,4] = obs_trends_preoz["rens"][2]   # upper limit
    out_data[1,:,5] = obs_trends_preoz["era5"]
    out_data[1,:,6] = hpr_means
    out_data[1,:,7] = hpr_perts[0]                  # lower limit
    out_data[1,:,8] = hpr_perts[1]                  # upper limit
    out_data[1,:,9] = apre_means
    out_data[1,:,10] = apre_perts[0]                # lower limit
    out_data[1,:,11] = apre_perts[1]                # upper limit 
    out_data[2,:,0] = obs_trends_postoz["raob"]
    out_data[2,:,1] = obs_trends_postoz["rich"]
    out_data[2,:,2] = obs_trends_postoz["rens"][0]
    out_data[2,:,3] = obs_trends_postoz["rens"][1]  # lower limit
    out_data[2,:,4] = obs_trends_postoz["rens"][2]  # upper limit
    out_data[2,:,5] = obs_trends_postoz["era5"]
    out_data[2,:,6] = hpo_means
    out_data[2,:,7] = hpo_perts[0]                  # lower limit
    out_data[2,:,8] = hpo_perts[1]                  # upper limit
    out_data[2,:,9] = apo_means
    out_data[2,:,10] = apo_perts[0]                 # lower limit
    out_data[2,:,11] = apo_perts[1]                 # upper limit 
    time_period_index = iris.coords.DimCoord(np.arange(3), long_name="time_period_index")
    pressure_levels = iris.coords.DimCoord(plevs, long_name="pressure_level", units="hPa")
    data_source_index = iris.coords.DimCoord(np.arange(12), long_name="data_source_index")
    out_cube = iris.cube.Cube(out_data, long_name="tropic_temperature_trend (celsius per decade)", \
                              dim_coords_and_dims=[(time_period_index, 0), (pressure_levels, 1), (data_source_index, 2)], \
                              attributes={"time_period_index 0":"1979-2014", "time_period_index 1":"1979-1997", \
                                          "time_period_index 2":"1998-2014", "data_source_index 0":"RAOBCORE v1.7", \
                                          "data_source_index 1":"RICH-obs v1.7 mean", \
                                          "data_source_index 2":"RICH-obs v1.5.1 mean", \
                                          "data_source_index 3":"RICH-obs v1.5.1 lower limit", \
                                          "data_source_index 4":"RICH-obs v1.5.1 upper limit", \
                                          "data_source_index 5":"ERA5/5.1", \
                                          "data_source_index 6":"Couple ocean models mean", \
                                          "data_source_index 7":"Couple ocean models lower limit", \
                                          "data_source_index 8":"Couple ocean models upper limit", \
                                          "data_source_index 9":"Prescribed SSTs models mean", \
                                          "data_source_index 10":"Prescribed SSTs models lower limit", \
                                          "data_source_index 11":"Prescribed SSTs models upper limit"}) 
    save_data(fstem, provenance_record, cfg, out_cube)
 
    return "Saved plot!" 


def main(cfg):
       
    ''' run the diagnostic '''
    
    # RAOBCORE v1.7
    input_raob_data = select_metadata(cfg['input_data'].values(), dataset="raobcore17")
    # cubelist of 1
    raob_cube = iris.load(input_raob_data[0]['filename'])
    # load in deg Celsius, they are anomales anyway
    raob_cube[0].units = 'celsius'

    # RICH-mean v1.7
    input_rich_data = select_metadata(cfg['input_data'].values(), dataset="rich17obs")
    # cubelist of 1
    rich_cube = iris.load(input_rich_data[0]['filename'])
    # load in deg Celsius, they are anomales anyway
    rich_cube[0].units = 'celsius'

    # record input files
    input_files = [input_raob_data[0]['filename'], input_rich_data[0]['filename']]    

    # RICH-obs v1.5.1 ensemble
    input_rens_data = select_metadata(cfg['input_data'].values(), dataset="rich")
    rens_cubes = iris.cube.CubeList()
    for (version, [data]) in group_metadata(input_rens_data, 'version').items():
        # load in deg Celsius, they are anomales anyway
        cube = iris.load_cube(data['filename'])
        cube.units = 'celsius'  
        rens_cubes.append(cube)
        # record input file
        input_files.append(data['filename'])

    # ERA5/5.1
    input_era5_data = select_metadata(cfg['input_data'].values(), dataset="era5.1")
    # cubelist of 1
    era5_cube = iris.load(input_era5_data[0]['filename'])
    # record input file
    input_files.append(input_era5_data[0]['filename'])   
 
    # CMIP6
    input_hist_data = select_metadata(cfg['input_data'].values(), exp="historical")
    input_amip_data = select_metadata(cfg['input_data'].values(), exp="amip")
    if not input_hist_data:
        raise ValueError("This diagnostic needs historical data")
    if not input_amip_data:
        raise ValueError("This diagnostic needs AMIP data")
    
    # get historical (coupled) data
    hist_models = []
    hist_cubes = iris.cube.CubeList()
    for (dataset, [data]) in group_metadata(input_hist_data, 'dataset').items():
        hist_models.append(dataset)
        # load in deg Celsius
        cube = iris.load_cube(data['filename'])
        cube.data -= 273.15
        cube.units = 'celsius'
        # mask observational missing data
        cube.data = ma.array(cube.data, mask=raob_cube[0].data.mask)
        # mask unrealistic Ts that are actually fill values
        cube.data.mask[np.abs(cube.data) >= 100.] = True
        hist_cubes.append(cube)
        # record input file
        input_files.append(data['filename'])
    
    # get amip (atmos only) data
    amip_models = []
    amip_cubes = iris.cube.CubeList()
    for (dataset, [data]) in group_metadata(input_amip_data, 'dataset').items():
        amip_models.append(dataset)
        # load in deg Celsius
        cube = iris.load_cube(data['filename'])
        cube.data -= 273.15
        cube.units = 'celsius'
        # mask observational missing data
        cube.data = ma.array(cube.data, mask=raob_cube[0].data.mask)
        # mask unrealistic Ts that are actually fill values
        cube.data.mask[np.abs(cube.data) >= 100.] = True
        amip_cubes.append(cube)
        # record input file
        input_files.append(data['filename'])
    
    # turn into annual values, discarding years with more than 3 months of data missing
    raob_annuals = annual_mean(raob_cube)
    rich_annuals = annual_mean(rich_cube)
    rens_annuals = annual_mean(rens_cubes)
    era5_annuals = annual_mean(era5_cube)
    hist_annuals = annual_mean(hist_cubes)     
    amip_annuals = annual_mean(amip_cubes)
    
    # tropical means
    raob_tropmeans = area_mean(raob_annuals)
    rich_tropmeans = area_mean(rich_annuals)
    rens_tropmeans = area_mean(rens_annuals)
    era5_tropmeans = area_mean(era5_annuals)
    hist_tropmeans = area_mean(hist_annuals)
    amip_tropmeans = area_mean(amip_annuals)

    # trends
    nlevs = len(hist_tropmeans[0].coord("air_pressure").points)
    
    # whole period
    yrs_all = np.arange(1979, 2014+1)
    raob_trends_allyrs = ta_trends(raob_tropmeans, nlevs, yrx=yrs_all)
    rich_trends_allyrs = ta_trends(rich_tropmeans, nlevs, yrx=yrs_all)
    rens_trends_allyrs = ta_trends(rens_tropmeans, nlevs, yrx=yrs_all)
    era5_trends_allyrs = ta_trends(era5_tropmeans, nlevs, yrx=yrs_all)
    # gather obs/reanalysis in a dict
    rens_trends_allyrs_l = np.nanmin(rens_trends_allyrs, axis=0)    # min
    rens_trends_allyrs_m = np.nanmean(rens_trends_allyrs, axis=0)   # mean
    rens_trends_allyrs_h = np.nanmax(rens_trends_allyrs, axis=0)    # max
    obs_trends_allyrs = {"raob":raob_trends_allyrs[0], "rich":rich_trends_allyrs[0], \
                         "rens":[rens_trends_allyrs_l, rens_trends_allyrs_m, rens_trends_allyrs_h], \
                         "era5":era5_trends_allyrs[0]}
    # models
    hist_trends_allyrs = ta_trends(hist_tropmeans, nlevs, yrx=yrs_all) 
    amip_trends_allyrs = ta_trends(amip_tropmeans, nlevs, yrx=yrs_all)
        
    # pre-ozone
    yrs_preoz = np.arange(1979, 1997+1)
    preoz_constraint = iris.Constraint(time=lambda cell: 1979 <= cell.point.year <= 1997)
    raob_trends_preoz = ta_trends(raob_tropmeans.extract(preoz_constraint), nlevs, yrx=yrs_preoz)
    rich_trends_preoz = ta_trends(rich_tropmeans.extract(preoz_constraint), nlevs, yrx=yrs_preoz)
    rens_trends_preoz = ta_trends(rens_tropmeans.extract(preoz_constraint), nlevs, yrx=yrs_preoz)
    era5_trends_preoz = ta_trends(era5_tropmeans.extract(preoz_constraint), nlevs, yrx=yrs_preoz)
    # gather obs/reanalysis in a dict
    rens_trends_preoz_l = np.nanmin(rens_trends_preoz, axis=0)
    rens_trends_preoz_m = np.nanmean(rens_trends_preoz, axis=0)
    rens_trends_preoz_h = np.nanmax(rens_trends_preoz, axis=0)
    obs_trends_preoz = {"raob":raob_trends_preoz[0], "rich":rich_trends_preoz[0], \
                        "rens":[rens_trends_preoz_l, rens_trends_preoz_m, rens_trends_preoz_h], \
                        "era5":era5_trends_preoz[0]}
    # models
    hist_trends_preoz = ta_trends(hist_tropmeans.extract(preoz_constraint), nlevs, yrx=yrs_preoz)
    amip_trends_preoz = ta_trends(amip_tropmeans.extract(preoz_constraint), nlevs, yrx=yrs_preoz)
    
    # post-ozone   
    yrs_postoz = np.arange(1998, 2014+1) 
    postoz_constraint = iris.Constraint(time=lambda cell: 1998 <= cell.point.year <= 2014)
    raob_trends_postoz = ta_trends(raob_tropmeans.extract(postoz_constraint), nlevs, yrx=yrs_postoz)
    rich_trends_postoz = ta_trends(rich_tropmeans.extract(postoz_constraint), nlevs, yrx=yrs_postoz)
    rens_trends_postoz = ta_trends(rens_tropmeans.extract(postoz_constraint), nlevs, yrx=yrs_postoz)
    era5_trends_postoz = ta_trends(era5_tropmeans.extract(postoz_constraint), nlevs, yrx=yrs_postoz)
    # gather obs/reanalysis in a dict
    rens_trends_postoz_l = np.nanmin(rens_trends_postoz, axis=0)
    rens_trends_postoz_m = np.nanmean(rens_trends_postoz, axis=0)
    rens_trends_postoz_h = np.nanmax(rens_trends_postoz, axis=0)
    obs_trends_postoz = {"raob":raob_trends_postoz[0], "rich":rich_trends_postoz[0], \
                         "rens":[rens_trends_postoz_l, rens_trends_postoz_m, rens_trends_postoz_h], \
                         "era5":era5_trends_postoz[0]}    
    # models
    hist_trends_postoz = ta_trends(hist_tropmeans.extract(postoz_constraint), nlevs, yrx=yrs_postoz)
    amip_trends_postoz = ta_trends(amip_tropmeans.extract(postoz_constraint), nlevs, yrx=yrs_postoz)
        
    # do the plotting
    # in hPa
    plevs = hist_tropmeans[0].coord("air_pressure").points/100.   
    caption = "Vertical profiles of air temperature trends between 1979 and 2014."
    provenance_record = get_provenance_record(caption, input_files) 
    plot_trends(cfg, provenance_record, \
                obs_trends_allyrs, hist_trends_allyrs, amip_trends_allyrs, \
                obs_trends_preoz, hist_trends_preoz, amip_trends_preoz, \
                obs_trends_postoz, hist_trends_postoz, amip_trends_postoz, \
                plevs=plevs, syr=1979, myr=1997, eyr=2014)

 
if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
