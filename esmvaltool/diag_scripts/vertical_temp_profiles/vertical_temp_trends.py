from __future__ import division
import logging
import os
import numpy as np
import numpy.ma as ma
import iris
import iris.coord_categorisation as icc
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from esmvaltool.diag_scripts.shared import run_diagnostic, select_metadata, group_metadata


''' Atmospheric temperature trends at pressure levels '''
''' from CMIP6 coupled and atmos-only models, and radiosonde and reanalysis data '''
''' Eunice Lo '''


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


def plot_trends(cfg, obs_trends_all, hist_trends_all, amip_trends_all, obs_trends_preoz, hist_trends_preoz, amip_trends_preoz, obs_trends_postoz, hist_trends_postoz, amip_trends_postoz, plevs, syr, myr, eyr):

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
    ax1.fill_betweenx(y=plevs[:-1], x1=obs_trends_all["rens"][0], x2=obs_trends_all["rens"][1], color='lightgrey')
    ax1.plot(obs_trends_all["rich"], plevs, color="black", linestyle="--", linewidth="4")
    ax1.plot(obs_trends_all["raob"], plevs, color="black", linestyle=":", linewidth="4")
    ax1.plot(obs_trends_all["era5"], plevs, color="black", linestyle="-", linewidth="4")
    # coupled
    ha_means = np.nanmean(hist_trends_all, axis=0)
    ha_perts = np.nanpercentile(hist_trends_all, [5, 95], axis=0)
    ax1.errorbar(ha_means, plevs, xerr=[ha_means-ha_perts[0], ha_perts[1]-ha_means], marker="s", \
                 markersize=5, color=colours["coupled"], elinewidth=3, linestyle="none")
    # atmos-only
    aa_means = np.nanmean(amip_trends_all, axis=0)
    aa_perts = np.nanpercentile(amip_trends_all, [5, 95], axis=0)
    ax1.errorbar(aa_means, plevs+(plevs*0.08), xerr=[aa_means-aa_perts[0], aa_perts[1]-aa_means], marker="s", \
                 markersize=5, color=colours["atmos"], elinewidth=3, linestyle="none")
    # overall
    ax1.set_title("a) "+str(syr)+"-"+str(eyr))
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.tick_params("both", length=10, width=1, which="major")
    ax1.tick_params("both", length=0, width=0, which="minor")
    ax1.set_ylim((1000, 9))
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
    ax2.fill_betweenx(y=plevs[:-1], x1=obs_trends_preoz["rens"][0], x2=obs_trends_preoz["rens"][1], color='lightgrey', label="RICH-obs v1.5 ensemble")
    ax2.plot(obs_trends_preoz["rich"], plevs, color="black", linestyle="--", linewidth="4", label="RICH-obs v1.5 mean")
    ax2.plot(obs_trends_preoz["raob"], plevs, color="black", linestyle=":", linewidth="4", label="RAOBCORE v1.5")
    ax2.plot(obs_trends_preoz["era5"], plevs, color="black", linestyle="-", linewidth="4", label="ERA5/5.1")
    # coupled
    hpr_means = np.nanmean(hist_trends_preoz, axis=0)
    hpr_perts = np.nanpercentile(hist_trends_preoz, [5, 95], axis=0)
    ax2.errorbar(hpr_means, plevs, xerr=[hpr_means-hpr_perts[0], hpr_perts[1]-hpr_means], marker="s", \
                 markersize=5, color=colours["coupled"], elinewidth=3, linestyle="none")
    # atmos-only
    apre_means = np.nanmean(amip_trends_preoz, axis=0)
    apre_perts = np.nanpercentile(amip_trends_preoz, [5, 95], axis=0)
    ax2.errorbar(apre_means, plevs+(plevs*0.08), xerr=[apre_means-apre_perts[0], apre_perts[1]-apre_means], marker="s", \
                 markersize=5, color=colours["atmos"], elinewidth=3, linestyle="none")
    # overall
    ax2.set_title("b) "+str(syr)+"-"+str(myr))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params("both", length=10, width=1, which="major")
    ax2.tick_params("both", length=0, width=0, which="minor")
    ax2.set_ylim((1000, 9))
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.yaxis.set_major_locator(ticker.FixedLocator(plevs))
    ax2.set_xlim((-1.5, 0.9))
    ax2.xaxis.set_major_locator(ticker.FixedLocator([-0.9, -0.6, -0.3, 0., 0.3, 0.6]))
    ax2.set_xlabel("Trend ($^{\circ}$C/decade)")
    # legend to be added
    ax2.legend(loc="lower left", fontsize=14, frameon=False)

    # post-ozone
    ax3.axvline(x=0, color='gray', alpha=0.5, linestyle='--')
    ax3.fill_betweenx(y=plevs[:-1], x1=obs_trends_postoz["rens"][0], x2=obs_trends_postoz["rens"][1], color='lightgrey')
    ax3.plot(obs_trends_postoz["rich"], plevs, color="black", linestyle="--", linewidth="4")
    ax3.plot(obs_trends_postoz["raob"], plevs, color="black", linestyle=":", linewidth="4")
    ax3.plot(obs_trends_postoz["era5"], plevs, color="black", linestyle="-", linewidth="4")
    # coupled
    hpo_means = np.nanmean(hist_trends_postoz, axis=0)
    hpo_perts = np.nanpercentile(hist_trends_postoz, [5, 95], axis=0)
    ax3.errorbar(hpo_means, plevs, xerr=[hpo_means-hpo_perts[0], hpo_perts[1]-hpo_means], marker="s", \
                 markersize=5, color=colours["coupled"], elinewidth=3, linestyle="none") 
    # atmos-only
    apo_means = np.nanmean(amip_trends_postoz, axis=0)
    apo_perts = np.nanpercentile(amip_trends_postoz, [5, 95], axis=0)
    ax3.errorbar(apo_means, plevs+(plevs*0.08), xerr=[apo_means-apo_perts[0], apo_perts[1]-apo_means], marker="s", \
                 markersize=5, color=colours["atmos"], elinewidth=3, linestyle="none") 
    # overall
    ax3.set_title("c) "+str(myr+1)+"-"+str(eyr))
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.tick_params("both", length=10, width=1, which="major")
    ax3.tick_params("both", length=0, width=0, which="minor")
    ax3.set_ylim((1000, 9))
    ax3.set_yscale("log")
    ax3.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3.yaxis.set_major_locator(ticker.FixedLocator(plevs))
    ax3.set_xlim((-1.2, 1.2))
    ax3.xaxis.set_major_locator(ticker.FixedLocator([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]))
    ax3.set_xlabel("Trend ($^{\circ}$C/decade)")

    # save
    plt.tight_layout()
    png_name = "vertical_temp_profiles_20S-20N_"+str(syr)+"_"+str(eyr)+"_rio_range_all_5-95.png"
    plt.savefig(os.path.join(local_path, png_name))
    plt.close()

    return "Saved plot!" 


def main(cfg):
       
    ''' run the diagnostic '''
    
    # extract the Tropics (-20 to 20N)
    # start_latitude and start_longitude defined in recipe    
 
    # load regridded data
    # start_year and end_year of data defined in recipe

    # RAOBCORE 1.5
    input_raob_data = select_metadata(cfg['input_data'].values(), dataset="raobcore17")
    # cubelist of 1
    raob_cube = iris.load(input_raob_data[0]['filename'])
   
    # RICH1.5-obs mean, with 10 hPa
    input_rich_data = select_metadata(cfg['input_data'].values(), dataset="rich17obs")
    # cubelist of 1
    rich_cube = iris.load(input_rich_data[0]['filename'])

    # RICH1.5-obs ensemble
    input_rens_data = select_metadata(cfg['input_data'].values(), dataset="rich")
    rens_cubes = iris.cube.CubeList()
    for (version, [data]) in group_metadata(input_rens_data, 'version').items():
        # load in deg Celsius
        cube = iris.load_cube(data['filename'])
        cube.data -= 273.15
        cube.units = 'celsius'  
        rens_cubes.append(cube)

    # ERA5/5.1
    input_era5_data = select_metadata(cfg['input_data'].values(), dataset="era5.1")
    # cubelist of 1
    era5_cube = iris.load(input_era5_data[0]['filename'])
    
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
        cube.data = ma.array(cube.data, mask=rich_cube[0].data.mask)
        # mask unrealistic Ts that are actually fill values
        cube.data.mask[np.abs(cube.data) >= 100.] = True
        hist_cubes.append(cube)
    
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
        cube.data = ma.array(cube.data, mask=rich_cube[0].data.mask)
        # mask unrealistic Ts that are actually fill values
        cube.data.mask[np.abs(cube.data) >= 100.] = True
        amip_cubes.append(cube)
    
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
    rens_trends_allyrs = ta_trends(rens_tropmeans, nlevs-1, yrx=yrs_all)
    era5_trends_allyrs = ta_trends(era5_tropmeans, nlevs, yrx=yrs_all)
    # gather obs/reanalysis in a dict
    rens_trends_allyrs_l = np.min(rens_trends_allyrs, axis=0)
    rens_trends_allyrs_h = np.max(rens_trends_allyrs, axis=0)
    obs_trends_allyrs = {"raob":raob_trends_allyrs[0], "rich":rich_trends_allyrs[0], "rens":[rens_trends_allyrs_l, rens_trends_allyrs_h], "era5":era5_trends_allyrs[0]}
    # models
    hist_trends_allyrs = ta_trends(hist_tropmeans, nlevs, yrx=yrs_all) 
    amip_trends_allyrs = ta_trends(amip_tropmeans, nlevs, yrx=yrs_all)
        
    # pre-ozone
    yrs_preoz = np.arange(1979, 1997+1)
    preoz_constraint = iris.Constraint(time=lambda cell: 1979 <= cell.point.year <= 1997)
    raob_trends_preoz = ta_trends(raob_tropmeans.extract(preoz_constraint), nlevs, yrx=yrs_preoz)
    rich_trends_preoz = ta_trends(rich_tropmeans.extract(preoz_constraint), nlevs, yrx=yrs_preoz)
    rens_trends_preoz = ta_trends(rens_tropmeans.extract(preoz_constraint), nlevs-1, yrx=yrs_preoz)
    era5_trends_preoz = ta_trends(era5_tropmeans.extract(preoz_constraint), nlevs, yrx=yrs_preoz)
    # gather obs/reanalysis in a dict
    rens_trends_preoz_l = np.min(rens_trends_preoz, axis=0)
    rens_trends_preoz_h = np.max(rens_trends_preoz, axis=0)
    obs_trends_preoz = {"raob":raob_trends_preoz[0], "rich":rich_trends_preoz[0], "rens":[rens_trends_preoz_l, rens_trends_preoz_h], "era5":era5_trends_preoz[0]}
    # models
    hist_trends_preoz = ta_trends(hist_tropmeans.extract(preoz_constraint), nlevs, yrx=yrs_preoz)
    amip_trends_preoz = ta_trends(amip_tropmeans.extract(preoz_constraint), nlevs, yrx=yrs_preoz)
    
    # post-ozone   
    yrs_postoz = np.arange(1998, 2014+1) 
    postoz_constraint = iris.Constraint(time=lambda cell: 1998 <= cell.point.year <= 2014)
    raob_trends_postoz = ta_trends(raob_tropmeans.extract(postoz_constraint), nlevs, yrx=yrs_postoz)
    rich_trends_postoz = ta_trends(rich_tropmeans.extract(postoz_constraint), nlevs, yrx=yrs_postoz)
    rens_trends_postoz = ta_trends(rens_tropmeans.extract(postoz_constraint), nlevs-1, yrx=yrs_postoz)
    era5_trends_postoz = ta_trends(era5_tropmeans.extract(postoz_constraint), nlevs, yrx=yrs_postoz)
    # gather obs/reanalysis in a dict
    rens_trends_postoz_l = np.min(rens_trends_postoz, axis=0)
    rens_trends_postoz_h = np.max(rens_trends_postoz, axis=0)
    obs_trends_postoz = {"raob":raob_trends_postoz[0], "rich":rich_trends_postoz[0], "rens":[rens_trends_postoz_l, rens_trends_postoz_h], "era5":era5_trends_postoz[0]}    
    # models
    hist_trends_postoz = ta_trends(hist_tropmeans.extract(postoz_constraint), nlevs, yrx=yrs_postoz)
    amip_trends_postoz = ta_trends(amip_tropmeans.extract(postoz_constraint), nlevs, yrx=yrs_postoz)
        
    # do the plotting
    # in hPa
    plevs = hist_tropmeans[0].coord("air_pressure").points/100.    
    plot_trends(cfg, obs_trends_allyrs, hist_trends_allyrs, amip_trends_allyrs, obs_trends_preoz, hist_trends_preoz, amip_trends_preoz, obs_trends_postoz, hist_trends_postoz, amip_trends_postoz, plevs=plevs, syr=1979, myr=1997, eyr=2014)

    # save models and their trend values
    local_path = cfg['work_dir']
    with open(os.path.join(local_path,"hist_models.txt"), "w") as f:
        for item in hist_models:
            f.write("%s\n" % item)
    with open(os.path.join(local_path,"amip_models.txt"), "w") as f:
        for item in amip_models:
            f.write("%s\n" % item)
    np.save(os.path.join(local_path,"rio_trends_allyrs.npy"), rens_trends_allyrs)
    np.save(os.path.join(local_path,"hist_trends_allyrs.npy"), hist_trends_allyrs)
    np.save(os.path.join(local_path,"amip_trends_allyrs.npy"), amip_trends_allyrs)
    np.save(os.path.join(local_path,"rio_trends_preoz.npy"), rens_trends_preoz)
    np.save(os.path.join(local_path,"hist_trends_preoz.npy"), hist_trends_preoz)
    np.save(os.path.join(local_path,"amip_trends_preoz.npy"), amip_trends_preoz)
    np.save(os.path.join(local_path,"rio_trends_postoz.npy"), rens_trends_postoz)
    np.save(os.path.join(local_path,"hist_trends_postoz.npy"), hist_trends_postoz)
    np.save(os.path.join(local_path,"amip_trends_postoz.npy"), amip_trends_postoz)

    # save other stuff for checking
    ens_no = np.arange(32)
    for nens, rc, ra, rt in zip(ens_no, rens_cubes, rens_annuals, rens_tropmeans):
        iris.save(rc, os.path.join(local_path,"rio"+"{:02d}".format(nens)+"_cube.nc"))
        iris.save(ra, os.path.join(local_path,"rio"+"{:02d}".format(nens)+"_annual.nc"))
        iris.save(rt, os.path.join(local_path,"rio"+"{:02d}".format(nens)+"_tropmean.nc"))
 
if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
