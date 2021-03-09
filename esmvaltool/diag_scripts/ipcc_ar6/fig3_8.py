"""Plots timeseries of masked and blended GMST from CMIP6 models, for comparison with obs."""
import logging
import os
from pprint import pformat

import iris

import numpy
import pandas
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ncblendmask_esmval as ncbm
import esmvaltool.diag_scripts.shared.plot as eplot

from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            select_metadata, sorted_metadata)
from esmvaltool.diag_scripts.shared._base import (
    ProvenanceLogger, get_diagnostic_filename, get_plot_filename)
from esmvaltool.diag_scripts.shared.plot import quickplot

logger = logging.getLogger(os.path.basename(__file__))


def get_provenance_record(attributes, ancestor_files):
    """Create a provenance record describing the diagnostic data and plot."""
    caption = ("Average {long_name} between {start_year} and {end_year} "
               "according to {dataset}.".format(**attributes))

    record = {
        'caption': caption,
        'statistics': ['mean'],
        'domains': ['global'],
        'plot_type': 'zonal',
        'authors': [
            'ande_bo',
            'righ_ma',
        ],
        'references': [
            'acknow_project',
        ],
        'ancestors': ancestor_files,
    }
    return record


def main(cfg):
    st_file = eplot.get_path_to_mpl_style(cfg.get('mpl_style'))
    plt.style.use(st_file)
    matplotlib.use('Agg')
    plt.ioff() #Turn off interactive plotting.
    """Compute the time average for each input dataset."""
    # Get a description of the preprocessed data that we will use as input.
    input_data = cfg['input_data'].values()
#    had4_file='/mnt/lustre02/work/bd0854/DATA/ESMValTool2/RAWOBS/Tier2/HadCRUT4/HadCRUT.4.6.0.0.median.nc'
#    had4_file='/home/rng/data/esmvaltool/HadCRUT.4.6.0.0.median.nc'
    had4_file='/home/rng/data/esmvaltool/HadCRUT.4.6.0.0.median.nc'
    sftlf_file='/home/rng/data/esmvaltool/CNRM-CM6-1-5x5-sftlf.nc'
#sftlf_file='/pf/b/b380746/CNRM-CM6-1-5x5-sftlf.nc' #Hard-coded path to sftlf file for CNRM-CM6 on a 5x5 grid. 
    
    grouped_input_data = group_metadata(
        input_data, 'dataset', sort='ensemble')
    logger.info(
        "Group input data by model and sort by ensemble:"
        "\n%s", pformat(grouped_input_data))
    type (grouped_input_data)
    print (len(grouped_input_data))
    nmodel=len(grouped_input_data)
#    experiments=['historical-ssp245','hist-GHG','hist-aer','hist-nat','hist-volc','hist-sol','hist-stratO3','hist-CO2']
    experiments=['historical-ssp245','hist-GHG','hist-aer','hist-nat','hist-CO2','hist-stratO3','hist-volc','hist-sol','hist-GHG-ssp245-GHG','hist-aer-ssp245-aer','hist-nat-ssp245-nat']
    long_labels=['Human and natural forcings','Greenhouse gases','Other human forcings','Natural forcings']
#    cols=['red','blue','green','orange','lime','peru','cyan','paleturquoise']
    cols=numpy.array([[0,0,0],[196,121,0],[178,178,178],[0,52,102],[0,79,0],[112,160,205],[200,0,0],[0,200,0],[0,0,200],[112,160,205],])/256.
    shade_cols=numpy.array([[128,128,128,128],[204,174,113,128],[191,191,191,128],[67,147,195,128],[223,237,195,128],[91,174,178,128],[255,150,150,128],[150,255,150,128],[150,150,255,128],[91,174,178,128]])/256.
    nexp=len(experiments)-3 #-3 to account for repetition of hist-nat, hist-nat-ssp245-nat etc.
    print ('Number of experiments', nexp)
    # Loop over variables/datasets in alphabetical order
#   Define variables for D&A analysis
    diag_name='gmst01' #Annual mean GMST.
    ldiag=170 #length of diagnostic,hard-coded for the moment.
    years=list(range(1850,2020,1)) #Used for plotting.
    anom_max=500 #arbitrary max size for number of anomalies.
    mean_diag=numpy.zeros((ldiag,nexp,nmodel))
    mean_dec_warming=numpy.zeros((nexp,nmodel))
    mean_ann_warming=numpy.zeros((ldiag,nexp,nmodel))
    mm_ann_warming=numpy.zeros((ldiag,nexp))
    range_ann_warming=numpy.zeros((ldiag,nexp,2)) # 5-95% range.
    nensmax=50
    msval=1e20
    all_ann_warming=numpy.full((ldiag,nexp,nmodel,nensmax),msval)
    ens_sizes=numpy.zeros((nexp,nmodel))

    

    nbar=6
    labels=['Observed warming','Net human influence','Greenhouse gases','Other human forcings','Natural forcings','Internal variability']
    min_assess=[0.88,0.8,1.0,-0.8,-0.1,-0.2]
    max_assess=[1.21,1.3,2.0, 0.0, 0.1, 0.2]
    min_ribes=[-99,0.89,1.12,-0.69,0.04,-0.18]
    max_ribes=[-99,1.17,1.76,-0.12,0.08,0.14]
    mean_ribes=[-99,1.03,1.44,-0.40,0.06,-0.02]
    min_haustein=[-99,0.941,-99,-99,0.001,-99]
    max_haustein=[-99,1.222,-99,-99,0.069,-99]
    mean_haustein=[-99,1.064,1.259,-0.195,0.026,-99]
    min_gillett=[-99,0.92,1.06,-0.71,-0.02,-99]
    max_gillett=[-99,1.30,1.94,-0.03,0.05,-99]
    mean_gillett=[-99,1.11,1.50,-0.37,0.01,-99]
    min_ch7=[-99,0.823021383,0.993864648,-0.628487091,0.04283156,-99]
    mean_ch7=[-99,1.066304612,1.341781251,-0.269287921,0.073580353,-99]
    max_ch7=[-99,1.353390492,1.836139027,-0.026618862,0.119195551,-99]

    xpos=range(nbar)
    lbl=['Ribes','Haustein','Gillett','Chapter 7','CMIP6']

# Get obs GSAT timeseries from Chapter 2.
    obs_gsat=pandas.read_csv('/home/rng/data/esmvaltool/AR6_GSAT.csv',header=None)
    print ('obs_gsat',obs_gsat)

    
    model_names=[]

    for mm, dataset in enumerate(grouped_input_data):
        model_names.append(dataset)
        logger.info("*************** Processing model %s", dataset)
        lbl=dataset
        grouped_model_input_data = group_metadata(
            grouped_input_data[dataset], 'exp', sort='ensemble')
        for exp in grouped_model_input_data:
            logger.info("***** Processing experiment %s", exp)
            exp_string = [experiments.index(i) for i in experiments if exp == i]
            experiment = exp_string[0]
            #Label hist-nat-ssp245-nat as hist-nat etc (some models' hist-nat ends in 2014 so is merged with ssp245-nat).
            if experiment > 7: experiment=experiment-7

            print ('*** Experiment',exp,'Index:',experiment)
            grouped_exp_input_data = group_metadata(
              grouped_model_input_data[exp], 'ensemble', sort='variable_group')
            nens=len(grouped_exp_input_data)
            ens_sizes[experiment,mm]=nens
            exp_diags=numpy.zeros((ldiag,nens))
            exp_ann_warming=numpy.zeros((ldiag,nens))
            exp_dec_warming=numpy.zeros(nens)
        
            for ee, ensemble in enumerate(grouped_exp_input_data):
                logger.info("** Processing ensemble %s", ensemble)
                files=[]
                for attributes in grouped_exp_input_data[ensemble]:
                    logger.info("Processing variable %s", attributes['variable_group'])
                    files.append(attributes['filename'])
                logger.info("*************** Files for blend and mask %s", files)
                dec_warming=[]
                had4_dec_warming=[]
                ann_warming=[]
                (exp_diags[:,ee],had4_diag)=ncbm.ncblendmask_esmval('max', files[0],files[1],files[2],sftlf_file,had4_file,dec_warming,had4_dec_warming,ann_warming,0,diag_name)
                exp_diags[:,ee]=exp_diags[:,ee]-numpy.mean(exp_diags[0:(1901-1850),ee]) #Take anomalies relative to 1850-1900.
                had4_diag=had4_diag-numpy.mean(had4_diag[0:(1901-1850)])
                exp_dec_warming[ee]=dec_warming[0]
                exp_ann_warming[:,ee]=ann_warming[0]
                lbl=""
            mean_diag[:,experiment,mm]=numpy.mean(exp_diags,axis=1)
            mean_dec_warming[experiment,mm]=numpy.mean(exp_dec_warming)
            mean_ann_warming[:,experiment,mm]=numpy.mean(exp_ann_warming,axis=1)
            all_ann_warming[:,experiment,mm,0:nens]=exp_ann_warming

#Overwrite aerosol response with other anthropogenic - historical-hist-GHG-hist-nat.
    all_ann_warming[:,2,:,:]= all_ann_warming[:,0,:,:]- all_ann_warming[:,1,:,:]-all_ann_warming[:,3,:,:]
    for mm in range(nmodel):
#Use minimum ensemble size of historical-SSP245,hist-nat,hist-GHG,hist-aer.
        ens_sizes[2,mm]=numpy.amin(ens_sizes[0:4,mm])
    for experiment in range(nexp):
        wts=numpy.zeros((nmodel,nensmax))
        for mm in range(nmodel):
            wts[mm,0:int(ens_sizes[experiment,mm])]=1./ens_sizes[experiment,mm]
        wts=numpy.reshape(wts,nmodel*nensmax)/numpy.sum(wts)
        print ('wts',wts)
        for yy in range(ldiag):
                year_warming=numpy.reshape(all_ann_warming[yy,experiment,:,:],nmodel*nensmax)
                sort_warming=numpy.sort(year_warming)
                sort_index=numpy.argsort(year_warming)
                cdf=numpy.cumsum(wts[sort_index])
                range_ann_warming[yy,experiment,:]=[sort_warming[cdf>=0.05][0],sort_warming[cdf>=0.95][0]]
                mm_ann_warming[yy,experiment]=numpy.sum(year_warming*wts)


                

    with open('/home/rng/plots/esmvaltool/faq3-3.csv', mode='w') as file:
        data_writer=csv.writer(file,delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for experiment in range(0,4):
            data_writer.writerow([long_labels[experiment]])
            data_writer.writerow(['Year, Mean, Min, Max'])
            for yy in range(ldiag):
                data_writer.writerow([years[yy],mm_ann_warming[yy,experiment],range_ann_warming[yy,experiment,0],range_ann_warming[yy,experiment,1]])
        data_writer.writerow(['Observations'])
        for yy in range(ldiag):
            data_writer.writerow([years[yy],obs_gsat[1][yy]])

    font = {'size'   : 9}
    mm_conv=0.0394
    matplotlib.rc('font', **font)
    plt.figure(0,figsize=[180*mm_conv,120*mm_conv])
                                 

    zzs=[3,1,0,2]
    for experiment in range(0,4):
        plt.fill_between(years,range_ann_warming[:,experiment,0],range_ann_warming[:,experiment,1],color=shade_cols[experiment+1,:],zorder=zzs[experiment])
        plt.plot([1850,2025],[0,0],color='black',linewidth=1)
        plt.plot(years,mm_ann_warming[:,experiment],color=cols[experiment+1,:],linewidth=1,zorder=zzs[experiment]+4,label=long_labels[experiment])

#    plt.plot(years,had4_diag*1.1719758280521986,color='black',linewidth=1,label='Observations',zorder=8)
    plt.plot(obs_gsat[0],obs_gsat[1],color='black',linewidth=1,label='Observations',zorder=8)
    plt.axis([1850,2020,-1.5,2.5])
    plt.xlabel('Year')
    plt.ylabel('Global temperature change ($^\circ$C)')
    plt.legend(loc="upper left")
    plt.savefig('/home/rng/plots/esmvaltool/faq3_3.pdf')

    plt.close()
   
    out_dec_warming=numpy.zeros((5,nmodel))
    out_dec_warming[1,:]=mean_dec_warming[0,:]-mean_dec_warming[3,:]
    out_dec_warming[2,:]=mean_dec_warming[1,:]
    out_dec_warming[3,:]=mean_dec_warming[0,:]-mean_dec_warming[1,:]-mean_dec_warming[3,:]
    out_dec_warming[4,:]=mean_dec_warming[3,:]
    out_dec_warming[0,:]=numpy.zeros(nmodel)-99. #Dummy data
 

#Make chapter figure (3.8)
                                 
    plt.figure(2,figsize=[180*mm_conv,120*mm_conv])
                                 
    lbl=['Ribes','Haustein','Gillett','Chapter 7','CMIP6']

    for bar in range(nbar):        
        plt.fill_between([xpos[bar]-0.45,xpos[bar]+0.45],[min_assess[bar],min_assess[bar]],[max_assess[bar],max_assess[bar]],color=shade_cols[bar,:])
        plt.plot([xpos[bar]-0.34,xpos[bar]-0.34],[min_ribes[bar],max_ribes[bar]],color=cols[bar,:],label=lbl[0],linewidth=4,solid_capstyle='butt')
        plt.plot([xpos[bar]-0.34],[mean_ribes[bar]],color=cols[bar,:],marker='+',linewidth=2)
        plt.plot([xpos[bar]-0.17,xpos[bar]-0.17],[min_haustein[bar],max_haustein[bar]],color=cols[bar,:],linewidth=3,label=lbl[1],solid_capstyle='butt')
        plt.plot([xpos[bar]-0.17],[mean_haustein[bar]],color=cols[bar,:],marker='+',linewidth=2)
        plt.plot([xpos[bar],xpos[bar]],[min_gillett[bar],max_gillett[bar]],color=cols[bar,:],linewidth=2,label=lbl[2],solid_capstyle='butt')
        plt.plot([xpos[bar]],[mean_gillett[bar]],color=cols[bar,:],marker='+',linewidth=2)
#        plt.plot([xpos[bar]+0.075,xpos[bar]+0.075],[min_jenkins[bar],max_jenkins[bar]],color=cols[bar,:],linewidth=2)
#        plt.plot([xpos[bar]+0.075],[mean_jenkins[bar]],color=cols[bar,:],marker='+',linewidth=2)
        plt.plot([xpos[bar]+0.17,xpos[bar]+0.17],[min_ch7[bar],max_ch7[bar]],color=cols[bar,:],label=lbl[3],linewidth=1,solid_capstyle='butt')
        plt.plot([xpos[bar]+0.17],[mean_ch7[bar]],color=cols[bar,:],marker='+',linewidth=1)
        lbl=['','','','','','']

    plt.plot([-10,10],[0,0],color='gray',linewidth=0.5,zorder=-1)
    plt.axis([-1,nbar,-1.5,2.5])
    plt.minorticks_off()

    plt.xticks(list(range(0,nbar)),labels, size='small',rotation=30.,ha="right")
    plt.ylabel('Attributable change in GSAT, 2010-2019 vs 1850-1900 ($^\circ$C)')
#    plt.xlabel('Forcing')
    plt.title('Drivers of observed warming')

    markers=['.','o','v','^','<','>','1','2','P','*','x','X','D']
    for bar in range(0,nbar-1):
        for mm in range (nmodel):
            plt.plot([xpos[bar]+0.34],out_dec_warming[bar,mm],color=cols[bar,:],marker=markers[mm],linewidth=0,label=model_names[mm])
        model_names=['','','','','','','','','','','','','','','','','','']
    plt.legend(loc="upper right",prop={'size':9},bbox_to_anchor=(1.45, 1),frameon=False)

# Add notation to show that OANT and GHG add to ANT.
    plt.arrow(1,-1.2,0,0.2,color="gray",head_width=0.06)
    plt.plot([1,2.5,2.5],[-1.2,-1.2,-1.1],color="gray")
    plt.plot([1.55,1.55,3.45,3.45],[-1,-1.1,-1.1,-1],color="gray")
    
    plt.tight_layout()


    plt.savefig('/home/rng/plots/esmvaltool/fig3_8.png')

    plt.close()






if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
