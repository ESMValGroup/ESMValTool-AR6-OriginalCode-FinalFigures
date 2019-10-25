"""Perform attribution using blended and masked temperatures."""
import logging
import os
from pprint import pformat

import iris

import numpy
import detatt_mk as da
import matplotlib
matplotlib.use('Agg') #Turn off interactive plots.
import matplotlib.pyplot as plt
import ncblendmask_esmval as ncbm

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
    plt.ioff() #Turn off interactive plotting.
    """Compute the time average for each input dataset."""
    # Get a description of the preprocessed data that we will use as input.
    input_data = cfg['input_data'].values()
#    had4_file='/mnt/lustre02/work/bd0854/DATA/ESMValTool2/RAWOBS/Tier2/HadCRUT4/HadCRUT.4.6.0.0.median.nc'
    had4_file='/home/rng/data/esmvaltool/HadCRUT.4.6.0.0.median.nc'
    sftlf_file='/home/rng/data/esmvaltool/CNRM-CM6-1-5x5-sftlf.nc'
#sftlf_file='/pf/b/b380746/CNRM-CM6-1-5x5-sftlf.nc' #Hard-coded path to sftlf file for CNRM-CM6 on a 5x5 grid. (Can't input through pre-processor at the moment. Update with sftlf for each model through preprocessor later.)
    
    grouped_input_data = group_metadata(
        input_data, 'dataset', sort='ensemble')
    logger.info(
        "Group input data by model and sort by ensemble:"
        "\n%s", pformat(grouped_input_data))
    print (type (grouped_input_data))
    print (len(grouped_input_data))
    nmodel=len(grouped_input_data)
    experiments=['historical-ssp245','hist-nat','hist-GHG','hist-aer','hist-CO2','hist-stratO3','hist-volc','hist-sol']
    nexp=len(experiments)
    print ('Number of experiments', nexp)
    # Loop over variables/datasets in alphabetical order
#   Define variables for D&A analysis
    diag_name='twoyr_mean_gmst'
    years=list(numpy.arange(1851,2021,2)) #Used for plotting.
    ldiag=85 #length of diagnostic,hard-coded for the moment.
#    diag_name='dec_mean_gmst'
#    years=list(numpy.arange(1855,2025,10)) #Used for plotting.
#    ldiag=17 #length of diagnostic,hard-coded for the moment.

    anom_max=500 #arbitrary max size for number of anomalies.
    mean_diag=numpy.zeros((ldiag,nexp,nmodel))
    mean_dec_warming=numpy.zeros((nexp,nmodel))
    anom=numpy.zeros((ldiag,anom_max))
    anom_index=0
    ens_sizes=numpy.zeros((nexp,nmodel))

    for mm, dataset in enumerate(grouped_input_data):
        logger.info("*************** Processing model %s", dataset)
        grouped_model_input_data = group_metadata(
            grouped_input_data[dataset], 'exp', sort='ensemble')
        for exp in grouped_model_input_data:
            logger.info("***** Processing experiment %s", exp)
            exp_string = [experiments.index(i) for i in experiments if exp == i]
            experiment = exp_string[0]
            print ('*** Experiment',exp,'Index:',experiment)
            grouped_exp_input_data = group_metadata(
              grouped_model_input_data[exp], 'ensemble', sort='variable_group')
            nens=len(grouped_exp_input_data)
            ens_sizes[experiment,mm]=nens
            exp_diags=numpy.zeros((ldiag,nens))
            exp_dec_warming=numpy.zeros(nens)
        
            for ee, ensemble in enumerate(grouped_exp_input_data):
                logger.info("** Processing ensemble %s", ensemble)
                files=[]
                for attributes in grouped_exp_input_data[ensemble]:
                    logger.info("Processing variable %s", attributes['variable_group'])
                    files.append(attributes['filename'])
                logger.info("*************** Files for blend and mask %s", files)
                dec_warming=[]
                (exp_diags[:,ee],had4_diag)=ncbm.ncblendmask_esmval('max', files[0],files[1],files[2],sftlf_file,had4_file,dec_warming,0,diag_name)
                exp_dec_warming[ee]=dec_warming[0]
            mean_diag[:,experiment,mm]=numpy.mean(exp_diags,axis=1)
            mean_dec_warming[experiment,mm]=numpy.mean(dec_warming)
            if nens>1: anom[:,anom_index:anom_index+nens-1]=(exp_diags[:,0:nens-1]-mean_diag[:,experiment:experiment+1,mm])*((nens/(nens-1))**0.5) #Intra-ensemble anomalies for use as pseudo-control. Only use nens-1 ensemble members to ensure independence, and inflate variance by sqrt (nens / (nens-1)) to account for subtraction of ensemble mean.
            anom_index=anom_index+nens-1
    anom=anom[:,0:anom_index]
    att_out={}
    att_out3={}
    model_names=[]
    fig={}
    mm_attrib=0 #Counter over those models used for attribution.
    for mm, dataset in enumerate(grouped_input_data):
        if mean_diag[0,1,mm] == 0: #If there is no hist-nat simulation skip over model.
            continue
        model_names.append(dataset)
        print ('Model:',dataset)
#        (xr,yr,cn1,cn2)=da.reduce_dim(mean_diag[:,[0,1],mm],had4_diag[:,None],anom[:,0:int(anom_index/2)],anom[:,int(anom_index/2):anom_index])
        (xr,yr,cn1,cn2)=da.reduce_dim(mean_diag[:,[0,1,2],mm],had4_diag[:,None],anom[:,list(range(1,anom_index,2))],anom[:,list(range(0,anom_index,2))])

        att_out[dataset]=da.tls(xr[:,0:2],yr,ne=ens_sizes[[0,1],mm],cn1=cn1,cn2=cn2,flag_2S=1)
        att_out3[dataset]=da.tls(xr[:,0:3],yr,ne=ens_sizes[[0,1,2],mm],cn1=cn1,cn2=cn2,flag_3S=1)
        if mm_attrib == 0:
            ant='ANT'
            nat='NAT'
            ghg='GHG'
            oth='OTH'
        else:
            ant=None
            nat=None
            ghg=None
            oth=None
#2-way regression coefficients.
        plt.figure(0)
        plt.plot([mm_attrib+0.9,mm_attrib+0.9],numpy.transpose(att_out[dataset]['betaCI'][0,:]),color='red',linewidth=4,label=ant)
        plt.plot([mm_attrib+1.1,mm_attrib+1.1],numpy.transpose(att_out[dataset]['betaCI'][1,:]),color='blue',linewidth=4,label=nat)
#2-way attributable warming.
        plt.figure(1)
        plt.plot([mm_attrib+0.9,mm_attrib+0.9],numpy.transpose(att_out[dataset]['betaCI'][0,:]*(mean_dec_warming[0,mm]-mean_dec_warming[1,mm])),color='red',linewidth=4,label=ant)
        plt.plot  ([mm_attrib+1.1,mm_attrib+1.1],numpy.transpose(att_out[dataset]['betaCI'][1,:]*mean_dec_warming[1,mm]),color='blue',linewidth=4,label=nat)
#3-way regression coefficients.
        plt.figure(2)
        plt.plot([mm_attrib+0.8,mm_attrib+0.8],numpy.transpose(att_out3[dataset]['betaCI'][2,:]),color='green',linewidth=4,label=ghg)
        plt.plot([mm_attrib+1.0,mm_attrib+1.0],numpy.transpose(att_out3[dataset]['betaCI'][1,:]),color='blue',linewidth=4,label=nat)    
        plt.plot([mm_attrib+1.2,mm_attrib+1.2],numpy.transpose(att_out3[dataset]['betaCI'][0,:]),color='yellow',linewidth=4,label=oth)
#3-way attributable warming.
        plt.figure(3)
        plt.plot([mm_attrib+0.8,mm_attrib+0.8],numpy.transpose(att_out3[dataset]['betaCI'][2,:]*mean_dec_warming[2,mm]),color='green',linewidth=4,label=ghg)
        plt.plot([mm_attrib+1.0,mm_attrib+1.0],numpy.transpose(att_out3[dataset]['betaCI'][1,:]*mean_dec_warming[1,mm]),color='blue',linewidth=4,label=nat)    
        plt.plot([mm_attrib+1.2,mm_attrib+1.2],numpy.transpose(att_out3[dataset]['betaCI'][0,:]*(mean_dec_warming[0,mm]-mean_dec_warming[1,mm]-mean_dec_warming[2,mm])),color='yellow',linewidth=4,label=oth)  
        mm_attrib=mm_attrib+1 #Counter over those models used for attribution.
        
    nmodel_attrib=mm_attrib
    if diag_name=='dec_mean_gmst':
      obs_warming=had4_diag[16]-numpy.mean(had4_diag[0:5])
    elif diag_name=='fiveyr_mean_gmst':
      obs_warming=numpy.mean(had4_diag[32:34])-numpy.mean(had4_diag[0:10])
    elif diag_name=='twoyr_mean_gmst':
      obs_warming=numpy.mean(had4_diag[80:85])-numpy.mean(had4_diag[0:25])
    print (obs_warming)
    print ('had4diag')
    print (had4_diag)
    print ('anom_index',anom_index)
    for ff in range(1,4,2):
        plt.figure(ff)
        plt.plot([0,nmodel_attrib+1],[obs_warming,obs_warming],color='black',linewidth=1,label='Had4 GMST')
        plt.ylabel('Attributable warming in GSAT 2010-2019 vs 1850-1900 in K') 
    for ff in range(0,4,2):
        plt.figure(ff)
        plt.plot([0,nmodel_attrib+1],[1,1],color='black',linewidth=1,ls=':')
        plt.ylabel('Regression coefficients for 1850-2019 GMST')

    filenames=['regcoeffs2','attrib2','regcoeffs3','attrib3']
    for ff in range(0,4,1):
        plt.figure(ff)
        plt.plot([0,nmodel_attrib+1],[0,0],color='black',linewidth=1,ls='--')
        plt.axis([0,nmodel_attrib+1,-1,2])
        plt.xticks(list(range(1,nmodel_attrib+1)),model_names, size='x-small')
        plt.xlabel('Model')
        plt.legend(loc="upper right")
        plt.savefig('/home/rng/plots/esmvaltool/'+filenames[ff]+'_'+diag_name+'.png')
        plt.close()
    


    print (att_out)
    print (mean_dec_warming[0,:])
    print (mean_diag[:,0,:])
    plt.plot(years,anom[:,0],color="Gray",label='Pseudo-control')
    for ann in range(1,anom_index):
        plt.plot(years,anom[:,ann],color="Gray")
    for mm, dataset in enumerate(grouped_input_data):
        if mean_diag[0,1,mm] == 0: #If there is no hist-nat simulation skip over model.
            continue
#        if mm == 0:
#            myls='-'
#        else:
#            myls=':'
        plt.plot(years,mean_diag[:,0,mm],color='C'+str(mm),linewidth=4,label=dataset+'_ALL')
        plt.plot(years,mean_diag[:,1,mm],color='C'+str(mm),linewidth=4,label=dataset+'_NAT',ls='--')
    plt.plot(years,had4_diag,color="Black",linewidth=4,label='HadCRUT4')
    plt.legend(loc="upper left")
    plt.xlabel('Year')
    plt.ylabel('Temperature anomaly (K)')
    plt.savefig('/home/rng/plots/esmvaltool/timeseries_ant_nat_'+diag_name+'_rev_anom.png')
#    plt.show()
    plt.close()
    
    print ('Anomaly variance - whole thing')
    print (numpy.mean(anom*anom,axis=0))

if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
