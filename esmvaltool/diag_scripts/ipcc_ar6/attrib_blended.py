"""Perform attribution using blended and masked temperatures."""
import logging
import os
from pprint import pformat
from scipy.stats import t

import iris

import numpy,time
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

#def attrib_warming(beta,betaCI,dec_warming,ci95_dec_warming):
#  print ('attrib_warming')
#  print ('attrib_warming inputs:',beta,betaCI,dec_warming,ci95_dec_warming)
#  ci95_beta=(betaCI[1]-betaCI[0])/2.
#  attrib_warming=beta*dec_warming
#  ci_attrib_warming=((attrib_warming**2)**0.5)*((ci95_beta/beta)**2+(ci95_dec_warming/dec_warming)**2)**0.5
#  print ('attrib_warming, ci_attrib_warming',attrib_warming,ci_attrib_warming)
#  return (attrib_warming,ci_attrib_warming)

def attrib_warming(beta,betaCI,dec_warming,ci95_dec_warming):
  print ('attrib_warming inputs:',beta,betaCI,dec_warming,ci95_dec_warming)
  attrib_warming=beta*dec_warming
  attrib_range=[attrib_warming-numpy.absolute(beta)*dec_warming*(((beta-betaCI[0])/beta)**2+(ci95_dec_warming/dec_warming)**2)**0.5,attrib_warming+numpy.absolute(beta)*dec_warming*(((betaCI[1]-beta)/beta)**2+(ci95_dec_warming/dec_warming)**2)**0.5]
  print ('attrib_warming, attrib_range',attrib_warming,numpy.sort(attrib_range))
  print ('betaCI*dec_warming',betaCI*dec_warming)
  return (attrib_warming,numpy.sort(attrib_range))



def main(cfg):
    plt.ioff() #Turn off interactive plotting.
    """Compute the time average for each input dataset."""
    # Get a description of the preprocessed data that we will use as input.
    input_data = cfg['input_data'].values()
#    had4_file='/mnt/lustre02/work/bd0854/DATA/ESMValTool2/RAWOBS/Tier2/HadCRUT4/HadCRUT.4.6.0.0.median.nc'
#    had4_file='/home/rng/data/esmvaltool/HadCRUT.4.6.0.0.median_2019.nc'
#    had4_file='/home/rng/data/esmvaltool/HadCRUT.4.6.0.0.median.nc'
    had4_file='/home/rng/data/esmvaltool/HadCRUT.5.0.0.0.analysis.anomalies.ensemble_median.nc'
    had5_flag=False
    if had5_flag:
        had4_file='/home/rng/data/esmvaltool/HadCRUT.5.0.0.0.analysis.anomalies.ensemble_median.nc'
    else:
        had4_file='/home/rng/data/esmvaltool/HadCRUT.4.6.0.0.median_2019.nc'
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
#    exp_flag='AER' #Use hist-aer for 3-way regression.
    rcplot=False
    exp_flag='GHG' #Use hist-AER
# IPCC colours
    colors=numpy.array([[0,0,0],[196,121,0],[178,178,178],[0,52,102],[0,79,0],[200,0,0],[0,200,0],[0,0,200],[112,160,205],])/256.
    shade_cols=numpy.array([[128,128,128,128],[204,174,113,128],[191,191,191,128],[67,147,195,128],[223,237,195,128],[255,150,150,128],[150,255,150,128],[150,150,255,128],[91,174,178,128]])/256.
    if exp_flag=='GHG':
      experiments=['historical-ssp245','hist-nat','hist-GHG','hist-aer','hist-CO2','hist-stratO3','hist-volc','hist-sol']
      label=['OTH','NAT','GHG']
      cols=colors[[3,4,2],:]
#      shade_cols=shade_cols[[3,4,2],:]
    else:
      label=['GHG','NAT','AER']
      cols=colors[[2,4,3],:]        
#      shade_cols=shade_cols[[2,4,3],:]        
      experiments=['historical-ssp245','hist-nat','hist-aer','hist-GHG','hist-CO2','hist-stratO3','hist-volc','hist-sol'] 
    nexp=len(experiments)
    print ('Number of experiments', nexp)
    # Loop over variables/datasets in alphabetical order
#   Define variables for D&A analysis
    diag_name='gmst05'
    av_yr=int(diag_name[4:6]) #Last two digits of diag_name are averaging period in yrs.
    years=list(numpy.arange(1850+av_yr/2,2020+av_yr/2,av_yr)) #Used for plotting.
    ldiag=int(170/av_yr) #length of diagnostic, assuming 170 years of data.
    if diag_name[0:4]=='hemi': ldiag=ldiag*2
#    diag_name='gmst10'
#    years=list(numpy.arange(1855,2025,10)) #Used for plotting.
#    ldiag=17 #length of diagnostic,hard-coded for the moment.
#Make summary figure
    plt.figure(1,figsize=[7,5])
    nbar=5
    labels=['Obs','ANT','GHG','OTH','NAT']
    min_assess=[0.97,0.8,0.9,-0.7,-0.1]
    max_assess=[1.25,1.4,2.0, 0.2, 0.1]
    min_ribes=[-99,0.88,1.06,-0.62,0.04]
    max_ribes=[-99,1.16,1.68,-0.08,0.08]
    mean_ribes=[-99,1.02,1.37,-0.35,0.06]
    min_haustein=[-99,0.911*1.043,-99,-99,-0.001*1.043]
    max_haustein=[-99,1.157*1.043,-99,-99,0.062*1.043]
    mean_haustein=[-99,1.013*1.043,1.199*1.043,-0.186*1.043,0.025*1.043]
    min_gillett=[-99,0.99826127,0.9975069,-0.6491212,-0.02016995]
    max_gillett=[-99,1.36291939,1.91118483,0.12322122,0.03770504]
    mean_gillett=[-99,1.1783,1.4520,-0.2610,0.0093]
    min_jenkins=[-99,0.98,-99,-99,-99]
    max_jenkins=[-99,1.31,-99,-99,-99]
    mean_jenkins=[-99,1.12,-99,-99,-99]
    min_ch7=[-99,0.63127327,1.02726714,-0.83781359,0.04701435]
    max_ch7=[-99,1.8415701,2.21846919,0.02637976,0.12845557]
    mean_ch7=[-99,1.09342969,1.42057137,-0.31795234,0.07424691]
#    min_ch7=[-99,0.449419412,0.992627059,-0.993207647,-0.050020784]
#    max_ch7=[-99,1.774519412,2.239827059,0.331892353,0.117979216]
#    mean_ch7=[-99,0.9857,1.4426,-0.4569,0.04818]
    min_ch7=[-99,0.32356418,1.02726714,-1.2665911,0.04701435]
    mean_ch7=[-99,0.90105274,1.42057137,-0.50680687,0.07424691]
    max_ch7=[-99,1.64593739,2.21846919,-0.04697724,0.1301182]

    xpos=range(nbar)
    lbl=['Ribes','Haustein','Gillett','Jenkins','Chapter 7','CMIP6']
    for bar in range(nbar):        
        plt.fill_between([xpos[bar]-0.5,xpos[bar]+0.5],[min_assess[bar],min_assess[bar]],[max_assess[bar],max_assess[bar]],color=shade_cols[bar,:])
        plt.plot([xpos[bar]-0.375,xpos[bar]-0.375],[min_ribes[bar],max_ribes[bar]],color=colors[bar,:],label=lbl[0],linewidth=4)
        plt.plot([xpos[bar]-0.375],[mean_ribes[bar]],color=colors[bar,:],marker='+',linewidth=4)
        plt.plot([xpos[bar]-0.225,xpos[bar]-0.225],[min_haustein[bar],max_haustein[bar]],color=colors[bar,:],label=lbl[1],linewidth=3)
        plt.plot([xpos[bar]-0.225],[mean_haustein[bar]],color=colors[bar,:],marker='+',linewidth=3)
        plt.plot([xpos[bar]-0.075,xpos[bar]-0.075],[min_gillett[bar],max_gillett[bar]],color=colors[bar,:],label=lbl[2],linewidth=2)
        plt.plot([xpos[bar]-0.075],[mean_gillett[bar]],color=colors[bar,:],marker='+',linewidth=2)
        plt.plot([xpos[bar]+0.075,xpos[bar]+0.075],[min_jenkins[bar],max_jenkins[bar]],color=colors[bar,:],label=lbl[3],linewidth=1)
        plt.plot([xpos[bar]+0.075],[mean_jenkins[bar]],color=colors[bar,:],marker='+',linewidth=1)
        plt.plot([xpos[bar]+0.225,xpos[bar]+0.225],[min_ch7[bar],max_ch7[bar]],color=colors[bar,:],label=lbl[4],linewidth=0.5)
        plt.plot([xpos[bar]+0.225],[mean_ch7[bar]],color=colors[bar,:],marker='+',linewidth=0.5)
        lbl=['','','','','','']
    plt.plot([-10,10],[0,0],color='black')
    plt.axis([-1,nbar,-1.5,2.5])
    plt.xticks(list(range(0,nbar)),labels, size='small')
    plt.ylabel('Attributable change in GSAT, 2010-2019 vs 1850-1900 ($^\circ$C)')
    plt.xlabel('Forcing')
   
    anom_max=500 #arbitrary max size for number of anomalies.
    nens_max=100
    mean_diag=numpy.zeros((ldiag,nexp,nmodel))
    mean_dec_warming=numpy.zeros((nexp,nmodel))
    ci95_dec_warming=numpy.zeros((nexp,nmodel))
    pseudo_obs=numpy.zeros((ldiag,nens_max,nmodel))
    anom=numpy.zeros((ldiag,anom_max))
    anom_index=0
    ens_sizes=numpy.zeros((nexp,nmodel))

    for mm, dataset in enumerate(grouped_input_data):
#        if mm > 3: continue
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
                had4_dec_warming=[]
                (exp_diags[:,ee],had4_diag)=ncbm.ncblendmask_esmval('max', files[0],files[1],files[2],sftlf_file,had4_file,dec_warming,had4_dec_warming,0,0,diag_name,had5_flag)
                if exp=='historical-ssp245': pseudo_obs[:,ee,mm]=exp_diags[:,ee]
                exp_dec_warming[ee]=dec_warming[0]
#Calculate attributable warming in 2010-2019 vs 1850-1899 GMST (assumes gmst05 is diagnostic).
#                exp_dec_warming[ee]=numpy.mean(exp_diags[32:34,ee])-numpy.mean(exp_diags[0:10,ee])
            mean_diag[:,experiment,mm]=numpy.mean(exp_diags,axis=1)
            mean_dec_warming[experiment,mm]=numpy.mean(exp_dec_warming)
            if nens==1:
                ci95_dec_warming[experiment,mm]=ci95_dec_warming[numpy.nonzero(ci95_dec_warming*(ens_sizes**0.5))].mean() #use mean of CIs already calculated, corrected for ensemble size.
            else:        
                ci95_dec_warming[experiment,mm]=(numpy.std(exp_dec_warming,ddof=1)/((nens)**0.5))*t.ppf(0.95,nens-1)
            if nens>1: anom[:,anom_index:anom_index+nens-1]=(exp_diags[:,0:nens-1]-mean_diag[:,experiment:experiment+1,mm])*((nens/(nens-1))**0.5) #Intra-ensemble anomalies for use as pseudo-control. Only use nens-1 ensemble members to ensure independence, and inflate variance by sqrt (nens / (nens-1)) to account for subtraction of ensemble mean.
            anom_index=anom_index+nens-1
    anom=anom[:,0:anom_index]
    att_out={}
    att_out3={}
    model_names=[]
    model_indices=[]
    fig={}
    mm_attrib=0 #Counter over those models used for attribution.
    print ('Number of anomaly segments',anom_index)
    for mm, dataset in enumerate(grouped_input_data):
#        if mm > 3: continue
        if mean_diag[0,1,mm] == 0: #If there is no hist-nat simulation skip over model.
            continue
        model_names.append(dataset)
        model_indices.append(mm)
        print ('Model:',dataset)
        start=time.time()
#        (xr,yr,cn1,cn2)=da.reduce_dim(mean_diag[:,[0,1],mm],had4_diag[:,None],anom[:,0:int(anom_index/2)],anom[:,int(anom_index/2):anom_index])
        (xr,yr,cn1,cn2)=da.reduce_dim(mean_diag[:,[0,1,2],mm],had4_diag[:,None],anom[:,list(range(0,anom_index,2))],anom[:,list(range(1,anom_index,2))])
        time1 =time.time()
        print ('1***',time1-start)
        att_out[dataset]=da.tls(xr[:,0:2],yr,cn1,ne=ens_sizes[[0,1],mm],cn2=cn2,flag_2S=1,RCT_flag=rcplot)
        time2=time.time()
        print ('2***',time2-time1)
        att_out3[dataset]=da.tls(xr[:,0:3],yr,cn1,ne=ens_sizes[[0,1,2],mm],cn2=cn2,flag_3S=1,RCT_flag=rcplot)
# Replace beta confidence interval bounds with large positive and negative numbers if NaNs.
        att_out3[dataset]['betaCI'][numpy.isnan(att_out3[dataset]['betaCI'][:,0]),0]=-1000. 
        att_out3[dataset]['betaCI'][numpy.isnan(att_out3[dataset]['betaCI'][:,1]),1]=1000.
# If upper end of confidence range < lower end, replace with large positive number.
        for pat in range(3):
            if att_out3[dataset]['betaCI'][pat,1]<att_out3[dataset]['betaCI'][pat,0]:
# If max of beta is less than min, draw unconstrained ranges.
                att_out3[dataset]['betaCI'][pat,1]=1000.
                att_out3[dataset]['betaCI'][pat,0]=-1000.
                
#print ([att_out3[dataset]['betaCI'][:,1]<att_out3[dataset]['betaCI'][:,0]])
#        att_out3[dataset]['betaCI'][[att_out3[dataset]['betaCI'][:,1]<att_out3[dataset]['betaCI'][:,0]],1]=1000.
        time3=time.time()
        print ('3***',time3-time2)
        if mm_attrib == 0:
            ant='ANT'
            nat='NAT'
            ghg=exp_flag #Set to GHG or AER.
            oth='OTH'
        else:
            ant=None
            nat=None
            ghg=None
            oth=None
        if rcplot:
            topleft=321
            topright=322
            bottomleft=325
            bottomright=326
        else:        
            topleft=221
            topright=222
            bottomleft=223
            bottomright=224
#2-way regression coefficients.
        if rcplot:
            plt.figure(0,figsize=[7,9]) 
        else:
            plt.figure(0,figsize=[7,6]) 
        plt.subplot(topleft)
        plt.plot([mm_attrib+0.9,mm_attrib+0.9],numpy.transpose(att_out[dataset]['betaCI'][0,:]),color=colors[1,:],linewidth=4,label=ant)
        plt.plot([mm_attrib+1.1,mm_attrib+1.1],numpy.transpose(att_out[dataset]['betaCI'][1,:]),color=colors[4,:],linewidth=4,label=nat)
        if rcplot:
          plt.subplot(323)
          plt.bar([mm_attrib+1],[att_out[dataset]['rc_pvalue']],color='gray')
#2-way attributable warming.
        plt.subplot(bottomleft)
        [att_warming,att_warming_range]=attrib_warming(att_out[dataset]['beta'][0],att_out[dataset]['betaCI'][0,:],mean_dec_warming[0,mm]-mean_dec_warming[1,mm],(ci95_dec_warming[0,mm]**2+ci95_dec_warming[1,mm]**2)**0.5)
        plt.plot([mm_attrib+0.9,mm_attrib+0.9],att_warming_range,color=colors[1,:],linewidth=4,label=ant)
#        plt.plot([mm_attrib+0.99,mm_attrib+0.99],att_out[dataset]['betaCI'][0,:]*(mean_dec_warming[0,mm]-mean_dec_warming[1,mm]),color=colors[1,:],linewidth=4,label=ant)
        [att_warming,att_warming_range]=attrib_warming(att_out[dataset]['beta'][1],att_out[dataset]['betaCI'][1,:],mean_dec_warming[1,mm],ci95_dec_warming[1,mm])
        plt.plot  ([mm_attrib+1.1,mm_attrib+1.1],att_warming_range,color=colors[4,:],linewidth=4,label=nat)
#        plt.plot  ([mm_attrib+1.19,mm_attrib+1.19],att_out[dataset]['betaCI'][1,:]*mean_dec_warming[1,mm],color=colors[4,:],linewidth=4,label=nat)
#3-way regression coefficients.
        plt.subplot(topright)
        plt.plot([mm_attrib+0.8,mm_attrib+0.8],numpy.transpose(att_out3[dataset]['betaCI'][2,:]),color=cols[2,:],linewidth=4,label=label[2])
        plt.plot([mm_attrib+1.0,mm_attrib+1.0],numpy.transpose(att_out3[dataset]['betaCI'][1,:]),color=cols[1,:],linewidth=4,label=label[1])    
        plt.plot([mm_attrib+1.2,mm_attrib+1.2],numpy.transpose(att_out3[dataset]['betaCI'][0,:]),color=cols[0,:],linewidth=4,label=label[0])
        if rcplot:
            plt.subplot(324)
            plt.bar([mm_attrib+1],[att_out3[dataset]['rc_pvalue']],color='gray')
#3-way attributable warming.
        plt.subplot(bottomright)
        [att_warming,att_warming_range]=attrib_warming(att_out3[dataset]['beta'][2],att_out3[dataset]['betaCI'][2,:],mean_dec_warming[2,mm],ci95_dec_warming[2,mm])
        plt.plot([mm_attrib+0.8,mm_attrib+0.8],att_warming_range,color=cols[2,:],linewidth=4,label=label[2])
#        plt.plot([mm_attrib+0.89,mm_attrib+0.89],att_out3[dataset]['betaCI'][2,:]*mean_dec_warming[2,mm],color=cols[2,:],linewidth=4,label=label[2])

        
        [att_warming,att_warming_range]=attrib_warming(att_out3[dataset]['beta'][1],att_out3[dataset]['betaCI'][1,:],mean_dec_warming[1,mm],ci95_dec_warming[1,mm])
        plt.plot([mm_attrib+1.0,mm_attrib+1.0],att_warming_range,color=cols[1,:],linewidth=4,label=label[1])
#        plt.plot([mm_attrib+1.09,mm_attrib+1.09],att_out3[dataset]['betaCI'][1,:]*mean_dec_warming[1,mm],color=cols[1,:],linewidth=4,label=label[1])

        
        [att_warming,att_warming_range]=attrib_warming(att_out3[dataset]['beta'][0],att_out3[dataset]['betaCI'][0,:],mean_dec_warming[0,mm]-mean_dec_warming[1,mm]-mean_dec_warming[2,mm],(ci95_dec_warming[0,mm]**2+ci95_dec_warming[1,mm]**2+ci95_dec_warming[2,mm]**2)**0.5)
        plt.plot([mm_attrib+1.2,mm_attrib+1.2],att_warming_range,color=cols[0,:],linewidth=4,label=label[0])
#        plt.plot([mm_attrib+1.29,mm_attrib+1.29],att_out3[dataset]['betaCI'][0,:]*(mean_dec_warming[0,mm]-mean_dec_warming[1,mm]-mean_dec_warming[2,mm]),color=cols[0,:],linewidth=4,label=label[0])
        
        mm_attrib=mm_attrib+1 #Counter over those models used for attribution.
        label=[None,None,None]

#Multi-model analysis.
    dataset='Multi'
    model_names.append(dataset)
#    anom=anom*1.5 #Scale internal variability.
    (xr,yr,cn1,cn2)=da.reduce_dim(numpy.mean(mean_diag[:,0:3,model_indices],axis=2),had4_diag[:,None],anom[:,list(range(1,anom_index,2))],anom[:,list(range(0,anom_index,2))])
    neff=mm_attrib**2/numpy.sum(1./ens_sizes[0:3,model_indices],axis=1) #Effective ensemble size when using multi-model mean.
    print ('*****attrib_blended ens_sizes',ens_sizes[0:3,model_indices])
    print ('*****attrib_blended neff',neff)
    att_out[dataset]=da.tls(xr[:,0:2],yr,cn1,ne=neff[[0,1]],cn2=cn2,flag_2S=1,RCT_flag=rcplot)
    att_out3[dataset]=da.tls(xr[:,0:3],yr,cn1,ne=neff[[0,1,2]],cn2=cn2,flag_3S=1,RCT_flag=rcplot)
    multim_mean_dec_warming=numpy.mean(mean_dec_warming[:,model_indices],axis=1)
    multim_ci95_dec_warming=numpy.mean(ci95_dec_warming[:,model_indices],axis=1)/len(model_indices)
    #2-way regression coefficients.
    plt.subplot(topleft)
    plt.plot([mm_attrib+0.9,mm_attrib+0.9],numpy.transpose(att_out[dataset]['betaCI'][0,:]),color=colors[1,:],linewidth=4,label=ant)
    plt.plot([mm_attrib+1.1,mm_attrib+1.1],numpy.transpose(att_out[dataset]['betaCI'][1,:]),color=colors[4,:],linewidth=4,label=nat)
    if rcplot:
        plt.subplot(323)
        plt.bar([mm_attrib+1],[att_out[dataset]['rc_pvalue']],color='gray')
#2-way attributable warming.
    plt.subplot(bottomleft)
    print ('Two-way attributable warming')
    [att_warming,att_warming_range]=attrib_warming(att_out[dataset]['beta'][0],att_out[dataset]['betaCI'][0,:],multim_mean_dec_warming[0]-multim_mean_dec_warming[1],(multim_ci95_dec_warming[0]**2+multim_ci95_dec_warming[1]**2)**0.5)
    plt.plot([mm_attrib+0.9,mm_attrib+0.9],att_warming_range,color=colors[1,:],linewidth=4,label=ant)
#    plt.plot([mm_attrib+0.99,mm_attrib+0.99],att_out[dataset]['betaCI'][0,:]*(multim_mean_dec_warming[0]-multim_mean_dec_warming[1]),color=colors[1,:],linewidth=4,label=ant)

    
    print ('ANT:',att_warming_range)
    [att_warming,att_warming_range]=attrib_warming(att_out[dataset]['beta'][1],att_out[dataset]['betaCI'][1,:],multim_mean_dec_warming[1],multim_ci95_dec_warming[1])
    plt.plot  ([mm_attrib+1.1,mm_attrib+1.1],att_warming_range,color=colors[4,:],linewidth=4,label=nat)
#    plt.plot  ([mm_attrib+1.19,mm_attrib+1.19],att_out[dataset]['betaCI'][1,:]*multim_mean_dec_warming[1],color=colors[4,:],linewidth=4,label=nat)

    
    print ('NAT:',att_warming_range)
#3-way regression coefficients.
    plt.subplot(topright)
    plt.plot([mm_attrib+0.8,mm_attrib+0.8],numpy.transpose(att_out3[dataset]['betaCI'][2,:]),color=cols[2,:],linewidth=4)
    plt.plot([mm_attrib+1.0,mm_attrib+1.0],numpy.transpose(att_out3[dataset]['betaCI'][1,:]),color=cols[1,:],linewidth=4)    
    plt.plot([mm_attrib+1.2,mm_attrib+1.2],numpy.transpose(att_out3[dataset]['betaCI'][0,:]),color=cols[0,:],linewidth=4)
    if rcplot:
        plt.subplot(324)
        plt.bar([mm_attrib+1],[att_out3[dataset]['rc_pvalue']],color='gray')
#3-way attributable warming.
    print ('Three-way attributable warming')
    plt.subplot(bottomright)
    [att_warming,att_warming_range]=attrib_warming(att_out3[dataset]['beta'][2],att_out3[dataset]['betaCI'][2,:],multim_mean_dec_warming[2],multim_ci95_dec_warming[2])
    plt.plot([mm_attrib+0.8,mm_attrib+0.8],att_warming_range,color=cols[2,:],linewidth=4)
#    plt.plot([mm_attrib+0.89,mm_attrib+0.89],att_out3[dataset]['betaCI'][2,:]*multim_mean_dec_warming[2],color=cols[2,:],linewidth=4)
    
    print (exp_flag,att_warming_range)
    [att_warming,att_warming_range]=attrib_warming(att_out3[dataset]['beta'][1],att_out3[dataset]['betaCI'][1,:],multim_mean_dec_warming[1],multim_ci95_dec_warming[1])
    plt.plot([mm_attrib+1.0,mm_attrib+1.0],att_warming_range,color=cols[1,:],linewidth=4)    
#    plt.plot([mm_attrib+1.09,mm_attrib+1.09],att_out3[dataset]['betaCI'][1,:]*multim_mean_dec_warming[1],color=cols[1,:],linewidth=4)    
    print ('NAT:',att_warming_range)
    [att_warming,att_warming_range]=attrib_warming(att_out3[dataset]['beta'][0],att_out3[dataset]['betaCI'][0,:],multim_mean_dec_warming[0]-multim_mean_dec_warming[1]-multim_mean_dec_warming[2],(multim_ci95_dec_warming[0]**2+multim_ci95_dec_warming[1]**2+multim_ci95_dec_warming[2]**2)**0.5)
    plt.plot([mm_attrib+1.2,mm_attrib+1.2],att_warming_range,color=cols[0,:],linewidth=4)
#    plt.plot([mm_attrib+1.29,mm_attrib+1.29],att_out3[dataset]['betaCI'][0,:]*(multim_mean_dec_warming[0]-multim_mean_dec_warming[1]-multim_mean_dec_warming[2]),color=cols[0,:],linewidth=4)
    
    print ('OTH:',att_warming_range)
    mm_attrib=mm_attrib+1 #Counter over those models used for attribution.


    
    nmodel_attrib=mm_attrib-1
      

    if had5_flag:
        obs_warming=had4_dec_warming[0]*1.1059 #Equivalent ratio for HadCRUT5
    else:
        obs_warming=had4_dec_warming[0]*1.1804689337859493 #Scale up by ensemble mean GSAT/GMST ratio in CMIP6.
#    obs_warming=had4_dec_warming[0] #Set for plotting GMST warming.

    print ('Obs warming',obs_warming)
    print ('Obs warming in GMST')

    for ff in [bottomleft,bottomright]:
        plt.subplot(ff)
        plt.plot([0,nmodel_attrib+2],[obs_warming,obs_warming],color='black',linewidth=1,label='Had4 GSAT')
        if ff == bottomleft: plt.ylabel('Attributable change 2010-2019 vs 1850-1900 ($^\circ$C)',size='x-small') 
        plt.plot([0,nmodel_attrib+2],[0,0],color='black',linewidth=1,ls='--')
        plt.axis([0,nmodel_attrib+2,-2,3])
        plt.xticks(list(range(1,nmodel_attrib+2)),model_names, size='xx-small',rotation=30.,ha="right")
#         plt.xlabel('Model')
#        plt.legend(loc="lower left")
    for ff in [topleft,topright]:
        plt.subplot(ff)
        plt.plot([0,nmodel_attrib+2],[1,1],color='black',linewidth=1,ls=':')
        if ff == topleft: plt.ylabel('Regression coefficients',size='x-small')
        plt.plot([0,nmodel_attrib+2],[0,0],color='black',linewidth=1,ls='--')
        plt.axis([0,nmodel_attrib+2,-1,3])
#        plt.xticks(list(range(1,nmodel_attrib+1)),model_names, fontsize=5,rotation="vertical")
        plt.xticks(list(range(1,nmodel_attrib+2)),['','','','','',''])
#        plt.xlabel('Model')
        plt.legend(loc="upper left")
    if rcplot:
        for ff in [323,324]:
            plt.subplot(ff)
            plt.plot([0,nmodel_attrib+2],[0.05,0.05],color='black',linewidth=1,ls='--')
            plt.plot([0,nmodel_attrib+2],[0.95,0.95],color='black',linewidth=1,ls='--')
            plt.axis([0,nmodel_attrib+2,0,1])
            plt.xticks(list(range(1,nmodel_attrib+2)),['','','','','',''])
            if ff == 323: plt.ylabel('RCT P-value',size='x-small')


    filenames=['regcoeffs2','attrib2','regcoeffs3','attrib3']
#    for ff in range(0,4,1):
#    plt.figure(0)
    if had5_flag:
        plt.savefig('/home/rng/plots/esmvaltool/reg_attrib_'+diag_name+'_'+exp_flag+'_HadCRUT5.png')
    else:
        plt.savefig('/home/rng/plots/esmvaltool/reg_attrib_'+diag_name+'_'+exp_flag+'.png')
    plt.close()
    out_dec_warming=numpy.zeros((5,nmodel_attrib))
    out_dec_warming[1,:]=mean_dec_warming[0,model_indices]-mean_dec_warming[1,model_indices]
    out_dec_warming[2,:]=mean_dec_warming[2,model_indices]
    out_dec_warming[3,:]=mean_dec_warming[0,model_indices]-mean_dec_warming[1,model_indices]-mean_dec_warming[2,model_indices]
    out_dec_warming[4,:]=mean_dec_warming[1,model_indices]
    out_dec_warming[0,:]=numpy.zeros(nmodel_attrib)-99. #Dummy data
    plt.figure(1)
    markers=['o','v','P','*','x','X','D']
    for bar in range(0,nbar):
        for mm in range (nmodel_attrib):
            plt.plot([xpos[bar]+0.375],out_dec_warming[bar,mm],color=colors[bar,:],marker=markers[mm],linewidth=0,label=model_names[mm])
        model_names=['','','','','','','','','']
    plt.legend(loc="upper right",prop={'size':8})  
    plt.savefig('/home/rng/plots/esmvaltool/summary_fig.png')
    plt.close()


    if diag_name[0:4]=='hemi':
      nlat=2
      nper=ldiag//2
      mean_diag=numpy.mean(numpy.reshape(mean_diag,(nlat,nper,nexp,nmodel)),axis=0)
      had4_diag=numpy.mean(numpy.reshape(had4_diag,(nlat,nper)),axis=0)
      anom=numpy.mean(numpy.reshape(anom,(nlat,nper,anom_index)),axis=0)
    print (att_out)
    print (att_out3)
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
        plt.plot(years,mean_diag[:,1,mm],color='C'+str(mm),linewidth=4,ls='--')
#Multi-model mean ANT response
    ant=numpy.mean(mean_diag[:,0,:]-mean_diag[:,1,:],axis=1)*att_out['Multi']['beta'][0]
    plt.plot(years,ant,color="Red",linewidth=4,label='Scaled anthropogenic')
    plt.plot(years,had4_diag,color="Black",linewidth=4,label='HadCRUT4')
    plt.legend(loc="upper left")
    plt.xlabel('Year')
    plt.ylabel('GMST anomaly ($^\circ$C)')
    plt.savefig('/home/rng/plots/esmvaltool/timeseries_ant_nat_'+diag_name+'_'+exp_flag+'.png')
#    plt.show()
    plt.close()
    print ('Regression coefficient',att_out['Multi']['beta'][0])
    
    print ('Anomaly variance - whole thing')
    print (numpy.mean(anom*anom,axis=0))

    
    

if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
