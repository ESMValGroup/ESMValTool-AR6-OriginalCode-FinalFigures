"""Plots timeseries of masked and blended GMST from CMIP6 models, for comparison with obs."""
import logging
import os
from pprint import pformat

import iris

import pandas,numpy
import csv
import matplotlib
matplotlib.use('Agg')
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
    matplotlib.use('Agg')
    plt.ioff() #Turn off interactive plotting.
    """Compute the time average for each input dataset."""
    # Get a description of the preprocessed data that we will use as input.
    input_data = cfg['input_data'].values()
#    had4_file='/mnt/lustre02/work/bd0854/DATA/ESMValTool2/RAWOBS/Tier2/HadCRUT4/HadCRUT.4.6.0.0.median.nc'
#    had4_file='/home/rng/data/esmvaltool/HadCRUT.4.6.0.0.median.nc'
    had4_file='/home/rng/data/esmvaltool/HadCRUT.4.6.0.0.median_2019.nc'
    sftlf_file='/home/rng/data/esmvaltool/CNRM-CM6-1-5x5-sftlf.nc'
#sftlf_file='/pf/b/b380746/CNRM-CM6-1-5x5-sftlf.nc' #Hard-coded path to sftlf file for CNRM-CM6 on a 5x5 grid. (Can't input through pre-processor at the moment. Update with sftlf for each model through preprocessor later.)
    
    grouped_input_data = group_metadata(
        input_data, 'dataset', sort='ensemble')
    logger.info(
        "Group input data by model and sort by ensemble:"
        "\n%s", pformat(grouped_input_data))
    type (grouped_input_data)
    print (len(grouped_input_data))
    nmodel=len(grouped_input_data)
    experiments=['historical-ssp245','hist-GHG','hist-aer','hist-nat','hist-volc','hist-sol','hist-stratO3','hist-CO2']
    long_labels=['Human and natural forcings','Greenhouse gases','Other human forcings','Natural forcings']
#    cols=['red','blue','green','orange','lime','peru','cyan','paleturquoise']
    cols=numpy.array([[0,0,0],[196,121,0],[178,178,178],[0,52,102],[0,79,0],[200,0,0],[0,200,0],[0,0,200],[112,160,205],])/256.
    shade_cols=numpy.array([[128,128,128,128],[204,174,113,128],[191,191,191,128],[67,147,195,128],[223,237,195,128],[255,150,150,128],[150,255,150,128],[150,150,255,128],[91,174,178,128]])/256.
    nexp=len(experiments)
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

#Plot up CH7 timeseries
    plt.figure(0,figsize=[12,12])
    plt.subplot(221)
    ch7=pandas.read_csv('/home/rng/data/esmvaltool/DA_timeseries_AR6_SODaero.csv')
    mn=[ch7['ALL_p50'],ch7['GHG_p50'],ch7['OTH_p50'],ch7['NAT_p50']]
    p95=[ch7['ALL_p95'],ch7['GHG_p95'],ch7['OTH_p95'],ch7['NAT_p95']]
    p05=[ch7['ALL_p05'],ch7['GHG_p05'],ch7['OTH_p05'],ch7['NAT_p05']]
    
    zzs=[3,1,0,2]
    for experiment in range(0,4):
        offset=0
        plt.fill_between(years,p05[experiment],p95[experiment],color=shade_cols[experiment+1,:],zorder=zzs[experiment])
        plt.plot([1850,2025],[0,0],color='black',linewidth=1)
        plt.plot(years,mn[experiment],color=cols[experiment+1,:],linewidth=1,label=long_labels[experiment],zorder=zzs[experiment]+4)

#    plt.plot(ch7['year'],ch7['ALL_p50'],color=cols[0+1,:],linewidth=2)
#    plt.plot(ch7['year'],ch7['GHG_p50'],color=cols[1+1,:],linewidth=2)
#    plt.plot(ch7['year'],ch7['OTH_p50'],color=cols[2+1,:],linewidth=2)
#    plt.plot(ch7['year'],ch7['NAT_p50'],color=cols[3+1,:],linewidth=2)
    

    obs_gsat=pandas.read_csv('/home/rng/data/esmvaltool/GSAT.csv')
    print (obs_gsat['Year'])
    print (obs_gsat['GSAT estimate'])
    plt.plot(obs_gsat['Year'],obs_gsat['GSAT estimate'],color='black')
    plt.axis([1850,2020,-1.5,2.5])
    plt.xlabel('Year')
    plt.ylabel('Global surface air temperature change ($^\circ$C)')
    plt.legend(loc="upper left")

#    plt.savefig('/home/rng/plots/esmvaltool/ts_13_ch7.png')

#    plt.figure(1,figsize=[7,5])
    plt.subplot(224)
    nbar=5
    labels=['Observed warming','Net human influence','Greenhouse gases','Other human forcings','Natural forcings']
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
    min_ch7=[-99,0.32356418,1.02726714,-1.2665911,0.04701435]
    mean_ch7=[-99,0.90105274,1.42057137,-0.50680687,0.07424691]
    max_ch7=[-99,1.64593739,2.21846919,-0.04697724,0.1301182]
#    min_ch7=[-99,0.449419412,0.992627059,-0.993207647,-0.050020784]
#    max_ch7=[-99,1.774519412,2.239827059,0.331892353,0.117979216]
#    mean_ch7=[-99,0.9857,1.4426,-0.4569,0.04818]

    xpos=range(nbar)
    lbl=['Ribes','Haustein','Gillett','Jenkins','Chapter 7','CMIP6']
    for bar in range(nbar):        
        plt.fill_between([xpos[bar]-0.5,xpos[bar]+0.5],[min_assess[bar],min_assess[bar]],[max_assess[bar],max_assess[bar]],color=shade_cols[bar,:])
        plt.plot([xpos[bar]-0.375,xpos[bar]-0.375],[min_ribes[bar],max_ribes[bar]],color=cols[bar,:],label='Attribution study',linewidth=2)
        plt.plot([xpos[bar]-0.375],[mean_ribes[bar]],color=cols[bar,:],marker='+',linewidth=2)
        plt.plot([xpos[bar]-0.225,xpos[bar]-0.225],[min_haustein[bar],max_haustein[bar]],color=cols[bar,:],linewidth=2)
        plt.plot([xpos[bar]-0.225],[mean_haustein[bar]],color=cols[bar,:],marker='+',linewidth=2)
        plt.plot([xpos[bar]-0.075,xpos[bar]-0.075],[min_gillett[bar],max_gillett[bar]],color=cols[bar,:],linewidth=2)
        plt.plot([xpos[bar]-0.075],[mean_gillett[bar]],color=cols[bar,:],marker='+',linewidth=2)
        plt.plot([xpos[bar]+0.075,xpos[bar]+0.075],[min_jenkins[bar],max_jenkins[bar]],color=cols[bar,:],linewidth=2)
        plt.plot([xpos[bar]+0.075],[mean_jenkins[bar]],color=cols[bar,:],marker='+',linewidth=2)
        plt.plot([xpos[bar]+0.225,xpos[bar]+0.225],[min_ch7[bar],max_ch7[bar]],color=cols[bar,:],label='Emulator',linewidth=1)
        plt.plot([xpos[bar]+0.225],[mean_ch7[bar]],color=cols[bar,:],marker='+',linewidth=1)
        lbl=['','','','','','']
    plt.plot([-10,10],[0,0],color='black')
    plt.axis([-1,nbar,-1.5,2.5])
    plt.xticks(list(range(0,nbar)),labels, size='small')
    plt.ylabel('Attributable change in GSAT, 2010-2019 vs 1850-1900 ($^\circ$C)')
    plt.xlabel('Forcing')    

    ANT      =[ 0.32356418,  0.90105274,  1.64593739]
    GHG      =[ 1.02445333,  1.42637698,  2.20385734]
    OTH      =[-1.2665911 , -0.50680687, -0.04697724]
    NAT      =[ 0.04739134,  0.07347462,  0.1301182 ]
 
    CO2      =[ 0.64239345,  0.90666198,  1.37744197]
    CH4      =[ 0.16643687,  0.24925419,  0.40997346]
    N2O      =[ 0.05940655,  0.08308749,  0.13049027]
    otherGHG =[ 0.13063127,  0.18473317,  0.28832533]
    
    aerosols =[-1.50730017, -0.68232325, -0.23551328]
    luc      =[-0.11196746, -0.05233459, -0.01371546]
    ozone    =[ 0.03941459,  0.13363669,  0.28357783]
    bcsnow   =[ 0.01604506,  0.04048831,  0.10126617]
    stwv     =[ 0.0077578 ,  0.03170974,  0.06420214]
    contrails=[ 0.00833304,  0.01798134,  0.04057628]
    
    Volcanic =[ 0.04016167,  0.06087261,  0.10497473]
    Solar    =[-0.00027681,  0.01281306,  0.03263096]
    plt.subplot(222)
    plt.fill_between([xpos[0]-0.5,xpos[0]+0.5],[min_assess[0],min_assess[0]],[max_assess[0],max_assess[0]],color=shade_cols[0,:])
    for bar in range(nbar):
        plt.plot([xpos[bar],xpos[bar]],[min_ch7[bar],max_ch7[bar]],color=cols[bar,:],linewidth=2)
        plt.plot([xpos[bar]],[mean_ch7[bar]],color=cols[bar,:],marker='+',linewidth=2)

    nghg=4
    ghgs=[CO2[1],CH4[1],N2O[1],otherGHG[1]]
    total=0
    for nn in range(nghg):
        colors=cols[2,:] if nn%2 == 0 else shade_cols[2,:]
        plt.fill_between([xpos[2]-0.2,xpos[2]+0.2],[total,total],[total+ghgs[nn],total+ghgs[nn]],color=colors)
        total=total+ghgs[nn]

    noth=6
    oths=[aerosols[1],luc[1],ozone[1],bcsnow[1],stwv[1],contrails[1]]
    total=0
    for nn in range(noth):
        colors=cols[3,:] if nn%2 == 0 else shade_cols[3,:]
        offset=0.4 if nn > 1 else 0.
        plt.fill_between([xpos[3]-0.6+offset,xpos[3]-0.2+offset],[total,total],[total+oths[nn],total+oths[nn]],color=colors)
        total=total+oths[nn]

    nnat=1
    nats=[Volcanic[1]+Solar[1]]
    total=0
    for nn in range(nnat):
        colors=cols[4,:] if nn%2 == 0 else shade_cols[4,:]
        plt.fill_between([xpos[4]-0.2,xpos[4]+0.2],[total,total],[total+nats[nn],total+nats[nn]],color=colors)
        total=total+nats[nn]
        

    

    
    plt.plot([-10,10],[0,0],color='black')
    plt.axis([-1,nbar,-1.5,2.5])
    plt.xticks(list(range(0,nbar)),labels, size='small')
    plt.ylabel('Estimated response in GSAT, 2010-2019 vs 1850-1900 ($^\circ$C)')
    plt.xlabel('Forcing')    

#SPM Figure
    
    plt.figure(1,figsize=[12,6]) #SPM Figure.
    plt.subplot(122)
#    plt.fill_between([xpos[0]-0.5,xpos[0]+0.5],[min_assess[0],min_assess[0]],[max_assess[0],max_assess[0]],color=shade_cols[0,:])
    for bar in range(nbar):
        if bar == 1:
            barno=5
        else:
            barno=bar
        plt.fill_between([xpos[bar]-0.3,xpos[bar]+0.3],[0.,0.],[min_assess[bar]/2.+max_assess[bar]/2.,min_assess[bar]/2.+max_assess[bar]/2.],color=shade_cols[barno,:])
        plt.plot([xpos[bar],xpos[bar]],[min_assess[bar],max_assess[bar]],color=cols[barno,:],linewidth=1)
        plt.plot([xpos[bar]],[min_assess[bar]],color=cols[barno,:],marker='_',linewidth=1)
        plt.plot([xpos[bar]],[max_assess[bar]],color=cols[barno,:],marker='_',linewidth=1)

    nghg=4
    ghgs=[CO2[1],CH4[1],N2O[1],otherGHG[1]]
    total=0
    for nn in range(nghg):
        colors=cols[2,:] if nn%2 == 0 else shade_cols[2,:]
        plt.fill_between([xpos[2]+0.35,xpos[2]+0.4],[total,total],[total+ghgs[nn]-0.05,total+ghgs[nn]-0.05],color=colors)
        plt.plot([xpos[2]+0.375],[total+ghgs[nn]],color=colors,marker=6,linewidth=1)
        total=total+ghgs[nn]

    noth=4
    oths=[aerosols[1],luc[1],ozone[1],bcsnow[1]+stwv[1]+contrails[1]]
    total=0
    for nn in range(noth):
        colors=cols[3,:] if nn%2 == 0 else shade_cols[3,:]
        offset=0.1 if nn > 1 else 0.
        if nn>1:
          plt.fill_between([xpos[3]+0.35+offset,xpos[3]+0.4+offset],[total,total],[total+oths[nn]-0.05,total+oths[nn]-0.05],color=colors)
          plt.plot([xpos[3]+0.375+offset],[total+oths[nn]],color=colors,marker=6,linewidth=1)
        else:
          plt.fill_between([xpos[3]+0.35+offset,xpos[3]+0.4+offset],[total,total],[total+oths[nn]+0.05,total+oths[nn]+0.05],color=colors)
          plt.plot([xpos[3]+0.375+offset],[total+oths[nn]],color=colors,marker=7,linewidth=1)
          

        total=total+oths[nn]

    nnat=2
    nats=[Volcanic[1],Solar[1]]
    total=0
    for nn in range(nnat):
        colors=cols[4,:] if nn%2 == 0 else shade_cols[4,:]
        plt.fill_between([xpos[4]+0.35,xpos[4]+0.4],[total,total],[total+nats[nn]-0.05,total+nats[nn]-0.05],color=colors)
        plt.plot([xpos[4]+0.375],[total+nats[nn]],color=colors,marker=6,linewidth=1)
        total=total+nats[nn]


    
    plt.plot([-10,10],[0,0],color='black')
    plt.axis([-1,nbar,-1.5,2.5])
    plt.xticks(list(range(0,nbar)),labels, size='small',rotation=20.,ha="right")
    plt.ylabel('Contributions to warming in 2010-2019 vs 1850-1900 ($^\circ$C)')
#    plt.xlabel('Forcing')    
    
#    plt.savefig('/home/rng/plots/esmvaltool/spm_fig.png')
#    exit()
    model_names=[]

    for mm, dataset in enumerate(grouped_input_data):
        model_names.append(dataset)
        logger.info("*************** Processing model %s", dataset)
        lbl=dataset
        grouped_model_input_data = group_metadata(
            grouped_input_data[dataset], 'exp', sort='ensemble')
        for exp in grouped_model_input_data:
#            if exp!="historical-ssp245":
#                continue
            logger.info("***** Processing experiment %s", exp)
            exp_string = [experiments.index(i) for i in experiments if exp == i]
            experiment = exp_string[0]
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

#Overwrite volcano response with other anthropogenic - historical-hist-GHG-hist-nat.
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
    plt.figure(0)
    plt.subplot(223)
    zzs=[3,1,0,2]
    for experiment in range(0,4):
#        offset=experiment*-2.
        offset=0
        plt.fill_between(years,range_ann_warming[:,experiment,0]+offset,range_ann_warming[:,experiment,1]+offset,color=shade_cols[experiment+1,:],zorder=zzs[experiment])
        plt.plot([1850,2025],[offset,offset],color='black',linewidth=1)
        plt.plot(years,mm_ann_warming[:,experiment]+offset,color=cols[experiment+1,:],linewidth=1,label=long_labels[experiment],zorder=zzs[experiment]+4)

#    plt.plot(years,had4_diag*1.1719758280521986,color='black',linewidth=1,label='Observations',zorder=8)
    plt.plot(obs_gsat['Year'],obs_gsat['GSAT estimate'],color='black',linewidth=1,label='Observations',zorder=8)
    plt.axis([1850,2020,-1.5,2.5])
    plt.xlabel('Year')
    plt.ylabel('Global surface air temperature change ($^\circ$C)')
    plt.legend(loc="upper left")

    out_dec_warming=numpy.zeros((5,nmodel))
    out_dec_warming[1,:]=mean_dec_warming[0,:]-mean_dec_warming[3,:]
    out_dec_warming[2,:]=mean_dec_warming[1,:]
    out_dec_warming[3,:]=mean_dec_warming[0,:]-mean_dec_warming[1,:]-mean_dec_warming[3,:]
    out_dec_warming[4,:]=mean_dec_warming[3,:]
    out_dec_warming[0,:]=numpy.zeros(nmodel)-99. #Dummy data
    plt.subplot(224)
    markers=['o','v','P','*','x','X','D']
    label='CMIP6 model'
    for bar in range(0,nbar):
        for mm in range (nmodel):
            plt.plot([xpos[bar]+0.375],out_dec_warming[bar,mm],color=cols[bar,:],marker='o',linewidth=0,label=label)
            label=''
        model_names=['','','','','','','','','']
    plt.legend(loc="upper right",prop={'size':8})  


    plt.savefig('/home/rng/plots/esmvaltool/ts_13_4x4.png')

    plt.close()

    plt.figure(1)
    plt.subplot(121)
    zzs=[3,1,0,2]
    for experiment in range(0,4):
#        offset=experiment*-2.
        offset=0
        plt.fill_between(years,range_ann_warming[:,experiment,0]+offset,range_ann_warming[:,experiment,1]+offset,color=shade_cols[experiment+1,:],zorder=zzs[experiment])
        plt.plot([1850,2025],[offset,offset],color='black',linewidth=1)
        plt.plot(years,mm_ann_warming[:,experiment]+offset,color=cols[experiment+1,:],linewidth=1,label=long_labels[experiment],zorder=zzs[experiment]+4)
        plt.plot(years,mn[experiment],color=cols[experiment+1,:],linewidth=1,zorder=zzs[experiment]+4,ls='--',label='_nolegend_')

#    plt.plot(years,had4_diag*1.1719758280521986,color='black',linewidth=1,label='Observations',zorder=8)
    plt.plot(obs_gsat['Year'],obs_gsat['GSAT estimate'],color='black',linewidth=1,label='Observations',zorder=8)
    plt.axis([1850,2020,-1.5,2.5])
    plt.xlabel('Year')
    plt.ylabel('Global surface air temperature change ($^\circ$C)')
    plt.legend(loc="upper left")



    plt.savefig('/home/rng/plots/esmvaltool/spm_fig.png')

    plt.close()
    with open('/home/rng/plots/esmvaltool/faq3-3.csv', mode='w') as file:
        data_writer=csv.writer(file,delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for experiment in range(0,4):
            data_writer.writerow([labels[experiment]])
            data_writer.writerow(['Year, Mean, Min, Max'])
            for yy in range(ldiag):
                data_writer.writerow([years[yy],mm_ann_warming[yy,experiment],range_ann_warming[yy,experiment,0],range_ann_warming[yy,experiment,1]])
        data_writer.writerow(['Observations'])
        for yy in range(ldiag):
            data_writer.writerow([years[yy],had4_diag[yy]*1.1719758280521986])
        
    print ('Ratio of GSAT to GMST warming',numpy.mean(mean_ann_warming[2010-1850:2020-1850,0,:])/numpy.mean(mean_diag[2010-1850:2020-1850,0,:]))
    print (numpy.polyfit(years[1970-1850:2015-1850],numpy.mean(mean_ann_warming[1970-1850:2015-1850,0,:],axis=1),1))
    print (numpy.polyfit(years[1970-1850:2015-1850],numpy.mean(mean_diag[1970-1850:2015-1850,0,:],axis=1),1))
    fit_gsat=numpy.polyfit(years[1970-1850:2015-1850],numpy.mean(mean_ann_warming[1970-1850:2015-1850,0,:],axis=1),1)
    fit_gmst=numpy.polyfit(years[1970-1850:2015-1850],numpy.mean(mean_diag[1970-1850:2015-1850,0,:],axis=1),1)
    print ('Ratio of trends:',fit_gsat[0]/fit_gmst[0])

    print ('ens_sizes',ens_sizes)    
if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
