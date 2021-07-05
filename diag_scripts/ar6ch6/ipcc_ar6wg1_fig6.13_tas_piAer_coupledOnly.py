
# ipcc_ar6wg1_fig6.13_tas_piAer_coupledOnly.py 

# Description
# Generates figures for IPCC Working Group I Contribution to the Sixth Assessment Report
#  IPCC AR6 WG1 Figure 6.13 

# Creator: Chaincy Kuo (CKuo@lbl.gov, chaincy.ipccwg1@gmail.com )
# Creation Date:  11 March 2021


import logging
import os
from pprint import pformat

import iris
from iris.experimental.equalise_cubes import equalise_attributes
from netCDF4 import Dataset

from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            select_metadata, sorted_metadata,Variables)
from esmvaltool.diag_scripts.shared._base import (
    ProvenanceLogger, get_diagnostic_filename, get_plot_filename)
from esmvaltool.diag_scripts.shared.plot import quickplot

import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import cmapipcc
import numpy as np
from scipy import stats
import cartopy.crs as ccrs

from ch6_fns import compute_multiModelStats, compute_multiModelDiffStats, get_signagreemap

logger = logging.getLogger(os.path.basename(__file__))

def get_areaweight(nlat,nlon,lat,lon):
	# determine the quandrangle area weight 

	areaweight = np.zeros((nlat,nlon),dtype=float);
	lat4wt = np.zeros((nlat,2),dtype=float);
	lon4wt = np.zeros((nlon,2),dtype=float);
	# assumption that cells are equally spaced., but this is not correct for latitudes
	dellon = 360.0/nlon;
	for ilat in range(0,nlat):
		if(ilat==0):
			dellat1 = (90.0-np.abs(lat[ilat]))/2;
		else:
			dellat1 = np.abs(lat[ilat]-lat[ilat-1])/2;
		if(ilat==nlat-1):
			dellat2 = (90.0-np.abs(lat[ilat]))/2;
		else:
			dellat2 = np.abs(lat[ilat+1]-lat[ilat])/2;

		lat4wt[ilat,0] = (lat[ilat]-dellat1)/180.0 * np.pi;
		lat4wt[ilat,1] = (lat[ilat]+dellat2)/180.0 * np.pi;
	for ilon in range(0,nlon):
		lon4wt[ilon,0] = (lon[ilon]-dellon/2)/180.0 * np.pi;
		lon4wt[ilon,1] = (lon[ilon]+dellon/2)/180.0 * np.pi;
	for ilat in range(0,nlat):
		for ilon in range(0,nlon):
			areaweight[ilat,ilon] = np.abs(lon4wt[ilon,1]-lon4wt[ilon,0]) * np.abs(np.sin(lat4wt[ilat,1])-np.sin(lat4wt[ilat,0])) / (4.0*np.pi);       
		
	np.sum(areaweight)
	return areaweight


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
            'kuo_chaincy',
        ],
        'references': [
            'acknow_project',
        ],
        'ancestors': ancestor_files,
    }
    return record

def checkstartend(selection):
	for attributes in selection:
		logger.info("Processing dataset %s", attributes['dataset'])
		input_file = attributes['filename']
		logger.info("Loading %s", input_file)
		# get start and end year.  all data should be the same, so do it once 
		#styear = ("{start_year}".format(**attributes))
		#endyear = ("{end_year}".format(**attributes))
		styear = attributes['start_year']
		endyear = attributes['end_year']
		logger.info("start %s ", styear)
		logger.info("end %s ", endyear)

#def compute_multiModelDiffStats(experiment, control):  
## historical is the experiment, and the pi-* exeriments control for specific species to pre-industrial levels
#
#    # compute differences between experiment and control for individual models
#    # input are cubes [time, nlat, nlon]
#	nmodels=0
#	for attributes in control:
#		logger.info("compute MM Processing dataset cntl %s", attributes['dataset'])
#		input_model = attributes['dataset']
#		input_filecntl = attributes['filename']
#		logger.info("opening %s",input_filecntl)
#		cubecntl=iris.load_cube(input_filecntl)
#		cshape=cubecntl.shape
#                # dims [ntime, nlat, nlon]
#		logger.info('cubecntl %d  %d %d ' % (cshape[0],cshape[1],cshape[2]))	
#		if(nmodels==0):
#			alltimedata = np.zeros((cshape[0],cshape[1],cshape[2]))
#		iflagexp = 0
#		for attributesexp in experiment:
#			if(attributesexp['dataset'] == input_model):
#				iflagexp = 1 # match found
#				logger.info("computeMM Processing dataset exp %s", attributesexp['dataset'])
#				input_fileexp = attributesexp['filename']
#				logger.info("opening %s",input_fileexp)
#				cubeexp=iris.load_cube(input_fileexp)
#				cshape=cubeexp.shape
#				logger.info('cubeexp %d  %d %d ' % (cshape[0],cshape[1],cshape[2]))	
#
#		#logger.info("computeMM iflagexp %d", iflagexp)
#		if(iflagexp == 1):
#			climateresponse= cubeexp.data - cubecntl.data 
#			if(nmodels==0):
#				alltimedata = climateresponse
#			else:
#                		# dims [nmodels x ntime, nlat, nlon]
#				alltimedata = np.append(alltimedata,climateresponse,axis=0)
#			nmodels=nmodels+1
#
#	alltimeavgcube = alltimedata.mean(axis=0)
#	alltimestdcube = alltimedata.std(axis=0)	
#
#	# set a cube for the metadata
#	timeavgcube = cubecntl.collapsed('time', iris.analysis.MEAN)
#	# replace with the data of interest
#	timeavgcube.data = alltimeavgcube
#	# set a cube for the metadata
#	timestdcube = cubecntl.collapsed('time', iris.analysis.VARIANCE)
#	# replace with the data of interest
#	timestdcube.data = alltimestdcube
#
#        # dims [nmodels x ntime, lat, lon]
#	return timeavgcube, alltimedata  


def plot_meanmap(cfg,cubeavg,exper,field):
    """
    Arguments:
        cube - the cube to plot

    Returns:

    """
    local_path = cfg['plot_dir']

    # coordinates
    Xplot1 = cubeavg.coord('longitude').points
    Yplot1 = cubeavg.coord('latitude').points
    Xplot, Yplot = np.meshgrid(Xplot1, Yplot1)

    # colomaps
    clevels=11
    cmapnow=cmapipcc.get_ipcc_cmap('precipitation',clevels)
    cmapip=ListedColormap(cmapnow)

    # do the plotting dance
#    plt.contourf(cube.data.mean(axis=0))
    #plt.imshow(cube.data.mean(axis=0))
    plotnow=cubeavg.data
#    dims = np.shape(plotnow)
#    print('size %f dim cube %f %f' % (plotnow.size,dims[0],dims[1]))
    nrows=1
    ncols=1

    fig = plt.figure(figsize=(ncols*7, nrows*5))
    ax = fig.add_subplot(nrows, ncols, 1, projection=ccrs.Robinson(central_longitude=180))
    im=ax.contourf(Xplot,Yplot,plotnow,clevels,cmap=cmapip, transform=ccrs.PlateCarree())
#   im=ax.contourf(plotnow,cmap=cmapip)
    ax.coastlines()
    ax.set_global()
    cb=fig.colorbar(im,fraction=0.05,pad=0.05,orientation='horizontal',format='%4.2f')
    if(field=='pr'):
	    cb.set_label(r' mm per day',fontsize=12)
	    plt.suptitle(r'Mean Surface precipitation',fontsize=16)
    elif(field=='tas'):
	    cb.set_label(r'$^{\circ}$C ',fontsize=12)
	    plt.suptitle(r'Mean near Surface Air Temperature',fontsize=16)
    #plt.title(dataset)
    plt.tight_layout()
    
    png_name = 'multimodelMean_%s_%s.png' % (field,exper)
    plt.savefig(os.path.join(local_path, png_name))
    plt.close()


def plot_diffmap(cfg,histavg,histall,histcntlavg,histcntall,field):
    """
    Arguments:
        cube - the cube to plot

    Returns:

    """
    local_path = cfg['plot_dir']
    # get hatch pattern
    cdims=histall.shape
    hatchnow = np.zeros((cdims[1],cdims[2]))
    hatchplot = np.zeros((cdims[1],cdims[2]))
    pthresh=0.050
    for ilat in range(0,cdims[1]):
        for ilon in range(0,cdims[2]):
            a = histall.data[:,ilat,ilon]
            b = histcntall.data[:,ilat,ilon]
		#ars=np.ma.masked_greater(np.resize(a,(cdims[0],)),10)
		#brs=np.ma.masked_greater(np.resize(b,(cdims[0],)),10)
            ars=np.resize(a,(cdims[0],))
            brs=np.resize(b,(cdims[0],))
            ttest = stats.ttest_ind(ars, brs, equal_var = False)
            hatchnow[ilat,ilon]=ttest[1]
	    #  ttest>pthresh means mark areas without significance
	    #  ttest<pthresh means mark significant areas 
            if(ttest[1]>pthresh):
                hatchplot[ilat,ilon] = 1

    

    # coordinates
    Xplot1 = histavg.coord('longitude').points
    Yplot1 = histavg.coord('latitude').points
    Xplot, Yplot = np.meshgrid(Xplot1, Yplot1)

    # colomaps
    clevels=11
    if(field=='pr'):
    	cmapnow=cmapipcc.get_ipcc_cmap('precipitation',clevels)
    	cmapip=ListedColormap((cmapnow))
    elif(field=='tas'):
    	cmapnow=cmapipcc.get_ipcc_cmap('temperature',clevels)
    	cmapip=ListedColormap(np.flipud(cmapnow))

#    plt.contourf(cube.data.mean(axis=0))
    #plt.imshow(cube.data.mean(axis=0))
     
    if(field=='pr'):
    	convert2mmperday = 1.e3*1e-4*10*3600*24   # kg/m^2/s to mm/day.   1.e3[g/kg]*1[cm^3/g]*1e-4*[m^2/cm^2]*10[mm/cm]*3600*24[s/day] 
    	plotnow=(histavg.data-histcntlavg.data)*convert2mmperday
    elif(field=='tas'):
    	plotnow=(histavg.data-histcntlavg.data)

    # cumulative sum
    plotflat=np.resize(plotnow,((cdims[1]*cdims[2]),))
    plothist= np.histogram(plotflat,bins=20)
    plotcum = np.cumsum(plothist[0][:])
    plotcumx = np.zeros((len(plotcum),))
    for i in range(0,len(plotcum)):
        plotcumx[i] = (plothist[1][i+1]+plothist[1][i])/2.0  # plothist is a lsit, not a tuple, so use this method of indexing


#    dims = np.shape(plotnow)
#    print('size %f dim cube %f %f' % (plotnow.size,dims[0],dims[1]))
    nrows=1
    ncols=1

    cmpcnt=0.95
    cmax = cmpcnt*np.max([np.abs(plotnow.max()),np.abs(plotnow.min())])
    cmin = -cmax
    cincr=2.0*cmax/float(clevels-1)
    crange=np.arange(cmin,cmax+cincr,cincr)

    fig = plt.figure(figsize=(ncols*7, nrows*5))
    ax = fig.add_subplot(nrows, ncols, 1, projection=ccrs.Robinson(central_longitude=180))
    im=ax.contourf(Xplot,Yplot,plotnow,crange,cmap=cmapip, transform=ccrs.PlateCarree())
    #imh=ax.contourf(Xplot,Yplot,hatchnow,hlevels,colors='none', hatch=['/','//','///','////','.','*',None,None,None,'/'], alpha=0.0)
    imh=ax.contourf(Xplot,Yplot,hatchplot,2,colors='none', hatches=[None,'xxxx','\\\\'], alpha=0.0, transform=ccrs.PlateCarree())
#    ax.pcolor(Xplot,Yplot,hatchnow, color='none',hatch='/', alpha=0.7)
    ax.coastlines()
    ax.set_global()
    cb=fig.colorbar(im,fraction=0.05,pad=0.05,orientation='horizontal',format='%4.2f')
    if(field=='pr'):
	    cb.set_label(r'(mm per day)',fontsize=12)
	    plt.suptitle(r'Surface precipitation',fontsize=16)
	    plt.title(r'Change in surface precip',fontsize=14)
    elif(field=='tas'):
	    cb.set_label(r'($^{\circ}$C) ',fontsize=12)
	    plt.suptitle(r'Change in Near Surface Air Temperature',fontsize=16)
	    plt.title(r'Change in near surface air temperature',fontsize=14)

    if(nrows==2):
            ax2 = fig.add_subplot(nrows, ncols, 2)
            ax2.plot(plotcumx,plotcum)
            if(field=='pr'):
                ax2.set_xlabel(r'(mm per day)',fontsize=12)
            elif(field=='tas'):
                ax2.set_xlabel(r'($^{\circ}$C)',fontsize=12)
                ax2.set_ylabel(r'Cumulative')

    plt.tight_layout()
    
    png_name = 'diff_%s.png' % field
    plt.savefig(os.path.join(local_path, png_name))
    plt.close()

def plot_fastslowresponses(cfg,fastslow,fastslowhatch,fast,fasthatch,field):
    """
    Arguments:
        cube - the cube to plot

    Returns:

    """
    local_path = cfg['plot_dir']
    # get hatch pattern

    # coordinates
    Xplot1 = histavg.coord('longitude').points
    Yplot1 = histavg.coord('latitude').points
    Xplot, Yplot = np.meshgrid(Xplot1, Yplot1)

    # colomaps
    clevels=11
    if(field=='pr'):
    	cmapnow=cmapipcc.get_ipcc_cmap('precipitation',clevels)
    	cmapip=ListedColormap((cmapnow))
    elif(field=='tas'):
    	cmapnow=cmapipcc.get_ipcc_cmap('temperature',clevels)
    	cmapip=ListedColormap(np.flipud(cmapnow))

#    plt.contourf(cube.data.mean(axis=0))
    #plt.imshow(cube.data.mean(axis=0))
     
    if(field=='pr'):
    	convert2mmperday = 1.e3*1e-4*10*3600*24   # kg/m^2/s to mm/day.   1.e3[g/kg]*1[cm^3/g]*1e-4*[m^2/cm^2]*10[mm/cm]*3600*24[s/day] 
    	plotnow=(histavg.data-histcntlavg.data)*convert2mmperday
    elif(field=='tas'):
    	plotnow=(histavg.data-histcntlavg.data)

    # cumulative sum
    plotflat=np.resize(plotnow,((cdims[1]*cdims[2]),))
    plothist= np.histogram(plotflat,bins=20)
    plotcum = np.cumsum(plothist[0][:])
    plotcumx = np.zeros((len(plotcum),))
    for i in range(0,len(plotcum)):
        plotcumx[i] = (plothist[1][i+1]+plothist[1][i])/2.0  # plothist is a lsit, not a tuple, so use this method of indexing


#    dims = np.shape(plotnow)
#    print('size %f dim cube %f %f' % (plotnow.size,dims[0],dims[1]))
    nrows=1
    ncols=1

    cmpcnt=0.95
    cmax = cmpcnt*np.max([np.abs(plotnow.max()),np.abs(plotnow.min())])
    cmin = -cmax
    cincr=2.0*cmax/float(clevels-1)
    crange=np.arange(cmin,cmax+cincr,cincr)

    fig = plt.figure(figsize=(ncols*7, nrows*5))
    ax = fig.add_subplot(nrows, ncols, 1, projection=ccrs.Robinson(central_longitude=180))
    im=ax.contourf(Xplot,Yplot,plotnow,crange,cmap=cmapip, transform=ccrs.PlateCarree())
    #imh=ax.contourf(Xplot,Yplot,hatchnow,hlevels,colors='none', hatch=['/','//','///','////','.','*',None,None,None,'/'], alpha=0.0)
    imh=ax.contourf(Xplot,Yplot,hatchplot,2,colors='none', hatches=[None,'xxxx','\\\\'],alpha=0.0, transform=ccrs.PlateCarree())
#    ax.pcolor(Xplot,Yplot,hatchnow, color='none',hatch='/', alpha=0.7)
    ax.coastlines()
    ax.set_global()
    cb=fig.colorbar(im,fraction=0.05,pad=0.05,orientation='horizontal',format='%4.2f')
    if(field=='pr'):
	    cb.set_label(r'(mm per day)',fontsize=12)
	    plt.suptitle(r'Surface precipitation',fontsize=16)
	    plt.title(r'Change in surface precip',fontsize=14)
    elif(field=='tas'):
	    cb.set_label(r'($^{\circ}$C) ',fontsize=12)
	    plt.suptitle(r'Change in Near Surface Air Temperature',fontsize=16)
	    plt.title(r'Change in near surface air temperature',fontsize=14)

    if(nrows==2):
            ax2 = fig.add_subplot(nrows, ncols, 2)
            ax2.plot(plotcumx,plotcum)
            if(field=='pr'):
                ax2.set_xlabel(r'(mm per day)',fontsize=12)
            elif(field=='tas'):
                ax2.set_xlabel(r'($^{\circ}$C) ',fontsize=12)
                ax2.set_ylabel(r'Cumulative')

    plt.tight_layout()
    
    png_name = 'fastslowresp_%s.png' % field
    plt.savefig(os.path.join(local_path, png_name))
    plt.close()

def get_diffmap(exptavg,exptall,cntlavg,cntlall):
    """
    Arguments:
	*avg are iris cubes
	*all are numpy array dims[nmodels x ntime,nlat,nlon] 

    Returns:
	

    """
    plotnow=(exptavg.data-cntlavg.data)
    # get hatch pattern
    cdims=exptall.shape
    hatchnow = np.zeros((cdims[1],cdims[2]))
    hatchplot = np.zeros((cdims[1],cdims[2]))
    pthresh=0.050
    for ilat in range(0,cdims[1]):
        for ilon in range(0,cdims[2]):
            #a = exptall.data[:,ilat,ilon]
            #b = cntlall.data[:,ilat,ilon]
            a = exptall[:,ilat,ilon]
            b = cntlall[:,ilat,ilon]
            ars=np.resize(a,(cdims[0],))
            brs=np.resize(b,(cdims[0],))
            ttest = stats.ttest_ind(ars, brs, equal_var = False)
            hatchnow[ilat,ilon]=ttest[1]
	    #  ttest>pthresh means mark areas without significance
	    #  ttest<pthresh means mark significant areas 
            if(ttest[1]>pthresh):
                hatchplot[ilat,ilon] = 1


    return plotnow,hatchplot
    
def get_slowmap(histexp,histcntl,histSSTexp,histSSTcntl,histexpall,histcntlall,histSSTexpall,histSSTcntlall):

    """
    Arguments:
    Arguments:
	*avg are iris cubes
	*all are numpy array dims[nmodels x ntime,nlat,nlon] 

    Returns:
	

    """
    plotnow1=(histexp.data-histcntl.data) 
    plotnow2=(histSSTexp.data-histSSTcntl.data)
    plotnow = plotnow1-plotnow2
    # get hatch pattern
    cdims=histexpall.shape
    hatchnow = np.zeros((cdims[1],cdims[2]))
    hatchplot = np.zeros((cdims[1],cdims[2]))
    pthresh=0.050
    for ilat in range(0,cdims[1]):
        for ilon in range(0,cdims[2]):
            a = histexpall[:,ilat,ilon] - histcntlall[:,ilat,ilon]
            b = histSSTexpall[:,ilat,ilon] - histSSTcntlall[:,ilat,ilon]
            ars=np.resize(a,(cdims[0],))
            brs=np.resize(b,(cdims[0],))
            ttest = stats.ttest_ind(ars, brs, equal_var = False)
            hatchnow[ilat,ilon]=ttest[1]
	    #  ttest>pthresh means mark areas without significance
	    #  ttest<pthresh means mark significant areas 
            if(ttest[1]>pthresh):
                hatchplot[ilat,ilon] = 1

    return plotnow,hatchplot


def run_some_diagnostic( experexpts):
	"""Diagnostic to be run."""

def main(cfg):
	"""Main diagnostic run function."""
	input_data = cfg['input_data'].values()
	variables = Variables(cfg)
	allvars = variables.short_names()

	# prepare figure matrix
#	fig = plt.figure(constrained_layout=True)
#	spec = gridspec.GridSpec(ncols=16, nrows=20, figure=fig)
	nrows=1
	ncols=2
	fig = plt.figure(figsize=(ncols*6.5, nrows*5))
	fig.subplots_adjust(top=0.6,hspace=0.3)
	#plt.suptitle('Climate Impacts of Ozone Forcing',fontsize=28)
	plt.rcParams.update({'hatch.color': 'grey'})	
	
 	# make scientific notation as powers of 10
	fp10 = mticker.ScalarFormatter(useOffset=False, useMathText=True)
	gp10 = lambda x,pos : "${}$".format(fp10._formatSciNotation('%1.1e' % x))
	fmt = mticker.FuncFormatter(gp10)         

	#for ifield in range(0,len(allvars)):
	ifield=0
	if(allvars[ifield] == 'tas'):

		histexp='historical'
		select_historical= select_metadata(input_data, exp=histexp, short_name=allvars[ifield])
		logger.info("Example of how to select only historical data:\n%s",
			pformat(select_historical))
		checkstartend(select_historical)
		#[hist_mdlmn,hist_mdlsd,hist_mdlall]=compute_multiModelStats(select_historical)
		[hist_mdlmn,hist_mdlall,hist_mdls,prov_rec]=compute_multiModelStats(cfg,select_historical)
		plot_meanmap(cfg,hist_mdlmn,histexp,allvars[ifield])

		histcntl='hist-piAer'
		select_histpiAer= select_metadata(input_data,exp=histcntl, short_name=allvars[ifield])
		logger.info("Example of how to select only hist-piAer data:\n%s",
			pformat(select_histpiAer))
		checkstartend(select_histpiAer)
		#[histcntl_mdlmn,histcntl_mdlsd,histcntl_mdlall]=compute_multiModelStats(select_histpiAer)
		[histcntl_mdlmn,histcntl_mdlall,histcntl_mdls,prov_rec]=compute_multiModelStats(cfg,select_histpiAer)
		plot_meanmap(cfg,histcntl_mdlmn,histcntl,allvars[ifield])
		#run_some_diagnostic(experexpts)
		[fastslow,fastslowhatch]=get_diffmap(hist_mdlmn,hist_mdlall,histcntl_mdlmn,histcntl_mdlall)

		[fastslowhatch, sigthresh] = get_signagreemap(hist_mdls,histcntl_mdls)

		histSSTexp='histSST'
		select_histSST= select_metadata(input_data, exp=histSSTexp, short_name=allvars[ifield])
		logger.info("Example of how to select only histSST data:\n%s",
			pformat(select_histSST))
		checkstartend(select_histSST)
		#[histSST_mdlmn,histSST_mdlsd,histSST_mdlall]=compute_multiModelStats(select_histSST)
		[histSST_mdlmn,histSST_mdlall,histSST_mdls,prov_rec]=compute_multiModelStats(cfg,select_histSST)
		plot_meanmap(cfg,histSST_mdlmn,histSSTexp,allvars[ifield])

		histSSTcntl='histSST-piAer'
		select_histSSTpiAer= select_metadata(input_data,exp=histSSTcntl, short_name=allvars[ifield])
		logger.info("Example of how to select only histSST-piAer data:\n%s",
			pformat(select_histSSTpiAer))
		checkstartend(select_histSSTpiAer)
		#[histSSTcntl_mdlmn,histSSTcntl_mdlsd,histSSTcntl_mdlall]=compute_multiModelStats(select_histSSTpiAer)
		[histSSTcntl_mdlmn,histSSTcntl_mdlall,histSSTcntl_mdls,prov_rec]=compute_multiModelStats(cfg,select_histSSTpiAer)
		plot_meanmap(cfg,histSSTcntl_mdlmn,histSSTcntl,allvars[ifield])

#		[fast,fasthatch]=get_diffmap(histSST_mdlmn,histSST_mdlall,histSSTcntl_mdlmn,histSSTcntl_mdlall)
#		[fasthatch, sigthresh]  = get_signagreemap(histSST_mdls,histSSTcntl_mdls)

#		[slow,slowhatch]=get_slowmap(hist_mdlmn,histcntl_mdlmn,histSST_mdlmn,histSSTcntl_mdlmn,
#			hist_mdlall,histcntl_mdlall,histSST_mdlall,histSSTcntl_mdlall)	

		#plot_meanmap(cfg,fastslowAeravg,histcntlAer,allvars[ifield])
		#plot_meanmap(cfg,fastslowNTCFavg,histcntlNTCF,allvars[ifield])
		#plot_meanmap(cfg,fastAeravg,histSSTcntlAer,allvars[ifield])
		#plot_meanmap(cfg,fastNTCFavg,histSSTcntlNTCF,allvars[ifield])

		# prep for plotting
		repcube = hist_mdlmn# respresentative cube
		Xplot1 = repcube.coord('longitude').points
		Yplot1 = repcube.coord('latitude').points
		Xplot, Yplot = np.meshgrid(Xplot1, Yplot1)


		awt = get_areaweight(len(Yplot1),len(Xplot1),Yplot1,Xplot1)

		# colormaps
		clevels=14
		if(allvars[ifield]=='pr'):
			cmapnow=cmapipcc.get_ipcc_cmap('precipitation',clevels)
			cmapip=ListedColormap((cmapnow))
			convert2mmperday = 1.e3*1e-4*10*3600*24   # kg/m^2/s to mm/day.   1.e3[g/kg]*1[cm^3/g]*1e-4*[m^2/cm^2]*10[mm/cm]*3600*24[s/day] 
			fastslow = fastslow*convert2mmperday
			fast     = fast*convert2mmperday
			slow     = slow*convert2mmperday
		elif(allvars[ifield]=='tas'):
			cmapnow=cmapipcc.get_ipcc_cmap('temperature',clevels)
			if(clevels<=11):
				cmapip=ListedColormap(np.flipud(cmapnow))
			else:
				cmapip=ListedColormap(cmapnow)

		
		cmpcnt=1
		cmax = cmpcnt*np.max([np.abs(fastslow.max()),np.abs(fastslow.min())])
		#if(allvars[ifield]=='pr'):
		#	# put on the same scale as for piNTCF plots
		#	cmax = 1.125 
		#elif(allvars[ifield]=='tas'):
			# put on the same scale as for piNTCF plots
		#	cmax = 2.4675
		cmin = -cmax
		cincr=2.0*cmax/float(clevels-1)
		cincr = 0.7
		cmin = -clevels/2*cincr
		cmax = -cmin
		crange=np.arange(cmin,cmax+cincr,cincr)

		# latitudinal profiles
		fastslowprof = np.zeros((len(Yplot1),))
		fastprof = np.zeros((len(Yplot1),))
		slowprof = np.zeros((len(Yplot1),))
		fastslowstd = np.zeros((len(Yplot1),))
		faststd = np.zeros((len(Yplot1),))
		slowstd = np.zeros((len(Yplot1),))
		for ilat in range(0,len(Yplot1)):
			fastslowprof[ilat] = np.sum(fastslow[ilat,:]*awt[ilat,:])/np.sum(awt[ilat,:])
			fastslowstd[ilat] = np.sqrt((float(len(Yplot1)/(len(Yplot1)-1)))*np.sum(np.power(fastslow[ilat,:]-fastslowprof[ilat],2)*awt[ilat,:])/np.sum(awt[ilat,:]))

		lateq = int(len(Yplot1)/2)
		#fastslow		
		plttype =''# 'fast + slow'# = [%s-%s]-[%s-%s]' % (histexp,histcntlNTCF,histexp,histcntlAer)	
		#ax = fig.add_subplot(spec[0:8,8:], projection=ccrs.Robinson(central_longitude=0))
		ax = fig.add_subplot(nrows, ncols, 1, projection=ccrs.Robinson(central_longitude=0))
		im=ax.contourf(Xplot,Yplot,fastslow,crange,cmap=cmapip, transform=ccrs.PlateCarree(),extend='both')
		imh=ax.contourf(Xplot,Yplot,fastslowhatch,2,colors='none', edgecolor=None, hatches=[None,'xxxx','\\\\\\'], alpha=0.0, transform=ccrs.PlateCarree())
		bbox = dict(boxstyle="square", fc="1.0")
		artists,labels = imh.legend_elements()
		lgd=ax.legend(artists,labels, handleheight=1.0, loc='center', bbox_to_anchor=(0.5,-0.5),fontsize=12  )
		lgd.get_texts()[0].set_text('Robust change')
		lgd.get_texts()[1].set_text('Conflicting signals')
		lgd.get_texts()[2].set_text('No change or no robust change')
		#bbox = dict(boxstyle="square", fc="1.0")
		#bbox = dict(boxstyle="square", fc="1.0")
		#plt.annotate('\\\\\\\\', xy=(0.225,-0.4), xycoords='axes fraction',color='black',fontsize=10,bbox=bbox)
		#plt.annotate(r' Lack of 2-$\sigma$ model significance and', xy=(0.275,-0.4), xycoords='axes fraction',color='black',fontsize=10)
		#plt.annotate(r' Lack of model agreement (threshold: 80% )', xy=(0.275,-0.45), xycoords='axes fraction',color='black',fontsize=10)

		fastslowglobal = np.sum(awt*fastslow)/np.sum(awt)
		fastslowSH = np.sum(awt[0:lateq,:]*fastslow[0:lateq,:])/np.sum(awt[0:lateq,:])
		fastslowNH = np.sum(awt[lateq:len(Yplot1),:]*fastslow[lateq:len(Yplot1),:])/np.sum(awt[lateq:len(Yplot1),:])
		fastslowglobal_std = np.sqrt((float(len(Yplot1)/(len(Yplot1)-1)))*np.sum(np.power((fastslow-fastslowglobal),2)*awt)/np.sum(awt))
		fastslowSH_std = np.sqrt((float(len(Yplot1)/(len(Yplot1)-1)))*np.sum(np.power(fastslow[0:lateq,:]-fastslowSH,2)*awt[0:lateq,:])/np.sum(awt[0:lateq,:]))
		fastslowNH_std = np.sqrt((float(len(Yplot1)/(len(Yplot1)-1)))*np.sum(np.power(fastslow[lateq:len(Yplot1),:]-fastslowNH,2)*awt[lateq:len(Yplot1),:])/np.sum(awt[lateq:len(Yplot1),:]))
				

		ax.coastlines()
		ax.set_global()
		cb=fig.colorbar(im,fraction=0.05,pad=0.05,orientation='horizontal',format='%4.2f')
		if(allvars[ifield]=='pr'):
			maptitle='Change in Surface Precipitation'+'\n'+'%s' % plttype
			plt.annotate('a)', xy=(0., 1.03), xycoords='axes fraction',fontsize=8)
			ax.set_ylabel(r'Change in surface precipitation',fontsize=10)
			cblabel='(mm per day)'
		elif(allvars[ifield]=='tas'):
			maptitle='Change in Surface Air Temperature'+'\n'+'%s' % plttype
			plt.annotate('a)', xy=(0., 1.03), xycoords='axes fraction',fontsize=8*2)
			ax.set_ylabel(r'Change in Near Surface Air Temperature',fontsize=10)
			cblabel='($^{\circ}$C) '
		cb.set_label(r'%s' % cblabel,fontsize=8*2)
		cb.ax.tick_params(labelsize=5*2)	
		fastslowmeanlabel=r'Global %4.1e $\pm$ %4.1e''\n''NH %4.1e $\pm$ %4.1e''\n''SH %4.1e $\pm$ %4.1e ' % (fastslowglobal,fastslowglobal_std,fastslowNH,fastslowNH_std,fastslowSH,fastslowSH_std)
		fastslowgblfmt = r'Global {}'.format(fmt(fastslowglobal)) + '$\pm${}'.format(fmt(fastslowglobal_std))
		fastslowNHfmt = r'NH {}'.format(fmt(fastslowNH)) + '$\pm${}'.format(fmt(fastslowNH_std))
		fastslowSHfmt = r'SH {}'.format(fmt(fastslowSH)) + '$\pm${}'.format(fmt(fastslowSH_std))
		fastslowmeanlabel = '%s\n%s\n%s\n' % (fastslowgblfmt,fastslowNHfmt,fastslowSHfmt)
		#plt.title(r'%s'% (plttype),fontsize=10)
		plt.title(r'Surface Air Temperature Response''\n''due to Aerosols',fontsize=8*2)


		# profiles
		prfmax = np.max([np.abs(fastslowprof)+np.max(fastslowstd)])
		if(allvars[ifield]=='pr'):
			# put on the same scale as for piNTCF plots
			prfmax = 0.65 
		elif(allvars[ifield]=='tas'):
			# put on the same scale as for piNTCF plots
			prfmax = 3.75
	

		ax = fig.add_subplot(nrows, ncols, 2)
		#lines, caps, bars = ax.errorbar(fastslowprof,Yplot1,fmt='k',linewidth=3,xerr=fastslowstd,ecolor='lightgrey',errorevery=3,label=r'fast+slow' )
		lines, caps, bars = ax.errorbar(fastslowprof,Yplot1,fmt='k',linewidth=3,xerr=fastslowstd,ecolor='lightgrey',errorevery=1) #label=r'fast+slow' )
		# loop through bars and caps and set the alpha value
		[bar.set_alpha(0.2) for bar in bars	]
		ax.plot(fastslowprof+fastslowstd,Yplot1,'k',linewidth=0.2, alpha=0.3 )
		ax.plot(fastslowprof-fastslowstd,Yplot1,'k',linewidth=0.2, alpha=0.3 )
		ax.plot(np.zeros((len(Yplot1),)),Yplot1,'k--',linewidth=1, alpha=1.0 )
		ax.set_xlim(-prfmax,prfmax)

		lblunits=' '
		if(allvars[ifield]=='pr'):
			plt.annotate('d)', xy=(0., 1.03), xycoords='axes fraction',fontsize=8)
			proftitle='Change in Surface Precipitation'
			lblunits=r'(mm per day)'
		elif(allvars[ifield]=='tas'):
			plt.annotate('b)', xy=(0., 1.03), xycoords='axes fraction',fontsize=8*2)
			proftitle=r'Zonal Mean Change in Surface Air Temperature''\n'' due to Aerosols'
			lblunits=r'$^{\circ}$C '
		ax.set_title(proftitle,fontsize=8*2)
		ax.set_xlabel(r'(%s)' % lblunits,fontsize=8*2)
		ax.set_ylabel(r'latitude ($^{\circ}$N)',fontsize=8*2)
		ax.tick_params(axis='both', which='major', labelsize=6*2)

		mnlblsz=10
		#ax.text(0.10*prfmax,10,r'$\mathbf{fast+slow}$ [%s]''\n''%s '% (lblunits,fastslowmeanlabel),fontsize=mnlblsz)
		ax.text(0.10*prfmax,10,r'$\mathbf{Means:}$ [%s]''\n\n''%s '% (lblunits,fastslowmeanlabel),fontsize=mnlblsz)
		#ax.legend(loc=3,fontsize=6)


	plt.tight_layout()
	local_path = cfg['plot_dir']
	png_name = 'fig6_fastslowOnly_piAer_signagree_tas.png'  
	plt.savefig(os.path.join(local_path, png_name),dpi=300)
	svg_name = 'fig6_fastslowOnly_piAer_signagree_tas.svg'  
	plt.savefig(os.path.join(local_path, svg_name),dpi=300)
	plt.close()


if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
