import logging
import os
from pprint import pformat

import iris
from iris.experimental.equalise_cubes import equalise_attributes
import netCDF4
from netCDF4 import Dataset

from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            select_metadata, sorted_metadata,Variables)
from esmvaltool.diag_scripts.shared._base import (
    ProvenanceLogger, get_diagnostic_filename, get_plot_filename)
from esmvaltool.diag_scripts.shared.plot import quickplot

import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cmapipcc
import numpy as np
from scipy import stats
import cartopy.crs as ccrs

from ch6_fns import compute_multiModelStats, compute_multiModelDiffStats, get_signagreemap, get_totalERFmap

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
               "according to {dataset}".format(**attributes))

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

##def compute_multiModelStats(cfg,selection):
#	# output:
#	# 	timeavgcube,  time-averaged iris cube
#	#	alltimedata,  all time data as iris cube
#	#	alltimemodels, all time data as numpy array
#	#	provenance_record 
#	nmodels=0
#	for attributes in selection:
#		input_file = attributes['filename']
#		logger.info("opening %s",input_file)
#		cube=iris.load_cube(input_file)
#		cshape=cube.shape
#		logger.info('cube %d  %d %d ' % (cshape[0],cshape[1],cshape[2]))	
#		if(nmodels==0):
#			alltimedata=cube.data
#		else:
#			alltimedata = np.append(alltimedata,cube.data,axis=0)
#		nmodels=nmodels+1
#
#		provenance_record = get_provenance_record(
#                        attributes, ancestor_files=[input_file])
#
#		output_basename = os.path.splitext(os.path.basename(input_file))[0]  
#		diagnostic_file = get_diagnostic_filename(output_basename, cfg)
#                #diagnostic_file = os.path.join(cfg['preproc_dir'],basename + '.' + extension,
#		with ProvenanceLogger(cfg) as provenance_logger:
#			provenance_logger.log(input_file,provenance_record)
#			provenance_logger.log(diagnostic_file,provenance_record)
#
#	alltimeavgcube = alltimedata.mean(axis=0)
#	alltimestdcube = alltimedata.std(axis=0)	
#
#	timeavgcube = cube.collapsed('time', iris.analysis.MEAN)
#	timeavgcube.data = alltimeavgcube
#	timestdcube = cube.collapsed('time', iris.analysis.VARIANCE)
#	timestdcube.data = alltimestdcube
#
#	alltimemodels= np.zeros((nmodels,cshape[0],cshape[1],cshape[2]))
#	for imdl in range(0,nmodels):
#		alltimemodels[imdl,:,:,:] = alltimedata[imdl*cshape[0]:(imdl+1)*cshape[0],:,:]
#
#	return timeavgcube, alltimedata, alltimemodels, provenance_record 


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
    cmapnow=cmapipcc.get_ipcc_cmap('temperature',clevels)
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
    cb=fig.colorbar(im,fraction=0.05,pad=0.05,orientation='horizontal')
    if(field=='rsut'):
	    cb.set_label(r'W-m$^{-1}$ ',fontsize=12)
	    plt.suptitle(r'Shortwave Effective Radiative Forcing',fontsize=16)
    elif(field=='rlut'):
	    cb.set_label(r'W-m$^{-1}$ ',fontsize=12)
	    plt.suptitle(r'Longwave Effective Radiative Forcing',fontsize=16)
    #plt.title(dataset)
    plt.tight_layout()
    
    png_name = 'multimodelMean_%s_%s.png' % (field,exper)
    plt.savefig(os.path.join(local_path, png_name))
    plt.close()


def plot_ch4erfresponses(cfg,ch4erf,ch4erfhatch,aererf,aererfhatch,field):
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
    cmapnow=cmapipcc.get_ipcc_cmap('temperature',clevels)
    cmapip=ListedColormap(np.flipud(cmapnow))

#    plt.contourf(cube.data.mean(axis=0))
    #plt.imshow(cube.data.mean(axis=0))
     
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

    cmax = np.max([np.abs(plotnow.max()),np.abs(plotnow.min())])
    cmin = -cmax
    cincr=2.0*cmax/float(clevels-1)
    crange=np.arange(cmin,cmax+cincr,cincr)

    fig = plt.figure(figsize=(ncols*7, nrows*5))
    ax = fig.add_subplot(nrows, ncols, 1, projection=ccrs.Robinson(central_longitude=180))
    im=ax.contourf(Xplot,Yplot,plotnow,crange,cmap=cmapip, transform=ccrs.PlateCarree())
    #imh=ax.contourf(Xplot,Yplot,hatchnow,hlevels,colors='none', hatch=['/','//','///','////','.','*',None,None,None,'/'], alpha=0.0)
    imh=ax.contourf(Xplot,Yplot,hatchplot,3,colors='none', hatches=['','xx','\\\\'], alpha=0.0, transform=ccrs.PlateCarree())
#    ax.pcolor(Xplot,Yplot,hatchnow, color='none',hatch='/', alpha=0.7)
    ax.coastlines()
    ax.set_global()
    cb=fig.colorbar(im,fraction=0.05,pad=0.05,orientation='horizontal')
    cb.set_label(r'W-m$^{-1}$ ',fontsize=12)
    if(field=='rsut'):
	    plt.suptitle(r'Shortwave Effective Radiative Forcing',fontsize=16)
	    plt.title(r'',fontsize=14)
    elif(field=='rlut'):
	    plt.suptitle(r'Longwave Effective Radiative Forcing',fontsize=16)
	    plt.title(r'',fontsize=14)

    if(nrows==2):
            ax2 = fig.add_subplot(nrows, ncols, 2)
            ax2.plot(plotcumx,plotcum)
            ax2.set_xlabel(r'W-m$^{-1}$ ',fontsize=12)
            ax2.set_ylabel(r'Cumulative')

    plt.tight_layout()
    
    png_name = 'diff_%s.png' % field
    plt.savefig(os.path.join(local_path, png_name))
    plt.close()

def get_erfmap(exptavg,exptall,cntlavg,cntlall):
    """
    Arguments:
        cube - the cubes to plot

    Returns:
	

    """

    plotnow=(exptavg.data-cntlavg.data)
    # ERF sign convention
    plotnow = -plotnow   

    # get hatch pattern
    cdims=exptall.shape
    hatchnow = np.zeros((cdims[1],cdims[2]))
    hatchplot = np.zeros((cdims[1],cdims[2]))
    for ilat in range(0,cdims[1]):
        for ilon in range(0,cdims[2]):
            a = exptall.data[:,ilat,ilon]
            b = cntlall.data[:,ilat,ilon]
            ars=np.resize(a,(cdims[0],))
            brs=np.resize(b,(cdims[0],))

            varthresh = np.sqrt(2)*1.645*b.std() 
	    # FGD guidelines on hatching threshold. hatch were not significant 
            if( np.abs(a.mean()) < (np.abs(b.mean())+varthresh)):		
                hatchplot[ilat,ilon] = 1   

    return plotnow,hatchplot

    
def writenetcdf_erfhatch(cfg,field,lat4wrt,lon4wrt,erf4wrt,hatch4wrt):
    local_path = cfg['plot_dir']
    nlat = lat4wrt.size	
    nlon = lon4wrt.size
   
    outfname = 'erf_hatch_%s.nc'  % field
    ncfile = netCDF4.Dataset(os.path.join(local_path,outfname), mode='w',format='NETCDF4_CLASSIC') 
    lat_dim = ncfile.createDimension('lat', nlat) # latitude axis
    lon_dim = ncfile.createDimension('lon', nlon) # longitude axis
    time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    time = ncfile.createVariable('time', np.float32, ('time',))
    time.units = 'one time point. Mean over models and time period'
    time.long_name = 'time'
    erf = ncfile.createVariable('erf_mn',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    erf.units = 'W/m^2' # 
    erf.standard_name = 'ERF models mean' # this is a CF standard name
    sig = ncfile.createVariable('significant',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    sig.units = 'truefalse' # 
    sig.standard_name = 'T-test significance p-val<0.05' # this is a CF standard name
    
    lat[:]= lat4wrt
    lon[:] = lon4wrt
    for ilat in range(0,nlat):
       	for ilon in range(0,nlon):	
            erf[0,ilat,ilon] = erf4wrt[ilat,ilon] 	
            sig[0,ilat,ilon] = hatch4wrt[ilat,ilon] 
    
    ncfile.close()

def run_some_diagnostic( experexpts):
	"""Diagnostic to be run."""

def main(cfg):
	"""Main diagnostic run function."""
	input_data = cfg['input_data'].values()
	variables = Variables(cfg)
	allvars = variables.short_names()

	# prepare figure matrix
	#nrows=3
	nrows=1
	ncols=1
	fig = plt.figure(figsize=(ncols*7, nrows*5))
	fig.subplots_adjust(hspace=0.3)
	plt.rcParams.update({'hatch.color': 'grey'})	
	
#	for ifield in range(0,len(allvars)):

	histSSTexp='histSST'
	select_histSSTSW= select_metadata(input_data, exp=histSSTexp, short_name='rsut')
	select_histSSTLW= select_metadata(input_data, exp=histSSTexp, short_name='rlut')
	logger.info("Example of how to select only histSST data:\n%s",
		pformat(select_histSSTSW))
	checkstartend(select_histSSTSW)
	logger.info("Example of how to select only histSST data:\n%s",
		pformat(select_histSSTLW))
	checkstartend(select_histSSTLW)
	[histSST_mdlmnSW,histSST_mdlallSW,histSST_mdlsSW,prov_rec]=compute_multiModelStats(cfg,select_histSSTSW)
	[histSST_mdlmnLW,histSST_mdlallLW,histSST_mdlsLW,prov_rec]=compute_multiModelStats(cfg,select_histSSTLW)
	plot_meanmap(cfg,histSST_mdlmnLW,histSSTexp,'rlut')
	plot_meanmap(cfg,histSST_mdlmnSW,histSSTexp,'rsut')

	histSSTaer='histSST-piAer'
	select_histSSTpiAerSW= select_metadata(input_data,exp=histSSTaer, short_name='rsut')
	select_histSSTpiAerLW= select_metadata(input_data,exp=histSSTaer, short_name='rlut')
	logger.info("Example of how to select only histSST-piAer data:\n%s",
		pformat(select_histSSTpiAerSW))
	checkstartend(select_histSSTpiAerSW)
	logger.info("Example of how to select only histSST-piAer data:\n%s",
		pformat(select_histSSTpiAerLW))
	checkstartend(select_histSSTpiAerLW)
	[histSSTaer_mdlmnLW,histSSTaer_mdlallLW,histSSTaer_mdlsLW,prov_rec]=compute_multiModelStats(cfg,select_histSSTpiAerLW)
	[histSSTaer_mdlmnSW,histSSTaer_mdlallSW,histSSTaer_mdlsSW,prov_rec]=compute_multiModelStats(cfg,select_histSSTpiAerSW)
	plot_meanmap(cfg,histSSTaer_mdlmnLW,histSSTaer,'rlut')
	plot_meanmap(cfg,histSSTaer_mdlmnSW,histSSTaer,'rsut')


	[aererfSW,aererfhatchSW]=get_erfmap(histSST_mdlmnSW,histSST_mdlallSW,histSSTaer_mdlmnSW,histSSTaer_mdlallSW)
	[aererfhatchSW, signagreethreshSW]  = get_signagreemap(histSSTaer_mdlsSW,histSST_mdlsSW)

	[aererfLW,aererfhatchLW]=get_erfmap(histSST_mdlmnLW,histSST_mdlallLW,histSSTaer_mdlmnLW,histSSTaer_mdlallLW)
	[aererfhatchLW, signagreethreshLW]  = get_signagreemap(histSSTaer_mdlsLW,histSST_mdlsLW)

	[aererfTOT,aererfhatchTOT]=get_totalERFmap(histSST_mdlmnLW,histSST_mdlmnSW,histSST_mdlallLW,histSST_mdlallSW,  
			histSSTaer_mdlmnLW,histSSTaer_mdlmnSW,histSSTaer_mdlallLW,histSSTaer_mdlallSW)
	[aererfhatchTOT, signagreethreshTOT]  = get_signagreemap(histSSTaer_mdlsSW+histSSTaer_mdlsLW,histSST_mdlsSW+histSST_mdlsLW)

	#aererfhatchSW=get_signagreemap(histSST_mdlsSW,histSSTaer_mdlsSW)
	#aererfhatchLW=get_signagreemap(histSST_mdlsLW,histSSTaer_mdlsLW)
	

	# prep for plotting
	repcube = histSST_mdlmnSW		 # respresentative cube
	Xplot1 = repcube.coord('longitude').points
	Yplot1 = repcube.coord('latitude').points
	Xplot, Yplot = np.meshgrid(Xplot1, Yplot1)

	# output erf and hatching
	writenetcdf_erfhatch(cfg,'SW',Yplot1,Xplot1,aererfSW,aererfhatchSW)
	writenetcdf_erfhatch(cfg,'LW',Yplot1,Xplot1,aererfLW,aererfhatchLW)

	awt = get_areaweight(len(Yplot1),len(Xplot1),Yplot1,Xplot1)

	# colormaps
	clevels=14
	cmapnow=cmapipcc.get_ipcc_cmap('temperature',clevels)
	if(clevels<=11):
		cmapip=ListedColormap(np.flipud(cmapnow))
	else:
		cmapip=ListedColormap(cmapnow)


	#piAer
	plttype =''# 'Aer  = [%s]-[%s]' % (histSSTexp,histSSTaer)	

	cmax = np.max([np.abs(aererfSW.max()),np.abs(aererfSW.min())])
	cmax = 10.5
	cmin = -cmax
	cincr=2.0*cmax/float(clevels)
	crange=np.arange(cmin,cmax+cincr,cincr)
	ax = fig.add_subplot(nrows, ncols, 1 , projection=ccrs.Robinson(central_longitude=0))
	#im=ax.contourf(Xplot,Yplot,aererfSW+aererfLW,crange,cmap=cmapip, transform=ccrs.PlateCarree())
	im=ax.contourf(Xplot,Yplot,aererfTOT,crange,cmap=cmapip, transform=ccrs.PlateCarree(),extend='both')
	imh=ax.contourf(Xplot,Yplot,aererfhatchTOT,2,colors='none', hatches=[None,'xx','\\\\'], alpha=0.0, transform=ccrs.PlateCarree())
	ax.coastlines()
	ax.set_global()
	cb=fig.colorbar(im,fraction=0.05,pad=0.05,orientation='horizontal',ticks=np.arange(cmin,cmax+cincr,2*cincr))
	cblabel=r'(W-m$^{-2}$)'
	maptitle='Net Effective Radiative Forcing'+'\n'+'Aerosols %s' % plttype
	plt.annotate('a)', xy=(0., 0.9), xycoords='axes fraction',fontsize=14)
	cb.set_label(r'%s' % cblabel,fontsize=12)
	plt.title(maptitle,fontsize=16)
	bbox = dict(boxstyle="square", fc="1.0")
	artists,labels = imh.legend_elements()
	lgd=ax.legend(artists,labels, handleheight=1.2, loc='upper center', bbox_to_anchor=(0.5,-0.25),fontsize=9  )
	lgd.get_texts()[0].set_text('Robust change')
	lgd.get_texts()[1].set_text('Conflicting signals')
	lgd.get_texts()[2].set_text('No change or no robust change')
	#bbox = dict(boxstyle="square", fc="1.0")
	#plt.annotate('xx', xy=(-0.01,0.03), xycoords='axes fraction',color='black',fontsize=10,bbox=bbox)
	#plt.annotate(r' Lack of %.2i' % (signagreethreshTOT*100) + r'$\%$', xy=(0.02,0.03), xycoords='axes fraction',color='black',fontsize=10)
	#plt.annotate(r' model sign agreement', xy=(0.02,-0.02), xycoords='axes fraction',color='black',fontsize=10)
	#plt.annotate('\\\\\\\\', xy=(-0.01,0.03), xycoords='axes fraction',color='black',fontsize=10,bbox=bbox)
	#plt.annotate(r' Lack of 2-$\sigma$ ', xy=(0.02,0.03), xycoords='axes fraction',color='black',fontsize=10)
	#plt.annotate(r' model significance ', xy=(0.02,-0.02), xycoords='axes fraction',color='black',fontsize=10)
	# area-weighted mean 
	areawtmn = np.sum(awt*aererfTOT)
	xann=0.85
	yann=0.03
	plt.annotate(r'Mean ERF ', xy=(xann,yann), xycoords='axes fraction',color='black',fontsize=10)
	plt.annotate(r'= %5.2f [W-m$^{-2}$]'% areawtmn, xy=(xann,yann-0.05), xycoords='axes fraction',color='black',fontsize=10)
	
#        # SW
#	cmax = np.max([np.abs(aererfSW.max()),np.abs(aererfSW.min())])
#	cmin = -cmax
#	cincr=2.0*cmax/float(clevels-1)
#	crange=np.arange(cmin,cmax+cincr,cincr)
#	ax = fig.add_subplot(nrows, ncols, 2 , projection=ccrs.Robinson(central_longitude=0))
#	im=ax.contourf(Xplot,Yplot,aererfSW,crange,cmap=cmapip, transform=ccrs.PlateCarree())
#	imh=ax.contourf(Xplot,Yplot,aererfhatchSW,3,colors='none', hatches=['','xx','\\\\'], alpha=0.0, transform=ccrs.PlateCarree())
#	ax.coastlines()
#	ax.set_global()
#	cb=fig.colorbar(im,fraction=0.05,pad=0.05,orientation='horizontal')
#	cblabel=r'(W-m$^{-2}$)'
#	maptitle='Shortwave Effective Radiative Forcing'+'\n'+'%s' % plttype
#	plt.annotate('b)', xy=(0., 0.9), xycoords='axes fraction',fontsize=14)
#	cb.set_label(r'%s' % cblabel,fontsize=12)
#	plt.title(maptitle,fontsize=16)
#	bbox = dict(boxstyle="square", fc="1.0")
#	plt.annotate('xx', xy=(-0.01,0.03), xycoords='axes fraction',color='black',fontsize=10,bbox=bbox)
#	plt.annotate(r' Lack of %.2i' % (signagreethreshSW*100) + r'$\%$', xy=(0.02,0.03), xycoords='axes fraction',color='black',fontsize=10)
#	plt.annotate(r' model sign agreement', xy=(0.02,-0.02), xycoords='axes fraction',color='black',fontsize=10)
#	#plt.annotate('\\\\\\\\', xy=(-0.01,0.03), xycoords='axes fraction',color='black',fontsize=10,bbox=bbox)
#	#plt.annotate(r' Lack of 2-$\sigma$ ', xy=(0.02,0.03), xycoords='axes fraction',color='black',fontsize=10)
#	#plt.annotate(r' model significance ', xy=(0.02,-0.02), xycoords='axes fraction',color='black',fontsize=10)
#	# area-weighted mean 
#	areawtmn = np.sum(awt*aererfSW)
#	plt.annotate(r'Mean ERF ', xy=(xann,yann), xycoords='axes fraction',color='black',fontsize=10)
#	plt.annotate(r'= %5.2f [W-m$^{-2}$]'% areawtmn, xy=(xann,yann-0.05), xycoords='axes fraction',color='black',fontsize=10)
#
#	# LW	
#	#cmax = np.max([np.abs(aererfLW.max()),np.abs(aererfLW.min())])
#	cmax = 4
#	cmin = -cmax
#	cincr=2.0*cmax/float(clevels-1)
#	crange=np.arange(cmin,cmax+cincr,cincr)
#	ax = fig.add_subplot(nrows, ncols, 3 , projection=ccrs.Robinson(central_longitude=0))
#	im=ax.contourf(Xplot,Yplot,aererfLW,crange,cmap=cmapip, transform=ccrs.PlateCarree())
#	imh=ax.contourf(Xplot,Yplot,aererfhatchLW,3,colors='none', hatches=['','xx','\\\\'], alpha=0.0, transform=ccrs.PlateCarree())
#	ax.coastlines()
#	ax.set_global()
#	cb=fig.colorbar(im,fraction=0.05,pad=0.05,orientation='horizontal')
#	cblabel=r'(W-m$^{-2}$)'
#	maptitle='Longwave Effective Radiative Forcing'+'\n'+'%s' % plttype
#	plt.annotate('c)', xy=(0., 0.9), xycoords='axes fraction',fontsize=14)
#	cb.set_label(r'%s' % cblabel,fontsize=12)
#	plt.title(maptitle,fontsize=16)
#	bbox = dict(boxstyle="square", fc="1.0")
#	plt.annotate('xx', xy=(-0.01,0.03), xycoords='axes fraction',color='black',fontsize=10,bbox=bbox)
#	plt.annotate(r' Lack of %.2i' % (signagreethreshLW*100) + r'$\%$', xy=(0.02,0.03), xycoords='axes fraction',color='black',fontsize=10)
#	plt.annotate(r' model sign agreement', xy=(0.02,-0.02), xycoords='axes fraction',color='black',fontsize=10)
#	#plt.annotate('\\\\\\\\', xy=(-0.01,0.03), xycoords='axes fraction',color='black',fontsize=10,bbox=bbox)
#	#plt.annotate(r' Lack of 2-$\sigma$ ', xy=(0.02,0.03), xycoords='axes fraction',color='black',fontsize=10)
#	#plt.annotate(r' model significance ', xy=(0.02,-0.02), xycoords='axes fraction',color='black',fontsize=10)
#	# area-weighted mean 
#	areawtmn = np.sum(awt*aererfLW)
#	plt.annotate(r'Mean ERF ', xy=(xann,yann), xycoords='axes fraction',color='black',fontsize=10)
#	plt.annotate(r'= %5.2f [W-m$^{-2}$]'% areawtmn, xy=(xann,yann-0.05), xycoords='axes fraction',color='black',fontsize=10)

	plt.tight_layout()
	local_path = cfg['plot_dir']
	png_name = 'fig6_erf_piAer.png'  
	svg_name = 'fig6_erf_piAer.svg'  
	plt.savefig(os.path.join(local_path, png_name),dpi=300,bbox_inches='tight')
	plt.savefig(os.path.join(local_path, svg_name),dpi=300,bbox_inches='tight')
	plt.close()


if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
