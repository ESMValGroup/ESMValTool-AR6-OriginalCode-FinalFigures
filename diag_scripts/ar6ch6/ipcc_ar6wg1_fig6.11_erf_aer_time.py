
# ipcc_ar6wg1_fig6.11_erf_aer_time.py 

# Description
# Generates ESMValTool preprocessed data for IPCC Working Group I Contribution to the Sixth Assessment Report
# IPCC AR6 WG1 Figure 6.11 preprosessed data 
# Final Figure 6.11 generatied by ipcc_ar6wg1_Fig6.11_FGD_submit.ipynb

# Creator: Chaincy Kuo (CKuo@lbl.gov, chaincy.ipccwg1@gmail.com )
# Creation Date:  11 March 2021



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

def compute_multiModelStats(cfg,selection):
	nmodels=0
	for attributes in selection:
		input_file = attributes['filename']
		logger.info("opening %s",input_file)
		cube=iris.load_cube(input_file)
		cshape=cube.shape
		logger.info('cube %d  %d %d ' % (cshape[0],cshape[1],cshape[2]))	
		if(nmodels==0):
			alltimedata=cube.data
		else:
			alltimedata = np.append(alltimedata,cube.data,axis=0)
		nmodels=nmodels+1

		provenance_record = get_provenance_record(
                        attributes, ancestor_files=[input_file])

		output_basename = os.path.splitext(os.path.basename(input_file))[0]  
		diagnostic_file = get_diagnostic_filename(output_basename, cfg)
		with ProvenanceLogger(cfg) as provenance_logger:
			provenance_logger.log(input_file,provenance_record)
			provenance_logger.log(diagnostic_file,provenance_record)

		lon4wrt = cube.coord('longitude').points
		lat4wrt = cube.coord('latitude').points


	alltimeavgcube = alltimedata.mean(axis=0)
	alltimestdcube = alltimedata.std(axis=0)	

	timeavgcube = cube.collapsed('time', iris.analysis.MEAN)
	timeavgcube.data = alltimeavgcube
	timestdcube = cube.collapsed('time', iris.analysis.VARIANCE)
	timestdcube.data = alltimestdcube

	alltimemodels= np.zeros((nmodels,cshape[0],cshape[1],cshape[2]))
	for imdl in range(0,nmodels):
		alltimemodels[imdl,:,:,:] = alltimedata[imdl*cshape[0]:(imdl+1)*cshape[0],:,:]

	return timeavgcube, alltimedata, alltimemodels, provenance_record 


def plot_meanmap(cfg,cubeavg,exper,field):
    """

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

    plotnow=cubeavg.data
    nrows=1
    ncols=1

    fig = plt.figure(figsize=(ncols*7, nrows*5))
    ax = fig.add_subplot(nrows, ncols, 1, projection=ccrs.Robinson(central_longitude=180))
    im=ax.contourf(Xplot,Yplot,plotnow,clevels,cmap=cmapip, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_global()
    cb=fig.colorbar(im,fraction=0.05,pad=0.05,orientation='horizontal')
    if(field=='rsut'):
	    cb.set_label(r'W-m$^{-1}$ ',fontsize=12)
	    plt.suptitle(r'Shortwave Effective Radiative Forcing',fontsize=16)
    elif(field=='rlut'):
	    cb.set_label(r'W-m$^{-1}$ ',fontsize=12)
	    plt.suptitle(r'Longwave Effective Radiative Forcing',fontsize=16)
    plt.tight_layout()
    
    png_name = 'multimodelMean_%s_%s.png' % (field,exper)
    plt.savefig(os.path.join(local_path, png_name))
    plt.close()


def plot_diffmaperf(cfg,histavg,histall,histcntlavg,histcntall,field):
    """

    """
    local_path = cfg['plot_dir']
    # get hatch pattern
    cdims=histall.shape
    hatchnow = np.zeros((cdims[1],cdims[2]))
    hatchplot = np.zeros((cdims[1],cdims[2]))
    pthresh=0.05
    for ilat in range(0,cdims[1]):
        for ilon in range(0,cdims[2]):
            a = histall.data[:,ilat,ilon]
            b = histcntall.data[:,ilat,ilon]
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

    # colormaps
    clevels=11
    cmapnow=cmapipcc.get_ipcc_cmap('temperature',clevels)
    cmapip=ListedColormap(np.flipud(cmapnow))

    plotnow=(histavg.data-histcntlavg.data)
    # erf sign convention
    plotnow = -plotnow

    # cumulative sum
    plotflat=np.resize(plotnow,((cdims[1]*cdims[2]),))
    plothist= np.histogram(plotflat,bins=20)
    plotcum = np.cumsum(plothist[0][:])
    plotcumx = np.zeros((len(plotcum),))
    for i in range(0,len(plotcum)):
        plotcumx[i] = (plothist[1][i+1]+plothist[1][i])/2.0  # plothist is a list, not a tuple, so use this method of indexing


    nrows=1
    ncols=1

    cmax = np.max([np.abs(plotnow.max()),np.abs(plotnow.min())])
    cmin = -cmax
    cincr=2.0*cmax/float(clevels-1)
    crange=np.arange(cmin,cmax+cincr,cincr)

    fig = plt.figure(figsize=(ncols*7, nrows*5))
    ax = fig.add_subplot(nrows, ncols, 1, projection=ccrs.Robinson(central_longitude=180))
    im=ax.contourf(Xplot,Yplot,plotnow,crange,cmap=cmapip, transform=ccrs.PlateCarree())
    imh=ax.contourf(Xplot,Yplot,hatchplot,3,colors='none', hatches=['','++'], alpha=0.0, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_global()
    cb=fig.colorbar(im,fraction=0.05,pad=0.05,orientation='horizontal')
    cb.set_label(r'(W-m$^{-1}$)',fontsize=12)
    if(field=='rsut'):
	    plt.suptitle(r'Shortwave Effective Radiative Forcing',fontsize=16)
	    plt.title(r'',fontsize=14)
    elif(field=='rlut'):
	    plt.suptitle(r'Longwave Effective Radiative Forcing',fontsize=16)
	    plt.title(r'',fontsize=14)

    if(nrows==2):
            ax2 = fig.add_subplot(nrows, ncols, 2)
            ax2.plot(plotcumx,plotcum)
            ax2.set_xlabel(r'(W-m$^{-1}$)',fontsize=12)
            ax2.set_ylabel(r'Cumulative')

    plt.tight_layout()
    
    png_name = 'diff_%s.png' % field
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

    # colormaps
    clevels=11
    cmapnow=cmapipcc.get_ipcc_cmap('temperature',clevels)
    cmapip=ListedColormap(np.flipud(cmapnow))

    plotnow=(histavg.data-histcntlavg.data)

    # cumulative sum
    plotflat=np.resize(plotnow,((cdims[1]*cdims[2]),))
    plothist= np.histogram(plotflat,bins=20)
    plotcum = np.cumsum(plothist[0][:])
    plotcumx = np.zeros((len(plotcum),))
    for i in range(0,len(plotcum)):
        plotcumx[i] = (plothist[1][i+1]+plothist[1][i])/2.0  # plothist is a lsit, not a tuple, so use this method of indexing

    nrows=1
    ncols=1

    cmax = np.max([np.abs(plotnow.max()),np.abs(plotnow.min())])
    cmin = -cmax
    cincr=2.0*cmax/float(clevels-1)
    crange=np.arange(cmin,cmax+cincr,cincr)

    fig = plt.figure(figsize=(ncols*7, nrows*5))
    ax = fig.add_subplot(nrows, ncols, 1, projection=ccrs.Robinson(central_longitude=180))
    im=ax.contourf(Xplot,Yplot,plotnow,crange,cmap=cmapip, transform=ccrs.PlateCarree())
    imh=ax.contourf(Xplot,Yplot,hatchplot,3,colors='none', hatches=['','++'], alpha=0.0, transform=ccrs.PlateCarree())
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

    """

    plotnow=(exptavg.data-cntlavg.data)
    # ERF sign convention
    plotnow = -plotnow   

    # get hatch pattern
    cdims=exptall.shape
    hatchnow = np.zeros((cdims[1],cdims[2]))
    hatchplot = np.zeros((cdims[1],cdims[2]))
    pthresh=0.05
    for ilat in range(0,cdims[1]):
        for ilon in range(0,cdims[2]):
            a = exptall.data[:,ilat,ilon]
            b = cntlall.data[:,ilat,ilon]
            ars=np.resize(a,(cdims[0],))
            brs=np.resize(b,(cdims[0],))
            ttest = stats.ttest_ind(ars, brs, equal_var = False)
            hatchnow[ilat,ilon]=ttest[1]
	    #  ttest>pthresh means mark areas without significance
	    #  ttest<pthresh means mark significant areas 
            if(ttest[1]>pthresh):
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

def writenetcdf_timemap(cfg,field,lat4wrt,lon4wrt,expmdls,cntlmdls):
    local_path = cfg['plot_dir']
    cdims=expmdls.shape
    nmodels = cdims[0]
    ntime = cdims[1]
    nlat = cdims[2] 
    nlon = cdims[3]
    nyears = int(ntime/12)
   
    outfname = '%s_diff_timemap.nc'  % field
    ncfile = netCDF4.Dataset(os.path.join(local_path,outfname), mode='w',format='NETCDF4_CLASSIC') 
    lat_dim = ncfile.createDimension('lat', nlat) # latitude axis
    lon_dim = ncfile.createDimension('lon', nlon) # longitude axis
    time_dim = ncfile.createDimension('time', nyears) # unlimited axis (can be appended to).
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'years since 1850-01-01'
    time.long_name = 'time'
    erf = ncfile.createVariable('ERF',np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
    erf.units = 'W/m^2' # 
    erf.standard_name = 'effective_rad_forcing means across models' # this is a CF standard name
   
    #
    erf_mdl2mdl = expmdls-cntlmdls
    erfalltime =  erf_mdl2mdl.mean(axis=0)
    # plotting ERF.  rsut=outgoing shortwave flux (positive outwards).  therefore sign convention for ERF is positive downwards.
    erfalltime = - erfalltime # sign convention

    lat[:]= lat4wrt
    lon[:] = lon4wrt

    # on monthly data
    for iyear in range(0,nyears):
        erf[iyear,:,:] = erfalltime[iyear*12:(iyear+1)*12,:,:].mean(axis=0) 
    
    ncfile.close()

def main(cfg):

	"""Main diagnostic run function."""
	input_data = cfg['input_data'].values()
	variables = Variables(cfg)


def main(cfg):
	"""Main diagnostic run function."""
	input_data = cfg['input_data'].values()
	variables = Variables(cfg)
	allvars = variables.short_names()


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



	# prep for plotting
	repcube = histSST_mdlmnSW		 # respresentative cube
	Xplot1 = repcube.coord('longitude').points
	Yplot1 = repcube.coord('latitude').points
	Xplot, Yplot = np.meshgrid(Xplot1, Yplot1)

	# output erf temporal maps 
	writenetcdf_timemap(cfg,'SW',Yplot1,Xplot1,histSST_mdlsSW,histSSTaer_mdlsSW)
	writenetcdf_timemap(cfg,'LW',Yplot1,Xplot1,histSST_mdlsLW,histSSTaer_mdlsLW)



if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
