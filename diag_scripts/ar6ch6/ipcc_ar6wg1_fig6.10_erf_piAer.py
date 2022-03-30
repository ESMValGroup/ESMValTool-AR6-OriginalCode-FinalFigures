# ipcc__ar6wg1_fig6.10_erf_piAer.py

# Description
# Generates figures and proprocessed data for IPCC Working Group I Contribution to the Sixth Assessment Report
#  IPCC AR6 WG1 Figure 6.10a 
#  IPCC AR6 WG1 Figure 6.10b preprosessed data 

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

from ch6_fns import compute_multiModelStats, compute_multiModelDiffStats, get_signagreemap, get_totalERFmap

logger = logging.getLogger(os.path.basename(__file__))

def get_areaweight(nlat,nlon,lat,lon):
	''' 
		determine the quandrangle area weight 
    Arguments:
				nlat - integer size of lat latitudes
				nlon - integer size of lon longitudes
				lat - 1D array of latitude values
				lon - 1D array of longitude values
	Returns::
				areaweight - 2D array of weights by lat/lon grid area 	
  	'''

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
	''' output to verify the starting and ending years '''
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

def plot_meanmap(cfg,cubeavg,exper,field):
    """
    Arguments:
        cubeavg - the iris cube to plot, for gridded means
    """
    local_path = cfg['plot_dir']

    # coordinates
    Xplot1 = cubeavg.coord('longitude').points
    Yplot1 = cubeavg.coord('latitude').points
    Xplot, Yplot = np.meshgrid(Xplot1, Yplot1)

    # colormaps
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


def get_erfmap(exptavg,exptall,cntlavg,cntlall):
    """
	Calculate the gridded effective radiative forcing map, and determine hatched areas.

	Arguments:
		exptavg - gridded mean of the model experiments (CMIP6 'historical')
		exptall - all the model experiment (CMIP6 'historical'), to determine hatching statistics
		cntlavg - gridded mean of the model control (AerChemMIP 'hist-piAer')
		cntlall - all the model experiment (AerChemMIP 'hist-piAer'), to determine hatching statistics
	Returns:
		erfmap - the effective radiative forcing map. [W/m^2]
		hatchplot - map of binary values. 1 - grid is hatched, 0 - grid is not hatched.	
    """

    erfmap =(exptavg.data-cntlavg.data)
    # ERF sign convention
    erfmap = -erfmap

    # get hatch pattern
    cdims=exptall.shape
    hatchplot = np.zeros((cdims[1],cdims[2]))
    for ilat in range(0,cdims[1]):
        for ilon in range(0,cdims[2]):
            a = exptall.data[:,ilat,ilon]
            b = cntlall.data[:,ilat,ilon]
            ars=np.resize(a,(cdims[0],))
            brs=np.resize(b,(cdims[0],))

            varthresh = np.sqrt(2)*1.645*b.std() 
	    # FGD guidelines on hatching threshold. hatch where not significant 
            if( np.abs(a.mean()) < (np.abs(b.mean())+varthresh)):		
                hatchplot[ilat,ilon] = 1   

    return erfmap,hatchplot

    
def writenetcdf_erfhatch(cfg,field,lat4wrt,lon4wrt,erf4wrt,hatch4wrt):
	'''
	Write out the erf and hatch patterns into a netCDF file, 
	netCDF file to be read into ipcc_ar6wg1_Fig6.10b_FGD_submit.ipynb to create IPCC AR6 WG1 Figure 6.10b
	'''
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

def main(cfg):
	"""Main diagnostic run function."""
	input_data = cfg['input_data'].values()
	variables = Variables(cfg)
	allvars = variables.short_names()

	# prepare figure matrix
	nrows=1
	ncols=1
	fig = plt.figure(figsize=(ncols*7, nrows*5))
	fig.subplots_adjust(hspace=0.3)
	plt.rcParams.update({'hatch.color': 'grey'})	
	

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
	# area-weighted mean 
	areawtmn = np.sum(awt*aererfTOT)
	xann=0.85
	yann=0.03
	plt.annotate(r'Mean ERF ', xy=(xann,yann), xycoords='axes fraction',color='black',fontsize=10)
	plt.annotate(r'= %5.2f [W-m$^{-2}$]'% areawtmn, xy=(xann,yann-0.05), xycoords='axes fraction',color='black',fontsize=10)
	
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
