
# ipcc_ar6wg1_interactiveatlas_erf_aer_time.py  

# Description
# Generates figures for IPCC Working Group I Contribution to the Sixth Assessment Report
# IPCC AR6 WG1 Interactive Atlas preprosessed data for Effective Radiative Forcing due to aerosols 

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
import numpy.ma as ma
from scipy import stats
import cartopy.crs as ccrs

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
	nmodels=0
	for attributes in selection:
		logger.info("Processing dataset %s", attributes['dataset'])
		input_file = attributes['filename']
		logger.info("Loading %s", input_file)
		# get start and end year.  all data should be the same, so do it once 
		styear = attributes['start_year']
		endyear = attributes['end_year']
		logger.info("start-end %d-%d ", (styear,endyear))
		if(nmodels > 0):
			if(styear != styearprev or endyear != endyearprev):
				logger.info("Error start-end from previous models", )
		else:
			styearprev = styear
			endyearprev = endyear
		nmodels = nmodels + 1

	return styear, endyear

def compute_ModelStats(cfg,selection):
	'''
	Arguments:
		cfg - ESMValTool Python dictionary of information on input data
		selection - selected metadata 			
	'''
	for attributes in selection:
		input_file = attributes['filename']
	logger.info("opening %s",input_file)
	cube=iris.load_cube(input_file)
	cshape=cube.shape
	logger.info('cube %d  %d %d ' % (cshape[0],cshape[1],cshape[2]))	
	alltimedata=cube.data

	provenance_record = get_provenance_record(
		attributes, ancestor_files=[input_file])

	output_basename = os.path.splitext(os.path.basename(input_file))[0]  
	diagnostic_file = get_diagnostic_filename(output_basename, cfg)
	#diagnostic_file = os.path.join(cfg['preproc_dir'],basename + '.' + extension,
	with ProvenanceLogger(cfg) as provenance_logger:
		provenance_logger.log(input_file,provenance_record)
		provenance_logger.log(diagnostic_file,provenance_record)

	alltimeavgcube = alltimedata.mean(axis=0)
	alltimestdcube = alltimedata.std(axis=0)	

	timeavgcube = cube.collapsed('time', iris.analysis.MEAN)
	timeavgcube.data = alltimeavgcube

	return timeavgcube, alltimedata, provenance_record 

def compute_multiModelStats(cfg,selection):
	'''
	Compute the multi-model statistics
	Arguments:
		cfg - ESMValTool Python dictionary of information on input data
		selection - selected metadata 			
	Returns:
		timeavgcube - iris cube of models, time-averaged 
		alltimedata - numpy array of all models. dims are (nmodels*time,lat,lon)
		alltimemodels - numpy array of all models. dims are (nmodels, time, lat, lon)
		provenance_record 
	'''
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
                #diagnostic_file = os.path.join(cfg['preproc_dir'],basename + '.' + extension,
		with ProvenanceLogger(cfg) as provenance_logger:
			provenance_logger.log(input_file,provenance_record)
			provenance_logger.log(diagnostic_file,provenance_record)

	alltimeavgcube = alltimedata.mean(axis=0)
	alltimestdcube = alltimedata.std(axis=0)	

    # set to an iris cube to use iris cube format of time-averaged data
	timeavgcube = cube.collapsed('time', iris.analysis.MEAN)
	# fill the data part of cube with time-averaged data
	timeavgcube.data = alltimeavgcube
    # set to an iris cube to use iris cube format of variance of time data 
	timestdcube = cube.collapsed('time', iris.analysis.VARIANCE)
	# fill the data part of cube with temporal standard-deviation of data
	timestdcube.data = alltimestdcube

	alltimemodels= np.zeros((nmodels,cshape[0],cshape[1],cshape[2]))
	for imdl in range(0,nmodels):
		alltimemodels[imdl,:,:,:] = alltimedata[imdl*cshape[0]:(imdl+1)*cshape[0],:,:]

	return timeavgcube, alltimedata, alltimemodels, provenance_record 


def writenetcdf_modeltimemap(cfg,dataset,styr,endyr,field,lat4wrt,lon4wrt,expmdls,cntlmdls):
	'''
	
	Arguments:
		cfg - ESMValTool Python dictionary of information on input data
		dataset - string of model name
		styr - start year of model data
		endyr - end year of model data
		field - model field string
		lat4wrt - 1D array of latitudes for writing 
		lon4wrt - 1D array of longitudes for writing 
		expmdls - numpy array of experiment models (histSST). dimension (time,lat,lon)
		cntlmdls - numpy array of control models (histSST-piAer). dimension (time,lat,lon)
	Results:
		writes out model data of Effective Radiative Forcing due to Aerosols gridded time series for use in IPCC AR6 WG1 Interactive Atlas	
	'''
    local_path = cfg['plot_dir']
    cdims=expmdls.shape
    ntime = cdims[0]
    nlat = cdims[1] 
    nlon = cdims[2]
   
    outfname = '%s_%s_Aer_ERF_%s-%s.nc'  % (dataset,field,styr,endyr)
    ncfile = netCDF4.Dataset(os.path.join(local_path,outfname), mode='w',format='NETCDF4_CLASSIC') 
    lat_dim = ncfile.createDimension('lat', nlat) # latitude axis
    lon_dim = ncfile.createDimension('lon', nlon) # longitude axis
    time_dim = ncfile.createDimension('time', ntime) # unlimited axis (can be appended to).

    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'months since %s-01-01' % styr
    time.long_name = 'time'

    erf = ncfile.createVariable('ERF',np.float64,('time','lat','lon'), fill_value = -9999) # note: unlimited dimension is leftmost
    erf.units = 'W/m^2' # 
    erf.standard_name = 'effective_rad_forcing due to Aerosols across model' # this is a CF standard name
   
    #
    erf_mdl2mdl = expmdls-cntlmdls
    # plotting ERF.  rsut=outgoing shortwave flux (positive outwards).  therefore sign convention for ERF is positive downwards.
    erf_mdl2mdl = - erf_mdl2mdl # sign convention

    lat[:]= lat4wrt
    lon[:] = lon4wrt

    time[:] = np.arange(0,ntime)

    # monthly data
    erf[:,:,:] = erf_mdl2mdl[:,:,:]
    
    ncfile.close()


def main(cfg):
	"""Main diagnostic run function."""
	input_data = cfg['input_data'].values()
	variables = Variables(cfg)
	allvars = variables.short_names()

#	for ifield in range(0,len(allvars)):
	esmmodels = group_metadata(cfg['input_data'].values(), 'dataset')

    	# iterate over key(dataset) and values(list of vars)
	for datasetname, inputmodel in esmmodels.items():

		histSSTexp='histSST'
		select_histSSTSW= select_metadata(inputmodel, exp=histSSTexp, short_name='rsut')
		select_histSSTLW= select_metadata(inputmodel, exp=histSSTexp, short_name='rlut')
		logger.info("Example of how to select only histSST data:\n%s",
			pformat(select_histSSTSW))
		[styear,endyear]=checkstartend(select_histSSTSW)
		logger.info("Example of how to select only histSST data:\n%s",
			pformat(select_histSSTLW))
		[styear,endyear]=checkstartend(select_histSSTLW)
		[histSST_mdlmnSW,histSST_mdlsSW,prov_rec]=compute_ModelStats(cfg,select_histSSTSW)
		[histSST_mdlmnLW,histSST_mdlsLW,prov_rec]=compute_ModelStats(cfg,select_histSSTLW)

		histSSTaer='histSST-piAer'
		select_histSSTpiAerSW= select_metadata(inputmodel,exp=histSSTaer, short_name='rsut')
		select_histSSTpiAerLW= select_metadata(inputmodel,exp=histSSTaer, short_name='rlut')
		logger.info("Example of how to select only histSST-piAer data:\n%s",
			pformat(select_histSSTpiAerSW))
		[styear,endyear]=checkstartend(select_histSSTpiAerSW)
		logger.info("Example of how to select only histSST-piAer data:\n%s",
			pformat(select_histSSTpiAerLW))
		[styear,endyear]=checkstartend(select_histSSTpiAerLW)
		[histSSTaer_mdlmnLW,histSSTaer_mdlsLW,prov_rec]=compute_ModelStats(cfg,select_histSSTpiAerLW)
		[histSSTaer_mdlmnSW,histSSTaer_mdlsSW,prov_rec]=compute_ModelStats(cfg,select_histSSTpiAerSW)


		# prep for plotting
		repcube = histSST_mdlmnSW		 # respresentative cube
		Xplot1 = repcube.coord('longitude').points
		Yplot1 = repcube.coord('latitude').points
		Xplot, Yplot = np.meshgrid(Xplot1, Yplot1)

		# output erf temporal maps 
		writenetcdf_modeltimemap(cfg,datasetname,styear,endyear,'Tot',Yplot1,Xplot1,histSST_mdlsSW+histSST_mdlsLW,histSSTaer_mdlsSW+histSSTaer_mdlsLW)
		writenetcdf_modeltimemap(cfg,datasetname,styear,endyear,'SW',Yplot1,Xplot1,histSST_mdlsSW,histSSTaer_mdlsSW)
		writenetcdf_modeltimemap(cfg,datasetname,styear,endyear,'LW',Yplot1,Xplot1,histSST_mdlsLW,histSSTaer_mdlsLW)



if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
