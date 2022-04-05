import logging
from esmvaltool.diag_scripts.shared._base import (
    ProvenanceLogger, get_diagnostic_filename, get_plot_filename)
import iris
import os
import numpy as np
from scipy import stats

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
            'kuo_chaincy',
        ],
        'references': [
            'acknow_project',
        ],
        'ancestors': ancestor_files,
    }
    return record

def compute_multiModelStats(cfg,selection):
	# output:
	# 	timeavgcube,  time-averaged iris cube
	#	alltimedata,  all time data as iris cube
	#	alltimemodels, all time data as numpy array
	#	provenance_record 
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

	timeavgcube = cube.collapsed('time', iris.analysis.MEAN)
	timeavgcube.data = alltimeavgcube
	timestdcube = cube.collapsed('time', iris.analysis.VARIANCE)
	timestdcube.data = alltimestdcube

	alltimemodels= np.zeros((nmodels,cshape[0],cshape[1],cshape[2]))
	for imdl in range(0,nmodels):
		alltimemodels[imdl,:,:,:] = alltimedata[imdl*cshape[0]:(imdl+1)*cshape[0],:,:]

	return timeavgcube, alltimedata, alltimemodels, provenance_record 

def compute_multiModelDiffStats(experiment, control):  
# historical is the experiment, and the pi-* exeriments control for specific species to pre-industrial levels

    # compute differences between experiment and control for individual models
    # input are cubes [time, nlat, nlon]
	nmodels=0
	for attributes in control:
		logger.info("compute MM Processing dataset cntl %s", attributes['dataset'])
		input_model = attributes['dataset']
		input_filecntl = attributes['filename']
		logger.info("opening %s",input_filecntl)
		cubecntl=iris.load_cube(input_filecntl)
		cshape=cubecntl.shape
                # dims [ntime, nlat, nlon]
		logger.info('cubecntl %d  %d %d ' % (cshape[0],cshape[1],cshape[2]))	
		if(nmodels==0):
			alltimedata = np.zeros((cshape[0],cshape[1],cshape[2]))
		iflagexp = 0
		for attributesexp in experiment:
			if(attributesexp['dataset'] == input_model):
				iflagexp = 1 # match found
				logger.info("computeMM Processing dataset exp %s", attributesexp['dataset'])
				input_fileexp = attributesexp['filename']
				logger.info("opening %s",input_fileexp)
				cubeexp=iris.load_cube(input_fileexp)
				cshape=cubeexp.shape
				logger.info('cubeexp %d  %d %d ' % (cshape[0],cshape[1],cshape[2]))	

		#logger.info("computeMM iflagexp %d", iflagexp)
		if(iflagexp == 1):
			climateresponse= cubeexp.data - cubecntl.data 
			if(nmodels==0):
				alltimedata = climateresponse
			else:
                		# dims [nmodels x ntime, nlat, nlon]
				alltimedata = np.append(alltimedata,climateresponse,axis=0)
			nmodels=nmodels+1

	alltimeavgcube = alltimedata.mean(axis=0)
	alltimestdcube = alltimedata.std(axis=0)	

	# set a cube for the metadata
	timeavgcube = cubecntl.collapsed('time', iris.analysis.MEAN)
	# replace with the data of interest
	timeavgcube.data = alltimeavgcube
	# set a cube for the metadata
	timestdcube = cubecntl.collapsed('time', iris.analysis.VARIANCE)
	# replace with the data of interest
	timestdcube.data = alltimestdcube

	# numpy arrays
	alltimemodels= np.zeros((nmodels,cshape[0],cshape[1],cshape[2]))
	for imdl in range(0,nmodels):
		alltimemodels[imdl,:,:,:] = alltimedata[imdl*cshape[0]:(imdl+1)*cshape[0],:,:]
        # dims [nmodels x ntime, lat, lon]

	return timeavgcube, alltimedata, alltimemodels  


def get_signagreemap(exptnp,cntlnp):
	"""
	Arguments:
	extnp = experiment as numpy arrays 
	cntlnp = control as numpy arrays 
	Returns:


	"""
	# get dimensions
	cdims=exptnp.shape 
	nmdls =  cdims[0] 
	ntime =  cdims[1]  
	nlat =  cdims[2] 
	nlon =  cdims[3] 
	  
	# hatch pattern
	# model agreement on sign
	hatchagreesign = np.zeros((cdims[2],cdims[3]))
	# model mean significance
	hatchsig = np.zeros((cdims[2],cdims[3]))
	# hatch pattern
	hatchmap = np.ones((cdims[2],cdims[3]))

	#multi-model difference 
	mmdiff = exptnp.mean(axis=(0,1))-cntlnp.mean(axis=(0,1))

	# significance level at 95%
	#pthresh=0.05
	# 
	for imdl in range(0,nmdls):
		for ilat in range(0,cdims[2]):
			for ilon in range(0,cdims[3]):
				a = exptnp[imdl,:,ilat,ilon]
				b = cntlnp[imdl,:,ilat,ilon]
				varthresh = np.sqrt(2)*1.645*b.std() 
				expgridmn = exptnp[imdl,:,ilat,ilon].mean()
				cntlgridmn = cntlnp[imdl,:,ilat,ilon].mean()
				griddiff = expgridmn - cntlgridmn 	
				if( np.abs(griddiff) < varthresh):		
					hatchsig[ilat,ilon] = hatchsig[ilat,ilon] + 1.0
				# sign agreement if 
				if((griddiff*mmdiff[ilat,ilon])>0):
					hatchagreesign[ilat,ilon] = hatchagreesign[ilat,ilon] + 1.0

		    #ars=np.resize(a,(cdims[0]*cdims[1]))
		    #brs=np.resize(b,(cdims[0]*cdims[1]))
		    #[tstat,pval] = stats.ttest_ind(ars, brs, equal_var = False)
		    #  pval>pthresh means mark areas without significance
		    #  pval<pthresh means mark significant areas 
		    #if(pval>pthresh):
		    #    hatchsig[ilat,ilon] = 1
		    # FGD guidelines on hatching threshold. hatch were not significant 

	hatchagreesign = hatchagreesign/float(nmdls) 
	hatchsig = hatchsig/float(nmdls)

	signifthresh = 0.66
	if( nmdls >= 10):
		signagreethresh = 0.9
	elif( nmdls >= 5):
		signagreethresh = 0.8
	elif( nmdls == 4):
		signagreethresh = 0.75
	else:
		signagreethresh = float(nmdls-1)/float(nmdls)

	for ilat in range(0,cdims[2]):
		for ilon in range(0,cdims[3]):
			if((hatchagreesign[ilat,ilon] > signagreethresh) and (hatchsig[ilat,ilon] > signifthresh)):
				hatchmap[ilat,ilon] = 0.0
			elif((hatchagreesign[ilat,ilon] < signagreethresh) and (hatchsig[ilat,ilon] > signifthresh)):
				hatchmap[ilat,ilon] = 0.5
			elif((hatchagreesign[ilat,ilon] < signagreethresh) and (hatchsig[ilat,ilon] < signifthresh)):
				hatchmap[ilat,ilon] = 1.0

	return hatchmap, signagreethresh


def get_diffdiffmap(histexp,histcntl,histSSTexp,histSSTcntl,histexpall,histcntlall,histSSTexpall,histSSTcntlall):

    """
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

def get_totalERFmap(histSSTLWexp,histSSTSWexp,histSSTLWexpall,histSSTSWexpall,
			histSSTLWcntl,histSSTSWcntl,histSSTLWcntlall,histSSTSWcntlall):
		

    """
    Arguments:
	*avg are iris cubes
	*all are numpy array dims[nmodels x ntime,nlat,nlon] 

    Returns:
	

    """
    plotnowLW=(histSSTLWexp.data-histSSTLWcntl.data)
    plotnowSW=(histSSTSWexp.data-histSSTSWcntl.data)
    plotnow = -(plotnowLW+plotnowSW)
    # get hatch pattern
    cdims=histSSTLWexpall.shape
    hatchnow = np.zeros((cdims[1],cdims[2]))
    hatchplot = np.zeros((cdims[1],cdims[2]))
    pthresh=0.050
    for ilat in range(0,cdims[1]):
        for ilon in range(0,cdims[2]):
            a = (histSSTLWexpall[:,ilat,ilon] + histSSTSWexpall[:,ilat,ilon])
            b = (histSSTLWcntlall[:,ilat,ilon] + histSSTSWcntlall[:,ilat,ilon])
            ars=np.resize(a,(cdims[0],))
            brs=np.resize(b,(cdims[0],))
            varthresh = np.sqrt(2)*1.654*b.std()
            # TSU FGD guidelines on hatching
            if((np.abs(a.mean()) < (np.abs(b.mean())+varthresh))):
                hatchplot[ilat,ilon] = 1

            #ttest = stats.ttest_ind(ars, brs, equal_var = False)
            #hatchnow[ilat,ilon]=ttest[1]
	    ##  ttest>pthresh means mark areas without significance
	    ##  ttest<pthresh means mark significant areas 
            #if(ttest[1]>pthresh):
            #    hatchplot[ilat,ilon] = 1

    return plotnow,hatchplot

def compute_ModelStats(cfg,selection):
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

