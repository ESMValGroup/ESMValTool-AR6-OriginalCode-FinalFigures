
"""Plots timeseries of wet and dry area means from CMIP6 models, for comparison with obs."""
import logging
import os
from pprint import pformat
import iris
import iris.coord_categorisation
import numpy
import numpy.ma as ma
import matplotlib
from scipy import stats
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import glob

from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            select_metadata, sorted_metadata)
from esmvaltool.diag_scripts.shared._base import (
    ProvenanceLogger, get_diagnostic_filename, get_plot_filename)
from esmvaltool.diag_scripts.shared.plot import quickplot



def get_provenance_record(ancestor_files):
    """Create a provenance record describing the diagnostic data and plot."""
    caption = ("Wet and dry region tropical mean (30°S-30°N) annual precipitation anomalies")

    record = {
        'caption': caption,
        'statistics': ['mean'],
        'domains': ['trop'],
        'plot_type': ['times'],
        'authors': [
            'schurer_andrew',
        ],
        'projects': ['ipcc_ar6'],
        'references': [
            'schurer20erl'
        ],
        'realms': ['atmos'],
        'themes': ['phys'],
        'ancestors': ancestor_files,
    }
    return record

#=================================
def addAuxCoords(cube):
	"""
	Add useful aux corrds to a cube
	:param cube: cube to be modified
	:return: nada as cube modified in place
	"""
	#cube.coord('longitude').circular=True # make longitude circular
	try:
		iris.coord_categorisation.add_year(cube, 'time') # add year
		iris.coord_categorisation.add_month(cube, 'time')  # add month
		iris.coord_categorisation.add_month_number(cube, 'time')  # add month
		iris.coord_categorisation.add_season_membership(cube, 'time', 'JFM', name='in_jfm')
		iris.coord_categorisation.add_season_membership(cube, 'time', 'AMJ', name='in_amj')
		iris.coord_categorisation.add_season_membership(cube, 'time', 'JAS', name='in_jas')
		iris.coord_categorisation.add_season_membership(cube, 'time', 'OND', name='in_ond')
		iris.coord_categorisation.add_season_year(cube, 'time', name='season_year', seasons=['JFM', 'AMJ','JAS','OND']) 
	except (iris.exceptions.CoordinateNotFoundError,ValueError):
		pass
	for bndCoord in ['time','longitude','latitude']:
		try:
			cube.coord(bndCoord).guess_bounds()
		except (ValueError, iris.exceptions.CoordinateNotFoundError):
			pass
def calc_globalmean(x):
        #Calculate a global mean
        grid_areas=iris.analysis.cartography.area_weights(x)
        x=x.collapsed(['latitude', 'longitude'], iris.analysis.MEAN,weights=grid_areas)
        return x
def smooth_vals(yr,wet1,dry1):
	#Calculate a running 12-year sum (i.e. annual total rainfall)
	wet=wet1.copy()
	dry=dry1.copy()
	Av_days_in_month=numpy.tile([31,28.25,31,30,31,30,31,31,30,31,30,31],2000)	
	Av_days_in_month=Av_days_in_month[:len(wet)]
	wet*=Av_days_in_month
	dry*=Av_days_in_month
	nmon=len(wet)-11
	wetsmooth=numpy.zeros(nmon)
	drysmooth=numpy.zeros(nmon)
	yrsmooth=numpy.zeros(nmon)
	for ijk in range(nmon):
		wetsmooth[ijk]=numpy.sum(wet[ijk:ijk+12])
		drysmooth[ijk]=numpy.sum(dry[ijk:ijk+12])
		yrsmooth[ijk]=yr[ijk+5]+(1/24.)
	#print(yr)
	#print(yrsmooth)
	return yrsmooth,wetsmooth,drysmooth
def CalcWetDry(data,grid_areas,option,daysinseas):
	#Define gridcells as "wet" and "dry" 
	#and then caluclate mean rainfall for wet and dry regions
	dataFLAT=data.data[:,:].flatten()
	AREAS=grid_areas.flatten()
	index=numpy.argsort(dataFLAT)
	data_sort=dataFLAT[index]	
	if option==1:
		AREAS_sort=AREAS[index]
		cum_AREAS=numpy.cumsum(AREAS_sort)
		threshInd=numpy.where(cum_AREAS <= PerOfAreaMin)[0][-1] 
		#print('MIN', threshInd)
		threshMIN=data_sort[threshInd]
		threshInd=numpy.where(cum_AREAS >= PerOfAreaMax)[0][0]
		threshMAX=data_sort[threshInd]
		#print('MAX', threshInd)
	if option==2:
		threshMIN=data_sort[int(PerOfAreaMin*len(data_sort))]
		#print('MIN', int(PerOfAreaMin*len(data_sort)))
		threshMAX=data_sort[int(PerOfAreaMax*len(data_sort))-1]
		#print('MAX', int(PerOfAreaMax*len(data_sort)))
	datawet=data.copy()
	datadry=data.copy()
	datawet.data[:,:]=ma.masked_less_equal(datawet.data[:,:],threshMAX)	
	datawet=calc_globalmean(datawet)
	datadry.data[:,:]=ma.masked_greater_equal(datadry.data[:,:],threshMIN)
	datadry=calc_globalmean(datadry)
	return datawet.data*daysinseas,datadry.data*daysinseas
def RunWetDryMonths(Filename,stdname,option,conversion,ClimPeriod):
	#take climatological mean off monthly values
	#Calcualte monthly wet and dry region means
	#Calultae 12-month running annual precip for wet & dry regions 
	OrigData=iris.load(Filename,stdname)[0]
	OrigData.data*=conversion
	addAuxCoords(OrigData)
	OrigData=OrigData.intersection(latitude=(latmin,latmax))
	grid_areas=iris.analysis.cartography.area_weights(OrigData[0])
	grid_areas/=numpy.sum(grid_areas)
	nmon=12
	climvals=numpy.zeros([12,2])
	for imn in range(nmon):
		#print('Month',imn)
		climdata=OrigData.copy()	
		climdata=climdata.extract( iris.Constraint(month_number=imn+1 ))
		if ClimPeriod: climdata=climdata.extract(iris.Constraint(year=lambda yr:stclim<=yr.point<=endclim ))
		wet=numpy.zeros([len(climdata.data)])
		dry=numpy.zeros([len(climdata.data)])
		for ii in range(len(climdata.data)):
			wet[ii],dry[ii]=CalcWetDry(climdata[ii],grid_areas,option,1)
			#print(wet[ii],dry[ii])	
		#print(wet)
		#print(numpy.mean(wet))	
		climvals[imn,0]=numpy.mean(wet)
		climvals[imn,1]=numpy.mean(dry)	
	#print(climvals)
	ALLwet=numpy.zeros( [len(OrigData.data[:,0,0])] )
	ALLdry=numpy.zeros( [len(OrigData.data[:,0,0])] ) 
	ANNyr=numpy.zeros(  [len(OrigData.data[:,0,0])] ) 
	for imn in range(len(OrigData.data[:,0,0])):
		ALLwet[imn],ALLdry[imn]=CalcWetDry(OrigData[imn],grid_areas,option,1)
		ANNyr[imn]=OrigData.coord('year').points[imn]+(OrigData.coord('month_number').points[imn]-0.5)/12.0
	for imn in range(len(OrigData.data[:,0,0])):
		ALLwet[imn]-=climvals[OrigData.coord('month_number').points[imn]-1,0]
		ALLdry[imn]-=climvals[OrigData.coord('month_number').points[imn]-1,1]
	ANNyrS,ALLwetS,ALLdryS=smooth_vals(ANNyr,ALLwet,ALLdry)
	return ANNyr,ALLwet,ALLdry,ANNyrS,ALLwetS,ALLdryS
#	
def calc_clim(yrdata,data,startyear,endyear):
	#Calculate mean of a specified period
	st= numpy.argwhere(yrdata>=startyear)[0][0]
	fi= numpy.argwhere(yrdata<endyear)[-1][0]
	#print(yrdata[st:fi+1])
	return numpy.mean(data[st:fi+1])	    

def getSamps(piCdata,NsampperpiC,LenC):
	#get NsampperpiC samples of length LenC from a given piControl simulation 
	Samps=numpy.zeros([NsampperpiC,LenC])
	for isamp in range(NsampperpiC):
		stindex=-(LenC*NsampperpiC)+(isamp*LenC)
		fiindex=-(LenC*NsampperpiC)+LenC+(isamp*LenC)
		if fiindex!=0:Samps[isamp,:]= ((piCdata[stindex:fiindex])-numpy.mean(piCdata[stindex:fiindex]))
		else:Samps[isamp,:]=((piCdata[stindex:])-numpy.mean(piCdata[stindex:]))
	return Samps
#################
#====================================
###Functions which calculate the TLS regression scaling factors
def calc_TLS(Z,xs):
	nsig=len(xs)
	beta=numpy.zeros(nsig)
	#DO TLS using results from a Singular Value Decomposition 
	U,s,V=numpy.linalg.svd(Z,full_matrices=False)
	V=numpy.transpose(V)
	for isig in range(nsig): beta[isig]= -V[isig,(nsig)]/V[(nsig),(nsig)]/xs[isig]
	# calculate noise reduced fingerprints
	v1=numpy.reshape(V[:,nsig],(nsig+1,1))
	v2=numpy.reshape(V[:,nsig],(1,nsig+1))
	zv=numpy.dot( Z,v1)
	zvv=numpy.dot(zv,v2)
	ztilde=numpy.transpose(Z-zvv)
	lam=s[nsig]**2#num/dem
	#
	return beta,ztilde,lam
def run_TLS(Zorig,xs,CNTRL):
	Z=Zorig.copy()
	#DO TLS using results from a Singular Value Decomposition 
	nsig=len(xs)
	Z=numpy.transpose(Z)
	beta,ztilde,lam=calc_TLS(Z,xs)
	#start uncertainty analysis
	nbetas=1000 # number of scaling factors to be calculated
	betas=numpy.zeros([nsig,nbetas])
	nc=len(CNTRL[:,0])
	for ic in range(nbetas):	
		#to calculate scaling factors add value from random samples of CNTRL to
		#noise reduced fingerprints ztilde
		zz=ztilde.copy()
		zz=zz.transpose()
		n=random.sample(range(nc),nsig+1)
		for ijk in range(nsig+1): 			
			zz[:,ijk]=zz[:,ijk]+((CNTRL[n[ijk],:]))
		#recalculate scaling factors
		beta_temp,z_nouse,lam_nouse=calc_TLS(zz,xs)
		betas[0,ic]=beta_temp[0]
		if nsig==2:betas[1,ic]=beta_temp[0]+beta_temp[1] 
	#find 5-95% range
	scalefact=numpy.zeros([nsig,3])
	for ijk in range(nsig):
		if ijk==0: bestbeta=beta[0]
		else: bestbeta=beta[0]+beta[1]
		scalefact[ijk,:]= [numpy.percentile(betas[ijk,:],5),bestbeta,numpy.percentile(betas[ijk,:],95)]
	return scalefact,ztilde,xs,lam
##################

def main(cfg):
    print('starting main program')
    logger = logging.getLogger(os.path.basename(__file__))
    matplotlib.use('Agg')
    plt.ioff() #Turn off interactive plotting.
    basename='WetDry_timeseries_scalefactors'
    plot_file = get_plot_filename(basename, cfg)
    logger.info("Plotting analysis results to %s", plot_file)
    ###########options for analysis###########
    global latmax,latmin,PerOfAreaMin,PerOfAreaMax,stclim,endclim,fiAnal,stAnal,option
    latmax=29.99
    latmin=-29.99
    PerOfAreaMin=1./3
    PerOfAreaMax=2./3.
    stclim=1988
    endclim=2019
    fiAnal=2019
    stAnal=1980
    option=1
    modconversion=86400
    NsampperpiC=10
    ##################################################
    # Read in observations
    preproc_dir = cfg["input_files"][0] # to get the metadata.yml in the preproc dir
    preproc_dir = os.path.dirname(preproc_dir) # to remove metadata.yml from the path 
    gpcp_filename = glob.glob(preproc_dir+"/*GPCP*")[0]
    obsANNyr,obswet,obsdry,obsANNyrS,obswetS,obsdryS=RunWetDryMonths(gpcp_filename,'precipitation_flux',option,modconversion,True)
    era5_filename = glob.glob(preproc_dir+"/*ERA5*")[0]
    eraANNyr,erawet,eradry,eraANNyrS,erawetS,eradryS=RunWetDryMonths(era5_filename,'precipitation_flux',option,modconversion,True)
    ##################################################
    # Get a description of the preprocessed data that we will use as input.
    input_data = cfg['input_data'].values()
    grouped_input_data = group_metadata(input_data, 'dataset', sort='ensemble')
    #print(grouped_input_data)
    grouped_obs_input_data = group_metadata(input_data, 'dataset')
    logger.info(
        "Group input data by model and sort by ensemble:"
        "\n%s", pformat(grouped_input_data))
    type (grouped_input_data)
    print (len(grouped_input_data))
    nmodel=len(grouped_input_data)
    model_names=[]

    firsttime=True
    count=0
    fig=plt.figure()
    grid = plt.GridSpec(2, 5, wspace=0.05, hspace=0.05)
    #Loop through all model simulations 
    #And calculate wet & dry regional timeseries
    for mm, dataset in enumerate(grouped_input_data):
        logger.info("*************** Processing model %s", dataset)
        if dataset!='ERA5' and dataset!='GPCP-SG': #ignore observations
           model_names.append(dataset)
           lbl=dataset
           grouped_model_input_data = group_metadata(
            grouped_input_data[dataset], 'exp', sort='ensemble')
           for exp in grouped_model_input_data:
              if exp=='historical-ssp245':
               logger.info("***** Processing experiment %s", exp)    
               grouped_exp_input_data = group_metadata(
                 grouped_model_input_data[exp], 'ensemble', sort='variable_group')
	            
               for ee, ensemble in enumerate(grouped_exp_input_data):
                   logger.info("** Processing ensemble %s", ensemble)
                   files=[]
                   for attributes in grouped_exp_input_data[ensemble]:
                       logger.info("Processing variable %s", attributes['variable_group'])
                       files.append(attributes['filename'])
                   modANNyr,modwet,moddry,modANNyrS,modwetS,moddryS=RunWetDryMonths(files[0],'precipitation_flux',option,modconversion,True)
                   if firsttime:
                      firsttime=False
                      MMMwet=numpy.zeros([len(modANNyr)])
                      MMMdry=numpy.zeros([len(modANNyr)])
                      MMMwetS=numpy.zeros([len(modANNyrS)])
                      MMMdryS=numpy.zeros([len(modANNyrS)])
                   plt.subplot(grid[0,0:4])
                   plt.plot(modANNyrS,modwetS,'lightblue')
                   plt.subplot(grid[1,0:4]) 
                   plt.plot(modANNyrS,moddryS,'pink')
                   MMMwet+=modwet
                   MMMdry+=moddry
                   MMMwetS+=modwetS
                   MMMdryS+=moddryS
                   count+=1		
    nsims=float(count)
    MMMwet/=nsims
    MMMdry/=nsims
    MMMwetS/=nsims
    MMMdryS/=nsims
    #			
    #Plot timeseries
    plt.subplot(grid[0,0:4])
    plt.title('Detection and attribution analysis of tropical precipitation')	
    plt.plot([1988,2021],[0,0],'k')
    plt.plot(modANNyrS,MMMwetS,'b')
    plt.plot(obsANNyrS,obswetS,'k')    
    plt.plot(eraANNyrS,erawetS,'grey')
    plt.ylabel('Annual precip\nin wet areas (mm)')
    plt.xticks([1990,2000,2010,2020], [' ',' ',' ',' '])
    plt.yticks([0,200,400,600,], ['0','200','400','600'])
    plt.ylim([-199,269])
    plt.xlim([1988,2021])    
    plt.plot([-1000,-900],[0,1],'k',label='GPCP')
    plt.plot([-1000,-900],[0,1],'grey',label='ERA5')
    #plt.plot([-1000,-900],[0,1],'b',label='Multi\nModel Mean')
    plt.plot([-1000,-900],[0,1],'lightblue',lw=6,label='CMIP6 Models')
    plt.legend(loc="upper left",fontsize=11,ncol=3,frameon=False)
    plt.gcf().text(0.02,0.85,'(a)') 
    plt.subplot(grid[1,0:4]) 
    plt.plot([1988,2021],[0,0],'k')  	
    plt.plot(modANNyrS,MMMdryS,'r')
    plt.plot(obsANNyrS,obsdryS,'k')    
    plt.plot(eraANNyrS,eradryS,'grey')
    plt.xlabel('Year')
    plt.ylabel('Annual precip\nin dry areas (mm)')
    plt.xticks([1990,2000,2010,2020], ['1990','2000','2010','2020'])
    plt.ylim([-20,37])
    plt.xlim([1988,2021]) 
    plt.plot([-1000,-900],[0,1],'k',label='GPCP')
    plt.plot([-1000,-900],[0,1],'grey',label='ERA5')
    #plt.plot([-1000,-900],[0,1],'r',label='Multi\nModel Mean')
    plt.plot([-1000,-900],[0,1],'pink',lw=6,label='CMIP6 Models')
    plt.legend(loc="upper left",fontsize=11,ncol=3,frameon=False)
    plt.gcf().text(0.02,0.45,'(b)')        
    ##########################################################################	

    #============================
    #Calculate the scalefactor
    startyear=stclim
    endyear=endclim
    #=================================================
    gap=50*' '
    for iobs in range(2):
    	if iobs==0:
    		ax2 = fig.add_axes([0.85, 0.51, 0.1, 0.35])
    		plt.title('GPCP')
    		plt.gcf().text(0.83,0.88,'(c)')
    	else:      
    		ax2 = fig.add_axes([0.85, 0.1, 0.1, 0.35])
    		plt.title('ERA5')
    		plt.gcf().text(0.83,0.47,'(d)') 
    		plt.ylabel(gap+'Scaling Factor\n'+gap+'combined wet & dry regions')
    	plt.ylim([-0.3,5.25])
    	plt.xlim([0,5])
    	plt.xticks([], [])
    	plt.plot([0,5],[1,1],'k:')
    	plt.plot([0,5],[0,0],'k')
    		
    	#unsmoothed
    	if iobs==0:
    		st= numpy.argwhere(obsANNyr>=startyear)[0][0]
    		yrobs=obsANNyr[st:]
    		wetobs=obswet[st:]-numpy.mean(obswet[st:])
    		dryobs=obsdry[st:]-numpy.mean(obsdry[st:])
    	if iobs==1:	
	    	st= numpy.argwhere(eraANNyr>=startyear)[0][0]
    		yrobs=eraANNyr[st:-1]
    		wetobs=erawet[st:-1]-numpy.mean(erawet[st:-1])
    		dryobs=eradry[st:-1]-numpy.mean(eradry[st:-1])
    	st= numpy.argwhere(modANNyr>=startyear)[0][0]
    	fi= st+len(yrobs)
    	yrmod=modANNyr[st:fi]
    	wetmod=MMMwet[st:fi]-numpy.mean(MMMwet[st:fi])
    	drymod=MMMdry[st:fi]-numpy.mean(MMMdry[st:fi])
    	nt=len(yrobs)
    	#Get piControl samples for the D&A
    	piCfiles=glob.glob("../../../preproc/*/pr/*piControl*.nc")
    	NpiC=len(piCfiles)
    	Nsamp=NpiC*NsampperpiC
    	LenC=nt#numpy.max([len(obswet),len(erawet)])
    	ControlsWet=numpy.zeros([Nsamp,LenC])
    	ControlsDry=numpy.zeros([Nsamp,LenC])
	#smoothed
    	if iobs==0:
    		st= numpy.argwhere(obsANNyrS>=startyear)[0][0]
    		yrobsS=obsANNyrS[st:]
    		wetobsS=obswetS[st:]-numpy.mean(obswetS[st:])
    		dryobsS=obsdryS[st:]-numpy.mean(obsdryS[st:])
    	if iobs==1:	
	    	st= numpy.argwhere(eraANNyrS>=startyear)[0][0]
    		yrobsS=eraANNyrS[st:-1]
    		wetobsS=erawetS[st:-1]-numpy.mean(erawetS[st:-1])
    		dryobsS=eradryS[st:-1]-numpy.mean(eradryS[st:-1])
    	st= numpy.argwhere(modANNyrS>=startyear)[0][0]
	#
    	fi= st+len(yrobsS)
    	yrmodS=modANNyrS[st:fi]
    	wetmodS=MMMwetS[st:fi]-numpy.mean(MMMwetS[st:fi])
    	drymodS=MMMdryS[st:fi]-numpy.mean(MMMdryS[st:fi])
	#
    	ntS=len(yrobsS)
    	#Get piControl samples for the D&A
    	piCfiles=glob.glob("../../../preproc/*/pr/*piControl*.nc")
    	NpiC=len(piCfiles)
    	Nsamp=NpiC*NsampperpiC
    	LenCS=ntS#numpy.max([len(obswet),len(erawet)])
    	ControlsWetS=numpy.zeros([Nsamp,LenCS])
    	ControlsDryS=numpy.zeros([Nsamp,LenCS])	
	#create control samples
    	count=0
    	for mm in range(NpiC):
    		files=piCfiles[mm]
    		logger.info("Processing piControl file %s", files)
    		piANNyr,picwet,picdry,piANNyrS,picwetS,picdryS=RunWetDryMonths(files,'precipitation_flux',option,modconversion,False)   
    		if len(picwet)>=NsampperpiC*LenC:
    			ControlsWet[count*NsampperpiC:(count+1)*NsampperpiC,:]=getSamps(picwet,NsampperpiC,LenC)
    			ControlsDry[count*NsampperpiC:(count+1)*NsampperpiC,:]=getSamps(picdry,NsampperpiC,LenC)
    			ControlsWetS[count*NsampperpiC:(count+1)*NsampperpiC,:]=getSamps(picwetS,NsampperpiC,LenCS)
    			ControlsDryS[count*NsampperpiC:(count+1)*NsampperpiC,:]=getSamps(picdryS,NsampperpiC,LenCS)			
    			count+=1

    	wetSamps=ControlsWet[:count*NsampperpiC,:]
    	drySamps=ControlsDry[:count*NsampperpiC,:]        
    	wetSampsS=ControlsWetS[:count*NsampperpiC,:]
    	drySampsS=ControlsDryS[:count*NsampperpiC,:]        	
    	#############
    	neff=nsims	
    	scennoise=1. / neff
    	xs=[numpy.sqrt(scennoise)]
    	nsig=len(xs)
    	#-----------
	#For regular and double TLS use smoothed timeseries to be consitent with Schurer et al 2020
    	Zmatrix=numpy.zeros([nsig+1,ntS*2])
    	nc=numpy.shape(wetSampsS)[0]
    	stdCont=numpy.zeros([nc])
    	for ic in range(nc): stdCont[ic]=numpy.std(wetSampsS[ic,:])
    	stdContwet=numpy.mean(stdCont)
    	for ic in range(nc): stdCont[ic]=numpy.std(drySampsS[ic,:])
    	stdContdry=numpy.mean(stdCont)
    	Zmatrix[-1,:]=numpy.concatenate((wetobsS/stdContwet,dryobsS/stdContdry))
    	Zmatrix[0,:]=numpy.concatenate((wetmodS/stdContwet,drymodS/stdContdry))/xs
    	#				
    	for ic in range(nc): 
    		wetSampsS[ic,:]/=stdContwet
    		drySampsS[ic,:]/=stdContdry		
    	scalefact,ztilde,xs,lam=run_TLS(Zmatrix,xs,numpy.concatenate((wetSampsS,drySampsS),axis=1))
    	print('Regular TLS')
    	print(scalefact)
    	plt.plot([2,2],[scalefact[0,0],scalefact[0,2] ],'g',lw=3)
    	plt.text(0.5,1.3,'Noise sampling',rotation=90,color='g') 
    	#-----------------
    	for ic in range(nc): wetSampsS[ic,:]*=numpy.sqrt(2)
    	for ic in range(nc): drySampsS[ic,:]*=numpy.sqrt(2)
    	scalefact,ztilde,xs,lam=run_TLS(Zmatrix,xs,numpy.concatenate((wetSampsS,drySampsS),axis=1))
    	print('Double TLS')
    	print(scalefact)
    	plt.plot([2,2],[scalefact[0,0],scalefact[0,2] ],'g',lw=1)
    	if iobs==0: 
    		plt.plot([2],  [scalefact[0,1]],marker='o',color='k')
    	else: 
    		plt.plot([2],  [scalefact[0,1]],marker='o',color='grey')   	
    	###BOOTSTRAP### use unsmooted values
    	nbetas=10000
    	betas=numpy.zeros([nbetas])
    	Zmatrix=numpy.zeros([nsig+1,nt*2])
    	nc=numpy.shape(wetSamps)[0]
    	stdCont=numpy.zeros([nc])
    	for ic in range(nc): stdCont[ic]=numpy.std(wetSamps[ic,:])
    	stdContwet=numpy.mean(stdCont)
    	for ic in range(nc): stdCont[ic]=numpy.std(drySamps[ic,:])
    	stdContdry=numpy.mean(stdCont)
    	Zmatrix[-1,:]=numpy.concatenate((wetobs/stdContwet,dryobs/stdContdry))
    	Zmatrix[0,:]=numpy.concatenate((wetmod/stdContwet,drymod/stdContdry))/xs
	#
    	newZ=numpy.zeros([nsig+1,nt*2])
    	for ii in range(nbetas):
    		ind = [0]*newZ.shape[1]
    		for y in range(0, newZ.shape[1]):
    			ind[y] = int(random.uniform(0, newZ.shape[1]))
    		newZ=Zmatrix[:,ind]
    		betas[ii],ztilde,lam=calc_TLS(numpy.transpose(newZ),xs)
    	betas=sorted(betas)
    	print('Bootstrap')
    	print(numpy.percentile(betas,5),numpy.percentile(betas,50),numpy.percentile(betas,95))
    	plt.plot([4.5,4.5],[numpy.percentile(betas,5),numpy.percentile(betas,95)],'m',lw=1)
    	if iobs==0: 
    		plt.plot([4.5],  [numpy.percentile(betas,50)],'k',marker='o')
    	else:
    		plt.plot([4.5],  [numpy.percentile(betas,50)],color='grey',marker='o')
    	plt.text(3,2.2,'Bootstrap',rotation=90,color='m')
    plt.savefig(plot_file)
    plt.close() 				

    ancestor_files = list(config['input_data'].keys())
    provenance_record = get_provenance_record(ancestor_files)

    logger.info("Recording provenance of %s:\n%s", plot_file,
                pformat(provenance_record))
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(plot_file, provenance_record)


if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)      
