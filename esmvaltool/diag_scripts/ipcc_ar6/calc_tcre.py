"""Python example diagnostic."""
import logging
import os
from pprint import pformat

import iris

import sys, numpy, scipy.stats, math
import netCDF4
import detatt_mk as da
import matplotlib.pyplot as plt

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


def compute_diagnostic(filename):
    """Compute an example diagnostic."""
    logger.debug("Loading %s", filename)
    cube = iris.load_cube(filename)

    logger.debug("Running example computation")
    return cube.collapsed('time', iris.analysis.MEAN)


def plot_diagnostic(cube, basename, provenance_record, cfg):
    """Create diagnostic data and plot it."""
    diagnostic_file = get_diagnostic_filename(basename, cfg)

    logger.info("Saving analysis results to %s", diagnostic_file)
    iris.save(cube, target=diagnostic_file)

    if cfg['write_plots'] and cfg.get('quickplot'):
        plot_file = get_plot_filename(basename, cfg)
        logger.info("Plotting analysis results to %s", plot_file)
        provenance_record['plot_file'] = plot_file
        quickplot(cube, filename=plot_file, **cfg['quickplot'])

    logger.info("Recording provenance of %s:\n%s", diagnostic_file,
                pformat(provenance_record))
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(diagnostic_file, provenance_record)


# Following two functions for blending and masking modified from Cowtan 2015.
# Calculate blended temperatures using general methods
# Usage:
#  python ncblendmask.py <mode> tas.nc tos.nc sic.nc sftof.nc [Had4.nc] > blend.temp
#  <mode> is one of xxx, mxx, xax, max, xxf, mxf, xaf, maf
#  see README for more details
# Nathan Gillett - Adapted from ncblendmask-nc4.py from Cowtan 2015
# http://www-users.york.ac.uk//~kdc3/papers/robust2015/methods.html


# cell areas, used for calculating area weighted averages
def areas( grid ):
  area = grid*[0.0]
  for i in range(grid):
    area[i] = ( ( math.sin(math.radians(180.0*(i+1)/grid-90.0)) -
                  math.sin(math.radians(180.0*(i  )/grid-90.0)) ) /
                math.sin(math.radians(180.0/grid)) )
  return area


def ncblendmask_esmval(options,sic_file,tas_file,tos_file,sftlf_file,had4_file,dec_warming):
# MAIN PROGRAM

# m = mask
# a = blend anomalies
# f = fix ice
# (use x for none)

  # read tas.nc
  nc = netCDF4.Dataset(tas_file, "r")
  print(nc.variables.keys(),file=sys.stderr)
  lats1 = nc.variables["lat"][:]
  lons1 = nc.variables["lon"][:]
  year=nc.variables["year"][:]
  y0=year[0]#NPG - Added since existing y0 definition below did not work on ESMValTool preprocessed files.   
  tas = numpy.ma.filled(nc.variables["tas"][:,:,:],-1.0e30)
  nc.close()

  # read tos.nc
  nc = netCDF4.Dataset(tos_file, "r")
  print(nc.variables.keys(),file=sys.stderr)
  lats2 = nc.variables["lat"][:]
  lons2 = nc.variables["lon"][:]
  tos = numpy.ma.filled(nc.variables["tos"][:,:,:],-1.0e30)
#  y0 = int(nc.variables["time"][:][0]/10000)
  nc.close()

  # read sic.nc
  nc = netCDF4.Dataset(sic_file, "r")
  print(nc.variables.keys(),file=sys.stderr)
  lats3 = nc.variables["lat"][:]
  lons3 = nc.variables["lon"][:]
  #Use siconca if it exists, otherwise use siconc.
  if 'siconca' in nc.variables:
    sic = numpy.ma.filled(nc.variables["siconca"][:,:,:],-1.0e30)
  else:
    sic = numpy.ma.filled(nc.variables["siconc"][:,:,:],-1.0e30)  
    nc.close()

  # read sftlf.nc (NPG - Changed from sftof, because of better data availability for sftlf).
  nc = netCDF4.Dataset(sftlf_file, "r")
  print(nc.variables.keys(),file=sys.stderr)
  lats4 = nc.variables["lat"][:]
  lons4 = nc.variables["lon"][:]
  sftof = 1-numpy.ma.filled(nc.variables["sftlf"][:,:],-1.0e30) #NPG - added '1-' to use lf.
  nc.close()



  if 'm' in options:
    # read HadCRUT4 data as mask
    nc = netCDF4.Dataset(had4_file, "r")
    print(nc.variables.keys(),file=sys.stderr)
    lats5 = nc.variables["latitude"][:]
    lons5 = nc.variables["longitude"][:]
    had4_tas = nc.variables["temperature_anomaly"][:,:,:]
    cvgmsk = numpy.ma.filled(nc.variables["temperature_anomaly"][:,:,:],-1.0e30)
    nc.close()
    #Simple regridding to agree with ESMValTool output, HadCRUT4 longitudes start from -177.5.
    regrid_index=list(range(int(lons5.shape[0]*0.5),lons5.shape[0]))+list(range(int(lons5.shape[0]*0.5)))
    lons5=lons5[regrid_index]
    had4_tas=had4_tas[:,:,regrid_index]
    cvgmsk=cvgmsk[:,:,regrid_index]


  print (tas.shape,file=sys.stderr)
  print (tos.shape,file=sys.stderr)
  print (sftof.shape,file=sys.stderr)
  print (sic.shape,file=sys.stderr)

  sic = sic[0:tas.shape[0],:,:]
  print (sic.shape,file=sys.stderr)


  # dates
  dates = (numpy.arange(tas.shape[0])+0.5)/12.0 + y0
  print (dates,file=sys.stderr)

  # force missing cells to be open water/land and scale if stored as percentage
  sic[sic<  0.0] = 0.0
  sic[sic>100.0] = 0.0
  if numpy.max(sic)>90.0: sic = 0.01*sic

  sftof[sftof<  0.0] = 0.0
  sftof[sftof>100.0] = 0.0
  if numpy.max(sftof)>90.0: sftof = 0.01*sftof

  print ("tos ", numpy.min(tos), numpy.max(tos), numpy.mean(tos),file=sys.stderr)

  print ("sic ", numpy.min(sic), numpy.max(sic), numpy.mean(sic),file=sys.stderr)
  print ("sftof ", numpy.min(sftof), numpy.max(sftof), numpy.mean(sftof),file=sys.stderr)

  # optional fixed ice mode
  if 'f' in options:
    # mask all cells with any ice post 1961
    for m0 in range(0,len(dates),12):
      if dates[m0] > 1961: break
      print (m0, dates[m0],file=sys.stderr)
    for i in range(sic.shape[1]):
      for j in range(sic.shape[2]):
        for m in range(12):
          cmax = sic[m0+m::12,i,j].max()
          if cmax > 0.01:
            sic[m::12,i,j] = 1.0

  # combine land/ice masks
  for m in range(sic.shape[0]):
    sic[m,:,:] = (1.0-sic[m,:,:])*sftof

  print (sic.shape)

  # print mask
  s = ""
  sicmax = numpy.max(sic)
  for i in range(sic.shape[1]-1,0,-sic.shape[1]//25):
    for j in range(0,sic.shape[2],sic.shape[2]//50):
      s += ".123456789#"[int(10*sic[-1,i,j]/sicmax)]
    s += "\n"
  print (s, "\n",file=sys.stderr)
  # print tos mask
  s = ""
  for i in range(tos.shape[1]-1,0,-tos.shape[1]//25):
    for j in range(0,tos.shape[2],tos.shape[2]//50):
      s += "#" if 100 < tos[-1,i,j] < 500 else "."
    s += "\n"
  print (s, "\n",file=sys.stderr)
  # print cvg mask
  if 'm' in options:
    s = ""
    for i in range(cvgmsk.shape[1]-1,0,-cvgmsk.shape[1]//25):
      for j in range(0,cvgmsk.shape[2],cvgmsk.shape[2]//50):
        s += "#" if -100 < cvgmsk[-1,i,j] < 500 else "."
      s += "\n"
    print (s, "\n",file=sys.stderr)

  # deal with missing tos through sic
  for m in range(sic.shape[0]):
    sic[m,tos[m,:,:]<-500.0] = 0.0
    sic[m,tos[m,:,:]> 500.0] = 0.0

  # baseline and blend in the desired order
  if 'a' in options:

    # prepare missing
    for m in range(sic.shape[0]):
#      tos[m,tos[m,:,:]<-500.0] = numpy.nan
      tos[m,abs(tos[m,:,:])> 500.0] = numpy.nan 

    # baseline
    mask = numpy.logical_and( dates > 1961, dates < 1991 )
    base = tas[mask,:,:]
    for m in range(12):
      norm = numpy.mean(base[m::12,:,:],axis=0)
      tas[m::12,:,:] = tas[m::12,:,:] - norm
    base = tos[mask,:,:]
    for m in range(12):
      norm = numpy.nanmean(base[m::12,:,:],axis=0)
      tos[m::12,:,:] = tos[m::12,:,:] - norm
    # blend
    for m in range(sic.shape[0]):
      tos[m,:,:] = tas[m,:,:]*(1.0-sic[m,:,:])+tos[m,:,:]*(sic[m,:,:])

  else:

    # blend
    for m in range(sic.shape[0]):
      tos[m,:,:] = tas[m,:,:]*(1.0-sic[m,:,:])+tos[m,:,:]*(sic[m,:,:])
    # baseline
    mask = numpy.logical_and( dates > 1961, dates < 1991 )
    base = tas[mask,:,:]
    for m in range(12):
      norm = numpy.mean(base[m::12,:,:],axis=0)
      tas[m::12,:,:] = tas[m::12,:,:] - norm
    base = tos[mask,:,:]
    for m in range(12):
      norm = numpy.mean(base[m::12,:,:],axis=0)
      tos[m::12,:,:] = tos[m::12,:,:] - norm

  print (sic.dtype, tos.dtype,file=sys.stderr)

  # deal with any remaining nans
  for m in range(sic.shape[0]):
    msk = numpy.isnan(tos[m,:,:])
    tos[m,msk] = tas[m,msk]
  # calculate area weights
  w = numpy.zeros_like(tas)
  wm = numpy.zeros_like(tas)
  a = areas(sftof.shape[0])
  for m in range(w.shape[0]):
#    for i in range(w.shape[1]):
      for j in range(w.shape[2]):
        w[m,:,j] = a[:]

  wm=w
  if 'm' in options: wm[ cvgmsk[0:wm.shape[0],:,:] < -100 ] = 0.0
  print (w[0,:,:],file=sys.stderr)
  print (wm[0,:,:],file=sys.stderr)
  diag_name='dec_mean_gmst'
  # calculate diagnostic
  diag=calc_diag(tos,wm,diag_name) #Diagnostic for attribution analysis.
  dec_warming.append(calc_dec_warming(tas,w)) #Diagnose SAT warming with global coverage for attributable trends.
  had4_diag=calc_diag(had4_tas[0:tos.shape[0],:,:],wm,diag_name)
  return (diag,had4_diag)

def calc_diag(tos,wm,diag_name):
  #compute diagnostic based on masked/blended temperatures.
  if diag_name=='dec_mean_gmst':
    ndec=math.ceil(tos.shape[0]/120) #Round up number of decades.
    diag=numpy.zeros(ndec)
    gmst_mon=numpy.zeros(tos.shape[0])
  # calculate temperatures
    for m in range(tos.shape[0]):
      s = numpy.sum( wm[m,:,:] )
      gmst_mon[m] = numpy.sum( wm[m,:,:] * tos[m,:,:] ) / s
#    print (gmst_mon)
    for m in range(ndec):
      diag[m]=numpy.mean(gmst_mon[m*120:(m+1)*120]) #Note - will calculate average over incomplete final decade.
    diag=diag-numpy.mean(diag) #Take anomalies over whole period.
  else:
    print ('Diagnostic ',diag_name,' not supported')
    exit ()
  return diag

def calc_dec_warming(tas,w):
  gmt_mon=numpy.zeros(tas.shape[0])
  # calculate 2010-2019 mean relative to 1850-1900, assuming data starts in 1850.
  # If last decade is incomplete, just computes mean from available data.
  for m in range(tas.shape[0]):
    s = numpy.sum( w[m,:,:] )
    gmt_mon[m] = numpy.sum( w[m,:,:] * tas[m,:,:] ) / s
#  print (gmt_mon)
  return (numpy.mean(gmt_mon[(2010-1850)*12:(2020-1850)*12])-numpy.mean(gmt_mon[0:(1901-1850)*12]))


def main(cfg):
    """Compute the time average for each input dataset."""
    # Get a description of the preprocessed data that we will use as input.
    input_data = cfg['input_data'].values()
#    had4_file='/mnt/lustre02/work/bd0854/DATA/ESMValTool2/OBS/Tier2/HadCRUT4/OBS_HadCRUT4_ground_1_Amon_tas_185001-201812.nc'
    had4_file='/mnt/lustre02/work/bd0854/DATA/ESMValTool2/RAWOBS/Tier2/HadCRUT4/HadCRUT.4.6.0.0.median.nc'
    sftlf_file='/pf/b/b380746/CNRM-CM6-1-5x5-sftlf.nc' #Hard-coded path to sftlf file for CNRM-CM6 on a 5x5 grid. (Can't input through pre-processor at the moment. Update with sftlf for each model through preprocessor later.)
    grouped_input_data = group_metadata(
        input_data, 'dataset', sort='ensemble')
    logger.info(
        "Group input data by model and sort by ensemble:"
        "\n%s", pformat(grouped_input_data))
    type (grouped_input_data)
    print (len(grouped_input_data))
    nmodel=len(grouped_input_data)
    experiments=['historical-ssp245','hist-nat','hist-GHG','hist-aer','hist-CO2']
    nexp=len(experiments)
    print ('Number of experiments', nexp)
    # Loop over variables/datasets in alphabetical order
#   Define variables for D&A analysis
    ldiag=17 #length of diagnostic,hard-coded for the moment.
    anom_max=500 #arbitrary max size for number of anomalies.
    mean_diag=numpy.zeros((ldiag,nexp,nmodel))
    mean_dec_warming=numpy.zeros((nexp,nmodel))
    anom=numpy.zeros((ldiag,anom_max))
    anom_index=0
    ens_sizes=numpy.zeros((nexp,nmodel))
    years=list(range(1855,2025,10)) #Used for plotting.

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
                (exp_diags[:,ee],had4_diag)=ncblendmask_esmval('max', files[0],files[1],files[2],sftlf_file,had4_file,dec_warming)
                exp_dec_warming[ee]=dec_warming[0]
                plt.plot(years,exp_diags[:,ee],color='C'+str(mm),linewidth=1)

            mean_diag[:,experiment,mm]=numpy.mean(exp_diags,axis=1)
            mean_dec_warming[experiment,mm]=numpy.mean(dec_warming)
            if nens>1: anom[:,anom_index:anom_index+nens-1]=(exp_diags[:,0:nens-1]-mean_diag[:,experiment:experiment+1,mm])*((nens/(nens-1))**0.5) #Intra-ensemble anomalies for use as pseudo-control. Only use nens-1 ensemble members to ensure independence, and inflate variance by sqrt (nens / (nens-1)) to account for subtraction of ensemble mean.
            anom_index=anom_index+nens-1
    anom=anom[:,0:anom_index]
    att_out={}
    print ('*******')
    print (type(grouped_input_data))
#    grouped_input_data=extract(['CanESM5'], grouped_input_data)
    for mm, dataset in enumerate(grouped_input_data):
        if mm == 1:
            break #Just use first model for D&A.
        (xr,yr,cn1,cn2)=da.reduce_dim(mean_diag[:,[0,4],mm],had4_diag[:,None],anom[:,0:int(anom_index/2)],anom[:,int(anom_index/2):anom_index])

        att_out[dataset]=da.tls(xr,yr,ne=ens_sizes[[0,4],mm],cn1=cn1,cn2=cn2,flag_2S=1)
        print (att_out[dataset]['betaCI'][0,:]*(mean_dec_warming[0,mm]-mean_dec_warming[4,mm]))
        print (att_out[dataset]['betaCI'][1,:]*mean_dec_warming[4,mm])
        if mm == 0:
            label1='OTH'
            label2='CO2'
        else:
            label1=None
            label2=None
 
        plt.plot([mm+0.9,mm+0.9],numpy.transpose(att_out[dataset]['betaCI'][0,:]*(mean_dec_warming[0,mm]-mean_dec_warming[4,mm])),color='red',linewidth=4,label=label1)
        plt.plot([mm+1.1,mm+1.1],numpy.transpose(att_out[dataset]['betaCI'][1,:]*mean_dec_warming[4,mm]),color='green',linewidth=4,label=label2)
    obs_warming=had4_diag[16]-numpy.mean(had4_diag[0:5])
    print (obs_warming)
    print ('had4diag')
    print (had4_diag)
    plt.plot([0,3],[obs_warming,obs_warming],color='black',linewidth=1,label='Had4 GMST')
    plt.plot([0,3],[0,0],color='black',linewidth=1,ls='--')
    plt.axis([0,nmodel,-1,2])
    plt.xticks([1],['CanESM5'], size='small')
    plt.xlabel('Model')
    plt.ylabel('Attributable warming in GSAT 2010-2018 vs 1850-1900 in K') 
    plt.legend(loc="upper right")
    plt.show()

    for mm, dataset in enumerate(grouped_input_data):
        if mm == 0:
            label1='OTH'
            label2='CO2'
        else:
            break
            label1=None
            label2=None
        plt.plot([mm+0.9,mm+0.9],numpy.transpose(att_out[dataset]['betaCI'][0,:]),color='red',linewidth=4,label=label1)
        plt.plot([mm+1.1,mm+1.1],numpy.transpose(att_out[dataset]['betaCI'][1,:]),color='green',linewidth=4,label=label2)
    plt.plot([0,3],[0,0],color='black',linewidth=1,ls='--')
    plt.plot([0,3],[1,1],color='black',linewidth=1,ls='-')
    plt.axis([0,nmodel,-1,2])
    plt.xticks([1],['CanESM5'], size='small')
    plt.xlabel('Model')
    plt.ylabel('Regression coefficient for 1850-2018 GMSAT')
    plt.legend(loc="upper left")
    plt.show()



    print (att_out)
    print (mean_dec_warming[0,:])
    print (mean_diag[:,0,:])
    years=list(range(1855,2025,10))
    plt.plot(years,anom[:,0],color="Gray",label='Pseudo-control')
    for ann in range(1,anom_index):
        plt.plot(years,anom[:,ann],color="Gray")
    for mm, dataset in enumerate(grouped_input_data):
        if mm == 0:
            myls='-'
        else:
            break
            myls=':'
        plt.plot(years,mean_diag[:,0,mm],color='red',linewidth=4,label=dataset+'_ALL',ls=myls)
        plt.plot(years,mean_diag[:,4,mm],color='green',linewidth=4,label=dataset+'_CO2',ls=myls)
    plt.plot(years,had4_diag,color="Black",linewidth=4,label='HadCRUT4')
    plt.legend(loc="upper left")
    plt.xlabel('Year')
    plt.ylabel('Temperature anomaly (K)')
    plt.show()

if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
