
import numpy as np 

def get_ipcc_cmap(field,nclrs):
# field is 'temperature' or 'precipitation'
# nclrs is the number of colors in the in colormap
	
	cmapnow=np.zeros((nclrs,4))

	if(field=='temperature' or field=='precipitation'):
		fn = '/pf/b/b381044/my_esmvt/diag_scripts/ar6ch6/ipccstylecolormaps_pr_tmp.txt'
	elif(field=='aerosols'):
		fn = '/pf/b/b381044/my_esmvt/diag_scripts/ar6ch6/chem_div_disc.txt'
	fp=open(fn,'r')

	alpha = 1.0

	nextline=0
	if(field=='temperature' or field=='precipitation'):
		for line in fp:
			#cols=line.split('\t')
			cols=line.split()
			head=cols[0].split('_')
			lnow=int(head[1])
			if(head[0]==field and lnow == nclrs):
				cmapnow[nextline,:] = [float(cols[1])/255.,float(cols[2])/255.,float(cols[3])/255.,alpha]
				nextline=nextline+1
	elif(field=='aerosols'):
		for line in fp:
			chemcols=line.split('_')
			if((len(chemcols)==3)):
				if(int(chemcols[2])==nclrs):
					for iclr in range(0,nclrs):
						line = fp.readline()
						chemcols=line.split()
						cmapnow[nextline,:] =  [float(chemcols[0])/255.,float(chemcols[1])/255.,float(chemcols[2])/255.,alpha]
						nextline = nextline + 1
	
	fp.close()

	return cmapnow

