#!/usr/bin/env python
#author @Marine Remaud
#  A changer: peut etre supprimer les outliers (excedent trois fois la variance au prealable)

#Define and load H
import numpy.ma as ma
import numpy as np
import os
from netCDF4 import Dataset
import datetime
from useful import *
import calendar
import pandas as pd
import copy
import xarray as xr
import math
import statsmodels.api as smf
from sys import argv
from dateutil.relativedelta import relativedelta
from scipy.optimize import curve_fit
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})

def define_O(obs_vec):
 """
 Matrix of observation errors : diagonal
 1 - Apply the ccgvu routine to find the smooth function
 2 - Substraction of the smooth function to the raw measurements
 3 - Half of the variance of the residuals
 obs_vec: observation vector
 """
 #Initialization
 matrix_O=np.zeros((len(obs_vec),len(obs_vec)))
 sig_O   =np.zeros(len(obs_vec))

 for cc in compound:
  for stat in cc.stations: 
    #Load of the high frequency observations
    stat2 = copy.copy(stat[:3])
    print cc.file_obs+stat2+'.pkl'
    tmp=pd.read_pickle(cc.file_obs+stat2+'.pkl')
    tmp.set_index('date',inplace=True)

    #Fitting curve from the observation vector
    masque=(obs_vec['compound']==cc.name)&(obs_vec['stat']==stat)
    index_mask=obs_vec[masque].index
    time_serie=obs_vec[masque].copy(deep=True)
    time_serie['frac']=time_serie.apply(lambda row: fraction_an(row),axis=1)
    file_out=mdl.homedir+'modules/tmp-ccgvu.txt'
    os.system('rm -f '+file_out)
    np.savetxt(file_out,time_serie[['frac','obs']].values)
    file_fit=mdl.homedir+'modules/tmp-fit.txt' 
    #Model: without cal and ori
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
   # time_serie=pd.DataFrame()
    time_serie["fit"]=np.loadtxt(file_fit)[:,5]
    time_serie["date"]=time_serie.apply(lambda row:to_date(row),axis=1) 
    time_serie.set_index("date",inplace=True)
    time_serie=time_serie.resample("H").interpolate()
    time_serie=time_serie.loc[tmp.index]
    time_serie.dropna(subset=['fit'],inplace=True)
    time_serie["orig"]=tmp.obs.copy(deep=True)
   # time=[datetime.datetime(int(serie_fit[ii,0]),int(serie_fit[ii,1]),int(serie_fit[ii,2])) for ii in range(len(serie_fit[:,0]))]
    #Residual are the errors
    erreur=np.var(time_serie['orig'].values-time_serie['fit'].values)/2. 
    #Add errors due to the diurnal cycle 
    #if stat2=="LEF": erreur=erreur+(12.5/500.*np.mean(time_serie.obs.values))**2
    #if stat2=="HFM": erreur=erreur+(7.5/500.*np.mean(time_serie.obs.values))**2
    #if stat2=="NWR": erreur=erreur+(5./500.*np.mean(time_serie.obs.values))**2
    #if stat2=="MLO": erreur=erreur+(5./500.*np.mean(time_serie.obs.values))**2
    matrix_O[index_mask,index_mask]=np.copy(erreur)
    sig_O[index_mask]=np.copy(erreur)

 np.save(mdl.storedir+'sig_O',sig_O)
 np.save(mdl.storedir+'C_O',matrix_O)
 return sig_O

def define_B(index_ctl_vec):
 """
 Matrix of prior errors : diagonal
 Coefficient par region: preciser le script
 """
 #matrix_B=np.zeros((len(index_ctl_vec),len(index_ctl_vec)))
 sig_B=np.zeros(len(index_ctl_vec))
 #for dd in range(len(index_ctl_vec)):
 # line_d=index_ctl_vec.iloc[dd]
 # error_d=np.copy(line_d.prior)
 # if line_d.parameter != 'offset':
 #   coef=getattr(Vector_control, line_d.parameter).coef
 #   region=getattr(Vector_control, line_d.parameter).region
 #   coef=coef[region==line_d.REG]
 #   if hasattr(getattr(Vector_control,line_d.parameter), 'coefPFT'): 
 #    #The error coef depends on the PFT
 #    coef=getattr(Vector_control, line_d.parameter).coefPFT
 #    coef=coef[int(line_d.PFT)-1]
 # else:
 #   coef=offset_coef
 for pp in index_ctl_vec.parameter.unique():
  if pp == "offset": continue
  index_pp=index_ctl_vec[index_ctl_vec.parameter==pp].copy(deep=True)
  coef=getattr(Vector_control, pp).coef
  coef=coef[0]
  for vv in index_pp.PFT.unique():
   index_vv=index_pp[index_pp.PFT==vv] if not np.isnan(vv) else index_pp.copy(deep=True)
   nbm=len(index_vv.month.unique())
   error_d=index_vv.groupby('year').sum().reset_index().mean().prior
   error_d/=nbm # per month
   if hasattr(getattr(Vector_control,pp), 'coefPFT'): 
    #The error coef depends on the PFT
    coef=getattr(Vector_control, pp).coefPFT
    coef=coef[int(vv)-1]
   sig_B[index_vv.index.values]=(error_d*coef)**2
   if (pp=="Gpp")|(pp=="OceanOCS"): sig_B[index_vv.index.values]=(index_vv.prior*coef)**2
 #Ajout de l offset
 prior_offset=index_ctl_vec[index_ctl_vec.parameter=="offset"]
 sig_B[prior_offset.index]=(prior_offset.prior.values*offset_coef)**2
 #np.save(mdl.storedir+'C_D',matrix_B)
 return sig_B



