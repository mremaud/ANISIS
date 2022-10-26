#!/usr/bin/env python
#author @Marine Remaud
#  A changer: peut etre supprimer les outliers (excedent trois fois la variance au prealable)

#Define and load H
from .useful import *
import numpy.ma as ma
import numpy as np
import os
from netCDF4 import Dataset
import datetime
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
 if mdl.only_season==0:
  for cc in compound:
   for stat in cc.stations: 
    #Load of the high frequency observations
    stat2 = copy.copy(stat[:3])
    #Raw data containing the whole variability
    tmp=pd.read_pickle(cc.file_obs+stat2+'.pkl')
    tmp=tmp.groupby("date").mean()

    #Fitting curve from the observation vector
    masque=(obs_vec['compound']==cc.name)&(obs_vec['stat']==stat)
    index_mask=obs_vec[masque].index
    index_summer=obs_vec[masque&(obs_vec.month>3)&(obs_vec.month<8)].index
    time_serie=obs_vec[masque].copy(deep=True)
    if time_serie.empty: continue
    time_serie["data"]=time_serie["obs"].copy(deep=True)
    time_serie=decomposition_ccgv(cc.name,time_serie,mdl.homedir)
    time_serie["date"]=time_serie.apply(lambda row:to_date(row),axis=1) 
    time_serie=time_serie.groupby("date").mean()
    time_serie=time_serie.resample("H").interpolate()
    time_serie=time_serie.reindex(tmp.index)
    time_serie.dropna(subset=['fit'],inplace=True)
    time_serie["orig"]=tmp.obs.copy(deep=True)
    time_serie.reset_index(inplace=True)
   # time=[datetime.datetime(int(serie_fit[ii,0]),int(serie_fit[ii,1]),int(serie_fit[ii,2])) for ii in range(len(serie_fit[:,0]))]
    #Residual are the errors
    erreur=np.var(time_serie['orig'].values-time_serie['fit'].values)
    if (stat2=="ALT")&(cc.name=="CO2"): erreur*=1.5
    if (stat2=="BRW")&(cc.name=="CO2"): erreur*=1.5

    if cc.name=="COS": erreur*=2
    #Add errors due to the diurnal cycle 
    if (stat2=="LEF")&(cc.name=="CO2"):
     erreur+=(6.5/500.*np.mean(time_serie.obs.values))**2
    elif (stat2=="LEF")&(cc.name=="COS"):
     erreur+=(6.5/500.*np.mean(time_serie.obs.values))**2
    elif (stat2=="HFM"):
     erreur*=2.
    elif (stat2=="NWR")&(cc.name=="CO2"):
     erreur*=2.
    elif (stat2=="SUM"):
     erreur*=1.5
    elif (stat2=="WIS"):
     erreur*=3.
    elif (stat2=="PSA")&(cc.name=="COS"):
     erreur*=3.
    elif (stat2=="SPO")&(cc.name=="COS"):
     erreur*=3
####EXPERIMENT REVIEW
    if stat2=="HFM": erreur+=(7.5/500.*np.mean(time_serie.obs.values))**2
    if (stat2=="MLO"): erreur+=(5./500.*np.mean(time_serie.obs.values))**2
#    if (stat2=="LEF")|(stat2=="WIS")|(stat2=="NWR"):
#     erreur/=2.
    ##=Final assignement####
    matrix_O[index_mask,index_mask]=np.copy(erreur)
    sig_O[index_mask]=np.copy(erreur)
 else:
  for cc in compound:
   for stat in cc.stations:
    stat2=stat[:3] 
    masque=(obs_vec['compound']==cc.name)&(obs_vec['stat']==stat)
    index_mask=obs_vec[masque].index
    time_serie=obs_vec[masque].copy(deep=True)
    sig_O[index_mask]=np.var(time_serie.res.values)*4
    #if cc.name== "CO2": sig_O[index_mask]*=3
    #if cc.name== "COS": sig_O[index_mask]=max(np.var(time_serie.res.values),(3.5*10**(-6))**2)
    #Add errors due to the diurnal cycle 
    #if (stat2=="LEF")&(cc.name=="CO2"):
    # sig_O[index_mask]+=(6.5/500.*np.mean(time_serie.orig.values))**2
    #elif (stat2=="LEF")&(cc.name=="COS"):
    # sig_O[index_mask]+=(6.5/500.*np.mean(time_serie.orig.values))**2
    # sig_O[index_mask]+=(15*10**(-6))**2
    #if (stat2=="NWR")&(cc.name=="CO2"):
    # sig_O[index_mask]+=(2.5/500.*np.mean(time_serie.orig.values))**2 
    #elif (stat2=="NWR")&(cc.name=="COS"):
    # sig_O[index_mask]+=(2.5/500.*np.mean(time_serie.orig.values))**2
    # sig_O[index_mask]+=(15*10**(-6))**2
    #if stat2=="HFM": sig_O[index_mask]+=(7.5/500.*np.mean(time_serie.orig.values))**2
    #if stat2=="MLO": sig_O[index_mask]+=(5./500.*np.mean(time_serie.orig.values))**2


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
 for pp in index_ctl_vec.parameter.unique():
  if pp == "offset": continue
  m1=(index_ctl_vec.parameter==pp).copy(deep=True)
  index_pp=index_ctl_vec[index_ctl_vec.parameter==pp].copy(deep=True)
  coef=getattr(Vector_control, pp).coef
  coef=coef[0]
  for vv in index_pp.PFT.unique():
   m2=m1&(index_ctl_vec.PFT==vv).copy(deep=True) if not np.isnan(vv) else m1.copy(deep=True)
   index_vv=index_pp[index_pp.PFT==vv] if not np.isnan(vv) else index_pp.copy(deep=True)
   for rr in index_vv.REG.unique():
    m3=m2&(index_ctl_vec.REG==rr).copy(deep=True)
    index_rr=index_ctl_vec[m3].copy(deep=True)
    ###Constant error bar
    nbm=len(index_rr.month.unique())
    YearlyPrior=index_rr.groupby('year').sum().reset_index().mean().prior
    YearlyPrior/=nbm # per month
    error_d=np.zeros(len(index_rr))
    for iv,erreur in enumerate(index_rr.prior.values):
      if not (index_rr.prior.values[iv] == 0): 
        error_d[iv]=YearlyPrior/index_rr.prior.values[iv]
    if hasattr(getattr(Vector_control,pp), 'coefPFT'): 
     #The error coef depends on the PFT
     coef=getattr(Vector_control, pp).coefPFT
     coef=coef[int(vv)-1]
   # if (vv == 15 ) &(pp=="Soil"):
   #  sig_B[index_rr.index.values]=(error_d*coef)**2
   # elif (vv == 15 ) &(pp=="Gpp"):
   #  ii_autumn=index_rr[(index_rr.month>7)&(index_rr.month<9)].index
   #  sig_B[index_rr.index.values]=(error_d*coef)**2
   #  sig_B[ii_autumn]/=3
   #   ii_spring=index_rr[(index_rr.month>4)&(index_rr.month<7)].index
   #  sig_B[ii_spring]*=2
   # elif (vv == 8 ) &(pp=="Gpp"):
   #  ii_autumn=index_rr[(index_rr.month>=7)&(index_rr.month<9)].index
   #  sig_B[index_rr.index.values]=(error_d*coef)**2
   #  sig_B[ii_autumn]/=10
   #  ii_spring=index_rr[(index_rr.month>4)&(index_rr.month<7)].index
   #  sig_B[ii_spring]*=2
   # elif (vv == 7 ) &(pp=="Gpp"):
   #  ii_autumn=index_rr[(index_rr.month>=7)&(index_rr.month<9)].index
   #  sig_B[index_rr.index.values]=(error_d*coef)**2
   #  sig_B[ii_autumn]/=10
   #  ii_spring=index_rr[(index_rr.month>4)&(index_rr.month<7)].index
   #  sig_B[ii_spring]*=2
   # elif (vv == 12 ) &(pp=="Soil"):
   #  sig_B[index_rr.index.values]=(error_d*coef)**2
    if (pp=="OceanOCS")&(rr!= "HN")&(rr!= "HS"):
     sig_B[index_rr.index.values]=(1.*coef*3.)**2
   #  sig_B[index_rr.index.values]=(error_d*coef)**2
   # elif (pp=="OceanOCS")&(rr== "HN"):
   # sig_B[index_rr.index.values]=(1.*coef)**2
    else:
     sig_B[index_rr.index.values]=(1.*coef)**2
   #if (pp=="Gpp")|(pp=="OceanOCS"): sig_B[index_vv.index.values]=(1.*coef)**2
 #Ajout de l offset
 if not index_ctl_vec[index_ctl_vec.parameter=="offset"].empty:
  prior_offset=index_ctl_vec[index_ctl_vec.parameter=="offset"]
  sig_B[prior_offset.index]=(offset_coef)**2
 #np.save(mdl.storedir+'C_D',matrix_B)
 return sig_B



