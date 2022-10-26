#!/usr/bin/env python
#author @Marine Remaud

from .useful import *
from .LRU_map import *

import numpy as np
import os
from netCDF4 import Dataset
import datetime
import calendar
import pandas as pd
import copy
import xarray as xr
from sys import argv
import sys

name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})

def budget_lat(index_g,all_priors,rep_da):
    """
    all_priors: dictionnary with all the prior maps
    rep_da: output directory
    Produce the necdf-files that contain the optimized fluxes in kg/m2/s
    at the LMDz resolution (96x95)
    Produce one netcdf file per process and per year at a
    Monthly resolution only!
    """
    area_LMDZ=get_area()
    BUDGET_LAT={'HN':[],'HS':[],'T':[],'year':[],'DA':[],'parameter':[],'compound':[]}

    #Posterior budget per latitude band
    list_process=index_g.parameter.unique()
    list_process=[x for x in list_process if not "offset" in x]
    for pp in list_process:
     for cc in compound:
      name_var='flx_'+cc.name.lower()
      convert=10**-(12) if cc.name=="CO2" else 10**(-6)
      convert*=np.sign(index_g[index_g.parameter==pp]['post_'+cc.name].iloc[0])
      for yy in range(begy,endy+1):
       nbj_m=[calendar.monthrange(yy,mm)[1] for mm in range(1,13)]
       nbj_m=np.asarray(nbj_m)
       list_nc=os.listdir(mdl.storedir+rep_da)
       list_nc=[ff for ff in list_nc if ".nc" in ff]
       if "post_"+pp+cc.name+"_"+str(yy)+".nc" in list_nc:
        file_name=mdl.storedir+rep_da+"post_"+pp+cc.name+"_"+str(yy)+".nc"
       else:
        file_name=eval("Sources."+cc.name+"."+pp).file_name
        if eval("Sources."+cc.name+"."+pp).clim != 1: file_name=file_name.replace('XXXX',str(yy))
       tmp_array = xr.open_dataset(file_name,decode_times=False)
       if (tmp_array[name_var].ndim == 4): tmp_array=tmp_array.sum(dim='veget')
       index_hs=np.where(tmp_array.lat<-30)[0]
       index_hn=np.where(tmp_array.lat>30)[0]
       index_tr=np.where((tmp_array.lat>=-30)&(tmp_array.lat<=30))[0]
       budget_t=np.squeeze(tmp_array[name_var].values[:,:,:-1])*area_LMDZ[np.newaxis,:,:]
       budget_t=np.multiply(budget_t,nbj_m[:,np.newaxis,np.newaxis])*86400.
       budget_t=np.squeeze(np.sum(budget_t,axis=0))
       BUDGET_LAT['HN'].append(np.sum(budget_t[index_hn])*convert)
       BUDGET_LAT['HS'].append(np.sum(budget_t[index_hs])*convert)
       BUDGET_LAT['T'].append(np.sum(budget_t[index_tr])*convert)
       BUDGET_LAT['DA'].append('post')
       BUDGET_LAT['year'].append(yy)
       BUDGET_LAT['parameter'].append(pp)
       BUDGET_LAT['compound'].append(cc.name)
    #Prior budget per latitude band
    for cc in compound:
     for pp  in all_priors:
      if pp=="offset":continue
      convert=10**-(12) if cc.name=="CO2" else 10**(-6)
      for yy in range(begy,endy+1):
       index_g_pp=index_g[(index_g.year==yy)&(index_g.parameter==pp)].copy(deep=True)
       if index_g_pp['factor_'+cc.name].iloc[0]==0: continue 
       prior=np.copy(all_priors[pp])
       prior=prior[(yy-begy)*12:(yy-begy+1)*12,:,:,:-1]
       prior*=index_g_pp['factor_'+cc.name].iloc[0]
       nbj_m=[calendar.monthrange(yy,mm)[1] for mm in range(1,13)]
       nbj_m=np.asarray(nbj_m)
       list_nc=os.listdir(mdl.storedir+rep_da)
       list_nc=[ff for ff in list_nc if ".nc" in ff]
       if (len(np.shape(prior)) == 4): prior=np.squeeze(np.sum(prior,axis=1))
       budget_t=np.squeeze(prior)*area_LMDZ[np.newaxis,:,:]
       budget_t=np.multiply(budget_t,nbj_m[:,np.newaxis,np.newaxis])*86400.
       budget_t=np.squeeze(np.sum(budget_t,axis=0))
       BUDGET_LAT['HN'].append(np.sum(budget_t[index_hn])*convert)
       BUDGET_LAT['HS'].append(np.sum(budget_t[index_hs])*convert)
       BUDGET_LAT['T'].append(np.sum(budget_t[index_tr])*convert)
       BUDGET_LAT['DA'].append('prior')
       BUDGET_LAT['year'].append(yy)
       BUDGET_LAT['parameter'].append(pp)
       BUDGET_LAT['compound'].append(cc.name)

    BUDGET_LAT=pd.DataFrame(BUDGET_LAT)
    BUDGET_LAT=BUDGET_LAT[BUDGET_LAT.year>begy+1].groupby(["DA","parameter"]).mean().reset_index()
    return BUDGET_LAT
