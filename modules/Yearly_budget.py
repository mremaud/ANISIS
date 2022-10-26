#!/usr/bin/env python
#author@Marine Remaud
#Purpose: paper plots


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
from sys import argv
import sys
import seaborn as sns


#Import the variables in the config.py
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]

  """
  Aim: Draw plots of the posterior, prior and observed atmospheric concentration
  Arguments: 
  obs_vec: observation vector
  sim_0 : prior concentration
  sim_opt: optimized simulation
  repa_da: directory of the experiment
  """ 
#Repertory in which are saved the plots
rep_fig=mdl.storedir+rep_da+'/FIG'
if not os.path.exists(rep_fig): os.system('mkdir '+rep_fig)

dico={'CO2':[],'COS':[]} 
dico['CO2']=[1,"ppm"]
dico['COS']=[10**6,"ppt"]
list_CO2=index_ctl_vec[index_ctl_vec["factor_CO2"]!=0].parameter.unique()
list_COS=index_ctl_vec[index_ctl_vec["factor_COS"]!=0].parameter.unique()

index_ctl_vec["REGION"]="Tropics"
index_ctl_vec.loc[index_ctl_vec.PFT==7,"REGION"]="Boreal"
index_ctl_vec.loc[index_ctl_vec.PFT==8,"REGION"]="Boreal"
index_ctl_vec.loc[index_ctl_vec.PFT==9,"REGION"]="Boreal"
index_ctl_vec.loc[index_ctl_vec.PFT==15,"REGION"]="Boreal"
index_ctl_vec.loc[index_ctl_vec.PFT==4,"REGION"]="Temperate"
index_ctl_vec.loc[index_ctl_vec.PFT==5,"REGION"]="Temperate"
index_ctl_vec.loc[index_ctl_vec.PFT==10,"REGION"]="Temperate"
index_ctl_vec.loc[index_ctl_vec.PFT==12,"REGION"]="Temperate"
index_ctl_vec.loc[index_ctl_vec.PFT==6,"REGION"]="Temperate"
index_ctl_vec=index_ctl_vec.groupby(["parameter","year","REGION"]).sum().reset_index()
index_ctl_vec=index_ctl_vec[index_ctl_vec.parameter!="offset"]
index_ctl_vec["post_CO2"]=index_ctl_vec["post_CO2"]-index_ctl_vec["prior_CO2"]
fig,ax=plt.subplots(1)

for pp in ["Gpp","Resp"]:
  index_pp=index_ctl_vec[index_ctl_vec.parameter==pp]
  for region in index_pp.REGION.unique():
   index_pp[index_pp.REGION==region].plot(x="year",y="post_CO2",ax=ax,label=region)

fig,ax=plt.subplots(1)
for pp in list_COS:
  index_pp=index_ctl_vec[index_ctl_vec.parameter==pp]
  for region in index_pp.REGION.unique():
   index_pp[index_pp.REGION==region].plot(x="year",y="post_COS",ax=ax,label=pp+region)


