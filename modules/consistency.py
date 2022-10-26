#!/usr/bin/env python
#author@Marine Remaud
#Script created the 20th of march
#Diagnostics aiming at evaluating the inversion results
#and tuning the error coefficients
import numpy.ma as ma
import numpy as np
import os
from netCDF4 import Dataset
import datetime
from .useful import *
from .LRU_map import *
from .build_C_m import *
import calendar
import pandas as pd
import copy
import xarray as xr
import math
from sys import argv
import sys
from scipy import signal
from scipy.stats import kurtosis,skew
#Import the variables in the config.py
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]



def consistency_diag(index_ctl_vec,obs_vec,sim_0,sim_opt,sig_B,sigma_t,Bpost,rep_da):
  """
  Desroziers et al., 2015
  Arguments: 
  - index_ctl_vec: control vector
  - obs_vec: observations
  - sim_0 : prior trajectory
  - sim_opt: optimized trajectory
  - sig_B : diagonal of the error covariance matrix
  - Bpost: Posterior error covariance matrix
  

  """

  rep_fig=mdl.storedir+rep_da+'/FIG/'
  detrend=0
  monthly=0
 
  G=np.load(mdl.storedir+"matrix_G.npy")
  R=np.load(mdl.storedir+'C_O.npy')

  #Get B
  if sigma_t is not None:
   B=build_C_m(index_ctl_vec,sig_B,sigma_t,False)
  else: 
   B=np.zeros((len(sig_B),len(sig_B)))
   np.fill_diagonal(B,sig_B)

  #Monthly average
  obs_vec['date']=obs_vec.apply(lambda row: to_date(row),axis=1)  
  obs_vec['post']=sim_opt
  obs_vec['prior']=sim_0
  if monthly:
   obs_vec=obs_vec.groupby(['compound','stat',"year","month"]).mean().reset_index()
   obs_vec["week"]=np.nan
   obs_vec['date']=obs_vec.apply(lambda row: to_date(row),axis=1)
  if detrend:
   for cc in compound:
    for stat in obs_vec.stat.unique():
     mask=(obs_vec.stat==stat)&(obs_vec["compound"]==cc.name)
     obs_tmp=obs_vec[mask].copy(deep=True)
     obs_tmpdt=obs_tmp.set_index("date").resample("D").mean().interpolate(how="linear")
     obs_tmpdt['post']=signal.detrend(obs_tmpdt.post.values)
     obs_tmpdt['obs']=signal.detrend(obs_tmpdt.obs.values)
     obs_tmpdt['prior']=signal.detrend(obs_tmpdt.prior.values)
     obs_tmp=obs_tmpdt.loc[obs_vec[mask].date]
     
     obs_vec.loc[mask,"prior"]=obs_tmp['prior'].values
     obs_vec.loc[mask,"obs"]=obs_tmp['obs'].values
     obs_vec.loc[mask,"post"]=obs_tmp['post'].values

 
  #diagnosed uncertainties (right side)
  d_s0=np.multiply((obs_vec["post"].values-obs_vec["prior"].values),(obs_vec["obs"].values-obs_vec["post"].values))
  d_s1=np.multiply((obs_vec["post"].values-obs_vec["prior"].values),(obs_vec.obs.values-obs_vec["prior"].values))
  d_s2=np.multiply((obs_vec.obs.values-obs_vec["post"].values),(obs_vec.obs.values-obs_vec["prior"].values))
  d_s3=np.multiply((obs_vec.obs.values-obs_vec["prior"].values),(obs_vec.obs.values-obs_vec["prior"].values))
 
  #Assigned uncertainties (left side)
  a_s0=np.matmul(G,Bpost)
  a_s0=np.matmul(a_s0,G.T).diagonal()

  a_s1=np.matmul(G,B)
  a_s1=np.matmul(a_s1,G.T).diagonal()

  a_s2=R.diagonal()

  a_s3=a_s1+R.diagonal()
  
  #Inovation
  IB=obs_vec["obs"].values-obs_vec["prior"].values
  
  dico={"CO2":[],"COS":[]}
  dico["CO2"]=[1,"ppm"]
  dico["COS"]=[10**12,"ppt"]
    
  #Diagnostics chevallier et al., 2017
  nbr=len(obs_vec["compound"].unique())
  for cc in [COS,CO2]:
   plt.figure()
   f,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,10))
   obs_cc=obs_vec[(obs_vec["compound"]==cc.name)].copy(deep=True)
   #Distribution of the innovation per compound: mean,skewness, kurtosis
   print(cc.name, "skewness :",round(skew(IB[obs_cc.index]),1),"kurtosis :",round(kurtosis(IB[obs_cc.index]),1))
   list_stat=obs_cc.stat.unique()
   list_stat=[ss for ss in list_stat]
   colors = plt.cm.jet(np.linspace(0,1,len(list_stat)))
   plt.suptitle(cc.name)
   plt.subplots_adjust(hspace=0.4)
   for ist,stat in enumerate(list_stat):
    i_stat=obs_cc[obs_cc.stat==stat].index
    #Subplot 1 : innovation statistics
    abs1=np.sqrt(np.abs(d_s3[i_stat])*dico[cc.name][0])
    ord1=np.sqrt(np.abs(a_s3[i_stat])*dico[cc.name][0])
    ax[0,0].scatter(abs1,ord1,label=stat[:3],s=1,color=colors[ist])
    x = np.linspace(*ax[0,0].get_xlim())
    if ist==1 :ax[0,0].plot(x, x,color="k")
    ax[0,0].set_ylabel("Assigned $\sigma$ ("+dico[cc.name][1]+")")
    ax[0,0].set_xlabel("Diagnosed $\sigma$ ("+dico[cc.name][1]+')')
    ax[0,0].set_title("(a) Innovation ")
    if cc.name=="COS":
     ax[0,0].set_ylim(0,30) #60)
     ax[0,0].set_xlim(0,350) #60)
    elif cc.name =="CO2":
     ax[1,1].set_ylim(0,8) #10)
     ax[1,1].set_xlim(0,30) #10)
    #Distribution of the innovation per compound and station: mean,skewness, kurtosis
    print(cc.name, stat, "skewness, ",round(skew(IB[i_stat]),1),"kurtosis, :",round(kurtosis(IB[i_stat]),1))

    #Subplot 2 : error statistics
    abs2=d_s0[i_stat]*dico[cc.name][0]
    ord2=a_s0[i_stat]*dico[cc.name][0]
    ineg=np.where(abs2>0)[0]
    abs2=abs2[ineg]; ord2=ord2[ineg]
    ineg=np.where(ord2>0)[0]
    abs2=np.sqrt(abs2[ineg]); ord2=np.sqrt(ord2[ineg])
    
    ax[1,0].scatter(abs2,ord2,label=stat[:3],s=1,color=colors[ist])
    ax[1,0].set_ylabel("Assigned $\sigma$ ("+dico[cc.name][1]+")")
    ax[1,0].set_xlabel("Diagnosed $\sigma$ ("+dico[cc.name][1]+')')
    ax[1,0].set_title("(b) Analysis ")
    x = np.linspace(*ax[1,0].get_xlim())
    if ist==1: ax[1,0].plot(x, x,color='k')
    if cc.name == "COS":
     ax[1,0].set_ylim(0,5) #15)
     ax[1,0].set_xlim(0, 150 ) #15)
    elif cc.name =="CO2":
     ax[1,1].set_ylim(0,1) #10)
     ax[1,1].set_xlim(0,14) #10)

    #Subplot 3: background 
    abs3=np.sqrt(np.abs(d_s1[i_stat])*dico[cc.name][0])
    ord3=np.sqrt(np.abs(a_s1[i_stat])*dico[cc.name][0])
    ax[0,1].scatter(abs3,ord3,label=stat[:3],s=1,color=colors[ist])
    ax[0,1].set_ylabel("Assigned $\sigma$ ("+dico[cc.name][1]+")")
    ax[0,1].set_xlabel("Diagnosed $\sigma$ ("+dico[cc.name][1]+')')
    ax[0,1].set_title("(c) Background")
    x = np.linspace(*ax[0,1].get_xlim())
    if ist==1: ax[0,1].plot(x, x,color="k")
    if cc.name=="COS":
     ax[0,1].set_ylim(0,25) #30)
     ax[0,1].set_xlim(0,350 ) #30)
    elif cc.name =="CO2":
     ax[0,1].set_ylim(0,10) #10)
     ax[0,1].set_xlim(0,20) #10)

    #Subplot 4: observation
    abs4=d_s2[i_stat]*dico[cc.name][0]
    ord4=a_s2[i_stat]*dico[cc.name][0]
    ineg=np.where(abs4>0)[0]
    abs4=abs4[ineg]; ord4=ord4[ineg]
    ineg=np.where(ord4>0)[0]
    abs4=np.sqrt(abs4[ineg]); ord4=np.sqrt(ord4[ineg])
    print("Selected values, ",len(abs4),len(d_s2[i_stat]))
    ax[1,1].scatter(abs4,ord4,label=stat[:3],s=1,color=colors[ist])
    ax[1,1].set_ylabel("Assigned $\sigma$ ("+dico[cc.name][1]+")")
    ax[1,1].set_xlabel("Diagnosed $\sigma$ ("+dico[cc.name][1]+")")
    ax[1,1].set_title("(d) Observations")
    ax[1,1].legend(fontsize=9)
    x = np.linspace(*ax[1,1].get_xlim())
    if ist==1: ax[1,1].plot(x, x,color="k")
    if cc.name =="COS":
     ax[1,1].set_ylim(0,50)   #60)
     ax[1,1].set_xlim(0,175)   #60)
    elif cc.name =="CO2":
     ax[1,1].set_ylim(0,13) #   10)
     ax[1,1].set_xlim(0,15)#  10)
    plt.savefig(rep_fig+"consistency"+cc.name+"-wt.pdf",format="pdf")






