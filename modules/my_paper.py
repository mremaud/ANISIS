#!/usr/bin/env python
#author@Marine Remaud
#Purpose: paper plots

from .useful import *
from .LRU_map import *
from .extract_SIF import *

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
from sklearn.linear_model import LinearRegression



#Import the variables in the config.py
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
rep_fig=mdl.storedir+rep_da+'/FIG'
if not os.path.exists(rep_fig): os.system('mkdir '+rep_fig)

#Computation of the tendancy of the COS and CO2 atmospheric concentrations at one site
station="ALT"
obs_vec["sim"]=sim_0
lr = LinearRegression()
for cc in compound: 
 conc=obs_vec[(obs_vec.stat=="ALT")&(obs_vec["compound"]==cc.name)].groupby("year").mean().reset_index()
 print(conc.obs)
 lr.fit(conc['year'].values.reshape(-1,1),conc['obs'].values.reshape(-1,1))
 print(cc.name,": Observed concentration: ", lr.coef_[0],lr.intercept_)
 lr.fit(conc['year'].values.reshape(-1,1),conc['sim'].values.reshape(-1,1))
 print(cc.name,"- Simulated concentrations: ", lr.coef_[0],lr.intercept_)

#Find the correlation between two consecutive months for each components: 
Matrix_B=np.load(mdl.storedir+rep_da+"corr.npy")
Matrix_B=np.multiply(Matrix_B,np.outer(1./sig_B**0.5,1./sig_B**0.5))
for pp in index_ctl_vec.parameter.unique():
  index_pp=index_ctl_vec[index_ctl_vec.parameter==pp].copy(deep=True)
  for vv in index_pp.PFT.unique():
   index_vv=index_pp[index_pp.PFT==vv].copy(deep=True).index
   if np.isnan(vv): index_vv=index_pp.copy(deep=True).index
   print(pp,vv,np.round(Matrix_B[index_vv[0],index_vv[1]]**0.5,2))


####Fluxes following Latitudinal bands#############
flux_ocean=["OceanOCS"]
flux_cont=["Gpp","Soil","Ant","Biomass1"]
nby=(endy-begy+2)
plt.rcParams.update({'font.size': 12})
f,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,3))
#Ocean
#####Posterior fluxes#####################
post_ocean   =np.zeros((96))
prior_ocean   =np.zeros((96))

for iff,flux in enumerate(flux_ocean):
  prior_tmp    =np.zeros((96))
  post_tmp     =np.zeros((96))
  for yy in range(begy+1,endy+1):
   tmp=xr.open_dataset(mdl.storedir+rep_da+"/post_"+flux+"COS_"+str(yy)+".nc",decode_times=False)
   tmp=tmp.mean("lon").mean("time")
   prior_tmp+=tmp.flx_cos_prior.values
   post_tmp+=tmp.flx_cos.values
  post_ocean+=post_tmp/nby
  prior_ocean+=prior_tmp/nby
axes.plot(tmp.lat.values,post_ocean,color="lightblue",label="Ocean: post")
axes.plot(tmp.lat.values,prior_ocean,color="lightblue",linestyle="--",label="Ocean: prior")

post_cont   =np.zeros((96))
prior_cont   =np.zeros((96))
for iff,flux in enumerate(flux_cont):
  prior_tmp    =np.zeros((96))
  post_tmp     =np.zeros((96))
  for yy in range(begy+1,endy+1):
   tmp=xr.open_dataset(mdl.storedir+rep_da+"/post_"+flux+"COS_"+str(yy)+".nc",decode_times=False)
   if len(np.shape(tmp.flx_cos.values))==4: tmp=tmp.sum("veget")
   tmp=tmp.mean("lon").mean("time")
   prior_tmp+=tmp.flx_cos_prior.values
   post_tmp+=tmp.flx_cos.values
  post_cont+=post_tmp/nby
  prior_cont+=prior_tmp/nby
axes.plot(tmp.lat.values,post_cont,color="brown",label="Continent: post")
#axes.fill_between(tmp.lat.values, post_min_cont, post_max_cont,color="brown",alpha=0.2)
axes.plot(tmp.lat.values,prior_cont,color="brown",linestyle="--",label="Continent: prior")
#axes.fill_between(tmp.lat.values, prior_min_cont, prior_max_cont,color="brown",alpha=0.2,linestyle="--")
axes.set_ylabel("COS flux [kg/m2/s]")
axes.set_xlabel("LATITUDE")
axes.set_xticks(np.arange(-90,100,30))
axes.set_xticklabels(["90S","60S","30S","0","30N","60N","90N"])
axes.set_xlim(-90,90)
plt.ylim(-1.1*10**(-13),1.1*10**(-13))
box = axes.get_position()
axes.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.7])
axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=True, ncol=2,fontsize=12)
plt.savefig(rep_fig+"/Gradient_flux.pdf",format="pdf",bbox_inches='tight')

#########################################################################################################
######NPP
flux_cont=["Gpp","Resp"]
nby=(endy-begy+2)
plt.rcParams.update({'font.size': 12})
f,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,3))
post_cont   =np.zeros((96))
prior_cont   =np.zeros((96))
for iff,flux in enumerate(flux_cont):
  prior_tmp    =np.zeros((96))
  post_tmp     =np.zeros((96))
  for yy in range(begy+1,endy+1):
   tmp=xr.open_dataset(mdl.storedir+rep_da+"/post_"+flux+"CO2_"+str(yy)+".nc",decode_times=False)
   if len(np.shape(tmp.flx_co2.values))==4: tmp=tmp.sum("veget")
   tmp=tmp.mean("lon").mean("time")
   prior_tmp+=tmp.flx_co2_prior.values
   post_tmp+=tmp.flx_co2.values
  post_cont+=post_tmp/nby
  prior_cont+=prior_tmp/nby
axes.plot(tmp.lat.values,post_cont,color="brown",label="Respiration + GPP: post")
axes.plot(tmp.lat.values,prior_cont,color="brown",linestyle="--",label="Respiration + GPP: prior")


axes.set_ylabel("CO2 flux [kg/m2/s]")
axes.set_xlabel("LATITUDE")
axes.set_xticks(np.arange(-90,100,30))
axes.set_xticklabels(["90S","60S","30S","0","30N","60N","90N"])
axes.set_xlim(-90,90)

#####GLOBAL CARBON BUDGET
area=get_area()
flux_cont=["Gpp","Resp"]
nby=(endy-begy)
plt.rcParams.update({'font.size': 12})
f,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,3))
post_cont   =np.zeros((3,nby))
prior_cont   =np.zeros((3,nby))
iy=0
for yy in range(begy+1,endy+1):
 for iff,flux in enumerate(flux_cont):
   tmp=xr.open_dataset(mdl.storedir+rep_da+"/post_"+flux+"CO2_"+str(yy)+".nc",decode_times=False)
   latitude=tmp.lat.values
   if len(np.shape(tmp.flx_co2.values))==4: tmp=tmp.sum("veget")
   prior_tmp=tmp.flx_co2_prior.values[:,:,:-1]*area[np.newaxis,:,:]
   prior_tmp=prior_tmp[:,latitude>50,:]
   prior_tmp=np.sum(prior_tmp,axis=1)
   prior_tmp=np.sum(prior_tmp,axis=1)
   prior_tmp=prior_tmp*[calendar.monthrange(yy,ii)[1] for ii in range(1,13)]*86400
   prior_cont[0,iy]+=np.sum(prior_tmp)
   prior_tmp=tmp.flx_co2_prior.values[:,:,:-1]*area[np.newaxis,:,:]
   prior_tmp=prior_tmp[:,(latitude<30)&(latitude>-30),:]
   prior_tmp=np.sum(prior_tmp,axis=1)
   prior_tmp=np.sum(prior_tmp,axis=1)
   prior_tmp=prior_tmp*[calendar.monthrange(yy,ii)[1] for ii in range(1,13)]*86400
   prior_cont[1,iy]+=np.sum(prior_tmp)
   prior_tmp=tmp.flx_co2_prior.values[:,:,:-1]*area[np.newaxis,:,:]
   prior_tmp=prior_tmp[:,latitude<-30,:]
   prior_tmp=np.sum(prior_tmp,axis=1)
   prior_tmp=np.sum(prior_tmp,axis=1)
   prior_tmp=prior_tmp*[calendar.monthrange(yy,ii)[1] for ii in range(1,13)]*86400
   prior_cont[2,iy]+=np.sum(prior_tmp)

   post_tmp=tmp.flx_co2.values[:,:,:-1]*area[np.newaxis,:,:]
   post_tmp=post_tmp[:,latitude>50,:]
   post_tmp=np.sum(post_tmp,axis=1)
   post_tmp=np.sum(post_tmp,axis=1)
   post_tmp=post_tmp*[calendar.monthrange(yy,ii)[1] for ii in range(1,13)]*86400
   post_cont[0,iy]+=np.sum(post_tmp)
   post_tmp=tmp.flx_co2.values[:,:,:-1]*area[np.newaxis,:,:]
   post_tmp=post_tmp[:,(latitude<30)&(latitude>-30),:]
   post_tmp=np.sum(post_tmp,axis=1)
   post_tmp=np.sum(post_tmp,axis=1)
   post_tmp=post_tmp*[calendar.monthrange(yy,ii)[1] for ii in range(1,13)]*86400
   post_cont[1,iy]+=np.sum(post_tmp)
   post_tmp=tmp.flx_co2.values[:,:,:-1]*area[np.newaxis,:,:]
   post_tmp=post_tmp[:,latitude<-30,:]
   post_tmp=np.sum(post_tmp,axis=1)
   post_tmp=np.sum(post_tmp,axis=1)
   post_tmp=post_tmp*[calendar.monthrange(yy,ii)[1] for ii in range(1,13)]*86400
   post_cont[2,iy]==np.sum(post_tmp)
 iy+=1


#####PLOT the prior oceanic fluxes
from calendar import isleap
from copy import copy

plt.rcParams.update({'font.size': 12})
f,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,3))
dir_flx="/home/surface1/mremaud/COS/INPUT_sflx/Ocean/Sinikka/"
dms=xr.open_dataset(dir_flx+"sflx_odms_2010_dyn.nc",decode_times=False)
dms=dms.mean("lon").mean("time")
ocs=xr.open_dataset(dir_flx+"flx_odirect_2010_dyn.nc",decode_times=False)
ocs=ocs.mean("lon").mean("time")
cs2=xr.open_dataset(dir_flx+"flx_ocs2_2010_dyn.nc",decode_times=False)
cs2=cs2.mean("lon").mean("time")
axes.plot(dms.lat.values,dms.flx_cos.values,color="brown",label="DMS")
axes.plot(cs2.lat.values,cs2.flx_cos.values,color="darkgreen",label="CS2")
axes.plot(ocs.lat.values,ocs.flx_cos.values,color="lightblue",label="Direct COS")
axes.set_ylabel("COS flux [kg/m2/s]")
#axes.set_xlabel("LATITUDE")
axes.set_xticks(np.arange(-90,100,30))
axes.set_xticklabels(["90S","60S","30S","0","30N","60N","90N"])
axes.set_xlim(-90,90)
plt.ylim(-1.*10**(-14),4*10**(-14))
plt.legend()
plt.show()
plt.savefig("Prior_ocean.pdf",format="pdf",bbox_inches='tight')




f,axes=plt.subplots(nrows=1,ncols=2)
for nc,cc in  enumerate(compound):
   table={'Source':[],'Year':[],'Prior':[],'Post':[]}
   table=pd.DataFrame(table)
   index_cc=index_g.groupby(['parameter','year']).sum().reset_index()
   #Remove the first year
   index_cc=index_cc[index_cc.year>=begy+1]
   index_cc=index_cc[index_cc['prior_'+cc.name]!=0]   
   table['parameter']=index_cc['parameter'].copy(deep=True)
   table['Year']=index_cc['year'].values
   table.loc[table.Source!="offset","Prior"]=index_cc['prior_'+cc.name]*dico[cc.name][0]
   table.loc[table.Source!="offset","Post"] =index_cc['post_'+cc.name]*dico[cc.name][0]
   table.loc[table.Source=="offset","Post"] =index_cc['post_'+cc.name]*dico[cc.name][2]
   table.loc[table.Source=="offset","Prior"]=index_cc['prior_'+cc.name]*dico[cc.name][2]
   table=table.round(1)
   for pp in table.parameter.unique():
     if  (pp  in index_unopt.parameter.unique())&(cc.name=="COS"):
      table=table.drop(index=table[(table.parameter==pp)&(table.Year>begy+1)].index)
     
   table['parameter']=table.apply(lambda row: to_name(row),axis=1)
   table["Source"]=table["parameter"]; del table["parameter"]
   if cc.name=="COS": table.loc[table.Source=="GPP","Source"]="vegetation"

   table=table[['Source','Year','Prior','Post']]
   render_mpl_table(table,axes[nc], col_width=200, row_height=10.0, font_size=9,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,ax=None)
    
   axes[nc].set_title(cc.name+" fluxes "+dico[cc.name][1])

