#!/usr/bin/env python
# Translate the config.py into some dictionnary or pandas structure

from .LRU_map import *
from .useful import *

from calendar import isleap
import numpy as np
import os
from netCDF4 import Dataset
import datetime
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import matplotlib.pyplot as plt
import sys
import copy
import json
from netCDF4 import Dataset

def class_topd_obs():
  """
  Creation of an observation vector and a control vector 
  in the pandas format
  Loading the observation 
  Return : dataframe of the observation vector
  """
  #Loading the obs vector
  obs_vec=pd.DataFrame()
  for cc in compound:
   for stat in cc.stations:
    stat2 = copy.copy(stat[:3])
    tmp=pd.read_pickle(cc.file_obs+stat2+'.pkl')
    print(cc.file_obs+stat2+'.pkl')
    tmp.set_index('date',inplace=True)
    tmp=tmp.resample('D').mean().dropna()
    tmp.reset_index(inplace=True)
    tmp["obserror"]=1
    tmp=tmp[['date','obs','obserror','lat']]
    tmp['compound']=cc.name
    tmp['stat']=np.copy(stat)
    obs_vec=obs_vec.append(tmp.copy(deep=True),ignore_index=True,sort=None)
  obs_vec=obs_vec[(obs_vec['date'].dt.year>=begy) & (obs_vec['date'].dt.year<=endy)]
  
  #Average per week/ two week /month depending on the config.py
  resol_temp=compound[0].resolution
  obs_vec['week']=1
  adays=[1,9,17,25,32]
  for kk in range(4):
   mask=(obs_vec['date'].dt.day>=adays[kk])&(obs_vec['date'].dt.day<adays[kk+1])
   obs_vec.loc[mask,'week']=kk+1
  obs_vec['month']=obs_vec.date.dt.month
  obs_vec['year']=obs_vec.date.dt.year
  obs_vec=obs_vec.groupby(['compound','stat','year','month','week']).mean()
  obs_vec.reset_index(inplace=True)
  if resol_temp == '2W':
   obs_vec.loc[(obs_vec['week']==2),'week']=1
   obs_vec.loc[(obs_vec['week']==4),'week']=3
  elif resol_temp == 'M': 
   obs_vec['week']=1
  obs_vec=obs_vec.groupby(['compound','stat','year','month','week']).mean()
  obs_vec.reset_index(inplace=True)
  n_ligne=len(obs_vec)
  return obs_vec

def class_topd_obs_seasonal():
  """
  Creation of an observation vector and a control vector 
  in the pandas format
  Loading the observation 
  Return : dataframe of the observation vector
  """
  #Loading the obs vector
  obs_vec=pd.DataFrame()
  for cc in compound:
   for stat in cc.stations:
    stat2 = copy.copy(stat[:3])
    tmp=pd.read_pickle(cc.file_obs+stat2+'.pkl')
    tmp.set_index('date',inplace=True)
    tmp=tmp.resample('D').mean().dropna()
    tmp.reset_index(inplace=True)
    tmp["data"]=tmp["obs"].copy(deep=True)
    
    #Extraction of the seasonal cycle
    time_serie=decomposition_ccgv2(cc.name,tmp,mdl.homedir)
    time_serie.reset_index(inplace=True)
    tmp["res"]=time_serie['orig']-time_serie['fit']
    tmp["obs"]=time_serie['fit']-time_serie['trend']
    tmp["orig2"]=time_serie['orig'].copy(deep=True)
    tmp["obserror"]=1
    tmp=tmp[['date','obs','res','orig2','lat','obserror']]
    ##########
    tmp.insert(2, "compound",[cc.name for ii in range(len(tmp))] , True)
    tmp.insert(2, "stat",[stat for ii in range(len(tmp))] , True)
    obs_vec=obs_vec.append(tmp.copy(deep=True),ignore_index=True,sort=None)
  obs_vec=obs_vec[(obs_vec['date'].dt.year>=begy) & (obs_vec['date'].dt.year<=endy)]

  #Average per week/ two week /month depending on the config.py
  resol_temp=compound[0].resolution
  obs_vec['week']=1
  adays=[1,9,17,25,32]
  for kk in range(4):
   mask=(obs_vec['date'].dt.day>=adays[kk])&(obs_vec['date'].dt.day<adays[kk+1])
   obs_vec.loc[mask,'week']=kk+1
  obs_vec['month']=obs_vec.date.dt.month
  obs_vec['year']=obs_vec.date.dt.year
  obs_vec=obs_vec.groupby(['compound','stat','year','month','week']).mean()
  obs_vec.reset_index(inplace=True)
  if resol_temp == '2W':
   obs_vec.loc[(obs_vec['week']==2),'week']=1
   obs_vec.loc[(obs_vec['week']==4),'week']=3
  elif resol_temp == 'M':
   obs_vec['week']=1
  obs_vec=obs_vec.groupby(['compound','stat','year','month','week']).mean()
  obs_vec.reset_index(inplace=True)
  n_ligne=len(obs_vec)
  return obs_vec



def class_topd_ctl():
  """
  Creation of an observation vector and a control vector 
  in the pandas format
  1) Creation of the control vector
  2) Taking into account the unoptimized component
  """
  ###Control vector/ Calculate the horizontal dimension of the matrix H
  ctl_processus=[]
  index_ctl_vec={"parameter":[],"PFT":[],"REG":[],"month":[],"year":[],"week":[]}
  for cc in compound:
   index_ctl_vec["factor_"+cc.name]=[]
  nb_colonnes=0
  for attr,items in Vector_control.__dict__.items():
    if attr.startswith('__'): continue
    type_veget=range(1,num_pft+1) if (items.veget != "no") else [np.nan]
    ctl_processus.extend(items.groupe)
    #Temporal resolution of the fluxes to be optmized
    nbm=strings_to_month(items.resol)  #Number of months
    nbm=[np.nan] if np.isnan(nbm) else range(1,nbm+1)
    nbw=strings_to_week(items.resol)   #Number of weeks
    nbw= [np.nan] if np.isnan(nbw) else range(1,nbw+1)  #Number of weeks
    for yy in range(begy,endy+1):
     for vv in type_veget:
      #Define the region
      if not hasattr(items, 'pftreg'):
       list_reg=items.region
      elif (hasattr(items, 'pftreg')):
       list_reg=[]
       for nreg,lveg in items.pftreg.items():
        if vv in lveg: list_reg.append(nreg)
      for rr,nr in enumerate(list_reg):
        for mm in nbm:
         for ww in nbw:
          index_ctl_vec['year'].append(yy)
          index_ctl_vec['parameter'].append(attr)
          index_ctl_vec['REG']      .append(nr)
          index_ctl_vec['PFT']      .append(vv)
          index_ctl_vec['month']    .append(mm)
          index_ctl_vec['week']     .append(ww)
          for cc in compound:
           if (cc.name in items.compound):
            index_ctl_vec['factor_'+cc.name]   .append(1)
           else:
            index_ctl_vec['factor_'+cc.name]   .append(0)

  #Optimisation of the offset: add a supplementary column that contains 1
  #The offset represent the background concentration uniform all over the globe
  if only_season == 0: 
   for cc in compound:
    index_ctl_vec['year'].append(begy-1)
    index_ctl_vec['parameter'].append("offset")
    index_ctl_vec['REG'].append("GLOBE")
    index_ctl_vec['PFT'].append(np.nan)
    index_ctl_vec['month'].append(12) #Arbitraire
    index_ctl_vec['week'].append(4)   #Arbitraire
    index_ctl_vec['factor_'+cc.name].append(1)
    compound2=[xx for xx in compound if cc.name not in xx.name]
    for xx in compound2:
     index_ctl_vec['factor_'+xx.name].append(0)
  index_ctl_vec=pd.DataFrame(index_ctl_vec)
  #Special case for GPP and COS
  list_c=[cc.name for cc in compound]
  mask=(index_ctl_vec['parameter']=='Gpp')
  #Warning
  if (not index_ctl_vec[mask].empty)&('COS' in list_c):
   index_ctl_vec.loc[mask,'factor_COS']=index_ctl_vec[mask].apply(lambda row: factor_vcos(mdl.LRU_case,row['REG'],int(row['PFT']),row['year'],row['month']),axis=1)
  index_ctl_vec.sort_values(by=['parameter','PFT','REG'],inplace=True)
  index_ctl_vec.reset_index(drop=True,inplace=True)

  #Simulated concentration: add the contribution of the source (vs sink) which are not optimized
  #list proc: optimized fluxes
  #Warning: Gpp and Resp in unoptimized flux
  index_unopt={"parameter":[],"PFT":[],"REG":[],"month":[],"year":[],"week":[]}
  for cc in compound:
    index_unopt["factor_"+cc.name]=[]
  for compound_attr,compound_items in Sources.__dict__.items():
    if compound_attr.startswith('__'): continue
    for attr,items in compound_items.__dict__.items():
      if (attr.startswith('__'))|(attr in ctl_processus): continue  #if already in the control vector 
      #Temporal resolution
      nbm=strings_to_month(items.resol) if hasattr(items, 'property') else index_ctl_vec.month.unique()
      nbm=[np.nan] if (len(nbm[np.isnan(nbm)])!=0) else range(1,np.max(nbm)+1)
      nbw=strings_to_week(items.resol) if hasattr(items, 'property') else index_ctl_vec.week.unique()
      nbw= [np.nan] if (len(nbw[np.isnan(nbw)])!=0) else range(1,np.max(nbw)+1)  #Number of weeks
      for yy in range(begy,endy+1):
       for mm in nbm:
        for ww in nbw:
         index_unopt['year']     .append(yy)
         index_unopt['parameter'].append(attr)
         index_unopt['REG']      .append('GLOBE')
         index_unopt['PFT']      .append(np.nan)
         index_unopt['month']    .append(mm)
         index_unopt['week']    .append(ww)
         for cc in compound:
          if compound_attr == cc.name:
           index_unopt['factor_'+cc.name]   .append(1)
          else:
           index_unopt['factor_'+cc.name]   .append(0)

  #Offset
  if only_season == 1:
   for cc in compound:
    index_unopt['year'].append(begy-1)
    index_unopt['parameter'].append("offset")
    index_unopt['REG'].append("GLOBE")
    index_unopt['PFT'].append(np.nan)
    index_unopt['month'].append(12) #Arbitraire
    index_unopt['week'].append(4)   #Arbitraire
    index_unopt['factor_'+cc.name].append(1)
    compound2=[xx for xx in compound if cc.name not in xx.name]
    for xx in compound2:
     index_unopt['factor_'+xx.name].append(0)
  index_unopt=pd.DataFrame(index_unopt)

  #Calculation of the deposition velocity: unoptimized parameters
  if (not index_unopt.empty)&('COS' in list_c ):
   if (not index_unopt[index_unopt.parameter=='Gpp'].empty):
    mask=(index_unopt['parameter']=='Gpp')
    index_unopt.loc[mask,'factor_COS']=Sources.COS.Gpp*index_unopt[mask].apply(lambda row: factor_vcos(mdl.LRU_case,row['REG'],int(row['PFT']),row['year'],row['month']),axis=1)
  index_unopt.sort_values(by=['parameter','PFT','REG'],inplace=True)
  index_unopt.reset_index(drop=True,inplace=True)
  

  #Convert the PFT, month and year
  index_ctl_vec.loc[index_ctl_vec['PFT'].notnull(), 'PFT'] = index_ctl_vec.loc[index_ctl_vec['PFT'].notnull(), 'PFT'].astype(int)
  index_ctl_vec.loc[index_ctl_vec['month'].notnull(), 'month'] = index_ctl_vec.loc[index_ctl_vec['month'].notnull(), 'month'].astype(int)
  index_ctl_vec.loc[index_ctl_vec['year'].notnull(), 'year'] = index_ctl_vec.loc[index_ctl_vec['year'].notnull(), 'year'].astype(int)

  index_unopt.loc[index_unopt['month'].notnull(), 'month'] = index_unopt.loc[index_unopt['month'].notnull(), 'month'].astype(int)
  index_unopt.loc[index_unopt['year'].notnull(), 'year']   = index_unopt.loc[index_unopt['year'].notnull(), 'year'].astype(int)
  return index_ctl_vec,index_unopt






def get_sigmat():
  """
  Get the autocorrelation-coefficient (sigma-t) from the config file
  Return: sigma_t= dictionnary or None if the matrix B is chosen to be diagonal
  """

  sigma_t={}
  for attr,items in Vector_control.__dict__.items():
   if (attr.startswith('__')): continue
   if  hasattr(items, 'sigma_t'): 
    #One coefficient for the process
    sigma_t[attr]=items.sigma_t 
   elif hasattr(items, 'sigma_tPFT'):
    #One different coef for each PFT:
    #Number of coefficient = number of PFT
    for vv in range(mdl.num_pft):
     sigma_t[attr+'-PFT'+str(vv+1)]=items.sigma_tPFT[vv]
   else: 
    print("The prior error matrix is diagonal")
    continue
   
  if not sigma_t: 
   sigma_t=None
  elif only_season==0:
   sigma_t['offset']=1
  return sigma_t

def load_prior(index_unopt,only_season):
  """
  Stockage of all priors (open once the files)
  return a dictionnary
  """
  all_priors={}
  for attr,items in Vector_control.__dict__.items():
    if attr.startswith('__'): continue
    for pp in  items.groupe:
     comp=items.compound[:3]
     file_name=eval("Sources."+comp+"."+pp).file_name
     if eval("Sources."+comp+"."+pp).clim == 1:
      nc_fid = Dataset(file_name, 'r')
      prior_fluxtmp = nc_fid.variables["flx_"+comp.lower()][:].data
      lat_prior = nc_fid.variables["lat"][:]
      nc_fid.close()
      prior_flux=np.tile(prior_fluxtmp, ((endy-begy+1), 1,1))
     else:
      for yy in range(begy,endy+1):
       file_name2=file_name.replace('XXXX',str(yy))
       if yy== begy:
        nc_fid = Dataset(file_name2, 'r')
        prior_flux = nc_fid.variables["flx_"+comp.lower()][:].data
        lat_prior = nc_fid.variables["lat"][:]
        nc_fid.close()
       else:
        nc_fid = Dataset(file_name2, 'r')
        tmp = nc_fid.variables["flx_"+comp.lower()][:].data
        lat_prior = nc_fid.variables["lat"][:]
        nc_fid.close()
        prior_flux=np.copy(np.concatenate((prior_flux,tmp),axis=0))
     prior_flux=prior_flux*eval("Sources."+comp+"."+pp).sign
     if len(np.shape(prior_flux))==3 :prior_flux=prior_flux[:,np.newaxis,:,:]
     if lat_prior[0]<lat_prior[1]:
       print("Inverse lat for flux"),pp
       prior_flux=np.flip(prior_flux,axis=2) if len(np.shape(prior_flux))==4 else np.flip(prior_flux,axis=1)
     all_priors[attr]=np.copy(prior_flux) if pp == items.groupe[0] else np.add(all_priors[attr],prior_flux)
     all_priors[attr]=np.array(all_priors[attr],dtype='float64')
    #Conversion d unite
    all_priors[attr]*=getattr(Masse_molaire,"CO2")/getattr(Masse_molaire,comp)
  #Addition of the offset
  offset={'compound':[],'value':[]}
  for cc in compound:
    tmp_o=pd.read_pickle(cc.file_obs+'hemi.pkl')
    tmp_o=tmp_o[(tmp_o['date'].dt.year == begy)].iloc[0]
    offset['compound'].append(cc.name)
    if only_season:
     offset['value'].append(0)
    else:
     offset['value'].append(tmp_o.Global)
  all_priors['offset']=pd.DataFrame(offset)

  #Unoptimized priors: sommer sur les PFT la respiration ou la GPP
  if not index_unopt.empty:
   for compound_attr,compound_items in Sources.__dict__.items():
    if compound_attr.startswith('__'): continue
    for attr,items in compound_items.__dict__.items():
      if attr.startswith('__'): continue
      ligne_source=index_unopt[index_unopt.parameter==attr].copy(deep=True)
      if ligne_source.empty : continue
      if (ligne_source['factor_'+compound_attr].values[0] !=0) :
       file_name=eval("Sources."+compound_attr+"."+attr).file_name
       if eval("Sources."+compound_attr+"."+attr).clim == 1:
        prior_flux = xr.open_dataset(file_name,decode_times=False)
        prior_flux=prior_flux.astype(np.float64)
        prior_flux=np.tile(prior_flux["flx_"+compound_attr.lower()].values, ((endy-begy+1), 1,1))
       else:
        for yy in range(begy,endy+1):
         file_name2=file_name.replace('XXXX',str(yy))
         if yy== begy:
          prior_flux = xr.open_dataset(file_name2,decode_times=False)
          prior_flux=prior_flux.astype(np.float64)
          lat_prior=prior_flux.lat.values
          prior_flux=prior_flux["flx_"+compound_attr.lower()].values
          
         else:
          tmp = xr.open_dataset(file_name2,decode_times=False)
          prior_flux=np.concatenate((prior_flux,tmp["flx_"+compound_attr.lower()].values),axis=0)
      if len(np.shape(prior_flux))==4 :prior_flux=np.sum(prior_flux,axis=1)
      if len(np.shape(prior_flux))==3 :prior_flux=prior_flux[:,np.newaxis,:,:]
      if lat_prior[0]<lat_prior[1]:
       prior_flux=np.flip(prior_flux,axis=2)
      all_priors[attr]=np.array(prior_flux,dtype='float64')  
      all_priors[attr]*=getattr(Masse_molaire,"CO2")/getattr(Masse_molaire,compound_attr)

  return all_priors

def write_output(rep_da):
  """
  Create the file output.txt which contains the... 
  ...inversion set-ups and results (cost, gradient and x2)
  """
  rep_fig=mdl.storedir+rep_da
  list_compound=[cc.name for cc in compound]
  
  with open(rep_fig+'output.txt', 'w') as outfile:
     outfile.write("Inversion period :"+str(begy)+"-"+str(endy)+"\n")
     for cc in compound:
      outfile.write("Assimilated stations for :"+cc.name +"\n")
      outfile.write("     "+ json.dumps(cc.stations)+"\n")
     if ("COS" in list_compound)&("LRU_case" in globals()): 
      outfile.write("LRU for each PFT :"+"\n")
      for vv in range(num_pft):
        outfile.write("     "+table_LRU[vv][0]+': '+str(table_LRU[vv][LRU_case])+"\n")
     outfile.write("Optimized fluxes :"+"\n")
     for attr,items in Vector_control.__dict__.items():
      if attr.startswith('__'): continue
      outfile.write("* "+items.name+"\n")
      if  hasattr(items, 'region'):
       if items.region[0]!= "GLOBE": outfile.write("     Regions : "+json.dumps(items.region)+ "\n")
      elif hasattr(items, 'pftreg'):
       outfile.write("     Regions : "+json.dumps([rr for rr in items.pftreg])+ "\n")
      if "PFT" in items.veget:
       outfile.write("     Number of PFTs : "+str(num_pft)+"\n")
      if hasattr(items, 'coefPFT'):
       outfile.write("     Prior error (ratio of the prior) :"+json.dumps(items.coefPFT)+"\n")
      else:
       outfile.write("     Prior error coefficient :"+str(items.coef)+"\n")
      if hasattr(items, 'sigma_tPFT'):
       outfile.write("     Temporal autocorrelation length (in days): "+json.dumps(items.sigma_tPFT)+"\n")
      else:
       outfile.write("     Temporal autocorrelation length (in days): :"+str(items.sigma_t)+"\n")




