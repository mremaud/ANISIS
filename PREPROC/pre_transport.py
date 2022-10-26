#!/usr/bin/env python
#author @Marine Remaud
#Calculate the g(Fprior): transport of the prior fluxes
#NOT USED ANYMORE
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import os
import datetime
import calendar
import pandas as pd
import copy
import math
from sys import argv
import sys
import matplotlib.pyplot as plt
from useful import *
#################PARAMETER############################################################################

beg_y=2010
end_y=2013
#Simulation parameter
is_sim=1 
dir_CO2='/home/satellites16/mremaud/'
dir_COS='/home/satellites16/mremaud/'
process={'CO2':[],'COS':[]}
process['CO2']     =['ocean_pos2_co2','ocean_neg2_co2','fossil2_co2', 'resp2_co2','gppSIF_co2','fire2_co2']
process['CO2_sign']=[   1           ,    -1         ,      1     ,     1     ,   -1    , 1         ]

process['COS']     =['OH','odms_wOH','ocs2_wOH','odirect_S+_wOH','odirect_S-_wOH','ant_wOH','soilOg+_wOH','soilOg-_wOH','solanox_wOH','biomass_wOH','SIF_trendy4_wOH']
process['COS_sign']=[ -1  , 1          ,  1   ,    1           ,        -1     ,  1         ,  -1         ,  -1        ,      1     ,      1   ,   -1    ]

######################################################################################################
vec_lmdz={"stat":[],"sim":[],'compound':[],"year":[],"month":[],"week":[]}

list_stat=['HFM','SPO',"NWR_afternoon",'ALT','BRW','SPO','KUM','MHD','LEF','SMO','CGO','WIS','MLO_afternoon','SUM','PSA','THD']
for cc in ['COS','CO2']:

  #Open LMDz simultion
  #Collect all the processes
  icomp=0
  for pp in range(len(process[cc])):
    outdir=eval('dir_'+cc)+'/Pyvar_'+process[cc][pp]+'/obsoperator/fwd_0000/'
    sim_tmp=pd.read_csv(outdir+'monitor.csv', sep=', ', delimiter=",",  skiprows=None, usecols=[0,1,10,12], na_values=None, skip_blank_lines=True)
    vec=sim_tmp['date'].values
    vec=sim_tmp['date'].valuesvec=[datetime.datetime(int(str(vec[ii])[:4]),int(str(vec[ii])[4:6]),int(str(vec[ii])[6:8]),int(str(vec[ii])[8:10]),0,0) for ii in range(len(vec))]
    sim_tmp['temps']=vec
    sim_tmp=sim_tmp[(sim_tmp.temps.dt.year>=beg_y)&(sim_tmp.temps.dt.year<=end_y)]
   #sim_tmp.set_index('temps',inplace=True,drop=True)
    if icomp==0:
     sim=sim_tmp.copy(deep=True)
     sim['sim']*=process[cc+'_sign'][pp]
     sim.set_index(['station','temps'],inplace=True,drop=True)
    else:
     sim['sim']+=sim_tmp.set_index(['station','temps'])['sim']*process[cc+'_sign'][pp]
    icomp+=1
  sim.reset_index(inplace=True)
#  sim['nb_week']=1
  adays=[1,9,17,25,32]
#  for kk in range(4):
#    mask=(sim['temps'].dt.day>=adays[kk])&(sim['temps'].dt.day<adays[kk+1])
#    sim.loc[mask,"nb_week"]=kk+1
#  sim['station']=sim.apply(lambda row : row['station'].upper(),axis=1)
  compteur=0  
  sim['station']=sim['station'].apply(lambda x: x.upper())
  for stat in list_stat:
    stat2 = copy.copy(stat[0:3])
    sim_stat=sim[sim['station']==stat2].copy(deep=True)
    sim_stat['sim']-=0 #sim_stat['sim'].iloc[0]
    sim_stat.set_index("temps",inplace=True)
    tmp=pd.read_pickle("/home/surface1/mremaud/"+cc+"/SURFACE/STATION/"+stat2+'.pkl')
    tmp=tmp[(tmp.date.dt.year>=beg_y)&(tmp.date.dt.year<=end_y)]
    tmp.set_index('date',inplace=True)

    sim_stat.fillna(value=0,inplace=True)
    tmp=tmp.resample('D').mean().dropna()
    tmp.reset_index(inplace=True)
    sim_stat=sim_stat.resample('D').mean()
    sim_stat.dropna(axis=0,inplace=True)
    sim_stat=sim_stat.loc[tmp["date"]].dropna() #; sim_stat.reset_index(inplace=True)
    sim_stat.reset_index(inplace=True)
    sim_stat['year']=sim_stat.temps.dt.year;sim_stat['month']=sim_stat.temps.dt.month
    sim_stat['week']=0
    for kk in range(4):
     mask=(sim_stat['temps'].dt.day>=adays[kk])&(sim_stat['temps'].dt.day<adays[kk+1])
     sim_stat.loc[mask,"week"]=kk+1
    sim_stat=sim_stat.groupby(['year','month','week']).mean()
    sim_stat.reset_index(inplace=True)
    sim_stat['date']=sim_stat.apply(lambda row: datetime.datetime(int(row['year']),int(row['month']),(int(row['week'])-1)*8+4,0,0,0),axis=1)
    a,f=plt.subplots(1,1)
    sim_stat.plot(x="date",y="sim",ax=f)
    sim_stat=outlier_filter(sim_stat)
    sim_stat.plot(x="date",y="sim",ax=f)
    plt.title(stat2)

#    sim_stat.plot(x='date',y="sim",ax=f)

    vec_lmdz['sim'].extend(sim_stat.sim.values ) 
    vec_lmdz['stat'].extend([stat for x in range(len(sim_stat))] )
    vec_lmdz['compound'].extend([cc for x in range(len(sim_stat))] )
    vec_lmdz['month'].extend( sim_stat.month.values)
    vec_lmdz['week'].extend(sim_stat.week.values)
    vec_lmdz['year'].extend( sim_stat.year.values)

vec_lmdz=pd.DataFrame(vec_lmdz)
vec_lmdz.to_pickle('vec_lmdz.pkl') 

  

