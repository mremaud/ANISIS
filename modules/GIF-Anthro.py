#!/usr/bin/env python
#author @Marine Remaud
#Sensitivity analysis
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.stats import linregress
import random
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
import sys
from matplotlib import ticker, cm

#plt.rcParams.update({'font.size': 14})

begy=2015
endy=2015

compound="COS"
dir_obs="/home/surface1/mremaud/"+compound+"/SURFACE/STATION/"
dir_adjoint="/home/satellites4/mremaud/FOOTPRINT/"
list_station=["GIF","MHD"]
list_day= ["01","09","17","25"]

legend_seas={"Jan-Feb-Mar":[0,1,2]}  #,"Sep-Oct-Nov":[10,9,8],"Mar-Avr-Mai":[2,3,4],"Jun-Jul-Aug":[6,5,7]}

##Open the region
region=xr.open_dataset("/home/surface1/mremaud/COS/PFTMAP/region-LMDZ9696.nc")
lon=region.lon.values
region=region.BASIN.values
region[region!=190]=0.
region[:,:45]=0.
region=region[::-1,:]

obs_vec=pd.DataFrame()
for ss in list_station:
 obs_vec=obs_vec.append(pd.read_pickle(dir_obs+ss+".pkl"))
#obs_vec=pd.read_pickle("/home/surface1/mremaud/COS/SURFACE/Japan.pkl")
obs_vec["station"]=obs_vec.apply(lambda row: row.station[:3].upper(),axis=1)
#obs_vec=obs_vec[obs_vec.station=="MIY"]
list_stat=obs_vec['station'].unique()
nstat=len(list_stat)
adjoint_annual=np.zeros( (nstat,12,96,96))
compteur_annual=np.zeros( (nstat,12))
ad=0
for iss,station in enumerate(list_stat):
 for yy in range(begy,endy+1):
   #Adjoint_annual: 
   for month_a in range(1,13):
    compteur=0.
    for day_i,day_a in enumerate(list_day):
      sel_obs=obs_vec[(obs_vec.station==station)&(obs_vec.date>=datetime.datetime(yy,month_a,day_i*8+1))&(obs_vec.date<=datetime.datetime(yy,month_a,min(day_i*8+8,calendar.monthrange(yy,month_a)[1])))]
      if sel_obs.empty: continue
      compteur+=1
      date_a=datetime.datetime(yy,month_a,day_i*8+3)
      dir_adjoint3= dir_adjoint+station+'/'+str(yy)+'/'+station+'_'+str(yy)+'%02d' %(month_a)+day_a
      #loop backward over the months
       #Adjoint climatologique qu une seule fois
      n_month=3
      for mm in range(n_month):
          date_open=datetime.datetime(yy,month_a,1)-relativedelta(months=int(mm))
          if date_open<datetime.datetime(yy,1,1): continue
          tmp = xr.open_dataset(dir_adjoint3+'/ad_'+str(date_open.year)+'-'+'%02d' %(date_open.month)+'.nc',decode_times=False)
          tmp=tmp.mean("time")
          tmp2=np.copy(tmp[station.upper()].values)[np.newaxis,:,:] if (mm == 0) else np.concatenate((tmp2,tmp[station.upper()].values[np.newaxis,:,:]),axis=0)
      datei_clim2=max(datetime.datetime(yy,month_a,1)-relativedelta(months=int(n_month-1)),datetime.datetime(yy,1,1))
      datef_clim2=datetime.datetime(yy,month_a,1)+relativedelta(months=int(1)) 
       #Probleme annee bissextil e
      adclim_array = xr.Dataset({station: (['time', 'latitude', 'longitude'], tmp2[::-1,:,:])},
                        coords={'latitude': (['latitude'], tmp.latitude.values),
                                'longitude': (['longitude'], tmp.longitude.values),
                                'time': (['time'], pd.date_range(start=datei_clim2, end=datef_clim2,freq='M',closed='left'))})
      adclim_array.time.values=np.flip(adclim_array.time.values)
      adclim_array[station].values=np.flip(adclim_array[station].values,axis=0)
      if adclim_array.latitude.values[0]<adclim_array.latitude.values[1]:
          adclim_array['latitude']=np.flip(adclim_array['latitude'].values)
          adclim_array[station].values=np.flip(adclim_array[station].values,axis=1)
      adjoint=np.copy(adclim_array[station].values)
      adjoint_annual[iss,month_a-1,:,:]+=np.squeeze(np.sum(adjoint,axis=0))
      compteur+=1
    print(month_a,compteur)
    if compteur!=0: 
     compteur_annual[iss,month_a-1]+=1
     adjoint_annual[iss,month_a-1,:,:]/=compteur
 #adjoint_annual=np.mean(adjoint_annual,axis=0)
adjoint_annual/=compteur_annual[:,:,np.newaxis,np.newaxis]

for iss,seas in enumerate(legend_seas):
  print(iss)


  list_stat2=[ff for ff in list_stat if (ff!="KUM")]
  for stat in list_stat2:
   data2=obs_vec[obs_vec.station==stat].mean()

  adjoint_annual2=np.copy(np.squeeze(adjoint_annual[0,:,:,:])-np.squeeze(adjoint_annual[1,:,:,:]))
  adjoint_annual2=np.squeeze(np.nanmean(adjoint_annual2[legend_seas[seas],:,:],axis=0))
  
  FANT=1./np.sum(adjoint_annual2)*13.4*10**(-6)*12./32.065
  print(FANT)

  ###Load the old fluxes
  FANT_old=xr.open_dataset("/home/surface1/mremaud/COS/INPUT_sflx/Anthro/flx_2015_dyn.nc",decode_times=False)
  area=xr.open_dataset("/home/surface1/mremaud/CO2/LMDZREF/start-96-L39.nc",decode_times=False) 
  area=area.aire.values
  FANT_old=FANT_old.flx_cos.values
  FANT_old=FANT_old[:,:,:-1]
  area=area[:,:-1]
  FANT_old=FANT_old[0,::-1,:]
  FANT_old[np.where(region==0)]=0.
  area2=np.copy(area)
  area[np.where(region==0)]=0.
  FANT_old=np.sum(np.squeeze(FANT_old[:,:])*area )*86400*365.
  print(FANT_old*10**(-6))
  FANT_new=np.sum(area)*FANT*86400*365.
  print("New uniform",FANT_new*10**(-6))


  FANT_IASI=xr.open_dataset("/home/users/mremaud/PYTHON/COS/EXTRACT/IASI/IASI-flux.nc",decode_times=False)
  FANT_IASI=FANT_IASI.__xarray_dataarray_variable__.values
  fac_IASI=1./np.sum(adjoint_annual2*FANT_IASI)*13.4*10**(-6)*12./32.065
  print(np.sum(np.squeeze(FANT_IASI*area2*fac_IASI )*86400*365.))  
