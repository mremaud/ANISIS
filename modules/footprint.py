#!/usr/bin/env python
#author @Marine Remaud
#Sensitivity analysis

from mpl_toolkits.basemap import Basemap
from scipy.stats import linregress
import random
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
import sys
plt.rcParams.update({'font.size': 14})


def footprint(obs_vec):
 repf=mdl.storedir+"/"
 ncfile="region.nc"
 reg_array = xr.open_dataset(repf+ncfile,decode_times=False)
 if reg_array.lat.values[0]<reg_array.lat.values[1]:
     reg_array['lat']=np.flip(reg_array['lat'].values)
     reg_array['BASIN'].values=np.flip(reg_array['BASIN'].values,axis=0)

 dataset = Dataset(mdl.file_area,'r',format='NETCDF4_CLASSIC')
 area_LMDZ=np.squeeze(dataset.variables['aire'][:])[:,:-1]
 dataset.close()

 list_stat=obs_vec['stat'].unique()
 nstat=len(list_stat)
 adjoint_annual=np.zeros( (nstat,12,np.shape(area_LMDZ)[0],np.shape(area_LMDZ)[1]))

 for iss,station in enumerate(list_stat):
   #Adjoint_annual: 
   for month_a in range(1,13):
      date_a=datetime.datetime(2017,month_a,1)
      dir_adjoint3= dir_adjoint+station+'/2017/'+station+'_2017'+'%02d' %(month_a)
      #loop backward over the months
       #Adjoint climatologique qu une seule fois
      datei_clim=datetime.datetime(2017,month_a,1) #-relativedelta(months=int(1))
      datef_clim=datetime.datetime(2017,month_a,1)#+relativedelta(months=int(1))
      for mm in range(1):
          date_open=datetime.datetime(2017,month_a,1)-relativedelta(months=int(1-mm))
          tmp = xr.open_dataset(dir_adjoint3+'/ad_'+str(date_open.year)+'-'+'%02d' %(date_open.month)+'.nc',decode_times=False)
          tmp2=np.copy(np.squeeze(tmp[station.upper()].values)) if (mm == 0) else np.concatenate((tmp2,tmp[station.upper()].values),axis=0)
      datei_clim2=datetime.datetime(2017,month_a,1)-relativedelta(months=int(1))
      datef_clim2=datetime.datetime(2017,month_a,1) # +relativedelta(months=int(1))
       #Probleme annee bissextil e
      adclim_array = xr.Dataset({station: (['time', 'latitude', 'longitude'], tmp2)},
                        coords={'latitude': (['latitude'], tmp.latitude.values),
                                'longitude': (['longitude'], tmp.longitude.values),
                                'time': (['time'], pd.date_range(start=datei_clim2, end=datef_clim2,freq='D',closed='left'))})
      adclim_array=adclim_array.resample(time="M").sum()
      adclim_array.time.values=pd.date_range(start=datei_clim, end=datef_clim,freq='M',closed='left')
      adclim_array.time.values=np.flip(adclim_array.time.values)
      adclim_array[station].values=np.flip(adclim_array[station].values,axis=0)
      if adclim_array.latitude.values[0]<adclim_array.latitude.values[1]:
          adclim_array['latitude']=np.flip(adclim_array['latitude'].values)
          adclim_array[station].values=np.flip(adclim_array[station].values,axis=1)
      adjoint=np.copy(adclim_array[station].values)
    #  adjoint=np.copy(adjoint)*(10**12)/(np.sum(area_LMDZ)*31.*86400)
      adjoint_annual[iss,month_a-1,:,:]=np.squeeze(np.sum(adjoint,axis=0))
      #Annual adjoint

 
 #adjoint_annual=np.mean(adjoint_annual,axis=0)
 #Stations coordinates
 coo_stat=pd.DataFrame()
 for cc in compound:
   for stat in cc.stations:
    #Test marine: NWR
    if (stat=='NWR_afternoon')|(stat=='MLO_afternoon'):
     stat2 = stat[:3] 
    else:
     stat2=copy.copy(stat) 
    tmp=pd.read_pickle(cc.file_obs+stat2+'.pkl')
    tmp=tmp.mean().dropna() 
    tmp=tmp[['lon','lat']]
    tmp['compound']=cc.name
    tmp['stat']=np.copy(stat)
    coo_stat=coo_stat.append(tmp.copy(deep=True),ignore_index=True)

 couleur=["k","brown","darkblue","darkgreen","lightgreen","yellow","red","rose","orange","purple","grey","darkgrey"]
 legend_seas={"Annual":np.arange(12),"Jan-Feb-Mar":np.arange(3),"Jul-Aug-Sep":np.arange(5,8)}

 for iss,seas in enumerate(legend_seas):
  plt.figure()
  #print adjoint_annual2.shape
  adjoint_annual2=np.mean(adjoint_annual,axis=0)
#  plt.title(seas)
  m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
              llcrnrlon=-180,urcrnrlon=180,resolution='c')
  m.drawcoastlines()
  n_graticules = 18
  parallels = np.arange(-90., 170, 30)
  meridians = np.arange(0., 480., 60)
  lw = 0.5
  dashes = [1,1] # 5 dots, 7 spaces... repeat
  graticules_color = 'grey'
  m.drawparallels(parallels, linewidth=0.5,labels=[1,0,0,0], dashes=dashes, color=graticules_color,zorder=20)
  m.drawmeridians(meridians,linewidth=0.5, labels=[0,0,0,1],dashes=dashes, color=graticules_color,zorder=20)

  list_stat2=[ff for ff in list_stat if (ff!="KUM")]
  for stat in list_stat2:
   data2=coo_stat[coo_stat.stat==stat].mean()
   x, y = m(data2["lon"],data2["lat"])
   m.scatter(x,y,marker='o',color="sandybrown",zorder=100,alpha=1,label=stat)
   #for i, (x, y) in enumerate(zip(X, Y), start=1):
   plt.annotate(stat[:3], xy=(x-3, y+4),color="brown",fontsize=8,weight="bold")


  Y, X=np.meshgrid(reg_array.lon.values,reg_array.lat.values)
  Y, X = m(Y,X)
  adjoint_annual2=np.mean(adjoint_annual2[legend_seas[seas],:,:],axis=0)
  v=np.arange(0,5,0.5)
  print(np.max(adjoint_annual2),np.min(adjoint_annual2))
  ax=m.contourf(Y,X,adjoint_annual2/10**5,v,cmap=plt.cm.Blues,extend='max')
  cb=plt.colorbar(orientation="horizontal")
  cb.set_label("$ppm/kg/m^{2}/s$")
  os.system("mkdir "+mdl.storedir+"/FOOTPRINT/")
  plt.savefig(mdl.storedir+"/FOOTPRINT/All_stat"+str(iss)+".pdf",format="pdf")


 plt.rcParams.update({'font.size': 7})

 fig, axes = plt.subplots(nrows=5, ncols=3)
 for iss,ax in enumerate(axes.flat):
  if iss>len(list_stat)-1: continue
# for iss,stat in enumerate(list_stat):
  data2=coo_stat[coo_stat.stat==list_stat[iss]]
  x, y = m(data2["lon"],data2["lat"])
#  plt.figure()
  m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
              llcrnrlon=-180,urcrnrlon=180,resolution='c',ax=ax)
#  m.scatter(x,y,marker='o',color="r",zorder=80,alpha=1,s=0.8,label=stat,ax=ax)
  m.drawcoastlines(ax=ax)
  n_graticules = 18
  parallels = np.arange(-90., 170, 30)
  meridians = np.arange(0., 480., 60)
  lw = 0.5
  dashes = [1,1] # 5 dots, 7 spaces... repeat
  graticules_color = 'grey'
  m.drawparallels(parallels, linewidth=0.5,labels=[0,0,0,0], dashes=dashes, color=graticules_color,zorder=20)
  m.drawmeridians(meridians,linewidth=0.5, labels=[0,0,0,0],dashes=dashes, color=graticules_color,zorder=20)
  adjoint_annual2=adjoint_annual[iss,:,:,:]
  adjoint_annual2=np.mean(adjoint_annual2,axis=0)
  v=np.arange(0,15,1)
  print(np.max(adjoint_annual2),np.min(adjoint_annual2))
  m.contourf(Y,X,adjoint_annual2/10**5,v,cmap=plt.cm.Blues,extend='max')
  ax.title.set_text(list_stat[iss])
  plt.tight_layout() 
  #plt.title(stat)
  if iss==len(list_stat)-4:
   cb=ax.colorbar(orientation="horizontal")
   cb.set_label("$ppm/kg/m^{2}/s$")

plt.tight_layout()
plt.savefig(mdl.storedir+"/FOOTPRINT/Stat.pdf",format="pdf")

