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

begy=2016
endy=2019

compound="COS"
dir_obs="/home/surface1/mremaud/"+compound+"/SURFACE/STATION/"
dir_adjoint="/home/satellites4/mremaud/FOOTPRINT/"
list_station=["MHD","MLO_afternoon","WIS",'THD',"SMO","PSA","CGO","NWR_afternoon","LEF","HFM","ALT","BRW","SUM","SPO"]
#list_station=["MIY"]
nstat=len(list_station)

list_day= ["01","09","17","25"]

###OBS VEC...
couleur=["k","brown","darkblue","darkgreen","lightgreen","yellow","red","rose","orange","purple","grey","darkgrey"]
legend_seas={"Dec-Jan-Feb":[11,0,1],"Sep-Oct-Nov":[10,9,8],"Mar-Avr-Mai":[2,3,4],"Jun-Jul-Aug":[6,5,7]}

obs_vec=pd.DataFrame()
for ss in list_station:
 obs_vec=obs_vec.append(pd.read_pickle(dir_obs+ss[:3]+".pkl"))
#obs_vec=pd.read_pickle("/home/surface1/mremaud/COS/SURFACE/Japan.pkl")
obs_vec["station"]=obs_vec.apply(lambda row: row.station.upper()[:3],axis=1)
#obs_vec["date"]=obs_vec.apply(lambda row: datetime.datetime(row.year,row.month,row.day,row.hour,0,0),axis=1)
list_stat=obs_vec['station'].unique()
print(list_stat)
adjoint_annual=np.zeros( (nstat,12,96,96))
compteur_annual=np.zeros( (nstat,12))
ad=0
for iss,station in enumerate(list_station):
 print(station) 
 for yy in range(begy,endy+1):
   #Adjoint_annual: 
   for month_a in range(1,13):
    compteur=0.
    for day_i,day_a in enumerate(list_day):
      sel_obs=obs_vec[(obs_vec.station==station[:3])&(obs_vec.date>=datetime.datetime(yy,month_a,day_i*8+1))&(obs_vec.date<=datetime.datetime(yy,month_a,min(day_i*8+8,calendar.monthrange(yy,month_a)[1])))]
      if sel_obs.empty: continue
      compteur+=1
      date_a=datetime.datetime(yy,month_a,day_i*8+3)
      dir_adjoint3= dir_adjoint+station+'/'+str(yy)+'/'+station+'_'+str(yy)+'%02d' %(month_a)+day_a
      #loop backward over the months
      nb_month=8
      for mm in range(nb_month):
         date_open=datetime.datetime(yy,month_a,1)-relativedelta(months=int(mm))
         print(dir_adjoint3+'/ad_'+str(date_open.year)+'-'+'%02d' %(date_open.month)+'.nc')
         tmp = xr.open_dataset(dir_adjoint3+'/ad_'+str(date_open.year)+'-'+'%02d' %(date_open.month)+'.nc',decode_times=False)
         tmp=tmp.mean("time")
         if mm==0:
          tmp2=np.copy(tmp[station.upper()].values)
          tmp2=tmp2[np.newaxis,:,:]
         else:
          tmp2=np.concatenate((tmp2,tmp[station.upper()].values[np.newaxis,:,:]),axis=0)
      datei_clim2=datetime.datetime(yy,month_a,1)-relativedelta(months=int(nb_month-1))
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
    #print(month_a,compteur)
    if compteur!=0: 
     compteur_annual[iss,month_a-1]+=1
     adjoint_annual[iss,month_a-1,:,:]/=compteur
 #adjoint_annual=np.mean(adjoint_annual,axis=0)
adjoint_annual/=compteur_annual[:,:,np.newaxis,np.newaxis]

for iss,seas in enumerate(legend_seas):
  print(iss,seas)
  plt.figure()
#  plt.title(seas)
  m = Basemap(projection='cyl',llcrnrlat=-89,urcrnrlat=90,\
              llcrnrlon=-180,urcrnrlon=180,resolution='c')
  #m = Basemap(projection='cyl',llcrnrlat=15,urcrnrlat=60,llcrnrlon=100,urcrnrlon=160,resolution='c')

  m.drawcoastlines()
  n_graticules = 18
  parallels = np.arange(-90., 170, 30)
  meridians = np.arange(0., 480., 30)
  parallels = np.arange(-90., 170, 40)
  meridians = np.arange(0., 480., 40)
  lw = 0.5
  dashes = [1,1] # 5 dots, 7 spaces... repeat
  graticules_color = 'grey'
  #m.drawparallels(parallels, linewidth=0.5,labels=[1,0,0,0], dashes=dashes, color=graticules_color,zorder=20)
  #m.drawmeridians(meridians,linewidth=0.5, labels=[0,0,0,1],dashes=dashes, color=graticules_color,zorder=20)
  m.drawparallels(parallels, linewidth=0.5,labels=[1,0,0,0], dashes=dashes, color=graticules_color,zorder=40)
  m.drawmeridians(meridians,linewidth=0.5, labels=[0,0,0,1],dashes=dashes, color=graticules_color,zorder=40)
  list_stat2=[ff for ff in list_station if (ff!="KUM")]
  for stat in list_stat2:
   data2=obs_vec[obs_vec.station==stat].mean()
   x, y = m(data2["lon"],data2["lat"])
   m.scatter(x,y,marker='o',color="sandybrown",zorder=100,alpha=1,label=stat)
   print(stat,data2["lon"],data2["lat"])
   plt.annotate(stat[:3], xy=(x-3, y+4),color="goldenrod",fontsize=8,weight="bold")

  Y, X=np.meshgrid(adclim_array.longitude.values,adclim_array.latitude.values)
  Y, X = m(Y,X)
  ##Moyenne sur toutes les stations
  adjoint_annual2=np.copy(np.sum(adjoint_annual[:,:,:,:],axis=0))
#  adjoint_annual2=np.squeeze(np.nanmean(adjoint_annual2[iss,:,:],axis=0))

  #adjoint_annual2=np.squeeze(np.nanmean(adjoint_annual2[legend_seas[seas],:,:],axis=0))
  adjoint_annual2=np.squeeze(np.nanmean(adjoint_annual2[:,:,:],axis=0))
#  adjoint_annual2=np.copy(np.squeeze(adjoint_annual)[6,:,:])
  adjoint_annual2=adjoint_annual2/10**5
  #adjoint_annual2[adjoint_annual2>=2]=2.
  #adjoint_annual2[adjoint_annual2<0.05]=np.nan
  v=np.arange(0.01,100,1)
  v=np.arange(0.06,30,)
  v=[0.06,0.08,1e-1,0.2,0.4,0.6,0.8 ,1e0,2,3]
  v=[0.3,0.4,0.5,0.6,0.7,0.8,0.9 ,1e0,2,3]
 # v=[3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,1e1]
  v2=v[::2]
  plt.title(seas)
  adjoint_annual2[adjoint_annual2<0]=0.

  if len(adjoint_annual2[~np.isnan(adjoint_annual2)])==0: continue
  ax=m.contourf(Y,X,adjoint_annual2,v,levels=v,locator=ticker.LogLocator(),cmap=plt.cm.Blues,extend='both')
#  ax=m.contourf(Y,X,adjoint_annual2,levels=v,cmap=plt.cm.Blues,extend='both')

  cb=plt.colorbar(ticks=v,orientation="horizontal")
  cb.set_ticks(v)
  cb.ax.minorticks_off()

  cb.set_label("$10^{5} ppm/kg/m^{2}/s$")
  #os.system("mkdir "+mdl.storedir+"/FOOTPRINT/")
  plt.savefig(stat+seas+"-"+str(iss)+".pdf",format="pdf")


