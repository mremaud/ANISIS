#!/usr/bin/env python
"""
author @Marine Remaud
Computation of the transport matrix G (tangent linear matrix) linking the weekly concentrations to the monthly prescribed fluxes
Each coefficient of the matrix are computed from the adjoint

"""
import importlib
import sys
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})

sys.path.append(mdl.homedir+'/modules')

#Define and load H
from .useful import *
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
import time
from netCDF4 import Dataset

name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})


def define_Hm(obs_vec,index_ctl_vec,index_unopt,all_priors):
 """
 Calcul of the transport matrix g and the
   associated linear tangent (matrix_G) (normalized for 1GtC)
 obs_vec: observation vector
 index_ctl_vec: control vector containing all the information to fill the tangent linear
 index_nopt: fixed (unoptimized) flux 
 all_priors: prior fluxes in kg/m2/s
 """

 #Warning:ajouter des exceptions pour les cas non lineaires
 asymptote_value=0.48
 area_LMDZ=get_area()

 adays=['01','09','17','25']
 #Variables pour moyenner les adjoints
 for cc in compound:
  if (cc.resolution=='W'):
   select_w=[1],[2],[3],[4]
  elif (cc.resolution == '2W'):
   select_w=[1,2],[3,4]
  elif (cc.resolution == 'M'):
   select_w=np.arange(1,5)
   select_w=select_w[np.newaxis,:]
 index_g=index_ctl_vec.append(index_unopt,ignore_index=True)
 matrix_g=np.zeros((len(obs_vec),len(index_g)))
 offset     =index_g[index_g['parameter']=='offset']
 index_g=index_g[index_g.parameter!='offset']
 list_process=index_g.parameter.unique()
 list_regtot=index_g.REG.unique()
 begya=obs_vec['year'].min()
 endya=obs_vec['year'].max()
 
 #Open all the region files at once in order to limit 
 # the number of files to be opened
 DIC_REG,AR_REG=get_dicregion(index_ctl_vec)

 #Remplissage de la matrice H en limitant louverture des adjoints
 for station in obs_vec['stat'].unique():
  print(station)
  start=time.time() #DEBUG
  for year_a in range(begya,endya+1):
   start_year=time.time()
   for month_a in range(1,13):
    for day_a in range(np.shape(select_w)[0]):
      date_a=datetime.datetime(year_a,month_a,1)
      #Selectionner les obs en fonction de la resolution temporelle
      if len(select_w[day_a])==1:
       mask_a=(obs_vec['year']==year_a)&(obs_vec["month"]==month_a)&(obs_vec['week']==select_w[day_a][0])&(obs_vec['stat']==station)
      elif len(select_w[day_a])==2:
       mask_a=(obs_vec['year']==year_a)&(obs_vec["month"]==month_a)&(obs_vec['stat']==station)
       mask_a=mask_a & (obs_vec['week']>=select_w[day_a][0]) & (obs_vec['week']<=select_w[day_a][1])
      elif len(select_w[day_a])==4:
       mask_a=(obs_vec['year']==year_a)&(obs_vec["month"]==month_a)&(obs_vec['stat']==station)
       mask_a=mask_a & (obs_vec['week']>=select_w[day_a,0]) & (obs_vec['week']<=select_w[day_a,-1])
      if obs_vec[mask_a].empty: continue
      index_line=np.copy(obs_vec[mask_a].index.values)
      dir_adjoint2=np.chararray((len(select_w[day_a])),itemsize=100)
      for ff in range(len(select_w[day_a])):
       dir_adjoint2[ff]= dir_adjoint+station+'/'+str(year_a)+'/'+station+'_'+str(year_a)+'%02d' %(month_a)+adays[select_w[day_a][ff]-1]
      dir_adjoint3= dir_adjoint+station+'/2017/'+station+'_2017'+'%02d' %(month_a)

      datef       = date_a-relativedelta(months=int(8))   #Limite des footprint 
      datef2      = date_a-relativedelta(months=int(24))  #Limite des footprint climatologiques
      date_col=copy.copy(date_a)
      #loop backward over the months
      while date_col >= datetime.datetime(begy,1,1) :   #Respect de la periode d optimisation
        #Ouverture des footprints
        if date_col>=datef:
         start2=time.time() #DEBUG
         #Moyenne des adjoints
         for ff in range(len(dir_adjoint2)):
          file_adjoint=dir_adjoint2[ff].decode("utf-8")+'/ad_'+str(date_col.year)+'-'+'%02d' %(date_col.month)+'.nc'
          while not os.path.exists(dir_adjoint2[ff]): 
             print(dir_adjoint2[ff].decode("utf-8")+' is absent.')
             ryear=random.randrange(max(year_a-5,begy), min(year_a+5,endy))
             dir_adjoint2[ff]= dir_adjoint+station+'/'+str(ryear)+'/'+station+'_'+str(ryear)+'%02d' %(month_a)+adays[select_w[day_a][ff]-1]
             file_adjoint=dir_adjoint2[ff].decode("utf-8")+'/ad_'+str((ryear-year_a)+date_col.year)+'-'+'%02d' %(date_col.month)+'.nc'
          #Open netcdf file
          nc_fid = Dataset(file_adjoint, 'r')
          tmp = nc_fid.variables[station.upper()][:].data
          lats = nc_fid.variables["latitude"][:]
          nc_fid.close()
          #Average over the weeks in case of biweekly or monthly
          if dir_adjoint2[ff]==dir_adjoint2[0]:
           adjoint=np.copy(tmp)
          else: 
           adjoint+= tmp
         adjoint/=len(dir_adjoint2)
         del tmp
         if lats[0]<lats[1]:
           lats=np.flip(lats)
           adjoint=np.flip(adjoint,axis=1)
         adjoint=np.sum(adjoint,axis=0)
         adjoint=adjoint[np.newaxis,:,:]
        elif date_col == (datef -relativedelta(months=int(1))):
         #Ouverture de l adjoint climatologique qu une seule fois
         datei_clim=datetime.datetime(year_a,month_a,1)-relativedelta(months=int(24))
         datef_clim=datetime.datetime(year_a,month_a,1)+relativedelta(months=int(1))
         tmp2=np.zeros((25,96,96))
         for mm in range(25):
          date_open=datetime.datetime(2017,month_a,1)-relativedelta(months=int(24-mm))
          namef_clim=dir_adjoint3+'/ad_'+str(date_open.year)+'-'+'%02d' %(date_open.month)+'.nc'
          nc_fid = Dataset(namef_clim, 'r')
          lats = nc_fid.variables['latitude'][:] 
          ad_clim = nc_fid.variables[station.upper()][:].data 
          nc_fid.close()
          ad_clim=np.sum(ad_clim,axis=0)[np.newaxis,:,:]
          tmp2[mm,:,:]=np.copy(ad_clim)  #  np.copy(ad_clim) if (mm == 0) else np.concatenate((tmp2,ad_clim),axis=0)
          if lats[0]<lats[1]:
           lats=np.flip(lats)
           tmp2=np.flip(tmp2,axis=1)
         #datei_clim2=datetime.datetime(2017,month_a,1)-relativedelta(months=int(24))
         #datef_clim2=datetime.datetime(2017,month_a,1)+relativedelta(months=int(1))
         time_clim=np.flip(pd.date_range(start=datei_clim, end=datef_clim,freq='M',closed='left'))
         tmp2=np.flip(tmp2,axis=0)
         adjoint=np.copy(tmp2)
        elif date_col.date()<(datef.date() -relativedelta(months=int(1))): 
         adjoint=np.copy(tmp2)
        #DEBOG
        for pp in list_process:
         index_adjoint_pp=index_g[(index_g['parameter']==pp)].copy(deep=True) #Select the annual process
         #Loop over the PFT
         for vv in index_adjoint_pp.PFT.unique(): 
          if np.isnan(vv):
           vv=1; index_adjoint_ppvv=index_adjoint_pp.copy(deep=True)
           if pp in DIC_REG:
            dic_reg  =DIC_REG[pp].copy(deep=True)
            reg_array=AR_REG[pp].copy(deep=True)
          else:
           index_adjoint_ppvv=index_adjoint_pp[index_adjoint_pp.PFT==vv].copy(deep=True)
           name_var=pp+"_"+str(int(vv))
           if name_var in DIC_REG:
            dic_reg  =DIC_REG[pp+"_"+str(int(vv))].copy(deep=True)
            reg_array=AR_REG[pp+"_"+str(int(vv))].copy(deep=True)
          listreg= index_adjoint_ppvv.REG.unique()
          #Loop over regions
          for rr in listreg:
              start3=time.time()
              index_adjoint_ppvvrr=index_adjoint_ppvv[index_adjoint_ppvv.REG==rr].copy(deep=True)
              if index_adjoint_ppvvrr.empty: continue
              if rr !=  "GLOBE": code=dic_reg[dic_reg['region']==rr].code.iloc[0]
              rows,columns=np.where(reg_array['BASIN'].values == code) if rr != 'GLOBE' else np.where(reg_array['BASIN'].values != np.nan)
              area=area_LMDZ[rows,columns]
              adjoint_reg=np.copy(adjoint[:,rows,columns])
              F1Gt=np.copy(np.array(np.squeeze(all_priors[pp][:,int(vv-1),rows,columns]),dtype='float64'))
              time_prior=[datetime.datetime(begy,1,1)+relativedelta(months=x) for x in range((endy-begy+1)*12)]
              if date_col.date()>=datef.date():
               mask_month=(index_adjoint_ppvvrr.month == date_col.month)&(index_adjoint_ppvvrr.year==date_col.year).copy(deep=True) 
               if not index_adjoint_ppvvrr[mask_month].empty: index_adjoint_ppvvrr=index_adjoint_ppvvrr[mask_month].copy(deep=True)
               adjoint_regp=np.copy(np.squeeze(adjoint_reg)) #only one month
               F1Gt=np.squeeze(F1Gt[(date_col.year-begy)*12+date_col.month-1,:])
               adjoint_regp=np.copy(np.nansum(np.multiply(adjoint_regp,F1Gt)))
               adjoint_regp=np.ones(1)*adjoint_regp
               time_ad=[date_col]
              elif (date_col.date()<datef.date()) & (date_col.date()>=datef2.date()):
               #mask_month=(index_adjoint_ppvvrr.month == date_col.month)&(index_adjoint_ppvvrr.year==date_col.year).copy(deep=True)
               #if not index_adjoint_ppvvrr[mask_month].empty: index_adjoint_ppvvrr=index_adjoint_ppvvrr[mask_month].copy(deep=True)
               time_ad=pd.to_datetime(time_clim)
               sel_time_ad=[x for x in range(len(time_ad)) if (time_ad[x]>=datetime.datetime(begy,1,1))&(time_ad[x]>=datef2)&(time_ad[x]<datef)]
               time_ad=time_ad[sel_time_ad]
               adjoint_regp=np.copy(adjoint_reg[sel_time_ad,:])
               index_time1=(time_ad[-1].year-begy)*12+time_ad[-1].month-1  #A lenvers
               index_time2=(time_ad[0].year-begy)*12+time_ad[0].month-1
               F1Gt=np.flip(F1Gt[index_time1:index_time2+1,:],axis=0)
               adjoint_regp=np.nansum(np.multiply(adjoint_regp,F1Gt),axis=1)
               #mask_month=(index_adjoint_ppvvrr.month == date_col.month)&(index_adjoint_ppvvrr.year==date_col.year).copy(deep=True)
               date_col=copy.copy(datef2)
               #i_month=[x for x in range(len(timet)) if (timet[x].month==date_col.month)&(timet[x].year==date_col.year)]
               #adjoint_regp=np.copy(adjoint_regp[i_month])
              elif date_col.date()<datef2.date(): #Au de la des footprint climatologique: extrapolation
               start4=time.time()
               #Computation of the final masse
               Mtot=np.sum(np.multiply(np.copy(F1Gt),area[np.newaxis,:]),axis=1,dtype='float64')
               Mtot=np.squeeze(Mtot)*86400.             
               index_time=[int(x) for x in range(len(time_prior)) if (time_prior[x]>=datetime.datetime(begy,1,1))&(time_prior[x]<datef2)]
               time_prior2=[time_prior[x] for x in index_time]
               Mtot_f=np.copy(Mtot[index_time])
               Mtot_f=[Mtot_f[x]*calendar.monthrange(time_prior[x].year,time_prior[x].month)[1] for x in range(len(time_prior2))]
               Mtot_f=np.flip(Mtot_f)

               timet=pd.to_datetime(time_clim)
               index_time1=(timet[-1].year-begy)*12+timet[-1].month-1  #A lenvers
               index_time2=(timet[0].year-begy)*12+timet[0].month-1
               F1Gt=np.flip(F1Gt[index_time1:index_time2+1,:],axis=0)
               Mtot=np.flip(Mtot[index_time1:index_time2+1])
               
               monthpery=np.asarray([calendar.monthrange(x.year,x.month)[1] for x in timet])
               Mtot=np.multiply(Mtot,monthpery)
               if len(Mtot[Mtot!=0])==0 : continue
               scaling_1G=np.copy(10**(12)/Mtot)
               F1Gt=np.multiply(F1Gt,scaling_1G[:,np.newaxis])
               adjoint_regp=np.nansum(np.multiply(adjoint_reg,F1Gt),axis=1)
               #imax_ad=8+np.where(  np.abs(adjoint_regp[8:]) == np.max(np.abs(adjoint_regp[8:])) )[0]
               #diff_max=np.abs(adjoint_regp[imax_ad]-adjoint_regp[imax_ad-1])
               #if (imax_ad>15)&(diff_max[0] >= 0.5):
               # if (Mtot[imax_ad]!=0)&(Mtot[imax_ad-1]!=0):
               #  if (orderOfMagnitude(np.abs(Mtot[imax_ad]))!=orderOfMagnitude(np.abs(Mtot[imax_ad-1]))) :
               #   print("outlier occurs 2 at: ",station,date_a,date_col,pp,rr,vv,adjoint_regp)
               #   adjoint_regp[imax_ad]=adjoint_regp[imax_ad-1]
               imax=8+np.where(adjoint_regp[8:]==np.max(adjoint_regp[8:]))[0][0]
               back_in_time=200
               #interpolation2=1
               #if (len(adjoint_regp[imax:])>3)|(adjoint_regp[-1]>=asymptote_value):
                #To avoid the sharp drop of the smoothing function
                #Do not begin with imax
               #interpolation2=0
               time_ad=copy.copy(timet[imax:])
               time_ad=time_ad.append(pd.to_datetime([time_ad[-1]-relativedelta(months=x+1) for x in range(back_in_time)]))
               time_ad=[x for x in time_ad if (x>=datetime.datetime(begy,1,1))&(x.date()<datef2.date())]
               adjoint_regp=np.ones(len(time_ad))*asymptote_value
              # if interpolation2:
                #print("interpolation2")
                #case of the remote stations
                #We take the latest point in time of the climatological adjoint
                #and we extroplate linearly until the datecol point 
             #   xdata=np.copy(np.arange(len(adjoint_regp)))
             #   xdata=np.copy(np.arange(xdata[-1],xdata[-1]+back_in_time))
             #   adjoint_extrapol=affine(xdata, adjoint_regp[-1],asymptote_value)
                # adjoint_extrapol=np.concatenate((adjoint_regp[:-1],adjoint_extrapol))
                #timet2=pd.to_datetime([timet[-1]-relativedelta(months=x) for x in range(back_in_time)])
               #i_month=[x for x in range(len(timet2)) if (timet2[x].month==date_col.month)&(timet2[x].year==date_col.year)]
               #Supprimer les points bizarre de lextrapolation
               #OUTLIERS
               #imax_ad=np.where(  np.abs(adjoint_extrapol) == np.max(np.abs(adjoint_extrapol)) )[0]
               #diff_max=np.abs(adjoint_extrapol[imax_ad]-adjoint_extrapol[imax_ad-1])
               #if (diff_max[0] >= 3) :
               #  print("outlier 3 occurs at: ",station,date_a,date_col,pp,vv,adjoint_extrapol)
               #  adjoint_extrapol[imax_ad]=adjoint_extrapol[imax_ad-1]
               adjoint_regp=np.multiply(adjoint_regp,Mtot_f)/10**(12)
               date_col=datetime.datetime(begy,1,1)
              ###Matrix affectation
              #Loop over index (parameters) for one specific region and date
              for ii in index_line:
               for tt in range(len(time_ad)):
                #Remplissage de la matrice G (tarantola) ou H: linear tangent
                col=index_adjoint_ppvvrr[(index_adjoint_ppvvrr.month==time_ad[tt].month)&(index_adjoint_ppvvrr.year==time_ad[tt].year)].copy(deep=True)
                matrix_g[ii,col.index]=np.copy(adjoint_regp[tt]*col["factor_"+obs_vec.loc[ii]['compound']])
        date_col=date_col-relativedelta(months=int(1))
 #Offset
 offset_prior =all_priors["offset"] 
 for jj in offset.index:
  for cc in compound:
   index_compound=obs_vec[obs_vec['compound']==cc.name].index
   matrix_g[index_compound,jj]=offset["factor_"+cc.name].loc[jj]*offset_prior[offset_prior["compound"]==cc.name].value.values[0]
 
 matrix_G=matrix_g[:,:len(index_ctl_vec)]
 np.savez_compressed(mdl.storedir+'matrix_G_i',matrix_G)
 np.savez_compressed(mdl.storedir+'matrix_g_i',matrix_g)

 ###


