#!/usr/bin/env python
#author @Marine Remaud
#Code de transport pour optimiser des flux hebdomadaires

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

name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})


def define_Hw(obs_vec,index_ctl_vec,index_unopt,all_priors):
 """
 Calcul of the transport matrix g and the
   associated linear tangent (matrix_G) (normalized for 1GtC)
 obs_vec: observation vector
 index_ctl_vec: control vector containing all the information to fill the tangent linear
 index_nopt: fixed (unoptimized) flux 
 all_priors: prior fluxes in kg/m2/s
 """
 #for pp in newdict.keys()::
 #  plt.figure()
 #  plt.contourf(np.squeeze(all_priors[pp][0,:,:]))
 #Warning:ajouter des exceptions pour les cas non lineaires
 asymptote_value=0.48
 area_LMDZ=get_area()
 adays=['01','09','17','25']

 #Add variables to adjust the timestep of the observations
 # If the timestep> 1 week: need to average the adjoints over the 
 #month

 #Timestep of the optimized flux: taking or not account for the weeks
 odays=[1,9,17,25]

 index_g=index_ctl_vec.append(index_unopt,ignore_index=True)
 matrix_g=np.zeros((len(obs_vec),len(index_g)))
 index_g=index_g[index_g.parameter!='offset']
# index_g.reset_index(inplace=True,drop=True)
 list_process=index_g.parameter.unique()  #Kind of flux to be optimized
 list_regtot=index_g.REG.unique() 

 begya=obs_vec['year'].min()
 endya=obs_vec['year'].max()
 #Open all the region files at once in order to limit 
 # the number of files to be opened
 DIC_REG,AR_REG=get_dicregion(index_ctl_vec)

 #Remplissage de la matrice H en limitant louverture des adjoints
 for station in obs_vec['stat'].unique():
  for year_a in range(begya,endya+1):
   for month_a in range(1,13):
    for week_a in range(1,5):
      date_a=datetime.datetime(year_a,month_a,odays[week_a-1])  #date de l observation qui sera assimilee
      mask_a=(obs_vec['year']==year_a)&(obs_vec["month"]==month_a)&(obs_vec['week']==week_a)&(obs_vec['stat']==station)
      if obs_vec[mask_a].empty: continue
      index_line=np.copy(obs_vec[mask_a].index.values)
      
      # Fichiers et repertoires des adjoints
      dir_adjoint2= dir_adjoint+station+'/'+str(year_a)+'/'+station+'_'+str(year_a)+'%02d' %(month_a)+adays[week_a-1]
      dir_adjoint3= dir_adjoint+station+'/2017/'+station+'_2017'+'%02d' %(month_a)
      #datef : date limites des footprints hebdomadaires (8 mois la plupart du temps)
      #datef2: date limites des footprints climatologiques (24 mois la plupart du temps) 
      datef       = datetime.datetime(date_a.year,date_a.month,1)-relativedelta(months=int(8))   #Limite des footprint 
      datef2      = datetime.datetime(date_a.year,date_a.month,1)-relativedelta(months=int(24))  #Limite des footprint climatologiques
      #initialisation: premier adjoint a ouvrir
      if adays[week_a-1]=="01":
       mcol=copy.copy((date_a-relativedelta(months=1)).month)
       ycol=copy.copy((date_a-relativedelta(months=1)).year)
       deltad=1+calendar.monthrange(ycol,mcol)[1]-25
      else:
       deltad=8
      date_col=date_a-relativedelta(days=deltad)
      #loop backward over the months
      while (date_col >= datef)&(date_col>=datetime.datetime(begy,1,1)) :   #Respect de la periode d optimisation
        ycol=copy.copy(date_col.year)
        mcol=copy.copy(date_col.month)
        wcol=copy.copy(np.where(date_col.day==np.asarray(odays))[0]+1)[0]
        #Ouverture des adjoints hebdomadaires
        file_adjoint=dir_adjoint2.decode("utf-8")+'/ad_'+str(date_col.year)+'-'+'%02d' %(date_col.month)+'.nc'
        while not os.path.exists(dir_adjoint2): 
         ryear=random.randrange(year_a-1, year_a+1)
         dir_adjoint2= dir_adjoint+station+'/'+str(ryear)+'/'+station+'_'+str(ryear)+'%02d' %(month_a)+adays[week_a-1]
         file_adjoint=dir_adjoint2.decode("utf-8")+'/ad_'+str((ryear-year_a)+ycol)+'-'+'%02d' %(mcol)+'.nc'
        ad_array = xr.open_dataset(file_adjoint,decode_times=False)
        if ad_array.latitude.values[0]<ad_array.latitude.values[1]:
         ad_array['latitude']=np.flip(ad_array['latitude'].values)
         ad_array[station.upper()].values=np.flip(ad_array[station.upper()].values,axis=1)
        
        adjoint=np.copy(ad_array[station.upper()].values[date_col.day-1:date_col.day+deltad-1,:,:])
        adjoint=np.squeeze(np.sum(adjoint,axis=0))
        #Loop over the processes
        for pp in all_priors.keys():
         if pp == "offset": continue
         index_adjoint_pp=index_g[index_g.parameter==pp].copy(deep=True) 
         mask_month=(index_adjoint_pp.month == mcol)&(index_adjoint_pp.year == ycol)&(index_adjoint_pp.week == wcol).copy(deep=True)
         if index_adjoint_pp[mask_month].empty: print("Not here",index_line,date_col)
         for vv in index_adjoint_pp.PFT.unique(): 
          if np.isnan(vv): 
           vv=1
           index_adjoint_ppvv=index_adjoint_pp.copy(deep=True)
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
          for rr in  listreg: 
           index_adjoint_ppvvrr=index_g[index_adjoint_ppvv.REG==rr].copy(deep=True)
           if index_adjoint_ppvvrr.empty: continue
           mask_month=(index_adjoint_ppvvrr.month == mcol)&(index_adjoint_ppvvrr.year == ycol)&(index_adjoint_ppvvrr.week == wcol).copy(deep=True)
           if index_adjoint_ppvvrr[mask_month].empty: print("Absent",index_line,date_col)
           if not index_adjoint_ppvvrr[mask_month].empty: index_adjoint_ppvvrr=index_adjoint_ppvvrr[mask_month]
           if rr !=  "GLOBE": code=dic_reg[dic_reg['region']==rr].code.iloc[0]
           rows,columns=np.where(reg_array['BASIN'].values == code) if rr != 'GLOBE' else np.where(reg_array['BASIN'].values != np.nan)
           area=area_LMDZ[rows,columns]
           adjoint_reg=np.copy(adjoint[rows,columns])
           F1Gt=np.copy(np.array(np.squeeze(all_priors[pp][:,int(vv-1),rows,columns]),dtype='float64'))
           Mtot=np.sum(np.multiply(np.copy(F1Gt),area),axis=1,dtype='float64')
          
           #Selection du flux prior: Prior sont en moyene mensuelle
           F1Gt=np.copy(np.array(np.squeeze(all_priors[pp][:,int(vv-1),rows,columns]),dtype='float64'))
           F1Gt=np.squeeze(F1Gt[(ycol-begy)*12+mcol-1,:])
           Mtot=np.sum(np.multiply(np.copy(F1Gt),area),dtype='float64')
           Mtot=np.squeeze(Mtot)*86400.*deltad  #Masse totale emise pendant une semaine
           adjoint_regp=np.copy(np.squeeze(adjoint_reg)) #only one month
           if Mtot==0: continue
           F1Gt=F1Gt*10**(12)/np.copy(Mtot)
           adjoint_regp=np.copy(np.nansum(np.multiply(adjoint_regp,F1Gt[np.newaxis,:])))

           #Loop over index (parameters) for one specific region and date
           col=index_adjoint_ppvvrr.copy(deep=True)
           for ii in index_line:
            #Remplissage de la matrice G (tarantola) ou H: linear tangent 
            matrix_g[ii,col.index]=np.copy(adjoint_regp*col["factor_"+obs_vec.loc[ii]['compound']])
        if not adays[wcol-1]=="01":
         deltad=8
        else:
         mcol=copy.copy((date_col-relativedelta(months=1)).month) 
         ycol=copy.copy((date_col-relativedelta(months=1)).year)
         deltad=calendar.monthrange(ycol,mcol)[1]-25+1
        date_col=date_col-relativedelta(days=deltad)

#####      #Ouverture des adjoints climatologiques
      first_clim=1
      while (date_col<datef)&(date_col >= datetime.datetime(begy,1,1)) :   #Respect de la periode d optimisation
       #Annee, mois, semaine du flux a optimiser
       ycol=copy.copy(date_col.year)
       mcol=copy.copy(date_col.month)
       wcol=copy.copy(np.where(date_col.day==np.asarray(odays))[0]+1)[0]
       if first_clim:
         #Ouverture de l adjoint climatologique qu une seule fois
         datei_clim=datetime.datetime(year_a,month_a,1)-relativedelta(months=int(24))
         datef_clim=datetime.datetime(year_a,month_a,1)+relativedelta(months=int(1))
         for mm in range(25):
          date_open=datetime.datetime(2017,month_a,1)-relativedelta(months=int(24-mm))
          tmp = xr.open_dataset(dir_adjoint3+'/ad_'+str(date_open.year)+'-'+'%02d' %(date_open.month)+'.nc',decode_times=False)
          tmp2=np.copy(np.squeeze(tmp[station.upper()].values)) if (mm == 0) else np.concatenate((tmp2,tmp[station.upper()].values),axis=0)
         datei_clim2=datetime.datetime(2017,month_a,1)-relativedelta(months=int(24))
         datef_clim2=datetime.datetime(2017,month_a,1)+relativedelta(months=int(1))
         #Probleme annee bissextile
         adclim_array = xr.Dataset({station: (['time', 'latitude', 'longitude'], tmp2)},
                        coords={'latitude': (['latitude'], tmp.latitude.values),
                                'longitude': (['longitude'], tmp.longitude.values),
                                'time': (['time'], pd.date_range(start=datei_clim2, end=datef_clim2,freq='D',closed='left'))})
         if adclim_array.latitude.values[0]<adclim_array.latitude.values[1]:
          adclim_array['latitude']=np.flip(adclim_array['latitude'].values)
          adclim_array[station].values=np.flip(adclim_array[station].values,axis=1)
         time_clim=pd.date_range(start=datei_clim, end=datef_clim,freq='D',closed='left')
         first_loc=1; time_clim2=[]
         for yy in range(datei_clim.year,datef_clim.year+1):
          for mm in range(1,13):
           for ww in range(1,5):
            index_clim=[x for x in range(len(time_clim))  if (time_clim[x].month==mm)&(time_clim[x].year==yy) ]
            if not index_clim: continue
            index_clim=index_clim[(ww-1)*8:ww*8] if (ww!=4) else index_clim[24:]
            tmp=np.copy(np.sum(adclim_array[station].values[index_clim,:,:],axis=0))
            adjoint=np.copy(tmp[np.newaxis,:,:]) if first_loc else np.concatenate((adjoint,tmp[np.newaxis,:,:]),axis=0)
            time_clim2.append(time_clim[index_clim[0]])
            first_loc=0
         time_clim2=pd.to_datetime(time_clim2)
         time_clim2=np.flip(time_clim2)
         adjoint=np.flip(adjoint,axis=0)
         first_clim=0
#MARINE
       for pp in all_priors.keys():
        if pp == "offset": continue
        mask_adjoint    =(index_g.parameter==pp)&(index_g.year == ycol).copy(deep=True)
        index_adjoint_pp=index_g[mask_adjoint].copy(deep=True)
        mask_month=(index_adjoint_pp.month == mcol)&(index_adjoint_pp.week == wcol).copy(deep=True)
        if not index_adjoint_pp[mask_month].empty: index_adjoint_pp=index_adjoint_pp[mask_month].copy(deep=True)
        for vv in index_adjoint_pp.PFT.unique():
         if np.isnan(vv):
           vv=1
           index_adjoint_ppvv=index_adjoint_pp.copy(deep=True)
           if pp in DIC_REG:
            dic_reg  =DIC_REG[pp].copy(deep=True)
            reg_array=AR_REG[pp].copy(deep=True)
         else:
           index_adjoint_ppvv=index_adjoint_pp[index_adjoint_pp.PFT==vv].copy(deep=True)
           name_var=pp+"_"+str(int(vv))
           if name_var in DIC_REG:
            dic_reg  =DIC_REG[pp+"_"+str(int(vv))].copy(deep=True)
            reg_array=AR_REG[pp+"_"+str(int(vv))].copy(deep=True)
         listreg=index_adjoint_ppvv.REG.unique()
         for rr in listreg:
          index_adjoint_ppvvrr=(index_adjoint_ppvv.REG==rr).copy(deep=True)
          if index_adjoint_ppvvrr.empty: continue
          if rr !=  "GLOBE": code=dic_reg[dic_reg['region']==rr].code.iloc[0]
          rows,columns=np.where(reg_array['BASIN'].values == code) if rr != 'GLOBE' else np.where(reg_array['BASIN'].values != np.nan)
          area=area_LMDZ[rows,columns]
          F1Gt=np.copy(np.array(np.squeeze(all_priors[pp][:,int(vv-1),rows,columns]),dtype='float64'))
          Mtot=np.sum(np.multiply(np.copy(F1Gt),area[np.newaxis,:]),axis=1,dtype='float64')
          Mtot=np.squeeze(Mtot)*86400.  #kg/day
          adjoint_reg=np.copy(adjoint[:,rows,columns])
          if (date_col.date()<datef.date()) & (date_col.date()>=datef2.date()):
               #Adjoint climatologique
               #Select the month
               timet=time_clim2.copy(deep=True)
               i_month=[x for x in range(len(timet)) if (timet[x].month==mcol)&(timet[x].year==ycol)]
               i_month=[x for x in i_month if (timet[x].day==odays[wcol-1])]
               adjoint_regp=np.copy(np.squeeze(adjoint_reg[i_month,:]))
               F1Gt=np.squeeze(F1Gt[(ycol-begy)*12+mcol-1,:])
               #Masse totale sur la semaine
               Mtot=Mtot[(ycol-begy)*12+mcol-1]*deltad
               if Mtot==0: continue
               F1Gt=F1Gt*10**(12)/np.copy(Mtot) #Flux egal a un 1Gt sur une semaine
               adjoint_regp=np.nansum(np.multiply(adjoint_regp,F1Gt))
          elif date_col.date()<datef2.date(): #Au de la des footprint climatologique: extrapolation
               #Inverser les flux
               index_time1=(timet[-1].year-begy)*12+timet[-1].month-1  #A lenvers
               index_time2=(timet[0].year-begy)*12+timet[0].month-1
               F1Gt=np.flip(F1Gt[index_time1:index_time2+1,:],axis=0)
               F1Gt=np.repeat(F1Gt,4,axis=0)
               Mtot=np.flip(Mtot[index_time1:index_time2+1],axis=0)
               if Mtot[Mtot!=0].size ==0 : "empty";continue
               Mtot=np.repeat(Mtot,4)
               dperw=[8 if x.day != 25 else (calendar.monthrange(x.year,x.month)[1]-25+1)  for x in timet]
               dperw=np.asarray(dperw)
               Mtot=np.multiply(Mtot,dperw)
               scaling_1G=np.copy(10**(12)/Mtot)
               F1Gt=np.multiply(F1Gt,scaling_1G[:,np.newaxis])
               adjoint_regp=np.nansum(np.multiply(adjoint_reg,F1Gt),axis=1)
              # if np.abs(0.48-adjoint_regp[-1])> np.abs(0.48-adjoint_regp[10]): print "problem time"; sys.exit()
               imax=np.where(adjoint_regp==np.max(adjoint_regp))[0][0]
               back_in_time=1000
               interpolation2=1
               if len(adjoint_regp[imax:])>12:
                #To avoid the sharp drop of the smoothing function
                #Do not begin with imax
                interpolation2=0
                ydata=np.copy(adjoint_regp[imax+10:])
                timet2=copy.copy(timet[imax+10:])
                xdata=np.copy(np.arange(len(ydata)))
                epsilon=0.0000001
                #Resolution de bogges:ajout cas supplementaire pour les pentes ascendantes 

                if linregress(ydata, xdata).slope >0:
                 #Addition of a linear segment to avoid a sharp drop
                 ydata=np.copy(adjoint_regp[-1]+(asymptote_value-adjoint_regp[-1])/160.*np.arange(160))
                 xdata=np.arange(160)
                 nb_month2=np.repeat(np.arange(160/4),4)
                 timet2=[timet[-1]-relativedelta(months=x) for x in nb_month2]
                 timet2=[datetime.datetime(timet2[n].year,timet2[n].month,odays[(3-n)%4]) for n in range(len(timet2)) if n!=0]
                 timet2=pd.to_datetime(timet2)
                try:
                 popt, pcov = curve_fit(expo, xdata, ydata,bounds= \
                            ((-np.inf,-np.inf,asymptote_value-epsilon), (np.inf,np.inf,asymptote_value+epsilon)))
                except (RuntimeError, TypeError, NameError):
                 #np.save(mdl.storedir+'ydata.npy',ydata)
                 #np.save(mdl.storedir+'xdata.npy',xdata)
                 np.save(mdl.storedir+'adjoint_regp.npy',adjoint_regp.values)
                 interpolation2=1
                xdata2=np.copy(np.arange(len(ydata)+back_in_time))
                adjoint_extrapol=expo(xdata2, *popt)
                if adjoint_extrapol[-1]< asymptote_value:
                  interpolation2=2
                time2_ad=np.repeat(np.arange(1,back_in_time/4.+1),4)
                time2_ad=pd.to_datetime([timet2[-1]-relativedelta(months=x) for x in time2_ad])
                time2_ad=[datetime.datetime(time2_ad[n].year,time2_ad[n].month,odays[(3-n)%4]) for n in range(len(time2_ad))]
                time2_ad=pd.to_datetime(time2_ad)
                timet2=timet2.append(time2_ad)
               if interpolation2:
                #case of the remote stations
                #We take the latest point in time of the climatological adjoint
                #and we extroplate linearly until the datecol point 
                xdata=np.copy(np.arange(len(adjoint_regp)))
                xdata=np.copy(np.arange(xdata[-1],xdata[-1]+back_in_time))
                adjoint_extrapol=affine(xdata, adjoint_regp[-1],asymptote_value)
                nb_month2=np.repeat(np.arange(back_in_time/4),4)
                timet2=pd.to_datetime([timet[-1]-relativedelta(months=x) for x in nb_month2])
                timet2=[datetime.datetime(timet2[n].year,timet2[n].month,odays[(3-n)%4]) for n in range(len(timet2)) if n!=0]
               i_month=[x for x in range(len(timet2)) if (timet2[x].month==mcol)&(timet2[x].year==ycol)&(timet2[x].day==date_col.day)]
               #Supprimer les points bizarre de lextrapolation
               imax_ad=np.where(  np.abs(adjoint_extrapol) == np.max(np.abs(adjoint_extrapol)) )[0]
               diff_max=np.abs(adjoint_extrapol[imax_ad]-adjoint_extrapol[imax_ad-1])
               if diff_max[0] >= 4 :
                 print("outlier occurs at: ",date_a,date_col,pp,vv)
                 print("adjoint",adjoint_extrapol)
                 adjoint_extrapol[imax_ad]=adjoint_extrapol[imax_ad-1]
               adjoint_regp=np.copy(np.squeeze(adjoint_extrapol[i_month]))
              #Loop over index (parameters) for one specific region and date
          col=index_adjoint_ppvvrr.copy(deep=True)
          for ii in index_line:
            #Remplissage de la matrice G (tarantola) ou H: linear tangent 
            matrix_g[ii,col.index]=np.copy(adjoint_regp*col["factor_"+obs_vec.loc[ii]['compound']])
       #Increment d une semaine en arriere
       if not adays[wcol-1]=="01":
         deltad=8
       else:
         mcol=copy.copy((date_col-relativedelta(months=1)).month)
         ycol=copy.copy((date_col-relativedelta(months=1)).year)
         deltad=calendar.monthrange(ycol,mcol)[1]-25+1
       date_col=date_col-relativedelta(days=deltad)

# print index_ctl_vec.loc[np.where(matrix_g[170,:]!=0)]
 #Offset
 offset=index_ctl_vec[index_ctl_vec['parameter']=='offset']
 for jj in offset.index:
  for cc in compound:
   index_compound=obs_vec[obs_vec['compound']==cc.name].index
   matrix_g[index_compound,jj]=offset["factor_"+cc.name].loc[jj]
 matrix_G=matrix_g[:,:len(index_ctl_vec)]
 np.savez_compressed(mdl.storedir+'matrix_G_i',matrix_G)
 np.savez_compressed(storedir+'matrix_g_i',matrix_g)

 ###


