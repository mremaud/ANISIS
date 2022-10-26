#!/usr/bin/env python
#author@Marine Remaud
#Script created the 20th of february
#Purpose: subset of diagnostics aiming at evaluating the inversion results
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
from scipy.stats.stats import pearsonr
import seaborn

#Import the variables in the config.py
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
######Comparison CO2 alone and CO2
#Difference RMSE

dir1="/home/satellites4/mremaud/DA_SYSTEM/LRU4-20082019-2/Gpp0.2Resp0.2Ant0.5Biomass10.9Soil0.3OceanOCS2.0/"
dir2="/home/satellites4/mremaud/DA_SYSTEM/CO2-20082019/Gpp0.2Resp0.2/"
RMSE1=pd.read_pickle(dir1+"rmse.pkl")
RMSE2=pd.read_pickle(dir2+"rmse.pkl")
RMSE1=RMSE1[RMSE1["compound"]=="CO2"]
RMSE2=RMSE2[RMSE2["compound"]=="CO2"]
RMSE1["lat"]=0
RMSE1["$RMSE_{post}^{INV-CO2COS}/RMSE_{post}^{INV-CO2}$"]=0
RMSE1['RE^{sea}']=0

plt.figure(figsize=(10, 2))
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
for stat in RMSE2.station.unique():
 lat=pd.read_pickle("/home/surface1/mremaud/COS/SURFACE/STATION/"+stat+".pkl").lat.mean()
 RMSE1.loc[(RMSE1.station==stat),"lat"]=lat
 RMSE1.loc[(RMSE1.station==stat),"$RMSE_{post}^{INV-CO2COS}/RMSE_{post}^{INV-CO2}$"]=RMSE1[(RMSE1.station==stat)]["$RMSE_{post}$"].iloc[0]/RMSE2[(RMSE2.station==stat)]["$RMSE_{post}$"].iloc[0]
RMSE1=RMSE1.sort_values("lat")
seaborn.scatterplot(x='station', y='$RMSE_{post}^{INV-CO2COS}/RMSE_{post}^{INV-CO2}$', data=RMSE1,color="k")
plt.xlabel("")

#Difference posterior error bar
dir1="/home/satellites4/mremaud/DA_SYSTEM/LRU4-20082019-2//Gpp0.2Resp0.2Ant0.5Biomass10.9Soil0.3OceanOCS2.0/"
dir2="/home/satellites4/mremaud/DA_SYSTEM/CO2-20082019/Gpp0.2Resp0.2//"
index_g1=pd.read_pickle(dir1+"../index_ctl_vec.pkl")
index_g1["sig_P"]=np.load(dir1+"sig_P.npy")
index_g1["sig_B"]=np.load(dir1+"sig_B.npy")
index_g2=pd.read_pickle(dir2+"../index_ctl_vec.pkl")
index_g2["sig_P"]=np.load(dir2+"sig_P.npy")
index_g2["sig_B"]=np.load(dir2+"sig_B.npy")

index_g1=index_g1[(index_g1.parameter=="Gpp")&(index_g1.year>=2009)&(index_g1.REG!="HS")]
index_g2=index_g2[(index_g2.parameter=="Gpp")&(index_g2.year>=2009)&(index_g2.REG!="HS")]
index_g1=index_g1.groupby(["PFT","month"]).mean().reset_index()
index_g2=index_g2.groupby(["PFT","month"]).mean().reset_index()

for vv in range(2,16):
 index_vv=index_g2[index_g2.PFT==vv]
 maxmon=index_vv[index_vv.prior_CO2==index_vv.prior_CO2.min()].month.iloc[0]
 index_g2=index_g2.drop(index_g2[(index_g2.month!=maxmon)&(index_g2.PFT==vv)].index)
 index_g1=index_g1.drop(index_g1[(index_g1.month!=maxmon)&(index_g1.PFT==vv)].index)

index_g1=index_g1.groupby(["PFT"]).mean().reset_index()
index_g2=index_g2.groupby(["PFT"]).mean().reset_index()



index_g1["Reduction of error (%)"]=100*(1-index_g1["sig_P"].values**0.5/index_g1["sig_B"].values**0.5)
index_g1[" "]="INV-CO2COS"
index_g2["Reduction of error (%)"]   =100*(1-index_g2["sig_P"].values**0.5/index_g2["sig_B"].values**0.5)
index_g2[" "]="INV-CO2"
index_tot=index_g1.append(index_g2)
index_tot["PFT"]=index_tot.apply(lambda row: int(row.PFT),axis=1)

plt.figure(figsize=(8, 2))
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 

seaborn.barplot(x='PFT', y='Reduction of error (%)', data=index_tot,hue=" ",palette=['darkorange','royalblue'])


#################################################################################

index_g1=pd.read_pickle(dir1+"/index_g.pkl")
index_ctl_vec1=pd.read_pickle(dir1+"..//index_ctl_vec.pkl")
index_ctl_vec1=index_g1.loc[index_ctl_vec1.index]
index_g2=pd.read_pickle(dir2+"/index_g.pkl")
index_ctl_vec2=pd.read_pickle(dir2+"../index_ctl_vec.pkl")
index_ctl_vec2=index_g2.loc[index_ctl_vec2.index]
sig_B1=np.load(dir1+"sig_B.npy")
sig_P1=np.load(dir1+"sig_P.npy")
sig_B2=np.load(dir2+"sig_B.npy")
sig_P2=np.load(dir2+"sig_P.npy")
add_SIF=1  #Add SIF for the GPP? 
linestyles = ['-', '--', '-.', ':']
ymax=np.asarray([2,0.5,0.1,0,0,0,0,0,0,0,0,0,0,0])*(-1)
ymin=np.asarray([4,1.2,0.5,0.75,1,1.9,1.5,0.6,0.75,1.5,2.5,1,0.3,2])*(-1)
dico={'CO2':[]} 
dico['CO2']=[1,"GtC"]

rep_fig=mdl.storedir+rep_da+'/FIG/'
if not os.path.exists(rep_fig): os.system('mkdir '+rep_fig)
if add_SIF:
  SIF=extract_SIFLMDZ(index_ctl_vec)
f,ax=plt.subplots(nrows=4,ncols=4,figsize=(6,5))
index_veget1=index_ctl_vec1[(index_ctl_vec1.parameter=="Gpp")&(index_ctl_vec1.REG!="HS")]
index_veget2=index_ctl_vec2[(index_ctl_vec2.parameter=="Gpp")&(index_ctl_vec2.REG!="HS")]

ir=0; ic=0 
for vv in range(2,16):
  index_vv1=index_veget1[index_veget1.PFT==vv].copy(deep=True)
  index_vv1['date']=index_vv1.apply(lambda row: to_date(row) ,axis=1)
  index_vv1["error_P"]=sig_P1[index_vv1.index]**0.5*index_vv1["prior_CO2"]
  index_vv1["error_B"]=sig_B1[index_vv1.index]**0.5*index_vv1["prior_CO2"]
  #A priori and a posteriori fluxes 
  index_vv1["flux_B"]=index_vv1["prior_CO2"]
  index_vv1["flux_P"]= index_vv1["post_CO2"]
  #Monthly average after remobing the first year
  cycle1=index_vv1[index_vv1.date.dt.year>=(mdl.begy+1)].groupby("month").mean()
  cycle1["std_B"]=index_vv1[index_vv1.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_B
  cycle1["std_P"]=index_vv1[index_vv1.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_P
  cycle1.reset_index(inplace=True)
  ax[ir,ic].plot(cycle1.month.values,cycle1["flux_B"],color='k',label='a priori GPP')
  ax[ir,ic].plot(cycle1.month.values,cycle1["flux_P"],color='darkorange',label='a posteriori GPP (INV-CO2COS)')
  ax[ir,ic].fill_between(cycle1.month.values,cycle1["flux_B"]-cycle1["error_B"].values, cycle1["flux_B"]+cycle1["error_B"].values, facecolor='grey', alpha=0.2)
  ax[ir,ic].fill_between(cycle1.month.values,cycle1["flux_P"]-cycle1["error_P"].values, cycle1["flux_P"]+cycle1["error_P"].values, facecolor='darkorange', alpha=0.2)
###
  index_vv2=index_veget2[index_veget2.PFT==vv].copy(deep=True)
  index_vv2['date']=index_vv2.apply(lambda row: to_date(row) ,axis=1)
  index_vv2["error_P"]=sig_P2[index_vv2.index]**0.5*index_vv2["prior_CO2"]
  index_vv2["error_B"]=sig_B2[index_vv2.index]**0.5*index_vv2["prior_CO2"]
  #A priori and a posteriori fluxes 
  index_vv2["flux_B"]= index_vv2["prior_CO2"]
  index_vv2["flux_P"]= index_vv2["post_CO2"]
  #Monthly average after remobing the first year
  cycle2=index_vv2[index_vv2.date.dt.year>=(mdl.begy+1)].groupby("month").mean()
  cycle2["std_B"]=index_vv2[index_vv2.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_B
  cycle2["std_P"]=index_vv2[index_vv2.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_P
  cycle2.reset_index(inplace=True)
  ax[ir,ic].plot(cycle2.month.values,cycle2["flux_P"],color='royalblue',label='a posteriori GPP (INV-CO2)')
  ax[ir,ic].fill_between(cycle2.month.values,cycle2["flux_P"]-cycle2["error_P"].values, cycle2["flux_P"]+cycle2["error_P"].values, facecolor='royalblue', alpha=0.2)

  ax[ir,ic].xaxis.set_ticks(np.arange(1, 13, 2))
  ax[ir,ic].set_xlim(1,12)

  if (add_SIF): 
   namereg=index_vv1.REG.unique()[0]
   SIFtmp=SIF[(SIF.PFT==vv)&(SIF.REG==namereg)&(SIF.year>=mdl.begy)].copy(deep=True)
   SIFtmp=SIFtmp.groupby(["month"]).mean().reset_index()
   SIFtmp["date"]=SIFtmp.apply(lambda row:datetime.datetime(int(row.year),int(row.month),1),axis=1)
   SIFtmp.set_index("date",inplace=True)
   imin=np.where(SIFtmp.SIF.values*(-1)==np.min(SIFtmp.SIF.values*(-1)))[0]
   if not SIFtmp.empty:
    ax[ir,ic].vlines(imin+1,ymin[vv-2],ymax[vv-2],colors='darkgreen', linestyles='solid', label='min SIF GOME-2')
    if vv==2: 
     imin=np.where(SIFtmp.SIF.values[:5]*(-1)==np.min(SIFtmp.SIF.values[:5]*(-1)))[0]
     ax[ir,ic].vlines(imin+1,ymin[vv-2],ymax[vv-2],colors='darkgreen', linestyles='solid', label='min SIF GOME-2')
  ###Correlation coefficient
  print("Pearson coefficient a posteriori: ",pearsonr(SIFtmp.SIF.values,cycle2["flux_P"].values)[0])
  print("Pearson coefficient  a priori: ",pearsonr(SIFtmp.SIF.values,cycle2["flux_B"].values)[0])

  ########################## 
  ax[ir,ic].set_title(table_ORC[int(vv)-1][1] ,fontsize=10)
  ax[ir,ic].tick_params(direction="in")
  ax[ir,ic].set_ylim(ymin[vv-2],ymax[vv-2])
  if ic==0: ax[ir,ic].set_ylabel(dico["CO2"][1])
  if (ir==3)&(ic<=2): ax[ir,ic].tick_params(bottom="off")
  if (ir==2)&(ic==3): ax[ir,ic].tick_params(bottom="off")
  if ic!=3:
    ic+=1
  else:
    ic=0; ir+=1
 
plt.subplots_adjust(wspace=0.5,hspace=0.8)
ax[-1,2].set_axis_off()
ax[-1,3].set_axis_off()
ax[3,1].legend(bbox_to_anchor=[1.5, 1])
 #plt.suptitle("a) GPP")  
 #ax[ir,ic].legend(bbox_to_anchor=[1.5, 0.3])
plt.savefig(rep_fig+"Cycle_GPP2.pdf",format="pdf",bbox_inches='tight')
########Diagnostics of the rerview################################################
#
dir2="/home/satellites4/mremaud/DA_SYSTEM/LRU4-20082019-2/"
exp2="/Gpp0.2Resp0.2Ant0.5Biomass10.9Soil0.3OceanOCS2.0_standart/"
obs_vec2=pd.read_pickle(dir2+"/obs_vec.pkl")
obs_vec2["stat"]=obs_vec2.apply(lambda row: row.stat[:3],axis=1)
index_g2=pd.read_pickle(dir2+exp2+"/index_g.pkl")
sim_02=np.load(dir2+"sim_0.npy")
sim_opt2=np.load(dir2+exp2+"sim_opt.npy")

rep_fig=mdl.storedir+rep_da+'/FIG/'
dico={'CO2':[],'COS':[]} 
dico['CO2']=[1,"ppm"]
dico['COS']=[10**6,"ppt"]

list_stations=["NWR_afternoon","LEF","WIS"]
list_fig=[]
for cc in compound:
  f,ax=plt.subplots(nrows=3,ncols=1)
  ff=0
  convert_unit=dico[cc.name][0]
  units=dico[cc.name][1]
  nstat=0
  for nstat,stat in enumerate(list_stations): 
    stat2 = stat[0:3] #For afternoon stations
    mask=(obs_vec['stat']==stat2)&(obs_vec['compound']==cc.name)
    data=obs_vec[mask].copy(deep=True)
    mask2=(obs_vec2['stat']==stat2)&(obs_vec2['compound']==cc.name)
    data2=obs_vec2[mask2].copy(deep=True)
    if data.empty: continue
    data['obs']=data['obs']*convert_unit #Conversion des observation
    data2['obs']=data2['obs']*convert_unit #Conversion des observation
    data['date']=0
    data2['date']=0
    data['date']=data.apply(lambda row: datetime.datetime(row['year'],row['month'],(row['week']-1)*7+3,0,0,0),axis=1)     
    data2['date']=data2.apply(lambda row: datetime.datetime(row['year'],row['month'],(row['week']-1)*7+3,0,0,0),axis=1)
    #Remove the first year
    data=data[(data['date'].dt.year>=begy+1)&(data['date'].dt.year<=endy+5)]
    data2=data2[(data2['date'].dt.year>=begy+1)&(data2['date'].dt.year<=endy+5)]
    #prior
    sim_0_stat=(sim_0[data.index])*convert_unit
    offset_prior=index_g[(index_g['parameter']=='offset')&(index_g['factor_'+cc.name]!=0)].prior.values[0]
    #post
    sim_opt_stat=(sim_opt[data.index])*convert_unit
    sim_opt_stat2=(sim_opt2[data.index])*convert_unit
    offset_post=index_g.loc[(index_g['parameter']=='offset')&(index_g['factor_'+cc.name]!=0),'prior_'+cc.name].values[0]
    offset_post2=index_g2.loc[(index_g2['parameter']=='offset')&(index_g2['factor_'+cc.name]!=0),'prior_'+cc.name].values[0]

    #Observation
    obs_stat=np.copy(data.obs.values)
    error_o=obs_vec.sig_O.values**0.5*convert_unit
    error_stat=(error_o[data.index])

    #detrending
    data['frac']=data.apply(lambda row: fraction_an(row),axis=1)
    time_serie=data[['frac','obs']]
    file_out=mdl.homedir+'modules/tmp-ccgvu.txt'
    os.system('rm -f '+file_out)
    np.savetxt(file_out,time_serie,fmt='%4.8f %3.15f')
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    time_serie=np.loadtxt(file_fit)
    time=[datetime.datetime(int(time_serie[ii,0]),int(time_serie[ii,1]),int(time_serie[ii,2])) for ii in range(len(time_serie[:,0]))]
    obs_stat_seas=(time_serie[:,5]-time_serie[:,6])

    #detrending= prior
    c=[data['frac'].values,sim_0_stat]
    os.system('rm -f '+file_out)
    with open(file_out, "w") as file:
      for x in zip(*c):
        file.write("{0}\t{1}\n".format(*x))
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    sim_0_stat=np.loadtxt(file_fit)
    sim_0_stat_seas=sim_0_stat[:,5]-sim_0_stat[:,6]

    #detrending: post
    c=[data['frac'].values,sim_opt_stat]
    os.system('rm -f '+file_out)
    with open(file_out, "w") as file:
      for x in zip(*c):
        file.write("{0}\t{1}\n".format(*x))
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    sim_opt_stat=np.loadtxt(file_fit)
    sim_opt_stat_seas=sim_opt_stat[:,5]-sim_opt_stat[:,6]

    c=[data['frac'].values,sim_opt_stat2]
    os.system('rm -f '+file_out)
    with open(file_out, "w") as file:
      for x in zip(*c):
        file.write("{0}\t{1}\n".format(*x))
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    sim_opt_stat2=np.loadtxt(file_fit)
    sim_opt_stat_seas2=sim_opt_stat2[:,5]-sim_opt_stat2[:,6]



    list_fig.append(ax[ff].plot(data.date,obs_stat_seas,color='k',label='Obs'))
    ax[ff].fill_between(data.date.values, obs_stat_seas-error_stat, obs_stat_seas+error_stat, facecolor='grey', alpha=0.5)
    list_fig.append(ax[ff].plot(data.date,sim_opt_stat_seas,label='post 2',color='violet'))
    list_fig.append(ax[ff].plot(data.date,sim_opt_stat_seas2,label='post',color='darkorange'))
    list_fig.append(ax[ff].plot(data.date,sim_0_stat_seas,color='darkblue',label='prior')) 
    ax[ff].set_ylabel(units,fontsize=13)
    ax[ff].set_title(stat2,fontsize=12) 
    ax[ff].grid()
    ax[ff].set_xlim(datetime.datetime(begy+1,1,1),datetime.datetime(begy+3,12,31))

    if (ff !=2)&(nstat != len(obs_vec.stat.unique())): 
     ax[ff].xaxis.set_ticklabels([])
     labels = ax[ff].get_xticklabels()
    elif (ff == 2)|(nstat == len(obs_vec.stat.unique())):
     labels = ax[ff].get_xticklabels()
     for i in labels:
       i.set_rotation(30)
    box = ax[ff].get_position()

    ff+=1  
    if ff==3: 
      plt.subplots_adjust(hspace=0.25,right=0.9)
      f.suptitle(cc.name)
      ff=0
      for ii in range(3):
       box = ax[ii].get_position()
       ax[ii].set_position([box.x0, box.y0 ,
                 box.width*0.85, box.height])
      if cc.name=="CO2":
        ax[0].legend(loc='right', bbox_to_anchor=(1.3, 0.6), 
                fancybox=True, shadow=True, fontsize=12)                                                                                                   

      plt.savefig(rep_fig+"/FitStat"+cc.name+".pdf",format='pdf',bbox_inches='tight')
######################################################################################
dir1="/home/satellites4/mremaud/DA_SYSTEM/LRU4-20082019/Gpp0.2Resp0.2Ant0.5Biomass10.9Soil0.3OceanOCS2.0/"
dir2="/home/satellites4/mremaud/DA_SYSTEM/LRU4-20082019-order1/Bios0.2Ant0.5Biomass10.9Soil0.3OceanOCS2.0/"
dico={'CO2':[],'COS':[]} 
dico['CO2']=[1,"GtC"]
dico['COS']=[10**6*32.065/12.,"GgS"]

index_g1=pd.read_pickle(dir1+"/index_g.pkl")
index_ctl_vec1=pd.read_pickle(dir1+"..//index_ctl_vec.pkl")
index_ctl_vec1=index_g1.loc[index_ctl_vec1.index]
index_g2=pd.read_pickle(dir2+"/index_g.pkl")
index_ctl_vec2=pd.read_pickle(dir2+"../index_ctl_vec.pkl")
index_ctl_vec2=index_g2.loc[index_ctl_vec2.index]
sig_B1=np.load(dir1+"sig_B.npy")
sig_P1=np.load(dir1+"sig_P.npy")
sig_B2=np.load(dir2+"sig_B.npy")
sig_P2=np.load(dir2+"sig_P.npy")

linestyles = ['-', '--', '-.', ':']
ymax=np.asarray([2,0.5,0.1,0,0,0,0,0,0,0,0,0,0,0])*(-1)
ymin=np.asarray([4,1.2,0.5,0.75,1,1.9,1.5,0.6,0.75,1.5,2.5,1,0.3,2])*(-1)

rep_fig=mdl.storedir+rep_da+'/FIG/'
if not os.path.exists(rep_fig): os.system('mkdir '+rep_fig)

f,ax=plt.subplots(nrows=4,ncols=4,figsize=(6,5))
index_veget1=index_ctl_vec1[(index_ctl_vec1.parameter=="Gpp")&(index_ctl_vec1.REG!="HS")]
index_veget2=index_ctl_vec2[(index_ctl_vec2.parameter=="Bios")&(index_ctl_vec2.REG!="HS")]

ir=0; ic=0 
for vv in range(2,16):
  index_vv1=index_veget1[index_veget1.PFT==vv].copy(deep=True)
  index_vv1['date']=index_vv1.apply(lambda row: to_date(row) ,axis=1)
  index_vv1["error_P"]=sig_P1[index_vv1.index]**0.5*index_vv1["prior_COS"]*dico["COS"][0]
  index_vv1["error_B"]=sig_B1[index_vv1.index]**0.5*index_vv1["prior_COS"]*dico["COS"][0]
  #A priori and a posteriori fluxes 
  index_vv1["flux_B"]=index_vv1["prior_COS"]*dico["COS"][0]
  index_vv1["flux_P"]= index_vv1["post_COS"]*dico["COS"][0]
  #Monthly average after remobing the first year
  cycle1=index_vv1[index_vv1.date.dt.year>=(2008+1)].groupby("month").mean()
  cycle1["std_B"]=index_vv1[index_vv1.date.dt.year>=(2008+1)].groupby("month").std().flux_B
  cycle1["std_P"]=index_vv1[index_vv1.date.dt.year>=(2008+1)].groupby("month").std().flux_P
  cycle1.reset_index(inplace=True)
  ax[ir,ic].plot(cycle1.month.values,cycle1["flux_B"],color='k',label='a priori')
  #ax[ir,ic].plot(cycle1.month.values,cycle1["flux_P"],color='darkorange',label='a posteriori COS flux')
  ax[ir,ic].fill_between(cycle1.month.values,cycle1["flux_B"]-cycle1["error_B"].values, cycle1["flux_B"]+cycle1["error_B"].values, facecolor='grey', alpha=0.2)
  #ax[ir,ic].fill_between(cycle1.month.values,cycle1["flux_P"]-cycle1["error_P"].values, cycle1["flux_P"]+cycle1["error_P"].values, facecolor='darkorange', alpha=0.2)
###
  index_vv2=index_veget2[index_veget2.PFT==vv].copy(deep=True)
  index_vv2['date']=index_vv2.apply(lambda row: to_date(row) ,axis=1)
  index_vv2["error_P"]=sig_P2[index_vv2.index]**0.5*index_vv2["prior_COS"]*dico["COS"][0]
  index_vv2["error_B"]=sig_B2[index_vv2.index]**0.5*index_vv2["prior_COS"]*dico["COS"][0]
  #A priori and a posteriori fluxes 
  index_vv2["flux_B"]= index_vv2["prior_COS"]*dico["COS"][0]
  index_vv2["flux_P"]= index_vv2["post_COS"]*dico["COS"][0]
  #Monthly average after remobing the first year
  cycle2=index_vv2[index_vv2.date.dt.year>=(2008+1)].groupby("month").mean()
  cycle2["std_B"]=index_vv2[index_vv2.date.dt.year>=(2008+1)].groupby("month").std().flux_B
  cycle2["std_P"]=index_vv2[index_vv2.date.dt.year>=(2008+1)].groupby("month").std().flux_P
  cycle2.reset_index(inplace=True)
  ax[ir,ic].plot(cycle2.month.values,cycle2["flux_B"],color='darkorange',linestyle="--",label='a priori (order 1)')
  ax[ir,ic].fill_between(cycle2.month.values,cycle2["flux_B"]-cycle2["error_B"].values, cycle2["flux_B"]+cycle2["error_B"].values,facecolor='darkorange', alpha=0.2)
 # ax[ir,ic].plot(cycle2.month.values,cycle2["flux_P"],color='royalblue',label='a posteriori flux (order 1)')
  #ax[ir,ic].fill_between(cycle2.month.values,cycle2["flux_P"]-cycle2["error_P"].values, cycle2["flux_P"]+cycle2["error_P"].values, facecolor='royalblue', alpha=0.2)

  ax[ir,ic].xaxis.set_ticks(np.arange(1, 13, 2))
  ax[ir,ic].set_xlim(1,12)

  ax[ir,ic].set_title(table_ORC[int(vv)-1][1] ,fontsize=10)
  ax[ir,ic].tick_params(direction="in")
  #ax[ir,ic].set_ylim(ymin[vv-2],ymax[vv-2])
  if ic==0: ax[ir,ic].set_ylabel(dico["COS"][1])
  if (ir==3)&(ic<=2): ax[ir,ic].tick_params(bottom="off")
  if (ir==2)&(ic==3): ax[ir,ic].tick_params(bottom="off")
  if ic!=3:
    ic+=1
  else:
    ic=0; ir+=1
 
plt.subplots_adjust(wspace=0.5,hspace=0.8)
ax[-1,2].set_axis_off()
ax[-1,3].set_axis_off()
ax[3,1].legend(bbox_to_anchor=[1.5, 1])
 #plt.suptitle("a) GPP")  
 #ax[ir,ic].legend(bbox_to_anchor=[1.5, 0.3])
#plt.savefig(rep_fig+"Cycle_GPP2.pdf",format="pdf",bbox_inches='tight')
######################################################################################################
dir2="/home/satellites4/mremaud/DA_SYSTEM/LRU4-20082019-order1-post/"
exp2="Bios0.2Ant0.5Biomass10.9Soil0.3OceanOCS2.0/"
obs_vec2=pd.read_pickle(dir2+"/obs_vec.pkl")
obs_vec2["stat"]=obs_vec2.apply(lambda row: row.stat[:3],axis=1)
index_g2=pd.read_pickle(dir2+exp2+"/index_g.pkl")
sim_02=np.load(dir2+"sim_0.npy")
sim_opt2=np.load(dir2+exp2+"sim_opt.npy")

rep_fig=mdl.storedir+rep_da+'/FIG/'
dico={'COS':[]} 
dico['COS']=[10**6,"ppt"]

list_stations=["ALT","LEF","MLO_afternoon"]
list_fig=[]
for cc in [COS]:
  f,ax=plt.subplots(nrows=3,ncols=1)
  ff=0
  convert_unit=dico[cc.name][0]
  units=dico[cc.name][1]
  nstat=0
  for nstat,stat in enumerate(list_stations): 
    stat2 = stat[0:3] #For afternoon stations
    mask=(obs_vec['stat']==stat2)&(obs_vec['compound']==cc.name)
    data=obs_vec[mask].copy(deep=True)
    mask2=(obs_vec2['stat']==stat2)&(obs_vec2['compound']==cc.name)
    data2=obs_vec2[mask2].copy(deep=True)
    if data.empty: continue
    data['obs']=data['obs']*convert_unit #Conversion des observation
    data2['obs']=data2['obs']*convert_unit #Conversion des observation
    data['date']=0
    data2['date']=0
    data['date']=data.apply(lambda row: datetime.datetime(row['year'],row['month'],(row['week']-1)*7+3,0,0,0),axis=1)     
    data2['date']=data2.apply(lambda row: datetime.datetime(row['year'],row['month'],(row['week']-1)*7+3,0,0,0),axis=1)
    #Remove the first year
    data=data[(data['date'].dt.year>=begy+1)&(data['date'].dt.year<=endy+5)]
    data2=data2[(data2['date'].dt.year>=begy+1)&(data2['date'].dt.year<=endy+5)]
    #prior
    sim_0_stat=(sim_02[data2.index])*convert_unit
    offset_prior=index_g[(index_g['parameter']=='offset')&(index_g['factor_'+cc.name]!=0)].prior.values[0]
    #post
    sim_opt_stat=(sim_opt[data.index])*convert_unit
    sim_opt_stat2=(sim_02[data2.index])*convert_unit
    offset_post=index_g.loc[(index_g['parameter']=='offset')&(index_g['factor_'+cc.name]!=0),'prior_'+cc.name].values[0]
    offset_post2=index_g2.loc[(index_g2['parameter']=='offset')&(index_g2['factor_'+cc.name]!=0),'prior_'+cc.name].values[0]

    #Observation
    obs_stat=np.copy(data.obs.values)
    error_o=obs_vec.sig_O.values**0.5*convert_unit
    error_stat=(error_o[data.index])

    #detrending
    data2['frac']=data2.apply(lambda row: fraction_an(row),axis=1)
    data['frac']=data.apply(lambda row: fraction_an(row),axis=1)
    time_serie=data[['frac','obs']]
    file_out=mdl.homedir+'modules/tmp-ccgvu.txt'
    os.system('rm -f '+file_out)
    np.savetxt(file_out,time_serie,fmt='%4.8f %3.15f')
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    time_serie=np.loadtxt(file_fit)
    time=[datetime.datetime(int(time_serie[ii,0]),int(time_serie[ii,1]),int(time_serie[ii,2])) for ii in range(len(time_serie[:,0]))]
    obs_stat_seas=(time_serie[:,5]-time_serie[:,6])

    #detrending= prior
    c=[data['frac'].values,sim_0_stat]
    os.system('rm -f '+file_out)
    with open(file_out, "w") as file:
      for x in zip(*c):
        file.write("{0}\t{1}\n".format(*x))
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    sim_0_stat=np.loadtxt(file_fit)
    sim_0_stat_seas=sim_0_stat[:,5]-sim_0_stat[:,6]

    #detrending: post
    c=[data['frac'].values,sim_opt_stat]
    os.system('rm -f '+file_out)
    with open(file_out, "w") as file:
      for x in zip(*c):
        file.write("{0}\t{1}\n".format(*x))
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    sim_opt_stat=np.loadtxt(file_fit)
    sim_opt_stat_seas=sim_opt_stat[:,5]-sim_opt_stat[:,6]

    c=[data2['frac'].values,sim_opt_stat2]
    os.system('rm -f '+file_out)
    with open(file_out, "w") as file:
      for x in zip(*c):
        file.write("{0}\t{1}\n".format(*x))
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    sim_opt_stat2=np.loadtxt(file_fit)
    sim_opt_stat_seas2=sim_opt_stat2[:,5]-sim_opt_stat2[:,6]



    list_fig.append(ax[ff].plot(data.date,obs_stat_seas,color='k',label='Obs'))
    ax[ff].fill_between(data.date.values, obs_stat_seas-error_stat, obs_stat_seas+error_stat, facecolor='grey', alpha=0.5)
    list_fig.append(ax[ff].plot(data.date,sim_opt_stat_seas,label='post',color='darkorange'))
    list_fig.append(ax[ff].plot(data2.date,sim_opt_stat_seas2,label='post (order 1)',color='violet'))
    ax[ff].set_ylabel(units,fontsize=13)
    ax[ff].set_title(stat2,fontsize=12) 
    ax[ff].grid()
    ax[ff].set_xlim(datetime.datetime(begy+1,1,1),datetime.datetime(begy+3,12,31))

    if (ff !=2)&(nstat != len(obs_vec.stat.unique())): 
     ax[ff].xaxis.set_ticklabels([])
     labels = ax[ff].get_xticklabels()
    elif (ff == 2)|(nstat == len(obs_vec.stat.unique())):
     labels = ax[ff].get_xticklabels()
     for i in labels:
       i.set_rotation(30)
    box = ax[ff].get_position()

    ff+=1  
    if ff==3: 
      plt.subplots_adjust(hspace=0.25,right=0.9)
      f.suptitle(cc.name)
      ff=0
      for ii in range(3):
       box = ax[ii].get_position()
       ax[ii].set_position([box.x0, box.y0 ,
                 box.width*0.85, box.height])
      if cc.name=="CO2":
        ax[0].legend(loc='right', bbox_to_anchor=(1.3, 0.6), 
                fancybox=True, shadow=True, fontsize=12)                                                                                                   

      plt.savefig(rep_fig+"/FitStat"+cc.name+".pdf",format='pdf',bbox_inches='tight')



