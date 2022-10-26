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


#Import the variables in the config.py
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]

########Diagnostics of the paper################################################

def budget(index_g,index_unopt,Errors,rep_da):
 rep_fig=mdl.storedir+rep_da+'/FIG/'
 if not os.path.exists(rep_fig): os.system('mkdir '+rep_fig)
 dico={'CO2':[],'COS':[]} 
 dico['CO2']=[1,"[GtC/y]",1]
 dico['COS']=[10**6*getattr(Masse_molaire, 'COS')/getattr(Masse_molaire, 'CO2'),"[GgS/y]",10**6] 


 f,axes=plt.subplots(nrows=1,ncols=2,figsize=(15,3))
 for nc,cc in  enumerate(compound):
   table={'Year':[],'Prior':[],'Post':[],'Error_B':[],'Error_P':[]}
   table=pd.DataFrame(table)
   index_cc=index_g[index_g.year>=begy+1]
   index_cc=index_cc[index_cc['prior_'+cc.name]!=0]
   index_cc=index_cc.groupby(['parameter','year']).sum().reset_index()
   
   #Remove the first year
   table['parameter']=index_cc['parameter']
   table['Year']=index_cc['year'].values
   table=table.round(1)
   table["Prior"]=index_cc['prior_'+cc.name]*dico[cc.name][0] 
   table["Post"] =index_cc['post_'+cc.name]*dico[cc.name][0] 
   table["Error_B"] = 0.
   table["Error_P"] = 0.
   
   table.loc[table.parameter=="offset","Post"] =index_cc['post_'+cc.name]*dico[cc.name][2]
   table.loc[table.parameter=="offset","Prior"]=index_cc['prior_'+cc.name]*dico[cc.name][2]
 
   for pp in table.parameter.unique():
     if  (pp in index_unopt.parameter.unique())&(cc.name=="COS"):
      table=table.drop(index=table[(table.parameter==pp)&(table.Year>begy+1)].index)

   table=table.groupby(["parameter"]).mean().reset_index()
   print(table)
   for pp in table.parameter.unique():
    if pp in index_unopt.parameter.unique(): continue
    table.loc[(table.parameter==pp),"Error_B"]=Errors[(Errors.parameter==pp)&(Errors["compound"]==cc.name)].Error_B.values
    table.loc[(table.parameter==pp),"Error_P"]=Errors[(Errors.parameter==pp)&(Errors["compound"]==cc.name)].Error_P.values
   table['parameter']=table.apply(lambda row: to_name(row),axis=1)
   table=table.rename(columns={"parameter":"Source"})
   if cc.name=="COS": table.loc[table.Source=="Gpp","Source"]="Vegetation"

   table=table[['Source','Prior','Post','Error_B','Error_P']]
   table=table.round(1)

   table.loc[table.Source!="offset","Prior"]=table.loc[table.Source!="offset","Prior"].astype(str)+ "±" + table.loc[table.Source!="offset","Error_B"].round(2).astype(str) 
   table.loc[table.Source!="offset","Post"] =table['Post'].round(2).astype(str)+ "±" + table.loc[table.Source!="offset","Error_P"].astype(str) 
   table=table[['Source','Prior','Post']]
   render_mpl_table(table,axes[nc], col_width=200, row_height=10.0, font_size=8,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,ax=None)
   
   axes[nc].set_title(cc.name+" fluxes "+dico[cc.name][1],fontsize=12)
 plt.savefig(rep_fig+"/Budget.pdf",format="pdf",bbox_inches='tight')



def Error_tot(index_ctl_vec,rep_da):
 list_process=index_ctl_vec.parameter.unique()
 list_process=[ff for ff in list_process if "ffset" not in ff]
 Matrix_B=np.load(mdl.storedir+rep_da+"corr.npy")
 Matrix_P=np.load(mdl.storedir+rep_da+"A.npy")

 Table_error={"compound":[],"parameter":[],"Error_P":[],"Error_B":[]}
 for cc in compound:
  X=index_ctl_vec['prior_'+cc.name].values
  Matrix_B2=np.multiply(np.multiply(X.reshape(-1,1),X.T),Matrix_B)
  Matrix_P2=np.multiply(np.multiply(X.reshape(-1,1),X.T),Matrix_P)

  for yy in range(begy+1,endy):
   for pp in list_process:
    index_pp=index_ctl_vec[(index_ctl_vec.parameter==pp)&(index_ctl_vec.year==yy)].copy(deep=True).index
    Error_B=np.sum(Matrix_B2[np.ix_(index_pp,index_pp)])**0.5
    Error_P=np.sum(Matrix_P2[np.ix_(index_pp,index_pp)])**0.5
    if Error_B==0: continue
    if (cc.name != "CO2"):
     Error_B*=35.065/12.*10**6
     Error_P*=35.065/12.*10**6
    Table_error["compound"].append(cc.name)
    Table_error["parameter"].append(pp)
    Table_error["Error_B"].append(Error_B)
    Table_error["Error_P"].append(Error_P)
 Table_error=pd.DataFrame(Table_error)
 Table_error=Table_error.groupby(["compound","parameter"]).mean().reset_index()
 return Table_error

def ts_station(obs_vec,index_g,sim_0,sim_opt,rmse,sig_o,rep_da):
 """
  obs_vec: observation data frame
  sim_0:  a priori concentrations
  
 """
 rep_fig=mdl.storedir+rep_da+'/FIG/'
 dico={'CO2':[],'COS':[]} 
 dico['CO2']=[1,"ppm"]
 dico['COS']=[10**6,"ppt"]
 list_stations=["BRW","NWR_afternoon","LEF"]
 list_fig=[]
 for cc in compound:
  f,ax=plt.subplots(nrows=3,ncols=1)
  ff=0
  convert_unit=dico[cc.name][0]
  units=dico[cc.name][1]
  nstat=0
  for nstat,stat in enumerate(list_stations): 
    stat2 = stat[0:3] #For afternoon stations
    mask=(obs_vec['stat']==stat)&(obs_vec['compound']==cc.name)
    data=obs_vec[mask].copy(deep=True)
    if data.empty: continue
    data['obs']=data['obs']*convert_unit #Conversion des observation
    data['date']=0
    data['date']=data.apply(lambda row: datetime.datetime(row['year'],row['month'],(row['week']-1)*7+3,0,0,0),axis=1)     
    #Remove the first year
    data=data[(data['date'].dt.year>=begy+1)&(data['date'].dt.year<=endy+5)]
    #prior
    sim_0_stat=(sim_0[data.index])*convert_unit
    offset_prior=index_g[(index_g['parameter']=='offset')&(index_g['factor_'+cc.name]!=0)].prior.values[0]

    #post
    sim_opt_stat=(sim_opt[data.index])*convert_unit
    offset_post=index_g.loc[(index_g['parameter']=='offset')&(index_g['factor_'+cc.name]!=0),'prior_'+cc.name].values[0]

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

    list_fig.append(ax[ff].plot(data.date,obs_stat_seas,color='k',label='Obs'))
    ax[ff].fill_between(data.date.values, obs_stat_seas-error_stat, obs_stat_seas+error_stat, facecolor='grey', alpha=0.5)
    list_fig.append(ax[ff].plot(data.date,sim_opt_stat_seas,label='post',color='darkorange'))
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

 #Misfit tabular: Add RMSE, supress Nan
 f,axes=plt.subplots(nrows=1,ncols=2,figsize=(13,6))
 for ic,cc in enumerate(compound):
   rmse_cc=rmse[rmse['compound']==cc.name]
   rmse_cc=rmse_cc[['station','$RE$','$RMSE_{prior}^{seas}$','$RMSE_{post}^{seas}$','$RE^{seas}$','$\chi^2$']]
   rmse_cc=rmse_cc.round(2)
   rmse_cc.dropna(inplace=True)
   render_mpl_table(rmse_cc,axes[ic], col_width=35, row_height=6, font_size=9,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0., 0, 1, 1], header_columns=0,ax=None)
   axes[ic].set_title(cc.name)
 f.subplots_adjust(wspace=0.05) 
 plt.savefig(rep_fig+"/RMSE.pdf",format="pdf",bbox_inches='tight')


 obs_vec["stat"]=obs_vec.apply(lambda row: row.stat.upper()[:3],axis=1)
 for cc in ["CO2","COS"]:
  obs_cc=obs_vec[obs_vec["compound"]==cc]
  sig_cc=sig_o[obs_cc.index]**0.5
  if cc == "COS": sig_cc*=10**6
  obs_cc["error"]=0

  obs_cc["error"]=sig_cc
  obs_cc=obs_cc.groupby("stat").mean().reset_index()
  obs_cc=obs_cc.sort_values("lat").reset_index()
  fig, ax = plt.subplots(figsize=(10,8), facecolor='white', dpi= 80)
  ax.vlines(x=obs_cc.index, ymin=0, ymax=obs_cc.error, color='firebrick', alpha=0.7, linewidth=20)
  for i, cty in enumerate(obs_cc.error):
    ax.text(i, cty+0.5, round(cty, 1), horizontalalignment='center',fontsize=14)


  plt.xticks(obs_cc.index, obs_cc.stat.str.upper(), rotation=60, horizontalalignment='right', fontsize=14)
  plt.yticks(fontsize=14)
  #ax.set_ylabel(fontsize=14)
  if cc == "CO2":
   ax.set_title("a) "+cc, fontdict={'size':18})
   ax.set(ylabel='Error (ppm)', ylim=(0, 15))
   plt.ylabel(ylabel='Error (ppm)',fontsize=14)
  else:
   ax.set_title("b) "+cc, fontdict={'size':18})
   ax.set(ylabel='Error (ppt)', ylim=(0, 48))
   plt.ylabel('Error (ppt)',fontsize=14)
  plt.savefig(rep_fig+"/Error_"+cc+".pdf",format="pdf",bbox_inches='tight')


def cycle_GPP2(index_ctl_vec,index_g,sig_B,sig_P,rep_da):
 """
 Arguments: 
  - index_ctl_vec: details about the control vector (and the prior vector)
  - index_g: same as above but with the unoptimized fluxes
  - sig_B : variance of the prior fluxes
  - sig_P : variance of posterior fluxes
 return: plots of the prior and posterior seasonal cycles and their associated incertitudes
 """          
      
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
 index_veget=index_ctl_vec[(index_ctl_vec.parameter=="Gpp")&(index_ctl_vec.REG!="HS")]
 ir=0; ic=0 
 for vv in range(2,16):
  index_vv=index_veget[index_veget.PFT==vv].copy(deep=True)
  index_vv['date']=index_vv.apply(lambda row: to_date(row) ,axis=1)
  index_vv["error_P"]=sig_P[index_vv.index]**0.5*index_vv["prior_CO2"]
  index_vv["error_B"]=sig_B[index_vv.index]**0.5*index_vv["prior_CO2"]
  #A priori and a posteriori fluxes 
  index_vv["flux_B"]=index_vv["prior_CO2"]
  index_vv["flux_P"]= index_vv["post_CO2"]
  #Monthly average after remobing the first year
  cycle=index_vv[index_vv.date.dt.year>=(mdl.begy+1)].groupby("month").mean()
  cycle["std_B"]=index_vv[index_vv.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_B
  cycle["std_P"]=index_vv[index_vv.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_P
  cycle.reset_index(inplace=True)
  ax[ir,ic].plot(cycle.month.values,cycle["flux_B"],color='k',label='a priori GPP')
  ax[ir,ic].plot(cycle.month.values,cycle["flux_P"],color='darkorange',label='a posteriori GPP')
  ax[ir,ic].fill_between(cycle.month.values,cycle["flux_B"]-cycle["error_B"].values, cycle["flux_B"]+cycle["error_B"].values, facecolor='grey', alpha=0.2)
  ax[ir,ic].fill_between(cycle.month.values,cycle["flux_P"]-cycle["error_P"].values, cycle["flux_P"]+cycle["error_P"].values, facecolor='darkorange', alpha=0.2)
  ax[ir,ic].xaxis.set_ticks(np.arange(1, 13, 2))
  ax[ir,ic].set_xlim(1,12)

  if (add_SIF): 
   namereg=index_vv.REG.unique()[0]
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
  print("Pearson coefficient a posteriori: ",pearsonr(SIFtmp.SIF.values,cycle["flux_P"].values)[0])
  print("Pearson coefficient  a priori: ",pearsonr(SIFtmp.SIF.values,cycle["flux_B"].values)[0])

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
 
 plt.subplots_adjust(wspace=0.45,hspace=0.8)
 ax[-1,2].set_axis_off()
 ax[-1,3].set_axis_off()
 ax[3,1].legend(bbox_to_anchor=[1.5, 1])
 plt.suptitle("a) GPP")  
 #ax[ir,ic].legend(bbox_to_anchor=[1.5, 0.3])
 plt.savefig(rep_fig+"Cycle_GPP2.pdf",format="pdf",bbox_inches='tight')





###################################################################################
def cycle_flux(index_ctl_vec,index_g,sig_B,sig_P,rep_da):
 """
 Arguments: 
  - index_ctl_vec: details about the control vector (and the prior vector)
  - index_g: same as above but with the unoptimized fluxes
  - sig_B : variance of the prior fluxes
  - sig_P : variance of posterior fluxes
 return: plots of the prior and posterior seasonal cycles and their associated incertitudes
 """          
      
 add_SIF=1  #Add SIF for the GPP? 
 linestyles = ['-', '--', '-.', ':']

 dico={'CO2':[],'COS':[]} 
 dico['CO2']=[1,"GtC"]
 dico['COS']=[10**6*getattr(Masse_molaire, 'COS')/getattr(Masse_molaire, 'CO2'),"GgS"]
 rep_fig=mdl.storedir+rep_da+'/FIG/'
 if not os.path.exists(rep_fig): os.system('mkdir '+rep_fig)
 if add_SIF:
  SIF=extract_SIFLMDZ(index_ctl_vec)

 for cc in compound:
  index_cc=index_ctl_vec[index_ctl_vec['factor_'+cc.name]!=0].copy(deep=True)
  processus=index_cc.parameter.unique()
  ir=0; ic=0
  ifig=1  #Nombre de pages
  ic_fig=1 # Nombre de suplots
  nb_fig=len(index_cc.groupby(['parameter','PFT','REG']).mean())
  nb_fig+=len(index_cc[index_cc.PFT.isnull()].groupby(['parameter','REG']).mean())-1
  f,ax=plt.subplots(nrows=5,ncols=2)
  f.set_size_inches((8.27,11.7))
  f.suptitle(cc.name+" fluxes")

  for pp in processus:
   if pp == 'offset': continue
   index_pp=index_cc[index_cc.parameter==pp].copy(deep=True)
   veget=index_pp.PFT.unique()
   for vv in veget:
    index_vv=index_pp[index_pp.PFT==vv].copy(deep=True) if (not np.isnan(vv)) else index_pp.copy(deep=True)
    list_reg=index_vv.REG.unique()
    index_vv['date']=index_vv.apply(lambda row: to_date(row) ,axis=1)
    for ireg,namereg in enumerate(list_reg):
     index_rr=index_vv[index_vv.REG==namereg].copy(deep=True)
     #A priori and a posteriori errors
     index_rr["error_P"]=sig_P[index_rr.index]**0.5*dico[cc.name][0]*index_rr["prior_"+cc.name]
     index_rr["error_B"]=sig_B[index_rr.index]**0.5*dico[cc.name][0]*index_rr["prior_"+cc.name]
     #A priori and a posteriori fluxes 
     index_rr["flux_B"]=index_rr["prior_"+cc.name].values*dico[cc.name][0]
     index_rr["flux_P"]= index_rr["post_"+cc.name].values*dico[cc.name][0]
     #Monthly average after remobing the first year
     index_rr["month"]=index_rr.date.dt.month
     cycle=index_rr[index_rr.date.dt.year>=(mdl.begy+1)].groupby("month").mean()
     cycle["std_B"]=index_rr[index_rr.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_B
     cycle["std_P"]=index_rr[index_rr.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_P
     cycle.reset_index(inplace=True)
     #Draw plots
     ax[ir,ic].plot(cycle.month.values,cycle["flux_B"],color='k',label='prior')
     ax[ir,ic].plot(cycle.month.values,cycle["flux_P"],color='darkorange',label='post')
 #    ax[ir,ic].fill_between(cycle.month.values, cycle["flux_B"]-cycle["std_B"].values, cycle["flux_B"]+cycle["std_B"].values, facecolor='grey', alpha=0.2)
 #    ax[ir,ic].fill_between(cycle.month.values, cycle["flux_P"]-cycle["std_P"].values, cycle["flux_P"]+cycle["std_P"].values, facecolor='darkorange', alpha=0.2)
     ax[ir,ic].fill_between(cycle.month.values, cycle["flux_B"]-cycle["error_B"].values, cycle["flux_B"]+cycle["error_B"].values, facecolor='grey', alpha=0.2)
     ax[ir,ic].fill_between(cycle.month.values, cycle["flux_P"]-cycle["error_P"].values, cycle["flux_P"]+cycle["error_P"].values, facecolor='darkorange', alpha=0.2)
     ax[ir,ic].xaxis.set_ticks(np.arange(1, 13, 1))

     #Grid
     if (pp == "Gpp")&(add_SIF):
      ax[ir,ic].grid(axis="x")
     else:
      ax[ir,ic].grid()
     #Add SIF diagnostics
     if (add_SIF)&(pp=="Gpp"):
      SIFtmp=SIF[(SIF.PFT==vv)&(SIF.REG==namereg)&(SIF.year>=mdl.begy)].copy(deep=True)
      SIFtmp=SIFtmp.groupby(["month"]).mean().reset_index()
      if not  SIFtmp.empty:
       ax1=ax[ir,ic]
       ax2 = ax[ir,ic].twinx()
       ax2.plot(SIFtmp.month, SIFtmp.SIF*(-1), color="darkgreen")
       #ax2.set_yticklabels(fontsize=9)
       ax2.tick_params(direction="in",labelsize=6)
       if ic ==1 : ax2.set_ylabel("SIF ($mW/nm/sr/m^2$)",color="darkgreen")
       for t in ax2.get_yticklabels():
        t.set_color("darkgreen")
     #End SIF diagnostics
     if pp=="Biosphere": ax[ir,ic].set_ylim(0,-25)
     if ic==0: ax[ir,ic].set_ylabel(dico[cc.name][1])

     #Subplot titles
     try:
      title=getattr(Vector_control, pp).name
     except:
      title=pp
     if (pp=="Gpp")&(cc.name=="COS"): title="plant uptake"
     title= title+"\n "+table_ORC[int(vv)-1][0] if not np.isnan(vv) else pp
     title=title if namereg=="GLOBE" else title+", "+namereg

     ax[ir,ic].set_title(title,fontsize=10)
     ax[ir,ic].tick_params(direction="in")
     plt.subplots_adjust(hspace=1.)

     for tick in ax[ir,ic].get_xticklabels():
      tick.set_rotation(30)
     if (ir==4)&(ic==0):
      ir=0; ic=1
     elif (ir==4)&(ic==1):
      ir=0; ic=0
      ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(1.1, -0.5), ncol=2)
      plt.savefig(rep_fig+"/CycleSais_"+cc.name+str(ifig)+".pdf",format='pdf')
      f,ax=plt.subplots(nrows=5,ncols=2)
      f.set_size_inches((8.27,11.7))
      f.suptitle(cc.name+" fluxes")
      ifig+=1
     else:
      ir+=1
     ic_fig+=1

  #Eliminate the frames
  if (nb_fig<10)&(nb_fig%10!=0):
     nb_frest=10-nb_fig%10
     if nb_frest/5 == 0:
      for ii in range(5):
       ax[ii,1].set_axis_off()
     elif nb_frest/5 == 1:
      for ii in range(nb_frest%5,5):
       ax[ii,1].set_axis_off()
     plt.savefig(rep_fig+"/CycleSais_"+cc.name+str(ifig)+".pdf",format='pdf')
  elif (ifig==(nb_fig/10+1))&(nb_fig%10!=0):
     nb_frest=nb_fig%10
     if nb_frest/5 == 0:
      for ii in range(5):
       ax[ii,1].set_axis_off()
      for ii in range(nb_frest,5):
       ax[ii,0].set_axis_off()
     elif nb_frest/5 == 1:
      for ii in range(nb_frest%5,5):
       ax[ii,1].set_axis_off()
     plt.savefig(rep_fig+"/CycleSais_"+cc.name+str(ifig)+".pdf",format='pdf')

############################################################################
def cycle_GPP(index_ctl_vec,index_g,sig_B,sig_P,rep_da):
 """
 Arguments: 
  - index_ctl_vec: details about the control vector (and the prior vector)
  - index_g: same as above but with the unoptimized fluxes
  - sig_B : variance of the prior fluxes
  - sig_P : variance of posterior fluxes
 return: plots of the prior and posterior seasonal cycles and their associated incertitudes
 """          
      
 add_SIF=1  #Add SIF for the GPP? 
 linestyles = ['-', '--', '-.', ':']

 dico={'CO2':[]} 
 dico['CO2']=[1,"GtC"]

 rep_fig=mdl.storedir+rep_da+'/FIG/'
 if not os.path.exists(rep_fig): os.system('mkdir '+rep_fig)
 if add_SIF:
  SIF=extract_SIFLMDZ(index_ctl_vec)
 f,ax=plt.subplots(nrows=4,ncols=4,figsize=(8,5))
 index_veget=index_ctl_vec[(index_ctl_vec.parameter=="Gpp")&(index_ctl_vec.REG!="HS")]
 #Remove the last year
 index_veget=index_veget[(index_veget.year>=2009)&(index_veget.year<=2018)]
 SIF=SIF[(SIF.year>=2009)&(SIF.year<=2018)]
 ir=0; ic=0 
 for vv in range(2,16):
  index_vv=index_veget[index_veget.PFT==vv].copy(deep=True)
  index_vv['date']=index_vv.apply(lambda row: to_date(row) ,axis=1)
  index_vv["error_P"]=sig_P[index_vv.index]**0.5*index_vv["prior_CO2"]
  index_vv["error_B"]=sig_B[index_vv.index]**0.5*index_vv["prior_CO2"]
  #A priori and a posteriori fluxes 
  index_vv["flux_B"]=index_vv["prior_CO2"]
  index_vv["flux_P"]= index_vv["post_CO2"]
  #Monthly average after remobing the first year
  cycle=index_vv[index_vv.date.dt.year>=(mdl.begy+1)].groupby("month").mean()
  cycle["std_B"]=index_vv[index_vv.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_B
  cycle["std_P"]=index_vv[index_vv.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_P
  cycle.reset_index(inplace=True)
  ax[ir,ic].plot(cycle.month.values,cycle["flux_B"],color='k',label='a priori GPP')
  ax[ir,ic].plot(cycle.month.values,cycle["flux_P"],color='darkorange',label='a posteriori GPP')
  ax[ir,ic].fill_between(cycle.month.values,cycle["flux_B"]-cycle["error_B"].values, cycle["flux_B"]+cycle["error_B"].values, facecolor='grey', alpha=0.2)
  ax[ir,ic].fill_between(cycle.month.values,cycle["flux_P"]-cycle["error_P"].values, cycle["flux_P"]+cycle["error_P"].values, facecolor='darkorange', alpha=0.2)
  ax[ir,ic].xaxis.set_ticks(np.arange(1, 13, 2))
  ax[ir,ic].set_xlim(1,12)
  #if vv in [1,4,5,6,9,10,14]:
  # ax[ir,ic].set_ylim(-1,0)
  #if vv !=2:ax[ir,ic].yaxis.set_ticks(np.arange(int(min(min(cycle["flux_B"]),min(cycle["flux_P"]))), 0, 0.5))
  #Grid
  if (add_SIF):
    ax[ir,ic].grid(axis="x")
  else:
    ax[ir,ic].grid()
  #Add SIF diagnostics
  if (add_SIF): 
   namereg=index_vv.REG.unique()[0]
   SIFtmp=SIF[(SIF.PFT==vv)&(SIF.REG==namereg)&(SIF.year>=mdl.begy)].copy(deep=True)
   SIFtmp=SIFtmp.groupby(["month"]).mean().reset_index()
   #Normalized SIF: linear scaling
   SIFtmp["SIF"]=(SIFtmp.SIF.values-np.min(SIFtmp.SIF.values) )/( np.max(SIFtmp.SIF.values) - np.min(SIFtmp.SIF.values) )
   SIFtmp["date"]=SIFtmp.apply(lambda row:datetime.datetime(int(row.year),int(row.month),1),axis=1)
   ###Correlation coefficient
   print("Pearson coefficient a posteriori: ",pearsonr(SIFtmp.SIF.values,cycle["flux_P"].values)[0])
   print("Pearson coefficient  a priori: ",pearsonr(SIFtmp.SIF.values,cycle["flux_B"].values)[0])
   if not SIFtmp.empty:
    ax1=ax[ir,ic]
    ax2 = ax[ir,ic].twinx()
    ax2.plot(SIFtmp.month, SIFtmp.SIF*(-1), color="darkgreen",label="Normalized SIF GOME-2")
    ax2.tick_params(direction="in",labelsize=8,color='darkgreen')
    ax1.tick_params(direction="in",labelsize=8)
    ax2.spines['right'].set_color('darkgreen')


    if ic ==3 : ax2.set_ylabel("SIF \n $mW/nm/sr/m^2$",color="darkgreen",fontsize=8)
    for t in ax2.get_yticklabels():
      t.set_color("darkgreen")
  #End SIF diagnostics
  ax[ir,ic].set_title(table_ORC[int(vv)-1][1] ,fontsize=10)
  ax[ir,ic].tick_params(direction="in")

  if ic==0: ax[ir,ic].set_ylabel(dico["CO2"][1])
  if (ir==3)&(ic<=2): ax[ir,ic].tick_params(bottom="off")
  if (ir==2)&(ic==3): ax[ir,ic].tick_params(bottom="off")
  if ic!=3:
    ic+=1
  else:
    ic=0; ir+=1
 
 plt.subplots_adjust(wspace=0.7,hspace=0.6)
 ax[-1,2].set_axis_off()
 ax[-1,3].set_axis_off()
 ax[3,1].legend(bbox_to_anchor=[1.5, 1])
 ax2.legend(bbox_to_anchor=[1.5, 0.3])
 plt.savefig(rep_fig+"Cycle_GPP.pdf",format="pdf",bbox_inches='tight')


def cycle_Soil(index_ctl_vec,index_g,sig_B,sig_P,rep_da):
 """
 Arguments: 
  - index_ctl_vec: details about the control vector (and the prior vector)
  - index_g: same as above but with the unoptimized fluxes
  - sig_B : variance of the prior fluxes
  - sig_P : variance of posterior fluxes
 return: plots of the prior and posterior seasonal cycles and their associated incertitudes
 """          
 
 ymin=[2,0.5,0.1,0,0,0,0,0,0,0,0,0,0,0]
 ymax=[4,1.2,0.5,0.75,1,1.9,1.5,0.6,0.75,1.5,2.5,1,0.3,2]
 linestyles = ['-', '--', '-.', ':']

 dico={'CO2':[]} 
 dico['CO2']=[1,"GtC"]

 rep_fig=mdl.storedir+rep_da+'/FIG/'
 if not os.path.exists(rep_fig): os.system('mkdir '+rep_fig)

 f,ax=plt.subplots(nrows=4,ncols=4,figsize=(6,5))
 index_veget=index_ctl_vec[(index_ctl_vec.parameter=="Soil")&(index_ctl_vec.REG!="HS")]
 ir=0; ic=0 
 for vv in range(2,16):
  index_vv=index_veget[index_veget.PFT==vv].copy(deep=True)
  index_vv['date']=index_vv.apply(lambda row: to_date(row) ,axis=1)
  index_vv["error_P"]=sig_P[index_vv.index]**0.5*index_vv["prior_COS"]
  index_vv["error_B"]=sig_B[index_vv.index]**0.5*index_vv["prior_COS"]
  #A priori and a posteriori fluxes 
  index_vv["flux_B"]=index_vv["prior_COS"]
  index_vv["flux_P"]= index_vv["post_COS"]
  #Monthly average after remobing the first year
  cycle=index_vv[index_vv.date.dt.year>=(mdl.begy+1)].groupby("month").mean()
  cycle["std_B"]=index_vv[index_vv.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_B
  cycle["std_P"]=index_vv[index_vv.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_P
  cycle.reset_index(inplace=True)
  ax[ir,ic].plot(cycle.month.values,cycle["flux_B"],color='k',label='a priori Soil')
  ax[ir,ic].plot(cycle.month.values,cycle["flux_P"],color='darkorange',label='a posteriori Soil')
  ax[ir,ic].fill_between(cycle.month.values,cycle["flux_B"]-cycle["error_B"].values, cycle["flux_B"]+cycle["error_B"].values, facecolor='grey', alpha=0.2)
  ax[ir,ic].fill_between(cycle.month.values,cycle["flux_P"]-cycle["error_P"].values, cycle["flux_P"]+cycle["error_P"].values, facecolor='darkorange', alpha=0.2)
  ax[ir,ic].xaxis.set_ticks(np.arange(1, 13, 2))
  ax[ir,ic].set_xlim(1,12)
#  ax[ir,ic].set_ylim(ymin[vv-2],ymax[vv-2])

  ax[ir,ic].set_title(table_ORC[int(vv)-1][1] ,fontsize=10)
  ax[ir,ic].tick_params(direction="in")

  if ic==0: ax[ir,ic].set_ylabel("GgS/y")
  if (ir==3)&(ic<=2): ax[ir,ic].tick_params(bottom="off")
  if (ir==2)&(ic==3): ax[ir,ic].tick_params(bottom="off")
  if ic!=3:
    ic+=1
  else:
    ic=0; ir+=1
 
 plt.subplots_adjust(wspace=0.45,hspace=0.8)
 ax[-1,2].set_axis_off()
 ax[-1,3].set_axis_off()
 ax[3,1].legend(bbox_to_anchor=[1.5, 1])
 plt.suptitle("Soil fluxes") 

 plt.savefig(rep_fig+"Cycle_Soil.pdf",format="pdf",bbox_inches='tight')


def cycle_Resp(index_ctl_vec,index_g,sig_B,sig_P,rep_da):
 """
 Arguments: 
  - index_ctl_vec: details about the control vector (and the prior vector)
  - index_g: same as above but with the unoptimized fluxes
  - sig_B : variance of the prior fluxes
  - sig_P : variance of posterior fluxes
 return: plots of the prior and posterior seasonal cycles and their associated incertitudes
 """          
 
 ymin=[2,0.5,0.1,0,0,0,0,0,0,0,0,0,0,0]
 ymax=[4,1.2,0.5,0.75,1,1.9,1.5,0.6,0.75,1.5,2.5,1,0.3,2]
 linestyles = ['-', '--', '-.', ':']

 dico={'CO2':[]} 
 dico['CO2']=[1,"GtC"]

 rep_fig=mdl.storedir+rep_da+'/FIG/'
 if not os.path.exists(rep_fig): os.system('mkdir '+rep_fig)

 f,ax=plt.subplots(nrows=4,ncols=4,figsize=(6,5))
 index_veget=index_ctl_vec[(index_ctl_vec.parameter=="Resp")&(index_ctl_vec.REG!="HS")]
 ir=0; ic=0 
 for vv in range(2,16):
  index_vv=index_veget[index_veget.PFT==vv].copy(deep=True)
  index_vv['date']=index_vv.apply(lambda row: to_date(row) ,axis=1)
  index_vv["error_P"]=sig_P[index_vv.index]**0.5*index_vv["prior_CO2"]
  index_vv["error_B"]=sig_B[index_vv.index]**0.5*index_vv["prior_CO2"]
  #A priori and a posteriori fluxes 
  index_vv["flux_B"]=index_vv["prior_CO2"]
  index_vv["flux_P"]= index_vv["post_CO2"]
  #Monthly average after remobing the first year
  cycle=index_vv[index_vv.date.dt.year>=(mdl.begy+1)].groupby("month").mean()
  cycle["std_B"]=index_vv[index_vv.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_B
  cycle["std_P"]=index_vv[index_vv.date.dt.year>=(mdl.begy+1)].groupby("month").std().flux_P
  cycle.reset_index(inplace=True)
  ax[ir,ic].plot(cycle.month.values,cycle["flux_B"],color='k',label='a priori Resp')
  ax[ir,ic].plot(cycle.month.values,cycle["flux_P"],color='darkorange',label='a posteriori Resp')
  ax[ir,ic].fill_between(cycle.month.values,cycle["flux_B"]-cycle["error_B"].values, cycle["flux_B"]+cycle["error_B"].values, facecolor='grey', alpha=0.2)
  ax[ir,ic].fill_between(cycle.month.values,cycle["flux_P"]-cycle["error_P"].values, cycle["flux_P"]+cycle["error_P"].values, facecolor='darkorange', alpha=0.2)
  ax[ir,ic].xaxis.set_ticks(np.arange(1, 13, 2))
  ax[ir,ic].set_xlim(1,12)
  ax[ir,ic].set_ylim(ymin[vv-2],ymax[vv-2])

  ax[ir,ic].set_title(table_ORC[int(vv)-1][1] ,fontsize=10)
  ax[ir,ic].tick_params(direction="in")

  if ic==0: ax[ir,ic].set_ylabel(dico["CO2"][1])
  if (ir==3)&(ic<=2): ax[ir,ic].tick_params(bottom="off")
  if (ir==2)&(ic==3): ax[ir,ic].tick_params(bottom="off")
  if ic!=3:
    ic+=1
  else:
    ic=0; ir+=1
 
 plt.subplots_adjust(wspace=0.45,hspace=0.8)
 ax[-1,2].set_axis_off()
 ax[-1,3].set_axis_off()
 ax[3,1].legend(bbox_to_anchor=[1.5, 1])
 plt.suptitle("b) Respiration") 

 plt.savefig(rep_fig+"Cycle_Resp.pdf",format="pdf",bbox_inches='tight')



def diag_flux(index_ctl_vec,index_g,sig_B,sig_P,rep_da):
 """
 Arguments: 
  - index_ctl_vec: details about the control vector (and the prior vector)
  - index_g: same as above but with the unoptimized fluxes
  - sig_B : variance of the prior fluxes
  - sig_P : variance of posterior fluxes
 return: plots of the prior and posterior fluxes and their associated incertitudes
 Biosphere first
 """

 add_SIF=1  #Add SIF for the GPP? 
 linestyles = ['-', '--', '-.', ':']
 rep_fig=mdl.storedir+rep_da+'/FIG/'
 if not os.path.exists(rep_fig): os.system('mkdir '+rep_fig)
 dico={'CO2':[],'COS':[]} 
 dico['CO2']=[1,"GtC"]
 dico['COS']=[10**6*getattr(Masse_molaire, 'COS')/getattr(Masse_molaire, 'CO2'),"GgS"]
 if add_SIF:
  SIF=extract_SIF(index_ctl_vec)
 for cc in compound:
  index_cc=index_ctl_vec[index_ctl_vec['factor_'+cc.name]!=0].copy(deep=True)
  processus=index_cc.parameter.unique()
  ir=0; ic=0
  ifig=1  #Nombre de pages
  ic_fig=1 # Nombre de suplots
  nb_fig=len(index_cc.groupby(['parameter','PFT','REG']).mean())
  nb_fig+=len(index_cc[index_cc.PFT.isnull()].groupby(['parameter','REG']).mean())-1
  f,ax=plt.subplots(nrows=5,ncols=2)
  f.set_size_inches((8.27,11.7))
  f.suptitle(cc.name+" fluxes")

  for pp in processus:
   if pp == 'offset': continue
   index_pp=index_cc[index_cc.parameter==pp].copy(deep=True)  
   veget=index_pp.PFT.unique()
   for vv in veget:
    index_vv=index_pp[index_pp.PFT==vv].copy(deep=True) if (not np.isnan(vv)) else index_pp.copy(deep=True)
    list_reg=index_vv.REG.unique()
    index_vv['date']=index_vv.apply(lambda row: to_date(row) ,axis=1)
    for ireg,namereg in enumerate(list_reg):
     index_rr=index_vv[index_vv.REG==namereg].copy(deep=True)
     error_P=sig_P[index_rr.index]**0.5*dico[cc.name][0]
     error_B=sig_B[index_rr.index]**0.5*dico[cc.name][0]
     #A priori and a posteriori errors 
     if (pp=="Gpp")&(cc.name=='COS'):
      error_B*=np.abs(index_rr['factor_'+cc.name].values)
      error_P*=np.abs(index_rr['factor_'+cc.name].values)

     #A priori and a posteriori fluxes 
     flux_B=index_rr["prior_"+cc.name].values*dico[cc.name][0]
     flux_P= index_rr["post_"+cc.name].values*dico[cc.name][0]

     #Draw plots
     ax[ir,ic].plot(index_rr.date.values,flux_B,color='k',label='prior')
     ax[ir,ic].plot(index_rr.date,flux_P,color='darkorange',label='post')
     ax[ir,ic].fill_between(index_rr.date.values, flux_B-error_B, flux_B+error_B, facecolor='grey', alpha=0.2)
     ax[ir,ic].fill_between(index_rr.date.values, flux_P-error_P, flux_P+error_P, facecolor='darkorange', alpha=0.2)
     #Grid
     if (pp == "Gpp")&(add_SIF):
      ax[ir,ic].grid(axis="x")
     else:
      ax[ir,ic].grid()
     #Add SIF diagnostics
     if (add_SIF)&(pp=="Gpp"):
      SIFtmp=SIF[(SIF.PFT==vv)&(SIF.REG==namereg)].copy(deep=True)
      if not  SIFtmp.empty: 
       ax1=ax[ir,ic]
       ax2 = ax[ir,ic].twinx()
       ax2.plot(SIFtmp.date, SIFtmp.SIF*(-1), color="darkgreen")
       ####Coefficient of correlation
       if cc.name=="CO2":
        print("Pearson coefficient apriori: ",pearsonr(SIFtmp.SIF.values,flux_B))
        print("Pearson coefficient a posteriori: ",pearsonr(SIFtmp.SIF.values,flux_P))
       #ax2.set_yticklabels(fontsize=9)
       ax2.tick_params(direction="in",labelsize=9)
       if ic ==1 : ax2.set_ylabel("SIF ($mW/nm/sr/m^{2}$)",color="darkgreen")
       for t in ax2.get_yticklabels():
        t.set_color("darkgreen")
     #End SIF diagnostics
     if pp=="Biosphere": ax[ir,ic].set_ylim(0,-25)
     if ic==0: ax[ir,ic].set_ylabel(dico[cc.name][1])

     #Subplot titles
     try:
      title=getattr(Vector_control, pp).name
     except:
      title=pp
     if (pp=="Gpp")&(cc.name=="COS"): title="plant uptake"
     title= title+", "+table_ORC[int(vv)-1][0] if not np.isnan(vv) else pp
     title=title if namereg=="GLOBE" else title+", "+namereg
    
     ax[ir,ic].set_title(title,fontsize=10)
     ax[ir,ic].tick_params(direction="in")
     if (ir!=4)&(ic_fig!=  nb_fig): 
     #ax[ir,ic].axes.get_xaxis().set_visible(False)
      ax[ir,ic].tick_params(labelbottom=False)
     elif (ir==4)|(ic_fig == nb_fig ):
      for tick in ax[ir,ic].get_xticklabels():
       tick.set_rotation(30)
     if (ir==4)&(ic==0): 
      ir=0; ic=1
     elif (ir==4)&(ic==1):
      ir=0; ic=0
      ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(1.1, -0.5), ncol=2) 
      if (pp=="Gpp")&(add_SIF):  plt.subplots_adjust(hspace=0.3)
      plt.savefig(rep_fig+"/PostF_"+cc.name+str(ifig)+".pdf",format='pdf')
      f,ax=plt.subplots(nrows=5,ncols=2)
      f.set_size_inches((8.27,11.7))
      f.suptitle(cc.name+" fluxes")
      ifig+=1
     else: 
      ir+=1
     ic_fig+=1

  #Eliminate the frames
  if (nb_fig<10)&(nb_fig%10!=0): 
     nb_frest=10-nb_fig%10
     if nb_frest/5 == 0:
      for ii in range(5):
       ax[ii,1].set_axis_off()
     elif nb_frest/5 == 1: 
      for ii in range(nb_frest%5,5):
       ax[ii,1].set_axis_off()
     plt.savefig(rep_fig+"/PostF_"+cc.name+str(ifig)+".pdf",format='pdf')
  elif (ifig==(nb_fig/10+1))&(nb_fig%10!=0):
     nb_frest=nb_fig%10
     if nb_frest/5 == 0:
      for ii in range(5):
       ax[ii,1].set_axis_off()
      for ii in range(nb_frest,5):
       ax[ii,0].set_axis_off()
     elif nb_frest/5 == 1: 
      for ii in range(nb_frest%5,5):
       ax[ii,1].set_axis_off()
     plt.savefig(rep_fig+"/PostF_"+cc.name+str(ifig)+".pdf",format='pdf')


 #####GLOBAL BUDGET#########
 index_ctl_vec['sig_B']=sig_B
 index_ctl_vec['sig_P']=sig_P
 for cc in compound:
  index_cc=index_ctl_vec[index_ctl_vec['factor_'+cc.name]!=0].copy(deep=True)
  processus=index_cc.parameter.unique()
  ir=0; ic=0
  ifig=1  #Nombre de pages
  ic_fig=1 # Nombre de suplots
  nb_fig=len(index_cc.groupby(['parameter']).sum())-1  #-1 offset
  f,ax=plt.subplots(nrows=6,ncols=1)
  f.set_size_inches((8.27,11.7))
  f.suptitle(cc.name+" fluxes")
  for pp in processus:
   if pp == 'offset': continue
   index_pp=index_cc[index_cc.parameter==pp].copy(deep=True)
   index_pp['date']=index_pp.apply(lambda row: to_date(row) ,axis=1)
   #A-priori and a-posteriori errors
   if (pp=='Gpp')&(cc.name=="COS"):  
     index_pp['sig_B']*=(index_pp['factor_'+cc.name])**2
     index_pp['sig_P']*=(index_pp['factor_'+cc.name])**2
   error_B=(index_pp.groupby(['date']).sum()['sig_B'].values)**0.5*dico[cc.name][0]
   error_P=(index_pp.groupby(['date']).sum()['sig_P'].values)**0.5*dico[cc.name][0]

   #A-priori and a-posteriori fluxes
   index_pp=index_pp.groupby(['date']).sum().reset_index()
   flux_B=index_pp["prior_"+cc.name]*dico[cc.name][0]
   flux_P=index_pp["post_"+cc.name]*dico[cc.name][0]
   #Drawing plots
   ax[ir].plot(index_pp.date.values,flux_B,color='k',label='prior')
   ax[ir].plot(index_pp.date.values,flux_P,color='darkorange',label='post')
   ax[ir].grid()
   ax[ir].fill_between(index_pp.date.values,flux_B-error_B, flux_B+error_B, facecolor='grey', alpha=0.5)
   ax[ir].fill_between(index_pp.date.values,flux_P-error_P, flux_P+error_P, facecolor='darkorange', alpha=0.5)
   ax[ir].set_ylabel(dico[cc.name][1])
   title= pp 
   ax[ir].set_title(title,fontsize=10)
   ax[ir].tick_params(axis='both', which='major',direction="in", labelsize=9)
   if (ir!=6)&(ic_fig!=  nb_fig): 
     #ax[ir,ic].axes.get_xaxis().set_visible(False)
     ax[ir].tick_params(labelbottom=False)
   elif (ir==6)|(ic_fig == nb_fig ):
     for tick in ax[ir].get_xticklabels():
       tick.set_rotation(30)
   if (ir==5):
     ir=0
     ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.3, -1.7), ncol=2) 
     plt.savefig(rep_fig+"/PostGlobal_"+str(ifig)+".pdf",format='pdf')
     f,ax=plt.subplots(nrows=6,ncols=1)
     f.set_size_inches((8.27,11.7))
     ifig+=1
   else: 
     ir+=1
   ic_fig+=1
   if (nb_fig/6==0)&(nb_fig%6!=0): 
    for ii in range(nb_fig,6):
      ax[ii].set_axis_off()
   elif (ifig==(nb_fig/6+1))&(nb_fig%6!=0):
    nb_frest=nb_fig%6
    if nb_frest/6 == 0:
     for ii in range(nb_frest%6,6):
       ax[ii].set_axis_off()


def diag_fit(index_ctl_vec,index_g,obs_vec,sim_0,sim_opt,rep_da):
  """
  Aim: Draw plots of the posterior, prior and observed atmospheric concentration
  Arguments: 
  obs_vec: observation vector
  sim_0 : prior concentration
  sim_opt: optimized simulation
  repa_da: directory of the experiment
  """ 
  rep_fig=mdl.storedir+rep_da+'/FIG/'
  if not os.path.exists(rep_fig): os.system('mkdir '+rep_fig)
  dico={'CO2':[],'COS':[]} 
  dico['CO2']=[1,"ppm"]
  dico['COS']=[10**6,"ppt"]
  ifig=0
  for cc in compound:
   ff=0
   convert_unit=dico[cc.name][0]
   units=dico[cc.name][1]
   nstat=0
   f,ax=plt.subplots(nrows=6,ncols=1)
   f.set_size_inches((8.27,11.7))
   ifig+=1
   for stat in  obs_vec.stat.unique(): 
    nstat+=1
    stat2 = stat[0:3] #For afternoon stations
    mask=(obs_vec['stat']==stat)&(obs_vec['compound']==cc.name)
    data=obs_vec[mask].copy(deep=True)
    if data.empty: continue
    data['obs']=data['obs']*convert_unit #Conversion des observation
    data['date']=0
    data['date']=data.apply(lambda row: datetime.datetime(row['year'],row['month'],(row['week']-1)*7+3,0,0,0),axis=1)     
    #Remove the first year
    data=data[data['date'].dt.year>=begy+1]
    obs_stat=np.copy(data.obs.values)
    error_o=obs_vec.sig_O.values**0.5*convert_unit
    offset_post=index_g.loc[(index_g['parameter']=='offset')&(index_g['factor_'+cc.name]!=0),'prior_'+cc.name].values[0]
    offset_prior=index_g[(index_g['parameter']=='offset')&(index_g['factor_'+cc.name]!=0)].prior.values[0]
    sim_0_stat=(sim_0[data.index])*convert_unit
    error_stat=(error_o[data.index])
    sim_opt_stat=(sim_opt[data.index])*convert_unit

    #obs
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
    #prior
    c=[data['frac'].values,sim_0_stat]
    os.system('rm -f '+file_out)
    with open(file_out, "w") as file:
      for x in zip(*c):
        file.write("{0}\t{1}\n".format(*x))
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    sim_0_stat=np.loadtxt(file_fit)
    sim_0_stat_seas=sim_0_stat[:,5]-sim_0_stat[:,6]
    #post
    c=[data['frac'].values,sim_opt_stat]
    os.system('rm -f '+file_out)
    with open(file_out, "w") as file:
      for x in zip(*c):
        file.write("{0}\t{1}\n".format(*x))
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    sim_opt_stat=np.loadtxt(file_fit)
    sim_opt_stat_seas=sim_opt_stat[:,5]-sim_opt_stat[:,6]

    ax[ff].plot(data.date,sim_opt_stat[:,3],label='post',color='darkorange') 
    ax[ff].plot(data.date,obs_stat,color='k',label='observation')
    ax[ff].fill_between(data.date.values, obs_stat-error_stat, obs_stat+error_stat, facecolor='grey', alpha=0.5)

    ax[ff].plot(data.date,sim_0_stat[:,3],color='darkgreen',label='prior') 
    ax[ff].set_ylabel(units,fontsize=13)
    ax[ff].set_title(stat2,fontsize=12) 
    ax[ff].xaxis.set_ticklabels([])
    ax[ff].grid()
    labels = ax[ff].get_xticklabels()
    ax[ff].set_xlim(datetime.datetime(begy+1,1,1),datetime.datetime(endy,12,31))


    ff+=1
    ax[ff].plot(data.date,sim_opt_stat_seas,label='post',color='darkorange')
    ax[ff].plot(data.date,obs_stat_seas,color='k',label='observation')
    ax[ff].fill_between(data.date.values, obs_stat_seas-error_stat, obs_stat_seas+error_stat, facecolor='grey', alpha=0.5)
    ax[ff].plot(data.date,sim_0_stat_seas,color='darkgreen',label='prior') 
    ax[ff].set_ylabel(units,fontsize=13)
    ax[ff].set_title(stat2+': seasonal cycle',fontsize=12) 
    ax[ff].grid()
    ax[ff].set_xlim(datetime.datetime(begy+1,1,1),datetime.datetime(endy,12,31))
    if (ff != 5)&(nstat != len(obs_vec.stat.unique())): 
     ax[ff].xaxis.set_ticklabels([])
     labels = ax[ff].get_xticklabels()
    elif (ff == 5)|(nstat == len(obs_vec.stat.unique())):
     labels = ax[ff].get_xticklabels()
     for i in labels:
       i.set_rotation(30)
    ff+=1  
    if ff==6: 
      plt.subplots_adjust(hspace=0.3,right=0.9)
      f.suptitle(cc.name)
      ff=0
      ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.3, -1.7), ncol=3) 
      plt.savefig(rep_fig+"/FitStat_"+str(ifig)+".pdf",format='pdf')
      f,ax=plt.subplots(nrows=6,ncols=1)
      f.set_size_inches((8.27,11.7))
      ifig+=1
      f.suptitle(cc.name)
   if ((2*nstat)%6 != 0):
    for ii in range((nstat*2)%6,6):
      ax[ii].set_axis_off()
   plt.savefig(rep_fig+"/FigStat_"+str(ifig)+".pdf",format='pdf')



def diag_gradient(index_ctl_vec,index_g,obs_vec,sim_0,sim_opt,rep_da):
  """
  Aim: Draw plots of the posterior, prior and observed atmospheric concentration
  Arguments: 
  obs_vec: observation vector
  sim_0 : prior concentration
  sim_opt: optimized simulation
  repa_da: directory of the experiment
  """
  rep_fig=mdl.storedir+rep_da+'/FIG/'
  if not os.path.exists(rep_fig): os.system('mkdir '+rep_fig)

  unit={'CO2':[],'COS':[]}
  unit['CO2']=[1,"ppm"]
  unit['COS']=[10**6,"ppt"]


  dico={'stat':[],'value':[],'std':[],'version':[],'lat':[],"compound":[]}
  for cc in compound:
   for stat in obs_vec.station.unique():
    stat2 = stat[0:3] 
    mask=(obs_vec['stat']==stat)&(obs_vec['compound']==cc.name).copy(deep=True)
    data=obs_vec[mask].copy(deep=True)
    if data.empty: continue
    convert_unit=unit[cc.name][0]
    data['obs']=data['obs']*convert_unit #Conversion des observation
    data['date']=0
    data['date']=data.apply(lambda row: datetime.datetime(row['year'],row['month'],(row['week']-1)*7+3,0,0,0),axis=1)
    #Remove the first year
    data=data[data['date'].dt.year>=begy+1]
    #Temporal series
    obs_stat    = np.copy(data.obs.values)
    sim_0_stat  = (sim_0[data.index])*convert_unit
    sim_opt_stat= (sim_opt[data.index])*convert_unit
    #CCGV OBS
    data['frac']=data.apply(lambda row: fraction_an(row),axis=1)
    time_serie=data[['frac','obs']]
    file_out=mdl.homedir+'modules/tmp-ccgvu.txt'
    os.system('rm -f '+file_out)
    np.savetxt(file_out,time_serie,fmt='%4.8f %3.15f')
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    time_serie=np.loadtxt(file_fit)
    time=[datetime.datetime(int(time_serie[ii,0]),int(time_serie[ii,1]),int(time_serie[ii,2])) for ii in range(len(time_serie[:,0]))]
    obs_stat_seas=(time_serie[:,5])
    dico["stat"].append(stat)
    dico["value"].append(np.mean(obs_stat_seas))
    dico["version"].append("obs")
    dico["std"].append(np.std(obs_stat_seas))
    dico["compound"].append(cc.name)
    dico["lat"].append(data.lat.mean())  
    #CCGV PRIOR
    data['frac']=data.apply(lambda row: fraction_an(row),axis=1)
    c=[data['frac'].values,sim_0_stat]
    os.system('rm -f '+file_out)
    with open(file_out, "w") as file:
      for x in zip(*c):
        file.write("{0}\t{1}\n".format(*x))
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    sim_0_stat=np.loadtxt(file_fit)
    time=[datetime.datetime(int(time_serie[ii,0]),int(time_serie[ii,1]),int(time_serie[ii,2])) for ii in range(len(time_serie[:,0]))]
    sim_0_seas=(time_serie[:,5])
    dico["stat"].append(stat)
    dico["value"].append(np.mean(sim_0_seas))
    dico["version"].append("prior")
    dico["std"].append(np.std(sim_0_seas))
    dico["compound"].append(cc.name)
    dico["lat"].append(data.lat.mean())
    #CCGV POST
    data['frac']=data.apply(lambda row: fraction_an(row),axis=1)
    c=[data['frac'].values,sim_opt_stat]
    os.system('rm -f '+file_out)
    with open(file_out, "w") as file:
      for x in zip(*c):
        file.write("{0}\t{1}\n".format(*x))
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    sim_opt_stat=np.loadtxt(file_fit)
    time=[datetime.datetime(int(time_serie[ii,0]),int(time_serie[ii,1]),int(time_serie[ii,2])) for ii in range(len(time_serie[:,0]))]
    sim_opt_seas=(time_serie[:,5])
    dico["stat"].append(stat)
    dico["value"].append(np.mean(sim_opt_seas))
    dico["version"].append("post")
    dico["std"].append(np.std(sim_opt_seas))
    dico["compound"].append(cc.name)
    dico["lat"].append(data.lat.mean())
  dico=pd.DataFrame(dico) 
 
  f,ax=plt.subplots(nrows=2,ncols=1)
  f.set_size_inches((8.27,11.7))
  mean_obs=dico[dico.version=="obs"].value.mean()
  version=["obs","prior","post"]
  couleur=["k","darkgreen","darkorange"]
  for icc in enumerate(compound):
   for iv2,vv2 in enumerate(version):
     mask=(dico.version==vv2)&(dico["compound"]==cc.name).copy(deep=True)
     offset=mean_obs-dico[mask].value.mean()
     dico.loc[mask,"value"]+=offset
     if vv2 == "obs":
      dico[mask].plot(x="lat",y="value",yerr="std",ax=ax[ic],capsize=4,capthick=1,color="k",label='_nolegend_',linewidth=1,markersize=3)
     dico[mask].plot(x="lat",y="value",                                color=couleur[iv2],style='D-',label=nversion[iv2],linewidth=1,markersize=3)
   ax[ic].tick_params(axis=u'x', which=u'both',length=0,labelbottom="off")
   ax[ic].set_xticks(np.arange(-90,100,30))

   ax[ic].tick_params(direction="in",axis=u'x', which=u'both',length=3,labelbottom="on",labeltop="on")
   ax[ic].set_xticklabels(["90S","60S","30S","0","30N","60N","90N"])
   ax[ic].set_xlabel("latitude")
   ax[ic].set_xlim(-92,92)
   ax[ic].set_ylabel(unit[cc.name][1])
   ax[ic].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4,fontsize=14)
   ax2 = ax[ic].twiny()

   ax2.tick_params(axis=u'x', which=u'top',length=3,labeltop="on")
   ax2.set_xticks(dico['lat'].unique())
   station2=['SPO', 'PSA', 'CGO', 'SMO', ' ', 'MLO', 'WIS', ' ', ' ',
       'HFM', ' ', 'MHD', 'BRW', ' ', 'ALT']
   ax2.set_xticklabels(station2,rotation="vertical",fontsize=14,color="k")
   ax2.set_xlim((-92,92))
   plt.savefig(rep_fig+"/Gradient.pdf",format='pdf')


def display_tabular(index_g,rmse,rep_da):
  """
  Draw plots with the tabular of the observation-sim misfits,
  annual budgets per compound
  repa_da: directory of the experiment
  """

  print(mdl.storedir+rep_da+'/FIG/')
  rep_fig=mdl.storedir+rep_da+'/FIG/'
  if not os.path.exists(rep_fig): os.system('mkdir '+rep_fig)
  dico={'CO2':[],'COS':[]} 
  dico['CO2']=[1,"[GtC/y]",1]
  dico['COS']=[10**6*getattr(Masse_molaire, 'COS')/getattr(Masse_molaire, 'CO2'),"[GgS/y]",10**6]

  #Calculate the number of plot in advance:
  nbf1=len(compound)
  nbf2=0
  for cc in compound: 
    index_cc=index_g[index_g['prior_'+cc.name]!=0]
    nbf2+=len(index_cc[~index_cc.PFT.isna()].parameter.unique())
  
  ifig=0
  f,axes=plt.subplots(nrows=2,ncols=2)
  f.set_size_inches((8.27,11.7))
  nr=0; nc=0; nf=0
  ifig+=1
  for cc in compound:
   table={'Source':[],'year':[],'Prior':[],'Post':[]}
   table=pd.DataFrame(table)
   index_cc=index_g.groupby(['parameter','year']).sum()
   index_cc.reset_index(inplace=True)
   index_cc=index_cc[index_cc['prior_'+cc.name]!=0]
   index_cc['name']=index_cc.apply(lambda row: to_name(row),axis=1)
   table['Source']=index_cc['name']
   table['year']=index_cc['year'].values
   table.loc[table.Source!="offset","Prior"]=index_cc['prior_'+cc.name]*dico[cc.name][0]
   table.loc[table.Source!="offset","Post"] =index_cc['post_'+cc.name]*dico[cc.name][0]
   table.loc[table.Source=="offset","Post"] =index_cc['post_'+cc.name]*dico[cc.name][2]
   table.loc[table.Source=="offset","Prior"]=index_cc['prior_'+cc.name]*dico[cc.name][2]
   table=table.round(1)
   table=table[['Source','year','Prior','Post']]
   render_mpl_table(table,axes[nr,nc], col_width=19, row_height=10.0, font_size=9,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,ax=None)
    
   axes[nr,nc].set_title(cc.name+" fluxes "+dico[cc.name][1])
   nf+=1
   nc+=1
   if nc ==2: nr=1
   nc=nc%2

  if nf == 4:
   plt.savefig(rep_fig+"/FitTab_"+str(ifig)+".pdf",format='pdf') 
   f,axes=plt.subplots(nrows=2,ncols=2)
   f.set_size_inches((8.27,11.7))
   ifig+=1
   nr=0; nc=0
  for cc in compound:
   index_cc=index_g[index_g['prior_'+cc.name]!=0]
   for pp in index_cc.parameter.unique():
    if index_cc[index_cc.parameter==pp].empty: continue
    npft=index_cc[index_cc.parameter==pp].PFT.unique()
    if (len(npft) ==1) : continue
    try: 
     name=getattr(Vector_control, pp).name
    except:
     name=pp
    index_pp=index_cc[index_cc.parameter==pp].groupby(['PFT','year']).sum()
    index_pp.reset_index(inplace=True)
    table_PFT=pd.DataFrame({'PFT':[],'year':[],'Prior':[],'Post':[]})
    table_PFT['Prior']=index_pp['prior_'+cc.name].values*dico[cc.name][0]
    table_PFT['Post']=index_pp['post_'+cc.name].values*dico[cc.name][0]
    table_PFT['PFT'] =index_pp['PFT'].values
    table_PFT['year'] =index_pp['year'].values
    table_PFT['year'].astype(int,inplace=True)
    table_PFT['PFT']=table_PFT.apply(lambda row: table_ORC[int(row['PFT'])-1][1],axis=1)
    table_PFT=table_PFT[table_PFT.year>=begy+1]
    table_PFT=table_PFT.groupby('PFT').mean().reset_index()
    del table_PFT['year']
    table_PFT=table_PFT.round(1)
    table_PFT=table_PFT[['PFT','Prior','Post']]
    render_mpl_table(table_PFT,axes[nr,nc], col_width=8, row_height=4, font_size=9,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,ax=None)
    nf+=1
    axes[nr,nc].set_title(cc.name+" "+name+" "+dico[cc.name][1])
    nc+=1
    if nc ==2: nr=1
    nc=nc%2
    if nf == 4: 
     plt.savefig(rep_fig+"/FitTab_"+str(ifig)+".pdf",format='pdf')
     f,axes=plt.subplots(nrows=2,ncols=2)
     ifig+=1
     f.set_size_inches((8.27,11.7))
     nr=0; nc=0

  #Misfit tabular
  for cc in compound:
   rmse_cc=rmse[rmse['compound']==cc.name]
#   rmse_cc=rmse_cc[['station','prior','post','prior_sea','post_sea']]
   rmse_cc=rmse_cc[['station','$RE$','$RE^{seas}$']]
   rmse_cc=rmse_cc.round(2)
   render_mpl_table(rmse_cc,axes[nr,nc], col_width=26, row_height=6, font_size=7,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0., 0, 1, 1], header_columns=0,ax=None)
   nf+=1
   axes[nr,nc].set_title(cc.name)
   nc+=1
   if nc ==2: nr=1
   nc=nc%2
   if nf == 4: 
     plt.savefig(rep_fig+"/FitTab_"+str(ifig)+".pdf",format='pdf')
     f,axes=plt.subplots(nrows=2,ncols=2)
     f.set_size_inches((8.27,11.7))
     ifig+=1
     nr=0; nc=0
  #Remove the frame 
  if nf%4 != 0:
   if nf%4<=2:
    for jj in range(2):
     axes[1,jj].set_axis_off()
    if nf%4%2 != 0: axes[0,1].set_axis_off()
   else:
    if nf%4%2 != 0: axes[1,1].set_axis_off()
  plt.savefig(rep_fig+"/FitTab_"+str(ifig)+".pdf",format='pdf')


def post_matrix(index_ctl_vec,diag_prior,diag_post,rep_da):
  """
  Draw plots with the tabular of the observation-sim misfits,
  annual budgets per compound
  repa_da: directory of the experiment
  """

  Reduc=diag_post/diag_prior
  index_ctl_vec["Reduc"]=Reduc
  index_offset=index_ctl_vec[index_ctl_vec.parameter!="offset"]
  index_ctl_vec2=index_ctl_vec[index_ctl_vec.year>mdl.begy]
  #index_ctl_vec2=index_ctl_vec2.groupby(["parameter","PFT","REG","year","month"]).mean()
  #index_ctl_vec2.reset_index(inplace=True)
  for cc in compound:
   tmp=index_ctl_vec2[index_ctl_vec2["factor_"+cc.name]!=0].copy(deep=True)
   tmp.reset_index(inplace=True)
   tmp["season"]="JFM"
   tmp.loc[(tmp.month==1)|(tmp.month==2)|(tmp.month==3),"season"]="JFM"
   tmp.loc[(tmp.month==4)|(tmp.month==5)|(tmp.month==6),"season"]="AMJ"
   tmp.loc[(tmp.month==7)|(tmp.month==8)|(tmp.month==9),"season"]="JAS"
   tmp.loc[(tmp.month==10)|(tmp.month==11)|(tmp.month==12),"season"]="OND"
   tmp["Reduc"]=100-tmp["Reduc"]*100

   n_param=len(tmp.parameter.unique())
   list_p=tmp.parameter.unique()
   for ip,pp in enumerate(list_p):
    tmp2=tmp[tmp.parameter==pp].copy(deep=True)
    if np.isnan(tmp2.PFT.iloc[0]): continue
    tmp2=tmp2[tmp2.REG!="HS"]
    tmp2=tmp2[tmp2.PFT!=1]
    tmp2=tmp2.groupby(["parameter","PFT","season"]).mean()
    tmp2.reset_index(inplace=True)
    tmp2.reset_index(drop=True,inplace=True)
    tmp2["PFT"]=tmp2.apply(lambda row: int(row["PFT"]),axis=1)
    tmp2=tmp2.groupby(["parameter","PFT","season"]).mean()
    tmp2.reset_index(inplace=True)
    hatches = ['-', '+', 'x']
    #axes[ip].plot(tmp2.month,tmp2.Reduc)
    if not np.isnan(tmp2.PFT.iloc[0]):
     n_pft=len(tmp2.PFT.unique())
     f,axe=plt.subplots(figsize=(5,2))
     axe.set_position([0.1,0.1,0.6,0.8])
     g=sns.barplot(data=tmp2,x="season", y="Reduc", hue="PFT",order=["JFM","AMJ","JAS","OND"], linewidth=2.5,palette="Paired",ax=axe)
     sns.despine( ax=axe, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    # plt.ylabel("Reduction of uncertainty (%)")
     plt.title(pp)
    # handles = g._legend_data.values()
   #  labels = g._legend_data.keys()
     axe.legend(bbox_to_anchor=(1., 1.), ncol=2,fontsize=7)
     axe.set( xlabel="",
       ylabel="Reduction of uncertainty (%)")
     
   tmp=tmp[tmp.PFT.isnull()]
   tmp=tmp.groupby(["parameter","REG","season"]).mean()
   tmp.reset_index(inplace=True)
   if tmp.empty: continue
   tmp["name"]="l"
   tmp['name']=tmp.apply(lambda row: (row.parameter +" "+row.REG),axis=1)
   f,axe=plt.subplots(figsize=(5,2))
   axe.set_position([0.1,0.1,0.5,0.8])
   g=sns.barplot(x="season", y="Reduc", hue="name",order=["JFM","AMJ","JAS","OND"], data=tmp,linewidth=2.5,palette="Paired",ax=axe)
   sns.despine( ax=axe, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
   axe.set( xlabel="",ylabel="Reduction of uncertainty (%)")
   axe.legend(bbox_to_anchor=(1., 1.), ncol=1,fontsize=7)

    # for i,thisbar in enumerate(bar.patches):
      # Set a different hatch for each bar
    #  thisbar.set_hatch(hatches[i%3])

   #  for iv,vv in enumerate(tmp2.PFT.unique()):
   #   a=tmp2[tmp2.PFT==vv].index[0]
   #   b=tmp2[tmp2.PFT==vv].index[-1]
   #   c1='#1f77b4' #blue
   #   c2='green' #green
   #   if not tmp2[(tmp2.PFT==vv)&(tmp2.Reduc>0.9)].empty:
   #   axes[ip].plot(tmp2[tmp2.PFT==vv].month,tmp2[tmp2.PFT==vv].Reduc,label=vv)
   # else:
   #   axes[ip].plot(tmp2.month,tmp2.Reduc)
   # axes[ip].xaxis.set_ticks(np.arange(0,13 , 1))
   # axes[ip].set_xlim(1,12)

     # n=n_pft
     # axes[ip].axvspan(a, b, color=colorFader(c1,c2,iv/n_pft), alpha=0.5, lw=0)
