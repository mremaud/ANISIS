"""

Author: Marine Remaud
Aim: Define and load the transport matrix G (linear tangent), that links the fluxes to the atmospheric concentrations

"""

#!/usr/bin/env python

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
sys.path.append(mdl.homedir+'/modules/')

from .useful import *
from .define_Hm import *
from .define_Hw import *
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

def define_H(obs_vec,index_ctl_vec,index_unopt,all_priors):
 """
 Calcul of the transport matrix g and the
   associated linear tangent (matrix_G) (normalized for 1GtC)
 obs_vec: observation vector
 index_ctl_vec: control vector containing all the information to fill the tangent linear
 index_nopt: fixed (unoptimized) flux 
 all_priors: prior fluxes in kg/m2/s
 """
 freq=np.copy(index_ctl_vec.week.unique())
 freq='W' if len(freq[np.isnan(freq)])==0 else 'M'
 if freq == "M":
   #Monthly frequency of the fluxes to be optimized
   define_Hm(obs_vec,index_ctl_vec,index_unopt,all_priors)
 elif freq== "W":
   #Weekly frequency of the fluxes to be optimized
   define_Hw(obs_vec,index_ctl_vec,index_unopt,all_priors)





def pruning(index_ctl_vec,obs_vec,index_unopt):
 """
 Supression of the null lines in the tangent linear matrix
 and the observations that can't be assimilated
 """
 area_LMDZ=get_area()
 matrix_G=np.load(storedir+'matrix_G_i.npz')
 matrix_G=matrix_G[list(matrix_G.keys())[0]]
 matrix_g=np.load(storedir+'matrix_g_i.npz')
 matrix_g=matrix_g[list(matrix_g.keys())[0]]
 index_g=index_ctl_vec.append(index_unopt,ignore_index=True)

 index_offset=index_g[index_g.parameter=='offset'].index
 #Get the columns of offset
 offset=matrix_g[:,index_offset]
 matrix_g[:,index_offset]=0

 line_sup=np.where(~matrix_g.any(axis=1))[0]
 offset=np.delete(offset,line_sup, axis=0)
 matrix_G=np.delete(matrix_G,line_sup, axis=0)
 matrix_g=np.delete(matrix_g,line_sup, axis=0)

 obs_vec.drop(index=line_sup, axis=0,inplace=True)
 obs_vec.reset_index(inplace=True,drop=True)

 #Replace offset
 matrix_g[:,index_offset]=offset
 if index_offset[-1]<=len(index_ctl_vec):
  matrix_G[:,index_offset]=offset
 #Suppression des colonnes
 col_sup=np.where(~matrix_G.any(axis=0))[0]
 matrix_G=np.delete(matrix_G,col_sup, axis=1)
 matrix_g=np.delete(matrix_g,col_sup, axis=1)
 index_ctl_vec.drop(index=col_sup, axis=0,inplace=True)
 index_ctl_vec.reset_index(inplace=True,drop=True)
 np.save(storedir+'matrix_G',matrix_G)
 np.save(storedir+'matrix_g',matrix_g)

 return line_sup,col_sup,index_ctl_vec,obs_vec


def calc_prior(obs_vec,index_ctl_vec,index_unopt,all_priors):
 """
 obs_vec: observation vector 
 index_ctl_vec: index of the control vector in the tangent linear matrix
 index_nopt:   index of the model which has not been linearized beforehand
 all_priors: prior fluxes that are taken into account in the CO2 ans COS budget (4 dimensiosn per default)
 return index_g: columns of the transport matrix (not linearized) + the prior budget
                 add columns 
 """
 area_LMDZ=get_area()

 index_g=index_ctl_vec.append(index_unopt,ignore_index=True)
 index_g.reset_index(inplace=True,drop=True)
 index_g['year']=index_g.apply(lambda row: int(row['year']) if not np.isnan(row['year']) else np.nan,axis=1)
 index_g['month']=index_g.apply(lambda row: int(row['month']) if not np.isnan(row['month']) else np.nan,axis=1)
 index_g['week']=index_g.apply(lambda row: int(row['week']) if not np.isnan(row['week']) else np.nan,axis=1)
 index_g['PFT']=index_g.apply(lambda row: int(row['PFT']) if not np.isnan(row['PFT']) else np.nan,axis=1)
 
 #Open all the region files at once in order to limit 
 # the number of files to be opened
 DIC_REG,AR_REG=get_dicregion(index_ctl_vec)

 time_prior=pd.date_range(start=datetime.datetime(begy,1,1), end=datetime.datetime(endy+1,1,1),closed='left',freq='M')
 for cc in compound:
  index_g['prior_'+cc.name]=pd.Series(np.zeros(len(index_g)))
  offset=all_priors['offset']
  ligne_col=index_g[(index_g.parameter=='offset')&(index_g['factor_'+cc.name]!=0)]
  pd.reset_option('mode.chained_assignment')
  with pd.option_context('mode.chained_assignment', None):
    index_g['prior_'+cc.name].loc[ligne_col.index]=offset[offset['compound']==cc.name].value.values[0]
 vector_time=pd.date_range(datetime.datetime(begy,1,1),datetime.datetime(endy+1,1,1),freq='M')
 jperm=vector_time.day.values  #Nombres de jours par mois
 for pp in index_g.parameter.unique():
  if pp == "offset":continue
  index_pp=index_g[index_g.parameter==pp].copy(deep=True)
  prior_orig=np.copy(all_priors[pp])
  if np.shape(prior_orig)[-1]== 97: prior_orig=prior_orig[:,:,:,:-1]
  prior_orig*=area_LMDZ[np.newaxis,np.newaxis,:,:]
  for vv,veget in enumerate(index_pp.PFT.unique()):
   index_ppvv=index_pp.copy(deep=True) if np.isnan(veget)  else index_pp[(index_pp.PFT==veget)].copy(deep=True)
   prior_fv=np.squeeze(np.copy(prior_orig[:,vv,:,:]))
   listreg=np.copy(index_ppvv[index_ppvv.parameter==pp].REG.unique())
   name_p=pp+"_"+str(int(veget)) if not np.isnan(veget) else pp
   if name_p in AR_REG:
    reg_array = AR_REG[name_p].copy(deep=True) 
    region    = DIC_REG[name_p].copy(deep=True)
   for rr in listreg:
    index_ppvvrr=index_ppvv[(index_ppvv.REG==rr)].copy(deep=True)
    if index_ppvvrr.empty: continue
    code=region[region['region']==rr].code.values
    rows,columns=np.where(reg_array['BASIN'].values == code) if rr != 'GLOBE' else np.where(reg_array['BASIN'].values != np.nan)
    prior_fvr=np.sum(prior_fv[:,rows,columns],axis=1)*86400*10**(-12) #On obtient des Gt/day
    prior_fvr=np.squeeze(prior_fvr)
    #Discrimination temporelle
    for yy in range(begy,endy+1):
     for cc in compound:
      prior_fvy=np.copy(prior_fvr)
      if not np.isnan(index_ppvvrr.month.iloc[0]):
       #Mensuel 
       for mm in range(1,13):
        index_time=np.where((vector_time.year==yy)&(vector_time.month==mm))[0]
        index_ppvvrryy=index_ppvvrr[(index_ppvvrr.month==mm)&(index_ppvvrr.year==yy)].copy(deep=True) 
        if np.isnan(index_ppvvrryy.week.iloc[0]): 
          prior_fmv=np.copy(prior_fvy[index_time])*jperm[index_time]
          with pd.option_context('mode.chained_assignment', None):
           index_g['prior_'+cc.name].loc[index_ppvvrryy.index]=np.copy(prior_fmv)
           index_g['prior_'+cc.name].loc[index_ppvvrryy.index]*=index_g['factor_'+cc.name].loc[index_ppvvrryy.index]
        else:
          for ww in range(1,5): 
           index_ppvvrryyww=index_ppvvrryy[index_ppvvrryy.week==ww].copy(deep=True)
           deltad=8 if ww!=4 else jperm[index_time]-25+1
           prior_fmv=np.copy(prior_fvy[index_time])*deltad
           with pd.option_context('mode.chained_assignment', None):
            index_g['prior_'+cc.name].loc[index_ppvvrryyww.index]=np.copy(prior_fmv)
            index_g['prior_'+cc.name].loc[index_ppvvrryyww.index]*=index_g['factor_'+cc.name].loc[index_ppvvrryyww.index]
      else:
       #Annuel
       index_time=np.where(vector_time.year==yy)[0]
       prior_fmv=np.copy(prior_fvy)
       prior_fmv=np.sum(np.multiply(prior_fmv[index_time],jperm[index_time]))
       index_ppvvrryy=index_ppvvrr[(index_ppvvrr.year==yy)].copy(deep=True)  
       pd.reset_option('mode.chained_assignment')
       with pd.option_context('mode.chained_assignment', None):
        index_g['prior_'+cc.name].loc[index_ppvvrryy.index]=float(prior_fmv)*index_g['factor_'+cc.name].loc[index_ppvvrryy.index]
 prior_global=pd.read_pickle(storedir+'prior_global.pkl') 
 
 #Elimination des regions qui sont zeros
 #Reequilibrage du budget
# for pp in index_ctl_vec.parameter.unique():
#  for yy in range(begy,endy+1):
#     mask=(index_g.parameter==pp)&(index_g.year==yy)
#     list_reg=index_g[mask].REG.unique()
#     if len(listreg) ==1 : continue
#     seuil=abs(np.copy(index_g[mask]['prior_CO2'].describe().loc['25%']))
#     if pp == "Gpp": seuil=seuil/10.
#     index_ctl_vec.drop(index_g[mask&(index_g['prior_CO2'].abs()<seuil)].index,inplace=True)
#     index_ctl_vec.reset_index(inplace=True,drop=True)
#     index_g.drop(index_g[mask&(index_g['prior_CO2'].abs()<seuil)].index,inplace=True)
#     index_g.reset_index(inplace=True,drop=True)
#     for cc in compound:
#      if index_g[mask]['prior_'+cc.name].sum() == 0: continue
#      exedent=prior_global[mask&(prior_global['compound']==cc.name)].value.values[0]-index_g[mask]['prior_'+cc.name].sum()
#      pd.reset_option('mode.chained_assignment')
#      with pd.option_context('mode.chained_assignment', None):
#       index_g.loc[mask,'prior_'+cc.name]+=exedent/len(index_g.loc[mask,'prior_'+cc.name])
#      print "BALANCE CO2 BUDGET"
#      print cc.name,pp,exedent, prior_global[mask&(prior_global['compound']==cc.name)].value.values[0] 
 #Fin reequilibrage
 index_ctl_vec.reset_index(inplace=True,drop=True)
 index_g.reset_index(inplace=True,drop=True)
 index_unopt=index_g[index_g.index>=len(index_ctl_vec)]
 index_ctl_vec=index_g[index_g.index<len(index_ctl_vec)]
 #Selectionner les vrais priors
 pd.reset_option('mode.chained_assignment')
 with pd.option_context('mode.chained_assignment', None):
  index_ctl_vec['prior']=0
  index_unopt['prior']=0
 for cc in compound:
  pd.reset_option('mode.chained_assignment')
  with pd.option_context('mode.chained_assignment', None):
   index_ctl_vec['prior']+=index_ctl_vec['prior_'+cc.name]*index_ctl_vec['factor_'+cc.name]
   index_unopt['prior']+=index_unopt['prior_'+cc.name]*index_ctl_vec['factor_'+cc.name]
   #Conversion unite
 #  index_ctl_vec["prior_"+cc.name]*=getattr(Masse_molaire,cc.name)/getattr(Masse_molaire,"CO2")
 #  index_g["prior_"+cc.name]*=getattr(Masse_molaire,cc.name)/getattr(Masse_molaire,"CO2")

 return index_ctl_vec,index_unopt


def calcul_sim0(obs_vec,index_ctl_vec,index_g,only_season):
 "Computation of the apriori trajectory"
 vec_prior=np.ones(len(index_g))
 sim_0=delta_sim(obs_vec,vec_prior,only_season)
 return sim_0




def delta_sim(obs_vec,vec_flux,only_season):
 """
 Case 1 : transport insured entirely by the adjoint
 Simplified transport: Return the simulated values by multiplying the matrix g (unlinearized) by the fluxes 
 obs_vec: observation vector that gives the index
 vec_flux: flux to be transported for all the observation components (CO2, COS,...)
 Return: simulated concentrations
 """
 #Matrix g which has not been linearized
 matrix_g=np.load(mdl.storedir+"matrix_g.npy") 
 for cc in compound:
  index_compound=np.copy(obs_vec[obs_vec['compound']==cc.name].index)
  matrix_g[index_compound,:]=np.multiply(matrix_g[index_compound,:],vec_flux[np.newaxis,:])
   
 conc_s=np.squeeze(np.sum(matrix_g,axis=1))
 tmp=[]
 if only_season:
  for cc in obs_vec["compound"].unique():
   for stat in obs_vec.stat.unique():
     time_serie=obs_vec[(obs_vec.stat==stat)&(obs_vec["compound"]==cc)].copy(deep=True)
     if time_serie.empty: continue
     time_serie["data"]=conc_s[obs_vec[(obs_vec.stat==stat)&(obs_vec["compound"]==cc)].index]
     time_serie=decomposition_ccgv(cc,time_serie,mdl.homedir)
     time_serie["fit"]=time_serie["fit"]-time_serie["trend"]
     tmp.extend(time_serie.fit.values)
  conc_s=np.copy(tmp) 
 np.save(mdl.storedir+'prior',matrix_g)
 return conc_s         




