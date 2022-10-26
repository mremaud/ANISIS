#!/usr/bin/env python
"""

Author: Marine Remaud
This is the main programm of the analytic inverse system that optimizes the surface fluxes over the globe by minimizing the observation-model mistmatch. The observations are the observed concentrations at the surface. The atmospheric transport model that links the surface fluxes to the atmospheric concentrations at the surface is LMDz. The 

"""

import importlib
import os 
import sys
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
from netCDF4 import Dataset
import datetime
import calendar
import pandas as pd
import copy
import xarray as xr
import math
from sys import argv
import scipy.optimize
import pickle
from sklearn.metrics import mean_squared_error
from modules import *

#Import the variables in the config.py
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})
sys.path.append(mdl.homedir+'modules/')
################################################################################

if not os.path.exists(storedir):os.makedirs(storedir)
if os.path.exists(homedir+"/INPUTF/vec_lmdz.pkl") : os.system("cp "+homedir+"/INPUTF/vec_lmdz.pkl "+storedir)
os.system("cp "+homedir+"/INPUTF/*nc "+storedir)
os.system("cp "+homedir+"/INPUTF/*csv "+storedir)


if mdl.version == 'tangent_linear':

 prior_budget()
 #Interpreting the config.card 
 obs_vec=class_topd_obs() if not only_season else class_topd_obs_seasonal() 
 obs_vec.to_pickle(mdl.storedir+'obs_vec.pkl')
 try:
   #Download a matrix from a previous experiment: same prior fluxes but different stations
   mdl.from_exp
   dir_exp=mdl.storedir+'../'+mdl.from_exp+'/'
   with open(dir_exp+'all_priors', 'rb') as handle:
    all_priors = pickle.load(handle,encoding="latin1")
   index_ctl_vec=pd.read_pickle(dir_exp+'index_ctl_vec.pkl')
   index_unopt=pd.read_pickle(dir_exp+'index_unopt.pkl')
   print('REDIMENSIONING THE MATRIX FROM THE REPERTORY:'),mdl.from_exp 
   obs_vec=redim_H(obs_vec,index_ctl_vec,index_unopt)
 except:
   index_ctl_vec,index_unopt=class_topd_ctl()
   print("LOADING ALL THE PRIOR FLUXES AVAILABLE IN CONFIG.PY")
   all_priors=load_prior(index_unopt,only_season) #Stockage of all priors in a dictionnary: in kg/m2/s
   print('NORMALIZATION OF THE PRIOR')
   index_ctl_vec,index_unopt=calc_prior(obs_vec,index_ctl_vec,index_unopt,all_priors)
   print('TANGENT LINEAR COMPUTATION')
   #Additional 
   index_ctl_vec.to_pickle(mdl.storedir+'index_ctl_vec.pkl')
   index_unopt.to_pickle(mdl.storedir+'index_unopt.pkl')

   define_H(obs_vec,index_ctl_vec,index_unopt,all_priors)
   #index_ctl_vec.to_pickle(mdl.storedir+'index_ctl_vec1.pkl')

   print('PRUNING')
   line_sup,col_sup,index_ctl_vec,obs_vec=pruning(index_ctl_vec,obs_vec,index_unopt)
   #index_ctl_vec.to_pickle(mdl.storedir+'index_ctl_vec4.pkl')

 index_g=index_ctl_vec.append(index_unopt,ignore_index=True)
 index_g.reset_index(inplace=True)
 #Enregistrement des variables
 obs_vec.to_pickle(mdl.storedir+'obs_vec.pkl')
 index_g.to_pickle(mdl.storedir+'index_g.pkl')
 index_unopt.to_pickle(mdl.storedir+'index_unopt.pkl')
 index_ctl_vec.to_pickle(mdl.storedir+'index_ctl_vec.pkl')
 #np.save(mdl.storedir+'line_sup.npy',line_sup)
 with open(mdl.storedir+'all_priors', 'wb') as handle:
    pickle.dump(all_priors, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
 with open(mdl.storedir+'all_priors', 'rb') as handle:
    all_priors = pickle.load(handle,encoding="latin1")
 obs_vec=pd.read_pickle(mdl.storedir+'obs_vec.pkl')
 index_ctl_vec=pd.read_pickle(mdl.storedir+'index_ctl_vec.pkl')
 index_g=pd.read_pickle(mdl.storedir+'index_g.pkl')
 index_unopt=pd.read_pickle(mdl.storedir+'index_unopt.pkl')

#Creation of a subdirectory in which the inversion results are saved: to be modified depending on the configuration
rep_da=""; flag_start=1
for attr,items in Vector_control.__dict__.items():
 if attr.startswith('__'): continue
 if not flag_start: rep_da+='_'
 rep_da+=attr+str(items.coef[0])
 flag_start=1
rep_da+="/"
print("Files saved in:", storedir+rep_da)
if not os.path.exists(mdl.storedir+rep_da): os.system('mkdir '+mdl.storedir+rep_da)

print('COMPUTATION OF THE SIMULATED VALUES')
sim_0=calcul_sim0(obs_vec,index_ctl_vec,index_g,mdl.only_season)
np.save(mdl.storedir+'sim_0.npy',sim_0)
output_transport=pd.DataFrame()
output_transport["mod"]=sim_0
output_transport["meas"]=obs_vec.obs
output_transport["station"]=obs_vec.stat
output_transport["year"]=obs_vec.year
output_transport["month"]=obs_vec.month
output_transport["day"]=obs_vec.week
output_transport.to_csv(mdl.storedir+rep_da+"/output_sim.csv")
print("COMPUTATION OF THE COVARIANCE ERROR MATRIX")
sig_B=define_B(index_ctl_vec)
sigma_t=get_sigmat()
sig_o=define_O(obs_vec)
obs_vec['sig_O']=sig_o

write_output(rep_da)
###################INVERSION########################################################
x_opt,Bpost,chi=AnalyticInv(rep_da,obs_vec,index_ctl_vec,index_unopt,sig_B,sigma_t,only_season)
sig_P=np.diagonal(Bpost)
######################################################################################
#Optimized simulated concentration
x_opt=np.append(x_opt,np.ones(len(index_unopt)))
sim_opt=delta_sim(obs_vec,x_opt,only_season)
np.save(mdl.storedir+rep_da+'sim_opt.npy',sim_opt)
np.save(mdl.storedir+rep_da+'sig_P.npy',sig_P)
np.save(mdl.storedir+rep_da+'sig_B.npy',sig_B)
################DIAGNOSTICS#####################################
Errors=Error_tot(index_ctl_vec,rep_da)
print(Errors)
rmse=misfit(obs_vec,sim_0,sim_opt,chi)
rmse.to_pickle(mdl.storedir+rep_da+"/rmse.pkl")
print('TOTAL MISFIT AFTER OPTIMIZATION')
#print(rmse)
index_g,index_ctl_vec=post_budget(index_g,index_ctl_vec,x_opt,sig_B,sig_P)
index_g.to_pickle(mdl.storedir+rep_da+"index_g.pkl")
print("POSTERIOR FLUXES")
#diag_flux(index_ctl_vec,index_g,sig_B,sig_P,rep_da)
#cycle_flux(index_ctl_vec,index_g,sig_B,sig_P,rep_da)
#cycle_GPP(index_ctl_vec,index_g,sig_B,sig_P,rep_da)
print("FIT TO NOAA STATIONS")
#diag_fit(index_ctl_vec,index_g,obs_vec,sim_0,sim_opt,rep_da)
#display_tabular(index_g,rmse,rep_da)
print("POSTERIOR MATRIX")
#diag_map(index_ctl_vec,all_priors,rep_da)

#cycle_GPP2(index_ctl_vec,index_g,sig_B,sig_P,rep_da)
#cycle_Resp(index_ctl_vec,index_g,sig_B,sig_P,rep_da)
#cycle_Soil(index_ctl_vec,index_g,sig_B,sig_P,rep_da)
#budget(index_g,index_unopt,Errors,rep_da)
#ts_station(obs_vec,index_g,sim_0,sim_opt,rmse,sig_o,rep_da)

#VISUALISATION
os.system("enscript -E -q -Z -p - -f Courier10 "+name_config+".py| ps2pdf - "+storedir+rep_da+"/FIG/aconfig.pdf")
os.system("enscript -E -q -Z -p - -f Courier10 "+storedir+rep_da+"/output.txt| ps2pdf - "+storedir+rep_da+"/FIG/acost.pdf")
os.system("rm -f "+storedir+rep_da+"/FIG/AllFigs.pdf")
os.system("pdfunite "+storedir+rep_da+"/FIG/*pdf "+storedir+rep_da+"/FIG/AllFigs.pdf")
#post_matrix(index_ctl_vec,sig_B,sig_P,rep_da)
index_ctl_vec.to_pickle(mdl.storedir+rep_da+"index_ctl_vec.pkl")
print(mdl.storedir+rep_da+"index_g.pkl")
################################################################
index_g.to_pickle(mdl.storedir+rep_da+"index_g.pkl")
print("PRODUCTION OF NEW SOURCES NETCDF FILES")
#posterior_map(index_ctl_vec,all_priors,rep_da,sig_B)
#launch_lmdz(index_g,index_ctl_vec,index_unopt,"COS",rep_da)
#launch_lmdz(index_g,index_ctl_vec,index_unopt,"CO2",rep_da)


