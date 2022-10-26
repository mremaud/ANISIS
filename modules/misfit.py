from sklearn.metrics import mean_squared_error
import numpy as np
import os
import pandas as pd
import datetime
from .useful import *
import copy

name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__

def misfit(obs_vec,sim_0,sim_opt,chi):
  """
  Computation of the observation-model misfit for each station/or each year
  """

  rmse_stat={"station":[],"compound":[],"$RMSE_{prior}$":[],"$RMSE_{prior}^{seas}$":[],"$RMSE_{post}$":[],"$RMSE_{post}^{seas}$":[],'$\chi^2$':[]}
  for cc in compound:
    convert=10**6 if cc.name== 'COS' else 1
    index_c=obs_vec[obs_vec['compound']==cc.name].index
    for stat in obs_vec.stat.unique():
     obs_stat=obs_vec[(obs_vec['compound']==cc.name)&(obs_vec['stat']==stat)].copy(deep=True)
     index_stat=obs_stat.index
     if not obs_stat.empty: obs_stat['frac']=obs_stat.apply(lambda row: fraction_an(row),axis=1)
     #Seasonal fit 
     time_serie=obs_stat.copy(deep=True)
     time_serie['$RMSE_{post}$']=sim_opt[index_stat]
     time_serie['$RMSE_{prior}$']=sim_0[index_stat]
     rmse_stat['station'].append(stat)
     rmse_stat['compound'].append(cc.name)
     nombre_chi=chi[(chi.station==stat)&(chi["compound"]==cc.name)].chi.iloc[0] if not chi[(chi.station==stat)&(chi["compound"]==cc.name)].empty else 0
     rmse_stat['$\chi^2$'].append(nombre_chi)

     for vv in ['obs','$RMSE_{post}$','$RMSE_{prior}$']:
      if (obs_stat.empty):
       if (vv=="obs"): continue
       rmse_stat[vv].append(999)
       rmse_stat[vv[:-1]+'^{seas}$'].append(999)
       continue
      #RMSE total
      rmse=math.sqrt(mean_squared_error(time_serie.obs.values, time_serie[vv].values))
      rmse*=convert
      #Seasonal RMSE
      tmp_serie=time_serie[['frac',vv]].values
      file_out=mdl.homedir+'modules/tmp-ccgvu.txt'
      os.system('rm -f '+file_out)
      np.savetxt(file_out,tmp_serie)
      file_fit=mdl.homedir+'modules/tmp-fit.txt'
      #Model: without cal and ori
      os.system('cd '+mdl.homedir+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
      fit=np.loadtxt(file_fit)
      fit=fit[:,5]-fit[:,6]
      if vv == 'obs':
       obs_fit=np.copy(fit)
      else:
       rmse_sea=math.sqrt(mean_squared_error(obs_fit, fit))
       rmse_sea*=convert
       rmse_stat[vv].append(rmse)
       rmse_stat[vv[:-1]+'^{seas}$'].append(rmse_sea)
  rmse_stat=pd.DataFrame(rmse_stat)
  rmse_stat[rmse_stat==999]=np.nan
  
  rmse_stat=rmse_stat.round(2)
  rmse_stat["station"]=rmse_stat.apply(lambda row: row['station'][:3],axis=1)
  #Reduction d erreur
  rmse_stat["$RE$"]=rmse_stat.apply(lambda row: (row["$RMSE_{prior}$"]-row["$RMSE_{post}$"])/row["$RMSE_{prior}$"],axis=1)
  rmse_stat["$RE^{seas}$"]=rmse_stat.apply(lambda row: (row["$RMSE_{prior}^{seas}$"]-row["$RMSE_{post}^{seas}$"])/row["$RMSE_{prior}^{seas}$"],axis=1)

  return rmse_stat
