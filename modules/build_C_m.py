#Marine Remaud
#Correlation error matrix
import sys
import numpy as np
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import copy
import itertools
from scipy.linalg import pinvh
import importlib
import json

name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})


def to_datetime(row):
    """
    Transform a row with the columns (year,month,day) to a timestamp date
    To be used in a dataframe to add a new columns (timestamp)
    """

    days_w=[1,9,17,25]
    if np.isnan(row['month']):
      date=datetime.datetime(int(row['year']),6,15)
    elif np.isnan(row['week']):
      date=datetime.datetime(int(row['year']),int(row['month']),15)
    else:
      date=datetime.datetime(int(row['year']),int(row['month']),int(row['week'])+4)
    return date


def build_C_m( index_ctl_vec,sig_B, sigma_t,inverse,rep_da,
                        **kwargs):
    """Build temporal norrelation matrix based on timedelta between periods.
    For period i and j, the corresponding correlation is:
    c(i,j) = exp(-timedelta(i, j) / sigma)*sigma(i)*sigma(j)
   
    sigma(i) (=corr_b['sig_B']**0.5[i]): standard deviation of the parameter i
    sigma(j) (=corr_b['sig_B']**0.5[j]): standard deviation of the parameter i
    timedelta= 1 / sigma_t
    Args:
        rep_da: experiment directory
        dates (np.array): dates sub-dividing the control vector periods
        sigma_t (float): decay distance for correlation between periods
                         (in days)
        correlation of B 
        inverse = True: return the inverse of c_nn    
    """
    add_corr=1.
    groups={"tropics":[2,3,11,14],"temperate":[4,5,6,10],"boreal":[7,8,9,15]}
    corr_groups={"tropics":0.6,"temperate":0.5,"boreal":0.6}
    if add_corr: 
      with open(mdl.storedir+rep_da+'output.txt', 'a') as outfile:
       outfile.write("Correlation between PFTs : \n")
       for gg in groups:
         outfile.write("      "+gg+" : correlation coefficient of "+str(corr_groups[gg])+" between the PFTs "+ json.dumps(groups[gg])+"\n")

    evalmin=0. #Flag out the egenvalues
    prec=1. #10**6 #Factor to  increase the precision
    sig_B=sig_B**0.5*prec
    sig_B=np.double(sig_B)
    #Standard deviation
    index_ctl_vec['date']=index_ctl_vec.apply(lambda row: to_datetime(row) ,axis=1) 
    dates=index_ctl_vec['date'].values
    # Else build correlations
    corr=np.zeros((len(index_ctl_vec),len(index_ctl_vec)))
    for pp in index_ctl_vec.parameter.unique():
         for rr in index_ctl_vec[index_ctl_vec.parameter==pp].REG.unique():
          for kk in index_ctl_vec[(index_ctl_vec.parameter==pp)&(index_ctl_vec.REG==rr)].PFT.unique():
           #On correlation length per PFT and region
           if np.isnan(kk):
            sigma_t2=copy.copy(sigma_t[pp])
           elif (pp+'-PFT'+str(int(kk))) not in sigma_t: 
            sigma_t2=copy.copy(sigma_t[pp])
           else:
            sigma_t2=copy.copy(sigma_t[pp+'-PFT'+str(int(kk))])
           mask_index=(index_ctl_vec.parameter==pp)&(index_ctl_vec.REG==rr)&(index_ctl_vec.PFT==kk)
           if index_ctl_vec[mask_index].empty: mask_index=(index_ctl_vec.parameter==pp)&(index_ctl_vec.REG==rr)
           index_pp=index_ctl_vec[mask_index]
           dates=index_pp.date.values
           # Compute matrix of distance (put all in days)
           dt = (pd.DatetimeIndex(dates)[:, np.newaxis]
              - pd.DatetimeIndex(dates)[np.newaxis, :]) \
            / np.timedelta64(sigma_t2, 'D')
           ip=np.copy(np.asarray(index_pp.index).tolist())
           # Compute the correlation matrix itself
           if pp == 'offset': dt[~np.eye(len(pd.DatetimeIndex(dates)),dtype=bool)]=8000000000
           corr[np.ix_(ip,ip)] = (np.exp(-dt ** 2))
    corr_t=np.copy(corr)
    corr=np.multiply(corr,np.outer(sig_B,sig_B))
    ##############################################################################################
    #Add autocorrelation between PFT
    if add_corr:
     for pp in index_ctl_vec.parameter.unique():
      index_pp=index_ctl_vec[index_ctl_vec.parameter==pp].copy(deep=True)
      if np.isnan(index_pp.PFT.iloc[0]): continue
      for rr in index_pp.REG.unique():
       index_pprr=index_pp[index_pp.REG==rr].copy(deep=True)
       for gg in groups:
        for pair in itertools.product(groups[gg], repeat=2):
         if pair[0]==pair[1]:continue
         cols=index_pprr[index_pprr.PFT==pair[0]].copy(deep=True)
         ligns=index_pprr[index_pprr.PFT==pair[1]].copy(deep=True)
         for cc in cols.index: 
          for ll in ligns.index:
           if (cols.loc[cc].date==ligns.loc[ll].date):
            corr[cc,ll]=np.copy(sig_B[cc]*sig_B[ll])*corr_groups[gg]
           else:
            #Find the index:
            auto1=np.copy(corr_t[cols[ cols.date ==ligns.loc[ll].date].index,cc]**0.5)
            auto2=np.copy(corr_t[ligns[ligns.date==cols.loc[cc].date].index,ll]**0.5)
            corr[cc,ll]=np.copy(auto1*auto2)*corr_groups[gg]*np.copy(sig_B[cc]*sig_B[ll])
    np.save(mdl.storedir+rep_da+'/corr.npy',corr)
##############################################################################################      
    #corr=corr*np.dot(sig_B.reshape(-1,1),sig_B)
    # Component analysis
#    np.linalg.cholesky(corr)

    evalues, evectors = np.linalg.eigh(corr)
#    print "Valeurs propres:",evalues
    # Re-ordering values
    # (not necessary in principle in recent numpy versions)
    index = np.argsort(evalues)[::-1]
    evalues = evalues[index]
    evectors = evectors[:, index]
    mask = evalues >= evalmin
    evalues=evalues[mask]
    evectors=evectors[:, mask]


    if inverse: 
     print ("inverse")
#     corr2=np.linalg.inv(corr)
 #    corr2=pinvh(corr)
     corr2=np.dot(evectors*1./evalues,evectors.T)
    else:
     print("correlation")
     corr2=np.dot(evectors*evalues,evectors.T)
    # corr=corr2
   # corr=np.dot(sig_B.reshape(-1,1)*evectors*evalues,sig_B*evectors.T)
    #corr=corr*np.outer(sig_B,sig_B)    
    #corr2: inverse of corr
 #   corr2=np.dot(evectors*1/sig_B.reshape(-1,1)*1./evalues,evectors.T*1/sig_B)
    return corr2


