#!/usr/bin/env python
#author @Marine Remaud

from .useful import *
from .LRU_map import *
import numpy as np
import os
from netCDF4 import Dataset
import datetime
import calendar
import pandas as pd
import copy
import xarray as xr
from sys import argv
import sys

name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})

def posterior_map(index_ctl_vec,all_priors,rep_da,sig_B):
    """
    all_priors: dictionnary with all the prior maps
    rep_da: output directory
    Produce the necdf-files that contain the optimized fluxes in kg/m2/s
    at the LMDz resolution (96x95)
    Produce one netcdf file per process and per year at a
    Monthly resolution only!
    """
    DIC_REG,AR_REG=get_dicregion(index_ctl_vec)
    os.system("rm -f "+mdl.storedir+rep_da+"post*.nc")
    list_process=index_ctl_vec.parameter.unique()
    list_process=[x for x in list_process if not "offset" in x]
    for pp in list_process:
     ens_map=np.copy(all_priors[pp]) 
     index_pp=index_ctl_vec[index_ctl_vec.parameter==pp].copy(deep=True)
     list_veget=index_pp.PFT.unique()
     for cc in compound:
      name_compound=cc.name
      if (index_pp['prior_'+name_compound].mean() == 0): continue
      for yy in range(begy,endy+1):
       index_ppyy=index_pp[index_pp.year==yy].copy(deep=True)
       prior_map=np.copy(ens_map[(yy-begy)*12:(yy-begy)*12+12,:,:,:])
       post_map    =np.copy(prior_map)
       post_map_min=np.copy(prior_map)
       post_map_max=np.copy(prior_map)
       #####A priori  variable#######
       apr_map    =np.copy(prior_map)
       apr_map_min=np.copy(prior_map)
       apr_map_max=np.copy(prior_map)
       for vv in list_veget:
        nreg=pp if np.isnan(vv) else pp+"_"+str(int(vv))
        ar_reg = AR_REG[nreg] .copy(deep=True) 
        dic_reg= DIC_REG[nreg].copy(deep=True)
        compteur_veget=0 if np.isnan(vv) else int(vv-1)
        index_ppyyvv=index_ppyy[index_ppyy.PFT == vv] if not index_ppyy[index_ppyy.PFT == vv].empty else index_ppyy[index_ppyy.PFT.isnull()]
        list_region=index_ppyyvv.REG.unique()
        for rr in list_region:
          index_ppyyvvrr=index_ppyyvv[index_ppyyvv.REG == rr].copy(deep=True)
          code=dic_reg[dic_reg['region']==rr].code.values
          rows,columns=np.where(ar_reg['BASIN'].values == code) if rr != 'GLOBE' else np.where(ar_reg['BASIN'].values != np.nan)
          for mm in range(1,13):
               prior_estimate=1; post_estimate=1; fac_estimate=1
               if index_ppyyvvrr[index_ppyyvvrr.month.isnull()].empty: #mensual
                if not index_ppyyvvrr[(index_ppyyvvrr.month==mm)].empty: # Not optimized after the pruning
                 index_mm=index_ppyyvvrr[(index_ppyyvvrr.month==mm)].copy(deep=True)
                 prior_estimate=copy.copy(index_ppyyvvrr.loc[(index_ppyyvvrr.month==mm),'prior_'+name_compound].iloc[0])
                 post_estimate =copy.copy(index_ppyyvvrr.loc[(index_ppyyvvrr.month==mm),'post_'+name_compound].iloc[0])
                 #post_estimate_min =copy.copy(index_ppyyvvrr.loc[(index_ppyyvvrr.month==mm),'post_min_'+name_compound].iloc[0])
                 #post_estimate_max =copy.copy(index_ppyyvvrr.loc[(index_ppyyvvrr.month==mm),'post_max_'+name_compound].iloc[0])
                 if (pp== "Gpp") &(cc.name=="COS"): 
                  fac_estimate=copy.copy(index_ppyyvvrr.loc[(index_ppyyvvrr.month==mm),'factor_'+name_compound].iloc[0])
               else: #Annual
                index_mm=index_ppyyvvrr.copy(deep=True)
                prior_estimate=np.copy(index_ppyyvvrr['prior_'+name_compound].values)
                post_estimate=np.copy(index_ppyyvvrr['post_'+name_compound].values)
               # post_estimate_min=np.copy(index_ppyyvvrr['post_min_'+name_compound].values)
               # post_estimate_max=np.copy(index_ppyyvvrr['post_max_'+name_compound].values)
                if (pp== "Gpp") &(cc.name=="COS"): 
                 fac_estimate=np.copy(index_ppyyvvrr['factor_'+name_compound].values)
               scaling_factor=post_estimate/prior_estimate
               #scaling_factor_max=post_estimate_max/prior_estimate
               #scaling_factor_min=post_estimate_min/prior_estimate
               post_map[mm-1,compteur_veget,rows,columns]=np.copy(scaling_factor*prior_map[mm-1,compteur_veget,rows,columns])
               #post_map_max[mm-1,compteur_veget,rows,columns]=np.copy(scaling_factor_max*prior_map[mm-1,compteur_veget,rows,columns])
               #post_map_min[mm-1,compteur_veget,rows,columns]=np.copy(scaling_factor_min*prior_map[mm-1,compteur_veget,rows,columns])
               ####Apriori map
               apr_map[mm-1,compteur_veget,rows,columns]    =np.copy(prior_map[mm-1,compteur_veget,rows,columns])
               #apr_map_max[mm-1,compteur_veget,rows,columns]=np.copy(prior_map[mm-1,compteur_veget,rows,columns]*(1+sig_B[index_mm.index]**0.5))
               #apr_map_min[mm-1,compteur_veget,rows,columns]=np.copy(prior_map[mm-1,compteur_veget,rows,columns]*(1-sig_B[index_mm.index]**0.5))
               if (pp== "Gpp") &(cc.name=="COS"): 
                post_map[mm-1,compteur_veget,rows,columns]*=fac_estimate
                #post_map_max[mm-1,compteur_veget,rows,columns]*=fac_estimate
                #post_map_min[mm-1,compteur_veget,rows,columns]*=fac_estimate
                apr_map[mm-1,compteur_veget,rows,columns]*=fac_estimate
                #apr_map_max[mm-1,compteur_veget,rows,columns]*=fac_estimate
                #apr_map_min[mm-1,compteur_veget,rows,columns]*=fac_estimate

       post_map*=getattr(Masse_molaire,cc.name)/getattr(Masse_molaire, "CO2")         
       post_map=np.squeeze(post_map)
       #post_map_max*=getattr(Masse_molaire,cc.name)/getattr(Masse_molaire, "CO2")
       #post_map_max=np.squeeze(post_map_max)
       #post_map_min*=getattr(Masse_molaire,cc.name)/getattr(Masse_molaire, "CO2")
       #post_map_min=np.squeeze(post_map_min)
       apr_map*=getattr(Masse_molaire,cc.name)/getattr(Masse_molaire, "CO2")
       apr_map=np.squeeze(apr_map)
       #apr_map_max*=getattr(Masse_molaire,cc.name)/getattr(Masse_molaire, "CO2")
       #apr_map_max=np.squeeze(apr_map_max)
       #apr_map_min*=getattr(Masse_molaire,cc.name)/getattr(Masse_molaire, "CO2")
       #apr_map_min=np.squeeze(apr_map_min)

       dimtoflip=1 if len(np.shape(post_map))==3 else 2
       if len(np.shape(post_map)) == 3 : 
        post_map[:,:,-1]=post_map[:,:,0]
        #post_map_min[:,:,-1]=post_map_min[:,:,0]
       # post_map_max[:,:,-1]=post_map_max[:,:,0]
       # apr_map[:,:,-1]=apr_map[:,:,0]
       # apr_map_min[:,:,-1]=apr_map_min[:,:,0]
       # apr_map_max[:,:,-1]=apr_map_max[:,:,0]
       elif len(np.shape(post_map)) == 4:
        post_map[:,:,:,-1]    =post_map[:,:,:,0]
        #post_map_min[:,:,:,-1]=post_map_min[:,:,:,0]
        #post_map_max[:,:,:,-1]=post_map_max[:,:,:,0]
        apr_map[:,:,:,-1]    =apr_map[:,:,:,0]
        #apr_map_min[:,:,:,-1]=apr_map_min[:,:,:,0]
        #apr_map_max[:,:,:,-1]=apr_map_max[:,:,:,0]

       #Produce the netcdf file with xarray (same dimensions as the prior flux)
       if hasattr(eval("Sources."+name_compound), pp):
        file_name=eval("Sources."+name_compound+"."+pp).file_name 
        if eval("Sources."+name_compound+"."+pp).clim != 1: file_name=file_name.replace('XXXX',str(yy))
       else:
        pp_group=eval('Vector_control.'+pp+'.groupe')[0]
        file_name=eval("Sources."+name_compound+"."+pp_group).file_name
        if eval("Sources."+name_compound+"."+pp_group).clim != 1: file_name=file_name.replace('XXXX',str(yy))

       
       ds_in = xr.open_dataset(file_name,decode_times=False)
       if ds_in.lat[0]<ds_in.lat[1]: 
        ds_in=ds_in.reindex(lat=ds_in.lat[::-1])
       nom_variable='flx_'+name_compound.lower()
       if (pp== "Gpp") &(cc.name=="COS"): ds_in=ds_in.rename({"flx_co2":nom_variable})
       #ds_in[nom_variable+"_min"]=ds_in[nom_variable].copy(deep=True)
       #ds_in[nom_variable+"_max"]=ds_in[nom_variable].copy(deep=True)
       #ds_in[nom_variable+"_prior_min"]=ds_in[nom_variable].copy(deep=True)
       #ds_in[nom_variable+"_prior_max"]=ds_in[nom_variable].copy(deep=True)
       ds_in[nom_variable+"_prior"]=ds_in[nom_variable].copy(deep=True)
       ds_in[nom_variable].values=np.copy(post_map)
       #ds_in[nom_variable+"_min"].values=np.copy(post_map_min)
       #ds_in[nom_variable+"_max"].values=np.copy(post_map_max)
       #ds_in[nom_variable+"_prior_min"].values=np.copy(apr_map_min)
       #ds_in[nom_variable+"_prior_max"].values=np.copy(apr_map_max)
       ds_in[nom_variable+"_prior"].values=np.copy(apr_map)


       ds_in.to_netcdf(path=mdl.storedir+rep_da+"post_"+pp+cc.name+"_"+str(yy)+".nc", mode = 'w', format = 'NETCDF4_CLASSIC')  
       if (ds_in[nom_variable].ndim == 4): ds_in=ds_in.sum(dim='veget')
       latf=ds_in.lat.values; lonf=ds_in.lon.values; timef=[datetime.datetime(yy,mm+1,15) for mm in range(12)]       

       ds_out = xr.Dataset({ 'flx_'+name_compound.lower(): (['time','lat', 'lon'], np.squeeze(ds_in[nom_variable].values)  )},
                           coords={'lat': (['lat'], latf),
                                    'lon': (['lon'], lonf),
                                    'time': (['time'],timef)}).copy(deep=True)
       ds_out['lat']=latf; ds_out['lon']=lonf




       #NETCDF PRODUCTION: physical grid: with the good sign
       ds_out=lmdz2pyvar(ds_out,'flx_'+name_compound.lower())
       ds_out.to_netcdf(mdl.storedir+rep_da+"post_"+pp+cc.name+"_"+str(yy)+"_phy.nc", mode='w')
       if yy == mdl.endy:
         #Duplicate the las year
         os.system("cp "+mdl.storedir+rep_da+"post_"+pp+cc.name+"_"+str(yy)+"_phy.nc "+mdl.storedir+rep_da+"post_"+pp+cc.name+"_"+str(yy+1)+"_phy.nc")
def lmdz2pyvar(data_array,name_var):
    """
    Convert a flux vector from LMDz grid to Pyvar grid, i.e. vector 9026 = 94 * 96 + 2
    :param data_array: a DataArray (xarray)
    :return: flx : the DataArray (ntimes, 9026)
    """
    lon, lat = data_array.lon.values, data_array.lat.values
    flx = data_array[name_var].values
    time = data_array.time.values
    ntimes = data_array[name_var].shape[0]
    if len(lon) == 97:
       lon=lon[:-1]
       flx=flx[:,:,:-1]
    if lat[0] < lat[-1]:
        flx = np.flip(flx, 1)

    values_north = flx[:, 0, :].mean(1)  # longitude average
    values_south = flx[:, -1, :].mean(1)
    values_north = values_north[:, np.newaxis]
    values_south = values_south[:, np.newaxis]

    flx = flx[:, 1:-1, :]
    flx = np.reshape(flx, (ntimes, 9024))
    flx = np.append(values_north, flx, axis=1)
    flx = np.append(flx, values_south, axis=1)

    vector = np.linspace(1, 9026, 9026)
    ds_out = xr.Dataset({name_var: (['time', 'vector'], flx)},
                       coords={'vector': (['vector'], vector),
                                'time': (['time'], time)})
    return ds_out

