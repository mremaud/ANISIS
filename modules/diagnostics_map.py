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
os.environ['PROJ_LIB'] = r'/usr/local/install/python-3/pkgs/proj4-5.2.0-he6710b0_1/share/proj/' 
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})

def diag_map(index_ctl_vec,all_priors,rep_da):
    """
    all_priors: dictionnary with all the prior maps
    rep_da: output directory
    Produce the necdf-files that contain the optimized fluxes in kg/m2/s
    at the LMDz resolution (96x95)
    """

    n_graticules = 18
    parallels = np.arange(-90., 90, 30)
    meridians = np.arange(0., 360., 60)
    lw = 0.5
    dashes = [1,1] # 5 dots, 7 spaces... repeat
    graticules_color = 'grey'
    rep_fig=mdl.storedir+rep_da+'/FIG/'

    DIC_REG,AR_REG=get_dicregion(index_ctl_vec)
    list_process=index_ctl_vec.parameter.unique()
    list_process=[x for x in list_process if not "offset" in x]
    fig, axes = plt.subplots(len(list_process), 2)
    fig.set_size_inches((8.27,11.7))
    for ip,pp in enumerate(list_process):
     ens_map=np.copy(all_priors[pp]) 
     index_pp=index_ctl_vec[index_ctl_vec.parameter==pp].copy(deep=True)
     list_veget=index_pp.PFT.unique()
     for cc in compound:
      name_compound=cc.name
      if (index_pp['prior_'+name_compound].mean() == 0): continue
      for yy in range(endy,endy+1):
       index_ppyy=index_pp[index_pp.year==yy].copy(deep=True)
       prior_map=np.copy(ens_map[(yy-begy)*12:(yy-begy)*12+12,:,:,:])
       post_map=np.copy(prior_map)
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
          for mm in range(12):
               prior_estimate=1; post_estimate=1; fac_estimate=1
               if index_ppyyvvrr[index_ppyyvvrr.month.isnull()].empty: #mensual
                if not index_ppyyvvrr[(index_ppyyvvrr.month==mm+1)].empty: # Not optimized after the pruning
                 index_ppyyvvrr=index_ppyyvvrr.groupby(['year','month']).sum()
                 index_ppyyvvrr.reset_index(inplace=True)
                 prior_estimate=copy.copy(index_ppyyvvrr.loc[(index_ppyyvvrr.month==mm+1),'prior_'+name_compound].iloc[0])
                 post_estimate =copy.copy(index_ppyyvvrr.loc[(index_ppyyvvrr.month==mm+1),'post_'+name_compound].iloc[0])
                 if (pp== "Gpp") &(cc.name=="COS"): 
                  fac_estimate=copy.copy(index_ppyyvvrr.loc[(index_ppyyvvrr.month==mm+1),'factor_'+name_compound].iloc[0])
               else: #Annual
                prior_estimate=np.copy(index_ppyyvvrr['prior_'+name_compound].values)
                post_estimate=np.copy(index_ppyyvvrr['post_'+name_compound].values)
                if (pp== "Gpp") &(cc.name=="COS"): fac_estimate=np.copy(index_ppyyvvrr['factor_'+name_compound].values)
               
               scaling_factor=post_estimate/prior_estimate
               post_map[mm,compteur_veget,rows,columns]=np.copy(scaling_factor*prior_map[mm,compteur_veget,rows,columns])
               if (pp== "Gpp") &(cc.name=="COS")&(CO2 not in compound):
                post_map[mm,compteur_veget,rows,columns]*=(fac_estimate*getattr(Masse_molaire,"COS")/getattr(Masse_molaire, "CO2"))
                prior_map[mm,compteur_veget,rows,columns]*=(fac_estimate*getattr(Masse_molaire,"COS")/getattr(Masse_molaire, "CO2"))
               elif (pp== "Gpp") &(cc.name=="COS")&(CO2 in compound):
                continue
       if (pp== "Gpp")&(cc.name=="COS")&(CO2 in compound):continue

       post_map=np.squeeze(post_map)
       prior_map=np.squeeze(prior_map)
       dimtoflip=1 if len(np.shape(post_map))==3 else 2
       if len(np.shape(post_map)) == 3 : 
        post_map[:,:,-1]=post_map[:,:,0]
        prior_map[:,:,-1]=prior_map[:,:,0]
       elif len(np.shape(post_map)) == 4:
        post_map[:,:,:,-1]=post_map[:,:,:,0]
        prior_map[:,:,:,-1]=prior_map[:,:,:,0]

       if hasattr(eval("Sources."+name_compound), pp):
        file_name=eval("Sources."+name_compound+"."+pp).file_name 
        if eval("Sources."+name_compound+"."+pp).clim != 1: file_name=file_name.replace('XXXX',str(yy))
       else:
        pp_group=eval('Vector_control.'+pp+'.groupe')[0]
        file_name=eval("Sources."+name_compound+"."+pp_group).file_name
        if eval("Sources."+name_compound+"."+pp_group).clim != 1: file_name=file_name.replace('XXXX',str(yy))
 
      #NETCDF PRIOR: ds_in
       ds_in = xr.open_dataset(file_name,decode_times=False)
       if ds_in.lat[0]<ds_in.lat[1]: 
        ds_in=ds_in.reindex(lat=ds_in.lat[::-1])
       nom_variable='flx_'+name_compound.lower()
       if (pp== "Gpp") &(cc.name=="COS"): 
        ds_in=ds_in.rename({"flx_co2":nom_variable})
       ds_in[nom_variable].values=np.copy(prior_map)

       if (ds_in[nom_variable].ndim == 4): 
        ds_in=ds_in.sum(dim='veget')
        post_map=np.sum(post_map,axis=1)
       latf=ds_in.lat.values; lonf=ds_in.lon.values; timef=[datetime.datetime(yy,mm+1,15) for mm in range(12)]       

       ds_out = xr.Dataset({ nom_variable: (['time','lat', 'lon'], np.squeeze(post_map)  )},
                           coords={'lat': (['lat'], latf),
                                    'lon': (['lon'], lonf),
                                    'time': (['time'],timef)}).copy(deep=True)

       ####Fpost/(abs(Fpost-Fprior))
       WINTER=ds_out.copy(deep=True)
       WINTER[nom_variable].values=(ds_out[nom_variable].values-ds_in[nom_variable].values)/np.abs(ds_in[nom_variable].values)
       WINTER=WINTER[nom_variable].isel(time=[0,1,2]).mean("time").copy(deep=True).data

       SUMMER=ds_out.copy(deep=True)
       SUMMER[nom_variable].values=(ds_out[nom_variable].values-ds_in[nom_variable].values)/np.abs(ds_in[nom_variable].values)
       SUMMER=SUMMER[nom_variable].isel(time=[6,7,8]).mean("time").copy(deep=True).data

       axes[ip,0].set_title(pp+" Winter (JFM)",fontsize=8)
       #WINTER=  (post_win-prior_win)/prior_win
       WINTER[np.isnan(WINTER)]=0.
       m = Basemap(projection='cyl',llcrnrlat=-80,urcrnrlat=90,\
              llcrnrlon=-180,urcrnrlon=178,resolution='c',ax=axes[ip,0])
       m.drawcoastlines( linewidth=0.5)
       Y, X=np.meshgrid(lonf,latf)
       Y, X = m(Y,X)
       ax=m.contourf(Y,X,WINTER.data,np.arange(-0.45,0.55,0.1),cmap=plt.cm.PiYG_r,extend='both')
       ax.set_clim(-0.45,0.45)
       axes[ip,1].set_title(pp+" Summer (JAS)",fontsize=8)
       #prior_sum[prior_sum==0]=np.nan
       #SUMMER=  (post_sum-prior_sum)/prior_sum
       SUMMER[np.isnan(SUMMER)]=0.
       m = Basemap(projection='cyl',llcrnrlat=-80,urcrnrlat=90,\
              llcrnrlon=-180,urcrnrlon=178,resolution='c',ax=axes[ip,0])
       m.drawcoastlines( linewidth=0.5,zorder=90)
       Y, X=np.meshgrid(lonf,latf)
       Y, X = m(Y,X)
       ax=m.contourf(Y,X,SUMMER,np.arange(-0.45,0.55,0.1),cmap=plt.cm.PiYG_r,extend='both',ax=axes[ip,1])
       m.drawcoastlines( linewidth=0.5,zorder=90,ax=axes[ip,1])
       cb=fig.colorbar(ax,ax=axes[ip,1])
       cb.ax.tick_params(labelsize=5) 
       ax.set_clim(-0.45,0.45)
    fig.suptitle('a posteriori-a priori/(Flux a priori)')
    plt.savefig(rep_fig+"/Map_season.pdf",format='pdf')     


    


      

