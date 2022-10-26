#!/usr/bin/env python
#author @Marine Remaud

#Define and load H
from .useful import *
from .LRU_map import *

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
#Import the variables in the config.py
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})

def post_budget(index_g,index_ctl_vec,xpost,sig_B,sig_P):
 """
 Calculation of the total budget for each compound after optimization
 return: pandas dataframe index_g and index_ctl_vec with an additional 
 column that feature the optimized fluxes. 
 Stop the program if the units are not consistent 
 """
 n_param=len(index_ctl_vec)

 
 for cc in compound:
  index_g["post_"+cc.name]=index_g["prior_"+cc.name].copy(deep=True)  #Because unopt dont change
  ###MIN AND MAX FROM THE ERROR BAR
  index_g["post_max_"+cc.name]=index_g["prior_"+cc.name].copy(deep=True)  #Because unopt dont change
  index_g["post_min_"+cc.name]=index_g["prior_"+cc.name].copy(deep=True)  #Because unopt dont change
  index_g["prior_max_"+cc.name]=index_g["prior_"+cc.name].copy(deep=True)  #Because unopt dont change
  index_g["prior_min_"+cc.name]=index_g["prior_"+cc.name].copy(deep=True)  #Because unopt dont change

  pd.reset_option('mode.chained_assignment')
  with pd.option_context('mode.chained_assignment', None):
    index_g['post_'+cc.name].iloc[:len(index_ctl_vec)]    =xpost[:len(index_ctl_vec)]*index_ctl_vec['factor_'+cc.name]*index_ctl_vec['prior']
    index_g['post_min_'+cc.name].iloc[:len(index_ctl_vec)]=(xpost[:len(index_ctl_vec)]-sig_P[:]**0.5)*index_ctl_vec['factor_'+cc.name]*index_ctl_vec['prior']
    index_g['post_max_'+cc.name].iloc[:len(index_ctl_vec)]=(xpost[:len(index_ctl_vec)]+sig_P[:]**0.5)*index_ctl_vec['factor_'+cc.name]*index_ctl_vec['prior']
    index_g['prior_min_'+cc.name].iloc[:len(index_ctl_vec)]=(xpost[:len(index_ctl_vec)]-sig_B[:]**0.5)*index_ctl_vec['factor_'+cc.name]*index_ctl_vec['prior']
    index_g['prior_max_'+cc.name].iloc[:len(index_ctl_vec)]=(xpost[:len(index_ctl_vec)]+sig_B[:]**0.5)*index_ctl_vec['factor_'+cc.name]*index_ctl_vec['prior']

  index_ctl_vec['post_'+cc.name]=index_g['post_'+cc.name].iloc[:len(index_ctl_vec)]
  index_ctl_vec['post_min_'+cc.name]=index_g['post_min_'+cc.name].iloc[:len(index_ctl_vec)]
  index_ctl_vec['post_max_'+cc.name]=index_g['post_max_'+cc.name].iloc[:len(index_ctl_vec)]
  index_ctl_vec['prior_min_'+cc.name]=index_g['prior_min_'+cc.name].iloc[:len(index_ctl_vec)]
  index_ctl_vec['prior_max_'+cc.name]=index_g['prior_max_'+cc.name].iloc[:len(index_ctl_vec)]

 index_g['date']=index_g.apply(lambda row: datetime.datetime(int(row['year']),int(row['month']),15) if not np.isnan(row['month']) else datetime.datetime(int(row['year']),6,15) ,axis=1)
 #Only for display 
 index_g2=index_g.copy(deep=True)
 for cc in compound:
  del index_g2['factor_'+cc.name]
  index_g2.loc[(index_g2.parameter!='offset'),'post_'+cc.name]*=getattr(Masse_molaire,cc.name)/getattr(Masse_molaire, "CO2")
  index_g2.loc[(index_g2.parameter!='offset'),'prior_'+cc.name]*=getattr(Masse_molaire,cc.name)/getattr(Masse_molaire, "CO2")
 del index_g2['index'],index_g2['week'],index_g2['month']
 print(index_g2.groupby(['parameter','year']).sum())
 return index_g,index_ctl_vec

 

def prior_budget():
 """
 Calculation of the prior total budget
 Stop the program if the units are not consistent 
 """
 #region,reg_array=get_region()
 area_LMDZ=get_area()
 prior_global={'parameter':[],'compound':[],'value':[],'year':[]}
 print("PRIOR BUDGET (BEFORE OPTIMIZATION) FOR THE PERIOD",str(begy)+'-'+str(endy))
 for compound_attr,compound_items in Sources.__dict__.items():
  if compound_attr.startswith('__'): continue
  #Every compound
  for attr,items in compound_items.__dict__.items():
   if attr.startswith('__'): continue
   #Every sources
   time_range=pd.date_range(datetime.datetime(begy,1,1),datetime.datetime(endy+1,1,1),freq='M')
   var_name='flx_co2' if (attr == 'Gpp')&(compound_attr =='COS') else 'flx_'+compound_attr.lower()
   if items.clim == 1:
     tmp = xr.open_dataset(items.file_name,decode_times=False)
     prior_flux=np.tile(tmp[var_name].values,(endy-begy+1,1,1)) 
     prior_flux = xr.Dataset({var_name: (['time','lat', 'lon'], prior_flux)}, 
                            coords={'lat': (['lat'], tmp.lat),
                                    'lon': (['lon'], tmp.lon),
                                    'time': (['time'],time_range)})
     tmp.close()
   else:
     flag_start=1
     for yy in range(begy,endy+1):
      file_name=items.file_name.replace('XXXX',str(yy))
      tmp_array = xr.open_dataset(file_name,decode_times=False) #kg C m-2
      if flag_start:
       prior_flux=tmp_array.copy(deep=True)
       flag_start=0
      else:
       prior_flux=xr.concat([prior_flux,tmp_array],dim='time')
      tmp_array.close()
   prior_flux['time']=time_range
   if prior_flux.lon.size==97: prior_flux=prior_flux.sel(lon=prior_flux.lon[:-1])
   if 'veget' in prior_flux.dims.keys(): prior_flux=prior_flux.sum(dim='veget')
   if prior_flux.lat.values[0]<prior_flux.lat.values[1]: 
     prior_flux['lat']=np.flip(prior_flux['lat'].values)
     prior_flux[var_name].values=np.flip(prior_flux[var_name].values,axis=1)
   if (attr == "Gpp") & (compound_attr=='COS'): #Transformation des flux CO2 en flux de COS
    for yy in range(begy,endy+1):
     LRU=map_LRU(mdl.LRU_case,yy)*Masse_molaire.COS/Masse_molaire.CO2
     if mdl.LRU_case != 5: LRU=LRU[np.newaxis,:,:]
     prior_flux['flx_co2'].loc[(prior_flux.time.dt.year==yy)]*=LRU
    tmp=np.copy(prior_flux.flx_co2.values)
    if mdl.order==0:
     cos_o=pd.read_pickle(mdl.COS.file_obs+'hemi.pkl')
     co2_o=pd.read_pickle(mdl.CO2.file_obs+'hemi.pkl')
     if (co2_o.mean().mean()>500)|(co2_o.mean().mean()<300): 
      print('CO2 hemispheric values are erroneous. Verify that the units be in ppm:',co2_o.mean().mean())
      sys.exit()
     if (cos_o.mean().mean()*10**6<300)|(cos_o.mean().mean()*10**6>800): 
      print('COS hemispheric values are erroneous. Verify that the units be in ppm:',cos_o.mean().mean())
      sys.exist()
     #Remplir les nan avant mars 2000 car il  ny pas d'observations de cos
     t_index = pd.date_range(start='2000-01-01', end='2000-02-29', freq='1M')
     t_index=pd.DataFrame(t_index)
     t_index.rename(columns={0: "date"},inplace=True)
     cos_o=cos_o.append(t_index,ignore_index=True,sort=True) 
     cos_o.set_index('date',inplace=True)
     cos_o=cos_o.resample('M').mean()
     cos_o.fillna(method="backfill",inplace=True)
     ### #####Correction a faire dans les obs
     cos_o.reset_index(inplace=True);co2_o.reset_index(inplace=True)
     cos_o=cos_o[(cos_o['date'].dt.year >= begy)&(cos_o['date'].dt.year <= endy)]
     co2_o=co2_o[(co2_o['date'].dt.year >= begy)&(co2_o['date'].dt.year <= endy)]
     co2_o.set_index('date',inplace=True) ; cos_o.set_index('date',inplace=True)
     index_hn=np.where(prior_flux.lat.values>0)[0]
     index_hs=np.where(prior_flux.lat.values<=0)[0]
     tmp[np.isnan(tmp)]=0
     tmp[:,index_hn,:]=tmp[:,index_hn,:]*cos_o['HN'].values[:,np.newaxis,np.newaxis]/co2_o['HN'].values[:,np.newaxis,np.newaxis]
     tmp[:,index_hs,:]=tmp[:,index_hs,:]*cos_o['HS'].values[:,np.newaxis,np.newaxis]/co2_o['HS'].values[:,np.newaxis,np.newaxis]
    else:
     cos_o=xr.open_dataset(mdl.homedir+"/INPUTF"+"/COS_variable.nc")
     co2_o=xr.open_dataset(mdl.homedir+"/INPUTF"+"/CO2_variable.nc") 
     cos_o=cos_o.loc[dict(time=slice(datetime.datetime(begy,1,1), datetime.datetime(endy,12,1)))]
     co2_o=co2_o.loc[dict(time=slice(datetime.datetime(begy,1,1), datetime.datetime(endy,12,1)))]
     tmp=tmp*cos_o.COS.values[:,:,:-1]/co2_o.CO2.values[:,:,:-1]
    prior_flux[var_name] = (('time','lat', 'lon'), tmp)
   prior_flux[var_name]*=area_LMDZ[np.newaxis,:,:]
   prior_flux=prior_flux.sum(dim='lon')
   prior_flux=prior_flux.sum(dim='lat')
   prior_flux[var_name]*=time_range.day*86400*items.sign
   prior_flux=prior_flux.resample(time='A').sum()
   prior_flux[var_name].values*=10**(-6) if compound_attr== "COS" else 10**(-12)
   unit='GtC/year' if compound_attr== "CO2" else 'GgS/year'
   for yy in range(begy,endy+1):
    print(compound_attr,items.__doc__,yy,round(prior_flux[var_name].values[yy-begy],3),unit)
    convert_unit=10**(-6) if compound_attr== "COS" else 1
    prior_global['compound'].append(compound_attr)
    prior_global['parameter'].append(attr)
    prior_global['value'].append(prior_flux[var_name].values[yy-begy]*convert_unit*Masse_molaire.CO2/getattr(Masse_molaire,compound_attr))  
    prior_global['year'].append(yy)
 prior_global=pd.DataFrame(prior_global)
 prior_global.to_pickle(mdl.storedir+'prior_global.pkl')



def calcul_aires():

 """Stockage des aires pour chaque fraction de pft et chaque region
 Area of the PFT region, area_reg: area of the region
 Verify the areas covered by each pft and region"""

 reg_array=get_reg()
 areas_veg={'PFT':[],'region':[],'year':[],'area':[],'area_reg':[]}
 for yy in range(begy,endy+1):
  for reg in Vector_control.Gpp.region:
   code=region[region['region']==reg].code.values
   rows,columns=np.where(reg_array['BASIN'].values == code)
   for ii in range(1,num_pft):
    pft_frac=veget_array['maxvegetfrac'+str(yy)].copy(deep=True).values[:,ii,:,:]
    pft_frac[np.isnan(pft_frac)]=0
    pft_frac=np.squeeze(pft_frac)
    area_pft=veget_array['Areas'].copy(deep=True).values*veget_array['Contfrac'].copy(deep=True).values
    area_reg=area_pft[rows,columns]
    area_reg[np.isnan(area_reg)]=0
    area_reg=np.sum(area_reg)
    area_pft=np.copy(area_pft*pft_frac)
    area_pft[np.isnan(area_pft)]=0
    area_pft=area_pft[rows,columns]
    area_pft=np.squeeze(np.sum(area_pft))
    area_pft=np.sum(area_pft)
    if area_pft !=0 :
     areas_veg['PFT'].append(ii+1)
     areas_veg['region'].append(reg)
     areas_veg['year'].append(yy)
     areas_veg['area'].append(area_pft)
     areas_veg['area_reg'].append(area_reg)
 areas_veg=pd.DataFrame(areas_veg)

 if Vector_control.Gpp.veget == "PFTREG":
  nbreg_veget=len(areas_veg.groupby(['PFT','region']).mean())
 elif Vector_control.Gpp.veget == "REG":
  nbreg_veget=len(areas_veg.groupby(['region']).mean())
  areas_veg=areas_veg.groupby(['region']).sum()
  areas_veg.reset_index(inplace=True)
  areas_veg['PFT']=np.nan
 else:
  print("probleme region")
 return areas_veg



