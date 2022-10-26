#!/usr/bin/env python

from calendar import isleap
import numpy as np
import os
from netCDF4 import Dataset
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys
from .useful import *
import copy
#Import the variables in the config.py
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})


def map_LRU(icase,year):
  """ Return a LRU map 
      case 1 : Values from Seibt et al (2010) 
      case 2 : 0.5 * case 1
      case 3:   1.5*case 2 
      case 4 : Distinction C3/C4 according to whelan (2016)
      case 5 : LRU values function of the month from Maignan et al., 2020
  """
  
  veget_array=get_veget() 
  region,reg_array=get_region()
  pft_frac=veget_array['maxvegetfrac'+str(year)].copy(deep=True).values
  pft_frac=np.squeeze(pft_frac)
  pft_frac[np.isnan(pft_frac)]=0
  lon=veget_array.lon.values
  lat=veget_array.lat.values
  nlat=veget_array.lat.size
  nlon=veget_array.lon.size
  LRU=np.zeros((nlat,nlon)) 

  table_vcos_seibt_ORC = [
                       ['1 Bare soil',                      0.00,0.00,0.00,0.00,0.00],
                       ['2 Tropical broadleaf evergreen',   3.09,2.27,3.40,1.68,1.72],
                       ['3 Tropical broadleaf raingreen',   3.38,2.48,3.71,1.68,1.62],
                       ['4 Temperate needleleaf evergreen', 1.89,1.39,2.08,1.68,1.39],
                       ['5 Temperate broadleaf evergreen',  3.60,2.64,3.96,1.68,1.06],
                       ['6 Temperate broadleaf summergreen',3.60,2.64,3.96,1.68,1.31 ],
                       ['7 boreal needleleaf evergreen',    1.89,1.39,2.08,1.68,0.95],
                       ['8 boreal broadleaf summergreen',   1.94,1.43,2.14,1.68,1.03],
                       ['9 boreal needleleaf summergreen',  1.89,1.39,2.08,1.68,0.92],
                       ['10 Temperate C3 grass',            2.53,1.85,2.77,1.68,1.18],
                       ['11 C4 grass',                      2.00,1.46,2.19,1.21,1.45],
                       ['12 C3 Agriculture',                2.26,1.66,2.48,1.68,1.37],
                       ['13 C4 Agriculture',                2.00,1.46,2.19,1.21,1.72],
                       ['14 Tropical C3 grass',             2.39,1.75,2.62,1.68,1.52],
                       ['15 Boreal C3 grass',               2.02,1.48,2.22,1.68,0.97]
                       ]

  table_vcos_seibt_ORC = [
                       ['1 Bare soil',                      0.00,0.00,0.00,0.00,0.00,0.00],
                       ['2 Tropical broadleaf evergreen',   3.09,2.27,3.40,1.68,1.72,1.68],
                       ['3 Tropical broadleaf raingreen',   3.38,2.48,3.71,1.68,1.62,1.68],
                       ['4 Temperate needleleaf evergreen', 1.89,1.39,2.08,1.68,1.39,1.39],
                       ['5 Temperate broadleaf evergreen',  3.60,2.64,3.96,1.68,1.06,1.06],
                       ['6 Temperate broadleaf summergreen',3.60,2.64,3.96,1.68,1.31,1.31 ],
                       ['7 boreal needleleaf evergreen',    1.89,1.39,2.08,1.68,0.95,2.],
                       ['8 boreal broadleaf summergreen',   1.94,1.43,2.14,1.68,1.03,2.],
                       ['9 boreal needleleaf summergreen',  1.89,1.39,2.08,1.68,0.92,2.],
                       ['10 Temperate C3 grass',            2.53,1.85,2.77,1.68,1.18,1.18],
                       ['11 C4 grass',                      2.00,1.46,2.19,1.21,1.45,1.21],
                       ['12 C3 Agriculture',                2.26,1.66,2.48,1.68,1.37,1.37],
                       ['13 C4 Agriculture',                2.00,1.46,2.19,1.21,1.72,1.21],
                       ['14 Tropical C3 grass',             2.39,1.75,2.62,1.68,1.52,1.68],
                       ['15 Boreal C3 grass',               2.02,1.48,2.22,1.68,0.97,2.]
                       ]


  for pft_numb in range(num_pft):
       LRU[:,:] += np.squeeze(pft_frac[pft_numb,:,:])*table_vcos_seibt_ORC[pft_numb][icase]
  return LRU


def factor_vcos(icase,region,veget_type,yy,month):
  """Return thescale factors (one per month) for the COS 
     (deposition velocity)  in the tangent linear
      * icase: case of the LRU ratio table
      * region  : name of the region
      * yy: year
      * veget_type: PFT or nan      
  """
  dic_reg,reg_array=get_region("Gpp",veget_type,region)
  veget_array=get_veget()
  table_vcos_seibt_ORC = [
                       ['1 Bare soil',                      0.00,0.00,0.00,0.00,0.00],
                       ['2 Tropical broadleaf evergreen',   3.09,2.27,3.40,1.68,1.72],
                       ['3 Tropical broadleaf raingreen',   3.38,2.48,3.71,1.68,1.62],
                       ['4 Temperate needleleaf evergreen', 1.89,1.39,2.08,1.68,1.39],
                       ['5 Temperate broadleaf evergreen',  3.60,2.64,3.96,1.68,1.06],
                       ['6 Temperate broadleaf summergreen',3.60,2.64,3.96,1.68,1.31 ],
                       ['7 boreal needleleaf evergreen',    1.89,1.39,2.08,1.68,0.95],
                       ['8 boreal broadleaf summergreen',   1.94,1.43,2.14,1.68,1.03],
                       ['9 boreal needleleaf summergreen',  1.89,1.39,2.08,1.68,0.92],
                       ['10 Temperate C3 grass',            2.53,1.85,2.77,1.68,1.18],
                       ['11 C4 grass',                      2.00,1.46,2.19,1.21,1.45],
                       ['12 C3 Agriculture',                2.26,1.66,2.48,1.68,1.37],
                       ['13 C4 Agriculture',                2.00,1.46,2.19,1.21,1.72],
                       ['14 Tropical C3 grass',             2.39,1.75,2.62,1.68,1.52],
                       ['15 Boreal C3 grass',               2.02,1.48,2.22,1.68,0.97]
                       ]

  table_vcos_seibt_ORC = [
                       ['1 Bare soil',                      0.00,0.00,0.00,0.00,0.00,0.00],
                       ['2 Tropical broadleaf evergreen',   3.09,2.27,3.40,1.68,1.72,1.68],
                       ['3 Tropical broadleaf raingreen',   3.38,2.48,3.71,1.68,1.62,1.68],
                       ['4 Temperate needleleaf evergreen', 1.89,1.39,2.08,1.68,1.39,1.39],
                       ['5 Temperate broadleaf evergreen',  3.60,2.64,3.96,1.68,1.06,1.06],
                       ['6 Temperate broadleaf summergreen',3.60,2.64,3.96,1.68,1.31,1.31 ],
                       ['7 boreal needleleaf evergreen',    1.89,1.39,2.08,1.68,0.95,2.],
                       ['8 boreal broadleaf summergreen',   1.94,1.43,2.14,1.68,1.03,2.],
                       ['9 boreal needleleaf summergreen',  1.89,1.39,2.08,1.68,0.92,2.],
                       ['10 Temperate C3 grass',            2.53,1.85,2.77,1.68,1.18,1.18],
                       ['11 C4 grass',                      2.00,1.46,2.19,1.21,1.45,1.21],
                       ['12 C3 Agriculture',                2.26,1.66,2.48,1.68,1.37,1.37],
                       ['13 C4 Agriculture',                2.00,1.46,2.19,1.21,1.72,1.21],
                       ['14 Tropical C3 grass',             2.39,1.75,2.62,1.68,1.52,1.68],
                       ['15 Boreal C3 grass',               2.02,1.48,2.22,1.68,0.97,2.]
                       ]

  vegetfrac=np.copy(np.squeeze(veget_array['maxvegetfrac'+str(yy)].values))
  vcos_factor=np.zeros(np.shape(veget_array.Contfrac))
  #Ratio of CO2/COS mole concentration
  cos_o=pd.read_pickle(mdl.COS.file_obs+'hemi.pkl')
  co2_o=pd.read_pickle(mdl.CO2.file_obs+'hemi.pkl')
  #Correction 2000
  t_index = pd.date_range(start='2000-01-01', end='2000-02-29', freq='1M')
  t_index=pd.DataFrame(t_index)
  t_index.rename(columns={0: "date"},inplace=True)
  cos_o=cos_o.append(t_index,ignore_index=True,sort=False)
  cos_o.set_index('date',inplace=True)
  cos_o=cos_o.resample('M').mean()
  cos_o.fillna(method="backfill",inplace=True)
  cos_o.reset_index(inplace=True);co2_o.reset_index(inplace=True)
  cos_o=cos_o[(cos_o['date'].dt.year == yy)]
  cos_o=cos_o.groupby(cos_o['date'].dt.year).mean() if np.isnan(month) else cos_o[(cos_o['date'].dt.month == month)]
  co2_o=co2_o[(co2_o['date'].dt.year == yy)]
  co2_o=co2_o.groupby(co2_o['date'].dt.year).mean() if np.isnan(month) else co2_o[(co2_o['date'].dt.month == month)]
  index_hn=np.where(veget_array.lat.values>0)[0]
  index_hs=np.where(veget_array.lat.values<=0)[0]
  if not np.isnan(veget_type):
   #LRU map function of PFT
   vcos_factor[:,:] =table_vcos_seibt_ORC[veget_type-1][icase] 
  else:
   for vv in range(num_PFT):
    #Pas la meme table quen haut
    vcos_factor +=  vegetfrac[vv,:,:]*table_vcos_seibt_ORC[vv,icase]
   
  vcos_factor[index_hn,:]=vcos_factor[index_hn,:]*cos_o['HN'].values/co2_o['HN'].values
  vcos_factor[index_hs,:]=vcos_factor[index_hs,:]*cos_o['HS'].values/co2_o['HS'].values

  if region == "GLOBE":
   rows,columns=np.where(reg_array['BASIN'].values != np.nan) 
  else:
   code=dic_reg.loc[dic_reg['region']==region,"code"].values[0]
   rows,columns=np.where(reg_array['BASIN'].values == code)
  vcos_factor=np.mean(vcos_factor[rows,columns])
  return vcos_factor




