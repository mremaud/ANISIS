#!/usr/bin/env python
#Marine Remaud
#Computation of the COS fluxes by using the LRU parameterization

from calendar import isleap
import numpy as np
import os
from netCDF4 import Dataset
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys
from useful import *
import copy
import xarray as xr


def factor_vcos(icase,veget_type,yy,gpp):
  """Return thescale factors (one per month) for the COS 
     (deposition velocity)  in the tangent linear
      * icase: case of the LRU ratio table
      * yy: year
      * veget_type: PFT
      * gpp flux 
  """

  dir_COS="/home/surface1/mremaud/COS/SURFACE/STATION/"
  dir_CO2="/home/surface1/mremaud/CO2/SURFACE/STATION/"

  table_vcos_seibt_ORC = [
                       ['1 Bare soil',                      0.00,0.00,0.00,0.00,0.00],
                       ['2 Tropical broadleaf evergreen',   3.09,2.27,3.40,1.68,1.72],
                       ['3 Tropical broadleaf raingreen',   3.38,2.48,3.71,1.68,1.62],
                       ['4 Temperate needleleaf evergreen', 1.89,1.39,2.08,1.68,1.39],
                       ['5 Temperate broadleaf evergreen',  3.60,2.64,3.96,1.68,1.06],
                       ['6 Temperate broadleaf summergreen',3.60,2.64,3.96,1.68,1.31],
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


  #Ratio of CO2/COS mole concentration
  cos_o=pd.read_pickle(dir_COS+'hemi.pkl')
  co2_o=pd.read_pickle(dir_CO2+'hemi.pkl')
  #Correction 2000

  t_index = pd.DatetimeIndex(start='2000-01-01', end='2000-02-29', freq='1M')
  t_index=pd.DataFrame(t_index)
  t_index.rename(columns={0: "date"},inplace=True)
  cos_o=cos_o.append(t_index,ignore_index=True)
  cos_o.set_index('date',inplace=True)
  cos_o=cos_o.resample('M').mean()
  cos_o.fillna(method="backfill",inplace=True)
  cos_o.reset_index(inplace=True);co2_o.reset_index(inplace=True)
  cos_o=cos_o[(cos_o['date'].dt.year == yy)]
  co2_o=co2_o[(co2_o['date'].dt.year == yy)]
  rapport_hs=np.copy(cos_o.HS.values/co2_o.HS.values)
  rapport_hn=np.copy(cos_o.HN.values/co2_o.HN.values)
#  co2_o.set_index('date',inplace=True) ; cos_o.set_index('date',inplace=True)
  FCOS=np.zeros(np.shape(gpp.flx_co2.values))
  #LRU map function of PFT
  vcos_factor =table_vcos_seibt_ORC[veget_type-1][icase+1] 
  index_hs=np.where(gpp.lat<=0)[0]
  index_hn=np.where(gpp.lat>=0)[0] 
  FCOS[:,index_hn,:]=vcos_factor*gpp.flx_co2.values[:,index_hn,:]*rapport_hn[:,np.newaxis,np.newaxis]
  FCOS[:,index_hs,:]=vcos_factor*gpp.flx_co2.values[:,index_hs,:]*rapport_hs[:,np.newaxis,np.newaxis]
  return FCOS


def Vcos_order1(icase,veget_type,yy,gpp):
  """Return thescale factors (one per month) for the COS 
     (deposition velocity)  in the tangent linear
      * icase: case of the LRU ratio table
      * begy
      * endy
      * veget_type: PFT
  """

  table_vcos_seibt_ORC = [
                       ['1 Bare soil',                      0.00,0.00,0.00,0.00,0.00],
                       ['2 Tropical broadleaf evergreen',   3.09,2.27,3.40,1.68,1.72],
                       ['3 Tropical broadleaf raingreen',   3.38,2.48,3.71,1.68,1.62],
                       ['4 Temperate needleleaf evergreen', 1.89,1.39,2.08,1.68,1.39],
                       ['5 Temperate broadleaf evergreen',  3.60,2.64,3.96,1.68,1.06],
                       ['6 Temperate broadleaf summergreen',3.60,2.64,3.96,1.68,1.31],
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


  file_cos="../INPUTF/COS_variable.nc"
  file_co2="../INPUTF/CO2_variable.nc"
  cos_o=xr.open_dataset(file_cos)
  cos_o=cos_o.sel(time=cos_o.time.dt.year == yy)
  co2_o=xr.open_dataset(file_co2)
  co2_o=co2_o.sel(time=co2_o.time.dt.year == yy)
  vcos_factor=np.zeros(np.shape(co2_o.CO2.values[:,:,:]))
  FCOS=np.zeros(np.shape(co2_o.CO2.values[:,:,:]))
  if not np.isnan(veget_type):
    #LRU map function of PFT
    vcos_factor[:,:,:] =table_vcos_seibt_ORC[veget_type-1][icase+1]
    print(np.mean(vcos_factor))
  else:
    for vv in range(num_PFT):
     #Pas la meme table quen haut
     vcos_factor +=  vegetfrac[vv,:,:]*table_vcos_seibt_ORC[vv,icase+1]
  factor=np.divide(cos_o.COS.values,co2_o.CO2.values)
  vcos_factor=np.multiply(factor,vcos_factor)
  FCOS[:,:,:]=vcos_factor*gpp.flx_co2.values[:,:,:]
  return FCOS


