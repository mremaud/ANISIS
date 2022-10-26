#!/usr/bin/env python
#author @Marine Remaud
#Prepartion of the prior fluxes (as given in the inversion processes) to be transported with LMDz

homedir='/home/users/mremaud/PYTHON/COS/DA_SYSTEM/PREPROC'

from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import os
import datetime
import calendar
import pandas as pd
import copy
import math
from sys import argv
import sys
import matplotlib.pyplot as plt
import importlib
from regridding import *
from useful import *
from LRU_map import *

beg_y=2008
end_y=2019
dir_flux='/home/surface1/mremaud/COS/INPUT_sflx/GppResp/Trendy/'
dir_flux="/home/satellites4/mremaud/DA_SYSTEM/LRU4-20082019/Gpp0.2Resp0.2Ant0.5Biomass10.9Soil0.3OceanOCS2.0/"
LRU_case=3
MS=32.065
MC=12.0

for yy in range(beg_y,end_y+1):
 nc_file = xr.open_dataset(dir_flux+"post_GppCO2_"+str(yy)+'.nc',decode_times=False)
 #nc_file = xr.open_dataset(dir_flux+'flx_gpp_'+str(yy)+'.nc',decode_times=False)
 nc_file.time.values=[datetime.datetime(yy,x+1,1) for x in range(len(nc_file.time))]
 gpp=nc_file.flx_co2.values
 fcos=np.zeros(gpp.shape)

 for vv in nc_file.veget.data:
   FCOS=Vcos_order1(LRU_case,vv+1,yy,nc_file.sel(veget=vv))
   fcos[:,vv,:,:]= np.copy(FCOS)*MS/MC  #*10**(6)

 ds = xr.Dataset({'flx_cos': (['time','veget','lat', 'lon'], fcos)},
                            coords={'lat': (['lat'], nc_file.lat.values),
                                    'lon': (['lon'], nc_file.lon.values),
                                    'veget': (['veget'], nc_file.veget.values),
                                    'time': (['time'],nc_file.time.values)})

# ds["flx_cos"].values*=(-1)
 ds.to_netcdf(dir_flux+'/flx_cos_order1_'+str(yy)+'_LRU'+str(LRU_case)+'_PFT.nc', mode='w')
 ds=ds.sum("veget")
 ds.to_netcdf(dir_flux+'/flx_cos_order1_'+str(yy)+'_LRU'+str(LRU_case)+'_dyn.nc', mode='w')
 ds_out=lmdz2pyvar(ds,'flx_cos')
 ds_out=ds_out.get(['flx_cos'])
 ds_out.attrs["units"] = "kgS/m2/s"
 ds_out.to_netcdf(dir_flux+'/flx_cos_order1_'+str(yy)+'_LRU'+str(LRU_case)+'_phy.nc', mode='w')

