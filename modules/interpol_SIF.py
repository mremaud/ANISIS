"""
Marine Remaud the 1st of April 2020
Evaluation of the optimized GPP against the SIF data
SIF data are gridded at 0.5 degre resolution
Selection of grid cells within each PFT and average them
"""

import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
import os
from netCDF4 import Dataset
import datetime
import calendar
import datetime
import pandas as pd
import copy
import xarray as xr
import math
from useful import *

#Import the variables in the config.py
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})

def extract_SIF(index_ctl_vec):
   """
   Argument: index_ctl_vec (pd data)
   GOME 2 daily average product (only available product before 2013: the fluxnet dataset stops in 2013)
   OCO2 only after 2015
   """
   what='GOME2'
   map_05=mdl.file_PFT_05+'2000.nc'
   #thefile'/home/surface6/cbacour/DATA/GOME2-V27/CAMS41'
   N=3

   #SIF data
   time_sif=[]
   flag_start=1
   rfile=mdl.homedir+"/INPUTF/region-Gpp_0.5.nc"

   #Open the region file at 0.5 degre
   ffile = Dataset(rfile,'r',format="NETCDF4")
   ncreg = ffile.variables['BASIN'][:]
   ffile.close()

   #Open the PFT map to get the area at 0.5 degre
   ffile = Dataset(map_05,'r',format="NETCDF4")
   Areas = ffile.variables['Areas'][:]
   ffile.close()
   Areas=np.squeeze(Areas)

   print("Ouverture des netcdf qui contiennent la SIF")
   #Compute SIFm, a multiyear monthly SIF vector
   if what == "GOME2":
    directory='/home/satellites7/maignan/REMOTE_SENSING_DATA/GOME-2/JOINER/v28/GOME_F/v28/MetOp-A/level3/'
    for yy in range(mdl.begy,mdl.endy+1):
     for mon in range(1,13):
      name_file=directory+str(yy)+'/ret_f_nr5_nsvd12_v26_waves734_nolog.grid_SIF_v28_'+str(yy)+"{:02}".format(mon)+'01_31.nc'
      if not  os.path.exists(name_file): print(directory); continue
      ffile = Dataset(name_file,'r',format="NETCDF4")
      lon = ffile.variables['longitude'][:]
      lat = ffile.variables['latitude'][:]
      SIF = ffile.variables['SIF_daily_average'][:]
      cos= ffile.variables['cos(SZA)'][:]
      ffile.close()
      time_sif.append(datetime.datetime(yy,mon,1))
#     SIF[SIF.mask]=np.nan
      SIF[SIF<-200]=np.nan
      #SIF[cos< math.cos(math.radians(40))]=np.nan
     # SIF=np.multiply(np.squeeze(SIF),Areas)
      if flag_start:
       SIFm=SIF[None,:,:]
       flag_start=0
      else:
       SIFm=np.concatenate((SIFm,SIF[None,:,:]),axis=0)      
      if lat[0]<0:
       lat=lat[::-1]
       SIFm=SIFm[:,::-1,:]
    time_sif=pd.to_datetime(time_sif)
   elif what=="OCO-2":
    directory="/home/surface6/cbacour/DATA/Fluorescence/SATELLITES/OCO2/ori/"
    for yy in range(2015,2017):
     name_file=directory+"OCO2_SIF_"+str(yy)+"_map05.nc"
     ffile = Dataset(name_file,'r',format="NETCDF4")
     lon = ffile.variables['longitude'][:]
     lat = ffile.variables['latitude'][:]
     SIFm = ffile.variables['SIF_daily_average'][:]
     cos= ffile.variables['cos(SZA)'][:]
     ffile.close()
     if lat[0]<0:
       lat=lat[::-1]
       SIFm=SIFm[:,::-1,:]

lon=
lat=

#SIFm 
ntime=np.shape(SIFm)[0]
for tt in range(ntime):
 SIF





