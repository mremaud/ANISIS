#!/usr/bin/env python
#author @Marine Remaud
#Redimentioning of the transport matrix from a previous experiment
#Define and load H
from .useful import *

from scipy.stats import linregress
import random
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
import statsmodels.api as smf
from sys import argv
from dateutil.relativedelta import relativedelta
from scipy.optimize import curve_fit
import sys
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})


def redim_H(obs_vec,index_ctl_vec,index_unopt):
   """
   Redimentionate the transport matrix g and the
   associated linear tangent (matrix_G) (normalized for 1GtC) from a previous experiment
   from_exp: directory of the previous experiment
   obs_vec: observation vector
   index_ctl_vec: control vector containing all the information to fill the tangent linear
   index_nopt: fixed (unoptimized) flux 
   """
   dir_exp=mdl.storedir+"../"+mdl.from_exp
   obs_vec_ref      =pd.read_pickle(dir_exp+'/obs_vec.pkl')
   index_ctl_vec_ref=pd.read_pickle(dir_exp+'/index_ctl_vec.pkl')
   index_unopt_ref  =pd.read_pickle(dir_exp+'/index_unopt.pkl')
   #Select the line that are alike
   idx=diff_rows(obs_vec,obs_vec_ref)
   obs_vec=obs_vec.reindex(idx)
   obs_vec.reset_index(inplace=True,drop=True)
   idx=diff_rows(index_ctl_vec,index_ctl_vec_ref)
   index_ctl_vec=index_ctl_vec.reindex(idx)
   idx=diff_rows(index_unopt,index_unopt_ref)
   index_unopt=index_unopt.reindex(idx)
   index_unopt.reset_index(inplace=True,drop=True)
   #Redimensionner la matrice
   idx=diff_rows(obs_vec_ref,obs_vec)
   matrix_G=np.load(dir_exp+"/matrix_G.npy")
   matrix_g=np.load(dir_exp+"/matrix_g.npy")
   matrix_G=matrix_G[idx,:]
   matrix_g=matrix_g[idx,:]
   np.save(mdl.storedir+"matrix_G.npy",matrix_G)
   np.save(mdl.storedir+"matrix_g.npy",matrix_g)

   return obs_vec


