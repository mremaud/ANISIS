#!/usr/bin/env python
#Marine Remaud: Configuration file 
#Ne pas donner les memes noms de flux pour CO2 et COS que pour les sources communes
#Les flux sont en kg/m2/S

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

begy=2008
endy=2019
num_pft=15   #number of PFT

name_sim='LRU4-'+str(begy)+str(endy)+"-order1"
version="tangent_linear" #; from_exp="optim1_ocean_20102013_ref"
LRU_case=4               #Column number 4 of the LRU table (see at the end of the file)
only_season=0

homedir="/home/users/mremaud/PYTHON/COS/DA_SYSTEM/"
file_PFT='/home/surface1/mremaud/COS/PFTMAP/PFTMAPLMDZ-'
file_PFT_05='/home/surface1/mremaud/COS/PFTMAP/PFTMAP05-'
file_area="/home/surface1/mremaud/CO2/LMDZREF/start-96-L39.nc"
dir_adjoint="/home/satellites4/mremaud/FOOTPRINT/"
storedir="/home/satellites4/mremaud/DA_SYSTEM/"+name_sim+'/'

#***********************OBSERVATIONS***************************** 
"""
OBSERVATIONS: List of chemical coupounds with their stations. 
resolution: temporal resolution of the observations ("W","M","A")
"""

class COS:
    file_obs='/home/surface1/mremaud/COS/SURFACE/STATION/' 
    name="COS"
    resolution='W'
    stations=["CGO","LEF","NWR_afternoon","ALT",'WIS','BRW','SPO','SMO','MLO_afternoon','SUM','THD','PSA',"MHD","HFM"] 
compound=[COS]

# ********************PRIOR EMISSIONS ***************************
"""
List of emission files for each compound (ex COS, CO2) 
file name : prior netcdf file
clim      : 1 climatological prior file or 0, anuual emission files. 
"""
class Sources:
   "Sources and sinks to take into account in the CO2 and COS budget at the mensual resolution (default)"
   class COS:
     class Puit_OH:
       "Atmospheric OH sink"
       sign=1
       file_name="/home/surface1/mremaud/COS/INPUT_sflx/Chem/flx_OH_2010_dyn.nc"
       clim=1
     class Ocean_ind1:
       "ocean indirect production DMS"
       sign=1
       file_name="/home/surface1/mremaud/COS/INPUT_sflx/Ocean/Sinikka/sflx_odms_2010_dyn.nc"
       clim=1
     class Ocean_ind2:
       "ocean indirect production CS2"
       sign=1
       file_name="/home/surface1/mremaud/COS/INPUT_sflx/Ocean/Sinikka/flx_ocs2_XXXX_dyn.nc"
       clim=0
     class Ocean_direct:
       "ocean direct production"
       sign=1
       file_name="/home/surface1/mremaud/COS/INPUT_sflx/Ocean/Sinikka/flx_odirect_XXXX_dyn.nc"
       clim=0
     class Ant:
       "anthropogenic source"
       sign=1
       file_name="/home/surface1/mremaud/COS/INPUT_sflx/Anthro/flx_XXXX_dyn.nc"
       clim=0
     class Soil1:
       "Ogee/Whelan Soil"
       sign=1
       file_name="/home/surface1/mremaud/COS/INPUT_sflx/Soil/Ogee/flx_XXXX_PFT.nc"
       clim=0
     class Bios:
       "Vegetation uptake"
       sign=1
       file_name="/home/surface1/mremaud/COS/INPUT_sflx/GppResp/Trendy/flx_cos_order1_XXXX_LRU3_PFT.nc"
       clim=0
     class Biomass1:
       "Biomass burning"
       sign=1
       file_name="/home/surface1/mremaud/COS/INPUT_sflx/Biomass/Stin/flx_XXXX_dyn.nc"
       clim=0
#*********************************CONTROL VECTOR *****************************************
"""
groupe: list of the emission fluxes inclucled in the parameter to be optimized. The be chosen above. 
veget: discretisation of the vegetation: per PFT REG, PFT or REG
resol: "M" monthly, "A" annually
region: list of the regions to be optimized. The regions are further defined in a netcdf file
pdfreg: same as region but with the PFTs 
coef: prior errors (diagonal terms of the variance covariance error matrix)
coefPFT: Same as coef but with different coef for each pft (one per PFT)
sigma_t : length (in days) of temporal autocorrelation in case of monthly/weekly optimization
sigma_tPFT: Same as sigma_t but with different sigma_t for each pft (one per PFT)
 """
class Vector_control: 
   class Bios:
      name="Biospheric fluxes"
      pftreg={"HN":[4,5,6,10,11,12,13],"HS":[4,5,6,10,11,12,13],"GLOBE":[1,2,3,7,8,9,14,15]}
      coef=  [0.2]   #Coef region
      coefPFT=[0.2,0.1,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
      resol= "M"
      compound="COS"
      veget='PFTREG'
      groupe=["Bios"]
      sigma_t=100
      sigma_tPFT=[80,360,360,90,90,90,90,90,90,90,90,90,90,90,90]
   class Ant:
      name="Anthropogenic emissions"
      region=["GLOBE"]
      coef=[0.5]
      resol='A'
      compound='COS'
      groupe=["Ant"]
      veget="no"
      sigma_t=500
   class Biomass1:
     name="Biomass burning emissions"
     coef=[0.9]
     region=["GLOBE"]
     compound='COS'
     resol="M"
     groupe=["Biomass1"]
     veget="no"
     sigma_t=60
   class Soil:
      name="Soil fluxes"
      #region=['HN','Tr','HS']
      pftreg={"HN":[4,5,6,10,11,12,13],"HS":[4,5,6,10,11,12,13],"GLOBE":[1,2,3,7,8,9,14,15]}
      coef=[0.3]
      resol='M'
      compound='COS'
      groupe=["Soil1"]
      veget="PFTREG"
      sigma_t=100
   class OceanOCS:
      name="Oceanic emissions"
      region=['HN','Tr','HS']
      coef=[2.]
      resol='M'
      compound='COS'
      groupe=["Ocean_ind2","Ocean_ind1","Ocean_direct"]
      veget="no"
      sigma_t=60

offset_coef=0.003

class Masse_molaire:
   "Masse molaire moleculaire en g.mol"
   COS= 32.065
   CO2= 12.
# =============================================================================
#  INVERSION PARAMETERS
# ------------------------------------------------------------------------------
inversion="AnalyticInv"

#  Fixed variables used in the tarantola programm
# ------------------------------------------------------------------------------
class Tarantola:
  input_file_dat='tarantola_in.dat'
  input_file_bin='tarantola_in.bin'
  output_file_bin='tarantola_out.bin' 
  tempo_file_bin='tarantola_tmp.bin'
  file_matcov_out = 'matcov_out'
  file_matcor_out = 'matcor_out'
  log_file='tarantola.log'
  exec_dir=homedir+'modules/TARANTOLA/'
# ==============================================================================================

table_ORC = [
                       ['1 Bare soil'                       ,"1 BaSoil"],
                       ['2 Tropical broadleaf evergreen'    ,"2 TrBrE"  ],
                       ['3 Tropical broadleaf raingreen'    ,"3 TrBrR"  ],
                       ['4 Temperate needleleaf evergreen'  ,"4 TeNeE"  ],
                       ['5 Temperate broadleaf evergreen'   ,"5 TeBrE"  ],
                       ['6 Temperate broadleaf summergreen' ,"6 TeBrS"  ],
                       ['7 boreal needleleaf evergreen'     , "7 BoNeE"],
                       ['8 boreal broadleaf summergreen'    , "8 BoBrS"],
                       ['9 boreal needleleaf summergreen'   , "9 BoNeS"],
                       ['10 Temperate C3 grass'             ,  "10 TeC3g"],
                         ['11 C4 grass'                       ,  "11 C4g"],
                       ['12 C3 Agriculture'                 ,   "12 C3Ag"],
                       ['13 C4 Agriculture'                 ,   "13 C4Ag"],
                       ['14 Tropical C3 grass'              ,   "14 TrC3g"],
                       ['15 Boreal C3 grass'                ,   "15 BoC3g"]
                       ]


table_LRU = [
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


