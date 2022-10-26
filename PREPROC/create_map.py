#!/usr/bin/env python
"""
author: Marine Remaud
Purpose: Creation of the maps that define the regions to optimize for each processes in the inverse system
Return : netcdf file and a csv files containing the coordinates of the regions
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import math
import string
import copy
import calendar
from scipy.io import netcdf
from netCDF4 import Dataset

transcom=0
processus=["OceanOCS"]
nb_PFT=1
PFT=[np.nan]
interpolation=1 #Interpolation sur une grille 0.5

dirout='/home/users/mremaud/PYTHON/COS/DA_SYSTEM/INPUTF/region-'

#Fichier de reference Transcom
varname="BASIN"
thefile='/home/users/bousquet/INVERSION/MIOP/MAP/map_miop_gcp_LMDZ9696.nc'
ffile = Dataset(thefile,'r',format="NETCDF4")
lon = ffile.variables['LON'][:]
lat = ffile.variables['LAT'][:]
basin = ffile.variables[varname][:]
ffile.close()
if lat[0]<lat[1]:
 lat=lat[::-1]
 basin=basin[::-1,:]



for pp in processus:
 #Creation d un fichier csv concernant les infos sur les regions
 if nb_PFT>1:
  dico={'region':[],'code':[],'PFT':[]}
 else:
  dico={'region':[],'code':[]}
 for vv in range(1,nb_PFT+1):
  thefilef=dirout+pp+'.nc'
  thedicof=dirout+pp+'.csv'
  if transcom:
   #Regions defined as in Transcom experiments
   col=[10,240,130,40,150,60,210,80,190,100,180,120,30,140,50,160,20,110,90,200,70,220,5,170,250] 
   reg_short = ['m_ALK','m_CAN','m_cUS','m_CAm',\
        'm_TrSAm','m_BRAW','m_BRAE','m_TeSAm',\
        'EuB','m_EuW','m_EuE',\
        'm_ME','m_CAs','m_RUS','m_EAs','m_SAs','m_Chi','m_SAEs','m_Aus',\
        'm_NAf','m_EqAfN','m_EqAfS','m_SAf','m_OCE','m_ANT']
   regions = ['Alaska','Canada','cUSA','Central_America' \
           'Tropical_South_America_West','Brazil_West','Brazil_East','Temperate_South_America',\
           'Europe_boreal','Europe_West','Europe_East',\
           'Middle_East','Central_Asia','Russia','East_Asia','India','China','South_Asia',\
           'Oceania','Australia',\
           'Nothern_Africa','Equatorial_Africa_North','Equatorial_Africa_South','Southern_Africa','Oceans','Antarctica']

   #If you want to aggregate some regions: be carefull to the partition ocean/land
   reg_shortf = ['NAm','SAm','Eu','AsS','AsN', 'AfN','AfT','AfS','m_OCE','m_ANT']
   regionf = {"NAm": [] , "SAm": [] , "Eu": [],"Af":[],"Rus":['m_RUS'],'SAs':[],'Ocea':[],'m_OCE':['m_OCE'],'m_ANT':['m_ANT']}
   regionf['NAm']  = ['m_ALK','m_CAN','m_cUS','m_CAm']
   regionf['SAm']  = ['m_TrSAm','m_BRAW','m_BRAE','m_TeSAm']
   regionf['AfN']  = ['m_NAf','m_ME']
   regionf['AfT']  = ['m_EqAfN','m_EqAfS']
   regionf['AfS']  = ['m_SAf']
   regionf['Eu']   = ['EuB','m_EuW','m_EuE']
   regionf['AsN']  = ['m_RUS','m_SAs','m_Chi','m_EAs','m_CAs']
   regionf['AsS']  = ['m_SAEs','m_Aus']
   regionf['m_ANT']= ['m_ANT']
  else:
  # reg_shortf=['Am','Eu','As']
  # col=      [0   ,  1 ,2]
  # lon_min=[-180 , -50, 60]
  # lon_max=[-50  , 60,   180]
  # lat_min=[-90 ,  -90 ,-90]
  # lat_max=[90 , 90,90]

   reg_shortf=['HS','Tr','HN']
   col=      [0   ,  1, 2]
   lon_min=  [-180 ,-180, -180]
   lon_max=[180  ,180,   180]
   lat_min=[-90 ,-30,30]
   lat_max=[-30 ,30,90]
 #  reg_shortf=['HS','HN']
 #  col=      [0   ,  1  ]
 #  lon_min=  [-180 , -180]
 # lon_max=[180   ,   180]
 #  lat_min=[-90 ,0]
 #  lat_max=[0, 90]

  #Final map
  basinf=np.copy(basin)
  for nn in range(len(reg_shortf)):
   if transcom:
    compteur=0
    #Carte apres agregation des grandes regions
    for name_reg in regionf[nn]:
     nr=[i for i,jj in enumerate(reg_short) if reg_short[i]==name_reg][0]
     #--- Lecture des regions et affectation dans la grille
     rows,columns=np.where(basin == col[nr])
     if compteur == 0 :
      valeur=np.mean(basin[rows,columns]) 
     else: 
      basinf[rows,columns]=valeur
     compteur+=1
   else:
    columns=np.where((lon>=lon_min[nn])&(lon<=lon_max[nn]))[0]
    rows=np.where((lat<lat_max[nn])&(lat>lat_min[nn]))[0]
    basinf[np.ix_(rows,columns)]=np.copy(col[nn])
    valeur=np.copy(col[nn])
   dico['region'].append(reg_shortf[nn])
   dico['code'].append(valeur)
   if nb_PFT>1:
    dico['PFT'].append(int(vv))

  if (not np.isnan(vv)) and (vv > 1): 
    basinf2=np.concatenate((basinf2,basinf[np.newaxis,:,:]),axis=0)
  else:
    basinf2=basinf[np.newaxis,:,:]

 #Final csv and netcdf files
 dico=pd.DataFrame(dico)
 dico.to_csv(thedicof,index=None)

 resultfile = netcdf.netcdf_file(thefilef,'w')
 resultfile.createDimension('lon',len(lon))
 resultfile.createDimension('lat',len(lat))
 if nb_PFT>1: resultfile.createDimension('pft',nb_PFT)
 var1=resultfile.createVariable('lat','d', ('lat',) )
 var1[:] = lat[:]
 var2=resultfile.createVariable('lon','d', ('lon',) )
 var2[:] = lon[:]
 if nb_PFT>1:
  var3=resultfile.createVariable('pft','d', ('pft',) )
  var3[:] = np.arange(1,nb_PFT+1)
 if nb_PFT>1:
  var_coord = ('pft','lat', 'lon')
 else:
  var_coord = ('lat', 'lon')
 v = resultfile.createVariable( varname, 'd', var_coord )
 v[:] = basinf2[:]
 resultfile.close()
