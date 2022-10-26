#!/usr/bin/env python
#Marine Remaud

import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import time
import datetime 
import os
from calendar import isleap
import datetime
from math import cos, asin, sqrt
import calendar

def detect_dimensions(data_array):
    """
    Detect the dimensions of the DataArray (xarray)
    :param data_array:
    :return: a list with True if the dimension is detected, None if not. In the order [time, pres, lat, lon].
    """
    data_array = change_dim_name(data_array)
    dims = data_array.dims
    dims2detect = ['time', 'pres', 'lat', 'lon']
    detection = [dim in dims for dim in dims2detect]
    return detection

def data2data_array(data_array, data, dataset=False):
    """
    Convert a Numpy array into a DataArray (xarray). Retrieve the dimensions from
    the DataArray. The two arrays need to have the same shape.
    Assuming there is longitude and latitude dimensions.

    :param data_array: A DataArray (xarray) with dimensions needed
    :param data: Numpy array data
    :param dataset: Return the DataSet instead of the DataArray
    :return: The DataArray with the data from the Numpy array.
    """
    data_array = change_dim_name(data_array)
    dims = detect_dimensions(data_array)
    lats, lons = data_array.lat.values, data_array.lon.values
    if dims[0]:
        time = data_array.time.values
        if dims[1]:
            pres = data_array.pres.values
            ds = xr.Dataset({'data': (['time', 'pres', 'lat', 'lon'], data)},
                            coords={'lat': (['lat'], lats),
                                    'pres': (['pres'], pres),
                                    'lon': (['lon'], lons),
                                    'time': (['time'], time)})
        else:
            ds = xr.Dataset({'data': (['time', 'lat', 'lon'], data)},
                            coords={'time': (['time'], time),
                                    'lat': (['lat'], lats),
                                    'lon': (['lon'], lons)})
    else:
        if dims[1]:
            pres = data_array.pres.values
            ds = xr.Dataset({'data': (['pres', 'lat', 'lon'], data)},
                            coords={'lat': (['lat'], lats),
                                    'pres': (['pres'], pres),
                                    'lon': (['lon'], lons)})
        else:
            ds = xr.Dataset({'data': (['lat', 'lon'], data)},
                            coords={'lat': (['lat'], lats),
                                    'lon': (['lon'], lons)})
    if dataset:
        return ds
    else:
        return ds['data']




def change_dim_name(data_array):
    """
    Change the dimensions names to 'time', 'presnivs', 'lat', 'lon'

    :param data_array: A DataArray (xarray)
    :return: Same DataArray with modified dimensions names
    """
    data_array=data_array.squeeze( drop=True)
    dims = data_array.dims
    dims_new = ('time', 'pres', 'lat', 'lon')
    for i, dim in enumerate(dims):
        for dim_new in dims_new:
            if dim.lower().find(dim_new.lower()) > -1:
                data_array = data_array.rename({dim: dim_new})
        if dim.lower().find('lev') > -1:
            data_array = data_array.rename({dim: 'pres'})
        elif dim.lower().find('sig') > -1:
            data_array = data_array.rename({dim: 'pres'})
    return data_array


def fraction_an(row):
  """
  Calculation of the fraction of year before applying the ccgvu routine 
  """

  day_in_year = calendar.isleap(row['date'].year) and 366 or 365
  start_date=datetime.datetime(row['date'].year,1,1,0,0)
  result=row['date'].year+(row['date']-start_date).total_seconds()/(day_in_year*86400.)
  return result


def outlier_filter(data):
  """
  Fit a curve with ccgvu and remove outliers
  """
  data['frac']=data.apply(lambda row: fraction_an(row),axis=1)
  time_serie=data[['frac','sim']]
  file_out='tmp.txt'
  os.system('rm -f '+file_out)
  np.savetxt(file_out,time_serie,fmt="%4.8f %4.12f")
  file_fit='tmp-fit.txt'
  os.system('./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
  data2=np.loadtxt(file_fit)
  time=[datetime.datetime(int(data2[ii,0]),int(data2[ii,1]),int(data2[ii,2])) for ii in range(len(data2[:,0]))]
  fit=data2[:,3]
  data2=data2[:,3]-data2[:,5]
  
  STD=1.5*np.std(data2)
  indice=np.where(np.abs(data2)>STD)[0]
  data['sim'].iloc[indice]=fit[indice]

  time_serie=data[['frac','sim']]
  os.system('rm -f '+file_out)
  np.savetxt(file_out,time_serie,fmt="%4.8f %4.12f")
  file_fit='tmp-fit.txt'
  os.system('./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
  data2=np.loadtxt(file_fit)
  time=[datetime.datetime(int(data2[ii,0]),int(data2[ii,1]),int(data2[ii,2])) for ii in range(len(data2[:,0]))]
  fit=np.copy(data2[:,5])
  data2=data2[:,3]-data2[:,5]
  data['sim'].iloc[indice]=fit[indice]
  return data

