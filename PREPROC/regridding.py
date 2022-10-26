#!/usr/bin/env python

import numpy
import sys
import math
import string
import copy
import calendar
from scipy.io import netcdf
import xarray as xr


def lmdz2pyvar(data_array,name_var):
    """
    Convert a flux vector from LMDz grid to Pyvar grid, i.e. vector 9026 = 94 * 96 + 2
    :param data_array: a DataArray (xarray)
    :return: flx : the DataArray (ntimes, 9026)
    """
    lon, lat = data_array.lon.values, data_array.lat.values
    flx = data_array[name_var].values
    time = data_array.time.values
    ntimes = data_array[name_var].shape[0]
    if len(lon) == 97: 
       lon=lon[:-1]
       flx=flx[:,:,:-1]
    if lat[0] < lat[-1]:
        flx = numpy.flip(flx, 1)

    values_north = flx[:, 0, :].mean(1)  # longitude average
    values_south = flx[:, -1, :].mean(1)
    values_north = values_north[:, numpy.newaxis]
    values_south = values_south[:, numpy.newaxis]

    flx = flx[:, 1:-1, :]
    flx = numpy.reshape(flx, (ntimes, 9024))
    flx = numpy.append(values_north, flx, axis=1)
    flx = numpy.append(flx, values_south, axis=1)

    vector = numpy.linspace(1, 9026, 9026)
    ds_out = xr.Dataset({name_var: (['time', 'vector'], flx)},
                       coords={'vector': (['vector'], vector),
                                'time': (['time'], time)})
    return ds_out



def interpolation(zlati,zloni,bigtabi,time): 
   """
    param: zlati,zloni: longitude, latitude initiales
   : bigtabi : champs a interpoler de dimension (zlati,zloni,time) 
   : Return bigtabf : Interpolated field (ntimes, 96,96)
   """
   rearth = 6356752.3142   # radius of the earth (m) at the Equator
   pi = math.pi
   nloni = zloni.shape[0]
   nlati = zlati.shape[0]
   if numpy.size(time) ==1: 
    ntime=1
    bigtabi=numpy.expand_dims(bigtabi,axis=0)
   else:
    ntime = time.shape[0]
   if zlati[0] > zlati[1] :
     latreversei = 1
     zlati=zlati[::-1]
   # calculate box corners
   latlowrigi = copy.copy( zlati )
   lonlowrigi = copy.copy( zloni )
   latuplefti = copy.copy( zlati )
   lonuplefti = copy.copy( zloni )
   latlowrigi[0] = -90.
   for i in range( nlati-1 ): 
     latuplefti[i] = (zlati[i] + zlati[i+1])/2.
     latlowrigi[i+1] = latuplefti[i]
   latuplefti[nlati-1] = 90.
   lonuplefti[0] = -180.
   for i in range( nloni-1 ): 
     lonlowrigi[i] = (zloni[i] + zloni[i+1])/2.
     lonuplefti[i+1] = lonlowrigi[i]
   lonlowrigi[nloni-1] = 180.
   latreversei = 0
   # Define final horizontal grid
   nlonf = 97
   zlonf = numpy.zeros(nlonf)
   for i in range(nlonf): zlonf[i] = -180. + i*3.75
   nlatf = 96
   zlatf = numpy.zeros(nlatf)
   for i in range(nlatf): zlatf[i] = 90.0 - i*180./95.

   # Area of the final grid (to be consistent with LMDZ)
   airefile = '/home/surface1/mremaud/CO2/LMDZREF/start-96-L39.nc'
   f=netcdf.netcdf_file(airefile,'r')
   areaf=f.variables['aire'][:]
   f.close()

   latreversef = 0
   if zlatf[0] > zlatf[1] :
    latreversef = 1
    zlatf = zlatf[::-1]

   # calculate box corners
   latlowrigf = copy.copy( zlatf )
   lonlowrigf = copy.copy( zlonf )
   latupleftf = copy.copy( zlatf )
   lonupleftf = copy.copy( zlonf )
   latlowrigf[0] = -90.
   for i in range( nlatf-1 ): 
    latupleftf[i] = (zlatf[i] + zlatf[i+1])/2.
    latlowrigf[i+1] = latupleftf[i]
   latupleftf[nlatf-1] = 90.
   lonupleftf[0] = -180.
   for i in range( nlonf-1 ): 
    lonlowrigf[i] = (zlonf[i] + zlonf[i+1])/2.
    lonupleftf[i+1] = lonlowrigf[i]
   lonlowrigf[nlonf-1] = 180.

   ilonmin = numpy.zeros( nlonf, numpy.int )
   ilonmax = numpy.zeros( nlonf, numpy.int )
   ilatmin = numpy.zeros( nlatf, numpy.int )
   ilatmax = numpy.zeros( nlatf, numpy.int )

# For each box in the final grid
#   select the boxes in the initial grid that spread over it
# -> longitude local limits
   for ilon in range( nlonf ):
    ilonmin[ilon] = 0
    ilonmax[ilon] = nloni -1
    for j in range( nloni ):
      if ilonmin[ilon] == 0 and \
        lonuplefti[j] <= lonupleftf[ilon] and \
        lonlowrigi[j] > lonupleftf[ilon]: ilonmin[ilon] = j
      if ilonmax[ilon] == nloni -1 and \
        lonuplefti[j] < lonlowrigf[ilon] and \
        lonlowrigi[j] >= lonlowrigf[ilon]: ilonmax[ilon] = j

  # -> latitude local limits
   for ilat in range( nlatf ):
    ilatmin[ilat] = 0
    ilatmax[ilat] = nlati -1
    for j in range( nlati ):
      if ilatmin[ilat] == 0 and \
        latlowrigi[j] <= latlowrigf[ilat] and \
        latuplefti[j] > latlowrigf[ilat]: ilatmin[ilat] = j
      if ilatmax[ilat] == nlati -1 and \
        latlowrigi[j] < latupleftf[ilat] and \
        latuplefti[j] >= latupleftf[ilat]: ilatmax[ilat] = j
    if ilatmin[ilat] == 0 and ilatmax[ilat] == nlati -1: ilatmax[ilat] = -1

   # Loop over the final grid points
   print('Initialisation done: start interpolation')
   bigtabf = numpy.zeros( (ntime,nlatf, nlonf) )
   fieldi  = numpy.zeros( (nlati, nloni) )
   for itime in range(ntime):
    # normalize with the box area
    for i in range (nlati):
      dy = (latuplefti[i]-latlowrigi[i])*pi/180. *rearth
      for j in range (nloni):
        dx = (lonlowrigi[j]-lonuplefti[j])*pi/180. *rearth * math.cos( zlati[i] * pi/180. )
        fieldi[i,j] = bigtabi[itime,i,j] * dx * dy
    fullweight = 0.
    if latreversei: fieldi = fieldi[::-1,:]

    fieldf  = numpy.zeros( (nlatf, nlonf) )
    for ilat in range( nlatf ):
      for ilon in range( nlonf ):
        totweight = 0.
        nb = 0
        # Loop over the selected initial grid points
        for i in range(ilatmin[ilat],ilatmax[ilat]+1): 
          dlat = min(latuplefti[i],latupleftf[ilat]) - max(latlowrigi[i],latlowrigf[ilat])
          fraclat = dlat / ( latuplefti[i] - latlowrigi[i] )
          for j in range(ilonmin[ilon],ilonmax[ilon]+1): 
            dlon = \
              min(lonlowrigi[j],lonlowrigf[ilon]) - max(lonuplefti[j],lonupleftf[ilon])
            fraclon = dlon / ( lonlowrigi[j] - lonuplefti[j] )
            fieldf[ilat,ilon] += fraclon * fraclat * fieldi[i,j] 
            totweight += fraclon * fraclat
            nb += 1
        if totweight == 0.: fieldf[ilat,ilon] = 0.
        fullweight += totweight

    # extra longitude box
    fieldf[:,0] += fieldf[:,-1]
    fieldf[:,-1] = fieldf[:,0]

    # Check global total
    if 1:
      toti = sum(sum(fieldi))
      totf = sum(sum(fieldf[:,:-1]))
      fieldf[:,:] = fieldf[:,:] / areaf[:,:]
      print("%23.17e"%toti, "%23.17e"%totf)
      zacc = 1.
      while (1 + zacc) > 1. : zacc = zacc /2
      zval = abs(toti-totf) /zacc /toti
      print(' The difference is '+str( abs(zval) )+' times the zero of the machine')

   # if latreversef: fieldf = fieldf[::-1,:]
    bigtabf[itime,:,:]  = copy.copy(fieldf[:,:])

   # save final field
   if latreversef: zlatf = zlatf[::-1]
   return bigtabf,zlonf,zlatf
