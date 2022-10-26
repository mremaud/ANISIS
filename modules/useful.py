#!/usr/bin/env python

from calendar import isleap
import numpy as np
import os
from netCDF4 import Dataset
import datetime
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sys import argv
import importlib
import xarray as xr
import six
import math

#Import the variables in the config.py
name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})

def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))


def get_veget():
    """
    Return: reg_array a map featuring the region
          veget_array: vegetation map
          area_LMDZ: area of LMDz mesh grid
    """ 
    veget_array = xr.open_dataset(mdl.file_PFT+str(begy)+'.nc',decode_times=False)
    if veget_array.lat.values[0]<veget_array.lat.values[1]:
     veget_array['lat']=np.flip(veget_array['lat'].values)
     veget_array['maxvegetfrac'].values=np.flip(veget_array['maxvegetfrac'].values,axis=2)
     veget_array['Contfrac'].values    =np.flip(veget_array['Contfrac'].values,axis=1)
     veget_array['Areas'].values    =np.flip(veget_array['Areas'].values,axis=1)
 
    veget_array['maxvegetfrac'+str(begy)]=veget_array['maxvegetfrac']
    for yy in range(begy+1,endy+1):
      tmp = xr.open_dataset(mdl.file_PFT+str(yy)+'.nc',decode_times=False)
      if tmp.lat.values[0]<tmp.lat.values[1]:
       tmp['maxvegetfrac'].values=np.flip(tmp['maxvegetfrac'].values,axis=2)
      veget_array['maxvegetfrac'+str(yy)]=(('time_counter','veget','lat', 'lon'), tmp['maxvegetfrac'].values)
    return veget_array


def get_pftsif(i_pft,year):
    """
    Argument: i_pdf (from 0 to num_pft -1)
    Use a 0.5 map for the SIF data
    Select the grid cells with a PFT fraction greater than 0.6 and greater than 0.3 for PFT 8 
    Return: the coordinates of the grid cells for each year
    """
    map_05=mdl.file_PFT_05+str(year)+'.nc'
    #Opening the netcdf file
    veget_array = xr.open_dataset(map_05,decode_times=False)
    if veget_array.lat.values[0]<veget_array.lat.values[1]:
     veget_array['lat']=np.flip(veget_array['lat'].values)
     veget_array['maxvegetfrac'].values=np.flip(veget_array['maxvegetfrac'].values,axis=2)
    #Select the PFT i_pft
    veget_array=veget_array.isel(pft=int(i_pft-1))  
    map_pft=np.squeeze(veget_array.maxvegetfrac.values)
    rows,columns=np.where(map_pft>=0.6)
    if len(map_pft[map_pft>0.6].flatten())<10:
     rows,columns=np.where(map_pft>=0.3)
    return rows,columns



def get_region(processus=None,PFT=None,reg=None):
    """ 
    Arguments (optional):
     - processus: the regions depends on the process to optimize
     - PFT      : the regions depends on the PFTs to optimize
     If any are defined, the region.csv and region.nc files are taken. 
     It happens if we  want to optimize the global flux per region
    Return: reg_array a map featuring the region
    """

    repf=mdl.storedir+"/"
    if (processus) : 
     ncfile = "region-"+processus+".nc"  
     csvfile="region-"+processus+".csv"
    else:
     #Default region file
     csvfile='region.csv'
     ncfile="region.nc"
    region=pd.read_csv(repf+csvfile)
    reg_array = xr.open_dataset(repf+ncfile,decode_times=False)
    if reg_array.lat.values[0]<reg_array.lat.values[1]:
     reg_array['lat']=np.flip(reg_array['lat'].values)
     reg_array['BASIN'].values=np.flip(reg_array['BASIN'].values,axis=0)
    if PFT : 
      reg_array=reg_array.isel(pft=int(PFT-1))
      region=region[region.PFT==PFT]
    return region,reg_array


def get_regionsif(PFT=None,reg=None):
    """ 
    Arguments (optional):
     - PFT      : the regions depends on the PFTs to optimize
     If any are defined, the region.csv and region.nc files are taken. 
     It happens if we  want to optimize the global flux per region
    Return: reg_array a map featuring the region 
    """ 
    
    repf=mdl.storedir+"/"
    region=pd.read_csv(repf+"region-Gpp.csv")
    reg_array = xr.open_dataset(repf+"region-Gpp_0.5.nc",decode_times=False)
    if reg_array.lat.values[0]<reg_array.lat.values[1]:
     reg_array['lat']=np.flip(reg_array['lat'].values)
     reg_array['BASIN'].values=np.flip(reg_array['BASIN'].values,axis=0)
    if PFT:
      reg_array=reg_array.isel(pft=int(PFT-1))
      region=region[region.PFT==PFT]
    return region,reg_array


def get_dicregion(index_g):
    """ 
    Arguments:
    - index_ctl_vec: control vector
    If any are defined, the region.csv and region.nc files are taken. 
    It happens if we  want to optimize the global flux per region
    Return: dictionnaries REG_AR reg_array a map featuring the region
          and DIC_REG with the associated coordonates. 
    """

    repf=mdl.storedir+"/"
    #REFERENCE REGION MAP
    region_ref=pd.read_csv(repf+"region.csv")
    reg_array_ref = xr.open_dataset(repf+"region.nc",decode_times=False)
    if reg_array_ref.lat.values[0]<reg_array_ref.lat.values[1]:
     reg_array_ref['lat']         =np.flip(reg_array_ref['lat'].values)
     reg_array_ref['BASIN'].values=np.flip(reg_array_ref['BASIN'].values,axis=0)
  
    list_process=index_g.parameter.unique()
    list_process=list_process[list_process !="offset"]
    DIC_REG={}; AR_REG={}
    for pp in list_process:
     index_pp=index_g[index_g.parameter==pp].copy(deep=True)
     #Opening region maps specific to each processus
     ncfile = "region-"+pp+".nc"
     csvfile= "region-"+pp+".csv"
     list_pft=index_pp.PFT.unique()
     list_reg=index_pp.REG.unique()

     if (len(list_reg) == 1)&(len(list_pft) == 1) :
      name        =pp
      AR_REG[name]=reg_array_ref
      DIC_REG[name]=region_ref
     elif (len(list_reg) == 1)&(len(list_pft) > 1) :
      for vv,veget in enumerate(index_pp.PFT.unique()):
       name        =pp+"_"+str(int(veget))
       AR_REG[name]=reg_array_ref
       DIC_REG[name]=region_ref
     elif (len(list_reg)>1)&(np.isnan(list_pft[0])):
      name    =pp
      region=pd.read_csv(repf+csvfile)
      reg_array = xr.open_dataset(repf+ncfile,decode_times=False)
      if reg_array.lat.values[0]<reg_array.lat.values[1]:
       reg_array['lat']         =np.flip(reg_array['lat'].values)
       reg_array['BASIN'].values=np.flip(reg_array['BASIN'].values,axis=0)
      AR_REG[name] = reg_array
      DIC_REG[name]= region
     elif (len(list_reg)>1)&(not np.isnan(list_pft[0])):
      reg_array = xr.open_dataset(repf+ncfile,decode_times=False)
      region=pd.read_csv(repf+csvfile)
      for vv,veget in enumerate(index_pp.PFT.unique()):
       name    =pp+"_"+str(int(veget))
       index_vv=index_pp[index_pp.PFT==veget].copy(deep=True)
       DIC_REG[name]= region[region.PFT==veget].copy(deep=True)
       AR_REG[name] = reg_array.isel(pft=int(veget-1)).copy(deep=True)
    return DIC_REG,AR_REG

def get_area():
    """Mesh grids of the LMDz model at standart resolution (96)"""
    dataset = Dataset(mdl.file_area,'r',format='NETCDF4_CLASSIC')
    area_LMDZ=np.squeeze(dataset.variables['aire'][:])[:,:-1]
    dataset.close()
    return area_LMDZ.data

#Extrapolation des courbes de footprint
def logi(x, b, c,asymptote_value):
    return 1/(1/asymptote_value+b*x**c)
def expo(x, a, c,asymptote_value):
    return a*np.exp(-c*x)+asymptote_value


def affine(x,x1,asymptote_value):
    """
    args: x=[value1,value2],x1
    Linear extrapolation written for the program define_H.py
    Extrapolate the response functions to a later date, that is 
    not covered by the climatological adjoints
    asymptote_value : value of the asymptote in time fixed 
    beforehand by running an adjoint for a very long time
    y=(asymptote_value-x1)/(x[-1]-x[0])*x+x1
    """

    y=(asymptote_value-x1)/(x[-1]-x[0])*x+x1
    return y

def strings_to_month(argument):
    """
    Interpreter
    Transform the resolution to number of months
    """
    switcher={    "M" : 12,
        "A" : np.nan, "W" : 12, "2W": 12
    }
    return switcher.get(argument, "nothing")

def strings_to_week(argument):
    """
    Interpreter
    Transform the resolution to number of weeks
    """
    switcher={    "M" : np.nan,
        "A" : np.nan, "W" : 4, "2W": 2
    }
    return switcher.get(argument, "nothing")


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

 
def  interpolate_reg(lon_lmdz,lat_lmdz,basin,lon,lat):
    """Interpolation to the nearest neighbour
    Return a higher resolution map that indicates the regions
    Basin : raw map
    lat_lmdz, lon_lmdz : raw coordinates
    lon,lat: final grid to define 
    return gridf: interpolated map"""

    gridf=np.zeros((len(lat),len(lon)))
    nlon=len(lon)
    nlat=len(lat)
    my_interpolating_function = RegularGridInterpolator((lat_lmdz, lon_lmdz), basin,method='nearest',bounds_error=False,fill_value=None)
    for ii in range(len(lon)):
     for jj in range(len(lat)):
      gridf[jj,ii]= my_interpolating_function([lat[jj],lon[ii]])
    return gridf


def fraction_an(row):
    """
    Argument: row["year","month","week"], week the num of the week within a month (row of a pandas data frame)
    Return: result (float)
    Calculation of the fraction of year before applying the ccgvu routine 
    NB: Used mainly for the observations
    """
  
    day_in_year = isleap(row['year']) and 366 or 365
    mon_in_year=isleap(row['year']) and [31,29,31,30,31,30,31,31,30,31,30,31] or [31,28,31,30,31,30,31,31,30,31,30,31]
    start_date=datetime.datetime(row['year'],1,1,0,0)
    if  np.isnan(row['week']):
     end_date = datetime.datetime(row['year'],row['month'],15)
    else:
     end_date = datetime.datetime(row['year'],row['month'],(row['week']-1)*7+3)
    result=row['year']+(end_date-start_date).total_seconds()/(day_in_year*86400.)
    return result

def fraction_an_fromdate(row):
    """
    Argument: row["year","month","week"], week the num of the week within a month (row of a pandas data frame)
    Return: result (float)
    Calculation of the fraction of year before applying the ccgvu routine 
    NB: Used mainly for the observations
    """

    day_in_year = isleap(row.date.year) and 366 or 365
    mon_in_year=isleap(row.date.year) and [31,29,31,30,31,30,31,31,30,31,30,31] or [31,28,31,30,31,30,31,31,30,31,30,31]
    start_date=datetime.datetime(row.date.year,1,1,0,0)
    result=row.date.year+(row.date-start_date).total_seconds()/(day_in_year*86400.)
    return result


def to_date(row):
    """
    Argument: row['year','month]
    To be used for a pandas dataframe, for each row
    Return date (timestamp format)
    """
    if not np.isnan(row['month']):
     date=datetime.datetime(int(row['year']),int(row['month']),15)
     if not np.isnan(row["week"]): 
      date = datetime.datetime(row['year'],row['month'],(int(row['week'])-1)*7+3)
    else:
     #Annual
     date=datetime.datetime(int(row['year']),6,15)
    return date

def decomposition_ccgv(name_compound,time_serie,path_directory):
    """
    Arguments: 
     - time serie: pandas data frame with the columns data, year, month, week
     - directory
    Time series decomposition: extraction of the trend and the smoothing curve 
    """
    time_serie['frac']=time_serie.apply(lambda row: fraction_an(row),axis=1)
    file_out=mdl.homedir+'modules/tmp-ccgvu.txt'
    os.system('rm -f '+file_out)
    u_convert=10**6 if name_compound== "COS" else 1
    time_serie['data']*=u_convert
    np.savetxt(file_out,time_serie[['frac','data']].values)
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+path_directory+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    time_serie["fit"]=np.loadtxt(file_fit)[:,5]
    time_serie["orig"]=np.loadtxt(file_fit)[:,3]
    time_serie["trend"]=np.loadtxt(file_fit)[:,6]

    time_serie['fit']/=u_convert
    time_serie['orig']/=u_convert
    time_serie['trend']/=u_convert

    time_serie["date"]=time_serie.apply(lambda row:to_date(row),axis=1)
    return time_serie

def decomposition_ccgv2(name_compound,time_serie,path_directory):
    """
    Arguments: 
     - time serie: pandas data frame with the columns data, year, month, week
     - directory
    Time series decomposition: extraction of the trend and the smoothing curve 
    """
    time_serie['frac']=time_serie.apply(lambda row: fraction_an_fromdate(row),axis=1)
    file_out=mdl.homedir+'modules/tmp-ccgvu.txt'
    os.system('rm -f '+file_out)
    u_convert=10**6 if name_compound== "COS" else 1
    time_serie['data']*=u_convert
    np.savetxt(file_out,time_serie[['frac','data']].values)
    file_fit=mdl.homedir+'modules/tmp-fit.txt'
    os.system('cd '+path_directory+'/modules ;./ccgcrv -cal -orig -short 80 -smooth -npoly 2 -nharm 4  -func -trend  -s  '+file_fit+' '+file_out)
    time_serie["fit"]=np.loadtxt(file_fit)[:,5]
    time_serie["orig"]=np.loadtxt(file_fit)[:,3]
    time_serie["trend"]=np.loadtxt(file_fit)[:,6]

    time_serie['fit']*=1./u_convert
    time_serie['orig']*=1./u_convert
    time_serie['trend']*=1./u_convert

    return time_serie



def diff_rows(df1,df2):
    """
    Arguments: df1,df2 (2 pd dataframes)
    Selection of the rows which are different between two dataframe df1 df2
    Return: index of the selected rows (array)
    """
    df=pd.concat([df1,df2])
    df=df.reset_index(drop=True)
    df_gpby=df.groupby(list(df.columns))
    idx=[x[0] for x in df_gpby.groups.values() if len(x) != 1 ]
    return sorted(idx)


def render_mpl_table(data,axes, col_width=50, row_height=17, font_size=7,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):

    """ Draw a tabular on a figure (later: include the multiindex) """

    if ax is None:
     size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
     axes.axis('off')

    mpl_table = axes.table(cellText=data.values, bbox=bbox, colLabels=data.columns,cellLoc='center', **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight="bold" ,color='w')
            cell.set_facecolor(header_color)
            cell.set_fontsize(12)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax


def to_name(row):
    
    try: 
     name=getattr(Vector_control, row['parameter']).name
    except:
     for cc in compound:
      try:
       name=getattr(getattr(Sources,cc.name),row['parameter']).__doc__
       break
      except:
       name="nan"
       continue
    return name 

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

