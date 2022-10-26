import numpy as np
import os
from netCDF4 import Dataset
import datetime
import calendar
import pandas as pd
import copy
import xarray as xr
from sys import argv
import sys
import re
import importlib

name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})






def launch_lmdz(index_g,index_ctl_vec,index_unopt,chemical,rep_da):

    filein=homedir+"INPUTF/"+"config_"+chemical+".yml"
    fileout=storedir+rep_da+"/config_"+chemical+".yml"
    filesh=storedir+rep_da+"/lmdz_"+chemical+".sh"

    #Set the list of every fluxes to be prescribed to LMDz
    index_ctl_vec=index_ctl_vec[index_ctl_vec["factor_"+chemical]!=0.]
    list_parameter=index_ctl_vec.parameter.unique()
    list_parameter=[pp for pp in list_parameter if "offset" not in pp]
    list_parameters_x="["
    for pp in list_parameter:
     list_parameters_x+=pp+","
    index_unopt=index_unopt[index_unopt["factor_"+chemical]!=0.]
    list_unopt=index_unopt.parameter.unique()
    list_unopt=[pp for pp in list_unopt if "offset" not in pp]
    for ip,pp in enumerate(list_unopt):
     list_parameters_x+=pp
     if ip != len(list_unopt)-1:
      list_parameters_x+=","
     else:
      list_parameters_x+="]"

    ####
    f=open(filein,'r')
    newtxt = f.read()
    newlines=f.readlines()
    f.close()

    #Period
    newtxt = newtxt.replace("start_year",datetime.datetime(begy,1,1).strftime("%Y-%m-%d"))
    newtxt = newtxt.replace("end_year",datetime.datetime(endy+1,1,1).strftime("%Y-%m-%d"))

    #Liste de parametres a optimiser
    newtxt = newtxt.replace("list_parameters",list_parameters_x)

    #Load the paragraph just ones
    match = re.search('(        parameter_x:.*?varname: flx_'+chemical.lower()+'.*?)$', newtxt, flags=re.MULTILINE|re.DOTALL)

    for ip,parameter in enumerate(list_parameter):
      #Directory
      dir_parameter=storedir+rep_da
      #Name of the file
      file_name="post_"+parameter+chemical+"_%Y_phy.nc"
      if ip != 0:
       #Load the paragraph just ones
       add_para=match.group()
       add_para=add_para.replace("dir_parameter_x",dir_parameter)
       add_para=add_para.replace("file_parameter_x",file_name)
       add_para=add_para.replace("parameter_x",parameter)
       f=open(fileout,'r')
       newlines=f.readlines()
       f.close()
       newlines.insert(278,"\n"+add_para+"\n")
       #Add this part several times
       f = open(fileout,'w')
       f.writelines(newlines)
       f.close()
      else:
       print(parameter)
       #Liste de parametres a optimiser
       newtxt = newtxt.replace("dir_parameter_x",dir_parameter)
       newtxt = newtxt.replace("file_parameter_x",file_name)
       newtxt = newtxt.replace("parameter_x",parameter)

       #Add this part several times
       f = open(fileout,'w')
       f.write(newtxt)
       f.close()

    for ip,parameter in enumerate(list_unopt):
     Flx=getattr(Sources,chemical)
     Flx=getattr(Flx,parameter)
     nn=len(Flx.file_name)-1
     letter=Flx.file_name[-1]
     while letter != "/":
      nn-=1
      letter=Flx.file_name[nn]

     #Directory
     dir_parameter=Flx.file_name[:nn+1]
     #Name of the file
     file_name=Flx.file_name[nn+1:]
     file_name=file_name.replace("dyn","phy")
     if Flx.sign==-1 : print("WARNING","sign")
     add_para=match.group()
     add_para=add_para.replace("dir_parameter_x",dir_parameter)
     file_name=file_name.replace("XXXX","%Y")
     add_para=add_para.replace("file_parameter_x",file_name)
     add_para=add_para.replace("parameter_x",parameter)
     f=open(fileout,'r')
     newlines=f.readlines()
     f.close()
     newlines.insert(278,"\n"+add_para)
     #Add this part several times
     f = open(fileout,'w')
     f.writelines(newlines)
     f.close()

########################END FLUXES#########################################################


####INITIAL CONDITION########
#In ppm
    IC=index_g[index_g.parameter=="offset"]["post_"+chemical].values*10**(-6)*getattr(Masse_molaire,chemical)/28.965
    IC=IC[IC!=0]
    print(IC)
    name_lon='rlonv'
    num_lon=97
    num_lat=96
    name_lat='rlatu'
    name_t='temps'
    num_lev=39
    name_lev='sigs'
    s = (1,num_lev,num_lat,num_lon)


    file_start=storedir+rep_da+'restart_'+chemical+"_"+str(begy)+".nc"
    restart='/home/satellites13/aberchet/RESTART/LMDZ/39L/lmdz5.inca.restart.an2009.m04j01.nc'
    f = Dataset(restart, 'r')
    CO2BIM=f.variables["q06"][:]
    f.close()
    os.system('rm -f  '+file_start)
    os.system('ncks -x -v q27,q28,q33,q06 '+restart+' '+file_start)
    CC=np.ones(CO2BIM.shape)*IC


    dataset = Dataset(file_start,'a',format='NETCDF3_CLASSIC')
    ATT=dataset.createVariable('q08', np.dtype('d'),(name_t,name_lev,name_lat,name_lon,))
    ATT[:,:,:,:]=CC
    dataset.close()

    f = open(fileout,'r')
    newtxt = f.read()
    f.close()
    #Liste de parametres a optimiser
    newtxt = newtxt.replace("file_init",'restart_'+chemical+"_"+str(begy)+".nc")
    newtxt = newtxt.replace("dir_init",storedir+rep_da+"/")
    newtxt = newtxt.replace("repoutput","/home/satellites16/mremaud/"+chemical+rep_da[:-1]+"_"+str(begy)+str(endy))
    f = open(fileout,'w')
    f.write(newtxt)
    f.close()

    #Write the sh file to 
    f = open(filesh,'w')
    f.write("#QSUB -s /bin/tcsh \n")
    f.write("#PBS -q long \n")
    f.write("rm -rdf /home/satellites16/mremaud/"+chemical+rep_da[:-1]+"_"+str(begy)+str(endy)+"\n")
    f.write("cd /home/users/mremaud/CIF"+"\n")
    f.write("python -m pycif "+fileout)
    f.close()



