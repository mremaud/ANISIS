#*******************************************************************************
# PROGRAMME     : POSTERIOR
# AUTEUR        : M REMAUD
# CREATION      : 09/2019
# COMPILATEUR   : PYTHON
# COS DATA ASSIMilATION
#
# Posterior informations on the observation and parameters
#
#*******************************************************************************
from .build_C_m import *
from .define_H import *

import os
import numpy as np
from numpy.linalg import inv
import copy
import io
import importlib
import matplotlib.pyplot as plt
import sys

name_config=sys.argv[1]
mdl=importlib.import_module(name_config)
module_dict = mdl.__dict__
try:
    to_import = mdl.__all__
except AttributeError:
    to_import = [name for name in module_dict if not name.startswith('_')]
globals().update({name: module_dict[name] for name in to_import})


# ==============================================================================
# 1. Determine the posterior uncertainties on parameters
# 2. Determine the n+1 estimation of the parameter set according to Tarantola
#    (p195 - eq 4.19)
#
# The algorithm compute the vector of estimated parameters m as well as the
# posterior covariance matrix on parameters according to Tarantola (eq 4.26a & 4.26b) :
#
# The posterior covariance matrix on parameters is defined as B' = [B^-1 + G^t*R^-1+G ]^-1
# with B : the prior covariance matrix on the parameters
#      R : the prior covariance matrix on observations
#      G : the jacobian matrix of the model at the solution
#
# ------------------------------------------------------------------------------

def AnalyticInv(rep_da,obs_vec,index_ctl_vec,index_unopt,sig_B,sigma_t,only_season):

    """
    Arguments: 
     obs_vec (pd object): observation vector 
     index_ctl_vec (pd object) : reference of the control vector that contains also the prior vector 
     index_unopt (pd object) :  reference of the flux that are not optimized
     sig_B (dictionnary) : variance covariance of the prior
     sigma_t (dictionnary): temporal correlation of the prior fluxes
    Loads:
     matrix_H.npy: numpy object, file containing inside the jacobian matrix, calculated in define_H.py
     sim.npy: numpy object; file containing the simulated values, produced in define_H.py
    Compute:
     Cm: Prior error covariance matrix
    Return: 
     - x_opt: optimized fluxes
     - Bpost: posterior error matrix (dimension of the optimized fluxes)
    """

    prec=1. #0**6

    mdiag=0 if sigma_t == None else 1
    if mdiag==0: print("B est diagonale")
    nobs=len(obs_vec)
    obs=np.double(obs_vec.obs.values)*prec
    n_param=len(index_ctl_vec)
    sig_R=np.double(obs_vec.sig_O.values )*prec*prec              #Au carree
    sim=np.load(mdl.storedir+'sim_0.npy')    #matrix_g*x
    Jac=np.load(mdl.storedir+'matrix_G.npy') 
    sim*=prec
    n_param=len(index_ctl_vec.prior)
    prior_vector=np.ones(n_param)
    prior_vector=np.double(prior_vector)*prec

    if sigma_t is not None:
     invB=build_C_m( index_ctl_vec,sig_B, sigma_t,True,rep_da)
    else: 
     invB=np.eye(n_param)*1./(sig_B*prec*prec)
    invR=np.eye(nobs)/np.copy(sig_R)
    post_B=np.matmul(np.transpose(np.copy(Jac)),invR)
    post_B=np.matmul(post_B,Jac)
    post_B+=np.copy(invB)
    post_B=inv(post_B)
    #Store the posterior error matrix
    np.save(mdl.storedir+rep_da+"A",post_B)
    delta=obs-sim
    x_opt=np.transpose(np.copy(Jac))
    x_opt=np.matmul(x_opt,invR)
    x_opt=np.dot(x_opt,np.copy(delta))
    x_opt=np.dot(np.copy(post_B),x_opt)
    x_opt+=np.copy(prior_vector)

    x_opt/=prec
    #Computation of the transported optimized flux
    sim_opt=delta_sim(obs_vec,np.append(x_opt,np.ones(len(index_unopt))  ),only_season)
 

    #Computation of the cost function aprior and aposteriori 
    #And their gradients
    J1,Jo,Jf=cost_function(prior_vector,prior_vector,obs,sim,invB,invR)
    print("Cost function before optimization:",round(J1,2))
    J2,Jo,Jf=cost_function(x_opt,prior_vector,obs,sim_opt,invB,invR)
    print("Cost function after optimization:",round(J2,2))
    grad1=gradient_function(prior_vector,prior_vector,obs,sim,invB,invR,Jac)
    print("Gradient of the cost function before:",round(np.linalg.norm(grad1),2))
    grad2=gradient_function(x_opt,prior_vector,obs,sim_opt,invB,invR,Jac)
    print("Gradient of the cost function after:",round(np.linalg.norm(grad2),2))

    #CHI 2 test
    Chi=2*J2/nobs; Chi_o=2*Jo/nobs; Chi_f=2*Jf/nobs
    print("Statistical chi-square:",round(Chi,2),str(round(Chi_o,2)),str(round(Chi_f,2)))
    with open(mdl.storedir+rep_da+'output.txt', 'a') as outfile:
     outfile.write("Cost function before optimization : "+str(round(J1,2))+"\n")
     outfile.write("Cost function after optimization : "+str(round(J2,2))+"\n")
     outfile.write("Gradient of the cost function before:"+str(round(np.linalg.norm(grad1),2))+"\n")
     outfile.write("Gradient of the cost function after:"+str(round(np.linalg.norm(grad2),2))+"\n")
     outfile.write("Statistical chi-square:"+str(round(Chi,2))+", "+str(round(Chi_o,2))+", "+str(round(Chi_f,2))+"\n")
    chi=chi_stat(obs,sim_opt,sig_R,obs_vec)
    return x_opt,post_B,chi    



def cost_function(x,xB,y,y_sim,invB,invR):
    """
     J(x)=1/2[(x-xb)TxinvBx(x-xb) +(y-Hx)TR-1(y-Hx)]
    """  
    ###Flux cost function
    J1=np.matmul(np.transpose(x-xB),invB)
    J1=np.matmul(J1,(x-xB))
    ###Observation cost function
    J2=np.matmul(np.transpose(y-y_sim),invR)
    J2=np.matmul(J2,(y-y_sim))
    J=1./2.*(J1+J2)
    return J,J2,J1

def chi_stat(y,y_sim,sig_R,obs_vec):
    """
     chi=2/Nobs*[(y-Hx)TR-1(y-Hx)]
     Observation cost function
    """
    chi={"station":[],"chi":[],"compound":[]}
    for cc in ["COS","CO2"]:
     for stat in obs_vec.stat.unique():
      if obs_vec[(obs_vec.stat==stat)&(obs_vec["compound"]==cc)].empty: continue
      index_stat=obs_vec[(obs_vec.stat==stat)&(obs_vec["compound"]==cc)].index
      invR=np.eye(len(index_stat))/np.copy(sig_R[index_stat])
      J2=np.matmul(np.transpose(y[index_stat]-y_sim[index_stat]),invR)
      J2=np.matmul(J2,(y[index_stat]-y_sim[index_stat]))
      J2/=len(index_stat)
      chi["station"].append(stat)
      chi["chi"].append(round(J2,1))
      chi["compound"].append(cc)
    chi=pd.DataFrame(chi)
    return chi


def gradient_function(x,xB,y,y_sim,invB,invR,H):
    """
     deltaJ(x)=np.multiply(invB,x-xb) - Ht invR(y-Hx)
    """
    J1=np.matmul(invB,(x-xB))
    J2=np.matmul(np.transpose(H),invR)
    J2=np.matmul(J2,(y-y_sim))
    J=J1-J2
    return J

