#!/usr/bin/env python
# Marine Remaud, august 2020
# Applications of the desroziers 2001 scheme to better characterize the obs and prior errors

# BUILT-IN MODULES
import Numeric
import sys
import os
import math
import string
import copy
import calendar
from Scientific.IO import NetCDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
import mpl_toolkits.basemap as basemap
from scipy.stats.stats import pearsonr
from scipy.stats.stats import chisqprob
from scipy import statsl] for l in range(nobs)],0)
from scipy.interpolate import splprep, splevea)],0)
import random


# HOME MADE MODULES

#LOAD THE DATA
dir_exp="/home/satellites4/mremaud/DA_SYSTEM/"
obs_vec=pd.read_pickle(dir_exp+"obs_vec.pkl")
index_ctl_vec=pd.read_pickle(dir_exp+"index_ctl_vec.pkl")
index_g=pd.read_pickle(dir_exp+"index_ctl_vec.pkl")
H=np.load(dir_exp+"matrix_G.npy")

n_param=len(index_ctl_vec)
nobs=len(obs_vec)
Niterations=200
Npert=1

##################################
# Coding the needed sub_routines #
##################################
def J(Xin,Yin,Rin,Bin,Hin,Kin):
 """
 Cost function
 Arguments:
  * Xin : prior fluxes
  * Yin : observation vector
  * Rin: prior error covariance
  * Hin: tranport matrix
  * Kin: Kalman gain
 Return:
  * Jo: observation part
  * Jb : prior part of the cost function
  * X opt: opt fluxes
 """
 Rinv=np.diag([1/Rin[l,l] for l in range(nobs)],0)
 Binv=np.diag([1/Bin[l,l] for l in range(n_param)],0)
 Ktmp=Kin
 Ymodtmp=np.reshape(np.dot(Hin,Xin),(nobs,1))
 Xatmp=Xin+np.dot(Ktmp,Yin-Ymodtmp)
 Yposttmp=np.reshape(np.dot(Hin,Xatmp),(nobs,1))
 Jotmp=0.5*np.dot(np.transpose(Yposttmp-Yin),np.dot(Rinv,Yposttmp-Yin))
 Jbtmp=0.5*np.dot(np.transpose(Xatmp-Xin),np.dot(Binv,Xatmp-Xin))
 return [Jotmp,Jbtmp,Xatmp]

def desroziers_RvsB(Rin,Bin,Npertin,Hin):
 """
 Defining Desroziers scheme for balancing R and B
 Args:
  * Rin: error matrix
  * Bin: prior matrix
  * Npertin: number of perturbation
  * Hin matrix transport
 Returns: 
  * Rout
  * Bout
  * Jo_pert
  * Jb_pert
  * Jo_pert/(nobs/2-tracetmp/2)
  * Jb_pert/(tracetmp/2) 
 """
 # We need Npertin perturbations 
 print("Generating the perturbations")
 #deltaBtmp=np.random.multivariate_normal([0. for k in range(n_param)],Bin,Npertin)
 #deltaB=[np.reshape(dx,(n_param,1)) for dx in deltaBtmp]
 #deltaRtmp=np.random.multivariate_normal([0. for k in range(nobs)],Rin,Npertin)
 #deltaR=[np.reshape(dy,(nobs,1)) for dy in deltaRtmp]
 deltaB=[np.reshape([0. for k in range(n_param)],(n_param,1))]
 deltaR=[np.reshape([0. for k in range(nobs)],(nobs,1))]
 Xpert=[Xb+dx for dx in deltaB]
 Ypert=[Y0+dy for dy in deltaR]

 # Calculating expectations
 Jo_pert=0.
 Jb_pert=0.
 K=np.dot(np.dot(Bin,np.transpose(Hin)), \
   np.linalg.inv(Rin+np.dot(Hin,np.dot(Bin,np.transpose(Hin)))))
 print("Calculating expectation")
 for k in range(Npertin):
  [Jotmp,Jbtmp,Xatmp]=J(Xpert[k],Ypert[k],Rin,Bin,Hin,K)
  Jo_pert+=Jotmp/Npertin
  Jb_pert+=Jbtmp/Npertin

 # Calculating theory
 HBHTtmp=np.dot(Hin,np.dot(Bin,np.transpose(Hin)))
 HKtmp=np.dot(HBHTtmp,np.linalg.inv(HBHTtmp+Rin))
 tracetmp=np.trace(HKtmp)
 
 # Preparing the output
 Rout=Rin*(Jo_pert/(nobs/2-tracetmp/2))
 Bout=Bin*(Jb_pert/(tracetmp/2))
 print(nobs/2-tracetmp/2,Jo_pert)
 print(tracetmp/2,Jb_pert)
 return Rout,Bout, Jo_pert,Jb_pert,Jo_pert/(nobs/2-tracetmp/2),Jb_pert/(tracetmp/2)

################
# Computations #
################
# First, find the balance between obs and background
print("First iterative tuning: obs vs background balance")
Ropt=R
Bopt=B
#[Jotmp,Jbtmp,Xatmp]=J(Xb,Y0,R,B,H,K)
#print Jotmp,Jbtmp
#sys.exit()
list_Rfactor=[]
list_Bfactor=[]
Rfactor=1.
Bfactor=1.
for iter in range(Niterations):
 print("\n\n\n")
 print("Iteration",iter)
 [Rtmp,Btmp,Jotmp,Jbtmp,Rfacttmp,Bfacttmp]=desroziers_RvsB(Ropt,Bopt,Npert,H)
 print(Jotmp,Jbtmp,Jotmp+Jbtmp,nobs/2)
 Ropt=Rtmp #Rfacttmp*R
 Bopt=Btmp #Bfacttmp*B
 list_Rfactor.append(Rfacttmp[0,0])
 list_Bfactor.append(Bfacttmp[0,0])
 Rfactor=Rfactor*Rfacttmp[0,0]
 Bfactor=Bfactor*Bfacttmp[0,0]
 print(Rfactor,Bfactor)
 iter+=1
 if Jotmp+Jbtmp>0.99*nobs/2:
  print("Chi-square test above 0.99. Exiting Desroziers' scheme")
  print(list_Rfactor)
  print(list_Bfactor)
  print(Rfactor,Bfactor)
  sys.exit()

# Test new CHI2
K=np.dot(np.dot(Bopt,np.transpose(H)), \
  np.linalg.inv(Ropt+np.dot(H,np.dot(Bopt,np.transpose(H)))))
[Joend,Jbend,Xaend]=J(Xb,Y0,Ropt,Bopt,H,K)
print Joend,Jbend,nobs
print "Chi-square prior=",2*(Joend+Jbend)/nobs

sys.exit()

print "Diagnostic"
print np.shape(R),np.shape(Pa),np.shape(H)
KH=np.dot(np.dot(Pa,np.dot(np.transpose(H),np.linalg.inv(R))) ,H)
HK=np.dot(H,np.dot(Pa,np.dot(np.transpose(H),np.linalg.inv(R))))
print "Jpost=",np.mean(J_pert), nobs/2 
print "Jo_post=",np.mean(Jo_pert),np.mean(Jo_approx), 0.5*(nobs-np.trace(HK))
print "Jb_post=",np.mean(Jb_pert),np.mean(Jb_approx),0.5*np.trace(KH)
print "Jo_sub at",statsubset,"=",np.mean(Jo_subset),np.mean(Jo_approxsubset),math.sqrt(np.mean(Jo_subset)/np.mean(Jo_approxsubset))
print "Chi-square post=",2*Jpost/nobs

sys.exit()

