import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
import sys
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm
import plotly.graph_objects as go
from scipy.special import lpmv

def chebgrid(nr,a,b):#will look back at this if it really produces chebyshev grids
    rst=(a+b)/(b-a)
    rr=0.5*(rst+np.cos(np.pi*(1.-np.arange(nr+1.)/nr)))*(b-a)# /nr here
    return rr

def ff(i,nr):
    if i==0:
        return 0
    ff=float(nr)*0.5/np.tan(np.pi*float(i)/float(2.0*nr))

    a=i%2
    if a==0:
        ff=-ff
    return ff

def dnum(k,j,nr):
    if k==0:
        if(j==0 or j==nr):
            dnum=0.5
            a=nr%2
            if a==1:
                dnum=-dnum
            if j==0:
                dnum=1.0/3.0*float(nr*nr)+1.0/6.0
            return dnum
        dnum=0.5*(float(nr)+0.5)*((float(nr)+0.5)+(1.0/np.tan(np.pi*float(j)\
             /float(2.0*nr)))**2)+1.0/8.0-0.25/(np.sin(np.pi*float(j)/ \
             float(2*nr))**2)-0.5*float(nr*nr)
        return dnum
    dnum=ff(k+j,nr)+ff(k-j,nr)
    return dnum

def den(k,j,nr):
    if k==0:
        den=0.5*float(nr)
        a=j%2
        if a==1:
            den=-den
        if(j==0 or j==nr):
            den=1.0
        return den
    den=float(nr)*np.sin(np.pi*float(k)/float(nr))
    if(j==0 or j==nr):
        den=2.0*den
    return den

def spdel(kr,jr,nr,zl):
    if kr!=nr:
        fac=1
        k=kr
        j=jr
    else:
        fac=-1.0
        k=0.0
        j=nr-jr
    spdel=fac*dnum(k,j,nr)/den(k,j,nr)
    return -spdel*(2.0/zl)

def matder(nr,z1,z2):
    print("inside matder")
    nrp=nr+1
    w1=np.zeros((nrp,nrp),dtype=np.float64)
    zl=z2-z1
    for i in range(nrp):
        for j in range(nrp):
            w1[i,j]=spdel(i,j,nr,zl)           
    return w1

def rderavg(data,rad,exclude=False):
    r1=rad[0]
    r2=rad[-1]
    nr=data.shape[-1]
    grid=chebgrid(nr-1,r1,r2)
    tol=1e-6
    diff=abs(grid-rad).max()
    if diff>tol:
        spectral=False
        grid=rad
    else:
        spectral=True
    if spectral:
        d1=matder(nr-1,r1,r2)
        if (data.shape)==1:
            der=np.dot(d1,data)
        elif len(data.shape)==2:
            der=np.tensordot(data,d1,axes=[1,1])
        else:
            der=np.tensordot(data,d1,axes=[2,1])
    else:
        print("rederavg spectral = false")
        denom=np.roll(grid,-1)-np.roll(grid,1)
        denom[0]=grid[1]-grid[0]
        denom[-1]=grid[-1]-grid[-2]
        der=(np.roll(data,-1,axis=-1)-np.roll(data,1,axis=-1))/denom
        der[...,0]=(data[...,1]-data[...,0])/(grid[1]-grid[0])
        der[...,-1]=(data[...,-1]-data[...,-2])/(grid[-1]-grid[-2])
    return der

def area_avg(data,phi,theta):
    sum=0.0
    dtheta,dphi=theta[1]-theta[0],phi[1]-phi[0]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            sum+=data[i,j]*np.sin(theta[i])*dtheta*dphi
    sum=(sum)/(4*np.pi)
    return sum


