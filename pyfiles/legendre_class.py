import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
import sys
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm
#import plotly.graph_objects as go
from scipy.special import lpmv


def read_stream(filename):
    f=open(filename,'rb')
    version=np.fromfile(f,np.int32,count=1)[0]#read the version
    time,Ra,Pr,Raxi,Sc,Prmag,Ek,radratio,sigma_ratio=np.fromfile(f,np.float32,count=9)#read the parameters
    print("time from potential file=",time)
    n_r_max,n_r_ic_max,l_max,minc,lm_max=np.fromfile(f,np.int32,count=5)
    m_min,m_max=np.fromfile(f,"{}2i4".format('<'),count=1)[0]
    omega_ic,omega_max=np.fromfile(f,np.float32,count=2)#rotation
    radius=np.fromfile(f,"{}{}f4".format('<',n_r_max),count=1)[0]#radius
    radius=radius[::-1]
    rho=np.fromfile(f,"{}{}f4".format('<',n_r_max),count=1)[0]#background density
    pol=np.fromfile(f,"{}({},{})c8".format('<',n_r_max,lm_max),count=1)[0]#poloidal potential
    pol=pol.T
    tor=np.fromfile(f,"{}({},{})c8".format('<',n_r_max,lm_max),count=1)[0]#toroidal potential
    tor=tor.T
    params=[time,Ra,Pr,Raxi,Sc,Prmag,Ek,radratio,sigma_ratio]
    rad_sph_params=[n_r_max,n_r_ic_max,l_max,minc,lm_max,m_max,m_min]
    omega=[omega_ic,omega_max]
    rad=[radius]
    rho_list=[rho]
    potentials=[pol,tor]
    return version,params,rad_sph_params,omega,rad,rho_list,potentials

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

class Legendre:
  def __init__(self,l_max,minc,n_theta_max,m_max,m_min):#initialize function
    self.l_max=l_max#maximum degree of expansion
    self.m_max=m_max#maximum azimuthal mode
    self.minc=minc#azimuthal symmetry
    self.n_theta_max=n_theta_max
    self.n_m_max=m_max//minc+1# Eg: m_max=96 => n_m_max=96//2+1=49
    self.n_phi_max=n_theta_max*2//2
    self.m_min=m_min#included m_min earlier we excluded this

    self.get_index,self.offsets,self.lm_max=self._build_index_map()#called all of them itself here as class objects rather than in main routine
    self.idx,self.lm2l,self.lm2m=self._LMmapping()
    self.LM_list=self._getLM(self.idx)
    self.lStart,self.lStop,self.lmOdd=self._getblocks()

    #declare arrays
    self.Plm=np.zeros((self.lm_max,self.n_theta_max//2))    #P_l^m(cos(theta))
    self.wPlm=np.zeros_like(self.Plm)   #weighted P_l^m
    self.wdPlm=np.zeros_like(self.Plm)  #weighted derivative
    self.dPlm=np.zeros_like(self.Plm)   #derivative wrt theta
    self.dPhi=np.zeros_like(self.Plm)   #derivative wrt phi
    self.sinTh=np.zeros(n_theta_max)  #sin(theta) values
    self.gauss=np.zeros(n_theta_max)  #gauss-legendre polynomial

    #theta and weights
    self.theta_ord,self.gauss=self._gauleg(self.n_theta_max)

    dpi=np.pi

    for ntheta in range(self.n_theta_max//2):
      colat=self.theta_ord[ntheta]
      plma,dtheta_plma=self._plm_theta(colat)
      lm=0
      for m in range(0,self.m_max+1,self.minc):
        for l in range(m,self.l_max+1):
          self.Plm[lm,ntheta]=((-1.0)**m)*plma[lm]
          self.dPlm[lm,ntheta]=((-1.0)**m)*dtheta_plma[lm]/(np.sin(colat))
          self.dPhi[lm,ntheta]=m/(np.sin(colat))
          self.wPlm[lm,ntheta]=2.0*dpi*self.gauss[ntheta]*self.Plm[lm,ntheta]
          self.wdPlm[lm,ntheta]=2.0*dpi*self.gauss[ntheta]*self.dPlm[lm,ntheta]
          lm+=1


  def _build_index_map(self):#previously it took m_max and minc as arguments
    m_values=list(range(0,self.m_max+1,self.minc))
    offsets={}
    idx=0
    for m in m_values:
      offsets[m]=idx
      idx+=(self.l_max-m+1)
    total=idx
    def get_index(m,l):
      return offsets[m]+(l-m)
    return get_index,offsets,total

  def _LMmapping(self):
    idx=np.zeros((self.l_max+1,self.m_max+1),np.int32)
    lm2l=np.zeros(self.lm_max,np.int16)
    lm2m=np.zeros(self.lm_max,np.int16)
    k=0
    for m in range(self.m_min,self.m_max+1,self.minc):#earlier the last slot was 0 as loop was only upto m_max
      for l in range(m,self.l_max+1):
        idx[l,m]=k
        lm2l[k]=l
        lm2m[k]=m
        k+=1
    return idx,lm2l,lm2m

  def _getLM(self,idx):
    LM_list=[]
    for m in range(self.m_min,self.m_max+1,self.minc):
      for l in range(m,self.l_max+1):
        LM_list.append(idx[l,m])
    LM_list=np.array(LM_list)
    return LM_list# the last entry is lm_max-1 as python idexing starts from 0

  def _getblocks(self):#start and end index of each m block, the first block m=0 has all l and last will have only 1
    lStart=np.zeros(self.n_m_max,dtype=int)
    lStop=np.zeros(self.n_m_max,dtype=int)
    lmOdd=np.zeros(self.n_m_max,dtype=int)

    lStart[0]=0#lStart[0]=1
    lStop[0]=self.l_max#lStop[0]=l_max+1
    lmOdd[0]=(self.l_max%2==0)

    for mc in range(1,self.n_m_max):
      m=mc*self.minc
      lStart[mc]=lStop[mc-1]+1
      lStop[mc]=lStart[mc]+(self.l_max-m)
      lmOdd[mc]=((lStop[mc]-lStart[mc])%2==0)

    return lStart,lStop,lmOdd

  def _specfilter(self,lcut=None,mcut=None):
      mask=np.ones(self.lm_max,dtype=bool)
      if lcut is not None:
          mask&=(self.lm2l==lcut)
      if mcut is not None:
          mask&=(self.lm2m==mcut)
      return mask

  def _spectraavgl(self,radius,poloidal,toroidal):
      return 0

  def _spectra(self,radius,radial_level,poloidal,toroidal,lspec,mspec):
      dpoldr=np.zeros_like(poloidal)
      dpoldr=rderavg(poloidal,radius,False)
      inputLM=np.zeros_like(poloidal)
      for i in range(radius.shape[0]):
          inputLM[:,i]=poloidal[:,i]*self.lm2l*(self.lm2l+1)/radius[i]**2
      m_list,l_list=[],[]
      Eltot,Elphi,Elthe,Elrad=[],[],[],[]
      Emtot,Emphi,Emthe,Emrad=[],[],[],[]
      if radial_level=='outer':
          spec_inputLM,spec_dpoldr,spec_toroidal=inputLM[:,-1],dpoldr[:,-1],toroidal[:,-1]
      if radial_level=='inner':
          spec_inputLM,spec_dpoldr,spec_toroidal=inputLM[:,0],dpoldr[:,0],toroidal[:,0]

      if lspec is not None:
          for l in range(0,self.l_max):
              mask=self._specfilter(l,None)
              dpoldr_masked,toroidal_masked,inputLM_masked=spec_dpoldr*mask,spec_toroidal*mask,spec_inputLM*mask
              vt,vp=self._specspat_vec(dpoldr_masked,toroidal_masked,self.n_theta_max,self.n_phi_max)
              vt=np.fft.ifft(vt,axis=0)*self.n_phi_max
              vp=np.fft.ifft(vp,axis=0)*self.n_phi_max
              vt,vp=vt.real,vp.real
              vt,vp=vt.T,vp.T
              vr=self._specspat_scal(inputLM_masked,self.n_theta_max,self.n_phi_max)
              vr=np.fft.ifft(vr,axis=0)*self.n_phi_max
              vr=vr.real
              vr=vr.T
              vt,vp,vr=np.tile(vt,(1,2)),np.tile(vp,(1,2)),np.tile(vr,(1,2))
              nlon,nlat=vt.shape[1],vt.shape[0]
              phi_plot,theta_plot=np.linspace(0,self.minc*np.pi,nlon),np.linspace(0,np.pi,nlat)
              v2=vp**2/16+vt**2/16+vr**2
              val_out=area_avg(v2,phi_plot,theta_plot)
              Eltot.append(val_out)
              val_out=area_avg(vp**2/16,phi_plot,theta_plot)
              Elphi.append(val_out)
              val_out=area_avg(vt**2/16,phi_plot,theta_plot)
              Elthe.append(val_out)
              val_out=area_avg(vr**2,phi_plot,theta_plot)
              Elrad.append(val_out)
              l_list.append(l)
          return l_list,Eltot,Elphi,Elthe,Elrad
      if mspec is not None:
          for m in range(self.m_min,self.m_max,self.minc):
              mask=self._specfilter(None,m)
              dpoldr_masked,toroidal_masked,inputLM_masked=spec_dpoldr*mask,spec_toroidal*mask,spec_inputLM*mask
              vt,vp=self._specspat_vec(dpoldr_masked,toroidal_masked,self.n_theta_max,self.n_phi_max)
              vt=np.fft.ifft(vt,axis=0)*self.n_phi_max
              vp=np.fft.ifft(vp,axis=0)*self.n_phi_max
              vt,vp=vt.real,vp.real
              vt,vp=vt.T,vp.T
              vr=self._specspat_scal(inputLM_masked,self.n_theta_max,self.n_phi_max)
              vr=np.fft.ifft(vr,axis=0)*self.n_phi_max
              vr=vr.real
              vr=vr.T
              vt,vp,vr=np.tile(vt,(1,2)),np.tile(vp,(1,2)),np.tile(vr,(1,2))
              nlon,nlat=vt.shape[1],vt.shape[0]
              phi_plot,theta_plot=np.linspace(0,self.minc*np.pi,nlon),np.linspace(0,np.pi,nlat)
              v2=vp**2/16+vt**2/16+vr**2
              val_out=area_avg(v2,phi_plot,theta_plot)
              Emtot.append(val_out)
              val_out=area_avg(vp**2/16,phi_plot,theta_plot)
              Emphi.append(val_out)
              val_out=area_avg(vt**2/16,phi_plot,theta_plot)
              Emthe.append(val_out)
              val_out=area_avg(vr**2,phi_plot,theta_plot)
              Emrad.append(val_out)
              m_list.append(m)
          return m_list,Emtot,Emphi,Emthe,Emrad


  def _gauleg(self,n_theta_max):
    theta_ord=np.zeros(n_theta_max)
    gauss=np.zeros(n_theta_max)

    dpi=np.pi
    M=(n_theta_max+1)//2
    XXM=0.0
    XXL=1.0
    eps=3e-14

    for i in range(1,M+1):
      zz=np.cos(dpi*((i-0.25)/(n_theta_max+0.5)))
      zz1=0.0

      while abs(zz-zz1)>eps:
        p1=1.0
        p2=0.0
        for j in range(1,n_theta_max+1):
          p3=p2
          p2=p1
          p1=((2*j-1)*zz*p2-(j-1)*p3)/j
        pp=n_theta_max*(zz*p1-p2)/(zz*zz-1.0)
        zz1=zz
        zz=zz1-p1/pp

      theta_ord[i-1]=np.arccos(XXM+XXL*zz)
      theta_ord[n_theta_max-i]=np.arccos(XXM-XXL*zz)
      w=2.0*XXL/((1.0-zz**2)*pp**2)
      gauss[i-1]=w
      gauss[n_theta_max-i]=w

    return theta_ord,gauss

  def _plm_theta(self,theta):#see notes for derivation of the recurrence relation
    m_values=list(range(self.m_min,self.m_max+1,self.minc))#produces m_values and ndim_req produces
    ndim_req=sum((self.l_max-m+1) for m in m_values)#same output as previous LM routines

    plma=np.zeros(ndim_req,dtype=np.float64)
    dtheta_plma=np.zeros(ndim_req,dtype=np.float64)

    dnorm=1.0/np.sqrt(16.0*np.arctan(1.0))#need to know why this is multiplied and how it is linked to derivation
    pos=-1

    for m in m_values:
      fac=1.0
      for j in range(3,2*m+2,2):#this section computes P^m_m for recurrence to buid higher P^m_m+1,P^m_m+2....P^m_lmax
        fac*=float(j)/float(j-1)
      plm0=np.sqrt(fac)
      s=np.sin(theta)
      if abs(s)>0.0:
        plm0=plm0*((-s)**m)
      elif m!=0:
        plm0=0.0

      l=m
      pos+=1
      plma[pos]=dnorm*plm0
      plm1=0.0

      for l in range(m+1,self.l_max+1):
        plm2=plm1#for l=m second term becomes zeros and hence recurrence remains consistent
        plm1=plm0
        num1=(2*l-1)*(2*l+1)
        den1=(l-m)*(l+m)
        term1=np.cos(theta)*np.sqrt(float(num1)/float(den1))*plm1
        num2=(2*l+1)*(l+m-1)*(l-m-1)
        den2=(2*l-3)*(l-m)*(l+m)
        term2=np.sqrt(float(num2)/float(den2))*plm2 if (den2!=0) else 0
        plm0=term1-term2
        pos+=1
        plma[pos]=dnorm*plm0

      l=self.l_max+1
      plm2=plm1
      plm1=plm0
      num1=(2*l-1)*(2*l+1)
      den1=(l-m)*(l+m)
      term1=np.cos(theta)*np.sqrt(float(num1)/float(den1))*plm1
      num2=(2*l+1)*(l+m-1)*(l-m-1)
      den2=(2*l-3)*(l-m)*(l+m)
      term2=np.sqrt(float(num2)/float(den2))*plm2
      plm0=term1-term2
      dtheta_plma[pos]=dnorm*plm0

    pos=-1
    for m in m_values:
      l=m
      pos+=1
      if m<self.l_max:
        dtheta_plma[pos]=(l/np.sqrt(float(2*l+3)))*plma[pos+1]
      else:
        dtheta_plma[pos]=(l/np.sqrt(float(2*l+3)))*plma[pos]

      for l in range(m+1,self.l_max):
        pos+=1
        termA=l*np.sqrt(float((l+m+1)*(l-m+1))/float((2*l+1)*(2*l+3)))*plma[pos+1]
        termB=(l+1)*np.sqrt(float((l+m)*(l-m))/float((2*l-1)*(2*l+1)))*plma[pos-1]
        dtheta_plma[pos]=termA-termB

      if m<self.l_max:
        l=self.l_max
        pos+=1
        termA=l*np.sqrt(float((l+m+1)*(l-m+1))/float((2*l+1)*(2*l+3)))*dtheta_plma[pos]
        termB=(l+1)*np.sqrt(float((l+m)*(l-m))/float((2*l-1)*(2*l+1)))*plma[pos-1]
        dtheta_plma[pos]=termA-termB

    return plma,dtheta_plma

  def _specspat_vec(self,dpoldr_LM,tor_LM,nth,nph):
      ii=1j
      lm_max=len(dpoldr_LM)

      PlmG=np.zeros(lm_max,dtype=np.float64)
      PlmC=np.zeros(lm_max,dtype=np.float64)
      vhG=np.zeros(lm_max,dtype=np.complex128)
      vhC=np.zeros(lm_max,dtype=np.complex128)

      vt=np.zeros((nph,nth),dtype=np.complex128)
      vp=np.zeros((nph,nth),dtype=np.complex128)

      vhG[:]=dpoldr_LM-ii*tor_LM
      vhC[:]=dpoldr_LM+ii*tor_LM

      dPlm,dPhi,Plm=self.dPlm,self.dPhi,self.Plm

      nThetaNHS=0
      for nThetaN in range(nth//2):
          nThetaS=nth-nThetaN-1
          for n_m in range(self.n_m_max):
              lms=self.lStop[n_m]
              for lm in range(self.lStart[n_m],lms-1,2):
                  PlmG[lm]=dPlm[lm,nThetaNHS]-dPhi[lm,nThetaNHS]*Plm[lm,nThetaNHS]
                  PlmC[lm]=dPlm[lm,nThetaNHS]+dPhi[lm,nThetaNHS]*Plm[lm,nThetaNHS]
                  PlmG[lm+1]=dPlm[lm+1,nThetaNHS]-dPhi[lm+1,nThetaNHS]*Plm[lm+1,nThetaNHS]
                  PlmC[lm+1]=dPlm[lm+1,nThetaNHS]+dPhi[lm+1,nThetaNHS]*Plm[lm+1,nThetaNHS]
              if self.lmOdd[n_m]:
                  PlmG[lms]=dPlm[lms,nThetaNHS]-dPhi[lms,nThetaNHS]*Plm[lms,nThetaNHS]
                  PlmC[lms]=dPlm[lms,nThetaNHS]+dPhi[lms,nThetaNHS]*Plm[lms,nThetaNHS]

          for n_m in range(self.n_m_max):
              lms=self.lStop[n_m]
              vhN1,vhS1,vhN2,vhS2=0.0+0.0j,0.0+0.0j,0.0+0.0j,0.0+0.0j
              for lm in range(self.lStart[n_m],lms-1,2):
                  vhN1+=vhG[lm]*PlmG[lm]+vhG[lm+1]*PlmG[lm+1]
                  vhS1+=-vhG[lm]*PlmC[lm]+vhG[lm+1]*PlmC[lm+1]
                  vhN2+=vhC[lm]*PlmC[lm]+vhC[lm+1]*PlmC[lm+1]
                  vhS2+=-vhC[lm]*PlmG[lm]+vhC[lm+1]*PlmG[lm+1]

              if self.lmOdd[n_m]:
                  vhN1+=vhG[lms]*PlmG[lms]
                  vhS1-=vhG[lms]*PlmC[lms]
                  vhN2+=vhC[lms]*PlmC[lms]
                  vhS2-=vhC[lms]*PlmG[lms]

              vt[n_m,nThetaN]=0.5*(vhN1+vhN2)
              vt[n_m,nThetaS]=0.5*(vhS1+vhS2)
              vp[n_m,nThetaN]=-0.5*ii*(vhN1-vhN2)
              vp[n_m,nThetaS]=-0.5*ii*(vhS1-vhS2)

          nThetaNHS+=1

      if self.n_phi_max>1:
          vt[self.n_m_max:self.n_phi_max//2+1,:]=0.0
          vp[self.n_m_max:self.n_phi_max//2+1,:]=0.0
          for nThetaN in range(self.n_theta_max):
              for n_m in range(self.n_phi_max//2+1,self.n_phi_max):
                  vt[n_m,nThetaN]=np.conj(vt[self.n_phi_max-n_m,nThetaN])
                  vp[n_m,nThetaN]=np.conj(vp[self.n_phi_max-n_m,nThetaN])
      return vt,vp

  def _specspat_scal(self,pol,nth,nph):
      ii=1j
      vr=np.zeros((nph,nth),dtype=np.complex128)
      n_m_max_loc=1 if nph==1 else self.n_m_max
      Plm=self.Plm
      for nThetaN in range(nth//2):
          nThetaS=nth-nThetaN-1
          for n_m in range(self.n_m_max):
              lms=self.lStop[n_m]
              s12=0.0+0.0j
              z12=0.0+0.0j
              for lm in range(self.lStart[n_m],lms-1,2):
                  s12+=pol[lm]*Plm[lm,nThetaN]
                  z12+=pol[lm+1]*Plm[lm+1,nThetaN]
              if self.lmOdd[n_m]:
                  s12+=pol[lms]*Plm[lms,nThetaN]
              vr[n_m,nThetaN]=s12+z12
              vr[n_m,nThetaS]=s12-z12

      if nph>1:
          vr[n_m_max_loc:nph//2+1,:]=0.0
          for nThetaN in range(nth):
              for n_m in range(nph//2+1,nph):
                  vr[n_m,nThetaN]=np.conj(vr[nph-n_m,nThetaN])

      return vr
#print("testing on cluster")

