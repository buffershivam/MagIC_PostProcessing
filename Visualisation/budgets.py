from read_File import ReadFile
import numpy as np
import math

class GenerateBudgets:
    def __init__(self, reader: ReadFile):
        self.reader = reader
    def _avg_phi(self, data):
        data_avg = np.mean(data,axis=0)
        return data_avg
    def _fluc(self, data):
        data_avg = self._avg_phi(data)
        fluc = data-data_avg[None,:,:]
        return fluc
    def _gau_leg(self, ntheta):
        theta_ord=np.zeros(ntheta)
        gauss=np.zeros(ntheta)

        dpi=np.pi
        M=(ntheta+1)//2
        XXM=0.0
        XXL=1.0
        eps=3e-14

        for i in range(1,M+1):
            zz=np.cos(dpi*((i-0.25)/(ntheta+0.5)))
            zz1=0.0

            while abs(zz-zz1)>eps:
                p1=1.0
                p2=0.0
                for j in range(1,ntheta+1):
                    p3=p2
                    p2=p1
                    p1=((2*j-1)*zz*p2-(j-1)*p3)/j
                pp=ntheta*(zz*p1-p2)/(zz*zz-1.0)
                zz1=zz
                zz=zz1-p1/pp
            theta_ord[i-1]=np.arccos(XXM+XXL*zz)
            theta_ord[ntheta-i]=np.arccos(XXM-XXL*zz)
            w=2.0*XXL/((1.0-zz**2)*pp**2)
            gauss[i-1]=w
            gauss[ntheta-i]=w

        return theta_ord,gauss
    def meanBuoyFlux(self):
        radius, nphi, minc = self.reader.radius, self.reader.nphi, self.reader.minc
        vr, entropy = self.reader.vr, self.reader.entropy
        # print("minc=",minc,"nphi=",nphi,"ntheta=",ntheta,"nr=",nr)
        ro = radius.max()
        factor = radius/ro
        Ra = self.reader.ra

        vr_primes, temp_primes = self._fluc(vr), self._fluc(entropy)

        Bflux=Ra*vr_primes*temp_primes*factor[None,None,:]
        
        # print(f"Bflux shape = {Bflux.shape}\n")
        # print(Bflux)
        Bflux_mean = self._avg_phi(Bflux)
        # print(f"Bflux_mean shape = {Bflux_mean.shape}\n")
        # print(Bflux_mean)
        Bflux_mean = np.tile(Bflux_mean, (nphi, 1, 1))
        return Bflux_mean
    def meanDiss(self):
        radius, nphi, ntheta, minc = self.reader.radius, self.reader.nphi, self.reader.ntheta, self.reader.minc
        vr, vtheta, vphi = self.reader.vr, self.reader.vtheta, self.reader.vphi
        
        theta_ord, gauss = self._gau_leg(ntheta)
        phi_ord = np.linspace(0,2*np.pi/minc,nphi//minc)

        vphi_primes, vtheta_primes, vr_primes = self._fluc(vphi), self._fluc(vtheta), self._fluc(vr)

        # vr_primes = np.tile(vr_primes, (minc, 1, 1))
        # vtheta_primes = np.tile(vtheta_primes, (minc, 1, 1))
        # vphi_primes = np.tile(vphi_primes, (minc, 1, 1))

        # print(f"vr shape = {vr.shape}\nvr_primes shape = {vr_primes.shape}\nphi_ord shape = {phi_ord.shape}")
        R,sin_theta = radius[None,None,:],np.sin(theta_ord[None,:,None])
        dur_dphi = np.gradient(vr_primes,phi_ord,axis=0)
        dur_dtheta = np.gradient(vr_primes,theta_ord,axis=1)
        # dur_dr=np.gradient(vr_primes,radius,axis=2)

        dut_dphi=np.gradient(vtheta_primes,phi_ord,axis=0)
        # dut_dtheta=np.gradient(vtheta_primes,theta_ord,axis=1)
        # dut_dr=np.gradient(vtheta_primes,radius,axis=2)

        # dup_dphi=np.gradient(vphi_primes,phi_ord,axis=0)
        # dup_dtheta=np.gradient(vphi_primes,theta_ord,axis=1)
        # dup_dr=np.gradient(vphi_primes,radius, axis=2)

        omega_r = ((np.gradient(vphi_primes*sin_theta,theta_ord,axis=1)-dut_dphi)/(R*sin_theta))
        omega_t = ((dur_dphi/sin_theta-np.gradient(R*vphi_primes,radius,axis=2))/R)
        omega_p = ((np.gradient(R*vtheta_primes,radius,axis=2)-dur_dtheta)/R)

        Diss_local = (omega_r**2+omega_t**2+omega_p**2)
        Diss_mean = np.mean(Diss_local,axis=0)
        Diss_mean = np.tile(Diss_mean, (nphi, 1, 1))
        return Diss_mean
