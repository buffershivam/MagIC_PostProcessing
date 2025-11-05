from legendre_class import *
if __name__=="__main__":
    version,params,rad_sph_params,omega,rad,rho_list,potentials=read_stream('V_lmr_1000.test_BIS')
    poloidal,toroidal=potentials[0],potentials[1]
    poloidal,toroidal=poloidal[:,::-1],toroidal[:,::-1]
    print("shape of polodal field=",poloidal.shape)
    radius=rad[0]
    l_max,minc,lm_max,m_max,m_min=rad_sph_params[2],rad_sph_params[3],rad_sph_params[4],rad_sph_params[5],rad_sph_params[6]
    print("data from the potential files:\n","l_max=",l_max,"\t minc=",minc,"\t m_max=",m_max,"\t m_min=",m_min)
    n_theta=144
    print("\n n_theta max given manually =",n_theta)
    leg=Legendre(l_max,minc,n_theta,m_max,m_min)
    print("started")
    #l_list_outer,Eltot_outer,Elphi_outer,Elthe_outer,Elrad_outer=leg._spectra(radius,'outer',poloidal,toroidal,1,None)
    #Eltot_avg,Elphi_avg,Elthe_avg,Elrad_avg=leg._spectraavgl(radius,poloidal,toroidal)
    Emtot_avg,Emphi_avg,Emthe_avg,Emrad_avg=leg._spectraavgm(radius,poloidal,toroidal)
    #data=np.column_stack((Eltot_avg,Elphi_avg,Elthe_avg,Elrad_avg))
    data=np.column_stack((Emtot_avg,Emphi_avg,Emthe_avg,Emrad_avg))
    np.savetxt("trying_Ra6_radavg_m.dat",data)
    print("stopped")
    #m_list_outer,Emtot_outer,Emphi_outer,Emthe_outer,Emrad_outer=leg._spectra(radius,'outer',poloidal,toroidal,None,1)
    #l_list_inner,Eltot_inner,Elphi_inner,Elthe_inner,Elrad_inner=leg._spectra(radius,'inner',poloidal,toroidal,1,None)
    #m_list_inner,Emtot_inner,Emphi_inner,Emthe_inner,Emrad_inner=leg._spectra(radius,'inner',poloidal,toroidal,None,1)
#print("testing main spectra")
