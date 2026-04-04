import derivatives as der
import read_File
import numpy as np
import plot_generator

f = read_File.ReadFile("G_100.test")
f.read_stream_G()   #reads the binary G file

vr = f.vr
vtheta = f.vtheta
vphi = f.vphi

rad = f.radius

dvrdr = der.rderavg(vr,rad)
dvphidphi = der.phideravg(vphi,minc=f.minc)
dvthdth = der.thetaderavg(vtheta)
theta, gau = der.gauleg(n_theta_max=f.ntheta)
div = np.zeros((f.nr, f.nphi, f.ntheta))

for i in range(f.nr):
    for j in range(f.nphi):
        for k in range(f.ntheta):
            div[i][j][k] = ( dvrdr[i][j][k] + (2/rad[i])*vr[i][j][k] 
                            + dvthdth[i][j][k]/rad[i] + vtheta[i][j][k]/(rad[i] * np.tan(theta[k]))
                            + dvphidphi[i][j][k]/(rad[i] * np.sin(theta[k])) )
            
plt = plot_generator.GeneratePlots(f)
plt.generate_meridional_plot(div,"divergence")
