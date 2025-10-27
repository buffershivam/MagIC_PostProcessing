import numpy as np
import math
import os

class ReadFile:
    def __init__(self,filename):
        self.filename = filename
        self.number = int(''.join(filter(str.isdigit, os.path.splitext(filename)[0])))
        self.ra = 0
        self.pr = 0
        self.raxi = 0
        self.sc = 0
        self.ek = 0
        self.stef = 0
        self.prmag = 0
        self.radratio = 0
        self.sigma = 0
        self.nr = 0
        self.ntheta = 0
        self.nphi = 0
        self.npI = 0
        self.minc = 0
        self.n_r_ic_max = 0

    def get_endianness(self):
        f = open(self.filename, 'rb')
        version = np.fromfile(f, np.int32, count=1)[0]
        endian = 'l'
        if abs(version) > 100:
            f.close()
            f = open(self.filename, 'rb')
            version = np.fromfile(f, '>i4', count=1)[0]
            endian = 'B'
        access = 'st'
        f.close()
        return endian, access

    # Use the below function when access = 'st'
    def read_stream_G(self):
        endian, access = self.get_endianness()
        if endian == 'B':
            prefix = '>'
        else:
            prefix = ''
        suffix = 4
        f = open(self.filename, 'rb')

        # Header
        fmt = '{}i{}'.format(prefix, suffix)
        version = np.fromfile(f, fmt, count=1)[0]
        fmt = '{}S64'.format(prefix)
        runID = np.fromfile(f, fmt, count=1)[0]
        fmt = '{}f{}'.format(prefix, suffix)
        time = np.fromfile(f, fmt, count=1)[0]


        if version > 13:
            self.ra, self.pr, self.raxi, self.sc, self.ek, self.stef, self.prmag, self.radratio, self.sigma = np.fromfile(f, fmt, count=9)
        else:
            self.ra, self.pr, self.raxi, self.sc, self.ek, self.prmag, self.radratio, self.sigma = np.fromfile(f, fmt, count=8)
            self.stef = 0.

        fmt = '{}i{}'.format(prefix, suffix)
        self.nr, self.ntheta, self.npI, self.minc, self.n_r_ic_max = np.fromfile(f, fmt, count=5)
        if self.npI == self.ntheta*2:
            self.npI = int(self.npI/self.minc)
        self.nphi = self.npI*self.minc+1

        if version > 13:
            l_heat, l_chem, l_phase, l_mag, l_press, l_cond_ic = np.fromfile(f, fmt, count=6)
        else:
            l_heat, l_chem, l_mag, l_press, l_cond_ic = np.fromfile(f, fmt, count=5)
            l_phase = False


        fmt = '{}f{}'.format(prefix, suffix)
        self.colatitude = np.fromfile(f, fmt, count=self.ntheta)
        self.radius = np.fromfile(f, fmt, count=self.nr)
        if ( l_mag != 0 and self.n_r_ic_max > 1 ):
            self.radius_ic = np.fromfile(f, fmt, count=self.n_r_ic_max)


        self.vr = np.zeros((self.npI, self.ntheta, self.nr), np.float32)
        self.vtheta = np.zeros_like(self.vr)
        self.vphi = np.zeros_like(self.vr)


        if l_heat != 0:
            self.entropy = np.zeros_like(self.vr)
        if l_chem != 0:
            self.xi = np.zeros_like(self.vr)
        if l_phase != 0:
            self.phase = np.zeros_like(self.vr)
        if l_press != 0:
            self.pre = np.zeros_like(self.vr)
        if l_mag != 0:
            self.Br = np.zeros_like(self.vr)
            self.Btheta = np.zeros_like(self.vr)
            self.Bphi = np.zeros_like(self.vr)
            if self.n_r_ic_max > 1:
                self.Br_ic  = np.zeros((self.npI, self.ntheta, self.n_r_ic_max), np.float32)
                self.Btheta_ic = np.zeros_like(self.Br_ic)
                self.Bphi_ic = np.zeros_like(self.Br_ic)

        # Outer core
        for i in range(self.nr):
            dat = np.fromfile(f, fmt, count=self.ntheta*self.npI)
            self.vr[:, :, i] = dat.reshape(self.npI, self.ntheta)
            dat = np.fromfile(f, fmt, count=self.ntheta*self.npI)
            self.vtheta[:, :, i] = dat.reshape(self.npI, self.ntheta)
            dat = np.fromfile(f, fmt, count=self.ntheta*self.npI)
            self.vphi[:, :, i] = dat.reshape(self.npI, self.ntheta)
            if l_heat != 0:
                dat = np.fromfile(f, fmt, count=self.ntheta*self.npI)
                self.entropy[:, :, i] = dat.reshape(self.npI, self.ntheta)
            if l_chem != 0:
                dat = np.fromfile(f, fmt, count=self.ntheta*self.npI)
                self.xi[:, :, i] = dat.reshape(self.npI, self.ntheta)
            if l_phase != 0:
                dat = np.fromfile(f, fmt, count=self.ntheta*self.npI)
                self.phase[:, :, i] = dat.reshape(self.npI, self.ntheta)
            if l_press != 0:
                dat = np.fromfile(f, fmt, count=self.ntheta*self.npI)
                self.pre[:, :, i] = dat.reshape(self.npI, self.ntheta)
            if l_mag != 0:
                dat = np.fromfile(f, fmt, count=self.ntheta*self.npI)
                self.Br[:, :, i] = dat.reshape(self.npI, self.ntheta)
                dat = np.fromfile(f, fmt, count=self.ntheta*self.npI)
                self.Btheta[:, :, i] = dat.reshape(self.npI, self.ntheta)
                dat = np.fromfile(f, fmt, count=self.ntheta*self.npI)
                self.Bphi[:, :, i] = dat.reshape(self.npI, self.ntheta)

        # Inner core
        if ( l_mag != 0 and self.n_r_ic_max > 1 ):
            for i in range(self.n_r_ic_max):
                dat = np.fromfile(f, fmt, count=self.ntheta*self.npI)
                self.Br_ic[:, :, i] = dat.reshape(self.npI, self.ntheta)
                dat = np.fromfile(f, fmt, count=self.ntheta*self.npI)
                self.Btheta_ic[:, :, i] = dat.reshape(self.npI, self.ntheta)
                dat = np.fromfile(f, fmt, count=self.ntheta*self.npI)
                self.Bphi_ic[:, :, i] = dat.reshape(self.npI, self.ntheta)

        f.close()

        # vel_list = [vr,vtheta,vphi]
        # n_list = [nr,ntheta,nphi]
        # r_list=[radius,radratio]
        return
    def getPotEndianness(self):
        f = open(self.filename, 'rb')
        ver = np.fromfile(f, dtype=np.int32, count=1)[0]
        if abs(ver) < 100:
            endian = 'l'
        else:
            endian = 'B'
        f.close()
        record_marker = False

        return endian, record_marker
    def read_Pot(self):#filename, endian, record_marker, precision = np.float32):
        endian, record_marker = self.getPotEndianness()
        if record_marker:
            version = 0
        else:
            version = 1

        if  (version == 0 and record_marker):
            print("oopsie")
            # infile = npfile(filename, endian=endian)

            # # Read header
            # l_max, n_r_max, n_r_ic_max, minc, lm_max = infile.fort_read(np.int32)
            # m_max = int((l_max/minc)*minc)
            # n_m_max = int(m_max/minc+1)
            # ra, ek, pr, prmag, radratio, sigma_ratio, omega_ma, omega_ic = infile.fort_read(precision)
            # time = infile.fort_read(precision)
            # dat = infile.fort_read(precision)

            # # Read radius and density
            # radius = dat[:n_r_max]
            # rho0 = dat[n_r_max:]

            # # Read fields in the outer core
            # shape = (lm_max, n_r_max)
            # pol = infile.fort_read(np.complex64, shape=shape, order='F')
            # tor = infile.fort_read(np.complex64, shape=shape, order='F') # field type is gonna be vel

            # # Read fields in the inner core
            # # if ic:
            # #     shape = (lm_max, n_r_ic_max)
            # #     pol_ic = infile.fort_read(np.complex64, shape=shape,
            # #                                     order='F')
            # #     tor_ic = infile.fort_read(np.complex64, shape=shape,
            # #                                     order='F')

            # infile.close()

        else:
            f = open(self.filename, 'rb')
            if endian == 'B':
                prefix = '>'
            else:
                prefix = '<'

            dt = np.dtype('{}i4'.format(prefix))
            version = np.fromfile(f, dtype=dt, count=1)[0]
            dt = np.dtype('{}9f4'.format(prefix))

            time, self.ra, self.pr, self.raxi, self.sc, self.prmag, self.ekman, self.radratio, self.sigma_ratio = np.fromfile(f, dtype=dt, count=1)[0]
            dt = np.dtype('{}5i4'.format(prefix))

            n_r_max, n_r_ic_max, l_max, minc, lm_max = np.fromfile(f, dtype=dt, count=1)[0]

            if version == 2:
                dt = np.dtype('{}2i4'.format(prefix))
                m_min, m_max = np.fromfile(f, dtype=dt, count=1)[0]
            dt = np.dtype('{}2f4'.format(prefix))
            omega_ic, omega_ma = np.fromfile(f, dtype=dt, count=1)[0]
            dt = np.dtype("{}{}f4".format(prefix, n_r_max))
            radius = np.fromfile(f, dtype=dt, count=1)[0]
            rho0 = np.fromfile(f, dtype=dt, count=1)[0]

            dt = np.dtype("{}({},{})c8".format(prefix, n_r_max,
                                                lm_max))
            pol = np.fromfile(f, dtype=dt, count=1)[0]
            pol = pol.T
            tor = np.fromfile(f, dtype=dt, count=1)[0] # No need to check for Field cuz we are assuming it is gonna be Vel potentials
            tor = tor.T

            # the below snippet is for inner core
            # if ic:
            #     dt = np.dtype("{}({},{})c8".format(prefix, n_r_ic_max,
            #                                         lm_max))
            #     pol_ic = np.fromfile(f, dtype=dt, count=1)[0]
            #     pol_ic = pol_ic.T
            #     tor_ic = np.fromfile(f, dtype=dt, count=1)[0]
            #     tor_ic = tor_ic.T

            f.close()

            params = [n_r_max, n_r_ic_max, l_max, m_min, m_max, minc, lm_max]
            potentials = [pol,tor]
            return params, potentials


    def get_map(lm_max, l_max, m_min, m_max, minc):
        """
        This routine determines the look-up tables to convert the indices
        (l, m) to the single index lm.
        """
        idx = np.zeros((l_max+1, m_max+1), np.int32)
        lm2l = np.zeros(lm_max, np.int16)
        lm2m = np.zeros(lm_max, np.int16)
        k = 0
        for m in range(m_min, m_max+1, minc):
            for l in range(m, l_max+1):
                idx[l, m] = k
                lm2l[k] = l
                lm2m[k] = m
                k += 1

        return idx, lm2l, lm2m
    def print_parameters(self):
        print("Physical Parameters :\n")

        phy_data = [
            ["Name", "Value", "Use"],
            ["Ra (Rayleigh Num)", self.ra, "Strength of buoyancy forces in flow"],
            ["Pr (Prandtl Num)", self.pr, "Ratio of thermal to viscous diffusion"],
            ["Ek (Ekman Num)", self.ek, "Ratio of viscous to coriolis forces"],
            ["Rad_ratio", self.radratio, "Ratio of outer radius to inner radius of shell"],
            ["Sc (Schmidt Num)", self.sc, "Ratio of momentum diffusivity to mass diffusivity"],
            ["Stefan Num", self.stef, "-"]
        ]

        # Get max column widths
        col_widths = [max(len(str(item)) for item in col) for col in zip(*phy_data)]

        # Function to print a row
        def print_row(row):
            print("| " + " | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row)) + " |")

        # Print table
        print("+-" + "-+-".join("-" * w for w in col_widths) + "-+")
        print_row(phy_data[0])  # headers
        print("+-" + "-+-".join("-" * w for w in col_widths) + "-+")
        for row in phy_data[1:]:
            print_row(row)
        print("+-" + "-+-".join("-" * w for w in col_widths) + "-+")

        print("\nSimulation Parameters :\n")

        sim_data = [
            ["Name", "Value", "Use"],
            ["N_r ", self.nr, "Number of points in radial direction"],
            ["N_phi", self.nphi, "Number of colatitude/longitude points"],
            ["N_theta", self.ntheta, "Number of latitude points"],
            ["minc", self.minc, "Azimuthal symmetry"],
            ["N_R_ic_max", self.sc, "Number of radial points in inner core"]
        ]

        # Get max column widths
        col_widths = [max(len(str(item)) for item in col) for col in zip(*sim_data)]

        # Print table
        print("+-" + "-+-".join("-" * w for w in col_widths) + "-+")
        print_row(sim_data[0])  # headers
        print("+-" + "-+-".join("-" * w for w in col_widths) + "-+")
        for row in sim_data[1:]:
            print_row(row)
        print("+-" + "-+-".join("-" * w for w in col_widths) + "-+")

        return