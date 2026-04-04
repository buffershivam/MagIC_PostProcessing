import numpy as np
import scipy.interpolate as S
def gauleg(n_theta_max: int):
    """
    Compute Gauss-Legendre quadrature nodes (as co-latitudes) and weights.

    Parameters
    ----------
    n_theta_max : int
        Number of quadrature points.

    Returns
    -------
    theta_ord : ndarray, shape (n_theta_max,)
        Co-latitudinal quadrature nodes in radians, ordered from north to south.
    gauss : ndarray, shape (n_theta_max,)
        Corresponding Gauss-Legendre weights.
    """
    theta_ord = np.zeros(n_theta_max, dtype=np.float64)
    gauss     = np.zeros(n_theta_max, dtype=np.float64)

    EPS = 3.0e-14
    M   = (n_theta_max + 1) // 2

    for i in range(1, M + 1):
        # Initial guess for i-th root (Fortran 1-based → Python 0-based internally)
        zz = np.cos(np.pi * (i - 0.25) / (n_theta_max + 0.5))

        # Newton-Raphson refinement
        zz1 = 0.0
        while abs(zz - zz1) > EPS:
            p1, p2 = 1.0, 0.0
            for j in range(1, n_theta_max + 1):
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * zz * p2 - (j - 1) * p3) / j
            pp  = n_theta_max * (zz * p1 - p2) / (zz * zz - 1.0)
            zz1 = zz
            zz  = zz1 - p1 / pp

        # Store symmetric pair (convert cosine abscissa → co-latitude)
        theta_ord[i - 1]                  = np.arccos(zz)
        theta_ord[n_theta_max - i]        = np.arccos(-zz)
        gauss[i - 1]                      = 2.0 / ((1.0 - zz * zz) * pp * pp)
        gauss[n_theta_max - i]            = gauss[i - 1]

    return theta_ord, gauss
def chebgrid(nr, a, b):
    """
    This function defines a Gauss-Lobatto grid from a to b.

    >>> r_icb = 0.5 ; r_cmb = 1.5; n_r_max=65
    >>> rr = chebgrid(n_r_max, r_icb, r_cmb)

    :param nr: number of radial grid points plus one (Nr+1)
    :type nr: int
    :param a: lower limit of the Gauss-Lobatto grid
    :type a: float
    :param b: upper limit of the Gauss-Lobatto grid
    :type b: float
    :returns: the Gauss-Lobatto grid
    :rtype: numpy.ndarray
    """
    rst = (a+b)/(b-a)
    rr = 0.5*(rst+np.cos(np.pi*(1.-np.arange(nr+1.)/nr)))*(b-a)
    return rr

def matder(nr, z1, z2):
    """
    This function calculates the derivative in Chebyshev space.

    >>> r_icb = 0.5 ; r_cmb = 1.5; n_r_max=65
    >>> d1 = matder(n_r_max, r_icb, r_cmb)
    >>> # Chebyshev grid and data
    >>> rr = chebgrid(n_r_max, r_icb, r_cmb)
    >>> f = sin(rr)
    >>> # Radial derivative
    >>> df = dot(d1, f)

    :param nr: number of radial grid points
    :type nr: int
    :param z1: lower limit of the Gauss-Lobatto grid
    :type z1: float
    :param z2: upper limit of the Gauss-Lobatto grid
    :type z2: float
    :returns: a matrix of dimension (nr,nr) to calculate the derivatives
    :rtype: numpy.ndarray
    """
    nrp = nr+1
    w1 = np.zeros((nrp, nrp), dtype=np.float64)
    zl = z2-z1
    for i in range(nrp):
        for j in range(nrp):
            w1[i, j] = spdel(i, j, nr, zl)

    return w1
def spdel(kr, jr, nr, zl):
    if kr != nr :
        fac = 1.
        k = kr
        j = jr
    else:
        fac = -1.
        k = 0.
        j = nr-jr

    spdel = fac*dnum(k, j, nr)/den(k, j, nr)
    return -spdel*(2./zl)

def dnum(k, j, nr):
    if k == 0:
        if (j == 0 or j == nr):
            dnum = 0.5
            a = nr % 2
            if a == 1:
                dnum = -dnum
            if j == 0:
                dnum = 1./3.*float(nr*nr)+1./6.
            return dnum

        dnum = 0.5*(float(nr)+0.5)*((float(nr)+0.5)+(1./np.tan(np.pi*float(j) \
               /float(2.*nr)))**2)+1./8.-0.25/(np.sin(np.pi*float(j)/ \
               float(2*nr))**2) - 0.5*float(nr*nr)
        return dnum

    dnum = ff(k+j, nr)+ff(k-j, nr)
    return dnum

def ff(i, nr):
    if i == 0:
        return 0
    ff = float(nr)*0.5/np.tan(np.pi*float(i)/float(2.*nr))

    a = i % 2
    if a == 0:
        ff = -ff
    return ff

def den(k, j, nr):
    if k == 0:
        den = 0.5*float(nr)
        a = j % 2
        if a == 1:
            den = -den
        if (j == 0 or j == nr):
            den = 1.
        return den

    den = float(nr)*np.sin(np.pi*float(k)/float(nr))
    if (j == 0 or j == nr):
        den = 2.*den
    return den

def rderavg(data, rad, exclude=False):
    """
    Radial derivative of an input array

    >>> gr = MagiGraph()
    >>> dvrdr = rderavg(gr.vr, gr.radius)

    :param data: input array
    :type data: numpy.ndarray
    :param rad: radial grid
    :type rad: numpy.ndarray
    :param exclude: when set to True, exclude the first and last radial grid points
                    and replace them by a spline extrapolation (default is False)
    :type exclude: bool
    :returns: the radial derivative of the input array
    :rtype: numpy.ndarray
    """
    r1 = rad[0]
    r2 = rad[-1]
    nr = data.shape[-1]
    grid = chebgrid(nr-1, r1, r2)
    tol = 1e-6 # This is to determine whether Cheb der will be used
    diff = abs(grid-rad).max()
    if diff > tol:
        spectral = False
        grid = rad
    else:
        spectral = True

    if exclude:
        g = grid[::-1]
        gnew = np.linspace(r2, r1, 1000)
        if len(data.shape) == 2:
            for i in range(data.shape[0]):
                val = data[i, ::-1]
                tckp = S.splrep(g[1:-1], val[1:-1])
                fnew = S.splev(gnew, tckp)
                data[i, 0] = fnew[-1]
                data[i, -1] = fnew[0]
        else:
            for j in range(data.shape[0]):
                for i in range(data.shape[1]):
                    val = data[j, i, ::-1]
                    tckp = S.splrep(g[1:-1], val[1:-1])
                    fnew = S.splev(gnew, tckp)
                    data[j, i, 0] = fnew[-1]
                    data[j, i, -1] = fnew[0]
    if spectral:
        d1 = matder(nr-1, r1, r2)
        if len(data.shape) == 1:
            der = np.dot(d1, data)
        elif len(data.shape) == 2:
            der = np.tensordot(data, d1, axes=[1, 1])
        else:
            der = np.tensordot(data, d1, axes=[2, 1])
    else:
        denom = np.roll(grid, -1) - np.roll(grid, 1)
        denom[0] = grid[1]-grid[0]
        denom[-1] = grid[-1]-grid[-2]
        der = (np.roll(data, -1,  axis=-1)-np.roll(data, 1, axis=-1))/denom
        der[..., 0] = (data[..., 1]-data[..., 0])/(grid[1]-grid[0])
        der[..., -1] = (data[..., -1]-data[..., -2])/(grid[-1]-grid[-2])

    return der
def phideravg(data, minc=1, order=4):
    """
    phi-derivative of an input array

    >>> gr = MagicGraph()
    >>> dvphidp = phideravg(gr.vphi, minc=gr.minc)

    :param data: input array
    :type data: numpy.ndarray
    :param minc: azimuthal symmetry
    :type minc: int
    :param order: order of the finite-difference scheme (possible values are 2 or 4)
    :type order: int
    :returns: the phi-derivative of the input array
    :rtype: numpy.ndarray
    """
    nphi = data.shape[0]
    dphi = 2.*np.pi/minc/(nphi-1.)
    if order == 2:
        der = (np.roll(data, -1,  axis=0)-np.roll(data, 1, axis=0))/(2.*dphi)
        der[0, ...] = (data[1, ...]-data[-2, ...])/(2.*dphi)
        der[-1, ...] = der[0, ...]
    elif order == 4:
        der = (   -np.roll(data,-2,axis=0) \
               +8.*np.roll(data,-1,axis=0) \
               -8.*np.roll(data, 1,axis=0)  \
                  +np.roll(data, 2,axis=0)   )/(12.*dphi)
        der[1, ...] = (-data[3, ...]+8.*data[2, ...]-\
                       8.*data[0, ...] +data[-2, ...])/(12.*dphi)
        der[-2, ...] = (-data[0, ...]+8.*data[-1, ...]-\
                       8.*data[-3, ...]+data[-4, ...])/(12.*dphi)
        der[0, ...] = (-data[2, ...]+8.*data[1, ...]-\
                       8.*data[-2, ...] +data[-3, ...])/(12.*dphi)
        der[-1, ...] = der[0, ...]
    return der

def thetaderavg(data, order=4):
    """
    Theta-derivative of an input array (finite differences)

    >>> gr = MagiGraph()
    >>> dvtdt = thetaderavg(gr.vtheta)

    :param data: input array
    :type data: numpy.ndarray
    :param order: order of the finite-difference scheme (possible values are 2 or 4)
    :type order: int
    :returns: the theta-derivative of the input array
    :rtype: numpy.ndarray
    """
    if len(data.shape) == 3: # 3-D
        ntheta = data.shape[1]
        dtheta = np.pi/(ntheta-1.)
        if order == 2:
            der = (np.roll(data, -1,  axis=1)-np.roll(data, 1, axis=1))/(2.*dtheta)
            der[:, 0, :] = (data[:, 1, :]-data[:, 0, :])/dtheta
            der[:, -1, :] = (data[:, -1, :]-data[:, -2, :])/dtheta
        elif order == 4:
            der = (   -np.roll(data,-2,axis=1) \
                   +8.*np.roll(data,-1,axis=1) \
                   -8.*np.roll(data, 1,axis=1)  \
                      +np.roll(data, 2,axis=1)   )/(12.*dtheta)
            der[:, 1, :] = (data[:, 2, :]-data[:, 0, :])/(2.*dtheta)
            der[:, -2, :] = (data[:, -1, :]-data[:, -3, :])/(2.*dtheta)
            der[:, 0, :] = (data[:, 1, :]-data[:, 0, :])/dtheta
            der[:, -1, :] = (data[:, -1, :]-data[:, -2, :])/dtheta

    elif len(data.shape) == 2: #2-D
        ntheta = data.shape[0]
        dtheta = np.pi/(ntheta-1.)
        if order == 2:
            der = (np.roll(data, -1,  axis=0)-np.roll(data, 1, axis=0))/(2.*dtheta)
            der[0, :] = (data[1, :]-data[0, :])/dtheta
            der[-1, :] = (data[-1, :]-data[-2, :])/dtheta
        elif order == 4:
            der = (-np.roll(data,-2,axis=0)+8.*np.roll(data,-1,axis=0)-\
                  8.*np.roll(data,1,axis=0)+np.roll(data,2,axis=0))/(12.*dtheta)
            der[1, :] = (data[2, :]-data[0, :])/(2.*dtheta)
            der[-2, :] = (data[-1, :]-data[-3, :])/(2.*dtheta)
            der[0, :] = (data[1, :]-data[0, :])/dtheta
            der[-1, :] = (data[-1, :]-data[-2, :])/dtheta

    return der