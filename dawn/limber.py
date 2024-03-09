# Performs Limber integral
# Based on Emma's code

import numpy as np
import scipy.integrate

import astropy.units as u
import astropy.constants as const
from astropy.cosmology import WMAP7
import astropy.cosmology.units as cu

h = WMAP7.h
H0 = 100*h*u.km/u.s/u.Mpc

def get_scale_factor(chi=None, z=None):
    if z is not None:
        return 1/(1 + z)
    
    elif chi is not None:
        pass

def get_chi(z):
    """Comoving distance at redshift z

    Parameters
    ----------
    z : float or array
        redshift

    Returns
    -------
    float or array
        Comoving distance in Mpc/h
    """
    ## Should be in Mpc/h
    chi = (z*cu.redshift).to(u.Mpc, cu.redshift_distance(WMAP7, kind="comoving"))*h/cu.littleh
    
    return chi

def get_weight_q_yy(z):
    """q_yy = sigma_T/(m_e * c^2) * 1/scale_factor^2

    Parameters
    ----------
    z : float
        redshift

    Returns
    -------
    float 
        _description_
    """
    return const.sigma_T/(const.m_e * const.c**2) * 1/get_scale_factor(z=z)**2


def get_integrand_yy(ell, z, Pk_interp):
    """Limber integrand: I(ell, z) = q_yy * q_yy /chi^2 * P_yy (k, z)

    Parameters
    ----------
    ell : float
        Multipole
    z : float
        redshift
    Pk_interpolator : scipy.interp1d object
        Power spectrum interpolator

    Returns
    -------
    float
        Integrand
    """
    chi = get_chi(z)
    
    q_yy = get_weight_q_yy(z)
    
    k = (ell+1/2)/chi # in Mpc/h
    Pk = Pk_interp(z, k)[0]
    
    kmin, kmax = Pk_interp.get_knots()[1].min(), Pk_interp.get_knots()[1].max()

    if np.any((k.value < kmin) | (k.value > kmax)):
        out_of_range_indices = np.where((k.value < kmin) | (k.value > kmax))[0]
        for idx in out_of_range_indices:
#             raise Exception(f'k = {k.value[idx]:.3f} h/Mpc is not in interpolator range: {kmin:.4f} - {kmax:.2f}')
            Pk[out_of_range_indices] = 0.0
    
    integrand = q_yy**2/chi**2 * Pk
    
    return integrand


def get_limber(ells, zs, Pk_interp, Pk_units=(u.eV/u.cm**3)**2*(u.Mpc/cu.littleh)**3):
    """Limber integral for electronic pressure

    Parameters
    ----------
    ells : array
        array of ell-modes
    zs : array
        array of redshifts
    Pk_interp : interpolator
        P(z, k) interpolator, typically scipy.interpolate.RectBivariateSpline
    Pk_units : astropy.units, optional
        units of the power spectrum, by default (u.eV/u.cm**3)**2*(u.Mpc/cu.littleh)**3

    Returns
    -------
    array
        Projected C_ell for given ell-modes and redshift range
    """
    integrand_z = np.zeros(shape=(len(zs), len(ells)))

    units_str = Pk_units.to_string(format="console")
    units_str = units_str.replace('\n', '\n\t\t')
    print(f'Assuming Pk in:  {units_str} \n')

    
    zmin, zmax = Pk_interp.get_knots()[0].min(), Pk_interp.get_knots()[0].max()
    if np.any((zs < zmin) | (zs > zmax)):
        out_of_range_indices = np.where((zs < zmin) | (zs > zmax))[0]
        for idx in out_of_range_indices:
            raise Exception(f'Redshift = {zs[idx]:.2f} h/Mpc is not in interpolator range: {zmin:.2f} - {zmax:.2f}')
 
    for i, zi in enumerate(zs):
        ## When integrating we will multiply by d_chi in Mpc/h
        ## This should be dimensionless
        this_integrand = get_integrand_yy(ells, zi, Pk_interp) * Pk_units * u.Mpc/cu.littleh
        this_integrand = this_integrand.to(u.dimensionless_unscaled, cu.with_H0(H0)).value
        integrand_z[i, :] = this_integrand
    
    chi = get_chi(zs).value  # Mpc/h is already absorbed above

    return np.array([scipy.integrate.trapezoid(integrand_z[:, i], chi) for i in range(len(ells))])