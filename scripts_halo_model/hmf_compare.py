'''Script for comparing halo mass function in the simulation (at z=0)
'''
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyhalomodel as halo
import pyccl

from dawn.sim_toolkit import tools
from dawn.theory.power_spectrum import build_CAMB_cosmology

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
# Axes options
matplotlib.rcParams['axes.titlesize'] = 'x-large'
matplotlib.rcParams['axes.labelsize'] = 'x-large'
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.grid'] = False
#
matplotlib.rcParams['legend.fontsize'] = 'large'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams['savefig.dpi'] = 300


def _get_sigmaR(R:np.ndarray, iz:int, CAMB_results, cold=False) -> np.ndarray:
    # From https://github.com/alexander-mead/HMcode-python/blob/main/hmcode/hmcode.py#L252
    var='delta_nonu' if cold else 'delta_tot'
    sigmaR = CAMB_results.get_sigmaR(R, z_indices=[iz], var1=var, var2=var)[0]
    return sigmaR

if __name__ == '__main__':
    MASS_FUNCTION = 'Sheth & Tormen (1999)'
    # MASS_FUNCTION  = 'Tinker et al. (2010)'

    halo_cat = joblib.load('../../magneticum-data/data/halo_catalog/Box1a/mr_bao_sub_144.pkl')
    halo_cat_dmo = joblib.load('../../magneticum-data/data/halo_catalog/Box1a/mr_dm_sub_144.pkl')
    halo_cat_box2 = joblib.load('../../magneticum-data/data/halo_catalog/Box2/hr_bao_sub_144.pkl')
    halo_cat_box2_dmo = joblib.load('../../magneticum-data/data/halo_catalog/Box2/hr_dm_sub_144.pkl')

    #----------------------------------------------------------------------------------------#
    #-------------------------1. Halo mass function from simulation--------------------------#
    #----------------------------------------------------------------------------------------#

    mf_box1a, error_box1a, edges_box1a = tools.get_hmf_from_halo_catalog(halo_cat, hmf='dndm', mr=1.3e10, boxsize=896)
    centers_box1a = 10**((edges_box1a[1:] + edges_box1a[:-1])/2)

    mf_box1a_dmo, error_box1a_dmo, edges_box1a_dmo = tools.get_hmf_from_halo_catalog(halo_cat_dmo, hmf='dndm', mr=1.3e10, boxsize=896)
    centers_box1a_dmo = 10**((edges_box1a_dmo[1:] + edges_box1a_dmo[:-1])/2)

    mf_box2, error_box2, edges_box2 = tools.get_hmf_from_halo_catalog(halo_cat_box2, hmf='dndm', mr=6.9e8, boxsize=352)
    centers_box2 = 10**((edges_box2[1:] + edges_box2[:-1])/2)

    mf_box2_dmo, error_box2_dmo, edges_box2_dmo = tools.get_hmf_from_halo_catalog(halo_cat_box2_dmo, hmf='dndm', mr=6.9e8, boxsize=352)
    centers_box2_dmo = 10**((edges_box2_dmo[1:] + edges_box2_dmo[:-1])/2)
    #----------------------------------------------------------------------------------------#
    #-------------------------2. Tinker+ 2010 Halo mass function --------------------------#
    #----------------------------------------------------------------------------------------#
    hmod= halo.model(z=0, Om_m=0.272, name=MASS_FUNCTION, Dv=330., dc=1.686, verbose=False)
    R = hmod.Lagrangian_radius(centers_box1a)
    CAMB_results = build_CAMB_cosmology()
    sigmaM = _get_sigmaR(R, 0, CAMB_results, cold=True)
    sigmaM = (sigmaM**2 - _get_sigmaR(896/(4/3*np.pi)**(1/3), 0, CAMB_results, cold=True)**2)**0.5
    hmf = hmod.mass_function(centers_box1a, sigmaM)

    R = hmod.Lagrangian_radius(centers_box2)
    sigmaM = _get_sigmaR(R, 0, CAMB_results, cold=True)
    sigmaM = (sigmaM**2- _get_sigmaR(352/(4/3*np.pi)**(1/3), 0, CAMB_results, cold=True)**2)**0.5
    hmf_box2 = hmod.mass_function(centers_box2, sigmaM)

    # camb = build_CAMB_cosmology()
    # params = camb.Params
    # cosmo_ccl = pyccl.Cosmology(Omega_c=params.omegac, Omega_b=params.omegab, Omega_g=0, Omega_k=params.omk,
    #                    h=params.h, sigma8=camb.get_sigma8_0(), n_s=camb.Params.InitPower.ns, Neff=params.N_eff, m_nu=0.0,
    #                    w0=-1, wa=0, T_CMB=params.TCMB, transfer_function='bbks')
    # mass_def = pyccl.halos.massdef.MassDef('vir', rho_type='matter')
    # dndlog10m_ccl = pyccl.halos.hmfunc.tinker10.MassFuncTinker10(mass_def=mass_def)(cosmo_ccl, centers_box1a, 1)  # In Mpc^-3
    # hmf = dndlog10m_ccl/centers_box1a/np.log(10)/params.h**3  # In Msun^-1 Mpc^-3 h^4

    # dndlog10m_ccl = pyccl.halos.hmfunc.tinker10.MassFuncTinker10(mass_def=mass_def)(cosmo_ccl, centers_box2, 1)  # In Mpc^-3
    # hmf_box2 = dndlog10m_ccl/centers_box2/np.log(10)/params.h**3  # In Msun^-1 Mpc^-3 h^4
    #----------------------------------------------------------------------------------------#
    # Plot hmf in big panel and plot ratio in small panel
    fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax[0].errorbar(centers_box1a, mf_box1a, yerr=error_box1a, ms=4, capsize=3, c='cornflowerblue', fmt='o-', label='Box1a (Med. Res.)', zorder=9)
    ax[0].errorbar(centers_box2, mf_box2, yerr=error_box2, ms=4, capsize=3, c='darkorange', fmt='o-', label='Box2 (High Res.)', zorder=9)
    ax[0].plot(centers_box1a, mf_box1a_dmo, c='cornflowerblue', ls='--', fillstyle='none', label='Box1a (Med. Res.) DMO', zorder=9)
    ax[0].plot(centers_box2, mf_box2_dmo, c='darkorange', ls='--', fillstyle='none', label='Box2 (High Res.) DMO', zorder=9)
    ax[0].plot(centers_box1a, hmf, c='k', label=MASS_FUNCTION, zorder=10)

    ax[0].set_xlim(0.5*centers_box2[0], 2*centers_box1a[-1])
    ax[0].set_xlabel('M$_\mathrm{vir} [\mathrm{M}_\odot/h$]')
    ax[0].set_ylabel('dn/dm $[h^4 \mathrm{M}_\odot^{-1} \mathrm{Mpc}^{-3}]$')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend()

    ax[1].errorbar(centers_box1a, mf_box1a/hmf, yerr=error_box1a/hmf, c='cornflowerblue')
    ax[1].errorbar(centers_box2, mf_box2/hmf_box2, yerr=error_box2/hmf_box2, c='darkorange')
    ax[1].errorbar(centers_box1a, mf_box1a_dmo/hmf, yerr=error_box1a_dmo/hmf, c='cornflowerblue', ls='--')
    ax[1].errorbar(centers_box2, mf_box2_dmo/hmf_box2, yerr=error_box2_dmo/hmf_box2, c='darkorange', ls='--')


    ax[1].axhline(1, c='darkgray', ls='--')
    ax[1].axhline(1.2, c='gray', lw=0.7, ls='--', alpha=0.3)
    ax[1].axhline(0.8, c='gray', lw=0.7, ls='--', alpha=0.3)
    # Draw 1% shaded region
    ax[1].fill_between([1e10, 1e16], 0.95, 1.05, color='lightgray', alpha=0.3)
    ax[1].text(1e11, 1.06, r'5%')
    ax[1].text(1e11, 1.21, '20%')
    ax[1].set_ylim(0.7, 1.3)
    ax[1].set_xlim(0.5*centers_box2[0], 2*centers_box1a[-1])
    ax[1].set_xscale('log')
    ax[1].set_xlabel('M$_\mathrm{vir} [\mathrm{M}_\odot/h$]')
    ax[1].set_ylabel('Ratio')
    plt.savefig(f'figures/hmf_comparison_{MASS_FUNCTION}.png', dpi=300, bbox_inches='tight')
