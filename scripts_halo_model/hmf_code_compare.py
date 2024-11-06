'''
Compare mass functions in CCL and in the pyhalomodel
'''
import matplotlib.pyplot as plt
import numpy as np

import pyhalomodel as halo
import pyccl
import hmf

from dawn.sim_toolkit import tools
from dawn.theory.power_spectrum import build_CAMB_cosmology

from scripts_halo_model.hmf_compare import _get_sigmaR

import matplotlib

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


M_edges = np.logspace(8, 16, 50)
Ms = 10**((np.log10(M_edges[1:]) + np.log10(M_edges[:-1]))/2)  # In Msun/h



# From pyhalomodel
# Takes mass in Msun/h
# hmod= halo.model(z=0, Om_m=0.272, name='Tinker et al. (2010)', Dv=330., dc=1.686, verbose=False)
hmod= halo.model(z=0, Om_m=0.272, name='Sheth & Tormen (1999)', Dv=330., dc=1.686, verbose=False)
R = hmod.Lagrangian_radius(Ms)
camb = build_CAMB_cosmology()
params = camb.Params
sigmaM = _get_sigmaR(R, 0, camb, cold=True)
dndm_pyhalo = hmod.mass_function(Ms, sigmaM) # units of dndm is h^4 Msun^{-1} Mpc^{-3}


# From CCL
# pyccl.spline_params["LOGM_SPLINE_NM"] = 512
# pyccl.spline_params['LOGM_SPLINE_DELTA'] = 0.005
# pyccl.gsl_params['INTEGRATION_SIGMAR_EPSREL'] = 1e-6
cosmo_ccl = pyccl.Cosmology(Omega_c=params.omegac, Omega_b=params.omegab, Omega_g=0, Omega_k=params.omk,
					   h=params.h, sigma8=camb.get_sigma8_0(), n_s=camb.Params.InitPower.ns, Neff=params.N_eff, m_nu=0.0,
					   w0=-1, wa=0, T_CMB=params.TCMB, transfer_function='boltzmann_camb', extra_parameters={'kmax':200.})
mass_def = pyccl.halos.massdef.MassDef('vir', rho_type='matter')
# dndlog10m_ccl = pyccl.halos.hmfunc.tinker08.MassFuncTinker08(mass_def=mass_def)(cosmo_ccl, Ms/params.h, 1)  # In Mpc^-3
dndlog10m_ccl = pyccl.halos.hmfunc.sheth99.MassFuncSheth99(mass_def=mass_def, mass_def_strict=False)(cosmo_ccl, Ms/params.h, 1)  # In Mpc^-3
dndm_ccl = dndlog10m_ccl/Ms/np.log(10)/params.h**3  # In Msun^-1 Mpc^-3 h^4


# From colossus
from colossus.cosmology import cosmology
from colossus.lss import mass_function

params_colossus = {'flat': True, 'H0': params.h*100, 'Om0': params.omegac+params.omegab, 'Ob0': params.omegab,
          'sigma8': camb.get_sigma8_0(), 'ns': camb.Params.InitPower.ns, 'Neff': params.N_eff, 'relspecies': True}
cosmo = cosmology.setCosmology('myCosmo', **params_colossus)
dndm_colossus = mass_function.massFunction(Ms, 0.0, q_out='dndlnM', mdef = 'vir', model = 'tinker08')/Ms
dndm_colossus = mass_function.massFunction(Ms, 0.0, q_out='dndlnM', mdef = 'fof', model = 'sheth99')/Ms


# Plot hmf in big panel and plot ratio in small panel
fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

ax[0].plot(Ms, dndm_pyhalo, c='r', ls='-.', label='Pyhalomodel')
ax[0].plot(Ms, dndm_ccl, c='lime', ls='--', label='CCL')


ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('M$_\mathrm{vir} [\mathrm{M}_\odot/h$]')
ax[0].set_ylabel('dn/dm $[h^4 \mathrm{M}_\odot^{-1} \mathrm{Mpc}^{-3}]$')
# ax[0].legend(title='Tinker et al. (2008)')
ax[0].legend(title='Sheth & Tormen (1999)')

ax[1].plot(Ms, dndm_pyhalo/dndm_colossus, c='r', ls='-.', label='Pyhalomodel/colossus')
ax[1].plot(Ms, dndm_ccl/dndm_colossus, c='lime', ls='-.', label='CCL/colossus')
ax[1].axhline(1, c='darkgray', ls='--')
ax[1].set_ylim(0.5, 1.5)
ax[1].set_xscale('log')
ax[1].set_ylabel('Ratio')
ax[1].set_xlabel('M$_\mathrm{vir} [\mathrm{M}_\odot/h$]')
ax[1].legend(ncol=2)

# plt.savefig('figures/hmf_code_comparison_Tinker.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/hmf_code_comparison_ST.png', dpi=300, bbox_inches='tight')