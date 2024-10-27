'''
Tests the impact of HMF on the predicted P(k) for a given cosmology.
'''
import matplotlib.pyplot as plt
import numpy as np

import pyccl

from dawn.theory.power_spectrum import build_CAMB_cosmology
from dawn.theory.ccl_tools import MassFuncSheth99Modified

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

#------------------------------------- 1. Load Simulation Data -------------------------------------#
Pk_magneticum_box1a = np.loadtxt('../../magneticum-data/data/Pylians/Pk_matter/Box1a/Pk_mr_bao_CIC_R1024.txt')
k = Pk_magneticum_box1a[:, 0]

#------------------------------------- 2. Setup settings -------------------------------------#
camb = build_CAMB_cosmology()
Pk_nonlin = camb.get_matter_power_interpolator(nonlinear=True)
params = camb.Params
cosmo_ccl = pyccl.Cosmology(Omega_c=params.omegac, Omega_b=params.omegab, Omega_g=0, Omega_k=params.omk,
					   h=params.h, sigma8=camb.get_sigma8_0(), n_s=camb.Params.InitPower.ns, Neff=params.N_eff, m_nu=0.0,
					   w0=-1, wa=0, T_CMB=params.TCMB, transfer_function='boltzmann_camb', extra_parameters={'kmax':200.})
mass_def = pyccl.halos.massdef.MassDef('vir', rho_type='matter')
concentration = pyccl.halos.concentration.duffy08.ConcentrationDuffy08(mass_def=mass_def)
NFW_profile = pyccl.halos.profiles.nfw.HaloProfileNFW(mass_def=mass_def, concentration=concentration)
halo_bias =  pyccl.halos.hbias.sheth99.HaloBiasSheth99(mass_def=mass_def, mass_def_strict=False)



#------------------------------------- 3. Compute Pk with default ST HMF -------------------------------------#
# First we compute the power spectrum using the default HMF
ST_mf = pyccl.halos.hmfunc.sheth99.MassFuncSheth99(mass_def=mass_def, mass_def_strict=False)
halo_model = pyccl.halos.halo_model.HMCalculator(mass_function=ST_mf, halo_bias=halo_bias, mass_def=mass_def,
												 log10M_min=8.0, log10M_max=16.0, nM=128, integration_method_M='spline')

Pk_halo_model = pyccl.halos.pk_2pt.halomod_power_spectrum(cosmo=cosmo_ccl, hmc=halo_model, prof=NFW_profile, k=k*0.704, a=1)


#------------------------------------- 4. Compute Pk with modified ST HMF -------------------------------------#
ST_mf_modified = MassFuncSheth99Modified(mass_def=mass_def, mass_def_strict=False)
halo_model2 = pyccl.halos.halo_model.HMCalculator(mass_function=ST_mf_modified, halo_bias=halo_bias, mass_def=mass_def,
												 log10M_min=8.0, log10M_max=16.0, nM=128, integration_method_M='spline')

Pk_halo_model2 = pyccl.halos.pk_2pt.halomod_power_spectrum(cosmo=cosmo_ccl, hmc=halo_model2, prof=NFW_profile, k=k*0.704, a=1)


#------------------------------------- 5. Plot the results -------------------------------------#
# Plot hmf in big panel and plot ratio in small panel

fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
ax[0].plot(k, Pk_halo_model*0.704**3, c='lime', ls='--', alpha=0.8, label='Sheth-Tormen (1999)')
ax[0].plot(k, Pk_halo_model2*0.704**3, c='r', ls='--', label='Modified Sheth-Tormen')
ax[0].scatter(k, Pk_magneticum_box1a[:, 1], c='dodgerblue', label='Magneticum')
ax[0].plot(k, Pk_nonlin.P(0, k), c='k', label='CAMB non-linear')

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('$k$ [Mpc$^{-1}$]')
ax[0].set_ylabel('$P(k)$ [Mpc$^3$]')
ax[0].legend()


# ax[1].plot(k, Pk_halo_model2/Pk_halo_model, c='r', ls='-.', label='Modified Sheth-Tormen (1999)/Sheth-Tormen (1999)')
ax[1].plot(k, Pk_magneticum_box1a[:, 1]/Pk_nonlin.P(0, k), c='k', label='CAMB non-linear/Sheth-Tormen (1999)')
ax[1].plot(k, Pk_magneticum_box1a[:, 1]/Pk_halo_model/0.704**3, c='lime', ls='--', label='Sheth-Tormen (1999)/Magneticum')
ax[1].plot(k, Pk_magneticum_box1a[:, 1]/Pk_halo_model2/0.704**3, c='r', ls='--', label='Modified Sheth-Tormen/Magneticum')

ax[1].axhline(1, c='darkgray', ls='--')
ax[1].set_ylim(0.2, 1.8)
ax[1].set_xscale('log')
ax[1].set_ylabel('$P(k)_\mathrm{Magneticum}/P(k)$')
ax[1].set_xlabel('$k$ [Mpc$^{-1}$]')
# ax[1].legend()
plt.savefig('figures/PK_hmf_comparison.png', dpi=300, bbox_inches='tight')

