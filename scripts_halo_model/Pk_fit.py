'''
Script for fitting power spectra while varying
parameters of HMF and concentration-mass relation
'''

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import joblib

import corner
import emcee
import pyccl

from dawn.sim_toolkit import tools
from dawn.theory.power_spectrum import build_CAMB_cosmology
from dawn.theory.ccl_tools import MassFuncSheth99Modified, MassFuncTinker08Modified, ConcentrationDuffy08Modified

import os
os.environ["OMP_NUM_THREADS"] = "1"

class PkFit:
	def __init__(self, fit_params, use_mead=False):
		self._priors_master = {'hmf_A': [0., 0.5],
				  		'hmf_p': [-0.5, 0.5],
						'hmf_a': [0.1, 2],
						'hmf_pA0': [0.1, 0.5],
						'hmf_pa0': [1., 3.],
						'hmf_pb0': [1, 3.],
						'hmf_pc': [1.0, 3.],
						'cM_A': [2, 10],
						'cM_B': [-1, 0],
						'kstar': [0.01, 0.1],
						'alpha': [0.1, 1.5]}

		self._initial_master = {'hmf_A': 0.27, # For Sheth99
						  		'hmf_p': -0.28, # For Sheth99
								'hmf_a': 1.05, # For Sheth99
								'hmf_pA0': 0.2, # For Tinker08
								'hmf_pa0': 1.52, # For Tinker08
								'hmf_pb0': 2.25, # For Tinker08
								'hmf_pc': 1.27, # For Tinker08
								'cM_A': 7.85,
								'cM_B': -0.081,
								'kstar': 0.07,
								'alpha': 0.719}
		self.fit_params = fit_params
		self.use_mead = use_mead
		if use_mead:
			self.kstar = 0.07*params.h
			self.alpha = 0.719
		else:
			self.kstar = None
			self.alpha = None
		self.get_initial_guess(fit_params)
		self.set_prior(fit_params)

	def set_prior(self, fit_params):
		self.priors = {}
		for name in fit_params:
			self.priors[name] = self._priors_master[name]

	def get_initial_guess(self, fit_params):
		self.initial_guess = []
		for name in fit_params:
			self.initial_guess.append(self._initial_master[name])

	def update_params(self, values, hmf, concentration):
		for i, value in enumerate(values):
			name = self.fit_params[i]
			if 'hmf' in name:
				hmf.__dict__[name.split('_')[1]] = value
			elif 'cM' in name:
				concentration.__dict__[name.split('_')[1]] = value

			elif self.use_mead:
				if name == 'kstar':
					self.kstar = value
				elif name == 'alpha':
					self.alpha = value

		return hmf, concentration

	def func_kstar(self):
		if self.kstar is None:
			return None
		return lambda a: self.kstar

	def func_alpha(self):
		if self.alpha is None:
			return None
		return lambda a: self.alpha

def log_likelihood(parameters, k, Pk_data):
	'''
	Log-likelihood function for the MCMC fit
	'''

	# Check if parameters are within the prior
	for i, name in enumerate(PkFit.fit_params):
		prior = PkFit.priors[name]
		if not (prior[0] <= parameters[i] <= prior[1]):
			return -np.inf

	# Update the parameters
	PkFit.update_params(parameters, hmf, concentration)
	# print(parameters)
	# print(concentration.A, concentration.B)


	# get_kstar = lambda a: 0.07*params.h
	# get_alpha = lambda a: 0.719

	NFW_profile = pyccl.halos.profiles.nfw.HaloProfileNFW(mass_def=mass_def, concentration=concentration)
	# Update the halo model
	halo_model = pyccl.halos.halo_model.HMCalculator(mass_function=hmf, halo_bias=halo_bias, mass_def=mass_def,
													 log10M_min=8.0, log10M_max=16.0, nM=128, integration_method_M='spline')

	# Compute the power spectrum
	Pk_halo_model = pyccl.halos.pk_2pt.halomod_power_spectrum(cosmo=cosmo_ccl, hmc=halo_model, prof=NFW_profile, k=k*params.h, a=1,
														suppress_1h=get_kstar, smooth_transition=get_alpha)*params.h**3

	if np.any(Pk_halo_model <= 0) or np.any(np.isnan(Pk_halo_model)):
		return -np.inf

	# Compute the chi^2
	chi2 = np.sum((Pk_halo_model - Pk_data)**2/variance)

	return -0.5*chi2

#------------------------------------- 0. Basic settings -------------------------------------#
# Turns on smooth transition smoothing and suppress_1h as defined in HMCode-2020
use_mead = False
mass_function_name = 'sheth99' # 'tinker08'

params_sheth = ['hmf_A', 'hmf_p', 'hmf_a']
params_tinker = ['hmf_pA0', 'hmf_pa0', 'hmf_pb0', 'hmf_pc']

if mass_function_name == 'tinker08':
	fit_params = params_tinker + ['cM_A', 'cM_B']

elif mass_function_name == 'sheth99':
	fit_params = params_sheth + ['cM_A', 'cM_B']

nwalkers = 40
nsteps = 2000

base_file_name = f'{mass_function_name}'
if use_mead:
	base_file_name += '_mead'
#------------------------------------- 1. Load Simulation Data -------------------------------------#
Pk_magneticum, box_size = np.loadtxt('../../magneticum-data/data/Pylians/Pk_matter/Box1a/Pk_mr_bao_CIC_R2048.txt'), 896
# Pk_magneticum, box_size = np.loadtxt('../../magneticum-data/data/Pylians/Pk_matter/Box2/Pk_hr_bao_CIC_R1024.txt'), 352

k = Pk_magneticum[:, 0]
Pk_sim = Pk_magneticum[:, 1]
kmax = 6 # h/Mpc
Pk_sim,	k = Pk_sim[k < kmax], k[k < kmax]

delta_k = 2*np.pi/box_size
Nk = 2*np.pi * (k/delta_k)**2
variance = Pk_sim**2/Nk

# Also compute hmf from sim
halo_cat = joblib.load('../../magneticum-data/data/halo_catalog/Box1a/mr_bao_sub_144.pkl')
dndlog10m_sim, error_box1a, M_edges = tools.get_hmf_from_halo_catalog(halo_cat, hmf='dndlog10m', mr=1.3e10, boxsize=896)
Ms = 10**((M_edges[1:] + M_edges[:-1])/2)
del halo_cat
#------------------------------------- 2. Setup settings -------------------------------------#
camb = build_CAMB_cosmology()
Pk_nonlin = camb.get_matter_power_interpolator(nonlinear=True)
params = camb.Params
cosmo_ccl = pyccl.Cosmology(Omega_c=params.omegac, Omega_b=params.omegab, Omega_g=0, Omega_k=params.omk,
					h=params.h, sigma8=camb.get_sigma8_0(), n_s=camb.Params.InitPower.ns, Neff=params.N_eff, m_nu=0.0,
					w0=-1, wa=0, T_CMB=params.TCMB, transfer_function='boltzmann_camb', extra_parameters={'kmax':200.})


mass_def = pyccl.halos.massdef.MassDef('vir', rho_type='matter')
# Custom classes for MCMC fit
if mass_function_name == 'tinker08':
	hmf = MassFuncTinker08Modified(mass_def=mass_def)

elif mass_function_name == 'sheth99':
	hmf = MassFuncSheth99Modified(mass_def=mass_def, mass_def_strict=False)

concentration = ConcentrationDuffy08Modified(mass_def=mass_def)

halo_bias =  pyccl.halos.hbias.sheth99.HaloBiasSheth99(mass_def=mass_def, mass_def_strict=False)

#------------------------------------- 3. Now fit Pk -------------------------------------#
# Initial guess for the parameters
PkFit = PkFit(fit_params, use_mead=use_mead)
initial_parameters = PkFit.initial_guess
priors = PkFit.priors
get_kstar = PkFit.func_kstar()
get_alpha = PkFit.func_alpha()
# priors = [(2, 10), (-1, 0)]
# Run the MCMC

ndim = len(initial_parameters)


# Initialize the walkers
initial = np.array(initial_parameters) + 1e-3*np.random.randn(nwalkers, ndim)


# with Pool() as pool:
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(k, Pk_sim))

load_samples = True
if load_samples:
	walkers = np.load('chain_fit_Pk.npy')
	# Discard 80% and flatten chain
	flat_chain = walkers[int(0.8*nsteps):, :, :].reshape((int(0.2*nsteps)*nwalkers, ndim))


else:
	sampler.run_mcmc(initial, nsteps, progress=True)
	walkers = sampler.get_chain(flat=False)
	np.save(f'figures/Pk_fits/chain_{base_file_name}.npy', walkers)
	flat_chain = sampler.get_chain(discard=int(0.8*nsteps), flat=True)

# Save the chain
fig = corner.corner(flat_chain, labels=PkFit.fit_params, show_titles=True)
plt.savefig(f'figures/Pk_fits/corner_{base_file_name}.png', dpi=300, bbox_inches='tight')


#------------------------------------- 4. Plot the results -------------------------------------#
# Plot the best fit model
best_fit = np.mean(flat_chain, axis=0)
PkFit.update_params(best_fit, hmf, concentration)

NFW_profile = pyccl.halos.profiles.nfw.HaloProfileNFW(mass_def=mass_def, concentration=concentration)
halo_model = pyccl.halos.halo_model.HMCalculator(mass_function=hmf, halo_bias=halo_bias, mass_def=mass_def,
												log10M_min=8.0, log10M_max=16.0, nM=128, integration_method_M='spline')
Pk_halo_model = pyccl.halos.pk_2pt.halomod_power_spectrum(cosmo=cosmo_ccl, hmc=halo_model, prof=NFW_profile, k=k*params.h, a=1,
													suppress_1h=get_kstar, smooth_transition=get_alpha)*params.h**3
bf_hmf =  hmf(cosmo_ccl, Ms/params.h, 1)/params.h**3

# Also get default Pk
ST_mf = pyccl.halos.hmfunc.sheth99.MassFuncSheth99(mass_def=mass_def, mass_def_strict=False)
concentration = pyccl.halos.concentration.duffy08.ConcentrationDuffy08(mass_def=mass_def)
NFW_profile = pyccl.halos.profiles.nfw.HaloProfileNFW(mass_def=mass_def, concentration=concentration)
halo_model = pyccl.halos.halo_model.HMCalculator(mass_function=ST_mf, halo_bias=halo_bias, mass_def=mass_def,
												log10M_min=8.0, log10M_max=16.0, nM=128, integration_method_M='spline')


Pk_halo_model2 = pyccl.halos.pk_2pt.halomod_power_spectrum(cosmo=cosmo_ccl, hmc=halo_model, prof=NFW_profile, k=k*params.h, a=1,
													suppress_1h=get_kstar, smooth_transition=get_alpha)*params.h**3
default_hmf = ST_mf(cosmo_ccl, Ms/params.h, 1)/params.h**3
#------------------------------------- 4. Compute Pk with HMCode -------------------------------------#
cosmo_ccl2 = pyccl.Cosmology(Omega_c=params.omegac, Omega_b=params.omegab, Omega_g=0, Omega_k=params.omk,
					   h=params.h, sigma8=camb.get_sigma8_0(), n_s=camb.Params.InitPower.ns, Neff=params.N_eff, m_nu=0.0,
					   w0=-1, wa=0, T_CMB=params.TCMB, transfer_function='boltzmann_camb', matter_power_spectrum='camb',
                       extra_parameters={"camb": {"halofit_version": "mead2020_feedback", "HMCode_logT_AGN": 7.4, 'kmax':200.}})

Pk_hmcode = pyccl.power.nonlin_power(cosmo_ccl2, k*params.h, a=1)*0.704**3


# Plot the Pk in one panel and Pk ratio in a smaller panel

fig, ax = plt.subplots(2, 2, figsize=(15, 6), sharex=False, gridspec_kw={'height_ratios': [3, 1]})

ax[0, 0].loglog(k, Pk_sim, c='dodgerblue', label='Simulation')
# Draw shaded region to indicate error
ax[0, 0].fill_between(k, Pk_sim - np.sqrt(variance), Pk_sim + np.sqrt(variance), color='lightgray', alpha=0.5)
ax[0, 0].loglog(k, Pk_halo_model, c='red', ls='--', label='Best fit')
ax[0, 0].loglog(k, Pk_halo_model2, c='lime', ls='--', label='Default')
ax[0, 0].loglog(k, Pk_hmcode, c='k', label='HMCode 2020')
ax[0, 0].set_ylabel('P(k) [Mpc/h]$^3$')
ax[0, 0].legend()

ax[1, 0].semilogx(k, Pk_halo_model/Pk_sim, c='red', ls='--', label='Best fit/Simulation')
ax[1, 0].semilogx(k, Pk_halo_model2/Pk_sim, c='lime', ls='--', label='Default/Simulation')
ax[1, 0].semilogx(k, Pk_hmcode/Pk_sim, c='k', label='HMCode/Simulation')


ax[1, 0].fill_between(k, 1 - np.sqrt(variance)/Pk_sim, 1 + np.sqrt(variance)/Pk_sim, color='lightgray', alpha=0.5)
ax[1, 0].axhline(1, c='gray', ls='--')
ax[1, 0].set_ylim(0.8, 1.1)
ax[1, 0].set_xlabel('k [h/Mpc]')
ax[1, 0].set_ylabel('Ratio [Theory/Simulation]')


# Right Panel (HMF)
ax[0, 1].errorbar(Ms, dndlog10m_sim, yerr=error_box1a, color='cornflowerblue', fmt='o', label='Box1a')
ax[0, 1].loglog(Ms, bf_hmf, c='red', ls='--', label='Best fit')
ax[0, 1].loglog(Ms, default_hmf, c='lime', ls='--', label='Default')
ax[0, 1].set_xlabel('M$_\mathrm{vir} [\mathrm{M}_\odot/h$]')
ax[0, 1].set_ylabel('dn/dm $[h^4 \mathrm{M}_\odot^{-1} \mathrm{Mpc}^{-3}]$')
ax[0, 1].set_title('Predicted HMF')

ax[1, 1].semilogx(Ms, bf_hmf/dndlog10m_sim, c='red', ls='--', label='Best fit/Simulation')
ax[1, 1].semilogx(Ms, default_hmf/dndlog10m_sim, c='lime', ls='--', label='Default/Simulation')
ax[1, 1].axhline(1, c='gray', ls='--')
ax[1, 1].fill_between(Ms, 1 - error_box1a/dndlog10m_sim, 1 + error_box1a/dndlog10m_sim, color='lightgray', alpha=0.5)
ax[1, 1].set_xlabel('M$_\mathrm{vir} [\mathrm{M}_\odot/h$]')
ax[1, 1].set_ylabel('Ratio [Theory/Simulation]')
plt.tight_layout()


plt.savefig(f'figures/Pk_fits/Best_fit_Pk_{base_file_name}.png', dpi=300, bbox_inches='tight')