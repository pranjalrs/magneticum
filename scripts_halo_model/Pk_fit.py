'''
Script for fitting power spectra while varying
parameters of HMF and concentration-mass relation
'''

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

import corner
import emcee
import pyccl

from dawn.theory.power_spectrum import build_CAMB_cosmology
from dawn.theory.ccl_tools import MassFuncSheth99Modified, ConcentrationDuffy08Modified

class PkFit:
	def __init__(self, fit_params, use_mead=False):
		self._priors_master = {'hmf_A': [0., 0.5],
				  		'hmf_p': [-0.5, 0.5],
						'hmf_a': [0.1, 2],
						'cM_A': [2, 10],
						'cM_B': [-1, 1],
						'kstar': [0.01, 0.1],
						'alpha': [0.1, 1.5]}

		self._initial_master = {'hmf_A': 0.27,
						  		'hmf_p': -0.28,
								'hmf_a': 1.05,
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
	print(parameters)
	print(concentration.A, concentration.B)


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

#------------------------------------- 2. Setup settings -------------------------------------#
camb = build_CAMB_cosmology()
Pk_nonlin = camb.get_matter_power_interpolator(nonlinear=True)
params = camb.Params
cosmo_ccl = pyccl.Cosmology(Omega_c=params.omegac, Omega_b=params.omegab, Omega_g=0, Omega_k=params.omk,
					h=params.h, sigma8=camb.get_sigma8_0(), n_s=camb.Params.InitPower.ns, Neff=params.N_eff, m_nu=0.0,
					w0=-1, wa=0, T_CMB=params.TCMB, transfer_function='boltzmann_camb', extra_parameters={'kmax':200.})


mass_def = pyccl.halos.massdef.MassDef('vir', rho_type='matter')
# Custom classes for MCMC fit
concentration = ConcentrationDuffy08Modified(mass_def=mass_def)
hmf = MassFuncSheth99Modified(mass_def=mass_def, mass_def_strict=False)

halo_bias =  pyccl.halos.hbias.sheth99.HaloBiasSheth99(mass_def=mass_def, mass_def_strict=False)

#------------------------------------- 3. Now fit Pk -------------------------------------#
# Initial guess for the parameters
PkFit = PkFit(['hmf_A', 'hmf_p', 'hmf_a', 'cM_A', 'cM_B'], use_mead=True)
initial_parameters = PkFit.initial_guess
priors = PkFit.priors
get_kstar = PkFit.func_kstar()
get_alpha = PkFit.func_alpha()
# priors = [(2, 10), (-1, 0)]
# Run the MCMC
nwalkers = 40
ndim = len(initial_parameters)
nsteps = 1000

# Initialize the walkers
initial = np.array(initial_parameters) + 1e-3*np.random.randn(nwalkers, ndim)


# with Pool() as pool:
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(k, Pk_sim))
sampler.run_mcmc(initial, nsteps, progress=True)

walkers = sampler.get_chain(flat=False)
np.save('chain_fit_Pk.npy', walkers)
flat_chain = sampler.get_chain(discard=int(0.5*nsteps), flat=True)
# Save the chain
fig = corner.corner(flat_chain, labels=PkFit.fit_params, show_titles=True)
plt.savefig('figures/corner_fit_Pk_vary_cm.png', dpi=300, bbox_inches='tight')


#------------------------------------- 4. Plot the results -------------------------------------#
# Plot the best fit model
best_fit = np.mean(flat_chain, axis=0)
PkFit.update_params(best_fit, hmf, concentration)

NFW_profile = pyccl.halos.profiles.nfw.HaloProfileNFW(mass_def=mass_def, concentration=concentration)
halo_model = pyccl.halos.halo_model.HMCalculator(mass_function=hmf, halo_bias=halo_bias, mass_def=mass_def,
												log10M_min=8.0, log10M_max=16.0, nM=128, integration_method_M='spline')
Pk_halo_model = pyccl.halos.pk_2pt.halomod_power_spectrum(cosmo=cosmo_ccl, hmc=halo_model, prof=NFW_profile, k=k*params.h, a=1,
													suppress_1h=get_kstar, smooth_transition=get_alpha)*params.h**3

# Also get default Pk
ST_mf = pyccl.halos.hmfunc.sheth99.MassFuncSheth99(mass_def=mass_def, mass_def_strict=False)
concentration = pyccl.halos.concentration.duffy08.ConcentrationDuffy08(mass_def=mass_def)
NFW_profile = pyccl.halos.profiles.nfw.HaloProfileNFW(mass_def=mass_def, concentration=concentration)
halo_model = pyccl.halos.halo_model.HMCalculator(mass_function=ST_mf, halo_bias=halo_bias, mass_def=mass_def,
												log10M_min=8.0, log10M_max=16.0, nM=128, integration_method_M='spline')


Pk_halo_model2 = pyccl.halos.pk_2pt.halomod_power_spectrum(cosmo=cosmo_ccl, hmc=halo_model, prof=NFW_profile, k=k*params.h, a=1,
													suppress_1h=get_kstar, smooth_transition=get_alpha)*params.h**3

# Plot the Pk in one panel and Pk ratio in a smaller panel

fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

ax[0].loglog(k, Pk_sim, c='dodgerblue', label='Simulation')
# Draw shaded region to indicate error
ax[0].fill_between(k, Pk_sim - np.sqrt(variance), Pk_sim + np.sqrt(variance), color='lightgray', alpha=0.5)
ax[0].loglog(k, Pk_halo_model, c='red', ls='--', label='Best fit')
ax[0].loglog(k, Pk_halo_model2, c='lime', ls='--', label='Default')
ax[0].set_ylabel('P(k) [Mpc/h]$^3$')
ax[0].legend()

ax[1].semilogx(k, Pk_halo_model/Pk_sim, c='red', ls='--', label='Best fit')
ax[1].semilogx(k, Pk_halo_model2/Pk_sim, c='lime', ls='--', label='Default')
ax[1].axhline(1, c='gray', ls='--')
ax[1].set_xlabel('k [h/Mpc]')
ax[1].set_ylabel('Ratio')

plt.tight_layout()


plt.savefig('figures/best_fit_Pk_mead.png', dpi=300, bbox_inches='tight')