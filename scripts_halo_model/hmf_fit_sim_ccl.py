import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
import corner

import emcee
import hmf
import joblib
import pyccl

from dawn.sim_toolkit import tools
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


def log_likelihood(model, data, volume, dlog10m):
	return np.sum(poisson.logpmf(data, mu=model*volume*dlog10m))



def log_prob(param_values, data, model, volume):
	model.a = param_values[0]
	model.p = param_values[1]
	model.A = param_values[2]

	if param_values[1]>0.5:
		return -np.inf#, []

	dndlog10m_ccl = model(cosmo_ccl, Ms, 1) / params.h**3

	return log_likelihood(dndlog10m_ccl, data, volume, dlog10m)#, derived

#------------------------------------- 1. Load the data -------------------------------------#
halo_cat = joblib.load('../../magneticum-data/data/halo_catalog/Box1a/mr_bao_sub_144.pkl')
counts, error_box1a, M_edges = tools.get_hmf_from_halo_catalog(halo_cat, hmf='counts', mr=1.3e10, boxsize=896)
dndlog10m_sim, error_box1a, M_edges = tools.get_hmf_from_halo_catalog(halo_cat, hmf='dndlog10m', mr=1.3e10, boxsize=896)
Ms = 10**((M_edges[1:] + M_edges[:-1])/2)
dlog10m = np.diff(M_edges)[0]
volume = 896**3

del halo_cat


#------------------------------------- 2. Instantiate the model -------------------------------------#
camb = build_CAMB_cosmology()
params = camb.Params
cosmo_ccl = pyccl.Cosmology(Omega_c=params.omegac, Omega_b=params.omegab, Omega_g=0, Omega_k=params.omk,
					   h=params.h, sigma8=camb.get_sigma8_0(), n_s=camb.Params.InitPower.ns, Neff=params.N_eff, m_nu=0.0,
					   w0=-1, wa=0, T_CMB=params.TCMB, transfer_function='bbks')
mass_def = pyccl.halos.massdef.MassDef('vir', rho_type='matter')

dndlog10m_ccl = pyccl.halos.hmfunc.tinker10.MassFuncTinker10(mass_def=mass_def)  # In Mpc^-3
ST_mf_modified = MassFuncSheth99Modified(mass_def=mass_def, mass_def_strict=False)
ST_mf = pyccl.halos.hmfunc.sheth99.MassFuncSheth99(mass_def=mass_def, mass_def_strict=False)

#------------------------------------- 3. Set up the sampler -------------------------------------#


sampler_hmf = emcee.EnsembleSampler(
	nwalkers = 50,
	ndim = 3,
	log_prob_fn = log_prob,
	kwargs = {'data': counts,
		'model': ST_mf_modified,
		'volume': volume})

init_pos = (np.array([
		0.7,
		0.3,
		0.21]) +
	1e-4 * np.random.normal(size=(sampler_hmf.nwalkers, sampler_hmf.ndim))
)


sampler_hmf.run_mcmc(init_pos, nsteps=1000)
flat_chain = sampler_hmf.get_chain(discard=600, flat=True)

#------------------------------------- 4. Plot the results -------------------------------------#
# Using corner to make triangle plot
fig = corner.corner(flat_chain, labels=['a', 'p', 'A'], show_titles=True)
plt.savefig('figures/corner_plot_hmf_box1a_ccl.png', dpi=300, bbox_inches='tight')

a, p, A = np.mean(flat_chain, axis=0)
ST_mf_modified.a = a
ST_mf_modified.p = p
ST_mf_modified.A = A

dndlog10m_to_count_factor = volume*dlog10m
# Plot the best fit model
plt.figure()
plt.errorbar(Ms, dndlog10m_sim, yerr=error_box1a, color='cornflowerblue', fmt='o', label='Data')
plt.plot(Ms, ST_mf(cosmo_ccl, Ms, 1)/params.h**3, c='lime', label=f'Default: a={ST_mf.a:.2f}, p={ST_mf.p:.2f}, A={ST_mf.A:.2f}')
plt.plot(Ms,  ST_mf_modified(cosmo_ccl, Ms, 1)/params.h**3, c='r', label=f'Best fit model: a={a:.2f}, p={p:.2f}, A={A:.2f}')
plt.xlabel('M$_\mathrm{vir} [\mathrm{M}_\odot/h$]')
plt.ylabel('dn/dm $[h^4 \mathrm{M}_\odot^{-1} \mathrm{Mpc}^{-3}]$')
plt.xscale('log')
plt.yscale('log')
plt.legend(title='Sheth & Tormen 1999')
plt.savefig('figures/best_fit_hmf_box1a_ccl.png', dpi=300, bbox_inches='tight')