'''Same as fit_HMCode_profile.py but fits concentration
on averaged profiles
'''
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d
import scipy.optimize
from tqdm import tqdm
from multiprocessing import Pool
from matplotlib.cm import ScalarMappable

import astropy.units as u
import astropy.cosmology.units as cu
import emcee
import glob
import lmfit


from dawn.halo_profile import HaloProfile
from dawn.sim_toolkit.profile_handler import HaloProfileHandler
import ipdb

#####-------------- Parse Args --------------#####
parser = argparse.ArgumentParser()
# parser.add_argument('--field', choices=['rho_dm'])
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--run', type=str)
parser.add_argument('--niter', type=int)
parser.add_argument('--chi2', type=str, choices=['log', 'linear'])
parser.add_argument('--mmin', type=float)
parser.add_argument('--mmax', type=float)
args = parser.parse_args()

test = args.test
chi2_type = args.chi2
run, niter = args.run, args.niter
mmin, mmax = args.mmin, args.mmax

field = ['rho_dm']

munit = u.Msun/cu.littleh
#####-------------- Likelihood --------------#####
def likelihood(theory_prof, args_dict):
	sim_prof = args_dict['data']
	sigma_prof = args_dict['sigma_prof']
	sigma_lnprof = args_dict['sigma_lnprof']
	chi2 = args_dict['chi2_type']
	return_sum = args_dict['return_sum']

	if chi2 == 'log':
		num = np.log(sim_prof / theory_prof.value)
		denom = sigma_lnprof

	elif chi2 == 'linear':
		num = (sim_prof - theory_prof.value)
		denom = sigma_prof

	idx = sim_prof==0

	residual = (num[~idx]/denom[~idx])  #Sum over radial bins

	if not np.all(np.isfinite(residual)):
		return np.inf

	if return_sum is True:
		return 0.5*np.sum(residual**2)

	elif return_sum is False:
		return 0.5*residual**2

	elif return_sum == 'residual':
		return residual

def joint_likelihood(x, args_dict):
	Mvir = args_dict['Mvir']
	Rvir = args_dict['Rvir']
	r_bins = args_dict['r_bins']
	z = args_dict['z']
    
	for i in range(len(fit_par)):
		lb, ub = bounds[fit_par[i]]
		if x[i]<lb or x[i]>ub:
			return -np.inf

	fitter.update_param(fit_par, x)

	## Get profile for each halo
	rho_dm_theory, r = fitter.get_rho_dm_profile(Mvir*munit, r_bins=r_bins, z=z)

	like_rho_dm, like_rho, like_Temp, like_Pe = 0., 0., 0., 0.

	if 'rho_dm' in field:
		like_rho_dm = likelihood(rho_dm_theory, args_dict)

	## Check if the the mass enclosed in the profile is consistent with 
	## The halo mass in the catalog
	M_nfw = get_NFW_mass(Rvir, fitter.get_concentration(Mvir*munit, z=0), 10**fitter.lognorm_rho)

# 	if M_nfw>Mvir: return -np.inf

	loglike = like_rho_dm + like_rho + like_Temp + like_Pe
	return -loglike

def joint_likelihood_DE(x, args_dict):
	return - joint_likelihood(x, args_dict)

def get_NFW_mass(rvir, conc, rho0):
	## Integral of the NFW profile up to 1Rvir
	rs = rvir/conc
	return 4*np.pi*rho0*rs**3*( np.log(1+conc) - conc/(1+conc))

def run_mcmc(x0, nsteps, nwalkers, args):
	ndim = len(fit_par)
	p0_walkers = emcee.utils.sample_ball(x0, std, size=nwalkers)

	for i, key in enumerate(fit_par):
		low_lim, up_lim = bounds[fit_par[i]]

	for walker in range(nwalkers):
		while p0_walkers[walker, i] < low_lim or p0_walkers[walker, i] > up_lim:
			p0_walkers[walker, i] = np.random.rand()*std[i] + starting_point[i]

	with Pool() as pool:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, joint_likelihood, args=args, pool=pool)
		sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=False)

	return sampler

def get_mcmc_std(x0, nsteps, nwalkers, args):
	sampler = run_mcmc(x0, nsteps, nwalkers, args)
	samples = sampler.get_chain(discard=200, flat=True)
	mean = np.mean(samples, axis=0)
	std = np.std(samples, axis=0)
	return std

## Need for lmfit
def likelihood_wrapper(par, args_dict):
	x = [par['conc_param'], par['lognorm_rho']]
	return joint_likelihood_DE(x, args_dict)    


def get_lmfit_sol(x0, bounds, args_dict):
	params = lmfit.Parameters()
	params.add('conc_param', value=x0[0], min=bounds[0][0], max=bounds[0][1])
	params.add('lognorm_rho', value=x0[1], min=bounds[1][0], max=bounds[1][1])

	wrap = lambda x: likelihood_wrapper(x, args_dict)
	res = lmfit.minimize(wrap, params, method='leastsq')

	return res


bounds = {'lognorm_rho': [1, 20],
			'conc_param': [0.2, 50]}

fid_val = {'lognorm_rho': 10,
			'conc_param': 7}

std_dev = {'lognorm_rho': 0.1,
			'conc_param': 0.1}

#####-------------- Load Data --------------#####
files = glob.glob('../../magneticum-data/data/profiles_median/*/cdm*.pkl')

profile_handler = HaloProfileHandler(['rho_dm'], files)

data = profile_handler.get_masked_profile(10**mmin, 10**mmax, 10, 'rho_dm')

rho_dm_sim = data.profile
sigma_rho_dm = data.sigma_prof
sigma_lnrho_dm = data.sigma_lnprof
Mvir_sim = data.mvir
Rvir_sim = data.rvir
r_bins_sim = data.rbins

print('Finished processing simulation data...')

#####-------------- Prepare for MCMC --------------#####
fitter = HaloProfile(use_interp=False, imass_conc=2)
print('Initialized profile fitter ...')


fit_par = ['conc_param', 'lognorm_rho']
par_latex_names = ['c(M)', 'log A_0']

starting_point = [fid_val[k] for k in fit_par]
std = [std_dev[k] for k in fit_par]
these_bounds = [bounds[k] for k in fit_par]
print(f'Using Likelihood for {field} field(s)')


#####-------------- RUN MCMC --------------#####
base_path = '../../magneticum-data/data/DM_conc/'
fit_result = []

fig = plt.figure()

map_array = np.linspace(1, 20, 50)
colors = plt.cm.viridis(np.linspace(0, 1, 50))

for i in tqdm(range(Mvir_sim.size)):
	this_rvir = Rvir_sim[i]
	this_mvir = Mvir_sim[i]
	this_r_bins = r_bins_sim[i]

	## Apply cut on rmin
	idx = (this_r_bins*this_rvir>10.) & (this_r_bins<=1.)
	this_r_bins = this_r_bins[idx]
	this_rho_dm_sim = rho_dm_sim[i][idx] # Don't change variable names; called in likelihood using `globals()`
	this_sigma_rho_dm = sigma_rho_dm[i][idx]
	this_sigma_lnrho_dm = sigma_lnrho_dm[i][idx]

    
	## Args to be passed to likelihood function
# 	args = (this_rho_dm_sim, this_mvir, this_rvir, this_r_bins)
# 	sol = scipy.optimize.differential_evolution(joint_likelihood_DE, bounds=np.array(these_bounds), x0=starting_point, args=args)

# 	std = get_mcmc_std(sol.x, nsteps=300, nwalkers=40, args=args)
	args_dict = {'Mvir': this_mvir,
                 'Rvir': this_rvir,
                 'r_bins': this_r_bins,
                 'z': 0.,
                 'data': this_rho_dm_sim,
                 'sigma_prof': this_sigma_rho_dm,
                 'sigma_lnprof': this_sigma_lnrho_dm,
                 'chi2_type': chi2_type,
                 'return_sum': 'residual'}

	res = get_lmfit_sol(starting_point, bounds=these_bounds, args_dict=args_dict)

	res_c, res_logrho0 = res.params['conc_param'].value, res.params['lognorm_rho'].value
	std = np.diagonal(res.covar)**0.5

	fit_result.append([this_mvir, res_c, res_logrho0, std[0], std[1], res.redchi])

	fitter.update_param(fit_par, [res_c, res_logrho0])
	rho_dm_theory, r = fitter.get_rho_dm_profile(this_mvir*u.Msun/cu.littleh, r_bins=this_r_bins, z=0)
# 	print(this_rho_dm_sim/rho_dm_theory.value)
# 	if i==25:#np.any(this_rho_dm_sim/rho_dm_theory.value)>4:#sol.x[0]>18.:
# 		fig2, ax2 = plt.subplots(1, 1)
# 		ax2.errorbar(r, np.log(this_rho_dm_sim), yerr=this_sigma_lnrho_dm, c='dodgerblue', alpha=0.2)
# 		ax2.semilogx(r, np.log(rho_dm_theory.value), c='orangered', alpha=0.2)
# 		fig2.savefig('temp.pdf')
# 		plt.close()    
# 		ipdb.set_trace()        
# 	else:
# 		pass

	c_idx = np.argmin(np.abs(map_array-res_c))
	plt.semilogx(r, this_rho_dm_sim/rho_dm_theory.value, c=colors[c_idx], alpha=0.4)

scalarmappaple = ScalarMappable(cmap=plt.cm.viridis)
scalarmappaple.set_array(map_array)
cbar = fig.colorbar(scalarmappaple, ax=plt.gca(), label='Concentration')

plt.axhline(1, ls='--', c='k', zorder=201)
plt.ylabel('$\\rho_{\mathrm{sim}}/\\rho_{\mathrm{NFW}}$')
plt.xlabel('$r/R_{\mathrm{vir}}$')
plt.savefig(f'{base_path}/DM_prof_residual_{mmin}_{mmax}_{chi2_type}.pdf', bbox_inches='tight')
plt.close()

result = np.vstack(fit_result)


## Plot Mass vs Concentration scatter
plt.figure()
plt.scatter(result[:,0], result[:, 1], c=np.log10(result[:, -1]), alpha=0.5)
plt.errorbar(result[:,0], result[:, 1], yerr=result[:, 3], fmt='o', mfc='white', alpha=0.)
plt.colorbar(label='$\log_{10}\,\, \chi^2_\\nu$')

plt.ylabel('Concentration')
plt.xlabel('Virial Mass [Msun]')
plt.xscale('log')
plt.yscale('log')
plt.ylim([0.15, 22])

plt.savefig(f'{base_path}/conc_{mmin}_{mmax}_{chi2_type}.pdf', bbox_inches='tight')
plt.close()

plt.figure()
plt.scatter(result[:,0], result[:, -1], c='dodgerblue', alpha=0.2)
plt.ylabel('Cost Fn.')
plt.xlabel('Virial Mass [Msun]')
plt.xscale('log')
plt.savefig(f'{base_path}/chi2_{mmin}_{mmax}_{chi2_type}.pdf', bbox_inches='tight')
plt.close()

joblib.dump(result, f'{base_path}/DM_conc_{mmin}_{mmax}_{chi2_type}.pkl')
