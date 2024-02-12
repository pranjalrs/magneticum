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

import astropy.units as u
import astropy.cosmology.units as cu

import glob

import sys
sys.path.append('../core/')

from analytic_profile import Profile
import post_processing
from fitting_utils import get_halo_data
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
def likelihood(theory_prof, field):
	sim_prof = globals()['this_'+field+'_sim'] # Simulation profile
	sim_sigma_lnprof = globals()[f'this_sigma_ln{field}']# Measurement uncertainty
	sim_sigma_prof = globals()[f'this_sigma_{field}']# Measurement uncertainty

	if chi2_type == 'log':
		num = np.log(sim_prof / theory_prof.value)**2
		denom = sim_sigma_lnprof**2

	elif chi2_type == 'linear':
		num = (sim_prof - theory_prof.value)**2
		denom = sim_sigma_prof**2

	idx = sim_prof==0
	num = ma.array(num, mask=idx, fill_value=0)
	
	chi2 = 0.5*np.sum(num/denom)  #Sum over radial bins

	if not np.isfinite(chi2):
		return np.inf

	return chi2

def joint_likelihood(x, data, Mvir, Rvir, r_bins, z=0):
	for i in range(len(fit_par)):
		lb, ub = bounds[fit_par[i]]
		if x[i]<lb or x[i]>ub:
			return -np.inf

	fitter.update_param(fit_par, x)

	## Get profile for each halo
	rho_dm_theory, r = fitter.get_rho_dm_profile(mvir*munit, r_bins=r_bins, z=z)

	like_rho_dm, like_rho, like_Temp, like_Pe = 0., 0., 0., 0.

	if 'rho_dm' in field:
		like_rho_dm = likelihood(rho_dm_theory, 'rho_dm')
	
	## Check if the the mass enclosed in the profile is consistent with 
	## The halo mass in the catalog
	M_nfw = get_NFW_mass(Rvir, fitter.get_concentration(mvir*munit, z=0), 10**fitter.lognorm_rho)

	if M_nfw>Mvir: return -np.inf

	loglike = like_rho_dm + like_rho + like_Temp + like_Pe
	return loglike

def joint_likelihood_DE(x, data, Mvir, Rvir, r_bins, z=0):
	return - joint_likelihood(x, data, Mvir, Rvir, r_bins, z=0)

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

	sampler = emcee.EnsembleSampler(nwalkers, ndim, joint_likelihood, args=args)
	sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=False)

	return sampler

bounds = {'lognorm_rho': [1, 20],
			'conc_param': [1, 20]}

fid_val = {'lognorm_rho': 10,
			'conc_param': 7}

std_dev = {'lognorm_rho': 0.1,
			'conc_param': 0.1}

#####-------------- Load Data --------------#####
files = glob.glob('/home/u31/pranjalrs/*.pkl')

rho_dm_sim= []
sigma_rho_dm = []
sigma_lnrho_dm = []
r_bins_sim = []
Mvir_sim = []
Rvir_sim = []

conversion_factor = (1*u.Msun/u.kpc**3).to(u.GeV/u.cm**3, u.mass_energy()).value
for f in files:
	this_prof_data = joblib.load(f)
	for halo in this_prof_data:

		r = halo['fields']['cdm'][1]/halo['rvir']
		profile = halo['fields']['cdm'][0]
		npart = halo['fields']['cdm'][2]
		sigma_prof = profile/npart**0.5        
		sigma_lnprof = sigma_prof/profile

		# These should be after all the if statements
		rho_dm_sim.append(profile)
		sigma_rho_dm.append(sigma_prof)
		sigma_lnrho_dm.append(sigma_lnprof)
		r_bins_sim.append(r)

		Rvir_sim.append(halo['rvir'])
		Mvir_sim.append(halo['mvir'])

Mvir_sim = np.array(Mvir_sim, dtype='float32')
Rvir_sim = np.array(Rvir_sim, dtype='float32')

rho_dm_sim = np.array(rho_dm_sim, dtype='float32')
sigma_rho_dm = np.vstack(sigma_rho_dm)
sigma_lnrho_dm = np.vstack(sigma_lnrho_dm)
r_bins_sim = np.vstack(r_bins_sim)

mask = (Mvir_sim>=10**(mmin)) & (Mvir_sim<10**mmax)
print(f'{np.log10(Mvir_sim[mask].min()):.2f}, {np.log10(Mvir_sim[mask].max()):.2f}')

rho_dm_sim = rho_dm_sim[mask]
sigma_rho_dm = sigma_rho_dm[mask]
sigma_lnrho_dm = sigma_lnrho_dm[mask]
Mvir_sim = Mvir_sim[mask]
r_bins_sim = r_bins_sim[mask]

print('Finished processing simulation data...')
print(f'Using {np.sum(mask)} halos for fit...')

#####-------------- Prepare for MCMC --------------#####
fitter = Profile(use_interp=False, imass_conc=2)
print('Initialized profile fitter ...')


fit_par = ['conc_param', 'lognorm_rho']
par_latex_names = ['c(M)', 'log A_0']

starting_point = [fid_val[k] for k in fit_par]
std = [std_dev[k] for k in fit_par]
these_bounds = [bounds[k] for k in fit_par]
print(f'Using Likelihood for {field} field(s)')


#####-------------- RUN MCMC --------------#####
base_path = '../../magneticum-data/data/DM_conc/test/'
fit_result = []

fig = plt.figure()

for i in tqdm(range(sum(mask))):
	this_rvir = Rvir_sim[i]
	this_mvir = Mvir_sim[i]
	this_r_bins = r_bins_sim[i]

	## Apply cut on rmin
	idx = (this_r_bins*this_rvir>30) & (this_r_bins<=1.)
	this_r_bins = this_r_bins[idx]
	this_rho_dm_sim = rho_dm_sim[i][idx] # Don't change variable names; called in likelihood using `globals()`
	this_sigma_rho_dm = sigma_rho_dm[i][idx]
	this_sigma_lnrho_dm = sigma_lnrho_dm[i][idx]

	## Args to be passed to likelihood function
	args = (this_rho_dm_sim, this_mvir, this_rvir,this_r_bins)
	sol = scipy.optimize.differential_evolution_DE(joint_likelihood, bounds=np.array(these_bounds), x0=starting_point, args=args)

	sampler = run_mcmc(sol.x, nsteps=300, nwalkers=40, args=args)
	samples = sampler.get_chain(discard=200, flat=True)
	mean = np.mean(samples, axis=0)
	std = np.std(samples, axis=0)

	assert np.abs(sol.x[0]-mean[0])<0.5*std[0]

	fit_result.append([this_halo_mass, sol.x[0], sol.x[1], sol.fun])

	fitter.update_param(fit_par, sol.x)
	rho_dm_theory, r = fitter.get_rho_dm_profile(this_halo_mass*u.Msun/cu.littleh, r_bins=this_r_bins, z=0)

# 	if sol.x[0]>15.:
# 		fig2, ax2 = plt.subplots(1, 1)
# 		ax2.errorbar(r, np.log(this_rho_dm_sim), yerr=this_sigma_lnrho_dm, c='dodgerblue', alpha=0.2)
# 		ax2.semilogx(r, np.log(rho_dm_theory.value), c='orangered', alpha=0.2)
# 		fig2.savefig('temp.pdf')
# 		ipdb.set_trace()        
# 	else:
# 		pass
	plt.semilogx(r, this_rho_dm_sim/rho_dm_theory.value, c='dodgerblue', alpha=0.4)

plt.ylabel('$\\rho_{\mathrm{sim}}/\\rho_{\mathrm{NFW}}$')
plt.xlabel('$r/R_{\mathrm{vir}}$')
plt.savefig(f'{base_path}/DM_prof_residual_{mmin}_{mmax}_{chi2_type}.pdf', bbox_inches='tight')
plt.close()

result = np.vstack(fit_result)

plt.figure()
plt.scatter(result[:,0], result[:, 1], c=result[:, -1], alpha=0.5)
plt.colorbar()
plt.ylabel('Concentration')
plt.xlabel('Virial Mass [Msun]')
plt.xscale('log')
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
