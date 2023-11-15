'''Same as fit_HMCode_profile.py but fits all profiles instead of 
the mean profile in each mass bin
'''
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d
from schwimmbad import MPIPool

import astropy.units as u
import astropy.cosmology.units as cu
import getdist
from getdist import plots
import glob
import emcee

import sys
sys.path.append('../core/')

from analytic_profile import Profile
import post_processing
from fitting_utils import get_halo_data
import ipdb

#####-------------- Parse Args --------------#####

parser = argparse.ArgumentParser()
parser.add_argument('--field')
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--run', type=str)
parser.add_argument('--niter', type=int)
parser.add_argument('--nsteps', type=int)
parser.add_argument('--mmin', type=float)
parser.add_argument('--mmax', type=float)
args = parser.parse_args()
test = args.test
run, niter = args.run, args.niter
mmin, mmax = args.mmin, args.mmax
field = args.field.strip('"').split(',')
#####-------------- Likelihood --------------#####

def nan_interp(x, y):
	idx = ((np.isnan(y)) | (y==0))
	return interp1d(x[~idx], y[~idx], kind='cubic', bounds_error=False, fill_value=0)

def get_scatter(x, xbar):
	# Calculate for radial bin at a time
	std  = []
	for i in range(x.shape[1]):
#		ipdb.set_trace()
		this_column = x[:, i]
		idx = (this_column>0) & (np.isfinite(this_column))
		this_column = this_column[idx]
#         std.append(np.mean((this_column-xbar[i])**2)**0.5)
		std.append((np.percentile(this_column-xbar[i], 84, axis=0) - np.percentile(this_column-xbar[i], 16, axis=0))/2)

	return np.array(std)

def recompute_best_fit_scatter(params):
	fitter.update_param(fit_par, params)
	(Pe_bestfit, rho_bestfit, Temp_bestfit), r_bestfit = fitter.get_Pe_profile_interpolated(Mvir_sim*u.Msun/cu.littleh, z=0, return_rho=True, return_Temp=True)

	median_Pe = np.median(Pe_bestfit[mask]/Pe_bestfit[mask][:, -1][:, np.newaxis], axis=0)
	median_rho = np.median(rho_bestfit[mask]/rho_bestfit[mask][:, -1][:, np.newaxis], axis=0)
	median_Temp = np.median(Temp_bestfit[mask]/Temp_bestfit[mask][:, -1][:, np.newaxis], axis=0)

	sigma_intr_Pe = get_scatter(np.log(Pe_rescale[mask]), np.log(median_Pe))
	sigma_intr_rho = get_scatter(np.log(rho_rescale[mask]), np.log(median_rho))
	sigma_intr_Temp = get_scatter(np.log(Temp_rescale[mask]), np.log(median_Temp))

	return sigma_intr_Pe, sigma_intr_rho, sigma_intr_Temp


def likelihood(theory_prof, field):
	sim_prof = globals()[field+'_sim'] # Simulation profile
	sim_sigma_lnprof = globals()[f'sigma_ln{field}'] # Measurement uncertainty

	num = np.log(sim_prof[mask] / theory_prof.value[mask])**2

	idx = sim_prof[mask]==0
	num = ma.array(num, mask=idx, fill_value=0)
    
	denom = globals()[f'sigma_intr_{field}']**2 + sim_sigma_lnprof[mask]**2
	chi2 = 0.5*np.sum(num/denom)  #Sum over radial bins

	if not np.isfinite(chi2):
		return -np.inf

	return -chi2

def joint_likelihood(x, mass_list, z=0):
	for i in range(len(fit_par)):
		lb, ub = bounds[fit_par[i]]
		if x[i]<lb or x[i]>ub:
			return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

	fitter.update_param(fit_par, x)

	mvir = mass_list*u.Msun/cu.littleh
	## Get profile for each halo
	rho_dm_theory, r = fitter.get_rho_dm_profile_interpolated(mvir, r_bins=r_bins, z=z)
	(Pe_theory, rho_theory, Temp_theory), r = fitter.get_Pe_profile_interpolated(mvir, r_bins=r_bins, z=z, return_rho=True, return_Temp=True)

	like_rho_dm, like_rho, like_Temp, like_Pe = 0., 0., 0., 0.

	if 'rho_dm' in field:
		like_rho_dm = likelihood(rho_dm_theory, 'rho_dm')

	if 'rho' in field or 'all' in field:
		like_rho = likelihood(rho_theory, 'rho')

	if 'Temp' in field  or 'all' in field:
		like_Temp = likelihood(Temp_theory, 'Temp')

	if 'Pe' in field  or 'all' in field:
		like_Pe = likelihood(Pe_theory, 'Pe')

	loglike = like_rho_dm + like_rho + like_Temp + like_Pe
	return loglike, like_rho_dm, like_rho, like_Temp, like_Pe

bounds = {'f_H': [0.65, 0.85],
                'gamma': [1.1, 5],
                'alpha': [0.1, 2.5],
                'log10_M0': [10, 17],
                'M0': [1e10, 1e17],
                'beta': [0.4, 0.8],
                'eps1_0': [-0.95, 3],
                'eps2_0': [-0.95, 3],
                'gamma_T': [1.1, 5.5],
                 'a': [0, 0.2],
                 'b': [0, 2],
                'alpha_nt': [0, 2],
                'n_nt': [0, 2]}

fid_val = {'f_H': 0.75,
                'gamma': 1.2,
                'alpha': 1,
                'log10_M0': 14,
                'M0': 1e14,
                'beta': 0.6,
                'eps1_0': 0.2,
                'eps2_0': -0.1,
                'gamma_T': 2,
                  'a': 0.1,
                  'b': 0.1,
                'alpha_nt':1,
                'n_nt':1}

std_dev = {'f_H': 0.2,
                'gamma': 0.2,
                'alpha': 0.5,
                'log10_M0': 2,
                'M0': 1e12,
                'beta': 0.2,
                'eps1_0': 0.2,
                'eps2_0': 0.2,
                'gamma_T':0.3,
                 'a': 0.02,
                 'b': 0.1,
                'alpha_nt':0.4,
                'n_nt':0.4}

#####-------------- Load Data --------------#####
save_path = f'../../magneticum-data/data/emcee_magneticum_cM/prof_{args.field}_halos_bin/{run}'
# save_path = f'../../magneticum-data/data/emcee_new/prof_{args.field}_halos_bin/{run}'
data_path = '../../magneticum-data/data/profiles_median'
files = glob.glob(f'{data_path}/Box1a/Pe_Pe_Mead_Temp_matter_cdm_gas_v_disp_z=0.00_mvir_1.0E+13_1.0E+16.pkl')
files += glob.glob(f'{data_path}/Box2/Pe_Pe_Mead_Temp_matter_cdm_gas_v_disp_z=0.00_mvir_1.0E+12_1.0E+13.pkl')

#files = [f'{data_path}/Box1a/Pe_Pe_Mead_Temp_matter_cdm_gas_z=0.00_mvir_1.0E+13_1.0E+15_coarse.pkl']
#files += [f'{data_path}/Box2/Pe_Pe_Mead_Temp_matter_cdm_gas_z=0.00_mvir_1.0E+12_1.0E+13_coarse.pkl']

## We will interpolate all measured profiles to the same r_bins as 
## the analytical profile for computational efficiency
rho_dm_sim= []
Pe_sim= []
rho_sim= []
Temp_sim= []

sigma_lnrho_dm = []
sigma_lnPe = []
sigma_lnrho = []
sigma_lnTemp = []

Mvir_sim = []

## Also need to rescale profile to guess intrinsic scatter 
rho_dm_rescale = []
Pe_rescale = []
rho_rescale = []
Temp_rescale = []

bin_edges = np.logspace(np.log10(0.1), np.log10(2), 17)
r_bins = (bin_edges[:-1] + bin_edges[1:])/2
r_bins = r_bins[3:-3]  #Exclude the last bin ~0.2-1.04 Rvir
r_bins[-1] = 0.98 # Make last bin slightly smaller than Rvir


for f in files:
	this_prof_data = joblib.load(f)

	for halo in this_prof_data:

		## Pressure	
		rho_dm_prof_interp, rho_dm_rescale_interp, unc_rho_dm = get_halo_data(halo, 'rho_dm', r_bins, return_sigma=True)
		if rho_dm_prof_interp is None or rho_dm_rescale_interp is None: continue

		## Gas density
		rho_prof_interp, rho_rescale_interp, unc_rho = get_halo_data(halo, 'rho', r_bins, return_sigma=True)
		if rho_prof_interp is None or rho_rescale_interp is None: continue


		## Temperature
		Temp_prof_interp, Temp_rescale_interp, unc_Temp  = get_halo_data(halo, 'Temp', r_bins, return_sigma=True)
		if Temp_prof_interp is None or Temp_rescale_interp is None: continue

		# These should be after all the if statements
		rho_dm_sim.append(rho_dm_prof_interp)
		rho_dm_rescale.append(rho_dm_rescale_interp)
		sigma_lnrho_dm.append(unc_rho_dm)

		Pe_sim.append(Pe_prof_interp)
		Pe_rescale.append(Pe_rescale_interp)
		sigma_lnPe.append(unc_Pe)

		rho_sim.append(rho_prof_interp)
		rho_rescale.append(rho_rescale_interp)
		sigma_lnrho.append(unc_rho)

		Temp_sim.append(Temp_prof_interp)
		Temp_rescale.append(Temp_rescale_interp)    
		sigma_lnTemp.append(unc_Temp)

		Mvir_sim.append(halo['mvir'])

# Since low mass halos have a large scatter we compute it separately for them

# Now we need to sort halos in order of increasing mass
# Since this is what the scipy interpolator expects
Mvir_sim = np.array(Mvir_sim, dtype='float32')
sorting_indices = np.argsort(Mvir_sim)

mask = (Mvir_sim>=10**(mmin)) & (Mvir_sim<10**mmax)
idx = np.arange(5, 10)
r_bins = r_bins[idx]



rho_dm_sim = np.array(rho_dm_sim, dtype='float32')[sorting_indices][:, idx]
Pe_sim = np.array(Pe_sim, dtype='float32')[sorting_indices][:, idx]
rho_sim = np.array(rho_sim, dtype='float32')[sorting_indices][:, idx]
Temp_sim = np.array(Temp_sim, dtype='float32')[sorting_indices][:, idx]
Mvir_sim = Mvir_sim[sorting_indices]

#---------------------- rho_dm ----------------------#
rho_dm_rescale = np.vstack(rho_dm_rescale)[sorting_indices][:, idx]
sigma_lnrho_dm = np.vstack(sigma_lnrho_dm)[sorting_indices][:, idx]

# High mass
median_prof = np.median(rho_dm_rescale[mask], axis=0)
sigma_intr_rho_dm = get_scatter(np.log(rho_dm_rescale[mask]), np.log(median_prof))

#---------------------- Pressure ----------------------#
Pe_rescale = np.vstack(Pe_rescale)[sorting_indices][:, idx]
sigma_lnPe = np.vstack(sigma_lnPe)[sorting_indices][:, idx]

# High mass
median_prof = np.median(Pe_rescale[mask], axis=0)
sigma_intr_Pe = get_scatter(np.log(Pe_rescale[mask]), np.log(median_prof))


####################### rho ###############################
rho_rescale = np.vstack(rho_rescale)[sorting_indices][:, idx]
sigma_lnrho = np.vstack(sigma_lnrho)[sorting_indices][:, idx]

# High mass
median_prof = np.median(rho_rescale[mask], axis=0)
sigma_intr_rho = get_scatter(np.log(rho_rescale[mask]), np.log(median_prof))


####################### Temp ###############################
Temp_rescale = np.vstack(Temp_rescale)[sorting_indices][:, idx]
sigma_lnTemp = np.vstack(sigma_lnTemp)[sorting_indices][:, idx]

# High mass
median_prof = np.median(Temp_rescale[mask], axis=0)
sigma_intr_Temp = get_scatter(np.log(Temp_rescale[mask]), np.log(median_prof))


sigma_intr_rho_dm[-1] = 0.1
sigma_intr_Pe[-1] = 0.1
sigma_intr_rho[-1] = 0.1
sigma_intr_Temp[-1] = 0.1


print('Finished processing simulation data...')
print(f'Using {np.sum(mask)} halos for fit...')

#####-------------- Prepare for MCMC --------------#####
fitter = Profile(use_interp=True, mmin=Mvir_sim.min()-1e10, mmax=Mvir_sim.max()+1e10, imass_conc=1)
print('Initialized profile fitter ...')
#fit_par = ['gamma', 'alpha', 'log10_M0', 'eps1_0', 'eps2_0', 'gamma_T_1', 'gamma_T_2', 'alpha_nt', 'n_nt']
#par_latex_names = ['\Gamma', '\\alpha', '\log_{10}M_0', '\epsilon_1', '\epsilon_2', '\Gamma_\mathrm{T}^1', '\Gamma_\mathrm{T}^2', '\\alpha_{nt}', 'n_{nt}']

#fit_par = ['gamma', 'alpha', 'log10_M0', 'eps1_0', 'eps2_0', 'gamma_T']#, 'a']
#par_latex_names = ['\Gamma', '\\alpha', '\log_{10}M_0', '\epsilon_1', '\epsilon_2', '\Gamma_\mathrm{T}']#, 'a']

fit_par = ['gamma', 'log10_M0', 'eps1_0', 'eps2_0']
par_latex_names = ['\Gamma', '\log_{10}M_0', '\epsilon_1', '\epsilon_2']

starting_point = [fid_val[k] for k in fit_par]
std = [std_dev[k] for k in fit_par]

ndim = len(fit_par)
nwalkers= 40
nsteps = args.nsteps

if niter>1:
	print('Recomputing scatter using best fit parameters from previous iteration...')
	best_params_prev_iter = np.loadtxt(f'{save_path}/best_params_{niter-1}.txt', skiprows=1)
	sigma_intr_Pe, sigma_intr_rho, sigma_intr_Temp = recompute_best_fit_scatter(best_params_prev_iter)
	starting_point = best_params_prev_iter


p0_walkers = emcee.utils.sample_ball(starting_point, std, size=nwalkers)

for i, key in enumerate(fit_par):
	low_lim, up_lim = bounds[fit_par[i]]

	for walker in range(nwalkers):
		while p0_walkers[walker, i] < low_lim or p0_walkers[walker, i] > up_lim:
			p0_walkers[walker, i] = np.random.rand()*std[i] + starting_point[i]

print(f'Finished initializing {nwalkers} walkers...')

print(f'Using Likelihood for {field} field(s)')

def test_interpolator(walkers):
	test_interp = Profile(use_interp=True, mmin=10**(mmin)-1e10, mmax=10**(mmax)+1e10, interp_error_tol = 0.1)

	for row in p0_walkers:
		test_interp.update_param(fit_par, np.array(row))
		test_interp._test_prof_interpolator()

#####-------------- RUN MCMC --------------#####
blobs_dtype = [('loglike_rho_dm', float), ('loglike_rho', float), ('loglike_Temp', float), ('loglike_Pe', float)]

if test is False:
	with MPIPool() as pool:
		if not pool.is_master():
			pool.wait()
			sys.exit(0)
		print('Testing interpolator...')
		test_interpolator(p0_walkers)
		print('Running MCMC with MPI..')
		sampler = emcee.EnsembleSampler(nwalkers, ndim, joint_likelihood, blobs_dtype=blobs_dtype, pool=pool, args=[Mvir_sim])
		sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=True)

else:
	print('Testing interpolator...')
	#test_interpolator(p0_walkers)
	print('Running MCMC...')

	sampler = emcee.EnsembleSampler(nwalkers, ndim, joint_likelihood, blobs_dtype=blobs_dtype, args=[Mvir_sim])
	sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=True)


#####-------------- Plot and Save --------------#####

if not os.path.exists(save_path):
# If the folder does not exist, create it and break the loop
	os.makedirs(save_path)

walkers = sampler.get_chain()
log_prob = sampler.get_log_prob()

chain = sampler.get_chain(flat=True)
log_prob_flat = sampler.get_log_prob(flat=True)
blobs_flat = sampler.get_blobs(flat=True)

idx = np.argmax(log_prob_flat)
best_params = chain[idx]

all_samples = np.concatenate((chain, log_prob_flat[:, None], blobs_flat['loglike_rho_dm'][:, None], 
							blobs_flat['loglike_rho'][:, None], 
							blobs_flat['loglike_Temp'][:, None], 
							blobs_flat['loglike_Pe'][:, None]), axis=1)


fig, ax = plt.subplots(len(fit_par), 1, figsize=(10, 1.5*len(fit_par)))
ax = ax.flatten()

for i in range(len(fit_par)):
	ax[i].plot(walkers[:, :, i])
	ax[i].set_ylabel(f'${par_latex_names[i]}$')
	ax[i].set_xlabel('Step #')

plt.savefig(f'{save_path}/trace_plot_{niter}.pdf')

#### Discard 0.9*steps and make triangle plot

gd_samples = getdist.MCSamples(samples=sampler.get_chain(flat=True, discard=int(0.9*nsteps)), names=fit_par, labels=par_latex_names)

np.save(f'{save_path}/all_walkers_{niter}.npy', walkers)
np.savetxt(f'{save_path}/all_samples_{niter}.txt', all_samples)
np.savetxt(f'{save_path}/sigma_intr_{niter}.txt',  np.column_stack((sigma_intr_rho, sigma_intr_Temp, sigma_intr_Pe)))
np.savetxt(f'{save_path}/best_params_{niter}.txt', best_params, header='\t'.join(fit_par))
########## Compare best-fit profiles ##########
c = ['r', 'b', 'g', 'k']

# Fiducial HMCode profiles
fitter.update_param(fit_par, best_params)
rho_dm_bestfit = fitter.get_rho_dm_profile_interpolated(Mvir_sim*u.Msun/cu.littleh, z=0)
(Pe_bestfit, rho_bestfit, Temp_bestfit), r_bestfit = fitter.get_Pe_profile_interpolated(Mvir_sim*u.Msun/cu.littleh, z=0, return_rho=True, return_Temp=True)


## Plot median Pe profiles
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax = ax.flatten()

rho_dm_sim[rho_dm_sim==0] = np.nan
ax[0].errorbar(r_bins, np.log(np.nanmedian(rho_sim[mask], axis=0)), yerr=sigma_intr_rho, ls='-.', label='Magneticum (median)')
ax[0].plot(r_bestfit, np.log(np.median(rho_bestfit[mask], axis=0).value), ls='-.', label='Best fit (median)')

rho_sim[rho_sim==0] = np.nan

ax[1].errorbar(r_bins, np.log(np.nanmedian(rho_sim[mask], axis=0)), yerr=sigma_intr_rho, ls='-.', label='Magneticum (median)')
ax[1].plot(r_bestfit, np.log(np.median(rho_bestfit[mask], axis=0).value), ls='-.', label='Best fit (median)')

Temp_sim[Temp_sim==0] = np.nan
ax[2].errorbar(r_bins, np.log(np.nanmedian(Temp_sim[mask], axis=0)), yerr=sigma_intr_Temp, ls='-.')
ax[2].plot(r_bestfit, np.log(np.median(Temp_bestfit[mask], axis=0).value), ls='-.')


Pe_sim[Pe_sim==0] = np.nan
ax[3].errorbar(r_bins, np.log(np.nanmedian(Pe_sim[mask], axis=0)), yerr=sigma_intr_Pe, ls='-.')
ax[3].plot(r_bestfit, np.log(np.median(Pe_bestfit[mask], axis=0).value), ls='-.')

for i in range(4):
	ax[i].set_xlim(0.1, 1.1)

for i in range(4):
	ax[i].set_xscale('log')

ax[0].set_ylabel('$\ln \\rho_{DM}$ [GeV/cm$^3$]')
ax[1].set_ylabel('$\ln \\rho_{gas}$ [GeV/cm$^3$]')
ax[2].set_ylabel('$\ln$ Temperature [K]')
ax[3].set_ylabel('$\ln P_e$ [keV/cm$^3$]')

ax[0].set_xlabel('$r/Rvir$')
ax[1].set_xlabel('$r/Rvir$')
ax[2].set_xlabel('$r/Rvir$')
ax[3].set_xlabel('$r/Rvir$')

ax[1].set_title(f'{mmin}<logM<{mmax}')
ax[0].legend()
plt.savefig(f'{save_path}/best_fit_profiles_{niter}.pdf', bbox_inches='tight')


#### make triangle plot
plt.figure()
g = plots.get_subplot_plotter()
g.triangle_plot(gd_samples, axis_marker_lw=2, marker_args={'lw':2}, line_args={'lw':1}, title_limit=2)
plt.savefig(f'{save_path}/triangle_plot_{niter}.pdf')