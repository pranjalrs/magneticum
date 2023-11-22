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

import ipdb
#####-------------- Parse Args --------------#####

parser = argparse.ArgumentParser()
parser.add_argument('--field')
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--run', type=str)
parser.add_argument('--nsteps', type=int)
args = parser.parse_args()
test = args.test
run = args.run

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

def update_sigma_intr(val1, val2):
	global sigma_intr_Pe
	global sigma_intr_rho
	
	sigma_intr_Pe = val1
	sigma_intr_rho = val2

def get_halo_data(halo, field):
	if field=='Pe':
		r = halo['fields']['Pe_Mead'][1]/halo['rvir']
		profile = halo['fields']['Pe_Mead'][0]
		this_sigma_lnP = halo['fields']['Pe'][3]

	if field=='rho':
		r = halo['fields']['gas'][1]/halo['rvir']
		profile = halo['fields']['gas'][0]
		sigma_lnrho = halo['fields']['gas'][3]

	if field=='Temp':
		r = halo['fields']['Temp'][1]/halo['rvir']
		profile = halo['fields']['Temp'][0]
		sigma_lnrho = halo['fields']['Temp'][3]

	#Rescale prof to get intr. scatter
	rescale_value = nan_interp(r, profile)(1)
	profile_rescale = (profile/ rescale_value)
	profile_rescale_interp = nan_interp(r, profile_rescale)(r_bins)
	profile_interp = nan_interp(r, profile)(r_bins)

	if np.any(profile_rescale<0) or np.any(profile_interp<0) or np.all(np.log(profile_rescale)<0):
		return None, None

	else:
		return profile_interp, profile_rescale_interp


def likelihood(theory_prof, field):
	sim_prof = globals()[field+'_sim']
	num = np.log(sim_prof[mask_low_mass] / theory_prof.value[mask_low_mass])**2
	denom = globals()[f'sigma_intr_{field}_init_high_mass']**2 #+ sigmalnP_sim**2
	chi2 = 0.5*np.sum(num/denom)  #Sum over radial bins

	idx = sim_prof[~mask_low_mass] ==0
	num = np.log(sim_prof[~mask_low_mass] / theory_prof.value[~mask_low_mass])**2
	denom = globals()[f'sigma_intr_{field}_init_low_mass']**2 #+ sigmalnP_sim**2
	num = ma.array(num, mask=idx, fill_value=0)
	chi2 += 0.5*np.sum(num/denom)  # Sum over radial bins

	if not np.isfinite(chi2):
		return -np.inf

	return -chi2


def joint_likelihood(x, mass_list, z=0):
	for i in range(len(fit_par)):
		lb, ub = bounds[fit_par[i]]
		if x[i]<lb or x[i]>ub:
			return -np.inf


	fitter_low_mass.update_param(fit_par_unique, np.array(x)[[0, 2, 4, 5, 6, 7]])
	fitter_high_mass.update_param(fit_par_unique, np.array(x)[[1, 3, 4, 5, 6, 8]])

	mvir = mass_list*u.Msun/cu.littleh
	## Get profile for each halo

	(Pe_theory1, rho_theory1, Temp_theory1), r = fitter_low_mass.get_Pe_profile_interpolated(mvir[~mask_low_mass], r_bins=r_bins, z=z, return_rho=True, return_Temp=True)

	(Pe_theory2, rho_theory2, Temp_theory2), r = fitter_high_mass.get_Pe_profile_interpolated(mvir[mask_low_mass], r_bins=r_bins, z=z, return_rho=True, return_Temp=True)

	Pe_theory = np.concatenate((Pe_theory1, Pe_theory2))
	rho_theory = np.concatenate((rho_theory1, rho_theory2))
	Temp_theory = np.concatenate((Temp_theory1, Temp_theory2))

	loglike = 0

	if 'Pe' in field  or 'all' in field:
		loglike += likelihood(Pe_theory, 'Pe')
        
        
	if 'rho' in field or 'all' in field:
		loglike += likelihood(rho_theory, 'rho')

	if 'Temp' in field  or 'all' in field:
		loglike += likelihood(Temp_theory, 'Temp')

	return loglike


bounds = {'f_H': [0.65, 0.85],
		'gamma': [1.1, 5],
		'gamma2': [1.1, 5],
		'alpha': [0.1, 2],
		'alpha2': [0.1, 2],
		'log10_M0': [10, 17],
		'M0': [1e10, 1e17],
		'beta': [0.4, 0.8],
		'eps1_0': [-0.95, 3],
		'eps2_0': [-0.95, 3],
		'gamma_T_1': [1.1, 5.5],
		'gamma_T_2': [1.1, 5.5],
		 'b': [0, 2],
		'alpha_nt': [0, 2],
		'n_nt': [0, 2]}

fid_val = {'f_H': 0.75,
		'gamma': 1.2,
		'gamma2': 1.2,
		'alpha': 1,
		'alpha2': 1,
		'log10_M0': 14,
		'M0': 1e14,
		'beta': 0.6,
		'eps1_0': 0.2,
		'eps2_0': -0.1,
		'gamma_T_1': 2,
		'gamma_T_2': 2,
		  'b': 0.1,
		'alpha_nt':1,
		'n_nt':1}

std_dev = {'f_H': 0.2,
		'gamma': 0.2,
		'gamma2': 0.2,
		'a': 0.1,
		'alpha': 0.5,
		'alpha2': 0.5,
		'log10_M0': 2,
		'M0': 1e12,
		'beta': 0.2,
		'eps1_0': 0.2,
		'eps2_0': 0.2,
		'gamma_T_1':0.3,
		'gamma_T_2':0.3,
		  'b': 0.1,
		'alpha_nt':0.4,
		'n_nt':0.4}


#####-------------- Load Data --------------#####
data_path = '../../magneticum-data/data/profiles_median'
files = glob.glob(f'{data_path}/Box1a/Pe_Pe_Mead_Temp_matter_cdm_gas_z=0.00_mvir_3.2E+13_1.0E+16.pkl')
files += glob.glob(f'{data_path}/Box2/Pe_Pe_Mead_Temp_matter_cdm_gas_z=0.00_mvir_1.0E+12_1.0E+13.pkl')

## We will interpolate all measured profiles to the same r_bins as 
## the analytical profile for computational efficiency
Pe_sim= []
rho_sim= []
Temp_sim= []

# r_sim = []
sigmaP_sim = []
sigmarho_sim = []
sigmaTemp_sim = []

sigmalnP_sim = []
sigmalnrho_sim = []
sigmalnTemp_sim = []

Mvir_sim = []

## Also need to rescale profile to guess intrinsic scatter 
Pe_rescale = []
rho_rescale = []
Temp_rescale = []

r_bins = np.logspace(np.log10(0.15), np.log10(1), 20)


for f in files:
	this_prof_data = joblib.load(f)

	for halo in this_prof_data:

		## Pressure	
		Pe_prof_interp, Pe_rescale_interp = get_halo_data(halo, 'Pe')
		if Pe_prof_interp is None or Pe_rescale_interp is None: continue

		## Gas density
		rho_prof_interp, rho_rescale_interp = get_halo_data(halo, 'rho')
		if rho_prof_interp is None or rho_rescale_interp is None: continue


		## Temperature
		Temp_prof_interp, Temp_rescale_interp = get_halo_data(halo, 'Temp')
		if Temp_prof_interp is None or Temp_rescale_interp is None: continue

		# These should be after all the if statements
		rho_sim.append(rho_prof_interp)
		rho_rescale.append(rho_rescale_interp)

		Pe_sim.append(Pe_prof_interp)
		Pe_rescale.append(Pe_rescale_interp)

		Temp_sim.append(Temp_prof_interp)
		Temp_rescale.append(Temp_rescale_interp)    

		Mvir_sim.append(halo['mvir'])

# Now we need to sort halos in order of increasing mass
# Since this is what the scipy interpolator expects
Mvir_sim = np.array(Mvir_sim, dtype='float32')
sorting_indices = np.argsort(Mvir_sim)

Pe_sim = np.array(Pe_sim, dtype='float32')[sorting_indices]
# sigmalnP_sim = np.array(sigmalnP_sim, dtype='float32')[sorting_indices]
rho_sim = np.array(rho_sim, dtype='float32')[sorting_indices]
# sigmalnrho_sim = np.array(sigmalnrho_sim, dtype='float32')[sorting_indices]
Temp_sim = np.array(Temp_sim, dtype='float32')[sorting_indices]
Mvir_sim = Mvir_sim[sorting_indices]


# Now compute intrinsic scatter
# Since low mass halos have a large scatter we compute it separately for them
mask_low_mass = Mvir_sim>=10**(13)

####################### Pressure ###############################
Pe_rescale = np.vstack(Pe_rescale)[sorting_indices]
# High mass
median_prof = np.median(Pe_rescale[mask_low_mass], axis=0)
sigma_intr_Pe_init_high_mass = get_scatter(np.log(Pe_rescale[mask_low_mass]), np.log(median_prof))
# Low mass
median_prof = np.median(Pe_rescale[~mask_low_mass], axis=0)
sigma_intr_Pe_init_low_mass = get_scatter(np.log(Pe_rescale[~mask_low_mass]), np.log(median_prof))


####################### rho ###############################
rho_rescale = np.vstack(rho_rescale)[sorting_indices]
# High mass
median_prof = np.median(rho_rescale[mask_low_mass], axis=0)
sigma_intr_rho_init_high_mass = get_scatter(np.log(rho_rescale[mask_low_mass]), np.log(median_prof))
# Low mass
median_prof = np.median(rho_rescale[~mask_low_mass], axis=0)
sigma_intr_rho_init_low_mass = get_scatter(np.log(rho_rescale[~mask_low_mass]), np.log(median_prof))
#update_sigma_intr(sigma_intr_Pe_init, sigma_intr_rho_init)


####################### Temp ###############################
Temp_rescale = np.vstack(Temp_rescale)[sorting_indices]
# High mass
median_prof = np.median(Temp_rescale[mask_low_mass], axis=0)
sigma_intr_Temp_init_high_mass = get_scatter(np.log(Temp_rescale[mask_low_mass]), np.log(median_prof))
# Low mass
median_prof = np.median(Temp_rescale[~mask_low_mass], axis=0)
sigma_intr_Temp_init_low_mass = get_scatter(np.log(Temp_rescale[~mask_low_mass]), np.log(median_prof))


sigma_intr_Pe_init_high_mass[-1] = 0.1
sigma_intr_Pe_init_low_mass[-1] = 0.1

sigma_intr_rho_init_high_mass[-1] = 0.1
sigma_intr_rho_init_low_mass[-1] = 0.1

sigma_intr_Temp_init_high_mass[-1] = 0.1
sigma_intr_Temp_init_low_mass[-1] = 0.1


print('Finished processing simulation data...')
#####-------------- Prepare for MCMC --------------#####
fitter_low_mass = Profile(use_interp=True, mmin=Mvir_sim.min()-1e10, mmax=1e13+1e10)
fitter_high_mass = Profile(use_interp=True, mmin=1e13-1e10, mmax=Mvir_sim.max()+1e10)

print('Initialized profile fitter ...')
fit_par_unique = ['gamma', 'alpha', 'log10_M0', 'eps1_0', 'eps2_0', 'gamma_T']
fit_par = ['gamma', 'gamma2', 'alpha', 'alpha2', 'log10_M0', 'eps1_0', 'eps2_0', 'gamma_T_1', 'gamma_T_2']
par_latex_names = ['\Gamma', '\Gamma2', '\\alpha', '\\alpha_2', '\log_{10}M_0', '\epsilon_1', '\epsilon_2', '\Gamma_\mathrm{T}^1', '\Gamma_\mathrm{T}^2']

starting_point = [fid_val[k] for k in fit_par]
std = [std_dev[k] for k in fit_par]

ndim = len(fit_par)
nwalkers= 40
nsteps = args.nsteps

p0_walkers = emcee.utils.sample_ball(starting_point, std, size=nwalkers)

for i, key in enumerate(fit_par):
	low_lim, up_lim = bounds[fit_par[i]]

	for walker in range(nwalkers):
		while p0_walkers[walker, i] < low_lim or p0_walkers[walker, i] > up_lim:
			p0_walkers[walker, i] = np.random.rand()*std[i] + starting_point[i]

print(f'Finished initializing {nwalkers} walkers...')


field = args.field.strip('"').split(',')
print(f'Using Likelihood for {field} field(s)')


def test_interpolator(walkers):
	test_interp_low_mass = Profile(use_interp=True, mmin=Mvir_sim.min()-1e10, mmax=1e13+1e10, interp_error_tol = 0.05)
	test_interp_high_mass = Profile(use_interp=True, mmin=1e13-1e10, mmax=Mvir_sim.max()+1e10, interp_error_tol = 0.05)

	for row in p0_walkers:
		test_interp_low_mass.update_param(fit_par_unique, np.array(row)[[0, 2, 4, 5, 6, 7]])
		test_interp_high_mass.update_param(fit_par_unique, np.array(row)[[1, 3, 4, 5, 6, 8]])

		test_interp_low_mass._test_prof_interpolator()
		test_interp_high_mass._test_prof_interpolator()

#####-------------- RUN MCMC --------------#####
if test is False:
	with MPIPool() as pool:
		if not pool.is_master():
			pool.wait()
			sys.exit(0)
		print('Testing interpolator...')
		test_interpolator(p0_walkers)

		print('Running MCMC with MPI...')
		sampler = emcee.EnsembleSampler(nwalkers, ndim, joint_likelihood, pool=pool, args=[Mvir_sim])
		sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=True)

else:
	print('Testing interpolator...')
	test_interpolator(p0_walkers)

	print('Running MCMC...')
	sampler = emcee.EnsembleSampler(nwalkers, ndim, joint_likelihood, args=[Mvir_sim])
	sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=True)

#####-------------- Plot and Save --------------#####
save_path = f'../../magneticum-data/data/emcee/prof_{args.field}_halos_all/{run}'
if not os.path.exists(save_path):
	# If the folder does not exist, create it and break the loop
	os.makedirs(save_path)

walkers = sampler.get_chain()
np.save(f'{save_path}/all_walkers.npy', walkers)

chain = sampler.get_chain(flat=True)
log_prob_samples = sampler.get_log_prob(flat=True)

all_samples = np.concatenate((chain, log_prob_samples[:, None]), axis=1)
np.savetxt(f'{save_path}/all_samples.txt', all_samples)



fig, ax = plt.subplots(len(fit_par), 1, figsize=(10, 1.5*len(fit_par)))
ax = ax.flatten()

for i in range(len(fit_par)):
	ax[i].plot(walkers[:, :, i])
	ax[i].set_ylabel(f'${par_latex_names[i]}$')
	ax[i].set_xlabel('Step #')

plt.savefig(f'{save_path}/trace_plot.pdf')

gd_samples = getdist.MCSamples(samples=sampler.get_chain(flat=True, discard=int(0.9*nsteps)), names=fit_par, labels=par_latex_names)

########## Temp ##########

# walkers = np.load(f'{save_path}/all_walkers.npy')

# shape = walkers.shape
# n_burn = int(shape[0]*0.9)
# n_sample = int(shape[1]*(shape[0]-n_burn))

# samples = walkers[n_burn:, :, :].reshape(n_sample, shape[2])
# gd_samples = getdist.MCSamples(samples=samples, names=fit_par, labels=par_latex_names)

########## Compare best-fit profiles ##########
c = ['r', 'b', 'g', 'k']

# Fiducial HMCode profiles
fitter_low_mass.update_param(fit_par_unique, np.array(gd_samples.getMeans())[[0, 2, 4, 5, 6, 7]])
fitter_high_mass.update_param(fit_par_unique, np.array(gd_samples.getMeans())[[1, 3, 4, 5, 6, 8]])

mvir = Mvir_sim*u.Msun/cu.littleh
## Get profile for each halo

(Pe_theory1, rho_theory1, Temp_theory1), r_bestfit = fitter_low_mass.get_Pe_profile_interpolated(mvir[~mask_low_mass], r_bins=r_bins, z=0, return_rho=True, return_Temp=True)

(Pe_theory2, rho_theory2, Temp_theory2), r_bestfit = fitter_high_mass.get_Pe_profile_interpolated(mvir[mask_low_mass], r_bins=r_bins, z=0, return_rho=True, return_Temp=True)

Pe_bestfit = np.concatenate((Pe_theory1, Pe_theory2))
rho_bestfit = np.concatenate((rho_theory1, rho_theory2))
Temp_bestfit = np.concatenate((Temp_theory1, Temp_theory2))


## Plot median Pe profiles
fig, ax = plt.subplots(3, 3, figsize=(18, 12))
### -------------------------- All halos -------------------------####
rho_sim[rho_sim==0] = np.nan
Temp_sim[Temp_sim==0] = np.nan
Pe_sim[Pe_sim==0] = np.nan

ax[0, 0].loglog(r_bins, np.nanmedian(rho_sim, axis=0), label='Magneticum (median)')
ax[0, 0].loglog(r_bestfit, np.median(rho_bestfit, axis=0), label='Best fit (median)')

ax[0, 1].loglog(r_bins, np.nanmedian(Temp_sim, axis=0))
ax[0, 1].loglog(r_bestfit, np.median(Temp_bestfit, axis=0))

ax[0, 2].loglog(r_bins, np.nanmedian(Pe_sim, axis=0))
ax[0, 2].loglog(r_bestfit, np.median(Pe_bestfit, axis=0))

ax[0, 0].legend()

ax[0, 0].set_ylabel('$\\rho_{gas}$ [GeV/cm$^3$]')
ax[0, 1].set_ylabel('Temperature [K]')
ax[0, 2].set_ylabel('$P_e$ [keV/cm$^3$]')

ax[1,1].set_title('All halos')

### -------------------------- High mass halos -------------------------####
ax[1, 0].errorbar(r_bins, np.log(np.nanmedian(rho_sim[mask_low_mass], axis=0)), yerr=sigma_intr_rho_init_high_mass, ls='-.', label='Magneticum (median)')
ax[1, 0].plot(r_bestfit, np.log(np.median(rho_bestfit[mask_low_mass], axis=0).value), ls='-.', label='Best fit (median)')

ax[1, 1].errorbar(r_bins, np.log(np.nanmedian(Temp_sim[mask_low_mass], axis=0)), yerr=sigma_intr_Temp_init_high_mass, ls='-.')
ax[1, 1].plot(r_bestfit, np.log(np.median(Temp_bestfit[mask_low_mass], axis=0).value), ls='-.')


ax[1, 2].errorbar(r_bins, np.log(np.nanmedian(Pe_sim[mask_low_mass], axis=0)), yerr=sigma_intr_Pe_init_high_mass, ls='-.')
ax[1, 2].plot(r_bestfit, np.log(np.median(Pe_bestfit[mask_low_mass], axis=0).value), ls='-.')


ax[1, 0].set_ylabel('$\\rho_{gas}$ [GeV/cm$^3$]')
ax[1, 1].set_ylabel('Temperature [K]')
ax[1, 2].set_ylabel('$P_e$ [keV/cm$^3$]')

ax[1,1].set_title('13<logM<15')

### -------------------------- Low mass halos -------------------------####
ax[2, 0].errorbar(r_bins, np.log(np.nanmedian(rho_sim[~mask_low_mass], axis=0)), yerr=sigma_intr_rho_init_low_mass, ls='-.', label='Magneticum (median)')
ax[2, 0].plot(r_bestfit, np.log(np.median(rho_bestfit[~mask_low_mass], axis=0).value), ls='-.', label='Best fit (median)')

ax[2, 1].errorbar(r_bins, np.log(np.nanmedian(Temp_sim[~mask_low_mass], axis=0)), yerr=sigma_intr_Temp_init_low_mass, ls='-.')
ax[2, 1].plot(r_bestfit, np.log(np.median(Temp_bestfit[~mask_low_mass], axis=0).value), ls='-.')


ax[2, 2].errorbar(r_bins, np.log(np.nanmedian(Pe_sim[~mask_low_mass], axis=0)), yerr=sigma_intr_Pe_init_low_mass, ls='-.')
ax[2, 2].plot(r_bestfit, np.log(np.median(Pe_bestfit[~mask_low_mass], axis=0).value), ls='-.')


ax[2, 0].set_ylabel('$\\rho_{gas}$ [GeV/cm$^3$]')
ax[2, 1].set_ylabel('Temperature [K]')
ax[2, 2].set_ylabel('$P_e$ [keV/cm$^3$]')

ax[2, 0].set_xlabel('$r/Rvir$')
ax[2, 1].set_xlabel('$r/Rvir$')
ax[2, 2].set_xlabel('$r/Rvir$')

ax[2,1].set_title('12<logM<13')

plt.savefig(f'{save_path}/best_fit_profiles.pdf')

#### Discard 0.9*steps and make triangle plot
plt.figure()

g = plots.get_subplot_plotter()
g.triangle_plot(gd_samples, axis_marker_lw=2, marker_args={'lw':2}, line_args={'lw':1}, title_limit=2)
plt.savefig(f'{save_path}/triangle_plot.pdf')
