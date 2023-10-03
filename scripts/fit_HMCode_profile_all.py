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

#####-------------- Parse Args --------------#####

parser = argparse.ArgumentParser()
parser.add_argument('--field')
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--run', type=int)
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
	profile_interp = nan_interp(r, profile)(r_bins)
	profile_rescale_interp = nan_interp(r, profile)(r_bins)

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
	chi2 = 0.5*np.sum(num/denom)  # Sum over radial bins

	return -chi2


def joint_likelihood(x, mass_list, z=0):
	for i in range(len(fit_par)):
		lb, ub = bounds[fit_par[i]]
		if x[i]<lb or x[i]>ub:
			return -np.inf

	fitter.update_param(fit_par, x)

	mvir = mass_list*u.Msun/cu.littleh
	## Get profile for each halo
	(Pe_theory, rho_theory, Temp_theory), r = fitter.get_Pe_profile_interpolated(mvir, r_bins=r_bins, z=z, return_rho=True, return_Temp=True)

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
		'alpha': [0.1, 2],
		'log10_M0': [10, 17],
		'M0': [1e10, 1e17],
		'beta': [0.4, 0.8],
		'eps1_0': [-0.95, 3],
		'eps2_0': [-0.95, 3],
		'gamma_T': [1.1, 5],
		 'b': [-3, 3],
		 'c': [-4, 4]}

fid_val = {'f_H': 0.75,
		'gamma': 1.2,
		'alpha': 1,
		'log10_M0': 14,
		'M0': 1e14,
		'beta': 0.6,
		'eps1_0': 0.2,
		'eps2_0': -0.1,
	'gamma_T': 2,
		  'b': 1,
		  'c': 1}

std_dev = {'f_H': 0.2,
		'gamma': 0.2,
		'alpha': 0.5,
		'log10_M0': 2,
		'M0': 1e12,
		'beta': 0.2,
		'eps1_0': 0.2,
		'eps2_0': 0.2,
	'gamma_T':0.3,
		  'b': 0.5,
		  'c':0.5}


#####-------------- Load Data --------------#####
data_path = '../../magneticum-data/data/profiles_median'
files = glob.glob(f'{data_path}/Box1a/Pe_Pe_Mead_Temp_matter_cdm_gas_z=0.00_mvir_3.2E+13_1.0E+16.pkl')
files += glob.glob(f'{data_path}/Box2/Pe_Pe_Mead_Temp_matter_cdm_gas_z=0.00_mvir_1.0E+11_1.0E+13.pkl')

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
		if Pe_prof_interp is None or Pe_prof_rescale_interp is None: continue
		Pe_sim.append(Pe_prof_interp)
		Pe_rescale.append(Pe_rescale_interp)

		## Gas density
		rho_prof_interp, rho_rescale_interp = get_halo_data(halo, 'rho')
		if Pe_prof_interp is None or Pe_prof_rescale_interp is None: continue
		rho_sim.append(rho_prof_interp)
		rho_rescale.append(rho_rescale_interp)


		## Temperature
		Temp_prof_interp, Temp_rescale_interp = get_halo_data(halo, 'Temp')
		if Temp_prof_interp is None or Temp_prof_rescale_interp is None: continue
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
mask_low_mass = Mvir_sim>=10**(13.5)

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
fitter = Profile(use_interp=True, mmin=Mvir_sim.min()-1e10, mmax=Mvir_sim.max()+1e10)
print('Initialized profile fitter ...')
fit_par = ['gamma', 'alpha', 'log10_M0', 'beta', 'eps1_0', 'eps2_0', 'gamma_T', 'b', 'c']
par_latex_names = ['\Gamma', '\\alpha', '\log_{10}M_0', '\\beta', '\epsilon_1', '\epsilon_2', '\Gamma_\mathrm{T}', 'b', 'c']

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

#####-------------- RUN MCMC --------------#####
print('Running MCMC..')

if test is False:
	with MPIPool() as pool:
		if not pool.is_master():
			pool.wait()
			sys.exit(0)
		
		print('Running MCMC with MPI...')
		sampler = emcee.EnsembleSampler(nwalkers, ndim, joint_likelihood, pool=pool, args=[Mvir_sim])
		sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=True)

else:
	print('Running MCMC...')
	sampler = emcee.EnsembleSampler(nwalkers, ndim, joint_likelihood, args=[Mvir_sim])
	sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=True)

#####-------------- Plot and Save --------------#####
save_path = f'../../magneticum-data/data/emcee/fit_{args.field}_all_halos/run{run}'
if not os.path.exists(save_path):
	# If the folder does not exist, create it and break the loop
	os.makedirs(save_path)

walkers = sampler.get_chain()
np.save(f'{save_path}/all_walkers.npy', walkers)

chain = sampler.get_chain(flat=True)
log_prob_samples = sampler.get_log_prob(flat=True)

all_samples = np.concatenate((chain, log_prob_samples[:, None]), axis=1)
np.savetxt(f'{save_path}/all_samples.txt', all_samples)

#sigma_data = np.column_stack((sigma_intr_Pe_init, sigma_intr_Pe, sigma_intr_rho_init, sigma_intr_rho))
#np.savetxt(f'{save_path}/sigma_intr.txt',  sigma_data, header='initial Pe \t final \t initial rho \t final')


fig, ax = plt.subplots(len(fit_par), 1, figsize=(10, 1.5*len(fit_par)))
ax = ax.flatten()

for i in range(len(fit_par)):
	ax[i].plot(walkers[:, :, i])
	ax[i].set_ylabel(f'${par_latex_names[i]}$')
	ax[i].set_xlabel('Step #')

plt.savefig(f'{save_path}/trace_plot.pdf')

#### Discard 0.9*steps and make triangle plot
plt.figure()

gd_samples = getdist.MCSamples(samples=sampler.get_chain(flat=True, discard=int(0.9*nsteps)), names=fit_par, labels=par_latex_names)
g = plots.get_subplot_plotter()
g.triangle_plot(gd_samples, axis_marker_lw=2, marker_args={'lw':2}, line_args={'lw':1}, title_limit=2)
plt.savefig(f'{save_path}/triangle_plot.pdf')

########## Compare best-fit profiles ##########
c = ['r', 'b', 'g', 'k']

bins = [13.5, 14, 14.5, 15]
# Fiducial HMCode profiles
fitter.update_param(fit_par, gd_samples.getMeans())
(Pe_bestfit, rho_bestfit, Temp_bestfit), r_bestfit = fitter.get_Pe_profile_interpolated(Mvir_sim*u.Msun/cu.littleh, z=0, return_rho=True, return_Temp=True)


## Plot median Pe profiles
fig, ax = plt.subplots(3, 3, figsize=(18, 12))
### -------------------------- All halos -------------------------####
ax[0, 0].loglog(r_bins, np.median(rho_sim, axis=0), label='Median sim prof')
ax[0, 0].loglog(r_bestfit, np.median(rho_bestfit, axis=0), label='Median theory prof')

ax[0, 2].loglog(r_bins, np.median(Temp_sim, axis=0), label='Median sim prof')
ax[0, 2].loglog(r_bestfit, np.median(Temp_bestfit, axis=0), label='Median theory prof')

ax[0, 1].loglog(r_bins, np.median(Pe_sim, axis=0), label='Median sim prof')
ax[0, 1].loglog(r_bestfit, np.median(Pe_bestfit, axis=0), label='Median theory prof')

ax[0, 0].legend()

ax[0, 0].set_ylabel('$\\rho_{gas}$ [GeV/cm$^3$]')
ax[0, 1].set_ylabel('Temperature [K]')
ax[0, 2].set_ylabel('$P_e$ [keV/cm$^3$]')

ax[1,1].set_title('All halos')

### -------------------------- High mass halos -------------------------####
ax[1, 0].loglog(r_bins, np.median(rho_sim[mask_low_mass], ls=':', axis=0))
ax[1, 0].loglog(r_bestfit, np.median(rho_bestfit[mask_low_mass], ls=':', axis=0))

ax[1, 1].loglog(r_bins, np.median(Temp_sim[mask_low_mass], ls=':', axis=0))
ax[1, 1].loglog(r_bestfit, np.median(Temp_bestfit[mask_low_mass], ls=':', axis=0))

ax[1, 2].loglog(r_bins, np.median(Pe_sim[mask_low_mass], ls=':', axis=0))
ax[1, 2].loglog(r_bestfit, np.median(Pe_bestfit[mask_low_mass], ls=':', axis=0))


ax[1, 0].set_ylabel('$\\rho_{gas}$ [GeV/cm$^3$]')
ax[1, 1].set_ylabel('Temperature [K]')
ax[1, 2].set_ylabel('$P_e$ [keV/cm$^3$]')

ax[1,1].set_title('13.5<logM<15')

### -------------------------- Low mass halos -------------------------####
ax[2, 0].loglog(r_bins, np.median(rho_sim[~mask_low_mass], ls='-.', axis=0))
ax[2, 0].loglog(r_bestfit, np.median(rho_bestfit[~mask_low_mass], ls='-.', axis=0))

ax[2, 1].loglog(r_bins, np.median(Temp_sim[~mask_low_mass], ls='-.', axis=0))
ax[2, 1].loglog(r_bestfit, np.median(Temp_bestfit[~mask_low_mass], ls='-.', axis=0))

ax[2, 2].loglog(r_bins, np.median(Pe_sim[~mask_low_mass], ls='-.', axis=0))
ax[2, 2].loglog(r_bestfit, np.median(Pe_bestfit[~mask_low_mass], ls='-.', axis=0))


ax[2, 0].set_ylabel('$\\rho_{gas}$ [GeV/cm$^3$]')
ax[2, 1].set_ylabel('Temperature [K]')
ax[2, 2].set_ylabel('$P_e$ [keV/cm$^3$]')

ax[2, 0].set_xlabel('$r/Rvir$')
ax[2, 1].set_xlabel('$r/Rvir$')
ax[2, 2].set_xlabel('$r/Rvir$')

ax[2,1].set_title('11<logM<13.5')

plt.savefig(f'{save_path}/best_fit_profiles.pdf')