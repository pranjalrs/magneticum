'''Same as fit_HMCode_profile_all.py but has same parameters
for halo masses
'''
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import glob
import joblib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d
from schwimmbad import MPIPool

import astropy.units as u
import astropy.cosmology.units as cu
import emcee
import getdist
from getdist import plots
import hmf

import sys
sys.path.append('../core/')

from analytic_profile import Profile
from fitting_utils import get_halo_data
import post_processing

import ipdb
#####-------------- Parse Args --------------#####

parser = argparse.ArgumentParser()
parser.add_argument('--field')
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--run', type=str)
parser.add_argument('--niter', type=int)
parser.add_argument('--nsteps', type=int)

args = parser.parse_args()
test = args.test
run, niter = args.run, args.niter
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

def update_sigma_intr(val1, val2):
	global sigma_intr_Pe
	global sigma_intr_rho
	
	sigma_intr_Pe = val1
	sigma_intr_rho = val2


def likelihood(theory_prof, field):
	sim_prof = globals()[field+'_sim']
	num = np.log(sim_prof[mask_low_mass] / theory_prof.value[mask_low_mass])**2
	denom = globals()[f'sigma_intr_{field}_init_high_mass']**2 #+ sigmalnP_sim**2
	chi2 = 0.5*np.sum(num/denom)  #Sum over radial bins

	idx = sim_prof[~mask_low_mass] ==0
	num = np.log(sim_prof[~mask_low_mass] / theory_prof.value[~mask_low_mass])**2
	denom = globals()[f'sigma_intr_{field}_init_low_mass']**2 #+ sigmalnP_sim**2
	num = ma.array(num, mask=idx, fill_value=0)
	chi2 += 0.5*np.sum(num/denom)*weight_factor  # Sum over radial bins

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
		'alpha': [0.1, 2],
		'log10_M0': [10, 17],
		'M0': [1e10, 1e17],
		'beta': [0.4, 0.8],
		'eps1_0': [-0.95, 3],
		'eps2_0': [-0.95, 3],
		'gamma_T': [1.1, 5.5],
		}

fid_val = {'f_H': 0.75,
		'gamma': 1.2,
		'alpha': 1,
		'log10_M0': 14,
		'M0': 1e14,
		'beta': 0.6,
		'eps1_0': 0.2,
		'eps2_0': -0.1,
		'gamma_T': 2,
		}

std_dev = {'f_H': 0.2,
		'gamma': 0.2,
		'alpha': 0.5,
		'log10_M0': 2,
		'M0': 1e12,
		'beta': 0.2,
		'eps1_0': 0.2,
		'eps2_0': 0.2,
		'gamma_T':0.3,
		}


#####-------------- Load Data --------------#####
save_path = f'../../magneticum-data/data/emcee_new/prof_{args.field}_halos_all/{run}'
data_path = '../../magneticum-data/data/profiles_median'
files = glob.glob(f'{data_path}/Box1a/Pe_Pe_Mead_Temp_matter_cdm_gas_v_disp_z=0.00_mvir_1.0E+13_1.0E+16.pkl')
files += glob.glob(f'{data_path}/Box2/Pe_Pe_Mead_Temp_matter_cdm_gas_v_disp_z=0.00_mvir_1.0E+12_1.0E+13.pkl')

#####-------------- Weight for low mass halos --------------#####

N_high = len(joblib.load(files[0]))
N_low = len(joblib.load(files[1]))

theory_mf = hmf.MassFunction(cosmo_model="WMAP7", mdef_model='SOVirial', z=0, sigma_8=0.809)
delta_m = np.array(0.01*theory_mf.m)

# 10^13<m<10^16
index = np.where((theory_mf.m<10**16) & (theory_mf.m>10**13.))  # Select dn/dM in mass bin
counts = np.sum(theory_mf.dndm[index]*delta_m[index])  # Sum (dn/dM * delta M)
N_high_hmf = counts

# 10^11<m<10^13
index = np.where((theory_mf.m<10**13) & (theory_mf.m>10**12))  # Select dn/dM in mass bin
counts = np.sum(theory_mf.dndm[index]*delta_m[index])  # Sum (dn/dM * delta M)
N_low_hmf = counts

N_expected = (N_low_hmf/N_high_hmf) * N_high
weight_factor = N_expected/N_low

print(f'Expected # of low mass halos given {N_high} high mass halos = {N_expected}')
print(f'# of low mass halos sampled = {N_low}')
print(f'Weight factor = {weight_factor}')

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

		## rho dm
		rho_dm_prof_interp, rho_dm_rescale_interp, unc_rho_dm = get_halo_data(halo, 'rho_dm', r_bins, return_sigma=True)
		if rho_dm_prof_interp is None or rho_dm_rescale_interp is None: continue

		## Pressure	
		Pe_prof_interp, Pe_rescale_interp, unc_Pe = get_halo_data(halo, 'Pe', r_bins, return_sigma=True)
		if Pe_prof_interp is None or Pe_rescale_interp is None: continue

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

# Now we need to sort halos in order of increasing mass
# Since this is what the scipy interpolator expects
Mvir_sim = np.array(Mvir_sim, dtype='float32')
sorting_indices = np.argsort(Mvir_sim)

rho_dm_sim = np.array(rho_dm_sim, dtype='float32')[sorting_indices]
Pe_sim = np.array(Pe_sim, dtype='float32')[sorting_indices]
# sigmalnP_sim = np.array(sigmalnP_sim, dtype='float32')[sorting_indices]
rho_sim = np.array(rho_sim, dtype='float32')[sorting_indices]
# sigmalnrho_sim = np.array(sigmalnrho_sim, dtype='float32')[sorting_indices]
Temp_sim = np.array(Temp_sim, dtype='float32')[sorting_indices]
Mvir_sim = Mvir_sim[sorting_indices]


# Now compute intrinsic scatter
# Since low mass halos have a large scatter we compute it separately for them
mask_low_mass = Mvir_sim>=10**(13)

#---------------------- rho_dm ----------------------#
rho_dm_rescale = np.vstack(rho_dm_rescale)[sorting_indices]
# High mass
median_prof = np.median(rho_dm_rescale[mask_low_mass], axis=0)
sigma_intr_rho_dm_init_high_mass = get_scatter(np.log(rho_dm_rescale[mask_low_mass]), np.log(median_prof))
# Low mass
median_prof = np.median(rho_dm_rescale[~mask_low_mass], axis=0)
sigma_intr_rho_dm_init_low_mass = get_scatter(np.log(rho_dm_rescale[~mask_low_mass]), np.log(median_prof))

#---------------------- Pressure ----------------------#
Pe_rescale = np.vstack(Pe_rescale)[sorting_indices]
# High mass
median_prof = np.median(Pe_rescale[mask_low_mass], axis=0)
sigma_intr_Pe_init_high_mass = get_scatter(np.log(Pe_rescale[mask_low_mass]), np.log(median_prof))
# Low mass
median_prof = np.median(Pe_rescale[~mask_low_mass], axis=0)
sigma_intr_Pe_init_low_mass = get_scatter(np.log(Pe_rescale[~mask_low_mass]), np.log(median_prof))


#---------------------- rho ----------------------#
rho_rescale = np.vstack(rho_rescale)[sorting_indices]
# High mass
median_prof = np.median(rho_rescale[mask_low_mass], axis=0)
sigma_intr_rho_init_high_mass = get_scatter(np.log(rho_rescale[mask_low_mass]), np.log(median_prof))
# Low mass
median_prof = np.median(rho_rescale[~mask_low_mass], axis=0)
sigma_intr_rho_init_low_mass = get_scatter(np.log(rho_rescale[~mask_low_mass]), np.log(median_prof))
#update_sigma_intr(sigma_intr_Pe_init, sigma_intr_rho_init)


#---------------------- Temperature ----------------------#
Temp_rescale = np.vstack(Temp_rescale)[sorting_indices]
# High mass
median_prof = np.median(Temp_rescale[mask_low_mass], axis=0)
sigma_intr_Temp_init_high_mass = get_scatter(np.log(Temp_rescale[mask_low_mass]), np.log(median_prof))
# Low mass
median_prof = np.median(Temp_rescale[~mask_low_mass], axis=0)
sigma_intr_Temp_init_low_mass = get_scatter(np.log(Temp_rescale[~mask_low_mass]), np.log(median_prof))

sigma_intr_rho_dm_init_high_mass[-1] = 0.1
sigma_intr_rho_dm_init_low_mass[-1] = 0.1

sigma_intr_Pe_init_high_mass[-1] = 0.1
sigma_intr_Pe_init_low_mass[-1] = 0.1

sigma_intr_rho_init_high_mass[-1] = 0.1
sigma_intr_rho_init_low_mass[-1] = 0.1

sigma_intr_Temp_init_high_mass[-1] = 0.1
sigma_intr_Temp_init_low_mass[-1] = 0.1


print('Finished processing simulation data...')
print(f'# of low mass halos = {len(rho_dm_sim[~mask_low_mass])}')
print(f'# of high mass halos = {len(rho_dm_sim[mask_low_mass])}')

#####-------------- Prepare for MCMC --------------#####
fitter = Profile(use_interp=True, mmin=Mvir_sim.min()-1e10, mmax=Mvir_sim.max()+1e10)

print('Initialized profile fitter ...')
fit_par = ['gamma', 'alpha', 'log10_M0', 'eps1_0', 'eps2_0', 'gamma_T']
par_latex_names = ['\Gamma', '\\alpha', '\log_{10}M_0', '\epsilon_1', '\epsilon_2', '\Gamma_\mathrm{T}']

#fit_par = ['log10_M0', 'eps1_0', 'eps2_0']
#par_latex_names = ['\log_{10}M_0', '\epsilon_1', '\epsilon_2']

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


print(f'Using Likelihood for {field} field(s)')


def test_interpolator(walkers):
	test_interp = Profile(use_interp=True, mmin=Mvir_sim.min()-1e10, mmax=Mvir_sim.max()+1e10, interp_error_tol = 0.1)

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
save_path = f'../../magneticum-data/data/emcee/prof_{args.field}_halos_all/{run}'
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

if 'all' in field: nfield=3
else: nfield = len(field)

chi2 = -all_samples[idx, -5] / (len(Mvir_sim)*len(r_bins)*nfield - len(fit_par))
chi2_rho_dm = -all_samples[idx, -4] / (len(Mvir_sim)*len(r_bins)*nfield - len(fit_par))
chi2_rho = -all_samples[idx, -3] / (len(Mvir_sim)*len(r_bins)*nfield - len(fit_par))
chi2_Temp = -all_samples[idx, -2] / (len(Mvir_sim)*len(r_bins)*nfield - len(fit_par))
chi2_Pe = -all_samples[idx, -1] / (len(Mvir_sim)*len(r_bins)*nfield - len(fit_par))

fig, ax = plt.subplots(len(fit_par), 1, figsize=(10, 1.5*len(fit_par)))
ax = ax.flatten()

for i in range(len(fit_par)):
	ax[i].plot(walkers[:, :, i])
	ax[i].set_ylabel(f'${par_latex_names[i]}$')
	ax[i].set_xlabel('Step #')

plt.savefig(f'{save_path}/trace_plot_{niter}.pdf')

gd_samples = getdist.MCSamples(samples=sampler.get_chain(flat=True, discard=int(0.9*nsteps)), names=fit_par, labels=par_latex_names)


#### Discard 0.9*steps and make triangle plot

gd_samples = getdist.MCSamples(samples=sampler.get_chain(flat=True, discard=int(0.9*nsteps)), names=fit_par, labels=par_latex_names)

np.save(f'{save_path}/all_walkers_{niter}.npy', walkers)
np.savetxt(f'{save_path}/all_samples_{niter}.txt', all_samples)

np.savetxt(f'{save_path}/sigma_intr_high_mass_{niter}.txt',  np.column_stack((sigma_intr_rho_dm_init_high_mass, sigma_intr_rho_init_high_mass, sigma_intr_Temp_init_high_mass, sigma_intr_Pe_init_high_mass)))

np.savetxt(f'{save_path}/sigma_intr_low_mass_{niter}.txt',  np.column_stack((sigma_intr_rho_dm_init_low_mass, sigma_intr_rho_init_low_mass, sigma_intr_Temp_init_low_mass, sigma_intr_Pe_init_low_mass)))

np.savetxt(f'{save_path}/best_params_{niter}.txt', best_params, header='\t'.join(fit_par))

########## Compare best-fit profiles ##########
# Fiducial HMCode profiles
fitter.update_param(fit_par, best_params)
rho_dm_bestfit, r = fitter.get_rho_dm_profile_interpolated(Mvir_sim*u.Msun/cu.littleh, z=0)
(Pe_bestfit, rho_bestfit, Temp_bestfit), r_bestfit = fitter.get_Pe_profile_interpolated(Mvir_sim*u.Msun/cu.littleh, z=0, return_rho=True, return_Temp=True)



n=1000
inds1 = np.sort(np.random.choice(np.arange(len(Mvir_sim)), n, replace=False))
inds2 = np.sort(np.random.choice(np.arange(np.sum(mask_low_mass)), n, replace=False)) # Random High mass
inds3 = np.sort(np.random.choice(np.arange(np.sum(~mask_low_mass)), n, replace=False)) # Random Low mass

c = plt.cm.winter(np.linspace(0,1,len(Mvir_sim)))

## Plot median Pe profiles
fig, ax = plt.subplots(3, 4, figsize=(18, 12))


for idx1, idx2, idx3 in zip(inds1, inds2, inds3):
    
    m = Mvir_sim[idx1]
    j = np.where(Mvir_sim==m)[0][0]
    ax[0, 0].plot(r_bins, np.log(rho_dm_sim[idx1]), c=c[j], alpha=0.2)
    ax[0, 1].plot(r_bins, np.log(rho_sim[idx1]), c=c[j], alpha=0.2)
    ax[0, 2].plot(r_bins, np.log(Temp_sim[idx1]), c=c[j], alpha=0.2)
    ax[0, 3].plot(r_bins, np.log(Pe_sim[idx1]), c=c[j], alpha=0.2)

    m = Mvir_sim[mask_low_mass][idx2]
    j = np.where(Mvir_sim==m)[0][0]
    ax[1, 0].plot(r_bins, np.log(rho_dm_sim[mask_low_mass][idx2]), c=c[j], alpha=0.2)
    ax[1, 1].plot(r_bins, np.log(rho_sim[mask_low_mass][idx2]), c=c[j], alpha=0.2)
    ax[1, 2].plot(r_bins, np.log(Temp_sim[mask_low_mass][idx2]), c=c[j], alpha=0.2)
    ax[1, 3].plot(r_bins, np.log(Pe_sim[mask_low_mass][idx2]), c=c[j], alpha=0.2)
    
    m = Mvir_sim[~mask_low_mass][idx3]
    j = np.where(Mvir_sim==m)[0][0]
    ax[2, 0].plot(r_bins, np.log(rho_dm_sim[~mask_low_mass][idx3]), c=c[j], alpha=0.2)
    ax[2, 1].plot(r_bins, np.log(rho_sim[~mask_low_mass][idx3]), c=c[j], alpha=0.2)
    ax[2, 2].plot(r_bins, np.log(Temp_sim[~mask_low_mass][idx3]), c=c[j], alpha=0.2)
    ax[2, 3].plot(r_bins, np.log(Pe_sim[~mask_low_mass][idx3]), c=c[j], alpha=0.2)

scalarmappaple = ScalarMappable(cmap=plt.cm.winter)
scalarmappaple.set_array(np.log10(Mvir_sim))

# Add colorbar to the plot
# cbar = plt.colorbar(scalarmappaple, ax=ax[0], location='top')
divider = make_axes_locatable(ax[0, 3])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(scalarmappaple, cax=cax)
plt.sca(ax[0, 3])
# colorbar(scalarmappaple)
cbar.set_label('log M')


### -------------------------- All halos -------------------------####
col1, col2 = 'red', 'orange'
rho_sim[rho_sim==0] = np.nan
Temp_sim[Temp_sim==0] = np.nan
Pe_sim[Pe_sim==0] = np.nan

ax[0, 0].plot(r_bins, np.log(np.nanmedian(rho_dm_sim, axis=0)), c=col1,label='Magneticum (median)')
ax[0, 0].plot(r_bestfit, np.log(np.median(rho_dm_bestfit, axis=0).value), c=col2, label='Best fit (median)')

ax[0, 1].plot(r_bins, np.log(np.nanmedian(rho_sim, axis=0)), c=col1,label='Magneticum (median)')
ax[0, 1].plot(r_bestfit, np.log(np.median(rho_bestfit, axis=0).value), c=col2, label='Best fit (median)')

ax[0, 2].plot(r_bins, np.log(np.nanmedian(Temp_sim, axis=0)), c=col1,)
ax[0, 2].plot(r_bestfit, np.log(np.median(Temp_bestfit, axis=0).value), c=col2, )

ax[0, 3].plot(r_bins, np.log(np.nanmedian(Pe_sim, axis=0)), c=col1,)
ax[0, 3].plot(r_bestfit, np.log(np.median(Pe_bestfit, axis=0).value), c=col2, )

ax[0, 0].legend()


ax[1,1].set_title('All halos')

### -------------------------- High mass halos -------------------------####
ax[1, 0].errorbar(r_bins, np.log(np.nanmedian(rho_dm_sim[mask_low_mass], axis=0)), yerr=sigma_intr_rho_dm_init_high_mass, c=col1, ls='-.', label='Magneticum (median)')
ax[1, 0].plot(r_bestfit, np.log(np.median(rho_dm_bestfit[mask_low_mass], axis=0).value), c=col2, ls='-.', label='Best fit (median)')

ax[1, 1].errorbar(r_bins, np.log(np.nanmedian(rho_sim[mask_low_mass], axis=0)), yerr=sigma_intr_rho_init_high_mass, c=col1, ls='-.', label='Magneticum (median)')
ax[1, 1].plot(r_bestfit, np.log(np.median(rho_bestfit[mask_low_mass], axis=0).value), c=col2, ls='-.', label='Best fit (median)')

ax[1, 2].errorbar(r_bins, np.log(np.nanmedian(Temp_sim[mask_low_mass], axis=0)), yerr=sigma_intr_Temp_init_high_mass, c=col1, ls='-.')
ax[1, 2].plot(r_bestfit, np.log(np.median(Temp_bestfit[mask_low_mass], axis=0).value), c=col2, ls='-.')


ax[1, 3].errorbar(r_bins, np.log(np.nanmedian(Pe_sim[mask_low_mass], axis=0)), yerr=sigma_intr_Pe_init_high_mass, c=col1, ls='-.')
ax[1, 3].plot(r_bestfit, np.log(np.median(Pe_bestfit[mask_low_mass], axis=0).value), c=col2, ls='-.')


ax[1,1].set_title('13<logM<15')

### -------------------------- Low mass halos -------------------------####
ax[2, 0].errorbar(r_bins, np.log(np.nanmedian(rho_dm_sim[~mask_low_mass], axis=0)), yerr=sigma_intr_rho_dm_init_low_mass, c=col1,  ls='-.', label='Magneticum (median)')
ax[2, 0].plot(r_bestfit, np.log(np.median(rho_dm_bestfit[~mask_low_mass], axis=0).value), c=col2, ls='-.', label='Best fit (median)')

ax[2, 1].errorbar(r_bins, np.log(np.nanmedian(rho_sim[~mask_low_mass], axis=0)), yerr=sigma_intr_rho_init_low_mass, c=col1, ls='-.', label='Magneticum (median)')
ax[2, 1].plot(r_bestfit, np.log(np.median(rho_bestfit[~mask_low_mass], axis=0).value), c=col2, ls='-.', label='Best fit (median)')

ax[2, 2].errorbar(r_bins, np.log(np.nanmedian(Temp_sim[~mask_low_mass], axis=0)), yerr=sigma_intr_Temp_init_low_mass, c=col1, ls='-.')
ax[2, 2].plot(r_bestfit, np.log(np.median(Temp_bestfit[~mask_low_mass], axis=0).value), c=col2, ls='-.')


ax[2, 3].errorbar(r_bins, np.log(np.nanmedian(Pe_sim[~mask_low_mass], axis=0)), yerr=sigma_intr_Pe_init_low_mass, c=col1, ls='-.')
ax[2, 3].plot(r_bestfit, np.log(np.median(Pe_bestfit[~mask_low_mass], axis=0).value), c=col2, ls='-.')


ax[2, 0].set_xlabel('$r/Rvir$')
ax[2, 1].set_xlabel('$r/Rvir$')
ax[2, 2].set_xlabel('$r/Rvir$')
ax[2, 3].set_xlabel('$r/Rvir$')

ax[2,1].set_title('12<logM<13')

for i in range(3):
	ax[i, 0].set_ylabel('$\\rho_{DM}$ [GeV/cm$^3$]')
	ax[i, 1].set_ylabel('$\\rho_{gas}$ [GeV/cm$^3$]')
	ax[i, 2].set_ylabel('Temperature [K]')
	ax[i, 3].set_ylabel('$P_e$ [keV/cm$^3$]')

fig.suptitle('Total $\chi^2_{\mathrm{d.o.f}}=$'+f'${chi2:.2f}$')
ax[0, 0].set_title('DM Density $\chi^2_{\mathrm{d.o.f}}=$'+f'${chi2_rho_dm:.2f}$')
ax[0, 1].set_title('Density $\chi^2_{\mathrm{d.o.f}}=$'+f'${chi2_rho:.2f}$')
ax[0, 2].set_title('Temperature  $\chi^2_{\mathrm{d.o.f}}=$'+f'${chi2_Temp:.2f}$')
ax[0, 3].set_title('Pressure $\chi^2_{\mathrm{d.o.f}}=$'+f'${chi2_Pe:.2f}$')

plt.savefig(f'{save_path}/best_fit_profiles.pdf')


#### Discard 0.9*steps and make triangle plot
plt.figure()

g = plots.get_subplot_plotter()
g.triangle_plot(gd_samples, axis_marker_lw=2, marker_args={'lw':2}, line_args={'lw':1}, title_limit=2)
plt.savefig(f'{save_path}/triangle_plot.pdf')
