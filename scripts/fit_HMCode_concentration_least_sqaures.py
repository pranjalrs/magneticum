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
parser.add_argument('--field', choices=['rho_dm'])
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


def likelihood(theory_prof, field):
	sim_prof = globals()['this_'+field+'_sim'] # Simulation profile
	sim_sigma_lnprof = globals()[f'this_sigma_ln{field}'] # Measurement uncertainty

	num = np.log(sim_prof / theory_prof.value)**2

	idx = sim_prof==0
	num = ma.array(num, mask=idx, fill_value=0)
    
	denom = globals()[f'sigma_intr_{field}']**2 + sim_sigma_lnprof**2
	chi2 = 0.5*np.sum(num/denom)  #Sum over radial bins

	if not np.isfinite(chi2):
		return -np.inf

	return -chi2

def joint_likelihood(x, mass_list, z=0):
	for i in range(len(fit_par)):
		lb, ub = bounds[fit_par[i]]
		if x[i]<lb or x[i]>ub:
			return -np.inf

	fitter.update_param(fit_par, x)

	mvir = mass_list*u.Msun/cu.littleh
	## Get profile for each halo
	rho_dm_theory, r = fitter.get_rho_dm_profile(mvir, r_bins=r_bins, z=z)

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
	return loglike

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
                'n_nt': [0, 2],
                 'conc_param': [0, 20]}

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
                'n_nt':1,
                'conc_param': 10}

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
                'alpha_nt': 0.4,
                'n_nt': 0.4,
                'conc_param': 8}

#####-------------- Load Data --------------#####
#save_path = f'../../magneticum-data/data/emcee_magneticum_cM/prof_{args.field}_halos_bin/{run}'
#save_path = f'../../magneticum-data/data/emcee_new/prof_{args.field}_halos_bin/{run}'
save_path = f'../../magneticum-data/data/emcee_concentration/prof_{args.field}_halos_bin/{run}'

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

# Since low mass halos have a large scatter we compute it separately for them

# Now we need to sort halos in order of increasing mass
# Since this is what the scipy interpolator expects
Mvir_sim = np.array(Mvir_sim, dtype='float32')
sorting_indices = np.argsort(Mvir_sim)
Mvir_sim = Mvir_sim[sorting_indices]

mask = (Mvir_sim>=10**(mmin)) & (Mvir_sim<10**mmax)
print(f'{np.log10(Mvir_sim[mask].min()):.2f}, {np.log10(Mvir_sim[mask].max()):.2f}')

idx = np.arange(10)
r_bins = r_bins[idx]


rho_dm_sim = np.array(rho_dm_sim, dtype='float32')[sorting_indices][:, idx]
sigma_lnrho_dm = np.vstack(sigma_lnrho_dm)[sorting_indices][:, idx]

rho_dm_sim = rho_dm_sim[mask]
sigma_lnrho_dm = sigma_lnrho_dm[mask]
Mvir_sim = Mvir_sim[mask]

rho_dm_rescale = np.vstack(rho_dm_rescale)[sorting_indices][:, idx]
median_prof = np.median(rho_dm_rescale[mask], axis=0)
sigma_intr_rho_dm = get_scatter(np.log(rho_dm_rescale[mask]), np.log(median_prof))

sigma_intr_rho_dm[-1] = 0.1

print('Finished processing simulation data...')
print(f'Using {np.sum(mask)} halos for fit...')

#####-------------- Prepare for MCMC --------------#####
fitter = Profile(use_interp=False, imass_conc=2)
print('Initialized profile fitter ...')


fit_par = ['conc_param']
par_latex_names = ['c(M)']

starting_point = [fid_val[k] for k in fit_par]
std = [std_dev[k] for k in fit_par]
these_bounds = [bounds[k] for k in fit_par][0]
print(f'Using Likelihood for {field} field(s)')


#####-------------- RUN MCMC --------------#####

result = []

fig = plt.figure()

for i in tqdm(range(sum(mask))):
    this_rho_dm_sim = rho_dm_sim[i]
    this_halo_mass = Mvir_sim[i]
    this_sigma_lnrho_dm = sigma_lnrho_dm[i]
    sol = scipy.optimize.least_squares(joint_likelihood, starting_point, bounds=these_bounds, args=([this_halo_mass]))
#     sol = scipy.optimize.least_squares(joint_likelihood, starting_point, bounds=these_bounds, args=([this_halo_mass]))

    result.append([this_halo_mass, sol.x[0], sol.cost])
    
    fitter.update_param(fit_par, sol.x)
    rho_dm_theory, r = fitter.get_rho_dm_profile(this_halo_mass*u.Msun/cu.littleh, r_bins=r_bins, z=0)
    plt.plot(r, this_rho_dm_sim/rho_dm_theory.value-1, c='dodgerblue', alpha=0.2);

plt.plot(r_bins, sigma_intr_rho_dm, c='k', ls=':', label='$\sigma_{int}$')
plt.plot(r_bins, -sigma_intr_rho_dm, c='k', ls=':')
plt.savefig('test_residual.pdf')
plt.close()

result = np.vstack(result)
result[:, 2][result[:, 2]>50] = 50


plt.figure()
plt.scatter(result[:,0], result[:, 1], c=result[:, 2], alpha=0.5)
plt.colorbar()
plt.ylabel('Concentration')
plt.xlabel('Virial Mass [Msun]')
plt.xscale('log')
plt.savefig('test_conc.pdf')
plt.close()

plt.figure()
plt.scatter(result[:,0], result[:, 2], c='dodgerblue', alpha=0.2)
plt.ylabel('Cost Fn.')
plt.xlabel('Virial Mass [Msun]')
plt.xscale('log')
plt.savefig('test_cost.pdf')
plt.close()
