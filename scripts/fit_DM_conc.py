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
parser.add_argument('--nsteps', type=int)
parser.add_argument('--mmin', type=float)
parser.add_argument('--mmax', type=float)
args = parser.parse_args()
test = args.test
run, niter = args.run, args.niter
mmin, mmax = args.mmin, args.mmax
field = ['rho_dm']

#####-------------- Likelihood --------------#####
def likelihood(theory_prof, field):
	sim_prof = globals()['this_'+field+'_sim'] # Simulation profile
	sim_sigma_lnprof = globals()[f'this_sigma_ln{field}']# Measurement uncertainty

	num = np.log(sim_prof / theory_prof.value)**2

	idx = sim_prof==0
	num = ma.array(num, mask=idx, fill_value=0)
	
	denom = 1.# sim_sigma_lnprof**2
	chi2 = 0.5*np.sum(num/denom)  #Sum over radial bins

	if not np.isfinite(chi2):
		return -np.inf

	return chi2

def joint_likelihood(x, mass, r_bins, z=0):
	for i in range(len(fit_par)):
		lb, ub = bounds[fit_par[i]]
		if x[i]<lb or x[i]>ub:
			return -np.inf

	fitter.update_param(fit_par, x)

	mvir = mass*u.Msun/cu.littleh
	## Get profile for each halo
	rho_dm_theory, r = fitter.get_rho_dm_profile(mvir, r_bins=r_bins, z=z)

	like_rho_dm, like_rho, like_Temp, like_Pe = 0., 0., 0., 0.

	if 'rho_dm' in field:
		like_rho_dm = likelihood(rho_dm_theory, 'rho_dm')

	loglike = like_rho_dm + like_rho + like_Temp + like_Pe
	return loglike

bounds = {'lognorm_rho': [-4., 10.],
			'conc_param': [0, 12]}

fid_val = {'lognorm_rho': -2.5,
			'conc_param': 2}

std_dev = {'lognorm_rho': 0.3,
			'conc_param': 8}

#####-------------- Load Data --------------#####
#save_path = f'../../magneticum-data/data/emcee_magneticum_cM/prof_{args.field}_halos_bin/{run}'
#save_path = f'../../magneticum-data/data/emcee_new/prof_{args.field}_halos_bin/{run}'
save_path = f'../../magneticum-data/data/emcee_concentration/prof_{field}_halos_bin/{run}'

data_path = '../../magneticum-data/data/profiles_median'
files = glob.glob(f'{data_path}/Box1a/Pe_Pe_Mead_Temp_matter_cdm_gas_v_disp_z=0.00_mvir_1.0E+13_1.0E+16.pkl')
files += glob.glob(f'{data_path}/Box2/Pe_Pe_Mead_Temp_matter_cdm_gas_v_disp_z=0.00_mvir_1.0E+12_1.0E+13.pkl')

#files = [f'{data_path}/Box1a/Pe_Pe_Mead_Temp_matter_cdm_gas_z=0.00_mvir_1.0E+13_1.0E+15_coarse.pkl']
#files += [f'{data_path}/Box2/Pe_Pe_Mead_Temp_matter_cdm_gas_z=0.00_mvir_1.0E+12_1.0E+13_coarse.pkl']

## We will interpolate all measured profiles to the same r_bins as 
## the analytical profile for computational efficiency
rho_dm_sim= []
sigma_lnrho_dm = []
r_bins_sim = []
Mvir_sim = []

for f in files:
	this_prof_data = joblib.load(f)
	for halo in this_prof_data:

		r = halo['fields']['cdm'][1]/halo['rvir']
		profile = halo['fields']['cdm'][0]
		sigma_lnprof = halo['fields']['cdm'][3]

		# These should be after all the if statements
		rho_dm_sim.append(profile)
		sigma_lnrho_dm.append(sigma_lnprof)
		r_bins_sim.append(r)

		Mvir_sim.append(halo['mvir'])

# Since low mass halos have a large scatter we compute it separately for them

# Now we need to sort halos in order of increasing mass
# Since this is what the scipy interpolator expects
Mvir_sim = np.array(Mvir_sim, dtype='float32')
sorting_indices = np.argsort(Mvir_sim)
Mvir_sim = Mvir_sim[sorting_indices]

mask = (Mvir_sim>=10**(mmin)) & (Mvir_sim<10**mmax)
print(f'{np.log10(Mvir_sim[mask].min()):.2f}, {np.log10(Mvir_sim[mask].max()):.2f}')

# # This is where we select which part of the profile (inner/outer) we want to fit for.
# idx = np.arange(10)
# r_bins = r_bins[idx]


rho_dm_sim = np.array(rho_dm_sim, dtype='float32')[sorting_indices]#[:, idx]
sigma_lnrho_dm = np.vstack(sigma_lnrho_dm)[sorting_indices]#[:, idx]

rho_dm_sim = rho_dm_sim[mask]
sigma_lnrho_dm = sigma_lnrho_dm[mask]
Mvir_sim = Mvir_sim[mask]

# rho_dm_rescale = np.vstack(rho_dm_rescale)[sorting_indices][:, idx]
# median_prof = np.median(rho_dm_rescale[mask], axis=0)
# sigma_intr_rho_dm = get_scatter(np.log(rho_dm_rescale[mask]), np.log(median_prof))
# sigma_intr_rho_dm[-1] = 0.1

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

fit_result = []

fig = plt.figure()

for i in tqdm(range(sum(mask))):
	this_rho_dm_sim = rho_dm_sim[i]
	this_halo_mass = Mvir_sim[i]
	this_sigma_lnrho_dm = sigma_lnrho_dm[i]
	this_r_bins = r_bins_sim[i]

	sol = scipy.optimize.least_squares(joint_likelihood, starting_point, bounds=np.array(these_bounds).T, args=(this_halo_mass, this_r_bins),
										max_nfev=1000, x_scale=std, xtol=None, gtol=None)

	fit_result.append([this_halo_mass, sol.x[0], sol.cost])
	
	fitter.update_param(fit_par, sol.x)
	rho_dm_theory, r = fitter.get_rho_dm_profile(this_halo_mass*u.Msun/cu.littleh, r_bins=this_r_bins, z=0)
	plt.plot(r, this_rho_dm_sim/rho_dm_theory.value-1, c='dodgerblue', alpha=0.2);


# plt.plot(r_bins, sigma_intr_rho_dm, c='k', ls=':', label='$\sigma_{int}$')
# plt.plot(r_bins, -sigma_intr_rho_dm, c='k', ls=':')
plt.savefig('test_residual.pdf')
plt.close()

result = np.vstack(fit_result)
# result[:, 2][result[:, 2]>50] = 50


plt.figure()
plt.scatter(result[:,0], result[:, 1], c=result[:, 2], alpha=0.5)
plt.colorbar()
plt.ylabel('Concentration')
plt.xlabel('Virial Mass [Msun]')
plt.xscale('log')
plt.savefig(f'test_conc_{mmin}_{mmax}.pdf')
plt.close()

plt.figure()
plt.scatter(result[:,0], result[:, 2], c='dodgerblue', alpha=0.2)
plt.ylabel('Cost Fn.')
plt.xlabel('Virial Mass [Msun]')
plt.xscale('log')
plt.savefig(f'test_cost_{mmin}_{mmax}.pdf')
plt.close()
