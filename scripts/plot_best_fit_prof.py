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
parser.add_argument('--mmin', type=int)
parser.add_argument('--mmax', type=int)

args = parser.parse_args()
mmin = args.mmin
mmax = args.mmax

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

mask = (Mvir_sim>=10**(mmin)) & (Mvir_sim<10**mmax)
####################### Pressure ###############################
Pe_rescale = np.vstack(Pe_rescale)[sorting_indices]
# High mass
median_prof = np.median(Pe_rescale[mask], axis=0)
sigma_intr_Pe_init = get_scatter(np.log(Pe_rescale[mask]), np.log(median_prof))


####################### rho ###############################
rho_rescale = np.vstack(rho_rescale)[sorting_indices]
# High mass
median_prof = np.median(rho_rescale[mask], axis=0)
sigma_intr_rho_init = get_scatter(np.log(rho_rescale[mask]), np.log(median_prof))


####################### Temp ###############################
Temp_rescale = np.vstack(Temp_rescale)[sorting_indices]
# High mass
median_prof = np.median(Temp_rescale[mask], axis=0)
sigma_intr_Temp_init = get_scatter(np.log(Temp_rescale[mask]), np.log(median_prof))

print('Finished processing simulation data...')
#####-------------- Prepare for MCMC --------------#####
fitter = Profile(use_interp=True, mmin=Mvir_sim.min()-1e10, mmax=Mvir_sim.max()+1e10)
print('Initialized profile fitter ...')
#fit_par = ['gamma', 'alpha', 'log10_M0', 'eps1_0', 'eps2_0', 'gamma_T_1', 'gamma_T_2', 'alpha_nt', 'n_nt']
#par_latex_names = ['\Gamma', '\\alpha', '\log_{10}M_0', '\epsilon_1', '\epsilon_2', '\Gamma_\mathrm{T}^1', '\Gamma_\mathrm{T}^2', '\\alpha_{nt}', 'n_{nt}']


#### Discard 0.9*steps and make triangle plot
field = args.field.strip('"').split(',')
save_path = f'../../magneticum-data/data/emcee/prof_{args.field}_halos_bin/mmin_{mmin}_mmax_{mmax}'

walkers = np.load(f'{save_path}/all_walkers.npy')

shape = walkers.shape
n_burn = int(shape[0]*0.9)
n_sample = int(shape[1]*(shape[0]-n_burn))

samples = walkers[n_burn:, :, :].reshape(n_sample, shape[2])


if mmin>=13:
    fit_par = ['gamma', 'alpha', 'log10_M0', 'eps1_0', 'eps2_0', 'gamma_T_2']
    par_latex_names = ['\Gamma', '\\alpha', '\log_{10}M_0', '\epsilon_1', '\epsilon_2', '\Gamma_\mathrm{T}^2']

if mmin<13:
    fit_par = ['gamma', 'alpha', 'log10_M0', 'eps1_0', 'eps2_0', 'gamma_T_1']
    par_latex_names = ['\Gamma', '\\alpha', '\log_{10}M_0', '\epsilon_1', '\epsilon_2', '\Gamma_\mathrm{T}^1']

gd_samples = getdist.MCSamples(samples=samples, names=fit_par, labels=par_latex_names)

########## Compare best-fit profiles ##########
c = ['r', 'b', 'g', 'k']

# Fiducial HMCode profiles
fitter.update_param(fit_par, gd_samples.getMeans())
(Pe_bestfit, rho_bestfit, Temp_bestfit), r_bestfit = fitter.get_Pe_profile_interpolated(Mvir_sim*u.Msun/cu.littleh, z=0, return_rho=True, return_Temp=True)


## Plot median Pe profiles
fig, ax = plt.subplots(1, 3, figsize=(14, 4))

rho_sim[rho_sim==0] = np.nan

ax[0].errorbar(r_bins, np.log(np.nanmedian(rho_sim[mask], axis=0)), yerr=sigma_intr_rho_init, ls='-.', label='Magneticum (median)')
ax[0].plot(r_bestfit, np.log(np.median(rho_bestfit[mask], axis=0).value), ls='-.', label='Best fit (median)')

Temp_sim[Temp_sim==0] = np.nan
ax[1].errorbar(r_bins, np.log(np.nanmedian(Temp_sim[mask], axis=0)), yerr=sigma_intr_Temp_init, ls='-.')
ax[1].plot(r_bestfit, np.log(np.median(Temp_bestfit[mask], axis=0).value), ls='-.')


Pe_sim[Pe_sim==0] = np.nan
ax[2].errorbar(r_bins, np.log(np.nanmedian(Pe_sim[mask], axis=0)), yerr=sigma_intr_Pe_init, ls='-.')
ax[2].plot(r_bestfit, np.log(np.median(Pe_bestfit[mask], axis=0).value), ls='-.')

ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[2].set_xscale('log')

ax[0].set_ylim(-11.8, -5.5)
ax[1].set_ylim(8, 17.5)
ax[2].set_ylim(-17, -5)

ax[0].set_ylabel('$\ln \\rho_{gas}$ [GeV/cm$^3$]')
ax[1].set_ylabel('$\ln$ Temperature [K]')
ax[2].set_ylabel('$\ln P_e$ [keV/cm$^3$]')

ax[0].set_xlabel('$r/Rvir$')
ax[1].set_xlabel('$r/Rvir$')
ax[2].set_xlabel('$r/Rvir$')

ax[1].set_title(f'{mmin}<logM<{mmax}')
ax[0].legend()
plt.savefig(f'{save_path}/best_fit_profiles.pdf', bbox_inches='tight')

#### make triangle plot
plt.figure()
g = plots.get_subplot_plotter()
g.triangle_plot(gd_samples, axis_marker_lw=2, marker_args={'lw':2}, line_args={'lw':1}, title_limit=2)
plt.savefig(f'{save_path}/triangle_plot.pdf')
