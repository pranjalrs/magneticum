'''Same as fit_HMCode_profile.py but fits all profiles instead of 
the mean profile in each mass bin
'''
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
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

def update_sigma_intr(val):
    global sigma_intr
    sigma_intr = val

def likelihood(x, mass_list, z=0):
    for i in range(len(fit_par)):
        lb, ub = bounds[fit_par[i]]
        if x[i]<lb or x[i]>ub:
            return -np.inf

    fitter.update_param(fit_par, x)

    mvir = mass_list*u.Msun/cu.littleh
    ## Get profile for each halo
    Pe_theory, r = fitter.get_Pe_profile_interpolated(mvir, r_bins=r_bins, z=z)
    
    chi2 = 0 

    num = np.log(Pe_sim / Pe_theory.value)**2
    denom = sigmalnP_sim**2 + sigma_intr**2
    chi2 = -0.5*np.sum(num/denom)  # Sum over radial bins
    
    # Compute new sigma_intr about the best fit mean
    median_prof = np.median(Pe_theory.value, axis=0)
    update = np.mean((np.log(Pe_sim)-np.log(median_prof))**2, axis=0)**0.5
    update_sigma_intr(update)

    return -chi2

bounds = {'f_H': [0.65, 0.85],
        'gamma': [1.1, 5],
        'alpha': [0.1, 1.5],
        'log10_M0': [10, 17],
        'M0': [1e10, 1e17],
        'beta': [0.4, 0.8],
        'eps1_0': [-.8, .8],
        'eps2_0': [-.8, .8]}

fid_val = {'f_H': 0.75,
        'gamma': 1.2,
        'alpha': 0.5,
        'log10_M0': 14,
        'M0': 1e14,
        'beta': 0.6,
        'eps1_0': 0.2,
        'eps2_0': -0.1}

std_dev = {'f_H': 0.2,
        'gamma': 0.2,
        'alpha': 0.2,
        'log10_M0': 2,
        'M0': 1e12,
        'beta': 0.2,
        'eps1_0': 0.02,
        'eps2_0': 0.02}

#####-------------- Parse Args --------------#####

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--run', type=int)
args = parser.parse_args()
test = args.test
run = args.run

#####-------------- Load Data --------------#####
files = glob.glob('../../magneticum-data/data/profiles_median/Box1a/Pe_Pe_Mead_Temp_matter_cdm_gas_z=0.00*')

## We will interpolate all measured profiles to the same r_bins as 
## the analytical profile for computational efficiency
Pe_sim= []
# r_sim = []
sigmalnP_sim = []
Mvir_sim = []

## Also need to rescale profile to guess intrinsic scatter 
Pe_rescale = []
r_bins = np.logspace(np.log10(0.1), np.log10(1), 20)

for f in files:
    this_prof_data = joblib.load(f)
    
    for halo in this_prof_data:
        this_prof_r = halo['fields']['Pe_Mead'][1]/halo['rvir']
        mask = this_prof_r<1
        this_prof_r = this_prof_r[mask]
        this_prof_field = halo['fields']['Pe_Mead'][0][mask]
        this_sigma_lnP = halo['fields']['Pe'][3][mask]

        Pe_sim.append(np.interp(r_bins, this_prof_r, this_prof_field))
        # r_sim.append(this_prof_r)
        sigmalnP_sim.append(this_sigma_lnP)
    
        #Rescale prof to get intr. scatter
        idx = np.argmin(np.abs(halo['rvir']-halo['fields']['Pe_Mead'][1]))
        prof_rescale = (halo['fields']['Pe_Mead'][0] / halo['fields']['Pe_Mead'][0][idx])[mask]

        Pe_rescale.append(np.interp(r_bins, this_prof_r, prof_rescale))

    Mvir_sim.append(this_prof_data['mvir'])

# Now we need to sort halos in order of increasing mass
# Since this is what the scipy interpolator expects
Mvir_sim = np.concatenate(Mvir_sim, dtype='float32')
sorting_indices = np.argsort(Mvir_sim)

Pe_sim = np.array(Pe_sim, dtype='float32')[sorting_indices]
# r_sim = np.array(r_sim, dtype='float32')[sorting_indices]
sigmalnP_sim = np.array(sigmalnP_sim, dtype='float32')[sorting_indices]
Mvir_sim = Mvir_sim[sorting_indices]

Pe_rescale = np.vstack(Pe_rescale)
median_prof = np.median(Pe_rescale, axis=0)
sigma_intr_init = np.mean((np.log(Pe_rescale)-np.log(median_prof))**2, axis=0)**0.5
update_sigma_intr(sigma_intr_init)

#####-------------- Prepare for MCMC --------------#####
fitter = Profile(use_interp=True)
fit_par = ['gamma', 'alpha', 'log10_M0', 'beta', 'eps1_0', 'eps2_0']
par_latex_names = ['\Gamma', '\\alpha', '\log_{10}M_0', '\\beta', '\epsilon_1', '\epsilon_2']

starting_point = [fid_val[k] for k in fit_par]
std = [std_dev[k] for k in fit_par]

ndim = len(fit_par)
nwalkers= 2 * ndim
nsteps = 3000

p0_walkers = emcee.utils.sample_ball(starting_point, std, size=nwalkers)

for i, key in enumerate(fit_par):
    low_lim, up_lim = bounds[fit_par[i]]

    for walker in range(nwalkers):
        while p0_walkers[walker, i] < low_lim or p0_walkers[walker, i] > up_lim:
            p0_walkers[walker, i] = np.random.rand()*std[i] + starting_point[i]


#####-------------- RUN MCMC --------------#####
if test is False:
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        
        print('Running MCMC with MPI...')
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, pool=pool, args=[Mvir_sim])
        sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=True)

else:
    print('Running MCMC...')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, args=[Mvir_sim])
    sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=True)

#####-------------- Plot and Save --------------#####
save_path = f'../../magneticum_data/data/emcee/fit_Pe_all/run{run}'
if not os.path.exists(save_path):
    # If the folder does not exist, create it and break the loop
    os.makedirs(save_path)

walkers = sampler.get_chain()
chain = sampler.get_chain(discard=int(0.8*nsteps), flat=True)

log_prob_samples = sampler.get_log_prob(discard=int(0.8*nsteps), flat=True)

all_samples = np.concatenate((chain, log_prob_samples[:, None]), axis=1)
np.savetxt(f'{save_path}/all_samples.txt', chain)
np.savetxt(f'{save_path}/sigma_intr.txt', np.column_stack((sigma_intr_init, sigma_intr)), header='initial guess \t best fit')


fig, ax = plt.subplots(len(fit_par), 1, figsize=(10, 1.5*len(fit_par)))
ax = ax.flatten()

for i in range(len(fit_par)):
    ax[i].plot(walkers[:, :, i])
    ax[i].set_ylabel(f'${par_latex_names[i]}$')
    ax[i].set_xlabel('Step #')

plt.savefig(f'{save_path}/trace_plot.pdf')

plt.figure()

gd_samples = getdist.MCSamples(samples=chain, names=fit_par, labels=par_latex_names)
g = plots.get_subplot_plotter()
g.triangle_plot(gd_samples, axis_marker_lw=2, marker_args={'lw':2}, line_args={'lw':1}, title_limit=2)
plt.savefig(f'{save_path}/triangle_plot.pdf')

########## Compare best-fit profiles ##########
c = ['r', 'b', 'g', 'k']

bins = [13.5, 14, 14.5, 15]
# Fiducial HMCode profiles
fitter.update_param(fit_par, gd_samples.getMeans())

# Randomly pick 5 profiles
nhalo = 5
inds = np.random.choice(Mvir_sim, nhalo, replace=False)
Pe_bestfit, r_bestfit = fitter.get_Pe_profile_interpolated(Mvir_sim[inds]*u.Msun/cu.littleh, z=0)

plt.figure(figsize=(7, 5))

lines = [None]*(len(nhalo)+1)

for i, j in zip(range(nhalo), inds):
    bin_label = f'{(bins[i]):.1f}$\leq\log$M$_{{vir}}<${(bins[i+1]):.1f}'
    plt.plot(r_bestfit[i], Pe_bestfit[i].value, c=c[i], ls='--')
    lines[i] = plt.errorbar(r_sim[j], Pe_sim[j], yerr=sigmaP_sim[j], c=c[i], label=f'log10 M = {np.log10(Mvir_sim[j]):.3f}')
    

plt.xscale('log')
plt.yscale('log')

lines[i+1], = plt.loglog([], [], '--', c='k', label='Best Fit')

legend1 = plt.legend(handles=lines, title='Box1a', fontsize=12, frameon=False)
# legend2 = plt.legend(handles=line_analytic, labels=['Best Fit'], fontsize=12, frameon=False, loc='lower center')

plt.gca().add_artist(legend1)
# plt.gca().add_artist(legend1)

plt.xlabel('$r/R_{\mathrm{vir}}$')
plt.ylabel('$P_e$ [keV/cm$^3$]');

plt.ylim([2e-6, 1.2e-2])
plt.savefig(f'{save_path}/best_fit.pdf')
