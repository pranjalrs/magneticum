import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import astropy.units as u
import astropy.cosmology.units as cu
import getdist
from getdist import plots
import emcee

import sys
sys.path.append('../src/')

from analytic_profile import Profile
import post_processing


def get_analytic_profile(class_instance, mass_list):
    Pe, r = [], []
    Rvir = []
    for m in mass_list:
        mvir = m*u.Msun/cu.littleh
        rvir = class_instance.get_rvirial(mvir)

        r_bins = np.logspace(np.log10(0.1), np.log10(1), 200)*rvir
        this_profile = class_instance.get_Pe(mvir, r_bins)
        Pe.append(this_profile)
        r.append(r_bins/rvir)
        Rvir.append(rvir)

    return Pe, r

def likelihood(x):
    for i in range(len(fit_par)):
        lb, ub = bounds[fit_par[i]]
        if x[i]<lb or x[i]>ub:
            return -np.inf
    
    fitter.update_param(fit_par, x)

    
    ## Get profile for each file
    Pe, r = get_analytic_profile(fitter, mass_list)
    
    residual = []

    for i in range(len(Pe)):
        interpolator = interp1d(r[i], Pe[i].value)
        Pe_theory = interpolator(r_sim[i])  # Get theory Pe at measured r values
        
        residual.append(np.sum(np.log(Pe_sim[i]/Pe_theory)**2/sigmalnP_sim[i]**2))
    
    return -np.sum(residual)


## Load Data
base = "../output_data/Profiles_median/Box1a/['Pe_Mead']_z=0.00_mvir_"
files=[f'{base}3.2E+13_1.0E+14.pkl',
       f'{base}1.0E+14_3.2E+14.pkl',
       f'{base}3.2E+14_1.0E+15.pkl']

mass_list = [np.mean(joblib.load(f)['mvir']) for f in files]

Pe_sim= []
r_sim = []
sigmaP_sim = []
sigmalnP_sim = []
for f in files:
    this_prof = post_processing.get_mean_profile_all_fields(joblib.load(f), r_name='rvir', rescale=False)
    ##Only select points r/Rvir <=1
    mask = this_prof['Pe_Mead'][1]<=1
    Pe_sim.append(this_prof['Pe_Mead'][0][mask])
    r_sim.append(this_prof['Pe_Mead'][1][mask])
    sigmaP_sim.append(this_prof['Pe_Mead'][2][mask])
    sigmalnP_sim.append(this_prof['Pe_Mead'][3][mask])

bounds = {'f_H': [0.65, 0.85],
        'gamma': [1.1, 5],
        'alpha': [0.1, 1.5],
        'log10_M0': [10, 17],
        'beta': [0.5, 0.8],
        'eps1_0': [-.8, .8],
        'eps2_0': [-.8, .8]}

fid_val = {'f_H': 0.75,
        'gamma': 1.2,
        'alpha': 0.5,
        'log10_M0': 14,
        'beta': 0.6,
        'eps1_0': 0.2,
        'eps2_0': -0.1}

std_dev = {'f_H': 0.2,
        'gamma': 0.2,
        'alpha': 0.2,
        'log10_M0': 0.5,
        'beta': 0.2,
        'eps1_0': 0.02,
        'eps2_0': 0.02}


fitter = Profile()

## Prepare for MCMC
fit_par = ['gamma', 'alpha', 'log10_M0', 'beta', 'eps1_0', 'eps2_0']
starting_point = [fid_val[k] for k in fit_par]
std = [std_dev[k] for k in fit_par]

ndim = len(fit_par)
nwalkers= 3 * ndim
nsteps = 8000

p0_walkers = emcee.utils.sample_ball(starting_point, std, size=nwalkers)

for i, key in enumerate(fit_par):
    low_lim, up_lim = bounds[fit_par[i]]

    for walker in range(nwalkers):
        while p0_walkers[walker, i] < low_lim or p0_walkers[walker, i] > up_lim:
            p0_walkers[walker, i] = np.random.rand()*std[i] + starting_point[i]

## Now run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood)
sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=True)

## Make and save plots
walkers = sampler.get_chain()
chain = sampler.get_chain(discard=int(0.9*nsteps), flat=True)
np.savetxt('emcee_output/samples_emcee.txt', chain)

par_latex_names = ['\Gamma', '\\alpha', '\log_{10}M_0', '\\beta', '\epsilon_1', '\epsilon_2']


fig, ax = plt.subplots(len(fit_par), 1, figsize=(10, 1.5*len(fit_par)))
ax = ax.flatten()

for i in range(len(fit_par)):
    ax[i].plot(walkers[:, :, i])
    ax[i].set_ylabel(f'${par_latex_names[i]}$')
    ax[i].set_xlabel('Step #')

plt.savefig('emcee_output/trace_plot.pdf')

plt.figure()

gd_samples = getdist.MCSamples(samples=chain, names=fit_par, labels=par_latex_names)
g = plots.get_subplot_plotter()
g.triangle_plot(gd_samples, axis_marker_lw=2, marker_args={'lw':2}, line_args={'lw':1}, title_limit=2)
plt.savefig('emcee_output/triangle_plot.pdf')

########## Compare best-fit profiles ##########
c = ['r', 'b', 'g', 'k']

bins = [13.5, 14, 14.5, 15]
# Fiducial HMCode profiles
fitter.update_param(fit_par, gd_samples.getMeans())

Pe_bestfit, r_bestfit = get_analytic_profile(fitter, mass_list)

plt.figure(figsize=(7, 5))

lines = [None]*(len(files)+1)

for i in range(len(files)):
    bin_label = f'{(bins[i]):.1f}$\leq\log$M$_{{vir}}<${(bins[i+1]):.1f}'
    plt.plot(r_bestfit[i], Pe_bestfit[i].value, c=c[i], ls='--')
    lines[i] = plt.errorbar(r_sim[i], Pe_sim[i], yerr=sigmaP_sim[i], c=c[i], label=bin_label)
    

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
plt.savefig('emcee_output/best_fit.pdf')
