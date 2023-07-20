import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import astropy.units as u
import astropy.cosmology.units as cu
import getdist
from getdist import plots
import ultranest

import sys
sys.path.append('../core/')

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

def prior(cube):
    params = cube.copy()
    for i in range(len(fit_par)):
        lb, ub = bounds[fit_par[i]]
        params[i] = cube[i] * (ub- lb) + lb
    
    return params


def likelihood(cube):
    fitter.update_param(fit_par, cube)

    
    ## Get profile for each file
    Pe, r = get_analytic_profile(fitter, mass_list)
    
    residual = []

    for i in range(len(Pe)):
        interpolator = interp1d(r[i], Pe[i].value)
        Pe_theory = interpolator(r_sim[i])  # Get theory Pe at measured r values
        
        residual.append(np.sum(np.log(Pe_sim[i]/Pe_theory)**2/sigmalnP_sim[i]**2))
    
    return -0.5*np.sum(residual)


## Load Data
base = "../../magneticum-data/data/profiles_median/Box1a/['Pe_Mead']_z=0.00_mvir_"
files=[f'{base}3.2E+13_1.0E+14.pkl',
       f'{base}1.0E+14_3.2E+14.pkl',
       f'{base}3.2E+14_1.0E+15.pkl']

mass_list = [np.mean(joblib.load(f)['mvir']) for f in files]

Pe_sim= []
r_sim = []
sigmalnP_sim = []
for f in files:
    this_prof = post_processing.get_mean_profile_all_fields(joblib.load(f), r_name='rvir', rescale=False)
    ##Only select points r/Rvir <=1
    mask = this_prof['Pe_Mead'][1]<=1
    Pe_sim.append(this_prof['Pe_Mead'][0][mask])
    r_sim.append(this_prof['Pe_Mead'][1][mask])
    sigmalnP_sim.append(this_prof['Pe_Mead'][2][mask])

bounds = {'f_H': [0.65, 0.85],
        'gamma': [1.1, 5],
        'alpha': [0.1, 5],
        'log10_M0': [10, 17],
        'beta': [0.2, 3],
        'eps1_0': [-.8, .8],
        'eps2_0': [-.8, .8]}

fitter = Profile()

## Prepare for MCMC
fit_par = ['gamma', 'alpha', 'log10_M0', 'beta', 'eps1_0', 'eps2_0']

## Now run MCMC
sampler = ultranest.ReactiveNestedSampler(fit_par, likelihood, prior, log_dir='./ultranest_output')
sampler.run(min_num_live_points=400)


chain = sampler.results['samples']
par_latex_names = ['\Gamma', '\\alpha', '\log_{10}M_0', '\\beta', '\epsilon_1', '\epsilon_2']


fig, ax = plt.subplots(len(fit_par), 1, figsize=(10, 1.5*len(fit_par)))
ax = ax.flatten()

gd_samples = getdist.MCSamples(samples=chain, names=fit_par, labels=par_latex_names, sampler='nested')

np.savetxt('multinest_samples.txt', chain)
########## Compare best-fit profiles ##########
c = ['r', 'b', 'g', 'k']

bins = [13, 13.5, 14, 14.5, 15]
# Fiducial HMCode profiles
fitter.update_param(fit_par, gd_samples.getMeans())

Pe_bestfit, r_bestfit = get_analytic_profile(fitter, mass_list)

plt.figure(figsize=(7, 5))

lines = [None]*(len(files)+1)

for i in range(len(files)):
    bin_label = f'{(bins[i]):.1f}$\leq\log$M$_{{vir}}<${(bins[i+1]):.1f}'
    plt.plot(r_bestfit[i], Pe_bestfit[i].value, c=c[i], ls='--')
    lines[i], = plt.plot(r_sim[i], Pe_sim[i], c=c[i], label=bin_label)

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
plt.savefig('best_fit_ultranest.pdf')


sampler.print_results()
sampler.plot()
sampler.plot_trace()
