'''Same as fit_HMCode_profile.py but fits all profiles instead of 
the mean profile in each mass bin
'''
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
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
import getdist
from getdist import plots
import glob
import emcee

import sys
sys.path.append('../core/')

from analytic_profile import Profile
import post_processing
from fitting_utils import get_halo_data, get_scatter, joint_likelihood

import ipdb

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

#####-------------- Parse Args --------------#####

parser = argparse.ArgumentParser()
parser.add_argument('--field')
parser.add_argument('--mmin', type=int)
parser.add_argument('--mmax', type=int)
parser.add_argument('--run', type=str)
parser.add_argument('--niter', type=str)

args = parser.parse_args()
mmin = args.mmin
mmax = args.mmax
run = args.run
niter = args.niter

bounds = {'f_H': [0.65, 0.85],
                'gamma': [1.1, 5],
                'alpha': [0.1, 2],
                'alpha2': [0.1, 2],
                'log10_M0': [10, 17],
                'M0': [1e10, 1e17],
                'beta': [0.4, 0.8],
                'eps1_0': [-0.95, 3],
                'eps2_0': [-0.95, 3],
                'gamma_T': [1.1, 5.5],
                 'b': [0, 2],
                'alpha_nt': [0, 2],
                'n_nt': [0, 2]}

#####-------------- Load Data --------------#####
data_path = '../../magneticum-data/data/profiles_median'
files = glob.glob(f'{data_path}/Box1a/Pe_Pe_Mead_Temp_matter_cdm_gas_v_disp_z=0.00_mvir_1.0E+13_1.0E+16.pkl')
files += glob.glob(f'{data_path}/Box2/Pe_Pe_Mead_Temp_matter_cdm_gas_v_disp_z=0.00_mvir_1.0E+12_1.0E+13.pkl')

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

mask = (Mvir_sim>=10**(mmin)) & (Mvir_sim<10**mmax)
idx = np.arange(10)
r_bins = r_bins[idx]



rho_dm_sim = np.array(rho_dm_sim, dtype='float32')[sorting_indices][:, idx]
Pe_sim = np.array(Pe_sim, dtype='float32')[sorting_indices][:, idx]
rho_sim = np.array(rho_sim, dtype='float32')[sorting_indices][:, idx]
Temp_sim = np.array(Temp_sim, dtype='float32')[sorting_indices][:, idx]
Mvir_sim = Mvir_sim[sorting_indices]

#---------------------- rho_dm ----------------------#
rho_dm_rescale = np.vstack(rho_dm_rescale)[sorting_indices][:, idx]
sigma_lnrho_dm = np.vstack(sigma_lnrho_dm)[sorting_indices][:, idx]
#---------------------- Pressure ----------------------#
Pe_rescale = np.vstack(Pe_rescale)[sorting_indices][:, idx]
sigma_lnPe = np.vstack(sigma_lnPe)[sorting_indices][:, idx]
####################### rho ###############################
rho_rescale = np.vstack(rho_rescale)[sorting_indices][:, idx]
sigma_lnrho = np.vstack(sigma_lnrho)[sorting_indices][:, idx]

####################### Temp ###############################
Temp_rescale = np.vstack(Temp_rescale)[sorting_indices][:, idx]
sigma_lnTemp = np.vstack(sigma_lnTemp)[sorting_indices][:, idx]

print('Finished processing simulation data...')

#####-------------- Load MCMC Data --------------#####
# fit_par = ['gamma', 'alpha', 'log10_M0', 'eps1_0', 'eps2_0', 'gamma_T']
# par_latex_names = ['\Gamma', '\\alpha', '\log_{10}M_0', '\epsilon_1', '\epsilon_2', '\Gamma_\mathrm{T}']

fit_par = ['gamma', 'log10_M0', 'eps1_0', 'eps2_0']
par_latex_names = ['\Gamma', '\log_{10}M_0', '\epsilon_1', '\epsilon_2']

fitter = Profile(use_interp=True, mmin=Mvir_sim.min()-1e10, mmax=Mvir_sim.max()+1e10, imass_conc=1)

field = args.field.strip('"').split(',')
#save_path = f'../../magneticum-data/data/emcee_new/prof_{args.field}_halos_bin/{run}'
save_path = f'../../magneticum-data/data/emcee_magneticum_cM/prof_{args.field}_halos_bin/{run}'

walkers = np.load(f'{save_path}/all_walkers_{niter}.npy')
flat_chain = np.loadtxt(f'{save_path}/all_samples_{niter}.txt')

sigma_int = np.loadtxt(f'{save_path}/sigma_intr_{niter}.txt')
sigma_intr_rho, sigma_intr_Temp, sigma_intr_Pe = sigma_int[:, 0], sigma_int[:, 1], sigma_int[:, 2]

loglike_walkers = flat_chain[:, -5].reshape(walkers.shape[0], walkers.shape[1])
max_loglike = loglike_walkers.max()

# remove stuck walkers
idx = [i for i in range(walkers.shape[1]) if max_loglike-loglike_walkers[-1, i]<1000]
walkers = walkers[:, idx, :]

shape = walkers.shape
n_burn = int(shape[0]*0.9)
n_sample = int(shape[1]*(shape[0]-n_burn))

samples = walkers[n_burn:, :, :].reshape(n_sample, shape[2])
idx = np.argmax(flat_chain[:, -5])
best_params = flat_chain[idx, :-5]

if 'all' in field: nfield=3
else: nfield = len(field)
chi2 = -flat_chain[idx, -5] / (np.sum(mask)*len(r_bins)*nfield - len(fit_par))
chi2_rho_dm = -flat_chain[idx, -4] / (np.sum(mask)*len(r_bins)*nfield - len(fit_par))
chi2_rho = -flat_chain[idx, -3] / (np.sum(mask)*len(r_bins)*nfield - len(fit_par))
chi2_Temp = -flat_chain[idx, -2] / (np.sum(mask)*len(r_bins)*nfield - len(fit_par))
chi2_Pe = -flat_chain[idx, -1] / (np.sum(mask)*len(r_bins)*nfield - len(fit_par))


gd_samples = getdist.MCSamples(samples=samples, names=fit_par, labels=par_latex_names)

#####-------------- Plot and Save --------------#####
fig, ax = plt.subplots(len(fit_par), 1, figsize=(10, 1.5*len(fit_par)))
ax = ax.flatten()
for i in range(len(fit_par)):
	ax[i].plot(walkers[:, :, i])
	ax[i].set_ylabel(f'${par_latex_names[i]}$')
	ax[i].set_xlabel('Step #')

plt.savefig(f'{save_path}/trace_plot_{niter}_post.pdf')

########## Compare best-fit profiles ##########
c = ['r', 'b', 'g', 'k']

# Fiducial HMCode profiles

fitter.update_param(fit_par, best_params)
rho_dm_bestfit, r = fitter.get_rho_dm_profile_interpolated(Mvir_sim*u.Msun/cu.littleh, z=0)
(Pe_bestfit, rho_bestfit, Temp_bestfit), r_bestfit = fitter.get_Pe_profile_interpolated(Mvir_sim*u.Msun/cu.littleh, z=0, return_rho=True, return_Temp=True, r_bins=r_bins)
print(fitter)

## Plot median Pe profiles
if np.sum(mask)>1000: n = 1000
else: n = np.sum(mask)
inds = np.sort(np.random.choice(np.arange(np.sum(mask)), n, replace=False))
#inds2 = np.sort(np.random.choice(np.arange(samples.shape[0]), 20, replace=False))


#-------------------------- Plot random halos --------------------------#
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax = ax.flatten()

c = plt.cm.winter(np.linspace(0,1,len(inds)))
c2 = plt.cm.magma_r(np.linspace(0,1,len(inds)))

for j, i in enumerate(inds):
    ax[0].plot(r_bins, np.log(rho_dm_sim[mask][i]), c=c[j], alpha=0.2)
    ax[1].plot(r_bins, np.log(rho_sim[mask][i]), c=c[j], alpha=0.2)
    ax[2].plot(r_bins, np.log(Temp_sim[mask][i]), c=c[j], alpha=0.2)
    ax[3].plot(r_bins, np.log(Pe_sim[mask][i]), c=c[j], alpha=0.2)

scalarmappaple = ScalarMappable(cmap=plt.cm.winter)
scalarmappaple.set_array(np.log10(Mvir_sim[mask][inds]))

# Add colorbar to the plot
# cbar = plt.colorbar(scalarmappaple, ax=ax[0], location='top')
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(scalarmappaple, cax=cax)
plt.sca(ax[2])
# colorbar(scalarmappaple)
cbar.set_label('log M')

#-------------------------- Plot median profile for random samples from posterior --------------------------#
#for i in inds2:
#    best_params = samples[i]
#    fitter.update_param(fit_par, best_params)
#    (Pe_bestfit_rand, rho_bestfit_rand, Temp_bestfit_rand), r_bestfit = fitter.get_Pe_profile_interpolated(Mvir_sim*u.Msun/cu.littleh, z=0, return_rho=True, return_Temp=True, r_bins=r_bins)
#    ax[0].plot(r_bestfit, np.log(np.median(rho_bestfit_rand[mask], axis=0).value), c='orange', alpha=0.1)
#    ax[1].plot(r_bestfit, np.log(np.median(Temp_bestfit_rand[mask], axis=0).value), c='orange', alpha=0.1)
#    ax[2].plot(r_bestfit, np.log(np.median(Pe_bestfit_rand[mask], axis=0).value), c='orange', alpha=0.1)


rho_dm_sim[rho_dm_sim==0] = np.nan
ax[0].errorbar(r_bins, np.log(np.nanmedian(rho_dm_sim[mask], axis=0)), yerr=sigma_intr_rho_dm, ls='-.', label='Magneticum (median)')
ax[0].plot(r_bestfit, np.log(np.median(rho_dm_bestfit[mask], axis=0).value), ls='-.', label='Best fit (median)')

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

for i in range(4):
    ax[i].set_xlabel('$r/Rvir$')


fig.suptitle(f'{mmin}<logM<{mmax}  '+ 'Total $\chi^2_{\mathrm{d.o.f}}=$'+f'${chi2:.2f}$')
ax[0].set_title('DM Density $\chi^2_{\mathrm{d.o.f}}=$'+f'${chi2_rho_dm:.2f}$')
ax[1].set_title('Density $\chi^2_{\mathrm{d.o.f}}=$'+f'${chi2_rho:.2f}$')
ax[2].set_title('Temperature  $\chi^2_{\mathrm{d.o.f}}=$'+f'${chi2_Temp:.2f}$')
ax[3].set_title('Pressure $\chi^2_{\mathrm{d.o.f}}=$'+f'${chi2_Pe:.2f}$')
ax[0].legend()

ylims = [subax.get_ylim() for subax in ax]

# ax = axs[1, :]

# for j, i in enumerate(inds):
#     ax[0].plot(r_bins, np.log(rho_bestfit[mask][i].value), c=c[j], alpha=0.2)
#     ax[1].plot(r_bins, np.log(Temp_bestfit[mask][i].value), c=c[j], alpha=0.2)
#     ax[2].plot(r_bins, np.log(Pe_bestfit[mask][i].value), c=c[j], alpha=0.2)

# scalarmappaple = ScalarMappable(cmap=plt.cm.magma_r)
# scalarmappaple.set_array(np.log10(Mvir_sim[mask][inds]))

# Add colorbar to the plot
# cbar = plt.colorbar(scalarmappaple, ax=ax[0], location='top')
# cbar.set_label('log M')

[subax.set_ylim(ylims[i]) for i, subax in enumerate(ax)]

ax[0].set_ylabel('$\ln \\rho_{DM}$ [GeV/cm$^3$]')
ax[1].set_ylabel('$\ln \\rho_{gas}$ [GeV/cm$^3$]')
ax[2].set_ylabel('$\ln$ Temperature [K]')
ax[3].set_ylabel('$\ln P_e$ [keV/cm$^3$]')


plt.savefig(f'{save_path}/best_fit_profiles_{niter}_post.pdf', bbox_inches='tight')
#plt.savefig('best_fit_profiles_re-plot2.pdf', bbox_inches='tight')

#-------------------------- Plot log ratio ------------------------------#
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax = ax.flatten()

label = '$ln(\\rho_{sim}/\\rho_{theory})$'
for j, i in enumerate(inds):
    if j>0: label='' 
    ax[0].plot(r_bins, np.log(rho_dm_sim/rho_dm_bestfit.value)[mask][i], c=c[j], alpha=0.2, label=label)
    ax[1].plot(r_bins, np.log(rho_sim/rho_bestfit.value)[mask][i], c=c[j], alpha=0.2, label=label)
    ax[2].plot(r_bins, np.log(Temp_sim/Temp_bestfit.value)[mask][i], c=c[j], alpha=0.2)
    ax[3].plot(r_bins, np.log(Pe_sim/Pe_bestfit.value)[mask][i], c=c[j], alpha=0.2)

scalarmappaple = ScalarMappable(cmap=plt.cm.winter)
scalarmappaple.set_array(np.log10(Mvir_sim[mask][inds]))

# Add colorbar to the plot
cbar = plt.colorbar(scalarmappaple, ax=ax[3])
cbar.set_label('log M')

# ax[0].plot(r_bins, sigma_intr_rho_dm, c='k', ls=':', label='$\sigma_{int}$')
# ax[0].plot(r_bins, -sigma_intr_rho_dm, c='k', ls=':')

ax[1].plot(r_bins, sigma_intr_rho, c='k', ls=':', label='$\sigma_{int}$')
ax[1].plot(r_bins, -sigma_intr_rho, c='k', ls=':')

ax[2].plot(r_bins, sigma_intr_Temp, c='k', ls=':')
ax[2].plot(r_bins, -sigma_intr_Temp, c='k', ls=':')

ax[3].plot(r_bins, sigma_intr_Pe, c='k', ls=':')
ax[3].plot(r_bins, -sigma_intr_Pe, c='k', ls=':')

ax[0].set_ylabel('$\ln \\rho_{DM}$ [GeV/cm$^3$]')
ax[1].set_ylabel('$\ln \\rho_{gas}$ [GeV/cm$^3$]')
ax[2].set_ylabel('$\ln$ Temperature [K]')
ax[3].set_ylabel('$\ln P_e$ [keV/cm$^3$]')

for i in range(4):
    ax[i].set_xlabel('$r/Rvir$')

ax[0].legend()
plt.savefig(f'{save_path}/individual_prof_comparison.pdf', bbox_inches='tight')


#-------------------------- Plot random sample ------------------------------#
#fig, ax = plt.subplots(3, 3, figsize=(14, 11))

#inds = np.sort(np.random.choice(np.arange(np.sum(mask)), 9, replace=False))

#ax = ax.flatten()

#N = len(Mvir_sim[mask])
#for j, h_id in enumerate(inds):
#    for i in inds2:
#        best_params = samples[i]
#        fitter.update_param(fit_par, best_params)
#        (Pe_bestfit_rand, rho_bestfit_rand, Temp_bestfit_rand), r_bestfit = fitter.get_Pe_profile_interpolated(Mvir_sim*u.Msun/cu.littleh, z=0, return_rho=True, return_Temp=True, r_bins=r_bins)

#        ax[j].plot(r_bins, np.log(rho_bestfit_rand.value)[mask][h_id], c=c[j], alpha=0.2, label=label)

#    ax[j].errorbar(r_bestfit, np.log(rho_bestfit[mask][h_id].value), yerr=sigma_intr_rho/N**0.5, c='orange', ls=':', label='best fit')
#    ax[j].errorbar(r_bestfit, np.log(rho_sim[mask][h_id]), yerr=sigma_intr_rho/N**0.5, c='orange', ls=':', label='best fit')

#    ax[j].set_ylabel('$\ln \\rho_{gas}$ [GeV/cm$^3$]')
#    ax[j].set_xlabel('$r/Rvir$')

#    ax[j].set_title(f'logM = {np.log10(Mvir_sim[mask][h_id]):.2f}')
#ax[0].legend()
#plt.savefig(f'{save_path}/individual_prof_scatter.pdf', bbox_inches='tight')



#----------------------------------------------------------------------------#

#### make triangle plot
plt.figure()
g = plots.get_subplot_plotter()
g.triangle_plot(gd_samples, axis_marker_lw=2, marker_args={'lw':2}, line_args={'lw':1}, title_limit=2)
plt.savefig(f'{save_path}/triangle_plot_{niter}_post.pdf')
