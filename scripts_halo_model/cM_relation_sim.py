'''Plot concentration from total matter profile
'''
import joblib
import matplotlib.pyplot as plt
import numpy as np


import dawn.utils as utils
import dawn.theory.power_spectrum as power_spectrum
from dawn.theory.power_spectrum import get_halomodel_Pk

def get_mean_mc(KDE_dict):
	grid = np.linspace(0, 50, 1000)
	c_bar_mean = np.zeros(len(KDE_dict.values()))
	c_bar_std = np.zeros(len(KDE_dict.values()))

	for i, kde in enumerate(KDE_dict.values()):
		if kde is not None:
			pdf = kde(grid)
			c_bar_mean[i] = np.trapz(grid*pdf, grid)/np.trapz(pdf, grid)
			c_bar_std[i] = np.sqrt(np.trapz((grid-c_bar_mean[i])**2*pdf, grid)/np.trapz(pdf, grid))

	return np.column_stack((c_bar_mean, c_bar_std))
#---------------------------- 1. Load concentration fits and plot histograms ----------------------------#
## Build the KDE for the concentration-mass relation scatter
mass_bins = np.logspace(11, 16, 11)
mass_bins = mass_bins[2:-1]
set = ['hr_bao_DM_conc_12.0_12.5_linear.pkl', # Box2
		'hr_bao_DM_conc_12.5_13.0_linear.pkl', # Box2
		'hr_bao_DM_conc_13.0_13.5_linear.pkl',
		'hr_bao_DM_conc_13.5_14.0_linear.pkl',
		'hr_bao_DM_conc_14.0_14.5_linear.pkl',
		'mr_bao_DM_conc_14.5_15.0_linear.pkl',
		'mr_bao_DM_conc_15.0_15.5_linear.pkl']

base_path = '../../magneticum-data/data/DM_conc/'
dm_data_paths = [base_path + f for f in set]
dm_fit_data = [joblib.load(f) for f in dm_data_paths]

base_path2 = '../../magneticum-data/data/matter_conc/'
matter_data_paths = [base_path2 + f for f in set]
matter_fit_data = [joblib.load(f) for f in dm_data_paths]


dm_fit_data = np.vstack(dm_fit_data)
mask = dm_fit_data[:, 1]<48
dm_fit_data = dm_fit_data[mask]
KDE_dict_dm = utils.build_KDE(dm_fit_data[:, 0], dm_fit_data[:, 1], mmin=1e12)#, weights=1/DM_fit_data[:, 3]**2)

matter_fit_data = np.vstack(matter_fit_data)
mask = matter_fit_data[:, 1]<48
matter_fit_data = matter_fit_data[mask]
KDE_dict_matter = utils.build_KDE(matter_fit_data[:, 0], matter_fit_data[:, 1], mmin=1e12)#, weights=1/DM_fit_data[:, 3]**2)

# Get mean concentration in eachh bin
c_bar_dm = get_mean_mc(KDE_dict_dm)
c_bar_hydro = get_mean_mc(KDE_dict_matter)

if __name__ == '__main__':
	# Plot histograms and KDE
	fig, ax = plt.subplots(3, 3, figsize=(12, 8))
	ax = ax.flatten()
	for i, (kde_matter, kde_dm) in enumerate(zip(KDE_dict_matter.values(), KDE_dict_dm.values())):
		if kde_matter is not None:
			# ax[i].hist(matter_fit_data[(matter_fit_data[:, 0]>mass_bins[i]) & (matter_fit_data[:, 0]<mass_bins[i+1]), 1],
				#  bins=30, density=True, alpha=0.5, color='orangered', label='DMO')
			# ax[i].hist(dm_fit_data[(dm_fit_data[:, 0]>mass_bins[i]) & (dm_fit_data[:, 0]<mass_bins[i+1]), 1],
				#  bins=30, density=True, alpha=0.5, color='dodgerblue', label='dm')

			grid = np.linspace(0, 50, 1000)
			ax[i].plot(grid, kde_matter(grid), c='orangered', label='KDE: total matter profile')
			ax[i].plot(grid, kde_dm(grid), c='dodgerblue', label='KDE: DM profile')
			ax[i].set_title(f'{np.log10(mass_bins[i]):.1f}<logM<{np.log10(mass_bins[i+1]):.1f}')

		ax[i].set_xlabel('Concentration')

	ax[0].legend()
	fig.tight_layout()
	# plt.savefig(f'../figures/KDE_concentration_mass_relation.pdf')
	# plt.close()
	plt.show()

	#---------------------------- 2. Mean mass-concentration relation  ----------------------------#

	## Build a mean concentration-mass relation using the median of the KDE
	bin_center = (mass_bins[1:] + mass_bins[:-1])/2


	Ragagnin_conc = power_spectrum.get_mass_concentration_relation('Ragagnin et al. (2023)')(np.logspace(11, 15.5, 100), 0)
	Duffy_conc = power_spectrum.get_mass_concentration_relation('Duffy et al. (2008)')(np.logspace(11, 15.5, 100), 0)


	# Now fit a power law to the mean concentration-mass relation
	def cm_powerlaw(mass, A, B):
		M0 = 10**13.5
		return A*(mass/M0)**B

	# Now fit using c_bar_mean and bin_center
	from scipy.optimize import curve_fit
	popt_dm, _ = curve_fit(cm_powerlaw, bin_center, c_bar_dm[:, 0], sigma=c_bar_dm[:, 1], absolute_sigma=True)
	popt_hydro, _ = curve_fit(cm_powerlaw, bin_center, c_bar_hydro[:, 0], sigma=c_bar_hydro[:, 1], absolute_sigma=True)


	plt.errorbar(bin_center, c_bar_dm[:, 0], yerr=c_bar_dm[:, 1], alpha=0.5, capsize=3, c='orangered', fmt='o', label='DMO')
	plt.plot(np.logspace(10, 16, 100), cm_powerlaw(np.logspace(11, 16, 100), *popt_dm), c='orangered', label='Power law fit')

	plt.errorbar(bin_center, c_bar_hydro[:, 0], yerr=c_bar_hydro[:, 1], alpha=0.5, capsize=3, c='dodgerblue', fmt='o', label='Hydro')
	plt.plot(np.logspace(10, 16, 100), cm_powerlaw(np.logspace(11, 16, 100), *popt_hydro), c='dodgerblue')

	plt.plot(np.logspace(10, 15.5, 100), Ragagnin_conc, 'k', ls='--', label='Ragagnin et al. (2023)')
	plt.plot(np.logspace(10, 15.5, 100), Duffy_conc, 'k', ls='-.', label='Duffy et al. (2008)')

	# plt.ylim([1, 20])
	plt.xscale('log')
	plt.xlabel('M [Msun/h]')
	plt.ylabel('c(M)')
	plt.legend()
	# plt.savefig(f'../figures/mean_concentration_mass_relation.pdf')
	plt.close()