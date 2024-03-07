import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import re
import scipy.stats

def build_KDE(mass, concentration, weights=None, bw_method='silverman'):
	'''
	Builds a kernel density estimator (KDE) for the given data.

	Parameters:
	- data (array): The data for which the KDE is to be built.
	- bw_method (str): The method to estimate the bandwidth. Default is 'scott'.

	Returns:
	- KDE (object): The KDE object.
	'''
	mass_bins = np.logspace(11, 16, 11)

	kde_models = {}
	for i in range(9):
		mmin, mmax = mass_bins[i], mass_bins[i+1]
		mask = (mass>mmin) & (mass<=mmax)
		x = concentration[mask]
		if len(x)==0:
			kde_models.append(None)
			continue

		scipy_kde = scipy.stats.gaussian_kde(x, bw_method=bw_method, weights=weights[mask] if weights is not None else None)

		kde_models[f'{np.log10(mmin)}<logM<{np.log10(mmax)}'] = scipy_kde

	return kde_models


def select_KDE(KDE_dict, mass):
	'''
	Selects the KDE to use for the marginalization over the concentration-mass relation scatter.

	Parameters:
	- KDE_dict (dict): Dictionary containing the KDEs for different halo masses.
	- mass (float): Halo mass.

	Returns:
	- KDE (object): The selected KDE object.
	'''
	# The dictionary keys will specify the mass range as a string i.e, mmin<logM<mmax
	# We need to convert the mass to log10(M)
	logM = np.log10(mass)

	# Find the mass bin
	for key in KDE_dict:
		mmin, mmax = map(float, key.split('<logM<'))
		if logM>=mmin and logM<mmax:
			return KDE_dict[key]


def lognormal_pdf(x, mu, sigma):
	norm = (2*np.pi)**0.5 * x * sigma
	return  1/norm * np.exp( - (np.log(x) - mu)**2 / (2 * sigma**2))

def get_cosmology_dict_from_path(path):
	if 'mr_bao' in path or 'hr_bao' in path or 'mr_dm' in path or 'hr_dm' in path:
		Om0, Ob0, sigma8, h = 0.272, 0.0456, 0.809, 0.704

	elif 'hr' in path:
		cosmo_pars = re.findall(r'hr_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)', path)[0]
		Om0, Ob0, sigma8, h = np.array(cosmo_pars, dtype=float)

	else:
		cosmo_pars = re.findall(r'mr_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)', path)[0]

		Om0, Ob0, sigma8, h = np.array(cosmo_pars, dtype=float)

	this_cosmo = {'flat': True, 'H0': h*100, 'Om0': Om0, 'Ob0': Ob0, 'sigma8': sigma8, 'ns': 0.963}

	return this_cosmo


def _assert_correct_field(fields):
	for f in fields:
		assert f in ['Pe', 'Temp', 'matter', 'cdm', 'gas', 'Pe_Mead', 'v_disp'], f'Field *{f}* is unknown!'


def set_storage_path():
	'''Sets the path to where the simulation data is stored
	'''
	global storage_path
	current_directory = os.getcwd()

	if 'pranjalrs' in current_directory:
		storage_path = f'/xdisk/timeifler/pranjalrs/magneticum_data/'

	if 'di75sic' in current_directory:
		storage_path = f'/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/'


def search_z_in_string(string):
	match = re.search(r'_z=([\d.]+)', string.split('/')[-1])

	if match:
		return float(match.group(1).rstrip('.'))

	else:
		print(f'No redshift found in {string}')
		return None

def search_mass_range_in_string(string):
	match = re.search(r'_([\d.Ee+-]+)_([\d.Ee+-]+)_z=\d+\.\d+', string.split('/')[-1])


	if match:
		return float(match.group(1)), float(match.group(2))
	
	else:
		print(f'No mass range found in {string}')
		return None, None

##### Functions mostly used in Jupyter notebook demo #####

def get_omega_m(filename):
	# extract the omega_b value from the filename
	cosmo = get_cosmology_dict_from_path(filename)
	omega_m = cosmo['Om0']
	return float(omega_m)

def get_omega_b(filename):
	# extract the omega_b value from the filename
	cosmo = get_cosmology_dict_from_path(filename)
	omega_b = cosmo['Ob0']
	return float(omega_b)

def get_sigma8(filename):
	# extract the omega_b value from the filename
	cosmo = get_cosmology_dict_from_path(filename)
	sigma8 = cosmo['sigma8']
	return float(sigma8)

def get_h(filename):
	# extract the omega_b value from the filename
	cosmo = get_cosmology_dict_from_path(filename)
	H0 = cosmo['H0']
	return float(H0)

def get_fb(filename):
	omega_b = get_omega_b(filename)
	omega_m = get_omega_m(filename)
	
	return omega_b/omega_m

def sigma_percentile(arr):
	return (np.percentile(arr, 84) - np.percentile(arr, 16))/2

def colorbar(mappable, ax):
	last_axes = plt.gca()
	fig = ax.figure
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = fig.colorbar(mappable, cax=cax)
	plt.sca(last_axes)
	return cbar
