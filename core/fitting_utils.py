import joblib
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d

import astropy.units as u
import astropy.cosmology.units as cu

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
# 		std.append(np.mean((this_column-xbar[i])**2)**0.5)
		std.append((np.percentile(this_column-xbar[i], 84, axis=0) - np.percentile(this_column-xbar[i], 16, axis=0))/2)

	return np.array(std)

def update_sigma_intr(val1, val2):
	global sigma_intr_Pe
	global sigma_intr_rho
	
	sigma_intr_Pe = val1
	sigma_intr_rho = val2


def get_halo_data(halo, field, r_bins, return_sigma=False, remove_outlier=True):
	if field=='rho_dm':
		r = halo['fields']['cdm'][1]/halo['rvir']
		profile = halo['fields']['cdm'][0]
		npart = halo['fields']['cdm'][2]
		sigma_prof = profile/npart**0.5
		sigma_lnprof = sigma_prof/profile

	if field=='Pe':
		r = halo['fields']['Pe_Mead'][1]/halo['rvir']
		profile = halo['fields']['Pe_Mead'][0]
		sigma_lnprof = halo['fields']['Pe'][3]


	elif field=='rho':
		r = halo['fields']['gas'][1]/halo['rvir']
		profile = halo['fields']['gas'][0]
		npart = halo['fields']['gas'][2]
		sigma_prof = profile/npart**0.5
		sigma_lnprof = sigma_prof/profile

	elif field=='Temp':
		r = halo['fields']['Temp'][1]/halo['rvir']
		profile = halo['fields']['Temp'][0]
		sigma_lnprof = halo['fields']['Temp'][3]


# 	elif field=='v_disp':
# 		r = halo['fields']['v_disp'][1]/halo['rvir']
# 		profile = halo['fields']['v_disp'][0]
# 		sigma_lnprof = halo['fields']['v_disp'][3]

	#Rescale prof to get intr. scatter
	rescale_value = nan_interp(r, profile)(1)
	profile_rescale = (profile/ rescale_value)
	profile_rescale_interp = nan_interp(r, profile_rescale)(r_bins)
	profile_interp = nan_interp(r, profile)(r_bins)

	if return_sigma is True:
		sigma_lnprof = sigma_lnprof[3:-3]
		assert len(sigma_lnprof) == len(r_bins)

	## Ensure that interpolated/rescaled profiles are not ill-behaved
	if np.any(profile_rescale<0) or np.any(profile_interp<0) or np.all(np.log(profile_rescale)<0):
		if return_sigma is True:
			return None, None, None
		return None, None

	elif np.any(profile_interp[:10]<0.9*profile_interp[-1]) and remove_outlier is True:
		if return_sigma is True:
			return None, None, None
		return None, None

	else:
		if return_sigma is True:
			return profile_interp, profile_rescale_interp, sigma_lnprof
		return profile_interp, profile_rescale_interp


def likelihood(theory_prof, field):
	sim_prof = globals()[field+'_sim']
	num = np.log(sim_prof[mask] / theory_prof.value[mask])**2

	idx = sim_prof[mask]==0
	num = ma.array(num, mask=idx, fill_value=0)

	denom = globals()[f'sigma_intr_{field}_init']**2 #+ globals()[f'sigmaln{field}_sim']**2
	chi2 = 0.5*np.sum(num/denom)  #Sum over radial bins

	if not np.isfinite(chi2):
		return -np.inf

	return -chi2


def joint_likelihood(x, field, fitter, params, bounds, halo_mass, z, r_bins):
	for i in range(len(params)):
		lb, ub = bounds[params[i]]
		if x[i]<lb or x[i]>ub:
			return -np.inf

	fitter.update_param(params, x)

	mvir = halo_mass*u.Msun/cu.littleh
	## Get profile for each halo
	(Pe_theory, rho_theory, Temp_theory), r = fitter.get_Pe_profile_interpolated(mvir, r_bins=r_bins, z=z, return_rho=True, return_Temp=True)

	loglike = 0

	if 'Pe' in field  or 'all' in field:
		loglike += likelihood(Pe_theory, 'Pe')

	if 'rho' in field or 'all' in field:
		loglike += likelihood(rho_theory, 'rho')

	if 'Temp' in field  or 'all' in field:
		loglike += likelihood(Temp_theory, 'Temp')

	return loglike

def nan_interp(x, y):
	idx = ((np.isnan(y)) | (y==0))
	return interp1d(x[~idx], y[~idx], kind='cubic', bounds_error=False, fill_value=0)

def get_scatter(x, xbar):
	# Calculate for radial bin at a time
	std  = []
	for i in range(x.shape[1]):
		this_column = x[:, i]
		idx = (this_column>0) & (np.isfinite(this_column))
		this_column = this_column[idx]
#         std.append(np.mean((this_column-xbar[i])**2)**0.5)
		std.append((np.percentile(this_column-xbar[i], 84, axis=0) - np.percentile(this_column-xbar[i], 16, axis=0))/2)

	return np.array(std)