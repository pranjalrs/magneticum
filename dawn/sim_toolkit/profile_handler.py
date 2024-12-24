import joblib
import numpy as np

from dawn.sim_toolkit.containers import ProfileContainer

class HaloProfileHandler():
	'''
	Class for easily loading and accessing profile data.

	Parameters:
	- fields (list): A list of fields to extract from the halo data.
	- data_path (str or list): The path(s) to the halo data file(s).

	Attributes:
	- fields (list): A list of fields to extract from the halo data.
	- data_path (str or list): The path(s) to the halo data file(s).

	Methods:
	- _get_profiles(data, fields): Extracts the halo profiles from the loaded data.

	'''

	def __init__(self, fields, path):
		self.fields = fields

		if not isinstance(path, list):
			path = [path]

		if not isinstance(fields, list):
			fields = [fields]

		halo_data = np.concatenate([joblib.load(f) for f in path])

		for field in fields:
			profile = self._get_profiles(halo_data, field)
			setattr(self, field, profile)


	@classmethod
	def _get_profiles(cls, data, field):
		'''
		Extracts the halo profiles from the loaded data.

		Parameters:
		- data (numpy.ndarray): The loaded halo data.
		- fields (str or list): The field(s) to extract from the halo data.

		Returns:
		- profile (Profile): The extracted halo profiles.

		'''

		profile, rescale = [], []
		rbins, xbins = [], []
		sigma_prof, sigma_lnprof = [], []
		mvir, rvir = [], []

		for halo in data:
			prof, r, x, prof_rescale, sigma, lnsigma = cls.read_halo_data(halo, field, return_sigma=True)
			profile.append(prof)
			rescale.append(prof_rescale)
			rbins.append(r)
			xbins.append(x)
			sigma_prof.append(sigma)
			sigma_lnprof.append(lnsigma)

			mvir.append(halo['mvir'])
			rvir.append(halo['rvir'])

		# convert to numpy arrays
		profile_dict = {
			'mvir': np.array(mvir),
			'rvir': np.array(rvir),
			'profile': np.array(profile),
			# 'units': prof.unit,
			'profile_rescale': np.array(rescale),
			'rbins': np.array(rbins),
			'xbins': np.array(xbins),
			'sigma_prof': np.array(sigma_prof),
			'sigma_lnprof': np.array(sigma_lnprof)
		}

		# Create Profile object
		profile = ProfileContainer(**profile_dict)

		return profile

	@staticmethod
	def read_halo_data(halo, field, return_sigma=False):
		'''
		Get halo data for a given field.
		For rho_dm, Pe, rho the uncertainty is calculated as sigma = mu/Npart**0.5=profile/Npart**0.5
		the uncertainty in log profile is calculated as sigma_lnprof = sigma/mu= sigma/profile
		Parameters:
		- halo (dict): Dictionary containing halo data.
		- field (str): Field name.
		- return_sigma (bool, optional): Whether to return sigma values. Default is False.

		Returns:
			tuple: Tuple containing profile, r, profile_rescale, sigma_prof, and sigma_lnprof (if return_sigma is True),
				or profile, r, and profile_rescale (if return_sigma is False).
		'''
		if field == 'rho_dm':
			r = halo['fields']['cdm'][1]
			x = halo['fields']['cdm'][1]/halo['rvir']
			profile = halo['fields']['cdm'][0]
			npart = halo['fields']['cdm'][2]
			sigma_prof = profile/npart**0.5
			sigma_lnprof = sigma_prof/profile

		elif field == 'Pe':
			r = halo['fields']['Pe_Mead'][1]
			x = halo['fields']['Pe_Mead'][1]/halo['rvir']
			profile = halo['fields']['Pe_Mead'][0]
			npart = halo['fields']['Pe_Mead'][2]
			sigma_prof = profile/npart**0.5
			sigma_lnprof = sigma_prof/profile

		elif field == 'gas':
			r = halo['fields']['gas'][1]
			x = halo['fields']['gas'][1]/halo['rvir']
			profile = halo['fields']['gas'][0]
			npart = halo['fields']['gas'][2]
			sigma_prof = profile/npart**0.5
			sigma_lnprof = sigma_prof/profile

		elif field == 'v_disp':
			raise NotImplementedError
	#         r = halo['fields']['v_disp'][1]/halo['rvir']
	#         profile = halo['fields']['v_disp'][0]
	#         sigma_lnprof = halo['fields']['v_disp'][3]

		elif field == 'matter':
			# We need to retrieve dm, gas, and star profiles and sum them
			r_dm = halo['fields']['cdm'][1]
			x_dm = halo['fields']['cdm'][1]/halo['rvir']
			profile_dm = halo['fields']['cdm'][0]
			npart_dm = halo['fields']['cdm'][2]

			r_gas = halo['fields']['gas'][1]
			x_gas = halo['fields']['gas'][1]/halo['rvir']
			profile_gas = halo['fields']['gas'][0]
			npart_gas = halo['fields']['gas'][2]

			r_star = halo['fields']['star'][1]
			x_star = halo['fields']['star'][1]/halo['rvir']
			profile_star = halo['fields']['star'][0]
			npart_star = halo['fields']['star'][2]

			profile = profile_dm + profile_gas + profile_star
			r = (r_dm*profile_dm + r_gas*profile_gas + r_star*profile_star)/profile
			x = (x_dm*profile_dm + x_gas*profile_gas + x_star*profile_star)/profile
			npart = npart_dm + npart_gas + npart_star
			sigma_prof = profile/npart**0.5
			sigma_lnprof = sigma_prof/profile

		else:
			try:
				r = halo['fields'][field][1]
				x = halo['fields'][field][1]/halo['rvir']
				profile = halo['fields'][field][0]
				npart = halo['fields'][field][2]
				sigma_prof = profile/npart**0.5
				sigma_lnprof = sigma_prof/profile

			except KeyError:
				raise KeyError(f"Field {field} not found in halo data.")

		#Rescale prof to get intr. scatter
		profile_rescale = (profile/ profile[-1])
		# profile_rescale = [0.]*len(r)

		if return_sigma:
			return profile, r, x, profile_rescale, sigma_prof, sigma_lnprof

		else:
			return profile, r, x, profile_rescale

	def get_masked_profile(self, mmin, mmax, rmin, field):
		"""
		Get a masked profile based on mass and radial range.

		Parameters:
		-----------
		mmin : float
			Minimum mass threshold for selecting halos.
		mmax : float
			Maximum mass threshold for selecting halos.
		rmin : float
			Minimum radial distance for selecting radial bins (in units of kpc/h).
		field : str
			The field name of the profile container to be accessed.

		Returns:
		--------
		ProfileContainer
			A ProfileContainer object containing the masked profiles and associated data.
		"""
		profile_container = getattr(self, field)

		# First we select all halos in mass range
		mass_mask = (profile_container.mvir > mmin) & (profile_container.mvir < mmax)

		rvir = profile_container.rvir[mass_mask]
		mvir = profile_container.mvir[mass_mask]
		profile = np.vstack(profile_container.profile[mass_mask])
		r = np.vstack(profile_container.rbins[mass_mask])
		x = np.vstack(profile_container.xbins[mass_mask])
		profile_rescale = np.vstack(profile_container.profile_rescale[mass_mask])
		sigma_lnprof = np.vstack(profile_container.sigma_lnprof[mass_mask])
		sigma_prof = np.vstack(profile_container.sigma_prof[mass_mask])


		# Then we select all radial bins in the range
		rmin = rmin / rvir
		xmax = 1. # in r/Rvir
		x_mask = (x >= rmin[:, np.newaxis]) & (r <= xmax)

		profile_args = {'mvir': mvir, 'rvir': rvir,
				  		'profile': np.where(x_mask, profile, np.nan),
						# 'units': profile_container.units,
						'profile_rescale': np.where(x_mask, profile_rescale, np.nan),
						'rbins':np.where(x_mask, r, np.nan),
						'xbins':np.where(x_mask, x, np.nan),
						'sigma_prof': sigma_prof*x_mask,
						'sigma_lnprof': sigma_lnprof*x_mask}

		return ProfileContainer(**profile_args)

	@classmethod
	def get_scatter(cls, x, xbar):
		"""
		Calculate the scatter in each radial bin (cloumn) in the input array `x` with respect to the corresponding element in `xbar`.

		Parameters:
		- x (ndarray): Input array of shape (n, m) where n is the number of rows and m is the number of columns.
		- xbar (ndarray): Array of shape (m,) containing the reference values for each column.

		Returns:
		- ndarray: Array of shape (m,) containing the scatter of each column.

		"""
		std = []
		for i in range(x.shape[1]):
			this_column = x[:, i]
			idx = (this_column > 0.) & (np.isfinite(this_column))
			this_column = this_column[idx]
			std.append(np.nanmean((this_column - xbar[i]) ** 2) ** 0.5)

		return np.array(std)