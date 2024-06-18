import numpy as np
import scipy.interpolate

import astropy.units as u
import astropy.cosmology.units as cu

class ProfileInterpolator():
	"""
	A class for interpolating profiles based on halo mass and radial bins.

	Attributes:
		interpolator (CloughTocher2DInterpolator): Interpolator object for the given data points.
		mass_unit (astropy.units.Unit): Unit of halo mass.
		profile_unit (astropy.units.Unit): Unit of profile values.

	Methods:
		_build_2Dinterpolator: Build a 2D interpolator for halo mass, radial bins, and profile values.
		eval: Evaluate the interpolated profiles for the given halo masses and radial bins.
	"""

	def __init__(self, mass, rbins, profile, mass_unit=u.Msun/cu.littleh):
		self.mass_unit = mass_unit
		self.interpolator = self._build_2Dinterpolator(mass, rbins, profile)

	def _build_2Dinterpolator(self, halo_mass, rbins, profile):
		"""
		Build a 2D interpolator for halo mass, radial bins, and profile values.

		Parameters:
			halo_mass (ndarray): Array of halo masses.
			rbins (ndarray): Array of radial bins.
			profile (ndarray): Array of profile values.

		Returns:
			CloughTocher2DInterpolator: Interpolator object for the given data points.
		"""

		if rbins.ndim == 1:
			# If the r_bins are same for all masses then repeat to get correct shape
			expanded_rbins = list(rbins) * len(halo_mass)
		else:
			expanded_rbins = np.concatenate(rbins)

		# We need to generate pairs of (mass, rbin) for the interpolator
		expanded_halo_mass = np.repeat(halo_mass, len(rbins)).T
		points = np.stack((expanded_halo_mass, expanded_rbins))  # obtain xy corrdinates for data points

		values = np.concatenate((profile))  # obtain values

		return scipy.interpolate.CloughTocher2DInterpolator(points.T, values, rescale=True)
		# return scipy.interpolate.LinearNDInterpolator(points.T, values, rescale=True)

	def eval(self, halo_mass, rbins):
			"""
			Evaluate the interpolated profiles for the given halo masses and radial bins.

			Parameters:
				halo_mass (array-like): Array of halo masses.
				rbins (array-like): Array of radial bins.

			Returns:
				interp_profiles (ndarray): Interpolated profiles for the given halo masses and radial bins.
			"""
			if isinstance(halo_mass, float):
				halo_mass = np.array([halo_mass])

			shape = [0, 0]
			shape[0] = len(halo_mass)

			if rbins.ndim == 1:
				expanded_rbins = list(rbins) * len(halo_mass)
				shape[1] = len(rbins)

			else:
				shape[1] = rbins.shape[1]
				expanded_rbins = np.concatenate(rbins)

			expanded_halo_mass = np.concatenate(np.repeat([halo_mass], shape[1], axis=0).T)

			interp_profiles = self.interpolator(expanded_halo_mass, expanded_rbins)

			if rbins.ndim == 1:
				interp_profiles = interp_profiles.reshape(len(halo_mass), len(rbins))
			else:
				interp_profiles = interp_profiles.reshape(np.array(rbins).shape)

			# if np.isnan(interp_profiles).any():
				# raise ValueError('Interpolated profiles contain NaNs. Halo mass/rbins might be out of interpolation range.')

			return interp_profiles
