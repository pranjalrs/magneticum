import numpy as np

class ProfileContainer():
	"""
	A class representing a container for profile data.

	Attributes:
		mvir (float): The virial mass.
		rvir (float): The virial radius.
		profile (array): The profile data.
		profile_rescale (float): The rescaled profile data.
        rbins (array): The radial bins in physical units.
		xbins (array): The radial bins in units of r/Rvir.
		sigma_prof (float): The profile standard deviation.
		sigma_lnprof (float): The natural logarithm of the profile standard deviation.
	"""

	def __init__(self, mvir, rvir, profile, profile_rescale, rbins, xbins, sigma_prof, sigma_lnprof, units=None):
		self.mvir = mvir
		self.rvir = rvir
		self.profile = profile
		self.profile_rescale = profile_rescale
		self.rbins = rbins
		self.xbins = xbins
		self.sigma_prof = sigma_prof
		self.sigma_lnprof = sigma_lnprof
		self.units = units

	@property
	def mean_profile_rescale(self):
		return np.nanmean(self.profile_rescale, axis=0)

	@property
	def median_profile_rescale(self):
		return np.nanmedian(self.profile_rescale, axis=0)
