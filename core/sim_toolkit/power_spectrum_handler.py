import numpy as np
import utils

class PowerSpectrumHandler:
	'''
	A class to handle power spectrum data.

	Parameters:
	- path (str): The path to the power spectrum data file.

	Attributes:
	- z (float): The redshift value extracted from the file path.
	- k (ndarray): Array of wavenumbers.
	- Pk_hydro (ndarray): Array of power spectrum values for hydro simulation.
	- Pk_dm (ndarray): Array of power spectrum values for dark matter only simulations.
	- Pk_ratio (ndarray): Array of power spectrum ratios between hydro and dark matter simulations.
	- Pk_ratio_mean (float): Median value of power spectrum ratios within a specified range.

	Methods:
	- _load_power_spectrum(path): Loads the power spectrum data from the given file path.
	- _loadtxt(path, **kwargs): Helper method to load data from a text file.

	'''

	def __init__(self, path):
		self._load_power_spectrum(path)

	def _load_power_spectrum(self, path):
		'''
		Loads the power spectrum data from the given file path.

		Parameters:
		- path (str): The path to the power spectrum data file.

		'''
		self.z = utils.search_z_in_string(path)

		self.k, self.Pk_hydro = self._loadtxt(path, unpack=True)

		if 'hr_bao' in path:
			path = path.replace('hr_bao', 'hr_dm')
		else:
			path = path.replace('hr', 'dm_hr')

		_, self.Pk_dm = self._loadtxt(path, unpack=True)

		if (self.Pk_hydro is not None) and (self.Pk_dm is not None):
			self.Pk_ratio = self.Pk_hydro / self.Pk_dm
			kmin, kmax = 5, 15
			k_idx = (self.k > kmin) & (self.k < kmax)
			self.Pk_ratio_mean = np.median(self.Pk_ratio[k_idx])
		else:
			self.Pk_ratio = None
			self.Pk_ratio_mean = None

	@staticmethod
	def _loadtxt(path, **kwargs):
		'''
		Helper method to load data from a text file.

		Parameters:
		- path (str): The path to the text file.
		- **kwargs: Additional keyword arguments to be passed to np.loadtxt().

		Returns:
		- ndarray: Array of loaded data.

		'''
		try:
			return np.loadtxt(path, **kwargs)
		except FileNotFoundError:
			print('File not found {path}')
			return None, None

