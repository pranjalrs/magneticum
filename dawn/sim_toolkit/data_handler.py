import glob

from dawn.sim_toolkit.power_spectrum_handler import PowerSpectrumHandler
from dawn.sim_toolkit.baryon_fraction_handler import BaryonFractionHandler
import dawn.utils as utils

class DataHandler:
	'''
	A class that handles simulation data for a given simulation.

	Parameters:
	- sim_name (str): The name of the simulation.
	- sim_num (str): The number associated with the simulation.
	- box (str): The box size of the simulation.
	- data_path (str): The path to the simulation data.

	Attributes:
	- sim_num_map (dict): A dictionary mapping simulation names to simulation numbers.
	- sim_name (str): The name of the simulation.
	- sim_num (str): The number associated with the simulation.
	- box (str): The box size of the simulation.
	- omega_m (float): The matter density parameter.
	- omega_b (float): The baryon density parameter.
	- sigma8 (float): The amplitude of the matter power spectrum.
	- h (float): The dimensionless Hubble constant.
	- fb (float): The baryon fraction.
	- cosmo_label (str): A formatted string representing the cosmological parameters.

	Methods:
	- _init_cosmology: Initializes the cosmological parameters.
	- _init_power_spectrum: Initializes the power spectrum.
	- _init_baryon_fraction: Initializes the baryon fraction.
	'''

	def __init__(self, sim_name=None, sim_num=None, box='Box3', data_path='./'):
		self.sim_num_map = {
			'hr_0.153_0.0408_0.614_0.666': 'C1',
			'hr_0.189_0.0455_0.697_0.703': 'C2',
			'hr_0.200_0.0415_0.850_0.730': 'C3',
			'hr_0.204_0.0437_0.739_0.689': 'C4',
			'hr_0.222_0.0421_0.793_0.676': 'C5',
			'hr_0.232_0.0413_0.687_0.670': 'C6',
			'hr_0.268_0.0449_0.721_0.699': 'C7',
			'hr_bao': 'WMAP7',
			'hr_0.301_0.0460_0.824_0.707': 'C9',
			'hr_0.304_0.0504_0.886_0.740': 'C10',
			'hr_0.342_0.0462_0.834_0.708': 'C11',
			'hr_0.363_0.0490_0.884_0.729': 'C12',
			'hr_0.400_0.0485_0.650_0.675': 'C13',
			'hr_0.406_0.0466_0.867_0.712': 'C14',
			'hr_0.428_0.0492_0.830_0.732': 'C15'
		}

		if sim_name is not None:
			self.sim_name = sim_name
			self.sim_num = self.sim_num_map[sim_name]
		elif sim_num is not None:
			self.sim_num = sim_num
			self.sim_name = [k for k, v in self.sim_num_map.items() if v == sim_num][0]
		else:
			raise ValueError('Must provide either sim_name or sim_num')

		self.box = box
		self._init_cosmology(self.sim_name)
		self.Pks = self._init_power_spectrum(self.box, self.sim_name, data_path)
		self.fbars = self._init_baryon_fraction(self.box, self.sim_name, data_path)

	def _init_cosmology(self, sim_name):
		'''
		Initializes the cosmological parameters based on the simulation name.

		Parameters:
		- sim_name (str): The name of the simulation.
		'''
		cosmo = utils.get_cosmology_dict_from_path(sim_name)
		self.omega_m = cosmo['Om0']
		self.omega_b = cosmo['Ob0']
		self.sigma8 = cosmo['sigma8']
		self.h = cosmo['H0'] / 100
		self.fb = cosmo['Ob0'] / cosmo['Om0']

		self.cosmo_label = '$\Omega_m={:.3f},\,\sigma_8={:.3f},\,h={:.3f},f_b={:.3f}$'.format(
			cosmo['Om0'], cosmo['sigma8'], self.h, self.fb
		)

	@staticmethod
	def _init_power_spectrum(box, sim_name, data_path):
		'''
		Initializes the power spectrum for the given simulation.

		Parameters:
		- box (str): The box size of the simulation.
		- sim_name (str): The name of the simulation.
		- data_path (str): The path to the simulation data.

		Returns:
		- Pks (dict): A dictionary mapping redshifts to power spectrum objects.
		'''
		path = f'{data_path}/Pylians/Pk_matter/{box}/{sim_name}/Pk_{sim_name}_z=*_R1024.txt'
		cosmo_snaps = sorted(glob.glob(path))

		print(f'Found Pk for {len(cosmo_snaps)} snapshots in {sim_name}')
		Pks = {}
		for snap in cosmo_snaps:
			Pk = PowerSpectrumHandler(snap)
			Pks[Pk.z] = Pk

		return Pks

	@staticmethod
	def _init_baryon_fraction(box, sim_name, data_path):
		'''
		Initializes the baryon fraction for the given simulation.

		Parameters:
		- box (str): The box size of the simulation.
		- sim_name (str): The name of the simulation.
		- data_path (str): The path to the simulation data.

		Returns:
		- baryon_fractions (dict): A dictionary mapping mass ranges to baryon fraction objects.
		'''
		path = f'{data_path}/gas_fraction/{box}/{sim_name}/gas_fraction_*_z=0.000.txt'
		cosmo_snaps = sorted(glob.glob(path))

		print(f'Found baryon fraction for {len(cosmo_snaps)} snapshots in {sim_name}')

		baryon_fractions = {}
		for snap in cosmo_snaps:
			bf = BaryonFractionHandler(snap)
			baryon_fractions[bf.mass_range_str] = bf

		return baryon_fractions
