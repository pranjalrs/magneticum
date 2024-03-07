# This will be a class for handling simulation data
# Each simulation will have cosmology parameters as attributes
# and attributes for power spectrum suppression at different redshifts
import glob
import numpy as np
import re

import utils

class PowerSpectrum:
	def __init__(self, path):
		self._load_power_spectrum(path)

	def _load_power_spectrum(self, path):
		self.z = utils.search_z_in_string(path)

		self.k, self.Pk_hydro = self._loadtxt(path, unpack=True)

		if 'hr_bao' in path:
			path = path.replace('hr_bao', 'hr_dm')

		else:
			path = path.replace('hr', 'dm_hr')

		_, self.Pk_dm = self._loadtxt(path, unpack=True)
		
		if (self.Pk_hydro is not None) and (self.Pk_dm is not None):
			self.Pk_ratio = self.Pk_hydro/self.Pk_dm
		
		else: self.Pk_ratio = None
	
	@staticmethod
	def _loadtxt(path, **kwargs):
		try:
			return np.loadtxt(path, **kwargs)
		
		except FileNotFoundError:
			print(f'File not found {path}')
			return None, None


# Class for handling gas and baryon fractions
class BaryonFraction():
	def __init__(self, path) -> None:
		self._load_baryon_fraction(path)
	
	def _load_baryon_fraction(self, path):
		self.z = utils.search_z_in_string(path)
		self.mmin, self.mmax = utils.search_mass_range_in_string(path)

		# save mass range as string in scientific notation
		self.mass_range_str = f'{self.mmin:.2E}_{self.mmax:.2E}'

		data = self._loadtxt(path)

		self.halo_mass = data[:, 0]
		self.fgas_r500c = data[:, 1]
		self.fgas_rvir = data[:, 2]
		self.fbar_r500c = data[:, 3]
		self.fbar_rvir = data[:, 4]

		self.mean_fgas_r500c = np.mean(self.fgas_r500c)
		self.mean_fgas_rvir = np.mean(self.fgas_rvir)
		self.mean_fbar_r500c = np.mean(self.fbar_r500c)
		self.mean_fbar_rvir = np.mean(self.fbar_rvir)


	@staticmethod
	def _loadtxt(path, **kwargs):
		try:
			return np.loadtxt(path, **kwargs)
		
		except FileNotFoundError:
			print(f'File not found {path}')
			return None, None


class SimDataHandler:
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
	'hr_0.428_0.0492_0.830_0.732': 'C15'}
		
		if sim_name is not None:
			self.sim_name = sim_name
			self.sim_num = self.sim_num_map[sim_name]
		
		elif sim_num is not None:
			self.sim_num = sim_num
			self.sim_name = [k for k,v in self.sim_num_map.items() if v == sim_num][0]
		
		else:
			raise ValueError('Must provide either sim_name or sim_num')

		self.box = box
		self._init_cosmology(self.sim_name)
		self.Pks = self._init_power_spectrum(self.box, self.sim_name, data_path)
		self.fbars = self._init_baryon_fraction(self.box, self.sim_name, data_path)

	def _init_cosmology(self, sim_name):
		cosmo = utils.get_cosmology_dict_from_path(sim_name)
		self.omega_m = cosmo['Om0']
		self.omega_b = cosmo['Ob0']
		self.sigma8 = cosmo['sigma8']
		self.h = cosmo['H0']/100
		self.fb = cosmo['Ob0']/cosmo['Om0']
		
		self.cosmo_label = "$\Omega_m={:.3f},\,\sigma_8={:.3f},\,h={:.3f},f_b={:.3f}$".format(cosmo['Om0'],cosmo['sigma8'],self.h,self.fb)
	
	@staticmethod
	def _init_power_spectrum(box, sim_name, data_path):
		# Assumes that Pk is stored in data/Pylians/Pk_matter/
		cosmo_snaps = sorted(glob.glob(f'{data_path}/data/Pylians/Pk_matter/{box}/Pk_{sim_name}_z=*_R1024.txt'))

		print(f'Found Pk for {len(cosmo_snaps)} snapshots in {sim_name}')
		Pks = {}
		for snap in cosmo_snaps:
			Pk = PowerSpectrum(snap)
			Pks[Pk.z] = Pk
		
		return Pks
	
	@staticmethod
	def _init_baryon_fraction(box, sim_name, data_path):
		cosmo_snaps = sorted(glob.glob(f'{data_path}/gas_fraction/{box}/{sim_name}/gas_fraction_*_z=0.000.txt'))

		print(f'Found baryon fraction for {len(cosmo_snaps)} snapshots in {sim_name}')

		baryon_fractions = {}
		for snap in cosmo_snaps:
			bf = BaryonFraction(snap)
			baryon_fractions[bf.mass_range_str] = bf