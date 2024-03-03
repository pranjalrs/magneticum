# This will be a class for handling simulation data
# Each simulation will have cosmology parameters as attributes
# and attributes for power spectrum suppression at different redshifts
import glob
import numpy as np
import re

import utils

class PowerSpectrum:
    def __init__(self, path):
        self._init_power_spectrum(path)

    def _init_power_spectrum(self, path):
        match = re.search(r'z=([0-9.]+)', path.split('/')[-1])
        z = match.group(1)
        self.z = float(z)

        self.k, self.Pk_hydro = np.loadtxt(self.path, unpack=True)

        if 'hr_bao' in path:
            snap = snap.replace('hr_bao', 'hr_dm')

        else:
            snap = snap.replace('hr', 'dm_hr')

        _, self.Pk_dm = np.loadtxt(self.path, unpack=True)

        self.Pk_ratio = self.Pk_hydro/self.Pk_dm


class Simulation:
    def __call__(self, sim_name, data_path='./'):
        
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
        
        self.sim_name = sim_name
        self.sim_num = self.sim_num_map[sim_name]
        self._init_cosmology(sim_name)
        self._get_power_spectrum(sim_name, data_path)

    def _init_cosmology(self, sim_name):
        cosmo = utils.get_cosmology_dict_from_path(sim_name)
        self.omega_m = cosmo['Om0']
        self.omega_b = cosmo['Ob0']
        self.sigma8 = cosmo['sigma8']
        self.h = cosmo['H0']/100
        self.fb = cosmo['Ob0']/cosmo['Om0']
    
    def _init_power_spectrum(self, sim_name, data_path):
        cosmo_snaps = sorted(glob.glob(f'{data_path}/{sim_name}/Pk_{sim_name}_z=*_R1024.txt'))

        self.Pks = {}
        for snap in cosmo_snaps:
            Pk = PowerSpectrum(snap)
            self.Pks[Pk.z] = Pk