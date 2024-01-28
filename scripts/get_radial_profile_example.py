'''
Same as `get_radial_profile.py` but returns profile for a specific halo
'''
import argparse
import astropy.units as u
import astropy.constants as constants
import joblib
import numpy as np
import os 
from tqdm import tqdm

from colossus.cosmology import cosmology
from colossus.lss import peaks
import g3read

import sys
sys.path.append('../core/')

import utils



# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--box', default='Box1a', type=str)
parser.add_argument('--sim', default='mr_bao', type=str)
parser.add_argument('--redshift_id', default=144, type=str)
parser.add_argument('--field')
parser.add_argument('--estimator', default='median', type=str)
args = parser.parse_args()

box = args.box
sim, redshift_id = args.sim, args.redshift_id
field, estimator = args.field.strip('"').split(','), args.estimator

bin_limits = {}
bin_limits['m500c'] = (args.m_min, args.m_max)
bin_limits['mvir'] = (args.m_min, args.m_max)
bin_limits['nu_m500c'] = (args.nu_min, args.nu_max)

low_bin, high_bin = bin_limits[binning][0], bin_limits[binning][1]

## Define internal units for GADGET, see:  https://wwwmpa.mpa-garching.mpg.de/~kdolag/GadgetHowTo/right.html#Format2
length_unit = 1*u.kpc 
mass_unit = 1e10*u.Msun
temp_unit = 1*u.K

field_dict = {'Pe': {'name': 'Electron Pressure', 'unit': 'keV/cm^3'},
              'Temp': {'name': 'Gas Temperature', 'unit': 'K'}}


## Define paths
utils.set_storage_path()
group_base = f'{utils.storage_path}{box}/{sim}/groups_{redshift_id}/sub_{redshift_id}'
snap_base = f'{utils.storage_path}{box}/{sim}/snapdir_{redshift_id}/snap_{redshift_id}'

fof = g3read.GadgetFile(snap_base + '.0')  # Read first file to get header information
z = fof.header.redshift
little_h = fof.header.HubbleParam

dtype = np.dtype({'names': ['Halo mass: Rvir (Msun/h)', 'Rvir (kpc/h)', 'Halo mass: R500c (Msun/h)',
 				'R500c (kpc/h)', 'Peak height (M500c)', 'Profiles for fields'],
 				'titles': ['mvir', 'rvir', 'm500c', 'r500c', 'nu_m500c', 'fields'],
 				'formats': ['float', 'float', 'float', 'float', 'float', object]})

print(f'Computing profile for field {field} and estimator {estimator} \n')
print(f'Using simulation {box}/{sim} at z={z:.2f}\n')
print(f'Binning in {binning} with min. = {low_bin} and max. = {high_bin}\n')

data = np.array(1, dtype=dtype)

this_cosmo = utils.get_cosmology_dict_from_path(sim)
cosmology.setCosmology('my_cosmo', this_cosmo)

halo_catalog = joblib.load(f'../../magneticum-data/data/halo_catalog/{box}/{sim}_sub_{redshift_id}.pkl')


halo_positions = halo_catalog['GPOS']
halo_mvir = halo_catalog['MVIR']*mass_unit
halo_m500c = halo_catalog['M5CC']*mass_unit
mask = halo_m500c.value!=0
halo_nu_m500c = np.zeros_like(halo_m500c.value)
halo_rvir = halo_catalog['RVIR']
halo_r500c = halo_catalog['R5CC']


# For each halo calculate pressure profile
this_pos = halo_positions[i]
this_mvir = halo_mvir[i]
this_rvir = halo_rvir[i]
this_m500c = halo_m500c[i]
this_nu_m500c = halo_nu_m500c[i]
this_r500c = halo_r500c[i]

this_profile_data = utils.get_profile_for_halo(snap_base, this_pos, this_rvir, fields=field, z=z, little_h=little_h, estimator=estimator)

this_halo_data = np.array(tuple([this_mvir.value, this_rvir, this_m500c.value, this_r500c, this_nu_m500c, this_profile_data]), dtype=dtype)

data = np.append(data, this_halo_data)
