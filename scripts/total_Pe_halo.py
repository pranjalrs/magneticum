'''
Computes the total Pein a mass bin and given cosmology
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
parser.add_argument('--field', nargs='*')
parser.add_argument('--nhalo', default=2000, type=int)
parser.add_argument('--file_start', default=0, type=int)
parser.add_argument('--file_stop', default=512, type=int)
parser.add_argument('--binning', default='mvir', type=str, choices=['m500c', 'mvir', 'nu_m500c'])
parser.add_argument('--m_min', default=1e11, type=float)
parser.add_argument('--m_max', default=1e16, type=float)
parser.add_argument('--isSave', default=True, type=bool)
args = parser.parse_args()

box = args.box
sim, redshift_id = args.sim, args.redshift_id
file_start, file_stop = args.file_start, args.file_stop
nhalo, binning = args.nhalo, args.binning

bin_limits = {}
bin_limits['m500c'] = (args.m_min, args.m_max)
bin_limits['mvir'] = (args.m_min, args.m_max)

low_bin, high_bin = bin_limits[binning][0], bin_limits[binning][1]

## Define internal units for GADGET, see:  https://wwwmpa.mpa-garching.mpg.de/~kdolag/GadgetHowTo/right.html#Format2
length_unit = 1*u.kpc 
mass_unit = 1e10*u.Msun
#time_unit = 
temp_unit = 1*u.K

# Select redshift and field
# if sim == 'mr_bao':
# 	if redshift == 0:
# 		redshift_id = '144'  # 144: z=0, 052: z=:1.18
# else:
# 	if redshift == 0:
# 		redshift_id = '014'  # 014: z=0

field_dict = {'Pe': {'name': 'Electron Pressure', 'unit': 'keV/cm^3'},
              'Temp': {'name': 'Gas Temperature', 'unit': 'K'}}


## Define paths
utils.set_storage_path()
snap_base = f'{utils.storage_path}{box}/{sim}/snapdir_{redshift_id}/snap_{redshift_id}'

fof = g3read.GadgetFile(group_base + '.0', is_snap=False)  # Read first file to get header information
z = fof.header.redshift
little_h = fof.header.HubbleParam
num_group_files = fof.header.num_files

dtype = np.dtype({'names': ['Halo mass: Rvir (Msun/h)', 'Rvir (kpc/h)', 'Total physical electron Pressure: keV/cm^3'],
 				'titles': ['mvir', 'rvir', 'Pe_total'],
 				'formats': ['float', 'float', 'float']})

print(f'Computing total electron pressure in halos \n')
print(f'Using simulation {box}/{sim} at z={z:.2f}\n')
print(f'Binning in {binning} with min. = {low_bin} and max. = {high_bin}\n')

data = np.array(1, dtype=dtype)

this_cosmo = utils.get_cosmology_dict_from_path(sim)
cosmology.setCosmology('my_cosmo', this_cosmo)

halo_catalog = joblib.load(f'../magneticum-data/data/halo_catalog/{box}/{sim}_sub_{redshift_id}.pkl')

with tqdm(total=nhalo) as pbar:
    inds = np.where((halo_catalog['MVIR']*mass_unit.value >= low_bin) & (halo_catalog['MVIR']*mass_unit.value < high_bin))[0]

    ## Randomly choose n halos
    if len(inds)>nhalo: inds = np.random.choice(inds, nhalo, replace=False)

    halo_positions = halo_catalog['GPOS'][inds]
    halo_mvir = halo_catalog['MVIR'][inds]*mass_unit
    halo_rvir = halo_catalog['RVIR'][inds]
    for i in range(len(halo_positions)):
        this_pos = halo_positions[i]
        this_mvir = halo_mvir[i] * mass_unit
        this_rvir = halo_rvir[i]

        this_Pe_total = utils.get_total_Pe_halo(snap_base, this_pos, this_rvir, z=z, little_h=little_h)

        this_halo_data = np.array(tuple([this_mvir.value, this_rvir, this_Pe_total]), dtype=dtype)

        data = np.append(data, this_halo_data)

        pbar.update(1)

data = np.delete(data, (0), axis=0)

if args.isSave is True:
	os.makedirs(f'../magneticum-data/data/Pe_total/{box}', exist_ok=True)

	if binning == 'mvir':
		joblib.dump(data, f'../magneticum-data/data/Pe_total/{box}/{sim}_z={z:.2f}_{binning}_{low_bin:.1E}_{high_bin:.1E}.pkl')

#	joblib.dump(data, f'../magneticum-data/data/{field}_profile_{estimator}/{sim}/{field}_z={z:.2f}_Mmin_{m_min:.1E}_Mmax_{m_max:.1E}.pkl')
# joblib.dump(data, '../magneticum-data/data/Pe_profile/Pe_start_%d'%file_start+'_stop_%d'%file_stop+'.pkl')
