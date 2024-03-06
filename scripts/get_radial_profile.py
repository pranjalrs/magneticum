'''
Computes the radial profile for a given field in a mass bin and given cosmology
By default only selects 2000 halos in a mass bin
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

import sim_tools
import utils



# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--box', default='Box1a', type=str)
parser.add_argument('--sim', default='mr_bao', type=str)
parser.add_argument('--redshift_id', default=144, type=str)
parser.add_argument('--field')
parser.add_argument('--estimator', default='median', type=str)
parser.add_argument('--nhalo', default=2000, type=int)
parser.add_argument('--file_start', default=0, type=int)
parser.add_argument('--file_stop', default=512, type=int)
parser.add_argument('--binning', default='mvir', type=str, choices=['m500c', 'mvir', 'nu_m500c'])
parser.add_argument('--nu_min', default=1., type=float)
parser.add_argument('--nu_max', default=4., type=float)
parser.add_argument('--mmin', default=1e11, type=float)
parser.add_argument('--mmax', default=1e16, type=float)
parser.add_argument('--isSave', default=True, type=bool)
parser.add_argument('--test', default=False, type=bool)
args = parser.parse_args()

box = args.box
sim, redshift_id = args.sim, args.redshift_id
field, estimator = args.field.strip('"').split(','), args.estimator
file_start, file_stop = args.file_start, args.file_stop
nhalo, binning = args.nhalo, args.binning

bin_limits = {}
bin_limits['m500c'] = (args.mmin, args.mmax)
bin_limits['mvir'] = (args.mmin, args.mmax)
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

dtype = np.dtype({'names': ['Halo mass: Rvir (Msun/h)', 'Rvir (ckpc/h)', 'Halo mass: R500c (Msun/h)',
 				'R500c (ckpc/h)', 'Halo Position (ckpc/h)', 'Profiles for fields'],
 				'titles': ['mvir', 'rvir', 'm500c', 'r500c', 'gpos', 'fields'],
 				'formats': ['float', 'float', 'float', 'float', object, object]})

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
halo_nu_m500c[mask] = peaks.peakHeight(halo_m500c[mask].value, z=z)
halo_rvir = halo_catalog['RVIR']
halo_r500c = halo_catalog['R5CC']


# Select binning
if binning == 'm500c':
	sort_by = halo_m500c.value

elif binning == 'mvir':
	sort_by = halo_mvir.value

elif binning == 'nu_m500c':
	sort_by = halo_nu_m500c

inds = np.where((sort_by >= low_bin) & (sort_by < high_bin))[0]

## Randomly choose n halos
np.random.seed(0)
if len(inds)>nhalo: inds = np.random.choice(inds, nhalo, replace=False)


halo_positions = halo_positions[inds]
halo_mvir = halo_mvir[inds]
halo_m500c = halo_m500c[inds]
halo_nu_m500c = halo_nu_m500c[inds]
halo_rvir = halo_rvir[inds]
halo_r500c = halo_r500c[inds]


with tqdm(total=nhalo) as pbar:
	# For each halo calculate pressure profile
	for i in range(len(halo_positions)):
		this_pos = halo_positions[i]
		this_mvir = halo_mvir[i]
		this_rvir = halo_rvir[i]
		this_m500c = halo_m500c[i]
		this_nu_m500c = halo_nu_m500c[i]
		this_r500c = halo_r500c[i]
		
		filename = f'../../magneticum-data/data/test/{box}_cent/figures/halo_proj_id_{i}'
		this_profile_data = sim_tools.get_profile_for_halo(snap_base, this_pos, this_rvir, fields=field, recal_cent=True, save_proj=False, filename=filename, estimator=estimator)
		
		this_halo_data = np.array(tuple([this_mvir.value, this_rvir, this_m500c.value, this_r500c, this_pos, this_profile_data]), dtype=dtype)

		data = np.append(data, this_halo_data)

		pbar.update(1)


data = np.delete(data, (0), axis=0)

if args.isSave is True:
	os.makedirs(f'../../magneticum-data/data/profiles_{estimator}/{box}/', exist_ok=True)

	if args.test is True:
		os.makedirs(f'../../magneticum-data/data/test/{box}/', exist_ok=True)
		np.save(f'../../magneticum-data/data/test/{box}/{field}_z={z:.2f}_{binning}_{low_bin:.1E}_{high_bin:.1E}_nhalo{nhalo}.npy', data)
		joblib.dump(data, f'../../magneticum-data/data/test/{box}/{field}_z={z:.2f}_{binning}_{low_bin:.1E}_{high_bin:.1E}_nhalo{nhalo}.pkl')

	elif binning == 'm500c':	
		joblib.dump(data, f'../../magneticum-data/data/profiles_{estimator}/{box}/{field}_z={z:.2f}_{binning}_{low_bin:.1E}_{high_bin:.1E}.pkl')

	elif binning == 'nu_m500c':
		joblib.dump(data, f'../../magneticum-data/data/profiles_{estimator}/{box}/{field}_z={z:.2f}_{binning}_{low_bin:.1f}_{high_bin:.1f}.pkl')

	elif binning == 'mvir':
		joblib.dump(data, f'../../magneticum-data/data/profiles_{estimator}/{box}/{"_".join(field)}_z={z:.2f}_{binning}_{low_bin:.1E}_{high_bin:.1E}_nhalo{nhalo}.pkl')


#	joblib.dump(data, f'../magneticum-data/data/{field}_profile_{estimator}/{sim}/{field}_z={z:.2f}_Mmin_{m_min:.1E}_Mmax_{m_max:.1E}.pkl')
# joblib.dump(data, '../magneticum-data/data/Pe_profile/Pe_start_%d'%file_start+'_stop_%d'%file_stop+'.pkl')
