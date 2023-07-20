'''
Computes the median WHIM (Warm hot intergalactic medium) temperature.
This is defnied as the gas paricles in a shell Rvir < r < 3Rvir.
'''
import argparse
import astropy.units as u
import astropy.constants as constants
import joblib
import numpy as np
import os 
from tqdm import tqdm

import g3read

import utils

# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--box', default='Box1a', type=str)
parser.add_argument('--redshift_id', default=144, type=str)
parser.add_argument('--mmin', default=1e11, type=float)
parser.add_argument('--mmax', default=1e16, type=float)

args = parser.parse_args()

box = args.box
redshift_id = args.redshift_id
sims = {'Box1a': 'mr_bao', 'Box2': 'hr_bao', 'Box3': 'hr_bao'}
sim = sims[box]

m_min, m_max = (args.mmin, args.mmax)

## Define internal units for GADGET, see:  https://wwwmpa.mpa-garching.mpg.de/~kdolag/GadgetHowTo/right.html#Format2
length_unit = 1*u.kpc 
mass_unit = 1e10*u.Msun
temp_unit = 1*u.K


## Define paths
utils.set_storage_path()
group_base = f'{utils.storage_path}{box}/{sim}/groups_{redshift_id}/sub_{redshift_id}'
snap_base = f'{utils.storage_path}{box}/{sim}/snapdir_{redshift_id}/snap_{redshift_id}'

fof = g3read.GadgetFile(snap_base + '.0')  # Read first file to get header information
z = fof.header.redshift
little_h = fof.header.HubbleParam



halo_catalog = joblib.load(f'../magneticum-data/data/halo_catalog/{box}/{sim}_sub_{redshift_id}.pkl')

data = []
halo_positions = halo_catalog['GPOS']
halo_mvir = halo_catalog['MVIR']*mass_unit
halo_m500c = halo_catalog['M5CC']*mass_unit
mask = halo_m500c.value!=0
halo_nu_m500c = np.zeros_like(halo_m500c.value)
halo_rvir = halo_catalog['RVIR']
halo_r500c = halo_catalog['R5CC']


sort_by = halo_mvir.value
## Choose halos in mass bin
inds = np.where((sort_by >= m_min) & (sort_by < m_max))[0]


halo_positions = halo_positions[inds]
halo_mvir = halo_mvir[inds]
halo_m500c = halo_m500c[inds]
halo_rvir = halo_rvir[inds]
halo_r500c = halo_r500c[inds]


with tqdm(total=len(inds)) as pbar:
	for i in range(len(halo_positions)):
		this_pos = halo_positions[i]
		this_mvir = halo_mvir[i]
		this_rvir = halo_rvir[i]
		this_m500c = halo_m500c[i]
		this_r500c = halo_r500c[i]

		radius = this_rvir
		halo_center = this_pos

		ptype = 0, 1  # For gas, DM
		particle_data = g3read.read_particles_in_box(snap_base, halo_center, 3*radius, ['TEMP', 'POS '], ptype, use_super_indexes=True)


		# gas mask: r<RVIR
		part_distance_from_center = g3read.to_spherical(particle_data[0]['POS '], halo_center).T[0]
		mask_gas = np.where(part_distance_from_center<radius)[0]
		Tgas_RVIR = np.median(particle_data[0]['TEMP'][mask_gas])

		# gas mask: r<3RVIR
		part_distance_from_center = g3read.to_spherical(particle_data[0]['POS '], halo_center).T[0]
		mask_gas = np.where(part_distance_from_center<3*radius)[0]
		Tgas_3RVIR = np.median(particle_data[0]['TEMP'][mask_gas])

		# gas mask: RVIR<r<3RVIR
		part_distance_from_center = g3read.to_spherical(particle_data[0]['POS '], halo_center).T[0]
		mask_gas = np.where((part_distance_from_center<3*radius) & (part_distance_from_center>radius))[0]
		Tgas_WHIM = np.median(particle_data[0]['TEMP'][mask_gas])


		data.append([this_mvir.value, Tgas_RVIR, Tgas_3RVIR, Tgas_WHIM])

		pbar.update(1)


header = f'''Gas Temperature {m_min:.1E}<Mvir<{m_max:.1E} for {box}
\n Mvir \t Temp_Rvir \t Temp_3Rvir \t Temp_WHIM'''
np.savetxt(f'../magneticum-data/data/Temperature/{box}/Temp_mvir_{m_min:.1E}_{m_max:.1E}.txt', np.vstack((data)), header=header, comments='#', delimiter='\t')