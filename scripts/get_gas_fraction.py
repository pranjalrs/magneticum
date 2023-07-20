'''
Computes the gas fraction of halos in a mass bins within R500c, Rvir, etc
'''
import argparse
import astropy.units as u
import astropy.constants as constants
import joblib
import numpy as np
import os 
from tqdm import tqdm

import g3read
import sys

import sys
sys.path.append('../core/')

import utils



# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--box', default='Box1a', type=str)
parser.add_argument('--redshift_id', default=144, type=str)
parser.add_argument('--m_min', default=1e11, type=float)
parser.add_argument('--m_max', default=1e16, type=float)

args = parser.parse_args()

box = args.box
redshift_id = args.redshift_id
sims = {'Box1a': 'mr_bao', 'Box2': 'hr_bao', 'Box3': 'hr_bao'}
sim = sims[box]

m_min, m_max = (args.m_min, args.m_max)

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


## Select halos in mass bin
sort_by = halo_mvir.value
inds = np.where((sort_by >= m_min) & (sort_by < m_max))[0]

with tqdm(total=len(inds)) as pbar:
	halo_positions = halo_positions[inds]
	halo_mvir = halo_mvir[inds]
	halo_m500c = halo_m500c[inds]
	halo_rvir = halo_rvir[inds]
	halo_r500c = halo_r500c[inds]


	# For each halo calculate pressure profile
	for i in range(len(halo_positions)):
		this_pos = halo_positions[i]
		this_mvir = halo_mvir[i]
		this_rvir = halo_rvir[i]
		this_m500c = halo_m500c[i]
		this_r500c = halo_r500c[i]

		this_fgas_rvir = utils.get_fgas_halo(snap_base, this_pos, this_rvir, z=z, little_h=little_h)
		this_fgas_3rvir = utils.get_fgas_halo(snap_base, this_pos, 3*this_rvir, z=z, little_h=little_h)

		this_fgas_r500c = utils.get_fgas_halo(snap_base, this_pos, this_r500c, z=z, little_h=little_h)
        
		data.append([this_mvir.value, this_fgas_r500c, this_fgas_rvir, this_fgas_3rvir-this_fgas_rvir])

		pbar.update(1)


header = f'''Gas fractions for {m_min:.1E}<Mvir<{m_max:.1E} for {box}
f_bnd = fgas(<R); f_ejec = fgas(<3R)-fgas(<R)
\n Mvir \t f_bnd_r500c \t f_bnd_rvir \t f_ejec_rvir'''
np.savetxt(f'../magneticum-data/data/gas_fraction/{box}/gas_fraction_mvir_{m_min:.1E}_{m_max:.1E}.txt', np.vstack((data)), header=header, comments='#', delimiter='\t')