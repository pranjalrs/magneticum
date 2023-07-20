'''Script for computing the pressure power spectrum using only
halos with mass > Mmin
'''
import argparse
from copy import deepcopy
import joblib
import numpy as np

import astropy.units as u
import Pk_library as PKL

import sys
sys.path.append('../src/')

import Pk_tools


def get_halo_only_cube(cube_path, catalog, box_size, mmin, mmax, n=1):
	cube = np.load(cube_path)
	cube_sum = np.sum(cube)  # For computing masked fraction in case of Pe

	resolution = np.shape(cube)[0]
	N = resolution**3
	boxsize = box_size*u.Mpc

	## Select halos
	ind = np.where((catalog['MVIR']*1e10>mmin) & (catalog['MVIR']*1e10<mmax))[0]

	pos = catalog['GPOS'][ind]*u.kpc  #in kpc/h
	rvir = catalog['RVIR'][ind]*u.kpc

	pos_in_pix = np.array(np.round((pos/boxsize).decompose()*resolution), int)
	rvir_in_pix = np.array(np.round(n*(rvir/boxsize).decompose()*resolution), int)

	halo_only_cube = np.zeros_like(cube)

	masked_fraction = 0.0
	for i, index in enumerate(pos_in_pix):
		nx, ny, nz = index
		mask_pix = rvir_in_pix[i]

		halo_only_cube[nx-mask_pix:nx+mask_pix, ny-mask_pix:ny+mask_pix, nz-mask_pix:nz+mask_pix] = cube[nx-mask_pix:nx+mask_pix, ny-mask_pix:ny+mask_pix, nz-mask_pix:nz+mask_pix]

   
	return halo_only_cube, np.sum(halo_only_cube)/np.sum(cube)


def save_halo_only_Pk(path, catalog, box_size, mmin, mmax, n):
	halo_only_cube, frac = get_halo_only_cube(path, catalog, box_size, float(mmin), float(mmax), n)
	Pk = PKL.Pk(halo_only_cube, box_size, axis=0, MAS='CIC', verbose=True)
	k, Pk = Pk.k3D, Pk.Pk[:, 0]
	
	metadata = f'Pressure power spectra from halos with {float(mmin):.2E} Msun < Mvir < {float(mmax):.2E}< Msun \n Only include halos up to n={n}*Rvir \n Fraction of total Pe in halos is {frac}'
	np.savetxt(f'{save_path}{mmin}_{mmax}_n{n}.txt', np.column_stack((k, Pk)), header=metadata, comments='#', delimiter='\t')


def save_everything_but_halo_Pk(path, catalog, box_size, mmin, mmax='1e17', n=1):
	halo_only_cube, frac = get_halo_only_cube(path, catalog, box_size, float(mmin), float(mmax), n)
	everything_but_halo_cube = 	np.load(cube_path) - halo_only_cube
                            
	Pk = PKL.Pk(everything_but_halo_cube, box_size, axis=0, MAS='CIC', verbose=True)
	k, Pk = Pk.k3D, Pk.Pk[:, 0]

	metadata = f'Pressure power spectra from excluding all halos halos with Mvir > {float(mmin):.2E} Msun \n Only include halos up to n={n}*Rvir \n Fraction of total Pe in halos is {frac}'
	np.savetxt(f'Pylians_output/Pk_pressure/{box}/Pk_exclude_all_halos_n{n}.txt', np.column_stack((k, Pk)), header=metadata, comments='#', delimiter='\t')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--box', default='Box1a', type=str)
	args = parser.parse_args()
	
	box = args.box
	
	## Load halo catalogs
	catalog ={}
	catalog['Box1a'] = joblib.load('../magneticum-data/data/halo_catalog/Box1a/mr_bao_sub_144.pkl')
	catalog['Box2'] = joblib.load('../magneticum-data/data/halo_catalog/Box2/hr_bao_sub_144.pkl')
	catalog['Box3'] = joblib.load('../magneticum-data/data/halo_catalog/Box3/hr_bao_sub_144.pkl')

	box_size = {}  # In Mpc/h
	box_size['Box1a'] = 896
	box_size['Box2'] = 352
	box_size['Box3'] = 128

	this_catalog = catalog[box]
	this_box_size = box_size[box]
	cube_path = f'/xdisk/timeifler/pranjalrs/cube/{box}_Pe_Mead_CIC_R1024.npy'
	save_path = f'Pylians_output/Pk_pressure/{box}/Pk_halos_only_'

	# ## Get halos up to only 1 Virial radius
	n = 1.2

	save_halo_only_Pk(cube_path, this_catalog, this_box_size, mmin='1e12', mmax='1e17', n=n)
	# save_halo_only_Pk(cube_path, this_catalog, this_box_size, mmin='1e15', mmax='1e17', n=n)
	# save_halo_only_Pk(cube_path, this_catalog, this_box_size, mmin='1e14', mmax='1e15', n=n)
	# save_halo_only_Pk(cube_path, this_catalog, this_box_size, mmin='1e13', mmax='1e14', n=n)
	# save_halo_only_Pk(cube_path, this_catalog, this_box_size, mmin='1e12', mmax='1e13', n=n)
	# save_everything_but_halo_Pk(cube_path, this_catalog, this_box_size, mmin='1e12', n=n)

	# ## Get halos up to only 1 Virial radius
	# n = 3

	# save_halo_only_Pk(cube_path, this_catalog, this_box_size, mmin='1e12', mmax='1e17', n=n)
	# save_halo_only_Pk(cube_path, this_catalog, this_box_size, mmin='1e15', mmax='1e17', n=n)
	# save_halo_only_Pk(cube_path, this_catalog, this_box_size, mmin='1e14', mmax='1e15', n=n)
	# save_halo_only_Pk(cube_path, this_catalog, this_box_size, mmin='1e13', mmax='1e14', n=n)
	# save_halo_only_Pk(cube_path, this_catalog, this_box_size, mmin='1e12', mmax='1e13', n=n)
	# save_everything_but_halo_Pk(cube_path, this_catalog, this_box_size, mmin='1e12', n=n)
