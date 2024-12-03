'''Script for computing the matter correlation function using
halos with mass > Mmin
'''
import argparse
from copy import deepcopy
import joblib
import numpy as np

import astropy.units as u
import Pk_library as PKL

def process_halos(cube_path, catalog, box_size, mmin, mmax, n, include_diffuse=True):
	"""
	Processes halo data by masking regions in a 3D cube based on halo positions and sizes.

	Parameters:
	cube_path (str): Path to the numpy file containing the 3D cube data.
	catalog (dict): Dictionary containing halo catalog data with keys 'MVIR', 'GPOS', and 'RVIR'.
	box_size (float): Size of the simulation box in Mpc.
	mmin (float): Minimum halo mass threshold in solar masses.
	mmax (float): Maximum halo mass threshold in solar masses.
	n (int): Scaling factor for the halo radius in pixels.
	include_diffuse (bool, optional): If True, includes diffuse regions (i.e. stuff outside halos) in the processed cube. Defaults to True.

	Returns:
	tuple: A tuple containing:
		- processed_cube (numpy.ndarray): The processed 3D cube with masked halo regions.
		- float: The ratio of the sum of the processed cube to the sum of the original cube.
	"""
	cube = np.load(cube_path)

	resolution = np.shape(cube)[0]
	boxsize = box_size * u.Mpc

	# Select halos
	if include_diffuse:
		ind = np.where((catalog['MVIR'] * 1e10 < mmin) & (catalog['MVIR'] * 1e10 > mmax))[0]
	else:
		ind = np.where((catalog['MVIR'] * 1e10 > mmin) & (catalog['MVIR'] * 1e10 < mmax))[0]

	pos = catalog['GPOS'][ind] * u.kpc  # in kpc/h
	rvir = catalog['RVIR'][ind] * u.kpc

	pos_in_pix = np.array(np.round((pos / boxsize).decompose() * resolution), int)
	rvir_in_pix = np.array(np.round(n * (rvir / boxsize).decompose() * resolution), int)

	processed_cube = cube.copy() if include_diffuse else np.zeros_like(cube)

	for i, index in enumerate(pos_in_pix):
		nx, ny, nz = index
		mask_pix = rvir_in_pix[i]

		# Define the range of indices to be masked
		x_min, x_max = max(nx - mask_pix, 0), min(nx + mask_pix, resolution)
		y_min, y_max = max(ny - mask_pix, 0), min(ny + mask_pix, resolution)
		z_min, z_max = max(nz - mask_pix, 0), min(nz + mask_pix, resolution)

		# Apply the mask to the processed_cube
		if include_diffuse:
			processed_cube[x_min:x_max, y_min:y_max, z_min:z_max] = 0
		else:
			processed_cube[x_min:x_max, y_min:y_max, z_min:z_max] = cube[x_min:x_max, y_min:y_max, z_min:z_max]

	return processed_cube, np.sum(processed_cube) / np.sum(cube)

def save_halo_xi(path, catalog, box_size, mmin, mmax, n):
	halo_only_cube, frac = process_halos(path, catalog, box_size, float(mmin), float(mmax), n, include_diffuse=True)
	CF = PKL.Xi(halo_only_cube, box_size, axis, MAS, verbose)
	r, xi = CF.r3D, CF.xi[:, 0]

	metadata = f'Matter correlation function from halos with {float(mmin):.2E} Msun < Mvir < {float(mmax):.2E}< Msun \n Only include halos up to n={n}*Rvir \n Fraction of total mass in halos is {frac}'
	np.savetxt(f'{save_path}_halos_only_{mmin}_{mmax}_n{n}.txt', np.column_stack((r, xi)), header=metadata, comments='#', delimiter='\t')


	halo_only_cube, frac = process_halos(path, catalog, box_size, float(mmin), float(mmax), n, exclude_halos=True)
	CF = PKL.Xi(halo_only_cube, box_size, axis, MAS, verbose)
	r, xi = CF.r3D, CF.xi[:, 0]

	metadata = f'Matter correlation function from halos (+diffuse component) with {float(mmin):.2E} Msun < Mvir < {float(mmax):.2E}< Msun \n Only include halos up to n={n}*Rvir \n Fraction of total mass in halos is {frac}'
	np.savetxt(f'{save_path}_halos_+_diffuse_{mmin}_{mmax}_n{n}.txt', np.column_stack((r, xi)), header=metadata, comments='#', delimiter='\t')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--box', default='Box1a', type=str)
	args = parser.parse_args()

	box = args.box
	MAS = 'CIC'
	verbose = True
	axis = 0

	## Load halo catalogs
	catalog ={}
	catalog['Box1a'] = joblib.load('../../magneticum-data/data/halo_catalog/Box1a/mr_bao_sub_144.pkl')
	catalog['Box2'] = joblib.load('../../magneticum-data/data/halo_catalog/Box2/hr_bao_sub_144.pkl')
	catalog['Box3'] = joblib.load('../../magneticum-data/data/halo_catalog/Box3/hr_bao_sub_144.pkl')

	box_size = {}  # In Mpc/h
	box_size['Box1a'] = 896
	box_size['Box2'] = 352
	box_size['Box3'] = 128

	this_catalog = catalog[box]
	this_box_size = box_size[box]
	cube_path = f'/xdisk/timeifler/pranjalrs/magneticum_data/{box}/cube_delta_hr_bao_z=0.00_2048_downsample.npy'
	save_path = f'../../magneticum-data/data/Pylians/xi_matter/{box}/xi_'

	# ## Get halos up to only 1 Virial radius
	n = 1.

	save_halo_xi(cube_path, this_catalog, this_box_size, mmin='1e12', mmax='1e17', n=n)