import argparse
import numpy as np
import os

import MAS_library as MASL
import Pk_library as PKL
import g3read

import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--box', default='Box1a', type=str)
parser.add_argument('--sim', default='mr_bao', type=str)
parser.add_argument('--snap_dir', default='144', type=str)
parser.add_argument('--grid', default=1024, type=int)
parser.add_argument('--MAS', type=str)
parser.add_argument('--threads', default=1, type=int)


def get_mass_cube(delta, ptype):
	pos = []
	mass = []

	for i in range(f.header.num_files):
		this_file = snap_path + str(i)

		for this in ptype:
			pos = np.array(g3read.read_new(this_file, ['POS '], [this])[this]['POS ']*1e-3)

			if this in [0, 1, 4, 2]:
				mass = np.array(g3read.read_new(this_file, ['MASS'], [this])[this]['MASS']*1e10)

			elif this==5:
				try:
					mass = np.array(g3read.read_new(this_file, ['BHMA'], [this])[this]['BHMA']*1e10)
				except:
					print(f'Block BHMA not found in file {i}')

			pos, mass = pos.astype('float32'), mass.astype('float32')

			MASL.MA(pos, delta, BoxSize, MAS, W=mass, verbose=verbose)


		print(i)

if __name__== '__main__':
	args = parser.parse_args()

	sim_box = args.box
	sim_name = args.sim
	snap_dir = args.snap_dir
	grid = args.grid
	threads = args.threads

	current_directory = os.getcwd()

	if 'pranjalrs' in current_directory:
		snap_path = f'/xdisk/timeifler/pranjalrs/magneticum_data/{sim_box}/{sim_name}/snapdir_{snap_dir}/snap_{snap_dir}.'

	if 'di75sic' in current_directory:
		snap_path = f'/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/{sim_box}/{sim_name}/snapdir_{snap_dir}/snap_{snap_dir}.'

	print(f'Path to snap shot files is: {snap_path}')

	f = g3read.GadgetFile(snap_path+'0')

	# density field parameters
	grid    = grid   #the 3D field will have grid x grid x grid voxels
	BoxSize = f.header.BoxSize/1e3 #Mpc/h ; size of box
	z = f.header.redshift
	MAS     = args.MAS  #mass-assigment scheme
	verbose = True   #print information on progress
	threads = threads
	axis = 0

	delta = np.zeros((grid,grid,grid), dtype=np.float32)

	if 'dm' not in sim_name:
		get_mass_cube(delta, [0, 1, 4, 5])

	else:
		if 'dm_hr' in sim_name:
			get_mass_cube(delta, [1])

		else:
			get_mass_cube(delta, [1, 2])

	delta /= np.mean(delta, dtype=np.float64)
	delta -= 1.0


	CF = PKL.Xi(delta, BoxSize, axis, MAS, verbose)

	np.savetxt(f'../../magneticum-data/data/Pylians/xi_matter/{sim_box}/xi_{sim_name}_z={z:.2f}_R{grid}.txt', np.column_stack((CF.r3D, CF.xi[:, 0])), delimiter='\t')
