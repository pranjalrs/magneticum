import argparse
import joblib
import numpy as np
import os

import astropy.units as u
import MAS_library as MASL
import Pk_library as PKL
import g3read

import sys
sys.path.append('../src/')

import Pk_tools
import Pylians_matter
import Pylians_pressure
import Pylians_pressure_halos

parser = argparse.ArgumentParser()
parser.add_argument('--box', default='Box1a', type=str)
parser.add_argument('--snap', default='144', type=str)
parser.add_argument('--grid', default=1024, type=int)
parser.add_argument('--MAS', default='CIC', type=str)
parser.add_argument('--threads', default=1, type=int)
args = parser.parse_args()

box = args.box
snap_dir = args.snap
grid = args.grid
threads = args.threads

current_directory = os.getcwd()
sim = {}
sim['Box1a'] = 'mr_bao'
sim['Box2'] = 'hr_bao'
sim['Box3'] = 'hr_bao'

this_sim = sim[box]

if 'pranjalrs' in current_directory:
	snap_path = f'/xdisk/timeifler/pranjalrs/magneticum_data/{box}/{this_sim}/snapdir_{snap_dir}/snap_{snap_dir}.'

if 'di75sic' in current_directory:
	snap_path = f'/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/{box}/{this_sim}/snapdir_{snap_dir}/snap_{snap_dir}.'

print(f'Path to snap shot files is: {snap_path}')

f = g3read.GadgetFile(snap_path+'0')


# density field parameters
grid    = grid   #the 3D field will have grid x grid x grid voxels
BoxSize = f.header.BoxSize/1e3 #Mpc/h ; size of box
MAS     = args.MAS  #mass-assigment scheme
verbose = True   #print information on progress
threads = threads
axis = 0

Pylians_matter.grid = grid
Pylians_matter.BoxSize = BoxSize
Pylians_matter.MAS = MAS
Pylians_matter.f = f
Pylians_matter.snap_path = snap_path
Pylians_matter.verbose = verbose

Pylians_pressure.grid = grid
Pylians_pressure.BoxSize = BoxSize
Pylians_pressure.MAS = MAS
Pylians_pressure.f = f
Pylians_pressure.snap_path = snap_path
Pylians_pressure.verbose = verbose

Pe_cube = np.zeros((grid,grid,grid), dtype=np.float32)
delta = np.zeros((grid,grid,grid), dtype=np.float32)

Pylians_matter.get_mass_cube(delta, [0, 1, 4, 5])
# Pylians_pressure.get_Pe_Mead_cube(Pe_cube, z=0.0, little_h=0.704)

delta /= np.mean(delta, dtype=np.float64)
delta -= 1.0

## If we want to Mask Halos
catalog ={}
catalog['Box1a'] = joblib.load('../magneticum-data/data/halo_catalog/Box1a/mr_bao_sub_144.pkl')
catalog['Box2'] = joblib.load('../magneticum-data/data/halo_catalog/Box2/hr_bao_sub_144.pkl')
catalog['Box3'] = joblib.load('../magneticum-data/data/halo_catalog/Box3/hr_bao_sub_144.pkl')
cube_path = f'/xdisk/timeifler/pranjalrs/cube/{box}_Pe_Mead_CIC_R1024.npy'


# Usual Pk
Pe_cube = np.load(cube_path)
Pk = PKL.XPk(np.array([delta, Pe_cube]), BoxSize, axis, [MAS, MAS], verbose)

np.savetxt(f'../../magneticum-data/data/Pylians/Pk_matterxpressure/{box}_{MAS}_R{grid}.txt', np.column_stack((Pk.k3D, Pk.XPk[:,0, 0])), delimiter='\t')

## Halo Only
## n=1
halos, _ = Pylians_pressure_halos.get_halo_only_cube(cube_path, catalog[box], BoxSize, mmin=1e14, mmax=1e16, n=1)
Pe_cube = np.load(cube_path) - halos

Pk = PKL.XPk(np.array([delta, Pe_cube]), BoxSize, axis, [MAS, MAS], verbose)

np.savetxt(f'../../magneticum-data/data/Pylians/Pk_matterxpressure/{box}_halo_only_n1_{MAS}_R{grid}.txt', np.column_stack((Pk.k3D, Pk.XPk[:,0, 0])), delimiter='\t')

# ## n=3
halos, _ = Pylians_pressure_halos.get_halo_only_cube(cube_path, catalog[box], BoxSize, mmin=1e14, mmax=1e16, n=3)
Pe_cube = np.load(cube_path) - halos

Pk = PKL.XPk(np.array([delta, Pe_cube]), BoxSize, axis, [MAS, MAS], verbose)

np.savetxt(f'../../magneticum-data/data/Pylians/Pk_matterxpressure/{box}_halo_only_n3_{MAS}_R{grid}.txt', np.column_stack((Pk.k3D, Pk.XPk[:,0, 0])), delimiter='\t')