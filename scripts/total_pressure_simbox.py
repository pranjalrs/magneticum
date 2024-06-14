'''Compute total electron pressure in a simulation box
'''
import argparse
import glob
import numpy as np
import os


import astropy.units as u
import astropy.cosmology.units as cu
import g3read

from dawn.sim_toolkit import tools
from dawn.gadget import Gadget
from dawn.sim_toolkit import tools

def get_total_pressure(snap_path):
	pressure = 0.

	for i in range(f.header.num_files):
		this_file = snap_path + str(i)
		ptype = 0 # For gas
		mass = np.array(g3read.read_new(this_file, ['MASS'], [ptype])[ptype]['MASS'])
		Temp = np.array(g3read.read_new(this_file, ['TEMP'], [ptype])[ptype]['TEMP'])
		Y = g3read.read_new(this_file, ['Zs  '], [ptype])[0]['Zs  '][:, 0]

		particle_volume = BoxSize**3 * Gadget.units.length**3
		Pe = tools.get_comoving_electron_pressure_Mead(mass, Temp, Y, particle_volume)
		pressure += np.sum(Pe)
		print(i)

	return pressure


def get_stellar_density(snap_path):
	stellar_mass = 0.
	particle_volume = BoxSize**3 * Gadget.units.length**3

	for i in range(f.header.num_files):
		this_file = snap_path + str(i)
		ptype = 4 # For stars
		mass = np.array(g3read.read_new(this_file, ['MASS'], [ptype])[ptype]['MASS']) * Gadget.units.mass
		stellar_mass += np.sum(mass)

		print(i)

	return (stellar_mass/particle_volume).to(u.Msun*cu.littleh**2/u.kpc**3)


# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--box', default='Box1a', type=str)
parser.add_argument('--sim', default='mr_bao', type=str)
parser.add_argument('--redshift_id', default='014', type=str)
args = parser.parse_args()

# sim_box = args.box
# sim_name = args.sim
snap_dir = args.redshift_id

sim_paths = glob.glob(f'/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/Box3/hr_0.*')
sim_paths += ['/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/Box3/hr_bao']

sim_paths = sorted(sim_paths)

result_dict = {}
for this_sim in sim_paths:
	sim_name = this_sim.split('/')[-1]
	print(sim_name)	

	if sim_name=='hr_bao':
		snap_dir = '144'
	else: snap_dir = '014'

	this_snap_path = this_sim + f'/snapdir_{snap_dir}/snap_{snap_dir}.'
	f = g3read.GadgetFile(this_snap_path+'0')
	BoxSize = f.header.BoxSize #Mpc/h ; size of box

	stellar_density = get_stellar_density(this_snap_path)
	total_pressure = get_total_pressure(this_snap_path)

	result_dict[sim_name] = [stellar_density, total_pressure]


#np.savetxt('Box3_MC_attributes.txt', 
