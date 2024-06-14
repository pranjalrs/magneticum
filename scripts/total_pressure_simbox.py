'''Compute total electron pressure in a simulation box
'''
import argparse
import glob
import numpy as np
import os

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
		Y = np.array(g3read.read_new(this_file, ['Zs  '], [ptype])[:, 0])  # Helium Fraction

		particle_volume = BoxSize**3 * Gadget.units.length**3
		Pe = tools.get_comoving_electron_pressure_Mead(mass, Temp, Y, particle_volume)

# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--box', default='Box1a', type=str)
parser.add_argument('--sim', default='mr_bao', type=str)
parser.add_argument('--redshift_id', default=144, type=str)
args = parser.parse_args()

# sim_box = args.box
# sim_name = args.sim
snap_dir = args.redshift_id

current_directory = os.getcwd()

sim_paths = glob.glob(f'/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/Box3/mr_0.*')

for this_sim in sim_paths:
	this_snap_path = this_sim + '/snapdir_{snap_dir}/snap_{snap_dir}.'
	f = g3read.GadgetFile(this_snap_path+'0')
	BoxSize = f.header.BoxSize/1e3 #Mpc/h ; size of box
	get_total_pressure(this_snap_path)