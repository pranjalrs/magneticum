import argparse
from copy import deepcopy
import joblib
import numpy as np

import astropy.units as u
import g3read
import Pk_library as PKL

import src.Pk_tools as Pk_tools

class PyliansTools():
	def __init__(self, box, sim, ):
		self.box = box
		self.sim = sim
		self.snap_dir = snap_dir
		

		self.num_files = g3read.GadgetFile(snap_path+'0').header.num_files
		self.boxsize = g3read.GadgetFile(snap_path+'0').header.BoxSize/1e3 #Mpc/h ; size of box

		current_directory = os.getcwd()
		if 'pranjalrs' in current_directory:
			self.snap_path = f'/xdisk/timeifler/pranjalrs/magneticum_data/{box}/{sim}/snapdir_{snap_dir}/snap_{snap_dir}.'

		if 'di75sic' in current_directory:
			self.snap_path = f'/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/{box}/{sim}/snapdir_{snap_dir}/snap_{snap_dir}.'

	def get_mass_cube(self, delta, ptype, MAS='CIC', verbose=True):
		pos = []
		mass = []

		for i in range(self.num_files):
			this_file = self.snap_path + str(i)

			for this in ptype:
				pos = np.array(g3read.read_new(this_file, ['POS '], [this])[this]['POS ']*1e-3)

				if this in [0, 1, 4, 2]:
					mass = np.array(g3read.read_new(this_file, ['MASS'], [this])[this]['MASS']*1e10)

				elif this==5:
					mass = np.array(g3read.read_new(this_file, ['BHMA'], [this])[this]['BHMA']*1e10)
				
				pos, mass = pos.astype('float32'), mass.astype('float32')
				MASL.MA(pos, delta, self.boxsize, MAS, W=mass, verbose=verbose)


	def get_Pe_cube(self, ptype=0):
		for i in range(self.num_files):
			this_snap = g3read.GadgetFile(self.snap_path + str(i))
			Pe = Pk_tools.get_field(ptype, this_snap, 'Pe')
			pos = this_snap.read_new('POS ', ptype)*1e-3
			m_over_rho = np.array(this_snap.read_new('MASS', ptype))/np.array(this_snap.read_new('RHO ', ptype))
	
			pos, Pe = pos.astype('float32'), Pe.astype('float32')
			m_over_rho = m_over_rho.astype('float32')
			MASL.MA(pos, Pe_cube, BoxSize, MAS, W=Pe*m_over_rho, verbose=verbose)
			MASL.MA(pos, norm, BoxSize, MAS, W=m_over_rho, verbose=verbose)


	def get_Pe_Mead_cube(self, Pe_cube, z, little_h, ptype=0):
		grid = Pe_cube.shape[0]        
		cell_volume = (self.boxsize*u.Mpc/grid)**3
		for i in range(self.num_files):
			this_snap = g3read.GadgetFile(self.snap_path + str(i))
			Pe = Pk_tools.get_field(ptype, this_snap, 'Pe_Mead', z, little_h, cell_volume)
			pos = this_snap.read_new('POS ', ptype)*1e-3


			pos, Pe = pos.astype('float32'), Pe.astype('float32')
			MASL.MA(pos, Pe_cube, BoxSize, MAS, W=Pe, verbose=verbose)


	def get_halo_only_cube(self, mmin, mmax, n=1):
		cube = np.load(self.cube_path)
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


	def mask_halos(self, mmax, n=1):
		"""Masks all halos with m>mmax
		"""
		full_cube = np.load(self.cube_path)
		halos_over_mmax, _ = self.get_halo_only_cube(mmin=mmax, mmax=n)  
		return full_cube - halos_over_mmax


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
			
 