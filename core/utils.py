import glob
import joblib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
from tqdm import tqdm


import astropy.units as u
import astropy.constants as constants
import g3read

import sys
sys.path.append('../core/')

from gadget import Gadget

import ipdb

def get_mean_mass_per_particle(Y):
	"""Calculates mean mass per particle assuming a completely ionized gas (H + He)

	Parameters
	----------
	Y : float
		Helium mass fraction

	Returns
	-------
	float
		Mean mass per particle
	"""	

	return 1/(2*(1-Y) + 3/4*Y)


def get_physical_density_in_spherical_shell(mass, volume, z, little_h):
	"""Calculates dark matter density in physical units by
	summing total mass and dividing by the shell volume
	"""
	mass = mass*Gadget.units.mass

	comoving_density = mass/volume
	physical_density = comoving_density*Gadget.convert.density_to_physical(z, little_h)
	return physical_density.to(u.GeV/u.cm**3, u.mass_energy())


def get_comoving_electron_pressure(rho, Temp, Y):
	"""Calculates electron pressure in comoving units

	P_gas = rho_gas/mu/m_H * k_B * T
	P_e = P_gas * (4-2Y)/(8-5Y)

	Parameters
	----------
	rho : float
		Gas mass density (in GADGET units)
	Temp : float
		Gas temperature (in GADGET units)
	Y : float
		Helium mass fraction

	Returns
	-------
	float
		Comoving electron pressure (in keV/cm^3 h^2)
	"""

	rho = rho * Gadget.units.mass_density
	Temp = Temp * Gadget.units.Temperature

	mu = get_mean_mass_per_particle(Y)
	ngas = rho/(mu*constants.m_p)
	P_thermal = ngas*constants.k_B*Temp
	Pe = ((4-2*Y)/(8-5*Y)*P_thermal).to(u.keV/u.cm**3)  # In comoving keV/cm^3 h^2

	return Pe

def get_comoving_electron_pressure_Mead(mass, Temp, Y, volume):


	Temp = Temp * Gadget.units.Temperature
	mass = mass * Gadget.units.mass
	mu_e = 2/(2-Y) # Mean mass per electron
	Ne = mass/constants.m_p/mu_e  # No. of electrons
	Pe = Ne*constants.k_B*Temp/volume

	Pe = Pe.to(u.keV/u.cm**3)

	return Pe


def get_physical_electron_pressure_Mead(mass, Temp, Y, volume, z, little_h):
	comoving_Pe = get_comoving_electron_pressure_Mead(mass, Temp, Y, volume)
	physical_Pe = comoving_Pe*Gadget.convert.pressure_to_physical(z, little_h)

	return physical_Pe


def get_physical_electron_pressure(rho, Temp, Y, z, little_h):
	"""Calls get_comoving_electron_pressure(rho, Temp, Y) and converts to physical units

	Parameters
	----------
	rho : float
		Gas mass density (in GADGET units)
	Temp : float
		Gas temperature (in GADGET units)
	Y : float
		Helium mass fraction
	z : float
		Redshift
	little_h : float
		H0/(100 km/s/Mpc)

	Returns
	-------
	float
		Physical electron pressure (in keV/cm^3)
	"""
	 
	comoving_Pe = get_comoving_electron_pressure(rho, Temp, Y)
	physical_Pe = comoving_Pe * Gadget.convert.pressure_to_physical(z, little_h)

	return physical_Pe


def get_field_for_halo(particle_data, mask, z, little_h, field, r1=None, r2=None, **kwargs):
	"""Gets specified field for particles in a halo, in physical units

	Parameters
	----------
	gas_data : Dict
		Dictionary of blocks necessary for calcualting field
	mask : 2D array
		2D array to mask particles not in halo/radial bin
	z : float
		Redshift
	little_h : float
		H0/(100 km/s/Mpc)
	field : str
		Field name: Electron Pressure (Pe), Temperature (Temp)

	Returns
	-------
	list
		List of field values
	"""
	if r1 is not None and r2 is not None:
		assert r1 < r2, "r1 must be smaller than r2 "
		r1, r2 = r1*Gadget.units.length, r2*Gadget.units.length
		shell_volume = 4*np.pi/3 * (r2**3 - r1**3)  # in (kpc/h)**3

	if field == 'Pe':
		rho = particle_data[0]['RHO '][mask[0]]
		Temp = particle_data[0]['TEMP'][mask[0]]
		Y = particle_data[0]['Zs  '][mask[0]][:, 0]  # Helium Fraction
		Pe = get_physical_electron_pressure(rho, Temp, Y, z, little_h)
		return Pe
	
	if field == 'Pe_Mead':
		
		mass = particle_data[0]['MASS'][mask[0]]
		Temp = particle_data[0]['TEMP'][mask[0]]
		Y = particle_data[0]['Zs  '][mask[0]][:, 0]  # Helium Fraction
		
		Pe_physical = get_physical_electron_pressure_Mead(mass, Temp, Y, shell_volume, z, little_h)
		return Pe_physical

	if field == 'v_disp':
		mass = particle_data[0]['MASS'][mask[0]]
		velocity = particle_data[0]['VEL '][mask[0]]*Gadget.units.velocity  # v_comoving / sqrt(1+z)

		velocity_comoving = (velocity/(1+z)**0.5).to(u.km/u.s).value
		v_dispersion = 1/3*mass*(velocity[:, 0]**2 + velocity[:, 1]**2 + velocity[:, 2]**2)/np.sum(mass)

		return v_dispersion

	if field == "Temp":
		Temp = particle_data[0]['TEMP'][mask[0]]*Gadget.units.Temperature
		return Temp

	if field=="cdm":
		rho_cdm = get_physical_density_in_spherical_shell(particle_data[1]['MASS'][mask[1]], shell_volume, z, little_h)
		return rho_cdm

	if field=="gas":
		rho_gas = get_physical_density_in_spherical_shell(particle_data[0]['MASS'][mask[0]], shell_volume, z, little_h)
		return rho_gas

	if field=="matter":
		rho_gas = get_physical_density_in_spherical_shell(particle_data[0]['MASS'][mask[0]], shell_volume, z, little_h)
		rho_cdm = get_physical_density_in_spherical_shell(particle_data[1]['MASS'][mask[1]], shell_volume, z, little_h)

		try:
			rho_star = get_physical_density_in_spherical_shell(particle_data[4]['MASS'][mask[4]], shell_volume, z, little_h)
		except:
			rho_star = []
		return np.concatenate((rho_gas, rho_star, rho_cdm))


def get_profile_for_halo(snap_base, halo_center, halo_radius, fields, recal_cent=False, save_proj=False, filename='', estimator='median'):
	"""Gets field profile for a given halo in physical units
	To Do: Update Docstring

	Parameters
	----------
	snap_base : str
		Base file path for snapshots
	halo_center : list
		3D cartesian coordinates of halo center (in GADGET units)
	halo_radius : float
		Halo radius used for gathering particles (in GADGET units)
		radial bins are from log10(0.1*halo_radius)-log10(3*halo_radius)
	z : float
		Redshift
	little_h : float
		H0/(100 km/s/Mpc)
	field : str
		Field name: Electron Pressure (Pe), Temperature (Temp)

	Returns
	-------
	list, list	
		Field value in each radial bin, radian bin
	"""
	if not isinstance(fields, list): fields = [fields]
	_assert_correct_field(fields)

	ptype = [0, 1]
	if 'matter' in fields:
		ptype += [4]

	try:
		particle_data = g3read.read_particles_in_box(snap_base, halo_center, halo_radius, ['POT ', 'POS ', 'TEMP', 'MASS', 'VEL ', 'RHO ', 'Zs  '], ptype, use_super_indexes=True)

	except FileNotFoundError:
		print(f'Snapshot directory {snap_base} not found!')
		sys.exit(1)
	
	## Check if the particle at potential min. is close to halo center	
	pot_min_idx = np.argmin( particle_data[1]['POT '])
	GPOS = particle_data[1]['POS '][pot_min_idx]
	
	if not np.all(np.isclose(halo_center, GPOS, atol=1.5)):
		print('Warning: Halo might be mis-centerd') 
		print('Delta X={:.2f}, Delta Y={:.2f}, Delta Z={:.2f}'.format(*(halo_center-GPOS)))
		if recal_cent is True:
			print('Recentering...')
			halo_center = GPOS
		else:
			print('Set recal_cent=True to compute a new halo center based on the position of DM particle with min. potential')


	profiles_dict = {field: [[], [], []] for field in fields}

	if save_proj is True:
		fig, ax = plt.subplots(len(fields), 3, figsize=(14, 4*len(fields)), gridspec_kw={'width_ratios': [1, 1, 1.05]})
		if len(fields) == 1: ax = [ax]

	else:
		ax = [None]*len(fields)
	for i, field in enumerate(fields):
		if field in ['Pe_Mead', 'matter', 'gas', 'cdm', 'v_disp']:
			profile, r, npart = _collect_profiles_for_halo(halo_center, halo_radius, particle_data, ptype, field, ax[i])

		else:
			profile, r, npart = _collect_profiles_for_halo(halo_center, halo_radius, particle_data, ptype, field, z, little_h, estimator)

		profiles_dict[field][0] = profile
		profiles_dict[field][1] = r
		profiles_dict[field][2] = npart

	if save_proj is True:
		plt.savefig(f'{filename}.pdf', bbox_inches='tight', dpi=100)
		plt.close()
	return profiles_dict



def _collect_profiles_for_halo(halo_center, halo_radius, particle_data, ptype, field_type, ax):
	rmin, rmax = 0.01*halo_radius, 1*halo_radius
	bins = np.logspace(np.log10(rmin), np.log10(rmax), 20)  # Radial bin edges

	## g3read.to_spherical returns an array of [r, theta, phi]
	# Compute particle pos w.r.t. halo center (as a fraction of Rvir)
	particle_pos = {}
	mask = {}
	for this_ptype in ptype:
		if particle_data[this_ptype]['POS '] is not None:
			distance = g3read.to_spherical(particle_data[this_ptype]['POS '], halo_center).T[0]
			particle_pos[this_ptype] = distance

			## Create mask for particle outside rmax	
			mask[this_ptype] = distance < rmax
		else:
			ptype.remove(this_ptype)


	# To assign e.g, pressure to each particle we also need its the volume of the shell it is in
	# This is only required for make 2D maps
	field, binned_field, bin_centers, npart = _get_field_for_halo(particle_pos, particle_data, field_type, bins, mask)
	
	if ax is None:
		return binned_field, bin_centers, npart
	
	## Hack: need to fix later
	if field_type == 'cdm':
		this_ptype = 1
		label = 'DM Density'	

	elif field_type in ['Pe_Mead', 'gas', 'Temp']:
		this_ptype = 0
		label = 'Electron Pressure'	
	
	x = particle_data[this_ptype]['POS '][:, 0][mask[this_ptype]]/1e3 - halo_center[0]/1e3
	y = particle_data[this_ptype]['POS '][:, 1][mask[this_ptype]]/1e3 - halo_center[1]/1e3
	z = particle_data[this_ptype]['POS '][:, 2][mask[this_ptype]]/1e3 - halo_center[2]/1e3


	## x-y projection
	ax[0].hist2d(x, y, weights=field, bins=100,  norm = colors.LogNorm(), rasterized=True)
	ax[0].set(xlabel='$\\Delta$X [cMpc/h]', ylabel='$\\Delta$Y [cMpc/h]')

	## x-z projection
	ax[1].hist2d(x, z, weights=field, bins=100,  norm = colors.LogNorm(), rasterized=True)
	ax[1].set(xlabel='$\\Delta$X [cMpc/h]', ylabel='$\\Delta$Z [cMpc/h]')
	ax[1].set_title(label)

	## y-z projection
	im = ax[2].hist2d(y, z, weights=field, bins=100,  norm = colors.LogNorm(), rasterized=True)
	ax[2].set(xlabel='$\\Delta$Y [cMpc/h]', ylabel='$\\Delta$Z [cMpc/h]')
	colorbar(im[3], ax=ax[2])	

	for i in range(3):
		circle = plt.Circle((0, 0), halo_radius/1e3, ls='--', color='orangered', fill=False)
		ax[i].add_patch(circle)
		ax[i].scatter(0., 0., marker='x', c='orangered')
		ax[i].set_xlim(-2*rmax/1e3, 2*rmax/1e3)
		ax[i].set_ylim(-2*rmax/1e3, 2*rmax/1e3)
		ax[i].set_aspect('equal')
	
	return binned_field, bin_centers, npart


def _get_field_for_halo(particle_pos, particle_data, field_type, bins, mask):

	if field_type == 'cdm':
		# First mask out al particles outside region of interest
		ptype = 1 # For DM
		these_pos = particle_pos[ptype][mask[ptype]]
		mass = particle_data[ptype]['MASS'][mask[ptype]]*Gadget.units.mass

		bin_centers, bins_shell, part_per_bin = _build_hist_bins(these_pos, bins)
		particle_volume = bins_shell[np.digitize(these_pos, bins)-1]*Gadget.units.length**3

		# Now compute `field`
		density = mass/particle_volume  # Per particle and in code units
		binned_density = np.histogram(these_pos, weights=density, bins=bins, density=False)[0]
		return density, binned_density, bin_centers, part_per_bin


	if field_type == 'gas':
		ptype = 0 # For gas
		these_pos = particle_pos[ptype][mask[ptype]]
		mass = particle_data[ptype]['MASS'][mask[ptype]]*Gadget.units.mass

		bin_centers, bins_shell, part_per_bin = _build_hist_bins(these_pos, bins)
		particle_volume = bins_shell[np.digitize(these_pos, bins)-1]*Gadget.units.length**3

		# Now compute `field`
		density = mass/particle_volume  # Per particle and in code units
		binned_density = np.histogram(these_pos, weights=density, bins=bins, density=False)[0]
		return density, binned_density, bin_centers, part_per_bin


	if field_type == 'Pe_Mead':
		ptype = 0  # For gas
		these_pos = particle_pos[ptype][mask[ptype]]
		mass = particle_data[ptype]['MASS'][mask[ptype]]
		Temp = particle_data[ptype]['TEMP'][mask[ptype]]
		Y = particle_data[ptype]['Zs  '][mask[ptype]][:, 0]  # Helium Fraction

		bin_centers, bins_shell, part_per_bin = _build_hist_bins(these_pos, bins)
		particle_volume = bins_shell[np.digitize(these_pos, bins)-1]*Gadget.units.length**3

		Pe = get_comoving_electron_pressure_Mead(mass, Temp, Y, particle_volume)
		binned_Pe = np.histogram(these_pos, weights=Pe.value, bins=bins, density=False)[0]
		return Pe.value, binned_Pe*Pe.unit, bin_centers, part_per_bin


def _build_hist_bins(pos, bins):
	"""
	Build histogram for a given set of particle positions and bins. 

	Parameters:
	pos (array-like): Array of particle positions.
	bins (int or array-like): Number of bins or bin edges.

	Returns:
	bin_centers (ndarray): Array of bin centers.
	bins_shell (ndarray): Array of shell volumes.
	part_per_bin (ndarray): Array of particle counts per bin.
	"""
	part_per_bin = np.histogram(pos, bins=bins, density=False)[0]
	bin_pos_sum = np.histogram(pos, weights=pos, bins=bins, density=False)[0]
	bin_centers = bin_pos_sum/part_per_bin  # Average bin center weighted by number of particles
	bins_shell = 4./3.*np.pi*(bins[1:]**3 - bins[:-1]**3) 
	

	return bin_centers, bins_shell, part_per_bin

def get_halo_catalog(group_base, blocks=['GPOS', 'MVIR', 'RVIR', 'M5CC', 'R5CC'], mmin=1e12):
	box_name = group_base.split('/')[-3]
	sim_name = group_base.split('/')[-2]
	redshift_id = group_base.split('/')[-1].split('_')[-1]

	file_paths = glob.glob(f'{group_base}/sub_{redshift_id}.*')
	
	print(f'Saving only halos with Mvir>{mmin:.1E} Msun/h')
	data = {key:[] for key in blocks}
	with tqdm(total=len(file_paths)) as pbar:
		for path in file_paths:
			group_data = g3read.read_new(path, blocks, 0, is_snap=False)
			idx = np.where(group_data['MVIR']>mmin/1e10)[0]
			for item in blocks:
				data[item].append(group_data[item][idx])
			pbar.update(1)

	data = {key:np.concatenate(data[key]) for key in blocks}

	joblib.dump(data, f'../../magneticum-data/data/halo_catalog/{box_name}/{sim_name}_sub_{redshift_id}.pkl', compress='lzma')


def get_hmf_from_halo_catalog(halo_catalog=None, mass=None, mr=1.3e10, return_dndlog10m=False, boxsize=None):
	'''
	Compute HMF from halo catalog for a given snapshot
	To Do: Add details to docstring
	'''
	if halo_catalog is not None:    
		halo_mass = (halo_catalog['MVIR']*Gadget.units.mass).value

	else:
		halo_mass = mass

	mass_min, mass_max = np.log10(55*mr), np.log10(max(halo_mass))  # In Msun/h
	bins_per_dex = 10  # Bins per dex

	halo_mass_cut = halo_mass[halo_mass>mr]

	mf, bin_edges = np.histogram(np.log10(halo_mass_cut), bins=int(bins_per_dex*(mass_max-mass_min)),
							 range=(mass_min, mass_max))

	center = np.array([(bin_edges[j]+bin_edges[j+1])/2 for j in range(len(mf))])

	if return_dndlog10m is True:
		assert boxsize is not None, 'Need boxsize (in Mpc/h) to compute dn/dm'
		Vbox = boxsize**3
		bin_widths = np.array([(10**center[i+1]-10**center[i]) for i in range(len(center)-1)])
		dndm = mf[:-1]/Vbox*10**center[:-1]/bin_widths*np.log(10)  # Count/Volume * M/delta_M
		return dndm, 10**center[:-1]

	return mf, bin_edges


def get_total_Pe_halo(snap_base, halo_center, halo_radius, z, little_h):
	ptype = 0  # For gas
	mask = []
	particle_data = []
	particle_data.append(g3read.read_particles_in_box(snap_base, halo_center, 2*halo_radius, ['POS ', 'TEMP', 'MASS', 'RHO ', 'Zs  '], ptype, use_super_indexes=True))
	part_distance_from_center = g3read.to_spherical(particle_data[0]['POS '], halo_center).T[0]
	mask.append(np.where(part_distance_from_center< halo_radius)[0])

	Pe = get_field_for_halo(particle_data, mask, z, little_h, 'Pe')

	return np.sum(Pe.value)


def get_fgas_halo(snap_base, halo_center, radius, z, little_h):
	ptype = 0, 1, 4, 5  # For gas, DM
	mask = []
	particle_data = []
	particle_data = g3read.read_particles_in_box(snap_base, halo_center, 2*radius, ['POS ', 'TEMP', 'MASS', 'BHMA', 'RHO ', 'Zs  '], ptype, use_super_indexes=True)

	
	# gas mask
	part_distance_from_center = g3read.to_spherical(particle_data[0]['POS '], halo_center).T[0]
	mask_gas = np.where(part_distance_from_center< radius)[0]

	# DM mask
	part_distance_from_center = g3read.to_spherical(particle_data[1]['POS '], halo_center).T[0]
	mask_dm = np.where(part_distance_from_center< radius)[0]

	# stellar mask
	if particle_data[4]['POS '] is not None:
		part_distance_from_center = g3read.to_spherical(particle_data[4]['POS '], halo_center).T[0]
		mask_stars = np.where(part_distance_from_center < radius)[0]
		m_stars = np.sum(particle_data[4]['MASS'][mask_stars])
	else: m_stars = 0.

	# BH mask
	if particle_data[5]['POS '] is not None:
		part_distance_from_center = g3read.to_spherical(particle_data[5]['POS '], halo_center).T[0]
		mask_bh = np.where(part_distance_from_center < radius)[0]
		m_bh = np.sum(particle_data[5]['BHMA'][mask_bh])
	else: m_bh = 0.

	m_gas = np.sum(particle_data[0]['MASS'][mask_gas])
	m_dm = np.sum(particle_data[1]['MASS'][mask_dm])

	m_total = m_gas + m_dm + m_stars + m_bh

	return m_gas/m_total


def get_cosmology_dict_from_path(path):
	if 'mr_bao' in path or 'hr_bao' in path or 'mr_dm' in path:
		Om0, Ob0, sigma8, h = 0.272, 0.0456, 0.809, 0.704

	elif 'hr' in path:
		cosmo_pars = re.findall(r'hr_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)', path)[0]
		Om0, Ob0, sigma8, h = np.array(cosmo_pars, dtype=float)

	else:
		cosmo_pars = re.findall(r'mr_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)', path)[0]

		Om0, Ob0, sigma8, h = np.array(cosmo_pars, dtype=float)

	this_cosmo = {'flat': True, 'H0': h*100, 'Om0': Om0, 'Ob0': Ob0, 'sigma8': sigma8, 'ns': 0.963}

	return this_cosmo


def _assert_correct_field(fields):
	for f in fields:
		assert f in ['Pe', 'Temp', 'matter', 'cdm', 'gas', 'Pe_Mead', 'v_disp'], f'Field *{f}* is unknown!'


def set_storage_path():
	'''Sets the path to where the simulation data is stored
	'''
	global storage_path
	current_directory = os.getcwd()

	if 'pranjalrs' in current_directory:
		storage_path = f'/xdisk/timeifler/pranjalrs/magneticum_data/'

	if 'di75sic' in current_directory:
		storage_path = f'/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/'

##### Functions mostly used in Jupyter notebook demo #####

def get_omega_m(filename):
	# extract the omega_b value from the filename
	cosmo = get_cosmology_dict_from_path(filename)
	omega_m = cosmo['Om0']
	return float(omega_m)

def get_omega_b(filename):
	# extract the omega_b value from the filename
	cosmo = get_cosmology_dict_from_path(filename)
	omega_b = cosmo['Ob0']
	return float(omega_b)

def get_sigma8(filename):
	# extract the omega_b value from the filename
	cosmo = get_cosmology_dict_from_path(filename)
	sigma8 = cosmo['sigma8']
	return float(sigma8)

def get_h(filename):
	# extract the omega_b value from the filename
	cosmo = get_cosmology_dict_from_path(filename)
	H0 = cosmo['H0']
	return float(H0)

def get_fb(filename):
	omega_b = get_omega_b(filename)
	omega_m = get_omega_m(filename)
	
	return omega_b/omega_m

def sigma_percentile(arr):
	return (np.percentile(arr, 84) - np.percentile(arr, 16))/2

def colorbar(mappable, ax):
    last_axes = plt.gca()
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar
