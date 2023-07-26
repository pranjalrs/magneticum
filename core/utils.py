import glob
import joblib
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


def get_physical_density_in_spherical_shell(mass, r1, r2, z, little_h):
	"""Calculates dark matter density in physical units by
	summing total mass and dividing by the shell volume
	"""
	mass = mass*Gadget.units.mass

	assert r1 < r2, "r1 must be smaller than r2 "
	r1, r2 = r1*Gadget.units.length, r2*Gadget.units.length
	total_volume = 4*np.pi/3 * (r2**3 - r1**3)

	comoving_density = mass/total_volume
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
	Pe = ((4-2*Y)/(8-5*Y)*P_thermal).to(u.keV/u.cm**3)  # In comoving keV/cm^3

	return Pe

def get_comoving_electron_pressure_Mead(mass, Temp, Y, r1, r2):
	assert r1 < r2, "r1 must be smaller than r2 "
	r1, r2 = r1*Gadget.units.length, r2*Gadget.units.length
	shell_volume = 4*np.pi/3 * (r2**3 - r1**3)

	Temp = Temp * Gadget.units.Temperature
	mass = mass * Gadget.units.mass
	mu_e = 2/(2-Y) # Mean mass per electron
	Ne = mass/constants.m_p/mu_e  # No. of electrons
	Pe = Ne*constants.k_B*Temp/shell_volume

	Pe = Pe.to(u.keV/u.cm**3)

	return Pe


def get_physical_electron_pressure_Mead(mass, Temp, Y, r1, r2, z, little_h):
	comoving_Pe = get_comoving_electron_pressure_Mead(mass, Temp, Y, r1, r2)
	physical_Pe = comoving_Pe* Gadget.convert.pressure_to_physical(z, little_h)

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
		
		Pe_physical = get_physical_electron_pressure_Mead(mass, Temp, Y, r1, r2, z, little_h)
		return Pe_physical

	if field == "Temp":
		Temp = particle_data[0]['TEMP'][mask[0]]*Gadget.units.Temperature
		return Temp

	if field=="cdm":
		rho_cdm = get_physical_density_in_spherical_shell(particle_data[1]['MASS'][mask[1]], r1, r2, z, little_h)
		return rho_cdm

	if field=="gas":
		rho_gas = get_physical_density_in_spherical_shell(particle_data[0]['MASS'][mask[0]], r1, r2, z, little_h)
		return rho_gas

	if field=="matter":
		rho_gas = get_physical_density_in_spherical_shell(particle_data[0]['MASS'][mask[0]], r1, r2, z, little_h)
		rho_cdm = get_physical_density_in_spherical_shell(particle_data[1]['MASS'][mask[1]], r1, r2, z, little_h)

		try:
			rho_star = get_physical_density_in_spherical_shell(particle_data[4]['MASS'][mask[4]], r1, r2, z, little_h)
		except:
			rho_star = []
		return np.concatenate((rho_gas, rho_star, rho_cdm))


def get_profile_for_halo(snap_base, halo_center, halo_radius, fields, z, little_h, estimator='median'):
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

	try:
		if 'matter' in fields:
			ptype = [0, 1, 4]

		elif 'cdm' in fields:
			ptype = [1]

		elif 'gas' in fields:
			pytpe = [0]

		else:
			ptype = [0]

		particle_data = g3read.read_particles_in_box(snap_base, halo_center, 3*halo_radius, ['POS ', 'TEMP', 'MASS', 'RHO ', 'Zs  '], ptype, use_super_indexes=True)

	except:
		print(f'Snapshot directory {snap_base} not found!')
		sys.exit(1)

	profiles_dict = {field: [[], [], [], []] for field in fields}

	for field in fields:
		if field in ['Pe_Mead', 'matter', 'gas', 'cdm']:
			profile, r, sigma_prof, sigma_lnprof = _collect_profiles_for_halo(halo_center, halo_radius, particle_data, ptype, field, z, little_h, estimator='sum')

		else:            
			profile, r, sigma_prof, sigma_lnprof = _collect_profiles_for_halo(halo_center, halo_radius, particle_data, ptype, field, z, little_h, estimator)

		profiles_dict[field][0] = profile
		profiles_dict[field][1] = r
		profiles_dict[field][2] = sigma_prof
		profiles_dict[field][3] = sigma_lnprof

	return profiles_dict


def _collect_profiles_for_halo(halo_center, halo_radius, particle_data, ptype, field, z, little_h, estimator):
	"""
	To Do: Update Doctstring
	"""
	rmin, rmax = 0.1*halo_radius, 3*halo_radius
	radial_bins = np.logspace(np.log10(rmin), np.log10(rmax), 31)  # Radial bin edges  (0.1-3)*R500c kpc/h

	## g3read.to_spherical returns an array of [r, theta, phi]
	part_distance_from_center = {}

	for this_ptype in ptype:
		if particle_data[this_ptype]['POS '] is not None:
			part_distance_from_center[this_ptype] = g3read.to_spherical(particle_data[this_ptype]['POS '], halo_center).T[0]
		else:
			#part_distance_from_center[this_ptype] = []
			ptype.remove(this_ptype)
			

	weighted_bin_center = np.ones(len(radial_bins)-1, dtype='float32')
	profile = np.zeros(len(radial_bins)-1, dtype='float32')
	sigma_prof = np.zeros(len(radial_bins)-1, dtype='float32')
	sigma_lnprof = np.zeros(len(radial_bins)-1, dtype='float32')

	## Calcualte field in each radial bin
	for bin_index in range(len(radial_bins)-1):
		r_low, r_up = radial_bins[bin_index], radial_bins[bin_index+1]

		# Construct mask to select particles in bin
		mask = {}
		for this_ptype in ptype:
			mask[this_ptype] = np.where((part_distance_from_center[this_ptype] >= r_low) & (part_distance_from_center[this_ptype] < r_up))[0]

		# Check if we have particles in the bin
		if np.sum([len(mask[this_ptype]) for this_ptype in ptype])!=0:
			this_bin_field = get_field_for_halo(particle_data, mask, z, little_h, field, r_low, r_up).value
			n_part = len(this_bin_field)  # No. of particles

			if estimator == 'median':
				profile[bin_index] = np.median(this_bin_field)
				sigma_prof[bin_index] = sigma_percentile(this_bin_field)/n_part**0.5
				sigma_lnprof[bin_index] = sigma_percentile(np.log(this_bin_field))/n_part**0.5


			elif estimator == 'mean':
				profile[bin_index] = np.mean(this_bin_field)
				sigma_prof[bin_index] = sigma_percentile(this_bin_field)/n_part**0.5

				sigma_lnprof[bin_index] = sigma_percentile(np.log(this_bin_field))/n_part**0.5


			elif estimator == 'sum':
				profile[bin_index] = np.sum(this_bin_field)
				sigma_prof[bin_index] = sigma_percentile(this_bin_field)*n_part**0.5
				sigma_lnprof[bin_index] = sigma_percentile(np.log(this_bin_field))*n_part**0.5

			# Concatenate positions and masses for all ptypes
			all_part_distance_from_center = []
			all_part_mass = []
			for part in ptype:
				all_part_distance_from_center.append(part_distance_from_center[part][mask[part]])
				all_part_mass.append(particle_data[part]['MASS'][mask[part]])

			all_part_distance_from_center = np.concatenate(all_part_distance_from_center)
			all_part_mass = np.concatenate(all_part_mass)

			weighted_bin_center[bin_index] = np.average(all_part_distance_from_center, weights=all_part_mass)


		else:
			profile[bin_index] = 0
			sigma_prof[bin_index] = np.nan
			sigma_lnprof[bin_index] = np.nan
			weighted_bin_center[bin_index] = radial_bins[bin_index]


	return  profile, weighted_bin_center, sigma_prof, sigma_lnprof


def get_halo_catalog(group_base, blocks=['GPOS', 'MVIR', 'RVIR', 'M5CC', 'R5CC']):
	box_name = group_base.split('/')[-3]
	sim_name = group_base.split('/')[-2]
	redshift_id = group_base.split('/')[-1].split('_')[-1]

	file_paths = glob.glob(f'{group_base}/sub_{redshift_id}.*')
	
	print('Saving only halos with Mvir>1e12Msun/h')
	data = {key:[] for key in blocks}
	with tqdm(total=len(file_paths)) as pbar:
		for path in file_paths:
			group_data = g3read.read_new(path, blocks, 0, is_snap=False)
			idx = np.where(group_data['MVIR']>100)[0]
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

	else:
		cosmo_pars = re.findall(r'mr_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)', path)[0]

		Om0, Ob0, sigma8, h = np.array(cosmo_pars, dtype=float)

	this_cosmo = {'flat': True, 'H0': h*100, 'Om0': Om0, 'Ob0': Ob0, 'sigma8': sigma8, 'ns': 0.963}

	return this_cosmo


def _assert_correct_field(fields):
	for f in fields:
		assert f in ['Pe', 'Temp', 'matter', 'cdm', 'gas', 'Pe_Mead'], f'Field *{f}* is unknown!'


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