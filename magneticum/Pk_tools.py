import g3read
import  astropy.units as u
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../src/')

from utils import get_physical_electron_pressure, get_physical_electron_pressure_Mead

# GADGET-2 Units: https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf
vel_units = 1.0  # km/s
len_units = 1.0  # kpc/h
mass_units = 1e10  # Msun/h

units_dict = {'POS ': 1e-3, 'MASS':1e10, 'BHMA':1e10}


def get_field(ptype, snapshot, field_name, z, little_h, cell_volume=None):
	"""_summary_

	Parameters
	----------
	ptype : int
		Particle type
	snapshot : object
		Snapshot file read using g3read
	field_name : str
		Field name

	Returns
	-------
	array
		Array of field values
	"""

	if field_name == 'Pe':
		assert ptype==0, 'Can calculate electron pressure only for gas particles (ptype=0)'
		rho = snapshot.read_new('RHO ', ptypes=0)
		Temp = snapshot.read_new('TEMP', ptypes=0)
		Y = snapshot.read_new('Zs  ', ptypes=0)[:, 0]

		return get_physical_electron_pressure(rho, Temp, Y, z, little_h).value

	elif field_name == 'rho':
		rho = snapshot.read_new('RHO ', ptypes=0)

		return rho

	elif field_name == 'Pe_Mead':
		assert ptype==0, 'Can calculate electron pressure only for gas particles (ptype=0)'
		mass = snapshot.read_new('MASS', ptypes=0)
		Temp = snapshot.read_new('TEMP', ptypes=0)
		Y = snapshot.read_new('Zs  ', ptypes=0)[:, 0]

		return get_physical_electron_pressure_Mead(mass, Temp, Y, cell_volume, z, little_h).value

	else:
		return snapshot.read_new(field_name, ptype)*units_dict[field_name]


def get_field_hist(ptype, field_name, boxsize, resolution, snap_base, nfiles):
	"""Generates histogram of the specified field, follows NN mass assignment

	Parameters
	----------
	ptype : int
		Particle type
	field_name : str
		Field for creating histogram
	boxsize : float
		Box size in Mpc/h
	resolution : int
		Grid resolution
	snap_base : str
		Base file path for snapshots
	nfiles : int
		Number of snapshot files

	Returns
	-------
	ndarray
		3D histogram of the field
	"""
	
	field_hist = 0
	num_part = 0
	for i in range(nfiles):
		this_snap = g3read.GadgetFile(snap_base + '.' + str(i))
		field = get_field(ptype, this_snap, field_name)
		coordinates = this_snap.read_new('POS ', ptype)*units_dict['POS ']

		this_hist, edges = np.histogramdd(coordinates, bins=resolution, range=[[0, boxsize], [0, boxsize], [0, boxsize]], weights=field)

		field_hist += this_hist
		print(i)
	return field_hist

def get_mass_hist(boxsize, resolution, snap_base, nfiles):
	"""Same as `get_field_hist()` but reads info for all particle types at once
	"""
	
	field_hist = 0
	num_part = 0
	for i in range(nfiles):
		data = g3read.read_new(snap_base + '.' + str(i), ['MASS', 'POS '], [0, 1, 4], do_flatten=False)
		data_bh = g3read.read_new(snap_base + '.' + str(i), ['BHMA', 'POS '], [5])
        
		for pt in [0, 1, 4]:
			this_hist, edges = np.histogramdd(data[pt]['POS ']*units_dict['POS '], bins=resolution, range=[[0, boxsize], [0, boxsize], [0, boxsize]], weights=data[pt]['MASS'])

			field_hist += this_hist
            
		this_hist, edges = np.histogramdd(data_bh[5]['POS ']*units_dict['POS '], bins=resolution, range=[[0, boxsize], [0, boxsize], [0, boxsize]], weights=data_bh[5]['BHMA'])
		field_hist += this_hist
		print(i)
	return field_hist


def get_field_hist_test(ptype, field_name, boxsize, resolution, snap_base, nfiles, weight=None):
	"""Generates histogram of the specified field, follows NN mass assignment

	Parameters
	----------
	ptype : int
		Particle type
	field_name : str
		Field for creating histogram
	boxsize : float
		Box size in Mpc/h
	resolution : int
		Grid resolution
	snap_base : str
		Base file path for snapshots
	nfiles : int
		Number of snapshot files

	Returns
	-------
	ndarray
		3D histogram of the field
	"""

	cell_volume = (boxsize*u.Mpc/resolution)**3    
	field_hist = 0#np.zeros((resolution, resolution, resolution))
	weight_hist = 0
	num_part = 0
    
	print(f'Field: {field_name} to be weighted by {weight}')

	for i in range(nfiles):
		this_snap = g3read.GadgetFile(snap_base + '.' + str(i))
		field = get_field(ptype, this_snap, field_name, cell_volume)
		coordinates = this_snap.read_new('POS ', ptype)*units_dict['POS ']

		if weight is None or weight=='mean':
			this_hist, edges = np.histogramdd(coordinates, bins=resolution, range=[[0, boxsize], [0, boxsize], [0, boxsize]], weights=field)

		if weight == 'volume':
			m_over_rho = np.array(this_snap.read_new('MASS', ptype))/np.array(this_snap.read_new('RHO ', ptype))
			this_hist, edges = np.histogramdd(coordinates, bins=resolution, range=[[0, boxsize], [0, boxsize], [0, boxsize]], weights=field*m_over_rho)
			this_vol_hist, edges = np.histogramdd(coordinates, bins=resolution, range=[[0, boxsize], [0, boxsize], [0, boxsize]], weights=m_over_rho)
			weight_hist += this_vol_hist

		if weight == 'hsml':
			hsml = this_snap.read_new('HSML', ptype)
			this_hist, edges = np.histogramdd(coordinates, bins=resolution, range=[[0, boxsize], [0, boxsize], [0, boxsize]], weights=field*hsml**3)

			this_hsml_hist, edges = np.histogramdd(coordinates, bins=resolution, range=[[0, boxsize], [0, boxsize], [0, boxsize]], weights=hsml**3)
			weight_hist += this_hsml_hist

		this_num_part, _ = np.histogramdd(coordinates, bins=resolution, range=[[0, boxsize], [0, boxsize], [0, boxsize]])

		field_hist += this_hist
		num_part += this_num_part
		print(i)

	return field_hist, weight_hist, num_part

def calc_amp_FFT(delta, resolution):
	"""Computes the Fourier transform of the field delta

	Parameters
	----------
	delta : array
		Field overdensity in real space
	resolution : int
		Frequency grid resolution

	Returns
	-------
	array
		Fourier transform of the field overdensity
	"""
	FFT = np.fft.fftn(delta)
	Amp_FFT = np.absolute(FFT)/resolution**3

	return Amp_FFT


def calc_freq_FFT(boxsize, resolution):
	"""Computes frequency grid and window function for a given boxsize and resolution

	Parameters
	----------
	boxsize : float
		Box size (in Mpc/h)
	resolution : int
		Resolution of frequency grid

	Returns
	-------
	array, array
		Frequency grid, window function
	"""	

	spatial_bin_size = boxsize/np.double(resolution)  # Bin size
	freq = np.fft.fftfreq(resolution, d=spatial_bin_size).astype('float32')[np.mgrid[0:resolution, 0:resolution, 0:resolution]]  # 3D freq grid

	Wk = np.sinc(freq[0]*spatial_bin_size)*np.sinc(freq[1]*spatial_bin_size)*np.sinc(freq[2]*spatial_bin_size)
	k = 2*np.pi*(freq[0]**2 + freq[1]**2 + freq[2]**2)**0.5
	
	return k, Wk


def calc_Pk(delta_k_hat, Wk, Vbox, k, kbins, CIC=False):
	"""Computes the auto-power spectrum of a field

	To Do: add functionality for computing cross power spectrum

	Parameters
	----------
	delta_k_hat : array
		Fourier transform of the field overdensity
	Wk : array
		Window function
	Vbox : float
		Box volume (in (Mpc/h)^3)
	k : float
		3D Frequency grid
	kbins : array
		k-bins for computing power spectrum

	Returns
	-------
	array, array, array
		Power spectrum, average k in each bin, number of modes in each bin
	"""	

	Nbins = len(kbins) - 1
	avgPk = np.empty(Nbins, dtype=float)
	avgk = np.empty(Nbins, dtype=float)
	Nk = np.empty(Nbins, dtype=float)
	
	for j in range(Nbins):
		takeout_ID = np.where((k>kbins[j]) & (k<=kbins[j+1]))
		k_now = k[takeout_ID]
		
		delta_k_now = delta_k_hat[takeout_ID]/Wk[takeout_ID]

		if CIC is True:
			delta_k_now = delta_k_hat[takeout_ID]
		delta_k2_now = delta_k_now*delta_k_now

		avgPk[j] = np.mean(delta_k2_now)*Vbox
		avgk[j] = np.mean(k_now)
		Nk[j] = len(k_now)
	
	return avgPk, avgk, Nk

def calc_Pk_cross(delta_k_hat1, delta_k_hat2, Wk, Vbox, k, kbins):
	"""Computes the auto-power spectrum of a field

	To Do: add functionality for computing cross power spectrum

	Parameters
	----------
	delta_k_hat : array
		Fourier transform of the field overdensity
	Wk : array
		Window function
	Vbox : float
		Box volume (in (Mpc/h)^3)
	k : float
		3D Frequency grid
	kbins : array
		k-bins for computing power spectrum

	Returns
	-------
	array, array, array
		Power spectrum, average k in each bin, number of modes in each bin
	"""	

	Nbins = len(kbins) - 1
	avgPk = np.empty(Nbins, dtype=float)
	avgk = np.empty(Nbins, dtype=float)
	Nk = np.empty(Nbins, dtype=float)

	for j in range(Nbins):
		takeout_ID = np.where((k>kbins[j]) & (k<=kbins[j+1]))
		k_now = k[takeout_ID]
		delta_k1_now = delta_k_hat1[takeout_ID]/Wk[takeout_ID]
		delta_k2_now = delta_k_hat2[takeout_ID]/Wk[takeout_ID]
		delta_k12_now = delta_k1_now*delta_k2_now

		avgPk[j] = np.mean(delta_k12_now)*Vbox
		avgk[j] = np.mean(k_now)
		Nk[j] = len(k_now)
	
	return avgPk, avgk, Nk
