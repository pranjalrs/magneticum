import numpy as np
import scipy
from scipy.special import sici

import camb
import pyhalomodel

import ipdb

def get_CLASS_Pk(k_sim, input_dict=None, binned=False, z=0.0):
	'''Returns Pk for WMAP7 cosmology, optionally a dictionary of cosmo
	parameters can also be passed
	'''
	from classy import Class

	zmax = 0.1

	if z>0.1:
		zmax = z + 0.2

	params = {
		'output': 'mPk', #which quantities we want CLASS to compute
		'H0':0.704*100, #cosmology paramater
		'z_max_pk' : str(zmax),
		'non linear' : 'halofit',#option for computing non-linear power spectum
		'P_k_max_1/Mpc':30.0,
		'sigma8': 0.809, #cosmology paramater
		'n_s': 0.963 , #cosmology paramater
		'Omega_b': 0.0456, #cosmology paramater
		'Omega_cdm': 0.272-0.0456 #cosmology paramater
	}
	
	if input_dict is not None:
		params['Omega_cdm'] = input_dict['Om0']-input_dict['Ob0']
		params['Omega_b'] = input_dict['Ob0']
		params['sigma8'] = input_dict['sigma8']
		params['H0'] = input_dict['H0']

	#initialize Class instance and set parameters
	cosmo = Class()
	cosmo.set(params)
	cosmo.compute()
	#now create second Class instance with parameters changed to linear power spectrum model
	params['non linear'] = 'none'
	cosmo_lin = Class()
	cosmo_lin.set(params)
	cosmo_lin.compute()
	# get P(k) at redhsift z=0
	Nk = 2000
	Plin = np.zeros(Nk)
	Pnl = np.zeros(Nk)
	h = cosmo.h() # get reduced Hubble for conversions to 1/Mpc
	kvec = np.exp(np.linspace(np.log(1.e-3),np.log(10.),Nk))
	for k in range(Nk):
		Pnl[k] = cosmo.pk(kvec[k]*h, z)*h**3 #using method .pk(k,z), convert to Mpc/h units
		Plin[k] = cosmo_lin.pk(kvec[k]*h, z)*h**3

	if binned is True:
		# Binning
		bin_edges = np.concatenate((k_sim, [k_sim[-1]+(k_sim[-1]-k_sim[-2])]))
		bins = np.digitize(kvec, bin_edges)-1
		P_theory_binned = np.zeros_like(k_sim)
		for i in range(len(k_sim)):
			P_theory_binned[i] = np.mean(Pnl[bins==i])

		return P_theory_binned, k_sim
		
		return Pk, k

	return Pnl, kvec

def build_CAMB_cosmology(input_cosmo={}, zs=[0.]):
	'''
	Builds a cosmology using the CAMB library.

	Parameters:
		input_cosmo (dict): Dictionary containing input cosmological parameters.
							If None, default cosmological parameters are used.
							Default is None.
		zs (list): List of redshifts at which to calculate the linear matter power spectrum.
				   Default is [0.].

	Returns:
		results: The results object obtained from running CAMB with the specified cosmological parameters.
	'''
	# default cosmology is WMAP7
	Omega_b = input_cosmo.get('Ob0', 0.0456)
	Omega_c = input_cosmo.get('Om0', 0.272) - Omega_b
	Omega_k = 0.0
	h = input_cosmo.get('H0', 70.4)/100
	ns = 0.963
	sigma_8 = input_cosmo.get('sigma8', 0.809)
	w0 = -1.
	wa = 0.
	m_nu = 0.
	set_sigma8 = True
	As = 2e-9

	# CAMB
	kmax_CAMB = 200.

	# Sets cosmological parameters in camb to calculate the linear power spectrum
	pars = camb.CAMBparams(WantTransfer=True, WantCls=False, Want_CMB_lensing=False, 
						DoLensing=False, NonLinearModel=camb.nonlinear.Halofit(halofit_version='mead2020'))
	wb, wc = Omega_b*h**2, Omega_c*h**2

	# This function sets standard and helium set using BBN consistency
	pars.set_cosmology(ombh2=wb, omch2=wc, H0=100.*h, mnu=m_nu, omk=Omega_k)
	pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')
	pars.InitPower.set_params(As=As, ns=ns, r=0.)
	pars.set_matter_power(redshifts=zs, kmax=kmax_CAMB) # Setup the linear matter power spectrum
	Omega_m = pars.omegam # Extract the matter density

	# Scale 'As' to be correct for the desired 'sigma_8' value if necessary
	if set_sigma8:
		results = camb.get_results(pars)
		sigma_8_init = results.get_sigma8_0()
		print('Running CAMB')
		print('Initial sigma_8:', sigma_8_init)
		print('Desired sigma_8:', sigma_8)
		scaling = (sigma_8/sigma_8_init)**2
		As *= scaling
		pars.InitPower.set_params(As=As, ns=ns, r=0.)

	sigma_8 = results.get_sigma8_0()
	print('Final sigma_8:', sigma_8)

	# Run
	results = camb.get_results(pars)

	return results

def get_pyHMCode_Pk(input_cosmo={}, z=[0.], fields=None, return_halo_terms=False, **kwargs):
	'''
	Calculate the nonlinear matter power spectrum using the HMcode.

	Parameters:
	- camb_pars (optional): CAMB parameters used to compute the linear matter power spectrum.
	- z (list): Redshifts at which to compute the power spectrum.
	- fields (optional): List of fields to include in the calculation.
	- return_halo_terms (bool): Whether to return the halo terms in addition to the total power spectrum.
	- **kwargs: Additional keyword arguments to customize the HMcode calculation.

	Returns:
	- Pk_hm (array): The total nonlinear matter power spectrum.
	- ks (array): The wavenumbers corresponding to the power spectrum.

	If return_halo_terms is True, the function also returns:
	- Pk_hm_1halo (array): The 1-halo term of the power spectrum.
	- Pk_hm_2halo (array): The 2-halo term of the power spectrum.
	'''
	import pyhmcode  # This the Python wrapper for the Fortran HMCode

	# Compute CAMB results
	r = camb.get_results(input_cosmo, z)
	ks, zs, Pk_lin = r.get_linear_matter_power_spectrum(nonlinear=False)

	# Need these Omegas
	# Note that the calculation of Omega_v is peculiar, but ensures flatness in HMcode
	# (very) small differences between CAMB and HMcode otherwise
	omv = r.omega_de+r.get_Omega("photon")+r.get_Omega("neutrino") 
	omm = r.Params.omegam

	# Setup HMcode internal cosmology
	c = pyhmcode.Cosmology()

	# Set HMcode internal cosmological parameters
	c.om_m = omm
	c.om_b = r.Params.omegab
	c.om_v = omv
	c.h = r.Params.h
	c.ns = r.InitPower.ns
	c.sig8 = r.get_sigma8_0()

	if r.num_nu_massive == 0.0:
		c.m_nu = 0.0
	else:
		raise NotImplementedError('Setting massive neutrinos is not implemented yet')

	# Set the linear power spectrum for HMcode
	c.set_linear_power_spectrum(ks, zs, Pk_lin)

	mode = pyhmcode.HMx2020_matter_pressure_w_temp_scaling
	hmod = pyhmcode.Halomodel(mode, verbose=False)

	for key in kwargs:
		hmod.__setattr__(key, kwargs[key])

	if fields is None:
		fields = [pyhmcode.field_electron_pressure]

	if return_halo_terms is False:
		Pk_hm = pyhmcode.calculate_nonlinear_power_spectrum(c, hmod, fields, verbose=False, return_halo_terms=False)
		return Pk_hm, ks

	elif return_halo_terms is True:
		Pk_hm, Pk_hm_1halo, Pk_hm_2halo = pyhmcode.calculate_nonlinear_power_spectrum(c, hmod, fields, verbose=False, return_halo_terms=True)
		return Pk_hm, Pk_hm_1halo, Pk_hm_2halo, ks


def get_suppresion_hmcode(input_cosmo={}, zs=[0.], T_AGNs=None):
	'''Matter power supression based on hmcode-the full Python
	implementation of the Fortran HMCode.
	'''
	import hmcode  # This is the full Python implementation of the Fortran HMCode

	zs = np.array(zs)

	# Wavenumbers [h/Mpc]
	kmin, kmax = 1e-3, 3e1
	nk = 128
	k = np.logspace(np.log10(kmin), np.log10(kmax), nk)

	# AGN-feedback temperature [K]
	if T_AGNs is None:
		T_AGNs = np.power(10, np.array([7.6, 7.8, 8.0, 8.3]))

	# Redshifts

	camb_results  = build_CAMB_cosmology(input_cosmo=input_cosmo, zs=zs)

	Pk_lin_interp = camb_results.get_matter_power_interpolator(nonlinear=False).P
	Pk_nonlin_interp = camb_results.get_matter_power_interpolator(nonlinear=True).P

	# Arrays for CAMB non-linear spectrum
	Pk_CAMB = np.zeros((len(zs), len(k)))
	for iz, z in enumerate(zs):
		Pk_CAMB[iz, :] = Pk_nonlin_interp(z, k)

	Rk_feedback = []
	for T_AGN in T_AGNs:
		Pk_feedback = hmcode.power(k, zs, camb_results, T_AGN=T_AGN, verbose=False)
		Pk_gravity = hmcode.power(k, zs, camb_results, T_AGN=None)
		Rk = Pk_feedback/Pk_gravity
		Rk_feedback.append(Rk)

	return Rk_feedback, k

def get_halomodel_Pk(input_cosmo={}, z=0., settings={}):
	'''
	This function calculates the power spectrum using pyhalomodel.

	Parameters:
	- input_cosmo (dict): Dictionary containing cosmological parameters.
	- z (float): Redshift value.
	- settings (dict): Additional settings for the calculation.

	Returns:
	- Pk_hm (array): Halo model power spectrum.
	- Pk_2h (array): Two-halo term power spectrum.
	- Pk_1h (array): One-halo term power spectrum.
	- ks (array): Wavenumber values.
	'''
	setup = setup_halomodel(input_cosmo=input_cosmo, z=z, settings=settings)

	ks = setup['ks']
	Ms = setup['Ms']
	rvs = setup['rvs']
	sigmaRs = setup['sigmaRs']
	cs = setup['cs']
	sigma_lnc = setup['sigma_lnc']
	hmod = setup['hmod']
	camb_results = setup['camb_results']

	cM_relation_name = settings.get('cM_relation_name', 'Ragagnin et al. (2023)')
	marginalize_cM_scatter = settings.get('marginalize_cM_scatter', False)
	print('Marginalize over c-M scatter:', marginalize_cM_scatter)

	# Create NFW profile
	Uk = win_NFW(ks, rvs, cs)

	# Marginalize over c-M relation scatter
	if marginalize_cM_scatter:
		Uk = win_NFW_marginalized(cM_relation_name=cM_relation_name, z=z, **setup)

	else:
		Uk = win_NFW(ks, rvs, cs)

	matter_profile = pyhalomodel.profile.Fourier(ks, Ms, Uk, amplitude=Ms, normalisation=hmod.rhom, mass_tracer=True) 

	# Get linear matter power spectrum from CAMB
	Pk_lin_interpolator = camb_results.get_matter_power_interpolator(nonlinear=False)
	Pk_linear = Pk_lin_interpolator.P(z, ks) # Single out the linear P(k) interpolator and evaluate linear power

	pack = ks, Pk_linear, Ms, sigmaRs
	Pk_2h, Pk_1h, Pk_hm = hmod.power_spectrum(*pack, {'m': matter_profile})

	return Pk_hm, Pk_2h, Pk_1h, ks

def setup_halomodel(input_cosmo={}, z=0., settings={}):
	if z!=0.:
		raise NotImplementedError('Redshifts other than z=0 are not implemented yet')

	# Get CAMB results
	camb_results = build_CAMB_cosmology(input_cosmo=input_cosmo, zs=[z])
	
	# Wavenumber range [h/Mpc]
	kmin, kmax = 1e-2, 50
	nk = 101
	ks = np.logspace(np.log10(kmin), np.log10(kmax), nk)

	# Halo mass range [Msun/h] over which to integrate
	Mmin, Mmax = 1e9, 1e17
	nM = 256
	Ms = np.logspace(np.log10(Mmin), np.log10(Mmax), nM)

	# Set up halo model
	Dv = settings.get('Dv', 330.)
	mass_function_name = settings.get('mass_function_name', 'Tinker et al. (2010)')
	cM_relation_name = settings.get('cM_relation_name', 'Ragagnin et al. (2023)')

	# print(f'Using delta_v = {Dv} (Bryan & Norman 1998 with Omega_m=0.3)')
	# print(f'using {mass_function_name} mass function')
	print(f'Using concentration mass relation from {cM_relation_name}')

	Omega_m = camb_results.Params.omegam
	hmod = pyhalomodel.model(z, Omega_m, name=mass_function_name, Dv=Dv, verbose=True)

	# Get sigma(R) from CAMB
	Rs = hmod.Lagrangian_radius(Ms)
	sigmaRs = camb_results.get_sigmaR(Rs, hubble_units=True, return_R_z=False)[[z].index(z)]

	rvs = hmod.virial_radius(Ms)
	cs, sigma_lnc = get_concentration_mass_relation(Ms, z, cM_relation_name, return_scatter=True)

	setup = {'ks':ks, 'Ms':Ms, 'rvs':rvs, 'sigmaRs':sigmaRs, 
		  'cs':cs, 'sigma_lnc': sigma_lnc,
		  'hmod':hmod, 'camb_results':camb_results}

	return setup

def win_NFW_marginalized(ks, cs, sigma_lnc, cM_relation_name, hmod, z, **_):
	'''
	Calculates the marginalized window function for the NFW profile.

	Parameters:
	- ks (array-like): array of wavenumbers
	- cs (array-like): array of concentrations
	- sigma_lnc (float): scatter in the concentration-mass relation
	- cM_relation_name (str): concentration-mass relation to use
	- hmod (object): pyhalomodel model object
	- z (float): redshift

	Returns:
	- Uk (array-like): The marginalized window function array with shape (k, cs.size).
	'''
	# Now we want to integrate over the scatter in the concentration-mass relation
	# Using the lognormal distribution, for each halo mass
	# Set up finer 1d grids for integration
	nsize = 1000  # Number of points for the integration
	mass_array = np.logspace(5, 20, nsize)  # between 1e5 and 1e20 Msun/h
	rvirial_array = hmod.virial_radius(mass_array)
	concentration_array = get_concentration_mass_relation(mass_array, z, cM_relation_name)
	Uk_grid = win_NFW(ks, rvirial_array, concentration_array) # shape is (k, M)

	concentration_grid = np.repeat(np.atleast_2d(concentration_array), ks.size, axis=0) # shape is (k, M)
	Uk = np.zeros(shape=(ks.size, cs.size))

	for i, this_cbar in enumerate(cs):
		# Compute the lognormal distribution
		ln_c_pdf = lognormal_pdf(concentration_grid, np.log(this_cbar), sigma_lnc)

		integrand = Uk_grid**2 * ln_c_pdf

		# Need to flip so that the grid is in increasing order
		# Otherwise integration is negative
		Uk[:, i] = np.trapz(np.flip(integrand, axis=1), np.flip(concentration_grid, axis=1), axis=1)**0.5

	return Uk

def win_NFW(k, rv, c):
	'''
	Computes the NFW window function for a given set of wavevectors.

	Parameters:
	- k (array-like): Array of wavevectors.
	- rv (float): Virial radius.
	- c (float): Concentration parameter.

	Returns:
	- Wk (array-like): NFW window function.
	'''
	rs = rv/c
	kv = np.outer(k, rv)
	ks = np.outer(k, rs)
	Sisv, Cisv = sici(ks+kv)
	Sis, Cis = sici(ks)
	f1 = np.cos(ks)*(Cisv-Cis)
	f2 = np.sin(ks)*(Sisv-Sis)
	f3 = np.sin(kv)/(ks+kv)
	f4 = np.log(1.+c)-c/(1.+c)
	Wk = (f1+f2-f3)/f4

	return Wk

def lognormal_pdf(x, mu, sigma):
	norm = (2*np.pi)**0.5 * x * sigma
	return  1/norm * np.exp( - (np.log(x) - mu)**2 / (2 * sigma**2))

def get_concentration_mass_relation(M, z, name, return_scatter=False):
	'''
	Calculate the concentration-mass relation for a given halo mass and redshift.

	Parameters:
	- M (float): Halo mass in Msun/h.
	- z (float): Redshift.
	- name (str): Name of the concentration-mass relation model.
	- return_scatter (bool, optional): Whether to return the scatter in the relation. Default is False.

	Returns:
	- float or tuple: Concentration or tuple of (concentration, scatter) depending on the value of return_scatter.

	'''

	if name == 'Duffy et al. (2008)':
		A = 7.85
		B = -0.081
		C = -0.71
		M0 = 2e12  # Msun/h
		sigma_lnc = 0.2
		c = A*(M/M0)**B*(1+z)**C

		if return_scatter:
			return c, sigma_lnc
		return c

	elif name == 'Ragagnin et al. (2023)':
		# Calirbrated from hydro simulations
		A = 1.503
		B = -0.043
		C = -0.516
		Mpivot = 19.9e13 # Msun
		a = 1/(1+z)
		ap = 0.877

		sigma_lnc = 0.388  # log-normal scatter
		ln_c = A + B* np.log(M/0.704/Mpivot) + C*np.log(a/ap)

		if return_scatter:
			return np.exp(ln_c), sigma_lnc

		return np.exp(ln_c)

def get_Pk_Pylians(cube, box_size, calc_delta, MAS, savefile=None):
	import Pk_library as PKL
	if calc_delta is False:
		if isinstance(cube, str):
			delta = np.load(cube) # this is the 3D data

		else:
			delta = cube
		Pk = PKL.Pk(delta, box_size, 0, MAS, True)

		return Pk.k3D, Pk.Pk[:,0]