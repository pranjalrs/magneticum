import numpy as np
import scipy
from scipy.special import sici

import camb
import pyhalomodel

import utils
from mass_concentration import get_mass_concentration_relation
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

	results = camb.get_results(pars)
	sigma_8 = results.get_sigma8_0()
	print('Final sigma_8:', sigma_8)

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
	hmod = setup['hmod']
	camb_results = setup['camb_results']

	marginalize_mc_scatter = settings.get('marginalize_mc_scatter', False)
	use_KDE = settings.get('use_KDE', False)
	print('Marginalize over c-M scatter:', marginalize_mc_scatter)

	# Create NFW profile
	Uk = win_NFW(ks, rvs, cs)

	# Marginalize over c-M relation scatter
	if marginalize_mc_scatter:
		if use_KDE is True:
			KDE = settings.get('KDE_dict', None)
			if KDE is None:
				raise ValueError('KDE must be provided if use_KDE is True') 
			# Use KDE to marginalize over the concentration-mass relation scatter
			Uk = win_NFW_marginalized(use_KDE=use_KDE, KDE=KDE, z=z, **setup)

		else:
			Uk = win_NFW_marginalized(z=z, **setup)

	else:
		Uk = win_NFW(ks, rvs, cs)

	matter_profile = pyhalomodel.profile.Fourier(ks, Ms, Uk, amplitude=Ms, normalisation=hmod.rhom, mass_tracer=True) 

	# Get linear matter power spectrum from CAMB
	Pk_lin_interpolator = camb_results.get_matter_power_interpolator(nonlinear=False)
	Pk_linear = Pk_lin_interpolator.P(z, ks) # Single out the linear P(k) interpolator and evaluate linear power

	pack = ks, Pk_linear, Ms, sigmaRs
	Pk_2h, Pk_1h, Pk_hm = hmod.power_spectrum(*pack, {'m': matter_profile})#, simple_twohalo=True)

	return Pk_hm, Pk_2h, Pk_1h, ks

def setup_halomodel(input_cosmo={}, z=0., settings={}):
	'''
	Set up the halo model for a given cosmology and redshift.

	Args:
	- input_cosmo (dict): Input cosmological parameters.
	- z (float): Redshift.
	- settings (dict): Additional settings for the halo model. May contain the following keys:
		kmin (float): Minimum wavenumber for the power spectrum calculation. Default is 1e-2.
		kmax (float): Maximum wavenumber for the power spectrum calculation. Default is 50.
		nk (int): Number of samples in k. Default is 101.
		Mmin (float): Minimum halo mass for the power spectrum calculation. Default is 1e9.
		Mmax (float): Maximum halo mass for the power spectrum calculation. Default is 1e17.
		nM (int): Number of samples in halo mass. Default is 256.
		Dv (float): Virial overdensity. Default is 330.
		mass_function_name (str): Name of the mass function to use. Default is 'Tinker et al. (2010)'.
		mc_relation_name (str): Name of the concentration-mass relation to use. Default is 'Ragagnin et al. (2023)'.
	Returns:
		dict: Dictionary containing the setup information for the halo model.
	'''
	if z!=0.:
		raise NotImplementedError('Redshifts other than z=0 are not implemented yet')

	# Get CAMB results
	camb_results = build_CAMB_cosmology(input_cosmo=input_cosmo, zs=[z])
	
	# Wavenumber range [h/Mpc]
	kmin, kmax = settings.get('kmin', 1e-2), settings.get('kmax', 50)
	nk = settings.get('nk', 101)
	ks = np.logspace(np.log10(kmin), np.log10(kmax), nk)

	# Halo mass range [Msun/h] over which to integrate
	Mmin, Mmax = settings.get('Mmin', 1e9), settings.get('Mmax', 1e17)
	nM = settings.get('nM', 256)
	Ms = np.logspace(np.log10(Mmin), np.log10(Mmax), nM)

	# Set up halo model
	Dv = settings.get('Dv', 330.)
	mass_function_name = settings.get('mass_function_name', 'Tinker et al. (2010)')
	mc_relation_name = settings.get('mc_relation_name', 'Ragagnin et al. (2023)')
	mc_relation_kwargs = settings.get('mc_relation_kwargs', {})

	# print(f'Using delta_v = {Dv} (Bryan & Norman 1998 with Omega_m=0.3)')
	# print(f'using {mass_function_name} mass function')
	print(f'Using mass concentration relation from {mc_relation_name}')

	Omega_m = camb_results.Params.omegam
	hmod = pyhalomodel.model(z, Omega_m, name=mass_function_name, Dv=Dv, verbose=True)

	# Get sigma(R) from CAMB
	Rs = hmod.Lagrangian_radius(Ms)
	sigmaRs = camb_results.get_sigmaR(Rs, hubble_units=True, return_R_z=False)[[z].index(z)]

	rvs = hmod.virial_radius(Ms)
	mc_relation = get_mass_concentration_relation(mc_relation_name, **mc_relation_kwargs)

	cs = mc_relation(Ms, z)
	sigma_lnc = mc_relation.sigma_lnc

	setup = {'ks':ks, 'Ms':Ms, 'rvs':rvs, 'sigmaRs':sigmaRs, 
		  'cs':cs, 'sigma_lnc': sigma_lnc, 'mc_relation':mc_relation,
		  'hmod':hmod, 'camb_results':camb_results}

	return setup

def win_NFW_marginalized(ks, cs, sigma_lnc, mc_relation, hmod, z, Ms=None, use_KDE=False, KDE=None, **_):
	'''
	Calculates the marginalized window function for the NFW profile.

	Parameters:
	- ks (array-like): array of wavenumbers
	- cs (array-like): array of concentrations
	- sigma_lnc (float): scatter in the concentration-mass relation
	- mc_relation (MassConcentrationRelation): MassConcentrationRelation class object
	- hmod (object): pyhalomodel model object
	- z (float): redshift
	- Ms (array-like, optional): array of halo masses (default: None)
	- use_KDE (bool, optional): flag to use Kernel Density Estimation (default: False)
	- KDE (object, optional): Kernel Density Estimation object (default: None)

	Returns:
	- Uk (array-like): The marginalized window function array with shape (k, cs.size).
	'''
	# Now we want to integrate over the scatter in the concentration-mass relation
	# Using the lognormal distribution, for each halo mass
	# Set up finer 1d grids for integration
	nsize = 1000  # Number of points for the integration
	mass_array = np.logspace(5, 20, nsize)  # between 1e5 and 1e20 Msun/h
	rvirial_array = hmod.virial_radius(mass_array)
	concentration_array = mc_relation(mass_array, z)
	Uk_grid = win_NFW(ks, rvirial_array, concentration_array) # shape is (k, M)

	concentration_grid = np.repeat(np.atleast_2d(concentration_array), ks.size, axis=0) # shape is (k, M)
	Uk = np.zeros(shape=(ks.size, cs.size))
	for i, this_cbar in enumerate(cs):

		if use_KDE is True:
			halo_mass = Ms[i]
			this_KDE = utils.select_KDE(KDE, halo_mass)(concentration_array)

			ln_c_pdf = np.repeat(np.atleast_2d(this_KDE), ks.size, axis=0) # shape is (k, M)

		else:
			# Compute the lognormal distribution
			ln_c_pdf = utils.lognormal_pdf(concentration_grid, np.log(this_cbar), sigma_lnc)

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
