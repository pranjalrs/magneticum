import numpy as np
import warnings

import astropy.units as u
import astropy.cosmology.units as cu

class Likelihood:
	'''
	Class computing the likelihood for profile fitting.

	Attributes:
	- chi2_type (str): Type of chi-squared calculation.
	- fit_params (list): List of parameter names to be fitted.
	- model (object): The model used to evaluate the likelihood.
	- profile_dict (dict): Dictionary containing the profiles of different fields.
	- fields (list): List of fields for which the likelihood is calculated.
	- priors (dict): Dictionary containing the prior ranges for the parameters.
	- mvir (float): Virial mass of the halo.
	- _model_args_return_rho (bool): Flag indicating if the model returns density profiles.
	- _model_args_return_Temp (bool): Flag indicating if the model returns temperature profiles.

	Methods:
	- __call__(self, theta): Calls the log_likelihood method.
	- log_likelihood(self, theta): Calculates the log likelihood for the given parameter values.
	- eval_model(self, theta): Evaluates the model for the given field and parameter values.
	- eval_chi2(self, data, model, sigma=None, lnsigma=None): Calculates the chi-squared value for the given data and model.
	- _init_priors(params, priors): Initializes the priors for the parameters.

	'''

	def __init__(self, profile_container_dict, model, fit_params, priors, chi2_type='log', return_blobs=True):
		'''
		Initializes the Likelihood class.

		Parameters:
		- profile_container_dict (dict): Dictionary containing the profiles of different fields.
		- model (object): The model used to evaluate the likelihood, should be a HaloProfile instance.
		- fit_params (list): List of parameter names to be fitted.
		- priors (dict): Dictionary containing the prior ranges for the parameters.
		- chi2_type (str, optional): Type of chi-squared calculation. Default is 'log'.
		- return_blobs (bool, optional): Flag indicating if additional information should be returned. Default is True.
		'''
		self.chi2_type = chi2_type
		self.fit_params = fit_params
		self.model = model

		self.priors = self._init_priors(fit_params, priors)
		self.return_blobs = return_blobs

		self.init_data(profile_container_dict)

	def init_data(self, container_dict):
		'''
		Initializes the data for likelihood calculation.

		Parameters:
		- container_dict (dict): Dictionary containing the profiles of different fields.

		Returns:
		- list: List of field names.

		Raises:
		- AssertionError: If fields are not among ["rho_dm", "rho_gas", "Pe", "Temp"].
		'''
		self.profile_dict = container_dict
		self.fields = list(container_dict.keys())

		assert all([field in ['rho_dm', 'rho_gas', 'Pe', 'Temp'] for field in self.fields]), 'Fields should be among ["rho_dm", "rho_gas", "Pe", "Temp"]'

		self.mvir = self.profile_dict[self.fields[0]].mvir * u.Msun / cu.littleh

		self._model_args_return_rho = True if 'rho_gas' in self.fields else False
		self._model_args_return_Temp = True if 'Temp' in self.fields else False

		return list(self.profile_dict.keys())

	def __call__(self, theta):
		'''
		Calls the log_likelihood method.

		Parameters:
		- theta (dict): Dictionary of parameter values.

		Returns:
		- float: The log likelihood value.
		'''
		return self.log_likelihood(theta)

	def log_likelihood(self, theta):
		'''
		Calculates the log likelihood for the given parameter values.

		Parameters:
		- theta (dict): Dictionary of parameter values.

		Returns:
		- float: The log likelihood value.
		'''
		for i, param in enumerate(self.fit_params):
			if not self.priors[param][0] <= theta[i] <= self.priors[param][1]:
			
				if self.return_blobs:
					return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf
				else:
					return -np.inf

		model_prediction = self.eval_model(theta)

		chi2 = {'rho_dm': 0, 'rho_gas': 0, 'Temp': 0, 'Pe': 0}

		for field in self.fields:
			data = self.profile_dict[field].profile
			sigma_prof = self.profile_dict[field].sigma_prof
			sigma_lnprof = self.profile_dict[field].sigma_lnprof
			sigma_intr = self.profile_dict[field].sigma_intr

			chi2[field] = self.eval_chi2(data, model_prediction[field].value, sigma_prof, sigma_lnprof, sigma_intr)

		if self.return_blobs:
			return -0.5 * np.sum(list(chi2.values())), -0.5 * chi2['rho_dm'], -0.5 * chi2['rho_gas'], -0.5 * chi2['Temp'], -0.5 * chi2['Pe']

		else:
			return -0.5 * np.sum(list(chi2.values()))


	def eval_model(self, theta):
		'''
		Evaluates the model for the given field and parameter values.

		Parameters:
		- theta (dict): Dictionary of parameter values.

		Returns:
		- dict: Dictionary containing the model predicted values for each field.
		'''
		self.model.update_param(self.fit_params, theta)

		model_prediction = {'rho_dm': None, 'Pe': None, 'rho_gas': None, 'Temp': None}

		for field in self.fields:
			if field == 'rho_dm':
				rbins = self.profile_dict['rho_dm'].rbins
				model_prediction['rho_dm'], _ = self.model.get_rho_dm_profile_interpolated(self.mvir, r_bins=rbins, z=0.)

			else:
				rbins = self.profile_dict[field].rbins
				profs, _ = self.model.get_Pe_profile_interpolated(self.mvir, r_bins=rbins, z=0.,
															return_rho=self._model_args_return_rho,
															return_Temp=self._model_args_return_Temp)
				model_prediction[field] = profs[field]

		return model_prediction

	def eval_chi2(self, data, model, sigma=None, lnsigma=None, sigma_intr=0.):
		'''
		Calculates the chi-squared value for the given data and model.

		Parameters:
		- data (ndarray): The observed data.
		- model (ndarray): The model predicted values.
		- sigma (ndarray, optional): The standard deviation of the data. Default is None.
		- lnsigma (ndarray, optional): The natural logarithm of the standard deviation of the data. Default is None.

		Returns:
		- float: The chi-squared value.
		'''
		if self.chi2_type == 'log':
			num = np.log(data / model)
			denom = (lnsigma**2 + sigma_intr**2)**0.5

		elif self.chi2_type == 'linear':
			num = (data - model)
			denom = (sigma**2 + sigma_intr**2)**0.5

		idx = data == 0

		residual = (num[~idx] / denom[~idx])  # Sum over radial bins

		if not np.all(np.isfinite(residual)):
			return np.inf

		else:
			return np.sum(residual**2)

	@staticmethod
	def _init_priors(params, priors):
		'''
		Initializes the priors for the parameters.

		Parameters:
		- params (list): List of parameter names.
		- priors (dict): Dictionary containing priors for the parameters.

		Returns:
		- dict: Updated dictionary of priors.

		Raises:
		- ValueError: If a parameter in `params` does not have a prior in `priors`.
		- Warning: If additional parameters are found in `priors` that are not in `params`.
		'''
		for param in params:
			if param not in priors:
				raise ValueError(f'No prior found for parameter {param}')

		for param in priors:
			if param not in params:
				warnings.warn(f'Prior dictionary contains additional parameter `{param}` !')

		return priors