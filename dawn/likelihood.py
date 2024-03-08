import numpy as np

class Likelihood:
	'''
	Class computing the likelihood for profile fitting.

	Attributes:
	- chi2_type (str): Type of chi-squared calculation.
	- fit_par (list): List of parameter names to be fitted.
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

	def __init__(self, profile_container_dict, model, fit_params, priors, chi2_type='log'):
		'''
		Parameters:
		- profile_container_dict (dict): Dictionary containing the profiles of different fields.
		- model (object): The model used to evaluate the likelihood, should be a HaloProfile instance.
		- fit_params (list): List of parameter names to be fitted.
		- priors (dict): Dictionary containing the prior ranges for the parameters.
		- chi2_type (str, optional): Type of chi-squared calculation. Default is 'log'.
		'''
		self.chi2_type = chi2_type
		self.fit_par = fit_params
		self.model = model
		self.profile_dict = profile_container_dict

		self.fields = profile_container_dict.keys()
		assert all([field in ['rho_dm', 'rho_gas', 'Pe', 'Temp'] for field in self.fields]), 'Fields should be among ["rho_dm", "rho_gas", "Pe", "Temp"]'

		self.priors = self._init_priors(fit_params, priors)
		self.mvir = self.profile_dict[self.fields[0]].mvir

		if 'rho_gas' in self.fields: 
			self._model_args_return_rho = True
		else: 
			self._model_args_return_rho = False

		if 'Temp' in self.fields: 
			self._model_args_return_Temp = True
		else: 
			self._model_args_return_Temp = False

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
		for param in self.fit_params:
			if not self.priors[param][0] <= theta[param] <= self.priors[param][1]:
				return -np.inf

		model_prediction = self.eval_model(theta)

		chi2 = 0
		for field in self.fields:
			data = self.profile_container_dict[field].profile
			sigma_prof = self.profile_container_dict[field].sigma_prof
			sigma_lnprof = self.profile_container_dict[field].sigma_lnprof

			chi2 += self.eval_chi2(data, model_prediction[field], sigma_prof, sigma_lnprof)

		return -0.5*chi2

	def eval_model(self, theta):
		'''
		Evaluates the model for the given field and parameter values.

		Parameters:
		- theta (dict): Dictionary of parameter values.

		Returns:
		- dict: Dictionary containing the model predicted values for each field.

		'''
		self.model.update_param(self.fit_params, theta)

		model_prediction = {'rho_dm': None,
							'Pe': None,
							'rho_gas': None,
							'Temp': None}

		if 'rho_dm' in self.fields:
			rbins = self.profile_dict['rho_dm'].rbins
			model_prediction['rho_dm'] = self.model.get_rho_dm_profile_interpolated(self.mvir, r_bins=rbins, z=0.)[0]

		rbins = self.profile_dict['rho_gas'].rbins
		profs, _ = self.get_Pe_profile_interpolated(self.mvir, r_bins=rbins, z=0., 
													return_rho=self._model_args_return_rho, return_Temp=self._model_args_return_Temp)

		for field in self.fields:
			if field == 'rho_dm': 
				continue
			model_prediction[field] = profs[field]

		return model_prediction

	def eval_chi2(self, data, model, sigma=None, lnsigma=None):
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
			denom = sigma
		elif self.chi2_type == 'linear':
			num = (data - model)
			denom = lnsigma

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
				raise Warning(f'Prior dictionary contains additional parameter `{param}` !')

		return priors
