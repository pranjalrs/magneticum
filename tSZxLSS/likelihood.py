import numpy as np

class Likelihood:
	def __init__(self, profile_container_dict, model, fit_params, priors, chi2_type='log'):
		self.chi2_type = chi2_type
		self.fit_par = fit_params
		self.model = model
		self.profile_dict = profile_container_dict

		self.fields = profile_container_dict.keys()
		# assert that fields among ['rho_dm', 'rho_gas', 'Pe', 'Temp']
		assert all([field in ['rho_dm', 'rho_gas', 'Pe', 'Temp'] for field in self.fields]), 'Fields should be among ["rho_dm", "rho_gas", "Pe", "Temp"]'

		self.priors = self._init_priors(fit_params, priors)
		self.mvir = self.profile_dict[self.fields[0]].mvir

		if 'rho_gas' in self.fields: 
			self._model_args_return_rho = True
		else: self._model_args_return_rho = False

		if 'Temp' in self.fields: 
			self._model_args_return_Temp = True
		else: self._model_args_return_Temp = False


	def __call__(self, theta):
		return self.log_likelihood(theta)


	def log_likelihood(self, theta):
		# check if the parameter values are within the prior range
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
		Evaluate the model for the given field and parameter values.

		Parameters:
		- field (str): The field for which the model is to be evaluated.
		- theta (dict): Dictionary of parameter values.

		Returns:
		- model (ndarray): The model predicted values for the given field.
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
			if field == 'rho_dm': continue
			model_prediction[field] = profs[field]

		return model_prediction


	def eval_chi2(self, data, model, sigma=None, lnsigma=None):
		'''
		Calculate the chi-squared value for the given data and model.

		Parameters:
		- data (ndarray): The observed data.
		- model (ndarray): The model predicted values.
		- sigma (ndarray, optional): The standard deviation of the data. Default is None.
		- lnsigma (ndarray, optional): The natural logarithm of the standard deviation of the data. Default is None.

		Returns:
			float: The chi-squared value.
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
		Initialize the priors for the parameters.

		Parameters:
		- params (list): List of parameter names.
		- priors (dict): Dictionary containing priors for the parameters.

		Returns:
		- priors (dict): Updated dictionary of priors.

		Raises:
		- ValueError: If a parameter in `params` does not have a prior in `priors`.
		- Warning: If additional parameters are found in `priors` that are not in `params`.
		'''
		for param in params:
			if param not in priors:
				raise ValueError(f'No prior found for parameter {param}')
		
		# Also raise warning if additional parameters are found in the priors dictionary
		for param in priors:
			if param not in params:
				raise Warning(f'Prior dictionary contains additional parameter `{param}` !')

		return priors