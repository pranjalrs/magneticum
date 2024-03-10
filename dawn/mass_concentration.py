from abc import ABC, abstractmethod
import numpy as np

class MassConcentrationRelation(ABC):
	def __init__(self):
		pass

	@abstractmethod
	def __call__(self, mass, redshift):
		pass

class Duffy08(MassConcentrationRelation):
	def __init__(self):
		self.A=7.85
		self.B=-0.081
		self.C=-0.71
		self.M0=2e12
		self.sigma_lnc=0.3

	def __call__(self, mass, redshift):
		return self.A*(mass/self.M0)**self.B*(1+redshift)**self.C

class Ragagnin23(MassConcentrationRelation):
	def __init__(self):
		self.A = 1.503
		self.B = -0.043
		self.C = -0.516
		self.M0 = 19.9e13
		self.sigma_lnc = 0.388

	def __call__(self, mass, redshift):
		a = 1/(1+redshift)
		ap = 0.877
		return np.exp(self.A + self.B*np.log(mass/0.704/self.M0) + self.C*np.log(a/ap))

class CustomMassConcentrationRelation(MassConcentrationRelation):
	def __init__(self, A, B, C, M0, sigma_lnc=None):
		self.A = A
		self.B = B
		self.C = C
		self.M0 = M0
		self.sigma_lnc = sigma_lnc

	def __call__(self, mass, redshift):
		return self.A*(mass/self.M0)**self.B*(1+redshift)**self.C

def get_mass_concentration_relation(name, **kwargs):
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
		return Duffy08()

	elif name == 'Ragagnin et al. (2023)':
		return Ragagnin23()
	
	elif name == 'custom':
		return CustomMassConcentrationRelation(**kwargs)