import numpy as np

import pyccl


class MassFuncSheth99Modified(pyccl.halos.halo_model_base.MassFunc):
	__repr_attrs__ = __eq_attrs__ = ("mass_def", "mass_def_strict",
									 "use_delta_c_fit",)
	name = 'Sheth99Modified'

	def __init__(self, *,
				 mass_def="fof",
				 mass_def_strict=True,
				 use_delta_c_fit=False):
		self.use_delta_c_fit = use_delta_c_fit
		super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

	def _check_mass_def_strict(self, mass_def):
		return mass_def.Delta != "fof"

	def _setup(self):
		self.A = 0.27
		self.p = -0.28
		self.a = 1.05

	def _get_fsigma(self, cosmo, sigM, a, lnM):
		if self.use_delta_c_fit:
			delta_c = pyccl.halos.halo_model_base.get_delta_c(cosmo, a, 'NakamuraSuto97')
		else:
			delta_c = pyccl.halos.halo_model_base.get_delta_c(cosmo, a, 'EdS')

		nu = delta_c / sigM
		return nu * self.A * (1. + (self.a * nu**2)**(-self.p)) * (
			np.exp(-self.a * nu**2/2.))


class MassFuncTinker08Modified(pyccl.halos.halo_model_base.MassFunc):
	"""Implements the mass function of `Tinker et al. 2008
	<https://arxiv.org/abs/0803.2706>`_. This parametrization accepts S.O.
	masses with :math:`200 < \\Delta < 3200`, defined with respect to the
	matter density. This can be automatically translated to S.O. masses
	defined with respect to the critical density.

	Args:
		mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
			a mass definition object, or a name string.
		mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
			definition will be ignored.
	"""
	name = 'Tinker08Modified'

	def __init__(self, *,
				 mass_def="200m",
				 mass_def_strict=True):
		super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

	def _check_mass_def_strict(self, mass_def):
		return mass_def.Delta == "fof"

	def _setup(self):
		self.pA0 = None #0.2
		self.pa0 = None #1.52
		self.pb0 = None #2.25
		self.pc = None #1.27

	def _get_fsigma(self, cosmo, sigM, a, lnM):
		ld = np.log10(self.mass_def._get_Delta_m(cosmo, a))
		pA = self.pA0 * a**0.14
		pa = self.pa0 * a**0.06
		pd = 10.**(-(0.75/(ld - 1.8750612633))**1.2)
		pb = self.pb0 * a**pd
		return pA * ((pb / sigM)**pa + 1) * np.exp(-self.pc/sigM**2)

class ConcentrationDuffy08Modified(pyccl.halos.halo_model_base.Concentration):
	"""Concentration-mass relation by `Duffy et al. 2008
	<https://arxiv.org/abs/0804.2486>`_. This parametrization is only
	valid for S.O. masses with :math:`\\Delta = \\Delta_{\\rm vir}`,
	of :math:`\\Delta=200` times the matter or critical density.
	By default it will be initialized for :math:`M_{200c}`.

	Args:
		mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`): a mass
			definition object, or a name string.
	"""
	name = 'Duffy08'

	def __init__(self, *, mass_def="200c"):
		super().__init__(mass_def=mass_def)

	def _check_mass_def_strict(self, mass_def):
		return mass_def.name not in ["vir", "200m", "200c"]

	def _setup(self):
		vals = {"vir": (7.85, -0.081, -0.71),
				"200m": (10.14, -0.081, -1.01),
				"200c": (5.71, -0.084, -0.47)}

		self.A, self.B, self.C = vals[self.mass_def.name]

	def _concentration(self, cosmo, M, a):
		M_pivot_inv = cosmo["h"] * 5E-13
		return self.A * (M * M_pivot_inv)**self.B * a**(-self.C)