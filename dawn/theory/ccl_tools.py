import numpy as np

import pyccl


class MassFuncSheth99Modified(pyccl.halos.halo_model_base.MassFunc):
	__repr_attrs__ = __eq_attrs__ = ("mass_def", "mass_def_strict",
									 "use_delta_c_fit",)
	name = 'Sheth99'

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