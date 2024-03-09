import astropy.cosmology.units as cu
import astropy.units as u

class GadgetUnits():
	def __init__(self):
		# Internal GADGET units, see:  https://wwwmpa.mpa-garching.mpg.de/~kdolag/GadgetHowTo/right.html#Format2
		self.length = 1*u.kpc/cu.littleh  # kpc/h
		self.mass = 1e10*u.Msun/cu.littleh  # Msun/h
		self.velocity = 1*u.km/u.second
		self.time = (self.length/self.velocity).to(u.second)
		self.Temperature = 1*u.K

		self.mass_density = self.mass/self.length**3


class GadgetConversion():
	def __init__(self):
		pass

	def density_to_physical(self, z, little_h):
		return (1+z)**3*little_h**2

	def pressure_to_physical(self, z, little_h):
		return (1+z)**3*little_h**2


class GADGET():
	def __init__(self):
		self.units = GadgetUnits()
		self.convert = GadgetConversion()

Gadget = GADGET()