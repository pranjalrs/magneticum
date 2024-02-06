import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import scipy
import scipy.optimize


import g3read as g3r
import g3matcha as g3m

## Select halo
base_folder = '/HydroSims/Magneticum/Box2b/hr_bao/'
snapnum = 144
catalog_path = base_folder + '/groups_%03d/sub_%03d'%(snapnum, snapnum)
snap_path = base_folder + '/snapdir_%03d/snap_%03d'%(snapnum, snapnum)



particles = g3r.read_particles_in_box(snap_path, gpos, Rvir, ['MASS','POS '], 1)

positions = particles['POS ']
distances = (
    (positions[:,0] - gpos[0]) **2 +  
    (positions[:,1] - gpos[1]) **2 +
    (positions[:,2] - gpos[2])**2 
) ** 0.5
mask = distances<Rvir

masses = particles['MASS'][mask]
positions = positions[mask]
distances = distances[mask]




bins = np.logspace(-1.5,0,20)
x = distances/Rvir

bins_n = np.histogram(x, bins=bins)[0]
bins_x_sum = np.histogram(x, weights=x, bins=bins)[0]
bins_m = np.histogram(x, weights=masses, bins=bins)[0]
bins_shell = 4./3.*np.pi*(bins[1:]**3 - bins[:-1]**3)
bins_rho = bins_m / bins_shell
bins_x = bins_x_sum/bins_n


def f(c):
    return np.log(1.+c) -  c/(1.+c)

myprofile_nfw=lambda r,rho0,rs: rho0 / ((r/rs) * (1.+r/rs)**2)

def minimize_me(x):

    v = np.sqrt(
        np.sum(
            np.abs(
                np.log(myprofile_nfw(bins_x[bins_x>0], *x)/bins_rho[bins_x>0])
                )**2
            )
    )
    #print(x,v)
    return v

x0 = [1e4,.3]
res = scipy.optimize.minimize(minimize_me,x0,bounds = [[1e3,1e6],[.01,1.]])
xrho0, xrs = res['x']
rs = xrs*Rvir
rho0 = xrho0/Rvir**3
c = Rvir/rs 

plt.loglog(bins_x, bins_rho/Rvir**3,label='data')
plt.loglog(bins_x, myprofile_nfw(bins_x*Rvir, rho0, rs), label='fit, c=%.1f'%(c))
plt.xlabel(r'$r/R_{\rm vir}$')
plt.ylabel(r'$\rho [h^2 10^{10}M_\odot {\rm ckpc}^{-3}]$')
plt.legend()
print('xrs %.2e'%xrs,'xrho0 %.2e'%xrho0,'rs %.2e'%rs, 'rho0 %.2e'%rho0, 'NFW rho0 %.2e'%(Mvir/(4.*np.pi*rs**3 * f(Mvir/rs))))