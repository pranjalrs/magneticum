import joblib
import matplotlib.pyplot as plt
import numpy as np

import g3read
import sys
sys.path.append('../src/')

from Pk_tools import get_field_hist, calc_amp_FFT, calc_freq_FFT, calc_Pk_cross

file_path = '/xdisk/timeifler/pranjalrs/magneticum_data/mr_bao/snapdir_144/snap_144'

f = g3read.GadgetFile(file_path+'.'+'0') # Read first file to get header information
redshift = f.header.redshift
nfiles = f.header.num_files
box_size = f.header.BoxSize/1000.  # in Mpc/h
Vbox = box_size**3
resolution = 512


##------------------------------------------------------------------------------------------##
delta_dm = np.load('maps/mr_bao_dm.npy')
Pe = np.load('maps/mr_bao_grid.npy')

delta_k_hat1 = calc_amp_FFT(delta_dm, resolution) # Fourier Transform of the density field
delta_k_hat2 = calc_amp_FFT(Pe, resolution) # Fourier Transform of the density field

del delta_dm, Pe

k, Wk = calc_freq_FFT(box_size, resolution)
kbins = np.logspace(-2, 1, 20)

avgPk, avgk, Nk = calc_Pk_cross(delta_k_hat1, delta_k_hat2, Wk, Vbox, k, kbins)

del delta_k_hat1, delta_k_hat2,  Wk, k, kbins


##------------------------------------------------------------------------------------------##
## Shot noise calculation
numerator = 0.  # sum(m_gas) + sum(m_dm) + sum(m_star) + sum(m_bh)
denominator = 0.  # sum(m_gas)**2 + sum(m_dm)**2 + sum(m_star)**2 + sum(m_bh)**2

for ptype in [0, 1, 4, 5]:
        
        for i in range(nfiles):
                this_snap = g3read.GadgetFile(file_path + '.' + str(i))
                
                if ptype == 5:
                    particle_mass = this_snap.read_new('BHMA', ptype)
                else:
                    particle_mass = this_snap.read_new('MASS', ptype)

                numerator += np.sum(particle_mass)
                denominator += np.sum(particle_mass**2)


Neff = numerator**2/denominator # (sum(m_gas) + sum(m_dm) + sum(m_star) + sum(m_bh))**2 / (sum(m_gas)**2 + sum(m_dm)**2 + sum(m_star)**2 + sum(m_bh)**2)

shot_noise = Vbox/Neff



##------------------------------------------------------------------------------------------##
## Save Data
data = {}

data = {'file_path'  : file_path,
        'redshift'   : redshift,
        'box_size'   : box_size,
        'resolution' : resolution,
        'Pk'         : avgPk,
        'k'          : avgk,
        'Nk'         : Nk,
        }



joblib.dump(data, '../magneticum-data/data/Pk_dmxPe/Pk_dmxPe_z=0_R512.pkl')
