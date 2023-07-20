import joblib
import matplotlib.pyplot as plt
import numpy as np
import sys

import g3read

from src.Pk_tools import get_field_hist, calc_amp_FFT, calc_freq_FFT, calc_Pk

sim_box = sys.argv[1]
sim_name = sys.argv[2]
snap_dir = sys.argv[3]

#file_path = '/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/Box1a/mr_0.406_0.0466_0.867_0.712/snapdir_014/snap_014'
file_path = f'/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/{sim_box}/{sim_name}/snapdir_{snap_dir}/snap_{snap_dir}'
#file_path = '/xdisk/timeifler/pranjalrs/magneticum_data/mr_bao/snapdir_144/snap_144'
#file_path = '/xdisk/timeifler/pranjalrs/magneticum_data/Box3/hr_dm/snapdir_144/snap_144'

f = g3read.GadgetFile(file_path+'.'+'0') # Read first file to get header information
redshift = f.header.redshift
nfiles = f.header.num_files
box_size = f.header.BoxSize/1000.  # in Mpc/h
Vbox = box_size**3
resolution = 512


##------------------------------------------------------------------------------------------##
## Pk calculation
mass_hist = 0.0

mass_hist += get_field_hist(1, 'MASS', box_size, resolution, file_path, nfiles) # Dark Matter
if 'dm' not in sim_name:
	print("Getting gas, star and BH fields...")
	mass_hist += get_field_hist(0, 'MASS', box_size, resolution, file_path, nfiles)  # Gas
	mass_hist += get_field_hist(4, 'MASS', box_size, resolution, file_path, nfiles) # Star
	mass_hist += get_field_hist(5, 'BHMA', box_size, resolution, file_path, nfiles) # Black hole

avgM_per_pix = np.sum(mass_hist)/resolution**3
delta = mass_hist/avgM_per_pix - 1 

del mass_hist
#np.save('maps/mr_bao_dm.npy', delta)

delta_k_hat = calc_amp_FFT(delta, resolution) # Fourier Transform of the density field

k, Wk = calc_freq_FFT(box_size, resolution)
kbins = np.logspace(-2, 1.5, 40)

avgPk, avgk, Nk = calc_Pk(delta_k_hat, Wk, Vbox, k, kbins)

del delta_k_hat, Wk, k, kbins


##------------------------------------------------------------------------------------------##
## Shot noise calculation
numerator = 0.  # sum(m_gas) + sum(m_dm) + sum(m_star) + sum(m_bh)
denominator = 0.  # sum(m_gas)**2 + sum(m_dm)**2 + sum(m_star)**2 + sum(m_bh)**2

ptype_list = [1]
if 'dm' not in sim_name:
	ptype_list = [0, 1, 4, 5]

for ptype in ptype_list:
        
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
        'shot_noise' : shot_noise
        }



joblib.dump(data, f'../magneticum-data/data/Pk_matter/{sim_box}/{sim_name}_z={redshift:.2f}_R512.pkl')
