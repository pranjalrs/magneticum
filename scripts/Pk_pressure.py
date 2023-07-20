import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np

import g3read

import sys
sys.path.append('../core/')

from Pk_tools import get_field_hist, calc_amp_FFT, calc_freq_FFT, calc_Pk, get_field_hist_test

parser = argparse.ArgumentParser()
parser.add_argument('--sub_vol', default=False, type=bool)
parser.add_argument('--lmn', default='000')
parser.add_argument('--save_cube_only', default=True, type=bool)
parser.add_argument('--box')
parser.add_argument('--field', type=str)
parser.add_argument('--weight', default='None', type=str)
args = parser.parse_args()

l, m, n = int(args.lmn[0]), int(args.lmn[1]), int(args.lmn[2])
box = args.box
save_cube_only = args.save_cube_only
field, weight = args.field, args.weight

if box=='Box3':
	sim = 'hr_bao'

elif box=='Box1a':
	sim = 'mr_bao'
#snap_base = '/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/Box1a/mr_bao/snapdir_144/snap_144'
snap_base = f'/xdisk/timeifler/pranjalrs/magneticum_data/{box}/{sim}/snapdir_144/snap_144'

f = g3read.GadgetFile(snap_base+'.'+'0') # Read first file to get header information
redshift = f.header.redshift
nfiles = f.header.num_files
box_size = f.header.BoxSize/1000.  # in Mpc/h
Vbox = box_size**3
resolution = 512


##------------------------------------------------------------------------------------------##
## Pk calculation
cube, weights, num_part = get_field_hist_test(0, field, box_size, resolution, snap_base, nfiles, weight=weight)  # Gas

step = 50

if args.sub_vol is True:
	box_size = box_size/0.704/512*50
	resolution = step
	cube = cube[step*l:step*(l+1), step*m:step*(m+1), step*n:step*(n+1)]

if save_cube_only is True:
	if weight == 'volume':
		cube[weights!=0] = cube[weights!=0]/weights[weights!=0]
		np.save(f'maps/test/{box}_{field}_vol_weighted_R{resolution}.npy', cube)

	if weight == 'hsml':
		cube[weights!=0] = cube[weights!=0]/weights[weights!=0]
		np.save(f'maps/test/{box}_{field}_hsml_weighted_R{resolution}.npy', cube)

	if weight == 'mean':
		cube[num_part!=0] = cube[num_part!=0]/num_part[num_part!=0]
		np.save(f'maps/test/{box}_{field}_mean_R{resolution}.npy', cube)

	if weight == 'None':
		np.save(f'maps/test/{box}_{field}_R{resolution}.npy', cube)
        
	np.save(f'/xdisk/timeifler/pranjalrs/cubes/{Box}_cube_{field}_{resolution}.npy', num_part)

else:
	delta_k_hat = calc_amp_FFT(cube, resolution) # Fourier Transform of the density field

	k, Wk = calc_freq_FFT(box_size, resolution)
	kbins = np.logspace(-1, 2, 30)

	avgPk, avgk, Nk = calc_Pk(delta_k_hat, Wk, Vbox, k, kbins, CIC=True)

	del delta_k_hat, Wk, k, kbins


	##------------------------------------------------------------------------------------------##
	## Save Data
	data = {}

	data = {'snap_base'  : snap_base,
		'redshift'   : redshift,
		'box_size'   : box_size,
		'resolution' : resolution,
		'Pk'         : avgPk,
		'k'          : avgk,
		'Nk'         : Nk,
	#        'shot_noise' : shot_noise
		}

	# joblib.dump(cube, '../magneticum-data/data/pressure_grid_same_as_matter_Box3.pkl')
	# joblib.dump(data, '../magneticum-data/data/pressure_power_spectrum_Box3.pkl')
	# joblib.dump(data, f'../magneticum-data/data/Pk_pressure/Pk_z={redshift:.2f}_R{resolution}_CIC.pkl')
	joblib.dump(data, f'../magneticum-data/data/Box3/Pk_pressure/Pk_z={redshift:.2f}_R{resolution}_CIC.pkl')
	#joblib.dump(data, f'../magneticum-data/data/Pk_pressure/Pk_z={redshift:.2f}_R{resolution}_CIC_masked.pkl')
	# joblib.dump(data, f'../magneticum-data/data/Pk_pressure/Pk_z={redshift:.2f}_R{resolution}_CIC_subvolume{l}{m}{n}.pkl')
