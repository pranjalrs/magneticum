'''Reads hydro and DMO power spectra and computes
logPk ratio for cocoa baryonic scenario
'''
import argparse
import glob
import numpy as np
import re
import scipy.interpolate

import ipdb

def get_logPkR(PkR, k):
    logk = np.log10(k)
    ## New kbins
    new_PkR = np.ones_like(k_new)

    ## Use extrapolation for k>25 h/Mpc (since k_Ny~25 for 128Mpc/h box and R=1024)
    kmax = 20
    inds_extr = np.where((logk>np.log10(8)) & (logk<np.log10(kmax)))  # ks to create extraplotor
    PkR_exterp = scipy.interpolate.UnivariateSpline(logk[inds_extr], PkR[inds_extr], k=2)
    new_PkR[k_new>kmax] = PkR_exterp(np.log10(k_new[k_new>kmax]))

    ## Use interpolation for k>kmin & k<25 h/Mpc
    PkR_interp = scipy.interpolate.interp1d(logk[k<kmax], PkR[k<kmax])

    inds = np.where((k_new>k.min()) & (k_new<kmax)) 
    new_PkR[inds] = PkR_interp(np.log10(k_new[inds]))

    if np.any(np.isnan(np.log10(new_PkR))):
        ipdb.set_trace()
    return np.log10(new_PkR)

def numpy_to_c_array(Pk, k, C):
    size = k.size
    c_array = f'static double logkBins_Mag{C}[{len(k_new)}] ='
    c_array += '{'
    c_array += ', '.join([str(k[i]) for i in range(size)])
    c_array += '};'

    # Now convert the 2D Pk array
    rows, cols = Pk.shape
    c_array += '\n '
    c_array += f'static double logPkR_Mag{C}[{len(k_new)}][{len(zs)}] ='
    c_array += '{'

    for row in range(rows):
        c_array += '{'
        c_array += ', '.join([str(Pk[row, col]) for col in range(cols)])
        c_array += '},\n'
    
    c_array += '};'
    return c_array

# zs = ['4.23', '2.79', '1.98', '1.18', '0.47', '0.25', '0.00']
k_new = np.logspace(-3.3, 3.176, 325)

# logPkRs = np.zeros((len(k_new), len(zs)))

# base_path = '../../magneticum-data/data/Pylians/Pk_matter/Box3'

# for i, z in enumerate(zs):
#     hr_bao = np.loadtxt(f'{base_path}/Pk_hr_bao_z={z}_R1024.txt')
#     hr_dm = np.loadtxt(f'{base_path}/Pk_hr_dm_z={z}_R1024.txt')

#     PkR = hr_bao[:, 1]/hr_dm[:, 1]
#     k = hr_bao[:, 0]

#     logPkRs[:, i] = get_logPkR(PkR, k)

# # Save as txt file
# c_array_representation = numpy_to_c_array(logPkRs, np.log10(k_new))

# text_file = open("../../magneticum-data/data/cocoa_logPkR/logPkR_MagWMAP7.txt", "w")
# text_file.write(c_array_representation)
# text_file.close()


# For Box 3 MC
name_key = {
    'hr_0.153_0.0408_0.614_0.666': 'C1',
    'hr_0.189_0.0455_0.697_0.703': 'C2',
    'hr_0.200_0.0415_0.850_0.730': 'C3',
    'hr_0.204_0.0437_0.739_0.689': 'C4',
    'hr_0.222_0.0421_0.793_0.676': 'C5',
    'hr_0.232_0.0413_0.687_0.670': 'C6',
    'hr_0.268_0.0449_0.721_0.699': 'C7',
    'hr_bao': 'WMAP7',
    'hr_0.301_0.0460_0.824_0.707': 'C9',
    'hr_0.304_0.0504_0.886_0.740': 'C10',
    'hr_0.342_0.0462_0.834_0.708': 'C11',
    'hr_0.363_0.0490_0.884_0.729': 'C12',
    'hr_0.400_0.0485_0.650_0.675': 'C13',
    'hr_0.406_0.0466_0.867_0.712': 'C14',
    'hr_0.428_0.0492_0.830_0.732': 'C15'}

files = sorted(glob.glob('../../magneticum-data/data/Pylians/Pk_matter/Box3/hr_0.*/'))

for f in files:
    sim = f.split('/')[-2]
    
    if sim in ['hr_0.304_0.0504_0.886_0.740', 'hr_0.342_0.0462_0.834_0.708']:
        continue
    
    cosmo_snaps = sorted(glob.glob(f+f'Pk_{sim}_z=*_R1024.txt'), reverse=True)

    logPkRs = np.zeros((len(k_new), len(cosmo_snaps)))
    zs = []
    for i, snap in enumerate(cosmo_snaps):
        match = re.search(r'z=([0-9.]+)', snap.split('/')[-1])
        z = match.group(1)
        
        # Due to strange wiggles in suppresion ignore z>2.3
        if float(z)>2.33:
            continue
        Pk_hydro = np.loadtxt(snap)
    
        snap = snap.replace('hr', 'dm_hr')
        sim = f.split('/')[-2]
        
        try:
            Pk_dm = np.loadtxt(snap)

        except FileNotFoundError:
            continue
            

        PkR = Pk_hydro[:, 1]/Pk_dm[:, 1]
        k = Pk_hydro[:, 0]

        logPkRs[:, i] = get_logPkR(PkR, k)
        
#         ipdb.set_trace()
        zs.append(z)
    
    zs = [float(z) for z in zs]
    print(sim, zs)

    #Make sure that redshifts are in increasing order
    assert np.all(zs == np.sort(zs)[::-1])
    c_array_representation = numpy_to_c_array(logPkRs, np.log10(k_new), name_key[sim])

    text_file = open(f"../../magneticum-data/data/cocoa_logPkR/logPkR_Mag{name_key[sim]}.txt", "w")
    text_file.write(c_array_representation)
    text_file.close()