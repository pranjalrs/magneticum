'''Reads hydro and DMO power spectra and computes
logPk ratio for cocoa baryonic scenario
'''
import argparse
import numpy as np
import scipy.interpolate

def get_logPkR(PkR, k):
    logk = np.log10(k)
    ## New kbins
    new_PkR = np.ones_like(k_new)

    ## Use extrapolation for k>25 h/Mpc (since k_Ny~25 for 128Mpc/h box and R=1024)
    inds_extr = np.where((logk>np.log10(8)) & (logk<np.log10(25)))  # ks to create extraplotor
    PkR_exterp = scipy.interpolate.UnivariateSpline(logk[inds_extr], PkR[inds_extr], k=2)
    new_PkR[k_new>25] = PkR_exterp(np.log10(k_new[k_new>25]))

    ## Use interpolation for k>kmin & k<25 h/Mpc
    PkR_interp = scipy.interpolate.interp1d(logk[k<25], PkR[k<25])

    inds = np.where((k_new>k.min()) & (k_new<25)) 
    new_PkR[inds] = PkR_interp(np.log10(k_new[inds]))

    return np.log10(new_PkR)

def numpy_to_c_array(Pk, k):
    size = k.size
    c_array = f'static double logkBins_MagWMAP7[{len(k_new)}] ='
    c_array += '{'
    c_array += ', '.join([str(k[i]) for i in range(size)])
    c_array += '};'

    # Now convert the 2D Pk array
    rows, cols = Pk.shape
    c_array += '\n '
    c_array += f'static double logPkR_MagWMAP7[{len(k_new)}][{len(zs)}] ='
    c_array += '{'

    for row in range(rows):
        c_array += '{'
        c_array += ', '.join([str(Pk[row, col]) for col in range(cols)])
        c_array += '},\n'
    
    c_array += '};'
    return c_array

zs = ['4.23', '2.79', '1.98', '1.18', '0.47', '0.25', '0.00']
k_new = np.logspace(-3.3, 3.176, 325)

logPkRs = np.zeros((len(k_new), len(zs)))

base_path = '../../magneticum-data/data/Pylians/Pk_matter/Box3'

for i, z in enumerate(zs):
    hr_bao = np.loadtxt(f'{base_path}/Pk_hr_bao_z={z}_R1024.txt')
    hr_dm = np.loadtxt(f'{base_path}/Pk_hr_dm_z={z}_R1024.txt')

    PkR = hr_bao[:, 1]/hr_dm[:, 1]
    k = hr_bao[:, 0]

    logPkRs[:, i] = get_logPkR(PkR, np.log10(k))

# Save as txt file
c_array_representation = numpy_to_c_array(logPkRs, k_new)

text_file = open("../../magneticum-data/data/cocoa_logPkR/logPkR_MagWMAP7.txt", "w")
text_file.write(c_array_representation)
text_file.close()
