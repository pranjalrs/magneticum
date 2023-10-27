'''Same as fit_HMCode_profile.py but fits all profiles instead of 
the mean profile in each mass bin
'''
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import matplotlib.pyplot as plt
import numpy as np

import pygtc

import getdist
from getdist import plots


import sys
sys.path.append('../core/')

from analytic_profile import Profile
import post_processing

import ipdb

#####-------------- Parse Args --------------#####

parser = argparse.ArgumentParser()
parser.add_argument('--field')
parser.add_argument('--run', type=str)

args = parser.parse_args()
run = args.run

#####-------------- Prepare for MCMC --------------#####
fitter = Profile(use_interp=True, mmin=1e11-1e10, mmax=1e15+1e10)
print('Initialized profile fitter ...')


param_names = ['gamma', 'alpha', 'log10_M0', 'eps1_0', 'eps2_0', 'gammaT']
param_latex_names = ['\Gamma', '\\alpha', '\log_{10}M_0', '\epsilon_1', '\epsilon_2', '\Gamma_\mathrm{T}']


#### Discard 0.9*steps and make triangle plot
field = args.field.strip('"').split(',')

gd_samples = []
chain_samples = []
legend = []

for m in [11, 12, 13, 14]:
    save_path = f'../../magneticum-data/data/emcee/prof_{args.field}_halos_bin/'

    walkers = np.load(f'{save_path}/mmin_{m}_mmax_{m+1}/all_walkers.npy')

    shape = walkers.shape
    n_burn = int(shape[0]*0.9)
    n_sample = int(shape[1]*(shape[0]-n_burn))

    samples = walkers[n_burn:, :, :].reshape(n_sample, shape[2])
    chain_samples.append(samples)
 
    legend.append(f'{m}<log M<{m+1}')
    this_sample = getdist.MCSamples(samples=samples, names=param_names, labels=param_latex_names)
    print(f'mmin={m}', this_sample.getMeans())
    gd_samples.append(this_sample)





## Make with PyGTC
font = {'family': 'DejaVu Sans', 'size':10}
names = [f'${i}$' for i in param_latex_names]

GTC = pygtc.plotGTC(chains=chain_samples, legendMarker='All',
                    paramNames=names, customLabelFont=font, chainLabels=legend,
                    customLegendFont=font, customTickFont=font, figureSize=10)
plt.savefig(f'{save_path}/triangle_plot_GTC.pdf')

#### make triangle plot
plt.figure()
g = plots.get_subplot_plotter()
g.triangle_plot(gd_samples, axis_marker_lw=2, marker_args={'lw':2}, line_args={'lw':1}, legend_labels=legend)
plt.savefig(f'{save_path}/triangle_plot.pdf')
