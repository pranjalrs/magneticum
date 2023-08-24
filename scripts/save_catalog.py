import argparse

import sys
sys.path.append('../core/')

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--box')
parser.add_argument('--sim')
parser.add_argument('--snap')
parser.add_argument('--mmin', default=1e12, type=float)

args = parser.parse_args()
box, sim , snap = args.box, args.sim, args.snap
mmin = args.mmin

# group_base = f'/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/{box}/{sim}/groups_{snap}'
group_base = f'/xdisk/timeifler/pranjalrs/Magneticum/{box}/{sim}/groups_{snap}'

utils.get_halo_catalog(group_base, ['GPOS', 'MVIR', 'RVIR', 'M5CC', 'R5CC', 'MCRI', 'RCRI', 'TGAS'], mmin)
