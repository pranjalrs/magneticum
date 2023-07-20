import argparse

import sys
sys.path.append('../core/')

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--box')
parser.add_argument('--sim')
parser.add_argument('--snap')

args = parser.parse_args()
box, sim , snap = args.box, args.sim, args.snap

# group_base = f'/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/{box}/{sim}/groups_{snap}'
group_base = f'/xdisk/timeifler/pranjalrs/magneticum_data/{box}/{sim}/groups_{snap}'

utils.get_halo_catalog(group_base, ['GPOS', 'MVIR', 'RVIR', 'M5CC', 'R5CC', 'MCRI', 'RCRI', 'TGAS'])
