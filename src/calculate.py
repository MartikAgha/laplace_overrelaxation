import os
import yaml
import argparse

import numpy as np

from experiments.coaxial_results import iter_vs_ppcm, optimize_tolerance,\
    iter_vs_width, optimize_mesh_size, optimum_relaxation, profile_ensemble, \
    method_comparison, tube_variation

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list-calculations', action='store_true')


def main():
    with open('default_experiment_params.yaml', 'r') as f:
        params = yaml.load(f, yaml.FullLoader)
    parser = get_argparser()
    args = parser.parse_args()

