import yaml
import argparse

from experiments.coaxial_results import iter_vs_ppcm, optimize_tolerance,\
    iter_vs_width, optimize_mesh_size, optimum_relaxation, profile_ensemble, \
    method_comparison, tube_variation


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--calculation', type=str, default=None)
    parser.add_argument('-l', '--list-calculations', action='store_true')
    return parser

def perform_calculation(calculation_type, pars):
    if calculation_type == 'iter_vs_ppcm':
        iter_vs_ppcm(max_mesh_density=pars['max_mesh_density'],
                     width=pars['width'],
                     wbar=pars['wbar'],
                     potential=pars['potential'])
    elif calculation_type == 'iter_vs_width':
        iter_vs_width(min_width=pars['min_width'],
                      max_width=pars['max_width'],
                      wbar=pars['wbar'],
                      ppcm=pars['ppcm'],
                      potential=pars['potential'])
    elif calculation_type == 'optimize_tolerance':
        optimize_tolerance(width=pars['width'],
                           ppcm=pars['ppcm'],
                           precision=pars['precision'],
                           lowest_tol_power=pars['lowest_tol_power'],
                           plot=pars['plot'])
    elif calculation_type == 'optimum_relaxation':
        optimum_relaxation(width=pars['width'])
    elif calculation_type == 'optimize_mesh_size':
        optimize_mesh_size(width=pars['width'],
                           precision=pars['precision'],
                           max_mesh_density=pars['max_mesh_density'])
    elif calculation_type == 'profile_ensemble':
        profile_ensemble(sizes=pars['sizes'])
    elif calculation_type == 'method_comparison':
        method_comparison(max_mesh_density=pars['max_mesh_density'],
                          width=pars['width'],
                          wbar=pars['wbar'],
                          potential=pars['potential'])
    elif calculation_type == 'tube_variation':
        tube_variation(ppcm=pars['ppcm'])
    else:
        raise ValueError("No calculation suggested.")

def main():
    with open('default_experiment_params.yaml', 'r') as f:
        params = yaml.load(f, yaml.FullLoader)
    parser = get_argparser()
    args = parser.parse_args()
    if args.list_calculations:
        print("Possible calculations: ")
        for key in params.keys():
            print('\t', key)
    else:
        pars = params[args.calculation]
        perform_calculation(args.calculation, pars)

if __name__ == '__main__':
    main()