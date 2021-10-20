import yaml
import argparse

from experiments.coaxial_calculator import perform_calculation

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--calculation', type=str, default=None)
    parser.add_argument('-l', '--list-calculations', action='store_true')
    return parser

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