"""
run_simulation.py
-----------------
Runs the power grid cascade simulation across four topologies
and three connectivity levels.

Usage:
    python run_simulation.py           # full run (~8 min)
    python run_simulation.py --fast    # quick test (~1 min)
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_model import run_experiment, DEFAULT_PARAMS

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

TOPOLOGIES = ['lattice', 'random', 'small_world', 'scale_free']
K_VALUES   = [3, 5, 8]

FULL_PARAMS = dict(n_nodes=30, n_steps=6000, burn_in=500,
                   load_mean=1.0, load_noise=0.05,
                   capacity_margin=1.4, seed=42)

FAST_PARAMS = dict(n_nodes=20, n_steps=1500, burn_in=150,
                   load_mean=1.0, load_noise=0.05,
                   capacity_margin=1.4, seed=42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true')
    args   = parser.parse_args()
    params = FAST_PARAMS if args.fast else FULL_PARAMS
    label  = 'FAST' if args.fast else 'FULL'
    print(f"=== Power Grid Cascade Simulation [{label}] ===\n")

    total, done = len(TOPOLOGIES) * len(K_VALUES), 0
    for topo in TOPOLOGIES:
        for k in K_VALUES:
            done += 1
            print(f"\n[{done}/{total}] topology={topo}  k={k}")
            run_experiment(topo, n_nodes=params['n_nodes'], k=k,
                           params=params, save_dir=DATA_DIR)

    print(f"\n=== All done. Results in {os.path.abspath(DATA_DIR)} ===")

if __name__ == '__main__':
    main()
