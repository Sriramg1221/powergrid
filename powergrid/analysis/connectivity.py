"""
connectivity.py
---------------
Produces:
  figures/load_curve.pdf          – grid stress vs cascade probability
  figures/cascade_timeseries.pdf  – blackout size over time
  figures/network_comparison.pdf  – four topology visualisations
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulation'))
from grid_model import PowerGrid, build_grid, DEFAULT_PARAMS

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {'lattice':'#185FA5','random':'#0F6E56',
          'small_world':'#BA7517','scale_free':'#A32D2D'}


def plot_load_curve():
    """Cascade probability vs. capacity margin (stress level)."""
    margins = np.linspace(1.05, 2.0, 16)
    cascade_probs = []
    params = {**DEFAULT_PARAMS, 'n_nodes':20, 'n_steps':800,
              'burn_in':100, 'load_noise':0.05}
    G = build_grid('random', n=20, k=4, seed=42)
    for cm in margins:
        p = {**params, 'capacity_margin': float(cm)}
        sim = PowerGrid(G, p)
        sim.run(verbose=False)
        avl = [a for a in sim.avalanche_log if a > 0]
        prob = len(avl) / max(params['n_steps'] - params['burn_in'], 1)
        cascade_probs.append(prob)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(margins, cascade_probs, 'o-', ms=5, color='#185FA5', lw=1.5)
    ax.axvline(margins[np.argmax(np.gradient(cascade_probs))],
               ls='--', color='#A32D2D', lw=1,
               label='steepest transition')
    ax.set_xlabel('Capacity margin $\\alpha$ (line capacity / initial flow)', fontsize=11)
    ax.set_ylabel('Cascade event rate (events/step)', fontsize=11)
    ax.set_title('Grid stress vs. cascade frequency', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', lw=0.4, alpha=0.6)
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, 'load_curve.pdf')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"  Saved {fname}"); plt.close(fig)


def plot_timeseries():
    """Cascade size time series for two topologies."""
    params = {**DEFAULT_PARAMS, 'n_nodes':25, 'n_steps':1200,
              'burn_in':200, 'load_noise':0.06}
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for ax, topo in zip(axes, ['lattice', 'scale_free']):
        G   = build_grid(topo, n=params['n_nodes'], k=4, seed=42)
        sim = PowerGrid(G, params)
        sizes = []
        for _ in range(params['n_steps']):
            prev = len(sim.avalanche_log)
            sim.step()
            if len(sim.avalanche_log) > prev:
                sizes.append(sim.avalanche_log[-1])
            else:
                sizes.append(0)
        t = np.arange(len(sizes))
        ax.fill_between(t, sizes, alpha=0.35, color=COLORS[topo])
        ax.plot(t, sizes, lw=0.6, color=COLORS[topo])
        ax.set_ylabel('Blackout size', fontsize=9)
        ax.set_title(topo.replace('_',' ').title(), fontsize=10)
        ax.grid(True, ls=':', lw=0.4, alpha=0.5)
    axes[-1].set_xlabel('Time step', fontsize=11)
    plt.suptitle('Cascade size time-series: intermittency across scales',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, 'cascade_timeseries.pdf')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"  Saved {fname}"); plt.close(fig)


def plot_network_comparison():
    """Draw all four topologies side-by-side."""
    topos = ['lattice','random','small_world','scale_free']
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, topo in zip(axes, topos):
        G = build_grid(topo, n=20, k=4, seed=42)
        if topo == 'scale_free':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        degs = dict(G.degree())
        sizes = [15 + 9*degs[n] for n in G.nodes()]
        nx.draw_networkx(G, pos=pos, ax=ax,
                         node_size=sizes,
                         node_color=COLORS[topo],
                         edge_color='#B4B2A9',
                         width=0.7, alpha=0.88,
                         with_labels=False)
        km = 2*G.number_of_edges()/G.number_of_nodes()
        ax.set_title(f"{topo.replace('_',' ')}\n$\\langle k\\rangle={km:.1f}$",
                     fontsize=9)
        ax.axis('off')
    plt.suptitle('Transmission network topologies', fontsize=11)
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, 'network_comparison.pdf')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"  Saved {fname}"); plt.close(fig)


if __name__ == '__main__':
    print("Plotting load curve..."); plot_load_curve()
    print("Plotting time series..."); plot_timeseries()
    print("Plotting network comparison..."); plot_network_comparison()
    print("\nDone.")
