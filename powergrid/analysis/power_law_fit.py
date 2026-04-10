"""
power_law_fit.py
----------------
MLE power-law fitting for blackout size distributions.
Produces log-log CCDF plots and tau vs connectivity figure.
"""

import os, sys, glob, pickle, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {'lattice':'#185FA5','random':'#0F6E56',
          'small_world':'#BA7517','scale_free':'#A32D2D'}


def mle_tau(data, s_min=2):
    x = np.array(data, dtype=float)
    x = x[x >= s_min]
    if len(x) < 10: return np.nan
    return 1.0 + len(x) / np.sum(np.log(x / (s_min - 0.5)))

def ks_stat(data, tau, s_min=2):
    x = np.sort(np.array(data, dtype=float))
    x = x[x >= s_min]
    if len(x) == 0: return np.nan
    ecdf = np.arange(1, len(x)+1) / len(x)
    tcdf = np.clip(1 - (x/s_min)**(1-tau), 0, 1)
    return float(np.max(np.abs(ecdf - tcdf)))

def ccdf(data):
    data = np.sort(np.array(data, dtype=float))
    n = len(data)
    u = np.unique(data)
    p = np.array([(data >= s).sum()/n for s in u])
    return u, p

def load_results():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.pkl')))
    results = []
    for f in files:
        with open(f,'rb') as fh:
            results.append(pickle.load(fh))
    results.sort(key=lambda r:(r['topology'], r['mean_degree']))
    return results


def plot_freq_distributions(results):
    topos = list(dict.fromkeys(r['topology'] for r in results))
    for topo in topos:
        tr = [r for r in results if r['topology']==topo]
        fig, ax = plt.subplots(figsize=(5,4))
        for i, res in enumerate(sorted(tr, key=lambda r:r['mean_degree'])):
            avl = np.array(res['avalanches'])
            avl = avl[avl > 0]
            if len(avl) < 10: continue
            s_vals, p_vals = ccdf(avl)
            col = plt.cm.Blues(0.35 + 0.3*i)
            ax.scatter(s_vals, p_vals, s=7, alpha=0.7, color=col,
                       label=f"$\\langle k\\rangle={res['mean_degree']:.1f}$")
            tau = mle_tau(avl)
            if not np.isnan(tau) and len(s_vals) > 1:
                s_fit = np.logspace(np.log10(max(s_vals.min(),1)),
                                    np.log10(s_vals.max()), 80)
                norm  = p_vals[0] * s_vals[0]**(tau-1)
                ax.plot(s_fit, norm*s_fit**(-(tau-1)), '--', lw=1.2,
                        color=COLORS[topo],
                        label=f'$\\tau={tau:.2f}$' if i==len(tr)-1 else None)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Blackout size $s$ (lines tripped)', fontsize=11)
        ax.set_ylabel('$P(S \\geq s)$', fontsize=11)
        ax.set_title(f'{topo.replace("_"," ").title()} network', fontsize=11)
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.5)
        plt.tight_layout()
        fname = os.path.join(FIGURES_DIR, f'freq_dist_{topo}.pdf')
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        print(f"  Saved {fname}"); plt.close(fig)


def plot_tau_vs_k(results):
    topos = list(dict.fromkeys(r['topology'] for r in results))
    fig, ax = plt.subplots(figsize=(5.5,4))
    for topo in topos:
        tr = sorted([r for r in results if r['topology']==topo],
                    key=lambda r:r['mean_degree'])
        ks, taus = [], []
        for res in tr:
            avl = np.array(res['avalanches'])
            avl = avl[avl > 0]
            t = mle_tau(avl)
            if not np.isnan(t):
                ks.append(res['mean_degree']); taus.append(t)
        if ks:
            ax.plot(ks, taus, 'o-', lw=1.5, ms=6,
                    color=COLORS[topo], label=topo.replace('_',' '))
    ax.set_xlabel('Mean degree $\\langle k\\rangle$ (connectivity)', fontsize=11)
    ax.set_ylabel('Power-law exponent $\\tau$', fontsize=11)
    ax.set_title('Blackout exponent vs. grid connectivity', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, ls=':', lw=0.5, alpha=0.6)
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, 'exponent_vs_connectivity.pdf')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"  Saved {fname}"); plt.close(fig)


def save_csv(results):
    fname = os.path.join(DATA_DIR, 'fit_results.csv')
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['topology','n_nodes','n_edges','mean_degree',
                    'n_cascades','mean_size','max_size','tau','ks'])
        for res in results:
            avl = np.array(res['avalanches']); avl = avl[avl>0]
            tau = mle_tau(avl)
            ks  = ks_stat(avl, tau) if not np.isnan(tau) else np.nan
            w.writerow([res['topology'], res['n_nodes'], res['n_edges'],
                        f"{res['mean_degree']:.2f}", len(avl),
                        f"{avl.mean():.2f}" if len(avl) else 'N/A',
                        int(avl.max()) if len(avl) else 'N/A',
                        f"{tau:.3f}" if not np.isnan(tau) else 'N/A',
                        f"{ks:.3f}"  if not np.isnan(ks)  else 'N/A'])
    print(f"  Saved {fname}")


if __name__ == '__main__':
    print("Loading results...")
    results = load_results()
    print(f"  {len(results)} files found.\n")
    print("Plotting frequency distributions...")
    plot_freq_distributions(results)
    print("\nPlotting tau vs connectivity...")
    plot_tau_vs_k(results)
    print("\nSaving CSV...")
    save_csv(results)
    print("\nDone.")
