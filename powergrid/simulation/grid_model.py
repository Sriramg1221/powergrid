"""
Cascading Failures in Power Grids
Core simulation: DC power-flow model with line overload removal.

Complexity science mapping (deliverable 4b):
  Noise        -> stochastic load fluctuations on each bus
  Avalanche    -> chain of line trips triggered by overload redistribution
  Connectivity -> mean degree <k> of the transmission network graph
"""

import numpy as np
import networkx as nx
import pickle, os

DEFAULT_PARAMS = dict(
    n_nodes      = 30,       # number of buses
    n_steps      = 6000,     # simulation time steps
    burn_in      = 500,      # discard initial transient
    load_mean    = 1.0,      # mean power demand per load bus (p.u.)
    load_noise   = 0.05,     # std of stochastic load fluctuation (p.u.)
    capacity_margin = 1.4,   # line capacity = capacity_margin * initial max flow
    seed         = 42,
)


# ── Network builders ──────────────────────────────────────────────────────────

def build_grid(topology: str, n: int = 30, k: int = 4, seed: int = 42) -> nx.Graph:
    """
    Return an undirected graph representing the transmission network.
    topology: 'random', 'small_world', 'scale_free', 'lattice'
    """
    if topology == 'lattice':
        s = int(np.ceil(np.sqrt(n)))
        G = nx.grid_2d_graph(s, s)
        G = nx.convert_node_labels_to_integers(G)
        G = G.subgraph(list(G.nodes)[:n]).copy()
    elif topology == 'random':
        m = max(n, int(n * k / 2))
        G = nx.gnm_random_graph(n, m, seed=seed)
    elif topology == 'small_world':
        G = nx.watts_strogatz_graph(n, k, 0.1, seed=seed)
    elif topology == 'scale_free':
        G = nx.barabasi_albert_graph(n, max(1, k // 2), seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    return nx.convert_node_labels_to_integers(G)


# ── DC power flow (linearised) ────────────────────────────────────────────────

def dc_power_flow(G: nx.Graph, injections: np.ndarray) -> dict:
    """
    Solve DC power flow: B * theta = P_inj.
    Returns dict of edge -> power flow (signed).

    B is the bus susceptance matrix (graph Laplacian for uniform reactances).
    One slack bus (node 0) is fixed at theta=0.
    """
    n = len(G.nodes())
    nodes = sorted(G.nodes())
    idx = {nd: i for i, nd in enumerate(nodes)}

    # Build susceptance matrix (uniform line reactance x=1 => b=1)
    B = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)

    # Remove slack bus row/col (node index 0)
    B_red = B[1:, 1:]
    P_red = injections[1:]

    # Solve for angles
    try:
        theta_red = np.linalg.lstsq(B_red, P_red, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {}

    theta = np.zeros(n)
    theta[1:] = theta_red

    # Compute line flows
    flows = {}
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        flows[(u, v)] = theta[i] - theta[j]   # flow ~ (theta_i - theta_j) / x
        flows[(v, u)] = -flows[(u, v)]

    return flows


# ── Grid simulation ───────────────────────────────────────────────────────────

class PowerGrid:
    """
    Power grid simulation with cascading line failures.

    At each step:
      1. Stochastic load fluctuation applied to all load buses.
      2. DC power flow solved on the surviving network.
      3. Lines with |flow| > capacity are tripped (removed).
      4. Steps 2-3 repeated until no more overloads (cascade).
      5. Avalanche size = number of lines tripped in one cascade.
    """

    def __init__(self, G: nx.Graph, params: dict):
        self.G_orig   = G.copy()
        self.params   = {**DEFAULT_PARAMS, **params}
        rng = np.random.default_rng(self.params['seed'])
        self.rng      = rng
        n = len(G.nodes())

        # Assign generator/load roles: ~30% generators, rest loads
        nodes = sorted(G.nodes())
        self.generators = set(nodes[:max(1, n // 3)])
        self.loads      = set(nodes[n // 3:])

        # Base injections: generators supply, loads consume
        self.base_injection = np.zeros(n)
        for nd in self.loads:
            self.base_injection[nd] = -self.params['load_mean']
        total_load = abs(self.base_injection.sum())
        for nd in self.generators:
            self.base_injection[nd] = total_load / len(self.generators)

        # Compute initial flows to set line capacities
        init_flows = dc_power_flow(G, self.base_injection)
        self.capacity = {}
        cm = self.params['capacity_margin']
        for u, v in G.edges():
            f = abs(init_flows.get((u, v), 0.01))
            self.capacity[(u, v)] = cm * max(f, 0.01)
            self.capacity[(v, u)] = self.capacity[(u, v)]

        self.avalanche_log = []
        self.step_count    = 0
        self.G_current     = G.copy()

    # ── Single simulation step ────────────────────────────────────────────────

    def step(self):
        # 1. Stochastic load fluctuation
        injection = self.base_injection.copy()
        noise_std = self.params['load_noise']
        n = len(self.G_orig.nodes())
        delta = self.rng.normal(0, noise_std, n)
        # Apply only to loads; rebalance generators
        for nd in self.loads:
            injection[nd] += delta[nd]
        load_imbalance = delta[[nd for nd in self.loads]].sum()
        for nd in self.generators:
            injection[nd] -= load_imbalance / len(self.generators)

        # 2. Cascade until no more overloads or network disconnected
        G_sim = self.G_current.copy()
        tripped_total = 0
        for _ in range(50):   # max cascade depth
            if G_sim.number_of_edges() == 0:
                break
            flows = dc_power_flow(G_sim, injection)
            overloaded = []
            for u, v in list(G_sim.edges()):
                f = abs(flows.get((u, v), 0))
                cap = self.capacity.get((u, v), 1e9)
                if f > cap:
                    overloaded.append((u, v))
            if not overloaded:
                break
            G_sim.remove_edges_from(overloaded)
            tripped_total += len(overloaded)

        self.step_count += 1

        # 3. Partial recovery: restore one random tripped line per step
        tripped_edges = set(self.G_orig.edges()) - set(G_sim.edges())
        if tripped_edges and self.rng.random() < 0.3:
            e = list(tripped_edges)[self.rng.integers(0, len(tripped_edges))]
            G_sim.add_edge(*e)

        self.G_current = G_sim

        # 4. Record avalanche
        if self.step_count > self.params['burn_in'] and tripped_total > 0:
            self.avalanche_log.append(tripped_total)

    def run(self, verbose: bool = True):
        total = self.params['n_steps']
        for t in range(total):
            self.step()
            if verbose and t % 1000 == 0:
                print(f"  step {t:5d}/{total}  | avalanches: {len(self.avalanche_log)}")
        if verbose:
            print(f"  Done. Total cascade events: {len(self.avalanche_log)}")

    @property
    def mean_degree(self):
        n = self.G_orig.number_of_nodes()
        return 2 * self.G_orig.number_of_edges() / n if n > 0 else 0


# ── Convenience runner ────────────────────────────────────────────────────────

def run_experiment(topology: str, n_nodes: int = 30, k: int = 4,
                   params: dict = None, save_dir: str = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    G = build_grid(topology, n=n_nodes, k=k, seed=p['seed'])
    print(f"[{topology}] nodes={G.number_of_nodes()}  "
          f"edges={G.number_of_edges()}  "
          f"mean_k={2*G.number_of_edges()/G.number_of_nodes():.2f}")

    sim = PowerGrid(G, p)
    sim.run(verbose=True)

    result = dict(
        topology    = topology,
        n_nodes     = G.number_of_nodes(),
        n_edges     = G.number_of_edges(),
        mean_degree = sim.mean_degree,
        avalanches  = sim.avalanche_log,
        params      = p,
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"{topology}_k{k}.pkl")
        with open(fname, 'wb') as f:
            pickle.dump(result, f)
        print(f"  Saved -> {fname}")
    return result


if __name__ == '__main__':
    r = run_experiment('random', n_nodes=20, k=4,
                       params={'n_steps': 500, 'burn_in': 50})
    print(f"Sample avalanches: {r['avalanches'][:10]}")
