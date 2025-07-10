# kuramoto_network_metrics.py
"""Utility functions to extract network‑science descriptors that are
known to govern Kuramoto synchronization dynamics.

All functions expect a **square (N×N) numpy array ``K``** whose (i,j)
entry is the coupling strength K_{ij}.  By default we treat edges as
undirected and positive (symmetric K); set ``directed=True`` to respect
orientation.

Dependencies
------------
* numpy
* networkx
* scipy (only `scipy.linalg.eigvals`)
* python‑louvain (`pip install python-louvain`)
"""
from __future__ import annotations
import warnings
from typing import Dict, Tuple, Union

import numpy as np
import networkx as nx
from scipy.linalg import eigvals  # dense eigensolver is OK up to ~5e3 nodes
import community as community_louvain  # Louvain algorithm for modularity

__all__ = [
    "graph_from_K",
    "spectral_metrics",
    "strength_metrics",
    "kcore_metrics",
    "community_metrics",
    "path_metrics",
    "betweenness_metrics",
    "compute_all_metrics",
]

################################################################################
# Helper: build NetworkX graph from coupling matrix
################################################################################

def graph_from_K(K: np.ndarray, *, directed: bool = False, threshold: float | None = None) -> Union[nx.Graph, nx.DiGraph]:
    """Create a `networkx` graph from coupling matrix ``K``.

    Parameters
    ----------
    K : np.ndarray (N×N)
        Coupling matrix.
    directed : bool, default False
        Treat K as asymmetric and build a DiGraph.  If False, we symmetrize
        by W = 0.5*(K + K.T).
    threshold : float | None, optional
        Edges with |K_ij| ≤ threshold are discarded.  `None` keeps all edges.
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")

    n = K.shape[0]
    if directed:
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                w = K[i, j]
                if threshold is None or abs(w) > threshold:
                    G.add_edge(i, j, weight=float(w))
    else:
        # symmetrize and keep upper triangle to avoid double edges
        W = 0.5 * (K + K.T)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        iu, ju = np.triu_indices(n, k=1)
        for i, j, w in zip(iu, ju, W[iu, ju]):
            if threshold is None or abs(w) > threshold:
                G.add_edge(i, j, weight=float(w))
    return G

################################################################################
# 1. Spectral metrics (Laplacian eigenstructure)
################################################################################

def _laplacian_spectrum(G: nx.Graph) -> np.ndarray:
    """Return Laplacian eigenvalues in ascending order (dense solver)."""
    W = nx.to_numpy_array(G, weight="weight")
    deg = np.sum(W, axis=1)
    L = np.diag(deg) - W
    return np.sort(eigvals(L).real)

def spectral_metrics(K: np.ndarray, *, directed: bool = False, threshold: float | None = None) -> Dict[str, float]:
    """Compute λ₂, λ_N and eigenratio λ_N/λ₂ of the (undirected) Laplacian.

    If `directed=True`, we first symmetrize K (Kuramoto MSF theory assumes
    an undirected, diffusive Laplacian).
    """
    G = graph_from_K(K, directed=False if not directed else False, threshold=threshold)
    lambdas = _laplacian_spectrum(G)
    # exclude the zero eigenvalue (numerical tolerance)
    lam2 = lambdas[1] if lambdas.size > 1 else 0.0
    lamN = lambdas[-1]
    return {
        "lambda_2": float(lam2),
        "lambda_N": float(lamN),
        "eigenratio": float(lamN / lam2) if lam2 > 0 else np.inf,
    }

################################################################################
# 2. Strength / degree distribution
################################################################################

def strength_metrics(K: np.ndarray, *, directed: bool = False) -> Dict[str, np.ndarray]:
    """Return in/out (or total) strength arrays.

    Strength s_i = Σ_j K_ij (out‑strength).  For undirected, in=out=total.
    """
    if directed:
        out_strength = K.sum(axis=1)
        in_strength = K.sum(axis=0)
        return {"in_strength": in_strength, "out_strength": out_strength}
    else:
        strength = K.sum(axis=1)
        return {"strength": strength}

################################################################################
# 3. k‑core decomposition
################################################################################

def kcore_metrics(K: np.ndarray, *, directed: bool = False) -> Dict[str, np.ndarray]:
    """Return coreness (k‑shell index) for each node.

    NetworkX implements k‑core for weighted graphs by ignoring weights; this
    is usually sufficient for predicting lock‑in cascade order.
    """
    G = graph_from_K(K, directed=directed)
    if directed:
        # Convert to undirected projection for core analysis
        G_u = G.to_undirected()
    else:
        G_u = G
    core_dict = nx.core_number(G_u)
    coreness = np.array([core_dict[v] for v in sorted(core_dict)])
    return {"coreness": coreness}

################################################################################
# 4. Community / modularity analysis
################################################################################

def community_metrics(K: np.ndarray, *, directed: bool = False, resolution: float = 1.0) -> Dict[str, Union[float, Dict[int, int]]]:
    """Run Louvain community detection and return partition + modularity Q."""
    G = graph_from_K(K, directed=directed)
    if directed:
        # Louvain expects undirected; use symmetrized graph
        G = G.to_undirected()
    part = community_louvain.best_partition(G, weight="weight", resolution=resolution)
    Q = community_louvain.modularity(part, G, weight="weight")
    return {"partition": part, "modularity": float(Q)}

################################################################################
# 5. Path‑length–based metrics
################################################################################

def _distance_graph(G: nx.Graph) -> nx.Graph:
    """Return a copy where edge attribute 'distance' = 1/|weight|."""
    H = G.copy()
    for u, v, d in H.edges(data=True):
        w = abs(d.get("weight", 0.0))
        d["distance"] = 1.0 / w if w > 0 else np.inf
    return H

def path_metrics(K: np.ndarray, *, directed: bool = False) -> Dict[str, float]:
    """Average shortest‑path length and diameter (largest geodesic)."""
    G = graph_from_K(K, directed=directed)
    if directed:
        G = G.to_undirected()
    H = _distance_graph(G)
    # Largest connected component only
    if not nx.is_connected(H):
        H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
        warnings.warn("Graph is disconnected; metrics computed on largest component.")
    avg_len = nx.average_shortest_path_length(H, weight="distance")
    diam = nx.diameter(H, e=None, weight="distance")
    return {"avg_shortest_path": float(avg_len), "diameter": float(diam)}

################################################################################
# 6. Betweenness‑based load measures
################################################################################

def betweenness_metrics(K: np.ndarray, *, directed: bool = False, normalized: bool = True) -> Dict[str, Dict[int, float]]:
    """Edge and node betweenness centrality calculated on distance graph."""
    G = graph_from_K(K, directed=directed)
    if directed:
        G = G.to_undirected()
    H = _distance_graph(G)
    node_bet = nx.betweenness_centrality(H, weight="distance", normalized=normalized)
    edge_bet = nx.edge_betweenness_centrality(H, weight="distance", normalized=normalized)
    return {"node_betweenness": node_bet, "edge_betweenness": edge_bet}

################################################################################
# Convenience: compute everything at once
################################################################################

def compute_all_metrics(K: np.ndarray, *, directed: bool = False, threshold: float | None = None) -> Dict[str, Dict]:
    """Return a nested dict with all descriptors."""
    return {
        "spectral": spectral_metrics(K, directed=directed, threshold=threshold),
        "strength": strength_metrics(K, directed=directed),
        "kcore": kcore_metrics(K, directed=directed),
        "community": community_metrics(K, directed=directed),
        "path": path_metrics(K, directed=directed),
        "betweenness": betweenness_metrics(K, directed=directed),
    }

################################################################################
# Example usage (will not execute on import)
################################################################################
if __name__ == "__main__":
    N = 10
    rng = np.random.default_rng(42)
    K = rng.random((N, N))
    K = 0.5 * (K + K.T)  # make symmetric
    metrics = compute_all_metrics(K)
    from pprint import pprint
    pprint(metrics)
