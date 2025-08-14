import numpy as np
import networkx as nx
from scipy.linalg import eigh
from scipy.stats import gaussian_kde
from modules import *
import networkx as nx
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
import pyflagser

def initialize_graph(tempG):
    """Efficiently initializes a graph from a temporal graph object."""
    G = nx.Graph()

    # Use defaultdict for efficient edge time storage
    edge_time_map = defaultdict(list)
    for u, v, t in zip(tempG.edge_index[0].tolist(), tempG.edge_index[1].tolist(), tempG.t.tolist()):
        edge_time_map[(u, v)].append(t)

    # Add edges with their corresponding timestamps
    for (u, v), times in edge_time_map.items():
        G.add_edge(u, v, time=sorted(times))  # Ensure sorted timestamps

    return G

def compute_dos(graph, num_bins=4, bandwidth=0.05):
    if graph.number_of_nodes() == 0:
        return np.zeros(num_bins)  # Handle empty graph gracefully

    L = nx.normalized_laplacian_matrix(graph).toarray()
    eigenvalues = eigh(L, eigvals_only=True)

    kde = gaussian_kde(eigenvalues, bw_method=bandwidth)
    x_min, x_max = min(eigenvalues), max(eigenvalues)
    bin_centers = np.linspace(x_min, x_max, num_bins)
    dos_values = kde(bin_centers)

    return dos_values

def sliding_window_dos(G, thresholds, window=3, jump=1, num_bins=4):
    edges_data = list(G.edges(data=True))  # (u, v, {'time': [t_start, t_end]})
    results = []

    num_windows = (len(thresholds) - window) // jump + 1

    for i in range(0, num_windows * jump, jump):
        t_start = thresholds[i]
        t_end = thresholds[i + window - 1]

        # Filter edges with min activation time in [t_start, t_end]
        active_edges = [
            (u, v)
            for u, v, attrs in edges_data
            if any(t_start <= t <= t_end for t in attrs['time'])
        ]

        subgraph = nx.Graph()
        subgraph.add_edges_from(active_edges)

        dos_vector = compute_dos(subgraph, num_bins=num_bins)
        results.append(dos_vector)

    return np.stack(results)

def dos_vectors(data_dir, data_name):
    """Computes Betti number sequences for multiple temporal graphs in parallel."""
    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(data_dir)
    temporal_graphs = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)
    all_timestamps = torch.cat([graph.t for graph in temporal_graphs]).unique().tolist()
    t2 = np.quantile(sorted(all_timestamps), np.linspace(0, 1, 20))
    v1 = []
    for i in tqdm(temporal_graphs):
      v1.append(sliding_window_dos(initialize_graph(i), t2, window = 4, jump=1))

    return v1
