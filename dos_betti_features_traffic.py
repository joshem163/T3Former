import networkx as nx
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
import pyflagser
from modules import *
from scipy.linalg import eigh
from scipy.stats import gaussian_kde

def initialize_graph_with_node_activity(tempG, speed_threshold):
    """
    Initializes a NetworkX graph where node attributes store the list of
    timestamps at which the node's speed exceeds the given threshold.

    Args:
        tempG (TemporalData): A temporal graph (from PyG)
        speed_threshold (float): Threshold for determining active timestamps

    Returns:
        G (nx.Graph): Graph with nodes having a 'time' attribute
    """
    G = nx.Graph()

    # Add edges (same as before — from edge_index)
    edge_index = tempG.edge_index
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    G.add_edges_from(edges)

    # Node features: [num_nodes, 288, 3]
    speeds = tempG.x[:, :, 2]  # Extract speed feature (0-th index)

    # For each node, find timestamps where speed > threshold
    for node_id in range(speeds.shape[0]):
        high_speed_times = (speeds[node_id] > speed_threshold).nonzero(as_tuple=False).squeeze().tolist()
        if isinstance(high_speed_times, int):
            high_speed_times = [high_speed_times]
        G.nodes[node_id]['time'] = sorted(high_speed_times)

    return G

def sliding_window_betti_node_filtered(G, thresholds, window=3, jump=1):
    node_time_data = dict(G.nodes(data='time'))  # node_id → list of active times
    results = []

    num_windows = (len(thresholds) - window) // jump + 1

    for i in range(0, num_windows * jump, jump):
        t_start = thresholds[i]
        t_end = thresholds[i + window - 1]

        # Filter nodes with activity within [t_start, t_end]
        active_nodes = {
            n for n, times in node_time_data.items()
            if any(t_start <= t <= t_end for t in times)
        }

        # Induced subgraph on active nodes
        subgraph = G.subgraph(active_nodes).copy()

        # Skip empty subgraphs
        if subgraph.number_of_nodes() == 0:
            results.append((0, 0, 0, 0))
            continue

        # Convert to adjacency matrix
        adj_matrix = nx.to_numpy_array(subgraph)
        homology = pyflagser.flagser_unweighted(
            adj_matrix, min_dimension=0, max_dimension=2, directed=False, coeff=2
        )
        b0, b1 = homology['betti'][0], homology['betti'][1]
        results.append((b0, b1, subgraph.number_of_nodes(), subgraph.number_of_edges()))

    return np.stack(results)

def betti_vectors_node_filtered(temporal_graphs, speed_threshold):
    """Computes Betti vectors for multiple temporal graphs, filtering by node activity."""
    from tqdm import tqdm

    # Gather all timestamps from edge index (they're consistent across graphs)
    all_timestamps = torch.cat([graph.t for graph in temporal_graphs]).unique().tolist()
    thresholds = np.quantile(sorted(all_timestamps), np.linspace(0, 1, 20))

    v1 = []
    for g in tqdm(temporal_graphs):
        G = initialize_graph_with_node_activity(g, speed_threshold)
        v1.append(sliding_window_betti_node_filtered(G, thresholds, window=3, jump=2))

    return v1

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

def sliding_window_dos_node_filtered(G, thresholds, window=3, jump=1):
    node_time_data = dict(G.nodes(data='time'))  # node_id → list of active times
    results = []

    num_windows = (len(thresholds) - window) // jump + 1

    for i in range(0, num_windows * jump, jump):
        t_start = thresholds[i]
        t_end = thresholds[i + window - 1]

        # Filter nodes with activity within [t_start, t_end]
        active_nodes = {
            n for n, times in node_time_data.items()
            if any(t_start <= t <= t_end for t in times)
        }

        # Induced subgraph on active nodes
        subgraph = G.subgraph(active_nodes).copy()

        # Skip empty subgraphs
        if subgraph.number_of_nodes() == 0:
            results.append((0, 0, 0, 0))
            continue

        # Convert to adjacency matrix
        dos_vector = compute_dos(subgraph)
        results.append(dos_vector)

    return np.stack(results)

def dos_vectors_node_filtered(temporal_graphs, speed_threshold):
    """Computes Betti vectors for multiple temporal graphs, filtering by node activity."""
    from tqdm import tqdm

    # Gather all timestamps from edge index (they're consistent across graphs)
    all_timestamps = torch.cat([graph.t for graph in temporal_graphs]).unique().tolist()
    thresholds = np.quantile(sorted(all_timestamps), np.linspace(0, 1, 20))

    v1 = []
    for g in tqdm(temporal_graphs):
        G = initialize_graph_with_node_activity(g, speed_threshold)
        v1.append(sliding_window_dos_node_filtered(G, thresholds, window=3, jump=2))

    return v1tr
