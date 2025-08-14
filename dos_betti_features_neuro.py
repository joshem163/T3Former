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
import NeuroGraph

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


def dos_extraction(data):
    """

    """
    ptr = data.ptr
    edge_index = data.edge_index

    results = []
    for j in range(len(ptr) - 1):  # For each time snapshot in this sample
        node_start, node_end = ptr[j].item(), ptr[j + 1].item()
        # Mask edges belonging to this snapshot
        mask = ((edge_index[0] >= node_start) & (edge_index[0] < node_end))
        edges = edge_index[:, mask] - node_start  # Relabel to local indices

        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(node_end - node_start))
        G.add_edges_from(edges.t().tolist())
        results.append(compute_dos(G))
    return results

def dos_vectors(dataset):
    """

    """
    dos = []
    for i in tqdm(dataset):
        dos.append(dos_extraction(i))
    return dos

def betti_extraction(data):
    """

    """
    ptr = data.ptr
    edge_index = data.edge_index

    results = []
    for j in range(len(ptr) - 1):  # For each time snapshot in this sample
        node_start, node_end = ptr[j].item(), ptr[j + 1].item()
        # Mask edges belonging to this snapshot
        mask = ((edge_index[0] >= node_start) & (edge_index[0] < node_end))
        edges = edge_index[:, mask] - node_start  # Relabel to local indices

        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(node_end - node_start))
        G.add_edges_from(edges.t().tolist())
        adj_matrix = nx.to_numpy_array(G)

        # Compute homology
        homology = pyflagser.flagser_unweighted(adj_matrix, min_dimension=0, max_dimension=2, directed=False, coeff=2)
        b0, b1 = homology['betti'][0], homology['betti'][1]
        results.append((b0, b1, G.number_of_nodes(), G.number_of_edges()))
    return results

def betti_vectors(dataset):
    """

    """
    betti = []
    for i in tqdm(dataset):
        betti.append(betti_extraction(i))
    return betti
