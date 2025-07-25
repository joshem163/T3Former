from modules import readTUds, temporal_graph_from_TUds
import pickle
import torch
import numpy as np
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import get_laplacian
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_sparse import coalesce
from torch_scatter import scatter_add
file_path_template = "datasets/{datasetname}/{datasetname}"
def load_data_feature(data_name):
    file_path = file_path_template.format(datasetname=data_name)
    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(file_path)
    temporal_graphs = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)
    with open('betti_vectors_mp_hks_allt.pkl', 'rb') as g:
        topo_vec = pickle.load(g)
    vec = np.array(topo_vec[data_name])
    vec = torch.tensor(vec)
    X0 = vec.permute(0, 3, 1, 2)
    y0 = np.array(graphs_label)
    return X0,y0
def load_MP_Dos(data_name):
    file_path = file_path_template.format(datasetname=data_name)
    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(file_path)
    temporal_graphs = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)
    with open('betti_vectors_mp_hks (1).pkl', 'rb') as g:
        MP_hks = pickle.load(g)
    vec = np.array(MP_hks[data_name])
    vec = torch.tensor(vec)
    X0 = vec.permute(0, 3, 1, 2)
    with open('dos_vec_4_1.pkl', 'rb') as g:
        dos_vec = pickle.load(g)
    X1 = torch.tensor(np.array(dos_vec[data_name]), dtype=torch.float32)
    y0 = np.array(graphs_label)
    return X0,X1,y0

############################### temporal node features
def get_time_range(temporal_graphs):
    max_time = max(graph.t.max().item() for graph in temporal_graphs)
    return max_time

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

def load_SW_Dos_Betti_traffic(data_name,task):
    file_path_tem = "./traffic_features/{datasetname}_{feature}_{tasks}.pkl"
    file_path_betti = file_path_tem.format(datasetname=data_name,feature='betti',tasks='binary')
    file_path_dos = file_path_tem.format(datasetname=data_name, feature='dos',tasks='binary')
    file_path_sage = file_path_tem.format(datasetname=data_name, feature='sage',tasks=task)
    with open(file_path_betti, 'rb') as g:
        sw_betti = pickle.load(g)
    with open(file_path_dos, 'rb') as g:
        sw_dos = pickle.load(g)
    with open(file_path_sage, 'rb') as g:
        data_list = pickle.load(g)
    X0 = torch.tensor(np.array(sw_betti), dtype=torch.float32)
    X1 = torch.tensor(np.array(sw_dos), dtype=torch.float32)
    y0=[data_list[i].y[0] for i in range(len(data_list))]
    return data_list, X0, X1, np.array(y0)

def load_SW_Dos_Betti_traffic_normalize(data_name,task):
    file_path_tem = "./traffic_features/{datasetname}_{feature}_{tasks}.pkl"
    file_path_betti = file_path_tem.format(datasetname=data_name,feature='betti',tasks='binary')
    file_path_dos = file_path_tem.format(datasetname=data_name, feature='dos',tasks='binary')
    file_path_sage = file_path_tem.format(datasetname=data_name, feature='sage',tasks=task)
    with open(file_path_betti, 'rb') as g:
        sw_betti = pickle.load(g)
    with open(file_path_dos, 'rb') as g:
        sw_dos = pickle.load(g)
    with open(file_path_sage, 'rb') as g:
        data_list = pickle.load(g)
    scaler = MinMaxScaler()
    norm_x=[]
    for i in range(len(data_list)):
        x = data_list[i].x
        if np.isnan(x).any():
            print(f"NaN found at index {i}")
            print(x)  # optional: to see the values
        else:
            x_scaled = scaler.fit_transform(x)
            norm_x.append(x_scaled)
            data_list[i].x = x_scaled
    # for i in range(len(data_list)):
    #     x_scaled = scaler.fit_transform(data_list[i].x)
    #     norm_x.append(x_scaled)
    #     data_list[i].x=x_scaled


    X0 = torch.tensor(np.array(sw_betti), dtype=torch.float32)
    X1 = torch.tensor(np.array(sw_dos), dtype=torch.float32)
    y0=[data_list[i].y[0] for i in range(len(data_list))]
    return data_list, X0, X1, np.array(y0)

def load_SW_Dos_Betti(data_name):
    file_path = file_path_template.format(datasetname=data_name)
    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(file_path)
    temporal_graphs, temp_edge_idx = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping,
                                                              graphs_edge)
    with open('sw_betti_3_2.pkl', 'rb') as g:
        sw_betti = pickle.load(g)
    X0 = torch.tensor(np.array(sw_betti[data_name]), dtype=torch.float32)
    with open('dos_vec_3_2.pkl', 'rb') as g:
        dos_vec = pickle.load(g)
    X1 = torch.tensor(np.array(dos_vec[data_name]), dtype=torch.float32)
    y0 = np.array(graphs_label)
    data_list = []
    max_t = get_time_range(temporal_graphs)

    for i in range(num_graphs):
        G = initialize_graph(temporal_graphs[i])
        edges_data = list(G.edges(data=True))
        node_features = []
        nodes = sorted(G.nodes())
        node_id_map = {node: idx for idx, node in enumerate(nodes)}
        for node in nodes:
            degree_list = []
            for t in range(max_t):
                temp_edges = [
                    (u, v)
                    for u, v, attrs in edges_data
                    if t in attrs['time']
                ]
                temp_graph = nx.Graph()
                temp_graph.add_edges_from(temp_edges)
                degree = temp_graph.degree(node) if node in temp_graph else 0
                degree_list.append(degree)
            node_features.append(degree_list)

        # node_features = []
        src, dst = temp_edge_idx[i]
        edge_index = torch.stack([src, dst], dim=0)

        # Use each row as node feature
        x = torch.tensor(node_features, dtype=torch.float)  # Shape: [num_nodes, num_nodes]
        y = torch.tensor(graphs_label[i])
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list, X0, X1, y0


###############################
def load_SW_Dos_Betti_old(data_name):
    file_path = file_path_template.format(datasetname=data_name)
    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(file_path)
    temporal_graphs,temp_edge_idx = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)
    with open('sw_betti_3_2.pkl', 'rb') as g:
        sw_betti = pickle.load(g)
    X0 = torch.tensor(np.array(sw_betti[data_name]), dtype=torch.float32)
    with open('dos_vec_3_2.pkl', 'rb') as g:
        dos_vec = pickle.load(g)
    X1 = torch.tensor(np.array(dos_vec[data_name]), dtype=torch.float32)
    y0 = np.array(graphs_label)
    data_list = []
    k = 10  # fallback dimensionality if needed

    for i in range(num_graphs):
        src, dst = temp_edge_idx[i]
        edge_index = torch.stack([src, dst], dim=0)
        num_nodes = edge_index.max().item() + 1

        # Get Laplacian as COO
        edge_index_lap, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)

        # Reconstruct sparse Laplacian matrix
        laplacian = torch.sparse_coo_tensor(
            edge_index_lap,
            edge_weight,
            size=(num_nodes, num_nodes)
        ).to_dense()  # Convert to dense matrix to extract rows

        # Use each row as node feature
        x = laplacian  # Shape: [num_nodes, num_nodes]

        # Optional: reduce dimension (if large) using PCA or similar
        if x.shape[1] > k:
            # Simple dimensionality reduction using SVD
            U, S, V = torch.linalg.svd(x)
            x = U[:, :k] @ torch.diag(S[:k])  # shape: [num_nodes, k]

        y = torch.tensor(graphs_label[i])
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list,X0,X1,y0

def load_Sage_Dos_Betti(data_name):
    file_path = file_path_template.format(datasetname=data_name)
    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(file_path)
    temporal_graphs,temp_edge_idx = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)
    with open('sw_betti_3_2.pkl', 'rb') as g:
        sw_betti = pickle.load(g)
    X0 = torch.tensor(np.array(sw_betti[data_name]), dtype=torch.float32)
    with open('dos_vec_3_2.pkl', 'rb') as g:
        dos_vec = pickle.load(g)
    X1 = torch.tensor(np.array(dos_vec[data_name]), dtype=torch.float32)
    y0 = np.array(graphs_label)
    with open('node_features.pkl', 'rb') as g:
        temp_data = pickle.load(g)

    graph_sequence = temp_data[data_name]  # list of Data objects, one per time step
    #
    # # Initialize sequences
    # x_seq = []
    # edge_index_seq = []
    # batch_seq = []
    # adj_t_seq = []
    #
    # for graph in graph_sequence:
    #     x_seq.append(graph.x)  # (num_nodes, num_features)
    #     edge_index_seq.append(graph.edge_index)
    #     num_nodes = graph.x.size(0)
    #     adj_t = SparseTensor.from_edge_index(graph.edge_index, sparse_sizes=(num_nodes, num_nodes))
    #     adj_t_seq.append(adj_t)
    #
    #     # Assuming each graph is a single component (no batches within time step)
    #     batch_seq.append(torch.zeros(num_nodes, dtype=torch.long))  # (num_nodes,)

    #return x_seq,edge_index_seq,adj_t_seq,X0,X1,y0
    return graph_sequence,X0,X1,y0
from NeuroGraph.datasets import NeuroGraphDynamic
feature_path_betti = "neuro_fe/betti_{datasetname}.data"
feature_path_dos = "neuro_fe/dos_{datasetname}.data"
def load_neuro_Dos_Betti(data_name):
    dataset = NeuroGraphDynamic(root="data/", name=data_name)#"DynHCPGender"
    Braindata = dataset.dataset
    merged_data = []
    label = []
    for graph in Braindata:
        label.append(graph.y[0])
        merged_data.append(Data(x=graph.x,edge_index=graph.edge_index,y=graph.y[0]))
    with open(feature_path_betti.format(datasetname=data_name), 'rb') as g:
        betti_vec = pickle.load(g)
    with open(feature_path_dos.format(datasetname=data_name), 'rb') as g:
        dos_vec = pickle.load(g)
    X0 = torch.tensor(np.array(betti_vec), dtype=torch.float32)
    X1 = torch.tensor(np.array(dos_vec), dtype=torch.float32)

    y0 = torch.tensor(dataset.labels)
    return merged_data, X0, X1, y0

def temporal_degree(temp_graph,max_time):
    src=temp_graph.src
    dst=temp_graph.dst
    time=temp_graph.t
    # Determine number of nodes and total time steps
    num_nodes = int(torch.max(torch.cat([src, dst]))) + 1
    #num_time_steps = int(torch.max(time)) + 1
    num_time_steps = max_time
    # Initialize node features [num_nodes, num_time_steps]
    node_features = torch.zeros((num_nodes, num_time_steps), dtype=torch.float)
    # For each edge, mark the time of appearance for src and dst nodes
    for s, d, t in zip(src, dst, time):
        node_features[s, t] = 1.0
        node_features[d, t] = 1.0
    #node_features = torch.clamp(node_features, max=1.0)
    return node_features

def load_temp_Dos_Betti(data_name):
    file_path = file_path_template.format(datasetname=data_name)
    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(file_path)
    temporal_graphs, temp_edge_idx = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping,
                                                              graphs_edge)
    with open('sw_betti_3_2.pkl', 'rb') as g:
        sw_betti = pickle.load(g)
    X0 = torch.tensor(np.array(sw_betti[data_name]), dtype=torch.float32)
    with open('dos_vec_3_2.pkl', 'rb') as g:
        dos_vec = pickle.load(g)
    X1 = torch.tensor(np.array(dos_vec[data_name]), dtype=torch.float32)
    y0 = np.array(graphs_label)
    data_list = []
    max_t = get_time_range(temporal_graphs)
    merged_data = []
    i = 0
    for graph in temporal_graphs:
        tem_deg = temporal_degree(graph, max_t + 1)  # tep_degree feaatures
        edge_index = torch.stack([graph.src, graph.dst], dim=0)
        merged_data.append(Data(x=tem_deg, edge_index=edge_index, y=graphs_label[i]))
        i += 1

    return merged_data, X0, X1, y0