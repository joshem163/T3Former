import torch
from torch_geometric.data import TemporalData
from torch_geometric.nn.models.tgn import LastNeighborLoader
from torch_geometric.loader import TemporalDataLoader
import pandas as pd
import numpy as np
import statistics
# import torch_scatter
from torch_geometric.nn.models.tgn import LastNeighborLoader
#from sklearn.model_selection import GridSearchCV
# from networkx.classes.graph import Graph
import networkx
def readTUds(folder_name):
    """
        return temporal graphs
    """
    num_graphs = 0
    graphs_label = []

    label_file = open(folder_name + "_graph_labels.txt")
    for line in label_file.readlines():
        num_graphs += 1
        graphs_label.append(int(line))

    graphs_node = [[] for _ in range(num_graphs)]
    node_mapping = {}
    indicator = open(folder_name + "_graph_indicator.txt")
    for idx, line in enumerate(indicator.readlines()):
        graph_id = int(line)
        graphs_node[graph_id - 1].append(idx + 1)
        if (idx + 1) not in node_mapping:
            node_mapping[idx + 1] = graph_id

    graphs_edge = [[] for _ in range(num_graphs)]
    check_edge = {}
    A = open(folder_name + "_A.txt")
    edge_t = open(folder_name + "_edge_attributes.txt")
    for line, t in zip(A.readlines(), edge_t.readlines()):
        u, v = line.split(", ")
        u, v = int(u), int(v)
        t = float(t)
        if (t, v, u) not in check_edge:
            graph_id = node_mapping[u]
            graphs_edge[graph_id - 1].append([t, u, v])
            check_edge[(t, v, u)] = 1
            check_edge[(t, u, v)] = 1

    return num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge


def temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge):
    for graph_id in range(num_graphs):
        graphs_edge[graph_id].sort()

    temporal_graphs = []
    temporal_edge = []

    for graph_id in range(num_graphs):
        src, dst, t_s = [], [], []
        for t, u, v in graphs_edge[graph_id]:
            src.append(u)
            dst.append(v)
            t_s.append(t)
        #         src = torch.tensor(src, dtype=torch.long)
        #         dst = torch.tensor(dst, dtype=torch.long)

        #         edge_index = torch.stack([src, dst], dim=0)  # shape: [2, num_edges]
        #         temporal_edge.append(edge_index)

        src = torch.Tensor(src).to(torch.long)
        dst = torch.Tensor(dst).to(torch.long)
        temporal_edge.append([src, dst])

        t_s = torch.Tensor(t_s).to(torch.long)

        min_id = torch.min(src.min(), dst.min())
        src -= min_id
        dst -= min_id

        temporal_graph = TemporalData(src=src, dst=dst, t=t_s)
        temporal_graphs.append(temporal_graph)

    return temporal_graphs, temporal_edge


def stat(acc_list, metric):
    mean = statistics.mean(acc_list)
    stdev = statistics.stdev(acc_list)
    print('Final', metric, f'using 5 fold CV: {mean:.4f} \u00B1 {stdev:.4f}%')


def print_stat(train_acc, test_acc):
    argmax = np.argmax(train_acc)
    best_result = test_acc[argmax]
    train_ac = np.max(train_acc)
    test_ac = np.max(test_acc)
    #print(f'Train accuracy = {train_ac:.4f}%,Test Accuracy = {test_ac:.4f}%\n')
    return test_ac, best_result
