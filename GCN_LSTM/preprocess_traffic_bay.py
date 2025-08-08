import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
import pandas as pd
import torch
from torch_geometric.data import TemporalData

data = np.load('pemsbay.npz')
x_all = data['data']  # shape: (16992, 307, 3)

with open("adj_mx_bay.pkl", "rb") as f:
    adjlist = pickle.load(f, encoding='latin1')

adj = adjlist[-1]

# Load distance.csv
# dist_df = pd.read_csv('distance04.csv')
num_nodes = x_all.shape[1]
timesteps_per_day = 24
num_days = x_all.shape[0] // timesteps_per_day
num_features = x_all.shape[2]

# Build adjacency matrix
# adj = np.zeros((num_nodes, num_nodes))
# for _, row in dist_df.iterrows():
#     i, j, cost = int(row['from']), int(row['to']), float(row['cost'])
#     if cost > 0:
#         adj[i, j] = 1 / cost  # inverse distance
# make symmetric

# Extract edges (src, dst)
src_all, dst_all = adj.nonzero()
edge_pairs = list(zip(src_all, dst_all))  # static edges

# --- 2. Build Per-Day Temporal Graphs ---
temporal_graphs = []
mean_link_speeds = []

for day in tqdm(range(num_days)):
    # Slice out 24 timesteps
    day_x = x_all[day * timesteps_per_day : (day + 1) * timesteps_per_day]  # shape: (24, 307, 3)

    # Transpose to shape (307, 288, 3)
    node_time_series = np.transpose(day_x, (1, 0, 2))  # [nodes, timesteps, features] = [307, 24, 3]

    # 2.1 Compute label: mean link speed (based only on speed feature)
    avg_speed_per_node = node_time_series[:, :, 0].mean(axis=1)  # (307,)
    edge_speeds = []
    for i, j in edge_pairs:
        speed_i = avg_speed_per_node[i] if not np.isnan(avg_speed_per_node[i]) else 0
        speed_j = avg_speed_per_node[j] if not np.isnan(avg_speed_per_node[j]) else 0
        edge_speeds.append((speed_i + speed_j) / 2)

    mean_link_speeds.append(np.mean(edge_speeds))

    # 2.2 Generate temporal edges (src, dst, t)
    src_list = []
    dst_list = []
    t_list = []
    x_list = []

    for t in range(timesteps_per_day):
        for i, j in edge_pairs:
            src_list.append(i)
            dst_list.append(j)
            t_list.append(t)
            x_list.append((node_time_series[i][t] + node_time_series[j][t]) / 2)

    # 2.3 Node features: full time-series with all features
    x_feat = torch.tensor(node_time_series, dtype=torch.float)  # shape: [307, 24, 3]

    # Build TemporalData
    td = TemporalData(
        src=torch.tensor(src_list, dtype=torch.long),
        dst=torch.tensor(dst_list, dtype=torch.long),
        t=torch.tensor(t_list, dtype=torch.long),
        msg=torch.tensor(np.array(x_feat),dtype=torch.float32)
    )
    temporal_graphs.append(td)

from torch_geometric.data import Data

def temporal_graph_to_data_list(temporal_graph,y):
    """
    Convert TemporalData object to list of Data objects, one per time step.
    """
    x_all = temporal_graph.msg  # shape: [N, T, F]
    src = temporal_graph.src
    dst = temporal_graph.dst
    t_all = temporal_graph.t  # edge time stamps, optional

    N, T, F = x_all.shape
    edge_index = torch.stack([src, dst], dim=0)

    data_list = []

    for t in range(T):
        x_t = x_all[:, t, :]  # shape: [N, F]

        # Optionally filter edge_index if edges change over time:
        t_mask = (t_all == t)
        edge_index_t = torch.stack([src[t_mask], dst[t_mask]], dim=0)
        # Use same edge_index if edges are static:
        data = Data(x=x_t, edge_index=edge_index_t,y=y)
        data_list.append(data)

    return data_list



# --- 3. Generate Binary Labels ---
mean_link_speeds = np.array(mean_link_speeds)
threshold = np.percentile(mean_link_speeds, 35)
labels_binary = [0 if m < threshold else 1 for m in mean_link_speeds]

torch.save(torch.tensor(labels_binary, dtype=torch.float32), 'pemsbay_binary_labels.pt')

# --- 3. Generate Multi Labels ---

# Define thresholds at 35% and 60%
low_thresh = np.percentile(mean_link_speeds, 35)
high_thresh = np.percentile(mean_link_speeds, 60)

# Assign class labels
labels_multi = []
for speed in mean_link_speeds:
    if speed < low_thresh:
        labels_multi.append(0)
    elif speed < high_thresh:
        labels_multi.append(1)
    else:
        labels_multi.append(2)

torch.save(torch.tensor(labels_multi, dtype=torch.float32), 'pemsbay_multi_labels.pt')

# Output:
# temporal_graphs → list of TemporalData objects (one per two hours)
# labels → list of binary labels

all_temporal_graphs_as_lists = [temporal_graph_to_data_list(g, y) for g, y in zip(temporal_graphs, labels_multi)]
with open('temporal_multi_pemsbay.pkl', 'wb') as f:
    pickle.dump(all_temporal_graphs_as_lists, f)


print(f"Generated {len(temporal_graphs)} temporal graphs.")
# print(f"First graph x.shape: {temporal_graphs[0].x.shape}")  # should be [307, 24, 3]
# print(f"Labels: {labels[:5]}")