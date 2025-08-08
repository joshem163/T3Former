import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
import pandas as pd
import torch
from torch_geometric.data import TemporalData

data = np.load('pems08.npz')
x_all = data['data']  # shape: (16992, 307, 3)

# Load distance.csv
dist_df = pd.read_csv('distance08.csv')
num_nodes = x_all.shape[1]
timesteps_per_day = 24
num_days = x_all.shape[0] // timesteps_per_day
num_features = x_all.shape[2]

# Build adjacency matrix
adj = np.zeros((num_nodes, num_nodes))
for _, row in dist_df.iterrows():
    i, j, cost = int(row['from']), int(row['to']), float(row['cost'])
    if cost > 0:
        adj[i, j] = 1 / cost  # inverse distance
adj = np.maximum(adj, adj.T)  # make symmetric

# Extract edges (src, dst)
src_all, dst_all = adj.nonzero()
edge_pairs = list(zip(src_all, dst_all))  # static edges

# --- 2. Build Per-Day Temporal Graphs ---
temporal_graphs = []
mean_link_speeds = []

for day in range(num_days):
    # Slice out 24 timesteps
    day_x = x_all[day * timesteps_per_day: (day + 1) * timesteps_per_day]  # shape: (24, 307, 3)

    # Transpose to shape (307, 288, 3)
    node_time_series = np.transpose(day_x, (1, 0, 2))  # [nodes, timesteps, features] = [307, 24, 3]

    # 2.1 Compute label: mean link speed (based only on speed feature)
    avg_speed_per_node = node_time_series[:, :, 2].mean(axis=1)  # (307,)
    edge_speeds = [(avg_speed_per_node[i] + avg_speed_per_node[j]) / 2 for i, j in edge_pairs]
    mean_link_speeds.append(np.mean(edge_speeds))

    # 2.2 Generate temporal edges (src, dst, t)
    src_list = []
    dst_list = []
    t_list = []

    for t in range(timesteps_per_day):
        for i, j in edge_pairs:
            src_list.append(i)
            dst_list.append(j)
            t_list.append(t)

    # 2.3 Node features: full time-series with all features
    x_feat = torch.tensor(node_time_series, dtype=torch.float)  # shape: [307, 24, 3]

    # Build TemporalData
    td = TemporalData(
        src=torch.tensor(src_list, dtype=torch.long),
        dst=torch.tensor(dst_list, dtype=torch.long),
        t=torch.tensor(t_list, dtype=torch.long),
        x=x_feat  # shape: [307, 24, 3]
    )
    temporal_graphs.append(td)

with open('temporal_data_pems08.pkl', 'wb') as f:
    pickle.dump(temporal_graphs, f)

# --- 3. Generate Binary Labels ---
mean_link_speeds = np.array(mean_link_speeds)
threshold = np.percentile(mean_link_speeds, 35)
labels_binary = [0 if m < threshold else 1 for m in mean_link_speeds]

torch.save(torch.tensor(labels_binary, dtype=torch.float32), 'pems08_binary_labels.pt')

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

torch.save(torch.tensor(labels_multi, dtype=torch.float32), 'pems08_multi_labels.pt')

# Output:
# temporal_graphs → list of TemporalData objects (one per two hours)
# labels → list of binary labels

print(f"Generated {len(temporal_graphs)} temporal graphs.")
# print(f"First graph x.shape: {temporal_graphs[0].x.shape}")  # should be [307, 24, 3]
# print(f"Labels: {labels[:5]}")