# This code achieves a performance of around 96.60%. However, it is not
# directly comparable to the results reported by the TGN paper since a
# slightly different evaluation setup is used here.
# In particular, predictions in the same batch are made in parallel, i.e.
# predictions for interactions later in the batch have no access to any
# information whatsoever about previous interactions in the same batch.
# On the contrary, when sampling node neighborhoods for interactions later in
# the batch, the TGN paper code has access to previous interactions in the
# batch.
# While both approaches are correct, together with the authors of the paper we
# decided to present this version here as it is more realistic and a better
# test bed for future methods.
import pickle
import os.path as osp
from tqdm import tqdm
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
# dataset = JODIEDataset(path, name='wikipedia')
# data = dataset[0]
# print(data.src)
# print(data.msg)
# print(data.t)

# For small datasets, we can put the whole dataset on GPU and thus avoid
# expensive memory transfer costs for mini-batches:
#data = data.to(device)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


memory_dim = time_dim = embedding_dim = 50




def train(loader,neighbor_loader,train_data,memory,gnn,link_pred,optimizer,assoc,criterion):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    #node_emb = None
    total_loss = 0
    node_emb_dict = {}
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        #node_emb = z
        for i, nid in enumerate(n_id):
            node_emb_dict[nid.item()] = z[i].detach().cpu()
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))
        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events
        sorted_ids = sorted(node_emb_dict.keys())
        train_node_emb = torch.stack([node_emb_dict[nid] for nid in sorted_ids])

    return total_loss / train_data.num_events,train_node_emb


@torch.no_grad()
def test(loader,neighbor_loader,memory,gnn,link_pred,assoc):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    #node_emb = None
    node_emb_dict = {}
    for batch in loader:
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        # node_emb = z
        # Store node embeddings in a dictionary
        for i, nid in enumerate(n_id):
            node_emb_dict[nid.item()] = z[i].detach().cpu()
        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
        sorted_ids = sorted(node_emb_dict.keys())
        train_node_emb = torch.stack([node_emb_dict[nid] for nid in sorted_ids])
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean()),train_node_emb

def run(data):
    train_data, val_data, test_data = data.train_val_test_split(
        val_ratio=0.15, test_ratio=0.15)

    train_loader = TemporalDataLoader(
        train_data,
        batch_size=200,
        neg_sampling_ratio=1.0,
    )
    val_loader = TemporalDataLoader(
        val_data,
        batch_size=200,
        neg_sampling_ratio=1.0,
    )
    test_loader = TemporalDataLoader(
        test_data,
        batch_size=200,
        neg_sampling_ratio=1.0,
    )
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)
    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters())
        | set(link_pred.parameters()), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
    for epoch in range(1, 10):
        best_val_ap = 0.0
        best_graph_emb = None
        #loss,-- = train(train_loader)
        #print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        loss, train_node_emb = train(train_loader,neighbor_loader,train_data,memory,gnn,link_pred,optimizer,assoc,criterion)
        #print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        val_ap, val_auc,val_node_emb = test(val_loader,neighbor_loader,memory,gnn,link_pred,assoc)
        test_ap, test_auc,test_node_emb = test(test_loader,neighbor_loader,memory,gnn,link_pred,assoc)
        #print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
        #print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_node_emb = torch.cat([train_node_emb, val_node_emb,test_node_emb], dim=0)
            # best_graph_emb = torch.sum(best_node_emb, dim=0).detach().cpu()
            best_graph_emb = torch.mean(best_node_emb, dim=0).detach().cpu()
    return best_graph_emb
if __name__ == "__main__":
    data_name='pemsbay'
    with open('temporal_data_pemsbay.pkl', 'rb') as f:
        temporalData = pickle.load(f)
    num_graphs = len(temporalData)
    all_graphs_emb = []
    for i in tqdm(range(num_graphs),):
        data = temporalData[i]
        data = data.to(device)
        graph_emb = run(data)
        all_graphs_emb.append(graph_emb.detach().cpu())
        #print(graph_emb)
        #all_graphs_emb = torch.cat((all_graphs_emb, graph_emb), dim=0)
    all_graphs_emb = torch.stack(all_graphs_emb, dim=0)
    torch.save(all_graphs_emb, "{}_graphs_emb_all_node.pt".format(data_name))

