import numpy as np
import torch
import torch.nn as nn

import torch.nn as nn

from torch.nn import Linear, Embedding
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv, TAGConv, ChebConv, ARMAConv,
    TransformerConv, GPSConv, global_mean_pool
)
def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

# class GraphBlock(nn.Module):
#     def __init__(self, input_dim, hidden_dim, model_type='GCN'):
#         super(GraphBlock, self).__init__()
#         self.model_type = model_type
#         if model_type == 'GIN':
#             nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
#             self.gnn = GINConv(nn1)
#         else:
#             self.gnn = GCNConv(input_dim, hidden_dim)
#
#     def forward(self, x, edge_index):
#         return self.gnn(x, edge_index)
    from torch_geometric.nn import GATConv

class GraphBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, model_type):
        super(GraphBlock, self).__init__()
        self.model_type = model_type

        if model_type == 'GIN':
            nn1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gnn = GINConv(nn1)

        elif model_type == 'GAT':
            self.gat1 = GATConv(input_dim, hidden_dim, heads=2, concat=True)
            self.gat2 = GATConv(2*hidden_dim, hidden_dim, heads=2, concat=True)
            self.relu = nn.ReLU()

        else:
            self.gcn1 = GCNConv(input_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
            self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        if self.model_type == 'GIN':
            return self.gnn(x, edge_index)
        elif self.model_type == 'GAT':
            x = self.gat1(x, edge_index)
            x = self.relu(x)
            x = self.gat2(x, edge_index)
            return x
        else:
            x = self.gcn1(x, edge_index)
            x = self.relu(x)
            x = self.gcn2(x, edge_index)
            return x

class TGATClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_layers=1, dropout=0.0, model_type='GAT'):
        super(TGATClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.model_type = model_type

        self.graph_block = GraphBlock(input_dim, hidden_dim, model_type)
        self.rnn = nn.GRU(2*hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        if self.graph_block.model_type == 'GIN':
            self.graph_block.gnn.reset_parameters()
        elif self.graph_block.model_type == 'GAT':
            self.graph_block.gat1.reset_parameters()
            self.graph_block.gat2.reset_parameters()
        else:
            self.graph_block.gcn1.reset_parameters()
            self.graph_block.gcn2.reset_parameters()

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x_seq, edge_index):
        batch_size, T, N, F = x_seq.shape
        outputs = []
        for t in range(T):
            xt = x_seq[:, t, :, :]
            xt_out = []
            for b in range(batch_size):
                x = xt[b]  # [N, F]
                x_gnn = self.graph_block(x, edge_index[t])  # [N, hidden]
                x_pool = torch.mean(x_gnn, dim=0)  # mean pooling per graph
                xt_out.append(x_pool)
            xt_out = torch.stack(xt_out)  # [B, hidden]
            outputs.append(xt_out)

        seq = torch.stack(outputs, dim=1)  # [B, T, hidden]
        rnn_out, _ = self.rnn(seq)  # [B, T, hidden]
        final_out = rnn_out.mean(dim=1)  # [B, hidden]
        final_out = self.dropout(final_out)
        return self.classifier(final_out)



class TGCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_layers=1, dropout=0.0, model_type='GCN'):
        super(TGCNClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.model_type = model_type

        self.graph_block = GraphBlock(input_dim, hidden_dim, model_type)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        if self.graph_block.model_type == 'GIN':
            self.graph_block.gnn.reset_parameters()
        else:
            self.graph_block.gcn1.reset_parameters()
            self.graph_block.gcn2.reset_parameters()

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    # def reset_parameters(self):
    #     self.graph_block.gnn.reset_parameters()
    #     for name, param in self.lstm.named_parameters():
    #         if 'weight' in name:
    #             nn.init.xavier_uniform_(param)
    #     nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x_seq, edge_index):
        # x_seq: [batch, T, N, F]
        batch_size, T, N, F = x_seq.shape
        # print(batch_size,T,N,F)
        # x_seq = x_seq.view(-1, N, F)  # [B*T, N, F]
        # x_seq = x_seq.transpose(0, 1)  # [N, B*T, F]
        #x_seq = x_seq.view(batch_size, T, N, F)

        outputs = []
        for t in range(T):
            #print(f"x_seq[t].shape = {x_seq[t].shape}, trying to reshape to ({batch_size}, {N}, {F})")
            # xt = x_seq[t].reshape(batch_size, N, F)  # [B, N, F]
            xt = x_seq[:, t, :, :]
            xt_out = []
            for b in range(batch_size):
                x = xt[b]  # [N, F]
                x_gnn = self.graph_block(x, edge_index[t])  # [N, hidden]
                x_pool = torch.mean(x_gnn, dim=0)  # mean pooling per graph
                xt_out.append(x_pool)
            xt_out = torch.stack(xt_out)  # [B, hidden]
            outputs.append(xt_out)

        seq = torch.stack(outputs, dim=1)  # [B, T, hidden]
        lstm_out, _ = self.lstm(seq)  # [B, T, hidden]
        #final_out = lstm_out[:, -1, :]  # [B, hidden]
        #final_out = lstm_out.reshape(batch_size, -1)
        final_out = lstm_out.mean(dim=1)
        # print(final_out)
        final_out = self.dropout(final_out)
        return self.classifier(final_out)

import torch
import torch.nn as nn

class TGINClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_layers=1, dropout=0.0, model_type='GIN'):
        super(TGINClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.model_type = model_type

        self.graph_block = GraphBlock(input_dim, hidden_dim, model_type)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        if self.graph_block.model_type == 'GIN':
            self.graph_block.gnn.reset_parameters()
        else:
            self.graph_block.gcn1.reset_parameters()
            self.graph_block.gcn2.reset_parameters()

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x_seq, edge_index):
        # x_seq: [batch, T, N, F]
        batch_size, T, N, F = x_seq.shape

        outputs = []
        for t in range(T):
            xt = x_seq[:, t, :, :]
            xt_out = []
            for b in range(batch_size):
                x = xt[b]  # [N, F]
                x_gnn = self.graph_block(x, edge_index[t])  # [N, hidden]
                x_pool = torch.mean(x_gnn, dim=0)  # mean pooling per graph
                xt_out.append(x_pool)
            xt_out = torch.stack(xt_out)  # [B, hidden]
            outputs.append(xt_out)

        seq = torch.stack(outputs, dim=1)  # [B, T, hidden]
        rnn_out, _ = self.rnn(seq)  # [B, T, hidden]
        final_out = rnn_out.mean(dim=1)  # [B, hidden]
        final_out = self.dropout(final_out)
        return self.classifier(final_out)
