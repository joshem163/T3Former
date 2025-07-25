import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv,global_mean_pool,TransformerConv
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool,ARMAConv,ChebConv,TAGConv,Linear,GATv2Conv,GPSConv
from torch.nn import Linear, Embedding
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, n_layers, num_timesteps,drop_out):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_timesteps, hidden_dim))
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=n_layers, num_decoder_layers=n_layers)
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)  # Flatten the output of the transformer
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, src):
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        src_emb = src_emb.permute(1, 0, 2)  # (seq_len, batch, feature)
        transformer_output = self.transformer.encoder(src_emb)
        transformer_output = transformer_output.permute(1, 0, 2).contiguous().view(src.size(0), -1)  # Flatten
        transformer_output = self.dropout(transformer_output)
        predictions = self.fc(transformer_output)
        return predictions


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)  # FIXED: include batch
        return x


class CNNTransformer(nn.Module):
    def __init__(self, num_classes, cnn_channels=64, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(4, cnn_channels, kernel_size=3, padding=1),  # (B, 64, 20, 10)
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),  # (B, 64, 20, 10)
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU()
        )

        # Flatten spatial grid into sequence of patches
        self.flatten_patches = lambda x: x.flatten(2).transpose(1, 2)  # (B, N=200, D=64)

        # Linear projection to d_model
        self.embedding = nn.Linear(cnn_channels, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, 4, 20, 10)
        x = self.cnn(x)  # (B, 64, 20, 10)
        x = self.flatten_patches(x)  # (B, 200, 64)
        x = self.embedding(x)  # (B, 200, d_model)

        # Add CLS token
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat((cls_token, x), dim=1)  # (B, 201, d_model)

        x = self.transformer(x)  # (B, 201, d_model)
        cls_output = x[:, 0]  # (B, d_model)
        return self.classifier(cls_output)


class T3Former(nn.Module):
    def __init__(self,
                 transformer_input_dim, transformer_hidden_dim, transformer_output_dim,
                 n_heads, n_layers, num_timesteps,
                 cnn_num_classes, dropout_p, cnn_channels=64, cnn_d_model=128, cnn_nhead=4, cnn_num_layers=4,
                 final_output_dim=10):  # Added dropout_p
        super().__init__()

        # Transformer branch
        self.transformer_branch = TransformerClassifier(
            input_dim=transformer_input_dim,
            hidden_dim=transformer_hidden_dim,
            output_dim=transformer_output_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            num_timesteps=num_timesteps,
            drop_out=dropout_p
        )

        # CNN-Transformer branch
        self.cnn_transformer_branch = CNNTransformer(
            num_classes=final_output_dim,
            cnn_channels=cnn_channels,
            d_model=cnn_d_model,
            nhead=cnn_nhead,
            num_layers=cnn_num_layers
        )

        # Final classifier
        combined_feature_dim = transformer_output_dim + final_output_dim
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_final = nn.Linear(combined_feature_dim, cnn_num_classes)

    def forward(self, cnn_input, transformer_input):
        out1 = self.transformer_branch(transformer_input)  # (B, transformer_output_dim)
        out2 = self.cnn_transformer_branch(cnn_input)  # (B, cnn_num_classes)

        combined = torch.cat([out1, out2], dim=1)  # (B, transformer_output_dim + cnn_num_classes)
        combined = self.dropout(combined)
        output = self.fc_final(combined)  # (B, final_output_dim)
        return output


class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, n_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        #return attn_output.mean(dim=1)  # mean over the 3 views
        return attn_output.reshape(attn_output.size(0), -1)


class T3SAGE(nn.Module):
    def __init__(self,
                 sage_input_dim,transformer_input_dim,transformer2_input_dim, hidden_dim, output_dim,
                 n_heads, n_layers, num_timesteps1,num_timesteps2, dropout_p,
                 final_output_dim=10):  # Added dropout_p
        super().__init__()

        # Transformer branch
        self.transformer_branch = TransformerClassifier(
            input_dim=transformer_input_dim,
            hidden_dim=hidden_dim,
            output_dim=final_output_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            num_timesteps=num_timesteps1,
            drop_out=dropout_p
        )
        self.transformer_branch2 = TransformerClassifier(
            input_dim=transformer2_input_dim,
            hidden_dim=hidden_dim,
            output_dim=final_output_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            num_timesteps=num_timesteps2,
            drop_out=dropout_p
        )
        # CNN-Transformer branch
        self.sage_branch = SAGE(sage_input_dim,hidden_dim,final_output_dim,n_layers,dropout_p
        )

        # Final classifier
        combined_feature_dim = 3*final_output_dim
        self.attn_fusion = AttentionFusion(embed_dim=final_output_dim, n_heads=1)

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_final = nn.Linear(combined_feature_dim, output_dim)

    def forward(self, x_gsage, edge_index, batch, transformer_input, transformer_input1):
        out1 = self.transformer_branch(transformer_input)
        out2 = self.transformer_branch2(transformer_input1)
        out3 = self.sage_branch(x_gsage, edge_index, batch)

        stacked = torch.stack([out1, out2, out3], dim=1)  # (batch_size, 3, final_output_dim)
        attn_out = self.attn_fusion(stacked)  # (batch_size, final_output_dim)
        # print(attn_out)
        output = self.fc_final(attn_out)

        #combined = torch.cat([out1, out2, out3], dim=1)
        #combined = self.dropout(combined)
        #output = self.fc_final(combined)
        return output

import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, n_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, query, key_value,key2):
        # query: (batch, seq_len, embed_dim) from SAGE
        # key_value: (batch, seq_len, embed_dim) from Betti or DoS
        out, _ = self.attn(query, key_value, key2)
        return out
import torch
import torch.nn as nn
from torch_geometric.data import Batch

class T3GNN(nn.Module):
    def __init__(self,
                 sage_input_dim, transformer_input_dim, hidden_dim, output_dim,
                 n_heads, n_layers, num_timesteps, dropout_p,
                 final_output_dim=10):
        super().__init__()

        self.sage_branch = SAGE(sage_input_dim, hidden_dim, final_output_dim, n_layers, dropout_p)

        self.betti_proj = nn.Linear(4, final_output_dim)
        self.dos_proj = nn.Linear(4, final_output_dim)

        self.cross_attn = CrossAttention(final_output_dim, n_heads=1)

        self.transformer = TransformerClassifier(
            input_dim=final_output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            num_timesteps=num_timesteps,
            drop_out=dropout_p
        )

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, data_seq, betti_seq, dos_seq):
        """
        data_seq: list of list of Data objects, len T, each item is a list of `Data` objects for that time step
        betti_seq, dos_seq: tensors of shape (batch_size, T, 4)
        """
        graph_embeds = []
        for t in range(len(data_seq)):
            batch_t = Batch.from_data_list(data_seq[t])  # Merge list of Data objects into a Batch
            out = self.sage_branch(batch_t.x, batch_t.edge_index, batch_t.batch)  # (batch_size, final_output_dim)
            graph_embeds.append(out)

        graph_embeds = torch.stack(graph_embeds, dim=1)  # (batch_size, T, final_output_dim)

        # Project Betti and DoS vectors
        betti_embeds = self.betti_proj(betti_seq)  # (batch_size, T, final_output_dim)
        dos_embeds = self.dos_proj(dos_seq)        # (batch_size, T, final_output_dim)

        # Apply cross-attention: query = GNN, key & value = betti + dos
        fused = self.cross_attn(graph_embeds, betti_embeds, dos_embeds)  # (batch_size, T, final_output_dim)

        fused = self.dropout(fused)
        output = self.transformer(fused)

        return output


import torch
# Standard GNNs
class GNN(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.model_type = model_type

        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=2, concat=False)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=2, concat=False)
        elif model_type == 'GIN':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels))
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels))
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
        elif model_type == 'TAG':
            self.conv1 = TAGConv(in_channels, hidden_channels)
            self.conv2 = TAGConv(hidden_channels, hidden_channels)
        elif model_type == 'Cheb':
            self.conv1 = ChebConv(in_channels, hidden_channels, K=3)
            self.conv2 = ChebConv(hidden_channels, hidden_channels, K=3)
        elif model_type == 'ARMA':
            self.conv1 = ARMAConv(in_channels, hidden_channels)
            self.conv2 = ARMAConv(hidden_channels, hidden_channels)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        #node_means = x.mean(dim=1)
        #print(len(node_means))
        return self.lin(x)

# Transformer-based GNNs
class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)

class T3GT(nn.Module):
    def __init__(self,
                 sage_input_dim,transformer_input_dim,transformer2_input_dim, hidden_dim, output_dim,
                 n_heads, n_layers, num_timesteps1,num_timesteps2, dropout_p,
                 final_output_dim=10):  # Added dropout_p
        super().__init__()

        # Transformer branch
        self.transformer_branch = TransformerClassifier(
            input_dim=transformer_input_dim,
            hidden_dim=hidden_dim,
            output_dim=final_output_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            num_timesteps=num_timesteps1,
            drop_out=dropout_p
        )
        self.transformer_branch2 = TransformerClassifier(
            input_dim=transformer2_input_dim,
            hidden_dim=hidden_dim,
            output_dim=final_output_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            num_timesteps=num_timesteps2,
            drop_out=dropout_p
        )
        # CNN-Transformer branch
        self.sage_branch = GraphTransformer(sage_input_dim,hidden_dim,final_output_dim)

        # Final classifier
        combined_feature_dim = 3*final_output_dim
        self.attn_fusion = AttentionFusion(embed_dim=final_output_dim, n_heads=1)

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_final = nn.Linear(combined_feature_dim, output_dim)

    def forward(self, x_gsage, edge_index, batch, transformer_input, transformer_input1):
        out1 = self.transformer_branch(transformer_input)
        out2 = self.transformer_branch2(transformer_input1)
        out3 = self.sage_branch(x_gsage, edge_index, batch)

        stacked = torch.stack([out1, out2, out3], dim=1)  # (batch_size, 3, final_output_dim)
        attn_out = self.attn_fusion(stacked)  # (batch_size, final_output_dim)
        # print(attn_out)
        output = self.fc_final(attn_out)

        return output

class GPSModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)

        self.conv1 = GPSConv(
            channels=hidden_channels,
            conv=GCNConv(hidden_channels, hidden_channels),
            heads=2
        )
        self.conv2 = GPSConv(
            channels=hidden_channels,
            conv=GCNConv(hidden_channels, hidden_channels),
            heads=2
        )
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)



class Graphormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=2, heads=4, max_degree=10):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)

        # Structural encodings (e.g., node degree encoding as Graphormer does)
        self.degree_emb = Embedding(max_degree + 1, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, concat=True)
            )

        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

        self.classifier = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, deg=None):
        x = self.input_proj(x)

        if deg is not None:
            deg = deg.clamp(max=self.degree_emb.num_embeddings - 1)
            x = x + self.degree_emb(deg)

        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = norm(x + residual)

        x = global_mean_pool(x, batch)
        return self.classifier(x)