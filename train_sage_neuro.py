import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import argparse
import random
from NeuroGraph.datasets import NeuroGraphDynamic
from torch_geometric.data import Data
from logger import *
from model import GNN,GraphTransformer

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Training function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        correct += (out.argmax(dim=1) == data.y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# Evaluation function with AUC
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1)
        labels = data.y.cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels)
        correct += (preds == data.y).sum().item()
    # auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    acc = correct / len(loader.dataset)
    return acc


def main():
    parser = argparse.ArgumentParser(description="GNN Model Trainer with Train/Val/Test Split")
    parser.add_argument('--model', type=str, choices=['GCN', 'SAGE', 'GAT', 'GIN', 'all'], default='GT')
    parser.add_argument('--data', type=str, default='DynHCPGender') #DynHCPActivity,DynHCPAge
    parser.add_argument('--hidden_dim', type=int, default=34)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset = NeuroGraphDynamic(root="data/", name=args.data)
    Braindata = dataset.dataset
    merged_data = []
    label=[]
    for graph in Braindata:
        label.append(graph.y[0])
        merged_data.append(Data(
            x=graph.x,
            edge_index=graph.edge_index,
            y=graph.y[0],  # graph-level label
        ))
    num_class = len(np.unique(label))
    data_list = merged_data

    print(f"\n================== Running on {args.data}")
    logger_acc = Logger(args.runs)
    logger_auc = Logger(args.runs)

    for run in range(args.runs):
        indices = np.arange(len(data_list))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=run)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.67, random_state=run)
        print(len(train_idx))
        print(len(val_idx))
        print(len(test_idx))

        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        test_data = [data_list[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        # for data in train_loader:
        #     print(data)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)


        # model = GNN(args.model, in_channels=data_list[0].x.size(1),
        #                 hidden_channels=args.hidden_dim, num_classes=num_class).to(device)
        model = GraphTransformer(in_channels=data_list[0].x.size(1),
                        hidden_channels=args.hidden_dim, num_classes=num_class).to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, args.epochs + 1):
            loss, train_acc = train(model, train_loader, optimizer, criterion)
            val_acc = evaluate(model, val_loader)
            test_acc = evaluate(model, test_loader)
            logger_acc.add_result(run, (train_acc, val_acc, test_acc))
            print(
                f"Epoch {epoch + 1:02d}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
        print("\nAccuracy results:")
        logger_acc.print_statistics(run)

    print("\n Final Accuracy results:")
    logger_acc.print_statistics()


if __name__ == "__main__":
    main()