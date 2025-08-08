import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle

from models import TGCNClassifier  # Adjust if your path differs

def train(model, dataset, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for sample in dataset:
        graph_sequence = sample
        label = sample[-1].y

        T = len(graph_sequence)
        N, F = graph_sequence[0].x.shape

        edge_index_seq = [data.edge_index.to(device) for data in graph_sequence]
        x_seq = torch.stack([data.x.to(device) for data in graph_sequence])  # [T, N, F]
        x_seq = x_seq.unsqueeze(0)  # [1, T, N, F]
        y = torch.tensor([label], dtype=torch.long).to(device)

        optimizer.zero_grad()
        output = model(x_seq, edge_index_seq)  # [1, output_dim]
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss

def evaluate(model, dataset, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in dataset:
            graph_sequence = sample
            edge_index_seq = [data.edge_index.to(device) for data in graph_sequence]
            x_seq = torch.stack([data.x.to(device) for data in graph_sequence])  # [T, N, F]
            x_seq = x_seq.unsqueeze(0).to(device)
            y = torch.tensor([sample[-1].y], dtype=torch.long).to(device)

            output = model(x_seq, edge_index_seq)
            _, preds = torch.max(output, 1)
            correct += (preds == y).sum().item()
            total += 1

    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='pems04')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--model_type', type=str, choices=['GCN', 'GIN'], default='GCN')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for bus_name in ['pems04']:
        print(f"\n====== Dataset: {bus_name} ======")
        with open('temporal_list_pems04.pkl', 'rb') as f:
            dataset = pickle.load(f)

        graphs_label = torch.load("pems04_binary_labels.pt")
        num_classes = len(np.unique(graphs_label))
        print(f"Num classes: {num_classes}")
        input_dim = len(dataset[0][0].x[0])
        num_timesteps = len(dataset[0])
        print(f"Num timesteps: {num_timesteps}")

        #kf = KFold(n_splits=5, shuffle=True, random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accuracies = []

        #for fold_id, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        for fold_id, (train_idx, test_idx) in enumerate(skf.split(dataset, graphs_label)):
            print(f"\n--- Fold {fold_id + 1} ---")
            train_data = [dataset[i] for i in train_idx]
            test_data = [dataset[i] for i in test_idx]
            test_label = [dataset[i][-1].y for i in test_idx]
            print(test_label)

            model = TGCNClassifier(
                input_dim=input_dim,
                hidden_dim=args.hidden_channels,
                output_dim=num_classes,
                num_timesteps=num_timesteps,
                num_layers=args.num_layers,
                dropout=args.dropout,
                model_type=args.model_type
            ).to(device)

            if hasattr(model, 'reset_parameters'):
                model.reset_parameters()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

            for epoch in tqdm(range(1, args.epochs + 1), desc=f"Fold {fold_id+1} Epoch"):
                loss = train(model, train_data, criterion, optimizer, device)
                train_acc = evaluate(model, train_data, device)
                acc = evaluate(model, test_data, device)

                print(f"Epoch {epoch} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {acc:.4f}")

            fold_accuracies.append(acc)
            print(f"Fold {fold_id + 1} Accuracy: {acc:.4f}")

        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        print(f"\n==> Final 5-Fold Accuracy for {bus_name}: {mean_acc:.4f} ± {std_acc:.4f}")

if __name__ == "__main__":
    main()
