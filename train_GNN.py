import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool, TransformerConv

from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import argparse
import random
from sklearn.model_selection import KFold
import statistics
from modules import print_stat,stat
from model import GNN,GraphTransformer,GPSModel,Graphormer
from data_loader import load_SW_Dos_Betti_traffic


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
    parser.add_argument('--model', type=str, choices=['GCN', 'SAGE', 'GAT', 'GIN', 'all'], default='all')
    parser.add_argument('--data', type=str, default='pemsbay')
    parser.add_argument('--task', type=str, default='multi')
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args([])

    set_seed(args.seed)
    data_list, X0, X1, y0 = load_SW_Dos_Betti_traffic(args.data, task=args.task)
    num_class = len(np.unique(y0))
    print(num_class)

    models_to_run = ['SAGE','GAT', 'UniMP','graphormer'] if args.model == 'all' else [args.model]
    #models_to_run = ['GAT','graphormer'] if args.model == 'all' else [args.model]
    for model_name in models_to_run:
        print(f"\n================== Running Model: {model_name} ==================on {args.data}")
        kfold = KFold(n_splits=args.folds, shuffle=True)
        loss_per_fold = []
        acc_per_fold = []
        fold_no = 1
        for train_idx, test_idx in kfold.split(data_list):
            train_data = [data_list[i] for i in train_idx]
            test_data = [data_list[i] for i in test_idx]

            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=args.batch_size)

            # Lists to store metrics
            train_losses = []
            train_accuracies = []
            test_accuracies = []

            if model_name in ['GCN', 'SAGE', 'GAT', 'GIN', 'TAG']:
                model = GNN(model_name, in_channels=data_list[0].x.size(1),
                            hidden_channels=args.hidden_dim, num_classes=num_class).to(device)
            elif model_name == 'UniMP':
                model = GraphTransformer(in_channels=data_list[0].x.size(1),
                                         hidden_channels=args.hidden_dim, num_classes=num_class).to(device)
            elif model_name == 'graphormer':
                model = Graphormer(in_channels=data_list[0].x.size(1),
                                   hidden_channels=args.hidden_dim, num_classes=num_class).to(device)
            elif model_name == 'GPS':
                model = GPSModel(in_channels=data_list[0].x.size(1),
                                 hidden_channels=args.hidden_dim, num_classes=num_class).to(device)
            else:
                raise ValueError(f"Model '{model_name}' is not defined.")

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(1, args.epochs + 1):
                loss, train_acc = train(model, train_loader, optimizer, criterion)
                test_acc = evaluate(model, test_loader)
                train_losses.append(loss)
                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)
                print('loss=', loss)
                print('test acc=',test_acc)
            print(f'Score for fold {fold_no}: ')
            accuracy = print_stat(train_accuracies, test_accuracies)
            print("acc=", accuracy[1])
            acc_per_fold.append(accuracy[1])
            fold_no += 1
        print('Result Statistics\n')
        stat(acc_per_fold, 'accuracy')


if __name__ == "__main__":
    main()