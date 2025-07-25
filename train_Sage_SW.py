import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import statistics
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Batch
import argparse
import sys
from model import CNNTransformer,T3Former,T3SAGE
from modules import *
from data_loader import load_MP_Dos,load_SW_Dos_Betti,load_SW_Dos_Betti_old,load_SW_Dos_Betti_traffic
import warnings
warnings.filterwarnings("ignore")
# Argument parsing
#sys.argv = [sys.argv[0]]
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--data', type=str, default='pems08')
parser.add_argument('--task', type=str, default='binary')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--folds', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--head', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--output_dim', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.0)
args = parser.parse_args()
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
print(device)

from torch.utils.data import Dataset
class CustomGraphDataset(Dataset):
    def __init__(self, X, X1, graphs, y):
        self.X = X
        self.X1 = X1
        self.graphs = graphs
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.X1[idx], self.graphs[idx], self.y[idx]

def custom_collate(batch):
    X_batch, X1_batch, graph_batch, y_batch = zip(*batch)
    return (
        torch.stack(X_batch),
        torch.stack(X1_batch),
        Batch.from_data_list(graph_batch),
        torch.stack(y_batch)
    )


def main():
    #data_names = ["infectious_ct1", "dblp_ct1", "facebook_ct1", "tumblr_ct1","mit_ct1","highschool_ct1"]
    data_names =['pemsbay']
    # Define hyperparameter search space
    learning_rates = [0.001]
    hidden_dims = [16]
    dropout_rates = [0.0]
    # learning_rates = [0.01, 0.005, 0.001]
    # hidden_dims = [16, 32, 64, 128]
    # dropout_rates = [0.5, 0.3, 0.0]
    for dataset_name in data_names:
        print(f'\n=== Running on dataset: {dataset_name} ===')
        #data_list,X0,X1,y0= load_SW_Dos_Betti(dataset_name)
        data_list, X0, X1, y0 = load_SW_Dos_Betti_traffic(dataset_name,task=args.task)
        X0 = torch.cat((X0[:841], X0[842:]), dim=0)
        X1 = torch.cat((X1[:841], X1[842:]), dim=0)
        print(np.unique(y0))
        num_class=len(np.unique(y0))

        sage_input_dim=data_list[0].x.shape[-1]
        num_samples = len(X1)
        num_timesteps = len(X0[0])
        num_timesteps2 = len(X1[0])
        num_features = len(X1[0][0])
        num_features2 = len(X1[0][0])

        print(X0.shape)
        print(len(data_list))
        print(len(X1[0]))

        X = torch.tensor(X0, dtype=torch.float32)
        X1 = torch.tensor(X1, dtype=torch.float32)
        y = torch.tensor(y0, dtype=torch.long)

        num_classes = len(torch.unique(y))
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)


        best_val_acc = 0
        best_hyperparams = {}

        # Grid search
        for lr in learning_rates:
            for hidden_dim in hidden_dims:
                for dropout in dropout_rates:
                    print(f"\n🔎 Trying: LR={lr}, Hidden={hidden_dim}, Dropout={dropout}")

                    acc_per_fold = []

                    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                        # X_train, X_val = X[train_idx], X[val_idx]
                        # X1_train, X1_val = X1[train_idx], X1[val_idx]
                        # y_train, y_val = y[train_idx], y[val_idx]
                        # train_graphs = [data_list[i] for i in train_idx]
                        # val_graphs = [data_list[i] for i in val_idx]

                        train_data = CustomGraphDataset(X[train_idx], X1[train_idx], [data_list[i] for i in train_idx],
                                                        y[train_idx])
                        val_data = CustomGraphDataset(X[val_idx], X1[val_idx], [data_list[i] for i in val_idx],
                                                      y[val_idx])

                        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=custom_collate)
                        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=custom_collate)

                        model = T3SAGE(sage_input_dim=sage_input_dim,
                                       transformer_input_dim=num_features,
                                       transformer2_input_dim=num_features2,
                                       hidden_dim=hidden_dim,
                                       output_dim=num_class,
                                       n_heads=args.head,
                                       n_layers=args.num_layers,
                                       num_timesteps1=num_timesteps,
                                       num_timesteps2=num_timesteps2,
                                       dropout_p=dropout
                        ).to(device)

                        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=5e-4)
                        criterion = nn.CrossEntropyLoss()

                        # Lists to store metrics
                        train_losses = []
                        train_accuracies = []
                        test_accuracies = []

                        for epoch in tqdm(range(1, args.epochs + 1), desc=f"Epochs (Fold {fold})"):
                        #for epoch in range(1, args.epochs + 1):
                            # Train
                            model.train()
                            correct_train = 0
                            total_train = 0
                            epoch_train_loss = 0
                            for xb, xb1, graph_batch, yb in train_loader:  # <-- Unpack X, X1, y
                                graph_batch,xb, xb1, yb = graph_batch.to(device),xb.to(device), xb1.to(device), yb.to(device)
                                optimizer.zero_grad()
                                output = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, xb, xb1) # <-- Pass both X and X1
                                loss = criterion(output, yb)
                                loss.backward()
                                optimizer.step()

                                epoch_train_loss += loss.item()

                                pred = output.argmax(dim=1)
                                correct_train += (pred == yb).sum().item()
                                total_train += yb.size(0)

                            avg_train_loss = epoch_train_loss / len(train_loader)
                            train_acc = correct_train / total_train
                            train_losses.append(avg_train_loss)
                            train_accuracies.append(train_acc)

                            # Validation
                            model.eval()
                            correct_val = 0
                            total_val = 0
                            with torch.no_grad():
                                for xb, xb1,graph_batch, yb in val_loader:  # <-- Again unpack all 3
                                    graph_batch,xb, xb1, yb =graph_batch.to(device), xb.to(device), xb1.to(device), yb.to(device)
                                    output = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, xb, xb1)
                                    pred = output.argmax(dim=1)
                                    correct_val += (pred == yb).sum().item()
                                    total_val += yb.size(0)

                            val_acc = correct_val / total_val
                            test_accuracies.append(val_acc)
                            tqdm.write(
                                f"Epoch {epoch}:, Train loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

                        accuracy = print_stat(train_accuracies, test_accuracies)
                        print("acc=",accuracy[1])
                        acc_per_fold.append(accuracy[1])
                    avg_val_acc = np.mean(acc_per_fold)
                    std=np.std(acc_per_fold)
                    print(f"✅ Finished: Avg Val Acc = {avg_val_acc:.4f}")

                    if avg_val_acc > best_val_acc:
                        best_val_acc = avg_val_acc
                        std_val=std
                        best_hyperparams = {
                            'learning_rate': lr,
                            'hidden_dim': hidden_dim,
                            'dropout': dropout
                        }
        print(f'\n=== 🏆 Best Results for {dataset_name} ===')
        print(f"Best Hyperparameters: {best_hyperparams}")
        print(f"Best Average Validation Accuracy: {best_val_acc:.4f}± {std_val:.4f}")
if __name__ == "__main__":
    main()