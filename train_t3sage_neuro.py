import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Batch
from logger import *
import argparse
import sys
from model import CNNTransformer,T3Former,T3SAGE
from modules import *
from data_loader import load_neuro_Dos_Betti
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")
# Argument parsing
#sys.argv = [sys.argv[0]]
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--folds', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--head', type=int, default=2)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--output_dim', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.0)
args = parser.parse_args()
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
print(device)


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for xb, xb1, graph_batch, yb in loader:
        graph_batch, xb, xb1, yb = graph_batch.to(device), xb.to(device), xb1.to(device), yb.to(device)
        optimizer.zero_grad()
        output = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, xb, xb1)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for xb, xb1, graph_batch, yb in loader:
        graph_batch, xb, xb1, yb = graph_batch.to(device), xb.to(device), xb1.to(device), yb.to(device)
        output = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, xb, xb1)
        pred = output.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    accuracy = correct / total
    return accuracy

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
    #data_names = ["DynHCPAge","infectious_ct1", "dblp_ct1", "facebook_ct1", "tumblr_ct1","mit_ct1","highschool_ct1"]
    data_names = ["DynHCPAge","DynHCPGender","DynHCPActivity"]
    # Define hyperparameter search space
    learning_rates = [0.01]
    hidden_dims = [16]
    dropout_rates = [0.5, 0.0]
    # learning_rates = [0.01, 0.005, 0.001]
    # hidden_dims = [16, 32, 64, 128]
    # dropout_rates = [0.5, 0.3, 0.0]
    for dataset_name in data_names:
        print(f'\n=== Running on dataset: {dataset_name} ===')
        #data_list,X0,X1,y0= load_SW_Dos_Betti(dataset_name)
        data_list, X0, X1, y0 = load_neuro_Dos_Betti(dataset_name)

        sage_input_dim=data_list[0].x.shape[-1]
        num_samples = len(X1)
        num_timesteps = len(X0[0])
        num_timesteps2 = len(X1[0])
        num_features = len(X0[0][0])
        num_features2 = len(X1[0][0])

        print(len(X0[0]))
        print(len(X1[0]))

        X = torch.tensor(X0, dtype=torch.float32)
        X1 = torch.tensor(X1, dtype=torch.float32)
        y = torch.tensor(y0, dtype=torch.long)
        logger_acc = Logger(args.runs)
        num_classes = len(torch.unique(y))



        best_val_acc = 0
        best_hyperparams = {}

        # Grid search
        for lr in learning_rates:
            for hidden_dim in hidden_dims:
                for dropout in dropout_rates:
                    print(f"\n🔎 Trying: LR={lr}, Hidden={hidden_dim}, Dropout={dropout}")
                    logger_acc = Logger(args.runs)

                    acc_per_fold = []

                    for run in range(args.runs):
                        indices = np.arange(len(data_list))
                        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=123)
                        val_idx, test_idx = train_test_split(temp_idx, test_size=0.33, random_state=123)

                        train_data = CustomGraphDataset(X[train_idx], X1[train_idx], [data_list[i] for i in train_idx],
                                                        y[train_idx])
                        val_data = CustomGraphDataset(X[val_idx], X1[val_idx], [data_list[i] for i in val_idx],
                                                      y[val_idx])
                        test_data = CustomGraphDataset(X[test_idx], X1[test_idx], [data_list[i] for i in test_idx],
                                                      y[test_idx])

                        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=custom_collate)
                        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=custom_collate)
                        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=custom_collate)

                        model = T3SAGE(sage_input_dim=sage_input_dim,
                                       transformer_input_dim=num_features,
                                       transformer2_input_dim=num_features2,
                                       hidden_dim=hidden_dim,
                                       output_dim=num_classes,
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

                        #for epoch in tqdm(range(1, args.epochs + 1), desc=f"Epochs (Fold {fold})"):
                        for epoch in range(1, args.epochs + 1):
                            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
                            val_acc = evaluate(model, val_loader, device)
                            test_acc = evaluate(model, test_loader, device)  # Optional: evaluate test every epoch

                            train_losses.append(train_loss)
                            train_accuracies.append(train_acc)
                            test_accuracies.append(val_acc)
                            logger_acc.add_result(run, (train_acc, val_acc, test_acc))
                            #tqdm.write(
                             #   f"Epoch {epoch}:, Train loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

                        accuracy=logger_acc.print_statistics(run)
                        #accuracy = print_stat(train_accuracies, test_accuracies)
                        acc_per_fold.append(accuracy)
                    print(acc_per_fold)
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