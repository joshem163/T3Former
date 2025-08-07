import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

# === Load your data ===
graphs_emb = torch.load("pems04_graphs_emb_all_node.pt")
#graphs_emb = torch.load("pems08_graphs_emb_all_node.pt").view(744, -1)
graphs_label = torch.load("pems04_multi_labels.pt")
#data = torch.load('your_graph_embeddings.pt')  # Replace with your path
X = graphs_emb  # shape: [num_graphs, emb_dim]
y = graphs_label  # shape: [num_graphs]
print(X[0])
# Ensure everything is float for input, long for labels
X = X.float()
y = y.long()


# === Simple MLP classifier ===
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# === 5-fold cross-validation ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_list = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = MLP(input_dim=X.shape[1], hidden_dim=64, num_classes=len(torch.unique(y)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    epochs = 300
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = accuracy_score(y_test.numpy(), preds.numpy())
        acc_list.append(acc)
        print(f'Fold {fold + 1} Accuracy: {acc:.4f}')

# === Final report ===
mean_acc = np.mean(acc_list)
std_acc = np.std(acc_list)
print(f'\n5-Fold CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}')
