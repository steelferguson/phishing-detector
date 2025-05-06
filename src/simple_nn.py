import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Binary classification 
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Binary classification to single logit
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  

def train_binary_classifier(
    X_train, y_train,
    reweight=False,
    batch_size=128,
    epochs=10,
    lr=1e-3,
    device='cpu',
    plot_loss=True,
    loss_plot_path="data/outputs/nn_training_loss.png"
):
    
    X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    if reweight:
        weights = torch.where(y_tensor == 1, 5.0, 1.0)
    else:
        weights = torch.ones_like(y_tensor)

    dataset = TensorDataset(X_tensor, y_tensor, weights)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    model = SimpleNN(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='none')

    model.train()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch, w_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            loss_per_sample = criterion(probs, y_batch)
            weighted_loss = (loss_per_sample * w_batch).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Plot loss curve
    if plot_loss:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs+1), loss_history, marker='o')
        plt.title("Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Saved loss plot to {loss_plot_path}")

    return model


def evaluate_binary_classifier(model, X_test, y_test, threshold=0.5, device='cpu'):

    model.eval()
    model.to(device)

    # Convert to tensors
    X_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_true = y_test.values

    # Get predicted probabilities
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    y_pred = (probs > threshold).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)

    print(classification_report(y_true, y_pred))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    }