import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ----------- Dataset Class -----------

class EEGDataset(Dataset):
    def __init__(self, X, y):
        """
        X: EEG data (n_samples, n_channels, n_times)
        y: labels (n_samples,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, C, T)
        self.y = torch.tensor(y, dtype=torch.long)     # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # LSTM expects (batch, time, features), so we transpose
        return self.X[idx].T, self.y[idx]  # (T, C), label


# ----------- LSTM Model -----------

class EEG_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super(EEG_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, (hn, _) = self.lstm(x)        # x: (B, T, C), out: (B, T, H)
        out = hn[-1]                        # Use last hidden state: (B, H)
        return self.classifier(out)        # (B, num_classes)


class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, input_time, lstm_hidden=64, num_classes=2, dropout=0.5):
        super(CNN_LSTM, self).__init__()

        # CNN block: input (N, 1, C, T)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(input_channels, 5)),  # spatial filter across channels
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # Compute output time dimension after CNN
        conv_out_time = input_time - 5 + 1

        # LSTM input shape: (N, T', features)
        self.lstm = nn.LSTM(input_size=8, hidden_size=lstm_hidden, batch_first=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (N, T, C)
        x = x.permute(0, 2, 1).unsqueeze(1)  # → (N, 1, C, T)
        x = self.cnn(x)                      # → (N, 8, 1, T')
        x = x.squeeze(2).permute(0, 2, 1)    # → (N, T', 8)
        out, _ = self.lstm(x)                # → (N, T', H)
        out = out[:, -1, :]                  # last time step
        return self.classifier(out)


# ----------- Training Function -----------

def train_model(model, train_loader, val_data, epochs=30, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_X, val_y = val_data
    val_X = torch.tensor(val_X, dtype=torch.float32).permute(0, 2, 1).to(device)
    val_y = torch.tensor(val_y, dtype=torch.long).to(device)

    train_acc_list, val_acc_list = [], []

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = 100. * correct / total
        with torch.no_grad():
            model.eval()
            val_outputs = model(val_X)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = 100. * (val_preds == val_y).sum().item() / val_y.size(0)
            model.train()

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

    return model, train_acc_list, val_acc_list


# ----------- Prediction Function -----------

def predict(model, X_new):
    """
    X_new: EEG data of shape (n_trials, n_channels, n_times)
    Returns: predicted class labels
    """
    model.eval()
    X_tensor = torch.tensor(X_new, dtype=torch.float32).permute(0, 2, 1)  # (N, T, C)
    with torch.no_grad():
        outputs = model(X_tensor.to(next(model.parameters()).device))
        preds = torch.argmax(outputs, dim=1)
    return preds.cpu().numpy()
