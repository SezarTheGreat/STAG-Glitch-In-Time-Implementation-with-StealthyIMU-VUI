import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

# ---------------------------------------------------------
# PyTorch CNN Regressor
# ---------------------------------------------------------
class CNNUpscaler(nn.Module):
    def __init__(self, seq_len=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(64 * seq_len, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, 4) -> convert to (batch_size, 4, seq_len)
        x = x.transpose(1, 2)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)

# ---------------------------------------------------------
# PyTorch RNN (GRU) Regressor
# ---------------------------------------------------------
class RNNUpscaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(input_size=4, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, 4)
        out, _ = self.rnn(x) # out shape: (batch_size, seq_len, 64)
        # Average pooling over time steps
        out = out.mean(dim=1) # shape: (batch_size, 64)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze(-1)

# ---------------------------------------------------------
# PyTorch Model Trainer
# ---------------------------------------------------------
def train_pytorch_model(model, X, Y, epochs=3, batch_size=256, lr=1e-3, device="cpu"):
    model.to(device)
    model.train()
    
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
            count += len(bx)
        # print(f"    Epoch {epoch+1}/{epochs} Loss: {total_loss/max(1, count):.4f}")

# ---------------------------------------------------------
# Tabular Model Trainers
# ---------------------------------------------------------
def train_random_forest(X, Y):
    # Flatten window feature representation for tabular models
    # X shape: (N, seq_len, 4) -> flatten to (N, seq_len * 4)
    N, seq_len, channels = X.shape
    X_flat = X.reshape(N, seq_len * channels)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_flat, Y)
    return model

def train_lightgbm(X, Y):
    N, seq_len, channels = X.shape
    X_flat = X.reshape(N, seq_len * channels)
    
    model = LGBMRegressor(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_flat, Y)
    return model
