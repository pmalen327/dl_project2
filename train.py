import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models import Autoencoder


X_train = np.load("data/processed/bearing_train.npy")
X_test  = np.load("data/processed/bearing_test.npy")

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

def per_window_znorm(x):
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std

train_loader = DataLoader(TensorDataset(X_train), batch_size=128, shuffle=True, drop_last=False)
test_loader  = DataLoader(TensorDataset(X_test),  batch_size=128, shuffle=False, drop_last=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(latent_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.SmoothL1Loss(beta=0.02)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
)


epochs = 50
best_test = float("inf")

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for (batch,) in train_loader:
        batch = batch.to(device)
        batch = per_window_znorm(batch)
        noisy = batch + 0.02 * torch.randn_like(batch)
        optimizer.zero_grad()
        recon = model(noisy)
        loss = loss_fn(recon, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for (batch,) in test_loader:
            batch = batch.to(device)
            batch = per_window_znorm(batch)
            noisy = batch + 0.02 * torch.randn_like(batch)  # and here
            recon = model(noisy)
            test_loss += loss_fn(recon, batch).item()

    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    scheduler.step(test_loss)

    print(f"Epoch {epoch:02d}/{epochs} | Train: {train_loss:.6f} | "
          f"Test: {test_loss:.6f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

    if test_loss < best_test:
        best_test = test_loss
        torch.save(model.state_dict(), "autoencoder.pth")