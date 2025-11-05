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

train_loader = DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test), batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(latent_dim=32).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 25
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch, in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, in test_loader:
            batch = batch.to(device)
            recon = model(batch)
            test_loss += loss_fn(recon, batch).item()

    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"Train Loss: {train_loss/len(train_loader):.6f} | "
          f"Test Loss: {test_loss/len(test_loader):.6f}")

torch.save(model.state_dict(), "autoencoder.pth")
print("Model saved to autoencoder.pth")