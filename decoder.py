import torch
import numpy as np
import glob
import os
from models import Autoencoder


print("Current working directory:", os.getcwd())

LOG_PATTERN = "latent_logs/*.npz"
files = sorted(glob.glob(LOG_PATTERN))
print(f"Found {len(files)} latent log files matching {LOG_PATTERN}")

if not files:
    print("No .npz files found. Make sure you copied them into latent_logs/")
    raise SystemExit


LATENT_DIM = 64  # must match training + encoder.onnx
model = Autoencoder(latent_dim=LATENT_DIM)
model.load_state_dict(torch.load("autoencoder.pth", map_location="cpu"))
model.eval()

def decode_latent(z):
    """
    z: numpy array shape (latent_dim,)
    returns: numpy array shape (512,)
    """
    z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0)  # (1, latent_dim)
    with torch.no_grad():
        recon = model.decode(z_t)  # (1, 1, 512)
    return recon.squeeze().numpy()

def reconstruction_loss(x_true, x_recon):
    """
    Smooth L1 loss between original Pi input and reconstructed signal.
    """
    x_true_t = torch.tensor(x_true, dtype=torch.float32).view(1, 1, -1)
    x_recon_t = torch.tensor(x_recon, dtype=torch.float32).view(1, 1, -1)
    loss = torch.nn.functional.smooth_l1_loss(x_recon_t, x_true_t)
    return float(loss)


losses = []

for path in files:
    data = np.load(path)
    x = data["x"]  # original 512-sample input from Pi
    z = data["z"]  # latent vector

    recon = decode_latent(z)
    loss = reconstruction_loss(x, recon)

    print(f"{os.path.basename(path)}: recon_loss = {loss:.6f}")
    losses.append(loss)

if losses:
    losses = np.array(losses)
    print("\nSummary over", len(losses), "samples:")
    print(f"  mean loss = {losses.mean():.6f}")
    print(f"  std  loss = {losses.std():.6f}")
    print(f"  min  loss = {losses.min():.6f}")
    print(f"  max  loss = {losses.max():.6f}")
