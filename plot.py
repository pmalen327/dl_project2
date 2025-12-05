import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import torch
from models import Autoencoder

LATENT_DIR = "latent_logs"
LATENT_DIM = 64


def load_x_and_z(latent_dir=LATENT_DIR):
    """
    Load (x, z) pairs from .npz files in latent_dir.
    Each file is expected to contain:
      - x: original input window, shape (512,)
      - z: latent vector, shape (latent_dim,)
    Returns:
      X: (N, 512)
      Z: (N, latent_dim)
    """
    pattern_npz = os.path.join(latent_dir, "*.npz")
    files = sorted(glob.glob(pattern_npz))

    if not files:
        raise RuntimeError(f"No .npz files found in {latent_dir}. "
                           f"Expected paired x,z logs (sample_*.npz).")

    X_list, Z_list = [], []
    print(f"Found {len(files)} .npz files in {latent_dir}")

    for path in files:
        data = np.load(path)
        if "x" not in data or "z" not in data:
            raise KeyError(f"'x' and/or 'z' not found in {path}")
        x = data["x"].reshape(-1)  # (512,)
        z = data["z"].reshape(-1)  # (latent_dim,)

        X_list.append(x)
        Z_list.append(z)

    X = np.stack(X_list, axis=0)  # (N, 512)
    Z = np.stack(Z_list, axis=0)  # (N, latent_dim)
    print(f"Loaded X shape: {X.shape}, Z shape: {Z.shape}")
    return X, Z


def load_model(latent_dim=LATENT_DIM, ckpt_path="autoencoder.pth"):
    """
    Load trained autoencoder for decoding latent vectors.
    """
    model = Autoencoder(latent_dim=latent_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model


def compute_reconstruction_losses(model, X, Z):
    """
    Given:
      X: (N, 512)
      Z: (N, latent_dim)
    Decode Z and compute Smooth L1 reconstruction loss vs X.
    Returns:
      losses: (N,)
    """
    X_t = torch.tensor(X, dtype=torch.float32).view(X.shape[0], 1, -1)
    Z_t = torch.tensor(Z, dtype=torch.float32)

    with torch.no_grad():
        recon = model.decode(Z_t)            # (N, 1, 512)
    recon = recon.view_as(X_t)

    loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    # per-sample loss: mean over sequence dimension
    losses_full = loss_fn(recon, X_t)       # (N, 1, 512)
    losses = losses_full.mean(dim=(1, 2))   # (N,)
    return losses.numpy()


def plot_pca_2d(Z, losses, save_path=None):
    """
    PCA to 2D and scatter plot with color = reconstruction loss.
    """
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)
    explained = pca.explained_variance_ratio_
    print("PCA explained variance ratios:", explained)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(Z_2d[:, 0], Z_2d[:, 1],
                     c=losses, cmap="viridis", s=35, alpha=0.9)
    cbar = plt.colorbar(sc)
    cbar.set_label("Reconstruction loss", rotation=90)

    plt.xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    plt.title("Latent Space (PCA 2D, colored by reconstruction loss)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved PCA figure to {save_path}")
    else:
        plt.show()


def plot_latent_stats(Z, save_path=None):
    """
    Plot per-dimension mean/std and histogram of all latent values.
    """
    means = Z.mean(axis=0)
    stds = Z.std(axis=0)
    dims = np.arange(Z.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Per-dimension stats
    ax = axes[0]
    ax.plot(dims, means, label="mean")
    ax.fill_between(dims, means - stds, means + stds,
                    alpha=0.3, label="mean ± std")
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Value")
    ax.set_title("Latent Dimension Statistics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[1]
    ax.hist(Z.flatten(), bins=40, alpha=0.8)
    ax.set_title("Histogram of Latent Values")
    ax.set_xlabel("Latent value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved stats figure to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    print(f"Loading (x, z) pairs from: {LATENT_DIR}")
    X, Z = load_x_and_z(LATENT_DIR)

    print("Loading trained autoencoder...")
    model = load_model()

    print("Computing reconstruction losses for each latent...")
    losses = compute_reconstruction_losses(model, X, Z)
    print(f"Loss stats: mean={losses.mean():.6f}, "
          f"std={losses.std():.6f}, "
          f"min={losses.min():.6f}, "
          f"max={losses.max():.6f}")

    plot_pca_2d(Z, losses, save_path="latent_pca_loss.png")

    plot_latent_stats(Z, save_path="latent_stats.png")

    print("Done.")