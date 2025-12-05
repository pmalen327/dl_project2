import onnxruntime as ort
import numpy as np
import time
import os
from datetime import datetime

MODEL_PATH = "encoder.onnx"
LOG_DIR = "latent_logs"   # will contain paired x + z

os.makedirs(LOG_DIR, exist_ok=True)

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def get_signal():
    """
    Synthetic 512-sample signal for now.
    Replace with real sensor data or a loaded bearing window later.
    """
    t = np.linspace(0, 1, 512)
    signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(512)
    return signal.astype(np.float32).reshape(1, 1, 512)

NUM_SAMPLES = 10  # how many pairs x,z to generate

for i in range(NUM_SAMPLES):
    x = get_signal()  # shape (1, 1, 512)

    start = time.time()
    latent = session.run(None, {"signal": x})[0]   # shape (1, latent_dim)
    elapsed_ms = (time.time() - start) * 1000.0

    x_flat = x.squeeze()        # (512,)
    z_flat = latent.squeeze()   # (latent_dim,)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(LOG_DIR, f"sample_{timestamp}.npz")

    # Save both input and latent together
    np.savez(out_path, x=x_flat, z=z_flat)

    print(f"[{i+1}/{NUM_SAMPLES}] Saved {out_path} | x shape={x_flat.shape}, z shape={z_flat.shape} | {elapsed_ms:.3f} ms")

print("\nDone generating latent logs.")
