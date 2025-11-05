import onnxruntime as ort
import numpy as np
import time

session = ort.InferenceSession("encoder.onnx")

def read_signal():
    t = np.linspace(0, 1, 512)
    sig = np.sin(2 * np.pi * 3 * t) + 0.05 * np.random.randn(512)
    return sig.astype(np.float32).reshape(1, 1, 512)

for i in range(5):
    sig = read_signal()
    start = time.time()
    latent = session.run(None, {"signal": sig})[0]
    print(f"Inference {i+1}: latent shape {latent.shape}, "
          f"time {time.time() - start:.4f}s")