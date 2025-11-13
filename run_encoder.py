import onnxruntime as ort
import numpy as np
import time

MODEL_PATH = "encoder.onnx"  # or encoder_quant.onnx

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def read_signal():
    t = np.linspace(0, 1, 512)
    signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(512)
    return signal.astype(np.float32).reshape(1, 1, 512)

_ = session.run(None, {"signal": read_signal()})

num_runs = 1000
times = []

for _ in range(num_runs):
    x = read_signal()
    start = time.perf_counter()       # high-res start
    latent = session.run(None, {"signal": x})[0]
    elapsed = (time.perf_counter() - start) * 1000  # convert to ms
    times.append(elapsed)

print(f"Model used: {MODEL_PATH}")
print(f"Output shape: {latent.shape}")
print(f"Average latency: {np.mean(times):.4f} ms over {num_runs} runs")
print(f"Min latency: {np.min(times):.4f} ms | Max latency: {np.max(times):.4f} ms")
