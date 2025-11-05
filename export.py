import torch
from models import Autoencoder
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. Load trained model
model = Autoencoder(latent_dim=32)
model.load_state_dict(torch.load("autoencoder.pth", map_location="cpu"))
encoder = model.encoder.eval()

# 2. Export to ONNX (float32 baseline)
dummy_input = torch.randn(1, 1, 512)
onnx_path = "encoder.onnx"
torch.onnx.export(
    encoder,
    dummy_input,
    onnx_path,
    input_names=["signal"],
    output_names=["latent"],
    opset_version=13
)
print(f"Exported base model: {onnx_path}")

# 3. Quantize to int8
quantized_path = "encoder_quant.onnx"
quantize_dynamic(
    model_input=onnx_path,
    model_output=quantized_path,
    weight_type=QuantType.QInt8  # use QuantType.QUInt8 if signed weights cause issues
)
print(f"Quantized model saved as: {quantized_path}")