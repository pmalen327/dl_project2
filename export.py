import torch
from models import Autoencoder
from onnxruntime.quantization import quantize_dynamic, QuantType


latent_dim = 64
model = Autoencoder(latent_dim=latent_dim)
model.load_state_dict(torch.load("autoencoder.pth", map_location="cpu"))
model.eval()


encoder = torch.nn.Sequential(model.encoder_convs, torch.nn.Flatten(), model.fc_enc)
encoder.eval()


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


quantized_path = "encoder_quant.onnx"
quantize_dynamic(
    model_input=onnx_path,
    model_output=quantized_path,
    weight_type=QuantType.QInt8
)

print(f"Quantized model saved as: {quantized_path}")