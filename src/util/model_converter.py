import onnx
import torch
import numpy as np
import onnxruntime as ort
from onnx2torch import convert
from torchinfo import summary

from src.backbone.iresnet import iresnet50, IResNet
from src.backbone.model_irse import IR_50

# === 1. Load ONNX model and convert to PyTorch ===
# onnx_model_path = '../../pretrained/IR50_Glint360K.onnx'
# onnx_model = onnx.load(onnx_model_path)
# pytorch_model = convert(onnx_model)
# pytorch_model.eval()
#
# print("✅ ONNX model successfully converted to PyTorch.")
#
# # === 2. Save the converted PyTorch model ===
# torch.save(pytorch_model.state_dict(), '../../pretrained/IR50_Glint360K.pt')
# print("✅ Model saved successfully.")
#
# # === 3. Consistency Check: ONNX vs PyTorch ===
#
# # Generate random test input (batch_size=1, 3 channels, 112x112)
# test_input_np = np.random.rand(1, 3, 112, 112).astype(np.float32)
# test_input_torch = torch.from_numpy(test_input_np)
#
# # ONNX Inference
# ort_sess = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
# onnx_output = ort_sess.run(None, {ort_sess.get_inputs()[0].name: test_input_np})[0]
#
# # PyTorch Inference
# with torch.no_grad():
#     torch_output = pytorch_model(test_input_torch).cpu().numpy()
#
# # Compare
# max_diff = np.max(np.abs(torch_output - onnx_output))
# print(f"\n🔍 Max difference between ONNX and PyTorch output: {max_diff:.6f}")
#
# if max_diff < 1e-3:
#     print("✅ Outputs match closely!")
# else:
#     print("❌ Significant output difference!")
#
# print(torch_output[0][:30])
# print(onnx_output[0][:30])
#
# model_stats_backbone = summary(pytorch_model, (1, 3, 112, 112), verbose=0)
# print(str(model_stats_backbone))
#
# model_stats_backbone = summary(iresnet50(), (1, 3, 112, 112), verbose=0)
# print(str(model_stats_backbone))


model = IR_50([112,112], 512)
model.load_state_dict(torch.load('../../pretrained/backbone_ir50_ms1m_epoch63.pth', weights_only=True))
dummy_input = torch.randn(1, 3, 112, 112)
torch.onnx.export(
    model,                      # model being run
    dummy_input,                 # model input
    "backbone_ir50_ms1m_epoch63.onnx",                # output file name
    export_params=True,          # store trained weights
    opset_version=11,            # ONNX version (11 is safe, 17+ is latest)
    do_constant_folding=True,
    input_names = ['input'],
    output_names = ['output'],
    dynamic_axes = {'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
)

onnx_model = onnx.load("backbone_ir50_asia.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid ✅")