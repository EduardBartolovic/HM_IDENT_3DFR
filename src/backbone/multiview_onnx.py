import numpy as np
import torch
import torch.nn as nn

import onnxruntime as ort

class ONNXBackboneWrapper:
    def __init__(self, onnx_path, device):
        self.device = device
        providers = ["CUDAExecutionProvider"] if "cuda" in device.type else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # get input name
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x):
        """
        x: (B, C, H, W) torch tensor
        returns: torch.tensor (B, 512)
        """
        x_np = x.detach().cpu().numpy().astype(np.float32)

        ort_out = self.session.run(None, {self.input_name: x_np})

        out = torch.from_numpy(ort_out[0]).to(x.device)
        return out

class MultiviewONNX(nn.Module):

    def __init__(self, device, onnx_path):
        super().__init__()

        self.device = device
        self.backbone_reg = ONNXBackboneWrapper(onnx_path, device)

    def forward(self, inputs, perspectives, face_corr, use_face_corr):
        """
        inputs: list of every view: [(B,C,H,W), (B,C,H,W), (B,C,H,W), ...]

        output:
            embeddings_reg: (V, B, 512)
            embeddings_agg  (B, 512)
        """
        all_views_embeddings = []
        for view in inputs:
            emb = self.backbone_reg(view.to(self.device))
            all_views_embeddings.append(emb)

        return all_views_embeddings, None


    def eval(self):
        pass


def onnx_mv(device, onnx_model_path):
    return MultiviewONNX(device, onnx_model_path)

