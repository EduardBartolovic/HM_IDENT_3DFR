import torch
import torch.nn as nn

from src.backbone.iresnet_insight import iresnet50, iresnet34, iresnet18, iresnet100


class IR_MV_V2_LF(nn.Module):
    def __init__(self, device, aggregator, backbone_fn, embedding_size=512, fp16=True):
        super().__init__()
        self.backbone_reg = backbone_fn(num_features=embedding_size, fp16=fp16)
        self.fp16 = fp16
        self.precision = torch.float16 if fp16 else torch.float32
        self.aggregator = aggregator
        self.device = device

    def forward(self, inputs, perspectives, face_corrs, use_face_corr):
        """
        inputs: list of every view: [(B,C,H,W), (B,C,H,W), (B,C,H,W), ...]

        output:
            embeddings_reg: (V, B, 512)
            embeddings_agg  (B, 512)
        """
        with torch.no_grad():
            all_views_embeddings = []
            for view in inputs:
                embeddings = self.backbone_reg(view.to(self.device))
                all_views_embeddings.append(embeddings)

        embeddings_concat = torch.stack(all_views_embeddings, dim=1)  # [B, V, dim]
        embeddings_agg = self.aggregator(embeddings_concat)
        embeddings_reg = all_views_embeddings

        return embeddings_reg, embeddings_agg

    def eval(self):
        self.backbone_reg.eval()
        self.aggregator.eval()


def IR_MV_V2_100_LF(device, aggregators, embedding_size=512, fp16=False, active_stages=None):
    return IR_MV_V2_LF(device, aggregators, iresnet100, embedding_size, fp16)


def IR_MV_V2_50_LF(device, aggregators, embedding_size=512, fp16=False, active_stages=None):
    return IR_MV_V2_LF(device, aggregators, iresnet50, embedding_size, fp16)


def IR_MV_V2_34_LF(device, aggregators, embedding_size=512, fp16=False, active_stages=None):
    return IR_MV_V2_LF(device, aggregators, iresnet34, embedding_size, fp16)


def IR_MV_V2_18_LF(device, aggregators, embedding_size=512, fp16=False, active_stages=None):
    return IR_MV_V2_LF(device, aggregators, iresnet18, embedding_size, fp16)
