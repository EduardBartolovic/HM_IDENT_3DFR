import torch
import torch.nn as nn

from src.backbone.facenet import ir_facenet_50
from src.backbone.iresnet_insight import iresnet50, iresnet34, iresnet18, iresnet100
from src.backbone.model_irse import ir_50


class MultiviewIResnetLF(nn.Module):
    def __init__(self, device, aggregator, backbone_fn, embedding_size=512, fp16=True):
        super().__init__()
        self.backbone_reg = backbone_fn(embedding_size=embedding_size, fp16=fp16)
        self.fp16 = fp16
        self.precision = torch.float16 if fp16 else torch.float32
        self.aggregator = aggregator
        self.device = device

    def forward(self, inputs, perspectives, face_corr, use_face_corr):
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


def ir_mv_50_lf(device, aggregator, embedding_size=512):
    return MultiviewIResnetLF(device, aggregator, ir_50, embedding_size, fp16=False)


def ir_mv_v2_100_lf(device, aggregator, embedding_size=512, fp16=False):
    return MultiviewIResnetLF(device, aggregator, iresnet100, embedding_size, fp16)


def ir_mv_v2_50_lf(device, aggregator, embedding_size=512, fp16=False):
    return MultiviewIResnetLF(device, aggregator, iresnet50, embedding_size, fp16)


def ir_mv_v2_34_lf(device, aggregator, embedding_size=512, fp16=False):
    return MultiviewIResnetLF(device, aggregator, iresnet34, embedding_size, fp16)


def ir_mv_v2_18_lf(device, aggregator, embedding_size=512, fp16=False):
    return MultiviewIResnetLF(device, aggregator, iresnet18, embedding_size, fp16)


def ir_mv_facenet_50_lf(device, aggregators, embedding_size=512, fp16=False):
    return MultiviewIResnetLF(device, aggregators, ir_facenet_50, embedding_size, fp16)

