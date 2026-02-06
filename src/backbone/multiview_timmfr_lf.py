import torch
import torch.nn as nn

from src.backbone.timmfr import timm_fr


class MultiviewTimmFRLF(nn.Module):

    def __init__(self, device, aggregators, backbone_fn):
        super().__init__()
        self.backbone_reg = backbone_fn()
        self.aggregators = aggregators
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
        self.aggregators.eval()


def timm_mv_lf(device, aggregators):
    return MultiviewTimmFRLF(device, aggregators, timm_fr)

