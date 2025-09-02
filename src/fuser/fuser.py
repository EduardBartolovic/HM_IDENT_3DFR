import torch.nn as nn
import torch.nn.functional as F


class SoftmaxFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attn = nn.Linear(dim, 1)  # learn per-view wights

    def forward(self, x):
        # x: [B, V, D]
        weights = F.softmax(self.attn(x), dim=1)  # [B, V, 1]
        fused = (weights * x).sum(dim=1)  # [B, D]
        return fused


class MLPFusion(nn.Module):
    def __init__(self, views=8, hidden=1024, out_dim=512, dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(views * dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        # x: [B, V, D]
        B, V, D = x.shape
        x = x.view(B, V * D)  # flatten views
        return self.fc(x)


class TransformerFusion(nn.Module):
    def __init__(self, dim=512, num_heads=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, V, D]
        x = self.transformer(x)  # [B, V, D]
        fused = x.mean(dim=1)  # global average pooling
        return fused


def make_mlp_fusion():
    return MLPFusion(views=8, hidden=1024, out_dim=512, dim=512)


def make_transformer_fusion():
    return TransformerFusion(dim=512, num_heads=8, num_layers=2)


def make_softmax_fusion():
    return SoftmaxFusion(dim=512)
