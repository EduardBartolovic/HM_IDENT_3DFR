import torch
import torch.nn as nn


class PoseFrontalizer(nn.Module):
    def __init__(self, emb_dim=512, pose_dim=2, hidden_dim=256, out_emb_dim=512):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_emb_dim)
        )

        self.normalize = True
        self.max_angle = 25

    def forward(self, emb, headpose):
        # allow headpose to be long (like angle bins)
        headpose = headpose.float()

        if self.normalize:
            headpose /= self.max_angle

        fused = torch.cat([emb, headpose], dim=1)
        return self.mlp(fused)


