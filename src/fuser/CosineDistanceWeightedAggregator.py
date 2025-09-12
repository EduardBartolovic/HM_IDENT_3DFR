import torch
import torch.nn.functional as f
from torch import nn


class CosineDistanceAggregator(nn.Module):
    def __init__(self, num_views, hidden_dim=16):
        super().__init__()
        # Input = flattened cosine similarity matrix (V*V)
        self.scorer = nn.Sequential(
            nn.Linear(num_views * num_views, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_views, bias=False)  # one score per view
        )
        # init last layer to zeros → start with uniform weights
        nn.init.zeros_(self.scorer[-1].weight)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, V, D] embeddings per view
        Returns:
            views_pooled_emb: [B, C, H, W] fused embedding
        """
        B, V, D = x.shape

        # Compute cosine similarity matrix [B, V, V]
        sims = f.cosine_similarity(
            x.unsqueeze(2),  # [B, V, 1, D]
            x.unsqueeze(1),  # [B, 1, V, D]
            dim=-1
        )  # [B, V, V]

        # Flatten similarity matrix per batch
        flat_sims = sims.reshape(B, V * V)

        # Get per-view scores
        scores = self.scorer(flat_sims)  # [B, V]

        # Normalize with softmax
        normalized_weights = f.softmax(scores, dim=1)  # [B, V]

        # Weighted pooling across views
        views_pooled_emb = torch.einsum('bvd,bv->bd', x, normalized_weights)

        return views_pooled_emb


def make_cosinedistance_fusion(num_views):
    return CosineDistanceAggregator(num_views, hidden_dim=64)
