import torch
import torch.nn.functional as f
from torch import nn

from src.aggregator.MeanAggregator import MeanAggregator


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

    def forward(self, all_view_stage: torch.Tensor, all_view_embeddings: torch.Tensor, **kwargs):
        """
        Args:
            all_view_stage: [B, V, C, H, W] feature maps
            all_view_embeddings: [B, V, D] embeddings per view
        Returns:
            views_pooled_stage: [B, C, H, W] fused feature map
        """
        B, V, D = all_view_embeddings.shape

        # Compute cosine similarity matrix [B, V, V]
        sims = f.cosine_similarity(
            all_view_embeddings.unsqueeze(2),  # [B, V, 1, D]
            all_view_embeddings.unsqueeze(1),  # [B, 1, V, D]
            dim=-1
        )  # [B, V, V]

        # Flatten similarity matrix per batch
        flat_sims = sims.reshape(B, V * V)

        # Get per-view scores
        scores = self.scorer(flat_sims)  # [B, V]

        # Normalize with softmax
        normalized_weights = f.softmax(scores, dim=1)  # [B, V]

        # Weighted pooling across views
        if all_view_stage.shape[1] == normalized_weights.shape[1]:
            views_pooled_stage = torch.einsum('bvchw,bv->bchw', all_view_stage, normalized_weights)
        else:
            # safety if mismatch
            views_pooled_stage = torch.einsum(
                'bvchw,bv->bchw', all_view_stage[:, :normalized_weights.shape[1]], normalized_weights
            )

        return views_pooled_stage


class EmbeddingWeightedAggregator(nn.Module):
    def __init__(self, num_views, emb_dim=512, hidden_dim=512):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(num_views * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_views, bias=False)  # one score per view
        )
        # init last layer to zeros → uniform start
        nn.init.zeros_(self.scorer[-1].weight)

    def forward(self, all_view_stage, all_view_embeddings):
        """
        Args:
            all_view_stage: [B, V, C, H, W] feature maps
            all_view_embeddings: [B, V, D] embeddings per view
        """
        B, V, D = all_view_embeddings.shape

        # Flatten embeddings of all views into one vector
        flat_emb = all_view_embeddings.reshape(B, V * D)  # [B, V*D]

        # Produce per-view scores
        scores = self.scorer(flat_emb)  # [B, V]

        # Normalize
        normalized_weights = f.softmax(scores, dim=1)  # [B, V]

        # Weighted sum pooling across views
        if all_view_stage.shape[1] == normalized_weights.shape[1]:
            views_pooled_stage = torch.einsum('bvchw,bv->bchw', all_view_stage, normalized_weights)
        else:
            views_pooled_stage = torch.einsum('bvchw,bv->bchw', all_view_stage[:,:-1,:,:,:], normalized_weights)

        return views_pooled_stage


def make_embeddingweighted_aggregator(agg_config):
    view_list = [8, 8, 8, 8, 8]
    aggregators = []
    for idx, views in enumerate(view_list):
        if idx == 5:
            aggregators.append(CosineDistanceAggregator(views))
        else:
            aggregators.append(MeanAggregator(use_aggregator_branch=False))

    return aggregators
