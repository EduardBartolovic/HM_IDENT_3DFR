import torch
from torch import nn


class RobustMeanAggregator(nn.Module):
    def __init__(self, z_thresh=2.0):
        """
        Args:
            z_thresh: is ~95 % of values lie within ±2 standard deviations
        """

        super().__init__()
        self.z_thresh = z_thresh

    def forward(self, all_view_stage):
        """
        Args:
            all_view_stage (torch.Tensor): [B, V, C, H, W]
        Returns:
            torch.Tensor: [B, C, H, W]
        """

        # Flatten spatial dims
        view_means = all_view_stage.mean(dim=[2, 3, 4])  # shape [B, V]

        mean = view_means.mean(dim=1, keepdim=True)  # [B, 1]
        std = view_means.std(dim=1, keepdim=True) + 1e-6  # [B, 1]

        z_scores = torch.abs((view_means - mean) / std)  # [B, V]

        # Mask where z-score < threshold
        mask = (z_scores < self.z_thresh).float()  # [B, V]

        # Reshape mask for broadcasting: [B, V, 1, 1, 1]
        mask = mask[:, :, None, None, None]

        masked = all_view_stage * mask
        valid_counts = mask.sum(dim=1)
        aggregated = masked.sum(dim=1) / valid_counts
        return aggregated


def make_rma(view_list):
    aggregators = []
    for _ in view_list:
        aggregators.append(RobustMeanAggregator())
    return aggregators
