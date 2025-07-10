import torch
from torch import nn


class L2Aggregator(nn.Module):
    def __init__(self):
        """
        Initialize the L2Aggregator.
        """
        super(L2Aggregator, self).__init__()

    def forward(self, all_view_stage):
        """
        Perform L2-norm pooling across views.

        Args:
            all_view_stage (torch.Tensor): Tensor of shape [batch, view, c, w, h].

        Returns:
            torch.Tensor: Aggregated tensor of shape [batch, c, w, h].
        """
        pooled = torch.sqrt(torch.mean(all_view_stage ** 2, dim=1) + 1e-12)  # small epsilon for stability
        return pooled


def make_l2_aggregator(view_list):
    aggregators = []
    for _ in view_list:
        aggregators.append(L2Aggregator())
    return aggregators
