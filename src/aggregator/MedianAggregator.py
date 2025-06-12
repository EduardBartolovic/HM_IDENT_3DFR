import torch
from torch import nn


class MedianAggregator(nn.Module):
    def __init__(self):
        """
        Initialize the MeanAggregator.

        """
        super(MedianAggregator, self).__init__()

    def forward(self, all_view_stage):
        """
        Perform median pooling.

        Args:
            all_view_stage (torch.Tensor): Tensor of shape [batch, view, c, w, h].

        Returns:
            torch.Tensor: Aggregated tensor of shape [batch, c, w, h].
        """
        return all_view_stage.median(dim=1).values


def make_median_aggregator(view_list):
    aggregators = []
    for _ in view_list:
        aggregators.append(MedianAggregator())
    return aggregators
