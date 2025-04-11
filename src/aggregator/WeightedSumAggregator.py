import torch
import torch.nn.functional as f
from torch import nn


class WeightedSumAggregator(nn.Module):
    def __init__(self, num_views, last_view_bias=2.0):
        """
        Initialize the WeightedSumAggregator.

        Args:
            num_views (int): Number of views to aggregate.
        """
        super(WeightedSumAggregator, self).__init__()
        initial_weights = torch.ones(num_views)  # Start with uniform weights
        initial_weights[-1] += last_view_bias  # Boost view
        self.weights = nn.Parameter(initial_weights)

    def forward(self, all_view_stage):
        """
        Perform weighted sum pooling.

        Args:
            all_view_stage (torch.Tensor): Tensor of shape [batch, view, c, w, h].

        Returns:
            torch.Tensor: Aggregated tensor of shape [batch, c, w, h].
        """
        # Normalize weights to ensure they sum to 1
        normalized_weights = f.softmax(self.weights, dim=0)

        # Apply weighted sum pooling
        views_pooled_stage = torch.einsum('bvchw,v->bchw', all_view_stage, normalized_weights)

        return views_pooled_stage


def make_weighted_sum_aggregator(view_list):
    aggregators = []
    for views, bias in view_list:
        aggregators.append(WeightedSumAggregator(views, last_view_bias=bias))
    return aggregators
