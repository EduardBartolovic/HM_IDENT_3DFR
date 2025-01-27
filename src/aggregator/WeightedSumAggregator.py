import torch
from torch import nn

class WeightedSumAggregator(nn.Module):
    def __init__(self, num_views):
        """
        Initialize the WeightedSumAggregator.

        Args:
            num_views (int): Number of views to aggregate.
        """
        super(WeightedSumAggregator, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_views) / num_views)

    def forward(self, all_view_stage):
        """
        Perform weighted sum pooling.

        Args:
            all_view_stage (torch.Tensor): Tensor of shape [batch, view, c, w, h].

        Returns:
            torch.Tensor: Aggregated tensor of shape [batch, c, w, h].
        """
        # Normalize weights to ensure they sum to 1
        #normalized_weights = F.softmax(self.weights, dim=0)
        normalized_weights = self.weights / self.weights.sum()

        # Apply weighted sum pooling
        views_pooled_stage = torch.einsum('bvchw,v->bchw', all_view_stage, normalized_weights)

        return views_pooled_stage



def make_weighted_sum_aggregator(view_list):
    aggregators = []
    for i in view_list:
        aggregators.append(WeightedSumAggregator(i))
    return aggregators
