import torch
from torch import nn

class MaxAggregator(nn.Module):
    def __init__(self):
        """
        Initialize the MeanAggregator.

        """
        super(MaxAggregator, self).__init__()


    def forward(self, all_view_stage):
        """
        Perform max pooling.

        Args:
            all_view_stage (torch.Tensor): Tensor of shape [batch, view, c, w, h].

        Returns:
            torch.Tensor: Aggregated tensor of shape [batch, c, w, h].
        """
        return all_view_stage.max(dim=1)[0]



def make_max_aggregator(view_list):
    aggregators = []
    for _ in view_list:
        aggregators.append(MaxAggregator())
    return aggregators
