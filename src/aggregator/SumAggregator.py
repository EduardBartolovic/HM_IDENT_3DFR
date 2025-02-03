import torch
from torch import nn

class SumAggregator(nn.Module):
    def __init__(self):
        """
        Initialize the MeanAggregator.

        """
        super(SumAggregator, self).__init__()


    def forward(self, all_view_stage):
        """
        Perform sum pooling.

        Args:
            all_view_stage (torch.Tensor): Tensor of shape [batch, view, c, w, h].

        Returns:
            torch.Tensor: Aggregated tensor of shape [batch, c, w, h].
        """
        return all_view_stage.sum(dim=1)



def make_max_aggregator(view_list):
    aggregators = []
    for _ in view_list:
        aggregators.append(MaxAggregator())
    return aggregators
