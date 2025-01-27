import torch
from torch import nn

class MeanAggregator(nn.Module):
    def __init__(self):
        """
        Initialize the MeanAggregator.

        """
        super(MeanAggregator, self).__init__()


    def forward(self, all_view_stage):
        """
        Perform mean pooling.

        Args:
            all_view_stage (torch.Tensor): Tensor of shape [batch, view, c, w, h].

        Returns:
            torch.Tensor: Aggregated tensor of shape [batch, c, w, h].
        """
        return all_view_stage.mean(dim=1)



def make_mean_aggregator(view_list):
    aggregators = []
    for _ in view_list:
        aggregators.append(MeanAggregator())
    return aggregators
