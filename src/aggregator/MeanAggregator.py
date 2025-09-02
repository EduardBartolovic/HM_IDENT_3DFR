import torch
from torch import nn


class MeanAggregator(nn.Module):
    def __init__(self, use_aggregator_branch=True):
        """
        Initialize the MeanAggregator.
        """
        super(MeanAggregator, self).__init__()

        self.use_aggregator_branch = use_aggregator_branch

    def forward(self, all_view_stage):
        """
        Perform mean pooling.

        Args:
            all_view_stage (torch.Tensor): Tensor of shape [batch, view, c, w, h].

        Returns:
            torch.Tensor: Aggregated tensor of shape [batch, c, w, h].
        """
        if self.use_aggregator_branch:
            return all_view_stage.mean(dim=1)
        else:
            return all_view_stage[:, :-1, :, :, :].mean(dim=1)


def make_mean_aggregator(agg_config):

    if agg_config["ACTIVE_STAGES"]:
        activate_stages = agg_config["ACTIVE_STAGES"]
    else:
        activate_stages = (True, True, True, True, True)
    aggregators = []
    for active in activate_stages:
        aggregators.append(MeanAggregator(active))
    return aggregators
