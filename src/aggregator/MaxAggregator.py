import torch
from torch import nn


class MaxAggregator(nn.Module):
    def __init__(self, use_aggregator_branch):
        """
        Initialize the MaxAggregator.

        """
        super(MaxAggregator, self).__init__()

        self.use_aggregator_branch = use_aggregator_branch

    def forward(self, all_view_stage, *arg, **kwargs):
        """
        Perform max pooling.

        Args:
            all_view_stage (torch.Tensor): Tensor of shape [batch, view, c, w, h].

        Returns:
            torch.Tensor: Aggregated tensor of shape [batch, c, w, h].
        """
        if self.use_aggregator_branch:
            return all_view_stage.max(dim=1)[0]
        else:
            return all_view_stage[:, -1, :, :, :]
            #return all_view_stage[:, :-1, :, :, :].max(dim=1)[0]


def make_max_aggregator(agg_config):

    if agg_config["ACTIVE_STAGES"]:
        activate_stages = agg_config["ACTIVE_STAGES"]
    else:
        activate_stages = (True, True, True, True, True)
    aggregators = []
    for active in activate_stages:
        aggregators.append(MaxAggregator(active))
    return aggregators
