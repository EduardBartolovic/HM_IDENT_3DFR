import torch
from torch import nn


class SEAggregator(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Squeeze-and-Excitation Aggregator to merge multi-view feature maps.
        This implements the Squeeze-and-Excitation mechanic from https://arxiv.org/abs/1709.01507

        Args:
            channels (int): Number of feature channels (e.g., 124).
            reduction (int): Reduction ratio for the SE block.
        """
        super(SEAggregator, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool3d((1, 1, 1))  # Pool across [view, h, w]
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, all_view_stage):
        """
        Args:
            all_view_stage (torch.Tensor): Tensor shape [batch, view, c, h, w].

        Returns:
            torch.Tensor: Aggregated tensor shape [batch, c, h, w].
        """
        b, v, c, h, w = all_view_stage.shape

        # Squeeze: Pool across view, height, and width
        pooled = self.squeeze(all_view_stage)  # [b, c, 1, 1, 1]
        pooled = pooled.view(b, c)  # [b, c]

        # Excitation: Learn attention per channel
        channel_weights = self.excitation(pooled).view(b, 1, c, 1, 1)  # [b, 1, c, 1, 1]

        # Reweight each view using the channel attention
        reweighted = all_view_stage * channel_weights  # [b, v, c, h, w]

        # Average the views TODO: Check weighted sum over views
        output = torch.mean(reweighted, dim=1)  # [b, c, h, w]

        return output


def make_se_aggregator(view_list):
    aggregators = []
    for views, bias in view_list:
        aggregators.append(SEAggregator(views))
    return aggregators
