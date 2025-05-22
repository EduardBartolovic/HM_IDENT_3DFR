import torch
from torch import nn


class SEAggregator(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Squeeze-and-Excitation Aggregator to merge multi-view feature maps.
        This implements the Squeeze-and-Excitation mechanic from https://arxiv.org/abs/1709.01507

        Args:
            channels (int): Number of feature channels
            reduction (int): Reduction ratio for the SE block.
        """
        super(SEAggregator, self).__init__()
        self.reduction = reduction
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, all_view_stage):
        """
        Args:
            all_view_stage (torch.Tensor): Tensor of shape [B, V, C, H, W].

        Returns:
            torch.Tensor: Aggregated tensor of shape [B, C, H, W].
        """

        # Squeeze: average spatial dimensions (H, W)
        pooled = all_view_stage.mean(dim=[3, 4])  # [B, V, C]

        # Excitation: apply FC layers to each view
        se_weights = self.fc1(pooled)  # [B, V, C//r]
        se_weights = self.relu(se_weights)
        se_weights = self.fc2(se_weights)  # [B, V, C]
        se_weights = self.sigmoid(se_weights)  # [B, V, C]

        # Reweight: apply channel-wise weights to each view
        se_weights = se_weights.unsqueeze(-1).unsqueeze(-1)  # [B, V, C, 1, 1]
        weighted = all_view_stage * se_weights  # [B, V, C, H, W]

        # Aggregate over views
        output = weighted.mean(dim=1)  # [B, C, H, W]

        return output

    def get_weights(self):
        return [self.fc1.weight.detach().cpu().numpy().copy(), self.relu.weight.detach().cpu().numpy().copy(), self.fc2.weight.detach().cpu().numpy().copy(), self.sigmoid.weight.detach().cpu().numpy().copy()]


def make_se_aggregator(channels_list):
    aggregators = []
    for channels in channels_list:
        aggregators.append(SEAggregator(channels))
    return aggregators
