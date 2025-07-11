from torch import nn


class ConvAggregator(nn.Module):
    def __init__(self, num_views, channels, use_activation=False, use_batchnorm=False, dropout_prob=0):
        super(ConvAggregator, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_views * channels, out_channels=channels, kernel_size=1, bias=False)

        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn = nn.BatchNorm2d(channels)

        self.use_activation = use_activation
        if self.use_activation:
            self.activation = nn.ReLU(inplace=True)

        self.use_dropout = dropout_prob > 0
        if self.use_dropout:
            self.dropout = nn.Dropout2d(p=dropout_prob)


    def forward(self, all_view_stage):
        """
        Args:
            all_view_stage: [B, V, C, W, H]
        Returns:
            [B, C, W, H]
        """
        B, V, C, W, H = all_view_stage.shape
        x = all_view_stage.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, V, W, H]
        x = x.view(B, V * C, W, H)  # [B, V*C, W, H]
        x = self.conv(x)  # [B, C, W, H]

        if self.use_batchnorm:
            x = self.bn(x)

        if self.use_activation:
            x = self.activation(x)

        if self.use_dropout > 0:
            x = self.dropout(x)

        return x


def make_conv_aggregator(agg_config: dict):


    if "VIEWS" in agg_config.keys():
        view_list = agg_config["VIEWS"]
    else:
        view_list = [8, 9, 9, 9, 9]

    if "USE_ACTIVATION" in agg_config.keys():
        use_activation = agg_config["USE_ACTIVATION"]
    else:
        use_activation = False

    if "USE_BATCHNORM" in agg_config.keys():
        use_batchnorm = agg_config["USE_BATCHNORM"]
    else:
        use_batchnorm = False

    if "DROPOUT" in agg_config.keys():
        dropout_prob = agg_config["DROPOUT"]
    else:
        dropout_prob = 0
    channel_list = [64, 64, 128, 256, 512]

    aggregators = []
    for views, channels in zip(view_list, channel_list):
        print(views, channels)
        aggregators.append(ConvAggregator(num_views = views, channels=channels, use_activation=use_activation, use_batchnorm=use_batchnorm, dropout_prob=dropout_prob))
    return aggregators
