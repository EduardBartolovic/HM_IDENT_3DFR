from torch import nn
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, PReLU, Dropout, Linear, BatchNorm1d

from src.backbone.model_irse import Bottleneck, bottleneck_IR, Flatten


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks():
    return [
        get_block(in_channel=64, depth=64, num_units=3),
        get_block(in_channel=64, depth=128, num_units=4),
        get_block(in_channel=128, depth=256, num_units=14),
        get_block(in_channel=256, depth=512, num_units=3)
    ]


class Backbone(Module):
    def __init__(self, input_size, num_layers, embedding_size=512):
        super(Backbone, self).__init__()
        assert input_size[0] in [112], "input_size should be [112, 112]"

        unit_module = bottleneck_IR
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 7 * 7, embedding_size),
                                           BatchNorm1d(embedding_size))

        modules = []
        blocks = get_blocks()
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
        self._initialize_weights()

    def forward(self, x, return_featuremaps=False, execute_input=True, execute_body=True, execute_output=True):
        feature_maps = {}

        if execute_input:
            x = self.input_layer(x)
            feature_maps['input_stage'] = x.cpu()

        if execute_body:
            for i, layer in enumerate(self.body):
                x = layer(x)
                if return_featuremaps:
                    if i == 2:
                        feature_maps[f'block_{i}'] = x.cpu()
                    elif i == 6:
                        feature_maps[f'block_{i}'] = x.cpu()
                    elif i == 20:
                        feature_maps[f'block_{i}'] = x.cpu()
                    elif i == 23:
                        feature_maps[f'block_{i}'] = x.cpu()
                    #feature_maps[f'block_{i}'] = x # Store feature maps at each block

        if execute_output:
            x = self.output_layer(x)
            if return_featuremaps:
                feature_maps['output_stage'] = x.cpu()
                return feature_maps

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()



def IR_MV_50(input_size, embedding_size):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, embedding_size)

    return model