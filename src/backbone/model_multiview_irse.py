import torch
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
    def __init__(self, input_size, embedding_size=512):
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

    def forward(self, x, return_featuremaps=False, execute_stage=None):
        if execute_stage is None:
            execute_stage = {0, 1, 2, 3, 4}

        feature_maps = {}

        # Stage 0: Input Layer
        if 0 in execute_stage:
            x = self.input_layer(x)
            feature_maps['input_stage'] = x

        # Body Layer Execution
        if {1, 2, 3} & execute_stage:  # Check if body should be executed
            body_layers_to_execute = set()

            # Define layers to execute for each stage
            if 1 in execute_stage:
                body_layers_to_execute.update({0, 1, 2})
            if 2 in execute_stage:
                body_layers_to_execute.update(range(3, 21))  # Layers 3 to 20
            if 3 in execute_stage:
                body_layers_to_execute.update({21, 22, 23})

            # Execute only selected layers
            for i, layer in enumerate(self.body):
                if i in body_layers_to_execute:
                    x = layer(x)
                    if return_featuremaps and i in {2, 6, 20, 23}:
                        feature_maps[f'block_{i}'] = x

        # Stage 4: Output Layer
        if 4 in execute_stage:
            x = self.output_layer(x)
            if return_featuremaps:
                feature_maps['output_stage'] = x
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
    model = Backbone(input_size, embedding_size)

    return model


def perform_aggregation_branch(device, backbone_agg, all_views_stage_features):

    # Average pooling across views for each stage
    embeddings = None
    for stage_index, stage_features in enumerate(all_views_stage_features):
        if not stage_features:
            continue  # Skip if no features for this stage

        # Stack features from all views
        all_view_stage = torch.stack(stage_features, dim=0)  # [view, batch, c, w, h]
        all_view_stage = all_view_stage.permute(1, 0, 2, 3, 4)  # [batch, view, c, w, h]

        # Perform average pooling across views
        views_pooled_stage = all_view_stage.mean(dim=1)  # [batch, c, w, h]

        # If the spatial dimensions match a specific criterion, process with BACKBONE_agg
        if views_pooled_stage.shape[-1] == 7:
            embeddings = backbone_agg(views_pooled_stage, execute_stage={4})
            break

    return embeddings


def execute_model(device, backbone_reg, backbone_agg, inputs):
    # Initialize a dictionary to hold stage features for all views
    stage_to_index = {
        "input_stage": 0,
        "block_2": 1,
        "block_6": 2,
        "block_20": 3,
        "block_23": 4,
    }
    all_views_stage_features = [[] for _ in stage_to_index]
    for view in inputs:
        view = view.to(device)
        features_stages = backbone_reg(view, return_featuremaps=True)
        for stage, index in stage_to_index.items():
            if stage in features_stages:
                all_views_stage_features[index].append(features_stages[stage])

    # print(all_views_stage_features.shape)
    # batch_size = stage_featuremaps["stage_0"].shape[0]
    # visualize_feature_maps(stage_featuremaps, "C:\\Users\\Eduard\\Desktop\\Face", stage_names=["stage_0"], batch_idx=0)
    # for k,v in stage_featuremaps.items():
    #    print(k, v.shape)

    embeddings = perform_aggregation_branch(device, backbone_agg, all_views_stage_features)

    return embeddings
