import time

import cv2
import numpy as np
import torch
from scipy.interpolate import Rbf
from torch import nn
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, PReLU, Dropout, Linear, BatchNorm1d
from tqdm import tqdm
import torch.nn.functional as F

from src.backbone.model_irse import Bottleneck, bottleneck_IR, Flatten
from src.util.visualize_feature_maps import visualize_feature_maps


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]


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
            execute_stage = {0, 1, 2, 3, 4, 5}

        feature_maps = {}

        # Stage 0: Input Layer
        if 0 in execute_stage:
            x = self.input_layer(x)
            feature_maps['input_stage'] = x

        # Body Layer Execution
        if {1, 2, 3, 4} & execute_stage:  # Check if body should be executed
            body_layers_to_execute = set()

            # Define layers to execute for each stage
            if 1 in execute_stage:
                body_layers_to_execute.update({0, 1, 2})  # Layers 0 to 2
            if 2 in execute_stage:
                body_layers_to_execute.update({3, 4, 5, 6})  # Layers 3 to 6
            if 3 in execute_stage:
                body_layers_to_execute.update(range(7, 21))  # Layers 7 to 20
            if 4 in execute_stage:
                body_layers_to_execute.update({21, 22, 23})
            # Execute only selected layers
            for i, layer in enumerate(self.body):
                if i in body_layers_to_execute:
                    x = layer(x)
                    if return_featuremaps and i in {2, 6, 20, 23}:
                        feature_maps[f'block_{i}'] = x

        # Stage 4: Output Layer
        if 5 in execute_stage:
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


def align_featuremap(featuremap, grid):
    """
    Align a single feature map using transformation grid
    """
    grid = grid.to("cuda")

    if grid.shape[:2] != featuremap.shape[1:]:
        _, target_height, target_width = featuremap.shape
        grid = grid.permute(2, 0, 1)  # [2, H, W]
        grid = F.interpolate(grid.unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=True)
        grid = grid.squeeze(0).permute(1, 2, 0)  # [H, W, 2]

    warped_featuremap = F.grid_sample(featuremap.unsqueeze(0), grid.unsqueeze(0), mode='bilinear', align_corners=True)

    return warped_featuremap.squeeze(0)


def align_featuremaps(featuremaps, face_corr, zero_position, device="cuda"):
    """
    Align feature maps in a batch using facial landmarks.

    Args:
        featuremaps: torch.Tensor of shape [B, V, C, H, W]
        face_corr: torch.Tensor of shape [B, V, H, W, 2]
        zero_position: position in array of the zero position
        device: device: Target device for the output tensor, e.g., 'cuda' or 'cpu'.

    Returns:
        Aligned feature maps as torch.Tensor of shape [B, V, C, H, W]
    """
    batch_size, num_views, num_channels, h, w = featuremaps.shape

    # Pre-allocate array for aligned feature maps
    aligned_batched_featuremaps = torch.empty((batch_size, num_views, num_channels, h, w), dtype=featuremaps.dtype, device=device)
    for b in range(batch_size):
        view_featuremaps = featuremaps[b]  # Shape: [V, C, H, W]
        view_grid = face_corr[b]  # Shape: [V, H, W, 2]
        for v in range(num_views):
            if v == zero_position or v >= view_grid.shape[0]: # Skip alignment if zero pose or merged features
                aligned_batched_featuremaps[b, v] = view_featuremaps[v]
            else:
                aligned_batched_featuremaps[b, v] = align_featuremap(
                    view_featuremaps[v],
                    view_grid[v]
                )

    return aligned_batched_featuremaps


def aggregator(aggregators, stage_index, all_view_stage, perspectives, face_corr):

    if face_corr.shape[1] > 0:
        zero_position = np.where(np.array(perspectives)[:,0] == '0_0')[0][0]
        if stage_index == 0:
            all_view_stage = align_featuremaps(all_view_stage, face_corr, zero_position)
        if stage_index == 1:
            all_view_stage = align_featuremaps(all_view_stage, face_corr, zero_position)

    views_pooled_stage = aggregators[stage_index](all_view_stage)

    # ========== Max ==========
    # views_pooled_stage = all_view_stage.max(dim=1)[0]

    # ========== Sum ==========
    # views_pooled_stage = all_view_stage.sum(dim=1)

    # ========== Weighted Average Pooling ==========
    # weights = torch.softmax(torch.randn(all_view_stage.size(1), device=all_view_stage.device), dim=0)  # Create weights for each view (shape: [view])
    # views_pooled_stage = torch.einsum('bvchw,v->bchw', all_view_stage, weights)  # Apply weights to views

    # ========== Global Context Pooling ==========
    # global_descriptor = all_view_stage.mean(dim=(1, 3, 4), keepdim=True)  # [batch, view, c, 1, 1]
    # weighted_views = all_view_stage * global_descriptor
    # views_pooled_stage = weighted_views.mean(dim=1)  # [batch, c, w, h]

    # ========== Attention ==========
    #attention_weights = torch.softmax(torch.matmul(all_view_stage.flatten(2), all_view_stage.flatten(2).transpose(-1, -2)), dim=-1)
    #views_pooled_stage = torch.matmul(attention_weights, all_view_stage.flatten(2)).view_as(all_view_stage)
    #views_pooled_stage = views_pooled_stage.mean(dim=1)  # [batch, c, w, h]

    return views_pooled_stage


def perform_aggregation_branch(backbone_agg, aggregators, all_views_stage_features, perspectives, face_corr):

    x_1 = None
    x_2 = None
    x_3 = None
    x_4 = None
    for stage_index, stage_features in enumerate(all_views_stage_features):
        # Stack features from all views
        all_view_stage = torch.stack(stage_features, dim=0)  # [view, batch, c, w, h]
        all_view_stage = all_view_stage.permute(1, 0, 2, 3, 4)  # [batch, view, c, w, h]

        if all_view_stage.shape[-1] == 56:
            all_view_stage = torch.cat((all_view_stage, x_1.unsqueeze(1)), dim=1)
        elif all_view_stage.shape[-1] == 28:
            all_view_stage = torch.cat((all_view_stage, x_2.unsqueeze(1)), dim=1)
        elif all_view_stage.shape[-1] == 14:
            all_view_stage = torch.cat((all_view_stage, x_3.unsqueeze(1)), dim=1)
        elif all_view_stage.shape[-1] == 7:
            all_view_stage = torch.cat((all_view_stage, x_4.unsqueeze(1)), dim=1)

        # Perform pooling across views
        views_pooled_stage = aggregator(aggregators, stage_index, all_view_stage, perspectives, face_corr)  # [batch, c, w, h]

        if views_pooled_stage.shape[-1] == 112:
            x_1 = backbone_agg(views_pooled_stage, execute_stage={1})
        elif views_pooled_stage.shape[-1] == 56:
            x_2 = backbone_agg(views_pooled_stage, execute_stage={2})
        elif views_pooled_stage.shape[-1] == 28:
            x_3 = backbone_agg(views_pooled_stage, execute_stage={3})
        elif views_pooled_stage.shape[-1] == 14:
            x_4 = backbone_agg(views_pooled_stage, execute_stage={4})
        elif views_pooled_stage.shape[-1] == 7:
            embeddings = backbone_agg(views_pooled_stage, execute_stage={5})
            return embeddings

    raise ValueError("Illegal State")


def execute_model(device, backbone_reg, backbone_agg, aggregators, inputs, perspectives, face_corr):
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

    # visualize_feature_maps(all_views_stage_features, "E:\\Download", batch_idx=0)

    embeddings = perform_aggregation_branch( backbone_agg, aggregators, all_views_stage_features, perspectives, face_corr)

    return embeddings
