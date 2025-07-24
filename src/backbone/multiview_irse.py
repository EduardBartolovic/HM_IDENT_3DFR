import numpy as np
import torch
from torch import nn
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, PReLU, Dropout, Linear, BatchNorm1d
import torch.nn.functional as F

from src.backbone.model_irse import Bottleneck, bottleneck_IR, Flatten
from src.util.visualize_feature_maps import visualize_feature_maps, visualize_all_views_and_pooled


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]


def get_blocks():
    return [
        get_block(in_channel=64, depth=64, num_units=3),
        get_block(in_channel=64, depth=128, num_units=4),
        get_block(in_channel=128, depth=256, num_units=14),
        get_block(in_channel=256, depth=512, num_units=3)
    ]


class MultiviewIResnet(Module):
    def __init__(self, input_size, embedding_size=512):
        super(MultiviewIResnet, self).__init__()
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

    def align_featuremap(self, featuremap, grid):
        """
        Align a single feature map using transformation grid
        """
        map_x, map_y = np.array(grid)

        C, H, W = featuremap.shape
        # Stack the maps to create a grid: shape [H, W, 2]
        grid = np.stack((map_y, map_x), axis=-1)
        # Convert to torch tensor, normalize grid from pixel coords to [-1, 1]
        grid = torch.from_numpy(grid).unsqueeze(0)  # [1, H, W, 2]
        grid = grid * 2 / torch.tensor([W - 1, H - 1]) - 1  # Normalize to [-1, 1]

        # Resize grid to match featuremap shape (H, W)
        grid = F.interpolate(grid.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=True)
        grid = grid.permute(0, 2, 3, 1)  # [1, H, W, 2]

        grid = grid.to(featuremap.device).float()
        # grid_sample needs [N, C, H, W] input, and [N, H, W, 2] grid
        input_tensor = featuremap.unsqueeze(0)  # [1, C, H, W]
        # grid is in (x, y) order, but PyTorch wants (y, x)
        grid = grid[..., [1, 0]]  # swap last dimension
        # Use grid_sample to warp
        remapped = F.grid_sample(input_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return remapped.squeeze(0)  # back to [C, H, W]

    def align_featuremaps(self, featuremaps, face_corr, zero_position, device="cuda"):
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
                    aligned_batched_featuremaps[b, v] = self.align_featuremap(
                        view_featuremaps[v],
                        view_grid[v]
                    )

        return aligned_batched_featuremaps

    def aggregator(self, aggregators, stage_index, all_view_stage, perspectives, face_corr, use_face_corr):

        if use_face_corr:
            zero_position = np.where(np.array(perspectives)[:, 0] == '0_0')[0][0]
            if stage_index == 0:
                all_view_stage = self.align_featuremaps(all_view_stage, face_corr, zero_position)
            if stage_index == 1:
                all_view_stage = self.align_featuremaps(all_view_stage, face_corr, zero_position)
            if stage_index == 2:
                all_view_stage = self.align_featuremaps(all_view_stage, face_corr, zero_position)

        views_pooled_stage = aggregators[stage_index](all_view_stage)
        #visualize_all_views_and_pooled(all_view_stage, views_pooled_stage, output_dir=f"feature_map_outputs_{stage_index}")

        return views_pooled_stage

    def perform_aggregation_branch(self, backbone_agg, aggregators, all_views_stage_features, perspectives, face_corr, use_face_corr):

        x_1 = None  # 56
        x_2 = None  # 28
        x_3 = None  # 14
        x_4 = None  # 7
        for stage_index, stage_features in enumerate(all_views_stage_features):

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
            views_pooled_stage = self.aggregator(aggregators, stage_index, all_view_stage, perspectives, face_corr, use_face_corr)  # [batch, c, w, h]

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

    def execute_model(self, device, backbone_reg, backbone_agg, aggregators, inputs, perspectives, face_corr, use_face_corr):
        # Dictionary to hold stage features for all views
        stage_to_index = {
            "input_stage": 0,
            "block_2": 1,
            "block_6": 2,
            "block_20": 3,
            "block_23": 4,
            "output_stage": 5,
        }
        with torch.no_grad():
            all_views_stage_features = [[] for _ in stage_to_index]
            for view in inputs:
                view = view.to(device)
                features_stages = backbone_reg(view, return_featuremaps=True)
                for stage, index in stage_to_index.items():
                    if stage in features_stages:
                        all_views_stage_features[index].append(features_stages[stage])

        # visualize_feature_maps(all_views_stage_features, "E:\\Download", view_idx=0)
        embeddings_agg = self.perform_aggregation_branch(backbone_agg, aggregators, all_views_stage_features, perspectives, face_corr, use_face_corr)  # Embeddings of aggregator branch
        embeddings_reg = all_views_stage_features[5]  # Embeddings of regular branch

        return embeddings_reg, embeddings_agg


def IR_MV_50(input_size, embedding_size):
    """Constructs a ir-50 model.
    """
    model = MultiviewIResnet(input_size, embedding_size)

    return model