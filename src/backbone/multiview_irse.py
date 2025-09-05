import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F

from src.backbone.model_irse import  IR_50
from src.util.visualize_feature_maps import visualize_feature_maps, visualize_all_views_and_pooled


class MultiviewIResnet(Module):
    def __init__(self, device, aggregators, input_size, embedding_size=512, active_stages=None):
        super(MultiviewIResnet, self).__init__()
        assert input_size[0] in [112], "input_size should be [112, 112]"

        self.backbone_agg = IR_50(input_size, embedding_size)
        self.backbone_reg = IR_50(input_size, embedding_size)
        self.aggregators = aggregators
        self.device = device
        if active_stages is None:
            self.active_stages = {1, 2, 3, 4, 5}
        else:
            self.active_stages = active_stages
        self.stage_to_index = {"input_stage": 0, "block_2": 1, "block_6": 2, "block_20": 3, "block_23": 4, "output_stage": 5}

    def forward(self, inputs, perspectives, face_corr, use_face_corr):
        """
        inputs: list of every view: [(B,C,H,W), (B,C,H,W), (B,C,H,W), ...]

        output:
            embeddings_reg: (V, B, 512)
            embeddings_agg  (B, 512)
        """

        with torch.no_grad():
            all_views_stage_features = [[] for _ in self.stage_to_index]
            for view in inputs:
                features_stages = self.backbone_reg(view.to(self.device), return_featuremaps=self.active_stages)
                for stage, index in self.stage_to_index.items():
                    if stage in features_stages:
                        all_views_stage_features[index].append(features_stages[stage])

        # visualize_feature_maps(all_views_stage_features, "E:\\Download", view_idx=0)
        embeddings_agg = self.perform_aggregation_branch(all_views_stage_features, perspectives, face_corr, use_face_corr)  # Embeddings of aggregator branch
        embeddings_reg = all_views_stage_features[5]  # Embeddings of regular branch

        return embeddings_reg, embeddings_agg

    def eval(self):
        self.backbone_reg.eval()
        self.backbone_agg.eval()
        [i.eval() for i in self.aggregators]

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

    def align_featuremaps(self, featuremaps, face_corr, zero_position):
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
        aligned_batched_featuremaps = torch.empty((batch_size, num_views, num_channels, h, w), dtype=featuremaps.dtype, device=self.device)
        for b in range(batch_size):
            view_featuremaps = featuremaps[b]  # Shape: [V, C, H, W]
            view_grid = face_corr[b]  # Shape: [V, H, W, 2]
            for v in range(num_views):
                if v == zero_position or v >= view_grid.shape[0]:  # Skip alignment if zero pose or merged features
                    aligned_batched_featuremaps[b, v] = view_featuremaps[v]
                else:
                    aligned_batched_featuremaps[b, v] = self.align_featuremap(
                        view_featuremaps[v],
                        view_grid[v]
                    )

        return aligned_batched_featuremaps

    def aggregate(self, stage_index, all_view_stage, perspectives, face_corr, use_face_corr):

        if use_face_corr and stage_index < 3:
            zero_position = np.where(np.array(perspectives)[:, 0] == '0_0')[0][0]
            all_view_stage = self.align_featuremaps(all_view_stage, face_corr, zero_position)

        views_pooled_stage = self.aggregators[stage_index](all_view_stage)
        #visualize_all_views_and_pooled(all_view_stage, views_pooled_stage, output_dir=f"feature_map_outputs_{stage_index}")

        return views_pooled_stage

    def perform_aggregation_branch(self, all_views_stage_features, perspectives, face_corr, use_face_corr):
        x_prev = None
        prev_res = None

        for stage_index, stage_features in enumerate(all_views_stage_features):
            if len(stage_features) == 0:
                continue

            all_view_stage = torch.stack(stage_features, dim=0)  # [view, batch, c, w, h]
            all_view_stage = all_view_stage.permute(1, 0, 2, 3, 4)  # [batch, view, c, w, h]
            res = all_view_stage.shape[-1]

            # concat with previous stage if res matches
            if x_prev is not None and res == prev_res:
                all_view_stage = torch.cat((all_view_stage, x_prev.unsqueeze(1)), dim=1)
                x_prev = None

            # aggregate views
            views_pooled_stage = self.aggregate(stage_index, all_view_stage, perspectives, face_corr, use_face_corr)
            res_out = views_pooled_stage.shape[-1]

            # pass through backbone_agg
            if res_out == 112:
                x_prev, prev_res = self.backbone_agg(views_pooled_stage, execute_stage={1}), 56
            elif res_out == 56:
                x_prev, prev_res = self.backbone_agg(views_pooled_stage, execute_stage={2}), 28
            elif res_out == 28:
                x_prev, prev_res = self.backbone_agg(views_pooled_stage, execute_stage={3}), 14
            elif res_out == 14:
                x_prev, prev_res = self.backbone_agg(views_pooled_stage, execute_stage={4}), 7
            elif res_out == 7:
                embeddings = self.backbone_agg(views_pooled_stage, execute_stage={5})
                return embeddings

        raise ValueError("Illegal state")


def IR_MV_50(device, aggregators, input_size, embedding_size=512, active_stages=None):
    """Constructs a ir-50 model.
    """
    return MultiviewIResnet(device, aggregators, input_size, embedding_size, active_stages=active_stages)
