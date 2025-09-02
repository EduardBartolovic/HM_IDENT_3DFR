import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.backbone.iresnet_insight import iresnet50, iresnet34, iresnet18, iresnet100


class IR_MV_V2(nn.Module):
    def __init__(self, device, aggregators, backbone_fn, embedding_size=512, fp16=True, active_stages=None):
        super().__init__()
        self.backbone_agg = backbone_fn(num_features=embedding_size, fp16=fp16)
        self.backbone_reg = backbone_fn(num_features=embedding_size, fp16=fp16)
        self.fp16 = fp16
        self.precision = torch.float16 if fp16 else torch.float32
        self.aggregators = aggregators
        self.device = device
        if active_stages is None:
            self.active_stages = {1, 2, 3, 4, 5}
        else:
            self.active_stages = active_stages

        self.stage_to_index = {"input_stage": 0, "stage_1": 1, "stage_2": 2, "stage_3": 3, "stage_4": 4, "output_stage": 5}

    def forward(self, inputs, perspectives, face_corr, use_face_corr):
        """
        inputs: list of every view: [(B,C,H,W), (B,C,H,W), (B,C,H,W), ...]

        output:
            embeddings_reg: (B, V*512)
            embeddings_agg  (B, 512)
        """
        with torch.no_grad():
            all_views_stage_features = [[] for _ in self.stage_to_index]
            for view in inputs:
                features_stages = self.backbone_reg(view.to(self.device), return_featuremaps=self.active_stages)
                for stage, index in self.stage_to_index.items():
                    if stage in features_stages:
                        all_views_stage_features[index].append(features_stages[stage])

        embeddings_agg = self.perform_aggregation_branch(all_views_stage_features, perspectives, face_corr, use_face_corr)
        embeddings_reg = all_views_stage_features[5]

        return embeddings_reg, embeddings_agg

    def eval(self):
        self.backbone_reg.eval()
        self.backbone_agg.eval()
        [i.eval() for i in self.aggregators]

    def align_featuremap(self, featuremap, grid):
        map_x, map_y = np.array(grid)
        C, H, W = featuremap.shape

        grid = np.stack((map_y, map_x), axis=-1)
        grid = torch.from_numpy(grid).unsqueeze(0)
        grid = grid * 2 / torch.tensor([W - 1, H - 1], dtype=torch.float32, device=featuremap.device) - 1

        grid = F.interpolate(grid.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=True)
        grid = grid.permute(0, 2, 3, 1)[..., [1, 0]].to(featuremap.device).float()

        input_tensor = featuremap.unsqueeze(0)
        remapped = F.grid_sample(input_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return remapped.squeeze(0)

    def align_featuremaps(self, featuremaps, face_corr, zero_position, device="cuda"):
        batch_size, num_views, num_channels, h, w = featuremaps.shape
        aligned = torch.empty((batch_size, num_views, num_channels, h, w), dtype=featuremaps.dtype, device=device)

        for b in range(batch_size):
            for v in range(num_views):
                if v == zero_position or v >= face_corr[b].shape[0]:
                    aligned[b, v] = featuremaps[b, v]
                else:
                    aligned[b, v] = self.align_featuremap(featuremaps[b, v], face_corr[b][v])
        return aligned

    def aggregate(self, stage_index, all_view_stage, perspectives, face_corr, use_face_corr):
        if use_face_corr and stage_index in {0, 1, 2}:
            zero_position = np.where(np.array(perspectives)[:, 0] == '0_0')[0][0]
            all_view_stage = self.align_featuremaps(all_view_stage, face_corr, zero_position)
        return self.aggregators[stage_index](all_view_stage)

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
                x_prev = None  # free reference

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


def IR_MV_V2_100(device, aggregators, embedding_size=512, fp16=False, active_stages=None):
    return IR_MV_V2(device, aggregators, iresnet100, embedding_size, fp16)


def IR_MV_V2_50(device, aggregators, embedding_size=512, fp16=False, active_stages=None):
    return IR_MV_V2(device, aggregators, iresnet50, embedding_size, fp16)


def IR_MV_V2_34(device, aggregators, embedding_size=512, fp16=False, active_stages=None):
    return IR_MV_V2(device, aggregators, iresnet34, embedding_size, fp16)


def IR_MV_V2_18(device, aggregators, embedding_size=512, fp16=False, active_stages=None):
    return IR_MV_V2(device, aggregators, iresnet18, embedding_size, fp16)
