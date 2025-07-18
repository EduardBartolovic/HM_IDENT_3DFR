import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.backbone.iresnet_insight import iresnet50, iresnet34, iresnet18


class IR_MV_V2(nn.Module):
    def __init__(self, backbone_fn, embedding_size=512, fp16=False):
        super().__init__()
        self.backbone = backbone_fn(num_features=embedding_size, fp16=fp16)
        self.fp16 = fp16
        self.precision = torch.float16 if fp16 else torch.float32

    def forward(self, x, return_featuremaps=False, execute_stage=None):
        if execute_stage is None:
            execute_stage = {0, 1, 2, 3, 4, 5}

        feature_maps = {}

        with torch.amp.autocast('cuda', dtype=self.precision):
            if 0 in execute_stage:
                x = self.backbone.conv1(x)
                x = self.backbone.bn1(x)
                x = self.backbone.prelu(x)
                feature_maps['input_stage'] = x

            if 1 in execute_stage:
                x = self.backbone.layer1(x)
                feature_maps['stage_1'] = x

            if 2 in execute_stage:
                x = self.backbone.layer2(x)
                feature_maps['stage_2'] = x

            if 3 in execute_stage:
                x = self.backbone.layer3(x)
                feature_maps['stage_3'] = x

            if 4 in execute_stage:
                x = self.backbone.layer4(x)
                feature_maps['stage_4'] = x

            if 5 in execute_stage:
                x = self.backbone.bn2(x)
                x = torch.flatten(x, 1)
                x = self.backbone.dropout(x)
                x = self.backbone.fc(x.float() if self.fp16 else x)
                x = self.backbone.features(x)
                feature_maps['output_stage'] = x

        return feature_maps if return_featuremaps else x

    def align_featuremap(self, featuremap, grid):
        map_x, map_y = np.array(grid)
        C, H, W = featuremap.shape

        grid = np.stack((map_y, map_x), axis=-1)
        grid = torch.from_numpy(grid).unsqueeze(0)
        grid = grid * 2 / torch.tensor([W - 1, H - 1]) - 1

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

    def aggregator(self, aggregators, stage_index, all_view_stage, perspectives, face_corr, use_face_corr):
        if use_face_corr and stage_index in {0, 1, 2}:
            zero_position = np.where(np.array(perspectives)[:, 0] == '0_0')[0][0]
            all_view_stage = self.align_featuremaps(all_view_stage, face_corr, zero_position)
        return aggregators[stage_index](all_view_stage)

    def perform_aggregation_branch(self, backbone_agg, aggregators, all_views_stage_features, perspectives, face_corr, use_face_corr):
        x_1, x_2, x_3, x_4 = None, None, None, None

        for stage_index, stage_features in enumerate(all_views_stage_features):
            all_view_stage = torch.stack(stage_features, dim=0).permute(1, 0, 2, 3, 4)

            if all_view_stage.shape[-1] == 56 and x_1 is not None:
                all_view_stage = torch.cat((all_view_stage, x_1.unsqueeze(1)), dim=1)
            elif all_view_stage.shape[-1] == 28 and x_2 is not None:
                all_view_stage = torch.cat((all_view_stage, x_2.unsqueeze(1)), dim=1)
            elif all_view_stage.shape[-1] == 14 and x_3 is not None:
                all_view_stage = torch.cat((all_view_stage, x_3.unsqueeze(1)), dim=1)
            elif all_view_stage.shape[-1] == 7 and x_4 is not None:
                all_view_stage = torch.cat((all_view_stage, x_4.unsqueeze(1)), dim=1)

            views_pooled_stage = self.aggregator(aggregators, stage_index, all_view_stage, perspectives, face_corr, use_face_corr)

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
        stage_to_index = {"input_stage": 0, "stage_1": 1, "stage_2": 2, "stage_3": 3, "stage_4": 4, "output_stage": 5}

        with torch.no_grad():
            all_views_stage_features = [[] for _ in stage_to_index]
            for view in inputs:
                features_stages = backbone_reg(view.to(device), return_featuremaps=True)
                for stage, index in stage_to_index.items():
                    if stage in features_stages:
                        all_views_stage_features[index].append(features_stages[stage])

        embeddings_agg = self.perform_aggregation_branch(backbone_agg, aggregators, all_views_stage_features, perspectives, face_corr, use_face_corr)
        embeddings_reg = all_views_stage_features[5]
        return embeddings_reg, embeddings_agg


def IR_MV_V2_50(embedding_size=512, fp16=False):
    return IR_MV_V2(iresnet50, embedding_size, fp16)


def IR_MV_V2_34(embedding_size=512, fp16=False):
    return IR_MV_V2(iresnet34, embedding_size, fp16)


def IR_MV_V2_18(embedding_size=512, fp16=False):
    return IR_MV_V2(iresnet18, embedding_size, fp16)
