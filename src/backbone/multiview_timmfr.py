import numpy as np
import torch
import torch.nn as nn

from src.backbone.timmfr import timm_fr
from src.util.align_featuremaps import align_featuremaps


class MultiviewTimmFR(nn.Module):

    def __init__(self, device, aggregators, backbone_fn, stage_to_index, active_stages=None):
        super().__init__()
        self.backbone_agg = backbone_fn()
        self.backbone_reg = backbone_fn()
        self.aggregators = aggregators
        self.device = device
        if active_stages is None:
            self.active_stages = {0, 1, 2, 3, 4, 5}
        else:
            self.active_stages = active_stages

        self.stage_to_index = stage_to_index

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

        embeddings_agg = self.perform_aggregation_branch(all_views_stage_features, perspectives, face_corr, use_face_corr)
        embeddings_reg = all_views_stage_features[5]

        return embeddings_reg, embeddings_agg

    def eval(self):
        self.backbone_reg.eval()
        self.backbone_agg.eval()
        [i.eval() for i in self.aggregators]

    def aggregate(self, stage_index, all_view_stage, perspectives, face_corr, use_face_corr, embs=None):
        if use_face_corr and stage_index in {0, 1, 2}:
            zero_position = np.where(np.array(perspectives)[:, 0] == '0_0')[0][0]
            all_view_stage = align_featuremaps(all_view_stage, face_corr, zero_position)

        if embs:
            all_view_embs = torch.stack(embs, dim=0)  # [view, batch, d]
            all_view_embs = all_view_embs.permute(1, 0, 2)  # [batch, view, d]
            return self.aggregators[stage_index](all_view_stage, all_view_embs)
        else:
            return self.aggregators[stage_index](all_view_stage)

    def perform_aggregation_branch(self, all_views_stage_features, perspectives, face_corr, use_face_corr):
        x_prev = None
        prev_stage = None

        for stage_index, stage_features in enumerate(all_views_stage_features):
            if len(stage_features) == 0:
                continue

            all_view_stage = torch.stack(stage_features, dim=0)  # [view, batch, c, w, h]
            all_view_stage = all_view_stage.permute(1, 0, 2, 3, 4)  # [batch, view, c, w, h]

            # concat with previous stage if res matches
            if x_prev is not None and prev_stage == stage_index:
                all_view_stage = torch.cat((all_view_stage, x_prev.unsqueeze(1)), dim=1)

            # aggregate views
            views_pooled_stage = self.aggregate(stage_index, all_view_stage, perspectives, face_corr, use_face_corr, embs=all_views_stage_features[5])

            # run through backbone_agg for the *next* stage
            if stage_index < 4:  # stages 0–4 produce features
                x_prev, prev_stage = self.backbone_agg(views_pooled_stage, execute_stage={stage_index + 1}), stage_index + 1
            else:  # if next stage 5 -> produce embeddings
                embeddings = self.backbone_agg(views_pooled_stage, execute_stage={5})
                return embeddings

        raise ValueError("Illegal state")


def timm_mv(device, aggregators, embedding_size=512, active_stages=None):
    return MultiviewTimmFR(device, aggregators, timm_fr, {"stage_0": 0, "stage_1": 1, "stage_2": 2, "stage_3": 3, "stage_4": 4, "output_stage": 5}, active_stages=active_stages)

