import numpy as np
import torch
import torch.nn as nn

from src.backbone.iresnet_insight import iresnet50, iresnet34, iresnet18, iresnet100
from src.backbone.model_irse import ir_50
from src.util.align_featuremaps import align_featuremaps


class MultiviewIResnet(nn.Module):

    def __init__(self, device, aggregators, backbone_fn, stage_to_index, embedding_size=512, fp16=True, active_stages=None):
        super().__init__()
        self.backbone_agg = backbone_fn(embedding_size=embedding_size, fp16=fp16)
        self.backbone_reg = backbone_fn(embedding_size=embedding_size, fp16=fp16)
        self.aggregators = aggregators
        self.device = device
        if active_stages is None:
            self.active_stages = {1, 2, 3, 4, 5}
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
            views_pooled_stage = self.aggregate(stage_index, all_view_stage, perspectives, face_corr, use_face_corr, embs=all_views_stage_features[5])
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


def ir_mv_50(device, aggregators, embedding_size=512, active_stages=None):
    return MultiviewIResnet(device, aggregators, ir_50, {"input_stage": 0, "block_2": 1, "block_6": 2, "block_20": 3, "block_23": 4, "output_stage": 5}, embedding_size, fp16=False, active_stages=active_stages)


def ir_mv_v2_100(device, aggregators, embedding_size=512, fp16=False, active_stages=None):
    return MultiviewIResnet(device, aggregators, iresnet100, {"input_stage": 0, "stage_1": 1, "stage_2": 2, "stage_3": 3, "stage_4": 4, "output_stage": 5}, embedding_size, fp16, active_stages)


def ir_mv_v2_50(device, aggregators, embedding_size=512, fp16=False, active_stages=None):
    return MultiviewIResnet(device, aggregators, iresnet50, {"input_stage": 0, "stage_1": 1, "stage_2": 2, "stage_3": 3, "stage_4": 4, "output_stage": 5}, embedding_size, fp16, active_stages)


def ir_mv_v2_34(device, aggregators, embedding_size=512, fp16=False, active_stages=None):
    return MultiviewIResnet(device, aggregators, iresnet34, {"input_stage": 0, "stage_1": 1, "stage_2": 2, "stage_3": 3, "stage_4": 4, "output_stage": 5}, embedding_size, fp16, active_stages)


def ir_mv_v2_18(device, aggregators, embedding_size=512, fp16=False, active_stages=None):
    return MultiviewIResnet(device, aggregators, iresnet18, {"input_stage": 0, "stage_1": 1, "stage_2": 2, "stage_3": 3, "stage_4": 4, "output_stage": 5}, embedding_size, fp16, active_stages)
