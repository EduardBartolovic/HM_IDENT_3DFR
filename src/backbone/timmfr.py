"""
========================== Original code from:=====================================================
Author: Anjith George
Institution: Idiap Research Institute, Martigny, Switzerland.

Copyright (C) 2023 Anjith George

This software is distributed under the terms described in the LICENSE file 
located in the parent directory of this source code repository. 

For inquiries, please contact the author at anjith.george@idiap.ch
===============================================================================
"""
import timm
import torch
import torch.nn as nn


class LoRaLin(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LoRaLin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        return x


def replace_linear_with_lowrank_recursive_2(model, rank_ratio=0.2):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and 'head' not in name:
            in_features = module.in_features
            out_features = module.out_features
            rank = max(2, int(min(in_features, out_features) * rank_ratio))
            bias = False
            if module.bias is not None:
                bias = True
            lowrank_module = LoRaLin(in_features, out_features, rank, bias)

            setattr(model, name, lowrank_module)
        else:
            replace_linear_with_lowrank_recursive_2(module, rank_ratio)


def replace_linear_with_lowrank_2(model, rank_ratio=0.2):
    replace_linear_with_lowrank_recursive_2(model, rank_ratio)
    return model


class TimmFRWrapperV2(nn.Module):
    """
    Wraps timm model
    """

    def __init__(self, model_name='edgenext_x_small', featdim=512):
        super().__init__()
        self.featdim = featdim
        self.model_name = model_name

        self.model = timm.create_model(self.model_name, features_only=True)

        self.channels = self.model.feature_info.channels()

        self.final_fc = nn.Linear(self.channels[-1], featdim)

    def forward(self, x, return_featuremaps=None, execute_stage=None):
        """
        Forward pass with optional featuremap return.

        Args:
            x (torch.Tensor): Input batch of images.
            return_featuremaps (set or None): Stages to return feature maps from (e.g., {1, 3}).
            execute_stage (set or None): Stages to actually execute.
        Returns:
            torch.Tensor or dict: Final embedding or dict of intermediate feature maps.
        """
        if execute_stage is None:
            execute_stage = {1, 2, 3, 4, 5}
        if return_featuremaps is None:
            return_featuremaps = set()

        feature_maps = {}

        if 1 in execute_stage:
            x = self.model.stem_0(x)
            x = self.model.stem_1(x)
            x = self.model.stages_0(x)
            if 1 in return_featuremaps:
                feature_maps['stage_1'] = x

        if 2 in execute_stage:
            x = self.model.stages_1(x)
            if 2 in return_featuremaps:
                feature_maps['stage_2'] = x

        if 3 in execute_stage:
            x = self.model.stages_2(x)
            if 3 in return_featuremaps:
                feature_maps['stage_3'] = x

        if 4 in execute_stage:
            x = self.model.stages_3(x)
            if 4 in return_featuremaps:
                feature_maps['stage_4'] = x

        if 5 in execute_stage:
            x = torch.mean(x, dim=[2, 3])
            x = self.final_fc(x)
            if 5 in return_featuremaps:
                feature_maps['output_stage'] = x

        return feature_maps if return_featuremaps else x


def get_timmfrv2(model_name, **kwargs):
    """
    Create an instance of TimmFRWrapperV2 with the specified `model_name` and additional arguments passed as `kwargs`.
    """
    return TimmFRWrapperV2(model_name=model_name, **kwargs)


def timm_fr(name="edgeface_xs_gamma_06"):
    if name == 'edgeface_xs_gamma_06':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_x_small'), rank_ratio=0.6)
    elif name == 'edgeface_xs_q':
        model = get_timmfrv2('edgenext_x_small')
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model
    elif name == 'edgeface_xxs':
        return get_timmfrv2('edgenext_xx_small')
    elif name == 'edgeface_base':
        return get_timmfrv2('edgenext_base')
    elif name == 'edgeface_xxs_q':
        model = get_timmfrv2('edgenext_xx_small')
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model
    elif name == 'edgeface_s_gamma_05':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_small'), rank_ratio=0.5)
    else:
        raise ValueError()
