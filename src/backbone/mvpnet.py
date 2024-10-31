# MIT License
#
# Copyright (c) 2024 Tobias Höfer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""Example custom cnn models. After importing this module, you should see the
models when using timm.list_models("custom*")

    Typical Usage:
        (simplest) - Train simplest cnn on fashion mnist
        poetry run python scripts/train.py --data-dir data/fmnist \
            --dataset torch/fashion_mnist --dataset-download \
            --model custom_simplest_model  --epochs 40 --input-size 1 28 28 \
            --log-wandb --lr 0.001 --no-aug



"""
from __future__ import annotations
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
from timm.layers import (
    trunc_normal_,
    AvgPool2dSame,
    DropPath,
    Mlp,
    GlobalResponseNormMlp,
    LayerNorm2d,
    LayerNorm,
    create_conv2d,
    get_act_layer,
    make_divisible,
    to_ntuple,
)
from timm.layers import NormMlpClassifierHead, ClassifierHead
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq
from timm.models._registry import generate_default_cfgs, register_model

__all__ = ["MVPNet"]  # model_registry will add each entrypoint fn to this

class Morph2d(nn.Module):
    """Summary of class here.

    Longer class information...
    Longer class information...

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding_mode: str = "constant",
        dilation: int | tuple | None = 1,
        percentile_init: float = 0.0,
        bias_init: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.percentile_scaling = 10.0
        self.bias_scaling = 10.0
        self.act_percentile = nn.Sigmoid()

        self.bias = nn.Parameter(
            nn.init.constant_(torch.empty(in_channels), val=bias_init)
        )
        self.percentile = nn.Parameter(
            nn.init.constant_(torch.empty(in_channels), val=percentile_init)
        )
        if isinstance(kernel_size, tuple):
            self.kernel_h = kernel_size[0]
            self.kernel_w = kernel_size[1]
        else:
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size

        self.kernel = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(1, in_channels, self.kernel_h, self.kernel_w),
                mean=1.0,
                std=0.05,
            )
        )

        # Replication padding is pretty similar to reflection padding, actually,
        # and attempts to achieve the same outcome: that the distribution of
        # your data is disturbed as little as possible (Liu et al., 2018).
        # Equivalent to TF symmetric padding.
        pad_h = int(torch.floor_divide(self.kernel_h, 2.0).item())
        pad_w = int(torch.floor_divide(self.kernel_w, 2.0).item())
        # (left, right, top, bottom)
        self.pad = (pad_w, pad_w, pad_h, pad_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (N, C, H, W)
        batch = x.shape[0]
        channel = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        x = F.pad(x, self.pad, mode=self.padding_mode)
        # Broadcasting bias.
        x = x + self.bias.view(self.bias.shape[0], 1, 1) * self.bias_scaling
        # Convolution is equivalent with:
        # Unfold + Matrix Multiplication + Fold (or view to output shape)
        # Consider a batched input tensor of shape (N,C,*), where N is the
        # batch dimension, C is the channel dimension, and * represent
        # arbitrary spatial dimensions. Outputs: (N, V, L)) with V equals
        # total number of values within each block, and L the total number
        # of blocks.
        x = F.unfold(
            input=x, kernel_size=(self.kernel_h, self.kernel_w)
        )  # todo stride dil
        # Blocks as rows
        x = x.transpose(1, 2)
        x = x.mul(self.kernel.reshape(self.kernel.shape[0], -1))

        # reshape: N, L, C, V
        x = x.view(x.shape[0], x.shape[1], channel, -1)
        x_max = torch.max(x, -1).values
        x_min = torch.min(x, -1).values

        x = x_min + (x_max - x_min) * self.act_percentile(
            self.percentile * self.percentile_scaling
        )

        x = x.transpose(1, 2)
        x = x.view(batch, channel, height, width)

        x = x - self.bias.view(self.bias.shape[0], 1, 1) * self.bias_scaling

        return x





class Aggregating(nn.Module):
    def __init__(self, dim=1):
        super(Aggregating, self).__init__()
        self.dim = dim

    def forward(self, *inputs):
        # Concatenate along the channel dimension
        return torch.cat(inputs, dim=1)
      

class Downsample(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = (
                AvgPool2dSame
                if avg_stride == 1 and dilation > 1
                else nn.AvgPool2d
            )
            self.pool = avg_pool_fn(
                2, avg_stride, ceil_mode=True, count_include_pad=False
            )
        else:
            self.pool = nn.Identity()

        if in_chs != out_chs:
            self.conv = create_conv2d(in_chs, out_chs, 1, stride=1)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class MVPBlock(nn.Module):
    """MVP Block.
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: Optional[int] = None,
        cardinality_weight: float = 0.1,
        naive_splitting: bool = False,
        mkernel_sizes: int = 3,
        kernel_size: int = 7,
        stride: int = 1,
        dilation: Union[int, Tuple[int, int]] = (1, 1),
        mlp_ratio: float = 4,
        conv_mlp: bool = False,
        conv_bias: bool = True,
        use_grn: bool = False,
        ls_init_value: Optional[float] = 1e-6,
        act_layer: Union[str, Callable] = "gelu",
        norm_layer: Optional[Callable] = None,
        drop_path: float = 0.0,
    ):
        """
        Args:
            in_chs: Block input channels.
            out_chs: Block output channels (same as in_chs if None).
            cardinality_weight: Denotes the ratio or proportion of each path.
            naive_splitting: If set, disable learned splitting.
            kernel_size: Depthwise convolution kernel size.
            stride: Stride of depthwise convolution.
            dilation: Tuple specifying input and output dilation of block.
            mlp_ratio: MLP expansion ratio.
            conv_mlp: Use 1x1 convolutions for MLP and a NCHW compatible norm layer if True.
            conv_bias: Apply bias for all convolution (linear) layers.
            use_grn: Use GlobalResponseNorm in MLP (from ConvNeXt-V2)
            ls_init_value: Layer-scale init values, layer-scale applied if not None.
            act_layer: Activation layer.
            norm_layer: Normalization layer (defaults to LN if not specified).
            drop_path: Stochastic depth probability.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            norm_layer = LayerNorm2d if conv_mlp else LayerNorm
        mlp_layer = partial(
            GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp
        )
        self.use_conv_mlp = conv_mlp
        self.naive_splitting = naive_splitting
        if cardinality_weight == 0.0:
            self.conv_chs = in_chs
            self.use_conv_layers = True
            self.use_morph_layers = False
        elif cardinality_weight == 1.0:
            self.morph_chs = in_chs
            self.use_conv_layers = False
            self.use_morph_layers = True
        else:
            self.morph_chs = int(cardinality_weight * in_chs)
            self.conv_chs = in_chs - self.morph_chs
            self.use_morph_layers = True
            self.use_conv_layers = True

        if self.use_conv_layers:
            # Transformation.
            self.conv_dw = create_conv2d(
                self.conv_chs,
                self.conv_chs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation[0],
                depthwise=True,
                bias=conv_bias,
            )
            self.conv_norm = norm_layer(self.conv_chs)
          

        if self.use_morph_layers:
            # Transformation.
            self.morph_dw = Morph2d(
                in_channels=self.morph_chs, kernel_size=mkernel_sizes
            )
            self.morph_norm = norm_layer(self.morph_chs) 
             
     
        if self.use_conv_layers and self.use_morph_layers:
            # Multi-Path.
            self.splitting = True
            # Splitting.
            if not self.naive_splitting:
                self.conv_splitting = nn.Conv2d(
                    in_chs, self.conv_chs, 1, padding="same"
                )
                self.morph_splitting = nn.Conv2d(
                    in_chs, self.morph_chs, 1, padding="same"
                )
            self.aggregating = Aggregating(dim=1)
        else:
            # Single-Path.
            self.splitting = False

        # Inverted bottleneck
        self.mlp = mlp_layer(
                in_chs, int(mlp_ratio * in_chs), act_layer=act_layer
                )

        self.gamma = (
            nn.Parameter(ls_init_value * torch.ones(out_chs))
            if ls_init_value is not None
            else None
        )
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(
                in_chs, out_chs, stride=stride, dilation=dilation[0]
            )
        else:
            self.shortcut = nn.Identity()
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x):
        shortcut = x
        # Transforming: Multi-path.
        if self.use_conv_layers:  # convolutional path.
            if self.splitting:
                # Splitting.
                if self.naive_splitting:
                    x_conv = x[:, : self.conv_chs, :, :]
                else:
                    x_conv = self.conv_splitting(x)
            else:
                x_conv = x
            # Transforming.
            x_conv = self.conv_dw(x_conv)
            x_conv = self.conv_norm(x_conv)
                
        if self.use_morph_layers:  # morphological path.
            if self.splitting:
                # Splitting.
                if self.naive_splitting:
                    x_morph = x[:, self.conv_chs :, :, :]
                else:
                    x_morph = self.morph_splitting(x)
            else:
                x_morph = x
            # Transforming.
            x_morph = self.morph_dw(x_morph)
            x_morph = self.morph_norm(x_morph)

        # Aggregating.
        if self.splitting:
            # Concatenate along the channel dimension
            x = self.aggregating(x_conv, x_morph)
        elif self.use_conv_layers and not self.use_morph_layers:
            x = x_conv
        else:
            x = x_morph

        # Inverted bottleneck.
        if self.use_conv_mlp:
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        # Residual connection.
        x = self.drop_path(x) + self.shortcut(shortcut)
        return x


class MVPStage(nn.Module):

    def __init__(
        self,
        in_chs,
        out_chs,
        cardinality_weight=0.1,
        naive_splitting=False,
        mkernel_sizes=3,
        kernel_size=7,
        stride=2,
        depth=2,
        dilation=(1, 1),
        drop_path_rates=None,
        ls_init_value=1.0,
        conv_mlp=False,
        conv_bias=True,
        use_grn=False,
        act_layer="gelu",
        norm_layer=None,
        norm_layer_cl=None,
    ):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = (
                "same" if dilation[1] > 1 else 0
            )  # same padding needed if dilation used
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                create_conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=ds_ks,
                    stride=stride,
                    dilation=dilation[0],
                    padding=pad,
                    bias=conv_bias,
                ),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.0] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(
                MVPBlock(
                    in_chs=in_chs,
                    out_chs=out_chs,
                    cardinality_weight=cardinality_weight,
                    naive_splitting=naive_splitting,
                    mkernel_sizes=mkernel_sizes,
                    kernel_size=kernel_size,
                    dilation=dilation[1],
                    drop_path=drop_path_rates[i],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    conv_bias=conv_bias,
                    use_grn=use_grn,
                    act_layer=act_layer,
                    norm_layer=norm_layer if conv_mlp else norm_layer_cl,
                )
            )
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class MVPNet(nn.Module):
    r"""ConvNeXt + Morphology
    A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 512,
        global_pool: str = "avg",
        output_stride: int = 32,
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        dims: Tuple[int, ...] = (96, 192, 384, 768),
        cardinality_weight: float = 0.1,
        naive_splitting: bool = False,
        mkernel_sizes: int = 3,
        kernel_sizes: Union[int, Tuple[int, ...]] = 7,
        ls_init_value: Optional[float] = 1e-6,
        stem_type: str = "patch",
        patch_size: int = 4,
        head_init_scale: float = 1.0,
        head_norm_first: bool = False,
        head_hidden_size: Optional[int] = None,
        conv_mlp: bool = False,
        conv_bias: bool = True,
        use_grn: bool = False,
        act_layer: Union[str, Callable] = "gelu",
        norm_layer: Optional[Union[str, Callable]] = None,
        norm_eps: Optional[float] = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            global_pool: Global pooling type.
            output_stride: Output stride of network, one of (8, 16, 32).
            depths: Number of blocks at each stage.
            dims: Feature dimension at each stage.
            cardinality_weight: Denotes the ratio or proportion of each path.
            kernel_sizes: Depthwise convolution kernel-sizes for each stage.
            ls_init_value: Init value for Layer Scale, disabled if None.
            stem_type: Type of stem.
            patch_size: Stem patch size for patch stem.
            head_init_scale: Init scaling value for classifier weights and biases.
            head_norm_first: Apply normalization before global pool + head.
            head_hidden_size: Size of MLP hidden layer in head if not None and head_norm_first == False.
            conv_mlp: Use 1x1 conv in MLP, improves speed for small networks w/ chan last.
            conv_bias: Use bias layers w/ all convolutions.
            use_grn: Use Global Response Norm (ConvNeXt-V2) in MLP.
            act_layer: Activation layer type.
            norm_layer: Normalization layer type.
            drop_rate: Head pre-classifier dropout rate.
            drop_path_rate: Stochastic depth drop rate.
        """
        super().__init__()
        assert output_stride in (8, 16, 32)
        kernel_sizes = to_ntuple(4)(kernel_sizes)
        if norm_layer is None:
            norm_layer = LayerNorm2d
            norm_layer_cl = norm_layer if conv_mlp else LayerNorm
            if norm_eps is not None:
                norm_layer = partial(norm_layer, eps=norm_eps)
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)
        else:
            assert (
                conv_mlp
            ), "If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input"
            norm_layer_cl = norm_layer
            if norm_eps is not None:
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        assert stem_type in ("patch", "overlap", "overlap_tiered")
        if stem_type == "patch":
            # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    dims[0],
                    kernel_size=patch_size,
                    stride=patch_size,
                    bias=conv_bias,
                ),
                norm_layer(dims[0]),
            )
            stem_stride = patch_size
        else:
            mid_chs = (
                make_divisible(dims[0] // 2)
                if "tiered" in stem_type
                else dims[0]
            )
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    mid_chs,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=conv_bias,
                ),
                nn.Conv2d(
                    mid_chs,
                    dims[0],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=conv_bias,
                ),
                norm_layer(dims[0]),
            )
            stem_stride = 4

        self.stages = nn.Sequential()
        dp_rates = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, sum(depths)).split(
                depths
            )
        ]
        stages = []
        prev_chs = dims[0]
        curr_stride = stem_stride
        dilation = 1
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(
                MVPStage(
                    prev_chs,
                    out_chs,
                    cardinality_weight=cardinality_weight,
                    naive_splitting=naive_splitting,
                    mkernel_sizes=mkernel_sizes,
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    dilation=(first_dilation, dilation),
                    depth=depths[i],
                    drop_path_rates=dp_rates[i],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    conv_bias=conv_bias,
                    use_grn=use_grn,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_layer_cl=norm_layer_cl,
                )
            )
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [
                dict(
                    num_chs=prev_chs,
                    reduction=curr_stride,
                    module=f"stages.{i}",
                )
            ]
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs

        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
        if head_norm_first:
            assert not head_hidden_size
            self.norm_pre = norm_layer(self.num_features)
            self.head = ClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
            )
        else:
            self.norm_pre = nn.Identity()
            self.head = NormMlpClassifierHead(
                self.num_features,
                512,
                hidden_size=head_hidden_size,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
                norm_layer=norm_layer,
                act_layer="gelu",
            )
        named_apply(
            partial(_init_weights, head_init_scale=head_init_scale), self
        )

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^stem",
            blocks=(
                r"^stages\.(\d+)"
                if coarse
                else [
                    (r"^stages\.(\d+)\.downsample", (0,)),  # blocks
                    (r"^stages\.(\d+)\.blocks\.(\d+)", None),
                    (r"^norm_pre", (99999,)),
                ]
            ),
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        nn.init.zeros_(module.bias)
        if name and "head." in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    """Remap FB checkpoints -> timm"""
    if "head.norm.weight" in state_dict or "norm_pre.weight" in state_dict:
        return state_dict  # non-FB checkpoint
    if "model" in state_dict:
        state_dict = state_dict["model"]

    out_dict = {}
    if "visual.trunk.stem.0.weight" in state_dict:
        out_dict = {
            k.replace("visual.trunk.", ""): v
            for k, v in state_dict.items()
            if k.startswith("visual.trunk.")
        }
        if "visual.head.proj.weight" in state_dict:
            out_dict["head.fc.weight"] = state_dict["visual.head.proj.weight"]
            out_dict["head.fc.bias"] = torch.zeros(
                state_dict["visual.head.proj.weight"].shape[0]
            )
        elif "visual.head.mlp.fc1.weight" in state_dict:
            out_dict["head.pre_logits.fc.weight"] = state_dict[
                "visual.head.mlp.fc1.weight"
            ]
            out_dict["head.pre_logits.fc.bias"] = state_dict[
                "visual.head.mlp.fc1.bias"
            ]
            out_dict["head.fc.weight"] = state_dict[
                "visual.head.mlp.fc2.weight"
            ]
            out_dict["head.fc.bias"] = torch.zeros(
                state_dict["visual.head.mlp.fc2.weight"].shape[0]
            )
        return out_dict

    import re

    for k, v in state_dict.items():
        k = k.replace("downsample_layers.0.", "stem.")
        k = re.sub(r"stages.([0-9]+).([0-9]+)", r"stages.\1.blocks.\2", k)
        k = re.sub(
            r"downsample_layers.([0-9]+).([0-9]+)",
            r"stages.\1.downsample.\2",
            k,
        )
        k = k.replace("dwconv", "conv_dw")
        k = k.replace("pwconv", "mlp.fc")
        if "grn" in k:
            k = k.replace("grn.beta", "mlp.grn.bias")
            k = k.replace("grn.gamma", "mlp.grn.weight")
            v = v.reshape(v.shape[-1])
        k = k.replace("head.", "head.fc.")
        if k.startswith("norm."):
            k = k.replace("norm", "head.norm")
        if v.ndim == 2 and "head" not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v

    return out_dict


def _create_mvpnet(variant, pretrained=False, **kwargs):
    if kwargs.get("pretrained_cfg", "") == "fcmae":
        # NOTE fcmae pretrained weights have no classifier or final norm-layer (`head.norm`)
        # This is workaround loading with num_classes=0 w/o removing norm-layer.
        kwargs.setdefault("pretrained_strict", False)

    model = build_model_with_cfg(
        MVPNet,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )
    return model


# Params (M): 3.7
# GMACs: 0.6
# Activations (M): 3.8
@register_model
def mvpnet_atto(pretrained=False, **kwargs) -> MVPNet:
    # timm femto variant (NOTE: still tweaking depths, will vary between 3-4M param, current is 3.7M
    model_args = dict(
        depths=(2, 2, 6, 2),
        dims=(40, 80, 160, 320),
        conv_mlp=True,
    )
    model = _create_mvpnet(
        "mvpnet_atto", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


# Params (M): 5.2
# GMACs: 0.8
# Activations (M): 4.6
@register_model
def mvpnet_femto(pretrained=False, **kwargs) -> MVPNet:
    # timm femto variant
    model_args = dict(
        depths=(2, 2, 6, 2),
        dims=(48, 96, 192, 384),
        conv_mlp=True,
    )
    model = _create_mvpnet(
        "mvpnet_femto", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


# Params (M): 9.1
# GMACs: 1.4
# Activations (M): 6.1
@register_model
def mvpnet_pico(pretrained=False, **kwargs) -> MVPNet:
    # timm pico variant
    model_args = dict(
        depths=(2, 2, 6, 2),
        dims=(64, 128, 256, 512),
        conv_mlp=True,
    )
    model = _create_mvpnet(
        "mvpnet_pico", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


# Params (M): 15.6
# GMACs: 2.5
# Activations (M): 8.4
@register_model
def mvpnet_nano(pretrained=False, **kwargs) -> MVPNet:
    # timm nano variant with standard stem and head
    model_args = dict(
        depths=(2, 2, 8, 2),
        dims=(80, 160, 320, 640),
        conv_mlp=True,
    )
    model = _create_mvpnet(
        "mvpnet_nano", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


# Params (M): 28.6
# GMACs: 4.5
# Activations (M): 13.4
@register_model
def mvpnet_tiny(pretrained=False, **kwargs) -> MVPNet:
    model_args = dict(
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        conv_mlp=True
    )
    model = _create_mvpnet(
        "mvpnet_tiny", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model
