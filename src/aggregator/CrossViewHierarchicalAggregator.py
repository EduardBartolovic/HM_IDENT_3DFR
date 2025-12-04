import torch
import torch.nn as nn
from torch.amp import autocast

from src.aggregator.MeanAggregator import MeanAggregator


class CrossViewHierarchicalAggregator(nn.Module):
    """
    Hierarchical multi-view aggregator with residual mean-pooling:
      1) Intra-view spatial Transformer
      2) Cross-view Transformer over per-view [CLS]
      3) Gated fusion to get attention_fusion
      4) Residual blend: fused = (1-α)*mean + α*attention_fusion  (α ~ 0.1 at init)
    """

    def __init__(self,
                 feature_dim=512,
                 spatial_size=7,
                 num_views=26,
                 d_model=512,
                 num_heads_spatial=8,
                 num_layers_spatial=2,
                 num_heads_view=8,
                 num_layers_view=2,
                 ff_dim=1024,
                 dropout=0.1,
                 use_mixed_precision=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_size = spatial_size
        self.num_views = num_views
        self.tokens_per_view = spatial_size * spatial_size
        self.use_mixed_precision = use_mixed_precision
        self.d_model = d_model

        # projections
        self.input_proj  = nn.Linear(feature_dim, d_model)
        self.output_proj = nn.Linear(d_model, feature_dim)

        # --- intra-view (spatial) encoder ---
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.spatial_pos_embed = nn.Parameter(torch.empty(1, self.tokens_per_view, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.spatial_pos_embed, std=0.02)

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads_spatial,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=num_layers_spatial)

        # --- cross-view encoder ---
        self.view_pos_embed = nn.Parameter(torch.empty(1, num_views, d_model))
        self.agg_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.view_pos_embed, std=0.02)
        nn.init.normal_(self.agg_token, std=0.02)

        view_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads_view,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.view_encoder = nn.TransformerEncoder(view_layer, num_layers=num_layers_view)

        # --- view gating for attention fusion ---
        self.view_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        for m in self.view_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # --- residual blend coefficient α (sigmoid -> ~0.1 at init) ---
        # logit(0.1) ≈ -2.1972246
        self.residual_logit = nn.Parameter(torch.tensor(-3))#-2.1972246))

    def forward(self, x):
        """
        x: (B, V, C, H, W)
        returns: (B, C, H, W)
        """
        with autocast('cuda', enabled=self.use_mixed_precision):
            B, V, C, H, W = x.shape
            assert V == self.num_views, f"Expected {self.num_views} views, got {V}"
            assert H == self.spatial_size and W == self.spatial_size, "spatial_size mismatch"

            # --- tokens per view ---
            x_tokens = x.view(B, V, C, H * W).permute(0, 1, 3, 2).reshape(B * V, H * W, C)
            x_tokens = self.input_proj(x_tokens)  # (B*V, H*W, d)
            x_tokens = x_tokens + self.spatial_pos_embed.expand(B * V, -1, -1)
            cls = self.cls_token.expand(B * V, 1, -1)
            x_tokens = torch.cat([cls, x_tokens], dim=1)  # (B*V, 1+H*W, d)

            # --- intra-view encoding ---
            x_tokens = self.spatial_encoder(x_tokens)
            view_cls = x_tokens[:, 0]                  # (B*V, d)
            view_spatial = x_tokens[:, 1:]             # (B*V, H*W, d)

            # --- cross-view encoding on per-view [CLS] ---
            view_cls = view_cls.view(B, V, self.d_model)
            view_seq = view_cls + self.view_pos_embed.expand(B, -1, -1)
            agg = self.agg_token.expand(B, 1, self.d_model)
            view_seq = torch.cat([agg, view_seq], dim=1)   # (B, 1+V, d)
            view_seq = self.view_encoder(view_seq)         # (B, 1+V, d)
            views_out = view_seq[:, 1:]                    # (B, V, d)

            # --- attention-based fusion weights over views ---
            gate_logits = self.view_gate(views_out).squeeze(-1)  # (B, V)
            gate = torch.softmax(gate_logits, dim=1)             # (B, V)

            # project spatial tokens back to feature_dim for both paths
            view_spatial_C = self.output_proj(view_spatial)      # (B*V, H*W, C)
            view_spatial_C = view_spatial_C.view(B, V, H * W, C) # (B, V, H*W, C)

            # 1) baseline = mean over views  (strong inductive bias)
            baseline = view_spatial_C.mean(dim=1)                # (B, H*W, C)

            # 2) attention fusion = weighted sum over views
            gate_exp = gate.unsqueeze(-1).unsqueeze(-1)          # (B, V, 1, 1)
            attn_fused = (view_spatial_C * gate_exp).sum(dim=1)  # (B, H*W, C)

            # 3) residual blend: start near baseline, allow attention to refine
            alpha = torch.sigmoid(self.residual_logit)           # scalar in (0,1)
            fused = (1.0 - alpha) * baseline + alpha * attn_fused  # (B, H*W, C)

            # reshape back to (B, C, H, W)
            fused = fused.permute(0, 2, 1).view(B, C, H, W)
            return fused


def make_crossview_hierarchical_aggregators(channels_list, num_views, agg_config, use_mixed_precision=False):
    if agg_config.get("ACTIVE_STAGES"):
        activate_stages = agg_config["ACTIVE_STAGES"]
    else:
        activate_stages = (False, False, False, False, True)
    if agg_config.get("NUM_LAYERS"):
        num_layers = agg_config["NUM_LAYERS"]
    else:
        num_layers = 2

    aggs = []
    for idx, channels in enumerate(channels_list):
        if idx == 4 and activate_stages[idx]:
            aggs.append(CrossViewHierarchicalAggregator(
                num_views=num_views+1, spatial_size=7, feature_dim=512,
                d_model=512, num_heads_spatial=8, num_layers_spatial=num_layers,
                num_heads_view=8, num_layers_view=num_layers, ff_dim=2048,
                use_mixed_precision=use_mixed_precision))
        elif idx == 3 and activate_stages[idx]:
            aggs.append(CrossViewHierarchicalAggregator(
                num_views=num_views+1, spatial_size=14, feature_dim=256,
                d_model=256, num_heads_spatial=8, num_layers_spatial=num_layers,
                num_heads_view=8, num_layers_view=num_layers, ff_dim=1024,
                use_mixed_precision=use_mixed_precision))
        elif idx == 2 and activate_stages[idx]:
            aggs.append(CrossViewHierarchicalAggregator(
                num_views=num_views+1, spatial_size=28, feature_dim=128,
                d_model=192, num_heads_spatial=6, num_layers_spatial=num_layers,
                num_heads_view=6, num_layers_view=num_layers, ff_dim=768,
                use_mixed_precision=use_mixed_precision))
        elif idx == 1 and activate_stages[idx]:
            aggs.append(CrossViewHierarchicalAggregator(
                num_views=num_views+1, spatial_size=56, feature_dim=64,
                d_model=128, num_heads_spatial=4, num_layers_spatial=num_layers,
                num_heads_view=4, num_layers_view=num_layers, ff_dim=512,
                use_mixed_precision=use_mixed_precision))
        elif idx == 0 and activate_stages[idx]:
            aggs.append(CrossViewHierarchicalAggregator(
                num_views=num_views, spatial_size=112, feature_dim=64,
                d_model=128, num_heads_spatial=4, num_layers_spatial=num_layers,
                num_heads_view=4, num_layers_view=num_layers, ff_dim=512,
                use_mixed_precision=use_mixed_precision))
        else:
            # fall back to mean aggregator for stages not to be transformed
            aggs.append(MeanAggregator())

    return aggs
