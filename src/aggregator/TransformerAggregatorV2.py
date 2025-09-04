import torch
from torch import nn
from torch.amp import autocast

from src.aggregator.MeanAggregator import MeanAggregator


class TransformerAggregatorV2(nn.Module):
    """
    Changes from V1 to V2: sinusoidal Positional Embeddings instead of learnable
    """
    def __init__(self,
                 feature_dim=512,
                 spatial_size=7,
                 num_views=26,
                 transformer_dim=512,
                 num_heads=8,
                 num_layers=3,
                 dropout=0.1,
                 use_mixed_precision=False):

        super().__init__()

        self.feature_dim = feature_dim
        self.spatial_size = spatial_size
        self.num_views = num_views
        self.tokens_per_view = spatial_size * spatial_size
        self.total_tokens = num_views * self.tokens_per_view
        self.use_mixed_precision = use_mixed_precision

        # Positional Embeddings
        self.register_buffer("spatial_pos_embed", get_2d_sincos_pos_embed(feature_dim, spatial_size).unsqueeze(0))  # (1, H*W, C)
        self.register_buffer("view_pos_embed", get_1d_sincos_pos_embed(feature_dim, torch.arange(num_views)).unsqueeze(0))  # (1, V, C)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=1024,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable fusion layer across views (e.g., X views → 1 aggregated vector)
        #self.fusion_layer = nn.Linear(num_views, 1)
        #self.fusion_layer = nn.Sequential(Linear(num_views, H), GELU(), Linear(H, 1))

    def forward(self, x):
        """
        x: (B, V, C, H, W)
        """
        with autocast('cuda', enabled=self.use_mixed_precision):
            B = x.size(0)

            # Flatten spatial dims
            x = x.view(B, self.num_views, self.feature_dim, self.tokens_per_view)  # (B, V, C, H*W)
            x = x.permute(0, 1, 3, 2)  # (B, V, H*W, C)
            x = x.reshape(B, self.total_tokens, self.feature_dim)  # (B, V*H*W, C)

            # Positional embeddings
            spatial_pos = self.spatial_pos_embed.repeat(1, self.num_views, 1)  # (1, V*H*W, C)
            view_pos = self.view_pos_embed.unsqueeze(2).repeat(1, 1, self.tokens_per_view, 1)  # (1, V, H*W, C)
            view_pos = view_pos.reshape(1, self.total_tokens, self.feature_dim)

            x = x + spatial_pos + view_pos  # Add positional encoding

            x = self.transformer(x)  # (B, V*H*W, C)

            # Reshape back to (B, V, H*W, C)
            x = x.view(B, self.num_views, self.tokens_per_view, self.feature_dim)

            # Fuse views together
            #x = x.permute(0, 2, 3, 1)  # (B, H*W, C, V)
            #x = self.fusion_layer(x)  # (B, H*W, C, 1)
            #x = x.squeeze(-1)  # (B, H*W, C)
            x = x.mean(dim=1)  # (B, H*W, C) mean pooling
            # x, _ = x.max(dim=1)  # (B, H*W, C) max pooling

            # Reshape to (B, C, H, W)
            x = x.permute(0, 2, 1).view(B, self.feature_dim, self.spatial_size, self.spatial_size)

            return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    embed_dim: output dimension for each position
    return:
    pos_embed: [grid_size*grid_size, embed_dim]
    """
    # generate grid
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.stack(grid, dim=0)  # (2, grid_size, grid_size)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    grid: (2, 1, H, W)
    """
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0].reshape(-1))  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1].reshape(-1))  # (H*W, D/2)
    return torch.cat([emb_h, emb_w], dim=1)


def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: (M,) position indices
    """
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim // 2.)
    omega = 1. / (10000**omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    return torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)


def make_transformerv2_aggregator(channels_list, num_views, agg_config, use_mixed_precision=False):
    """
    Factory method
    """
    if agg_config["ACTIVE_STAGES"]:
        activate_stages = agg_config["ACTIVE_STAGES"]
    else:
        activate_stages = (False, False, False, False, True)
    if agg_config["NUM_LAYERS"]:
        num_layers = agg_config["NUM_LAYERS"]
    else:
        num_layers = 3

    aggregators = []
    for idx, channels in enumerate(channels_list):
        if idx == 4 and activate_stages[idx]:
            aggregators.append(TransformerAggregatorV2(num_views=num_views+1, spatial_size=7, transformer_dim=512, feature_dim=512, num_layers=num_layers, use_mixed_precision=use_mixed_precision))
        elif idx == 3 and activate_stages[idx]:
            aggregators.append(TransformerAggregatorV2(num_views=num_views+1, spatial_size=14, transformer_dim=256, feature_dim=256, num_layers=num_layers, use_mixed_precision=use_mixed_precision))
        elif idx == 2 and activate_stages[idx]:
            aggregators.append(TransformerAggregatorV2(num_views=num_views+1, spatial_size=28, transformer_dim=128, feature_dim=128, num_layers=num_layers, use_mixed_precision=use_mixed_precision))
        elif idx == 1 and activate_stages[idx]:
            aggregators.append(TransformerAggregatorV2(num_views=num_views+1, spatial_size=56, transformer_dim=64, feature_dim=64, num_layers=num_layers, use_mixed_precision=use_mixed_precision))
        elif idx == 0 and activate_stages[idx]:
            aggregators.append(TransformerAggregatorV2(num_views=num_views, spatial_size=112, transformer_dim=64, feature_dim=64, num_layers=num_layers, use_mixed_precision=use_mixed_precision))
        else:
            aggregators.append(MeanAggregator(use_aggregator_branch=True))

    return aggregators