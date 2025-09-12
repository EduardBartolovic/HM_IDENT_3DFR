"""
Multiview transformer aggregator
- Input: embeddings tensor (B, V, D) where each view is a D-dim token (default D=512)
- Output: fused embedding (B, D)

Features:
- Optional learnable CLS token (default True)
- Choice of positional encoding: 'sine' (sin-cos 1D) or 'learned' or None
- Uses PyTorch nn.TransformerEncoder with batch_first=True
- Pooling: 'cls' or 'mean' (when cls exists, 'cls' recommended)

Example usage at bottom.
"""

import torch
from torch import nn
import math


class SinePositionalEncoding1D(nn.Module):
    """1D sine-cosine positional encoding like in "Attention is All You Need",
    adapted for tokens along the view dimension.
    Input shape: (B, V, D)
    Output: (B, V, D)
    """

    def __init__(self, dim, max_len=512):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Sine pos enc dimension must be even")
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, V, D)
        v = x.size(1)
        return x + self.pe[:, :v, :].to(x.dtype)


class MultiviewTransformer(nn.Module):
    """Transformer aggregator for multiview embeddings.

    Args:
        emb_dim: dimensionality of each view embedding (default 512)
        num_heads: number of attention heads
        num_layers: number of transformer encoder layers
        mlp_ratio: feed-forward expansion ratio
        dropout: dropout probability
        use_cls_token: prepend a learnable CLS token and return it as pooled output
        pos_enc: 'sine', 'learned', or None
        pool: 'cls' or 'mean' (if pool=='mean', average token outputs)
    """

    def __init__(self,
                 emb_dim=512,
                 num_heads=8,
                 num_layers=4,
                 mlp_ratio=4,
                 dropout=0.1,
                 use_cls_token=True,
                 pos_enc='sine',  # or 'learned' or None
                 max_views=32,
                 pool='cls',
                 layer_norm_eps=1e-6):
        super().__init__()
        assert pool in ('cls', 'mean')
        self.emb_dim = emb_dim
        self.use_cls_token = use_cls_token
        self.pool = pool

        # optional projection if input dim != emb_dim (keeps API flexible)
        self.input_proj = None

        # cls token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        else:
            self.register_parameter('cls_token', None)

        # positional encoding
        if pos_enc == 'sine':
            self.pos_enc = SinePositionalEncoding1D(emb_dim, max_len=max_views + (1 if use_cls_token else 0))
        elif pos_enc == 'learned':
            self.pos_enc = nn.Parameter(torch.zeros(1, max_views + (1 if use_cls_token else 0), emb_dim))
            nn.init.trunc_normal_(self.pos_enc, std=0.02)
        else:
            self.pos_enc = None

        # Transformer encoder
        d_ff = int(emb_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=d_ff,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # final layer norm for stability
        self.norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps)

        # optional pool projection (keeps output dim exactly emb_dim)
        self.pool_proj = nn.Identity()

        # init weights
        self._init_weights()

    def _init_weights(self):
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        # if learned pos enc is a parameter it's initialized above

    def forward(self, x):
        """Forward
        x: (B, V, D_in) or (B, V, emb_dim)
        returns: (B, emb_dim)
        """
        # check dims
        if x.dim() != 3:
            raise ValueError('Input must be (B, V, D)')
        B, V, D_in = x.shape
        if D_in != self.emb_dim:
            # project to model dim
            if self.input_proj is None:
                self.input_proj = nn.Linear(D_in, self.emb_dim).to(x.device)
            x = self.input_proj(x)

        # optionally prepend cls token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,emb_dim)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, V+1, emb_dim)

        # positional encoding
        if self.pos_enc is not None:
            if isinstance(self.pos_enc, nn.Parameter):
                x = x + self.pos_enc[:, : x.size(1), :]
            else:
                x = self.pos_enc(x)

        # transformer encoder
        # mask is None (no padding) — if you have variable number of views replace padding values and pass key_padding_mask
        x = self.encoder(x)  # (B, T, emb_dim)

        # pooling
        if self.pool == 'cls' and self.use_cls_token:
            out = x[:, 0, :]
        else:
            # mean pool across token dimension
            out = x.mean(dim=1)

        out = self.norm(out)
        out = self.pool_proj(out)
        return out


def make_transformer_fusion():
    return MultiviewTransformer(emb_dim=512, num_heads=8, num_layers=2, use_cls_token=True, pos_enc='sine', max_views=16, pool='cls')


if __name__ == '__main__':
    # quick shape test
    model = MultiviewTransformer(emb_dim=512, num_heads=8, num_layers=2, use_cls_token=True, pos_enc='sine', max_views=16, pool='cls')
    dummy = torch.randn(4, 8, 512)  # batch=4, views=8, dim=512
    out = model(dummy)
    print('out shape', out.shape)  # should be (4,512)

    # test without cls token and mean pooling
    model2 = MultiviewTransformer(emb_dim=512, num_heads=8, num_layers=2, use_cls_token=False, pos_enc='sine', pool='mean')
    out2 = model2(dummy)
    print('out2 shape', out2.shape)

    # test projection when input dim != emb_dim
    dummy2 = torch.randn(2, 5, 256)
    model3 = MultiviewTransformer(emb_dim=512, num_heads=8, num_layers=2, use_cls_token=True, pos_enc='learned')
    out3 = model3(dummy2)
    print('out3 shape', out3.shape)
