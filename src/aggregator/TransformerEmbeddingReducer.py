import torch
import torch.nn as nn


class TransformerEmbeddingReducer(nn.Module):
    def __init__(self, embedding_dim=512, seq_len=25, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Positional Encoding (hier einfach learned, kannst auch sinusoidal nehmen)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embedding_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: Tensor von Shape (batch_size, seq_len=25, embedding_dim=512)
        Output: Tensor von Shape (batch_size, embedding_dim)
        """
        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, E)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, S+1, E)

        x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.transformer_encoder(x)  # (B, S+1, E)

        return x[:, 0, :]  # (B, E)
