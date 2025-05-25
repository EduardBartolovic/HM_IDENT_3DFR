import torch
import torch.nn as nn


class TransformerEmbeddingReducer(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=8, num_layers=1, dropout=0.1):
        super().__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        """
        x: Tensor of shape (B, V, 512)
        Returns: logits (B, num_classes)
        """
        x = self.encoder(x)  # (B, V, 512)
        x = x.mean(dim=1)  # average fusion -> (B, 512)
        x = self.dropout(x)  # (B, 512)
        embedding = self.linear(x)  # (B, 512)
        return embedding
