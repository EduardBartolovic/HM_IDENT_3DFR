import torch
import torch.nn as nn


class ViewTransformer(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=8, num_layers=2, seq_length=25, dropout=0.1):
        super(ViewTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.seq_length = seq_length

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embedding_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection layer
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # x: (batch_size, seq_length, embedding_dim)
        x = x + self.positional_encoding[:, :self.seq_length, :]  # Add positional encoding
        x = x.permute(1, 0, 2)  # Transformer expects (seq_length, batch_size, embedding_dim)
        x = self.transformer_encoder(x)  # Pass through transformer
        x = x.mean(dim=0)  # Pooling (mean across sequence length)
        # Todo: Check other pooling
        x = self.fc(x)  # Final projection to 512-dimensional output
        return x

