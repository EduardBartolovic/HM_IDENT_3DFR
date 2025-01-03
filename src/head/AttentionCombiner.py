import torch
from torch import nn


class AttentionCombiner(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionCombiner, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)

    def forward(self, x):
        x = x.transpose(0, 1)
        attention_output, _ = self.attention(x,x,x)
        return torch.mean(attention_output, dim=0)