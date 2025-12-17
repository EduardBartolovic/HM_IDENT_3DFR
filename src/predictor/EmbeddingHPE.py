import torch.nn as nn

class EmbeddingHPE(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.01),

            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.1),

            nn.Linear(64, 2)  # yaw, pitch
        )

    def forward(self, x):
        return self.net(x)
