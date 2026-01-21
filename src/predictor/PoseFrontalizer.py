import torch.nn as nn


class PoseFrontalizer(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.05),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class PoseFrontalizerWithPose(nn.Module):
    def __init__(self, embedding_dim=512, pose_dim=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim + pose_dim, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)
