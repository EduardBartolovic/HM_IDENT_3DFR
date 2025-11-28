import torch.nn as nn


class PoseAndFrontalizer(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=1024, out_emb_dim=512):
        super().__init__()

        # Shared encoder that processes the 512-dim input embedding
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Headpose regressor: predicts (x, y)
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)   # x, y
        )

        # Frontal embedding generator: predicts a new 512-dim embedding
        self.frontal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_emb_dim)
        )

    def forward(self, emb):
        """
        emb: [batch, 512] input embedding
        returns:
            pred_pose: [batch, 2]
            emb_front: [batch, 512]
        """
        features = self.shared(emb)

        pred_pose = self.pose_head(features)
        emb_front = self.frontal_head(features)

        return pred_pose, emb_front
