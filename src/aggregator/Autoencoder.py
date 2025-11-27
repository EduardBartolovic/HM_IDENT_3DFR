import torch.nn as nn
import torch.nn.functional as F


class FusionAEHead(nn.Module):
    """
    Autoencoder-style fusion head for compressing VxEMB multiview embeddings
    into a single EMB-D discriminative embedding.

    Input:  (batch, 8, 512)
    Output: (batch, 512)
    """

    def __init__(self, views=8, embedding_size=512, hidden_dim=1024, out_dim=512):
        super().__init__()

        self.in_dim = views * embedding_size

        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.in_dim)
        )

        self.return_reconstruction = False

    def encode(self, x):
        """
        x: (B, V, EMB)
        returns: fused 512-D embedding
        """
        b = x.size(0)
        x = x.reshape(b, -1)  # concat → V*EMB-D
        z = self.encoder(x)  # compress → EMB-D
        z = F.normalize(z, p=2, dim=1)  # L2 norm for discriminative power
        return z

    def decode(self, z):
        """
        z: (B, EMB)
        returns reconstructed V*EMB-D to apply AE reconstruction loss
        """
        rec = self.decoder(z)
        return rec

    def forward(self, x, return_reconstruction=False):
        """
        x: (B, V, EMB)
        """
        z = self.encode(x)
        if self.training or return_reconstruction or self.return_reconstruction:
            rec = self.decode(z)
            return z, rec
        return z


def make_ae_head(views=8, embedding_size=512, hidden_dim=1024, out_dim=512):
    return FusionAEHead(views, embedding_size, hidden_dim, out_dim)

