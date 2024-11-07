import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ViT_B_16(nn.Module):
    """
    Constructs a SqueezeNet model.
    """
    def __init__(self, encoding_size=512, pretrained=None):
        super(ViT_B_16, self).__init__()

        if pretrained == "IMAGENET1K_V1":
            weights = ViT_B_16_Weights.IMAGENET1K_V1
        else:
            weights = None

        if not isinstance(encoding_size, int):
            raise ValueError("encoding_size must be an integer")

        self.base_model = vit_b_16(num_classes=512, weights=weights)

    def forward(self, x):
        embeddings = self.base_model(x)
        return embeddings


def vit_b_16_torch(encoding_size=512, pretrained=None):
    return ViT_B_16(encoding_size, pretrained)
