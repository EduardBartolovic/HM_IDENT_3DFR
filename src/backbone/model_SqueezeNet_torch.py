import torch
import torch.nn as nn
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights


class SqueezeNet(nn.Module):
    """
    Constructs a SqueezeNet model.
    """
    def __init__(self, encoding_size=512, pretrained=None):
        super(SqueezeNet, self).__init__()

        if pretrained == "IMAGENET1K_V1":
            weights = SqueezeNet1_1_Weights.IMAGENET1K_V1
        else:
            weights = None

        if not isinstance(encoding_size, int):
            raise ValueError("encoding_size must be an integer")

        self.base_model = squeezenet1_1(num_classes=512, weights=weights)

    def forward(self, x):
        embeddings = self.base_model(x)
        return embeddings


def squeezenet_torch(encoding_size=512, pretrained=None):
    return SqueezeNet(encoding_size, pretrained)
