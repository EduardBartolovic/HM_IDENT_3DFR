import torch.nn as nn
from torchvision.models import Swin_V2_S_Weights, swin_v2_s


class Swin_V2_S(nn.Module):
    """
    Constructs a SqueezeNet model.
    """
    def __init__(self, encoding_size=512, pretrained=None):
        super(Swin_V2_S, self).__init__()

        if pretrained == "IMAGENET1K_V1":
            weights = Swin_V2_S_Weights.IMAGENET1K_V1
        else:
            weights = None

        if not isinstance(encoding_size, int):
            raise ValueError("encoding_size must be an integer")

        self.base_model = swin_v2_s(num_classes=512, weights=weights)

    def forward(self, x):
        embeddings = self.base_model(x)
        return embeddings


def swin_v2_s_torch(encoding_size=512, pretrained=None):
    return Swin_V2_S(encoding_size, pretrained)
