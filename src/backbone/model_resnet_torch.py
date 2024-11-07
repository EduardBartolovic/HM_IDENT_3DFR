import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet_50_torch(nn.Module):
    """
    Constructs a ResNet-50 model.
    """
    def __init__(self, encoding_size=512, pretrained=None):
        super(ResNet_50_torch, self).__init__()

        if pretrained == "IMAGENET1K_V1":
            weights = ResNet50_Weights.IMAGENET1K_V1
        elif pretrained == "IMAGENET1K_V2":
            weights = ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        if not isinstance(encoding_size, int):
            raise ValueError("encoding_size must be an integer")

        self.base_model = torch.nn.Sequential(*(list(resnet50(weights=weights).children())[:-1]))
        self.fc = nn.Linear(in_features=2048, out_features=encoding_size)

    def forward(self, x):
        # Extract embeddings from the base model
        embeddings = self.base_model(x)
        embeddings = embeddings.view(embeddings.size(0), -1)  # Flatten the output
        # Pass through the fully connected layer
        embeddings = self.fc(embeddings)
        return embeddings
