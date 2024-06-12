import torch
import torchvision
from torchvision.datasets.folder import default_loader


class ImageFolder4Channel(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        super(ImageFolder4Channel, self).__init__(root, transform, target_transform, loader)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        # Convert 3-channel image to 4-channel by duplicating the last channel
        if img.shape[0] == 3:
            img = torch.cat((img, img[-1:, :, :]), dim=0)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
