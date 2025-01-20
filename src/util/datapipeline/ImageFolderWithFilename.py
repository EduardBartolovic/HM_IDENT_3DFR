import os

from torchvision.datasets import ImageFolder


class ImageFolderWithFilename(ImageFolder):
    def __getitem__(self, index):
        # Get the original tuple (image, label)
        img, label = super().__getitem__(index)

        # Get the image file path
        file_path, _ = self.samples[index]

        filename = os.path.basename(file_path)

        name, number = filename.rsplit('_', 1)

        number = int(number.split('.')[0])
        return img, name, number
