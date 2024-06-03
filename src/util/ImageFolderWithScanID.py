from pathlib import Path
from torchvision import datasets


class ImageFolderWithSanID(datasets.ImageFolder):
    """Custom dataset that includes scan_id and perspective. Extends torchvision.datasets.ImageFolder"""

    def __getitem__(self, index):
        # This is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithSanID, self).__getitem__(index)
        image_path = Path(self.imgs[index][0])
        filename = image_path.name
        scan_id = filename[:40]  # sha-1 hash is always 40 of length
        perspective = filename[40:-10]
        # Make a new tuple that includes original and the scan_id, perspective
        data_tuple = (original_tuple + (scan_id, perspective))
        return data_tuple


