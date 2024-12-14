from pathlib import Path

from src.util.datapipeline.ImageFolder4Channel import ImageFolder4Channel


class ImageFolderRGBDWithScanID(ImageFolder4Channel):
    """Custom dataset that includes scan_id and perspective. Extends torchvision.datasets.ImageFolder"""

    def __getitem__(self, index):
        original_tuple = super(ImageFolderRGBDWithScanID, self).__getitem__(index)
        image_path = Path(self.imgs[index][0])
        filename = image_path.name
        scan_id = filename[:40]  # sha-1 hash is always 40 of length
        perspective = filename[40:-10]
        # Make a new tuple that includes original and the scan_id, perspective
        data_tuple = (original_tuple + (scan_id, perspective))
        return data_tuple


