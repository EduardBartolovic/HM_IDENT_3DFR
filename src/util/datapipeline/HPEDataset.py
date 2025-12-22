import os
from torch.utils.data import Dataset
from PIL import Image


class CoordImageDataset(Dataset):
    """
    Dataset that reads images from a folder structure (class-wise folders)
    and extracts the last coordinate pair (pitch, yaw) from the filenames.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing class subfolders.
            transform (callable, optional): Transform to apply to images.
        """
        self.samples = []
        self.transform = transform

        for class_name in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for filename in os.listdir(class_folder):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    filepath = os.path.join(class_folder, filename)
                    # Extract the last coordinate pair from filename
                    # Filename format: something#x_y#pitch_yaw
                    try:
                        last_pair = filename.split('#')[-1]
                        pitch_str, yaw_str = last_pair[:-3].split('_')
                        pitch, yaw = float(pitch_str), float(yaw_str)
                        self.samples.append((filepath, (pitch, yaw)))
                    except Exception as e:
                        print(f"Skipping file {filename}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, coords = self.samples[idx]
        img = Image.open(filepath).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, coords, filepath
