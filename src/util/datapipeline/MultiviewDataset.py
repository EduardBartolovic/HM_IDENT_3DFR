import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class MultiviewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Path to the root directory of the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = self._get_class_to_idx()
        self.data = self._load_data()
        self.classes = self._find_classes()

    def _get_class_to_idx(self):
        """
        Maps class names to integers.
        """
        classes = sorted(os.listdir(self.root_dir))
        return {class_name: idx for idx, class_name in enumerate(classes)}

    def _find_classes(self):
        """
        Finds the class names in the dataset and assigns each a unique index.
        Returns:
            dict: A mapping from class names to class indices.
        """
        class_names = sorted(
            [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        return {class_name: idx for idx, class_name in enumerate(class_names)}

    def _load_data(self):
        data = []
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                class_idx = self.class_to_idx[class_name]
                # Loop through the sets in each class
                for set_name in os.listdir(class_path):
                    set_path = os.path.join(class_path, set_name)
                    if os.path.isdir(set_path):
                        images = []
                        # Loop through the images in each set
                        if len(os.listdir(set_path)) == 9:
                            for img_name in os.listdir(set_path):
                                img_path = os.path.join(set_path, img_name)
                                if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
                                    images.append(img_path)
                            # Store the class, set, and corresponding images
                            data.append((images, class_idx, set_name))
                        else:
                            print("Dataset Mistake:", set_path)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_paths, class_name, set_name = self.data[idx]

        # Load all images in the set
        images = [Image.open(img_path).convert("RGB") for img_path in img_paths]

        # Apply the transform if any
        if self.transform:
            images = [self.transform(img) for img in images]

        # Return images as a tensor batch along with class and set info
        #return images[0], class_name
        return images, class_name
