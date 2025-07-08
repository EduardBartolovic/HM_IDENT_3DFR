import random
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class MultiviewDataset(Dataset):
    def __init__(self, root_dir, num_views, transform=None, use_face_corr=True, shuffle_views=False):
        """
        Args:
            root_dir (string): Path to the root directory of the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_views = num_views
        self.face_cor_exist = False
        self.use_face_corr = use_face_corr
        self.class_to_idx = self._get_class_to_idx()
        self.data = self._load_data()
        self.classes = self._find_classes()
        self.shuffle_views = shuffle_views

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
        class_names = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        return {class_name: idx for idx, class_name in enumerate(class_names)}

    def _load_data(self):
        data = []
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                class_idx = self.class_to_idx[class_name]
                sha_groups = {}
                for filename in os.listdir(class_path):
                    if filename.endswith((".jpg", ".png", ".jpeg", ".webp")):
                        file_path = os.path.join(class_path, filename)
                        if os.path.isfile(file_path):
                            sha_hash = filename[:40]  # Extract SHA hash from filename
                            if sha_hash not in sha_groups:
                                sha_groups[sha_hash] = []
                            sha_groups[sha_hash].append(file_path)

                        assert re.match(r"^[-+]?\d+_[-+]?\d+$", os.path.basename(file_path)[40:-10]), f"perspective in dataset doesnt match convention: {file_path}"
                    else:
                        if filename.endswith(".npz"):
                            self.face_cor_exist = True

                # Append each grouped data point to the dataset
                for sha_hash, file_paths in sha_groups.items():
                    if len(file_paths) == self.num_views:
                        file_paths.sort()  # Sort the data so perspectives are always at the same position
                        data.append((file_paths, class_idx))
                    else:
                        raise ValueError(f"Dataset Mistake in: {file_paths} \n {len(file_paths)}: number of views doesnt match with {self.num_views}")

        if not self.use_face_corr:  # When use_face_corr is deactivated then ignore face_corr
            self.face_cor_exist = False

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_paths, class_name = self.data[idx]

        # Load all images in the set
        images = [Image.open(img_path).convert("RGB") for img_path in img_paths]

        # Load all face correspondences
        if self.face_cor_exist:
            facial_corr = torch.Tensor(np.array([np.load(img_path.replace("_image.jpg", "_corr.npz"))['corr'] for img_path in img_paths]))
        else:
            facial_corr = torch.Tensor([])

        # Load all perspectives
        perspectives = [os.path.basename(img_path)[40:-10] for img_path in img_paths]

        # Load Scan ids
        scan_id = os.path.basename(img_paths[0])[:40]

        if self.shuffle_views:
            if self.face_cor_exist:
                combined = list(zip(images, perspectives, facial_corr))
                random.shuffle(combined)
                images, perspectives, facial_corr = zip(*combined)
                facial_corr = torch.stack(facial_corr)
            else:
                combined = list(zip(images, perspectives))
                random.shuffle(combined)
                images, perspectives = zip(*combined)
            images = list(images)
            perspectives = list(perspectives)

        # Apply the transform if any
        if self.transform:
            images = [self.transform(img) for img in images]

        return images, class_name, perspectives, facial_corr, scan_id
