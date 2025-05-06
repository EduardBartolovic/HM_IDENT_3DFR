import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class FilteredMultiviewDataset(Dataset):
    def __init__(self, root_dir, num_views, transform=None, use_face_corr=True, allowed_views=None):
        """
        Args:
            root_dir (string): Path to the root directory of the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            allowed_views (list, optional): List of allowed view strings, e.g., ['0_0', '10_0'].
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_views = num_views
        self.face_cor_exist = False
        self.use_face_corr = use_face_corr
        self.allowed_views = allowed_views

        self.class_to_idx = self._get_class_to_idx()
        self.data = self._load_data()
        self.classes = self._find_classes()

    def _get_class_to_idx(self):
        classes = sorted(os.listdir(self.root_dir))
        return {class_name: idx for idx, class_name in enumerate(classes)}

    def _find_classes(self):
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
                        perspective = os.path.basename(filename)[40:-10]

                        if perspective not in self.allowed_views:
                            continue  # Skip if not in allowed views

                        file_path = os.path.join(class_path, filename)
                        if os.path.isfile(file_path):
                            sha_hash = filename[:40]
                            if sha_hash not in sha_groups:
                                sha_groups[sha_hash] = []
                            sha_groups[sha_hash].append(file_path)

                        assert re.match(r"^[-+]?\d+_[-+]?\d+$", perspective), \
                            f"Perspective format incorrect: {perspective} in file {filename}"
                    elif filename.endswith(".npz"):
                        self.face_cor_exist = True

                for sha_hash, file_paths in sha_groups.items():
                    if len(file_paths) == self.num_views:
                        file_paths.sort()  # Keep consistent order
                        data.append((file_paths, class_idx))
                    else:
                        raise ValueError(f"Mismatch in views for {sha_hash}: {len(file_paths)} found, expected {self.num_views}")

        if not self.use_face_corr:
            self.face_cor_exist = False

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_paths, class_idx = self.data[idx]
        images = [Image.open(img_path).convert("RGB") for img_path in img_paths]

        if self.face_cor_exist:
            facial_corr = torch.Tensor(np.array([
                np.load(img_path.replace("_image.jpg", "_corr.npz"))['corr'] for img_path in img_paths
            ]))
        else:
            facial_corr = torch.Tensor([])

        perspectives = [os.path.basename(img_path)[40:-10] for img_path in img_paths]
        scan_id = os.path.basename(img_paths[0])[:40]

        if self.transform:
            images = [self.transform(img) for img in images]

        return images, class_idx, perspectives, facial_corr, scan_id