import os
import re
import random
from collections import defaultdict

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
        self.use_face_corr = use_face_corr
        self.shuffle_views = shuffle_views
        self.face_cor_exist = False

        self.valid_image_ext = {".jpg", ".jpeg", ".png", ".webp"}
        self.filename_regex = re.compile(r"^[-+]?\d+_[-+]?\d+$")

        self.classes, self.class_to_idx = self._find_classes()
        self.data = self._load_data()

    def _find_classes(self):
        """
        Finds class names and maps them to integer labels.
        """
        class_names = sorted(
            entry.name for entry in os.scandir(self.root_dir) if entry.is_dir()
        )
        return class_names, {name: i for i, name in enumerate(class_names)}

    def _load_data(self):
        data = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            sha_groups = defaultdict(list)

            for filename in os.scandir(class_path):
                if filename.is_file():
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in self.valid_image_ext:
                        match = self.filename_regex.match(filename.name[16:-4].split("#")[0])
                        if not match:
                            raise ValueError(f"Invalid filename format: {filename.name}")
                        sha_hash = filename.name[:15]
                        sha_groups[sha_hash].append(filename.path)

                    elif ext == ".npz":
                        self.face_cor_exist = True

            # Append each grouped data point to the dataset
            for file_paths in sha_groups.values():
                if len(file_paths) == self.num_views:
                    data.append((sorted(file_paths, key=_sort_key), class_idx))  # Sort the data so perspectives are always at the same position
                else:
                    raise ValueError(f"Incorrect number of views ({len(file_paths)}), expected {self.num_views}: {file_paths}")

        if not self.use_face_corr:  # When use_face_corr is deactivated then ignore face_corr
            self.face_cor_exist = False

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_paths, class_idx = self.data[idx]

        # Load images and apply transform (if any)
        if self.transform:
            images = [self.transform(Image.open(p).convert("RGB")) for p in img_paths]
        else:
            images = [Image.open(p).convert("RGB") for p in img_paths]

        if self.face_cor_exist:
            facial_corr = torch.stack([
                torch.from_numpy(np.load(p.replace(".jpg", "_corr.npz"))["corr"]) for p in img_paths
            ])
        else:
            facial_corr = torch.empty(0)

        # Load all ref perspectives and true perspectives and scan ids
        ref_perspectives = [os.path.basename(img_path)[16:-4].split("#")[0] for img_path in img_paths]
        true_perspectives = [os.path.basename(img_path)[16:-4].split("#")[1] for img_path in img_paths]
        scan_id = os.path.basename(img_paths[0])[:15]

        if self.shuffle_views:
            if self.face_cor_exist:
                combined = list(zip(images, ref_perspectives, true_perspectives, facial_corr))
                random.shuffle(combined)
                images, ref_perspectives, true_perspectives, facial_corr = zip(*combined)
                facial_corr = torch.stack(facial_corr)
            else:
                combined = list(zip(images, ref_perspectives, true_perspectives))
                random.shuffle(combined)
                images, ref_perspectives, true_perspectives = zip(*combined)
            images = list(images)
            ref_perspectives = list(ref_perspectives)
            true_perspectives = list(true_perspectives)

        return images, class_idx, ref_perspectives, true_perspectives, facial_corr, scan_id


def _sort_key(filepath):
    return os.path.basename(filepath).split("#")[1]
