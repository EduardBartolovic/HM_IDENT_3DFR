import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiviewEmbeddingDataset(Dataset):
    def __init__(self, root_dir, num_views):
        """
        Args:
            root_dir (string): Path to the root directory of the dataset.
        """
        self.root_dir = root_dir
        self.num_views = num_views
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
                    if filename.endswith(".npy"):
                        file_path = os.path.join(class_path, filename)
                        if os.path.isfile(file_path):
                            sha_hash = filename[:40]  # Extract SHA hash from filename
                            if sha_hash not in sha_groups:
                                sha_groups[sha_hash] = []
                            sha_groups[sha_hash].append(file_path)

                        assert re.match(r"^[-+]?\d+_[-+]?\d+$", os.path.basename(file_path)[40:-8]), "perspective in dataset doesnt match convention"

                # Append each grouped data point to the dataset
                for sha_hash, file_paths in sha_groups.items():
                    if len(file_paths) == self.num_views:
                        file_paths.sort()  # Sort the data so perspectives are always at the same position
                        data.append((file_paths, class_idx))
                    else:
                        raise ValueError(f"Dataset Mistake in: {file_paths} \n {len(file_paths)}: number of views doesnt match with {self.num_views}")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        emb_paths, class_name = self.data[idx]

        # Load all images in the set
        embs = torch.Tensor(np.array([np.load(emb_path) for emb_path in emb_paths]))

        # Load all perspectives
        perspectives = [os.path.basename(emb_path)[40:-8] for emb_path in emb_paths]

        # Load Scan ids
        scan_id = os.path.basename(emb_paths[0])[:40]

        return embs, class_name, perspectives, scan_id
