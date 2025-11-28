import os

import numpy as np
import json
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing class folders with .npz files
            transform: Optional transform to apply on embeddings
        """
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        class_dirs = sorted(os.listdir(root_dir))
        self.classes = class_dirs

        for cls in class_dirs:
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            for fname in os.listdir(cls_path):
                if fname.endswith(".npz"):
                    self.samples.append(os.path.join(cls_path, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        data = np.load(path, allow_pickle=True)

        embeddings = data["embedding_reg"].astype(np.float32)
        label = int(data["label"])
        scan_id = str(data["scan_id"])

        embeddings = torch.tensor(embeddings)

        def convert_list(str_list):
            return [tuple(map(int, s.split("_"))) for s in str_list]

        ref_p = torch.tensor(convert_list(data["ref_perspective"].tolist()))
        true_p = torch.tensor(convert_list(data["true_perspective"].tolist()))

        return embeddings, label, scan_id, ref_p, true_p, path
