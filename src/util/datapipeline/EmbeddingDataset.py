import os
import numpy as np
import torch
from scripts.regsetup import description
from torch.utils.data import Dataset
from collections import defaultdict

from tqdm import tqdm


class EmbeddingDataset(Dataset):

    def __init__(self, root_dir, views: list[str] | None = None, perspective_as_string=False, disable_tqdm=True):
        self.root_dir = root_dir

        self.views = views
        if self.views is not None:
            self.view_tuples = {
                tuple(map(int, v.split("_"))) for v in self.views
            }
        else:
            self.view_tuples = None

        self.samples = []
        class_dirs = sorted(os.listdir(root_dir))
        self.classes = class_dirs

        def convert_list(str_list):
            return torch.tensor(
                [tuple(map(int, s.split("_"))) for s in str_list],
                dtype=torch.int32
            )

        for cls in tqdm(class_dirs, desc="Loading classes", disable=disable_tqdm):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            for fname in os.listdir(cls_path):
                if not fname.endswith(".npz"):
                    continue

                path = os.path.join(cls_path, fname)
                data = np.load(path, allow_pickle=True)

                emb_np = data["embedding_reg"].astype(np.float32)
                # --- load perspectives as STRINGS first ---
                true_p_str = data["true_perspective"].tolist()

                # --- FILTER VIEWS ---
                if self.views is not None:
                    mask = np.array([p in self.views for p in true_p_str])
                    emb_np = emb_np[mask]
                    true_p_str = [p for p, m in zip(true_p_str, mask) if m]

                # --- convert perspectives to int tuple if needed ---
                if perspective_as_string:
                    true_p = true_p_str
                else:
                    true_p = convert_list(true_p_str)

                emb = torch.from_numpy(emb_np)

                label = int(data["label"])
                scan_id = str(data["scan_id"])

                self.samples.append(
                    (emb, label, scan_id, true_p, path)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


def split_with_shared_labels(dataset, val_ratio=0.2, seed=42):
    rng = torch.Generator().manual_seed(seed)

    # group indices by label
    label_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        _, label, *_ = dataset[i]
        label_to_indices[label].append(i)

    train_indices = []
    val_indices = []

    # split inside each label group
    for label, idx_list in label_to_indices.items():
        idx_tensor = torch.randperm(len(idx_list), generator=rng)
        split = int(len(idx_list) * (1 - val_ratio))

        train_idx = [idx_list[i] for i in idx_tensor[:split]]
        val_idx   = [idx_list[i] for i in idx_tensor[split:]]

        # ensure at least one sample per set
        if len(val_idx) == 0:
            val_idx.append(train_idx.pop())
        if len(train_idx) == 0:
            train_idx.append(val_idx.pop())

        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    # create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset   = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset
