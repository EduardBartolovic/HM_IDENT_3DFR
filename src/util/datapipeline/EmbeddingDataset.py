import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class EmbeddingDataset(Dataset):

    def __init__(self, root_dir, perspective_range=None):
        self.root_dir = root_dir

        self.samples = []
        class_dirs = sorted(os.listdir(root_dir))
        self.classes = class_dirs

        def convert_list(str_list):
            return torch.tensor(
                [tuple(map(int, s.split("_"))) for s in str_list],
                dtype=torch.int32
            )

        for cls in class_dirs:
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            for fname in os.listdir(cls_path):
                if not fname.endswith(".npz"):
                    continue

                path = os.path.join(cls_path, fname)
                data = np.load(path, allow_pickle=True)

                emb = torch.tensor(data["embedding_reg"].astype(np.float32))
                #emb = torch.nn.functional.normalize(emb, p=2, dim=-1) # L2-normalize embeddings

                label = int(data["label"])
                scan_id = str(data["scan_id"])

                ref_p = convert_list(data["ref_perspective"].tolist())
                true_p = convert_list(data["true_perspective"].tolist())

                # ==============================================
                # Normalize angles if requested
                # ==============================================
                if perspective_range is not None:
                    # expects: perspective_range = ((min_yaw, max_yaw), (min_pitch, max_pitch))
                    (min_yaw, max_yaw), (min_pitch, max_pitch) = perspective_range

                    # normalize yaw
                    yaw_range = max_yaw - min_yaw
                    if yaw_range == 0:
                        raise ValueError("Yaw min/max cannot be equal.")

                    ref_yaw = (ref_p[:, 1] - min_yaw) / yaw_range
                    true_yaw = (true_p[:, 1] - min_yaw) / yaw_range

                    # normalize pitch
                    pitch_range = max_pitch - min_pitch
                    if pitch_range == 0:
                        raise ValueError("Pitch min/max cannot be equal.")

                    ref_pitch = (ref_p[:, 0] - min_pitch) / pitch_range
                    true_pitch = (true_p[:, 0] - min_pitch) / pitch_range

                    # scale both to [-1, 1]
                    ref_yaw = ref_yaw * 2 - 1
                    ref_pitch = ref_pitch * 2 - 1
                    true_yaw = true_yaw * 2 - 1
                    true_pitch = true_pitch * 2 - 1

                    # reassemble tensors
                    ref_p = torch.stack([ref_yaw, ref_pitch], dim=1)
                    true_p = torch.stack([true_yaw, true_pitch], dim=1)

                self.samples.append(
                    (emb, label, scan_id, ref_p, true_p, path)
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
