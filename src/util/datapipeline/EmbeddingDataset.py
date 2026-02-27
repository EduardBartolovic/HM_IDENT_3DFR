import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

from tqdm import tqdm


class EmbeddingDataset(Dataset):

    def __init__(self, root_dir, views: list[str] | None = None, shuffle_views=False, disable_tqdm=True):
        self.root_dir = root_dir

        self.views = views
        if self.views is not None:
            self.view_tuples = {
                tuple(map(int, v.split("_"))) for v in self.views
            }
        else:
            self.view_tuples = None

        self.samples = []
        self.classes, self.class_to_idx = self._find_classes()
        #class_dirs = sorted(os.listdir(root_dir))
        #self.classes = class_dirs

        for cls in tqdm(self.classes, desc="Loading classes", disable=disable_tqdm):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            cls_idx = self.class_to_idx[cls]  # <- get numeric class index

            for fname in os.listdir(cls_path):
                if not fname.endswith(".npz"):
                    continue

                path = os.path.join(cls_path, fname)
                data = np.load(path, allow_pickle=False)

                emb_np = data["embedding_reg"]
                true_p_np = data["true_perspective"]  # shape: (num_views, 2)
                ref_p_np = data["ref_perspective"]  # shape: (num_views, 2)

                # --- FILTER VIEW FILTERING ---
                if self.views is not None:
                    mask = np.array([tuple(p) in self.view_tuples for p in ref_p_np])
                    assert mask.any(), "No samples match the selected views. Check your selected views!"
                    emb_np = emb_np[mask]
                    true_p_np = true_p_np[mask]
                    ref_p_np = ref_p_np[mask]

                if shuffle_views:
                    perm = np.random.permutation(len(emb_np))
                    emb_np = emb_np[perm]

                # shape: [number_poses, 2] becaus a pose has yaw and pitch
                true_p = torch.from_numpy(true_p_np).to(torch.float16)
                # shape: [number_poses, 2] becaus a pose has yaw and pitch
                ref_p = torch.from_numpy(ref_p_np).to(torch.float16)
                emb = torch.from_numpy(emb_np)
                scan_id = str(data["scan_id"])

                self.samples.append(
                    (emb, cls_idx, scan_id, true_p, ref_p)
                )

    def _find_classes(self):
        """
        Finds class names and maps them to integer labels.
        """
        class_names = sorted(
            entry.name for entry in os.scandir(self.root_dir) if entry.is_dir()
        )
        return class_names, {name: i for i, name in enumerate(class_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample

    #def remove_by_ids(self, ids: set):
    #    self.samples = [s for s in self.samples if s[1] not in ids]
    #    self.classes = list(set([e[1] for e in self.samples]))

    #def remove_samples(self, euclidean_distance_thresh):
    #    new_samples = []
    #    removed_ids = set()

    #    for sample in self.samples:
    #        emb, label, scan_id, true_p, ref_p = sample

            # compute euclidean distance between true and ref perspective for each pose
    #        # true_p and ref_p are shape [num_poses, 2], dtype int16
    #        diff = (true_p.float() - ref_p.float())  # [num_poses, 2]
    #        distances = torch.norm(diff, dim=1)  # [num_poses]

    #        count_close = (distances < euclidean_distance_thresh).sum().item()

    #        if count_close >= 2:
    #            new_samples.append(sample)
    #        else:
    #            removed_ids.add(label)

    #    self.samples = new_samples
    #    self.classes = list(set([e[1] for e in self.samples]))
    #    return removed_ids


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
        val_idx = [idx_list[i] for i in idx_tensor[split:]]

        # ensure at least one sample per set
        if len(val_idx) == 0:
            val_idx.append(train_idx.pop())
        if len(train_idx) == 0:
            train_idx.append(val_idx.pop())

        train_indices.extend(train_idx)
        val_indices.extend(val_idx)

    # create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset
