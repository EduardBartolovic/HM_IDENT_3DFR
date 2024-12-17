import os

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, data_path):

        data = np.load(os.path.join(data_path, "embedding_library.npz"))
        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.scan_ids = data["scan_ids"]
        self.perspectives = data["perspectives"]
        assert len(self.embeddings) == len(self.labels) == len(self.scan_ids) == len(self.perspectives), "Embeddings, labels, and positional information must have the same number of samples."

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):

        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        #scan_id = np.array(self.scan_ids[idx])
        #perspectives = np.array(self.perspectives[idx])

        return embedding, label#, scan_id, perspectives
