import os

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, data_path, input_name="embedding"):

        self.root_dir = data_path
        self.input = input_name
        self.data = []

        for label in os.listdir(data_path):
            label_folder = os.path.join(data_path, label)
            if not os.path.isdir(label_folder):
                continue

            # Find all embedding/Featuremap files in the folder
            embedding_files = [
                f for f in os.listdir(label_folder) if f"_{input_name}" in f and f.endswith(".npz")
            ]

            # Match embeddings with perspectives by scan_name
            for embedding_file in embedding_files:
                scan_name = embedding_file.split(f"_{input_name}")[0]
                perspective_file = f"{scan_name}_perspective.npz"

                # Ensure the perspective file exists
                perspective_path = os.path.join(label_folder, perspective_file)
                if os.path.exists(perspective_path):
                    embedding_path = os.path.join(label_folder, embedding_file)
                    self.data.append((embedding_path, perspective_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        embedding_path, perspective_path, label = self.data[idx]

        embedding = np.load(embedding_path)["arr_0"]
        #perspective = np.load(perspective_path)#["arr_0"]

        embedding = torch.tensor(embedding, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        #scan_id = np.array(self.scan_ids[idx])
        #perspectives = np.array(self.perspectives[idx])

        return embedding, label#, scan_id, perspectives
