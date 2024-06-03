from collections import namedtuple

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def build_embedding_library(device, model, data_loader):
    embedding_library = []
    embedding_labels = []
    embedding_scan_id = []
    embedding_perspective = []

    for images, labels, scan_id, perspective in tqdm(data_loader, desc="Generate Embeddings"):
        # Enable autocast for the forward pass, running the model in mixed precision
        #with torch.cuda.amp.autocast():
        embeddings = model(images.to(device))
        embedding_library.extend(embeddings.cpu().numpy())
        embedding_labels.extend(labels.cpu().numpy())
        embedding_scan_id.extend(np.array(scan_id))
        embedding_perspective.extend(np.array(perspective))

    embeddings = np.asarray(embedding_library)
    labels = np.asarray(embedding_labels)
    scan_id = np.asarray(embedding_scan_id)
    perspective = np.asarray(embedding_perspective)

    library = namedtuple("library", ["embeddings", "labels", "scan_ids", "perspectives"])
    return library(embeddings, labels, scan_id, perspective)


def batched_distances(embeddings_val: np.array, embeddings_database: np.array, batch_size=1000):
    num_samples = len(embeddings_val)
    distances_list = []
    for i in tqdm(range(0, num_samples, batch_size), desc="Calculate Distances"):
        val_batch = embeddings_val[i:i + batch_size, :-1]
        dist_batch = np.linalg.norm(val_batch[:, None, :] - embeddings_database[None, :, :], axis=-1)
        distances_list.append(dist_batch)
    distances = np.concatenate(distances_list, axis=0)
    return distances


def batched_distances_gpu(device, embeddings_val: np.array, embeddings_database: np.array, batch_size=30):
    num_samples = len(embeddings_val)

    embeddings_val = torch.tensor(embeddings_val, device=device, dtype=torch.float32)
    embeddings_database = torch.tensor(embeddings_database, device=device, dtype=torch.float32)

    # Expand embeddings_database for broadcasting: (1, num_embeddings, num_features)
    embeddings_database_expanded = embeddings_database.unsqueeze(0)

    # Preallocate memory for distances
    distances = np.empty((num_samples, len(embeddings_database)), dtype=np.float32)

    for i in tqdm(range(0, num_samples, batch_size), desc="Calculate Distances"):
        # Get the current batch of embeddings, which could be smaller than batch_size for last batch
        val_batch = embeddings_val[i:i + batch_size]

        # Expand val_batch for broadcasting: (current_batch_size, 1, num_features)
        val_batch_expanded = val_batch.unsqueeze(1)

        # Calculate distances with broadcasting
        dist_batch = torch.cdist(val_batch_expanded, embeddings_database_expanded, p=2)  # L2-norm / Euclidean

        distances[i:i + batch_size] = dist_batch.squeeze(1).cpu().numpy()

    return distances
