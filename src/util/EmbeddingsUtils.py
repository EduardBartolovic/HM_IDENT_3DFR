from collections import namedtuple

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def build_embedding_library(device, model, data_loader, disable_bar):
    embedding_library = []
    embedding_labels = []
    embedding_scan_id = []
    embedding_perspective = []

    model.eval()

    for images, labels, scan_id, perspective in tqdm(data_loader, disable=disable_bar, desc="Generate Embeddings"):
        embeddings = model(images.to(device)).cpu().numpy()
        embedding_library.extend(embeddings)
        embedding_labels.extend(labels.cpu().numpy())
        embedding_scan_id.extend(np.array(scan_id))
        embedding_perspective.extend(np.array(perspective))

    embeddings = np.asarray(embedding_library)
    labels = np.asarray(embedding_labels)
    scan_id = np.asarray(embedding_scan_id)
    perspective = np.asarray(embedding_perspective)

    library = namedtuple("library", ["embeddings", "labels", "scan_ids", "perspectives"])
    return library(embeddings, labels, scan_id, perspective)


def batched_distances(embeddings_val: np.array, embeddings_database: np.array, batch_size=1000, disable_bar=True):
    num_samples = len(embeddings_val)
    distances_list = []
    for i in tqdm(range(0, num_samples, batch_size), disable=disable_bar, desc="Calculate Distances"):
        val_batch = embeddings_val[i:i + batch_size, :-1]
        dist_batch = np.linalg.norm(val_batch[:, None, :] - embeddings_database[None, :, :], axis=-1)
        distances_list.append(dist_batch)
    distances = np.concatenate(distances_list, axis=0)
    return distances


@torch.no_grad()
def batched_distances_gpu(device, embeddings_query: np.array, embeddings_enrolled: np.array, batch_size=64, distance_metric='cosine', disable_bar=True):

    if distance_metric == 'euclidean':
        distances = batched_distances_gpu_euclidian(device, embeddings_query, embeddings_enrolled, batch_size, disable_bar)
    elif distance_metric == 'cosine':
        distances = batched_distances_gpu_cosine(device, embeddings_query, embeddings_enrolled, batch_size, disable_bar)
    else:
        raise ValueError('Wrong Distance Metric Selected')
    return distances


@torch.no_grad()
def batched_distances_gpu_euclidian(device, embeddings_query: np.array, embeddings_enrolled: np.array, batch_size, disable_bar):
    num_samples = len(embeddings_query)

    embeddings_query = torch.tensor(embeddings_query, device=device, dtype=torch.float32)
    embeddings_enrolled = torch.tensor(embeddings_enrolled, device=device, dtype=torch.float32)

    distances = np.empty((num_samples, len(embeddings_enrolled)), dtype=np.float32)
    for start_idx in tqdm(range(0, num_samples, batch_size), disable=disable_bar, desc="Calculate Distances"):
        end_idx = min(start_idx + batch_size, num_samples)
        query_batch = embeddings_query[start_idx:end_idx]
        dist_batch = torch.cdist(query_batch, embeddings_enrolled, p=2)  # L2-norm / Euclidean
        distances[start_idx:end_idx] = dist_batch.cpu().numpy()

    return distances


@torch.no_grad()
def batched_distances_gpu_cosine(device, embeddings_query: np.array, embeddings_enrolled: np.array, batch_size, disable_bar):
    num_samples = len(embeddings_query)

    embeddings_query = torch.tensor(embeddings_query, device=device, dtype=torch.float32)
    embeddings_enrolled = torch.tensor(embeddings_enrolled, device=device, dtype=torch.float32)
    enrolled_norm = torch.nn.functional.normalize(embeddings_enrolled, p=2, dim=1)

    distances = np.empty((num_samples, len(embeddings_enrolled)), dtype=np.float32)
    for start_idx in tqdm(range(0, num_samples, batch_size), disable=disable_bar, desc="Calculate Distances"):
        end_idx = min(start_idx + batch_size, num_samples)
        query_batch = embeddings_query[start_idx:end_idx]
        query_batch_norm = torch.nn.functional.normalize(query_batch, p=2, dim=1)
        cos_sim_batch = torch.mm(query_batch_norm, enrolled_norm.t())
        dist_batch = 1 - cos_sim_batch  # Convert cosine similarity to cosine distance
        distances[start_idx:end_idx] = dist_batch.cpu().numpy()

    return distances


def process_unsorted_embeddings(scan_ids, embeddings, labels, perspectives):
    scan_to_data = {}
    for scan_id, embedding, label, perspective in zip(scan_ids, embeddings, labels, perspectives):
        if scan_id not in scan_to_data:
            scan_to_data[scan_id] = {'embeddings': [], 'perspectives': [], 'label': label}
        scan_to_data[scan_id]['embeddings'].append(embedding)
        scan_to_data[scan_id]['perspectives'].append(perspective)

    embeddings_shape = None
    num_scans = len(scan_to_data)
    concatenated_embeddings = None
    concatenated_labels = np.empty(num_scans, dtype=np.int64)
    for i, (scan_id, data) in enumerate(scan_to_data.items()):
        sorted_embs = np.array([emb for _, emb in sorted(zip(data['perspectives'], data['embeddings']), key=lambda x: x[0])])
        concatenated_embedding = np.concatenate(sorted_embs, axis=0)

        if embeddings_shape is None:
            embeddings_shape = concatenated_embedding.shape[0]
            concatenated_embeddings = np.empty((num_scans, embeddings_shape), dtype=np.float32)

        assert concatenated_embedding.shape[0] == embeddings_shape, (
            f"Embedding sizes are not correct: {embeddings_shape} != {concatenated_embedding.shape[0]} for {scan_id}. Check Dataset!"
        )

        concatenated_embeddings[i] = concatenated_embedding
        concatenated_labels[i] = data['label']

    return concatenated_embeddings, concatenated_labels