import os
from collections import defaultdict, namedtuple

import mlflow
import numpy as np
import time
import torch
import torchvision
from torchvision.transforms import transforms
from src.util.EmbeddingsUtils import build_embedding_library, batched_distances_gpu
from src.util.ImageFolderWithScanID import ImageFolderWithSanID
from src.util.Metrics import calc_metrics
from src.util.Plotter import plot_confusion_matrix
from src.util.utils import buffer_val_min


def get_embeddings_and_distances(device, model, library_loader, val_loader):

    enrolled = build_embedding_library(device, model, library_loader)

    # Compute mean embeddings for each label
    unique_labels = np.unique(enrolled.labels)
    enrolled_embeddings_mean = np.array([enrolled.embeddings[enrolled.labels == label].mean(axis=0) for label in unique_labels])

    val = build_embedding_library(device, model, val_loader)

    # Calculate distances between embeddings of validation and library data
    distances = batched_distances_gpu(device, val.embeddings, enrolled_embeddings_mean)

    Results = namedtuple("Results",
                         ["enrolled_embeddings", "enrolled_labels", "enrolled_scan_ids", "enrolled_perspectives",
                          "val_embeddings", "val_labels", "val_scan_ids", "val_perspectives", "distances"])
    return Results(enrolled.embeddings, enrolled.labels, enrolled.scan_ids, enrolled.perspectives, val.embeddings,
                   val.labels, val.scan_ids, val.perspectives, distances)


def voting(y_pred, scan_ids, val_labels):
    # Group validation embeddings by scan ID
    scan_id_to_idx = defaultdict(list)
    for idx, scan_id in enumerate(scan_ids):
        scan_id_to_idx[scan_id].append(idx)

    # For each scan ID, apply weighted voting among its embeddings
    weighted_top_per_scan_id = {}
    for scan_id, indices in scan_id_to_idx.items():
        # Retrieve predicted labels for embeddings corresponding to this scan ID
        scan_top_n_labels = y_pred[indices]

        # Count the occurrences of each label weighted by their rank
        label_weights = defaultdict(int)
        for all_labels in scan_top_n_labels:
            for rank, label in enumerate(all_labels):
                weight = len(all_labels) - rank  # Weight inversely proportional to the rank
                label_weights[label] += weight

        # Sort labels with the highest total weights
        weighted_top_per_scan_id[scan_id] = np.array(sorted(label_weights, key=label_weights.get, reverse=True))

    # Create a dictionary that maps scan_ids to their corresponding labels
    scan_id_to_label = dict(zip(scan_ids, val_labels))

    y_true_scan = [scan_id_to_label[scan_id] for scan_id in weighted_top_per_scan_id.keys()]
    y_pred_scan = [weighted_top_per_scan_id[scan_id] for scan_id in weighted_top_per_scan_id.keys()]

    return np.array(y_true_scan), np.array(y_pred_scan)


def load_data(data_dir, transform, max_batch_size: int) -> (
        torchvision.datasets.ImageFolder, torch.utils.data.dataloader.DataLoader):
    dataset = ImageFolderWithSanID(root=data_dir, transform=transform)

    dataset_size = len(dataset)
    # Ensure the last batch is always larger than 1
    batch_size = max_batch_size
    while (dataset_size % batch_size == 1) and (batch_size > 2):
        batch_size -= 1

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6,
                                              drop_last=False)
    return dataset, data_loader


def evaluate(device, batch_size, backbone, test_path):
    input_size = [112, 112]
    test_transform = transforms.Compose([
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
        transforms.CenterCrop([input_size[0], input_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset_enrolled_path = os.path.join(test_path, 'train')
    dataset_val_path = os.path.join(test_path, 'validation')
    dataset_enrolled, enrolled_loader = load_data(dataset_enrolled_path, test_transform, batch_size)
    _, val_loader = load_data(dataset_val_path, test_transform, batch_size)

    time.sleep(0.1)

    embedding_library = get_embeddings_and_distances(device, backbone, enrolled_loader, val_loader)

    # Sort indices/classes of the closest vectors for each validation embedding
    y_pred = np.argsort(embedding_library.distances, axis=1)

    y_pred_top1 = y_pred[:, 0]
    y_pred_top5 = y_pred[:, :5]
    metrics = calc_metrics(embedding_library.val_labels, y_pred_top1, y_pred_top5)
    plot_confusion_matrix(embedding_library.val_labels, y_pred_top1, dataset_enrolled, os.path.basename(test_path), matplotlib=False)

    y_true_voting, y_pred_voting = voting(y_pred, embedding_library.val_scan_ids, embedding_library.val_labels)
    y_pred_voting_top1 = y_pred_voting[:, 0]
    y_pred_voting_top5 = y_pred_voting[:, :5]
    metrics_voting = calc_metrics(y_true_voting, y_pred_voting_top1, y_pred_voting_top5)
    # plot_confusion_matrix(y_true_scan, y_pred_scan_top1, test_database_dataset, os.path.basename(test_path), matplotlib=False)

    return metrics, metrics_voting


def evaluate_and_log(device, backbone, data_root, dataset, writer, epoch, num_epoch):
    print(f"Perform 1:N Evaluation on {dataset}")
    metrics, metrics_voting = evaluate(device, 32, backbone, os.path.join(data_root, dataset))

    buffer_val_min(writer, dataset, metrics['Rank-1 Rate'], epoch + 1)
    buffer_val_min(writer, dataset, metrics_voting['Rank-1 Rate'], epoch + 1)

    mlflow.log_metric(f"{dataset}_RR1", metrics['Rank-1 Rate'], step=epoch + 1)
    mlflow.log_metric(f'{dataset}_RR5', metrics['Rank-5 Rate'], step=epoch + 1)
    mlflow.log_metric(f"{dataset}_Voting_RR1", metrics_voting['Rank-1 Rate'], step=epoch + 1)
    mlflow.log_metric(f'{dataset}_Voting_RR5', metrics_voting['Rank-1 Rate'], step=epoch + 1)

    print(
        f"Epoch {epoch + 1}/{num_epoch}, {dataset} Evaluation: RR1: {metrics['Rank-1 Rate']} RR5: {metrics['Rank-5 Rate']} ; Voting-RR1: {metrics_voting['Rank-1 Rate']} Voting-RR5: {metrics_voting['Rank-5 Rate']}")