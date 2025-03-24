import gc
import os
from collections import namedtuple

import mlflow
import numpy as np
import time
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

from src.util.EmbeddingsUtils import build_embedding_library, batched_distances_gpu
from src.util.datapipeline.ImageFolderRGBDWithScanID import ImageFolderRGBDWithScanID
from src.util.datapipeline.ImageFolderWithScanID import ImageFolderWithScanID
from src.util.Voting import knn_voting, voting, accuracy_front_perspective, concat
from src.util.Metrics import calc_metrics, error_rate_per_class
from src.util.Plotter import plot_confusion_matrix
from src.util.embeddungs_metrics import calc_embedding_analysis
from src.util.misc import colorstr


def get_embeddings(device, model, enrolled_loader, query_loader, disable_bar):

    enrolled = build_embedding_library(device, model, enrolled_loader, disable_bar)
    query = build_embedding_library(device, model, query_loader, disable_bar)

    Results = namedtuple("Results",
                         ["enrolled_embeddings", "enrolled_labels", "enrolled_scan_ids", "enrolled_perspectives",
                          "query_embeddings", "query_labels", "query_scan_ids", "query_perspectives"])
    return Results(enrolled.embeddings, enrolled.labels, enrolled.scan_ids, enrolled.perspectives, query.embeddings,
                   query.labels, query.scan_ids, query.perspectives)


def load_data(data_dir, transform, max_batch_size: int) -> (
        torchvision.datasets.ImageFolder, torch.utils.data.dataloader.DataLoader):
    if 'rgbd' in data_dir:
        dataset = ImageFolderRGBDWithScanID(root=data_dir, transform=transform)
    else:
        dataset = ImageFolderWithScanID(root=data_dir, transform=transform)

    dataset_size = len(dataset)
    # Ensure the last batch is always larger than 1
    batch_size = max_batch_size
    while (dataset_size % batch_size == 1) and (batch_size > 2):
        batch_size -= 1

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                              drop_last=False)  # Todo: Check why Shuffle False makes everything worse
    return dataset, data_loader


def get_topk_indices(distances, k=5, batch_size=1000, disable_bar=False):
    """Compute top-k indices in batches to reduce memory usage."""
    num_queries = distances.shape[0]
    y_pred_top5 = np.zeros((num_queries, k), dtype=np.int32)

    for start in tqdm(range(0, num_queries, batch_size), disable=disable_bar, desc="Find best Match"):
        end = min(start + batch_size, num_queries)
        batch_distances = distances[start:end]

        # Get the indices of the k smallest elements in each row
        batch_topk = np.argpartition(batch_distances, k, axis=1)[:, :k]

        # Store results in the output array
        y_pred_top5[start:end] = batch_topk

    # Extract the top-1 directly from top-5
    y_pred_top1 = y_pred_top5[:, 0]
    return y_pred_top1, y_pred_top5


def evaluate(device, batch_size, backbone, test_path, distance_metric, test_transform, disable_bar):
    """
    Evaluate 1:N Model Performance on given test dataset
    """
    dataset_enrolled_path = os.path.join(test_path, 'train')
    dataset_query_path = os.path.join(test_path, 'validation')
    dataset_enrolled, enrolled_loader = load_data(dataset_enrolled_path, test_transform, batch_size)
    _, query_loader = load_data(dataset_query_path, test_transform, batch_size)

    time.sleep(0.1)

    embedding_library = get_embeddings(device, backbone, enrolled_loader, query_loader, disable_bar)

    # Compute mean embeddings for each label
    unique_labels, indices = np.unique(embedding_library.enrolled_labels, return_inverse=True)
    enrolled_embeddings_mean = np.zeros((len(unique_labels), embedding_library.enrolled_embeddings.shape[1]), dtype=np.float32)
    for i, label in enumerate(unique_labels):
        enrolled_embeddings_mean[i] = embedding_library.enrolled_embeddings[indices == i].mean(axis=0)

    # Calculate distances between embeddings of query and library data
    distances = batched_distances_gpu(device, embedding_library.query_embeddings, enrolled_embeddings_mean, batch_size, distance_metric=distance_metric, disable_bar=disable_bar)

    # Sort indices/classes of the closest vectors for each query embedding
    y_pred_top1, y_pred_top5 = get_topk_indices(distances, k=5, batch_size=batch_size, disable_bar=disable_bar)
    del distances

    embedding_metrics = {}  # calc_embedding_analysis(embedding_library, enrolled_embeddings_mean, distance_metric)

    metrics = calc_metrics(embedding_library.query_labels, y_pred_top1, y_pred_top5)

    plot_confusion_matrix(embedding_library.query_labels, y_pred_top1, dataset_enrolled, os.path.basename(test_path), matplotlib=False)
    error_rate_per_class(embedding_library.query_labels, y_pred_top1, os.path.basename(test_path))

    # Front Only
    if 'texas' in test_path:
        metrics_front = metrics
    else:
        metrics_front = accuracy_front_perspective(embedding_library, distance_metric)

    # VotingV1 Single Encoding
    #if 'texas' in test_path:
    metrics_voting = {}
    #else:
    #    y_true_voting, y_pred_voting = voting(y_pred, embedding_library.query_scan_ids, embedding_library.query_labels)
    #    y_pred_voting_top1 = y_pred_voting[:, 0]
    #    y_pred_voting_top5 = y_pred_voting[:, :5]
    #    metrics_voting = calc_metrics(y_true_voting, y_pred_voting_top1, y_pred_voting_top5)

    # VotingV2 KNN
    if 'texas' in test_path:
        metrics_knn_voting = {}
    else:
        y_true_knn, y_pred_knn = knn_voting(embedding_library)
        metrics_knn_voting = calc_metrics(y_true_knn, y_pred_knn)
        plot_confusion_matrix(y_true_knn, y_pred_knn, dataset_enrolled, os.path.basename(test_path) + '_votingV2', matplotlib=False)

    # Concat
    if 'texas' in test_path or 'colorferet' in test_path:
        metric_concat = {}
    else:
        metric_concat = concat(embedding_library, disable_bar)

    return metrics, metrics_front, metrics_voting, metrics_knn_voting, metric_concat, embedding_metrics, embedding_library


def evaluate_and_log(device, backbone, data_root, dataset, epoch, distance_metric, test_transform_sizes, batch_size, disable_bar=False):

    test_transform = transforms.Compose([
        transforms.Resize(test_transform_sizes),
        transforms.CenterCrop([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(colorstr('bright_green', f"Perform 1:N Evaluation on {dataset} with cropping: {test_transform_sizes}"))
    metrics, metrics_front, metrics_voting, metrics_knn_voting, metric_concat, embedding_metrics, embedding_library = evaluate(device, batch_size*2, backbone, os.path.join(data_root, dataset), distance_metric, test_transform, disable_bar)

    neutral_dataset = dataset.replace('depth_', '').replace('rgbd_', '').replace('rgb_', '').replace('test_', '')

    mlflow.log_metric(f"{neutral_dataset}_RR1", metrics['Rank-1 Rate'], step=epoch + 1)
    mlflow.log_metric(f'{neutral_dataset}_RR5', metrics['Rank-5 Rate'], step=epoch + 1)

    if 'Rank-1 Rate' in metrics_front.keys():
        mlflow.log_metric(f"{neutral_dataset}_Front_RR1", metrics_front['Rank-1 Rate'], step=epoch + 1)
    if 'Rank-1 Rate' in metrics_voting.keys():
        mlflow.log_metric(f"{neutral_dataset}_Voting_RR1", metrics_voting['Rank-1 Rate'], step=epoch + 1)
    if 'Rank-1 Rate' in metrics_knn_voting.keys():
        mlflow.log_metric(f"{neutral_dataset}_KNNVoting_RR1", metrics_knn_voting['Rank-1 Rate'], step=epoch + 1)
    if 'Rank-1 Rate' in metric_concat.keys():
        mlflow.log_metric(f"{neutral_dataset}_Concat_RR1", metric_concat['Rank-1 Rate'], step=epoch + 1)
        mlflow.log_metric(f'{neutral_dataset}_Concat_RR5', metric_concat['Rank-5 Rate'], step=epoch + 1)

    #if 'bellus' in dataset:
    #    write_embeddings(embedding_library, neutral_dataset, epoch + 1)

    if embedding_metrics:
        mlflow.log_metric(f"{neutral_dataset}_intra_enrolled_avg_distance", embedding_metrics['intra_enrolled_avg_distance'], step=epoch + 1)
        mlflow.log_metric(f"{neutral_dataset}_intra_query_avg_distance", embedding_metrics['intra_query_avg_distance'], step=epoch + 1)
        mlflow.log_metric(f"{neutral_dataset}_intra_scan_avg_distance", embedding_metrics['intra_scan_avg_distance'], step=epoch + 1)
        mlflow.log_metric(f"{neutral_dataset}_inter_enrolled_center_avg_distance", embedding_metrics['inter_enrolled_center_avg_distance'], step=epoch + 1)
        if embedding_metrics['enrolled_silhouette_score'] is not None:
            mlflow.log_metric(f"{neutral_dataset}_enrolled_silhouette_score", embedding_metrics['enrolled_silhouette_score'], step=epoch + 1)
            mlflow.log_metric(f"{neutral_dataset}_query_silhouette_score", embedding_metrics['query_silhouette_score'], step=epoch + 1)
            mlflow.log_metric(f"{neutral_dataset}_enrolled_davies_bouldin_score", embedding_metrics['enrolled_davies_bouldin_score'], step=epoch + 1)
            mlflow.log_metric(f"{neutral_dataset}_query_davies_bouldin_score", embedding_metrics['query_davies_bouldin_score'], step=epoch + 1)
        mlflow.log_metric(f"{neutral_dataset}_enrolled_mean_norm", embedding_metrics['enrolled_mean_norm'], step=epoch + 1)
        mlflow.log_metric(f"{neutral_dataset}_enrolled_std_norm", embedding_metrics['enrolled_std_norm'], step=epoch + 1)
        mlflow.log_metric(f"{neutral_dataset}_query_mean_norm", embedding_metrics['query_mean_norm'], step=epoch + 1)
        mlflow.log_metric(f"{neutral_dataset}_query_std_norm", embedding_metrics['query_std_norm'], step=epoch + 1)

    rank_1 = metrics.get('Rank-1 Rate', 'N/A')
    rank_5 = metrics.get('Rank-5 Rate', 'N/A')
    front_rank_1 = metrics_front.get('Rank-1 Rate', 'N/A')
    front_rank_5 = metrics_front.get('Rank-5 Rate', 'N/A')
    voting_rank_1 = metrics_voting.get('Rank-1 Rate', 'N/A')
    voting_rank_5 = metrics_voting.get('Rank-5 Rate', 'N/A')
    knn_voting_rank_1 = metrics_knn_voting.get('Rank-1 Rate', 'N/A')
    concat_rank_1 = metric_concat.get('Rank-1 Rate', 'N/A')
    concat_rank_5 = metric_concat.get('Rank-5 Rate', 'N/A')

    print(colorstr(
        'bright_green',
        f"{neutral_dataset} Evaluation: "
        f"RR1: {rank_1} RR5: {rank_5} "
        f"Front-RR1: {front_rank_1} Front-RR5: {front_rank_5} "
        f"Voting-RR1: {voting_rank_1} Voting-RR5: {voting_rank_5} "
        f"KNN-Voting-RR1: {knn_voting_rank_1} "
        f"Concat-RR1: {concat_rank_1} Concat-RR5: {concat_rank_5}"
    ))
