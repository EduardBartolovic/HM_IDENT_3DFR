import os
from collections import defaultdict, namedtuple

import mlflow
import numpy as np
import time
import torch
import torchvision
from src.util.EmbeddingsUtils import build_embedding_library, batched_distances_gpu
from src.util.ImageFolderRGBDWithScanID import ImageFolderRGBDWithScanID
from src.util.ImageFolderWithScanID import ImageFolderWithScanID
from src.util.Voting import knn_voting, voting, accuracy_front_perspective, concat
from src.util.Metrics import calc_metrics, error_rate_per_class
from src.util.Plotter import plot_confusion_matrix, write_embeddings
from src.util.embeddungs_metrics import calc_embedding_analysis
from src.util.misc import colorstr
from src.util.utils import buffer_val_min


def get_embeddings_and_distances(device, model, library_loader, query_loader, distance_metric, batch_size):

    enrolled = build_embedding_library(device, model, library_loader)

    # Compute mean embeddings for each label
    unique_labels = np.unique(enrolled.labels)
    enrolled_embeddings_mean = np.array(
        [enrolled.embeddings[enrolled.labels == label].mean(axis=0) for label in unique_labels])

    query = build_embedding_library(device, model, query_loader)

    # Calculate distances between embeddings of query and library data
    distances = batched_distances_gpu(device, query.embeddings, enrolled_embeddings_mean, batch_size, distance_metric=distance_metric )

    Results = namedtuple("Results",
                         ["enrolled_embeddings", "enrolled_labels", "enrolled_scan_ids", "enrolled_perspectives",
                          "query_embeddings", "query_labels", "query_scan_ids", "query_perspectives", "distances"])
    return Results(enrolled.embeddings, enrolled.labels, enrolled.scan_ids, enrolled.perspectives, query.embeddings,
                   query.labels, query.scan_ids, query.perspectives, distances)


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


def evaluate(device, batch_size, backbone, test_path, distance_metric, test_transform):
    """
    Evaluate 1:N Model Performance on given test dataset
    """

    dataset_enrolled_path = os.path.join(test_path, 'train')
    dataset_query_path = os.path.join(test_path, 'validation')
    dataset_enrolled, enrolled_loader = load_data(dataset_enrolled_path, test_transform, batch_size)
    _, query_loader = load_data(dataset_query_path, test_transform, batch_size)

    time.sleep(0.1)

    embedding_library = get_embeddings_and_distances(device, backbone, enrolled_loader, query_loader, distance_metric, batch_size)

    # Sort indices/classes of the closest vectors for each query embedding
    y_pred = np.argsort(embedding_library.distances, axis=1)

    embedding_metrics = calc_embedding_analysis(embedding_library, distance_metric)

    y_pred_top1 = y_pred[:, 0]
    y_pred_top5 = y_pred[:, :5]
    metrics = calc_metrics(embedding_library.query_labels, y_pred_top1, y_pred_top5)
    plot_confusion_matrix(embedding_library.query_labels, y_pred_top1, dataset_enrolled, os.path.basename(test_path), matplotlib=False)
    error_rate_per_class(embedding_library.query_labels, y_pred_top1, os.path.basename(test_path))

    # Eval only Front
    if 'texas' not in test_path and 'colorferet' not in test_path:
        y_true_front, y_pred_front = accuracy_front_perspective(device, embedding_library, distance_metric)
        metrics_front = calc_metrics(y_true_front, y_pred_front)
    else:
        metrics_front = {}

    # VotingV1 Single Encoding
    y_true_voting, y_pred_voting = voting(y_pred, embedding_library.query_scan_ids, embedding_library.query_labels)
    y_pred_voting_top1 = y_pred_voting[:, 0]
    y_pred_voting_top5 = y_pred_voting[:, :5]
    metrics_voting = calc_metrics(y_true_voting, y_pred_voting_top1, y_pred_voting_top5)
    plot_confusion_matrix(y_true_voting, y_pred_voting_top1, dataset_enrolled,
                          (os.path.basename(test_path) + '_voting'), matplotlib=False)

    #VotingV2 KNN
    y_true_knn, y_pred_knn = knn_voting(embedding_library)
    metrics_knn_voting = calc_metrics(y_true_knn, y_pred_knn)

    # ConCat
    metric_concat = concat(embedding_library)

    return metrics, metrics_front, metrics_voting, metrics_knn_voting, metric_concat, embedding_metrics, embedding_library


def evaluate_and_log(device, backbone, data_root, dataset, writer, epoch, num_epoch, distance_metric, test_transform, batch_size):
    print(colorstr('bright_green', f"Perform 1:N Evaluation on {dataset}"))
    metrics, metrics_front, metrics_voting, metrics_knn_voting, metric_concat, embedding_metrics, embedding_library = evaluate(device, batch_size*2, backbone, os.path.join(data_root, dataset), distance_metric, test_transform)

    neutral_dataset = dataset.replace('depth_', '').replace('rgbd_', '').replace('rgb_', '').replace('test_', '')

    buffer_val_min(writer, neutral_dataset, metrics['Rank-1 Rate'], epoch + 1)
    buffer_val_min(writer, neutral_dataset, metrics_voting['Rank-1 Rate'], epoch + 1)

    mlflow.log_metric(f"{neutral_dataset}_RR1", metrics['Rank-1 Rate'], step=epoch + 1)
    mlflow.log_metric(f'{neutral_dataset}_RR5', metrics['Rank-5 Rate'], step=epoch + 1)
    if 'Rank-1 Rate' in metrics_front.keys():
        mlflow.log_metric(f"{neutral_dataset}_Front_RR1", metrics_front['Rank-1 Rate'], step=epoch + 1)
    mlflow.log_metric(f"{neutral_dataset}_Voting_RR1", metrics_voting['Rank-1 Rate'], step=epoch + 1)
    mlflow.log_metric(f"{neutral_dataset}_KNNVoting_RR1", metrics_knn_voting['Rank-1 Rate'], step=epoch + 1)
    mlflow.log_metric(f"{neutral_dataset}_RR1", metric_concat['Rank-1 Rate'], step=epoch + 1)
    mlflow.log_metric(f'{neutral_dataset}_RR5', metric_concat['Rank-5 Rate'], step=epoch + 1)

    #if 'bellus' in dataset:
    #    write_embeddings(embedding_library, neutral_dataset, epoch + 1)

    if 'texas' not in dataset:
        mlflow.log_metric(f"{neutral_dataset}_intra_enrolled_avg_distance", embedding_metrics['intra_enrolled_avg_distance'], step=epoch + 1)
        mlflow.log_metric(f"{neutral_dataset}_intra_query_avg_distance", embedding_metrics['intra_query_avg_distance'], step=epoch + 1)
        mlflow.log_metric(f"{neutral_dataset}_intra_scan_avg_distance", embedding_metrics['intra_scan_avg_distance'], step=epoch + 1)
        # mlflow.log_metric(f"{neutral_dataset}_intra_query_to_enrolled_center_avg_distance", embedding_metrics['intra_query_to_enrolled_center_avg_distance'], step=epoch + 1)
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

    if 'Rank-1 Rate' in metrics_front.keys():
        print(colorstr('bright_green', f"{neutral_dataset} Evaluation: RR1: {metrics['Rank-1 Rate']} RR5: {metrics['Rank-5 Rate']} Front-RR1: {metrics_front['Rank-1 Rate']} ; Voting-RR1: {metrics_voting['Rank-1 Rate']} Voting-RR5: {metrics_voting['Rank-5 Rate']} ; KNN-Voting-RR1: {metrics_knn_voting['Rank-1 Rate']} ; Concat-RR1: {metric_concat['Rank-1 Rate']} Concat-RR5: {metric_concat['Rank-5 Rate']}"))
    else:
        print(colorstr('bright_green', f"{neutral_dataset} Evaluation: RR1: {metrics['Rank-1 Rate']} RR5: {metrics['Rank-5 Rate']} ; Voting-RR1: {metrics_voting['Rank-1 Rate']} Voting-RR5: {metrics_voting['Rank-5 Rate']} ; KNN-Voting-RR1: {metrics_knn_voting['Rank-1 Rate']} ; Concat-RR1: {metric_concat['Rank-1 Rate']} Concat-RR5: {metric_concat['Rank-5 Rate']}"))
