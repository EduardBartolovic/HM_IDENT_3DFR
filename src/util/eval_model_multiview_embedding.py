import os
from collections import namedtuple
from copy import deepcopy

import mlflow
import numpy as np
import time
import torch
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm

from src.util.Metrics import error_rate_per_class
from src.util.Plotter import plot_confusion_matrix, plot_rrk_histogram
from src.util.Voting import calculate_embedding_similarity, compute_ranking_matrices, analyze_result, concat, accuracy_front_perspective
from src.util.datapipeline.EmbeddingDataset import EmbeddingDataset
from src.util.datapipeline.MultiviewDataset import MultiviewDataset
from src.util.datapipeline.MultiviewEmbeddingDataset import MultiviewEmbeddingDataset
from src.util.misc import colorstr, bold, underscore, smart_round


@torch.no_grad()
def get_embeddings_mv(backbone, enrolled_loader, query_loader, disable_bar=False):
    """
    Calculate Embeddings
    """
    backbone.eval()

    enrolled_embeddings = []
    enrolled_labels = []
    enrolled_scan_ids = []
    for inputs, labels, _, scan_id in tqdm(iter(enrolled_loader), disable=disable_bar, desc="Generate Enrolled Embeddings"):
        reduced_embeddings = backbone(inputs)  # -> (16, 512)
        enrolled_embeddings.extend(reduced_embeddings.cpu().numpy())
        enrolled_labels.extend(deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        enrolled_scan_ids.extend(deepcopy(scan_id))

    query_embeddings = []
    query_labels = []
    query_scan_ids = []
    for inputs, labels, _, scan_id in tqdm(iter(query_loader), disable=disable_bar, desc="Generate Query Embeddings"):
        reduced_embeddings = backbone(inputs)  # -> (16, 512)
        query_embeddings.extend(reduced_embeddings.cpu().numpy())
        query_labels.extend(deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        query_scan_ids.extend(deepcopy(scan_id))

    enrolled_embeddings = np.array(enrolled_embeddings)
    enrolled_labels = np.array([t.item() for t in enrolled_labels])
    enrolled_scan_ids = np.array(enrolled_scan_ids)
    query_embeddings = np.array(query_embeddings)
    query_labels = np.array([t.item() for t in query_labels])
    query_scan_ids = np.array(query_scan_ids)

    Results = namedtuple("Results", ["enrolled_embeddings", "enrolled_labels", "enrolled_scan_ids", "query_embeddings", "query_labels", "query_scan_ids"])
    return Results(enrolled_embeddings, enrolled_labels, enrolled_scan_ids, query_embeddings, query_labels, query_scan_ids)


def load_data_mv_emb(data_dir, max_batch_size: int, num_views: int) -> (torchvision.datasets.ImageFolder, torch.utils.data.dataloader.DataLoader):
    """
     Load Dataset and sets that batch size so that large batch is always greater than 1
    Args:
        data_dir: directory of dataset
        max_batch_size: target batch size
        num_views: num of views for mv

    Returns: dataset, data_loader
    """
    dataset = MultiviewEmbeddingDataset(data_dir, num_views=num_views)
    dataset_size = len(dataset)
    batch_size = max_batch_size  # Ensure the last batch is always larger than 1
    while (dataset_size % batch_size == 1) and (batch_size > 2):
        batch_size -= 1
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
    return dataset, data_loader


def evaluate_mv(device, backbone, test_path, batch_size, num_views: int, disable_bar: bool):
    """
    Evaluate 1:N Model Performance on given test dataset
    """
    dataset_enrolled_path = os.path.join(test_path, 'train')
    dataset_query_path = os.path.join(test_path, 'validation')
    dataset_enrolled, enrolled_loader = load_data_mv_emb(dataset_enrolled_path, batch_size, num_views)
    dataset_query, query_loader = load_data_mv_emb(dataset_query_path, batch_size, num_views)
    if len(dataset_enrolled.classes) != len(dataset_enrolled):
        raise Exception(f"len(dataset_enrolled.classes): {len(dataset_enrolled.classes)} doesnt match len(dataset_enrolled.samples): {len(dataset_enrolled)} -> Check your dataset: {test_path}")

    time.sleep(0.1)

    embedding_library = get_embeddings_mv(backbone, enrolled_loader, query_loader, disable_bar)

    enrolled_labels = embedding_library.enrolled_labels
    query_labels = embedding_library.query_labels

    # Multi View evaluation
    similarity_matrix = calculate_embedding_similarity(embedding_library.query_embeddings, embedding_library.enrolled_embeddings, chunk_size=batch_size, disable_bar=disable_bar)
    top_indices, top_values = compute_ranking_matrices(similarity_matrix)
    result_metrics = analyze_result(similarity_matrix, top_indices, enrolled_labels, query_labels, top_k_acc_k=5)
    plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix, os.path.basename(test_path), "mv")
    plot_confusion_matrix(query_labels, enrolled_labels[top_indices[:, 0]], dataset_enrolled, os.path.basename(test_path), matplotlib=False)
    error_rate_per_class(query_labels, enrolled_labels, top_indices, dataset_enrolled, embedding_library.query_scan_ids, similarity_matrix, os.path.basename(test_path), "_mv")

    # Single Front View
    #metrics_front, similarity_matrix_front, top_indices_front, y_true_front, y_pred_front = accuracy_front_perspective(embedding_library, pre_sorted=True)
    #plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_front, os.path.basename(test_path), "front")
    #error_rate_per_class(query_labels, enrolled_labels, top_indices_front, dataset_enrolled, embedding_library.query_scan_ids, similarity_matrix, os.path.basename(test_path), "_front")

    # Concat
    #metrics_concat, similarity_matrix_concat, top_indices_concat, y_true_concat, y_pred_concat = concat(embedding_library, disable_bar, pre_sorted=True)
    #metrics_concat_mean, similarity_matrix_concat_mean, top_indices_concat_mean, y_true_concat_mean, y_pred_concat_mean = concat(embedding_library, disable_bar, pre_sorted=True, reduce_with="mean")
    #metrics_concat_pca, similarity_matrix_concat_pca, top_indices_concat_pca, y_true_concat_pca, y_pred_concat_pca = concat(embedding_library, disable_bar, pre_sorted=True, reduce_with="pca")
    #plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat, os.path.basename(test_path), "concat")
    #plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat_pca, os.path.basename(test_path), "concat_pca")
    #error_rate_per_class(query_labels, enrolled_labels, top_indices_concat, dataset_enrolled, embedding_library.query_scan_ids, similarity_matrix, os.path.basename(test_path), "_concat")

    return result_metrics, embedding_library, dataset_enrolled, dataset_query


def evaluate_and_log_mv(device, backbone, data_root, dataset, epoch, batch_size, num_views: int, disable_bar: bool):

    print(colorstr('bright_green', f"Perform 1:N Evaluation on {dataset}:"))
    metrics_mv,  embedding_library, dataset_enrolled, dataset_query = evaluate_mv(device, backbone, os.path.join(data_root, dataset), batch_size, num_views, disable_bar)

    neutral_dataset = dataset.replace('depth_', '').replace('rgbd_', '').replace('rgb_', '').replace('test_', '')

    mlflow.log_metric(f'{neutral_dataset}_MV-RR1', metrics_mv['Rank-1 Rate'], step=epoch)
    mlflow.log_metric(f'{neutral_dataset}_MV-RR5', metrics_mv['Rank-5 Rate'], step=epoch)

    # if 'bellus' in dataset:
    #    write_embeddings(embedding_library, neutral_dataset, epoch + 1)

    print_results(neutral_dataset, dataset_enrolled, dataset_query, metrics_mv)


def print_results(neutral_dataset, dataset_enrolled, dataset_query, metrics_mv):

    rank_1_mv = smart_round(metrics_mv.get('Rank-1 Rate', 'N/A'))
    rank_5_mv = smart_round(metrics_mv.get('Rank-5 Rate', 'N/A'))

    string = (colorstr('bright_green', f"{neutral_dataset} E{len(dataset_enrolled)}Q{len(dataset_query)}: ")+ f"{bold('MV-RR1')}: {underscore(rank_1_mv)} {bold('MV-RR5')}: {rank_5_mv}")
    print(string)
