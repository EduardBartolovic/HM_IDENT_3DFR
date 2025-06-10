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

from src.backbone.model_multiview_irse import execute_model
from src.util.Metrics import error_rate_per_class
from src.util.Plotter import plot_confusion_matrix, plot_rrk_histogram
from src.util.Voting import calculate_embedding_similarity, compute_ranking_matrices, analyze_result, concat, accuracy_front_perspective
from src.util.datapipeline.EmbeddingDataset import EmbeddingDataset
from src.util.datapipeline.MultiviewDataset import MultiviewDataset
from src.util.misc import colorstr, bold, underscore, safe_round, smart_round


@torch.no_grad()
def get_embeddings_mv(device, backbone_reg, backbone_agg, aggregators, enrolled_loader, query_loader, use_face_corr: bool, disable_bar=False):
    """
    Calculate Embeddings
    """
    backbone_reg.eval()
    backbone_agg.eval()
    [i.eval() for i in aggregators]

    enrolled_embeddings_reg = []
    enrolled_embeddings_agg = []
    enrolled_labels = []
    enrolled_scan_ids = []
    enrolled_perspectives = 0
    for inputs, labels, perspectives, face_corr, scan_id in tqdm(iter(enrolled_loader), disable=disable_bar, desc="Generate Enrolled Embeddings"):

        if use_face_corr and face_corr.shape[1] == 0:
            raise ValueError("Please provide face correspondences if use_face_corr is True")

        embeddings_reg, embeddings_agg = execute_model(device, backbone_reg, backbone_agg, aggregators, inputs, perspectives, face_corr, use_face_corr)
        enrolled_embeddings_agg.extend(embeddings_agg.cpu().numpy())
        enrolled_embeddings_reg.append(np.array([t.cpu().numpy() for t in embeddings_reg]))
        enrolled_labels.extend(deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        enrolled_scan_ids.extend(deepcopy(scan_id))
        enrolled_perspectives = np.array(perspectives).T

    query_embeddings_reg = []
    query_embeddings_agg = []
    query_labels = []
    query_scan_ids = []
    query_perspectives = 0
    for inputs, labels, perspectives, face_corr, scan_id in tqdm(iter(query_loader), disable=disable_bar, desc="Generate Query Embeddings"):
        embeddings_reg, embeddings_agg = execute_model(device, backbone_reg, backbone_agg, aggregators, inputs, perspectives, face_corr, use_face_corr)
        query_embeddings_agg.extend(embeddings_agg.cpu().numpy())
        query_embeddings_reg.append(np.array([t.cpu().numpy() for t in embeddings_reg]))
        query_labels.extend(deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        query_scan_ids.extend(deepcopy(scan_id))
        query_perspectives = np.array(perspectives).T

    enrolled_embeddings_agg = np.array(enrolled_embeddings_agg)
    enrolled_embeddings_reg = np.concatenate(enrolled_embeddings_reg, axis=1)
    enrolled_labels = np.array([t.item() for t in enrolled_labels])
    enrolled_scan_ids = np.array(enrolled_scan_ids)
    enrolled_perspectives = np.array([enrolled_perspectives])
    query_embeddings_agg = np.array(query_embeddings_agg)
    query_embeddings_reg = np.concatenate(query_embeddings_reg, axis=1)
    query_labels = np.array([t.item() for t in query_labels])
    query_scan_ids = np.array(query_scan_ids)
    query_perspectives = np.array([query_perspectives])

    Results = namedtuple("Results", ["enrolled_embeddings_agg", "enrolled_embeddings", "enrolled_labels", "enrolled_scan_ids", "enrolled_perspectives", "query_embeddings_agg", "query_embeddings", "query_labels", "query_scan_ids", "query_perspectives"])
    return Results(enrolled_embeddings_agg, enrolled_embeddings_reg, enrolled_labels, enrolled_scan_ids, enrolled_perspectives, query_embeddings_agg, query_embeddings_reg, query_labels, query_scan_ids, query_perspectives)


def load_data(data_dir, max_batch_size: int) -> (torchvision.datasets.ImageFolder, torch.utils.data.dataloader.DataLoader):
    dataset = EmbeddingDataset(data_dir)
    dataset_size = len(dataset)
    # Ensure the last batch is always larger than 1
    batch_size = max_batch_size
    while (dataset_size % batch_size == 1) and (batch_size > 2):
        batch_size -= 1

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
    return dataset, data_loader


def load_data_mv(data_dir, max_batch_size: int, num_views: int, transform, use_face_corr: bool) -> (torchvision.datasets.ImageFolder, torch.utils.data.dataloader.DataLoader):
    """
     Load Dataset and sets that batch size so that large batch is always greater than 1
    Args:
        data_dir: directory of dataset
        max_batch_size: target batch size
        num_views: num of views for mv
        transform: pytorch data transformer
        use_face_corr: Should face correspondences getting loaded

    Returns: dataset, data_loader
    """
    dataset = MultiviewDataset(data_dir, num_views=num_views, transform=transform, use_face_corr=use_face_corr)
    dataset_size = len(dataset)
    batch_size = max_batch_size  # Ensure the last batch is always larger than 1
    while (dataset_size % batch_size == 1) and (batch_size > 2):
        batch_size -= 1
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
    return dataset, data_loader


def evaluate_mv(device, backbone_reg, backbone_agg, aggregators, test_path, test_transform, batch_size, num_views: int, use_face_corr: bool, disable_bar: bool, eval_all=True):
    """
    Evaluate 1:N Model Performance on given test dataset
    """
    dataset_enrolled_path = os.path.join(test_path, 'train')
    dataset_query_path = os.path.join(test_path, 'validation')
    dataset_enrolled, enrolled_loader = load_data_mv(dataset_enrolled_path, batch_size, num_views, test_transform, use_face_corr)
    dataset_query, query_loader = load_data_mv(dataset_query_path, batch_size, num_views, test_transform, use_face_corr)
    if len(dataset_enrolled.classes) != len(dataset_enrolled):
        raise Exception(f"len(dataset_enrolled.classes): {len(dataset_enrolled.classes)} doesnt match len(dataset_enrolled.samples): {len(dataset_enrolled)} -> Check your dataset: {test_path}")

    time.sleep(0.1)

    embedding_library = get_embeddings_mv(device, backbone_reg, backbone_agg, aggregators, enrolled_loader, query_loader, use_face_corr, disable_bar)

    enrolled_labels = embedding_library.enrolled_labels
    query_labels = embedding_library.query_labels

    # Multi View evaluation
    similarity_matrix = calculate_embedding_similarity(embedding_library.query_embeddings_agg, embedding_library.enrolled_embeddings_agg, chunk_size=batch_size, disable_bar=disable_bar)
    top_indices, top_values = compute_ranking_matrices(similarity_matrix)
    result_metrics = analyze_result(similarity_matrix, top_indices, enrolled_labels, query_labels, top_k_acc_k=5)
    plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix, os.path.basename(test_path), "mv")
    plot_confusion_matrix(query_labels, enrolled_labels[top_indices[:, 0]], dataset_enrolled, os.path.basename(test_path), matplotlib=False)
    error_rate_per_class(query_labels, enrolled_labels, top_indices, dataset_enrolled, embedding_library.query_scan_ids, similarity_matrix, os.path.basename(test_path), "_mv")

    if not eval_all:
        return result_metrics, {}, {}, {}, {}, embedding_library, dataset_enrolled, dataset_query

    # Single Front View
    metrics_front, similarity_matrix_front, top_indices_front, y_true_front, y_pred_front = accuracy_front_perspective(embedding_library, pre_sorted=True)
    plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_front, os.path.basename(test_path), "front")
    error_rate_per_class(query_labels, enrolled_labels, top_indices_front, dataset_enrolled, embedding_library.query_scan_ids, similarity_matrix, os.path.basename(test_path), "_front")
    del similarity_matrix_front, top_indices_front, y_true_front, y_pred_front

    # Concat
    metrics_concat, similarity_matrix_concat, top_indices_concat, y_true_concat, y_pred_concat = concat(embedding_library, disable_bar, pre_sorted=True)
    plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat, os.path.basename(test_path), "concat")
    error_rate_per_class(query_labels, enrolled_labels, top_indices_concat, dataset_enrolled, embedding_library.query_scan_ids, similarity_matrix, os.path.basename(test_path), "_concat")
    del similarity_matrix_concat, top_indices_concat, y_true_concat, y_pred_concat

    metrics_concat_mean, similarity_matrix_concat_mean, top_indices_concat_mean, y_true_concat_mean, y_pred_concat_mean = concat(embedding_library, disable_bar, pre_sorted=True, reduce_with="mean")
    metrics_concat_pca, similarity_matrix_concat_pca, top_indices_concat_pca, y_true_concat_pca, y_pred_concat_pca = concat(embedding_library, disable_bar, pre_sorted=True, reduce_with="pca")
    plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat_pca, os.path.basename(test_path), "concat_pca")

    return result_metrics, metrics_front, metrics_concat, metrics_concat_mean, metrics_concat_pca, embedding_library, dataset_enrolled, dataset_query


def evaluate_and_log_mv(device, backbone_reg, backbone_agg, aggregators, data_root, dataset, epoch, transform_sizes, batch_size, num_views: int, use_face_corr: bool, disable_bar: bool, eval_all=True):

    test_transform = transforms.Compose([
        transforms.Resize(transform_sizes),
        transforms.CenterCrop([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(colorstr('bright_green', f"Perform 1:N Evaluation on {dataset} with cropping: {transform_sizes} and face_corr: {use_face_corr}"))
    metrics_mv, metrics_front, metrics_concat, metrics_concat_mean, metrics_concat_pca, embedding_library, dataset_enrolled, dataset_query = evaluate_mv(
        device, backbone_reg, backbone_agg, aggregators, os.path.join(data_root, dataset), test_transform, batch_size,
        num_views, use_face_corr, disable_bar, eval_all)

    neutral_dataset = dataset.replace('depth_', '').replace('rgbd_', '').replace('rgb_', '').replace('test_', '')

    mlflow.log_metric(f'{neutral_dataset}_MV-RR1', metrics_mv['Rank-1 Rate'], step=epoch)
    mlflow.log_metric(f'{neutral_dataset}_MV-RR5', metrics_mv['Rank-5 Rate'], step=epoch)
    #mlflow.log_metric(f'{neutral_dataset}_MV-mean_true_match_similarity', metrics_mv['mean_true_match_similarity'], step=epoch)
    #mlflow.log_metric(f'{neutral_dataset}_MV-mean_false_match_similarity', metrics_mv['mean_false_match_similarity'], step=epoch)

    if metrics_front:
        mlflow.log_metric(f'{neutral_dataset}_Front-RR1', metrics_front['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Front-RR5', metrics_front['Rank-5 Rate'], step=epoch)
        #mlflow.log_metric(f'{neutral_dataset}_Front-mean_true_match_similarity', metrics_mv['mean_true_match_similarity'], step=epoch)
        #mlflow.log_metric(f'{neutral_dataset}_Front-mean_false_match_similarity', metrics_mv['mean_false_match_similarity'], step=epoch)

    if metrics_concat:
        mlflow.log_metric(f'{neutral_dataset}_Concat-RR1', metrics_concat['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Concat-RR5', metrics_concat['Rank-5 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Concat_Mean-RR1', metrics_concat_mean['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Concat_Mean-RR5', metrics_concat_mean['Rank-5 Rate'], step=epoch)
        #mlflow.log_metric(f'{neutral_dataset}_Concat-mean_true_match_similarity', metrics_mv['mean_true_match_similarity'], step=epoch)
        #mlflow.log_metric(f'{neutral_dataset}_Concat-mean_false_match_similarity', metrics_mv['mean_false_match_similarity'], step=epoch)

    if metrics_concat_pca:
        mlflow.log_metric(f'{neutral_dataset}_Concat_PCA-RR1', metrics_concat_pca['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Concat_PCA-RR5', metrics_concat_pca['Rank-5 Rate'], step=epoch)
        #mlflow.log_metric(f'{neutral_dataset}_Concat_PCA-mean_true_match_similarity',  metrics_concat_pca['mean_true_match_similarity'], step=epoch)
        #mlflow.log_metric(f'{neutral_dataset}_Concat_PCA-mean_false_match_similarity', metrics_concat_pca['mean_false_match_similarity'], step=epoch)

    # if 'bellus' in dataset:
    #    write_embeddings(embedding_library, neutral_dataset, epoch + 1)

    print_results(neutral_dataset, dataset_enrolled, dataset_query, metrics_front, metrics_concat, metrics_concat_mean, metrics_concat_pca, metrics_mv, eval_all)

    return metrics_mv


def print_results(neutral_dataset, dataset_enrolled, dataset_query, metrics_front, metrics_concat, metrics_concat_mean, metrics_concat_pca, metrics_mv, eval_all):
    rank_1_front = smart_round(metrics_front.get('Rank-1 Rate', 'N/A'))
    rank_5_front = smart_round(metrics_front.get('Rank-5 Rate', 'N/A'))

    rank_1_concat = smart_round(metrics_concat.get('Rank-1 Rate', 'N/A'))
    rank_5_concat = smart_round(metrics_concat.get('Rank-5 Rate', 'N/A'))

    rank_1_concat_mean = smart_round(metrics_concat_mean.get('Rank-1 Rate', 'N/A'))
    rank_5_concat_mean = smart_round(metrics_concat_mean.get('Rank-5 Rate', 'N/A'))

    rank_1_concat_pca = smart_round(metrics_concat_pca.get('Rank-1 Rate', 'N/A'))
    rank_5_concat_pca = smart_round(metrics_concat_pca.get('Rank-5 Rate', 'N/A'))

    rank_1_mv = smart_round(metrics_mv.get('Rank-1 Rate', 'N/A'))
    rank_5_mv = smart_round(metrics_mv.get('Rank-5 Rate', 'N/A'))

    if eval_all:
        string = (colorstr('bright_green', f"{neutral_dataset} E{len(dataset_enrolled)}Q{len(dataset_query)}: ") +
                  f"{bold('Front-RR1')}: {underscore(rank_1_front)} {bold('Front-RR5')}: {rank_5_front} "
                  f"{bold('Concat-RR1')}: {underscore(rank_1_concat)} {bold('Concat-RR5')}: {rank_5_concat} "
                  f"{bold('Concat_Mean-RR1')}: {underscore(rank_1_concat_mean)} {bold('Concat_Mean-RR5')}: {rank_5_concat_mean} "
                  f"{bold('Concat_PCA-RR1')}: {underscore(rank_1_concat_pca)} {bold('Concat_PCA-RR5')}: {rank_5_concat_pca} "
                  f"{bold('MV-RR1')}: {underscore(rank_1_mv)} {bold('MV-RR5')}: {rank_5_mv} "
                  )
    else:
        string = (colorstr('bright_green', f"{neutral_dataset} E{len(dataset_enrolled)}Q{len(dataset_query)}: ") +
                  f"{bold('MV-RR1')}: {underscore(rank_1_mv)} {bold('MV-RR5')}: {rank_5_mv} ")
    print(string)
