import os
from collections import namedtuple
from copy import deepcopy

import mlflow
import numpy as np
import time
import torch
import torchvision
from numpy.linalg import norm
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import normalize
from torchvision.transforms import transforms
from tqdm import tqdm

from src.util.Metrics import error_rate_per_class
from src.util.Plotter import plot_confusion_matrix, plot_rrk_histogram, plot_cmc, analyze_identification_distribution, \
    plot_all_cmc_from_txt
from src.util.Voting import calculate_embedding_similarity, compute_ranking_matrices, analyze_result, concat, \
    accuracy_front_perspective, analyze_result_verification, score_fusion, fuse_pairwise_scores
from src.util.analyse_perspective import calc_perspective_distances, analyze_perspective_error_correlation, \
    compute_per_view_distance_matrix, analyze_perspective_error_correlation_1v1
from src.util.datapipeline.MultiviewDataset import MultiviewDataset
from src.util.misc import colorstr, bold, underscore, smart_round

#torch.set_float32_matmul_precision('high')


@torch.no_grad()
def get_embeddings_mv(backbone, enrolled_loader, query_loader, use_face_corr: bool, disable_bar=False):
    """
    Calculate embeddings for enrolled and query datasets using a multi-view backbone.
    """
    backbone.eval()

    enrolled_embeddings_reg = []
    enrolled_embeddings_agg = []
    enrolled_labels = []
    enrolled_scan_ids = []
    enrolled_perspectives = 0
    enrolled_true_perspectives = []
    for inputs, labels, perspectives, true_perspectives, face_corr, scan_id in tqdm(iter(enrolled_loader), disable=disable_bar, desc="Generate Enrolled Embeddings"):

        if use_face_corr and face_corr.shape[1] == 0:
            raise ValueError("Please provide face correspondences if use_face_corr is True")

        embeddings_reg, embeddings_agg = backbone(inputs, perspectives, face_corr, use_face_corr)
        enrolled_embeddings_agg.extend(embeddings_agg.cpu().numpy())
        enrolled_embeddings_reg.append(np.stack([t.cpu().numpy() for t in embeddings_reg]))
        enrolled_labels.extend(deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        enrolled_scan_ids.extend(deepcopy(scan_id))
        enrolled_perspectives = np.array(perspectives).T[0]
        enrolled_true_perspectives.append(np.array(deepcopy(true_perspectives)).T)

    enrolled_embeddings_agg = np.array(enrolled_embeddings_agg)
    enrolled_embeddings_reg = np.concatenate(enrolled_embeddings_reg, axis=1)
    enrolled_labels = np.array([t.item() for t in enrolled_labels])
    enrolled_scan_ids = np.array(enrolled_scan_ids)
    enrolled_perspectives = np.array(enrolled_perspectives)
    enrolled_true_perspectives = np.concatenate(enrolled_true_perspectives, axis=0)

    if query_loader is None:
        Results = namedtuple("Results", ["enrolled_embeddings_agg", "enrolled_embeddings", "enrolled_labels", "enrolled_scan_ids", "enrolled_perspectives","enrolled_true_perspectives"])
        return Results(enrolled_embeddings_agg, enrolled_embeddings_reg, enrolled_labels, enrolled_scan_ids, enrolled_perspectives, enrolled_true_perspectives)

    query_embeddings_reg = []
    query_embeddings_agg = []
    query_labels = []
    query_scan_ids = []
    query_perspectives = 0
    query_true_perspectives = []
    for inputs, labels, perspectives, true_perspectives, face_corr, scan_id in tqdm(iter(query_loader), disable=disable_bar, desc="Generate Query Embeddings"):
        embeddings_reg, embeddings_agg = backbone(inputs, perspectives, face_corr, use_face_corr)
        query_embeddings_agg.extend(embeddings_agg.cpu().numpy())
        query_embeddings_reg.append(np.stack([t.cpu().numpy() for t in embeddings_reg]))
        query_labels.extend(deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        query_scan_ids.extend(deepcopy(scan_id))
        query_perspectives = np.array(perspectives).T[0]
        query_true_perspectives.append(np.array(deepcopy(true_perspectives)).T)

    query_embeddings_agg = np.array(query_embeddings_agg)
    query_embeddings_reg = np.concatenate(query_embeddings_reg, axis=1)
    query_labels = np.array([t.item() for t in query_labels])
    query_scan_ids = np.array(query_scan_ids)
    query_perspectives = np.array(query_perspectives)
    query_true_perspectives = np.concatenate(query_true_perspectives, axis=0)

    Results = namedtuple("Results", ["enrolled_embeddings_agg", "enrolled_embeddings", "enrolled_labels", "enrolled_scan_ids", "enrolled_perspectives", "enrolled_true_perspectives","query_embeddings_agg", "query_embeddings", "query_labels", "query_scan_ids", "query_perspectives", "query_true_perspectives"])
    return Results(enrolled_embeddings_agg, enrolled_embeddings_reg, enrolled_labels, enrolled_scan_ids, enrolled_perspectives, enrolled_true_perspectives, query_embeddings_agg, query_embeddings_reg, query_labels, query_scan_ids, query_perspectives, query_true_perspectives)


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


def evaluate_mv_1_n(backbone, test_path, test_transform, batch_size, num_views: int, use_face_corr: bool, disable_bar: bool, eval_all=True):
    """
    Evaluate 1:N Model Performance on given test dataset
    """

    dataset_name = os.path.basename(test_path)
    dataset_enrolled_path = os.path.join(test_path, 'enrolled')
    dataset_query_path = os.path.join(test_path, 'query')
    dataset_enrolled, enrolled_loader = load_data_mv(dataset_enrolled_path, batch_size, num_views, test_transform, use_face_corr)
    dataset_query, query_loader = load_data_mv(dataset_query_path, batch_size, num_views, test_transform, use_face_corr)
    if len(dataset_enrolled.classes) != len(dataset_enrolled):
        raise Exception(f"len(dataset_enrolled.classes): {len(dataset_enrolled.classes)} doesnt match len(dataset_enrolled.samples): {len(dataset_enrolled)} -> Check your dataset: {test_path}")

    time.sleep(0.1)

    embedding_library = get_embeddings_mv(backbone, enrolled_loader, query_loader, use_face_corr, disable_bar)
    enrolled_labels = embedding_library.enrolled_labels
    query_labels = embedding_library.query_labels

    enrolled_distances = calc_perspective_distances(embedding_library.enrolled_perspectives, embedding_library.enrolled_true_perspectives)
    query_distances = calc_perspective_distances(embedding_library.query_perspectives, embedding_library.query_true_perspectives)
    distance_matrix, distance_matrix_avg = compute_per_view_distance_matrix(embedding_library.query_true_perspectives, embedding_library.enrolled_true_perspectives)

    all_metrics = {}

    # --------- Multi View evaluation ---------
    similarity_matrix_mvfa = calculate_embedding_similarity(embedding_library.query_embeddings_agg, embedding_library.enrolled_embeddings_agg, chunk_size=batch_size, disable_bar=disable_bar)
    top_indices, top_values = compute_ranking_matrices(similarity_matrix_mvfa)
    metrics_mvfa = analyze_result(similarity_matrix_mvfa, top_indices, enrolled_labels, query_labels, top_k_acc_k=5)
    plot_cmc(similarity_matrix_mvfa, enrolled_labels, query_labels, dataset_name, "mvfa")
    plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_mvfa, dataset_name, "mvfa")
    analyze_identification_distribution(similarity_matrix_mvfa, query_labels, enrolled_labels, dataset_name, "mvfa", plot=True)
    plot_confusion_matrix(query_labels, enrolled_labels[top_indices[:, 0]], dataset_enrolled, dataset_name, matplotlib=False)
    error_rate_per_class(query_labels, enrolled_labels, top_indices, dataset_enrolled, embedding_library.query_scan_ids, similarity_matrix_mvfa, dataset_name, "_mvfa")
    all_metrics["metrics_mvfa"] = metrics_mvfa
    correlation_results = analyze_perspective_error_correlation(
        query_labels=query_labels,
        enrolled_labels=enrolled_labels,
        query_distances=query_distances,
        enrolled_distances=enrolled_distances,
        top_indices=top_indices,
        distance_matrix_avg=distance_matrix_avg,
        plot=True,
        extension="_" + dataset_name + "_mvfa"
    )
    print("MVFA", correlation_results)
    del similarity_matrix_mvfa, top_indices, top_values
    if not eval_all:
        return all_metrics, embedding_library, dataset_enrolled, dataset_query

    # --------- Single Front View ---------
    metrics_front, similarity_matrix_front, top_indices_front, y_true_front, y_pred_front = accuracy_front_perspective(embedding_library)
    plot_cmc(similarity_matrix_front, enrolled_labels, query_labels, dataset_name, "front")
    plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_front, dataset_name, "front")
    error_rate_per_class(query_labels, enrolled_labels, top_indices_front, dataset_enrolled, embedding_library.query_scan_ids, similarity_matrix_front, dataset_name, "_front")
    analyze_identification_distribution(similarity_matrix_front, query_labels, enrolled_labels, dataset_name, "front", plot=True)
    all_metrics["metrics_front"] = metrics_front
    correlation_results = analyze_perspective_error_correlation(
        query_labels=query_labels,
        enrolled_labels=enrolled_labels,
        query_distances=query_distances,
        enrolled_distances=enrolled_distances,
        top_indices=top_indices_front,
        distance_matrix_avg=distance_matrix_avg,
        plot=True,
        extension="_" + dataset_name + "_front"
    )
    print("Front", correlation_results)
    del similarity_matrix_front, top_indices_front, y_true_front, y_pred_front

    # --------- Concat Full ---------
    metrics_concat, similarity_matrix_concat, top_indices_concat, y_true_concat, y_pred_concat = concat(embedding_library, disable_bar)
    plot_cmc(similarity_matrix_concat, enrolled_labels, query_labels, dataset_name, "concat")
    plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat, dataset_name, "concat")
    error_rate_per_class(query_labels, enrolled_labels, top_indices_concat, dataset_enrolled, embedding_library.query_scan_ids, similarity_matrix_concat, dataset_name, "_concat")
    analyze_identification_distribution(similarity_matrix_concat, query_labels, enrolled_labels, dataset_name, "concat", plot=True)
    all_metrics["metrics_concat"] = metrics_concat
    correlation_results = analyze_perspective_error_correlation(
        query_labels=query_labels,
        enrolled_labels=enrolled_labels,
        query_distances=query_distances,
        enrolled_distances=enrolled_distances,
        top_indices=top_indices_concat,
        distance_matrix_avg=distance_matrix_avg,
        plot=True,
        extension="_" + dataset_name + "_concat"
    )
    print("CONCAT", correlation_results)
    del similarity_matrix_concat, top_indices_concat, y_true_concat, y_pred_concat

    # --------- Concat Mean ---------
    metrics_concat_mean, similarity_matrix_concat_mean, top_indices_concat_mean, y_true_concat_mean, y_pred_concat_mean = concat(embedding_library, disable_bar, reduce_with="mean")
    plot_cmc(similarity_matrix_concat_mean, enrolled_labels, query_labels, dataset_name, "concat_mean")
    plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat_mean, dataset_name, "concat_mean")
    analyze_identification_distribution(similarity_matrix_concat_mean, query_labels, enrolled_labels, dataset_name, "concat_mean", plot=True)
    all_metrics["metrics_concat_mean"] = metrics_concat_mean
    del similarity_matrix_concat_mean, top_indices_concat_mean, y_true_concat_mean, y_pred_concat_mean

    # --------- Concat PCA ---------
    metrics_concat_pca, similarity_matrix_concat_pca, top_indices_concat_pca, y_true_concat_pca, y_pred_concat_pca = concat(embedding_library, disable_bar, reduce_with="pca")
    plot_cmc(similarity_matrix_concat_pca, enrolled_labels, query_labels, dataset_name, "concat_pca")
    plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat_pca, dataset_name, "concat_pca")
    all_metrics["metrics_concat_pca"] = metrics_concat_pca
    del similarity_matrix_concat_pca, top_indices_concat_pca, y_true_concat_pca, y_pred_concat_pca

    # --------- Score fusion ---------
    metrics_score_sum, similarity_matrix_score, fused_scores, top_indices, predicted_labels, query_label = score_fusion(embedding_library, disable_bar, method="sum", similarity_matrix=None)
    all_metrics["metrics_score_sum"] = metrics_score_sum
    metrics_score_max, _, fused_scores, top_indices, predicted_labels, query_label = score_fusion(embedding_library, disable_bar,  method="max", similarity_matrix=similarity_matrix_score)
    all_metrics["metrics_score_max"] = metrics_score_max
    metrics_score_prod, _, fused_scores, top_indices, predicted_labels, query_label = score_fusion(embedding_library, disable_bar,  method="product", similarity_matrix=similarity_matrix_score)
    all_metrics["metrics_score_product"] = metrics_score_prod
    metrics_score_geomean, _, fused_scores, top_indices, predicted_labels, query_label = score_fusion(embedding_library, disable_bar, method="geom_mean", similarity_matrix=similarity_matrix_score)
    all_metrics["metrics_score_geomean"] = metrics_score_geomean
    metrics_score_lse, _, fused_scores, top_indices, predicted_labels, query_label = score_fusion(embedding_library, disable_bar, method="lse", similarity_matrix=similarity_matrix_score)
    all_metrics["metrics_score_lse"] = metrics_score_lse
    metrics_score_majority, _, fused_scores, top_indices, predicted_labels, query_label = score_fusion(embedding_library, disable_bar, method="majority", similarity_matrix=similarity_matrix_score)
    all_metrics["metrics_score_majority"] = metrics_score_majority
    metrics_score_mean, _, fused_scores, top_indices, predicted_labels, query_label = score_fusion(embedding_library, disable_bar, method="mean", similarity_matrix=similarity_matrix_score)
    all_metrics["metrics_score_mean"] = metrics_score_mean
    metrics_score_trimmedmean, _, fused_scores, top_indices, predicted_labels, query_label = score_fusion(embedding_library, disable_bar, method="trimmed_mean", similarity_matrix=similarity_matrix_score)
    all_metrics["metrics_score_trimmedmean"] = metrics_score_trimmedmean

    metrics_score_sdw, _, fused_scores, top_indices, predicted_labels, query_label = score_fusion(embedding_library, disable_bar, method="softmax_distance_weighting", similarity_matrix=similarity_matrix_score, distance_matrix=distance_matrix)
    print(metrics_score_sdw)

    plot_all_cmc_from_txt(dataset_name)

    return all_metrics, embedding_library, dataset_enrolled, dataset_query


def evaluate_mv_1_1(backbone, test_path, test_transform, batch_size, num_views: int, use_face_corr: bool, disable_bar: bool, eval_all=True):
    """
    Evaluate 1:1 Model Performance on given test dataset
    """
    dataset_name = os.path.basename(test_path)

    pair_list = []
    folds = []
    unique_sample_paths = set()
    with open(os.path.join(test_path, "split.txt"), "r") as f:
        next(f)  # skip header
        for line in f:
            split, pair, name1, name2, is_same, is_same_corrected = line.strip().split(",")
            pair_list.append((name1, name2, int(is_same_corrected)))
            folds.append(int(split))
            unique_sample_paths.add(name1)
            unique_sample_paths.add(name2)

    dataset_enrolled, enrolled_loader = load_data_mv(test_path, batch_size, num_views, test_transform, use_face_corr)
    time.sleep(0.1)

    embedding_library = get_embeddings_mv(backbone, enrolled_loader, None, use_face_corr, disable_bar)

    embeddings_agg = embedding_library.enrolled_embeddings_agg
    embeddings_reg = embedding_library.enrolled_embeddings
    embeddings_reg_mean = embedding_library.enrolled_embeddings.mean(axis=0)
    name_to_class_dict = dataset_enrolled.class_to_idx
    mask = np.array(["0_0" in perspective for perspective in embedding_library.enrolled_perspectives])
    embeddings_reg_pca = embeddings_reg.transpose(1, 0, 2).reshape(embeddings_reg.shape[1], -1)
    if embeddings_reg.shape[0] > 2048:
        pca = IncrementalPCA(n_components=512, batch_size=2048)
    else:
        pca = PCA(n_components=512)

    pca = pca.fit(embeddings_reg_pca)
    embeddings_reg_pca = normalize(pca.transform(embeddings_reg_pca))

    embedding_lookup = {(c, s): i for i, (c, s) in enumerate(zip(embedding_library.enrolled_labels, embedding_library.enrolled_scan_ids))}

    # Evaluate pairs
    similarities_mv = []
    similarities_front = []
    similarities_concat = []
    similarities_concat_mean = []
    similarities_concat_pca = []
    fusion_methods = ["sum", "product", "max", "geomean", "lse", "mean"]
    similarities_fusion = {m: [] for m in fusion_methods}
    labels = []
    for name1, name2, is_same in tqdm(pair_list, desc="Evaluating pairs", disable=disable_bar):
        class1, sample1 = name1.split("/")
        class2, sample2 = name2.split("/")

        sample1 = (sample1 + 'X' * 15)[:15]
        sample2 = (sample2 + 'X' * 15)[:15]
        class1 = class1.lstrip()
        class2 = class2.lstrip()

        try:
            class_idx1 = name_to_class_dict[class1]
            class_idx2 = name_to_class_dict[class2]
        except Exception:
            raise Exception("NOT IN DATASET: ", class1, "or", class2)

        i1 = embedding_lookup.get((class_idx1, sample1))
        i2 = embedding_lookup.get((class_idx2, sample2))
        if i1 is None or i2 is None:
            print(f"Warning: Could not find embeddings for pair {name1}, {name2}")

        emb1_agg, emb2_agg = embeddings_agg[i1], embeddings_agg[i2]
        emb1_reg, emb2_reg = embeddings_reg[:, i1, :], embeddings_reg[:, i2, :]
        emb1_reg_mean, emb2_reg_mean = embeddings_reg_mean[i1, :], embeddings_reg_mean[i2, :]
        emb1_reg_pca, emb2_reg_pca = embeddings_reg_pca[i1, :], embeddings_reg_pca[i2, :]

        # Multiview compute cosine similarity
        sim = np.dot(emb1_agg, emb2_agg) / (norm(emb1_agg) * norm(emb2_agg))
        similarities_mv.append(sim)
        labels.append(is_same)

        # Front
        emb1_front = emb1_reg[mask][0]
        emb2_front = emb2_reg[mask][0]
        sim = np.dot(emb1_front, emb2_front) / (norm(emb1_front) * norm(emb2_front))
        similarities_front.append(sim)

        # Concat
        emb1_flat = emb1_reg.reshape(-1)
        emb2_flat = emb2_reg.reshape(-1)
        sim = np.dot(emb1_flat, emb2_flat) / (norm(emb1_flat) * norm(emb2_flat))
        similarities_concat.append(sim)

        # Concat mean
        sim = np.dot(emb1_reg_mean, emb2_reg_mean) / (norm(emb1_reg_mean) * norm(emb2_reg_mean))
        similarities_concat_mean.append(sim)

        # Concat pca
        sim = np.dot(emb1_reg_pca, emb2_reg_pca) / (norm(emb1_reg_pca) * norm(emb2_reg_pca))
        similarities_concat_pca.append(sim)

        for m in fusion_methods:
            sim_fused = fuse_pairwise_scores(emb1_reg, emb2_reg, method=m)
            similarities_fusion[m].append(sim_fused)
    all_metrics = {"metrics_mvfa": analyze_result_verification(labels, similarities_mv, dataset_name, "_mv", folds=folds)}

    if not eval_all:
        return all_metrics, embedding_library, dataset_enrolled

    all_metrics["metrics_front"] = analyze_result_verification(labels, similarities_front, dataset_name, "_front", folds=folds)
    all_metrics["metrics_concat"] = analyze_result_verification(labels, similarities_concat, dataset_name, "_concat", folds=folds)
    all_metrics["metrics_concat_mean"] = analyze_result_verification(labels, similarities_concat_mean, dataset_name, "_mean", folds=folds)
    all_metrics["metrics_concat_pca"] = analyze_result_verification(labels, similarities_concat_pca, dataset_name, "_pca", folds=folds)

    for m in fusion_methods:
        metrics = analyze_result_verification(labels, similarities_fusion[m], dataset_name, f"_{m}", folds=folds)
        all_metrics[f"metrics_score_{m}"] = metrics

    # ---------- Perspective analysis ---------
    perspective_distances = calc_perspective_distances(
        embedding_library.enrolled_perspectives,
        embedding_library.enrolled_true_perspectives
    )  # shape (num_samples, num_views)

    preds_mv = (similarities_mv > all_metrics["metrics_mvfa"]["Best_thresh"]).astype(int)
    correlation_1v1 = analyze_perspective_error_correlation_1v1(
        pair_list=pair_list,
        enrolled_distances=perspective_distances,
        embedding_library=embedding_library,
        labels=preds_mv,
        name_to_class_dict=name_to_class_dict,
        true_perspectives=embedding_library.enrolled_true_perspectives,
        plot=True,
        extension="_"+dataset_name+"_mvfa"
    )
    print("1v1 Mvfa:", correlation_1v1)

    preds_front = (similarities_front > all_metrics["metrics_front"]["Best_thresh"]).astype(int)
    correlation_1v1 = analyze_perspective_error_correlation_1v1(
        pair_list=pair_list,
        enrolled_distances=perspective_distances,
        embedding_library=embedding_library,
        labels=preds_front,
        name_to_class_dict=name_to_class_dict,
        true_perspectives=embedding_library.enrolled_true_perspectives,
        plot=True,
        extension="_"+dataset_name + "_front"
    )
    print("1v1 Correlation front:", correlation_1v1)

    preds_concat = (similarities_concat > all_metrics["metrics_concat"]["Best_thresh"]).astype(int)
    correlation_1v1 = analyze_perspective_error_correlation_1v1(
        pair_list=pair_list,
        enrolled_distances=perspective_distances,
        embedding_library=embedding_library,
        labels=preds_concat,
        name_to_class_dict=name_to_class_dict,
        true_perspectives=embedding_library.enrolled_true_perspectives,
        plot=True,
        extension="_" + dataset_name + "_concat"
    )
    print("1v1 Correlation concat:", correlation_1v1)

    return all_metrics, embedding_library, dataset_enrolled


def evaluate_and_log_mv(backbone, data_root, dataset, epoch, transform_sizes, final_crop, batch_size, num_views: int, use_face_corr: bool, disable_bar: bool, eval_all=True):

    test_transform = transforms.Compose([
        transforms.Resize(transform_sizes),
        transforms.CenterCrop(final_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(colorstr('bright_green', f"Perform 1:N Evaluation on {dataset} with cropping: {transform_sizes} and face_corr: {use_face_corr}"))
    all_metrics, embedding_library, dataset_enrolled, dataset_query = evaluate_mv_1_n(backbone, os.path.join(data_root, dataset), test_transform, batch_size, num_views, use_face_corr, disable_bar, eval_all)

    neutral_dataset = dataset
    for prefix in ['depth_', 'rgbd_', 'rgb_', 'test_']:
        if neutral_dataset.startswith(prefix):
            neutral_dataset = neutral_dataset[len(prefix):]

    metrics_mv = all_metrics["metrics_mvfa"]
    mlflow.log_metric(f'{neutral_dataset}_MV-RR1', metrics_mv['Rank-1 Rate'], step=epoch)
    mlflow.log_metric(f'{neutral_dataset}_MV-RR5', metrics_mv['Rank-5 Rate'], step=epoch)
    mlflow.log_metric(f'{neutral_dataset}_MV-MRR', metrics_mv['MRR'], step=epoch)
    if "metrics_front" in all_metrics.keys():
        metrics_front = all_metrics["metrics_front"]
        mlflow.log_metric(f'{neutral_dataset}_Front-RR1', metrics_front['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Front-RR5', metrics_front['Rank-5 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Front-MRR', metrics_front['MRR'], step=epoch)

    if "metrics_concat" in all_metrics.keys():
        metrics_concat = all_metrics["metrics_concat"]
        metrics_concat_mean = all_metrics["metrics_concat_mean"]
        mlflow.log_metric(f'{neutral_dataset}_Concat-RR1', metrics_concat['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Concat-RR5', metrics_concat['Rank-5 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Concat_Mean-RR1', metrics_concat_mean['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Concat_Mean-RR5', metrics_concat_mean['Rank-5 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Concat-MRR', metrics_concat['MRR'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Concat_Mean-MRR', metrics_concat_mean['MRR'], step=epoch)

    if "metrics_concat_pca" in all_metrics.keys() and all_metrics["metrics_concat_pca"]:
        metrics_concat_pca = all_metrics["metrics_concat_pca"]
        mlflow.log_metric(f'{neutral_dataset}_Concat_PCA-RR1', metrics_concat_pca['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Concat_PCA-RR5', metrics_concat_pca['Rank-5 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Concat_PCA-MRR', metrics_concat_pca['MRR'], step=epoch)

    if "metrics_score_sum" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_sum-RR1', all_metrics["metrics_score_sum"]['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_sum-RR5', all_metrics["metrics_score_sum"]['Rank-5 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_sum-MRR', all_metrics["metrics_score_sum"]['MRR'], step=epoch)
    if "metrics_score_product" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_product-RR1', all_metrics["metrics_score_product"]['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_product-RR5', all_metrics["metrics_score_product"]['Rank-5 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_product-MRR', all_metrics["metrics_score_product"]['MRR'], step=epoch)
    if "metrics_score_max" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_max-RR1', all_metrics["metrics_score_max"]['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_max-RR5', all_metrics["metrics_score_max"]['Rank-5 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_max-MRR', all_metrics["metrics_score_max"]['MRR'], step=epoch)
    if "metrics_score_geomean" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_geom_mean-RR1', all_metrics["metrics_score_geomean"]['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_geom_mean-RR5', all_metrics["metrics_score_geomean"]['Rank-5 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_geom_mean-MRR', all_metrics["metrics_score_geomean"]['MRR'], step=epoch)
    if "metrics_score_lse" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_lse-RR1', all_metrics["metrics_score_lse"]['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_lse-RR5', all_metrics["metrics_score_lse"]['Rank-5 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_lse-MRR', all_metrics["metrics_score_lse"]['MRR'], step=epoch)
    if "metrics_score_majority" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_majority-RR1', all_metrics["metrics_score_majority"]['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_majority-RR5', all_metrics["metrics_score_majority"]['Rank-5 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_majority-MRR', all_metrics["metrics_score_majority"]['MRR'], step=epoch)
    if "metrics_score_mean" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_majority-RR1', all_metrics["metrics_score_majority"]['Rank-1 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_majority-RR5', all_metrics["metrics_score_majority"]['Rank-5 Rate'], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_majority-MRR', all_metrics["metrics_score_majority"]['MRR'], step=epoch)

    # if 'bellus' in dataset:
    #    write_embeddings(embedding_library, neutral_dataset, epoch + 1)

    print_results(neutral_dataset, dataset_enrolled, dataset_query, all_metrics, eval_all)

    return all_metrics


def evaluate_and_log_mv_verification(backbone, data_root, dataset, epoch, transform_sizes, final_crop, batch_size, num_views: int, use_face_corr: bool, disable_bar: bool, eval_all=True):

    test_transform = transforms.Compose([
        transforms.Resize(transform_sizes),
        transforms.CenterCrop(final_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(colorstr('bright_green', f"Perform 1:1 Evaluation on {dataset} with cropping: {transform_sizes} and face_corr: {use_face_corr} and folds: {10}"))
    all_metrics, embedding_library, dataset_enrolled = evaluate_mv_1_1(backbone, os.path.join(data_root, dataset), test_transform, batch_size, num_views, use_face_corr, disable_bar, eval_all)

    neutral_dataset = dataset
    for prefix in ['depth_', 'rgbd_', 'rgb_', 'test_']:
        if neutral_dataset.startswith(prefix):
            neutral_dataset = neutral_dataset[len(prefix):]

    mlflow.log_metric(f'{neutral_dataset}_MV-AUC', all_metrics["metrics_mvfa"]["AUC"], step=epoch)
    mlflow.log_metric(f'{neutral_dataset}_MV-ACC', all_metrics["metrics_mvfa"]["Accuracy"], step=epoch)

    if "metrics_front" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_Front-AUC', all_metrics["metrics_front"]["AUC"], step=epoch)
    if "metrics_concat" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_Concat-AUC', all_metrics["metrics_concat"]["AUC"], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Concat_Mean-AUC', all_metrics["metrics_concat_mean"]["AUC"], step=epoch)
        mlflow.log_metric(f'{neutral_dataset}_Concat_Mean-ACC', all_metrics["metrics_concat_mean"]["Accuracy"], step=epoch)
    if "metrics_concat_pca" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_Concat_PCA-AUC', all_metrics["metrics_concat_pca"]["AUC"], step=epoch)
    if "metrics_score_sum" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_sum-AUC', all_metrics["metrics_score_sum"]['AUC'], step=epoch)
    if "metrics_score_product" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_product-AUC', all_metrics["metrics_score_product"]['AUC'], step=epoch)
    if "metrics_score_max" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_max-AUC', all_metrics["metrics_score_max"]['AUC'], step=epoch)
    if "metrics_score_geomean" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_geom_mean-AUC', all_metrics["metrics_score_geomean"]['AUC'], step=epoch)
    if "metrics_score_lse" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_lse-AUC', all_metrics["metrics_score_lse"]['AUC'], step=epoch)
    if "metrics_score_mean" in all_metrics.keys():
        mlflow.log_metric(f'{neutral_dataset}_mean-AUC', all_metrics["metrics_score_mean"]['AUC'], step=epoch)

    print_results_verification(neutral_dataset, dataset_enrolled, all_metrics, eval_all)

    return all_metrics


def print_results(neutral_dataset, dataset_enrolled, dataset_query, all_metrics, eval_all):

    rank_1_mv = smart_round(all_metrics["metrics_mvfa"].get('Rank-1 Rate', 'N/A'))
    rank_5_mv = smart_round(all_metrics["metrics_mvfa"].get('Rank-5 Rate', 'N/A'))
    mrr_mv = smart_round(all_metrics["metrics_mvfa"].get('MRR', 'N/A'))

    if eval_all:
        rank_1_front = smart_round(all_metrics["metrics_front"].get('Rank-1 Rate', 'N/A'))
        rank_5_front = smart_round(all_metrics["metrics_front"].get('Rank-5 Rate', 'N/A'))
        mrr_front = smart_round(all_metrics["metrics_front"].get('MRR', 'N/A'))

        rank_1_concat = smart_round(all_metrics["metrics_concat"].get('Rank-1 Rate', 'N/A'))
        rank_5_concat = smart_round(all_metrics["metrics_concat"].get('Rank-5 Rate', 'N/A'))
        mrr_concat = smart_round(all_metrics["metrics_concat"].get('MRR', 'N/A'))

        rank_1_concat_mean = smart_round(all_metrics["metrics_concat_mean"].get('Rank-1 Rate', 'N/A'))
        rank_5_concat_mean = smart_round(all_metrics["metrics_concat_mean"].get('Rank-5 Rate', 'N/A'))
        mrr_concat_mean = smart_round(all_metrics["metrics_concat_mean"].get('MRR', 'N/A'))

        rank_1_concat_pca = smart_round(all_metrics["metrics_concat_pca"].get('Rank-1 Rate', 'N/A'))
        rank_5_concat_pca = smart_round(all_metrics["metrics_concat_pca"].get('Rank-5 Rate', 'N/A'))
        mrr_concat_pca = smart_round(all_metrics["metrics_concat_pca"].get('MRR', 'N/A'))

        mrr_score_sum = smart_round(all_metrics["metrics_score_sum"].get('MRR', 'N/A'))
        mrr_score_max = smart_round(all_metrics["metrics_score_max"].get('MRR', 'N/A'))
        mrr_score_prod = smart_round(all_metrics["metrics_score_product"].get('MRR', 'N/A'))
        mrr_score_mean = smart_round(all_metrics["metrics_score_mean"].get('MRR', 'N/A'))
        mrr_score_majority = smart_round(all_metrics["metrics_score_majority"].get('MRR', 'N/A'))
        string = (
            colorstr('bright_green', f"{neutral_dataset} E{len(dataset_enrolled)}Q{len(dataset_query)}: ") +
            f"{bold('Front RR1')}: {rank_1_front} {bold('MRR')}: {underscore(mrr_front)} | "
            f"{bold('Concat RR1')}: {rank_1_concat} {bold('MRR')}: {underscore(mrr_concat)} | "
            f"{bold('Concat_Mean RR1')}: {rank_1_concat_mean} {bold('MRR')}: {underscore(mrr_concat_mean)} | "
            f"{bold('Concat_PCA RR1')}: {rank_1_concat_pca} {bold('MRR')}: {underscore(mrr_concat_pca)} | "
            f"{bold('Score_sum MRR')}: {underscore(mrr_score_sum)} | "
            f"{bold('Score_prod MRR')}: {underscore(mrr_score_prod)} | "
            f"{bold('Score_mean MRR')}: {underscore(mrr_score_mean)} | "
            f"{bold('Score_max MRR')}: {underscore(mrr_score_max)} | "
            f"{bold('Score_maj MRR')}: {underscore(mrr_score_majority)} | "
            f"{bold('MV RR1')}: {rank_1_mv} {bold('MRR')}: {underscore(mrr_mv)} "
        )
    else:
        string = (
            colorstr('bright_green', f"{neutral_dataset} E{len(dataset_enrolled)}Q{len(dataset_query)}: ") +
            f"{bold('MV-RR1')}: {underscore(rank_1_mv)} {bold('MV-RR5')}: {rank_5_mv} {bold('MV-MRR')}: {mrr_mv} "
        )

    print(string)


def print_results_verification(neutral_dataset, dataset_enrolled, all_metrics, eval_all):

    def fmt_metric(metrics, key):
        """Format mean ± std string if available, otherwise fallback."""
        mean = metrics.get(key, 'N/A')
        std = metrics.get(f"{key}_std", None)
        if mean == 'N/A':
            return 'N/A'
        if std is not None:
            return f"{smart_round(mean)}±{smart_round(std)}"
        return smart_round(mean)

    auc_mvfa = smart_round(all_metrics["metrics_mvfa"].get("AUC", 'N/A'))
    acc_mvfa = fmt_metric(all_metrics["metrics_mvfa"], "Accuracy")

    if eval_all:
        auc_front = smart_round(all_metrics["metrics_front"].get("AUC", 'N/A'))
        acc_front = fmt_metric(all_metrics["metrics_front"], "Accuracy")
        auc_concat = smart_round(all_metrics["metrics_concat"].get("AUC", 'N/A'))
        acc_concat = fmt_metric(all_metrics["metrics_concat"], "Accuracy")
        auc_concat_mean = smart_round(all_metrics["metrics_concat_mean"].get("AUC", 'N/A'))
        acc_concat_mean = fmt_metric(all_metrics["metrics_concat_mean"], "Accuracy")
        auc_concat_pca = smart_round(all_metrics["metrics_concat_pca"].get("AUC", 'N/A'))
        auc_score_sum = smart_round(all_metrics["metrics_score_sum"].get('AUC', 'N/A'))
        auc_score_max = smart_round(all_metrics["metrics_score_max"].get('AUC', 'N/A'))
        auc_score_prod = smart_round(all_metrics["metrics_score_product"].get('AUC', 'N/A'))
        auc_score_mean = smart_round(all_metrics["metrics_score_mean"].get('AUC', 'N/A'))
        string = (
                colorstr('bright_green', f"{neutral_dataset} E{len(dataset_enrolled)}: ") +
                f"{bold('Front-AUC/Acc')}: {underscore(auc_front)} / {acc_front} | "
                f"{bold('Concat-AUC/Acc')}: {underscore(auc_concat)} / {acc_concat} | "
                f"{bold('Concat_Mean-AUC/Acc')}: {underscore(auc_concat_mean)} / {acc_concat_mean} | "
                f"{bold('Concat_PCA-AUC')}: {underscore(auc_concat_pca)} | "
                f"{bold('Score_sum-AUC')}: {underscore(auc_score_sum)} | "
                f"{bold('Score_prod-AUC')}: {underscore(auc_score_prod)} | "
                f"{bold('Score_mean-AUC')}: {underscore(auc_score_mean)} | "
                f"{bold('Score_max-AUC')}: {underscore(auc_score_max)} | "
                f"{bold('MV-AUC/Acc')}: {underscore(auc_mvfa)} / {acc_mvfa}"
        )
    else:
        string = (
                colorstr('bright_green', f"{neutral_dataset} E{len(dataset_enrolled)}: ") +
                f"{bold('MV-AUC/Acc')}: {underscore(auc_mvfa)} / {acc_mvfa}"
        )
    print(string)
