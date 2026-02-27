import time
from collections import namedtuple
from copy import deepcopy

import random
import numpy as np
import torch
import os
from numpy.linalg import norm
from tqdm import tqdm

from src.preprocess_datasets.rendering.Facerender import generate_rotation_matrices_cross_x_y, generate_rotation_matrices
from src.util.Plotter import analyze_embedding_distribution
from src.util.Voting import accuracy_front_perspective, concat, score_fusion, face_verification_from_similarity, \
    concat_mask, fuse_pairwise_scores, analyze_result_verification
from src.util.datapipeline.EmbeddingDataset import EmbeddingDataset
from src.util.misc import smart_round


def get_embeddings_mv(enrolled_loader, query_loader, disable_bar=False):
    """
    Calculate embeddings for enrolled and query datasets using a multi-view backbone.
    """
    enrolled_embeddings_reg = []
    enrolled_labels = []
    enrolled_scan_ids = []
    enrolled_true_perspectives = []
    enrolled_ref_perspectives = []
    for embeddings, labels, scan_id, true_perspectives, ref_perspectives in tqdm(iter(enrolled_loader), disable=disable_bar, desc="Generate Enrolled Embeddings"):
        enrolled_embeddings_reg.append(embeddings.permute(1, 0, 2))
        enrolled_labels.extend(deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        enrolled_scan_ids.extend(deepcopy(scan_id))
        enrolled_true_perspectives.append(np.asarray(deepcopy(true_perspectives)))
        enrolled_ref_perspectives.append(np.asarray(deepcopy(ref_perspectives)))

    enrolled_embeddings_reg = np.concatenate(enrolled_embeddings_reg, axis=1)
    enrolled_labels = np.array([t.item() for t in enrolled_labels])
    enrolled_scan_ids = np.array(enrolled_scan_ids)
    enrolled_true_perspectives = np.concatenate(enrolled_true_perspectives, axis=0)
    enrolled_ref_perspectives = np.concatenate(enrolled_ref_perspectives, axis=0)

    if query_loader is None:
        Results = namedtuple("Results", ["enrolled_embeddings", "enrolled_labels", "enrolled_scan_ids", "enrolled_true_perspectives", "enrolled_ref_perspectives"])
        return Results(enrolled_embeddings_reg, enrolled_labels, enrolled_scan_ids, enrolled_true_perspectives, enrolled_ref_perspectives)

    query_embeddings_reg = []
    query_labels = []
    query_scan_ids = []
    query_true_perspectives = []
    query_ref_perspectives = []
    for embeddings, labels, scan_id, true_perspectives, ref_perspectives in tqdm(iter(query_loader), disable=disable_bar,  desc="Generate Query Embeddings"):
        query_embeddings_reg.append(embeddings.permute(1, 0, 2))
        query_labels.extend(deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        query_scan_ids.extend(deepcopy(scan_id))
        query_true_perspectives.append(np.asarray(deepcopy(true_perspectives)))
        query_ref_perspectives.append(np.asarray(deepcopy(ref_perspectives)))

    query_embeddings_reg = np.concatenate(query_embeddings_reg, axis=1)
    query_labels = np.array([t.item() for t in query_labels])
    query_scan_ids = np.array(query_scan_ids)
    query_true_perspectives = np.concatenate(query_true_perspectives, axis=0)
    query_ref_perspectives = np.concatenate(query_ref_perspectives, axis=0)

    Results = namedtuple("Results",
                         ["enrolled_embeddings", "enrolled_labels", "enrolled_scan_ids", "enrolled_perspectives", "enrolled_ref_perspectives",
                          "query_embeddings", "query_labels", "query_scan_ids", "query_perspectives", "query_ref_perspectives"])
    return Results(enrolled_embeddings_reg, enrolled_labels, enrolled_scan_ids, enrolled_true_perspectives, enrolled_ref_perspectives,
                   query_embeddings_reg, query_labels, query_scan_ids, query_true_perspectives, query_ref_perspectives)


def evaluate_mv_emb_1_n(test_path, batch_size, views=None, shuffle_views=False, disable_bar: bool = True):
    """
    Evaluate 1:N Model Performance on given test dataset
    """
    dataset_enrolled_path = os.path.join(test_path, 'enrolled')
    dataset_query_path = os.path.join(test_path, 'query')

    # TODO: MASKED CONCAT
    # TODO make sure that it is no problem if query embedding has for some ids no instance!
    #euclidean_distance_thresh = 5
    dataset_enrolled = EmbeddingDataset(dataset_enrolled_path, views, shuffle_views=shuffle_views)
    #print("Enrolled Size before: "+ str(len(dataset_enrolled)))
    #removed_ids = dataset_enrolled.remove_samples(euclidean_distance_thresh=euclidean_distance_thresh)
    #print("Enrolled Size after: "+ str(len(dataset_enrolled)))
    enrolled_loader = torch.utils.data.DataLoader(dataset_enrolled, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

    dataset_query = EmbeddingDataset(dataset_query_path, views, shuffle_views=shuffle_views)
    #print("Query Size before: "+ str(len(dataset_query)))
    #removed_ids_query = dataset_query.remove_samples(euclidean_distance_thresh=euclidean_distance_thresh)
    #dataset_query.remove_by_ids(removed_ids)
    #print("Query Size after: "+ str(len(dataset_query)))
    query_loader = torch.utils.data.DataLoader(dataset_query, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

    if len(dataset_enrolled.classes) != len(dataset_enrolled):
        raise Exception(
            f"len(dataset_enrolled.classes): {len(dataset_enrolled.classes)} doesnt match len(dataset_enrolled.samples): {len(dataset_enrolled)} -> Check your dataset: {test_path}")

    time.sleep(0.1)

    embedding_library = get_embeddings_mv(enrolled_loader, query_loader, disable_bar)
    enrolled_labels, query_labels = embedding_library.enrolled_labels, embedding_library.query_labels

    all_metrics = {}

    # --------- Single Front View ---------
    metrics_front, sim_front, top_idx, y_true_front, y_pred_front = accuracy_front_perspective(embedding_library, string_mask=False)
    # plot_cmc(sim_front, enrolled_labels, query_labels, dataset_name, "front")
    # plot_rrk_histogram(query_labels, enrolled_labels, sim_front, dataset_name, "front")
    # error_rate_per_class(query_labels, enrolled_labels, top_idx, dataset_enrolled, embedding_library.query_scan_ids, sim_front, dataset_name, "_front")
    all_metrics["emb_dist_front"] = analyze_embedding_distribution(sim_front, query_labels, enrolled_labels, "", "front", plot=False)
    all_metrics["metrics_front"] = metrics_front
    all_metrics["verification_results_front"] = face_verification_from_similarity(sim_front, query_labels, enrolled_labels)
    del sim_front, top_idx, y_true_front, y_pred_front

    # --------- Concat Full ---------
    metrics_concat, sim_concat, top_idx, y_true_concat, y_pred_concat = concat(embedding_library, disable_bar)
    # plot_cmc(sim_concat, enrolled_labels, query_labels, dataset_name, "concat")
    # plot_rrk_histogram(query_labels, enrolled_labels, sim_concat, dataset_name, "concat")
    # error_rate_per_class(query_labels, enrolled_labels, top_idx, dataset_enrolled, embedding_library.query_scan_ids, sim_concat, dataset_name, "_concat")
    all_metrics["emb_dist_concat"] = analyze_embedding_distribution(sim_concat, query_labels, enrolled_labels, "", "concat", plot=False)
    all_metrics["metrics_concat"] = metrics_concat
    all_metrics["verification_results_concat"] = face_verification_from_similarity(sim_concat, query_labels, enrolled_labels)
    del sim_concat, top_idx, y_true_concat, y_pred_concat

    # --------- Masked Concat --------- TODO
    #metrics_concat_masked, sim_concat_masked, top_idx_masked, y_true_concat_masked, y_pred_concat_masked = concat_mask(embedding_library, disable_bar, euclidean_dist_thresh=20, mask_by_enrolled=True, mask_by_query=True)
    #all_metrics["emb_dist_concat_masked"] = analyze_embedding_distribution(sim_concat_masked, query_labels, enrolled_labels, "", "concat_masked", plot=False)
    #all_metrics["metrics_concat_masked"] = metrics_concat_masked
    #all_metrics["verification_results_concat_masked"] = face_verification_from_similarity(sim_concat_masked, query_labels, enrolled_labels)
    #del sim_concat_masked, top_idx_masked, y_true_concat_masked, y_pred_concat_masked

    # --------- Concat Mean ---------
    metrics_concat_mean, sim_concat_mean, top_indices_concat_mean, y_true_concat_mean, y_pred_concat_mean = concat(embedding_library, disable_bar, reduce_with="mean")
    # plot_cmc(similarity_matrix_concat_mean, enrolled_labels, query_labels, dataset_name, "concat_mean")
    # plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat_mean, dataset_name, "concat_mean")
    all_metrics["emb_dist_concat_mean"] = analyze_embedding_distribution(sim_concat_mean, query_labels, enrolled_labels, "", "concat_mean", plot=False)
    all_metrics["metrics_concat_mean"] = metrics_concat_mean
    all_metrics["verification_results_concat_mean"] = face_verification_from_similarity(sim_concat_mean, query_labels, enrolled_labels)
    del sim_concat_mean, top_indices_concat_mean, y_true_concat_mean, y_pred_concat_mean

    # --------- Concat Median ---------
    metrics_concat_median, similarity_matrix_concat_median, top_indices_concat_median, y_true_concat_median, y_pred_concat_median = concat(embedding_library, disable_bar, reduce_with="median")
    # plot_cmc(similarity_matrix_concat_median, enrolled_labels, query_labels, dataset_name, "concat_median")
    # plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat_median, dataset_name, "concat_median")
    # all_metrics["emb_dist_concat_median"] = analyze_embedding_distribution(similarity_matrix_concat_median, query_labels, enrolled_labels, "", "concat_median", plot=False)
    all_metrics["metrics_concat_median"] = metrics_concat_median
    del similarity_matrix_concat_median, top_indices_concat_median, y_true_concat_median, y_pred_concat_median

    # --------- Concat PCA ---------
    # metrics_concat_pca, similarity_matrix_concat_pca, top_indices_concat_pca, y_true_concat_pca, y_pred_concat_pca = concat(embedding_library, disable_bar, reduce_with="pca")
    # plot_cmc(similarity_matrix_concat_pca, enrolled_labels, query_labels, dataset_name, "concat_pca")
    # plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat_pca, dataset_name, "concat_pca")
    # all_metrics["metrics_concat_pca"] = metrics_concat_pca
    # del similarity_matrix_concat_pca, top_indices_concat_pca, y_true_concat_pca, y_pred_concat_pca

    # --------- Score fusion ---------
    fusion_methods = ["max", "product", "majority", "mean", "median"]
    sim_score = None
    for m in fusion_methods:
        metrics, sim_score, fused, top_idx, pred = score_fusion(embedding_library, disable_bar, method=m, similarity_matrix=sim_score, distance_matrix=None)
        all_metrics[f"metrics_score_{m}"] = metrics
        # all_metrics[f"emb_dist_score_{m}"] = analyze_embedding_distribution(fused, query_labels, enrolled_labels, dataset_name, f"score_{m}", plot=True)

    # plot_all_cmc_from_txt(dataset_name)

    return all_metrics, embedding_library, dataset_enrolled, dataset_query


def print_results(neutral_dataset, dataset_enrolled, dataset_query, all_metrics):
    rank_1_front = smart_round(all_metrics["metrics_front"].get('Rank-1 Rate', 'N/A'))
    mrr_front = smart_round(all_metrics["metrics_front"].get('MRR', 'N/A'))
    gbig_front = smart_round(all_metrics["emb_dist_front"].get('gbig', 'N/A'), rounding_prec=8)
    auc_front = smart_round(all_metrics["verification_results_front"].get('auc', 'N/A'), rounding_prec=8)

    rank_1_concat = smart_round(all_metrics["metrics_concat"].get('Rank-1 Rate', 'N/A'))
    mrr_concat = smart_round(all_metrics["metrics_concat"].get('MRR', 'N/A'))
    gbig_concat = smart_round(all_metrics["emb_dist_concat"].get('gbig', 'N/A'), rounding_prec=8)
    auc_concat = smart_round(all_metrics["verification_results_concat"].get('auc', 'N/A'), rounding_prec=8)

    #rank_1_concat_masked = smart_round(all_metrics["metrics_concat_masked"].get('Rank-1 Rate', 'N/A'))
    #mrr_concat_masked = smart_round(all_metrics["metrics_concat_masked"].get('MRR', 'N/A'))
    #gbig_concat_masked = smart_round(all_metrics["emb_dist_concat_masked"].get('gbig', 'N/A'), rounding_prec=8)
    #auc_concat_masked = smart_round(all_metrics["verification_results_concat_masked"].get('auc', 'N/A'), rounding_prec=8)

    rank_1_concat_mean = smart_round(all_metrics["metrics_concat_mean"].get('Rank-1 Rate', 'N/A'))
    mrr_concat_mean = smart_round(all_metrics["metrics_concat_mean"].get('MRR', 'N/A'))
    gbig_concat_mean = smart_round(all_metrics["emb_dist_concat_mean"].get('gbig', 'N/A'), rounding_prec=8)
    auc_concat_mean = smart_round(all_metrics["verification_results_concat_mean"].get('auc', 'N/A'), rounding_prec=8)

    rank_1_concat_median = smart_round(all_metrics["metrics_concat_median"].get('Rank-1 Rate', 'N/A'))
    mrr_concat_median = smart_round(all_metrics["metrics_concat_median"].get('MRR', 'N/A'))

    mrr_score_max = smart_round(all_metrics["metrics_score_max"].get('MRR', 'N/A'))

    mrr_score_prod = smart_round(all_metrics["metrics_score_product"].get('MRR', 'N/A'))

    mrr_score_mean = smart_round(all_metrics["metrics_score_mean"].get('MRR', 'N/A'))

    mrr_score_majority = smart_round(all_metrics["metrics_score_majority"].get('MRR', 'N/A'))

    # mrr_score_pdw = smart_round(all_metrics["metrics_score_pdw"].get('MRR', 'N/A'))
    string = (f"{neutral_dataset} E{len(dataset_enrolled)}Q{len(dataset_query)}: " +
              f"{'Front RR1'}: {rank_1_front} {'MRR'}: {mrr_front} {'GBIG'}: {gbig_front} {'AUC'}: {auc_front} | "  # {bold('GAIG')}: {underscore(gaig_front)} | "
              f"{'Concat RR1'}: {rank_1_concat} {'MRR'}: {mrr_concat} {'GBIG'}: {gbig_concat} {'AUC'}: {auc_concat} | "  # {bold('GAIG')}: {underscore(gaig_concat)} | "
              #f"{'Concat_Masked RR1'}: {rank_1_concat_masked} {'MRR'}: {mrr_concat_masked} {'GBIG'}: {gbig_concat_masked} {'AUC'}: {auc_concat_masked} | "
              f"{'Concat_Mean RR1'}: {rank_1_concat_mean} {'MRR'}: {mrr_concat_mean} {'GBIG'}: {gbig_concat_mean} {'AUC'}: {auc_concat_mean} | "
              f"{'Concat_Median RR1'}: {rank_1_concat_median} {'MRR'}: {mrr_concat_median} | "
              f"{'Score_prod MRR'}: {mrr_score_prod} | "
              f"{'Score_mean MRR'}: {mrr_score_mean} | "
              f"{'Score_max MRR'}: {mrr_score_max} | "
              f"{'Score_maj MRR'}: {mrr_score_majority} | "
              )
    print(string)


def evaluate_and_log_mv(data_root, test_views, batch_size, shuffle_views: bool, disable_bar: bool = True):
    print(f"Perform 1:N Evaluation on {data_root} with {test_views}")
    all_metrics, embedding_library, dataset_enrolled, dataset_query = evaluate_mv_emb_1_n(data_root, batch_size, test_views, shuffle_views, disable_bar)
    neutral_dataset = "Dataset: " + str(test_views)
    print_results(neutral_dataset, dataset_enrolled, dataset_query, all_metrics)
    return all_metrics


def evaluate_mv_1_1(dataset_enrolled_path, views, batch_size, disable_bar: bool, shuffle_views=False, norm_vector="view"):
    """
    Evaluate 1:1 Model Performance on given test dataset
    """
    dataset_name = os.path.basename(dataset_enrolled_path)

    def normalize_viewwise(x):
        # x shape: (views, dim)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    pair_list = []
    folds = []
    unique_sample_paths = set()
    with open(os.path.join(dataset_enrolled_path, "split.txt"), "r") as f:
        next(f)  # skip header
        for line in f:
            split, pair, name1, name2, is_same, is_same_corrected = line.strip().split(",")
            pair_list.append((name1, name2, int(is_same_corrected)))
            folds.append(int(split))
            unique_sample_paths.add(name1)
            unique_sample_paths.add(name2)

    dataset_enrolled = EmbeddingDataset(dataset_enrolled_path, views, shuffle_views=shuffle_views)
    enrolled_loader = torch.utils.data.DataLoader(dataset_enrolled, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
    time.sleep(0.1)

    embedding_library = get_embeddings_mv(enrolled_loader, None, disable_bar)

    embeddings_reg = embedding_library.enrolled_embeddings
    embeddings_reg_mean = embedding_library.enrolled_embeddings.mean(axis=0)
    name_to_class_dict = dataset_enrolled.class_to_idx
    #mask = np.array(["0_0" in perspective for perspective in embedding_library.enrolled_ref_perspectives])
    mask = np.all(embedding_library.enrolled_ref_perspectives == (0, 0), axis=(0, 2))

    embedding_lookup = {(c, s): i for i, (c, s) in enumerate(zip(embedding_library.enrolled_labels, embedding_library.enrolled_scan_ids))}

    # Evaluate pairs
    similarities_front = []
    similarities_concat = []
    similarities_concat_mean = []
    fusion_methods = ["product", "max", "mean"]
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

        emb1_reg, emb2_reg = embeddings_reg[:, i1, :], embeddings_reg[:, i2, :]
        emb1_reg_mean, emb2_reg_mean = embeddings_reg_mean[i1, :], embeddings_reg_mean[i2, :]

        # ----- Apply normalization mode -----
        if norm_vector == "view":
            emb1_reg = normalize_viewwise(emb1_reg)
            emb2_reg = normalize_viewwise(emb2_reg)

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

        # Fusion Methods
        for m in fusion_methods:
            sim_fused = fuse_pairwise_scores(emb1_reg, emb2_reg, method=m)
            similarities_fusion[m].append(sim_fused)

    # ---------- Verification Accuracy Analysis ----------
    all_metrics = {}

    sim_dict = {
        "_front": similarities_front,
        "_concat": similarities_concat,
        "_concat_mean": similarities_concat_mean,
    }

    for k, v in sim_dict.items():
        all_metrics[f"metrics{k}"] = analyze_result_verification(labels, v, dataset_name, k, folds=folds)

    for m in fusion_methods:
        all_metrics[f"metrics_score_{m}"] = analyze_result_verification(labels, similarities_fusion[m], dataset_name,f"_{m}", folds=folds)

    # ---------- Score Distribution Analysis ----------
    #for ext, sims in sim_dict.items():
    #    all_metrics[f"emb_dist_score_{ext}"] = analyze_verification_distribution(sims, labels, dataset_name, ext, plot=True)

    return all_metrics, embedding_library, dataset_enrolled


def print_results_verification(neutral_dataset, dataset_enrolled, all_metrics):

    def fmt_metric(metrics, key):
        """Format mean +- std string if available, otherwise fallback."""
        mean = metrics.get(key, 'N/A')
        std = metrics.get(f"{key}_std", None)
        if mean == 'N/A':
            return 'N/A'
        if std is not None:
            return f"{smart_round(mean)}±{smart_round(std)}"
        return smart_round(mean)

    auc_front = smart_round(all_metrics["metrics_front"].get("AUC", 'N/A'))
    acc_front = fmt_metric(all_metrics["metrics_front"], "Accuracy")
    auc_concat = smart_round(all_metrics["metrics_concat"].get("AUC", 'N/A'))
    acc_concat = fmt_metric(all_metrics["metrics_concat"], "Accuracy")
    auc_concat_mean = smart_round(all_metrics["metrics_concat_mean"].get("AUC", 'N/A'))
    acc_concat_mean = fmt_metric(all_metrics["metrics_concat_mean"], "Accuracy")
    auc_score_max = smart_round(all_metrics["metrics_score_max"].get('AUC', 'N/A'))
    acc_score_max = fmt_metric(all_metrics["metrics_score_max"], "Accuracy")
    auc_score_prod = smart_round(all_metrics["metrics_score_product"].get('AUC', 'N/A'))
    auc_score_mean = smart_round(all_metrics["metrics_score_mean"].get('AUC', 'N/A'))
    string = (
            f"{neutral_dataset} E{len(dataset_enrolled)}: " +
            f"{'Front-AUC/Acc'}: {auc_front} / {acc_front} | "
            f"{'Concat-AUC/Acc'}: {auc_concat} / {acc_concat} | "
            f"{'Concat_Mean-AUC/Acc'}: {auc_concat_mean} / {acc_concat_mean} | "
            f"{'Score_prod-AUC'}: {auc_score_prod} | "
            f"{'Score_mean-AUC'}: {auc_score_mean} | "
            f"{'Score_max-AUC'}: {auc_score_max} / {acc_score_max} | "
    )
    print(string)


def evaluate_and_log_mv_verification(data_root, test_views, batch_size, shuffle_views: bool, disable_bar: bool = True):

    print(f"Perform 1:1 Evaluation on {data_root} and folds: {10}")
    all_metrics, embedding_library, dataset_enrolled = evaluate_mv_1_1(os.path.join(data_root, data_root), test_views, batch_size, disable_bar=disable_bar, shuffle_views=shuffle_views)
    neutral_dataset = "Dataset: " + str(test_views)
    print_results_verification(neutral_dataset, dataset_enrolled, all_metrics)

    return all_metrics


def generate_view_subsets_sampled(
        base_set,
        max_k=30,
        samples_per_k=50,
        seed=0
):
    random.seed(seed)

    # Ensure 0_0 is present and fixed
    base_set = list(base_set)
    base_set.remove("0_0")
    other_elements = base_set

    sampled_subsets = []

    for k in range(2, max_k + 1):
        level = []

        if k == 1:
            level.append(["0_0"])
        else:
            seen = set()
            while len(level) < samples_per_k:
                perm = tuple(sorted(random.sample(other_elements, k - 1)))
                if perm in seen:
                    continue
                seen.add(perm)
                level.append(["0_0", *perm])

        sampled_subsets.append(level)

        print(f"Level {k}: {len(level)} sampled subsets")

    return sampled_subsets


def generate_cross_views(
        max_angle=35,
        step=5,
        yaw_zero = False,
        pitch_zero = False,
):
    """
    Generate views where yaw=0 OR pitch=0 (cross),
    sampled every `step` degrees up to `max_angle`.
    """
    angles = list(range(-max_angle, max_angle + 1, step))

    views = set()

    for a in angles:
        if pitch_zero:
            views.add((a, 0))  # yaw axis
        if yaw_zero:
            views.add((0, a))  # pitch axis

    return sorted(views)


def views_to_strings(views):
    return [f"{yaw}_{pitch}" for yaw, pitch in views]


def generate_cross_experiments(
        max_angles=(10, 15, 25, 35),
        steps=(1, 2, 3, 5, 10),
        yaw_zero=True,
        pitch_zero=True,
):
    experiments = []

    for max_a in max_angles:
        for step in steps:
            views = generate_cross_views(
                max_angle=max_a,
                step=step,
                yaw_zero=yaw_zero,
                pitch_zero=pitch_zero
            )
            experiments.append(views_to_strings(views))

    return experiments


def main_perspective_test():
    torch.multiprocessing.set_sharing_strategy('file_system')
    render_angles = [-35, -25, -15, -10, -5, 0, 5, 10, 15, 25, 35]
    all_views = (generate_rotation_matrices_cross_x_y() + generate_rotation_matrices(render_angles))

    yaw_pitch_pairs = [(x, y) for x, y, _ in all_views]
    unique_views = sorted(set(yaw_pitch_pairs))
    all_views_set = [f"{x}_{y}" for (x, y) in unique_views]

    allowed = [[['0_0']]]
    random_list = generate_view_subsets_sampled(all_views_set)
    allowed.extend(random_list)
    print(f"✅ Using {len(random_list)} random unique perspectives")

    for num_view_list in allowed:
        for selected_views in num_view_list:
            cfg_yaml = {"TEST_VIEWS": selected_views}
            DATA_ROOT = "F:\\Face\\data\\dataset15_emb\\test_rgb_bff_crop261_emb-irseglintr18\\"  # "/home/gustav/dataset14_emb/test_rgb_bff_crop261_emb-irseglintr18/"  # the parent root where the datasets are stored
            BATCH_SIZE = 16  # Batch size
            evaluate_and_log_mv(DATA_ROOT, cfg_yaml['TEST_VIEWS'], BATCH_SIZE, disable_bar=True)

    allowed = [['0_0']]

    cross_experiments1 = generate_cross_experiments(
        max_angles=[10, 15, 25, 35],
        steps=[1, 2, 3, 4, 5, 10, 15],
        yaw_zero=True,
        pitch_zero=True
    )
    print(f"✅ Using {len(cross_experiments1)} cross unique perspectives")
    allowed.extend(cross_experiments1)

    cross_experiments2 = generate_cross_experiments(
        max_angles=[10, 15, 25, 35],
        steps=[1, 2, 3, 4, 5, 10, 15],
        yaw_zero=True,
        pitch_zero=False
    )
    print(f"✅ Using {len(cross_experiments2)} cross unique perspectives")
    allowed.extend(cross_experiments2)

    cross_experiments3 = generate_cross_experiments(
        max_angles=[10, 15, 25, 35],
        steps=[1, 2, 3, 4, 5, 10, 15],
        yaw_zero=False,
        pitch_zero=True
    )
    print(f"✅ Using {len(cross_experiments3)} cross unique perspectives")
    allowed.extend(cross_experiments3)

    extras = [['0_0', '-25_0', '-10_0', '25_0', '10_0'],  # 1 Azimuth axis
              ['0_0', '0_-25', '0_-10', '0_10', '0_25'],  # 2 Alitude axis
              ['0_0', '-25_-25', '-10_-10', '10_10', '25_25'],  # 3 diagonal
              ['0_0', '-25_25', '-10_10', '10_-10', '25_-25'],  # 4 diagonal
              ['0_0', '-25_25', '-10_10', '10_-10', '25_-25', '-25_-25', '-10_-10', '10_10', '25_25'],  # 5 cross
              ['0_0', '-25_-25', '-25_25', '25_-25', '25_25'],  # 6 Only corners
              ['0_0', '-10_-10', '-10_10', '10_-10', '10_10'],  # 7 Only middle
              ['0_0', '25_-25', '25_-10', '25_0', '25_10', '25_25'],  # 8 Top row
              ['0_0', '10_-25', '10_-10', '10_0', '10_10', '10_25'],  # 9 2nd Top row
              ['0_0', '25_-25', '25_-10', '25_0', '25_10', '25_25', '10_-25', '10_-10', '10_0', '10_10', '10_25'],
              # 10 Top All
              ['0_0', '25_-25', '25_-10', '25_0', '25_10', '25_25', '10_-25', '10_-10', '10_0', '10_10', '10_25',
               '0_-25', '0_-10', '0_10', '0_25'],  # 11 Top All + Middle
              ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25'],  # 12 Top Corners + Sides
              ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0'],
              # 13 Top Corners + Sides + Mid  FAV! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
              ['0_0', '10_-10', '10_10', '-10_10', '-10_-10'],  # Inner Corners
              ['0_0', '10_-10', '10_10', '-10_10', '-10_-10', '-10_0', '10_0', '0_10', '0_-10'],  # Inner Ring
              ['0_0', '10_-10', '10_10', '10_0', '0_10', '0_-10'],  # Inner Top Ring
              ['-25_-25'],
              ['-25_-10'],
              ['-25_0'],
              ['-25_10'],
              ['-25_25'],
              ['-10_-25'],
              ['-10_-10'],
              ['-10_0'],
              ['-10_10'],
              ['-10_25'],
              ['0_-25'],
              ['0_-10'],
              ['0_0'],
              ['0_10'],
              ['0_25'],
              ['25_-25'],
              ['25_-10'],
              ['25_0'],
              ['25_10'],
              ['25_25'],
              ['10_-25'],
              ['10_-10'],
              ['10_0'],
              ['10_10'],
              ['10_25']
              ]
    allowed.extend(extras)

    extras2 = [
        # --- Pure large-angle axes ---
        ['0_0', '-35_0', '-15_0', '15_0', '35_0'],  # Yaw axis wide
        ['0_0', '0_-35', '0_-15', '0_15', '0_35'],  # Pitch axis wide

        # --- Large diagonals ---
        ['0_0', '-35_-35', '-15_-15', '15_15', '35_35'],  # Main diagonal
        ['0_0', '-35_35', '-15_15', '15_-15', '35_-35'],  # Anti-diagonal

        # --- Large cross ---
        ['0_0',
         '-35_0', '-15_0', '15_0', '35_0',
         '0_-35', '0_-15', '0_15', '0_35'],

        # --- Outer ring (big box) ---
        ['0_0',
         '-35_-35', '-35_0', '-35_35',
         '0_-35', '0_35',
         '35_-35', '35_0', '35_35'],

        # --- Outer corners only ---
        ['0_0', '-35_-35', '-35_35', '35_-35', '35_35'],

        # --- Top hemisphere emphasis (positive pitch) ---
        ['0_0',
         '-35_35', '-15_35', '15_35', '35_35',
         '-15_15', '0_15', '15_15'],

        # --- Bottom hemisphere emphasis (negative pitch) ---
        ['0_0',
         '-35_-35', '-15_-35', '15_-35', '35_-35',
         '-15_-15', '0_-15', '15_-15'],

        # --- Vertical slices at extreme yaw ---
        ['0_0', '-35_-35', '-35_-15', '-35_0', '-35_15', '-35_35'],
        ['0_0', '35_-35', '35_-15', '35_0', '35_15', '35_35'],

        # --- Horizontal slices at extreme pitch ---
        ['0_0', '-35_35', '-15_35', '0_35', '15_35', '35_35'],
        ['0_0', '-35_-35', '-15_-35', '0_-35', '15_-35', '35_-35'],

        # --- Asymmetric stress tests ---
        ['0_0', '-35_35', '-15_0', '15_-15', '35_-35'],
        ['0_0', '-35_-15', '-15_35', '15_0', '35_15'],

        # --- Sparse but extreme ---
        ['0_0', '-35_0', '0_35', '35_0'],
        ['0_0', '0_-35', '35_35', '-35_-35'],

        # --- Single extreme views (controls) ---
        ['35_35'],
        ['35_-35'],
        ['-35_35'],
        ['-35_-35'],

        # --- No-front (remove 0_0 entirely) ---
        ['-35_0', '-15_0', '15_0', '35_0'],
        ['0_-35', '0_-15', '0_15', '0_35'],
        ['-35_-35', '-15_-15', '15_15', '35_35'],

        # --- Only extremes, no mids ---
        ['0_0', '-35_0', '35_0'],
        ['0_0', '0_-35', '0_35'],
        ['0_0', '-35_-35', '35_35'],
        ['0_0', '-35_35', '35_-35'],

        # --- Horseshoe / U-shapes ---
        ['0_0', '-35_-35', '-35_0', '-35_35', '0_35'],
        ['0_0', '35_-35', '35_0', '35_35', '0_35'],
        ['0_0', '-35_-35', '0_-35', '35_-35', '35_0'],

        # --- L-shaped coverage ---
        ['0_0', '-35_0', '-35_35', '0_35'],
        ['0_0', '35_0', '35_-35', '0_-35'],
        ['0_0', '-15_0', '-15_15', '0_15'],

        # --- Skewed diagonals (non-symmetric) ---
        ['0_0', '-35_-15', '-15_35', '15_15', '35_-35'],
        ['0_0', '-35_15', '-15_-35', '15_-15', '35_35'],

        # --- One-side bias (yaw-heavy / pitch-light) ---
        ['0_0', '35_-15', '35_0', '35_15'],
        ['0_0', '-35_-15', '-35_0', '-35_15'],

        # --- Pitch-heavy / yaw-light ---
        ['0_0', '-15_35', '0_35', '15_35'],
        ['0_0', '-15_-35', '0_-35', '15_-35'],

        # --- Checkerboard sparse ---
        ['0_0', '-35_35', '-15_15', '15_-15', '35_-35'],
        ['0_0', '-35_-35', '-15_15', '15_-15', '35_35'],

        # --- Radius jump (center → extreme only) ---
        ['0_0', '-35_35'],
        ['0_0', '35_-35'],
        ['0_0', '0_35'],
        ['0_0', '35_0'],

        # --- Off-grid realism (simulates noisy capture bias) ---
        ['0_0', '-35_15', '-15_35', '15_35', '35_15'],
        ['0_0', '-35_-15', '-15_-35', '15_-35', '35_-15'],

        # --- Minimal but dangerous ---
        ['-35_35', '35_-35'],
        ['-35_-35', '35_35'],

        # --- Full main diagonal (yaw == pitch) ---
        ['-35_-35', '-25_-25', '-15_-15', '-10_-10', '-5_-5',
         '0_0',
         '5_5', '10_10', '15_15', '25_25', '35_35'],

        # --- Full anti-diagonal (yaw == -pitch) ---
        ['-35_35', '-25_25', '-15_15', '-10_10', '-5_5',
         '0_0',
         '5_-5', '10_-10', '15_-15', '25_-25', '35_-35'],

        # --- Full diagonal WITHOUT center ---
        ['-35_-35', '-25_-25', '-15_-15', '-10_-10', '-5_-5',
         '5_5', '10_10', '15_15', '25_25', '35_35'],

        # --- Full anti-diagonal WITHOUT center ---
        ['-35_35', '-25_25', '-15_15', '-10_10', '-5_5',
         '5_-5', '10_-10', '15_-15', '25_-25', '35_-35'],

        # --- Sparse diagonal (large steps only) ---
        ['-35_-35', '-15_-15', '0_0', '15_15', '35_35'],
        ['-35_35', '-15_15', '0_0', '15_-15', '35_-35'],

        # --- Extreme-only diagonal ---
        ['-35_-35', '35_35'],
        ['-35_35', '35_-35'],

        # --- Asymmetric diagonal sampling ---
        ['-35_-35', '-25_-25', '-10_-10', '0_0', '15_15', '35_35'],
        ['-35_35', '-25_25', '-10_10', '0_0', '15_-15', '35_-35'],

        # --- Half-diagonal (one-sided from center) ---
        ['0_0', '5_5', '10_10', '15_15', '25_25', '35_35'],
        ['0_0', '5_-5', '10_-10', '15_-15', '25_-25', '35_-35'],

        # --- Broken diagonal (gap stress test) ---
        ['-35_-35', '-25_-25', '0_0', '25_25', '35_35'],
        ['-35_35', '-25_25', '0_0', '25_-25', '35_-35'],
    ]
    allowed.extend(extras2)

    extras3 = [
        # Full Yaw
        ['0_0', '-35_0', '-30_0', '-25_0', '-20_0', '-15_0', '-10_0', '-5_0', '5_0', '10_0', '15_0', '20_0', '25_0', '30_0', '35_0'],  # Yaw
        ['0_0', '-35_0', '-30_0', '-20_0', '-10_0', '10_0', '20_0', '25_0', '30_0', '35_0'],  # Yaw less resolution
        ['0_0', '-35_0', '35_0'],  # Yaw less resolution
        ['0_0', '-25_0', '25_0'],  # Yaw less resolution
        ['0_0', '-20_0', '20_0'],  # Yaw less resolution
        ['0_0', '-15_0', '15_0'],  # Yaw less resolution
        ['0_0', '-35_0', '35_0', '-20_0', '20_0'],  # Yaw less resolution
        # Full Pitch
        ['0_0', '0_-35', '0_-30', '0_-25', '0_-20', '0_-15', '0_-10', '0_-5', '0_5', '0_10', '0_15', '0_20', '0_25', '0_30', '0_35'],  # Pitch
        ['0_0', '0_-35', '0_-30', '0_-20', '0_-10', '0_10', '0_20', '0_25', '0_30', '0_35'],  # Less resolution
        ['0_0', '0_-35', '0_35'],  # Less resolution
        ['0_0', '0_-25', '0_25'],  # Less resolution
        ['0_0', '0_-20', '0_20'],  # Less resolution
        ['0_0', '0_-15', '0_15'],  # Less resolution
        ['0_0', '0_-35', '0_35', '0_-20', '0_20'],  # Yaw less resolution
        # --- Full Cross ---
        ['0_0', '-35_0', '-30_0', '-25_0', '-20_0', '-15_0', '-10_0', '-5_0', '5_0', '10_0', '15_0', '20_0', '25_0', '30_0', '35_0', '0_-35', '0_-30', '0_-25', '0_-20', '0_-15', '0_-10', '0_-5', '0_5', '0_10', '0_15', '0_20', '0_25', '0_30', '0_35'],
        ['0_0', '-35_0', '-30_0', '-20_0', '-10_0', '10_0', '20_0', '25_0', '30_0', '35_0', '0_-35', '0_-30', '0_-20', '0_-10', '0_10', '0_20', '0_25', '0_30', '0_35'],
        ['0_0', '-35_0', '35_0', '0_-35', '0_35'],
        ['0_0', '-25_0', '25_0', '0_0', '0_-25', '0_25'],
        ['0_0', '-20_0', '20_0', '0_-20', '0_20'],
        ['0_0', '-15_0', '15_0', '0_-15', '0_15'],
        ['0_0', '-35_0', '35_0', '-20_0', '20_0', '0_-35', '0_35', '0_-20', '0_20'],
        ['0_0', '-35_0', '35_0', '0_-25', '0_25'],
        ['0_0', '-20_0', '20_0', '0_-10', '0_10'],
        ['0_0', '-35_0', '35_0', '-20_0', '20_0', '0_-10', '0_10', '0_-25', '0_25'],
    ]
    allowed.extend(extras3)

    extras4 = [
        ['-1_0', '0_0', '1_0'],
        ['-3_0', '0_0', '3_0'],
        ['-5_0', '0_0', '5_0'],
        ['-7_0', '0_0', '7_0'],
        ['-10_0', '0_0', '10_0'],
        ['-15_0', '0_0', '15_0'],
        ['-20_0', '0_0', '20_0'],
        ['-25_0', '0_0', '25_0'],
        ['-30_0', '0_0', '30_0'],
        ['-35_0', '0_0', '35_0'],

        ['-3_0', '-2_0', '0_0', '2_0', '3_0'],
        ['-5_0', '-2_0', '0_0', '2_0', '5_0'],
        ['-7_0', '-4_0', '0_0', '4_0', '7_0'],
        ['-10_0', '-5_0', '0_0', '5_0', '10_0'],
        ['-15_0', '-8_0', '0_0', '8_0', '15_0'],
        ['-20_0', '-10_0', '0_0', '10_0', '20_0'],
        ['-25_0', '-12_0', '0_0', '12_0', '25_0'],
        ['-30_0', '-15_0', '0_0', '15_0', '30_0'],
        ['-35_0', '-18_0', '0_0', '18_0', '35_0'],

        ['-3_0', '-2_0', '0_0', '2_0', '3_0', '0_5', '0_10'],
        ['-5_0', '-2_0', '0_0', '2_0', '5_0', '0_5', '0_10'],
        ['-7_0', '-4_0', '0_0', '4_0', '7_0', '0_5', '0_10'],
        ['-10_0', '-5_0', '0_0', '5_0', '10_0', '0_5', '0_10'],
        ['-15_0', '-8_0', '0_0', '8_0', '15_0', '0_5', '0_10'],
        ['-20_0', '-10_0', '0_0', '10_0', '20_0', '0_5', '0_10'],
        ['-25_0', '-12_0', '0_0', '12_0', '25_0', '0_5', '0_10'],
        ['-30_0', '-15_0', '0_0', '15_0', '30_0', '0_5', '0_10'],
        ['-35_0', '-18_0', '0_0', '18_0', '35_0', '0_5', '0_10'],

        ['-3_0', '-2_0', '0_0', '2_0', '3_0', '0_-5', '0_-10'],
        ['-5_0', '-2_0', '0_0', '2_0', '5_0', '0_-5', '0_-10'],
        ['-7_0', '-4_0', '0_0', '4_0', '7_0', '0_-5', '0_-10'],
        ['-10_0', '-5_0', '0_0', '5_0', '10_0', '0_-5', '0_-10'],
        ['-15_0', '-8_0', '0_0', '8_0', '15_0', '0_-5', '0_-10'],
        ['-20_0', '-10_0', '0_0', '10_0', '20_0', '0_-5', '0_-10'],
        ['-25_0', '-12_0', '0_0', '12_0', '25_0', '0_-5', '0_-10'],
        ['-30_0', '-15_0', '0_0', '15_0', '30_0', '0_-5', '0_-10'],
        ['-35_0', '-18_0', '0_0', '18_0', '35_0', '0_-5', '0_-10'],
    ]
    allowed.extend(extras4)

    extras5 = [
        ['0_-1', '0_0', '0_1'],
        ['0_-3', '0_0', '0_3'],
        ['0_-5', '0_0', '0_5'],
        ['0_-7', '0_0', '0_7'],
        ['0_-10', '0_0', '0_10'],
        ['0_-15', '0_0', '0_15'],
        ['0_-20', '0_0', '0_20'],
        ['0_-25', '0_0', '0_25'],
        ['0_-30', '0_0', '0_30'],
        ['0_-35', '0_0', '0_35'],

        ['0_-3', '0_-2', '0_0', '0_2', '0_3'],
        ['0_-5', '0_-2', '0_0', '0_2', '0_5'],
        ['0_-7', '0_-4', '0_0', '0_4', '0_7'],
        ['0_-10', '0_-5', '0_0', '0_5', '0_10'],
        ['0_-15', '0_-8', '0_0', '0_8', '0_15'],
        ['0_-20', '0_-10', '0_0', '0_10', '0_20'],
        ['0_-25', '0_-12', '0_0', '0_12', '0_25'],
        ['0_-30', '0_-15', '0_0', '0_15', '0_30'],
        ['0_-35', '0_-18', '0_0', '0_18', '0_35'],

        ['0_-3', '0_-2', '0_0', '0_2', '0_3', '5_0', '10_0'],
        ['0_-5', '0_-2', '0_0', '0_2', '0_5', '5_0', '10_0'],
        ['0_-7', '0_-4', '0_0', '0_4', '0_7', '5_0', '10_0'],
        ['0_-10', '0_-5', '0_0', '0_5', '0_10', '5_0', '10_0'],
        ['0_-15', '0_-8', '0_0', '0_8', '0_15', '5_0', '10_0'],
        ['0_-20', '0_-10', '0_0', '0_10', '0_20', '5_0', '10_0'],
        ['0_-25', '0_-12', '0_0', '0_12', '0_25', '5_0', '10_0'],
        ['0_-30', '0_-15', '0_0', '0_15', '0_30', '5_0', '10_0'],
        ['0_-35', '0_-18', '0_0', '0_18', '0_35', '5_0', '10_0'],

        ['0_-3', '0_-2', '0_0', '0_2', '0_3', '-5_0', '-10_0'],
        ['0_-5', '0_-2', '0_0', '0_2', '0_5', '-5_0', '-10_0'],
        ['0_-7', '0_-4', '0_0', '0_4', '0_7', '-5_0', '-10_0'],
        ['0_-10', '0_-5', '0_0', '0_5', '0_10', '-5_0', '-10_0'],
        ['0_-15', '0_-8', '0_0', '0_8', '0_15', '-5_0', '-10_0'],
        ['0_-20', '0_-10', '0_0', '0_10', '0_20', '-5_0', '-10_0'],
        ['0_-25', '0_-12', '0_0', '0_12', '0_25', '-5_0', '-10_0'],
        ['0_-30', '0_-15', '0_0', '0_15', '0_30', '-5_0', '-10_0'],
        ['0_-35', '0_-18', '0_0', '0_18', '0_35', '-5_0', '-10_0'],
    ]
    allowed.extend(extras5)

    for selected_views in allowed:
        cfg_yaml = {"TEST_VIEWS": selected_views}
        DATA_ROOT = "F:\\Face\\data\\dataset15_emb\\test_rgb_bff_crop261_emb-irseglintr18\\"  # "/home/gustav/dataset14_emb/test_rgb_bff_crop261_emb-irseglintr18/"  # the parent root where the datasets are stored
        BATCH_SIZE = 16  # Batch size
        evaluate_and_log_mv(DATA_ROOT, cfg_yaml['TEST_VIEWS'], BATCH_SIZE, disable_bar=True, shuffle_views=False)

    print("DONE")


def dataset_test():
    root = "/home/gustav/dataset15_emb/glint18/"  # "F:\\Face\\data\\dataset15_emb\\"
    TEST_SETS = [root+"test_rgb_bff_crop5E08_emb-glint_r18",
                 root+"test_rgb_bff_crop5E08_emb-glint_r50",
                 root+"test_rgb_bff_crop5E08_emb-glint_r100",
                 root+"test_rgb_bff_crop5E08_emb-ms1mv3_r18",
                 root+"test_rgb_bff_crop5E08_emb-ms1mv3_r50",
                 root+"test_rgb_bff_crop5E08_emb-ms1mv3_r100",
                 root+"test_rgb_bff_crop5E08_emb-adaface_ms1mv3",
                 root+"test_rgb_bff_crop5E08_emb-adaface_webface12m",
                 root+"test_rgb_bff_crop5E08_emb-edgeface_xs",
                 root+"test_rgb_bff_crop5E08_emb-hyperface50k",]


    print("Shuffle False")
    for DATA_ROOT in TEST_SETS:
        # cfg_yaml = {"TEST_VIEWS": ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0']}
        cfg_yaml = {"TEST_VIEWS": ['0_-25', '0_-10', '0_0', '0_10', '0_25']}
        BATCH_SIZE = 16
        evaluate_and_log_mv(DATA_ROOT, cfg_yaml['TEST_VIEWS'], BATCH_SIZE, shuffle_views=False, disable_bar=True)
        #evaluate_and_log_mv_verification(DATA_ROOT, cfg_yaml['TEST_VIEWS'], BATCH_SIZE, shuffle_views=False, disable_bar=True)


def single_dataset_test():
    cfg_yaml = {"TEST_VIEWS": ['0_-25', '0_-10', '0_0', '0_10', '0_25']}
    BATCH_SIZE = 16  # Batch size
    root = "F:\\Face\\data\\dataset15_emb\\"
    #DATA_ROOT = root+"test_nersemble_crop5-v15_emb-ms1mv3_r100" # "test_vox2test_crop5-v15_emb-glint_r18"##
    #evaluate_and_log_mv(DATA_ROOT, cfg_yaml['TEST_VIEWS'], BATCH_SIZE, shuffle_views=False, disable_bar=True)
    #DATA_ROOT = root+"test_nersemble_crop5-v15_emb-ms1mv3_r18"
    #evaluate_and_log_mv(DATA_ROOT, cfg_yaml['TEST_VIEWS'], BATCH_SIZE, shuffle_views=False, disable_bar=True)
    DATA_ROOT = root + "test_ytf_crop5_emb-ms1mv3_r100"
    evaluate_and_log_mv_verification(DATA_ROOT, cfg_yaml['TEST_VIEWS'], BATCH_SIZE, shuffle_views=False, disable_bar=True)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    SEED = 42
    torch.manual_seed(SEED)
    # main_perspective_test()
    # dataset_test()
    single_dataset_test()


