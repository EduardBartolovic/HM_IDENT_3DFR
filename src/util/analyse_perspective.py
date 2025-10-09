import os

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def str_to_xy(arr, reshape=True):
    """Convert 'x_y' strings to integer pairs"""
    xy = np.array([list(map(int, s.split('_'))) for s in arr.reshape(-1)])
    if reshape:
        return xy.reshape(*arr.shape, 2)
    else:
        return xy


def calc_perspective_distances(target_perspectives, true_perspectives):
    """
    Calculate Euclidean distances between enrolled and true perspectives.

    Args:
        target_perspectives: np.ndarray of shape (8,) with string values like 'x_y'
        true_perspectives: np.ndarray of shape (samples, 8) with same pattern

    Returns:
        np.ndarray of shape (samples, 8) with Euclidean distances
    """
    enrolled_xy = str_to_xy(target_perspectives)  # shape (8, 2)
    true_xy = str_to_xy(true_perspectives)  # shape (samples, 8, 2)

    # Compute Euclidean distance for each sample and each perspective
    distances = np.linalg.norm(true_xy - enrolled_xy, axis=2)  # shape (samples, 8)

    return distances


def compute_per_view_distance_matrix(query_perspectives, enrolled_perspectives):
    """
    Compute per-view Euclidean distances between query and enrolled perspectives.

    Args:
        query_perspectives (np.ndarray): shape (num_queries, num_views), strings like "x_y"
        enrolled_perspectives (np.ndarray): shape (num_enrolled, num_views), strings like "x_y"

    Returns:
        np.ndarray: shape (num_queries, num_enrolled, num_views), distances per view
        np.ndarray: shape (num_queries, num_enrolled), avg. distances per view
    """

    query_xy = str_to_xy(query_perspectives)        # shape (num_queries, num_views, 2)
    enrolled_xy = str_to_xy(enrolled_perspectives)  # shape (num_enrolled, num_views, 2)

    # Expand dimensions to broadcast subtraction
    query_exp = query_xy[:, np.newaxis, :, :]      # shape (num_queries, 1, num_views, 2)
    enrolled_exp = enrolled_xy[np.newaxis, :, :, :]  # shape (1, num_enrolled, num_views, 2)

    # Compute Euclidean distances per view
    distance_matrix = np.linalg.norm(query_exp - enrolled_exp, axis=-1)  # shape (num_queries, num_enrolled, num_views)
    distance_matrix_avg = np.mean(distance_matrix, axis=2)  # shape (num_queries, num_enrolled)

    return distance_matrix, distance_matrix_avg


def analyze_perspective_error_correlation(query_labels, enrolled_labels, query_distances, enrolled_distances, top_indices, distance_matrix_avg, save_path=None, plot=True):
    """
    Analyze how perspective distance (query + enrolled) correlates with recognition accuracy.

    Args:
        query_labels (np.ndarray): Labels of query samples.
        enrolled_labels (np.ndarray): Labels of enrolled samples.
        query_distances (np.ndarray): Euclidean distances between query perspective and its true perspective.
        enrolled_distances (np.ndarray): Euclidean distances between enrolled perspective and its true perspective.
        distance_matrix_avg (np.array): Matrix of distances between enrolled and query
        top_indices (np.ndarray): Top ranked indices (from compute_ranking_matrices).
        save_path (str, optional): Path to save the plots.
        plot (bool): Whether to generate plots.

    Returns:
        dict: Summary metrics (correlations + grouped stats).
    """
    enrolled_distances_grouped = np.mean(np.abs(enrolled_distances), axis=1)
    query_distances_grouped = np.mean(np.abs(query_distances), axis=1)

    # --- Per-query correctness
    top1_preds = enrolled_labels[top_indices[:, 0]]
    correct = (top1_preds == query_labels).astype(int)

    if np.all(enrolled_distances_grouped == enrolled_distances_grouped[0]) or np.all(correct == correct[0]):
        return {}

    # --- Combine distances per query
    # Map enrolled distances to top-1 matched sample
    top1_enrolled_dist = enrolled_distances_grouped[top_indices[:, 0]]

    # Compute combined (mean) perspective distance per pair
    combined_dist = (query_distances_grouped + top1_enrolled_dist) / 2

    # ---  Correlations
    pearson_q, _ = stats.pearsonr(query_distances_grouped, correct)
    pearson_e, _ = stats.pearsonr(top1_enrolled_dist, correct)
    pearson_c, _ = stats.pearsonr(combined_dist, correct)

    # --- Per-view analysis using distance_matrix_avg
    top1_distance_matrix_avg = distance_matrix_avg[np.arange(len(query_labels)), top_indices[:, 0]]
    mean_per_view_correct = np.mean(top1_distance_matrix_avg[correct == 1], axis=0)
    mean_per_view_incorrect = np.mean(top1_distance_matrix_avg[correct == 0], axis=0)

    pearson_corr_top1_avg, _ = stats.pearsonr(top1_distance_matrix_avg, correct)

    # --- Visualization
    if plot:
        correct_mask = correct == 1
        incorrect_mask = correct == 0

        fig, axs = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)

        # --- Query perspective histogram
        sns.histplot(
            [query_distances_grouped[correct_mask], query_distances_grouped[incorrect_mask]],
            bins=40,
            stat="percent",
            alpha=0.7,
            palette=["tab:blue", "tab:red"],
            multiple="dodge",
            common_norm=False,
            ax=axs[0],
            legend=True
        )
        mean_correct = np.mean(query_distances[correct_mask])
        mean_incorrect = np.mean(query_distances[incorrect_mask])
        axs[0].axvline(mean_correct, color="tab:blue", linestyle="--", linewidth=2, label=f"Correct μ={mean_correct:.2f}")
        axs[0].axvline(mean_incorrect, color="tab:red", linestyle="--", linewidth=2, label=f"Incorrect μ={mean_incorrect:.2f}")
        axs[0].set_title("Query")
        axs[0].set_xlabel("perspective angle error")
        axs[0].set_ylabel("Percentage")
        axs[0].legend()

        # --- Enrolled perspective histogram
        sns.histplot(
            [top1_enrolled_dist[correct_mask], top1_enrolled_dist[incorrect_mask]],
            bins=40,
            stat="percent",
            alpha=0.7,
            palette=["tab:blue", "tab:red"],
            multiple="dodge",
            common_norm=False,
            ax=axs[1],
            legend=True
        )
        mean_correct = np.mean(top1_enrolled_dist[correct_mask])
        mean_incorrect = np.mean(top1_enrolled_dist[incorrect_mask])
        axs[1].axvline(mean_correct, color="tab:blue", linestyle="--", linewidth=2, label=f"Correct μ={mean_correct:.2f}")
        axs[1].axvline(mean_incorrect, color="tab:red", linestyle="--", linewidth=2, label=f"Incorrect μ={mean_incorrect:.2f}")
        axs[1].set_title("Enrolled")
        axs[1].set_xlabel("perspective angle error")
        axs[1].set_ylabel("Percentage")
        axs[1].legend()

        # -- Matrix
        sns.histplot(
            [top1_distance_matrix_avg[correct_mask], top1_distance_matrix_avg[incorrect_mask]],
            bins=40,
            stat="percent",
            alpha=0.7,
            palette=["tab:blue", "tab:red"],
            multiple="dodge",
            common_norm=False,
            ax=axs[2],
            legend=True
        )
        mean_correct = np.mean(top1_distance_matrix_avg[correct_mask])
        mean_incorrect = np.mean(top1_distance_matrix_avg[incorrect_mask])
        axs[2].axvline(mean_correct, color="tab:blue", linestyle="--", linewidth=2, label=f"Correct μ={mean_correct:.2f}")
        axs[2].axvline(mean_incorrect, color="tab:red", linestyle="--", linewidth=2, label=f"Incorrect μ={mean_incorrect:.2f}")
        axs[2].set_title("Matrix Pairwise Distance")
        axs[2].set_xlabel("perspective angle error")
        axs[2].set_ylabel("Percentage")
        axs[2].legend()

        plt.savefig(os.path.join(save_path, "perspective_correlation_hist.png"), dpi=200)
        plt.close()

    return {
        "pearson_corr_query": pearson_q,
        "pearson_corr_enrolled": pearson_e,
        "pearson_corr_combined": pearson_c,
        "pearson_corr_top1_avg": pearson_corr_top1_avg,
        "mean_query_dist_correct": np.mean(query_distances[correct == 1]),
        "mean_query_dist_incorrect": np.mean(query_distances[correct == 0]),
        "mean_enrolled_dist_correct": np.mean(top1_enrolled_dist[correct == 1]),
        "mean_enrolled_dist_incorrect": np.mean(top1_enrolled_dist[correct == 0]),
        "mean_combined_dist_correct": np.mean(combined_dist[correct == 1]),
        "mean_combined_dist_incorrect": np.mean(combined_dist[correct == 0]),
        "mean_per_view_dist_correct": mean_per_view_correct,
        "mean_per_view_dist_incorrect": mean_per_view_incorrect
    }


def analyze_perspective_error_correlation_1v1(
    pair_list,
    enrolled_distances,
    embedding_library,
    labels,
    name_to_class_dict,
    true_perspectives,
    save_path=None,
    plot=True
):
    """
    Analyze how perspective distance correlates with 1:1 verification correctness.

    Args:
        pair_list (list of tuples): (name1, name2, is_same)
        enrolled_distances (np.ndarray): shape (num_samples, num_views) - per-sample perspective distances
        embedding_library: object containing enrolled_perspectives, enrolled_true_perspectives, enrolled_labels, enrolled_scan_ids
        labels (list[int]): ground truth 1/0 for each pair (same or not)
        name_to_class_dict: from name string to class id
        save_path (str, optional): directory to save figures
        plot (bool): whether to generate plots

    Returns:
        dict: correlation metrics and grouped means
    """
    # --- Extract useful mappings
    enrolled_labels = embedding_library.enrolled_labels
    enrolled_scan_ids = embedding_library.enrolled_scan_ids
    enrolled_dist_grouped = np.mean(np.abs(enrolled_distances), axis=1)  # mean distance per sample

    # Build lookup for distance per (class, scan)
    class_scan_to_dist = {
        (c, s): d for c, s, d in zip(enrolled_labels, enrolled_scan_ids, enrolled_dist_grouped)
    }

    true_perspectives = str_to_xy(true_perspectives)
    class_scan_to_true_perspectives = {
        (c, s): d for c, s, d in zip(enrolled_labels, enrolled_scan_ids, true_perspectives)
    }

    # --- Compute pairwise combined distances
    dist1, dist2, combined, pair_distances = [], [], [], []
    for name1, name2, _ in pair_list:
        class1, sample1 = name1.split("/")
        class2, sample2 = name2.split("/")

        sample1 = (sample1 + 'X' * 15)[:15]
        sample2 = (sample2 + 'X' * 15)[:15]
        class1 = class1.lstrip()
        class2 = class2.lstrip()

        class_idx1 = name_to_class_dict[class1]
        class_idx2 = name_to_class_dict[class2]

        d1 = class_scan_to_dist.get((class_idx1, sample1))
        d2 = class_scan_to_dist.get((class_idx2, sample2))

        dist1.append(d1)
        dist2.append(d2)

        p1 = class_scan_to_true_perspectives.get((class_idx1, sample1))
        p2 = class_scan_to_true_perspectives.get((class_idx2, sample2))
        pair_distance = np.linalg.norm(p1 - p2, axis=1).mean()
        pair_distances.append(pair_distance)
        combined.append(np.nanmean([d1, d2]))

    dist1 = np.array(dist1)
    dist2 = np.array(dist2)
    combined = np.array(combined)
    labels = np.array(labels)
    pair_distances = np.array(pair_distances)

    # --- Remove NaN pairs if any
    valid_mask = ~np.isnan(combined)
    dist1 = dist1[valid_mask]
    dist2 = dist2[valid_mask]
    combined = combined[valid_mask]
    labels = labels[valid_mask]

    # --- Compute Pearson correlations
    pearson_1, _ = stats.pearsonr(dist1, labels)
    pearson_2, _ = stats.pearsonr(dist2, labels)
    pearson_c, _ = stats.pearsonr(combined, labels)
    pearson_p, _ = stats.pearsonr(pair_distances, labels)

    # --- Mean distances for same vs diff
    mean_1_same = np.mean(dist1[labels == 1])
    mean_1_diff = np.mean(dist1[labels == 0])
    mean_2_same = np.mean(dist2[labels == 1])
    mean_2_diff = np.mean(dist2[labels == 0])
    mean_c_same = np.mean(combined[labels == 1])
    mean_c_diff = np.mean(combined[labels == 0])

    # --- Visualization
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)

        sns.histplot(
            [dist1[labels == 1], dist1[labels == 0]],
            bins=40, stat="percent", alpha=0.7, palette=["tab:blue", "tab:red"],
            multiple="dodge", ax=axs[0]
        )
        axs[0].axvline(mean_1_same, color="tab:blue", linestyle="--", label=f"Same μ={mean_1_same:.2f}")
        axs[0].axvline(mean_1_diff, color="tab:red", linestyle="--", label=f"Diff μ={mean_1_diff:.2f}")
        axs[0].set_title("Perspective Distance - Sample 1")
        axs[0].set_xlabel("Distance")
        axs[0].set_ylabel("Percentage")
        axs[0].legend()

        sns.histplot(
            [dist2[labels == 1], dist2[labels == 0]],
            bins=40, stat="percent", alpha=0.7, palette=["tab:blue", "tab:red"],
            multiple="dodge", ax=axs[1]
        )
        axs[1].axvline(mean_2_same, color="tab:blue", linestyle="--", label=f"Same μ={mean_2_same:.2f}")
        axs[1].axvline(mean_2_diff, color="tab:red", linestyle="--", label=f"Diff μ={mean_2_diff:.2f}")
        axs[1].set_title("Perspective Distance - Sample 2")
        axs[1].set_xlabel("Distance")
        axs[1].set_ylabel("Percentage")
        axs[1].legend()

        # -- Matrix
        sns.histplot(
            [pair_distances[labels == 1], pair_distances[labels == 0]],
            bins=40,
            stat="percent",
            alpha=0.7,
            palette=["tab:blue", "tab:red"],
            multiple="dodge",
            common_norm=False,
            ax=axs[2],
            legend=True
        )
        mean_correct = np.mean(pair_distances[labels == 1])
        mean_incorrect = np.mean(pair_distances[labels == 0])
        axs[2].axvline(mean_correct, color="tab:blue", linestyle="--", linewidth=2, label=f"Correct μ={mean_correct:.2f}")
        axs[2].axvline(mean_incorrect, color="tab:red", linestyle="--", linewidth=2, label=f"Incorrect μ={mean_incorrect:.2f}")
        axs[2].set_title("Matrix Pairwise Distance")
        axs[2].set_xlabel("perspective angle error")
        axs[2].set_ylabel("Percentage")
        axs[2].legend()

        if save_path:
            plt.savefig(os.path.join(save_path, "perspective_correlation_1v1.png"), dpi=200)
        plt.close()

    return {
        "pearson_corr_sample1": pearson_1,
        "pearson_corr_sample2": pearson_2,
        "pearson_corr_combined": pearson_c,
        "pearson_pairwise": pearson_p,
        "mean_sample1_same": mean_1_same,
        "mean_sample1_diff": mean_1_diff,
        "mean_sample2_same": mean_2_same,
        "mean_sample2_diff": mean_2_diff,
        "mean_combined_same": mean_c_same,
        "mean_combined_diff": mean_c_diff,
    }
