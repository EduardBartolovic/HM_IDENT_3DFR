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

    # --- Visualization
    if plot:
        correct_mask = correct == 1
        incorrect_mask = correct == 0

        fig, axs = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)

        # --- Query perspective histogram
        sns.histplot(
            [query_distances_grouped[correct_mask], query_distances_grouped[incorrect_mask]],
            bins=40,
            stat="percent",
            alpha=0.7,
            palette=["tab:green", "tab:red"],
            multiple="dodge",
            ax=axs[0],
            legend=True
        )
        axs[0].set_title("Query Perspective Distance")
        axs[0].set_xlabel("Query Distance")
        axs[0].set_ylabel("Percentage")
        axs[0].legend(labels=["Correct", "Incorrect"])

        # --- Enrolled perspective histogram
        sns.histplot(
            [top1_enrolled_dist[correct_mask], top1_enrolled_dist[incorrect_mask]],
            bins=40,
            stat="percent",
            alpha=0.7,
            palette=["tab:green", "tab:red"],
            multiple="dodge",
            ax=axs[1],
            legend=True
        )
        axs[1].set_title("Enrolled Perspective Distance")
        axs[1].set_xlabel("Enrolled Distance")
        axs[1].set_ylabel("Percentage")
        axs[1].legend(labels=["Correct", "Incorrect"])

        plt.savefig(os.path.join(save_path, "perspective_correlation_hist.png"), dpi=200)
        plt.close()

    # --- Per-view analysis using distance_matrix_avg
    top1_distance_matrix_avg = distance_matrix_avg[np.arange(len(query_labels)), top_indices[:, 0]]
    mean_per_view_correct = np.mean(top1_distance_matrix_avg[correct == 1], axis=0)
    mean_per_view_incorrect = np.mean(top1_distance_matrix_avg[correct == 0], axis=0)

    pearson_corr_top1_avg, _ = stats.pearsonr(top1_distance_matrix_avg, correct)

    # Optional plot
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(mean_per_view_correct, label="Correct", marker='o')
        plt.plot(mean_per_view_incorrect, label="Incorrect", marker='x')
        plt.title("Average Per-View Distance (Top-1)")
        plt.xlabel("View Index")
        plt.ylabel("Avg Euclidean Distance")
        plt.legend()
        if save_path:
            plt.savefig(os.path.join(save_path, "per_view_distance_analysis.png"), dpi=200)
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
