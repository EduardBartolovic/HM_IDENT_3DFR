import os

import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA, IncrementalPCA
import numba
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.util.Plotter import plot_verification

# Set environment variables to avoid OpenBLAS conflicts
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"


def compute_ranking_matrices(similarity_matrix):
    top_indices = np.argsort(-similarity_matrix, axis=1)  # Negative for descending order
    top_values = np.take_along_axis(similarity_matrix, top_indices, axis=1)
    return top_indices, top_values


def analyze_result(similarity_matrix, top_indices, reference_ids, ground_truth_ids, top_k_acc_k=5):
    num_inferences = similarity_matrix.shape[0]

    predicted_labels = reference_ids[top_indices[:, 0]]
    top_1_accuracy = accuracy_score(ground_truth_ids, predicted_labels)

    top_k_matches = reference_ids[top_indices[:, :top_k_acc_k]]
    top_k_accuracy = np.mean([ground_truth_ids[i] in top_k_matches[i] for i in range(num_inferences)])

    true_match_scores = np.array(
        [similarity_matrix[i, np.where(reference_ids == ground_truth_ids[i])[0][0]] for i in range(num_inferences)])
    mean_true_match_similarity = np.mean(true_match_scores)

    false_match_scores = []
    for i in range(num_inferences):
        top1_ref_idx = top_indices[i, 0]
        predicted_label = reference_ids[top1_ref_idx]
        if predicted_label != ground_truth_ids[i]:
            false_match_scores.append(similarity_matrix[i, top1_ref_idx])
    mean_false_match_similarity = np.mean(false_match_scores) if false_match_scores else 0

    # Mean Reciprocal Rank (MRR)
    reciprocal_ranks = []
    for i in range(num_inferences):
        gt_indices = np.where(reference_ids == ground_truth_ids[i])[0]
        if len(gt_indices) == 0:
            continue  # ground truth not in reference set

        ranks = np.where(np.isin(top_indices[i], gt_indices))[0]
        if len(ranks) > 0:
            reciprocal_ranks.append(1.0 / (ranks[0] + 1))  # rank is 1-based
        else:
            reciprocal_ranks.append(0.0)
    mean_reciprocal_rank = np.mean(reciprocal_ranks)

    return {
        "Rank-1 Rate": round(top_1_accuracy * 100, 4),
        f"Rank-{top_k_acc_k} Rate": round(top_k_accuracy * 100, 4),
        "mean_true_match_similarity": mean_true_match_similarity,
        "mean_false_match_similarity": mean_false_match_similarity,
        "MRR": round(mean_reciprocal_rank * 100, 6),
    }


def analyze_result_verification(labels, similarities_mv,
                                dataset_name,
                                method_appendix="",
                                far_targets=(1e-6, 1e-4),
                                k_folds=10,
                                random_state=42,
                                plot=False
                                ):
    """
    K-Fold evaluation for 1:1 face verification.

    Args:
        labels: Ground-truth binary labels (1=same, 0=different)
        similarities_mv: Similarity scores between pairs
        dataset_name: Name of the dataset (for logging/plotting)
        method_appendix: Extra string for logging/plotting
        far_targets: Tuple of FARs to compute TAR at (default: 1e-6 and 1e-4)
        k_folds: Number of cross-validation folds (default: 10)
        random_state: Random seed for reproducibility
        plot: If True, plots for the first fold only

    Returns:
        Dictionary with mean and std for each metric
    """

    labels = np.array(labels)
    similarities_mv = np.array(similarities_mv)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    all_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(similarities_mv)):
        train_labels = labels[train_idx]
        train_scores = similarities_mv[train_idx]
        test_labels = labels[test_idx]
        test_scores = similarities_mv[test_idx]

        # Threshold selection on train set using Youden’s J
        fpr_train, tpr_train, thresholds_train = roc_curve(train_labels, train_scores)
        best_idx = np.argmax(tpr_train - fpr_train)
        best_thresh = thresholds_train[best_idx]

        # Evaluate on test set
        test_preds = (test_scores > best_thresh).astype(int)
        accuracy = accuracy_score(test_labels, test_preds)

        fpr, tpr, thresholds = roc_curve(test_labels, test_scores)
        roc_auc = auc(fpr, tpr)

        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

        precision, recall, _ = precision_recall_curve(test_labels, test_scores)
        avg_precision = average_precision_score(test_labels, test_scores)

        tar_results = {}
        for far in far_targets:
            idxs = np.where(fpr <= far)[0]
            if len(idxs) > 0:
                tar_at_far = tpr[idxs[-1]]
            else:
                tar_at_far = 0.0
            tar_results[f"TAR@FAR={far:.0e}"] = tar_at_far

        # Optionally plot only for the first fold
        if plot and fold_idx == 0:
            plot_verification(
                recall, precision, avg_precision,
                fpr, tpr, best_idx, best_thresh,
                accuracy, eer, roc_auc,
                dataset_name, method_appendix + f"_fold{fold_idx}"
            )

        metrics = {
            "AUC": roc_auc * 100,
            "Best_thresh": best_thresh,
            "Accuracy": accuracy * 100,
            "EER": eer,
            "Average_Precision": avg_precision * 100
        }
        metrics.update({k: v * 100 for k, v in tar_results.items()})

        all_metrics.append(metrics)

    # Aggregate metrics: mean and std
    summary = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        summary[key] = round(np.mean(values), 4)
        summary[key + "_std"] = round(np.std(values), 4)

    return summary


@numba.njit(parallel=True, fastmath=True, nogil=True)
def process_chunk_embedding_similarity(query_data, enrolled_data, start_row, end_row, similarity_matrix):
    for i in numba.prange(start_row, end_row):
        similarity_matrix[i, :] = np.dot(query_data[i], enrolled_data.T)


def calculate_embedding_similarity(query_embeddings, enrolled_embeddings, chunk_size=500, disable_bar=True):
    query_data = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    enrolled_data = enrolled_embeddings / np.linalg.norm(enrolled_embeddings, axis=1, keepdims=True)
    similarity_matrix = np.empty((query_data.shape[0], enrolled_data.shape[0]), dtype=np.float32)
    with tqdm(total=query_data.shape[0], disable=disable_bar, desc="Calculating Embedding Similarity") as pbar:
        for start_row in range(0, query_data.shape[0], chunk_size):
            end_row = min(start_row + chunk_size, query_data.shape[0])
            process_chunk_embedding_similarity(query_data, enrolled_data, start_row, end_row, similarity_matrix)
            pbar.update(end_row - start_row)
    return similarity_matrix


def concat(embedding_library, disable_bar: bool, reduce_with=""):

    enrolled_embedding, enrolled_label = embedding_library.enrolled_embeddings, embedding_library.enrolled_labels
    enrolled_embedding = enrolled_embedding.transpose(1, 0, 2).reshape(enrolled_embedding.shape[1], -1)  # (views, ids, 512) -> (ids, views*512)
    query_embedding, query_label = embedding_library.query_embeddings, embedding_library.query_labels
    query_embedding = query_embedding.transpose(1, 0, 2).reshape(query_embedding.shape[1], -1)  # (views, ids, 512) -> (ids, views*512)

    if reduce_with == "pca":
        if enrolled_embedding.shape[0] <= 512:
            return {}, None, None, None, None
        if enrolled_embedding.shape[0] > 2048:
            pca = IncrementalPCA(n_components=512, batch_size=2048)
        else:
            pca = PCA(n_components=512)
        pca = pca.fit(enrolled_embedding)
        enrolled_embedding = normalize(pca.transform(enrolled_embedding))
        query_embedding = normalize(pca.transform(query_embedding))
    elif reduce_with == "mean":
        enrolled_embedding = enrolled_embedding.reshape(enrolled_embedding.shape[0], -1, 512).mean(axis=1)
        query_embedding = query_embedding.reshape(query_embedding.shape[0], -1, 512).mean(axis=1)

    similarity_matrix = calculate_embedding_similarity(query_embedding, enrolled_embedding, disable_bar=disable_bar)
    top_indices, top_values = compute_ranking_matrices(similarity_matrix)
    result = analyze_result(similarity_matrix, top_indices, enrolled_label, query_label, top_k_acc_k=5)
    predicted_labels = enrolled_label[top_indices[:, 0]]
    return result, similarity_matrix, top_indices, predicted_labels, query_label


def calculate_embedding_similarity_per_view(query_embeddings, enrolled_embeddings, disable_bar=True, batch_size=1024):
    """
    query_embeddings: (num_queries, num_views, dim)
    enrolled_embeddings: (num_enrolled, num_views, dim)
    returns similarity_matrix: (num_queries, num_enrolled, num_views)
    """
    num_queries, num_views, dim = query_embeddings.shape
    num_enrolled = enrolled_embeddings.shape[0]

    similarity_matrix = np.empty((num_queries, num_enrolled, num_views), dtype=np.float32)

    for v in range(num_views):
        qv = query_embeddings[:, v, :]
        qv /= np.linalg.norm(qv, axis=1, keepdims=True)

        ev = enrolled_embeddings[:, v, :]
        ev /= np.linalg.norm(ev, axis=1, keepdims=True)

        # process queries in smaller chunks
        for start in range(0, num_queries, batch_size):
            end = min(start + batch_size, num_queries)
            similarity_matrix[start:end, :, v] = np.dot(qv[start:end], ev.T)

    return similarity_matrix


def score_fusion(embedding_library, disable_bar=True, method="product", similarity_matrix=None):
    """
    enrolled_embedding: (views, ids, 512)
    query_embedding: (views, ids, 512)
    """
    enrolled_embedding, enrolled_label = embedding_library.enrolled_embeddings, embedding_library.enrolled_labels
    query_embedding, query_label = embedding_library.query_embeddings, embedding_library.query_labels
    if similarity_matrix is None:
        query_embedding = np.transpose(embedding_library.query_embeddings, (1, 0, 2))  # (views, ids, 512) -> (ids, views, 512)
        enrolled_embedding = np.transpose(embedding_library.enrolled_embeddings, (1, 0, 2))  # (views, ids, 512) -> (ids, views, 512)
        #enrolled_embedding = enrolled_embedding.transpose(1, 0, 2).reshape(enrolled_embedding.shape[1], -1)  # (views, ids, 512) -> (ids, views*512)
        #query_embedding = query_embedding.transpose(1, 0, 2).reshape(query_embedding.shape[1], -1)  # (views, ids, 512) -> (ids, views*512)
        similarity_matrix = calculate_embedding_similarity_per_view(query_embedding, enrolled_embedding, disable_bar)

    if method == "sum":
        fused_scores = np.sum(similarity_matrix, axis=-1)
    elif method == "product":
        fused_scores = np.prod(similarity_matrix, axis=-1)
    elif method == "max":
        fused_scores = np.max(similarity_matrix, axis=-1)
    elif method == "geom_mean":
        # geometric mean: avoid underflow by working in log-space
        fused_scores = np.exp(np.mean(np.log(np.clip(similarity_matrix, 1e-12, None)), axis=-1))
    elif method == "lse":
        # log-sum-exp fusion
        fused_scores = np.log(np.sum(np.exp(similarity_matrix), axis=-1))
    elif method == "borda":
        # rank-based fusion: high similarity = high rank
        ranks = np.argsort(np.argsort(-similarity_matrix, axis=1), axis=1)  # (Q, E, V)
        # convert ranks to scores: best rank=E-1 points, worst=0
        borda_scores = (similarity_matrix.shape[1] - 1) - ranks
        fused_scores = np.sum(borda_scores, axis=-1)
    elif method == "majority":
        # argmax per view, then vote
        winners = np.argmax(similarity_matrix, axis=1)  # (Q, V)
        num_queries, num_views = winners.shape
        num_enrolled = similarity_matrix.shape[1]
        fused_scores = np.zeros((num_queries, num_enrolled), dtype=np.float32)
        for i in range(num_queries):
            votes = np.bincount(winners[i], minlength=num_enrolled)
            fused_scores[i] = votes  # class with most votes gets largest score
    elif method == "mean":
        fused_scores = np.mean(similarity_matrix, axis=-1)
    elif method == "trimmed_mean":
        k = 1  # Remove most outer elements
        sorted_scores = np.sort(similarity_matrix, axis=-1)
        trimmed = sorted_scores[:, :, k:-k] if k < similarity_matrix.shape[-1] // 2 else similarity_matrix
        fused_scores = np.mean(trimmed, axis=-1)
    else:
        raise ValueError(f"Unknown fusion method '{method}'")

    # get rankings from fused scores
    top_indices, top_values = compute_ranking_matrices(fused_scores)
    result = analyze_result(fused_scores, top_indices, enrolled_label, query_label, top_k_acc_k=5)
    predicted_labels = enrolled_label[top_indices[:, 0]]

    return result, similarity_matrix, fused_scores, top_indices, predicted_labels, query_label


def fuse_pairwise_scores(emb1_reg, emb2_reg, method="sum"):
    """
    emb1_reg, emb2_reg: (num_views, dim)
    returns fused similarity score (float)
    """
    sims = []
    for v in range(emb1_reg.shape[0]):
        s = np.dot(emb1_reg[v], emb2_reg[v]) / norm(emb1_reg[v] * norm(emb2_reg[v]))
        sims.append(s)
    sims = np.array(sims)

    if method == "sum":
        return np.sum(sims)
    elif method == "product":
        return np.prod(sims)
    elif method == "max":
        return np.max(sims)
    elif method == "geomean":  # geometric mean
        return np.exp(np.mean(np.log(np.clip(sims, 1e-12, None))))
    elif method == "lse":
        return np.log(np.sum(np.exp(sims)))
    elif method == "mean":
        return np.mean(sims)
    else:
        raise ValueError(f"Unknown fusion method '{method}'")


def accuracy_front_perspective(embedding_library):
    # enrolled_embeddings.shape -> (views, num_samples, embedding_dim)
    # enrolled_labels.shape -> (num_samples,)
    # enrolled_perspectives.shape -> (num_samples, views)

    view_mask = np.array(embedding_library.enrolled_perspectives[0]) == "0_0"
    selected_view_index = np.argmax(view_mask)
    enrolled_embeddings = embedding_library.enrolled_embeddings[selected_view_index]
    enrolled_labels = embedding_library.enrolled_labels

    view_mask = np.array(embedding_library.query_perspectives[0]) == "0_0"
    selected_view_index = np.argmax(view_mask)
    query_embeddings = embedding_library.query_embeddings[selected_view_index]
    query_labels = embedding_library.query_labels

    similarity_matrix = calculate_embedding_similarity(query_embeddings, enrolled_embeddings)
    top_indices, top_values = compute_ranking_matrices(similarity_matrix)
    result = analyze_result(similarity_matrix, top_indices, enrolled_labels, query_labels, top_k_acc_k=5)
    predicted_labels = enrolled_labels[top_indices[:, 0]]
    return result, similarity_matrix, top_indices, predicted_labels, query_labels


def accuracy_best_case(embedding_library):

    # enrolled_embeddings.shape -> (views, num_samples, embedding_dim)
    # enrolled_labels.shape -> (num_samples,)
    # enrolled_perspectives.shape -> (num_samples, views)

    enrolled_embeddings = embedding_library.enrolled_embeddings  # [V, N, D]
    enrolled_labels = embedding_library.enrolled_labels          # [N]
    query_embeddings = embedding_library.query_embeddings        # [V, M, D]
    query_labels = embedding_library.query_labels                # [M]

    # Flatten across views so each row is (sample, view)
    V_enrolled, N, D = enrolled_embeddings.shape
    V_query, M, _ = query_embeddings.shape

    enrolled_embeddings = enrolled_embeddings.reshape(V_enrolled * N, D)
    enrolled_labels = np.repeat(enrolled_labels, V_enrolled)

    query_embeddings = query_embeddings.reshape(V_query * M, D)
    query_labels = np.repeat(query_labels, V_query)

    # Compute similarities across all possible query<->enrolled pairs
    similarity_matrix = calculate_embedding_similarity(query_embeddings, enrolled_embeddings)

    # For each *original query sample*, pick the best view (highest similarity)
    top_indices, top_values = compute_ranking_matrices(similarity_matrix)
    result = analyze_result(similarity_matrix, top_indices, enrolled_labels, query_labels, top_k_acc_k=5)
    predicted_labels = enrolled_labels[top_indices[:, 0]]

    return result, similarity_matrix, top_indices, predicted_labels, query_labels
