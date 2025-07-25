import os
from collections import defaultdict

import faiss
import numpy as np
from matplotlib import pyplot as plt
from sklearn import neighbors
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numba
from sklearn.metrics import accuracy_score, roc_curve, auc, DetCurveDisplay, precision_recall_curve, \
    average_precision_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.util.EmbeddingsUtils import process_unsorted_embeddings
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

    true_match_scores = np.array([similarity_matrix[i, np.where(reference_ids == ground_truth_ids[i])[0][0]] for i in range(num_inferences)])
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


def analyze_result_verification(labels, similarities_mv, dataset_name, method_appendix="", far_targets=(1e-6, 1e-4)):
    """
    Evaluate 1:1 face verification results.

    Args:
        labels: Ground-truth binary labels (1=same, 0=different)
        similarities_mv: Similarity scores between pairs
        far_targets: Tuple of FARs to compute TAR at (default: 1e-6 and 1e-4)

    Returns:
        Dictionary with evaluation metrics.
    """

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(labels, similarities_mv)
    roc_auc = auc(fpr, tpr)

    # Best threshold based on Youden’s J statistic
    best_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_idx]
    predictions = (similarities_mv > best_thresh).astype(int)
    accuracy = accuracy_score(labels, predictions)

    # Equal Error Rate (EER)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    # Average Precision (AP)
    precision, recall, _ = precision_recall_curve(labels, similarities_mv)
    avg_precision = average_precision_score(labels, similarities_mv)

    # TAR @ FAR
    tar_results = {}
    for far in far_targets:
        idxs = np.where(fpr <= far)[0]
        if len(idxs) > 0:
            tar_at_far = tpr[idxs[-1]]  # take the last one under the FAR
        else:
            tar_at_far = 0.0  # No FPR small enough
        tar_results[f"TAR@FAR={far:.0e}"] = tar_at_far

    plot_verification(recall, precision, avg_precision, fpr, tpr, best_idx, best_thresh, accuracy, eer, roc_auc, dataset_name, method_appendix)

    # Final results
    result = {
        "AUC": round(roc_auc * 100, 4),
        "Best_thresh": round(best_thresh, 4),
        "Accuracy": round(accuracy * 100, 4),
        "EER": round(eer * 100, 4),
        "Average_Precision": round(avg_precision * 100, 4),
    }

    # Add TAR@FARs
    for k, v in tar_results.items():
        result[k] = round(v * 100, 4)

    return result


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


def concat(embedding_library, disable_bar: bool, pre_sorted=False, reduce_with=""):

    if pre_sorted:
        enrolled_embedding, enrolled_label = embedding_library.enrolled_embeddings, embedding_library.enrolled_labels
        enrolled_embedding = enrolled_embedding.transpose(1, 0, 2).reshape(enrolled_embedding.shape[1], -1)  # (views, ids, 512) -> (ids, views*512)
        query_embedding, query_label = embedding_library.query_embeddings, embedding_library.query_labels
        query_embedding = query_embedding.transpose(1, 0, 2).reshape(query_embedding.shape[1], -1)  # (views, ids, 512) -> (ids, views*512)
    else:
        enrolled_embedding, enrolled_label = process_unsorted_embeddings(
            embedding_library.enrolled_scan_ids, embedding_library.enrolled_embeddings,
            embedding_library.enrolled_labels, embedding_library.enrolled_perspectives)
        query_embedding, query_label = process_unsorted_embeddings(
            embedding_library.query_scan_ids, embedding_library.query_embeddings,
            embedding_library.query_labels, embedding_library.query_perspectives)

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
        enrolled_embedding = enrolled_embedding.reshape(enrolled_embedding.shape[0], 512, -1).mean(axis=2)
        query_embedding = query_embedding.reshape(query_embedding.shape[0], 512, -1).mean(axis=2)

    similarity_matrix = calculate_embedding_similarity(query_embedding, enrolled_embedding, disable_bar=disable_bar)
    top_indices, top_values = compute_ranking_matrices(similarity_matrix)
    result = analyze_result(similarity_matrix, top_indices, enrolled_label, query_label, top_k_acc_k=5)
    predicted_labels = enrolled_label[top_indices[:, 0]]
    return result, similarity_matrix, top_indices, predicted_labels, query_label


def multidatabase_voting(embedding_library):
    # Group embeddings, perspectives, and match them with labels by scan_id
    scan_to_data = {}
    for scan_id, embedding, label, perspective in zip(
            embedding_library.enrolled_scan_ids,
            embedding_library.enrolled_embeddings,
            embedding_library.enrolled_labels,
            embedding_library.enrolled_perspectives
    ):
        if scan_id not in scan_to_data:
            scan_to_data[scan_id] = {'embeddings': [], 'perspectives': [], 'label': label}
        scan_to_data[scan_id]['embeddings'].append(embedding)
        scan_to_data[scan_id]['perspectives'].append(perspective)

    embeddings_databases = []
    embeddings_labels = []
    numberofposes = -1
    for scan_id, data in scan_to_data.items():
        # Combine embeddings and perspectives into sortable pairs
        combined = list(zip(data['perspectives'], data['embeddings']))
        combined.sort(key=lambda x: x[0])  # Sort by perspective
        _, sorted_embeddings = zip(*combined)  # Extract sorted embeddings

        if numberofposes < 0:
            numberofposes = len(sorted_embeddings)
        # Extend embeddings_databases to match the size of sorted_embeddings
        while len(embeddings_databases) < len(sorted_embeddings):
            embeddings_databases.append([])  # Add new inner lists as needed

        if len(sorted_embeddings) != numberofposes:
            print("continued")
            continue

        for i in range(len(sorted_embeddings)):
            embeddings_databases[i].append(sorted_embeddings[i])
            # Use the label associated with this scan_id
        embeddings_labels.append(data['label'])

    enrolled_embedding_databases = np.array(embeddings_databases)
    enrolled_label_database = np.array(embeddings_labels)

    # --------------QUERY --------------------------:

    # Group embeddings, perspectives, and match them with labels by scan_id
    scan_to_data = {}
    for scan_id, embedding, label, perspective in zip(
            embedding_library.query_scan_ids,
            embedding_library.query_embeddings,
            embedding_library.query_labels,
            embedding_library.query_perspectives
    ):
        if scan_id not in scan_to_data:
            scan_to_data[scan_id] = {'embeddings': [], 'perspectives': [], 'label': label}
        scan_to_data[scan_id]['embeddings'].append(embedding)
        scan_to_data[scan_id]['perspectives'].append(perspective)

    embeddings_databases = []
    embeddings_labels = []
    for scan_id, data in scan_to_data.items():
        # Combine embeddings and perspectives into sortable pairs
        combined = list(zip(data['perspectives'], data['embeddings']))
        combined.sort(key=lambda x: x[0])  # Sort by perspective
        _, sorted_embeddings = zip(*combined)  # Extract sorted embeddings

        # Extend embeddings_databases to match the size of sorted_embeddings
        while len(embeddings_databases) < len(sorted_embeddings):
            embeddings_databases.append([])  # Add new inner lists as needed

        if len(sorted_embeddings) != numberofposes:
            print("continued", len(sorted_embeddings))
            continue

        for i in range(len(sorted_embeddings)):
            embeddings_databases[i].append(sorted_embeddings[i])
            # Use the label associated with this scan_id
        embeddings_labels.append(data['label'])

    query_embedding_databases = np.array(embeddings_databases)
    query_label_database = np.array(embeddings_labels)

    majority_vote_predictions = []
    for i in range(len(query_embedding_databases)):
        similarity_matrix = calculate_embedding_similarity(query_embedding_databases[i], enrolled_embedding_databases[i], show_progress=False)
        top_indices, _ = compute_ranking_matrices(similarity_matrix)

        majority_vote_predictions.append(top_indices)

    majority_vote_predictions = np.array(majority_vote_predictions)
    num_views, num_queries, num_classes = majority_vote_predictions.shape

    view_weights = np.ones(num_views)  # change this to give different weights to different views
    view_weights /= view_weights.sum()  # Normalize weights to sum to 1
    # Initialize an empty array to store the weighted majority vote predictions
    weighted_majority_votes = np.zeros((num_queries, num_classes), dtype=int)
    # Iterate over each query
    for query_idx in range(num_queries):
        # Weighted vote: calculate scores for each class
        rank_scores = np.zeros(num_classes)
        for view_idx in range(num_views):
            # Add view weight to the scores of the ranked classes
            for rank_pos, class_idx in enumerate(majority_vote_predictions[view_idx, query_idx]):
                rank_scores[class_idx] += view_weights[view_idx] * (num_classes - rank_pos)

        # Get the sorted indices based on scores (highest to lowest)
        weighted_majority_votes[query_idx] = np.argsort(rank_scores)[::-1]

    result_metrics = analyze_result(similarity_matrix, weighted_majority_votes, enrolled_label_database, query_label_database, top_k_acc_k=5)
    return result_metrics


def knn_voting(embedding_library, k=1, d="cosine", faiss_method=True):

    if faiss_method:
        return faiss_knn_voting(embedding_library)

    knn_model = neighbors.KNeighborsClassifier(n_neighbors=k, n_jobs=-1, metric=d)
    knn_model.fit(embedding_library.enrolled_embeddings, embedding_library.enrolled_labels)

    y_preds = knn_model.predict(embedding_library.query_embeddings)

    vote_scan_id = {}
    label_scan_id = {}
    for idx, y_pred in enumerate(y_preds):
        if embedding_library.query_scan_ids[idx] in vote_scan_id.keys():
            vote_scan_id[embedding_library.query_scan_ids[idx]].append(y_pred)
        else:
            vote_scan_id[embedding_library.query_scan_ids[idx]] = [y_pred]
            label_scan_id[embedding_library.query_scan_ids[idx]] = embedding_library.query_labels[idx]

    y_pred_scan = []
    y_true_scan = []
    for key, value in vote_scan_id.items():
        votes = vote_scan_id[key]
        vote = max(set(votes), key=votes.count)
        y_true = label_scan_id[key]
        y_true_scan.append(y_true)
        y_pred_scan.append(vote)

    return np.array(y_true_scan), np.array(y_pred_scan)


def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def faiss_knn_voting(embedding_library, k=1):

    enrolled_embeddings = normalize_embeddings(embedding_library.enrolled_embeddings)
    query_embeddings = normalize_embeddings(embedding_library.query_embeddings)

    index = faiss.IndexFlatIP(enrolled_embeddings.shape[1])
    index.add(enrolled_embeddings)

    _, indices = index.search(query_embeddings, k)
    y_preds = embedding_library.enrolled_labels[indices.flatten()]

    # Majority voting
    vote_scan_id = {}
    label_scan_id = {}
    for scan_id, y_pred, y_true in zip(embedding_library.query_scan_ids, y_preds, embedding_library.query_labels):
        vote_scan_id.setdefault(scan_id, []).append(y_pred)
        label_scan_id[scan_id] = y_true

    y_pred_scan = np.array([max(set(votes), key=votes.count) for votes in vote_scan_id.values()])
    y_true_scan = np.array([label_scan_id[key] for key in vote_scan_id.keys()])

    return y_true_scan, y_pred_scan


def voting(y_pred, scan_ids, query_labels):
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
    scan_id_to_label = dict(zip(scan_ids, query_labels))

    y_true_scan = [scan_id_to_label[scan_id] for scan_id in weighted_top_per_scan_id.keys()]
    y_pred_scan = [weighted_top_per_scan_id[scan_id] for scan_id in weighted_top_per_scan_id.keys()]

    return np.array(y_true_scan), np.array(y_pred_scan)


def accuracy_front_perspective(embedding_library, pre_sorted=False):

    if pre_sorted:
        # enrolled_embeddings.shape -> (views, num_samples, embedding_dim)
        # enrolled_labels.shape -> (num_samples,)
        # enrolled_perspectives.shape -> (num_samples, views)

        # Important Assume all enrolled_perspectives are identical across samples
        view_mask = np.array(embedding_library.enrolled_perspectives[0]) == "0_0"
        selected_view_index = np.argmax(view_mask)
        enrolled_embeddings = embedding_library.enrolled_embeddings[selected_view_index]
        enrolled_labels = embedding_library.enrolled_labels

        view_mask = np.array(embedding_library.query_perspectives[0]) == "0_0"
        selected_view_index = np.argmax(view_mask)
        query_embeddings = embedding_library.query_embeddings[selected_view_index]
        query_labels = embedding_library.query_labels
    else:
        # enrolled_embeddings.shape -> (num_samples, embedding_dim)
        # enrolled_labels.shape -> (num_samples,)
        # enrolled_perspectives.shape -> (num_samples,)

        # Mask rows where the enrolled_perspectives contain the specific string
        mask = np.array(["0_0" in perspective for perspective in embedding_library.enrolled_perspectives])
        enrolled_embeddings = embedding_library.enrolled_embeddings[mask]
        enrolled_labels = embedding_library.enrolled_labels[mask]

        # Mask rows where the query_perspectives contain the specific string
        mask = np.array(["0_0" in perspective for perspective in embedding_library.query_perspectives])
        query_embeddings = embedding_library.query_embeddings[mask]
        query_labels = embedding_library.query_labels[mask]

    similarity_matrix = calculate_embedding_similarity(query_embeddings, enrolled_embeddings)
    top_indices, top_values = compute_ranking_matrices(similarity_matrix)
    result = analyze_result(similarity_matrix, top_indices, enrolled_labels, query_labels, top_k_acc_k=5)
    predicted_labels = enrolled_labels[top_indices[:, 0]]
    return result, similarity_matrix, top_indices, predicted_labels, query_labels
