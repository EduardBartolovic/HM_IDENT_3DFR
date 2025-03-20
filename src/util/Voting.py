from collections import defaultdict

import numpy as np
from sklearn import neighbors
import numba
from sklearn.metrics import accuracy_score
from tqdm import tqdm


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

    return {
        "Rank-1 Rate": round(top_1_accuracy * 100, 2),
        f"Rank-{top_k_acc_k} Rate": round(top_k_accuracy * 100, 2),
    }


@numba.njit(parallel=True)
def process_chunk_embedding_similarity(tabular_data, image_data, start_row, end_row, similarity_matrix):
    for i in numba.prange(start_row, end_row):
        similarity_matrix[i, :] = np.dot(tabular_data[i], image_data.T)


def calculate_embedding_similarity(tabular_embeddings, image_embeddings, chunk_size=100, show_progress=True):
    tabular_data = tabular_embeddings / np.linalg.norm(tabular_embeddings, axis=1, keepdims=True)
    image_data = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    similarity_matrix = np.empty((tabular_data.shape[0], image_data.shape[0]), dtype=np.float32)

    with tqdm(total=tabular_data.shape[0], disable=not show_progress, desc="Calculating Embedding Similarity") as pbar:
        for start_row in range(0, tabular_data.shape[0], chunk_size):
            end_row = min(start_row + chunk_size, tabular_data.shape[0])
            process_chunk_embedding_similarity(tabular_data, image_data, start_row, end_row, similarity_matrix)
            pbar.update(end_row - start_row)

    return similarity_matrix


def concat(embedding_library):
    def process_embeddings(scan_ids, embeddings, labels, perspectives):
        scan_to_data = {}
        for scan_id, embedding, label, perspective in zip(scan_ids, embeddings, labels, perspectives):
            if scan_id not in scan_to_data:
                scan_to_data[scan_id] = {'embeddings': [], 'perspectives': [], 'label': label}
            scan_to_data[scan_id]['embeddings'].append(embedding)
            scan_to_data[scan_id]['perspectives'].append(perspective)

        concatenated_embeddings, concatenated_labels = [], []
        for scan_id, data in scan_to_data.items():
            sorted_embeddings = [emb for _, emb in
                                 sorted(zip(data['perspectives'], data['embeddings']), key=lambda x: x[0])]
            concatenated_embedding = np.concatenate(sorted_embeddings, axis=0)
            if concatenated_embedding.shape[0] not in {512, 512 * 5, 512 * 25}:
                continue
            concatenated_embeddings.append(concatenated_embedding)
            concatenated_labels.append(data['label'])

        return np.array(concatenated_embeddings, dtype=np.float32), np.array(concatenated_labels)

    enrolled_embedding_database, enrolled_label_database = process_embeddings(
        embedding_library.enrolled_scan_ids, embedding_library.enrolled_embeddings,
        embedding_library.enrolled_labels, embedding_library.enrolled_perspectives)

    query_embedding_database, query_label_database = process_embeddings(
        embedding_library.query_scan_ids, embedding_library.query_embeddings,
        embedding_library.query_labels, embedding_library.query_perspectives)

    similarity_matrix = calculate_embedding_similarity(query_embedding_database, enrolled_embedding_database)
    top_indices, top_values = compute_ranking_matrices(similarity_matrix)
    return analyze_result(similarity_matrix, top_indices, enrolled_label_database, query_label_database, top_k_acc_k=5)


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


def knn_voting(embedding_library, k=1):
    k = 1
    d = "cosine"
    # for k in range(1, 7, 2):
    #    for d in ["cosine", "euclidean"]:

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

    # print("K", k, "d", d, "KNN Voting Accuracy={}%".format(accuracy_score(y_true_scan, y_pred_scan) * 100))

    return np.array(y_true_scan), np.array(y_pred_scan)


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


def accuracy_front_perspective(embedding_library, distance_metric=None):
    # enrolled_embeddings.shape -> (num_samples, embedding_dim)
    # enrolled_labels.shape -> (num_samples,)
    # enrolled_perspectives.shape -> (num_samples,)

    # Mask rows where the enrolled_perspectives contain the specific string
    mask = np.array(["0_0" in perspective or "-fa" in perspective for perspective in embedding_library.enrolled_perspectives])
    enrolled_embeddings = embedding_library.enrolled_embeddings[mask]
    enrolled_labels = embedding_library.enrolled_labels[mask]

    # Mask rows where the query_perspectives contain the specific string
    mask = np.array(["0_0" in perspective or "-fa" in perspective for perspective in embedding_library.query_perspectives])
    query_embeddings = embedding_library.query_embeddings[mask]
    query_labels = embedding_library.query_labels[mask]

    similarity_matrix = calculate_embedding_similarity(query_embeddings, enrolled_embeddings)
    top_indices, top_values = compute_ranking_matrices(similarity_matrix)
    result_metrics = analyze_result(similarity_matrix, top_indices, enrolled_labels, query_labels, top_k_acc_k=5)
    return result_metrics
