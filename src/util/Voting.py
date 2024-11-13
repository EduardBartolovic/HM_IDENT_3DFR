from collections import defaultdict

import numpy as np
from sklearn import neighbors

from src.util.EmbeddingsUtils import batched_distances_gpu


def knn_voting(embedding_library, k=1):

    scan_id_to_idx = defaultdict(list)
    for idx, scan_id in enumerate(embedding_library.query_scan_ids):
        scan_id_to_idx[scan_id].append(idx)

    k = 1
    d = "cosine"
    #for k in range(1, 7, 2):
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

    #print("K", k, "d", d, "KNN Voting Accuracy={}%".format(accuracy_score(y_true_scan, y_pred_scan) * 100))

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


def accuracy_front_perspective(device, embedding_library, distance_metric):
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

    d = "cosine"
    knn_model = neighbors.KNeighborsClassifier(n_neighbors=1, n_jobs=8, metric=d)
    knn_model.fit(enrolled_embeddings, enrolled_labels)
    y_preds = knn_model.predict(query_embeddings)

    return query_labels, y_preds
