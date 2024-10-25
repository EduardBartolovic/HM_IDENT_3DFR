from collections import defaultdict

import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score


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
