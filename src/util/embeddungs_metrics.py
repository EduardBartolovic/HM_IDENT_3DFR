import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score


def calc_embedding_analysis(embedding_library, enrolled_embeddings_mean, distance_metric):
    enrolled_embeddings = embedding_library.enrolled_embeddings
    enrolled_labels = embedding_library.enrolled_labels
    query_embeddings = embedding_library.query_embeddings
    query_labels = embedding_library.query_labels
    query_scan_ids = embedding_library.query_scan_ids

    unique_labels = np.unique(enrolled_labels)
    unique_query_labels = np.unique(query_scan_ids)

    embedding_metrics = {}

    #  ------------------- Calculate intra_enrolled_distances -------------------
    # Average Distance between enrolled centers
    intra_enrolled_total_distance = 0
    total_enrolled_pairs = 0

    for label in unique_labels:
        label_embeddings = enrolled_embeddings[enrolled_labels == label]
        if len(label_embeddings) > 1:
            distances = cdist(label_embeddings, label_embeddings, metric=distance_metric)
            upper_triangular_values = distances[np.triu_indices(len(label_embeddings), k=1)]
            intra_enrolled_total_distance += upper_triangular_values.sum()
            total_enrolled_pairs += len(upper_triangular_values)

    if total_enrolled_pairs > 0:
        embedding_metrics['intra_enrolled_avg_distance'] = intra_enrolled_total_distance / total_enrolled_pairs

    #  ------------------- Calculate intra_query_distances -------------------
    # Average Distances between queries of same labels
    intra_query_total_distance = 0
    total_query_pairs = 0

    for label in unique_labels:
        label_embeddings = query_embeddings[query_labels == label]
        if len(label_embeddings) > 1:
            distances = cdist(label_embeddings, label_embeddings, metric=distance_metric)
            upper_triangular_values = distances[np.triu_indices_from(distances, k=1)]
            intra_query_total_distance += upper_triangular_values.sum()
            total_query_pairs += len(upper_triangular_values)

    if total_query_pairs > 0:
        embedding_metrics['intra_query_avg_distance'] = intra_query_total_distance / total_query_pairs

    #  ------------------- Calculate intra_scan_distances -------------------
    # Average Distances between queries of same query scan_id
    intra_scan_total_distance = 0
    total_scan_pairs = 0

    for scan_id in unique_query_labels:
        scan_embeddings = query_embeddings[query_scan_ids == scan_id]
        if len(scan_embeddings) > 1:
            distances = cdist(scan_embeddings, scan_embeddings, metric=distance_metric)
            upper_triangular_values = distances[np.triu_indices_from(distances, k=1)]
            intra_scan_total_distance += upper_triangular_values.sum()
            total_scan_pairs += len(upper_triangular_values)

    if total_scan_pairs > 0:
        embedding_metrics['intra_scan_avg_distance'] = intra_scan_total_distance / total_scan_pairs

    #  ------------------- Calculate inter_enrolled_center_distances -------------------
    # Distance between enrolled centers
    if len(enrolled_embeddings_mean) > 1:
        distance_matrix = cdist(enrolled_embeddings_mean, enrolled_embeddings_mean, metric=distance_metric)
        upper_triangle_values = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        embedding_metrics['inter_enrolled_center_avg_distance'] = np.mean(upper_triangle_values)

    # ------------------- Clustering Metrics -------------------
    try:
        embedding_metrics['enrolled_silhouette_score'] = silhouette_score(enrolled_embeddings, enrolled_labels,
                                                                          metric='cosine')
        embedding_metrics['query_silhouette_score'] = silhouette_score(query_embeddings, query_labels, metric='cosine')

        embedding_metrics['enrolled_davies_bouldin_score'] = davies_bouldin_score(enrolled_embeddings, enrolled_labels)
        embedding_metrics['query_davies_bouldin_score'] = davies_bouldin_score(query_embeddings, query_labels)
    except ValueError:
        print("Size of clusters is not sufficient for silhouette or Davies-Bouldin scores.")

    #  ------------------- Distribution of Norms -------------------
    embedding_metrics['enrolled_mean_norm'] = np.mean(np.linalg.norm(enrolled_embeddings, axis=1))
    embedding_metrics['enrolled_std_norm'] = np.std(np.linalg.norm(enrolled_embeddings, axis=1))

    embedding_metrics['query_mean_norm'] = np.mean(np.linalg.norm(query_embeddings, axis=1))
    embedding_metrics['query_std_norm'] = np.std(np.linalg.norm(query_embeddings, axis=1))

    return embedding_metrics

