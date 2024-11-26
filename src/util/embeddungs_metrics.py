import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score


def calc_embedding_analysis(embedding_library, distance_metric):
    enrolled_embeddings = embedding_library.enrolled_embeddings
    enrolled_labels = embedding_library.enrolled_labels
    enrolled_perspective = embedding_library.enrolled_perspectives
    enrolled_scan_ids = embedding_library.enrolled_scan_ids

    query_embeddings = embedding_library.query_embeddings
    query_labels = embedding_library.query_labels
    query_perspective = embedding_library.query_perspectives
    query_scan_ids = embedding_library.query_scan_ids

    unique_labels = np.unique(enrolled_labels)
    unique_query_labels = np.unique(query_scan_ids)

    enrolled_embeddings_mean = np.array([enrolled_embeddings[enrolled_labels == label].mean(axis=0) for label in unique_labels])

    embedding_metrics = {}

    #  ------------------- Calculate intra_enrolled_distances -------------------
    # Average Distance between enrolled centers
    intra_enrolled_distances = {}
    intra_enrolled_total_distance = 0
    number_of_embeddings = 0
    for label in unique_labels:
        label_indices = np.where(enrolled_labels == label)[0]
        label_embeddings = enrolled_embeddings[label_indices]
        if len(label_embeddings) > 1:
            distances = cdist(label_embeddings, label_embeddings, metric=distance_metric)
            upper_triangular_values = distances[np.triu_indices(len(label_embeddings), k=1)]
            label_distance = np.sum(upper_triangular_values)

            intra_enrolled_distances[label] = label_distance
            intra_enrolled_total_distance += label_distance
            number_of_embeddings += len(label_embeddings)
    if number_of_embeddings > 0:
        intra_enrolled_avg_distance = intra_enrolled_total_distance / number_of_embeddings
        # print('intra_enrolled_avg_distance', intra_enrolled_avg_distance)
        embedding_metrics['intra_enrolled_avg_distance'] = intra_enrolled_avg_distance

    #  ------------------- Calculate intra_query_distances -------------------
    # Average Distances between queries of same labels
    intra_query_distances = {}
    intra_query_total_distance = 0
    number_of_embeddings = 0
    for label in unique_labels:
        label_indices = np.where(query_labels == label)[0]
        label_embeddings = query_embeddings[label_indices]
        if len(label_embeddings) > 1:
            distances = cdist(label_embeddings, label_embeddings, metric=distance_metric)
            # Extract the upper triangular part of the matrix, excluding the diagonal
            upper_triangular_indices = np.triu_indices_from(distances, k=1)
            upper_triangular_values = distances[upper_triangular_indices]
            # Calculate the total distance of the upper triangular matrix
            label_distance = np.sum(upper_triangular_values)
            intra_query_distances[label] = label_distance
            intra_query_total_distance += label_distance
            number_of_embeddings += len(label_embeddings)
    if number_of_embeddings > 0:
        intra_query_avg_distance = intra_query_total_distance / number_of_embeddings
        # print('intra_query_avg_distance', intra_query_avg_distance)
        embedding_metrics['intra_query_avg_distance'] = intra_query_avg_distance

    #  ------------------- Calculate intra_scan_distances -------------------
    # Average Distances between queries of same query scan_id
    intra_scan_distances = {}
    intra_scan_total_distance = 0
    number_of_embeddings = 0
    for scan_id in unique_query_labels:
        scan_id_indices = np.where(query_scan_ids == scan_id)[0]
        scan_id_embeddings = query_embeddings[scan_id_indices]

        if len(scan_id_embeddings) > 1:
            distances = cdist(scan_id_embeddings, scan_id_embeddings, metric=distance_metric)
            # Extract the upper triangular part of the matrix, excluding the diagonal
            upper_triangular_indices = np.triu_indices_from(distances, k=1)
            upper_triangular_values = distances[upper_triangular_indices]
            # Calculate the total distance of the upper triangular matrix
            scan_distance = np.sum(upper_triangular_values)
            intra_scan_distances[scan_id] = scan_distance
            intra_scan_total_distance += scan_distance
            number_of_embeddings += len(scan_id_embeddings)
    if number_of_embeddings > 0:
        intra_scan_avg_distance = intra_scan_total_distance / number_of_embeddings
        # print('intra_scan_avg_distance', intra_scan_avg_distance)
        embedding_metrics['intra_scan_avg_distance'] = intra_scan_avg_distance

    #  ------------------- Calculate inter_enrolled_center_distances -------------------
    # Distance between enrolled centers
    distance_matrix = cdist(enrolled_embeddings_mean, enrolled_embeddings_mean, metric=distance_metric)
    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    # Extract the upper triangle values
    upper_triangle_values = distance_matrix[triu_indices]
    # Calculate the average distance
    inter_enrolled_center_avg_distance = np.mean(upper_triangle_values)
    # print('inter_enrolled_center_avg_distance', inter_enrolled_center_avg_distance)
    embedding_metrics['inter_enrolled_center_avg_distance'] = inter_enrolled_center_avg_distance

    try:
        #  ------------------- Silhouette Score  -------------------
        #  Measures how similar an embedding is to its own cluster compared to other clusters. A higher score indicates more cohesive clusters.
        enrolled_silhouette_score = silhouette_score(enrolled_embeddings, enrolled_labels, metric='cosine')
        query_silhouette_score = silhouette_score(query_embeddings, query_labels, metric='cosine')
        embedding_metrics['enrolled_silhouette_score'] = enrolled_silhouette_score
        embedding_metrics['query_silhouette_score'] = query_silhouette_score
        #  ------------------- Davies-Bouldin Index -------------------
        #  Evaluates the average similarity ratio between each cluster and the cluster most similar to it. Lower values mean better clustering.
        enrolled_davies_bouldin_score = davies_bouldin_score(enrolled_embeddings, enrolled_labels)
        query_davies_bouldin_score = davies_bouldin_score(query_embeddings, query_labels)
        embedding_metrics['enrolled_davies_bouldin_score'] = enrolled_davies_bouldin_score
        embedding_metrics['query_davies_bouldin_score'] = query_davies_bouldin_score
    except ValueError:
        print("Size of Clusters in not sufficient")

    #  ------------------- Distribution of Norms -------------------
    norms = np.linalg.norm(enrolled_embeddings, axis=1)
    embedding_metrics['enrolled_mean_norm'] = np.mean(norms)
    embedding_metrics['enrolled_std_norm'] = np.std(norms)

    norms = np.linalg.norm(query_embeddings, axis=1)
    embedding_metrics['query_mean_norm'] = np.mean(norms)
    embedding_metrics['query_std_norm'] = np.std(norms)

    return embedding_metrics

