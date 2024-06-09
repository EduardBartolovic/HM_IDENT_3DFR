import json
import os
import time

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def calc_metrics(y_true, y_pred, y_pred_top5):
    # https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

    assert y_pred.shape[0] == len(y_true)
    assert y_pred_top5.shape[0] == len(y_true)
    assert y_pred_top5.shape[1] == 5

    labels = np.unique(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    rank_1_rate = accuracy * 100
    accuracy_top5 = np.mean(np.any(y_pred_top5 == y_true[:, None], axis=1))
    rank_5_rate = accuracy_top5 * 100
    precision = precision_score(y_true, y_pred, average='micro', labels=labels, zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', labels=labels, zero_division=0)

    cr = classification_report(y_true, y_pred, zero_division=0)

    metrics = {
        'Accuracy': round(accuracy, 3),
        'AccuracyTop5': round(accuracy_top5, 3),
        'Rank-1 Rate': round(rank_1_rate, 2),
        'Rank-5 Rate': round(rank_5_rate, 2),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1-score': round(f1, 3),
        'classification_report': cr
    }
    return metrics


def write_metrics_and_config(output_path, acc_val, acc5_val, prec_val, rec_val, f1_val, r1r_val, test_names, acc_test,
                             acc5_test, prec_test, rec_test,
                             f1_test, r1r_test, metrics_lists, train_loss_list, val_loss_list, hyperparameters):
    output_file_path = os.path.join(output_path, 'metrics.txt')
    with open(output_file_path, 'w') as file:
        file.write(f"Validation Accuracy: {acc_val:.4f}\n")
        file.write(f"Validation AccuracyTop5: {acc5_val:.4f}\n")
        file.write(f"Validation Precision: {prec_val:.4f}\n")
        file.write(f"Validation Recall: {rec_val:.4f}\n")
        file.write(f"Validation F1-score: {f1_val:.4f}\n")
        file.write(f"Validation Rank-1 Recognition Rate: {r1r_val:.4f}\n")

        for i, test_name in enumerate(test_names):
            file.write(f"{test_name} Test Accuracy: {acc_test[i]:.4f}\n")
            file.write(f"{test_name} Test AccuracyTop5: {acc5_test[i]:.4f}\n")
            file.write(f"{test_name} Test Precision: {prec_test[i]:.4f}\n")
            file.write(f"{test_name} Test Recall: {rec_test[i]:.4f}\n")
            file.write(f"{test_name} Test F1-score: {f1_test[i]:.4f}\n")
            file.write(f"{test_name} Test Rank-1 Recognition Rate: {r1r_test[i]:.4f}\n")

        file.write("\nMetric Lists:\n")
        for metric_name, metric_list in metrics_lists.items():
            file.write(f"\n{metric_name}: ")
            for metric_value in metric_list:
                file.write(f"{metric_value:.4f} ")
        file.write("\ntrain_loss_list: ")
        for i in train_loss_list:
            file.write(f"{i:.4f} ")
        file.write("\nval_loss_list: ")
        for i in val_loss_list:
            file.write(f"{i:.4f} ")

    # Write config to the file
    output_file_path = os.path.join(output_path, 'config.txt')
    with open(output_file_path, 'w') as file:
        json.dump(hyperparameters, file, indent=4)


def calc_embedding_facts(embedding_library):
    enrolled_embeddings = embedding_library.enrolled_embeddings
    enrolled_labels = embedding_library.enrolled_labels
    enrolled_perspective = embedding_library.enrolled_perspectives
    enrolled_scan_ids = embedding_library.enrolled_scan_ids

    query_embeddings = embedding_library.val_embeddings
    query_labels = embedding_library.val_labels
    query_perspective = embedding_library.val_perspectives
    query_scan_ids = embedding_library.val_scan_ids

    unique_labels = np.unique(enrolled_labels)

    # Calculate intra_enrolled_distances
    intra_enrolled_distances = {}
    intra_enrolled_total_distance = 0
    for label in unique_labels:
        label_indices = np.where(enrolled_labels == label)[0]
        label_embeddings = enrolled_embeddings[label_indices]

        if len(label_embeddings) > 1:

            #start_time = time.time()
            distances = cdist(label_embeddings, label_embeddings, 'cosine')
            #print("SciPy cdist time: %s seconds" % (time.time() - start_time))
            #start_time = time.time()
            #distances_torch = torch.cdist(torch.Tensor(label_embeddings), torch.Tensor(label_embeddings), p=2)
            #print("PyTorch cdist time: %s seconds" % (time.time() - start_time))

            # Extract the upper triangular part of the matrix, excluding the diagonal
            upper_triangular_indices = np.triu_indices_from(distances, k=1)
            upper_triangular_values = distances[upper_triangular_indices]
            # Calculate the total distance of the upper triangular matrix
            label_distance = np.sum(upper_triangular_values)
            intra_enrolled_distances[label] = label_distance
            intra_enrolled_total_distance += label_distance

    intra_enrolled_avg_distance = intra_enrolled_total_distance / len(unique_labels)
    print('intra_enrolled_avg_distance', intra_enrolled_avg_distance)

    # Calculate intra_query_distances
    intra_query_distances = {}
    intra_query_total_distance = 0
    for label in unique_labels:
        label_indices = np.where(enrolled_labels == label)[0]
        label_embeddings = query_embeddings[label_indices]

        if len(label_embeddings) > 1:
            distances = cdist(label_embeddings, label_embeddings, 'cosine') #euclidean
            # Extract the upper triangular part of the matrix, excluding the diagonal
            upper_triangular_indices = np.triu_indices_from(distances, k=1)
            upper_triangular_values = distances[upper_triangular_indices]
            # Calculate the total distance of the upper triangular matrix
            label_distance = np.sum(upper_triangular_values)
            intra_query_distances[label] = label_distance
            intra_query_total_distance += label_distance

    intra_query_avg_distance = intra_query_total_distance / len(unique_labels)
    print('intra_query_avg_distance', intra_query_avg_distance)

    # TODO: Calculate distance to mean

    return intra_enrolled_avg_distance, intra_enrolled_distances, intra_query_avg_distance, intra_query_distances

