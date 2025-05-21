import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def calc_metrics(y_true, y_pred, y_pred_top5=None):
    """
    Calculate various classification metrics.

    Parameters:
    - y_true: array-like of shape (n_samples,) - True class labels.
    - y_pred: array-like of shape (n_samples,) - Predicted class labels.
    - y_pred_top5: array-like of shape (n_samples, 5), optional - Top 5 predicted class labels.

    Returns:
    - metrics: dict - Dictionary containing calculated metrics.
    """
    # https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

    assert y_pred.shape[0] == len(y_true)
    if y_pred_top5 is not None:
        assert y_pred_top5.shape == (len(y_true), 5), "Top-5 predictions should have shape (n_samples, 5)"

    # Core metrics
    labels = np.unique(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {
        'Accuracy': round(accuracy, 3),
        'Rank-1 Rate': round(accuracy * 100, 2),
        'Precision': round(precision_score(y_true, y_pred, average='micro', labels=labels, zero_division=0), 3),
        'Recall': round(recall_score(y_true, y_pred, average='micro', labels=labels, zero_division=0), 3),
        'F1-score': round(f1_score(y_true, y_pred, average='micro', labels=labels, zero_division=0), 3),
        'classification_report': classification_report(y_true, y_pred, zero_division=0)
    }

    # Top-5 accuracy if available
    if y_pred_top5 is not None:
        accuracy_top5 = np.mean(np.any(y_pred_top5 == y_true[:, None], axis=1))
        metrics.update({
            'AccuracyTop5': round(accuracy_top5, 3),
            'Rank-5 Rate': round(accuracy_top5 * 100, 2)
        })

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


def error_rate_per_class(true_labels, enrolled_labels, top_indices, dataset, query_scan_ids, similarity_matrix, filename, method_appendix=""):

    pred_labels = enrolled_labels[top_indices[:, 0]]

    classes = np.unique(true_labels)

    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Calculate error rates for each class
    error_rates = defaultdict(float)
    for cls in classes:
        class_indices = np.where(true_labels == cls)[0]
        class_true = true_labels[class_indices]
        class_pred = pred_labels[class_indices]
        errors = np.sum(class_true != class_pred)
        error_rate = errors / len(class_true)
        error_rates[cls] = error_rate
    df_classes = pd.DataFrame(list(error_rates.items()), columns=['Class', 'Error Rate'])

    # Calculate error for each scan
    misclassified_scans = []
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        true_class = idx_to_class[true]
        pred_class = idx_to_class[pred]

        if true != pred:

            similarities = similarity_matrix[i]
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_labels = enrolled_labels[sorted_indices]
            correct_rank = np.where(sorted_labels == true)[0]
            rank = correct_rank[0] + 1

            misclassified_scans.append({
                "Scan ID": query_scan_ids[i],
                "True Class": true_class,
                "Predicted Class": pred_class,
                "Rank": rank
            })

    df_scans = pd.DataFrame(misclassified_scans)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        df_classes.to_csv(os.path.join(tmp_dir, f'{filename}{method_appendix}_error_rate_per_class.csv'), index=False)
        df_scans.to_csv(tmp_dir / f'{filename}{method_appendix}_misclassified_scans.csv', index=False)

        classes = list(error_rates.keys())
        error_values = list(error_rates.values())

        plt.figure(figsize=(20, 5))
        plt.bar(classes, error_values, color='skyblue')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Error Rate', fontsize=12)
        plt.title('Error Rate per Class', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(tmp_dir, f'{filename}{method_appendix}_error_rate_per_class.jpg'), format='jpg', dpi=300)
        plt.close()

        mlflow.log_artifacts(tmp_dir, artifact_path="error_rate_per_class")
