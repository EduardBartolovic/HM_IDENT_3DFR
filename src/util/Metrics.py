import os
import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
