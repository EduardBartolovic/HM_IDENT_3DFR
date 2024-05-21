import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calc_metrics(y_true, y_pred, y_pred_top5, prints=False):
    # https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

    labels = np.unique(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    rank_1_rate = accuracy * 100
    accuracy_top5 = np.mean(np.any(y_pred_top5 == y_true[:, None], axis=1))

    if prints:
        print(
            f"AccuracyTop5: {accuracy_top5:.4f} "
            f"Accuracy: {accuracy:.4f} "
            f"Precision: {precision:.4f}"
            f" Recall: {recall:.4f} "
            f"F1-score: {f1:.4f} "
            f"RR1: {rank_1_rate:.4f}")

    return accuracy, precision, recall, f1, rank_1_rate, accuracy_top5


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
