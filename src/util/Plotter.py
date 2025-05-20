import itertools
import os
from pathlib import Path
from typing import List, Dict

import mlflow
import numpy as np
from matplotlib import pyplot as plt
from pycm import ConfusionMatrix, ROCCurve
from openTSNE import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tempfile


def plot_metric(output_path, epochs, metric_values: List[float], metric_name: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, len(metric_values)), metric_values, label=metric_name, marker='o')
    plt.title(f'{metric_name} Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f'{metric_name.lower().replace(" ", "_")}.jpg'))
    plt.close('all')


def plot_rrk_histogram(true_labels, pred_labels, similarity_matrix, dataset_name, method_appendix=""):
    """
    Plots a histogram of the ranks where the correct match was found in the similarity matrix,
    always showing ranks 1–30, with an extra category for ranks beyond 30.

    Args:
        true_labels: List or array of true class labels for query embeddings.
        pred_labels: List or array of enrolled labels.
        similarity_matrix: 2D numpy array with shape (n_queries, n_enrolled) representing cosine similarity.
        dataset_name: Name of the dataset for title/saving.
        method_appendix: Optional string to append to the method name or title.
    """
    if true_labels is None or pred_labels is None or similarity_matrix is None:
        return
    ranks = []
    for i, true_label in enumerate(true_labels):
        similarities = similarity_matrix[i]
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_labels = pred_labels[sorted_indices]
        correct_rank = np.where(sorted_labels == true_label)[0]
        if len(correct_rank) > 0:
            rank = correct_rank[0] + 1
            ranks.append(rank if rank <= 30 else 31)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        plt.figure(figsize=(10, 6))
        bins = np.arange(1, 33) - 0.5  # Bin edges from 0.5 to 31.5 to center bins on integers 1–31
        plt.hist(ranks, bins=bins, edgecolor='black', color='skyblue')

        counts, bins_edges = np.histogram(ranks, bins=bins)
        for i, count in enumerate(counts):
            if count > 0:
                plt.text(i + 1, count + 5, str(count), ha='center', fontsize=8)

        plt.xlabel("Rank of First Correct Match")
        plt.ylabel("Number of Queries")
        plt.title(f"Histogram of Matching Ranks - {dataset_name} {method_appendix}")

        xticks = list(range(1, 31)) + [31]
        xtick_labels = [str(x) for x in range(1, 31)] + ["31+"]
        plt.xticks(xticks, xtick_labels, rotation=45)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(tmp_dir, 'RRK_Histogram-' +dataset_name + '_' + method_appendix + '.svg'), format='svg')
        plt.close()
        mlflow.log_artifacts(tmp_dir, artifact_path="errorhistogram")

        # Log scale

        plt.figure(figsize=(10, 6))
        bins = np.arange(1, 33) - 0.5  # Bin edges from 0.5 to 31.5 to center bins on integers 1–31
        plt.hist(ranks, bins=bins, edgecolor='black', color='skyblue')

        counts, bins_edges = np.histogram(ranks, bins=bins)
        for i, count in enumerate(counts):
            if count > 0:
                plt.text(i + 1, count + 5, str(count), ha='center', fontsize=8)

        plt.xlabel("Rank of First Correct Match")
        plt.yscale('log')
        plt.ylabel("Number of Queries (log scale)")
        plt.title(f"Histogram of Matching Ranks - {dataset_name} {method_appendix}")

        xticks = list(range(1, 31)) + [31]
        xtick_labels = [str(x) for x in range(1, 31)] + ["31+"]
        plt.xticks(xticks, xtick_labels, rotation=45)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(tmp_dir, 'RRK_Histogram_log_scale-' +dataset_name + '_' + method_appendix + '.svg'), format='svg')
        plt.close()
        mlflow.log_artifacts(tmp_dir, artifact_path="errorhistogram")


def plot_confusion_matrix(true_labels, pred_labels, dataset, extension='', matplotlib=True):

    assert len(true_labels) == len(pred_labels)

    if len(np.unique(true_labels)) > 1000:
        # print("Too many classes for confusion_matrix. Continuing...")
        return 0

    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    if len(np.unique(true_labels)) < len(np.unique(pred_labels)):
        print(len(np.unique(true_labels)), len(np.unique(pred_labels)))
        print(np.unique(true_labels))
        print(np.unique(pred_labels))
        print("true_labels")
        for i in np.unique(true_labels):
            print(idx_to_class[i])
        print("pred_labels")
        for i in np.unique(pred_labels):
            print(idx_to_class[i])
        raise ValueError("len(np.unique(true_labels)), len(np.unique(pred_labels))")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        if matplotlib:
            classes = dataset.classes
            cm = confusion_matrix(true_labels, pred_labels)
            row_sums = cm.sum(axis=1)
            # To avoid division by zero, set the zero elements to 1. The corresponding normalized row will be all zeros later on
            row_sums[row_sums == 0] = 1
            cm_normalized = cm.astype('float') / row_sums[:, np.newaxis]
            plt.figure(figsize=(20, 20))
            plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=90)
            plt.yticks(tick_marks, classes)
            threshold = cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm_normalized[i, j] > threshold else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(os.path.join(tmp_dir, 'confusion_matrix'+extension+'_mpl.jpg'))

        # https://github.com/sepandhaghighi/pycm
        cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels, classes=np.unique(true_labels).tolist())
        try:
            cm.relabel(idx_to_class)
        except:
            print(idx_to_class)
            raise Exception('')
        cm.save_html(os.path.join(tmp_dir, 'confusion_matrix'+extension))
        cm.save_html(os.path.join(tmp_dir, 'normalized_confusion_matrix_'+extension), normalize=True)

        mlflow.log_artifacts(tmp_dir, artifact_path="confusion_matrix")


def tsne_plot(embeddings, labels, style_markers, title, filename, output_path, annotations_flag=True):
    """
    Using: # https://opentsne.readthedocs.io/en/latest/tsne_algorithm.html
    Todo: Improve init params https://opentsne.readthedocs.io/en/latest/examples/03_preserving_global_structure/03_preserving_global_structure.html#standard-t-sne
    :param embeddings:
    :param labels:
    :param style_markers:
    :param title:
    :param filename:
    :param output_path:
    :param annotations_flag:
    :return:
    """
    embeddings_2d = TSNE(n_components=2,
                         random_state=42,
                         n_jobs=8,
                         verbose=False).fit(embeddings)

    plt.figure(figsize=(20, 20))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
                    hue=labels, style=style_markers,
                    s=5, palette='viridis', legend=False)

    if annotations_flag:
        for i, txt in enumerate(labels):
            plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                         textcoords="offset points", xytext=(0, 1),
                         ha='center', fontsize=2)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(os.path.join(output_path, filename), format='svg', dpi=300)
    plt.close()


def pca_plot(embeddings, labels, style_markers, title, filename, output_path, annotations_flag=True, pca_model=None):
    """
    :param pca_model:
    :param embeddings:
    :param labels:
    :param style_markers:
    :param title:
    :param filename:
    :param output_path:
    :param annotations_flag:
    :return:
    """
    if pca_model is None:
        pca = PCA(n_components=2, random_state=42)
    else:
        pca = pca_model
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(20, 20))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
                    hue=labels, style=style_markers,
                    s=5, palette='viridis', legend=False)

    if annotations_flag:
        for i, txt in enumerate(labels):
            plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                         textcoords="offset points", xytext=(0, 1),
                         ha='center', fontsize=2)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(os.path.join(output_path, filename), format='svg', dpi=300)
    plt.close()

    return pca


def plot_embeddings(output_path, embeddings_train, class_labels, embeddings_val, class_labels_val, correctness_list,
                    sub_name='final', compresser='pca'):
    train_marker = np.array(['x'] * len(class_labels))
    val_marker = np.array(['s' if c else 'o' for c in correctness_list])

    concatenated_embeddings = np.concatenate((embeddings_train, embeddings_val), axis=0)
    concatenated_class = np.concatenate((class_labels, class_labels_val), axis=0)
    marker = np.concatenate((train_marker, val_marker), axis=0)

    plot_output_path = os.path.join(output_path, 't-SNEPlots')
    os.makedirs(plot_output_path, exist_ok=True)

    # Plot for Train Embeddings
    if compresser == 'pca':
        pca_model = pca_plot(embeddings_train, class_labels,
                             style_markers=train_marker,
                             title=f'PCA Visualization of Train Embeddings at {sub_name}',
                             filename=f'{sub_name}-train.svg',
                             output_path=plot_output_path,
                             annotations_flag=True)

        # Plot for Train+Val Embeddings
        pca_plot(concatenated_embeddings, concatenated_class,
                 style_markers=marker,
                 title=f'PCA Visualization of Embeddings at {sub_name}',
                 filename=f'{sub_name}-all.svg',
                 output_path=plot_output_path,
                 annotations_flag=True,
                 pca_model=pca_model)
    else:
        tsne_plot(embeddings_train, class_labels,
                  style_markers=train_marker,
                  title=f't-SNE Visualization of Train Embeddings at {sub_name}',
                  filename=f'{sub_name}-train.svg',
                  output_path=plot_output_path,
                  annotations_flag=True)

        # Plot for Train+Val Embeddings
        tsne_plot(concatenated_embeddings, concatenated_class,
                  style_markers=marker,
                  title=f't-SNE Visualization of Embeddings at {sub_name}',
                  filename=f'{sub_name}-all.svg',
                  output_path=plot_output_path,
                  annotations_flag=True)


def plot_metrics(output_path, num_epochs, train_loss_list: List, val_loss_list: List, metrics_list: Dict):
    """Plot training loss and validation accuracy"""
    metrics = {
        "Training Loss": train_loss_list,
        "Validation Loss": val_loss_list,
        "Validation Accuracy": metrics_list["accuracy_list"],
        "Validation Precision": metrics_list["precision_list"],
        "Validation Recall": metrics_list["recall_list"],
        "Validation F1-Score": metrics_list["f1_list"]
    }
    epochs = range(1, num_epochs + 1)
    for metric_name, metric_values in metrics.items():
        plot_metric(output_path, epochs, metric_values, metric_name, ylabel=metric_name.split()[-1])


def write_embeddings(embedding_library, dataset, epoch):

    with tempfile.TemporaryDirectory() as tmp_dir:
        np.savez_compressed(os.path.join(tmp_dir, f'{dataset}_enrolled_embeddings_{epoch}.npz'), embedding_library.enrolled_embeddings)
        np.savez_compressed(os.path.join(tmp_dir, f'{dataset}_enrolled_labels_{epoch}.npz'), embedding_library.enrolled_labels)
        np.savez_compressed(os.path.join(tmp_dir, f'{dataset}_enrolled_scan_ids_{epoch}.npz'), embedding_library.enrolled_scan_ids)
        np.savez_compressed(os.path.join(tmp_dir, f'{dataset}_enrolled_perspectives_{epoch}.npz'), embedding_library.enrolled_perspectives)
        np.savez_compressed(os.path.join(tmp_dir, f'{dataset}_query_embeddings_{epoch}.npz'), embedding_library.val_embeddings)
        np.savez_compressed(os.path.join(tmp_dir, f'{dataset}_query_labels_{epoch}.npz'), embedding_library.val_labels)
        np.savez_compressed(os.path.join(tmp_dir, f'{dataset}_query_scan_ids_{epoch}.npz'), embedding_library.val_scan_ids)
        np.savez_compressed(os.path.join(tmp_dir, f'{dataset}_query_perspectives_{epoch}.npz'), embedding_library.val_perspectives)
        # np.savez_compressed(os.path.join(tmp_dir, f'{dataset}_distances_{epoch}.npz'), embedding_library.distances)

        mlflow.log_artifacts(tmp_dir, artifact_path="embeddings")


def plot_weight_evolution(weights_log, save_dir="weights_logs"):
    os.makedirs(save_dir, exist_ok=True)

    line_styles = ['-', '--', '-.', ':']
    color_map = plt.cm.get_cmap('tab20', 26)  # up to 26 distinct colors

    for i, weight_history in enumerate(weights_log):
        weight_history = np.array(weight_history)  # (epochs, num_views)
        num_views = weight_history.shape[1]
        plt.figure(figsize=(10, 6))

        style_cycle = itertools.cycle(line_styles)

        for v in range(num_views):
            color = color_map(v % 26)
            linestyle = next(style_cycle) if v % 4 == 0 else '-'
            plt.plot(
                weight_history[:, v],
                label=f"View {v}",
                color=color,
                linestyle=linestyle,
                linewidth=1.3,
                alpha=0.85
            )

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Weight", fontsize=12)
        plt.title(f"Weight Evolution - Aggregator {i}", fontsize=14)
        plt.legend(fontsize=8, ncol=3, loc='upper right', frameon=False)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, f"aggregator_{i}_weight_evolution.png"))
        plt.close()