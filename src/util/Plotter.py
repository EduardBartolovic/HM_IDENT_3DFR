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


def plot_confusion_matrix(true_labels, pred_labels, dataset, extension='', matplotlib=True):

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        """Plot confusion matrix"""
        if matplotlib:
            classes = dataset.classes
            cm = confusion_matrix(true_labels, pred_labels)
            row_sums = cm.sum(axis=1)
            # To avoid division by zero, set the zero elements to 1 (or a very small number)
            # The corresponding normalized row will be all zeros later on
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

        class_to_idx = dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}



        # https://github.com/sepandhaghighi/pycm
        cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)
        # Ensure the mapping includes all class indices

        # Relabel the confusion matrix
        cm.relabel(idx_to_class)
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
