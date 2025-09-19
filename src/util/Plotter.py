import itertools
import os
from pathlib import Path

import mlflow
import numpy as np
from matplotlib import pyplot as plt
from pycm import ConfusionMatrix, ROCCurve
from openTSNE import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, DetCurveDisplay, auc
import seaborn as sns
import tempfile


def plot_rrk_histogram(true_labels, enrolled_labels, similarity_matrix, dataset_name, method_appendix=""):
    """
    Plots a histogram of the ranks where the correct match was found in the similarity matrix,
    always showing ranks 1–30, with an extra category for ranks beyond 30.

    Args:
        true_labels: List or array of true class labels for query embeddings.
        enrolled_labels: List or array of enrolled labels.
        similarity_matrix: 2D numpy array with shape (n_queries, n_enrolled) representing cosine similarity.
        dataset_name: Name of the dataset for title/saving.
        method_appendix: Optional string to append to the method name or title.
    """
    if true_labels is None or enrolled_labels is None or similarity_matrix is None:
        return
    ranks = []
    for i, true_label in enumerate(true_labels):
        similarities = similarity_matrix[i]
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_labels = enrolled_labels[sorted_indices]
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
        plt.savefig(os.path.join(tmp_dir, 'RRK_Histogram-' + dataset_name + '_' + method_appendix + '.svg'), format='svg')
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


def plot_verification(all_fold_results, dataset_name, method_appendix):
    """
    Plot combined verification results across all folds with average curves.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # --- Score distributions ---
        plt.figure(figsize=(7, 5))
        genuine_scores = []
        imposter_scores = []
        for res in all_fold_results:
            gs = res["test_scores"][res["test_labels"] == 1]
            is_ = res["test_scores"][res["test_labels"] == 0]
            genuine_scores.append(gs)
            imposter_scores.append(is_)
            sns.kdeplot(gs, label=f"Genuine Fold {res['fold']}", fill=False)
            sns.kdeplot(is_, label=f"Imposter Fold {res['fold']}", fill=False)

        # plot average KDE
        all_genuine = np.concatenate(genuine_scores)
        all_imposter = np.concatenate(imposter_scores)
        sns.kdeplot(all_genuine, label="Genuine Average", color="black", lw=3)
        sns.kdeplot(all_imposter, label="Imposter Average", color="black", lw=3)

        plt.title(f"{dataset_name} {method_appendix} - Score Distributions")
        plt.xlabel("Similarity / Distance")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(tmp_dir / f'Distribution-{dataset_name}_{method_appendix}.svg', format='svg')
        plt.close()

        # --- Precision-Recall ---
        plt.figure(figsize=(7, 5))
        recall_list = []
        precision_list = []
        for res in all_fold_results:
            plt.plot(res["recall"], res["precision"], label=f"Fold {res['fold']} (AP={res['avg_precision']:.3f})")
            recall_list.append(res["recall"])
            precision_list.append(res["precision"])

        # average precision curve
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend()
        plt.grid()
        plt.savefig(tmp_dir / f'PR_Curve-{dataset_name}_{method_appendix}.svg', format='svg')
        plt.close()

        # --- DET Curve ---
        plt.figure(figsize=(7, 5))
        fpr_list = []
        fnr_list = []
        for res in all_fold_results:
            DetCurveDisplay(fpr=res["fpr"], fnr=1 - res["tpr"]).plot(ax=plt.gca(), name=f"Fold {res['fold']}")
            fpr_list.append(res["fpr"])
            fnr_list.append(1 - res["tpr"])

        DetCurveDisplay(fpr=res["fpr"], fnr=1 - res["tpr"]).plot(ax=plt.gca(), name="Average", color='black', linewidth=3)
        plt.title("DET Curves across folds")
        plt.grid(True)
        plt.legend()
        plt.savefig(tmp_dir / f'DET_Curve-{dataset_name}_{method_appendix}.svg', format='svg')
        plt.close()

        # --- ROC Curve ---
        plt.figure(figsize=(8, 6))
        for res in all_fold_results:
            plt.plot(res["fpr"], res["tpr"], lw=1.5, label=f"Fold {res['fold']} (AUC={res['roc_auc']:.3f}, EER={res['eer']:.3f})")

        plt.plot([0, 1], [0, 1], "k:", label="Random guess")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("1:1 Verification ROC across folds")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(tmp_dir / f'ROC_Curve-{dataset_name}_{method_appendix}.svg', format='svg')
        plt.close()

        # log to mlflow
        mlflow.log_artifacts(tmp_dir, artifact_path="Verification")


def plot_cmc(similarity_matrix, gallery_labels, probe_labels, dataset, extension='', top_k=100):
    """
    Compute and plot the CMC (Cumulative Matching Characteristic) curve.

    Args:
        similarity_matrix: (num_probes, num_gallery) similarity scores. Higher = more similar.
        gallery_labels: np.array (num_gallery,) - identity labels of gallery
        probe_labels: np.array (num_probes,) - identity labels of probe
        dataset: str - dataset name for title/saving
        extension: str - extra filename identifier
        top_k: int - max rank for CMC

    Returns:
        cmc_curve: np.array of shape (top_k,)
        auc_cmc: float, area under the CMC curve (normalized)
    """

    if similarity_matrix is None:
        return None

    num_probes = similarity_matrix.shape[0]
    ranks = np.zeros(top_k)

    for i in range(num_probes):
        sims = similarity_matrix[i]
        sorted_idx = np.argsort(sims)[::-1]
        sorted_labels = gallery_labels[sorted_idx]

        correct_label = probe_labels[i]
        rank = np.where(sorted_labels == correct_label)[0]
        if len(rank) > 0 and rank[0] < top_k:
            ranks[rank[0]:] += 1

    cmc_curve = ranks / num_probes

    x_ranks = np.arange(1, top_k + 1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        plt.figure(figsize=(9, 6))
        plt.plot(
            x_ranks,
            cmc_curve[:top_k] * 100,
            marker="o",
            color="royalblue",
            linewidth=2
        )
        plt.xlabel("Rank", fontsize=12)
        plt.ylabel("Identification Rate (%)", fontsize=12)
        plt.title(f"CMC Curve – {dataset} - {extension}", fontsize=14)
        xticks = [1, 5, 10, 25, 50, 100]
        xticks = [x for x in xticks if x <= top_k]  # filter out ticks beyond top_k
        plt.xticks(xticks, fontsize=10)
        plt.yticks(fontsize=10)
        if cmc_curve[0] > 0.995:
            plt.ylim((99.5, 100))
        elif cmc_curve[0] > 0.99:
            plt.ylim((99, 100))
        elif cmc_curve[0] > 0.95:
            plt.ylim((95, 100))
        elif cmc_curve[0] > 0.90:
            plt.ylim((90, 100))
        elif cmc_curve[0] > 0.85:
            plt.ylim((85, 100))
        elif cmc_curve[0] > 0.70:
            plt.ylim((70, 100))
        elif cmc_curve[0] > 0.50:
            plt.ylim((50, 100))
        elif cmc_curve[0] > 0.25:
            plt.ylim((25, 100))
        plt.grid(True, linestyle="--", alpha=0.6)

        # Annotation box
        rank1 = cmc_curve[0] * 100
        rank3 = cmc_curve[2] * 100 if top_k >= 3 else None
        rank5 = cmc_curve[4] * 100 if top_k >= 5 else None
        rank10 = cmc_curve[9] * 100 if top_k >= 10 else None
        rank25 = cmc_curve[24] * 100 if top_k >= 25 else None
        rank50 = cmc_curve[49] * 100 if top_k >= 50 else None

        annotation = f"Rank-1: {rank1:.2f}%"
        if rank3 is not None:
            annotation += f"\nRank-3: {rank3:.2f}%"
        if rank5 is not None:
            annotation += f"\nRank-5: {rank5:.2f}%"
        if rank10 is not None:
            annotation += f"\nRank-10: {rank10:.2f}%"
        if rank25 is not None:
            annotation += f"\nRank-25: {rank25:.2f}%"
        if rank50 is not None:
            annotation += f"\nRank-50: {rank50:.2f}%"

        plt.text(
            0.98,
            0.02,
            annotation,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
        )
        plt.tight_layout()
        plt.savefig(os.path.join(tmp_dir, 'CMC_Curve_-' + dataset + '_' + extension + '.svg'), format='svg')
        plt.close()

        mlflow.log_artifacts(tmp_dir, artifact_path="CMC_Curve")


def analyze_identification_distribution(similarity_matrix, query_labels, enrolled_labels, dataset_name, extension="", plot=True):
    """
    Analyze 1:N identification results by plotting score distributions
    for genuine (correct) vs impostor (incorrect) matches.

    Args:
        similarity_matrix: (num_queries x num_gallery) similarity scores
        query_labels: Labels of query samples
        enrolled_labels: Labels of gallery samples
        dataset_name: Name of the dataset (for logging/plotting)
        extension: Extra string for logging/plotting
        plot: If True, makes the KDE plot of distributions
    """

    # Broadcast comparison: shape (num_queries, num_gallery)
    matches = query_labels[:, None] == enrolled_labels[None, :]

    genuine_scores = similarity_matrix[matches]
    impostor_scores = similarity_matrix[~matches]

    if plot:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            plt.figure(figsize=(7, 5))
            sns.kdeplot(genuine_scores, label="Genuine", fill=False)
            sns.kdeplot(impostor_scores, label="Impostor", fill=False)
            plt.title(f"{dataset_name} {extension} - Identification Distributions")
            plt.xlabel("Similarity / Distance")
            plt.ylabel("Density")
            plt.legend()
            plt.savefig(os.path.join(tmp_dir, 'CMC_Curve_-' + dataset_name + '_' + extension + '.svg'), format='svg')

            mlflow.log_artifacts(tmp_dir, artifact_path="IdentificationDistributions")

    # return { # TODO: Maybe use these
    #    "genuine_mean": np.mean(genuine_scores),
    #    "genuine_std": np.std(genuine_scores),
    #    "impostor_mean": np.mean(impostor_scores),
    #    "impostor_std": np.std(impostor_scores)
    # }


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