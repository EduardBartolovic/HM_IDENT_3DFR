import time
from collections import namedtuple
from copy import deepcopy

import random
import numpy as np
import torch
import os
from dotenv import load_dotenv
from tqdm import tqdm

from src.preprocess_datasets.rendering.Facerender import generate_rotation_matrices_cross_x_y, \
    generate_rotation_matrices
from src.util.Voting import accuracy_front_perspective, concat, score_fusion
from src.util.datapipeline.EmbeddingDataset import EmbeddingDataset
from src.util.misc import colorstr, smart_round, bold, underscore

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def get_embeddings_mv(enrolled_loader, query_loader, disable_bar=False):
    """
    Calculate embeddings for enrolled and query datasets using a multi-view backbone.
    """
    enrolled_embeddings_reg = []
    enrolled_labels = []
    enrolled_scan_ids = []
    enrolled_perspectives = 0
    enrolled_true_perspectives = []
    for embeddings, labels, scan_id, true_perspectives, path in tqdm(iter(enrolled_loader), disable=disable_bar,
                                                                     desc="Generate Enrolled Embeddings"):
        enrolled_embeddings_reg.append(embeddings.permute(1, 0, 2))
        enrolled_labels.extend(
            deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        enrolled_scan_ids.extend(deepcopy(scan_id))
        # enrolled_perspectives = np.array(perspectives).T[0]
        enrolled_true_perspectives.append(np.array(deepcopy(true_perspectives)).T)

    enrolled_embeddings_reg = np.concatenate(enrolled_embeddings_reg, axis=1)
    enrolled_labels = np.array([t.item() for t in enrolled_labels])
    enrolled_scan_ids = np.array(enrolled_scan_ids)
    # enrolled_perspectives = np.array(enrolled_perspectives)
    enrolled_true_perspectives = np.concatenate(enrolled_true_perspectives, axis=0)

    # if query_loader is None:
    #    Results = namedtuple("Results", ["enrolled_embeddings", "enrolled_labels", "enrolled_scan_ids", "enrolled_perspectives","enrolled_true_perspectives"])
    #    return Results(enrolled_embeddings_reg, enrolled_labels, enrolled_scan_ids, enrolled_perspectives, enrolled_true_perspectives)

    query_embeddings_reg = []
    query_labels = []
    query_scan_ids = []
    query_perspectives = 0
    query_true_perspectives = []
    for embeddings, labels, scan_id, true_perspectives, path in tqdm(iter(query_loader), disable=disable_bar,
                                                                     desc="Generate Query Embeddings"):
        query_embeddings_reg.append(embeddings.permute(1, 0, 2))
        query_labels.extend(
            deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        query_scan_ids.extend(deepcopy(scan_id))
        # query_perspectives = np.array(perspectives).T[0]
        query_true_perspectives.append(np.array(deepcopy(true_perspectives)).T)

    query_embeddings_reg = np.concatenate(query_embeddings_reg, axis=1)
    query_labels = np.array([t.item() for t in query_labels])
    query_scan_ids = np.array(query_scan_ids)
    # query_perspectives = np.array(query_perspectives)
    query_true_perspectives = np.concatenate(query_true_perspectives, axis=0)

    Results = namedtuple("Results",
                         ["enrolled_embeddings", "enrolled_labels", "enrolled_scan_ids", "enrolled_perspectives",
                          "query_embeddings", "query_labels", "query_scan_ids", "query_perspectives"])
    return Results(enrolled_embeddings_reg, enrolled_labels, enrolled_scan_ids, enrolled_true_perspectives,
                   query_embeddings_reg, query_labels, query_scan_ids, query_true_perspectives)


def evaluate_mv_emb_1_n(test_path, batch_size, views=None, disable_bar: bool = True):
    """
    Evaluate 1:N Model Performance on given test dataset
    """
    dataset_enrolled_path = os.path.join(test_path, 'enrolled')
    dataset_query_path = os.path.join(test_path, 'query')

    dataset_enrolled = EmbeddingDataset(dataset_enrolled_path, views, perspective_as_string=True)
    enrolled_loader = torch.utils.data.DataLoader(dataset_enrolled, batch_size=batch_size, shuffle=True, num_workers=8,
                                                  drop_last=False)

    dataset_query = EmbeddingDataset(dataset_query_path, views, perspective_as_string=True)
    query_loader = torch.utils.data.DataLoader(dataset_query, batch_size=batch_size, shuffle=True, num_workers=8,
                                               drop_last=False)

    if len(dataset_enrolled.classes) != len(dataset_enrolled):
        raise Exception(
            f"len(dataset_enrolled.classes): {len(dataset_enrolled.classes)} doesnt match len(dataset_enrolled.samples): {len(dataset_enrolled)} -> Check your dataset: {test_path}")

    time.sleep(0.1)

    embedding_library = get_embeddings_mv(enrolled_loader, query_loader, disable_bar)
    # enrolled_labels, query_labels = embedding_library.enrolled_labels, embedding_library.query_labels

    all_metrics = {}

    # --------- Single Front View ---------
    metrics_front, sim_front, top_idx, y_true_front, y_pred_front = accuracy_front_perspective(embedding_library)
    # plot_cmc(sim_front, enrolled_labels, query_labels, dataset_name, "front")
    # plot_rrk_histogram(query_labels, enrolled_labels, sim_front, dataset_name, "front")
    # error_rate_per_class(query_labels, enrolled_labels, top_idx, dataset_enrolled, embedding_library.query_scan_ids, sim_front, dataset_name, "_front")
    # all_metrics["emb_dist_front"] = analyze_embedding_distribution(sim_front, query_labels, enrolled_labels, dataset_name, "front", plot=True)
    all_metrics["metrics_front"] = metrics_front
    del sim_front, top_idx, y_true_front, y_pred_front

    # --------- Concat Full ---------
    metrics_concat, sim_concat, top_idx, y_true_concat, y_pred_concat = concat(embedding_library, disable_bar)
    # plot_cmc(sim_concat, enrolled_labels, query_labels, dataset_name, "concat")
    # plot_rrk_histogram(query_labels, enrolled_labels, sim_concat, dataset_name, "concat")
    # error_rate_per_class(query_labels, enrolled_labels, top_idx, dataset_enrolled, embedding_library.query_scan_ids, sim_concat, dataset_name, "_concat")
    # all_metrics["emb_dist_concat"] = analyze_embedding_distribution(sim_concat, query_labels, enrolled_labels, dataset_name, "concat", plot=True)
    all_metrics["metrics_concat"] = metrics_concat
    del sim_concat, top_idx, y_true_concat, y_pred_concat

    # --------- Concat Mean ---------
    metrics_concat_mean, similarity_matrix_concat_mean, top_indices_concat_mean, y_true_concat_mean, y_pred_concat_mean = concat(
        embedding_library, disable_bar, reduce_with="mean")
    # plot_cmc(similarity_matrix_concat_mean, enrolled_labels, query_labels, dataset_name, "concat_mean")
    # plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat_mean, dataset_name, "concat_mean")
    # all_metrics["emb_dist_concat_mean"] = analyze_embedding_distribution(similarity_matrix_concat_mean, query_labels, enrolled_labels, dataset_name, "concat_mean", plot=True)
    all_metrics["metrics_concat_mean"] = metrics_concat_mean
    del similarity_matrix_concat_mean, top_indices_concat_mean, y_true_concat_mean, y_pred_concat_mean

    # --------- Concat Median ---------
    metrics_concat_median, similarity_matrix_concat_median, top_indices_concat_median, y_true_concat_median, y_pred_concat_median = concat(
        embedding_library, disable_bar, reduce_with="median")
    # plot_cmc(similarity_matrix_concat_median, enrolled_labels, query_labels, dataset_name, "concat_median")
    # plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat_median, dataset_name, "concat_median")
    # all_metrics["emb_dist_concat_median"] = analyze_embedding_distribution(similarity_matrix_concat_median, query_labels, enrolled_labels, dataset_name, "concat_median", plot=True)
    all_metrics["metrics_concat_median"] = metrics_concat_median
    del similarity_matrix_concat_median, top_indices_concat_median, y_true_concat_median, y_pred_concat_median

    # --------- Concat PCA ---------
    # metrics_concat_pca, similarity_matrix_concat_pca, top_indices_concat_pca, y_true_concat_pca, y_pred_concat_pca = concat(embedding_library, disable_bar, reduce_with="pca")
    # plot_cmc(similarity_matrix_concat_pca, enrolled_labels, query_labels, dataset_name, "concat_pca")
    # plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat_pca, dataset_name, "concat_pca")
    # all_metrics["metrics_concat_pca"] = metrics_concat_pca
    # del similarity_matrix_concat_pca, top_indices_concat_pca, y_true_concat_pca, y_pred_concat_pca

    # --------- Score fusion ---------
    fusion_methods = ["max", "product", "majority", "mean", "median"]
    sim_score = None
    for m in fusion_methods:
        metrics, sim_score, fused, top_idx, pred = score_fusion(embedding_library, disable_bar, method=m,
                                                                similarity_matrix=sim_score, distance_matrix=None)
        all_metrics[f"metrics_score_{m}"] = metrics
        # all_metrics[f"emb_dist_score_{m}"] = analyze_embedding_distribution(fused, query_labels, enrolled_labels, dataset_name, f"score_{m}", plot=True)

    # plot_all_cmc_from_txt(dataset_name)

    return all_metrics, embedding_library, dataset_enrolled, dataset_query


def print_results(neutral_dataset, dataset_enrolled, dataset_query, all_metrics):
    rank_1_front = smart_round(all_metrics["metrics_front"].get('Rank-1 Rate', 'N/A'))
    rank_5_front = smart_round(all_metrics["metrics_front"].get('Rank-5 Rate', 'N/A'))
    mrr_front = smart_round(all_metrics["metrics_front"].get('MRR', 'N/A'))
    # gbig_front = smart_round(all_metrics["emb_dist_front"].get('gbig', 'N/A')*100)

    rank_1_concat = smart_round(all_metrics["metrics_concat"].get('Rank-1 Rate', 'N/A'))
    rank_5_concat = smart_round(all_metrics["metrics_concat"].get('Rank-5 Rate', 'N/A'))
    mrr_concat = smart_round(all_metrics["metrics_concat"].get('MRR', 'N/A'))
    # gbig_concat = smart_round(all_metrics["emb_dist_concat"].get('gbig', 'N/A')*100)

    rank_1_concat_mean = smart_round(all_metrics["metrics_concat_mean"].get('Rank-1 Rate', 'N/A'))
    rank_5_concat_mean = smart_round(all_metrics["metrics_concat_mean"].get('Rank-5 Rate', 'N/A'))
    mrr_concat_mean = smart_round(all_metrics["metrics_concat_mean"].get('MRR', 'N/A'))
    # gbig_concat_mean = smart_round(all_metrics["emb_dist_concat_mean"].get('gbig', 'N/A')*100)

    rank_1_concat_median = smart_round(all_metrics["metrics_concat_median"].get('Rank-1 Rate', 'N/A'))
    rank_5_concat_median = smart_round(all_metrics["metrics_concat_median"].get('Rank-5 Rate', 'N/A'))
    mrr_concat_median = smart_round(all_metrics["metrics_concat_median"].get('MRR', 'N/A'))

    # rank_1_concat_pca = smart_round(all_metrics["metrics_concat_pca"].get('Rank-1 Rate', 'N/A'))
    # rank_5_concat_pca = smart_round(all_metrics["metrics_concat_pca"].get('Rank-5 Rate', 'N/A'))
    # mrr_concat_pca = smart_round(all_metrics["metrics_concat_pca"].get('MRR', 'N/A'))

    mrr_score_max = smart_round(all_metrics["metrics_score_max"].get('MRR', 'N/A'))
    # gbig_score_max = smart_round(all_metrics["emb_dist_score_max"].get('gbig', 'N/A')*100)

    mrr_score_prod = smart_round(all_metrics["metrics_score_product"].get('MRR', 'N/A'))
    # gbig_score_prod = smart_round(all_metrics["emb_dist_score_product"].get('gbig', 'N/A')*100)

    mrr_score_mean = smart_round(all_metrics["metrics_score_mean"].get('MRR', 'N/A'))
    # gbig_score_mean = smart_round(all_metrics["emb_dist_score_mean"].get('gbig', 'N/A')*100)

    mrr_score_majority = smart_round(all_metrics["metrics_score_majority"].get('MRR', 'N/A'))
    # gbig_score_majority = smart_round(all_metrics["emb_dist_score_majority"].get('gbig', 'N/A')*100)

    # mrr_score_pdw = smart_round(all_metrics["metrics_score_pdw"].get('MRR', 'N/A'))
    string = (f"{neutral_dataset} E{len(dataset_enrolled)}Q{len(dataset_query)}: " +
              f"{('Front RR1')}: {rank_1_front} {'MRR'}: {(mrr_front)} | "  # {bold('GAIG')}: {underscore(gaig_front)} | "
              f"{('Concat RR1')}: {rank_1_concat} {'MRR'}: {(mrr_concat)} | "  # {bold('GAIG')}: {underscore(gaig_concat)} | "
              f"{('Concat_Mean RR1')}: {rank_1_concat_mean} {('MRR')}: {(mrr_concat_mean)} | "
              f"{('Concat_Median RR1')}: {rank_1_concat_median} {('MRR')}: {(mrr_concat_median)} | "
              # f"{bold('Concat_PCA RR1')}: {rank_1_concat_pca} {bold('MRR')}: {underscore(mrr_concat_pca)} | "
              f"{('Score_prod MRR')}: {(mrr_score_prod)} | "
              f"{('Score_mean MRR')}: {(mrr_score_mean)} | "
              f"{('Score_max MRR')}: {(mrr_score_max)} | "
              f"{('Score_maj MRR')}: {(mrr_score_majority)} | "
              # f"{bold('Score_pdw MRR')}: {underscore(mrr_score_pdw)} | "
              # f"{bold('MV RR1')}: {rank_1_mv} {bold('MRR')}: {underscore(mrr_mv)} {bold('GBIG')}: {underscore(gbig_mv)}"
              )
    print(string)


def evaluate_and_log_mv(data_root, test_views, batch_size, disable_bar: bool = True):
    print(colorstr('bright_green', f"Perform 1:N Evaluation on {test_views}"))
    all_metrics, embedding_library, dataset_enrolled, dataset_query = evaluate_mv_emb_1_n(data_root, batch_size, test_views, disable_bar)

    neutral_dataset = "Dataset: " + str(test_views)

    print_results(neutral_dataset, dataset_enrolled, dataset_query, all_metrics)

    return all_metrics


def generate_view_subsets_sampled(
        base_set,
        max_k=30,
        samples_per_k=50,
        seed=0
):
    random.seed(seed)

    # Ensure 0_0 is present and fixed
    base_set = list(base_set)
    base_set.remove("0_0")
    other_elements = base_set

    sampled_subsets = []

    for k in range(2, max_k + 1):
        level = []

        if k == 1:
            level.append(["0_0"])
        else:
            seen = set()
            while len(level) < samples_per_k:
                perm = tuple(sorted(random.sample(other_elements, k - 1)))
                if perm in seen:
                    continue
                seen.add(perm)
                level.append(["0_0", *perm])

        sampled_subsets.append(level)

        print(f"Level {k}: {len(level)} sampled subsets")

    return sampled_subsets


def generate_cross_views(
        max_angle=35,
        step=5,
):
    """
    Generate views where yaw=0 OR pitch=0 (cross),
    sampled every `step` degrees up to `max_angle`.
    """
    angles = list(range(-max_angle, max_angle + 1, step))

    views = set()

    for a in angles:
        views.add((a, 0))  # yaw axis
        views.add((0, a))  # pitch axis

    return sorted(views)


def views_to_strings(views):
    return [f"{yaw}_{pitch}" for yaw, pitch in views]


def generate_cross_experiments(
        max_angles=(10, 15, 25, 35),
        steps=(1, 2, 3, 5, 10)
):
    experiments = []

    for max_a in max_angles:
        for step in steps:
            views = generate_cross_views(
                max_angle=max_a,
                step=step
            )
            experiments.append(views_to_strings(views))

    return experiments


def main(cfg):
    SEED = 42
    torch.manual_seed(SEED)

    DATA_ROOT = "F:\\Face\\data\\dataset14_emb\\test_rgb_bff_crop261_emb-irseglintr18\\"  # "/home/gustav/dataset14_emb/test_rgb_bff_crop261_emb-irseglintr18/"  # the parent root where the datasets are stored
    BATCH_SIZE = 16  # Batch size in training

    # ======= Validation =======
    evaluate_and_log_mv(DATA_ROOT, cfg['TEST_VIEWS'], BATCH_SIZE, disable_bar=True)


if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')
    render_angles = [-35, -25, -15, -10, -5, 0, 5, 10, 15, 25, 35]
    all_views = (
            generate_rotation_matrices_cross_x_y()
            + generate_rotation_matrices(render_angles)
    )

    yaw_pitch_pairs = [(x, y) for x, y, _ in all_views]
    unique_views = sorted(set(yaw_pitch_pairs))
    all_views_set = [f"{x}_{y}" for (x, y) in unique_views]

    allowed = []  # generate_view_subsets_sampled(all_views_set)
    print(f"✅ Using {len(allowed)} random unique perspectives")

    cross_experiments = generate_cross_experiments(
        max_angles=[10, 15, 25, 35],
        steps=[1, 2, 3, 4, 5, 10, 15]
    )
    print(f"✅ Using {len(cross_experiments)} cross unique perspectives")
    allowed.append(cross_experiments)

    # TODO Add more extras: Diagnonal and cross and no neg pitch
    extras = [['0_0', '-25_0', '-10_0', '25_0', '10_0'],  # 1 Azimuth axis
              ['0_0', '0_-25', '0_-10', '0_10', '0_25'],  # 2 Alitude axis
              ['0_0', '-25_-25', '-10_-10', '10_10', '25_25'],  # 3 diagonal
              ['0_0', '-25_25', '-10_10', '10_-10', '25_-25'],  # 4 diagonal
              ['0_0', '-25_25', '-10_10', '10_-10', '25_-25', '-25_-25', '-10_-10', '10_10', '25_25'],  # 5 cross
              ['0_0', '-25_-25', '-25_25', '25_-25', '25_25'],  # 6 Only corners
              ['0_0', '-10_-10', '-10_10', '10_-10', '10_10'],  # 7 Only middle
              ['0_0', '25_-25', '25_-10', '25_0', '25_10', '25_25'],  # 8 Top row
              ['0_0', '10_-25', '10_-10', '10_0', '10_10', '10_25'],  # 9 2nd Top row
              ['0_0', '25_-25', '25_-10', '25_0', '25_10', '25_25', '10_-25', '10_-10', '10_0', '10_10', '10_25'],
              # 10 Top All
              ['0_0', '25_-25', '25_-10', '25_0', '25_10', '25_25', '10_-25', '10_-10', '10_0', '10_10', '10_25',
               '0_-25', '0_-10', '0_10', '0_25'],  # 11 Top All + Middle
              ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25'],  # 12 Top Corners + Sides
              ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0'],
              # 13 Top Corners + Sides + Mid  FAV! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
              ['0_0', '10_-10', '10_10', '-10_10', '-10_-10'],  # Inner Corners
              ['0_0', '10_-10', '10_10', '-10_10', '-10_-10', '-10_0', '10_0', '0_10', '0_-10'],  # Inner Ring
              ['0_0', '10_-10', '10_10', '10_0', '0_10', '0_-10'],  # Inner Top Ring
              ['-25_-25'],
              ['-25_-10'],
              ['-25_0'],
              ['-25_10'],
              ['-25_25'],
              ['-10_-25'],
              ['-10_-10'],
              ['-10_0'],
              ['-10_10'],
              ['-10_25'],
              ['0_-25'],
              ['0_-10'],
              ['0_0'],
              ['0_10'],
              ['0_25'],
              ['25_-25'],
              ['25_-10'],
              ['25_0'],
              ['25_10'],
              ['25_25'],
              ['10_-25'],
              ['10_-10'],
              ['10_0'],
              ['10_10'],
              ['10_25']
              ]
    allowed.append(extras)

    extras2 = [
        # --- Pure large-angle axes ---
        ['0_0', '-35_0', '-15_0', '15_0', '35_0'],  # Yaw axis wide
        ['0_0', '0_-35', '0_-15', '0_15', '0_35'],  # Pitch axis wide

        # --- Large diagonals ---
        ['0_0', '-35_-35', '-15_-15', '15_15', '35_35'],  # Main diagonal
        ['0_0', '-35_35', '-15_15', '15_-15', '35_-35'],  # Anti-diagonal

        # --- Large cross ---
        ['0_0',
         '-35_0', '-15_0', '15_0', '35_0',
         '0_-35', '0_-15', '0_15', '0_35'],

        # --- Outer ring (big box) ---
        ['0_0',
         '-35_-35', '-35_0', '-35_35',
         '0_-35', '0_35',
         '35_-35', '35_0', '35_35'],

        # --- Outer corners only ---
        ['0_0', '-35_-35', '-35_35', '35_-35', '35_35'],

        # --- Top hemisphere emphasis (positive pitch) ---
        ['0_0',
         '-35_35', '-15_35', '15_35', '35_35',
         '-15_15', '0_15', '15_15'],

        # --- Bottom hemisphere emphasis (negative pitch) ---
        ['0_0',
         '-35_-35', '-15_-35', '15_-35', '35_-35',
         '-15_-15', '0_-15', '15_-15'],

        # --- Vertical slices at extreme yaw ---
        ['0_0', '-35_-35', '-35_-15', '-35_0', '-35_15', '-35_35'],
        ['0_0', '35_-35', '35_-15', '35_0', '35_15', '35_35'],

        # --- Horizontal slices at extreme pitch ---
        ['0_0', '-35_35', '-15_35', '0_35', '15_35', '35_35'],
        ['0_0', '-35_-35', '-15_-35', '0_-35', '15_-35', '35_-35'],

        # --- Asymmetric stress tests ---
        ['0_0', '-35_35', '-15_0', '15_-15', '35_-35'],
        ['0_0', '-35_-15', '-15_35', '15_0', '35_15'],

        # --- Sparse but extreme ---
        ['0_0', '-35_0', '0_35', '35_0'],
        ['0_0', '0_-35', '35_35', '-35_-35'],

        # --- Single extreme views (controls) ---
        ['35_35'],
        ['35_-35'],
        ['-35_35'],
        ['-35_-35'],

        # --- No-front (remove 0_0 entirely) ---
        ['-35_0', '-15_0', '15_0', '35_0'],
        ['0_-35', '0_-15', '0_15', '0_35'],
        ['-35_-35', '-15_-15', '15_15', '35_35'],

        # --- Only extremes, no mids ---
        ['0_0', '-35_0', '35_0'],
        ['0_0', '0_-35', '0_35'],
        ['0_0', '-35_-35', '35_35'],
        ['0_0', '-35_35', '35_-35'],

        # --- Horseshoe / U-shapes ---
        ['0_0', '-35_-35', '-35_0', '-35_35', '0_35'],
        ['0_0', '35_-35', '35_0', '35_35', '0_35'],
        ['0_0', '-35_-35', '0_-35', '35_-35', '35_0'],

        # --- L-shaped coverage ---
        ['0_0', '-35_0', '-35_35', '0_35'],
        ['0_0', '35_0', '35_-35', '0_-35'],
        ['0_0', '-15_0', '-15_15', '0_15'],

        # --- Skewed diagonals (non-symmetric) ---
        ['0_0', '-35_-15', '-15_35', '15_15', '35_-35'],
        ['0_0', '-35_15', '-15_-35', '15_-15', '35_35'],

        # --- One-side bias (yaw-heavy / pitch-light) ---
        ['0_0', '35_-15', '35_0', '35_15'],
        ['0_0', '-35_-15', '-35_0', '-35_15'],

        # --- Pitch-heavy / yaw-light ---
        ['0_0', '-15_35', '0_35', '15_35'],
        ['0_0', '-15_-35', '0_-35', '15_-35'],

        # --- Checkerboard sparse ---
        ['0_0', '-35_35', '-15_15', '15_-15', '35_-35'],
        ['0_0', '-35_-35', '-15_15', '15_-15', '35_35'],

        # --- Radius jump (center → extreme only) ---
        ['0_0', '-35_35'],
        ['0_0', '35_-35'],
        ['0_0', '0_35'],
        ['0_0', '35_0'],

        # --- Off-grid realism (simulates noisy capture bias) ---
        ['0_0', '-35_15', '-15_35', '15_35', '35_15'],
        ['0_0', '-35_-15', '-15_-35', '15_-35', '35_-15'],

        # --- Minimal but dangerous ---
        ['-35_35', '35_-35'],
        ['-35_-35', '35_35'],

        # --- Full main diagonal (yaw == pitch) ---
        ['-35_-35', '-25_-25', '-15_-15', '-10_-10', '-5_-5',
         '0_0',
         '5_5', '10_10', '15_15', '25_25', '35_35'],

        # --- Full anti-diagonal (yaw == -pitch) ---
        ['-35_35', '-25_25', '-15_15', '-10_10', '-5_5',
         '0_0',
         '5_-5', '10_-10', '15_-15', '25_-25', '35_-35'],

        # --- Full diagonal WITHOUT center ---
        ['-35_-35', '-25_-25', '-15_-15', '-10_-10', '-5_-5',
         '5_5', '10_10', '15_15', '25_25', '35_35'],

        # --- Full anti-diagonal WITHOUT center ---
        ['-35_35', '-25_25', '-15_15', '-10_10', '-5_5',
         '5_-5', '10_-10', '15_-15', '25_-25', '35_-35'],

        # --- Sparse diagonal (large steps only) ---
        ['-35_-35', '-15_-15', '0_0', '15_15', '35_35'],
        ['-35_35', '-15_15', '0_0', '15_-15', '35_-35'],

        # --- Extreme-only diagonal ---
        ['-35_-35', '35_35'],
        ['-35_35', '35_-35'],

        # --- Asymmetric diagonal sampling ---
        ['-35_-35', '-25_-25', '-10_-10', '0_0', '15_15', '35_35'],
        ['-35_35', '-25_25', '-10_10', '0_0', '15_-15', '35_-35'],

        # --- Half-diagonal (one-sided from center) ---
        ['0_0', '5_5', '10_10', '15_15', '25_25', '35_35'],
        ['0_0', '5_-5', '10_-10', '15_-15', '25_-25', '35_-35'],

        # --- Broken diagonal (gap stress test) ---
        ['-35_-35', '-25_-25', '0_0', '25_25', '35_35'],
        ['-35_35', '-25_25', '0_0', '25_-25', '35_-35'],
    ]
    allowed.append(extras2)

    for num_views in allowed:
        print(len(num_views))
        for selected_views in num_views:
            cfg_yaml = {"TEST_VIEWS": selected_views}
            main(cfg_yaml)

    print("DONE")
