import time
from collections import namedtuple
from copy import deepcopy

import numpy as np
import torch
import os
import yaml
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

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
    for embeddings, labels, perspectives, true_perspectives, face_corr, scan_id in tqdm(iter(enrolled_loader), disable=disable_bar, desc="Generate Enrolled Embeddings"):

        enrolled_embeddings_reg.append(np.stack([t.cpu().numpy() for t in embeddings]))
        enrolled_labels.extend(deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        enrolled_scan_ids.extend(deepcopy(scan_id))
        enrolled_perspectives = np.array(perspectives).T[0]
        enrolled_true_perspectives.append(np.array(deepcopy(true_perspectives)).T)

    enrolled_embeddings_reg = np.concatenate(enrolled_embeddings_reg, axis=1)
    enrolled_labels = np.array([t.item() for t in enrolled_labels])
    enrolled_scan_ids = np.array(enrolled_scan_ids)
    enrolled_perspectives = np.array(enrolled_perspectives)
    enrolled_true_perspectives = np.concatenate(enrolled_true_perspectives, axis=0)

    if query_loader is None:
        Results = namedtuple("Results", ["enrolled_embeddings", "enrolled_labels", "enrolled_scan_ids", "enrolled_perspectives","enrolled_true_perspectives"])
        return Results(enrolled_embeddings_reg, enrolled_labels, enrolled_scan_ids, enrolled_perspectives, enrolled_true_perspectives)

    query_embeddings_reg = []
    query_labels = []
    query_scan_ids = []
    query_perspectives = 0
    query_true_perspectives = []
    for embeddings, labels, perspectives, true_perspectives, face_corr, scan_id in tqdm(iter(query_loader), disable=disable_bar, desc="Generate Query Embeddings"):
        query_embeddings_reg.append(np.stack([t.cpu().numpy() for t in embeddings]))
        query_labels.extend(deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        query_scan_ids.extend(deepcopy(scan_id))
        query_perspectives = np.array(perspectives).T[0]
        query_true_perspectives.append(np.array(deepcopy(true_perspectives)).T)

    query_embeddings_reg = np.concatenate(query_embeddings_reg, axis=1)
    query_labels = np.array([t.item() for t in query_labels])
    query_scan_ids = np.array(query_scan_ids)
    query_perspectives = np.array(query_perspectives)
    query_true_perspectives = np.concatenate(query_true_perspectives, axis=0)

    Results = namedtuple("Results", ["enrolled_embeddings", "enrolled_labels", "enrolled_scan_ids", "enrolled_perspectives", "enrolled_true_perspectives", "query_embeddings", "query_labels", "query_scan_ids", "query_perspectives", "query_true_perspectives"])
    return Results(enrolled_embeddings_reg, enrolled_labels, enrolled_scan_ids, enrolled_perspectives, enrolled_true_perspectives, query_embeddings_reg, query_labels, query_scan_ids, query_perspectives, query_true_perspectives)


def evaluate_mv_emb_1_n(test_path, batch_size, disable_bar: bool):
    """
    Evaluate 1:N Model Performance on given test dataset
    """
    dataset_name = os.path.basename(test_path)
    dataset_enrolled_path = os.path.join(test_path, 'enrolled')
    dataset_query_path = os.path.join(test_path, 'query')

    dataset_enrolled = EmbeddingDataset(dataset_enrolled_path)
    enrolled_loader = torch.utils.data.DataLoader(dataset_enrolled, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

    dataset_query = EmbeddingDataset(dataset_query_path)
    query_loader = torch.utils.data.DataLoader(dataset_enrolled, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

    if len(dataset_enrolled.classes) != len(dataset_enrolled):
        raise Exception(f"len(dataset_enrolled.classes): {len(dataset_enrolled.classes)} doesnt match len(dataset_enrolled.samples): {len(dataset_enrolled)} -> Check your dataset: {test_path}")

    time.sleep(0.1)

    embedding_library = get_embeddings_mv(enrolled_loader, query_loader, disable_bar)
    #enrolled_labels, query_labels = embedding_library.enrolled_labels, embedding_library.query_labels

    all_metrics = {}

    # --------- Single Front View ---------
    metrics_front, sim_front, top_idx, y_true_front, y_pred_front = accuracy_front_perspective(embedding_library)
    #plot_cmc(sim_front, enrolled_labels, query_labels, dataset_name, "front")
    #plot_rrk_histogram(query_labels, enrolled_labels, sim_front, dataset_name, "front")
    #error_rate_per_class(query_labels, enrolled_labels, top_idx, dataset_enrolled, embedding_library.query_scan_ids, sim_front, dataset_name, "_front")
    #all_metrics["emb_dist_front"] = analyze_embedding_distribution(sim_front, query_labels, enrolled_labels, dataset_name, "front", plot=True)
    all_metrics["metrics_front"] = metrics_front
    del sim_front, top_idx, y_true_front, y_pred_front

    # --------- Concat Full ---------
    metrics_concat, sim_concat, top_idx, y_true_concat, y_pred_concat = concat(embedding_library, disable_bar)
    #plot_cmc(sim_concat, enrolled_labels, query_labels, dataset_name, "concat")
    #plot_rrk_histogram(query_labels, enrolled_labels, sim_concat, dataset_name, "concat")
    #error_rate_per_class(query_labels, enrolled_labels, top_idx, dataset_enrolled, embedding_library.query_scan_ids, sim_concat, dataset_name, "_concat")
    #all_metrics["emb_dist_concat"] = analyze_embedding_distribution(sim_concat, query_labels, enrolled_labels, dataset_name, "concat", plot=True)
    all_metrics["metrics_concat"] = metrics_concat
    del sim_concat, top_idx, y_true_concat, y_pred_concat

    # --------- Concat Mean ---------
    metrics_concat_mean, similarity_matrix_concat_mean, top_indices_concat_mean, y_true_concat_mean, y_pred_concat_mean = concat(embedding_library, disable_bar, reduce_with="mean")
    #plot_cmc(similarity_matrix_concat_mean, enrolled_labels, query_labels, dataset_name, "concat_mean")
    #plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat_mean, dataset_name, "concat_mean")
    #all_metrics["emb_dist_concat_mean"] = analyze_embedding_distribution(similarity_matrix_concat_mean, query_labels, enrolled_labels, dataset_name, "concat_mean", plot=True)
    all_metrics["metrics_concat_mean"] = metrics_concat_mean
    del similarity_matrix_concat_mean, top_indices_concat_mean, y_true_concat_mean, y_pred_concat_mean

    # --------- Concat Median ---------
    metrics_concat_median, similarity_matrix_concat_median, top_indices_concat_median, y_true_concat_median, y_pred_concat_median = concat(embedding_library, disable_bar, reduce_with="median")
    #plot_cmc(similarity_matrix_concat_median, enrolled_labels, query_labels, dataset_name, "concat_median")
    #plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat_median, dataset_name, "concat_median")
    #all_metrics["emb_dist_concat_median"] = analyze_embedding_distribution(similarity_matrix_concat_median, query_labels, enrolled_labels, dataset_name, "concat_median", plot=True)
    all_metrics["metrics_concat_median"] = metrics_concat_median
    del similarity_matrix_concat_median, top_indices_concat_median, y_true_concat_median, y_pred_concat_median

    # --------- Concat PCA ---------
    #metrics_concat_pca, similarity_matrix_concat_pca, top_indices_concat_pca, y_true_concat_pca, y_pred_concat_pca = concat(embedding_library, disable_bar, reduce_with="pca")
    #plot_cmc(similarity_matrix_concat_pca, enrolled_labels, query_labels, dataset_name, "concat_pca")
    #plot_rrk_histogram(query_labels, enrolled_labels, similarity_matrix_concat_pca, dataset_name, "concat_pca")
    #all_metrics["metrics_concat_pca"] = metrics_concat_pca
    #del similarity_matrix_concat_pca, top_indices_concat_pca, y_true_concat_pca, y_pred_concat_pca

    # --------- Score fusion ---------
    fusion_methods = ["max", "product", "majority", "mean", "median"]
    sim_score = None
    for m in fusion_methods:
        metrics, sim_score, fused, top_idx, pred = score_fusion(embedding_library, disable_bar, method=m, similarity_matrix=sim_score, distance_matrix=None)
        all_metrics[f"metrics_score_{m}"] = metrics
        #all_metrics[f"emb_dist_score_{m}"] = analyze_embedding_distribution(fused, query_labels, enrolled_labels, dataset_name, f"score_{m}", plot=True)

    #plot_all_cmc_from_txt(dataset_name)

    return all_metrics, embedding_library, dataset_enrolled, dataset_query


def print_results(neutral_dataset, dataset_enrolled, dataset_query, all_metrics):


    rank_1_front = smart_round(all_metrics["metrics_front"].get('Rank-1 Rate', 'N/A'))
    rank_5_front = smart_round(all_metrics["metrics_front"].get('Rank-5 Rate', 'N/A'))
    mrr_front = smart_round(all_metrics["metrics_front"].get('MRR', 'N/A'))
    gbig_front = smart_round(all_metrics["emb_dist_front"].get('gbig', 'N/A')*100)

    rank_1_concat = smart_round(all_metrics["metrics_concat"].get('Rank-1 Rate', 'N/A'))
    rank_5_concat = smart_round(all_metrics["metrics_concat"].get('Rank-5 Rate', 'N/A'))
    mrr_concat = smart_round(all_metrics["metrics_concat"].get('MRR', 'N/A'))
    gbig_concat = smart_round(all_metrics["emb_dist_concat"].get('gbig', 'N/A')*100)

    rank_1_concat_mean = smart_round(all_metrics["metrics_concat_mean"].get('Rank-1 Rate', 'N/A'))
    rank_5_concat_mean = smart_round(all_metrics["metrics_concat_mean"].get('Rank-5 Rate', 'N/A'))
    mrr_concat_mean = smart_round(all_metrics["metrics_concat_mean"].get('MRR', 'N/A'))
    gbig_concat_mean = smart_round(all_metrics["emb_dist_concat_mean"].get('gbig', 'N/A')*100)

    rank_1_concat_median = smart_round(all_metrics["metrics_concat_median"].get('Rank-1 Rate', 'N/A'))
    rank_5_concat_median = smart_round(all_metrics["metrics_concat_median"].get('Rank-5 Rate', 'N/A'))
    mrr_concat_median = smart_round(all_metrics["metrics_concat_median"].get('MRR', 'N/A'))

    #rank_1_concat_pca = smart_round(all_metrics["metrics_concat_pca"].get('Rank-1 Rate', 'N/A'))
    #rank_5_concat_pca = smart_round(all_metrics["metrics_concat_pca"].get('Rank-5 Rate', 'N/A'))
    #mrr_concat_pca = smart_round(all_metrics["metrics_concat_pca"].get('MRR', 'N/A'))

    mrr_score_max = smart_round(all_metrics["metrics_score_max"].get('MRR', 'N/A'))
    gbig_score_max = smart_round(all_metrics["emb_dist_score_max"].get('gbig', 'N/A')*100)

    mrr_score_prod = smart_round(all_metrics["metrics_score_product"].get('MRR', 'N/A'))
    gbig_score_prod = smart_round(all_metrics["emb_dist_score_product"].get('gbig', 'N/A')*100)

    mrr_score_mean = smart_round(all_metrics["metrics_score_mean"].get('MRR', 'N/A'))
    gbig_score_mean = smart_round(all_metrics["emb_dist_score_mean"].get('gbig', 'N/A')*100)

    mrr_score_majority = smart_round(all_metrics["metrics_score_majority"].get('MRR', 'N/A'))
    gbig_score_majority = smart_round(all_metrics["emb_dist_score_majority"].get('gbig', 'N/A')*100)

    # mrr_score_pdw = smart_round(all_metrics["metrics_score_pdw"].get('MRR', 'N/A'))
    string = (
        colorstr('bright_green', f"{neutral_dataset} E{len(dataset_enrolled)}Q{len(dataset_query)}: ") +
        f"{bold('Front RR1')}: {rank_1_front} {bold('MRR')}: {underscore(mrr_front)} {bold('GBIG')}: {underscore(gbig_front)} | "# {bold('GAIG')}: {underscore(gaig_front)} | "
        f"{bold('Concat RR1')}: {rank_1_concat} {bold('MRR')}: {underscore(mrr_concat)} {bold('GBIG')}: {underscore(gbig_concat)} | "# {bold('GAIG')}: {underscore(gaig_concat)} | "
        f"{bold('Concat_Mean RR1')}: {rank_1_concat_mean} {bold('MRR')}: {underscore(mrr_concat_mean)} {bold('GBIG')}: {underscore(gbig_concat_mean)} | "
        f"{bold('Concat_Median RR1')}: {rank_1_concat_median} {bold('MRR')}: {underscore(mrr_concat_median)} | "
        #f"{bold('Concat_PCA RR1')}: {rank_1_concat_pca} {bold('MRR')}: {underscore(mrr_concat_pca)} | "
        f"{bold('Score_prod MRR')}: {underscore(mrr_score_prod)} {bold('GBIG')}: {underscore(gbig_score_prod)} | "
        f"{bold('Score_mean MRR')}: {underscore(mrr_score_mean)} {bold('GBIG')}: {underscore(gbig_score_mean)} | "
        f"{bold('Score_max MRR')}: {underscore(mrr_score_max)} {bold('GBIG')}: {underscore(gbig_score_max)} | "
        f"{bold('Score_maj MRR')}: {underscore(mrr_score_majority)} {bold('GBIG')}: {underscore(gbig_score_majority)} | "
        # f"{bold('Score_pdw MRR')}: {underscore(mrr_score_pdw)} | "
        #f"{bold('MV RR1')}: {rank_1_mv} {bold('MRR')}: {underscore(mrr_mv)} {bold('GBIG')}: {underscore(gbig_mv)}"
    )
    print(string)


def evaluate_and_log_mv(data_root, dataset, batch_size, disable_bar: bool):


    print(colorstr('bright_green', f"Perform 1:N Evaluation on {dataset}"))
    all_metrics, embedding_library, dataset_enrolled, dataset_query = evaluate_mv_emb_1_n(os.path.join(data_root, dataset), batch_size, disable_bar)

    neutral_dataset = next((dataset[len(p):] for p in ['depth_', 'rgbd_', 'rgb_', 'test_'] if dataset.startswith(p)), dataset)

    print_results(neutral_dataset, dataset_enrolled, dataset_query, all_metrics)

    return all_metrics


def eval_loop(data_root, batch_size, test_set):
    evaluate_and_log_mv(data_root, test_set, batch_size * 4, disable_bar=True)


def main(cfg):
    SEED = cfg['SEED']
    torch.manual_seed(SEED)

    DATA_ROOT = "/home/gustav/dataset11-bff/"  # the parent root where the datasets are stored
    BATCH_SIZE = cfg['BATCH_SIZE']  # Batch size in training

    # ======= Validation =======
    eval_loop(DATA_ROOT, BATCH_SIZE, cfg_copy['TEST_SET'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file', default='config_exp_X.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        cfg_yaml = yaml.safe_load(file)

    # Resolve root folder
    base_data_root = "/home/gustav/dataset11-bff/"

    # find all subdirectories
    subfolders = [f for f in os.listdir(base_data_root) if os.path.isdir(os.path.join(base_data_root, f))]

    if not subfolders:
        print(f"No subfolders found in {base_data_root}. Exiting.")
        exit()

    print(f"Found {len(subfolders)} datasets: {subfolders}")
    print("=" * 80)

    # Loop through each folder and run main(cfg) for each
    for folder in subfolders:
        print(f"\n🚀 Running for dataset: {folder}")

        num_views_cfg = len(folder.replace("test_rgb_bff_crop_new_", "").split())

        cfg_copy = dict(cfg_yaml)
        cfg_copy['TEST_SET'] = folder
        cfg_copy['NUM_VIEWS'] = num_views_cfg

        main(cfg_copy)

    print("\n✅ All dataset runs finished.")
