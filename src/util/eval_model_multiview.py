import os
from collections import namedtuple

import mlflow
import numpy as np
import time
import torch
import torchvision
from tqdm import tqdm

from src.util.EmbeddingsUtils import batched_distances_gpu
from src.util.datapipeline.EmbeddingDataset import EmbeddingDataset
from src.util.Metrics import calc_metrics, error_rate_per_class
from src.util.Plotter import plot_confusion_matrix
from src.util.embeddungs_metrics import calc_embedding_analysis
from src.util.misc import colorstr


@torch.no_grad()
def get_embeddings(device, model, enrolled_loader, query_loader):

    model.eval()

    enrolled_embeddings = []
    enrolled_labels = []
    for inputs, labels in tqdm(iter(enrolled_loader)):
        inputs = inputs.to(device)
        embeddings = model(inputs).cpu().numpy()
        enrolled_embeddings.extend(embeddings)
        enrolled_labels.extend(labels)

    query_embeddings = []
    query_labels = []
    for inputs, labels in tqdm(iter(query_loader)):
        inputs = inputs.to(device)
        embeddings = model(inputs).cpu().numpy()
        query_embeddings.extend(embeddings)
        query_labels.extend(labels)

    Results = namedtuple("Results", ["enrolled_embeddings", "enrolled_labels", "query_embeddings", "query_labels"])
    return Results(enrolled_embeddings, enrolled_labels, np.array(query_embeddings), query_labels)


def load_data(data_dir, max_batch_size: int) -> (torchvision.datasets.ImageFolder, torch.utils.data.dataloader.DataLoader):

    dataset = EmbeddingDataset(data_dir)

    dataset_size = len(dataset)
    # Ensure the last batch is always larger than 1
    batch_size = max_batch_size
    while (dataset_size % batch_size == 1) and (batch_size > 2):
        batch_size -= 1

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
    return dataset, data_loader


def evaluate(device, batch_size, backbone, test_path, distance_metric):
    """
    Evaluate 1:N Model Performance on given test dataset
    """
    dataset_enrolled_path = os.path.join(test_path, 'train')
    dataset_query_path = os.path.join(test_path, 'validation')
    dataset_enrolled, enrolled_loader = load_data(dataset_enrolled_path, batch_size)
    _, query_loader = load_data(dataset_query_path, batch_size)

    time.sleep(0.1)

    embedding_library = get_embeddings(device, backbone, enrolled_loader, query_loader)

    # Calculate distances between embeddings of query and library data
    distances = batched_distances_gpu(device, embedding_library.query_embeddings, embedding_library.enrolled_embeddings, batch_size, distance_metric=distance_metric)
    # Sort indices/classes of the closest vectors for each query embedding
    y_pred = np.argsort(distances, axis=1)
    y_pred_top1 = y_pred[:, 0]
    # TODO: y_pred_top5 = y_pred[:, :5]
    metrics = calc_metrics(embedding_library.query_labels, y_pred_top1)
    # TODO:
    # plot_confusion_matrix(embedding_library.query_labels, y_pred_top1, dataset_enrolled, os.path.basename(test_path), matplotlib=False)
    # error_rate_per_class(embedding_library.query_labels, y_pred_top1, os.path.basename(test_path))

    return metrics, embedding_library


def evaluate_and_log_multiview(device, backbone, data_root, dataset, epoch, distance_metric, batch_size):

    print(colorstr('bright_green', f"Perform 1:N Evaluation on {dataset}"))
    metrics, embedding_library = evaluate(device, batch_size*2, backbone, os.path.join(data_root, dataset), distance_metric)

    neutral_dataset = dataset.replace('depth_', '').replace('rgbd_', '').replace('rgb_', '').replace('test_', '')

    mlflow.log_metric(f"{neutral_dataset}_RR1", metrics['Rank-1 Rate'], step=epoch + 1)
    # TODO: mlflow.log_metric(f'{neutral_dataset}_RR5', metrics['Rank-5 Rate'], step=epoch + 1)

    #if 'bellus' in dataset:
    #    write_embeddings(embedding_library, neutral_dataset, epoch + 1)

    rank_1 = metrics.get('Rank-1 Rate', 'N/A')
    # TODO: rank_5 = metrics.get('Rank-5 Rate', 'N/A')

    print(colorstr(
        'bright_green',
        f"{neutral_dataset} Evaluation: "
        f"RR1: {rank_1} " # TODO: RR5: {rank_5} "
    ))
