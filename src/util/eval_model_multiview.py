import os
from collections import namedtuple
from copy import deepcopy

import mlflow
import numpy as np
import time
import torch
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm

from src.backbone.model_multiview_irse import execute_model
from src.util.Metrics import error_rate_per_class
from src.util.Plotter import plot_confusion_matrix
from src.util.Voting import calculate_embedding_similarity, compute_ranking_matrices, analyze_result
from src.util.datapipeline.EmbeddingDataset import EmbeddingDataset
from src.util.datapipeline.MultiviewDataset import MultiviewDataset
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


@torch.no_grad()
def get_embeddings_mvs(device, backbone_reg, backbone_agg, aggregators, enrolled_loader, query_loader):

    backbone_reg.eval()
    backbone_agg.eval()
    [i.eval() for i in aggregators]

    enrolled_embeddings = []
    enrolled_labels = []
    enrolled_perspectives = []
    use_face_corr = False
    for inputs, labels, perspectives, face_corr in tqdm(iter(enrolled_loader), desc="Generate Enrolled Embeddings"):

        if not use_face_corr and face_corr.shape[1] > 0:
            print("Using Feature Alignment")
            use_face_corr = True

        embeddings = execute_model(device, backbone_reg, backbone_agg, aggregators, inputs, perspectives, face_corr, use_face_corr).cpu().numpy()
        enrolled_embeddings.extend(embeddings)
        enrolled_labels.extend(deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        enrolled_perspectives.append(np.array(perspectives).T)

    query_embeddings = []
    query_labels = []
    query_perspectives= []
    for inputs, labels, perspectives, face_corr in tqdm(iter(query_loader), desc="Generate Query Embeddings"):
        embeddings = execute_model(device, backbone_reg, backbone_agg, aggregators, inputs, perspectives, face_corr, use_face_corr).cpu().numpy()
        query_embeddings.extend(embeddings)
        query_labels.extend(deepcopy(labels))  # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
        query_perspectives.append(np.array(perspectives).T)

    enrolled_embeddings = np.array(enrolled_embeddings)
    enrolled_labels = np.array([t.item() for t in enrolled_labels])
    enrolled_perspectives = np.concatenate(enrolled_perspectives, axis=0)
    query_embeddings = np.array(query_embeddings)
    query_labels = np.array([t.item() for t in query_labels])
    query_perspectives = np.concatenate(query_perspectives, axis=0)

    Results = namedtuple("Results", ["enrolled_embeddings", "enrolled_labels", "enrolled_perspectives", "query_embeddings", "query_labels", "query_perspectives"])
    return Results(enrolled_embeddings, enrolled_labels, enrolled_perspectives, query_embeddings, query_labels, query_perspectives)


def load_data(data_dir, max_batch_size: int) -> (torchvision.datasets.ImageFolder, torch.utils.data.dataloader.DataLoader):

    dataset = EmbeddingDataset(data_dir)

    dataset_size = len(dataset)
    # Ensure the last batch is always larger than 1
    batch_size = max_batch_size
    while (dataset_size % batch_size == 1) and (batch_size > 2):
        batch_size -= 1

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
    return dataset, data_loader


def evaluate(device, batch_size, backbone, test_path, distance_metric, disable_bar):
    """
    Evaluate 1:N Model Performance on given test dataset
    """
    dataset_enrolled_path = os.path.join(test_path, 'train')
    dataset_query_path = os.path.join(test_path, 'validation')
    dataset_enrolled, enrolled_loader = load_data(dataset_enrolled_path, batch_size)
    _, query_loader = load_data(dataset_query_path, batch_size)

    time.sleep(0.1)

    embedding_library = get_embeddings(device, backbone, enrolled_loader, query_loader)

    enrolled_embedding = np.array(embedding_library.enrolled_embeddings)
    enrolled_label = np.array(embedding_library.enrolled_labels)

    query_embedding = np.array(embedding_library.query_embeddings)
    query_label = np.array(embedding_library.query_labels)

    similarity_matrix = calculate_embedding_similarity(query_embedding, enrolled_embedding, disable_bar=disable_bar)
    top_indices, top_values = compute_ranking_matrices(similarity_matrix)
    result_metrics = analyze_result(similarity_matrix, top_indices, enrolled_label, query_label, top_k_acc_k=5)

    # metrics = calc_metrics(embedding_library.query_labels, top_indices[:, 0])
    # plot_confusion_matrix(embedding_library.query_labels, top_indices[:, 0], dataset_enrolled, os.path.basename(test_path), matplotlib=False)
    # error_rate_per_class(embedding_library.query_labels, top_indices[:, 0], os.path.basename(test_path))

    return result_metrics, embedding_library


def evaluate_and_log_multiview(device, backbone, data_root, dataset, epoch, distance_metric, batch_size):

    print(colorstr('bright_green', f"Perform 1:N Evaluation on {dataset}"))
    metrics, embedding_library = evaluate(device, batch_size*2, backbone, os.path.join(data_root, dataset), distance_metric)

    neutral_dataset = dataset.replace('depth_', '').replace('rgbd_', '').replace('rgb_', '').replace('test_', '')

    mlflow.log_metric(f'{neutral_dataset}_RR5', metrics['Rank-5 Rate'], step=epoch + 1)

    #if 'bellus' in dataset:
    #    write_embeddings(embedding_library, neutral_dataset, epoch + 1)

    rank_1 = metrics.get('Rank-1 Rate', 'N/A')
    rank_5 = metrics.get('Rank-5 Rate', 'N/A')

    print(colorstr(
        'bright_green',
        f"{neutral_dataset} Evaluation: "
        f"RR1: {rank_1} "
        f"RR5: {rank_5} "
    ))


# MVS for reg and agg
def load_data_mvs(data_dir, max_batch_size: int, test_transform, use_face_corr: bool) -> (torchvision.datasets.ImageFolder, torch.utils.data.dataloader.DataLoader):

    dataset = MultiviewDataset(data_dir, num_views=25, transform=test_transform, use_face_corr=use_face_corr)

    dataset_size = len(dataset)
    # Ensure the last batch is always larger than 1
    batch_size = max_batch_size
    while (dataset_size % batch_size == 1) and (batch_size > 2):
        batch_size -= 1

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
    return dataset, data_loader


def evaluate_mvs(device, backbone_reg, backbone_agg, aggregators, test_path, test_transform, batch_size, use_face_corr: bool, disable_bar: bool):
    """
    Evaluate 1:N Model Performance on given test dataset
    """
    dataset_enrolled_path = os.path.join(test_path, 'train')
    dataset_query_path = os.path.join(test_path, 'validation')
    dataset_enrolled, enrolled_loader = load_data_mvs(dataset_enrolled_path, batch_size, test_transform, use_face_corr)
    _, query_loader = load_data_mvs(dataset_query_path, batch_size, test_transform, use_face_corr)

    time.sleep(0.1)

    embedding_library = get_embeddings_mvs(device, backbone_reg, backbone_agg, aggregators, enrolled_loader, query_loader)

    enrolled_embedding = embedding_library.enrolled_embeddings
    enrolled_label = embedding_library.enrolled_labels

    query_embedding = embedding_library.query_embeddings
    query_label = embedding_library.query_labels

    similarity_matrix = calculate_embedding_similarity(query_embedding, enrolled_embedding, chunk_size=batch_size, disable_bar=disable_bar)
    top_indices, top_values = compute_ranking_matrices(similarity_matrix)
    result_metrics = analyze_result(similarity_matrix, top_indices, enrolled_label, query_label, top_k_acc_k=5)

    # result_metrics_front = accuracy_front_perspective(embedding_library) #TODO Use correct emebdding

    #metrics = calc_metrics(embedding_library.query_labels, top_indices[:, 0])
    plot_confusion_matrix(embedding_library.query_labels, enrolled_label[top_indices[:, 0]], dataset_enrolled, os.path.basename(test_path), matplotlib=False)
    error_rate_per_class(embedding_library.query_labels, enrolled_label[top_indices[:, 0]], os.path.basename(test_path))

    #metric_concat = concat(embedding_library)  #TODO Use correct emebddin
    #print(metric_concat)

    return result_metrics, embedding_library


def evaluate_and_log_mvs(device, backbone_reg, backbone_agg, aggregators, data_root, dataset, epoch, test_transform_sizes, batch_size, use_face_corr: bool, disable_bar: bool):

    test_transform = transforms.Compose([
        transforms.Resize(test_transform_sizes),
        transforms.CenterCrop([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(colorstr('bright_green', f"Perform 1:N Evaluation on {dataset} with cropping: {test_transform_sizes}"))
    metrics, embedding_library = evaluate_mvs(device, backbone_reg, backbone_agg, aggregators, os.path.join(data_root, dataset), test_transform, batch_size, use_face_corr, disable_bar)

    neutral_dataset = dataset.replace('depth_', '').replace('rgbd_', '').replace('rgb_', '').replace('test_', '')

    mlflow.log_metric(f'{neutral_dataset}_RR1', metrics['Rank-1 Rate'], step=epoch)
    mlflow.log_metric(f'{neutral_dataset}_RR5', metrics['Rank-5 Rate'], step=epoch)
    #mlflow.log_metric(f'{neutral_dataset}_Front_RR1', result_metrics_front['Rank-1 Rate'], step=epoch)
    #mlflow.log_metric(f'{neutral_dataset}_Front_RR5', result_metrics_front['Rank-5 Rate'], step=epoch)

    #if 'bellus' in dataset:
    #    write_embeddings(embedding_library, neutral_dataset, epoch + 1)

    rank_1 = metrics.get('Rank-1 Rate', 'N/A')
    rank_5 = metrics.get('Rank-5 Rate', 'N/A')
    #rank_1_front = result_metrics_front.get('Rank-1 Rate', 'N/A')
    #rank_5_front = result_metrics_front.get('Rank-5 Rate', 'N/A')

    print(colorstr(
        'bright_green',
        f"{neutral_dataset} Evaluation: "
        f"RR1: {rank_1} "
        f"RR5: {rank_5} "
        #f"Front RR1: {rank_1_front} "
        #f"Front RR5: {rank_5_front} "
    ))

