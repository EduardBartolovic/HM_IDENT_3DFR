import os

import numpy as np
import time
import torch
import torchvision
from torchvision.transforms import transforms
from src.util.EmbeddingsUtils import build_embedding_library, batched_distances_gpu
from src.util.Metrics import calc_metrics


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5


def evaluate_triplet_model(DEVICE, model, library_loader, val_loader):

    embeddings_train, embeddings_train_labels = build_embedding_library(DEVICE, model, library_loader)

    # Compute mean embeddings for each label
    labels_database = np.unique(embeddings_train_labels)
    embedding_train_mean = np.zeros((len(labels_database), embeddings_train.shape[1]))
    for i, label in enumerate(labels_database):
        mask = (embeddings_train_labels == label)
        mean_embedding = np.mean(embeddings_train[mask], axis=0)
        embedding_train_mean[i] = mean_embedding

    embeddings_val, embeddings_val_labels = build_embedding_library(DEVICE, model, val_loader)

    # Calculate distances between embeddings of validation and training data
    distances = batched_distances_gpu(DEVICE, embeddings_val, embedding_train_mean)

    # Find top 5 indices/classes of the closest vectors for each validation embedding
    y_pred_top5 = np.argsort(distances, axis=1)[:, :5]

    return embeddings_train, embeddings_train_labels, embeddings_val, embeddings_val_labels, y_pred_top5


def load_data(data_dir, transform, batch_size: int) -> (
        torchvision.datasets.ImageFolder, torch.utils.data.dataloader.DataLoader):
    """Load dataset from the specified directory"""
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6, drop_last = True)
    return dataset, data_loader


def eval_model(DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, test_path):

    INPUT_SIZE = [112,112]
    test_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.CenterCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5],
                             std = [0.5, 0.5, 0.5]),
    ])

    dataset_test_database = os.path.join(test_path, 'train')
    dataset_test_val = os.path.join(test_path, 'validation')
    test_database_dataset, test_database_loader = load_data(dataset_test_database, test_transform, BATCH_SIZE)
    test_val_dataset, test_val_loader = load_data(dataset_test_val, test_transform, BATCH_SIZE)

    time.sleep(0.5)
    embeddings_train, embeddings_train_labels, embeddings_val, embeddings_val_labels, y_pred_top5 = evaluate_triplet_model(DEVICE, BACKBONE, test_database_loader, test_val_loader)
    y_pred_top1 = y_pred_top5[:, 0]
    #plot_confusion_matrix(self.output_path, embeddings_val_labels, y_pred_top1, td, ('_test_'+test_name), matplotlib=False)
    acc_t, prec_t, rec_t, f1_t, r1r_t, acc5_t = calc_metrics(embeddings_val_labels, y_pred_top1, y_pred_top5, prints=True)
    return acc_t, acc5_t