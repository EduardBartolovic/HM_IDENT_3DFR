import os

import mlflow
import numpy as np
import torch
from scipy.spatial.distance import pdist, cdist, cosine
from sklearn.model_selection import KFold
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.util.ImageFolderWithFilename import ImageFolderWithFilename
from src.util.misc import colorstr
from src.util.utils import gen_plot


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_thresholds = len(thresholds)

    tprs = np.zeros(nrof_thresholds)
    fprs = np.zeros(nrof_thresholds)

    dist = np.array([cosine(embeddings1[i], embeddings2[i]) for i in range(len(embeddings1))])

    acc_train = np.zeros(nrof_thresholds)
    for threshold_idx, threshold in enumerate(thresholds):
        _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist, actual_issame)
    best_threshold_index = np.argmax(acc_train)
    best_thresholds = thresholds[best_threshold_index]
    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], _ = calculate_accuracy(threshold, dist, actual_issame)
    _, _, accuracy = calculate_accuracy(thresholds[best_threshold_index], dist, actual_issame)

    return tprs, fprs, accuracy, best_thresholds


def evaluate(embeddings1, embeddings2, actual_issame):
    thresholds = np.arange(0, 1.5, 0.001)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame))
    return tpr, fpr, accuracy, best_thresholds


def evaluate_verification(device, backbone, data_root, dataset_path, writer, epoch, num_epoch, distance_metric,
                          rgb_mean, rgb_std):
    print(colorstr('bright_green', f"Perform 1:1 Evaluation on {dataset_path}"))
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std),
    ])

    dataset = ImageFolderWithFilename(os.path.join(data_root, dataset_path), transform=transform)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    embeddings_storage = {}
    with torch.no_grad():
        for imgs, name, number in tqdm(data_loader, desc="Generate Embeddings"):
            images = imgs.to(device)
            embeddings = backbone(images).cpu()
            for i in range(len(name)):
                embeddings_storage[(name[i], number[i].item())] = embeddings[i]

    pairs = []
    data_root = os.path.expanduser(data_root)
    with open(os.path.join(data_root, dataset_path, 'pairs.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split('\t')

            if len(parts) == 3:
                person = parts[0]
                img1 = int(parts[1])
                img2 = int(parts[2])
                pairs.append(((person, img1), (person, img2), True))
            elif len(parts) == 4:
                person1 = parts[0]
                img1 = int(parts[1])
                person2 = parts[2]
                img2 = int(parts[3])
                pairs.append(((person1, img1), (person2, img2), False))

    embeddings_pairs1 = []
    embeddings_pairs2 = []
    is_same_pairs = []
    for p1, p2, is_same in pairs:
        is_same_pairs.append(is_same)
        embeddings_pairs1.append(embeddings_storage[p1])
        embeddings_pairs2.append(embeddings_storage[p2])

    tpr, fpr, accuracy, best_thresholds = evaluate(np.array(embeddings_pairs1), np.array(embeddings_pairs2),
                                                   np.array(is_same_pairs))
    roc_curve = gen_plot(fpr, tpr)

    print(colorstr('bright_green',
                   f"Epoch {epoch + 1}/{num_epoch}, {dataset_path} Evaluation: Acc: {accuracy}, best_thresholds:{best_thresholds}, tpr: {tpr.mean()}, fpr: {fpr.mean()}"))

    mlflow.log_metric(f"{dataset_path}_Acc", accuracy, step=epoch + 1)
    mlflow.log_figure(roc_curve, "roc_curve.png")

    return accuracy, best_thresholds, roc_curve  # _tensor
