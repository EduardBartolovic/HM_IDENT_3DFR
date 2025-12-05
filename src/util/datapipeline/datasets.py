import os
import cv2
import random
import numpy as np
from scipy import io
from PIL import Image, ImageFilter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.preprocess_datasets.headPoseEstimation.utils.general import get_rotation_matrix


def load_filenames(root_dir):
    filenames = []
    removed_count = 0

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg'):
                mat_path = os.path.join(root, file.replace('.jpg', '.mat'))
                label = io.loadmat(mat_path)
                pitch, yaw, roll = label['Pose_Para'][0][:3]

                # Convert radians to degrees
                pitch *= 180 / np.pi
                yaw *= 180 / np.pi
                roll *= 180 / np.pi

                # Only add the file if the conditions are met
                limit = 25 # 99
                if abs(pitch) <= limit and abs(yaw) <= limit and abs(roll) <= limit:
                    filenames.append(os.path.join(root, file[:-4]))
                else:
                    removed_count += 1

    return filenames, removed_count

class AFLW2000(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.filenames, removed_items = load_filenames(root)
        print(f"AFLW200: {removed_items} items removed from dataset that have an angle > 99 degrees. Loaded {len(self)} files.")

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        img_path = f"{filename}.jpg"
        mat_path = f"{filename}.mat"

        image = Image.open(img_path).convert("RGB")
        lbl = io.loadmat(mat_path)
        pt2d = lbl['pt2d']
        pitch, yaw, roll = lbl['Pose_Para'][0][:3]

        x_min, x_max = min(pt2d[0, :]), max(pt2d[0, :])
        y_min, y_max = min(pt2d[1, :]), max(pt2d[1, :])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)

        x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))
        image = image.crop((x_min, y_min, x_max, y_max))

        rotation_matrix = get_rotation_matrix(pitch, yaw, roll)
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)

        labels = torch.tensor([pitch, yaw, roll], dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        return image, rotation_matrix, labels, filename

    def __len__(self):
        return len(self.filenames)


def load_filenames_npz(root_dir):
    filenames = []
    removed_count = 0

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.npz'):
                mat_path = os.path.join(root, file.replace('.npz', '.mat'))
                label = io.loadmat(mat_path)
                pitch, yaw, roll = label['Pose_Para'][0][:3]

                # Convert radians to degrees
                pitch *= 180 / np.pi
                yaw *= 180 / np.pi
                roll *= 180 / np.pi

                # Only add the file if the conditions are met
                if abs(pitch) <= 99 and abs(yaw) <= 99 and abs(roll) <= 99:
                    filenames.append(os.path.join(root, file[:-4]))
                else:
                    removed_count += 1

    return filenames, removed_count

class AFLW2000EMB(Dataset):
    def __init__(self, root):
        self.root = root
        self.filenames, removed_items = load_filenames(root)
        print(f"AFLW200: {removed_items} items removed from dataset that have an angle > 99 degrees. Loaded {len(self)} files.")

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        npz_path = f"{filename}.npz"
        mat_path = f"{filename}.mat"

        emb = np.load(npz_path)["embedding"]
        lbl = io.loadmat(mat_path)
        pitch, yaw, roll = lbl['Pose_Para'][0][:3]

        rotation_matrix = get_rotation_matrix(pitch, yaw, roll)
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)

        labels = torch.tensor([pitch, yaw, roll], dtype=torch.float32)

        return emb, rotation_matrix, labels, filename

    def __len__(self):
        return len(self.filenames)