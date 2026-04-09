import os
import shutil

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt

from src.preprocess_datasets.headPoseEstimation.headpose_estimation import get_model
from src.preprocess_datasets.headPoseEstimation.utils.general import compute_euler_angles_from_rotation_matrices


class MS1MV3Folder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root)
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)

        if self.transform:
            img = self.transform(img)

        return img, path

def plot_distribution(values, name, filename):
    left = np.sum(values < -20)
    middle = np.sum((values >= -20) & (values <= 20))
    right = np.sum(values > 20)

    total = len(values)

    left_pct = 100 * left / total
    middle_pct = 100 * middle / total
    right_pct = 100 * right / total

    print(f"{name} < -20°: {left} ({left_pct:.2f}%)")
    print(f"{name} -20° to 20°: {middle} ({middle_pct:.2f}%)")
    print(f"{name} > 20°: {right} ({right_pct:.2f}%)")

    plt.figure()
    plt.hist(values, bins=50)
    plt.title(f"{name} distribution")
    plt.xlabel(f"{name} (degrees)")
    plt.ylabel("Count")

    plt.axvline(-20, linestyle='--')
    plt.axvline(20, linestyle='--')

    text = (
        f"< -20°: {left} ({left_pct:.1f}%)\n"
        f"-20° to 20°: {middle} ({middle_pct:.1f}%)\n"
        f"> 20°: {right} ({right_pct:.1f}%)"
    )

    plt.gca().text(
        0.02, 0.95, text,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle="round", alpha=0.6)
    )

    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':

    # ---- CONFIG ----
    BATCH_SIZE = 1536
    NUM_WORKERS = 12
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_FILE = "headpose.txt"

    # ---- TRANSFORM ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    model_path_hpe = r"C:\Users\Eduard\Desktop\Face\HM_IDENT_3DFR\src\preprocess_datasets\headPoseEstimation\weights\resnet50.pt" # "/home/gustav/HM_IDENT_3DFR/src/preprocess_datasets/headPoseEstimation/weights/resnet50.pt"
    ms1mv3_folder = r"C:\Users\Eduard\Downloads\ms1mv3" # "/home/gustav/ms1mv3/ms1mv3/" #

    # Load HPE model
    head_pose_model = get_model("resnet50", num_classes=6)
    state_dict = torch.load(model_path_hpe, map_location=DEVICE, weights_only=True)
    head_pose_model.load_state_dict(state_dict)
    head_pose_model.to(DEVICE)
    head_pose_model.eval()

    dataset = MS1MV3Folder(ms1mv3_folder, transform=transform)
    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,
                        shuffle=False,
                        pin_memory=True)

    with open(OUTPUT_FILE, "w") as f:
        with torch.no_grad():
            for imgs, paths in tqdm(loader):
                imgs = imgs.to(DEVICE)

                R_pred = head_pose_model(imgs)
                euler = compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi

                p_pred_deg = -euler[:, 0].cpu()
                y_pred_deg = -euler[:, 1].cpu()
                r_pred_deg = -euler[:, 2].cpu()

                for i, path in enumerate(paths):
                    #print(f"{path} {p_pred_deg[i]:.4f} {y_pred_deg[i]:.4f} {r_pred_deg[i]:.4f}\n")
                    f.write(f"{path} {p_pred_deg[i]:.4f} {y_pred_deg[i]:.4f} {r_pred_deg[i]:.4f}\n")

    yaw_values = []
    pitch_values = []
    roll_values = []

    try:
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    pitch = float(parts[1])
                    yaw = float(parts[2])
                    roll = float(parts[3])

                    pitch_values.append(pitch)
                    yaw_values.append(yaw)
                    roll_values.append(roll)
    except FileNotFoundError:
        yaw_values = None

    pitch_values = np.array(pitch_values)
    yaw_values = np.array(yaw_values)
    roll_values = np.array(roll_values)

    plot_distribution(yaw_values, "Yaw", "yaw_distribution.jpg")
    plot_distribution(pitch_values, "Pitch", "pitch_distribution.jpg")
    plot_distribution(roll_values, "Roll", "roll_distribution.jpg")

   #  ---------------- Create Subsets ----------------
    LEFT_DIR = "MS1MV3_left"
    FRONTAL_DIR = "MS1MV3_frontal"
    RIGHT_DIR = "MS1MV3_right"

    os.makedirs(LEFT_DIR, exist_ok=True)
    os.makedirs(FRONTAL_DIR, exist_ok=True)
    os.makedirs(RIGHT_DIR, exist_ok=True)

    with open(OUTPUT_FILE, "r") as f:
        for line in tqdm(f, total=len(yaw_values)):
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            path = parts[0]
            pitch = float(parts[1])
            yaw = float(parts[2])
            roll = float(parts[3])

            # Decide target folder
            if yaw < -20:
                target_dir = LEFT_DIR
            elif yaw > 20:
                target_dir = RIGHT_DIR
            else:
                target_dir = FRONTAL_DIR

            class_folder = os.path.basename(os.path.dirname(path))
            target_class_dir = os.path.join(target_dir, class_folder)
            os.makedirs(target_class_dir, exist_ok=True)
            filename = os.path.basename(path)
            dst_path = os.path.join(target_class_dir, filename)

            try:
                shutil.copy2(path, dst_path)
            except Exception as e:
                print(f"Error copying {path}: {e}")