import os

import numpy as np
import torch
import cv2
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from sixdrepnet.model import SixDRepNet

from src.preprocess_datasets.headPoseEstimation.headpose_estimation import get_model
from src.preprocess_datasets.headPoseEstimation.utils.general import compute_euler_angles_from_rotation_matrices

# ---- CONFIG ----
BATCH_SIZE = 128
NUM_WORKERS = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_FILE = "headpose.txt"

# ---- MODEL ----
model = SixDRepNet()
model.load_state_dict(torch.load("6DRepNet_300W_LP_AFLW2000.pth"))
model.to(DEVICE)
model.eval()

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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



if __name__ == '__main__':

    device = "cuda"
    model_path_hpe = "/home/gustav/HM_IDENT_3DFR/src/preprocess_datasets/headPoseEstimation/weights/resnet50.pt"

    # Load HPE model
    head_pose_model = get_model("resnet50", num_classes=6)
    state_dict = torch.load(model_path_hpe, map_location=device, weights_only=True)
    head_pose_model.load_state_dict(state_dict)
    head_pose_model.to(device)
    head_pose_model.eval()

    dataset = MS1MV3Folder("ms1mv3")

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

                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                for i, path in enumerate(paths):
                    print(f"{path} {y_pred_deg[i]:.4f} {p_pred_deg[i]:.4f} {r_pred_deg[i]:.4f}\n")
                    f.write(f"{path} {y_pred_deg[i]:.4f} {p_pred_deg[i]:.4f} {r_pred_deg[i]:.4f}\n")