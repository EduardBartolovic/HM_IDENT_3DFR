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



if __name__ == '__main__':

    # ---- CONFIG ----
    BATCH_SIZE = 128
    NUM_WORKERS = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_FILE = "headpose.txt"

    # ---- TRANSFORM ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    model_path_hpe = r"C:\Users\Eduard\Desktop\Face\HM_IDENT_3DFR\src\preprocess_datasets\headPoseEstimation\weights\resnet50.pt"#"/home/gustav/HM_IDENT_3DFR/src/preprocess_datasets/headPoseEstimation/weights/resnet50.pt"

    # Load HPE model
    head_pose_model = get_model("resnet50", num_classes=6)
    state_dict = torch.load(model_path_hpe, map_location=DEVICE, weights_only=True)
    head_pose_model.load_state_dict(state_dict)
    head_pose_model.to(DEVICE)
    head_pose_model.eval()

    dataset = MS1MV3Folder("C:\\Users\\Eduard\\Downloads\\ms1mv3", transform=transform)

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

    yaw_values = []

    try:
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # format: path yaw pitch roll (based on user's print order)
                    yaw = float(parts[1])
                    yaw_values.append(yaw)
    except FileNotFoundError:
        yaw_values = None

    yaw_values = np.array(yaw_values)

    left = np.sum(yaw_values < -20)
    middle = np.sum((yaw_values >= -20) & (yaw_values <= 20))
    right = np.sum(yaw_values > 20)

    total = len(yaw_values)

    left_pct = 100 * left / total
    middle_pct = 100 * middle / total
    right_pct = 100 * right / total

    print(f"< -20°: {left} ({left_pct:.2f}%)")
    print(f"-20° to 20°: {middle} ({middle_pct:.2f}%)")
    print(f"> 20°: {right} ({right_pct:.2f}%)")

    plt.figure()
    plt.hist(yaw_values, bins=50)
    plt.title("Yaw distribution")
    plt.xlabel("Yaw (degrees)")
    plt.ylabel("Count")

    plt.axvline(-20, color='red', linestyle='--')
    plt.axvline(20, color='red', linestyle='--')

    # add text box
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

    plt.savefig("distribution.jpg")