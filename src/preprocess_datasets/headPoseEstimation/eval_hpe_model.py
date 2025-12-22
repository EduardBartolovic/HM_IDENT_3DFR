import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from src.preprocess_datasets.headPoseEstimation.headpose_estimation import get_model
from src.preprocess_datasets.headPoseEstimation.utils.general import compute_euler_angles_from_rotation_matrices, \
    draw_axis
from src.util.datapipeline.HPEDataset import CoordImageDataset


def evaluate_headpose_mae(
        model,
        dataloader,
        device,
):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, (pitch, yaw), path in tqdm(dataloader):

            inputs = inputs.to(device)

            rotation_matrices = model(inputs)

            eulers = compute_euler_angles_from_rotation_matrices(rotation_matrices).detach().cpu().numpy()
            eulers_deg = np.degrees(eulers)
            pred_angles = eulers_deg[:, [2, 1]]  # Ignore roll

            #img = cv2.imread(path[0])
            #draw_axis(
            #    img,
            #    eulers_deg[0,0],
            #    eulers_deg[0,1],
            #    eulers_deg[0,2],
            #    bbox=[0, 0, 256, 256],
            #    size_ratio=0.5
            #)

            # Draw ground truth
            #gt_pitch = -pitch[0].item()
            #gt_yaw = -yaw[0].item()
            #gt_roll = 0.0  # usually unknown / ignored
            #draw_axis(
            #    img,
            #    gt_yaw,
            #    gt_pitch,
            #    gt_roll,
            #    bbox=[0, 0, 100, 100],
            #    size_ratio=0.45,  # slightly smaller so both are visible
            #)
            #cv2.imshow('image', img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            all_preds.append(pred_angles)
            all_targets.append(np.stack([-pitch.numpy(), -yaw.numpy()], axis=1))

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute MAE per angle
    mae_pitch = np.mean(np.abs(all_preds[:, 0] - all_targets[:, 0]))
    mae_yaw = np.mean(np.abs(all_preds[:, 1] - all_targets[:, 1]))

    # Optional: average MAE over both angles
    mae_mean = np.mean(np.abs(all_preds - all_targets))

    print(f"MAE Pitch: {mae_pitch:.4f}°")
    print(f"MAE Yaw: {mae_yaw:.4f}°")
    print(f"MAE Overall: {mae_mean:.4f}°")


if __name__ == '__main__':
    input_folder = "F:\\Face\\data\\dataset14\\rgb_bff_crop261" #rgb_tmp261"# # Folder containing original preprocessed files
    #input_folder = "F:\\Face\\data\\dataset14\\rgb_tmp261"  # # Folder containing original preprocessed files
    model_path_hpe = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"

    device = torch.device("cuda")
    head_pose_model = get_model("resnet50", num_classes=6)
    state_dict = torch.load(model_path_hpe, map_location=device, weights_only=True)
    head_pose_model.load_state_dict(state_dict)
    head_pose_model.to(device)
    head_pose_model.eval()

    train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset_train = CoordImageDataset(input_folder, train_transform)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, pin_memory=False, num_workers=8, drop_last=False)

    evaluate_headpose_mae(
        model=head_pose_model,
        dataloader=train_loader,
        device=device,
    )
