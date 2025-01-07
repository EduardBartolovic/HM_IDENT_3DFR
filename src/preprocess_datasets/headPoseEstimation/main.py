import logging

import numpy as np
import torch

from src.preprocess_datasets.headPoseEstimation.headpose_estimation import get_model, headpose_estimation
from src.preprocess_datasets.headPoseEstimation.hpe_videos_to_dataset import generate_voxceleb_dataset
from src.preprocess_datasets.headPoseEstimation.match_hpe_angles_to_reference import find_matches

if __name__ == '__main__':
    input_folder = "E:\\Download\\face\\VoxCeleb1_test"  # Folder containing original preprocessed files   # --------------------------------------------------------------
    output_folder = "hpe"  # Folder to save cropped videos
    model_path_hpe = "F:\\Face\\HPE\\weights\\resnet50.pt"   # --------------------------------------------------------------

    device = torch.device("cuda")
    try:
        head_pose = get_model("resnet50", num_classes=6)
        state_dict = torch.load(model_path_hpe, map_location=device)
        head_pose.load_state_dict(state_dict)
        logging.info("Head Pose Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading weights of head pose estimation model: {e}")
        raise Exception()

    head_pose.to(device)
    head_pose.eval()

    headpose_estimation(input_folder, output_folder, head_pose, device, fix_rotation=True)

    references = [
        [25, 25, 0],
        [25, 0, 0],
        [25, -25, 0],
        [0, 25, 0],
        [0, 0, 0],
        [0, -25, 0],
        [-25, 25, 0],
        [-25, 0, 0],
        [-25, -25, 0],

    ]  # --------------------------------------------------------------
    references = np.array(references)

    find_matches(input_folder, references)

    dataset_output_folder = "E:\\Download\\face\\VoxCeleb1_test_dataset"  # --------------------------------------------------------------

    generate_voxceleb_dataset(input_folder, dataset_output_folder)
