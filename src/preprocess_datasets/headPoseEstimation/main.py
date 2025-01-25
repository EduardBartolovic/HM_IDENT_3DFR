import itertools
import logging
import os

import numpy as np
import torch

from src.preprocess_datasets.face_correspondences.CalculateFaceCorrespondences import \
    calculate_face_correspondences_dataset
from src.preprocess_datasets.headPoseEstimation.create_test_dataset import create_train_test_split
from src.preprocess_datasets.headPoseEstimation.face_analysis import filter_wrong_faces
from src.preprocess_datasets.headPoseEstimation.headpose_estimation import get_model, headpose_estimation
from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import generate_voxceleb_dataset
from src.preprocess_datasets.headPoseEstimation.match_hpe_angles_to_reference import find_matches

if __name__ == '__main__':
    folder_root = "C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb2_train"
    model_path_hpe = "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    dataset_output_folder = "C:\\Users\\Eduard\\Desktop\\Face\\VoxCeleb2_train_dataset"
    output_test_dataset = "C:\\Users\\Eduard\\Desktop\\Face\\test_VoxCeleb2_train_dataset"
    backbone_face_model = "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\pretrained\\backbone_ir50_asia.pth"

    device = torch.device("cuda")
    try:
        head_pose = get_model("resnet50", num_classes=6)
        state_dict = torch.load(model_path_hpe, map_location=device, weights_only=True)
        head_pose.load_state_dict(state_dict)
        head_pose.to(device)
        head_pose.eval()
        logging.info("Head Pose Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading weights of head pose estimation model: {e}")
        raise Exception()

    print("##################################")
    print("#############HPE##################")
    print("##################################")
    headpose_estimation(folder_root, "hpe", head_pose, device, fix_rotation=True, draw=True)

    print("##################################")
    print("##########Filter wrong faces###########")
    print("##################################")
    filter_wrong_faces(folder_root, "frames_filtered", backbone_face_model, "cuda")

    ref_angles = [-25, -10, 0, 10, 25]
    permutations = np.array([(x, y, 0) for x, y in itertools.product(ref_angles, repeat=2)])
    print(len(permutations))
    print(permutations)
    print("##################################")
    print("###########FINDMATCHES############")
    print("##################################")
    find_matches(folder_root, permutations)

    print("##################################")
    print("###########GEN DATASET############")
    print("##################################")
    generate_voxceleb_dataset(folder_root, "frames_cropped", dataset_output_folder)

    print("##################################")
    print("######face_correspondences########")
    print("##################################")
    calculate_face_correspondences_dataset(dataset_output_folder)

    print("##################################")
    print("###########GEN TEST DATASET############")
    print("for:", dataset_output_folder, "to:", output_test_dataset)
    print("##################################")
    create_train_test_split(dataset_output_folder, output_test_dataset)

    print("##################################")
    print("######face_correspondences########")
    print("##################################")
    calculate_face_correspondences_dataset(os.path.join(output_test_dataset,"train"))
    calculate_face_correspondences_dataset(os.path.join(output_test_dataset,"validation"))
