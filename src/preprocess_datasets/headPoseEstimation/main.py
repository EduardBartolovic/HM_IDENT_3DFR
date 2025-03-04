import itertools
import logging
import os

import numpy as np
import torch

from src.preprocess_datasets.face_correspondences.CalculateFaceCorrespondences import \
    calculate_face_landmarks_dataset, calculate_face_correspondences_dataset
from src.preprocess_datasets.headPoseEstimation.create_test_dataset import create_train_test_split
from src.preprocess_datasets.headPoseEstimation.face_analysis import filter_wrong_faces
from src.preprocess_datasets.headPoseEstimation.headpose_estimation import get_model, headpose_estimation, \
    headpose_estimation_from_video
from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import generate_voxceleb_dataset, \
    generate_voxceleb_dataset_from_video
from src.preprocess_datasets.headPoseEstimation.match_hpe_angles_to_reference import find_matches

if __name__ == '__main__':
    #folder_root = "C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb2_train"
    folder_root = "E:\\Download\\vox2_mp4_6\\dev\\mp4"
    model_path_hpe = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    dataset_output_folder = "E:\\Download\\vox2"
    output_test_dataset = "C:\\Users\\Eduard\\Desktop\\Face\\test_VoxCeleb2_train_dataset"
    backbone_face_model = "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\pretrained\\backbone_ir50_asia.pth"

    #print("##################################")
    #print("##########Filter wrong faces###########")
    #print("##################################")
    #filter_wrong_faces(folder_root, "frames_filtered", backbone_face_model, "cuda")

    device = torch.device("cuda")
    print("##################################")
    print("#############HPE##################")
    print("##################################")
    #headpose_estimation(folder_root, "frames_filtered", "hpe", model_path_hpe, device, fix_rotation=True, draw=True)
    headpose_estimation_from_video(folder_root, "frames_filtered", "hpe", model_path_hpe, device, fix_rotation=True, draw=True)

    ref_angles = [-25, -10, 0, 10, 25]
    permutations = np.array([(x, y, 0) for x, y in itertools.product(ref_angles, repeat=2)])
    print(len(permutations))
    print(permutations)
    print("##################################")
    print("###########FINDMATCHES############")
    print("##################################")
    #find_matches(folder_root, permutations, txt_name="hpe.txt")


    print("##################################")
    print("###########GEN DATASET############")
    print("##################################")
    generate_voxceleb_dataset_from_video(folder_root, "frames_cropped", dataset_output_folder)
    print("######face_correspondences########")

    calculate_face_landmarks_dataset(dataset_output_folder)
    calculate_face_correspondences_dataset(dataset_output_folder)

    print("##################################")
    print("###########GEN TEST DATASET############")
    print("for:", dataset_output_folder, "to:", output_test_dataset)
    print("##################################")
    create_train_test_split(dataset_output_folder, output_test_dataset)
    print("######face_correspondences########")
    calculate_face_landmarks_dataset(os.path.join(output_test_dataset, "train"))
    calculate_face_landmarks_dataset(os.path.join(output_test_dataset, "validation"))
    calculate_face_correspondences_dataset(os.path.join(output_test_dataset, "train"))
    calculate_face_correspondences_dataset(os.path.join(output_test_dataset, "validation"))
