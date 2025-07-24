import itertools

import numpy as np
import torch

from src.preprocess_datasets.create_test_dataset import create_train_test_split
from src.preprocess_datasets.face_correspondences.CalculateFaceCorrespondences import calculate_face_landmarks_dataset, \
    calculate_face_correspondences_dataset
from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import generate_nersemble_dataset_from_video
from src.preprocess_datasets.headPoseEstimation.match_hpe_angles_to_reference import find_matches
from src.preprocess_datasets.preprocess_video import analyse_video_new
from src.preprocess_datasets.process_dataset_retinaface import face_crop_and_alignment
from src.preprocess_datasets.rendering import PrepareDataset

if __name__ == '__main__':

    root = "C:\\Users\\Eduard\\Desktop\\Face\\dataset10\\"
    folder_root = root + "data_raw"
    dataset_output_folder = root + "nersemble_out"
    dataset_output_folder_crop = root+"nersemble_crop"
    dataset_output_folder_filtered = root+"nersemble_filtered"
    output_test_dataset = root + "test_nersemble8"
    model_path_hpe = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    batch_size = 8  # 512 for 48GB # 256 for 24GB  # 48 for 8 GB VRAM
    frame_skip = 8  # 8 which is ~10FPS
    poses = 8  # Number of poses
    device = torch.device("cuda")

    print("##################################")
    print("##### Analyse Video ##############")
    print("##################################")
    analyse_video_new(folder_root, "analysis", model_path_hpe, device, keep=True, frame_skip=frame_skip, downscale=True)

    print("##################################")
    print("##### FIND MATCHES ###############")
    print("##################################")
    ref_angles = [-25, -10, 0, 10, 25]
    permutations = np.array([(x, y, 0) for x, y in itertools.product(ref_angles, repeat=2)])
    print("number of permutations:", len(permutations))
    print(permutations)
    find_matches(folder_root, permutations, txt_name="analysis.txt", correct_angles=True)

    print("##################################")
    print("##### GEN DATASET ################")
    print("##################################")
    generate_nersemble_dataset_from_video(folder_root, dataset_output_folder, keep=False)

    print("##################################")
    print("##### Crop Frames ################")
    print("##################################")
    face_crop_and_alignment(dataset_output_folder, dataset_output_folder_crop, face_factor=0.8, device='cuda' if torch.cuda.is_available() else 'cpu', resize_size=(2000, 3208), det_threshold=0.6)

    perspective_filter = ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0']
    PrepareDataset.filter_views(dataset_output_folder_crop, dataset_output_folder_filtered, perspective_filter, target_views=8)

    #print("##################################")
    #print("##### FACE CORRESPONDENCES #######")
    #print("##################################")
    #calculate_face_landmarks_dataset(dataset_output_folder)
    #perspective_filter = ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0']
    #calculate_face_correspondences_dataset(dataset_output_folder, keep=True, filter_keywords=perspective_filter, target_views=len(perspective_filter))

    print("##################################")
    print("##### GEN TEST DATASET ###########")
    print("##################################")
    create_train_test_split(dataset_output_folder_filtered, output_test_dataset, poses=8, ignore_face_corr=True)
