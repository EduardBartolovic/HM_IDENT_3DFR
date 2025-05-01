import itertools

import numpy as np
import torch

from src.preprocess_datasets.blazeface.face_crop import face_crop_full_frame
from src.preprocess_datasets.face_correspondences.CalculateFaceCorrespondences import calculate_face_landmarks_dataset, \
    calculate_face_correspondences_dataset
from src.preprocess_datasets.headPoseEstimation.create_test_dataset import create_train_test_split
from src.preprocess_datasets.headPoseEstimation.headpose_estimation import headpose_estimation_from_video
from src.preprocess_datasets.headPoseEstimation.match_hpe_angles_to_reference import find_matches, find_matches
from src.preprocess_datasets.nersemble.collect_frames import generate_voxceleb_dataset_from_video_nersemble
from src.preprocess_datasets.preprocess_video import analyse_video_nersemble

if __name__ == '__main__':

    root = "F:\\Face\\nersemble\\"
    folder_root = root + "data"
    dataset_output_folder = root + "data_out"
    output_test_dataset = root + "test_data"
    model_path_hpe = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    face_detect_model_root = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\blazeface"
    batch_size = 48 # 128  # 48 for 8 GB VRAM
    poses = 25  # Number of poses
    device = torch.device("cuda")

    print("##################################")
    print("##### Analyse Video ##############")
    print("##################################")
    cams = ["cam_222200037",#["cam_220700191",
            "cam_221501007",
            "cam_222200042"]
            #"cam_222200036",
            #"cam_222200038",
            #"cam_222200039",
            #"cam_222200040",
            #"cam_222200041",
            #"cam_222200043",
            #"cam_222200044",
            #"cam_222200045",
            #"cam_222200046",
            #"cam_222200047",
            #"cam_222200048",
            #"cam_222200049"]
    for i in cams:
        analyse_video_nersemble(folder_root, "analysis_"+i, model_path_hpe, face_detect_model_root, device, batch_size=batch_size, filter=i, keep=False)

    exit()
    print("##################################")
    print("##### FIND MATCHES ###############")
    print("##################################")
    ref_angles = [-25, -10, 0, 10, 25]
    permutations = np.array([(x, y, 0) for x, y in itertools.product(ref_angles, repeat=2)])
    print("number of permutations:", len(permutations))
    print(permutations)
    find_matches_new(folder_root, permutations, txt_name="analysis.txt")


    exit()
    print("##################################")
    print("###########GEN DATASET############")
    print("##################################")
    generate_voxceleb_dataset_from_video_nersemble(folder_root, dataset_output_folder, keep=True)

    print("##################################")
    print("##########Better Face Crop###########")
    print("##################################")
    face_crop_full_frame(dataset_output_folder, dataset_output_folder_crop, face_detect_model_root)

    print("##################################")
    print("######face_correspondences########")
    print("##################################")
    calculate_face_landmarks_dataset(dataset_output_folder_crop)
    calculate_face_correspondences_dataset(dataset_output_folder_crop, keep=True)

    print("##################################")
    print("###########GEN TEST DATASET############")
    print("for:", dataset_output_folder_crop, "to:", output_test_dataset)
    print("##################################")
    create_train_test_split(dataset_output_folder_crop, output_test_dataset, ignore_face_corr=False)
    print("######face_correspondences########")
