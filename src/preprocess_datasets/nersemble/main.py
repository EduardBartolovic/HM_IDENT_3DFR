import itertools

import numpy as np
import torch

from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import generate_nersemble_dataset_from_video
from src.preprocess_datasets.headPoseEstimation.match_hpe_angles_to_reference import find_matches
from src.preprocess_datasets.preprocess_video import analyse_video_nersemble

if __name__ == '__main__':

    root = "F:\\Face\\nersemble\\"
    folder_root = root + "data"
    dataset_output_folder = root + "data_out"
    output_test_dataset = root + "test_data"
    model_path_hpe = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    face_detect_model_root = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\blazeface"
    batch_size = 48 # 256 for 24GB  # 48 for 8 GB VRAM
    poses = 25  # Number of poses
    device = torch.device("cuda")

    print("##################################")
    print("##### Analyse Video ##############")
    print("##################################")
    analyse_video_nersemble(folder_root, "analysis", model_path_hpe, face_detect_model_root, device, batch_size=batch_size, keep=False)

    print("##################################")
    print("##### FIND MATCHES ###############")
    print("##################################")
    ref_angles = [-25, -10, 0, 10, 25]
    permutations = np.array([(x, y, 0) for x, y in itertools.product(ref_angles, repeat=2)])
    print("number of permutations:", len(permutations))
    print(permutations)
    find_matches(folder_root, permutations, txt_name="analysis.txt")

    print("##################################")
    print("##### GEN DATASET ################")
    print("##################################")
    generate_nersemble_dataset_from_video(folder_root, dataset_output_folder, keep=False)

    exit()

    #print("##################################")
    #print("######face_correspondences########")
    #print("##################################")
    #calculate_face_landmarks_dataset(dataset_output_folder_crop)
    #calculate_face_correspondences_dataset(dataset_output_folder_crop, keep=True)

    print("##################################")
    print("###########GEN TEST DATASET############")
    print("##################################")
    create_train_test_split(dataset_output_folder_crop, output_test_dataset, ignore_face_corr=False)
    print("######face_correspondences########")
