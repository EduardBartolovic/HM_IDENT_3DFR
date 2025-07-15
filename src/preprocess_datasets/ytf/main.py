import itertools

import numpy as np
import torch

from src.preprocess_datasets.create_test_dataset import create_train_test_split
from src.preprocess_datasets.face_correspondences.CalculateFaceCorrespondences import calculate_face_landmarks_dataset, calculate_face_correspondences_dataset
from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import generate_voxceleb_dataset_from_video, \
    generate_ytf_dataset_from_video
from src.preprocess_datasets.headPoseEstimation.match_hpe_angles_to_reference import find_matches
from src.preprocess_datasets.preprocess_video import analyse_video_ytf
from src.preprocess_datasets.process_dataset_retinaface import face_crop_and_alignment, \
    face_crop_and_alignment_deepfolder
from src.preprocess_datasets.rendering import PrepareDataset


def preprocessing():
    root = "F:\\Face\\data\\dataset10\\"
    folder_root = root+"aligned_images_DB"
    folder_root_crop = root+"aligned_images_DB_crop"
    dataset_output_folder = root+"aligned_images_DB_out"
    output_test_dataset = root+"test_ytf8"
    model_path_hpe = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    face_detect_model_root = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\blazeface"
    batch_size = 8  # 256 for 24GB  # 48 for 8 GB VRAM
    device = torch.device("cuda")

    print("##################################")
    print("##### Crop Frames ################")
    print("##################################")
    #face_crop_and_alignment_deepfolder(folder_root, folder_root_crop, face_factor=0.75, device='cuda' if torch.cuda.is_available() else 'cpu')

    print("##################################")
    print("##### Analyse Video ##############")
    print("##################################")
    analyse_video_ytf(folder_root_crop, "analysis", model_path_hpe, face_detect_model_root, device, batch_size=batch_size, keep=True, max_workers=16, face_confidence=0.6)

    print("##################################")
    print("##### FIND MATCHES ###############")
    print("##################################")
    ref_angles = [-25, -10, 0, 10, 25]
    permutations = np.array([(x, y, 0) for x, y in itertools.product(ref_angles, repeat=2)])
    print("number of permutations:", len(permutations))
    print(permutations)
    find_matches(folder_root_crop, permutations, txt_name="analysis.txt")
    # evaluate_gaze_coverage(folder_root)

    print("##################################")
    print("##### GEN DATASET ################")
    print("##################################")
    generate_ytf_dataset_from_video(folder_root_crop, dataset_output_folder, keep=True)

    perspective_filter = ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0']
    PrepareDataset.filter_views(dataset_output_folder, output_test_dataset, perspective_filter, target_views=8)

    exit()
    print("##################################")
    print("###### face_correspondences ######")
    print("##################################")
    calculate_face_landmarks_dataset(dataset_output_folder)
    perspective_filter = ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0']
    calculate_face_correspondences_dataset(dataset_output_folder, keep=True, filter_keywords=perspective_filter, target_views=len(perspective_filter))

    print("##################################")
    print("###### Create Test Dataset #######")
    print("##################################")
    create_train_test_split(dataset_output_folder, output_test_dataset, poses=poses, ignore_face_corr=True)


if __name__ == '__main__':
    preprocessing()
