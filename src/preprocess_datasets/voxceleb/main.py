import itertools
import numpy as np
import torch

from src.preprocess_datasets.create_test_dataset import create_train_test_split
from src.preprocess_datasets.face_correspondences.CalculateFaceCorrespondences import calculate_face_landmarks_dataset, calculate_face_correspondences_dataset
from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import generate_voxceleb_dataset_from_video
from src.preprocess_datasets.headPoseEstimation.match_hpe_angles_to_reference import find_matches
from src.preprocess_datasets.preprocess_video import analyse_video_vox


def preprocessing():
    root = "F:\\Face\\vox2\\"
    folder_root = root+"vox2test"
    dataset_output_folder = root+"vox2test_out"
    output_test_dataset = root+"test_vox2test"
    model_path_hpe = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    face_detect_model_root = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\blazeface"
    batch_size = 8 # 256 for 24GB  # 48 for 8 GB VRAM
    poses = 25  # Number of poses
    device = torch.device("cuda")

    print("##################################")
    print("##### Analyse Video ##############")
    print("##################################")
    analyse_video_vox(folder_root, "analysis", model_path_hpe, face_detect_model_root, device, batch_size=batch_size, keep=True, max_workers=16, face_confidence=0.6)

    print("##################################")
    print("##### FIND MATCHES ###############")
    print("##################################")
    ref_angles = [-25, -10, 0, 10, 25]
    permutations = np.array([(x, y, 0) for x, y in itertools.product(ref_angles, repeat=2)])
    print("number of permutations:", len(permutations))
    print(permutations)
    find_matches(folder_root, permutations, txt_name="analysis.txt")
    # evaluate_gaze_coverage(folder_root)

    print("##################################")
    print("##### GEN DATASET ################")
    print("##################################")
    generate_voxceleb_dataset_from_video(folder_root, dataset_output_folder, keep=True)

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
