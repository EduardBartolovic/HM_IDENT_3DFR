import itertools
import numpy as np
import torch

from src.preprocess_datasets.headPoseEstimation.create_test_dataset import create_train_test_split
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
    batch_size = 48 # 128  # 48 for 8 GB VRAM
    poses = 25  # Number of poses
    device = torch.device("cuda")

    print("##################################")
    print("##### Analyse Video ##############")
    print("##################################")
    analyse_video_vox(folder_root, "analysis", model_path_hpe, face_detect_model_root, device, batch_size=batch_size, keep=False)

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
    generate_voxceleb_dataset_from_video(folder_root, dataset_output_folder, keep=False)

    print("##################################")
    print("###### face_correspondences ######")
    print("##################################")
    #calculate_face_landmarks_dataset(dataset_output_folder_crop)
    #calculate_face_correspondences_dataset(dataset_output_folder_crop, keep=True)

    print("##################################")
    print("###### Create Test Dataset #######")
    print("##################################")
    create_train_test_split(dataset_output_folder, output_test_dataset, poses=poses, ignore_face_corr=True)


if __name__ == '__main__':
    preprocessing()
