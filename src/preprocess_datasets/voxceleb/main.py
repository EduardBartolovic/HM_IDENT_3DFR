import itertools
import numpy as np
import torch

from src.preprocess_datasets.create_test_dataset import create_train_test_split
from src.preprocess_datasets.face_correspondences.CalculateFaceCorrespondences import calculate_face_landmarks_dataset, calculate_face_correspondences_dataset
from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import generate_voxceleb_dataset_from_video
from src.preprocess_datasets.headPoseEstimation.match_hpe_angles_to_reference import find_matches
from src.preprocess_datasets.preprocess_video import analyse_video_hpe
from src.preprocess_datasets.process_dataset_retinaface import face_crop_and_alignment
from src.preprocess_datasets.rendering import PrepareDataset


def preprocessing():
    root = "F:\\Face\\data\\datasets9\\"
    folder_root = root+"vox2test_raw"
    dataset_output_folder = root+"vox2test_out-2"
    dataset_output_folder_crop = root+"vox2test_crop-2"
    dataset_output_folder_filtered = root+"vox2test_crop8-2"
    output_test_dataset = root+"test_vox2test_crop8-2"
    model_path_hpe = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    batch_size = 8 # 256 for 24GB  # 48 for 8 GB VRAM
    poses = 25  # Number of poses
    device = torch.device("cuda")

    print("##################################")
    print("##### Analyse Video ##############")
    print("##################################")
    analyse_video_hpe(folder_root, "analysis", model_path_hpe, device, batch_size=batch_size, keep=False, max_workers=16, face_confidence=0.5, padding=True)

    print("##################################")
    print("##### FIND MATCHES ###############")
    print("##################################")
    ref_angles = [-25, -10, 0, 10, 25]
    permutations = np.array([(x, y, 0) for x, y in itertools.product(ref_angles, repeat=2)])
    print("number of permutations:", len(permutations))
    print(permutations)
    find_matches(folder_root, permutations, txt_name="analysis.txt", correct_angles=True)
    # evaluate_gaze_coverage(folder_root)

    print("##################################")
    print("##### GEN DATASET ################")
    print("##################################")
    generate_voxceleb_dataset_from_video(folder_root, dataset_output_folder, keep=True)

    print("##################################")
    print("##### Crop Frames ################")
    print("##################################")
    face_crop_and_alignment(dataset_output_folder, dataset_output_folder_crop, face_factor=0.8, device='cuda' if torch.cuda.is_available() else 'cpu', resize_size=(224, 224), det_threshold=0.05)

    perspective_filter = ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0']
    PrepareDataset.filter_views(dataset_output_folder_crop, dataset_output_folder_filtered, perspective_filter, target_views=8)

    #print("##################################")
    #print("###### face_correspondences ######")
    #print("##################################")
    #calculate_face_landmarks_dataset(dataset_output_folder+"\\train")
    #perspective_filter = ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0']
    #calculate_face_correspondences_dataset(dataset_output_folder+"\\train", keep=True, target_views=8, processes=1)

    print("##################################")
    print("###### Create Test Dataset #######")
    print("##################################")
    create_train_test_split(dataset_output_folder_filtered, output_test_dataset, poses=poses, ignore_face_corr=True)


if __name__ == '__main__':
    preprocessing()
