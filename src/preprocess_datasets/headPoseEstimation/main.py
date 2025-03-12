import itertools
import numpy as np
import torch

from src.preprocess_datasets.face_correspondences.CalculateFaceCorrespondences import \
    calculate_face_landmarks_dataset, calculate_face_correspondences_dataset
from src.preprocess_datasets.headPoseEstimation.create_test_dataset import create_train_test_split
from src.preprocess_datasets.headPoseEstimation.eval_hpe import evaluate_gaze_coverage
from src.preprocess_datasets.headPoseEstimation.headpose_estimation import headpose_estimation_from_video
from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import generate_voxceleb_dataset_from_video
from src.preprocess_datasets.headPoseEstimation.match_hpe_angles_to_reference import find_matches

if __name__ == '__main__':

    folder_root = "E:\\Download\\vox2_mp4_6\\dev\\mp42"
    model_path_hpe = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    dataset_output_folder = "E:\\Download\\vox2"
    output_test_dataset = "C:\\Users\\Eduard\\Downloads\\test_VoxCeleb2_train_dataset"
    #backbone_face_model = "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\pretrained\\backbone_ir50_asia.pth"
    batch_size = 128
    device = torch.device("cuda")

    #print("##################################")
    #print("##########Filter wrong faces###########")
    #print("##################################")
    #filter_wrong_faces(folder_root, "frames_filtered", backbone_face_model, "cuda")

    print("##################################")
    print("#############HPE##################")
    print("##################################")
    #headpose_estimation(folder_root, "frames_filtered", "hpe", model_path_hpe, device, fix_rotation=True, draw=True)
    headpose_estimation_from_video(folder_root, "hpe", model_path_hpe, device, batch_size=batch_size)

    ref_angles = [-25, -10, 0, 10, 25]
    permutations = np.array([(x, y, 0) for x, y in itertools.product(ref_angles, repeat=2)])
    print(len(permutations))
    print(permutations)
    print("##################################")
    print("###########FINDMATCHES############")
    print("##################################")
    find_matches(folder_root, permutations, txt_name="hpe.txt")
    evaluate_gaze_coverage(folder_root)

    print("##################################")
    print("###########GEN DATASET############")
    print("##################################")
    generate_voxceleb_dataset_from_video(folder_root, dataset_output_folder)

    print("##################################")
    print("######face_correspondences########")
    print("##################################")
    calculate_face_landmarks_dataset(dataset_output_folder)
    calculate_face_correspondences_dataset(dataset_output_folder)

    print("##################################")
    print("###########GEN TEST DATASET############")
    print("for:", dataset_output_folder, "to:", output_test_dataset)
    print("##################################")
    create_train_test_split(dataset_output_folder, output_test_dataset)
    print("######face_correspondences########")
