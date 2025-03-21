import itertools
import numpy as np
import torch

from src.preprocess_datasets.face_correspondences.CalculateFaceCorrespondences import \
    calculate_face_landmarks_dataset, calculate_face_correspondences_dataset
from src.preprocess_datasets.headPoseEstimation.create_test_dataset import create_train_test_split
from src.preprocess_datasets.headPoseEstimation.eval_hpe import evaluate_gaze_coverage
from src.preprocess_datasets.blazeface.face_crop import better_face_crop
from src.preprocess_datasets.headPoseEstimation.headpose_estimation import headpose_estimation_from_video
from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import generate_voxceleb_dataset_from_video
from src.preprocess_datasets.headPoseEstimation.match_hpe_angles_to_reference import find_matches

if __name__ == '__main__':

    folder_root = "E:\\Download\\vox2test"
    model_path_hpe = "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    dataset_output_folder = "E:\\Download\\vox2test_out"
    dataset_output_folder_crop = "E:\\Download\\vox2test_out_crop"
    output_test_dataset = "E:\\Download\\test_vox2test"
    face_detect_model = "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\det_10g.onnx.pt"
    batch_size = 128
    device = torch.device("cuda")

    print("##################################")
    print("#############HPE##################")
    print("##################################")
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
    generate_voxceleb_dataset_from_video(folder_root, dataset_output_folder, keep=True)

    print("##################################")
    print("##########Better Face Crop###########")
    print("##################################")
    better_face_crop(dataset_output_folder, dataset_output_folder_crop, face_detect_model)

    print("##################################")
    print("######face_correspondences########")
    print("##################################")
    calculate_face_landmarks_dataset(dataset_output_folder)
    calculate_face_correspondences_dataset(dataset_output_folder, keep=True)

    print("##################################")
    print("###########GEN TEST DATASET############")
    print("for:", dataset_output_folder, "to:", output_test_dataset)
    print("##################################")
    create_train_test_split(dataset_output_folder, output_test_dataset)
    print("######face_correspondences########")
