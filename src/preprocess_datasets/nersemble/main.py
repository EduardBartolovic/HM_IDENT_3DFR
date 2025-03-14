import torch

from src.preprocess_datasets.face_correspondences.CalculateFaceCorrespondences import calculate_face_landmarks_dataset, \
    calculate_face_correspondences_dataset
from src.preprocess_datasets.headPoseEstimation.create_test_dataset import create_train_test_split
from src.preprocess_datasets.nersemble.collect_frames import extract_and_group_frames

if __name__ == '__main__':

    folder_root = "C:\\Users\\Eduard\\Downloads\\nersemble\\sequence_EXP-1-head_part-1"
    model_path_hpe = "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    dataset_output_folder = "C:\\Users\\Eduard\\Downloads\\nersemble\\sequence_EXP-1-head_part-1-out"
    output_test_dataset = "C:\\Users\\Eduard\\Downloads\\nersemble\\sequence_EXP-1-head_part-1-out-test"
    batch_size = 128
    device = torch.device("cuda")

    print("##################################")
    print("#######Extract Frames#############")
    print("##################################")
    extract_and_group_frames(folder_root, dataset_output_folder)

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
