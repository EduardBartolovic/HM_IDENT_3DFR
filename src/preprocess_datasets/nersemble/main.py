import itertools
import numpy as np
import torch

from src.preprocess_datasets.nersemble.collect_frames import extract_and_group_frames

if __name__ == '__main__':

    folder_root = "C:\\Users\\Eduard\\Downloads\\nersemble\\sequence_EXP-1-head_part-1"
    model_path_hpe = "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    dataset_output_folder = "C:\\Users\\Eduard\\Downloads\\nersemble\\sequence_EXP-1-head_part-1-out"
    output_test_dataset = "C:\\Users\\Eduard\\Downloads\\nersemble\\sequence_EXP-1-head_part-1-out-test"
    batch_size = 128
    device = torch.device("cuda")

    print("##################################")
    print("#############Extract Frames##################")
    print("##################################")
    extract_and_group_frames(folder_root, dataset_output_folder)

    exit()
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
