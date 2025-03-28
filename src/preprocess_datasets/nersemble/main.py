import torch

from src.preprocess_datasets.headPoseEstimation.headpose_estimation import headpose_estimation_from_video

if __name__ == '__main__':

    folder_root = "E:\\Download\\nersemble\\dev"
    model_path_hpe = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    #dataset_output_folder = "C:\\Users\\Eduard\\Downloads\\nersemble\\sequence_EXP-1-head_part-1-out"
    #output_test_dataset = "C:\\Users\\Eduard\\Downloads\\nersemble\\sequence_EXP-1-head_part-1-out-test"
    batch_size = 128
    device = torch.device("cuda")

    print("##################################")
    print("#######    HPE       #############")
    print("##################################")
    cams = ["cam_220700191",
            "cam_221501007",
            "cam_222200036",
            "cam_222200037",
            "cam_222200038",
            "cam_222200039",
            "cam_222200040",
            "cam_222200041",
            "cam_222200042",
            "cam_222200043",
            "cam_222200044",
            "cam_222200045",
            "cam_222200046",
            "cam_222200047",
            "cam_222200048",
            "cam_222200049"]
    for i in cams:
        headpose_estimation_from_video(folder_root, "hpe_"+i, model_path_hpe, device, batch_size=batch_size, filter=i)

    exit()

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
