import itertools
from pathlib import Path

import numpy as np
import torch

from src.preprocess_datasets.cropping.croppingv3.cropping_and_alignment import run_batch_alignment
from src.preprocess_datasets.cropping.croppingv3.face_detection import FaceAligner
from src.preprocess_datasets.misc.create_test_dataset import create_train_test_split
from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import generate_nersemble_dataset_from_video, \
    generate_voxceleb_dataset_from_video
from src.preprocess_datasets.headPoseEstimation.match_hpe_angles_to_reference import find_matches
from src.preprocess_datasets.preprocess_video import analyse_video_hpe
from src.preprocess_datasets.cropping.process_dataset_retinaface import face_crop_and_alignment
from src.preprocess_datasets.rendering import PrepareDataset

if __name__ == '__main__':

    root = "/home/gustav/nersemble/"
    folder_root = root + "data_raw"
    dataset_output_folder = root + "nersemble_out5R-v15"
    dataset_output_folder_crop = root+"nersemble_crop5R-v15"
    dataset_output_folder_filtered = root+"nersemble_crop5R-v15"
    output_test_dataset = root + "test_nersemble_crop5R-v15"

    model_path_hpe = "/home/gustav/HM_IDENT_3DFR/src/preprocess_datasets/headPoseEstimation/weights/resnet50.pt"
    batch_size = 256  # 256 for 24GB  # 48 for 8 GB VRAM
    poses = 5  # Number of poses
    random_choice = False
    allow_flip = False
    discard_threshold = 15
    device = torch.device("cuda")

    print("##################################")
    print("##### Analyse Video ##############")
    print("##################################")
    #analyse_video_hpe(folder_root, "analysis", model_path_hpe, device, keep=True, frame_skip=frame_skip, downscale=True, face_confidence=0.5)

    print("##################################")
    print("##### FIND MATCHES ###############")
    print("##################################")
    print("##################################")
    print("##### FIND MATCHES ###############")
    print("##################################")
    # ref_angles = [-25, -10, 0, 10, 25]
    # permutations = np.array([(x, y, 0) for x, y in itertools.product(ref_angles, repeat=2)])
    # if random_choice:
    #    permutations = permutations[:poses]
    permutations = np.array([
        [0, -25, 0],
        [0, -10, 0],
        [0, 0, 0],
        [0, 10, 0],
        [0, 25, 0]
    ])
    print("number of permutations:", len(permutations))
    print(permutations)
    find_matches(folder_root, permutations, pkl_name="analysis.pkl", correct_angles=True, allow_flip=allow_flip, random_choice=random_choice, avg_dist_threshold=discard_threshold)

    print("##################################")
    print("##### GEN DATASET ################")
    print("##################################")
    #generate_nersemble_dataset_from_video(folder_root, dataset_output_folder, keep=False)
    generate_voxceleb_dataset_from_video(folder_root, dataset_output_folder, keep=False)

    print("##################################")
    print("##### Crop Frames ################")
    print("##################################")
    MODEL_FILE = Path('mobile0.25.onnx')
    DEVICE = "cpu"

    folder_paths = [p for p in Path(dataset_output_folder).iterdir() if p.is_dir()]

    run_batch_alignment(
        data_folders=folder_paths,
        model_path=str(MODEL_FILE),
        align_method=FaceAligner.AlignmentMethod.AURA_FACE_ALIGNER,
        batch_size=32,
        output_dir=Path(dataset_output_folder_crop),
        num_processes=4,
        device=DEVICE
    )
    # face_crop_and_alignment(dataset_output_folder, dataset_output_folder_crop, face_factor=0.8, device='cuda' if torch.cuda.is_available() else 'cpu', resize_size=(224, 224), det_threshold=0.05)

    # perspective_filter = ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0']
    # PrepareDataset.filter_views(dataset_output_folder_crop, dataset_output_folder_filtered, perspective_filter, target_views=poses)

    print("##################################")
    print("##### GEN TEST DATASET ###########")
    print("##################################")
    create_train_test_split(dataset_output_folder_filtered, output_test_dataset, poses=5, ignore_face_corr=True)
