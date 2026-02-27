from pathlib import Path
import itertools
import numpy as np
import torch
from src.preprocess_datasets.cropping.croppingv3.cropping_and_alignment import run_batch_alignment
from src.preprocess_datasets.cropping.croppingv3.face_detection import FaceAligner
from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import generate_ytf_dataset_from_video
from src.preprocess_datasets.headPoseEstimation.match_hpe_angles_to_reference import find_matches
from src.preprocess_datasets.preprocess_video import analyse_images_hpe


def preprocessing():
    root = "F:\\Face\\data\\dataset15\\"
    folder_root = "H:\\Maurer\\YouTubeFaces\\YouTubeFaces\\" + "frame_images_DB"
    folder_root_crop = root+"ytf_crop"
    dataset_output_folder = root+"aligned_images_DB_out"
    folder_root_crop = root+"ytf_crop"
    output_test_dataset = root+"test_ytf_crop8"
    model_path_hpe = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\resnet50.pt"
    model_path_cropping = Path("F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\cropping/croppingv3/mobile0.25.onnx")
    #model_path_cropping = Path("/home/gustav/HM_IDENT_3DFR/src/preprocess_datasets/cropping/croppingv3/mobile0.25.onnx")
    batch_size = 48  # 256 for 24GB  # 48 for 8 GB VRAM
    random_choice = False
    device = torch.device("cuda")

    print("##################################")
    print("##### Analyse Images ##############")
    print("##################################")
    #analyse_images_hpe(folder_root, "analysis", model_path_hpe, device, batch_size=batch_size, keep=True, face_confidence=0.3, padding=True)

    print("##################################")
    print("##### FIND MATCHES ###############")
    print("##################################")
    #ref_angles = [-25, -10, 0, 10, 25]
    #permutations = np.array([(x, y, 0) for x, y in itertools.product(ref_angles, repeat=2)])
    permutations = np.array([
        [0, -25, 0],
        [0, -10, 0],
        [0, 0, 0],
        [0, 10, 0],
        [0, 25, 0]
    ])
    print("number of permutations:", len(permutations))
    print(permutations)
    find_matches(folder_root, permutations, pkl_name="analysis.pkl", allow_flip=False, random_choice=random_choice)

    print("##################################")
    print("##### GEN DATASET ################")
    print("##################################")
    generate_ytf_dataset_from_video(folder_root, dataset_output_folder, keep=True)

    print("##################################")
    print("##### Crop Frames ################")
    print("##################################")
    DEVICE = "cpu"
    folder_paths = [p for p in Path(dataset_output_folder).iterdir() if p.is_dir()]
    run_batch_alignment(
        data_folders=folder_paths,
        model_path=str(model_path_cropping),
        align_method=FaceAligner.AlignmentMethod.AURA_FACE_ALIGNER,
        batch_size=32,
        output_dir=Path(folder_root_crop),
        num_processes=4,
        device=DEVICE
    )

    #perspective_filter = ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0']
    #PrepareDataset.filter_views(dataset_output_folder, output_test_dataset, perspective_filter, target_views=8)


if __name__ == '__main__':
    preprocessing()
