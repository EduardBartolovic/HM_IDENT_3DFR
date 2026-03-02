import hashlib
from collections import defaultdict

import os
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from src.preprocess_datasets.cropping.croppingv3.cropping_and_alignment import run_batch_alignment
from src.preprocess_datasets.cropping.croppingv3.face_detection import FaceAligner
from src.preprocess_datasets.misc.create_test_dataset import create_train_test_split


def preprocess_multipie(
    input_dir,
    output_dir,
    valid_cams,
    cam_to_coords,
):

    os.makedirs(output_dir, exist_ok=True)

    # group by (illumination, recording, subject)
    groups = defaultdict(list)

    for fname in tqdm(os.listdir(input_dir), desc="Sorting files"):
        if not fname.endswith(".png"):
            continue

        # Example: 001_01_01_010_00_crop_128.png
        parts = fname.split("_")[:-2]
        subject = parts[0]
        session = parts[1]  # unused
        recording = parts[2]
        camera = parts[3]
        illumination = parts[4]

        # only use valid cams
        if camera not in valid_cams:
            continue

        if illumination == "019":
            continue

        key = (illumination, recording, subject)
        groups[key].append((fname, camera))

    # Now restructure
    for (illumination, recording, subject), items in tqdm(groups.items(), desc="Copying files"):
        if len(items) != len(valid_cams):
            print("Missing Items", len(items), items)
            continue  # skip incomplete groups

        # Create class folder for this illumination+recording
        class_name = subject
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Consistent hash per subject
        sha = hashlib.sha1(f"{subject}_{illumination}_{recording}".encode()).hexdigest()[:15]

        for i, (fname, camera) in enumerate(items):
            src = os.path.join(input_dir, fname)

            coords = cam_to_coords[camera]  # map camera → ~ coordinates
            dst_name = f"{sha}#{coords}#{coords}.png"
            dst = os.path.join(class_dir, dst_name)

            shutil.copy(src, dst)


if __name__ == '__main__':

    print("##################################")
    print("######## Prepare Dataset #########")
    print("##################################")

    # keep only these cameras
    valid_cams = {"050", "051", "140"}
    #valid_cams = {"041", "050", "051", "130", "140"}
    #valid_cams = {"041", "050", "051", "080", "090", "130", "140", "190"}

    cam_to_coords = {
        "041": "-15_0",
        "050": "-10_0",
        "051": "0_0",
        "080": "25_0",
        "090": "35_0",
        "130": "15_0",
        "140": "10_0",
        "190": "-25_0",
    }

    input_dir = "H:\\Maurer\\CMU_Multi_pie\\Multi_Pie\\HR_128"
    output_dir = "F:\\Face\\data\\dataset11\\multipie3"
    output_folder_crop = "F:\\Face\\data\\dataset11\\multipie_crop3"
    output_test_dataset = "F:\\Face\\data\\dataset11\\test_multipie_crop3"
    poses = 3
    model_path_cropping = Path("/home/gustav/HM_IDENT_3DFR/src/preprocess_datasets/cropping/croppingv3/mobile0.25.onnx")

    preprocess_multipie(input_dir, output_dir, valid_cams, cam_to_coords)

    print("##################################")
    print("##### Crop Frames ################")
    print("##################################")
    DEVICE = "cpu"
    folder_paths = [p for p in Path(output_dir).iterdir() if p.is_dir()]
    run_batch_alignment(
        data_folders=folder_paths,
        model_path=str(model_path_cropping),
        align_method=FaceAligner.AlignmentMethod.AURA_FACE_ALIGNER,
        batch_size=32,
        output_dir=Path(output_folder_crop),
        num_processes=4,
        device=DEVICE
    )

    print("##################################")
    print("###### Create Test Dataset #######")
    print("##################################")
    create_train_test_split(output_folder_crop, output_test_dataset, poses=poses)
