import hashlib
from collections import defaultdict

import os
import shutil
import torch
from tqdm import tqdm

from src.preprocess_datasets.misc.create_test_dataset import create_train_test_split
from src.preprocess_datasets.process_dataset_retinaface import face_crop_and_alignment


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
    valid_cams = {"041", "050", "051", "080", "090", "130", "140", "190"}

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
    output_dir = "F:\\Face\\data\\dataset11\\multipie8"
    output_folder_crop = "F:\\Face\\data\\dataset11\\multipie_crop8"
    output_test_dataset = "F:\\Face\\data\\dataset11\\test_multipie8"
    poses = 8

    preprocess_multipie(input_dir, output_dir, valid_cams, cam_to_coords)

    print("##################################")
    print("##### Crop Frames ################")
    print("##################################")
    face_crop_and_alignment(output_dir, output_folder_crop, face_factor=0.8, device='cuda' if torch.cuda.is_available() else 'cpu', resize_size=(112, 112), det_threshold=0.05)

    print("##################################")
    print("###### Create Test Dataset #######")
    print("##################################")
    create_train_test_split(output_folder_crop, output_test_dataset, poses=poses, ignore_face_corr=True)
