import hashlib
import time

import cv2
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def find_min_error_per_line(file_paths):
    data_frames = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        data_frames.append(df)

    if not data_frames:
        print("No valid data found.")
        return None

    combined_df = pd.concat(data_frames, axis=0).groupby(['Ref_X', 'Ref_Y', 'Ref_Z']).min().reset_index()
    return combined_df


def generate_voxceleb_dataset_from_video_nersemble(input_folder, output_folder, keep=True):

    start_time = time.time()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    counter = 0
    for id_folder in tqdm(os.listdir(input_folder), desc="Iterating over ids"):
        for exp_folder in os.listdir(os.path.join(input_folder, id_folder)):
            video_folder_path = os.path.join(input_folder, id_folder, "sequences", exp_folder)
            txts = []
            for root, _, files in os.walk(video_folder_path):
                for f in files:
                    if f.endswith('matched_angles.txt') and "cam_222200037" in root:
                        txts.append(os.path.join(root, f))

            df = find_min_error_per_line(txts)

            #sample_name = os.path.abspath(os.path.join(root, os.pardir))
            ##id_name = os.path.abspath(os.path.join(sample_name, os.pardir))
            #sample_name = os.path.basename(sample_name)
            #id_name = os.path.basename(id_name)

            destination = os.path.join(output_folder, id_folder)
            os.makedirs(destination, exist_ok=True)

            for info in df.iterrows():
                info = np.array(info[1])
                video_path = os.path.join(video_folder_path, "cam_222200037.mp4")
                frame_index = int(info[6].split('#')[1])

                hash_name = hashlib.sha1((id_folder + exp_folder).encode()).hexdigest()
                dst = os.path.join(destination, f'{hash_name}{info[0]}_{info[1]}_image.jpg')

                if keep and os.path.exists(dst):
                    continue  # Skip if file already exists

                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()

                if ret:
                    cv2.imwrite(dst, frame)
                    counter += 1

                cap.release()

        elapsed_time = time.time() - start_time
        print("Copied", counter, "files in", round(elapsed_time / 60, 2), "min")