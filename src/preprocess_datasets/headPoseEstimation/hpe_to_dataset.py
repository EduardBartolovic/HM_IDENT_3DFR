import csv
import hashlib
import os
import time

import cv2
import numpy as np
from tqdm import tqdm

from src.preprocess_datasets.blazeface.face_crop import resize_with_padding


def read_file(file_path, remove_header=True):
    data = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        if remove_header:
            _ = next(reader)
        for row in reader:
            data.append(row)

    return np.array(data)


def generate_voxceleb_dataset_from_video(folder_root, dataset_output_folder, keep=True):

    start_time = time.time()
    counter = 0
    folders = list(os.walk(os.path.join(folder_root)))
    errors = 0
    for root, _, files in tqdm(folders, desc="Processing folders"):
        if "analysis" in root:
            for txt_file in files:
                if txt_file.endswith('matched_angles.txt'):
                    file_path = os.path.join(root, txt_file)
                    data = read_file(file_path)
                    video_folder_path = os.path.join(root, "..")

                    sample_name = os.path.abspath(os.path.join(root, os.pardir))
                    id_name = os.path.abspath(os.path.join(sample_name, os.pardir))
                    sample_name = os.path.basename(sample_name)
                    id_name = os.path.basename(id_name)

                    destination = os.path.join(dataset_output_folder, id_name)
                    os.makedirs(destination, exist_ok=True)

                    for info in data:
                        video_path = os.path.join(video_folder_path, info[7].split('#')[0])
                        frame_index = int(info[7].split('#')[1])
                        x_min, y_min, x_max, y_max = info[8:]
                        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                        hash_name = hashlib.sha1((id_name + sample_name).encode()).hexdigest()
                        dst = os.path.join(destination, f'{hash_name}{info[0]}_{info[1]}_image.jpg')

                        if keep and os.path.exists(dst):
                            continue  # Skip if file already exists

                        cap = cv2.VideoCapture(video_path)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ret, frame = cap.read()

                        if ret:
                            pad_size = 16  # Since (256-224)/2 = 16
                            padded_image = cv2.copyMakeBorder(frame, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                            face_crop = padded_image[y_min:y_max, x_min:x_max]
                            face_crop_resized = cv2.resize(face_crop, (112, 112), cv2.INTER_AREA)  # maybe better? cv2.INTER_LANCZOS4
                            cv2.imwrite(dst, face_crop_resized)
                            counter += 1
                        else:
                            errors += 1

                        cap.release()

    elapsed_time = time.time() - start_time
    print("Copied", counter, "files in", round(elapsed_time/60, 2), "min, with ", errors, "errors")


def generate_nersemble_dataset_from_video(folder_root, dataset_output_folder, keep=True):

    start_time = time.time()
    counter = 0
    folders = list(os.walk(os.path.join(folder_root)))
    errors = 0
    for root, _, files in tqdm(folders, desc="Processing folders"):
        if "analysis" in root:
            for txt_file in files:
                if txt_file.endswith('matched_angles.txt'):
                    file_path = os.path.join(root, txt_file)
                    data = read_file(file_path)

                    video_folder_path = os.path.join(root, "..")

                    sample_name = os.path.abspath(os.path.join(root, os.pardir))
                    id_name = os.path.abspath(os.path.join(sample_name, os.pardir, os.pardir))
                    sample_name = os.path.basename(sample_name)
                    id_name = os.path.basename(id_name)

                    destination = os.path.join(dataset_output_folder, id_name)
                    os.makedirs(destination, exist_ok=True)

                    for info in data:
                        video_path = os.path.join(video_folder_path, info[7].split('#')[0])
                        frame_index = int(info[7].split('#')[1])
                        x_min, y_min, x_max, y_max = info[8:]
                        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                        hash_name = hashlib.sha1((id_name + sample_name).encode()).hexdigest()
                        dst = os.path.join(destination, f'{hash_name}{info[0]}_{info[1]}_image.jpg')

                        if keep and os.path.exists(dst):
                            continue  # Skip if file already exists

                        cap = cv2.VideoCapture(video_path)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ret, frame = cap.read()

                        if ret:
                            padded_image, _, _ = resize_with_padding(frame)
                            face_crop = padded_image[y_min:y_max, x_min:x_max]
                            face_crop_resized = cv2.resize(face_crop, (112, 112), cv2.INTER_AREA)  # maybe better? cv2.INTER_LANCZOS4
                            cv2.imwrite(dst, face_crop_resized)
                            counter += 1
                        else:
                            errors += 1

                        cap.release()

    elapsed_time = time.time() - start_time
    print("Copied", counter, "files in", round(elapsed_time/60, 2), "min, with ", errors, "errors")