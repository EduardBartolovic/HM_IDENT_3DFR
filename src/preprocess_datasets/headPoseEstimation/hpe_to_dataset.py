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
    folders = list(os.walk(folder_root))
    jobs = []
    total_files = 0
    total_success = 0
    total_errors = 0

    for root, _, files in tqdm(folders, desc="Generating Dataset"):
        if "analysis" not in root:
            continue

        relevant_files = [f for f in files if f.endswith('matched_angles.txt')]
        if not relevant_files:
            continue

        sample_name_path = os.path.abspath(os.path.join(root, os.pardir))
        id_name_path = os.path.abspath(os.path.join(sample_name_path, os.pardir))
        sample_name = os.path.basename(sample_name_path)
        id_name = os.path.basename(id_name_path)
        video_folder_path = os.path.abspath(os.path.join(root, ".."))

        destination = os.path.join(dataset_output_folder, id_name)
        os.makedirs(destination, exist_ok=True)

        hash_name = hashlib.sha1((id_name + sample_name).encode()).hexdigest()

        for txt_file in relevant_files:
            file_path = os.path.join(root, txt_file)
            jobs.append((file_path, destination, hash_name, keep, video_folder_path))
            total_files += 1

        for job in jobs:
            success_count, errors = process_txt_file_to_video_voxceleb(job)
            total_success += success_count
            total_errors += errors

    elapsed_time = time.time() - start_time
    print(f"Copied {total_success} files in {elapsed_time / 60:.2f} min, with {total_errors} errors.")


def process_txt_file_to_video_voxceleb(args):
    file_path, destination, hash_name, keep, video_folder_path = args
    data = read_file(file_path)
    errors = 0
    success_count = 0
    video_cache = {}  # Cache for open VideoCapture objects
    for info in data:
        video_name, frame_index = info[7].split('#')
        frame_index = int(frame_index)
        x_min, y_min, x_max, y_max = map(int, info[8:])
        dst_filename = f'{hash_name}{info[0]}_{info[1]}_image.jpg'
        dst_path = os.path.join(destination, dst_filename)

        if keep and os.path.exists(dst_path):
            continue

        if video_name not in video_cache:
            video_path = os.path.join(video_folder_path, video_name)
            cap = cv2.VideoCapture(video_path)
            video_cache[video_name] = cap
        else:
            cap = video_cache[video_name]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            errors += 1
            continue

        pad_size = 16  # Since (256-224)/2 = 16
        padded_image = cv2.copyMakeBorder(frame, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        face_crop = padded_image[y_min:y_max, x_min:x_max]
        assert face_crop.size > 0

        face_crop_resized = cv2.resize(face_crop, (112, 112), cv2.INTER_AREA)  # maybe better? cv2.INTER_LANCZOS4
        cv2.imwrite(dst_path, face_crop_resized)
        success_count += 1

    for cap in video_cache.values():
        cap.release()

    return success_count, errors


def generate_nersemble_dataset_from_video(folder_root, dataset_output_folder, keep=True):

    start_time = time.time()
    folders = list(os.walk(folder_root))
    jobs = []
    total_files = 0
    total_success = 0
    total_errors = 0

    for root, _, files in tqdm(folders, desc="Generating Dataset"):
        if "analysis" not in root:
            continue

        relevant_files = [f for f in files if f.endswith('matched_angles.txt')]
        if not relevant_files:
            continue

        sample_name_path = os.path.abspath(os.path.join(root, os.pardir))
        id_name_path = os.path.abspath(os.path.join(sample_name_path, os.pardir, os.pardir))
        sample_name = os.path.basename(sample_name_path)
        id_name = os.path.basename(id_name_path)
        video_folder_path = os.path.abspath(os.path.join(root, ".."))

        destination = os.path.join(dataset_output_folder, id_name)
        os.makedirs(destination, exist_ok=True)

        hash_name = hashlib.sha1((id_name + sample_name).encode()).hexdigest()

        for txt_file in relevant_files:
            file_path = os.path.join(root, txt_file)
            jobs.append((file_path, destination, hash_name, keep, video_folder_path))
            total_files += 1

        for job in jobs:
            success_count, errors = process_txt_file_to_video_voxceleb(job)
            total_success += success_count
            total_errors += errors

    elapsed_time = time.time() - start_time
    print(f"Copied {total_success} files in {elapsed_time / 60:.2f} min, with {total_errors} errors.")


def process_txt_file_to_video_nersemble(args):
    file_path, destination, hash_name, keep, video_folder_path = args
    data = read_file(file_path)
    errors = 0
    success_count = 0
    video_cache = {}  # Cache for open VideoCapture objects
    for info in data:
        video_name, frame_index = info[7].split('#')
        frame_index = int(frame_index)
        x_min, y_min, x_max, y_max = map(int, info[8:])
        dst_filename = f'{hash_name}{info[0]}_{info[1]}_image.jpg'
        dst_path = os.path.join(destination, dst_filename)

        if keep and os.path.exists(dst_path):
            continue

        if video_name not in video_cache:
            video_path = os.path.join(video_folder_path, video_name)
            cap = cv2.VideoCapture(video_path)
            video_cache[video_name] = cap
        else:
            cap = video_cache[video_name]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            errors += 1
            continue

        padded_image, _, _ = resize_with_padding(frame)
        face_crop = padded_image[y_min:y_max, x_min:x_max]
        assert face_crop.size > 0

        face_crop_resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst_path, face_crop_resized)
        success_count += 1

    for cap in video_cache.values():
        cap.release()

    return success_count, errors
