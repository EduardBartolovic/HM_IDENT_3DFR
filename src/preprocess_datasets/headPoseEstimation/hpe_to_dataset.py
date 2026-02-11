import csv
import hashlib
import os
import shutil
import time

import cv2
import numpy as np
from tqdm import tqdm


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

    for root, _, files in tqdm(folders, desc="Parsing Match files"):
        if "analysis" not in root:
            continue

        for file in files:
            if not file.endswith('matched_angles.txt'):
                continue

            sample_name_path = os.path.abspath(os.path.join(root, os.pardir))
            id_name_path = os.path.abspath(os.path.join(sample_name_path, os.pardir))
            sample_name = os.path.basename(sample_name_path)
            id_name = os.path.basename(id_name_path)
            video_folder_path = os.path.abspath(os.path.join(root, ".."))

            destination = os.path.join(dataset_output_folder, id_name)
            os.makedirs(destination, exist_ok=True)

            hash_name = hashlib.sha1((id_name + sample_name).encode()).hexdigest()

            file_path = os.path.join(root, file)
            jobs.append((file_path, destination, hash_name, keep, video_folder_path))
            total_files += 1

    for job in tqdm(jobs, desc="Generate Dataset"):
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
        try:
            video_name, frame_index = info[7].split('#')
            frame_index = int(frame_index)

            val = info[12]
            was_flipped = str(val).lower() in ["true"]

            flip_tag = "f" if was_flipped else ""

            dst_filename = (
                f'{hash_name[:15]}#{info[0]}_{info[1]}'
                f'#{clean_zero(info[3].split(".")[0])}_{clean_zero(info[4].split(".")[0])}'
                f'{flip_tag}.jpg'
            )
            dst_path = os.path.join(destination, dst_filename)

        except ValueError as e:
            print("Error in :", file_path)
            raise e

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

        # apply horizontal flip if needed
        if was_flipped:
            frame = cv2.flip(frame, 1)

        cv2.imwrite(dst_path, frame)
        success_count += 1

    for cap in video_cache.values():
        cap.release()

    return success_count, errors


def clean_zero(x):
    return str(int(float(x))) if float(x) == 0 else str(int(float(x)))


def generate_nersemble_dataset_from_video(folder_root, dataset_output_folder, keep=True):

    start_time = time.time()
    folders = list(os.walk(folder_root))
    jobs = []
    total_files = 0
    total_success = 0
    total_errors = 0

    for root, _, files in tqdm(folders, desc="Parsing Match files"):
        if "analysis" not in root:
            continue

        for file in files:
            if not file.endswith('matched_angles.txt'):
                continue

            sample_name_path = os.path.abspath(os.path.join(root, os.pardir))
            id_name_path = os.path.abspath(os.path.join(sample_name_path, os.pardir, os.pardir))
            sample_name = os.path.basename(sample_name_path)
            id_name = os.path.basename(id_name_path)
            video_folder_path = os.path.abspath(os.path.join(root, ".."))

            destination = os.path.join(dataset_output_folder, id_name)
            os.makedirs(destination, exist_ok=True)

            hash_name = hashlib.sha1((id_name + sample_name).encode()).hexdigest()

            file_path = os.path.join(root, file)
            jobs.append((file_path, destination, hash_name, keep, video_folder_path))
            total_files += 1

    for job in tqdm(jobs, desc="Generate Dataset"):
        success_count, errors = process_txt_file_to_video_nersemble(job)
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
        try:
            video_name, frame_index = info[7].split('#')
            frame_index = int(frame_index)
            #x_min, y_min, x_max, y_max = map(int, info[8:])
            dst_filename = f'{hash_name[:15]}#{info[0]}_{info[1]}#{info[3].split(".")[0]}_{info[4].split(".")[0]}.jpg'
            dst_path = os.path.join(destination, dst_filename)
        except ValueError:
            raise ValueError("Error in :", file_path)

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

        cv2.imwrite(dst_path, frame)
        success_count += 1

    for cap in video_cache.values():
        cap.release()

    return success_count, errors


def generate_ytf_dataset_from_video(folder_root, dataset_output_folder, keep=True):

    start_time = time.time()
    folders = list(os.walk(folder_root))
    jobs = []
    total_files = 0
    total_success = 0
    total_errors = 0

    for root, _, files in tqdm(folders, desc="Parsing Match files"):
        if "analysis" not in root:
            continue

        for file in files:
            if not file.endswith('matched_angles.txt'):
                continue

            sample_name_path = os.path.abspath(os.path.join(root, os.pardir))
            id_name_path = os.path.abspath(os.path.join(sample_name_path, os.pardir))
            sample_name = os.path.basename(sample_name_path)
            id_name = os.path.basename(id_name_path)
            video_folder_path = os.path.abspath(os.path.join(root, ".."))

            destination = os.path.join(dataset_output_folder, id_name)
            os.makedirs(destination, exist_ok=True)

            hash_name = (sample_name + 'X' * 15)[:15]

            file_path = os.path.join(root, file)
            jobs.append((file_path, destination, hash_name, keep, video_folder_path))
            total_files += 1

    for job in tqdm(jobs, desc="Generate Dataset"):
        success_count, errors = process_txt_file_to_video_ytf(job)
        total_success += success_count
        total_errors += errors

    elapsed_time = time.time() - start_time
    print(f"Copied {total_success} files in {elapsed_time / 60:.2f} min, with {total_errors} errors.")


def process_txt_file_to_video_ytf(args):
    file_path, destination, hash_name, keep, image_folder_path = args
    data = read_file(file_path)
    errors = 0
    success_count = 0
    for info in data:
        try:
            img_name = info[7]
            #x_min, y_min, x_max, y_max = map(int, info[8:])
            dst_filename = f'{hash_name[:15]}#{info[0]}_{info[1]}#{info[3].split(".")[0]}_{info[4].split(".")[0]}.jpg'
            dst_path = os.path.join(destination, dst_filename)
        except ValueError:
            raise ValueError("Error in :", file_path)

        if keep and os.path.exists(dst_path):
            continue

        img_source = str(os.path.join(image_folder_path, img_name))
        shutil.copy2(img_source, dst_path)

        success_count += 1

    return success_count, errors
