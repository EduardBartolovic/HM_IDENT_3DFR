import hashlib
import os
import time

from tqdm import tqdm

from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import process_txt_file_to_video_voxceleb_nersemble


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
        success_count, errors = process_txt_file_to_video_voxceleb_nersemble(job)
        total_success += success_count
        total_errors += errors

    elapsed_time = time.time() - start_time
    print(f"Copied {total_success} files in {elapsed_time / 60:.2f} min, with {total_errors} errors.")
