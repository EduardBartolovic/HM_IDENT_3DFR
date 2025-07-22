import hashlib

import os
import time
from collections import defaultdict
import shutil

from tqdm import tqdm


def sanity_check(folder_path, views):
    """
       Checks if the number of files in the given folder is a multiple of `multiple_of`.

       Args:
           folder_path (str): Path to the folder.
           views (int): The number to check multiplicity against.

       Returns:
           bool: True if file count is a multiple of `multiple_of`, False otherwise.
       """
    try:
        # List all files (ignoring directories)
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        file_count = len(files)
        print(f"Found {file_count} files in '{folder_path}'.")
        return file_count % views == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def unique_views_score(file_paths):
    """
    Counts the number of unique images based on file content.
    """
    def hash_file(path):
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    unique_hashes = {hash_file(f) for f in file_paths}
    return len(unique_hashes)


def create_train_test_split(input_folder, output_folder, filter_strings=None, poses=25, ignore_face_corr=True):
    """
    Splits images into enrolled and query sets, always using the first group in each class for training
    and the remaining groups for testing. Only files containing a specific string from filter_strings
    in their name are included.

    Args:
        ignore_face_corr:
        input_folder (str): Path to the folder containing the class subfolders with images.
        output_folder (str): Path to the output folder where train and test folders will be created.
        filter_strings (list): List of strings; only files containing one of these strings in their name are included.
        poses (int): The required number of poses per group for inclusion.
    """
    if filter_strings is None:
        filter_strings = []

    if ignore_face_corr:
        file_ext = (".jpg", ".png", ".jpeg")
    else:
        poses = poses*2
        file_ext = (".jpg", "corr.npz", ".png", ".jpeg")
    start_time = time.time()
    counter = 0
    os.makedirs(output_folder, exist_ok=True)
    train_folder = os.path.join(output_folder, "enrolled")
    test_folder = os.path.join(output_folder, "query")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    ignored = 0

    for class_name in tqdm(os.listdir(input_folder), desc="Copy files"):
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        # Collect groups by hash prefix
        groups = defaultdict(list)
        for filename in os.listdir(class_path):
            if (not filter_strings or any(fstr in filename for fstr in filter_strings)) and filename.endswith(file_ext):
                hash_prefix = filename[:40]
                groups[hash_prefix].append(os.path.join(class_path, filename))

        # Remove groups that do not match the required number of poses
        filtered_groups = {k: v for k, v in groups.items() if len(v) == poses}
        ignored += len(groups) - len(filtered_groups)
        groups = filtered_groups

        # Sanity check: Ensure all groups have the same number of perspectives
        if len(set(len(v) for v in groups.values())) > 1:
            raise ValueError(
                f"Inconsistent number of perspectives in class {class_name}: {[len(v) for v in groups.values()]}")

        # Sort groups to ensure deterministic order
        sorted_groups = sorted(groups.items())

        if len(sorted_groups) < 2:
            print("Not enough groups", sorted_groups[:2])
            continue

        scored_groups = [(hash_prefix, file_paths, unique_views_score(file_paths)) for hash_prefix, file_paths in sorted_groups]
        scored_groups.sort(key=lambda x: x[2], reverse=True)

        # Use the best group for train and the rest for testing
        for idx, (hash_prefix, file_paths, _) in enumerate(scored_groups):
            if len(file_paths) != poses:
                assert f"views {len(file_paths)} dont match required {poses} poses"

            dest_folder = train_folder if idx == 0 else test_folder
            class_dest_folder = os.path.join(dest_folder, class_name)
            os.makedirs(class_dest_folder, exist_ok=True)

            for file_path in file_paths:
                shutil.copy(file_path, class_dest_folder)
                counter += 1

    elapsed_time = time.time() - start_time
    print(f"Train-test split created in {output_folder}. {ignored} ignored groups in {counter} files in", round(elapsed_time/60, 2), "minutes")
    sanity_check(output_folder, views=poses)
    print("Sanity Check completed successfully")
