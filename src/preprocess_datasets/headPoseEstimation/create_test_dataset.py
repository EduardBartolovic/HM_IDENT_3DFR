import os
import time
from collections import defaultdict
import shutil

from tqdm import tqdm


def create_train_test_split(input_folder, output_folder, filter_strings=None, poses=25, ignore_face_corr=True):
    """
    Splits images into train and test sets, always using the first group in each class for training
    and the remaining groups for testing. Only files containing a specific string from filter_strings
    in their name are included.

    Args:
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
    train_folder = os.path.join(output_folder, "train")
    test_folder = os.path.join(output_folder, "validation")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    ignored = 0

    # Iterate over class subfolders
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

        # Sort groups to ensure deterministic order
        sorted_groups = sorted(groups.items())

        # Not enough groups
        if len(sorted_groups) < 2:
            print("Not enough groups", sorted_groups[:2])
            continue

        # Use the first group for training and the rest for testing
        for idx, (hash_prefix, file_paths) in enumerate(sorted_groups):
            if len(file_paths) != poses:
                ignored += 1
                continue  # Skip groups that do not match the required number of poses

            dest_folder = train_folder if idx == 0 else test_folder
            class_dest_folder = os.path.join(dest_folder, class_name)
            os.makedirs(class_dest_folder, exist_ok=True)

            for file_path in file_paths:
                shutil.copy(file_path, class_dest_folder)
                counter += 1

    elapsed_time = time.time() - start_time
    print(f"Train-test split created in {output_folder}. {ignored} groups in {counter} files in", round(elapsed_time/60, 2), "minutes")


if __name__ == '__main__':
    input_dir = "E:\\Download\\test_out"  # input folder path
    output_dir = "E:\\Download\\test_out_TEST"  # output folder path
    filter_angles = ["-25_0", "-15_0", "0_0", "15_0" "25_0"]  # which angles should be included
    create_train_test_split(input_dir, output_dir, filter_angles)
