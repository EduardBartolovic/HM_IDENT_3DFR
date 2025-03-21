import os
import time
from collections import defaultdict
import shutil

from tqdm import tqdm


def create_train_test_split(input_folder, output_folder, filter_strings=None):
    """
    Splits images into train and test sets, always using the first group in each class for training
    and the remaining groups for testing. Only files containing a specific string from filter_strings
    in their name are included.

    Args:
        input_folder (str): Path to the folder containing the class subfolders with images.
        output_folder (str): Path to the output folder where train and test folders will be created.
        filter_strings (list): List of strings; only files containing one of these strings in their name are included.
    """
    if filter_strings is None:
        filter_strings = []
    start_time = time.time()
    counter = 0
    os.makedirs(output_folder, exist_ok=True)
    train_folder = os.path.join(output_folder, "train")
    test_folder = os.path.join(output_folder, "validation")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Iterate over class subfolders
    for class_name in tqdm(os.listdir(input_folder), desc="Copy files"):
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        # Collect groups by hash prefix
        groups = defaultdict(list)
        for filename in os.listdir(class_path):
            if (not filter_strings or any(fstr in filename for fstr in filter_strings)) and filename.endswith((".jpg", "corr.npz", ".png", ".jpeg")):
                hash_prefix = filename[:40]
                groups[hash_prefix].append(os.path.join(class_path, filename))

        # Sort groups to ensure deterministic order
        sorted_groups = sorted(groups.items())

        # Not enough groups
        if len(sorted_groups) < 2:
            print("Not enough groups", sorted_groups[:2])
            continue

        # Use the first group for training and the rest for testing
        for idx, (hash_prefix, file_paths) in enumerate(sorted_groups):
            dest_folder = train_folder if idx == 0 else test_folder
            class_dest_folder = os.path.join(dest_folder, class_name)
            os.makedirs(class_dest_folder, exist_ok=True)

            for file_path in file_paths:
                shutil.copy(file_path, class_dest_folder)
                counter += 1

    elapsed_time = time.time() - start_time
    print(f"Train-test split created in {output_folder} for {counter} files in", round(elapsed_time/60, 2), "minutes")


if __name__ == '__main__':
    input_dir = "E:\\Download\\test_out"  # input folder path
    output_dir = "E:\\Download\\test_out_TEST"  # output folder path
    filter_angles = ["-25_0", "-15_0", "0_0", "15_0" "25_0"]  # which angles should be included
    create_train_test_split(input_dir, output_dir, filter_angles)
