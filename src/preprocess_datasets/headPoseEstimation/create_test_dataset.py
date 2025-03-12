import os
from collections import defaultdict
import shutil


def create_train_test_split(input_folder, output_folder):
    """
    Splits images into train and test sets, always using the first group in each class for training
    and the remaining groups for testing.

    Args:
        input_folder (str): Path to the folder containing the class subfolders with images.
        output_folder (str): Path to the output folder where train and test folders will be created.
    """
    counter = 0
    os.makedirs(output_folder, exist_ok=True)
    train_folder = os.path.join(output_folder, "train")
    test_folder = os.path.join(output_folder, "validation")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Iterate over class subfolders
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        # Collect groups by hash prefix
        groups = defaultdict(list)
        for filename in os.listdir(class_path):
            if filename.endswith((".jpg", ".png", ".jpeg", '.npz')):  # Adjust extensions as needed
                hash_prefix = filename[:40]
                groups[hash_prefix].append(os.path.join(class_path, filename))

        # Sort groups to ensure deterministic order
        sorted_groups = sorted(groups.items())

        # Not enough groups
        if len(sorted_groups) < 2:
            print(sorted_groups[:2])
            continue

        # Use the first group for training and the rest for testing
        for idx, (hash_prefix, file_paths) in enumerate(sorted_groups):
            dest_folder = train_folder if idx == 0 else test_folder
            class_dest_folder = os.path.join(dest_folder, class_name)
            os.makedirs(class_dest_folder, exist_ok=True)

            for file_path in file_paths:
                shutil.copy(file_path, class_dest_folder)
                counter += 1

    print(f"Train-test split created in {output_folder} for {counter} files")


if __name__ == '__main__':
    input_folder = "E:\\Download\\test_out"  # Replace with your input folder path
    output_folder = "E:\\Download\\test_out_TEST"  # Replace with your output folder path
    create_train_test_split(input_folder, output_folder)