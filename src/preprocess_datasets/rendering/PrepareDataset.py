import hashlib
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from torchvision import datasets
import torch
from torchvision.transforms import transforms

from src.preprocess_datasets.detect_face import cut_face
from src.preprocess_datasets.headPoseEstimation.models.scrfd import SCRFD


def collect_data_files_depth(input_path):
    file_paths = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith('_depth.jpg') or file.endswith('_depth.png'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    print('Collected:', len(file_paths), 'Photos')
    return file_paths


def prepare_dataset_depth(input_path, output_dir, mode=''):
    print('input_dir:', input_path)
    print('output_dir:', output_dir)
    file_paths = collect_data_files_depth(input_path)

    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    model_cache = []
    set_cache = []

    for p in file_paths:
        splited_path = Path(p).parts
        model = splited_path[-4]
        scan_set = splited_path[-2] + splited_path[-3]
        file_name = hashlib.sha1((splited_path[-2] + splited_path[-3] + splited_path[-4]).encode()).hexdigest() + splited_path[-1]

        if mode == '':

            if model not in model_cache or scan_set in set_cache:
                target_dir = train_dir
                model_cache.append(model)
                set_cache.append(scan_set)
            else:
                target_dir = val_dir

        elif mode == 'facescape':

            if 'neutral' in scan_set:
                target_dir = train_dir
                model_cache.append(model)
                set_cache.append(scan_set)
            else:
                target_dir = val_dir

        # Create the target directory for the model if it doesn't exist
        target = os.path.join(target_dir, model)
        os.makedirs(target, exist_ok=True)

        # Copy the file to the target directory
        shutil.copyfile(Path(p), os.path.join(target, file_name))


def collect_data_files_rgb(input_path):
    file_paths = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if (file.endswith('.jpg') or file.endswith('.png')) and not (
                    file.endswith('_depth.jpg') or file.endswith('_depth.png')):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    print('Collected:', len(file_paths), 'Photos')
    return file_paths


def prepare_dataset_rgb(input_path, output_dir, mode=''):
    print('input_dir:', input_path)
    print('output_dir:', output_dir)
    file_paths = collect_data_files_rgb(input_path)

    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    model_cache = []
    set_cache = []

    for p in file_paths:
        splited_path = Path(p).parts
        scan_set = splited_path[-2] + splited_path[-3]
        model = splited_path[-4]
        file_name = hashlib.sha1((splited_path[-2] + splited_path[-3] + splited_path[-4]).encode()).hexdigest() + splited_path[-1]

        if mode == '':

            if model not in model_cache or scan_set in set_cache:
                target_dir = train_dir
                model_cache.append(model)
                set_cache.append(scan_set)
            else:
                target_dir = val_dir

        elif mode == 'facescape':

            if 'neutral' in scan_set:
                target_dir = train_dir
                model_cache.append(model)
                set_cache.append(scan_set)
            else:
                target_dir = val_dir

        # Create the target directory for the model if it doesn't exist
        target = os.path.join(target_dir, model)
        os.makedirs(target, exist_ok=True)

        # Copy the file to the target directory
        shutil.copyfile(Path(p), os.path.join(target, file_name))


def normalize_path(file_path):
    base, ext = os.path.splitext(file_path)
    if base.endswith('_depth'):
        base = base[:-6]  # Remove the '_depth' part
    if base.endswith('_image'):
        base = base[:-6]  # Remove the '_depth' part
    return base + ext


def prepare_dataset_rgbd(input_path, input_path_depth, output_dir):
    print('input_dir:', input_path)
    print('input_path_depth:', input_path_depth)
    print('output_dir:', output_dir)
    file_paths = collect_data_files_rgb(input_path)
    file_paths_depth = collect_data_files_depth(input_path_depth)

    # Normalize the file paths by removing the '_depth' suffix
    normalized_file_paths = [path.split('\\')[-1][:-9] for path in file_paths]
    normalized_file_paths_depth = [path.split('\\')[-1][:-9] for path in file_paths_depth]
    # Compute differences
    diff_files_rgb = list(set(normalized_file_paths) - set(normalized_file_paths_depth))
    diff_files_depth = list(set(normalized_file_paths_depth) - set(normalized_file_paths))
    print("Files only in RGB list:")
    for file in diff_files_rgb:
        print(file)
    print("Files only in Depth list:")
    for file in diff_files_depth:
        print(file)

    print(len(file_paths), len(file_paths_depth))
    assert len(file_paths) == len(file_paths_depth)

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for rgb, depth in tqdm(zip(file_paths, file_paths_depth), desc="Create dirs", total=len(file_paths)):
        tmp = rgb.replace('.jpg', '.png').split(os.path.sep)
        os.makedirs(os.path.join(output_dir, *tmp[-3:-1]), exist_ok=True)

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for rgb, depth in tqdm(zip(file_paths, file_paths_depth), desc="Create merge images task",
                               total=len(file_paths)):
            futures.append(executor.submit(merge_image, rgb, depth, output_dir))

        for future in tqdm(futures, desc="Processing Merge Images", total=len(futures)):
            future.result()


def merge_image(rgb, depth, output_dir):
    rgb_image = np.array(Image.open(rgb))
    depth_image = np.array(Image.open(depth).convert("L"))

    # Stack RGB and grayscale arrays to create a 4-channel image
    four_channel_image = np.dstack((rgb_image, depth_image))

    # Convert the numpy array back to PIL image
    four_channel_image_pil = Image.fromarray(four_channel_image)

    # Save the 4-channel image to disk
    tmp = rgb.replace('.jpg', '.webp').split(os.path.sep)
    image_path = os.path.join(output_dir, *tmp[-3:])
    four_channel_image_pil.save(image_path, format='WebP')


def collect_data_files_photos(input_path):
    file_paths = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    print('Collected:', len(file_paths), 'Photos')
    return file_paths


def prepare_dataset_photos(input_path, output_dir):
    print('input_dir:', input_path)
    print('output_dir:', output_dir)
    file_paths = collect_data_files_photos(input_path)

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    model_cache = []
    set_cache = []

    for p in file_paths:
        splited_path = Path(p).parts
        photo_name = splited_path[-1][6:-4]+'_photo.jpg'
        file_name = hashlib.sha1((splited_path[-2] + splited_path[-3] + splited_path[-4]).encode()).hexdigest() + '-' +photo_name
        set = splited_path[-2] + splited_path[-3]
        model = splited_path[-4]

        if model not in model_cache or set in set_cache:
            target_dir = train_dir
            model_cache.append(model)
            set_cache.append(set)
        else:
            target_dir = val_dir

        # Create the target directory for the model if it doesn't exist
        target = os.path.join(target_dir, model)
        os.makedirs(target, exist_ok=True)

        # Copy the file to the target directory
        shutil.copyfile(Path(p), os.path.join(target, file_name))


def prepare_dataset_bff(source_folders, destination_folder):

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for folder in source_folders:
        # Use the base name of each source folder as a prefix
        folder_prefix = os.path.basename(folder).split('_')[-1]
        for root, dirs, files in tqdm(os.walk(folder)):
            # Calculate the relative path of the current folder from the source folder root
            relative_path = os.path.relpath(root, folder)

            # Split the relative path to identify levels
            path_parts = relative_path.split(os.sep)
            if len(path_parts) == 2:
                path_parts[1] = f"{folder_prefix}_{path_parts[1]}"

            # Reconstruct the destination path with the modified second-level folder name
            destination_path = os.path.join(destination_folder, *path_parts)

            # Create the corresponding subfolder in the destination
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)

            # Copy each file, handling duplicates
            for file in files:
                source_file = os.path.join(root, file)
                destination_file = os.path.join(destination_path, file)

                # Handle file name conflicts by appending a counter
                if os.path.exists(destination_file):
                    raise AttributeError('File Exists')

                # Copy file to the destination path
                shutil.copy2(source_file, destination_file)


def collect_data_files_texas3d(input_path, img_type):
    file_paths = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(img_type + '.png'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    print('Collected:', len(file_paths), 'Images')
    return file_paths


def collect_data_files_colorferet(input_path):
    file_paths = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith('.ppm'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    print('Collected:', len(file_paths), 'Images')
    return file_paths


def prepare_dataset_colorferet_1_1(input_path, output_dir):
    print('input_dir:', input_path)
    print('output_dir_rgb:', output_dir)

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    file_paths = collect_data_files_colorferet(input_path)

    for p in tqdm(file_paths):
        splited_path = Path(p).parts
        file_name = splited_path[-1][:-4]
        splited_file_name = file_name.split('_')

        if len(splited_file_name) == 4:
            flag = splited_file_name[3]
        set = splited_file_name[1]
        model = splited_file_name[0]

        # Create the target directory for the model if it doesn't exist
        target = os.path.join(output_dir, model, set)
        os.makedirs(target, exist_ok=True)

        img = Image.open(Path(p))
        file_name = Path(p).stem
        target_path = os.path.join(target, f"{file_name}.jpg")
        img.save(target_path, 'JPEG')


def prepare_dataset_colorferet_1_n(input_path, output_dir):
    print('input_dir:', input_path)
    print('output_dir_rgb:', output_dir)

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    file_paths = collect_data_files_colorferet(input_path)

    face_detector = SCRFD(model_path="/src/preprocess_datasets/headPoseEstimation/weights/det_10g.onnx")

    model_cache = []
    set_cache = []
    for p in tqdm(file_paths):
        splited_path = Path(p).parts
        file_name = splited_path[-1][:-4]
        splited_file_name = file_name.split('_')

        scan = splited_file_name[1]
        model = splited_file_name[0]

        if model in model_cache and scan in set_cache:  # One set of one model exists
            target_dir = train_dir
        elif model in model_cache and scan not in set_cache:
            target_dir = val_dir
        elif model not in model_cache:
            target_dir = train_dir
            model_cache.append(model)
            set_cache.append(scan)
        else:
            raise Exception('Illegal State')

        # Create the target directory for the model if it doesn't exist
        target = os.path.join(target_dir, model)
        os.makedirs(target, exist_ok=True)

        img = Image.open(Path(p))
        file_name = hashlib.sha1((model + scan + Path(p).stem[:7]).encode()).hexdigest() + "-" + Path(p).stem[13:].replace("_a", "") + "_photo"

        cutted_face, worked_fine = cut_face(face_detector, np.array(img))
        img = Image.fromarray(cutted_face)
        if not worked_fine:
            print(Path(p))

        target_path = os.path.join(target, f"{file_name}.jpg")
        img.save(target_path, 'JPEG')

    # Get a list of folder names in train and val directories
    train_folders = set(os.listdir(train_dir))
    val_folders = set(os.listdir(val_dir))

    # Find unmatched folders
    unmatched_folders = train_folders - val_folders

    # Remove unmatched folders from train
    for folder in unmatched_folders:
        folder_path = os.path.join(train_dir, folder)
        if os.path.isdir(folder_path):  # Check if it's a folder
            shutil.rmtree(folder_path)  # Delete the folder and its contents
            print(f"Removed folder: {folder}")

    print("All unmatched folders have been removed.")


def prepare_dataset_texas3d(input_path, output_dir_rgb, output_dir_depth):
    print('input_dir:', input_path)
    print('output_dir_rgb:', output_dir_rgb)
    print('output_dir_depth:', output_dir_depth)

    os.makedirs(output_dir_rgb, exist_ok=True)
    os.makedirs(output_dir_depth, exist_ok=True)

    train_dir_rgb = os.path.join(output_dir_rgb, 'train')
    val_dir_rgb = os.path.join(output_dir_rgb, 'validation')
    os.makedirs(train_dir_rgb, exist_ok=True)
    os.makedirs(val_dir_rgb, exist_ok=True)

    train_dir_depth = os.path.join(output_dir_depth, 'train')
    val_dir_depth = os.path.join(output_dir_depth, 'validation')
    os.makedirs(train_dir_depth, exist_ok=True)
    os.makedirs(val_dir_depth, exist_ok=True)

    file_paths_rgb = collect_data_files_texas3d(input_path, img_type='Portrait')

    model_cache = []
    set_cache = []
    for p in file_paths_rgb:
        splited_path = Path(p).parts
        file_name = splited_path[-1]
        splited_file_name = file_name.split('_')

        set = splited_file_name[1]
        model = splited_file_name[2]

        # Models with only one image get ignored:
        if model in ['001', '002', '006', '007', '008', '014', '035', '069', '076', '085', '087', '102', '105', '106',
                     '118']:
            continue

        if model not in model_cache or set in set_cache:
            target_dir = train_dir_rgb
            model_cache.append(model)
            set_cache.append(set)
        else:
            target_dir = val_dir_rgb

        # Create the target directory for the model if it doesn't exist
        target = os.path.join(target_dir, model)
        os.makedirs(target, exist_ok=True)

        # Copy the file to the target directory
        shutil.copyfile(Path(p), os.path.join(target, file_name))

    file_paths_depth = collect_data_files_texas3d(input_path, img_type='Range')

    model_cache = []
    set_cache = []
    for p in file_paths_depth:
        splited_path = Path(p).parts
        file_name = splited_path[-1]
        splited_file_name = file_name.split('_')

        set = splited_file_name[1]
        model = splited_file_name[2]

        # Models with only one image get ignored:
        if model in ['001', '002', '006', '007', '008', '014', '035', '069', '076', '085', '087', '102', '105', '106',
                     '118']:
            continue

        if model not in model_cache or set in set_cache:
            target_dir = train_dir_depth
            model_cache.append(model)
            set_cache.append(set)
        else:
            target_dir = val_dir_depth

        # Create the target directory for the model if it doesn't exist
        target = os.path.join(target_dir, model)
        os.makedirs(target, exist_ok=True)

        # Copy the file to the target directory
        shutil.copyfile(Path(p), os.path.join(target, file_name.replace('_Range', '_depth')))


def prepare_datasets_test(dir_path):
    entries = os.listdir(dir_path)
    print(entries)

    for entry in entries:
        if "test_" in entry and not entry == 'test_colorferet' and not entry == 'test_lfw_deepfunneled':
            print(f"Processing {entry}...")
            entry_path = os.path.join(dir_path, entry)
            target = os.path.join(dir_path, entry.replace("test_", ""))
            if not os.path.exists(target):
                os.makedirs(target, exist_ok=True)

                for dataset_type in ['enrolled', 'query']:
                    dataset_path = os.path.join(entry_path, dataset_type)
                    class_dirs = os.listdir(dataset_path)

                    for class_dir in class_dirs:
                        class_dir_path = os.path.join(dataset_path, class_dir)
                        target_class_dir_path = os.path.join(target, class_dir)

                        # Create class dir in the target if it doesn't exist
                        os.makedirs(target_class_dir_path, exist_ok=True)

                        # Copy or move files from the test_ dirs to the target dirs
                        image_files = os.listdir(class_dir_path)
                        for image_file in image_files:
                            src_file_path = os.path.join(class_dir_path, image_file)
                            dest_file_path = os.path.join(target_class_dir_path, image_file)
                            shutil.copy(src_file_path, dest_file_path)


class ImageFolderWithScanID(datasets.ImageFolder):
    """Custom dataset that includes scan_id and perspective. Extends torchvision.datasets.ImageFolder"""

    def __getitem__(self, index):
        # This is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithScanID, self).__getitem__(index)
        image_path = Path(self.imgs[index][0])
        filename = image_path.name
        scan_id = filename[:40]  # sha-1 hash is always 40 of length
        perspective = filename[40:-10]
        # Make a new tuple that includes original and the scan_id, perspective
        data_tuple = (original_tuple + (scan_id, perspective))
        return data_tuple


def load_data(data_dir, transform, batch_size: int, shuffle=True):
    dataset = ImageFolderWithScanID(root=data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6,
                                              drop_last=False)
    return dataset, data_loader


def sanity_check(dir_path):
    entries = os.listdir(dir_path)
    print(entries)

    for dataset in entries:
        print('testing:', dataset)

        input_size = [112, 112]
        test_transform = transforms.Compose([
            # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        dataset_path = os.path.join(dir_path, dataset)
        dataset, loader = load_data(dataset_path, test_transform, 32)

        scan_id_label_map = {}
        scan_id_perspective_map = {}

        for images, labels, scan_id, perspective in tqdm(loader, desc="Check Hashs for Collisions"):
            labels_np = labels.cpu().numpy()
            scan_id = np.array(scan_id)
            perspective = np.array(perspective)

            for scan, label, persp in zip(scan_id, labels_np, perspective):
                if scan in scan_id_label_map:
                    if scan_id_label_map[scan] != label:
                        print(f"Scan ID {scan} is used with different labels: {scan_id_label_map[scan]} and {label}")
                else:
                    scan_id_label_map[scan] = label

                if scan in scan_id_perspective_map:
                    if persp in scan_id_perspective_map[scan]:
                        print(f"Scan ID {scan} has duplicated perspective: {persp}")
                    else:
                        scan_id_perspective_map[scan].append(persp)
                else:
                    scan_id_perspective_map[scan] = [persp]

        # Print the number of perspectives per scan
        for scan, perspectives in scan_id_perspective_map.items():
            if len(perspectives) == 9 or len(perspectives) == 25:
                continue
            print(f"Scan ID {scan} has {len(perspectives)} unique perspectives.")


def filter_views(dataset_folder, output_folder, filter_keywords, target_views=8):
    """
    Filters images in dataset_folder based on keywords, and only copies scans where the
    required keywords are fully available. A scan is identified by the first 40 characters
    of the filename.

    Args:
        dataset_folder (str): Path to dataset.
        filter_keywords (list): List of keywords to filter filenames.
        target_views (int): Required number of views or multiple.
        output_folder (str): Destination for filtered images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_scans = 0
    copied_scans = 0
    skipped_scans = 0

    for class_name in tqdm(os.listdir(dataset_folder), desc="Filtering Dataset"):
        class_path = os.path.join(dataset_folder, class_name)
        if os.path.isdir(class_path):

            # group files by scan_id = first 40 chars
            scan_groups = {}
            for f in os.listdir(class_path):
                scan_id = f[:20]
                scan_groups.setdefault(scan_id, []).append(f)

            output_class_path = os.path.join(output_folder, class_name)
            os.makedirs(output_class_path, exist_ok=True)

            for scan_id, files in scan_groups.items():
                total_scans += 1
                # pick only files whose keyword is in filter_keywords
                filtered_files = [
                    f for f in files if f[20:-3].split("#")[1] in filter_keywords
                ]
                keywords_present = set(f[20:-3].split("#")[1] for f in filtered_files)

                # check completeness
                if all(k in keywords_present for k in filter_keywords):
                    assert len(filtered_files) % target_views == 0
                    # copy only filtered files
                    for filename in filtered_files:
                        src_file = os.path.join(class_path, filename)
                        dst_file = os.path.join(output_class_path, filename)
                        shutil.copy2(src_file, dst_file)
                    copied_scans += 1
                else:
                    skipped_scans += 1
                    missing = [k for k in filter_keywords if k not in keywords_present]
                    print(f"⚠️ Scan {scan_id} in class {class_name} skipped (missing keywords {missing})")

    print("\n📊 Processing Summary")
    print(f"  ➤ Total scans processed: {total_scans}")
    print(f"  ➤ Scans copied:         {copied_scans}")
    print(f"  ➤ Scans skipped:        {skipped_scans}")
    print(f"\n✅ Filtered dataset saved in: {output_folder}")
