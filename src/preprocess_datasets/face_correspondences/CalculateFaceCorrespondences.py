import os
import time

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from tqdm import tqdm
from multiprocessing import Pool


def create_dense_flow(src_landmarks, dst_landmarks, w, h):
    """
    Create dense flow field from dst → src using scattered landmark correspondence.
    """

    # Create grid of (x,y) coords for destination image
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid = np.vstack((grid_x.ravel(), grid_y.ravel())).T  # shape: (H*W, 2)

    # Interpolate x and y separately using griddata
    map_x = griddata(dst_landmarks, src_landmarks[:, 0], grid, method='cubic', fill_value=0)
    map_y = griddata(dst_landmarks, src_landmarks[:, 1], grid, method='cubic', fill_value=0)

    # Reshape to image shape
    map_x = map_x.reshape((h, w)).astype(np.float32)
    map_y = map_y.reshape((h, w)).astype(np.float32)

    return np.array((map_x, map_y))


def calculate_face_correspondences_between_two_faces2(source_landmarks, target_landmarks):
    w, h = 112, 112
    grid = create_dense_flow(source_landmarks, target_landmarks, w, h)
    return grid


def tps_transform(source_points, target_points, grid_x, grid_y, smooth=0.0):
    """
    Perform Thin-Plate Spline Transformation
    """
    rbf_x = Rbf(source_points[:, 0], source_points[:, 1], target_points[:, 0], function='thin_plate', smooth=smooth)
    rbf_y = Rbf(source_points[:, 0], source_points[:, 1], target_points[:, 1], function='thin_plate', smooth=smooth)

    warped_x = rbf_x(grid_x, grid_y)
    warped_y = rbf_y(grid_x, grid_y)
    return warped_x, warped_y


def calculate_face_correspondences_between_two_faces(source_landmarks, target_landmarks):
    w, h = 112, 112

    source_landmarks = np.array([[x * w, y * h] for x, y in source_landmarks])
    target_landmarks = np.array([[x * w, y * h] for x, y in target_landmarks])

    extra_points = np.array([
        [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],  # Corners of the image
        [w // 2, 0], [w // 2, h - 1],  # Mid-top and mid-bottom
        [0, h // 2], [w - 1, h // 2]  # Mid-left and mid-right
    ])
    source_landmarks = np.vstack([source_landmarks, extra_points])
    target_landmarks = np.vstack([target_landmarks, extra_points])

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    warped_x, warped_y = tps_transform(source_landmarks, target_landmarks, grid_x, grid_y, smooth=0.3)

    warped_x, warped_y = torch.Tensor(warped_x), torch.Tensor(warped_y)

    map_x = torch.clamp(warped_x, 0, w - 1)
    map_y = torch.clamp(warped_y, 0, h - 1)

    map_x = 2.0 * map_x / (w - 1) - 1.0
    map_y = 2.0 * map_y / (h - 1) - 1.0
    grid = torch.stack((map_x, map_y), dim=-1)  # Shape: (H, W, 2)

    return grid


def process_file_paths(file_paths, draw=False):
    perspectives = [os.path.basename(file_path)[40:-10] for file_path in file_paths]
    npzs = [np.load(file_path.replace(".jpg", ".npz").replace(".png", ".npz"))["landmarks"] for file_path in file_paths]

    zero_position = np.where(np.array(perspectives) == '0_0')[0][0]

    for v in range(len(file_paths)):
        if v == zero_position:  # Skip alignment if zero pose or merged features
            grid = calculate_face_correspondences_between_two_faces2(npzs[v], npzs[v])
            np.savez_compressed(file_paths[v].replace("_image.npz", "_corr.npz"), corr=grid)
        else:
            grid = calculate_face_correspondences_between_two_faces2(npzs[zero_position], npzs[v])
            np.savez_compressed(file_paths[v].replace("_image.npz", "_corr.npz"), corr=grid)
            if draw:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(grid[:, :, 0], cmap='viridis')
                axes[0].set_title("Slice 1")
                axes[0].axis('off')
                axes[1].imshow(grid[:, :, 1], cmap='viridis')
                axes[1].set_title("Slice 2")
                axes[1].axis('off')
                plt.tight_layout()
                plt.show()
        # print('Done', file_paths)
    return 0


def calculate_face_correspondences_dataset(dataset_folder, keep=True, processes=4, filter_keywords=None, target_views=25):
    start_time = time.time()
    data = []
    for class_name in os.listdir(dataset_folder):
        class_path = os.path.join(dataset_folder, class_name)
        if os.path.isdir(class_path):
            sha_groups = {}
            for filename in os.listdir(class_path):
                if not filename.endswith("_image.npz"):
                    continue

                # Apply filter if keywords are provided
                if filter_keywords and not any(keyword == filename[40:-10] for keyword in filter_keywords):
                    continue
                # Skip if correspondence already exists
                corr_path = os.path.join(class_path, filename.replace("_image.npz", "_corr.npz"))
                if keep and os.path.exists(corr_path):
                    continue

                file_path = os.path.join(class_path, filename)
                if os.path.isfile(file_path):
                    sha_hash = filename[:40]  # Extract SHA hash from filename
                    if sha_hash not in sha_groups:
                        sha_groups[sha_hash] = []
                    sha_groups[sha_hash].append(file_path)

            # Append each grouped data point to the dataset
            for sha_hash, file_paths in sha_groups.items():
                if len(file_paths) == target_views:
                    data.append(file_paths)
                else:
                    print("skipped:", file_paths)

    with Pool(processes=processes) as p, tqdm(total=len(data), desc="Creating Face Correspondences") as pbar:
        for _ in p.imap(process_file_paths, data):
            pbar.update()
            pbar.refresh()

    elapsed_time = time.time() - start_time
    print(f"Created Face Correspondences in {dataset_folder} in", round(elapsed_time / 60, 2), "minutes")


def calculate_face_landmarks_dataset(dataset_folder, keep=True):
    start_time = time.time()
    counter = 0
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.1, refine_landmarks=True)

    class_names = os.listdir(dataset_folder)
    failed_landmark_counter = 0
    for class_name in tqdm(class_names, desc="Processing Classes"):
        class_path = os.path.join(dataset_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                npz_path = os.path.join(class_path, (filename[:-4] + '.npz'))
                if keep and os.path.exists(npz_path):
                    continue  # Skip if landmarks already exist
                image = cv2.imread(os.path.join(class_path, filename))
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                h, w, _ = image.shape
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    points = np.array([[lm.x, lm.y] for lm in landmarks])
                    np.savez_compressed(os.path.join(class_path, (filename[:-4] + '.npz')), landmarks=points)
                    counter += 1
                else:
                    points = np.load(os.path.join(os.path.join(os.path.dirname(__file__)), "default_landmarks.npz"))['landmarks']
                    np.savez_compressed(os.path.join(class_path, (filename[:-4] + '.npz')), landmarks=points)
                    failed_landmark_counter += 1
                    #print(f"No landmarks found for: {os.path.join(class_path, filename)} using default landmarks")
                    #raise Exception(f"No landmarks found for: {os.path.join(class_path, filename)}")

    elapsed_time = time.time() - start_time
    print(f"Created landmarks in {dataset_folder} for {counter} images, {failed_landmark_counter} failed,  in", round(elapsed_time/60, 2), "minutes")
