import os
import time

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import Rbf
from scipy.spatial import KDTree
from tqdm import tqdm





# Create a grid of points over the image
def add_points_to_empty_areas(points1, points2, h, w, density=20, min_distance=10):
    """
    Adds points to the image in areas without landmarks.
    - points: Existing landmark points (Nx2 array).
    - h, w: Image height and width.
    - density: Number of points per row/column in the grid.
    - min_distance: Minimum distance between landmarks and new points.
    """
    # Create a grid of points
    y, x = np.linspace(0, h, density), np.linspace(0, w, density)
    grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    # Use a KDTree to filter out points near existing landmarks
    tree = KDTree(np.concatenate([points1, points2]))
    new_points = []
    for point in grid:
        dist, _ = tree.query(point)
        if dist > min_distance:
            new_points.append(point)

    return np.vstack([points1, new_points])

def calculate_face_correspondences_prototype(face_mesh, image1, image2, draw_corr=False):

    results1 = face_mesh.process(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    results2 = face_mesh.process(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    if results1.multi_face_landmarks and results2.multi_face_landmarks:
        landmarks1 = results1.multi_face_landmarks[0].landmark
        landmarks2 = results2.multi_face_landmarks[0].landmark

        h1, w1, _ = image1.shape
        h2, w2, _ = image2.shape

        points1 = np.array([[lm.x * w1, lm.y * h1] for lm in landmarks1])
        points2 = np.array([[lm.x * w2, lm.y * h2] for lm in landmarks2])

        points1_filled = add_points_to_empty_areas(points1, points2, h1, w1, density=50, min_distance=5)
        points2_filled = add_points_to_empty_areas(points2, points1, h2, w2, density=50, min_distance=5)

        # Compute the TPS transformation and Warp the image
        grid_x, grid_y = np.meshgrid(np.arange(w2), np.arange(h2))
        warped_x, warped_y = tps_transform(points1_filled, points2_filled, grid_x, grid_y, smooth=0.5)
        map_x = np.clip(warped_x, 0, w1 - 1).astype(np.float32)
        map_y = np.clip(warped_y, 0, h1 - 1).astype(np.float32)
        warped_image2 = cv2.remap(image2, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        # Plot corresponding landmarks -------------------------------
        if draw_corr:
            points1 = np.array([(int(lm.x * w1), int(lm.y * h1)) for lm in landmarks1])
            points2 = np.array([(int(lm.x * w2), int(lm.y * h2)) for lm in landmarks2])

            #print(f"Nose tip in image 1: {points1[1]}, image 2: {points2[1]}")
            combined_width = w1 + w2
            combined_height = max(h1, h2)
            canvas = 255 * np.ones((combined_height, combined_width, 3), dtype=np.uint8)
            canvas[:h1, :w1] = image1
            canvas[:h2, w1:] = image2

            for p1, p2 in zip(points1, points2):
                cv2.circle(canvas, p1, 1, (0, 255, 0), -1)
                cv2.circle(canvas, (p2[0] + w1, p2[1]), 1, (0, 255, 0), -1)

            important_indices = [1, 33, 61, 199, 263, 291]
            important_points1 = [points1[i] for i in important_indices]
            important_points2 = [points2[i] for i in important_indices]
            for p1, p2 in zip(important_points1, important_points2):
                cv2.circle(canvas, p1, 3, (255, 255, 255), -1)
                cv2.circle(canvas, (p2[0] + w1, p2[1]), 2, (255, 255, 255), -1)
                cv2.line(canvas, p1, (p2[0] + w1, p2[1]), (255, 0, 0), 1)

            plt.figure(figsize=(12, 6))
            plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("Corresponding Landmarks Between Two Images")
            plt.show()
        # ---------------------------------------

            plt.figure(figsize=(24, 12))
            plt.subplot(1, 4, 1)
            plt.title("Original Image 1")
            plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.subplot(1, 4, 2)
            plt.title("Warped Image 2 (Aligned)")
            plt.scatter(points1[:, 0], points1[:, 1], s=0.4, c='red', label='Source Landmarks')
            #plt.scatter(points2[:, 0], points2[:, 1], s=0.4, c='blue', label='Target Landmarks')
            plt.imshow(cv2.cvtColor(warped_image2, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.subplot(1, 4, 3)
            plt.title("Original Image 2")
            plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.savefig('C:\\Users\\Eduard\\Desktop\\Face\\merge.png')
            plt.show()

    else:
        raise ValueError("Face landmarks not detected in one or both images.")

# Thin-Plate Spline Transformation
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
    w, h = 112,112

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


def calculate_face_correspondences_dataset(dataset_folder, draw=False):
    data = []
    for class_name in os.listdir(dataset_folder):
        class_path = os.path.join(dataset_folder, class_name)
        if os.path.isdir(class_path):
            sha_groups = {}
            for filename in os.listdir(class_path):
                if filename.endswith("_image.npz"):
                    file_path = os.path.join(class_path, filename)
                    if os.path.isfile(file_path):
                        sha_hash = filename[:40]  # Extract SHA hash from filename
                        if sha_hash not in sha_groups:
                            sha_groups[sha_hash] = []
                        sha_groups[sha_hash].append(file_path)

            # Append each grouped data point to the dataset
            for sha_hash, file_paths in sha_groups.items():
                if len(file_paths) == 25:
                    data.append(file_paths)

    for file_paths in tqdm(data, desc="Generate Face Correspondences"):
        perspectives = [os.path.basename(file_path)[40:-10] for file_path in file_paths]
        npzs = [np.load(file_path.replace(".jpg", ".npz").replace(".png", ".npz"))["landmarks"] for file_path in file_paths]

        zero_position = np.where(np.array(perspectives) == '0_0')[0][0]

        for v in range(25):
            if v == zero_position: # Skip alignment if zero pose or merged features
                grid = calculate_face_correspondences_between_two_faces(npzs[v], npzs[v])
                np.savez_compressed(file_paths[v].replace("_image.npz", "_corr.npz"), corr=grid)
            else:
                grid = calculate_face_correspondences_between_two_faces(npzs[zero_position], npzs[v])
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


def calculate_face_landmarks_dataset(dataset_folder):
    start_time = time.time()
    counter = 0
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # Iterate over class subfolders
    folder = list(os.listdir(dataset_folder))
    for class_name in tqdm(folder, desc="Processing folders"):
        class_path = os.path.join(dataset_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                image = cv2.imread(os.path.join(class_path, filename))
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                h, w, _ = image.shape
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    points = np.array([[lm.x, lm.y] for lm in landmarks])
                else:
                    points = np.load(os.path.join(os.path.join(os.path.dirname(__file__)), "default_landmarks.npz"))['landmarks']        # TODO: Temporary solution for bad dataset
                    print(f"No landmarks found for: {os.path.join(class_path, filename)} using default landmarks")
                    #raise Exception(f"No landmarks found for: {os.path.join(class_path, filename)}")

                np.savez_compressed(os.path.join(class_path, (filename[:-4]+'.npz')), landmarks=points)
                counter += 1

    elapsed_time = time.time() - start_time
    print(f"Created landmarks in {dataset_folder} for {counter} images in", round(elapsed_time/60, 2), "minutes")


if __name__ == '__main__':

    dataset = "C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test_dataset_TEST\\validation"
    #calculate_face_landmarks_dataset(dataset)
    #calculate_face_correspondences_dataset(dataset)
    dataset = "C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test_dataset_TEST\\train"
    #calculate_face_landmarks_dataset(dataset)
    #calculate_face_correspondences_dataset(dataset)

    # raise Exception("")
    #
    # image1_ori = cv2.imread("C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test_dataset_TEST\\validation\\id10270\\f03463d2c721f7f05304c9826ef170c99b8a78850_0_image.jpg")
    # image1_ori = cv2.cvtColor(image1_ori, cv2.COLOR_BGR2RGB)
    # image2_ori = cv2.imread("C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test_dataset_TEST\\validation\\id10270\\f03463d2c721f7f05304c9826ef170c99b8a788525_-25_image.jpg")
    # image2_ori = cv2.cvtColor(image2_ori, cv2.COLOR_BGR2RGB)
    #
    # ori_landmarks1 = np.load("C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test_dataset_TEST\\validation\\id10270\\f03463d2c721f7f05304c9826ef170c99b8a78850_0_image.npz")["landmarks"]
    # ori_landmarks2 = np.load("C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test_dataset_TEST\\validation\\id10270\\f03463d2c721f7f05304c9826ef170c99b8a788525_-25_image.npz")["landmarks"]
    #
    # ori_grid = calculate_face_correspondences_between_two_faces(ori_landmarks2, ori_landmarks1)

    # h1, w1, _ = image1_ori.shape
    # h2, w2, _ = image2_ori.shape
    #
    # points1 = np.array([[x * w1, y * h1] for x,y in ori_landmarks1])
    # points2 = np.array([[x * w2, y * h2] for x,y in ori_landmarks2])
    #
    # extra_points = np.array([
    #     [0, 0], [w1 - 1, 0], [0, h1 - 1], [w1 - 1, h1 - 1],  # Corners of the image
    #     [w1 // 2, 0], [w1 // 2, h1 - 1],  # Mid-top and mid-bottom
    #     [0, h1 // 2], [w1 - 1, h1 // 2]  # Mid-left and mid-right
    # ])
    # points1 = np.vstack([points1, extra_points])
    # points2 = np.vstack([points2, extra_points])
    #
    # grid_x, grid_y = np.meshgrid(np.arange(w2), np.arange(h2))
    # warped_x, warped_y = tps_transform(points1, points2, grid_x, grid_y, smooth=0.3)
    #
    # warped_x, warped_y = torch.Tensor(warped_x), torch.Tensor(warped_y)
    #
    # map_x = torch.clamp(warped_x, 0, w1 - 1)
    # map_y = torch.clamp(warped_y, 0, h1 - 1)
    #
    # map_x = 2.0 * map_x / (w1 - 1) - 1.0
    # map_y = 2.0 * map_y / (h1 - 1) - 1.0
    # grid = torch.stack((map_x, map_y), dim=-1)  # Shape: (H, W, 2)

    # # Add batch dimension and permute to (B, H, W, 2)
    # grid = grid.unsqueeze(0)
    #
    # # Perform grid sampling
    # image2 = transforms.ToTensor()(image2_ori)
    # image2 = transforms.Resize((112,112))(image2)
    # image2 = image2.unsqueeze(0)  # Add batch dimension
    # warped_image = F.grid_sample(image2, grid, mode='bilinear', align_corners=True).squeeze(0)
    # warped_image = np.transpose(warped_image.numpy(), (1, 2, 0))
    #
    #
    # #map_x = np.clip(warped_x, 0, w1 - 1).astype(np.float32)
    # #map_y = np.clip(warped_y, 0, h1 - 1).astype(np.float32)
    # #warped_image2 = cv2.remap(image2_ori, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    #
    # fig, axes = plt.subplots(1, 4, figsize=(10, 5))
    # axes[0].imshow(image1_ori)
    # axes[0].axis('off')
    # axes[1].imshow(warped_image)
    # axes[1].axis('off')
    # #axes[2].imshow(warped_image2)
    # axes[2].axis('off')
    # axes[3].imshow(image2_ori)
    # axes[3].axis('off')
    # plt.tight_layout()
    # plt.show()
    # raise Exception()

    from torchvision.transforms import transforms
    import torch.nn.functional as F

    image1_ori = cv2.imread("C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test_dataset_TEST\\validation\\id10270\\f03463d2c721f7f05304c9826ef170c99b8a78850_0_image.jpg")
    image1_ori = cv2.cvtColor(image1_ori, cv2.COLOR_BGR2RGB)
    image2_ori = cv2.imread("C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test_dataset_TEST\\validation\\id10270\\f03463d2c721f7f05304c9826ef170c99b8a788525_-25_image.jpg")
    image2_ori = cv2.cvtColor(image2_ori, cv2.COLOR_BGR2RGB)
    image2 = transforms.ToTensor()(image2_ori)
    image2 = transforms.Resize((112,112))(image2)

    ori_grid = np.load("C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test_dataset_TEST\\validation\\id10270\\f03463d2c721f7f05304c9826ef170c99b8a788525_-25_corr.npz")["corr"]
    grid = torch.Tensor(ori_grid)
    print(grid.shape[:2], image2.shape[1:])
    if grid.shape[:2] != image2.shape[1:]:
        print("RESIZE")
        _, target_height, target_width = image2.shape
        # Reshape grid for interpolation
        grid = grid.permute(2, 0, 1)  # [1, 2, H, W]
        # Resize using bilinear interpolation
        grid = F.interpolate(grid, size=(target_height, target_width), mode='bilinear', align_corners=True)
        # Reshape back to grid format [1, target_height, target_width, 2]
        grid = grid.permute(1, 2, 0)

    grid = grid.unsqueeze(0)

    image2 = transforms.ToTensor()(image2_ori)
    image2 = transforms.Resize((112,112))(image2)
    image2 = image2.unsqueeze(0)  # Add batch dimension
    warped_image = F.grid_sample(image2, grid, mode='bilinear', align_corners=True).squeeze(0)
    warped_image = np.transpose(warped_image.numpy(), (1, 2, 0))

    fig, axes = plt.subplots(1, 5, figsize=(10, 5))
    axes[0].imshow(ori_grid[:, :, 0], cmap='viridis')
    axes[0].set_title("Slice 1")
    axes[0].axis('off')
    axes[1].imshow(ori_grid[:, :, 1], cmap='viridis')
    axes[1].set_title("Slice 2")
    axes[1].axis('off')
    axes[2].imshow(warped_image)
    axes[2].set_title("Slice 2")
    axes[2].axis('off')
    axes[3].imshow(image2_ori)
    axes[3].axis('off')
    axes[4].imshow(image1_ori)
    axes[4].axis('off')
    plt.tight_layout()
    plt.show()