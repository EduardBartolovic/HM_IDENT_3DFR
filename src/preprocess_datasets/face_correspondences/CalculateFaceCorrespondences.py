import os
from collections import defaultdict

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial import KDTree
from torchvision.transforms import transforms

from src.backbone.model_irse import IR_50


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

def calculate_face_correspondences(face_mesh, image1, image2, draw_corr=False):

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


def calculate_face_correspondences_dataset(dataset_folder):

    counter = 0
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # train_transform = transforms.Compose([
    #     transforms.Resize((150, 150)),
    #     transforms.CenterCrop((112, 112)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])
    # model = IR_50([112,112], 512)
    # model = model.to("cuda")
    # model.eval()

    # Iterate over class subfolders
    for class_name in os.listdir(dataset_folder):
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
                    #points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
                    points = np.array([[lm.x, lm.y] for lm in landmarks])
                else:
                    points = np.load(os.path.join(os.path.join(os.path.dirname(__file__)), "default_landmarks.npz"))['landmarks']        # TODO: Temporary solution for bad dataset
                    print(f"No landmarks found for: {os.path.join(class_path, filename)} using default landmarks")
                    #raise Exception(f"No landmarks found for: {os.path.join(class_path, filename)}")

                np.savez_compressed(os.path.join(class_path, (filename[:-4]+'.npz')), landmarks=points)
                counter += 1

    print(f"Created landmarks in {dataset_folder} for {counter} images")

if __name__ == '__main__':

    dataset = "C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test_dataset_TEST\\validation"
    calculate_face_correspondences_dataset(dataset)
    dataset = "C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test_dataset_TEST\\train"
    calculate_face_correspondences_dataset(dataset)


    #face_mesh = mp.solutions.face_mesh.FaceMesh( static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    #image1 = cv2.imread("C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test_dataset\\id10276\\136b071b44cb84892818a85b7cb43f21d0ab3e76-10_0_image.jpg")
    #image2 = cv2.imread("C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test_dataset\\id10276\\136b071b44cb84892818a85b7cb43f21d0ab3e76-25_10_image.jpg")
    #calculate_face_correspondences(face_mesh, image1, image2, draw_corr=True)
