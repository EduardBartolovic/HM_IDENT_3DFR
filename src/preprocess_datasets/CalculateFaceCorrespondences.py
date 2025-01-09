import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf


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

        # Compute the TPS transformation and Warp the image
        grid_x, grid_y = np.meshgrid(np.arange(w2), np.arange(h2))
        warped_x, warped_y = tps_transform(points1, points2, grid_x, grid_y, smooth=0.5)
        map_x = np.clip(warped_x, 0, w1 - 1).astype(np.float32)
        map_y = np.clip(warped_y, 0, h1 - 1).astype(np.float32)
        warped_image2 = cv2.remap(image2, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        # Plot corresponding landmarks -------------------------------
        if draw_corr:
            points1 = [(int(lm.x * w1), int(lm.y * h1)) for lm in landmarks1]
            points2 = [(int(lm.x * w2), int(lm.y * h2)) for lm in landmarks2]
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



            plt.figure(figsize=(12, 6))
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
            plt.show()




    else:
        raise ValueError("Face landmarks not detected in one or both images.")

if __name__ == '__main__':
    face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True)

    image1 = cv2.imread("C:\\Users\\Eduard\\Downloads\\VoxCeleb1_test_dataset\\id10285\\5nGO1Mq_iGM\\0_0_0_-3.jpg")
    image2 = cv2.imread("C:\\Users\\Eduard\\Downloads\\VoxCeleb1_test_dataset\\id10285\\5nGO1Mq_iGM\\-25_0_0_0.jpg")
    calculate_face_correspondences(face_mesh, image1, image2)