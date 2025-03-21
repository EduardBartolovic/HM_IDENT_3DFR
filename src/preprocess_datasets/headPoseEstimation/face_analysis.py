import shutil

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from torchvision.transforms import transforms

from src.backbone.model_irse import IR_50
from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import read_file
import os
import cv2
import mediapipe as mp
import umap


def process_landmarks(multi_face_landmarks, dst, r_pred_deg, image):
    for face_landmarks in multi_face_landmarks:

        h, w, _ = image.shape
        x_coords = [int(landmark.x * w) for landmark in face_landmarks.landmark]
        y_coords = [int(landmark.y * h) for landmark in face_landmarks.landmark]

        # Calculate bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Expand bounding box with padding
        padding = int(w*0.25)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Apply roll compensation to the full frame
        center_x = int(x_max - x_min) // 2
        center_y = int(y_max - y_min) // 2
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), r_pred_deg, 1.0)
        image = cv2.warpAffine(
            image,
            rotation_matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Crop and resize the face
        face_crop = image[y_min:y_max, x_min:x_max]
        resized_face = cv2.resize(face_crop, (224, 224))  # Adjust as needed

        cv2.imwrite(dst, resized_face)


def face_analysis(input_folder, output_folder, device):

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    for root, _, files in os.walk(input_folder):
        if "hpe" in root:
            for txt_file in files:
                if txt_file.endswith('hpe.txt'):
                    file_path = os.path.join(root, txt_file)
                    data = read_file(file_path, remove_header=False)

                    folder_image_path = os.path.join(root, "..", output_folder)
                    folder_image_output_path = os.path.join(root, "..", output_folder)

                    os.makedirs(folder_image_output_path, exist_ok=True)

                    last_land_marks = None
                    for info in data:
                        r_pred_deg = int(info[2])
                        src = os.path.join(folder_image_path, info[3])
                        dst = os.path.join(folder_image_output_path, info[3])

                        image = cv2.imread(src)
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb_image)

                        if results.multi_face_landmarks:
                            process_landmarks(results.multi_face_landmarks, dst, r_pred_deg, image)
                            last_land_marks = results.multi_face_landmarks
                        else:
                            if last_land_marks:
                                print("NO LANDMARKS BUT USED PREV", dst)
                                process_landmarks(last_land_marks, dst, r_pred_deg, image)
                            else:
                                print("NO LANDMARKS")
                                raise Exception("NO LANDMARKS")



def filter_wrong_faces(input_folder, output_folder, backbone_save_path, device, distance_threshold=0.6):

    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    backbone = IR_50((112,112), 512)
    backbone.load_state_dict(torch.load(backbone_save_path, weights_only=True))
    backbone = backbone.to("cuda")
    backbone.eval()

    for id_name in os.listdir(input_folder):
        id_path = os.path.join(input_folder, id_name)
        if not os.path.isdir(id_path):
            continue

        folder_image_output_path = os.path.join(id_path, output_folder)
        os.makedirs(folder_image_output_path, exist_ok=True)

        embeddings = []
        image_paths = []

        for video_name in os.listdir(id_path):
            video_path = os.path.join(id_path, video_name, "frames_cropped")
            if not os.path.isdir(video_path):
                continue

            for img_file in os.listdir(video_path):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(video_path, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Failed to read image: {img_path}")
                    continue

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = transforms.ToPILImage()(rgb_image)
                processed_image = train_transform(pil_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = backbone(processed_image).cpu().numpy().flatten()
                embeddings.append(embedding)
                image_paths.append(img_path)

        if len(embeddings) == 0:
            continue

        embeddings = np.array(embeddings)
        distances = cosine_distances(embeddings)
        avg_distances = distances.mean(axis=1)
        inliers = avg_distances < distance_threshold

        if True:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
            embeddings_2d = reducer.fit_transform(embeddings)
            plt.figure(figsize=(10, 8))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=inliers, cmap='coolwarm', s=50, alpha=0.7)
            plt.title(f"Embeddings Visualization for ID: {id_name}")
            plt.colorbar(label='Inliers (True = 1, False = 0)')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.show()


        if not np.any(inliers):
            continue

        for idx, is_inlier in enumerate(inliers):
            if is_inlier:
                src = image_paths[idx]
                dst = os.path.join(folder_image_output_path, os.path.basename(src))
                shutil.copy(src, dst)
            else:
                print(f"Filtered outlier: {image_paths[idx]} with distance {round(avg_distances[idx], 4)}")


if __name__ == '__main__':

    input_folder = "C:\\Users\\Eduard\\Desktop\\Face\\dataset8\\VoxCeleb1_test"
    filter_wrong_faces(input_folder, "filtered_frames", "C:\\Users\\Eduard\\Desktop\\Face\\HM_IDENT_3DFR\\pretrained\\backbone_ir50_asia.pth", "cuda")
