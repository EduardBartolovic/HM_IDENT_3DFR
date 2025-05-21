import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.preprocess_datasets.detect_face import expand_bbox

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.preprocess_datasets.blazeface.blazeface import BlazeFace


def plot_detections(img, detections, with_keypoints=True):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)

    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])

    largest_area = 0
    largest_bbox = None

    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

        area = (xmax - xmin) * (ymax - ymin)
        if area > largest_area:
            largest_area = area
            largest_bbox = (xmin, ymin, xmax, ymax)

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor="r", facecolor="none",
                                 alpha=detections[i, 16])
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k * 2] * img.shape[1]
                kp_y = detections[i, 4 + k * 2 + 1] * img.shape[0]
                circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1,
                                        edgecolor="lightskyblue", facecolor="none",
                                        alpha=detections[i, 16])
                ax.add_patch(circle)

    # Mark the largest bounding box with a green box
    if largest_bbox:
        xmin, ymin, xmax, ymax = largest_bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor="g", facecolor="none")
        ax.add_patch(rect)

    plt.show()


def resize_with_padding(image, target_size=(256, 256), pad_color=(0, 0, 0)):
    h, w = image.shape[:2]

    # Calculate padding to make the image square
    if h > w:
        pad = (h - w) // 2
        left, right = pad, h - w - pad
        top, bottom = 0, 0
    else:
        pad = (w - h) // 2
        top, bottom = pad, w - h - pad
        left, right = 0, 0

    # Add border to make image square
    squared_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    # Resize the square image to target size
    resized_image = cv2.resize(squared_image, target_size, interpolation=cv2.INTER_AREA)

    return resized_image, (h, w), (top, bottom, left, right)


def face_crop_full_frame(input_folder, output_folder, model_root):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    back_net = BlazeFace(back_model=True).to(device)
    back_net.load_weights(os.path.join(model_root, "blazefaceback.pth"))
    back_net.load_anchors(os.path.join(model_root, "anchorsback.npy"))

    back_net.min_score_thresh = 0.45
    back_net.min_suppression_threshold = 0.3

    os.makedirs(output_folder, exist_ok=True)

    class_names = os.listdir(input_folder)
    missing_faces = 0
    more_faces = 0
    total_faces = 0
    for class_name in tqdm(class_names, desc="Cropping Faces - Processing Classes"):
        class_path = os.path.join(input_folder, class_name)
        target_class_path = os.path.join(output_folder, class_name)

        os.makedirs(target_class_path, exist_ok=True)
        for filename in os.listdir(class_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):

                face_crop_path = os.path.join(target_class_path, filename)
                if os.path.exists(face_crop_path):
                    continue  # Skip already cropped images

                image_path = os.path.join(class_path, filename)
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Resize for BlazeFace detection
                resized_image, (h, w), (top, bottom, left, right) = resize_with_padding(image_rgb)
                detections = back_net.predict_on_image(resized_image).cpu().numpy()

                if detections.shape[0] == 0:
                    missing_faces += 1
                    print("Missing face for", class_path, filename)
                elif detections.shape[0] > 1:
                    # plot_detections(padded_image, detections)
                    more_faces += 1
                    detections = max(detections, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                else:
                    detections = detections[0]
                total_faces += 1

                y_min = int(detections[0] * resized_image.shape[0])
                x_min = int(detections[1] * resized_image.shape[1])
                y_max = int(detections[2] * resized_image.shape[0])
                x_max = int(detections[3] * resized_image.shape[1])

                x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max, factor=0.1)
                assert (x_max - x_min) == (y_max - y_min)
                min_accepted_face_size = 112
                if x_max - x_min < min_accepted_face_size:
                    print("Too small face for", filename)

                face_crop = resized_image[y_min:y_max, x_min:x_max]
                face_crop_resized = cv2.resize(face_crop, (112, 112))
                final_image = cv2.cvtColor(face_crop_resized, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(face_crop_path), final_image)
    print(f"Done! Total_faces: {total_faces}, missing_faces: {missing_faces}, more_faces: {more_faces}")
