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


def better_face_crop_voxceleb(input_folder, output_folder, model_root):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    back_net = BlazeFace(back_model=True).to(device)
    back_net.load_weights(os.path.join(model_root, "blazefaceback.pth"))
    back_net.load_anchors(os.path.join(model_root, "anchorsback.npy"))

    back_net.min_score_thresh = 0.6
    back_net.min_suppression_threshold = 0.3

    os.makedirs(output_folder, exist_ok=True)

    class_names = os.listdir(input_folder)
    missing_faces = 0
    more_faces = 0
    total_faces = 0
    for class_name in tqdm(class_names, desc="Processing Classes"):
        class_path = os.path.join(input_folder, class_name)
        target_class_path = os.path.join(output_folder, class_name)

        os.makedirs(target_class_path, exist_ok=True)
        for filename in os.listdir(class_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):

                face_crop_path = os.path.join(target_class_path, filename)
                if os.path.exists(face_crop_path):
                    continue  # Skip already cropped images

                image = cv2.imread(os.path.join(class_path, filename), cv2.COLOR_BGR2RGB)

                # Add padding using cv2.copyMakeBorder
                pad_size = 16  # Since (256-224)/2 = 16
                padded_image = cv2.cvtColor(cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0]), cv2.COLOR_BGR2RGB)

                detections = back_net.predict_on_image(padded_image).cpu().numpy()

                if detections.shape[0] == 0:
                    missing_faces += 1
                    continue
                elif detections.shape[0] > 1:
                    # plot_detections(padded_image, detections)
                    more_faces += 1
                    detections = max(detections, key=lambda det: (det[2] - det[0]) * (det[3] - det[1]))
                else:
                    detections = detections[0]
                total_faces += 1

                y_min = int(detections[0] * image.shape[0])
                x_min = int(detections[1] * image.shape[1])
                y_max = int(detections[2] * image.shape[0])
                x_max = int(detections[3] * image.shape[1])

                x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max, factor=0.3)
                assert (x_max-x_min) == (y_max-y_min)

                face_crop = padded_image[y_min:y_max, x_min:x_max]
                face_crop_resized = cv2.cvtColor(cv2.resize(face_crop, (112, 112)), cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(face_crop_path), face_crop_resized)
    print(f"Done. total_faces: {total_faces}, missing_faces: {missing_faces}, more_faces: {more_faces}")


def resize_with_padding(image, target_size=(256, 256), pad_color=(0, 0, 0)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_w = target_size[1] - new_w
    pad_h = target_size[0] - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded_image, scale, left, top


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
    for class_name in tqdm(class_names, desc="Processing Classes"):
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

                original_h, original_w = image_rgb.shape[:2]

                # Resize for BlazeFace detection
                resized_image, scale, pad_left, pad_top = resize_with_padding(image_rgb)
                detections = back_net.predict_on_image(resized_image).cpu().numpy()

                if detections.shape[0] == 0:
                    missing_faces += 1
                    print("Missing face:", face_crop_path)
                    continue
                elif detections.shape[0] > 1:
                    more_faces += 1
                    detections = max(detections, key=lambda det: (det[2] - det[0]) * (det[3] - det[1]))
                else:
                    detections = detections[0]

                total_faces += 1

                # Undo normalization and padding
                y_min = int((detections[0] * 256 - pad_top) / scale)
                x_min = int((detections[1] * 256 - pad_left) / scale)
                y_max = int((detections[2] * 256 - pad_top) / scale)
                x_max = int((detections[3] * 256 - pad_left) / scale)

                x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max, factor=0.1)

                # Make sure the bounding box is square
                box_w = x_max - x_min
                box_h = y_max - y_min
                side = max(box_w, box_h)
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                x_min = max(center_x - side // 2, 0)
                y_min = max(center_y - side // 2, 0)
                x_max = min(x_min + side, original_w)
                y_max = min(y_min + side, original_h)

                face_crop = image_rgb[y_min:y_max, x_min:x_max]
                face_crop_resized = cv2.resize(face_crop, (112, 112))
                final_image = cv2.cvtColor(face_crop_resized, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(face_crop_path), final_image)
    print(f"Done. total_faces: {total_faces}, missing_faces: {missing_faces}, more_faces: {more_faces}")


if __name__ == '__main__':
    dataset_folder = "E:\\Download\\vox2test_out"
    dataset_output_folder_crop = "E:\\Download\\vox2test_out_crop"
    face_detect_model = "F:\\Face\\HM_IDENT_3DFR\\src\\preprocess_datasets\\headPoseEstimation\\weights\\det_10g.onnx"

    better_face_crop_voxceleb(dataset_folder, dataset_output_folder_crop, face_detect_model)
