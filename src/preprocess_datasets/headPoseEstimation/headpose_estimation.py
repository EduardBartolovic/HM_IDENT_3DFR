import logging
import os
import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from src.preprocess_datasets.headPoseEstimation.models.resnet import resnet50
from src.preprocess_datasets.headPoseEstimation.utils.general import draw_axis, \
    compute_euler_angles_from_rotation_matrices


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    return image


def get_model(arch, num_classes=6, pretrained=True):
    """Return the model based on the specified architecture."""
    if arch == 'resnet50':
        model = resnet50(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Please choose available model architecture, currently chosen: {arch}")
    return model


def process(cropped_face, device, head_pose_model):
    """Process a frame."""

    processed_face = pre_process(cropped_face)

    batched_images = processed_face.to(device).unsqueeze(0)
    rotation_matrices = head_pose_model(batched_images)

    eulers = compute_euler_angles_from_rotation_matrices(rotation_matrices).detach().cpu().numpy()[0]
    eulers_deg = np.degrees(eulers)

    return [int(eulers_deg[1]), int(eulers_deg[0]), int(eulers_deg[2])]


def process_batch(cropped_faces, device, head_pose_model):
    """Process a batch of frames."""

    processed_faces = [pre_process(face) for face in cropped_faces]
    batched_images = torch.stack(processed_faces).to(device)
    rotation_matrices = head_pose_model(batched_images)

    eulers = compute_euler_angles_from_rotation_matrices(rotation_matrices).detach().cpu().numpy()
    eulers_deg = np.degrees(eulers)

    return [[int(deg[1]), int(deg[0]), int(deg[2])] for deg in eulers_deg]


def headpose_estimation(input_folder, image_folder, output_folder, model_path_hpe, device, fix_rotation=False,
                        draw=True, make_vid=False):
    try:
        head_pose_model = get_model("resnet50", num_classes=6)
        state_dict = torch.load(model_path_hpe, map_location=device, weights_only=True)
        head_pose_model.load_state_dict(state_dict)
        head_pose_model.to(device)
        head_pose_model.eval()
        logging.info("Head Pose Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading weights of head pose estimation model: {e}")
        raise Exception()

    counter = 0
    for root, _, files in os.walk(input_folder):  # Recursively walk through directories

        if image_folder in root and "hpe" not in root:

            output_video_folder = os.path.join(os.path.dirname(root), output_folder)
            output_video_path = os.path.join(output_video_folder, "hpe.mp4")
            output_txt_path = os.path.join(output_video_folder, "hpe.txt")
            os.makedirs(output_video_folder, exist_ok=True)

            imgs = []
            frame_infos = []
            for img_file_name in files:
                if ".jpg" in img_file_name or ".png" in img_file_name:
                    img_path = os.path.join(root, img_file_name)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img is not None:
                        imgs.append(img)
                        info = process(img, device, head_pose_model)
                        info.append(img_file_name)
                        frame_infos.append(info)

            assert len(frame_infos) == len(imgs)
            assert len(frame_infos) > 0

            fps = 3
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 videos
            out = None

            frames_written = 0
            for frame, info in zip(imgs, frame_infos):
                y_pred_deg, p_pred_deg, r_pred_deg, _ = info
                w, h, c = frame.shape

                if make_vid:
                    if fix_rotation:
                        bbox_center_x = w // 2
                        bbox_center_y = h // 2
                        rotation_matrix = cv2.getRotationMatrix2D((bbox_center_x, bbox_center_y), r_pred_deg, 1.0)

                        # Apply rotation to the full frame
                        frame = cv2.warpAffine(
                            frame,
                            rotation_matrix,
                            (frame.shape[1], frame.shape[0]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0)
                        )
                        r_pred_deg = 0

                    if draw:
                        draw_axis(
                            frame,
                            y_pred_deg,
                            p_pred_deg,
                            r_pred_deg,
                            bbox=[0, 0, w, h],
                            size_ratio=0.5
                        )

                    if out is None:
                        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
                    out.write(frame)
                    frames_written += 1

                with open(output_txt_path, 'w') as txt_file:
                    for i in frame_infos:
                        txt_file.write(','.join(map(str, i)) + '\n')
                        counter += 1

            if make_vid:
                out.write(frame)
                out.release()
                print(f"Processed and saved cropped video: {output_video_path} frames_written: {frames_written}")
            else:
                print(f"Processed: {output_txt_path}")

    print(f"HPE for {counter} frames")


def get_frames(video_path, frame_skip=1):
    cap = cv2.VideoCapture(video_path)
    counter = 0
    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if counter % frame_skip == 0:
            frame_list.append(frame)
        counter += 1

    cap.release()
    cv2.destroyAllWindows()
    return frame_list


def headpose_estimation_from_video(input_folder, output_folder, model_path_hpe, device, batch_size=64, filter=None):
    start_time = time.time()
    try:
        head_pose_model = get_model("resnet50", num_classes=6)
        state_dict = torch.load(model_path_hpe, map_location=device, weights_only=True)
        head_pose_model.load_state_dict(state_dict)
        head_pose_model.to(device)
        head_pose_model.eval()
        logging.info("Head Pose Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occurred while loading weights of head pose estimation model: {e}")
        raise Exception()

    folders = list(os.walk(input_folder))
    num_folders = len(folders)
    for root, _, files in tqdm(folders, desc="Processing folders"):

        frame_infos = []
        output_hpe_folder = os.path.join(root, output_folder)
        output_txt_path = os.path.join(output_hpe_folder, "hpe.txt")

        # Skip video if hpe.txt already exists
        if os.path.exists(output_txt_path):
            # print(f"Skipping already processed folder: {output_hpe_folder}")
            continue

        video_frames = []
        video_names = []
        for video in files:
            if ".mp4" in video:
                if filter is not None:
                    if filter not in video:
                        continue
                os.makedirs(output_hpe_folder, exist_ok=True)
                imgs = get_frames(os.path.join(root, video))

                for counter, img in enumerate(imgs):
                    if img is not None:
                        video_frames.append(img)
                        video_names.append(video + "#" + str(counter))

                if not video_frames:
                    print("Error for", os.path.join(root, output_folder))

        if not video_frames:
            continue

        for i in range(0, len(imgs), batch_size):
            batch = [img for img in imgs[i:i + batch_size] if img is not None]
            if batch:
                batch_infos = process_batch(batch, device, head_pose_model)
                for j, info in enumerate(batch_infos):
                    info.append(video + "#" + str(i + j))
                    frame_infos.append(info)

            if len(frame_infos) == 0:
                print("Error for ", os.path.join(root, output_folder))

        if frame_infos:
            with open(output_txt_path, 'w') as txt_file:
                for info in frame_infos:
                    txt_file.write(','.join(map(str, info)) + '\n')

            print(f"Processed: {output_txt_path}")

    elapsed_time = time.time() - start_time
    print("HPE for ", num_folders, " in", round(elapsed_time / 60, 2), "minutes")


if __name__ == '__main__':
    input_folder = "E:\\Download\\face\\VoxCeleb1_test"  # Folder containing original preprocessed files
    output_folder = "hpe"  # Folder to save cropped videos
    model_path_hpe = "F:\\Face\\HPE\\weights\\resnet50.pt"

    device = torch.device("cuda")
    try:
        head_pose = get_model("resnet50", num_classes=6)
        state_dict = torch.load(model_path_hpe, map_location=device, weights_only=True)
        head_pose.load_state_dict(state_dict)
        logging.info("Head Pose Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading weights of head pose estimation model: {e}")
        raise Exception()

    head_pose.to(device)
    head_pose.eval()

    headpose_estimation(input_folder, output_folder, device, fix_rotation=True, make_vid=False)
