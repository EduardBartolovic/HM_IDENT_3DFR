import logging
import os

import cv2
import numpy as np
import torch
from torchvision import transforms

from src.preprocess_datasets.headPoseEstimation.models.resnet import resnet50
from src.preprocess_datasets.headPoseEstimation.utils.general import draw_axis, \
    compute_euler_angles_from_rotation_matrices


def pre_process(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        print("Error:", image)
        raise Exception()
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


def process(cropped_face, device, head_pose):
    """Process a frame."""

    processed_face = pre_process(cropped_face)

    batched_images = processed_face.to(device).unsqueeze(0)
    rotation_matrices = head_pose(batched_images)

    eulers = compute_euler_angles_from_rotation_matrices(rotation_matrices).detach().cpu().numpy()[0]
    eulers_deg = np.degrees(eulers)

    return int(eulers_deg[1]), int(eulers_deg[0]), int(eulers_deg[2])


def headpose_estimation(input_folder, output_folder, head_pose_model, device, fix_rotation=False, draw=True):

    for root, _, files in os.walk(input_folder):  # Recursively walk through directories

        if "frames_cropped" in root and "hpe" not in root:

            output_video_folder = os.path.join(os.path.dirname(root), output_folder)
            output_video_path = os.path.join(output_video_folder, "hpe.mp4")
            output_txt_path = os.path.join(output_video_folder, "hpe.txt")
            os.makedirs(output_video_folder, exist_ok=True)

            imgs = []
            for img_file in files:
                if ".png" in img_file:
                    img_path = os.path.join(root, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    imgs.append(img)

            frame_infos = []
            for img in imgs:
                info = process(img, device, head_pose_model)
                frame_infos.append(info)

            assert len(frame_infos) == len(imgs)
            assert len(frame_infos) > 0

            fps = 2  # int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 videos
            # Prepare VideoWriter for cropped video
            out = None

            for frame, info in zip(imgs, frame_infos):
                y_pred_deg, p_pred_deg, r_pred_deg = info

                w, h, c = frame.shape

                if fix_rotation:  # r_pred_deg > 1 or r_pred_deg < -1:

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

                with open(output_txt_path, 'w') as txt_file:
                    for i in frame_infos:
                        txt_file.write(','.join(map(str, i)) + '\n')

            if out:
                out.release()
            print(f"Processed and saved cropped video: {output_video_path}")


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

    headpose_estimation(input_folder, output_folder, device, fix_rotation=True)
