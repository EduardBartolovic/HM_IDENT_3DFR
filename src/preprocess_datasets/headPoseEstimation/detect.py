import os
import time
import logging
import argparse
import warnings

import cv2
import numpy as np

import torch
from torchvision import transforms

from src.preprocess_datasets.headPoseEstimation.models.resnet import resnet50
from src.preprocess_datasets.headPoseEstimation.models.scrfd import SCRFD
from src.preprocess_datasets.headPoseEstimation.utils.general import compute_euler_angles_from_rotation_matrices

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation inference.')
    parser.add_argument("--arch", type=str, default="resnet18", help="Model name, default `resnet18`")
    parser.add_argument(
        "--input",
        type=str,
        default='assets/in_video.mp4',
        help="Path to input video file or camera id"
    )
    parser.add_argument("--view", action="store_true", help="Display the inference results")
    parser.add_argument(
        "--draw-type",
        type=str,
        default='cube',
        choices=['cube', 'axis'],
        help="Draw cube or axis for head pose"
    )
    parser.add_argument('--weights', type=str, required=True, help='Path to head pose estimation model weights')
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")

    return parser.parse_args()


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


def expand_bbox(x_min, y_min, x_max, y_max, factor=0.2):
    """Expand the bounding box by a given factor and make it square."""
    width = x_max - x_min
    height = y_max - y_min

    # Expand the bbox dimensions
    x_min_new = x_min - int(factor * height)
    y_min_new = y_min - int(factor * width)
    x_max_new = x_max + int(factor * height)
    y_max_new = y_max + int(factor * width)

    # Ensure square by finding the max side length
    square_size = max(x_max_new - x_min_new, y_max_new - y_min_new)
    center_x = (x_min_new + x_max_new) // 2
    center_y = (y_min_new + y_max_new) // 2

    # Calculate new square bounding box
    half_size = square_size // 2
    x_min_square = max(0, center_x - half_size)
    y_min_square = max(0, center_y - half_size)
    x_max_square = x_min_square + square_size
    y_max_square = y_min_square + square_size

    return x_min_square, y_min_square, x_max_square, y_max_square


def get_model(arch, num_classes=6, pretrained=True):
    """Return the model based on the specified architecture."""
    if arch == 'resnet50':
        model = resnet50(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Please choose available model architecture, currently chosen: {arch}")
    return model


def save_frame(image, angles, output_dir, counter):
    yaw, pitch, roll = angles
    filename = f"frame_{counter}_yaw_{yaw:.2f}_pitch_{pitch:.2f}_roll_{roll:.2f}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)


def save_frame(image, angles, output_dir, counter):
    yaw, pitch, roll = angles
    filename = f"frame_{counter}_yaw_{yaw:.2f}_pitch_{pitch:.2f}_roll_{roll:.2f}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)


def video_to_pyr(face_detector, head_pose, device, video_source, output_dir, frame_count_start, save_frames=False):
    cap = cv2.VideoCapture(video_source)
    counter = frame_count_start

    frame_infos = []
    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                break

            bboxes, keypoints = face_detector.detect(frame)
            for bbox, keypoint in zip(bboxes, keypoints):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max)

                image_ori = frame[y_min:y_max, x_min:x_max]
                image = pre_process(image_ori)
                image = image.to(device)

                rotation_matrix = head_pose(image)

                euler = np.degrees(compute_euler_angles_from_rotation_matrices(rotation_matrix))
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()
                angles = [int(y_pred_deg.item()), int(p_pred_deg.item()), int(r_pred_deg.item())]

                if save_frames:
                    save_frame(image_ori, angles, output_dir, counter)
                else:
                    frame_infos.append(
                        [counter, x_min, y_min, x_max, y_max, int(y_pred_deg.item()), int(p_pred_deg.item()),
                         int(r_pred_deg.item())])
                counter += 1

    if not save_frames:
        txt_output_file = os.path.join(output_dir, "frame_infos.txt")
        with open(txt_output_file, 'w') as txt_file:
            for i in frame_infos:
                txt_file.write(','.join(map(str, i)) + '\n')

    cap.release()
    cv2.destroyAllWindows()


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        face_detector = SCRFD(model_path="./weights/det_10g.onnx")
        logging.info("Face Detection model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of face detection model. Exception: {e}")
        raise Exception()

    try:
        head_pose = get_model(params.arch, num_classes=6)
        state_dict = torch.load(params.weights, map_location=device)
        head_pose.load_state_dict(state_dict)
        logging.info("Head Pose Estimation model weights loaded.")
    except Exception as e:
        logging.info(
            f"Exception occured while loading pre-trained weights of head pose estimation model. Exception: {e}")
        raise Exception()

    head_pose.to(device)
    head_pose.eval()

    """
    Process videos in a folder structure and save frames in an identical folder structure.
    """
    input_dir = "/home/gustav/voxceleb/VoxCeleb2_test"
    output_dir = "/home/gustav/voxceleb_out/"
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):  # Add other formats if needed
                video_path = os.path.join(root, file)
                frame_count_start = int(file.split('#')[2].split('-')[0])
                relative_path = os.path.relpath(root, input_dir)
                save_path = os.path.join(output_dir, relative_path.replace("/chunk_videos", ""))
                os.makedirs(save_path, exist_ok=True)
                start = time.time()
                video_to_pyr(face_detector, head_pose, device, video_path, save_path, frame_count_start)
                logging.info(f'Head pose estimation for Video {file}: %.2f s' % (time.time() - start))


if __name__ == '__main__':
    args = parse_args()
    main(args)
