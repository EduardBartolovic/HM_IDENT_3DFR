import os
import time
import logging
import argparse
import warnings

import cv2
import numpy as np
import onnxruntime

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
    return parser.parse_args()


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
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


def video_to_pyr(face_detector, head_pose, device, video_source, output_dir, frame_count_start, batch_size=2048):
    cap = cv2.VideoCapture(video_source)
    counter = frame_count_start
    frames = []
    frame_infos = []

    def process_batch(frames_batch, frame_indices):
        """Process a batch of frames."""
        nonlocal counter
        batch_frame_infos = []

        # Detect faces for the entire batch
        batch_bboxes = []
        for frame in frames_batch:
            bboxes = face_detector.detect(frame)
            batch_bboxes.append(bboxes)

        # Preprocess and move to GPU
        batched_images = []
        for frame, bboxes in zip(frames_batch, batch_bboxes):
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max)
                cropped_face = frame[y_min:y_max, x_min:x_max]
                processed_face = pre_process(cropped_face)
                batched_images.append(processed_face)

        # Stack and move the batch to the GPU
        batched_images = torch.stack(batched_images).to(device)
        batched_images = batched_images.squeeze(1)

        # Head pose estimation in batch
        rotation_matrices = head_pose(batched_images)

        eulers = compute_euler_angles_from_rotation_matrices(rotation_matrices).cpu().numpy()  # Assumes batched output
        eulers_deg = np.degrees(eulers)

        # Process results back to frame info
        idx = 0
        for frame_idx, (frame, bboxes) in enumerate(zip(frames_batch, batch_bboxes)):
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max)
                angles = eulers_deg[idx]
                batch_frame_infos.append(
                    [frame_indices[frame_idx], x_min, y_min, x_max, y_max, int(angles[1]), int(angles[0]),
                     int(angles[2])]
                )
                idx += 1

        counter += len(frames_batch)
        return batch_frame_infos

    frame_indices = []

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                break

            frames.append(frame)
            frame_indices.append(counter)
            counter += 1

            # Process when batch is full
            if len(frames) == batch_size:
                batch_infos = process_batch(frames, frame_indices)
                frame_infos.extend(batch_infos)
                frames = []
                frame_indices = []

        # Process remaining frames
        if frames:
            batch_infos = process_batch(frames, frame_indices)
            frame_infos.extend(batch_infos)

    # Save frame information to a text file
    txt_output_file = os.path.join(output_dir, f"{frame_count_start}_frame_infos.txt")
    with open(txt_output_file, 'w') as txt_file:
        for info in frame_infos:
            txt_file.write(','.join(map(str, info)) + '\n')

    cap.release()
    cv2.destroyAllWindows()
    logging.info(f"Processing complete. Output saved to {txt_output_file}")


def main(params):

    model_path_scrd= "~/HM_IDENT_3DFR/src/preprocess_datasets/headPoseEstimation/weights/det_10g.onnx"
    model_path_hpe = "~/HM_IDENT_3DFR/src/preprocess_datasets/headPoseEstimation/weights/resnet50.pt"
    input_dir = "/home/gustav/voxceleb/VoxCeleb2_test"
    output_dir = "/home/gustav/voxceleb_out/"

    model_path_scrd = "C:\\Users\\Eduard\\Downloads\\voxceleb_preprocessing\\FaceHPE\\weights\\det_10g.onnx"
    model_path_hpe = "C:\\Users\\Eduard\\Downloads\\voxceleb_preprocessing\\FaceHPE\\weights\\resnet50.pt"
    input_dir = "C:\\Users\\Eduard\\Downloads\\voxceleb_preprocessing\\FaceHPE\\VoxCeleb1_test"
    output_dir = "C:\\Users\\Eduard\\Downloads\\voxceleb_preprocessing\\FaceHPE\\VoxCeleb1_test_out"

    if onnxruntime.get_device() != "GPU":
        print(onnxruntime.get_device(), "is not GPU")
        raise Exception("ONNX runs on CPU and not GPU -> pip install onnxruntime-gpu")
    try:
        face_detector = SCRFD(model_path=model_path_scrd)
        logging.info("Face Detection model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of face detection model. Exception: {e}")
        raise Exception()
    device = torch.device("cuda")
    try:
        head_pose = get_model(params.arch, num_classes=6)
        state_dict = torch.load(model_path_hpe, map_location=device)
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
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):  # Add other formats if needed
                video_path = os.path.join(root, file)
                frame_count_start = int(file.split('#')[2].split('-')[0])
                relative_path = os.path.relpath(root, input_dir)
                save_path = os.path.join(output_dir, relative_path.replace("/chunk_videos", ""))
                os.makedirs(save_path, exist_ok=True)
                # Check if the .txt file already exists
                txt_file_path = os.path.join(save_path, f"{frame_count_start}_frame_infos.txt")
                if os.path.exists(txt_file_path):
                    logging.info(f"Skipping Video {file}: Output file already exists.")
                    continue
                start = time.time()
                video_to_pyr(face_detector, head_pose, device, video_path, save_path, frame_count_start)
                logging.info(f'Head pose estimation for Video {file}: %.2f s' % (time.time() - start))


if __name__ == '__main__':
    args = parse_args()
    main(args)
