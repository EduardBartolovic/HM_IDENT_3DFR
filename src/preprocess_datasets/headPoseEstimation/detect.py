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


def read_txt_files(txt_dir):
    frames = []
    x_coords = []
    y_coords = []
    widths = []
    heights = []
    for root, dirs, files in os.walk(txt_dir):
        for file in files:
            if file.endswith(".txt"):
                txt_path = os.path.join(root, file)

                # Open and process the file
                with open(txt_path, "r") as f:
                    lines = f.readlines()

                # Start processing after the header line
                for line in lines:
                    # Strip whitespace and split by whitespace
                    parts = line.strip().split()
                    # Check if the line starts with a numeric frame index
                    if parts and parts[0].isdigit():
                        # Append the data as a dictionary
                        frames.append(int(parts[0]))
                        x_coords.append(int(parts[1]))
                        y_coords.append(int(parts[2]))
                        widths.append(int(parts[3]))
                        heights.append(int(parts[4]))

    return frames, x_coords, y_coords, widths, heights


def pre_process(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        print(image)
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


def save_frame(image, angles, output_dir, counter):
    yaw, pitch, roll = angles
    filename = f"frame_{counter}_yaw_{yaw:.2f}_pitch_{pitch:.2f}_roll_{roll:.2f}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)


def process(frame, frame_idx, x, y, w, h, device, head_pose):
    """Process a frame."""

    if w > h:
        w -= 1
    elif w < h:
        h -= 1
    assert w == h

    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h

    cropped_face = frame[y_min:y_max, x_min:x_max]

    if cropped_face.shape != (h, w, 3):
        raise Exception("cropped_face.shape != (h, w, 3):", cropped_face.shape, (h, w, 3))

    processed_face = pre_process(cropped_face)

    batched_images = processed_face.to(device).unsqueeze(0)
    rotation_matrices = head_pose(batched_images)

    eulers = compute_euler_angles_from_rotation_matrices(rotation_matrices).cpu().numpy()[0]
    eulers_deg = np.degrees(eulers)

    return [frame_idx, x_min, y_min, x_max, y_max, int(eulers_deg[1]), int(eulers_deg[0]), int(eulers_deg[2])]


def video_to_pyr(head_pose, device, video_source, txt_dir, output_dir, batch_size=32):

    frames_to_use, x_coords, y_coords, widths, heights = read_txt_files(txt_dir)
    cap = cv2.VideoCapture(video_source)
    current_frame_index = 0
    current_list_index = -1
    frame_infos = []

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                break

            current_frame_index += 1
            if current_frame_index in frames_to_use:
                current_list_index += 1
                infos = process(frame, frames_to_use[current_list_index], x_coords[current_list_index], y_coords[current_list_index], widths[current_list_index], heights[current_list_index], device, head_pose)
                frame_infos.append(infos)

    if len(frame_infos) == 0:
        raise Exception("frame_infos has length 0. No headpose extraction done for:", video_source, txt_dir)

    txt_output_file = os.path.join(output_dir, "frame_infos.txt")
    with open(txt_output_file, 'w') as txt_file:
        for info in frame_infos:
            txt_file.write(','.join(map(str, info)) + '\n')

    cap.release()
    cv2.destroyAllWindows()


def main(params):

    #model_path_hpe = "~/HM_IDENT_3DFR/src/preprocess_datasets/headPoseEstimation/weights/resnet50.pt"
    #input_dir = "/home/gustav/voxceleb/VoxCeleb2_test"
    #output_dir = "/home/gustav/voxceleb_out/"

    model_path_hpe = "F:\\Face\\HPE\\weights\\resnet50.pt"
    input_dir = "F:\\Face\\HPE\\VoxCeleb1_test"
    output_dir = "F:\\Face\\HPE\\VoxCeleb1_test_out"

    device = torch.device("cuda")
    try:
        head_pose = get_model("resnet50", num_classes=6)
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
    for root, dirs, files in os.walk(os.path.join(input_dir, "video")):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):  # Add other formats if needed
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, os.path.join(input_dir, "video"))
                txt_dir = os.path.join(input_dir, "txt", relative_path)

                relative_path = os.path.relpath(root, input_dir)
                save_path = os.path.join(output_dir, relative_path)
                os.makedirs(save_path, exist_ok=True)

                # Check if the .txt file already exists
                txt_file_path = os.path.join(save_path, "frame_infos.txt")
                if os.path.exists(txt_file_path):
                    logging.info(f"Skipping Video {file}: Output file already exists.")
                else:
                    logging.info(f"Processing Video {txt_file_path}...")
                    start = time.time()
                    try:
                        video_to_pyr(head_pose, device, video_path, txt_dir, save_path)
                    except Exception:
                        print(f"Error for Video {txt_file_path}")
                    logging.info(f'Head pose estimation for Video {file}: %.2f s' % (time.time() - start))


if __name__ == '__main__':
    args = parse_args()
    main(args)
