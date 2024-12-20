import os
import time
import logging
import argparse
import warnings
import mediapipe as mp
import face_recognition

import cv2
import numpy as np

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


def extract_face_landmarks(frame, face_mesh):
    """Extract facial landmarks using mediapipe."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]  # Assume only one face for simplicity
    return None


def calculate_landmark_distance(landmarks1, landmarks2):
    """Calculate Euclidean distance between corresponding landmarks."""
    distances = []
    for lm1, lm2 in zip(landmarks1, landmarks2):
        distance = np.linalg.norm(np.array([lm1.x, lm1.y, lm1.z]) - np.array([lm2.x, lm2.y, lm2.z]))
        distances.append(distance)
    return np.mean(distances)


def is_same_person(landmarks1, landmarks2, threshold=0.05):
    """Compare two faces based on their landmarks."""
    distance = calculate_landmark_distance(landmarks1, landmarks2)
    return distance < threshold


def detect_faces(video_source, txt_dir, output_dir):
    frames_to_use, x_coords, y_coords, widths, heights = read_txt_files(txt_dir)
    first_frame = min(frames_to_use)
    last_frame = max(frames_to_use)

    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:

        cap = cv2.VideoCapture(video_source)
        frame_number = 0
        txt_output_file = os.path.join(output_dir, "frame_infos.txt")
        first_face_embedding = None

        with open(txt_output_file, 'w') as file:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                if frame_number < first_frame:
                    continue
                if frame_number > last_frame:
                    break

                frame_number += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if results.detections:
                    for detection in results.detections:
                        # Extract bounding box
                        bboxC = detection.location_data.relative_bounding_box
                        x_min = int(bboxC.xmin * frame.shape[1])
                        y_min = int(bboxC.ymin * frame.shape[0])
                        width = int(bboxC.width * frame.shape[1])
                        height = int(bboxC.height * frame.shape[0])

                        # Make the bounding box square
                        max_side = max(width, height)
                        x_min -= (max_side - width) // 2
                        y_min -= (max_side - height) // 2
                        x_max = x_min + max_side
                        y_max = y_min + max_side

                        # Ensure bounding box stays within frame
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(frame.shape[1], x_max)
                        y_max = min(frame.shape[0], y_max)

                        landmarks = extract_face_landmarks(frame, face_mesh)

                        # Extract the face embedding for the first detected face
                        if first_face_embedding is None:
                            first_face_embedding = extract_face_embedding(face)
                            if first_face_embedding is not None:
                                logging.info(f"First face embedding extracted at frame {frame_number}.")
                        else:
                            # For subsequent faces, compare embeddings
                            current_face_embedding = extract_face_embedding(face)
                            if current_face_embedding is not None:
                                if is_same_person(first_face_embedding, current_face_embedding):
                                    # The face is the same person, process it
                                    file.write(f"{frame_number},{x_min},{y_min},{x_max},{y_max}\n")

        cap.release()


def main(params):

    #model_path_hpe = "~/HM_IDENT_3DFR/src/preprocess_datasets/headPoseEstimation/weights/resnet50.pt"
    #input_dir = "/home/gustav/voxceleb/VoxCeleb2_test"
    #output_dir = "/home/gustav/voxceleb_out/"

    input_dir = "F:\\Face\\HPE\\VoxCeleb1_test"
    output_dir = "F:\\Face\\HPE\\VoxCeleb1_test_faces_out"

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
                    detect_faces(video_path, txt_dir, save_path)
                    logging.info(f'Facedetection for Video {file}: %.2f s' % (time.time() - start))


if __name__ == '__main__':
    args = parse_args()
    main(args)
