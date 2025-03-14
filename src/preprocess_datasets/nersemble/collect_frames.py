import hashlib

import cv2
import os

from tqdm import tqdm_pandas, tqdm


def extract_and_group_frames(input_folder, output_folder, frame_interval=100):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for id_folder in tqdm(os.listdir(input_folder), desc="Iterating over ids"):
        # Walk through all subfolders to find videos, skipping those with 'stacked' in their filename
        videos = []
        for root, _, files in os.walk(os.path.join(input_folder,id_folder)):
            for f in files:
                if f.endswith(('.mp4', '.avi', '.mov')) and 'head_stacked' not in f:
                    videos.append(os.path.join(root, f))
        id_output_folder = os.path.join(output_folder, id_folder)
        if videos and not os.path.exists(id_output_folder):
            os.makedirs(id_output_folder)

        for video_path in videos:
            video_name = os.path.basename(video_path)
            cap = cv2.VideoCapture(video_path)

            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Create a unique hash for each frame group
                    frame_hash = hashlib.sha256(f"frame_{frame_count}".encode()).hexdigest()[40:]

                    frame_filename = os.path.join(id_output_folder,
                                                  f"{frame_hash}_{os.path.splitext(video_name)[0]}.jpg")
                    cv2.imwrite(frame_filename, frame)

                frame_count += 1

            cap.release()

        print("Frames extracted and grouped successfully!")

