import hashlib
import time

import cv2
import os

from tqdm import tqdm_pandas, tqdm


def extract_and_group_frames(input_folder, output_folder, frame_interval=100):
    start_time = time.time()
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
            target_name = os.path.splitext(video_name)[0]
            if "00042" in target_name:
                target_name = "25_-25"
            elif "00046" in target_name:
                target_name = "25_-10"
            elif "00036" in target_name:
                target_name = "10_-10"
            elif "00191" in target_name:
                target_name = "0_0"
            elif "00037" in target_name:
                target_name = "25_0"
            elif "00047" in target_name:
                target_name = "10_10"
            elif "00049" in target_name:
                target_name = "25_10"
            elif "1007" in target_name:
                target_name = "25_25"
            elif "00044" in target_name:
                target_name = "-25_-25"
            elif "00040" in target_name:
                target_name = "-25_-10"
            elif "00048" in target_name:
                target_name = "-10_-10"
            elif "00041" in target_name:
                target_name = "-10_0"
            elif "00038" in target_name:
                target_name = "-25_0"
            elif "00043" in target_name:
                target_name = "-10_10"
            elif "00039" in target_name:
                target_name = "-25_10"
            elif "00045" in target_name:
                target_name = "-25_25"
            else:
                raise Exception(f"{target_name} can not be converted to angle")


            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Create a unique hash for each frame group
                    frame_hash = hashlib.sha256(f"frame_{frame_count}".encode()).hexdigest()[40:]

                    frame_filename = os.path.join(id_output_folder,
                                                  f"{frame_hash}{target_name}_image.jpg")
                    cv2.imwrite(frame_filename, frame)

                frame_count += 1

            cap.release()

    elapsed_time = time.time() - start_time
    print(f"Collected Frames in {output_folder} in", round(elapsed_time/60, 2), "minutes")


