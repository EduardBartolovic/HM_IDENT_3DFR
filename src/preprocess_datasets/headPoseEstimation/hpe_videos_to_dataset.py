import os
import re
import cv2

import numpy as np


def read_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            matches = re.findall(r'[-\d]+', line)
            if matches:
                parsed_line = [int(value) for value in matches]
                data.append(parsed_line)
    return np.array(data)


def generate_voxceleb_dataset(input_folder, dataset_output_folder):

    for root, _, files in os.walk(input_folder):
        if "hpe" in root:
            for txt_file in files:
                if txt_file.endswith('matched_angles.txt'):
                    file_path = os.path.join(root, txt_file)
                    data = read_file(file_path)

                    video_path = os.path.join(root, "hpe.mp4")
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print("Error opening video stream or file")

                    sample_name = os.path.abspath(os.path.join(root, os.pardir))
                    id_name = os.path.abspath(os.path.join(sample_name, os.pardir))
                    sample_name = os.path.basename(sample_name)
                    id_name = os.path.basename(id_name)

                    destination = os.path.join(dataset_output_folder, id_name, sample_name)
                    os.makedirs(destination, exist_ok=True)
                    frame_counter = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            for info in data:
                                if frame_counter == info[-1]:
                                    cv2.imwrite(os.path.join(destination, f'{info[0]}_{info[1]}_{info[2]}_{info[5]}.jpg'), frame)

                        else:
                            break
                        frame_counter += 1


if __name__ == '__main__':
    input_folder = "E:\\Download\\face\\VoxCeleb1_test"  # Folder containing original preprocessed files
    dataset_output_folder = "E:\\Download\\face\\VoxCeleb1_test_dataset"

    generate_voxceleb_dataset(input_folder, dataset_output_folder)
