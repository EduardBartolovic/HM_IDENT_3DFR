import os
import shutil

import cv2
from mononphm import env_paths


def move_video_to_new_folder(input_dir, output_dir):
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        print(f"The input directory {input_dir} does not exist.")
        return

    # Ensure the output directory exists, if not create it
    os.makedirs(output_dir, exist_ok=True)

    for dir_name in os.listdir(input_dir):

        video_list = os.listdir(os.path.join(input_dir, dir_name))
        video_list2 = os.listdir(os.path.join(input_dir, dir_name, video_list[0], 'chunk_videos'))

        # Full path of the input file
        input_file_path = os.path.join(input_dir, dir_name, video_list[0], 'chunk_videos', video_list2[0])

        # Skip directories, we are only interested in files
        if not os.path.isfile(input_file_path):
            raise AttributeError('Folder found!')

        # Full path of the new directory in the output directory
        new_folder_path = os.path.join(output_dir, dir_name)
        new_folder_path_source = os.path.join(output_dir, dir_name, 'source')

        # Check if the frames are already extracted
        if os.path.exists(new_folder_path_source) and len(os.listdir(new_folder_path_source)) > 0:
            print(f"Frames already extracted for {dir_name}. Skipping...")
            continue

        try:
            # Create the new directory
            os.makedirs(new_folder_path, exist_ok=True)
            os.makedirs(new_folder_path_source, exist_ok=True)

            cap = cv2.VideoCapture(input_file_path)
            frame_number = 0
            while True:
                # Read the next frame from the video
                ret, frame = cap.read()

                # If reading a frame was not successful, break the loop
                if not ret:
                    break

                # Define the filename with an increasing number
                frame_filename = os.path.join(new_folder_path_source, f"{frame_number:05d}.png")

                # Write the frame to the output directory
                cv2.imwrite(frame_filename, frame)

                # Increment the frame number
                frame_number += 1

            # Release the video capture object
            cap.release()
            # Rename and Move the file into the new directory
            print(f"Extracted {frame_number - 1} frames to {output_dir}")

        except Exception as e:
            print(f"An error occurred while processing {dir_name}: {e}")



def apply_pre_processing(working_dir):

    for dir_name in os.listdir(working_dir):
        print(dir_name)
        os.system(f'cd {env_paths.CODE_BASE}/scripts/preprocessing/; bash run.sh {dir_name} --no-intrinsics_provided')
        break


############
input_dir = "/home/gustav/voxceleb/VoxCeleb2_test/"
MonoNPHM_dataset_tracking_dir = "/home/gustav/MonoNPHM/dataset_tracking"

move_video_to_new_folder(input_dir, MonoNPHM_dataset_tracking_dir)
apply_pre_processing(MonoNPHM_dataset_tracking_dir)

