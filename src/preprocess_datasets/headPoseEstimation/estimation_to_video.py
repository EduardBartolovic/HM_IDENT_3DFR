import os
import re

import cv2

from src.preprocess_datasets.headPoseEstimation.utils.general import draw_axis


def process_txt_and_create_videos(txt_root_folder, video_root_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for root, _, files in os.walk(txt_root_folder):  # Recursively walk through directories
        for txt_file in files:
            if txt_file.endswith('frame_infos.txt'):
                txt_path = os.path.join(root, txt_file)

                video_dir = os.path.join(video_root_folder, os.path.relpath(root, txt_root_folder))
                video_file = None
                for file in os.listdir(video_dir):
                    if file.endswith('.mp4'):
                        video_file = file
                        break

                if not video_file:
                    print(f"No matching video found for {txt_file}. Skipping...")
                    continue

                video_path = os.path.join(video_dir, video_file)

                # Create output video path
                relative_dir = os.path.relpath(root, txt_root_folder)
                save_dir = os.path.join(output_folder, relative_dir)
                os.makedirs(save_dir, exist_ok=True)
                output_video_path = os.path.join(save_dir, f"{os.path.splitext(txt_file)[0]}_cropped.mp4")

                # Open the video file
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Failed to open video {video_path}. Skipping...")
                    continue

                # Get video properties
                fps = 2#int(cap.get(cv2.CAP_PROP_FPS))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 videos

                # Prepare VideoWriter for cropped video
                out = None

                video_frame_counter = 0
                # Read bounding box data from txt file
                with open(txt_path, 'r') as file:
                    lines = file.readlines()

                for line in lines:
                    frame_info = list(map(int, line.strip().split(',')))
                    frame_number, x_min, y_min, x_max, y_max = frame_info[:5]
                    y_pred_deg, p_pred_deg, r_pred_deg = frame_info[5:]

                    while video_frame_counter != frame_number:
                        video_frame_counter += 1
                        success, frame = cap.read()
                        if not success:
                            break

                    if r_pred_deg > 1 or r_pred_deg < -1:
                        bbox_center_x = (x_min + x_max) // 2
                        bbox_center_y = (y_min + y_max) // 2
                        rotation_matrix = cv2.getRotationMatrix2D((bbox_center_x, bbox_center_y), -r_pred_deg, 1.0)

                        # Apply rotation to the full frame
                        frame = cv2.warpAffine(
                            frame,
                            rotation_matrix,
                            (frame.shape[1], frame.shape[0]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0)
                        )

                    cropped_frame = frame[y_min:y_max, x_min:x_max]

                    if True:
                        draw_axis(
                            frame,
                            y_pred_deg,
                            p_pred_deg,
                            r_pred_deg,
                            bbox=[x_min, y_min, x_max, y_max],
                            size_ratio=0.5
                        )

                    if out is None:
                        crop_width = x_max - x_min
                        crop_height = y_max - y_min
                        out = cv2.VideoWriter(output_video_path, fourcc, fps, (crop_width, crop_height))
                    out.write(cropped_frame)

                cap.release()
                if out:
                    out.release()
                print(f"Processed and saved cropped video: {output_video_path}")


def extract_video_index(file_name):
    """
    Extract the index number from a video filename using regex.
    Example: '0jWENYvy7HM#00001#313-462' => 1
    """
    match = re.search(r"#(\d+)", file_name)
    if match:
        return int(match.group(1))  # Extract the number and convert to integer
    return 0  # Fallback if no match is found


def merge_videos_in_folder(output_folder, final_output_folder):
    """
    Merge all cropped videos in each subfolder into one video using OpenCV,
    ensuring they are sorted based on their filenames' extracted index numbers.
    """
    os.makedirs(final_output_folder, exist_ok=True)

    for root, _, files in os.walk(output_folder):
        video_paths = [
            os.path.join(root, file) for file in files if file.endswith('_cropped.mp4')
        ]

        # Sort video paths by extracting the index number
        video_paths.sort(key=lambda x: extract_video_index(os.path.basename(x)))

        if video_paths:
            # Get properties from the first video
            first_video = cv2.VideoCapture(video_paths[0])
            if not first_video.isOpened():
                print(f"Failed to open video {video_paths[0]}. Skipping...")
                continue

            # Video properties
            fps = 1  # You can dynamically set this if needed
            width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 videos

            # Prepare the merged video path
            relative_path = os.path.relpath(root, output_folder)
            merged_video_path = os.path.join(final_output_folder, f"{relative_path}_merged.mp4")
            os.makedirs(os.path.dirname(merged_video_path), exist_ok=True)

            # Initialize VideoWriter
            out = cv2.VideoWriter(merged_video_path, fourcc, fps, (width, height))

            # Process and write frames from each sorted video
            for video_path in video_paths:
                cap = cv2.VideoCapture(video_path)
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    out.write(frame)
                cap.release()

            # Release resources
            out.release()
            print(f"Merged video saved: {merged_video_path}")


def merge_txt_files_in_folder(txt_root_folder, merged_txt_folder):
    """
    Merge all frame_infos.txt files in each subfolder into a single .txt file,
    sorted by the first number in filenames, and renumber frames sequentially.
    """
    os.makedirs(merged_txt_folder, exist_ok=True)

    for root, _, files in os.walk(txt_root_folder):
        # Sort files numerically by the first number in the filename
        txt_files = sorted(
            [file for file in files if file.endswith('frame_infos.txt')],
            key=lambda x: int(x.split('_')[0])
        )

        merged_lines = []
        new_frame_number = 0  # Counter for new sequential frame numbers

        for txt_file in txt_files:
            txt_path = os.path.join(root, txt_file)
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # Parse the line
                    frame_info = line.strip().split(',')
                    frame_info[0] = str(new_frame_number)  # Update the frame number
                    new_frame_number += 1
                    merged_lines.append(','.join(frame_info) + '\n')

        if merged_lines:
            # Save the merged file
            relative_dir = os.path.relpath(root, txt_root_folder)
            merged_txt_path = os.path.join(merged_txt_folder, f"{relative_dir}_merged.txt")
            os.makedirs(os.path.dirname(merged_txt_path), exist_ok=True)

            with open(merged_txt_path, 'w') as merged_txt_file:
                merged_txt_file.writelines(merged_lines)

            print(f"Merged .txt file saved: {merged_txt_path}")


# Example usage
txt_folder = "F:\\Face\\HPE\\VoxCeleb1_test_out\\video"  # Folder containing frame_infos.txt files
video_folder = "F:\\Face\\HPE\\VoxCeleb1_test\\video"  # Folder containing original video files
output_folder = "F:\\Face\\HPE\\hpe_cropped"  # Folder to save cropped videos


process_txt_and_create_videos(txt_folder, video_folder, output_folder)

#merge_txt_files_in_folder(txt_folder, merged_txt_folder)

#merge_videos_in_folder(output_folder, final_output_folder)