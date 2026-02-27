from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import os

import time
import torch
from tqdm import tqdm

from src.preprocess_datasets.blazeface.blazeface import BlazeFace
from src.preprocess_datasets.misc.detect_face import expand_bbox
from src.preprocess_datasets.headPoseEstimation.headpose_estimation import get_model, process_hpe_batch, \
    get_images_from_dir
import pickle
from insightface.app import FaceAnalysis


def process_video(video_path, frame_skip, output_analysis_folder, downscale=False):
    os.makedirs(output_analysis_folder, exist_ok=True)
    frames, names = get_frames(video_path, frame_skip=frame_skip, downscale=downscale)

    if not frames:
        print("Error processing", video_path)
        return [], []

    return frames, names


def get_frames(video_path, frame_skip=1, downscale=False):
    cap = cv2.VideoCapture(video_path)
    frames = []
    names = []
    video_filename = os.path.basename(video_path)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_skip == 0:

            if downscale:
                height, width = frame.shape[:2]
                if height > 1500 or width > 1500:
                    frame = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            names.append(f"{video_filename}#{frame_index}")
        frame_index += 1

    cap.release()
    return frames, names


def pad_to_minimum_size(img, min_size=640):
    h, w = img.shape[:2]
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded_img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded_img


def analyse_video_hpe(input_folder, output_folder, model_path_hpe, device, batch_size=64, keep=True, min_accepted_face_size=64, frame_skip=8, downscale=True, max_workers=8, face_confidence=0.5, padding=False):
    start_time = time.time()

    # Load HPE model
    head_pose_model = get_model("resnet50", num_classes=6)
    state_dict = torch.load(model_path_hpe, map_location=device, weights_only=True)
    head_pose_model.load_state_dict(state_dict)
    head_pose_model.to(device)
    head_pose_model.eval()

    # Setup InsightFace
    face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
    face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    face_app.det_model.conf_threshold = face_confidence

    missing_faces = 0
    more_faces = 0
    found_one_face = 0
    too_small = 0
    already_exist = 0

    folders = list(os.walk(input_folder))
    num_folders = len(folders)
    print("Searching in:", input_folder, "with", num_folders, "folders")
    for root, _, files in tqdm(folders, desc="Processing folders"):

        video_files = [f for f in files if os.path.splitext(f)[1].lower() in {'.mp4', '.mov', '.avi', '.mkv'}]
        if not video_files:
            continue

        output_analysis_folder = os.path.join(root, output_folder)
        output_pkl_path = os.path.join(output_analysis_folder, "analysis.pkl")

        if keep and os.path.exists(output_pkl_path):
            already_exist += 1
            continue

        os.makedirs(output_analysis_folder, exist_ok=True)

        frame_dets = []
        frame_hpe = []

        video_frames = []
        video_names = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for video in files:
                if video.endswith(".mp4"):
                    video_path = os.path.join(root, video)
                    futures.append(executor.submit(process_video, video_path, frame_skip, output_analysis_folder, downscale))
        for future in futures:
            frames, names = future.result()
            video_frames.extend(frames)
            video_names.extend(names)

        if not video_frames:
            continue

        for i in range(0, len(video_frames), batch_size):
            img_batch = [img for img in video_frames[i:i + batch_size] if img is not None]
            name_batch = [name for name in video_names[i:i + batch_size] if name is not None]
            if not img_batch:
                continue

            cropped_batch = []
            valid_names_batch = []
            embeddings_batch = []
            for img_bgr, name in zip(img_batch, name_batch):
                if padding:
                    img_bgr = pad_to_minimum_size(img_bgr)

                faces = face_app.get(img_bgr)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                if not faces:
                    missing_faces += 1
                    continue

                # Filter by landmark visibility (≥90%)
                valid_faces = []
                h, w = img_rgb.shape[:2]
                for face in faces:
                    kps = face.kps
                    if kps is None or len(kps) == 0:
                        continue
                    visible = [(0 <= x < w) and (0 <= y < h) for x, y in kps]
                    if sum(visible) / len(visible) >= 0.9:
                        valid_faces.append(face)

                if len(valid_faces) == 0:
                    missing_faces += 1
                    continue

                if len(valid_faces) > 1:
                    more_faces += 1
                    face = max(valid_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                else:
                    found_one_face += 1
                    face = valid_faces[0]

                embeddings_batch.append(face.normed_embedding)

                x_min, y_min, x_max, y_max = map(int, face.bbox)
                x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max, factor=0.0)
                face_size = min(x_max - x_min, y_max - y_min)
                if face_size < min_accepted_face_size:
                    too_small += 1
                    continue

                face_crop = img_rgb[y_min:y_max, x_min:x_max]
                face_crop_resized = cv2.resize(face_crop, (224, 224))
                cropped_batch.append(face_crop_resized)
                frame_dets.append(np.array([x_min, y_min, x_max, y_max]))
                valid_names_batch.append(name)

            if len(cropped_batch) == 0:
                continue

            cropped_batch = np.array(cropped_batch)

            # HPE inference
            batch_infos = process_hpe_batch(cropped_batch, device, head_pose_model)
            for j, info in enumerate(batch_infos):
                info.append(valid_names_batch[j])
                info.append(embeddings_batch[j])
                frame_hpe.append(info)

        assert len(frame_hpe) == len(frame_dets)
        if frame_dets:
            try:
                with open(output_pkl_path, 'wb') as pkl_file:
                    pickle.dump({
                        "frame_hpe": frame_hpe,
                        "frame_dets": frame_dets
                    }, pkl_file)
            except KeyboardInterrupt:
                print(f"\nInterrupted while saving at {output_pkl_path}. Attempting to finish saving and exited cleanly.")
                with open(output_pkl_path, 'wb') as pkl_file:
                    pickle.dump({
                        "frame_hpe": frame_hpe,
                        "frame_dets": frame_dets
                    }, pkl_file)
                exit()
        else:
            print(f"Processed {root} with no frames usable")

    elapsed_time = time.time() - start_time
    print("Video Analysis for", num_folders, "in", round(elapsed_time / 60, 2), "minutes")
    print("missing_faces:", missing_faces, ", multiple_faces:", more_faces, ", total_faces:", missing_faces + more_faces + found_one_face, ", too_small:", too_small, ", already_exist", already_exist)


def analyse_images_hpe(input_folder, output_folder, model_path_hpe, device,
                       batch_size=64, keep=True, min_accepted_face_size=64,
                       downscale=True, face_confidence=0.5, padding=False):

    start_time = time.time()

    # Load HPE model
    head_pose_model = get_model("resnet50", num_classes=6)
    state_dict = torch.load(model_path_hpe, map_location=device, weights_only=True)
    head_pose_model.load_state_dict(state_dict)
    head_pose_model.to(device)
    head_pose_model.eval()

    # Setup InsightFace
    face_app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition']
    )
    face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    face_app.det_model.conf_threshold = face_confidence

    missing_faces = 0
    more_faces = 0
    found_one_face = 0
    too_small = 0
    already_exist = 0

    folders = list(os.walk(input_folder))
    num_folders = len(folders)

    print("Searching in:", input_folder, "with", num_folders, "folders")

    for root, _, files in tqdm(folders, desc="Processing folders"):

        image_files = [
            f for f in files
            if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        ]

        if not image_files:
            continue

        output_analysis_folder = os.path.join(root, output_folder)
        output_pkl_path = os.path.join(output_analysis_folder, "analysis.pkl")

        if keep and os.path.exists(output_pkl_path):
            already_exist += 1
            continue

        os.makedirs(output_analysis_folder, exist_ok=True)

        frame_dets = []
        frame_hpe = []

        # Load all images in folder
        images = []
        image_names = []

        for img_name in image_files:
            img_path = os.path.join(root, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            if downscale:
                h, w = img.shape[:2]
                if max(h, w) > 1280:
                    scale = 1280 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))

            images.append(img)
            image_names.append(img_name)

        if not images:
            continue

        # Batch processing
        for i in range(0, len(images), batch_size):

            img_batch = images[i:i + batch_size]
            name_batch = image_names[i:i + batch_size]

            cropped_batch = []
            valid_names_batch = []
            embeddings_batch = []

            for img_bgr, name in zip(img_batch, name_batch):

                if padding:
                    img_bgr = pad_to_minimum_size(img_bgr)

                faces = face_app.get(img_bgr)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                if not faces:
                    missing_faces += 1
                    continue

                valid_faces = []
                h, w = img_rgb.shape[:2]

                for face in faces:
                    kps = face.kps
                    if kps is None or len(kps) == 0:
                        continue

                    visible = [(0 <= x < w) and (0 <= y < h) for x, y in kps]
                    if sum(visible) / len(visible) >= 0.9:
                        valid_faces.append(face)

                if len(valid_faces) == 0:
                    missing_faces += 1
                    continue

                if len(valid_faces) > 1:
                    more_faces += 1
                    face = max(valid_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                else:
                    found_one_face += 1
                    face = valid_faces[0]

                embeddings_batch.append(face.normed_embedding)

                x_min, y_min, x_max, y_max = map(int, face.bbox)
                x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max, factor=0.0)

                face_size = min(x_max - x_min, y_max - y_min)
                if face_size < min_accepted_face_size:
                    too_small += 1
                    continue

                face_crop = img_rgb[y_min:y_max, x_min:x_max]
                face_crop_resized = cv2.resize(face_crop, (224, 224))

                cropped_batch.append(face_crop_resized)
                frame_dets.append(np.array([x_min, y_min, x_max, y_max]))
                valid_names_batch.append(name)

            if len(cropped_batch) == 0:
                continue

            cropped_batch = np.array(cropped_batch)

            # HPE inference
            batch_infos = process_hpe_batch(cropped_batch, device, head_pose_model)

            for j, info in enumerate(batch_infos):
                info.append(valid_names_batch[j])
                info.append(embeddings_batch[j])
                frame_hpe.append(info)

        assert len(frame_hpe) == len(frame_dets)

        if frame_dets:
            with open(output_pkl_path, 'wb') as pkl_file:
                pickle.dump({
                    "frame_hpe": frame_hpe,
                    "frame_dets": frame_dets
                }, pkl_file)
        else:
            print(f"Processed {root} with no usable images")

    elapsed_time = time.time() - start_time
    print("Image Analysis for", num_folders, "folders in", round(elapsed_time / 60, 2), "minutes")
    print("missing_faces:", missing_faces, ", multiple_faces:", more_faces, ", total_faces:", missing_faces + more_faces + found_one_face, ", too_small:", too_small, ", already_exist:", already_exist)
