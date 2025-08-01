from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import os

import time
import torch
from sklearn.utils import deprecated
from tqdm import tqdm

from src.preprocess_datasets.blazeface.blazeface import BlazeFace
from src.preprocess_datasets.blazeface.face_crop import resize_with_padding
from src.preprocess_datasets.detect_face import expand_bbox
from src.preprocess_datasets.headPoseEstimation.headpose_estimation import get_model, get_frames, process_hpe_batch, \
    process_video, get_images_from_dir
from insightface.app import FaceAnalysis

@deprecated
def analyse_video_vox(input_folder, output_folder, model_path_hpe, model_path_blazeface, device, batch_size=64, filter=None, keep=True, min_accepted_face_size=112, frame_skip=2, max_workers=8, face_confidence=0.6):
    start_time = time.time()

    head_pose_model = get_model("resnet50", num_classes=6)
    state_dict = torch.load(model_path_hpe, map_location=device, weights_only=True)
    head_pose_model.load_state_dict(state_dict)
    head_pose_model.to(device)
    head_pose_model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    back_net = BlazeFace(back_model=True).to(device)
    back_net.load_weights(os.path.join(model_path_blazeface, "blazefaceback.pth"))
    back_net.load_anchors(os.path.join(model_path_blazeface, "anchorsback.npy"))
    back_net.min_score_thresh = 0.45
    back_net.min_suppression_threshold = 0.3

    missing_faces = 0
    more_faces = 0
    found_one_face = 0
    too_small = 0

    hpe_counter = 0

    folders = list(os.walk(input_folder))
    num_folders = len(folders)
    for root, _, files in tqdm(folders, desc="Processing folders"):

        frame_dets = []
        frame_hpe = []
        output_analysis_folder = os.path.join(root, output_folder)
        output_txt_path = os.path.join(output_analysis_folder, "analysis.txt")

        # Skip video if analysis.txt already exists
        if keep and os.path.exists(output_txt_path):
            continue

        video_frames = []
        video_names = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for video in files:
                if video.endswith(".mp4"):
                    video_path = os.path.join(root, video)
                    futures.append(executor.submit(process_video, video_path, frame_skip, output_analysis_folder))
        for future in futures:
            frames, names = future.result()
            video_frames.extend(frames)
            video_names.extend(names)

        if not video_frames:
            continue

        for i in range(0, len(video_frames), batch_size):
            img_batch = [img for img in video_frames[i:i + batch_size] if img is not None]
            name_batch = [name for name in video_names[i:i + batch_size] if name is not None]
            if img_batch:

                # Blazeface
                padded_batch = []
                for image in img_batch:
                    pad_size = 16  # Pad the image Since (256-224)/2 = 16
                    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    padded_batch.append(padded_image)

                padded_batch = np.array(padded_batch)
                detections = back_net.predict_on_batch(padded_batch)

                cropped_batch = []
                valid_names_batch = []
                for det, img, name in zip(detections, padded_batch, name_batch):  # Iterate over batch
                    det = np.array([d for d in det.cpu().numpy() if d[-1] >= face_confidence])  # Remove det below a set confidence
                    if det.shape[0] == 0:
                        missing_faces += 1
                        continue
                    elif det.shape[0] > 1:
                        # plot_detections(padded_image, detections)
                        more_faces += 1
                        det = max(det, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                    else:
                        if det[0][-1] < 0.5:
                            print(det)
                        found_one_face += 1
                        det = det[0]

                    y_min = int(det[0] * img.shape[0])
                    x_min = int(det[1] * img.shape[1])
                    y_max = int(det[2] * img.shape[0])
                    x_max = int(det[3] * img.shape[1])

                    x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max, factor=0.1)
                    assert (x_max - x_min) == (y_max - y_min)
                    if x_max - x_min < min_accepted_face_size:
                        too_small += 1
                        continue

                    face_crop = img[y_min:y_max, x_min:x_max]
                    face_crop_resized = cv2.resize(face_crop, (224, 224))
                    cropped_batch.append(face_crop_resized)
                    frame_dets.append(np.array([x_min, y_min, x_max, y_max]))
                    valid_names_batch.append(name)

                if len(cropped_batch) == 0:
                    continue

                cropped_batch = np.array(cropped_batch)

                # +++++++++++++++++++  HPE ++++++++++++++++++++
                batch_infos = process_hpe_batch(cropped_batch, device, head_pose_model)
                for j, info in enumerate(batch_infos):
                    info.append(valid_names_batch[j])
                    frame_hpe.append(info)

            assert len(frame_hpe) == len(frame_dets)
            if frame_dets:
                try:
                    with open(output_txt_path, 'w') as txt_file:
                        for info_hpe, info_dets in zip(frame_hpe, frame_dets):
                            txt_file.write(','.join(map(str, info_hpe)) + "," + ','.join(map(str, info_dets)) + '\n')
                except KeyboardInterrupt:
                    print(
                        f"\nInterrupted while saving at {output_txt_path}. Attempting to finish saving and exit cleanly.")
                    with open(output_txt_path, 'w') as txt_file:
                        for info_hpe, info_dets in zip(frame_hpe, frame_dets):
                            txt_file.write(','.join(map(str, info_hpe)) + "," + ','.join(map(str, info_dets)) + '\n')
                    exit()
                # print(f"Processed: {root}")
            else:
                print(f"Processed {root} with no frames usable")

    elapsed_time = time.time() - start_time
    print("Video Analysis for ", num_folders, " in", round(elapsed_time / 60, 2), "minutes, missing_faces:", missing_faces, ", multiple_faces:", more_faces, ", total_faces:", missing_faces+more_faces+found_one_face, ", too_small:", too_small, "hpe on", hpe_counter, "frames")


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
    face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'], allowed_modules=['detection'])
    face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    face_app.det_model.conf_threshold = face_confidence

    missing_faces = 0
    more_faces = 0
    found_one_face = 0
    too_small = 0

    folders = list(os.walk(input_folder))
    num_folders = len(folders)
    for root, _, files in tqdm(folders, desc="Processing folders"):

        frame_dets = []
        frame_hpe = []
        output_analysis_folder = os.path.join(root, output_folder)
        output_txt_path = os.path.join(output_analysis_folder, "analysis.txt")

        if keep and os.path.exists(output_txt_path):
            continue

        os.makedirs(output_analysis_folder, exist_ok=True)

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

            for img, name in zip(img_batch, name_batch):
                if padding:
                    img = pad_to_minimum_size(img)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                faces = face_app.get(img_rgb)

                if not faces:
                    missing_faces += 1
                    continue

                # Filter by landmark visibility (≥90%)
                valid_faces = []
                h, w = img.shape[:2]
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

                x_min, y_min, x_max, y_max = map(int, face.bbox)
                x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max, factor=0.0)
                face_size = min(x_max - x_min, y_max - y_min)
                if face_size < min_accepted_face_size:
                    too_small += 1
                    continue

                face_crop = img[y_min:y_max, x_min:x_max]
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
                frame_hpe.append(info)

        assert len(frame_hpe) == len(frame_dets)
        if frame_dets:
            try:
                with open(output_txt_path, 'w') as txt_file:
                    for info_hpe, info_dets in zip(frame_hpe, frame_dets):
                        txt_file.write(','.join(map(str, info_hpe)) + "," + ','.join(map(str, info_dets)) + '\n')
            except KeyboardInterrupt:
                print(f"\nInterrupted while saving at {output_txt_path}. Attempting to finish saving and exit cleanly.")
                with open(output_txt_path, 'w') as txt_file:
                    for info_hpe, info_dets in zip(frame_hpe, frame_dets):
                        txt_file.write(','.join(map(str, info_hpe)) + "," + ','.join(map(str, info_dets)) + '\n')
                exit()
        else:
            print(f"Processed {root} with no frames usable")

    elapsed_time = time.time() - start_time
    print("Video Analysis for", num_folders, "in", round(elapsed_time / 60, 2), "minutes")
    print("missing_faces:", missing_faces, ", multiple_faces:", more_faces, ", total_faces:", missing_faces + more_faces + found_one_face, ", too_small:", too_small)


@deprecated
def analyse_video_nersemble(input_folder, output_folder, model_path_hpe, model_path_blazeface, device, batch_size=64, keep=True, min_accepted_face_size=64, frame_skip=8, downscale=True, max_workers=8, face_confidence=0.6):
    start_time = time.time()

    head_pose_model = get_model("resnet50", num_classes=6)
    state_dict = torch.load(model_path_hpe, map_location=device, weights_only=True)
    head_pose_model.load_state_dict(state_dict)
    head_pose_model.to(device)
    head_pose_model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    back_net = BlazeFace(back_model=True).to(device)
    back_net.load_weights(os.path.join(model_path_blazeface, "blazefaceback.pth"))
    back_net.load_anchors(os.path.join(model_path_blazeface, "anchorsback.npy"))
    back_net.min_score_thresh = 0.45
    back_net.min_suppression_threshold = 0.3

    missing_faces = 0
    more_faces = 0
    found_one_face = 0
    too_small = 0

    folders = list(os.walk(input_folder))
    num_folders = len(folders)
    for root, _, files in tqdm(folders, desc="Processing folders"):

        frame_dets = []
        frame_hpe = []
        output_analysis_folder = os.path.join(root, output_folder)
        output_txt_path = os.path.join(output_analysis_folder, "analysis.txt")

        # Skip video if analysis.txt already exists
        if keep and os.path.exists(output_txt_path):
            continue

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
            if img_batch:

                # Blazeface
                padded_batch = []
                for image in img_batch:
                    resized_image, _, _ = resize_with_padding(image)  # Resize for BlazeFace detection
                    padded_batch.append(resized_image)

                padded_batch = np.array(padded_batch)
                detections = back_net.predict_on_batch(padded_batch)

                cropped_batch = []
                valid_names_batch = []
                for det, img, name in zip(detections, padded_batch, name_batch):  # Iterate over batch
                    det = np.array([d for d in det.cpu().numpy() if d[-1] >= face_confidence])  # Remove det below a set confidence
                    if det.shape[0] == 0:
                        missing_faces += 1
                        continue
                    elif det.shape[0] > 1:
                        # plot_detections(padded_image, detections)
                        more_faces += 1
                        det = max(det, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                    else:
                        found_one_face += 1
                        det = det[0]

                    y_min = int(det[0] * img.shape[0])
                    x_min = int(det[1] * img.shape[1])
                    y_max = int(det[2] * img.shape[0])
                    x_max = int(det[3] * img.shape[1])

                    x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max, factor=0.1)
                    assert (x_max - x_min) == (y_max - y_min)
                    if x_max - x_min < min_accepted_face_size:
                        too_small += 1
                        continue

                    face_crop = img[y_min:y_max, x_min:x_max]
                    face_crop_resized = cv2.resize(face_crop, (224, 224))
                    cropped_batch.append(face_crop_resized)
                    frame_dets.append(np.array([x_min, y_min, x_max, y_max]))
                    valid_names_batch.append(name)

                if len(cropped_batch) == 0:
                    continue

                cropped_batch = np.array(cropped_batch)

                # +++++++++++++++++++  HPE ++++++++++++++++++++
                batch_infos = process_hpe_batch(cropped_batch, device, head_pose_model)
                for j, info in enumerate(batch_infos):
                    info.append(valid_names_batch[j])
                    frame_hpe.append(info)

        assert len(frame_hpe) == len(frame_dets)
        if frame_dets:
            try:
                with open(output_txt_path, 'w') as txt_file:
                    for info_hpe, info_dets in zip(frame_hpe, frame_dets):
                        txt_file.write(','.join(map(str, info_hpe)) + "," + ','.join(map(str, info_dets)) + '\n')
            except KeyboardInterrupt:
                print(f"\nInterrupted while saving at {output_txt_path}. Attempting to finish saving and exit cleanly.")
                with open(output_txt_path, 'w') as txt_file:
                    for info_hpe, info_dets in zip(frame_hpe, frame_dets):
                        txt_file.write(','.join(map(str, info_hpe)) + "," + ','.join(map(str, info_dets)) + '\n')
                exit()
            # print(f"Processed: {root}")
        else:
            print(f"Processed {root} with no frames usable")

    elapsed_time = time.time() - start_time
    print("Video Analysis for ", num_folders, " in", round(elapsed_time / 60, 2), "minutes, missing_faces:", missing_faces, ", multiple_faces:", more_faces, ", total_faces:", missing_faces+more_faces+found_one_face, ", too_small:", too_small)


def analyse_video_ytf(input_folder, output_folder, model_path_hpe, model_path_blazeface, device, batch_size=64, filter=None, keep=True, min_accepted_face_size=112, frame_skip=2, max_workers=8, face_confidence=0.6):
    start_time = time.time()

    head_pose_model = get_model("resnet50", num_classes=6)
    state_dict = torch.load(model_path_hpe, map_location=device, weights_only=True)
    head_pose_model.load_state_dict(state_dict)
    head_pose_model.to(device)
    head_pose_model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    back_net = BlazeFace(back_model=True).to(device)
    back_net.load_weights(os.path.join(model_path_blazeface, "blazefaceback.pth"))
    back_net.load_anchors(os.path.join(model_path_blazeface, "anchorsback.npy"))
    back_net.min_score_thresh = 0.25
    back_net.min_suppression_threshold = 0.3

    missing_faces = 0
    more_faces = 0
    found_one_face = 0
    too_small = 0

    hpe_counter = 0

    folders = list(os.walk(input_folder))
    num_folders = len(folders)
    for root, _, files in tqdm(folders, desc="Processing folders"):

        if len(files) == 0:
            continue
        if '.jpg' not in files[0]:
            continue

        frame_dets = []
        frame_hpe = []
        output_analysis_folder = os.path.join(root, output_folder)
        output_txt_path = os.path.join(output_analysis_folder, "analysis.txt")

        # Skip video if analysis.txt already exists
        if keep and os.path.exists(output_txt_path):
            continue
        os.makedirs(output_analysis_folder, exist_ok=True)

        video_frames = get_images_from_dir(root, files)
        video_names = files

        if not video_frames:
            continue

        for i in range(0, len(video_frames), batch_size):
            img_batch = [img for img in video_frames[i:i + batch_size] if img is not None]
            name_batch = [name for name in video_names[i:i + batch_size] if name is not None]
            if img_batch:

                # Blazeface
                padded_batch = []
                for image in img_batch:
                    target_size = 256
                    h, w = image.shape[:2]
                    # Step 1: Resize if needed (scale down if larger than target size)
                    if max(h, w) > target_size:
                        scaling_factor = target_size / max(h, w)
                        new_w = int(w * scaling_factor)
                        new_h = int(h * scaling_factor)
                        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        h, w = new_h, new_w

                    # Step 2: Pad if needed (pad smaller images to target size)
                    pad_h = max(target_size - h, 0)
                    pad_w = max(target_size - w, 0)

                    top = pad_h // 2
                    bottom = pad_h - top
                    left = pad_w // 2
                    right = pad_w - left

                    padded_image = cv2.copyMakeBorder(
                        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                    )

                    # Just in case: crop to 256x256 (shouldn't be needed, but safe)
                    padded_image = cv2.resize(padded_image, (target_size, target_size))

                    padded_batch.append(padded_image)

                padded_batch = np.array(padded_batch)
                detections = back_net.predict_on_batch(padded_batch)

                cropped_batch = []
                valid_names_batch = []
                for det, img, name in zip(detections, padded_batch, name_batch):  # Iterate over batch
                    det = np.array([d for d in det.cpu().numpy() if d[-1] >= face_confidence])  # Remove det below a set confidence
                    if det.shape[0] == 0:
                        missing_faces += 1
                        y_min, x_min, y_max, x_max = 0, 0, img.shape[0], img.shape[1]
                    elif det.shape[0] > 1:
                        # plot_detections(padded_image, detections)
                        more_faces += 1
                        det = max(det, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                        y_min = int(det[0] * img.shape[0])
                        x_min = int(det[1] * img.shape[1])
                        y_max = int(det[2] * img.shape[0])
                        x_max = int(det[3] * img.shape[1])
                    else:
                        if det[0][-1] < 0.5:
                            print(det)
                        found_one_face += 1
                        det = det[0]
                        y_min = int(det[0] * img.shape[0])
                        x_min = int(det[1] * img.shape[1])
                        y_max = int(det[2] * img.shape[0])
                        x_max = int(det[3] * img.shape[1])

                    x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max, factor=0)
                    assert (x_max - x_min) == (y_max - y_min)
                    if x_max - x_min < min_accepted_face_size:
                        too_small += 1
                        continue

                    face_crop = img[y_min:y_max, x_min:x_max]
                    face_crop_resized = cv2.resize(face_crop, (224, 224))
                    cropped_batch.append(face_crop_resized)
                    frame_dets.append(np.array([x_min, y_min, x_max, y_max]))
                    valid_names_batch.append(name)

                if len(cropped_batch) == 0:
                    continue

                cropped_batch = np.array(cropped_batch)

                # +++++++++++++++++++  HPE ++++++++++++++++++++
                batch_infos = process_hpe_batch(cropped_batch, device, head_pose_model)
                for j, info in enumerate(batch_infos):
                    info.append(valid_names_batch[j])
                    frame_hpe.append(info)

            assert len(frame_hpe) == len(frame_dets)
            if frame_dets:
                try:
                    with open(output_txt_path, 'w') as txt_file:
                        for info_hpe, info_dets in zip(frame_hpe, frame_dets):
                            txt_file.write(','.join(map(str, info_hpe)) + "," + ','.join(map(str, info_dets)) + '\n')
                except KeyboardInterrupt:
                    print(
                        f"\nInterrupted while saving at {output_txt_path}. Attempting to finish saving and exit cleanly.")
                    with open(output_txt_path, 'w') as txt_file:
                        for info_hpe, info_dets in zip(frame_hpe, frame_dets):
                            txt_file.write(','.join(map(str, info_hpe)) + "," + ','.join(map(str, info_dets)) + '\n')
                    exit()
                # print(f"Processed: {root}")
            else:
                print(f"Processed {root} with no frames usable")

    elapsed_time = time.time() - start_time
    print("Video Analysis for ", num_folders, " in", round(elapsed_time / 60, 2), "minutes, missing_faces:", missing_faces, ", multiple_faces:", more_faces, ", total_faces:", missing_faces+more_faces+found_one_face, ", too_small:", too_small, "hpe on", hpe_counter, "frames")
