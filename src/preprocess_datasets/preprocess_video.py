import cv2
import numpy as np
import os

import time
import torch
from tqdm import tqdm

from src.preprocess_datasets.blazeface.blazeface import BlazeFace
from src.preprocess_datasets.blazeface.face_crop import resize_with_padding
from src.preprocess_datasets.detect_face import expand_bbox
from src.preprocess_datasets.headPoseEstimation.headpose_estimation import get_model, get_frames, process_hpe_batch


def analyse_video_vox(input_folder, output_folder, model_path_hpe, model_path_blazeface, device, batch_size=64, filter=None, keep=True, min_accepted_face_size=112):
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
        for video in files:
            if ".mp4" in video:
                if filter is not None:
                    if filter not in video:
                        continue
                os.makedirs(output_analysis_folder, exist_ok=True)
                imgs = get_frames(os.path.join(root, video))

                for counter, img in enumerate(imgs):
                    if img is not None:
                        video_frames.append(img)
                        video_names.append(video + "#" + str(counter))

                if not video_frames:
                    print("Error for", os.path.join(root, output_folder))

        if not video_frames:
            continue

        for i in range(0, len(imgs), batch_size):
            batch = [img for img in imgs[i:i + batch_size] if img is not None]
            if batch:

                # Blazeface
                padded_batch = []
                for image in batch:
                    pad_size = 16  # Pad the image Since (256-224)/2 = 16
                    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    padded_batch.append(padded_image)

                padded_batch = np.array(padded_batch)

                detections = back_net.predict_on_batch(padded_batch)

                cropped_batch = []
                for det, img in zip(detections, padded_batch):

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

                if len(cropped_batch) == 0:
                    continue

                cropped_batch = np.array(cropped_batch)

                # +++++++++++++++++++  HPE ++++++++++++++++++++
                batch_infos = process_hpe_batch(cropped_batch, device, head_pose_model)
                for j, info in enumerate(batch_infos):
                    info.append(video + "#" + str(i + j))
                    frame_hpe.append(info)

        assert len(frame_hpe) == len(frame_dets)
        if frame_dets:
            with open(output_txt_path, 'w') as txt_file:
                for info_hpe, info_dets in zip(frame_hpe, frame_dets):
                    txt_file.write(','.join(map(str, info_hpe)) + "," + ','.join(map(str, info_dets)) + '\n')

            #print(f"Processed: {root}")
        else:
            print(f"Processed {root} with no frames usable")

    elapsed_time = time.time() - start_time
    print("Video Analysis for ", num_folders, " in", round(elapsed_time / 60, 2), "minutes, missing_faces:", missing_faces, ", multiple_faces:", more_faces, ", total_faces:", missing_faces+more_faces+found_one_face, ", too_small:", too_small)


def analyse_video_nersemble(input_folder, output_folder, model_path_hpe, model_path_blazeface, device, batch_size=64, filter=None, keep=True, min_accepted_face_size=224):
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
        for video in files:
            if ".mp4" in video:
                if filter is not None:
                    if filter not in video:
                        continue
                os.makedirs(output_analysis_folder, exist_ok=True)
                imgs = get_frames(os.path.join(root, video))

                for counter, img in enumerate(imgs):
                    if img is not None:
                        video_frames.append(img)
                        video_names.append(video + "#" + str(counter))

                if not video_frames:
                    print("Error for", os.path.join(root, output_folder))

        if not video_frames:
            continue

        for i in range(0, len(imgs), batch_size):
            batch = [img for img in imgs[i:i + batch_size] if img is not None]
            if batch:

                # Blazeface
                padded_batch = []
                for image in batch:
                    resized_image, _, _, _ = resize_with_padding(image)  # Resize for BlazeFace detection
                    padded_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                    padded_batch.append(padded_image)

                padded_batch = np.array(padded_batch)
                detections = back_net.predict_on_batch(padded_batch)

                cropped_batch = []
                for det, img in zip(detections, padded_batch):

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

                if len(cropped_batch) == 0:
                    continue

                cropped_batch = np.array(cropped_batch)

                # +++++++++++++++++++  HPE ++++++++++++++++++++
                batch_infos = process_hpe_batch(cropped_batch, device, head_pose_model)
                for j, info in enumerate(batch_infos):
                    info.append(video + "#" + str(i + j))
                    frame_hpe.append(info)

        assert len(frame_hpe) == len(frame_dets)
        if frame_dets:
            with open(output_txt_path, 'w') as txt_file:
                for info_hpe, info_dets in zip(frame_hpe, frame_dets):
                    txt_file.write(','.join(map(str, info_hpe)) + "," + ','.join(map(str, info_dets)) + '\n')

            print(f"Processed: {root}")
        else:
            print(f"Processed {root} with no frames usable")

    elapsed_time = time.time() - start_time
    print("Video Analysis for ", num_folders, " in", round(elapsed_time / 60, 2), "minutes, missing_faces:", missing_faces, ", multiple_faces:", more_faces, ", total_faces:", missing_faces+more_faces+found_one_face, ", too_small:", too_small)
