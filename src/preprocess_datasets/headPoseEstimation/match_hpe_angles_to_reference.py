import pickle

import csv
import os
import random
import time

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm


def parse_analysis_file(file_path, correct_angles=False):
    """Parse one analysis.pkl file into a NumPy array of objects."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    frame_hpe = data["frame_hpe"]
    frame_dets = data["frame_dets"]

    parsed_lines = []
    for info, det in zip(frame_hpe, frame_dets):
        angles = info[:3]  # HPE angles
        if correct_angles:
            angles[0], angles[1] = correct_angle_pair(angles[0], angles[1])

        filename_frame = info[3]  # filename
        embedding = info[4]       # embedding

        bbox = list(det)  # x_min, y_min, x_max, y_max

        parsed_lines.append(angles + [filename_frame] + bbox + [embedding])

    return np.array(parsed_lines, dtype=object)


def match_hpe_angles_to_references(data, references, ignore_roll=True, allow_flip=False, random_choice=False):
    used_indices = set()
    matches = []

    for reference in references:

        # consider roll or not
        if ignore_roll:
            ref_vec = np.array(reference[:2], dtype=float)
            data_vecs = np.array(data[:, :2], dtype=float)
        else:
            ref_vec = np.array(reference[:3], dtype=float)
            data_vecs = np.array(data[:, :3], dtype=float)

        # compute distances
        distances = np.linalg.norm(data_vecs - ref_vec, axis=1)
        was_flipped_array = np.zeros(len(data), dtype=bool)

        if allow_flip:
            flipped_ref = ref_vec.copy()
            flipped_ref[1] = -flipped_ref[1]  # flip yaw (index 1)

            flipped_distances = np.linalg.norm(data_vecs - flipped_ref, axis=1)

            # check where flipped is better
            better_is_flipped = flipped_distances < distances
            was_flipped_array = better_is_flipped.copy()

            # use better distance
            distances = np.where(better_is_flipped, flipped_distances, distances)

        if random_choice:
            available_indices = [i for i in range(len(data)) if i not in used_indices]
            if not available_indices:
                chosen_index = random.choice(range(len(data)))
            else:
                chosen_index = random.choice(available_indices)
        else:
            chosen_index = int(np.argmin(distances))

        min_distance = distances[chosen_index]
        was_flipped = bool(was_flipped_array[chosen_index]) if allow_flip else False

        # add flip info to matches
        matches.append((reference, data[chosen_index], min_distance, was_flipped))
        used_indices.add(chosen_index)

    return matches


def correct_angle_pair(x, y):
    """
    Corrects a pair of angles (x, y) based only on their signs,
    ignoring their actual magnitude values.

    Mapping logic based on signs:
    - If signs of x and y are the same (both positive or both negative),
      mirror both by flipping their signs.
    - Otherwise, leave them unchanged.

    This corresponds to your example where:
    (-25, -25) -> (25, 25)
    (25, 25) -> (-25, -25)
    but
    (-25, 25) and (25, -25) remain unchanged.
    """
    # TODO: Check if this makes sense for pitch.
    return -y, -x
    # Check if signs are the same
    #if (x >= 0 and y >= 0) or (x < 0 and y < 0):
    #    return -x, -y
    #else:
    #    return x, y


def remove_embedding_outliers_lof(data, n_neighbors=20, contamination=0.05):
    embeddings = np.stack(data[:, -1])
    if n_neighbors > embeddings.shape[0]:
        n_neighbors = embeddings.shape[0]
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, metric='cosine')
    labels = lof.fit_predict(embeddings)
    keep_mask = labels == 1
    removed_count = np.sum(~keep_mask)
    return data[keep_mask], removed_count


def save_matches(output_file, infos):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Ref_X", "Ref_Y", "Ref_Z",
            "Hpe_X", "Hpe_Y", "Hpe_Z",
            "Error",
            "file_name",
            "BBox_x_min", "BBox_y_min", "BBox_x_max", "BBox_y_max", "Flipped"
        ])
        for ref_angles, hpe_data, error, flipped in infos:
            ref_angles = ref_angles.tolist()
            hpe_angles = hpe_data[:3].tolist()
            filename = hpe_data[3]
            if len(filename.split("#")) >= 2:
                filename = "#".join(filename.split("#")[0:2])
            bbox = hpe_data[4:8]
            writer.writerow(ref_angles + hpe_angles + [error] + [filename] + list(bbox) + [flipped])


def find_matches(input_folder, references, pkl_name="analysis.pkl", correct_angles=False, remove_outliers=True, threshold_std=4.0, ignore_roll=True, allow_flip=True, random_choice=False, avg_dist_threshold=None):

    start_time = time.time()
    all_errors = []
    discarded_samples = 0
    counter = 0

    for root, _, files in tqdm(os.walk(input_folder), desc="Find Matches"):
        if "analysis" in root:
            for pkl_file in files:
                if pkl_file.endswith(pkl_name):
                    counter += 1
                    file_path = os.path.join(root, pkl_name)
                    output_file = os.path.join(root, "matched_angles.txt")

                    data = parse_analysis_file(file_path, correct_angles)

                    #if remove_outliers:
                    #    data, removed_count = remove_embedding_outliers_lof(data, n_neighbors=20, contamination=0.05)
                    #    total_removed += removed_count
                    #    #print(f"Removed {removed_count} frames ({removed_percentage:.2f}%) from {file_path}")

                    if len(data) == 0:
                        continue

                    infos = match_hpe_angles_to_references(data, references, ignore_roll=ignore_roll, allow_flip=allow_flip, random_choice=random_choice)

                    # extract euclidean errors
                    sample_errors = [row[2] for row in infos]
                    avg_sample_error = np.mean(sample_errors)

                    # discard entire sample if above threshold
                    if avg_dist_threshold is not None and avg_sample_error > avg_dist_threshold:
                        discarded_samples += 1
                        if os.path.isfile(output_file):
                            os.remove(output_file)
                        continue

                    # keep sample
                    counter += 1
                    all_errors.extend(sample_errors)
                    save_matches(output_file, infos)

    elapsed_time = time.time() - start_time
    if not all_errors:
        raise RuntimeError("No analysis txts found")

    print(f"Processed {counter+discarded_samples} files in {round(elapsed_time, 2)}s")
    if avg_dist_threshold:
        overall_percentage = (discarded_samples / (counter + discarded_samples)) * 100
        print(f"Total discarded samples: {discarded_samples} from {counter} ({overall_percentage:.2f}%) with threshold above {avg_dist_threshold}")
    avg_error = np.mean(all_errors)
    print(f"Average angle error: {avg_error:.4f} in {counter} valid files")

