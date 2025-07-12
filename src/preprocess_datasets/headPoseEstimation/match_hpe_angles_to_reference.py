import csv
import itertools
import os
import random
import time

import numpy as np
from tqdm import tqdm


def match_hpe_angles_to_references(data, references, ignore_roll=False):
    closest_rows = []
    for reference in references:
        if ignore_roll:
            distances = np.linalg.norm(np.array(data[:, :2], dtype=float) - reference[:2], axis=1)
        else:
            distances = np.linalg.norm(np.array(data[:, :3], dtype=float) - reference[:3], axis=1)

        min_distance = np.min(distances)
        min_indices = np.where(distances == min_distance)[0]
        closest_index = random.choice(min_indices)

        closest_rows.append((reference, data[closest_index], min_distance))
        assert closest_index < len(data)

    # print("####")
    # for ref, closest_row, distance, index in closest_rows:
    #     print(f"Closest row to {ref}: {closest_row} (index {index}), distance_error: {int(distance)}")

    return closest_rows


def find_matches(input_folder, references, txt_name="analysis.txt"):
    start_time = time.time()
    counter = 0
    all_errors = []
    for root, _, files in tqdm(os.walk(input_folder), desc="Find Matches"):
        if "analysis" in root:
            for txt_file in files:
                if txt_file.endswith(txt_name):
                    counter += 1
                    file_path = os.path.join(root, txt_file)
                    with open(file_path) as file:
                        lines = []
                        for line in file:
                            parts = line.strip().split(',')
                            if len(parts) != 8:
                                raise Exception("Unexpected line format:", line, "in", file_path)
                            try:
                                angles = list(map(float, parts[0:3]))
                                filename_frame = parts[3]  # Keep as string
                                bbox = list(map(int, parts[4:8]))
                            except:
                                raise Exception("Error in Conversion! at:", file_path)

                            lines.append(angles + [filename_frame] + bbox)

                    data = np.array(lines, dtype=object)
                    infos = match_hpe_angles_to_references(data, references)

                    assert len(infos) == len(references)

                    output_file = os.path.join(root, "matched_angles.txt")
                    with open(output_file, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            "Ref_X", "Ref_Y", "Ref_Z",
                            "Hpe_X", "Hpe_Y", "Hpe_Z",
                            "Error",
                            "file_name",
                            "BBox_x_min", "BBox_y_min", "BBox_x_max", "BBox_y_max"
                        ])
                        for row in infos:
                            ref_angles = row[0].tolist()
                            hpe_angles = row[1][:3].tolist()
                            filename = row[1][3]
                            bbox = row[1][4:]
                            error = row[2]
                            all_errors.append(error)
                            writer.writerow(ref_angles + hpe_angles + [error] + [filename] + list(bbox))

    elapsed_time = time.time() - start_time
    if len(all_errors) == 0:
        raise Exception("No analysis txts found")
    avg_error = sum(all_errors) / len(all_errors)
    print("Found matches for", counter, "files in", round(elapsed_time, 2), "seconds, Average angle error:", round(avg_error, 4))


if __name__ == '__main__':

    input_dir = "VoxCeleb2_test"  # Folder containing original preprocessed files
    references_angles = [-25, -10, 0, 10, 25]
    permutations = np.array([(x, y, 0) for x, y in itertools.product(references_angles, repeat=2)])
    print(len(permutations))
    print(permutations)
    find_matches(input_dir, permutations)
