import os

import numpy as np


def match_hpe_angles_to_referecenes(data, references, ignore_roll=False):

    closest_rows = []
    for reference in references:
        if ignore_roll:
            distances = np.linalg.norm(data[:, :2] - reference[:2], axis=1)
        else:
            distances = np.linalg.norm(data-reference, axis=1)
        closest_index = np.argmin(distances)
        closest_rows.append((reference, data[closest_index], int(distances[closest_index]), closest_index))

    print("####")
    for ref, closest_row, distance, index in closest_rows:
        print(f"Closest row to {ref}: {closest_row} (index {index}), distance_error: {int(distance)}")

    return closest_rows


def find_matches(input_folder, references):

    for root, _, files in os.walk(input_folder):
        if "hpe" in root:
            for txt_file in files:
                if txt_file.endswith('hpe.txt'):
                    file_path = os.path.join(root, txt_file)
                    with open(file_path) as file:
                        lines = [list(map(int, line.strip().split(','))) for line in file]
                    data = np.array(lines)
                    infos = match_hpe_angles_to_referecenes(data, references)

                    file_path = os.path.join(root, "matched_angles.txt")
                    with open(file_path, 'w') as file:
                        for i in infos:
                            file.write(','.join(map(str, i)) + '\n')


if __name__ == '__main__':

    input_folder = "E:\\Download\\face\\VoxCeleb1_test"  # Folder containing original preprocessed files
    references = [
        [25, 25, 0],
        [25, 0, 0],
        [25, -25, 0],
        [0, 25, 0],
        [0, 0, 0],
        [0, -25, 0],
        [-25, 25, 0],
        [-25, 0, 0],
        [-25, -25, 0],

    ]
    references = np.array(references)

    find_matches(input_folder, references)