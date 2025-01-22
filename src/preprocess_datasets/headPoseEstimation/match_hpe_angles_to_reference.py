import csv
import itertools
import os

import numpy as np


def match_hope_angles_to_references(data, references, ignore_roll=False):
    closest_rows = []
    for reference in references:
        if ignore_roll:
            distances = np.linalg.norm(np.array(data[:, :2], dtype=int) - reference[:2], axis=1)
        else:
            distances = np.linalg.norm(np.array(data[:, :3], dtype=int) - reference[:3], axis=1)
        closest_index = np.argmin(distances)
        closest_rows.append((reference, data[closest_index], int(distances[closest_index])))
        assert closest_index < len(data)

    # print("####")
    # for ref, closest_row, distance, index in closest_rows:
    #     print(f"Closest row to {ref}: {closest_row} (index {index}), distance_error: {int(distance)}")

    return closest_rows


def find_matches(input_folder, references):
    counter = 0
    for root, _, files in os.walk(input_folder):
        if "hpe" in root:
            for txt_file in files:
                if txt_file.endswith('hpe.txt'):
                    counter += 1
                    file_path = os.path.join(root, txt_file)
                    with open(file_path) as file:
                        lines = []
                        for line in file:
                            parts = line.strip().split(',')
                            integers = list(map(int, parts[:-1]))  # Convert all but the last part to integers
                            last_part = parts[-1]  # Keep the last part as a string
                            lines.append(integers + [last_part])  # Combine integers with the string part
                    data = np.array(lines, dtype=object)
                    infos = match_hope_angles_to_references(data, references)

                    assert len(infos) == len(references)

                    output_file = os.path.join(root, "matched_angles.txt")
                    with open(output_file, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["Ref_X", "Ref_Y", "Ref_Z", "Hpe_X", "Hpe_Y", "Hpe_Z", "file_name", "Error"])
                        for row in infos:
                            array1 = row[0].tolist()  # Convert first array to a list
                            array2 = row[1].tolist()  # Convert second array to a list
                            other_values = row[2:]  # Remaining values
                            writer.writerow(array1 + array2 + list(other_values))
    print("Found matches for", counter, "files")

if __name__ == '__main__':

    input_folder = "E:\\Download\\face\\VoxCeleb1_test"  # Folder containing original preprocessed files
    ref_angles = [-25, -10, 0, 10, 25]
    permutations = np.array([(x, y, 0) for x, y in itertools.product(ref_angles, repeat=2)])
    print(len(permutations))
    print(permutations)
    find_matches(input_folder, permutations)
