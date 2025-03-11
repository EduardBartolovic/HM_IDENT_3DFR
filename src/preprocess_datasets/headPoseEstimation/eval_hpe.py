import os
import time


def classify_gaze(x, y):
    if y > 10:  # Looking up
        if x < -10:
            return 'top_left'
        elif x > 10:
            return 'top_right'
        else:
            return 'top_middle'
    elif y < -10:  # Looking down
        if x < -10:
            return 'below_left'
        elif x > 10:
            return 'below_right'
        else:
            return 'below_middle'
    else:  # Centered vertically
        if x < -10:
            return 'middle_left'
        elif x > 10:
            return 'middle_right'
        else:
            return 'middle'


def classify_gaze_25(x, y):
    angles = [-25, -10, 0, 10, 25]
    x_index = min(range(len(angles)), key=lambda i: abs(angles[i] - x))
    y_index = min(range(len(angles)), key=lambda i: abs(angles[i] - y))
    return f"x{angles[x_index]}_y{angles[y_index]}"


def evaluate_gaze_coverage(input_folder, txt_name="hpe.txt"):

    print("#################9########################")

    start_time = time.time()
    counter = 0
    coverage_counts = {i: 0 for i in range(10)}  # Tracks counts for 0 to 9 areas covered
    for root, _, files in os.walk(input_folder):
        if "hpe" in root:
            for txt_file in files:
                if txt_file.endswith(txt_name):
                    counter += 1
                    file_path = os.path.join(root, txt_file)
                    with open(file_path) as file:
                        covered_areas = set()
                        for line in file:
                            parts = line.strip().split(',')
                            integers = list(map(int, parts[:-1]))  # Convert all but the last part to integers
                            x, y, z = integers[:3]  # Assuming these are head pose angles
                            area = classify_gaze(x, y)
                            covered_areas.add(area)

                        coverage_counts[len(covered_areas)] += 1

    elapsed_time = time.time() - start_time
    print("Evaluated", counter, "files in", round(elapsed_time, 2), "seconds")
    for areas, count in sorted(coverage_counts.items(), reverse=True):
        percentage = (count / counter * 100) if counter > 0 else 0
        print(f"{count} videos cover {areas} gaze areas ({percentage:.2f}%)")

    print("#################25########################")

    start_time = time.time()
    counter = 0
    coverage_counts = {i: 0 for i in range(26)}  # Tracks counts for 0 to 25 areas covered
    for root, _, files in os.walk(input_folder):
        if "hpe" in root:
            for txt_file in files:
                if txt_file.endswith(txt_name):
                    counter += 1
                    file_path = os.path.join(root, txt_file)
                    with open(file_path) as file:
                        covered_areas = set()
                        for line in file:
                            parts = line.strip().split(',')
                            integers = list(map(int, parts[:-1]))  # Convert all but the last part to integers
                            x, y, z = integers[:3]  # Assuming these are head pose angles
                            area = classify_gaze_25(x, y)
                            covered_areas.add(area)

                        coverage_counts[len(covered_areas)] += 1

    elapsed_time = time.time() - start_time
    print("Evaluated", counter, "files in", round(elapsed_time, 2), "seconds")
    for areas, count in sorted(coverage_counts.items(), reverse=True):
        percentage = (count / counter * 100) if counter > 0 else 0
        print(f"{count} videos cover {areas} gaze areas ({percentage:.2f}%)")


if __name__ == '__main__':
    input_folder = "E:\\Download\\vox2_mp4_6\\dev\\mp42"  # Folder containing original preprocessed files
    evaluate_gaze_coverage(input_folder)
