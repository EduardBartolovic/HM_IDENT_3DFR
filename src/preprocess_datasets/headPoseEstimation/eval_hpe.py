import os
import time
from collections import Counter


def classify_gaze_9(x, y):
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


def print_gaze_grid(area_counts, total_triggers, num_areas):
    if num_areas == 10:
        grid = {
            "top_left": "   ", "top_middle": "   ", "top_right": "   ",
            "middle_left": "   ", "middle": "   ", "middle_right": "   ",
            "below_left": "   ", "below_middle": "   ", "below_right": "   "
        }

        for area, count in area_counts.items():
            percentage = (count / total_triggers * 100) if total_triggers > 0 else 0
            grid[area] = f"{percentage:5.1f}%"

        print("\nGaze Heatmap (Percentage)")
        print("====================")
        print(f"| {grid['top_left']} | {grid['top_middle']} | {grid['top_right']} |")
        print(f"| {grid['middle_left']} | {grid['middle']} | {grid['middle_right']} |")
        print(f"| {grid['below_left']} | {grid['below_middle']} | {grid['below_right']} |")
        print("====================\n")
    else:
        print("\nGaze Heatmap (Percentage - 5x5 Grid)")
        print("====================================")
        for y in [25, 10, 0, -10, -25]:
            row = "|"
            for x in [-25, -10, 0, 10, 25]:
                key = f"x{x}_y{y}"
                percentage = (
                            area_counts[key] / total_triggers * 100) if key in area_counts and total_triggers > 0 else 0
                row += f" {percentage:5.1f}% |"
            print(row)
        print("====================================\n")

def print_gaze_coverage(coverage_counts, counter, title):
    print(f"\n{title}")
    print("=" * len(title))
    max_count = max(coverage_counts.values(), default=1)
    for areas, count in sorted(coverage_counts.items(), reverse=True):
        percentage = (count / counter * 100) if counter > 0 else 0
        bar_length = count * 40 // max_count if max_count > 0 else 0
        print(f"Areas Covered: {areas:2} | Videos: {count:5} | {percentage:6.2f}% | {'█' * bar_length}")
    print("\n")


def print_most_triggered_areas(area_counts, title, total_triggers):
    print(f"\nMost Triggered Gaze Areas ({title})")
    print("=" * (len(title) + 30))
    for area in sorted(area_counts.keys()):  # Keep areas in natural order
        count = area_counts[area]
        percentage = (count / total_triggers * 100) if total_triggers > 0 else 0
        bar_length = int((percentage / 100) * 40)
        print(f"Area: {area:12} | Triggered: {count:7} | {percentage:6.2f}% | {'█' * bar_length}")


def evaluate_gaze_coverage(input_folder, txt_name="hpe.txt"):
    for num_areas, classify_func in [(10, classify_gaze_9), (25, classify_gaze_25)]:
        print(f"#################{num_areas - 1}########################")

        start_time = time.time()
        counter = 0
        coverage_counts = {i: 0 for i in range(num_areas)}
        area_counts = Counter()
        total_triggers = 0

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
                                integers = list(map(int, parts[:-1]))  # Convert all but last part to int
                                x, y, _ = integers[:3]  # Assuming head pose angles
                                area = classify_func(x, y)
                                covered_areas.add(area)
                                area_counts[area] += 1
                                total_triggers += 1

                            coverage_counts[len(covered_areas)] += 1

        elapsed_time = time.time() - start_time
        print("Evaluated", counter, "files in", round(elapsed_time, 2), "seconds")

        print_gaze_coverage(coverage_counts, counter, f"Gaze Coverage Distribution ({num_areas} Areas)")
        print_most_triggered_areas(area_counts, f"{num_areas} Areas", total_triggers)
        print_gaze_grid(area_counts, total_triggers, num_areas)


if __name__ == '__main__':
    input_folder = "C:\\Users\\Eduard\\Downloads\\vox2_test_mp4\\mp4"
    evaluate_gaze_coverage(input_folder)
