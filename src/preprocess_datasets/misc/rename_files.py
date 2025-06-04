import shutil

import os

import re
from tqdm import tqdm


def rename_angles(input_dir, output_dir):
    pattern = re.compile(r"^([a-fA-F0-9]{40})(-?\d+)_(-?\d+)(_image\.jpg)$")

    def fix_coords(x, y):
        x, y = int(x), int(y)
        if x < 0 and y < 0:  # -a, -a -> a, a
            return str(abs(x)), str(abs(y))
        elif x > 0 and y < 0:  # a, -a -> a, -a
            return str(x), str(y)
        elif x > 0 and y > 0:  # a, a -> -a, -a
            return str(-x), str(-y)
        else:  # -a, a -> -a, a
            return str(x), str(y)

    all_files = []
    for foldername, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if pattern.match(filename):
                all_files.append((foldername, filename))

    for foldername, filename in tqdm(all_files, desc="Processing Files", unit="file"):
        match = pattern.match(filename)
        if match:
            hash_str, x, y, appendix = match.groups()
            new_x, new_y = fix_coords(x, y)

            old_path = os.path.join(foldername, filename)
            new_filename = f"{hash_str}{new_x}_{new_y}{appendix}"
            relative_path = os.path.relpath(foldername, input_dir)
            new_folder = os.path.join(output_dir, relative_path)
            os.makedirs(new_folder, exist_ok=True)
            new_path = os.path.join(new_folder, new_filename)

            #print(f"Copying: {old_path} -> {new_path}")
            shutil.copy2(old_path, new_path)


if __name__ == '__main__':
    rename_angles("F:\\Face\\data\\datasets9\\test_nersemble", "F:\\Face\\data\\datasets9\\test_nersemble_fix")