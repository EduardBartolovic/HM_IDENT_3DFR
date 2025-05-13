import os

import shutil

import random

import itertools

from tqdm import tqdm


def generate_allowed_perspectives(ref_angles):
    base_set = list(itertools.product(ref_angles, repeat=2))
    base_set = [tuple(x) for x in base_set]
    base_set = [f"{t[0]}_{t[1]}" for t in base_set]
    # Ensure (0, 0) is first
    base_set.remove("0_0")
    base_set = ["0_0"] + base_set

    # Generate nested subsets
    nested_subsets = []

    for k in tqdm(range(1, len(base_set) + 1)):
        # Fix "0_0" as the first element
        other_elements = base_set[1:]
        subsets_k = []

        for perm in itertools.combinations(other_elements, k - 1):
            subset = ["0_0"] + list(perm)
            subsets_k.append(subset)

        nested_subsets.append(subsets_k)

    # Print max 24 combos from each level
    sampled_subsets = []
    for i, level in enumerate(nested_subsets, start=1):
        print(f"\nLevel {i} - Sample {min(30, len(level))} from {len(level)} combinations:")
        to_show = random.sample(level, min(30, len(level))) if len(level) > 24 else level
        sampled_subsets.append(to_show)
        for combo in to_show:
            print(combo)

    return sampled_subsets


def filter_images(SOURCE_DIR, DEST_DIR, split, allowed_perspectives):
    src_dir = os.path.join(SOURCE_DIR, split)
    dst_dir = os.path.join(DEST_DIR, split)

    for identity in os.listdir(src_dir):
        src_id_path = os.path.join(src_dir, identity)
        dst_id_path = os.path.join(dst_dir, identity)
        os.makedirs(dst_id_path, exist_ok=True)

        allowed_set = set(allowed_perspectives)
        for file in os.listdir(src_id_path):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            perspective = os.path.basename(file)[40:-10]
            if perspective in allowed_set:
                #shutil.copy2(
                #    os.path.join(src_id_path, file),
                #    os.path.join(dst_id_path, file)
                #)
                os.symlink(
                    os.path.join(src_id_path, file),
                    os.path.join(dst_id_path, file)
                )


if __name__ == '__main__':
    # --------------------------- CONFIG --------------------------- #
    SOURCE_DIR = 'F:\\Face\\data\\datasets9\\test_rgb_bff_crop'
    ANGLES = [-25, -10, 0, 10, 25]
    MAX_SAMPLES_PER_LEVEL = 25
    # -------------------------------------------------------------- #
    allowed = generate_allowed_perspectives(ANGLES)
    print(f"✅ Using {len(allowed)} unique perspectives")
    extras = [['0_0', '-25_0', '-10_0', '25_0', '10_0'],  # 1 Azimuth axis
              ['0_0', '0_-25', '0_-10', '0_10', '0_25'],  # 2 Alitude axis
              ['0_0', '-25_-25', '-10_-10', '10_10', '25_25'],  # 3 diagonal
              ['0_0', '-25_25', '-10_10', '10_-10', '25_-25'],  # 4 diagonal
              ['0_0', '-25_25', '-10_10', '10_-10', '25_-25', '-25_-25', '-10_-10', '10_10', '25_25'],  # 5 cross
              ['0_0', '-25_-25', '-25_25', '25_-25', '25_25'],  # 6 Only corners
              ['0_0', '-10_-10', '-10_10', '10_-10', '10_10'],  # 7 Only middle
              ['0_0', '25_-25', '25_-10', '25_0', '25_10', '25_25'],  # 8 Top row
              ['0_0', '10_-25', '10_-10', '10_0', '10_10', '10_25'],  # 9 2nd Top row
              ['0_0', '25_-25', '25_-10', '25_0', '25_10', '25_25', '10_-25', '10_-10', '10_0', '10_10', '10_25'],  # 10 Top All
              ['0_0', '25_-25', '25_-10', '25_0', '25_10', '25_25', '10_-25', '10_-10', '10_0', '10_10', '10_25', '0_-25', '0_-10', '0_10', '0_25'],  # 11 Top All + Middle
              ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25'],  # 12 Top Corners + Sides
              ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0'],  # 13 Top Corners + Sides + Mid  FAV?
              ]
    allowed.append(extras)
    for num_perspectives in allowed:
        print(len(num_perspectives))
        for allowed_perspectives in tqdm(num_perspectives):

            DEST_DIR = f'F:\\Face\\data\\datasets9\\test_rgb_bff_crop_new_{allowed_perspectives}'.replace("[", "").replace("]", "").replace("\'", "").replace(",", "")
            for split in ['train', 'validation']:
                filter_images(SOURCE_DIR, DEST_DIR, split, allowed_perspectives)

    print("🎉 Filtering complete. New dataset at:", DEST_DIR)

    # Deleting later : find . -type l -delete
