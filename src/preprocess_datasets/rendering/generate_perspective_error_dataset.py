import os
import random
import shutil
from collections import defaultdict
from tqdm import tqdm


def swap_reference_angles(
    dataset_path: str,
    output_path: str,
    swap_ratio: float = 0.2,
):
    """
    Swaps reference angles within each (class_name, group_id) group,
    while keeping the true angles fixed.

    Args:
        dataset_path (str): Root folder containing class subdirectories.
        output_path (str): Output directory to save swapped images.
        swap_ratio (float): Fraction of items in each group to swap references.
    """

    os.makedirs(output_path, exist_ok=True)
    groups = defaultdict(list)

    # --- Step 1: Group images by (class_name, group_id) ---
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        for fname in os.listdir(class_path):
            if "#" not in fname:
                continue

            try:
                group_id, ref_angle, true_angle = fname.split("#")
                true_angle = os.path.splitext(true_angle)[0]
            except ValueError:
                continue

            groups[(class_name, group_id)].append((ref_angle, true_angle, fname))

    # --- Step 2: Swap REFERENCE angles within each group ---
    for (class_name, group_id), items in tqdm(groups.items(), desc="Swapping reference angles"):
        class_in = os.path.join(dataset_path, class_name)
        class_out = os.path.join(output_path, class_name)
        os.makedirs(class_out, exist_ok=True)

        ref_angles = [r for r, t, f in items]
        n_items = len(items)
        n_swaps = int(n_items * swap_ratio)

        if n_swaps < 2:
            # Too small to swap
            for (_, _, fname) in items:
                shutil.copy(os.path.join(class_in, fname),
                            os.path.join(class_out, fname))
            continue

        # Pick subset of indices to swap reference angles
        swap_indices = list(range(n_items)) if swap_ratio >= 1.0 else random.sample(range(n_items), n_swaps)
        subset_ref_angles = [ref_angles[i] for i in swap_indices]

        # Shuffle the reference angles (ensuring actual change)
        shuffled_refs = subset_ref_angles[:]
        while shuffled_refs == subset_ref_angles and n_swaps > 1:
            random.shuffle(shuffled_refs)

        # Apply swapped reference angles to subset
        for idx, new_ref in zip(swap_indices, shuffled_refs):
            ref_angles[idx] = new_ref

        # --- Write the new files ---
        for (old_ref, true, fname), new_ref in zip(items, ref_angles):
            new_fname = f"{group_id}#{new_ref}#{true}.jpg"
            shutil.copy(os.path.join(class_in, fname), os.path.join(class_out, new_fname))

    print("✅ Done! Swapped dataset saved to:", output_path)


def main():
    DATASET_PATH = r"C:\Users\Eduard\Desktop\Face\dataset11\test_rgb_bff_crop8\enrolled"
    OUTPUT_PATH = r"C:\Users\Eduard\Desktop\Face\dataset11\test_rgb_bff_crop8_swap_ref10\enrolled"
    SWAP_RATIO = 1.0
    swap_reference_angles(DATASET_PATH, OUTPUT_PATH, SWAP_RATIO)
    DATASET_PATH = r"C:\Users\Eduard\Desktop\Face\dataset11\test_rgb_bff_crop8\query"
    OUTPUT_PATH = r"C:\Users\Eduard\Desktop\Face\dataset11\test_rgb_bff_crop8_swap_ref10\query"
    swap_reference_angles(DATASET_PATH, OUTPUT_PATH, SWAP_RATIO)


if __name__ == "__main__":
    main()
