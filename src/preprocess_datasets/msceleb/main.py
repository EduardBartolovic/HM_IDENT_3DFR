import shutil

import os
import hashlib
import random
from pathlib import Path
from tqdm import tqdm

# Config
dataset_path = "F:\\Face\\data\\datasets9\\MV_MSCELEB85KO8\\"
appendices = ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0']
batch_size = len(appendices)


def generate_hash():
    return hashlib.sha1(str(random.random()).encode()).hexdigest()


def get_image_files(path):
    return sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


def rename_images_in_class_folder(class_path):
    images = get_image_files(class_path)
    total_images = len(images)
    num_batches = total_images // batch_size
    num_to_process = num_batches * batch_size
    leftovers = images[num_to_process:]

    processed_images = images[:num_to_process]

    # Step 1: Rename full batches
    for i in range(num_batches):
        batch = processed_images[i * batch_size:(i + 1) * batch_size]
        group_hash = generate_hash()
        for img_name, appendix in zip(batch, appendices):
            ext = Path(img_name).suffix
            new_name = f"{group_hash}{appendix}_image{ext}"
            src = os.path.join(class_path, img_name)
            dst = os.path.join(class_path, new_name)
            os.rename(src, dst)

    # Step 2: Process leftovers
    if leftovers:
        # Refresh file list (only renamed images are available now)
        all_images_now = get_image_files(class_path)
        needed = batch_size - len(leftovers)
        fill_ins = random.choices(all_images_now, k=needed)
        final_batch = leftovers + fill_ins
        group_hash = generate_hash()

        for img_name, appendix in zip(final_batch, appendices):
            ext = Path(img_name).suffix
            new_name = f"{group_hash}{appendix}_image{ext}"
            dst = os.path.join(class_path, new_name)

            if img_name in leftovers:
                # This is a real leftover → rename
                src = os.path.join(class_path, img_name)
                if os.path.exists(src):
                    os.rename(src, dst)
            else:
                # This is a fill-in → copy
                src = os.path.join(class_path, img_name)
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        print(f"Leftover batch filled (copied {needed} fill-ins) in '{class_path}'.")

    print(f"Processed total of {(num_batches + (1 if leftovers else 0)) * batch_size} images in '{class_path}'.")


def main():
    for class_name in tqdm(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            rename_images_in_class_folder(class_path)


if __name__ == "__main__":
    main()
