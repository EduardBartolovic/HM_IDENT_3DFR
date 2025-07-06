import shutil

import os
import hashlib
import random
from pathlib import Path
from tqdm import tqdm

# Config
dataset_path = "F:\\Face\\data\\datasets9\\MV_MSCELEBFULL\\photo_MS-Celeb-1M_Align_112x112" #MV_MSCELEB85KO8\\"
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

    renamed_images = []

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
            renamed_images.append(new_name)

    # Step 2: Process leftovers
    if leftovers:
        if not renamed_images:
            print(f"Not enough images to fill batch in '{class_path}'. Deleting folder.")
            # Delete all files
            for file_name in get_image_files(class_path):
                os.remove(os.path.join(class_path, file_name))
            # Delete folder
            os.rmdir(class_path)
            return

        needed = batch_size - len(leftovers)
        fill_ins = random.choices(renamed_images, k=needed)
        final_batch = leftovers + fill_ins
        group_hash = generate_hash()

        for img_name, appendix in zip(final_batch, appendices):
            ext = Path(img_name).suffix
            new_name = f"{group_hash}{appendix}_image{ext}"
            dst = os.path.join(class_path, new_name)

            src = os.path.join(class_path, img_name)

            if img_name in leftovers:
                if os.path.exists(src):
                    os.rename(src, dst)
            else:
                # Fill-in → Copy from renamed_images
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        print(f"Leftover batch filled (only using non-leftovers) in '{class_path}'.")

    print(f"Processed total of {(num_batches + (1 if leftovers else 0)) * batch_size} images in '{class_path}'.")


def main():
    for class_name in tqdm(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            rename_images_in_class_folder(class_path)


if __name__ == "__main__":
    main()
