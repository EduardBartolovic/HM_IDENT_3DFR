import os
import hashlib
import random
from pathlib import Path

# Config
dataset_path = "F:\\Face\\data\\datasets8\\photo_MS-Celeb-1M_Align_112x112_5K8\\"
appendices = ['0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0']
batch_size = len(appendices)


def generate_hash():
    return hashlib.sha1(str(random.random()).encode()).hexdigest()


def rename_images_in_class_folder(class_path):
    images = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    total_images = len(images)
    num_batches = total_images // batch_size
    num_to_keep = num_batches * batch_size

    for i in range(num_batches):
        batch = images[i * batch_size:(i + 1) * batch_size]
        group_hash = generate_hash()

        for img_name, appendix in zip(batch, appendices):
            ext = Path(img_name).suffix
            new_name = f"{group_hash}{appendix}{ext}"
            src = os.path.join(class_path, img_name)
            dst = os.path.join(class_path, new_name)
            os.rename(src, dst)

    # Delete leftover images
    leftovers = images[num_to_keep:]
    for leftover in leftovers:
        os.remove(os.path.join(class_path, leftover))

    print(f"Processed {num_batches * batch_size} images in '{class_path}'. Deleted {len(leftovers)} leftovers.")


def main():
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            rename_images_in_class_folder(class_path)


if __name__ == "__main__":
    main()

