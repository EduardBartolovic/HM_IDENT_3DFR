import os
import torch
from tqdm import tqdm
from face_crop_plus import Cropper


def face_crop_and_alignment(input_folder, output_folder, face_factor=0.75, device='cuda' if torch.cuda.is_available() else 'cpu'):
    cropper = Cropper(
        resize_size=(256, 256),
        output_size=(256, 256),
        output_format="jpg",
        face_factor=face_factor,
        strategy="largest",
        padding="Constant",
        det_threshold=0.45,
        device=device,
        attr_groups=None,
        mask_groups=None,
    )

    class_names = os.listdir(input_folder)

    total_faces = 0
    for class_name in tqdm(class_names, desc="Cropping & Aligning Faces - Processing Classes"):
        class_path = os.path.join(input_folder, class_name)
        target_class_path = os.path.join(output_folder, class_name)
        os.makedirs(target_class_path, exist_ok=True)

        # Process all images in the class folder and save results to target_class_path
        cropper.process_dir(input_dir=class_path, output_dir=target_class_path, desc=None)

        # Sanity check
        input_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        output_images = [f for f in os.listdir(target_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_faces += len(input_images)
        if len(input_images) == 0 or len(input_images) != len(output_images):
            print(f"⚠️ Warning: Mismatch in image count for class '{class_name}': Input={len(input_images)} vs Output={len(output_images)}")

    print(f"Done! Total_faces: {total_faces}")


def face_crop_and_alignment_deepfolder(input_folder, output_folder, face_factor=0.75, device='cuda' if torch.cuda.is_available() else 'cpu'):
    cropper = Cropper(
        resize_size=(256, 256),
        output_size=(256, 256),
        output_format="jpg",
        face_factor=face_factor,
        strategy="largest",
        padding="Constant",
        det_threshold=0.45,
        device=device,
        attr_groups=None,
        mask_groups=None,
    )

    total_faces = 0

    class_names = os.listdir(input_folder)

    for class_name in tqdm(class_names, desc="Processing Classes"):
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue  # Skip non-folder items

        subfolder_names = os.listdir(class_path)

        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(class_path, subfolder_name)
            if not os.path.isdir(subfolder_path):
                continue  # Skip non-folder items

            target_subfolder_path = os.path.join(output_folder, class_name, subfolder_name)
            os.makedirs(target_subfolder_path, exist_ok=True)

            cropper.process_dir(input_dir=subfolder_path, output_dir=target_subfolder_path, desc=None)

            # Sanity check
            input_images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            output_images = [f for f in os.listdir(target_subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_faces += len(input_images)
            if len(input_images) == 0 or len(input_images) != len(output_images):
                print(f"⚠️ Warning: Mismatch in image count for '{class_name}/{subfolder_name}': Input={len(input_images)} vs Output={len(output_images)}")

    print(f"✅ Done! Total images processed: {total_faces}")