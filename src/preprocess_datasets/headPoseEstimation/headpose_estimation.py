import os
import cv2
import numpy as np
import torch
from torchvision import transforms

from src.preprocess_datasets.headPoseEstimation.models.resnet import resnet50
from src.preprocess_datasets.headPoseEstimation.utils.general import compute_euler_angles_from_rotation_matrices


def pre_process(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    return image


def get_model(arch, num_classes=6, pretrained=True):
    """Return the model based on the specified architecture."""
    if arch == 'resnet50':
        model = resnet50(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Please choose available model architecture, currently chosen: {arch}")
    return model


def process_hpe_batch(cropped_faces, device, head_pose_model):
    """Process a batch of frames."""

    processed_faces = [pre_process(face) for face in cropped_faces]
    batched_images = torch.stack(processed_faces).to(device)
    rotation_matrices = head_pose_model(batched_images)

    eulers = compute_euler_angles_from_rotation_matrices(rotation_matrices).detach().cpu().numpy()
    eulers_deg = np.degrees(eulers)

    return [[(deg[1]), (deg[0]), (deg[2])] for deg in eulers_deg]


def get_images_from_dir(image_dir, files_names):
    images = []
    for image_file in files_names:
        image_path = os.path.join(image_dir, image_file)
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)

    return images
