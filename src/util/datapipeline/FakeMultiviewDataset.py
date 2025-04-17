import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageOps


class FakeMultiviewDataset(Dataset):
    def __init__(self, root_dir, num_views, augmentation_type='rotation', transform=None, use_face_corr=True):
        """
        Args:
            root_dir (string): Path to the root directory of the dataset.
            num_views (int): Number of views (augmentations of front view) to return.
            augmentation_type (string): Type of augmentation ('rotation', 'shift', or 'brightness').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_views = num_views
        self.face_cor_exist = False
        self.use_face_corr = use_face_corr
        self.class_to_idx = self._get_class_to_idx()
        self.data = self._load_data()
        self.classes = self._find_classes()

        # Define consistent augmentation factors
        self.augmentation_type = augmentation_type
        self.augmentation_factors = self._generate_factors(0.7, 1.3, self.num_views)

    def _get_class_to_idx(self):
        classes = sorted(os.listdir(self.root_dir))
        return {class_name: idx for idx, class_name in enumerate(classes)}

    def _generate_factors(self, min_val, max_val, n):
        if n == 1:
            return [1.0]
        return [min_val + (max_val - min_val) * i / (n - 1) for i in range(n)]

    def _find_classes(self):
        class_names = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        return {class_name: idx for idx, class_name in enumerate(class_names)}

    def _load_data(self):
        data = []
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                class_idx = self.class_to_idx[class_name]
                for filename in os.listdir(class_path):
                    if filename.endswith((".jpg", ".png", ".jpeg", ".webp")) and "0_0" == filename[40:-10]:
                        file_path = os.path.join(class_path, filename)
                        if os.path.isfile(file_path):
                            data.append((file_path, class_idx))

                    elif filename.endswith(".npz"):
                        self.face_cor_exist = True

        if not self.use_face_corr:
            self.face_cor_exist = False

        return data

    def _apply_augmentation(self, img, idx):
        """
        Apply specified augmentations: rotation, shifting, or brightness based on the input parameter.
        """
        factor = self.augmentation_factors[idx]

        if self.augmentation_type == 'rotation':
            # Apply fixed rotation based on the augmentation factor (e.g., factor * 30 degrees)
            angle = factor * 30  # Maximum rotation will be 30 degrees
            img = img.rotate(angle, resample=Image.BICUBIC)

        elif self.augmentation_type == 'shift':
            # Apply fixed shift (translation) based on the augmentation factor
            max_shift = int(0.1 * img.width)  # Shift up to 10% of image width/height
            shift_x = int(factor * max_shift)
            shift_y = int(factor * max_shift)
            img = ImageOps.offset(img, shift_x, shift_y)

        elif self.augmentation_type == 'brightness':
            # Apply deterministic brightness change based on index (as before)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)

        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_idx = self.data[idx]

        # Load the front-view image
        img = Image.open(img_path).convert("RGB")

        # 1. Base image (unaltered)
        images = [self.transform(img) if self.transform else img]

        # 2. Augmented images
        for i in range(self.num_views - 1):
            aug_img = self._apply_augmentation(img, i)
            if self.transform:
                aug_img = self.transform(aug_img)
            images.append(aug_img)

        # 3. Face correspondence
        if self.face_cor_exist:
            face_corr = np.load(img_path.replace("_image.jpg", "_corr.npz"))['corr']
            face_corr_tensor = torch.Tensor(np.stack([face_corr] * self.num_views))
        else:
            face_corr_tensor = torch.Tensor([])

        # 4. Perspective labels
        perspectives = [f"0_{i}" for i in range(self.num_views)]

        return images, class_idx, perspectives, face_corr_tensor