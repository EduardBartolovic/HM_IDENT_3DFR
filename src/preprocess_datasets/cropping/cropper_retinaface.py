from functools import partial

from multiprocessing.pool import ThreadPool

import torch
from tqdm import tqdm
from enum import Enum
import os
import cv2
import numpy as np
import onnxruntime as ort
import sys

def face_crop_and_alignment(input_folder, output_folder, face_factor=0.7, device='cuda' if torch.cuda.is_available() else 'cpu', resize_size=(256, 256), output_size=(256, 256), det_threshold=0.45):

    cropper = FaceAligner(model_path, alignment_method=FaceAligner.AlignmentMethod.AFFINE_TRANSFORM)
    #cropper = FaceAligner(model_path, alignment_method=FaceAligner.AlignmentMethod.BOUNDING_BOX_SCALING)

    class_names = os.listdir(input_folder)
    total_faces = 0
    for class_name in tqdm(class_names, desc="Cropping & Aligning Faces - Processing Classes"):
        class_path = os.path.join(input_folder, class_name)
        target_class_path = os.path.join(output_folder, class_name)
        os.makedirs(target_class_path, exist_ok=True)

        # Process all images in the class folder and save results to target_class_path
        # TODO
        cropper.process_dir(input_dir=class_path, output_dir=target_class_path, desc=None)

        # Sanity check
        input_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        output_images = [f for f in os.listdir(target_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_faces += len(input_images)
        if len(input_images) == 0 or len(input_images) != len(output_images):
            print(f"⚠️ Warning: Mismatch in image count for class '{class_name}': Input={len(input_images)} vs Output={len(output_images)}")

    print(f"Done! Total_faces: {total_faces}")



CONF_THRESHOLD = 0.8

STANDARD_LANDMARKS_5 = np.float32([
    [0.31556875000000000, 0.4615741071428571],
    [0.68262291666666670, 0.4615741071428571],
    [0.50026249999999990, 0.6405053571428571],
    [0.34947187500000004, 0.8246919642857142],
    [0.65343645833333330, 0.8246919642857142],
])


# ========================
# Helper functions (standalone)
# ========================

def resize_and_pad(image: np.ndarray, max_side: int = 512):
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    top = (max_side - new_h) // 2
    left = (max_side - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized
    return padded, scale, top, left


def preprocess_image_for_retinaface(img: np.ndarray):
    cv2.imwrite("retina_input.png", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img.astype(np.float32)
    img_float -= (104, 117, 123)
    img_transposed = np.transpose(img_float, (2, 0, 1))
    return np.expand_dims(img_transposed, axis=0)


def generate_priors(img_height, img_width, min_sizes=[[16, 32], [64, 128], [256, 512]], steps=[8, 16, 32]):
    priors = []
    for i, step in enumerate(steps):
        f_h = int(np.ceil(img_height / step))
        f_w = int(np.ceil(img_width / step))
        for y in range(f_h):
            for x in range(f_w):
                for size in min_sizes[i]:
                    s_kx = size / img_width
                    s_ky = size / img_height
                    cx = (x + 0.5) * step / img_width
                    cy = (y + 0.5) * step / img_height
                    priors.append([cx, cy, s_kx, s_ky])
    return np.array(priors, dtype=np.float32)


def decode_bboxes(pred, priors, variances=[0.1, 0.2]):
    boxes = np.concatenate((
        priors[:, :2] + pred[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(pred[:, 2:] * variances[1])
    ), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landmarks(pred, priors, variances=[0.1, 0.2]):
    landmarks = []
    for i in range(5):
        landmarks.append(
            priors[:, :2] + pred[:, i * 2:(i + 1) * 2] * variances[0] * priors[:, 2:]
        )
    return np.concatenate(landmarks, axis=1)


class FaceAligner:
    class AlignmentMethod(Enum):
        AFFINE_TRANSFORM = 1
        BOUNDING_BOX_SCALING = 2

    def __init__(self, model_path: str, target_size=(256, 256), max_side=512, conf_threshold=0.5,
                 inference_type=InferenceType.WINDOWS, face_factor=0.7,
                 alignment_method=AlignmentMethod.AFFINE_TRANSFORM):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if inference_type == InferenceType.WINDOWS:
            self.onnx_inference = ONNX_Inference_Windows(model_path)
        elif inference_type == InferenceType.Android:
            self.onnx_inference = ONNX_Inference_Android()

        self.target_size = target_size
        self.max_side = max_side
        self.conf_threshold = conf_threshold
        self.face_factor = face_factor
        self.alignment_method = alignment_method
        self._init_landmarks_target()
        print(f"[FaceAligner] Model loaded from {model_path}")

    def process_dir(
            self,
            input_dir: str,
            output_dir: str | None = None,
            desc: str | None = "Processing",
    ):
        """Processes images in the specified input directory.

        Splits all the file names in the input directory to batches
        and processes batches on multiple cores. For every file name
        batch, images are loaded, some are optionally enhanced,
        landmarks are generated and used to optionally align and
        center-crop faces, and grouping is optionally applied based on
        face attributes. For more details, check
        :meth:`process_batch`.

        Note:
            There might be a few seconds delay before the actual
            processing starts if there are a lot of files in the
            directory - it takes some time to split all the file names
            to batches.

        Args:
            input_dir: Path to input directory with image files.
            output_dir: Path to output directory to save the extracted
                (and optionally grouped to sub-directories) face images.
                If None, then the same path as for ``input_dir`` is used
                and additionally "_faces" suffix is added to the name.
            desc: The description to use for the progress bar. If
                specified as ``None``, no progress bar is shown.
                Defaults to "Processing".
        """
        if output_dir is None:
            # Create a default output dir name
            output_dir = input_dir + "_faces"

        for i in os.listdir(input_dir):
            cropped_image = bb_aligner.align_face(image_path, "BoundingBoxAlignment.png")


        # Create batches of image file names in input dir
        #files, bs = os.listdir(input_dir), self.batch_size
        #file_batches = [files[i:i + bs] for i in range(0, len(files), bs)]

        #if len(file_batches) == 0:
        #    # Empty
        #    return

        # Define worker function and its additional arguments
        #kwargs = {"input_dir": input_dir, "output_dir": output_dir}
        #worker = partial(self.process_batch, **kwargs)
        #with ThreadPool(self.num_processes, self._init_models) as pool:
        #    # Create imap object and apply workers to it
        #    imap = pool.imap_unordered(worker, file_batches)
        #    if desc is not None:
        #        # If description is provided, wrap progress bar around
        #        imap = tqdm.tqdm(imap, total=len(file_batches), desc=desc)
        #    # Process
        #    list(imap)

    def _init_landmarks_target(self):
        std_landmarks = STANDARD_LANDMARKS_5.copy()

        # Apply appropriate scaling based on face factor and out size
        std_landmarks[:, 0] *= self.target_size[0] * self.face_factor
        std_landmarks[:, 1] *= self.target_size[1] * self.face_factor

        # Add an offset to standard landmarks to center the cropped face
        std_landmarks[:, 0] += (1 - self.face_factor) * self.target_size[0] / 2
        std_landmarks[:, 1] += (1 - self.face_factor) * self.target_size[1] / 2

        # Pass STD landmarks as target landms
        self.landmarks_target = std_landmarks

    def detect_face(self, img: np.ndarray):

        padded, scale, top, left = resize_and_pad(img, self.max_side)

        input_blob = preprocess_image_for_retinaface(padded)

        result = self.onnx_inference.perform_inference(input_blob)
        loc, conf, landms = result

        scores = conf[0][:, 1]
        h, w = padded.shape[:2]
        priors = generate_priors(h, w)
        bboxes = decode_bboxes(loc[0], priors)
        landmarks = decode_landmarks(landms[0], priors)

        bboxes[:, [0, 2]] *= w
        bboxes[:, [1, 3]] *= h
        landmarks[:, 0::2] *= w
        landmarks[:, 1::2] *= h

        inds = np.where(scores > self.conf_threshold)[0]
        if len(inds) == 0:
            return None

        best_idx = inds[np.argmax(scores[inds])]
        score = scores[best_idx]
        bbox = bboxes[best_idx]
        ldmk = landmarks[best_idx].reshape(-1, 2)

        bbox[0::2] -= left
        bbox[1::2] -= top
        ldmk[:, 0] -= left
        ldmk[:, 1] -= top
        bbox /= scale
        ldmk /= scale

        return bbox, ldmk

    def align_face_from_landmarks(self, img: np.ndarray, landmarks: np.ndarray):
        assert landmarks.shape == (5, 2)
        M, _ = cv2.estimateAffinePartial2D(landmarks, self.landmarks_target, method=cv2.LMEDS)
        if M is None:
            print("[FaceAligner] Could not estimate affine transform.")
            return None
        return cv2.warpAffine(img, M, self.target_size, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _expand_bounding_box(self, img_shape: tuple, bounding_box: np.ndarray) -> np.ndarray:
        """
        Expands the bounding box by a given ratio, clipped to the image boundaries.

        Args:
            img_shape: The (Height, Width) of the original image.
            bounding_box: A NumPy array [x_min, y_min, x_max, y_max].

        Returns:
            The expanded and clipped bounding box [x_min, y_min, x_max, y_max] as a NumPy array.
        """
        H, W = img_shape[:2]
        x_min, y_min, x_max, y_max = bounding_box.astype(int)

        # Calculate current dimensions
        w = x_max - x_min
        h = y_max - y_min

        # Calculate expansion amount
        # We expand by the smaller of the two dimensions to keep the expansion proportional
        expansion_amount = int(max(w, h) * (1 - self.face_factor))

        # Calculate new coordinates
        new_x_min = x_min - expansion_amount
        new_y_min = y_min - expansion_amount
        new_x_max = x_max + expansion_amount
        new_y_max = y_max + expansion_amount

        # Clip coordinates to image boundaries (0 and W/H)
        x_min_clipped = max(0, new_x_min)
        y_min_clipped = max(0, new_y_min)
        x_max_clipped = min(W, new_x_max)
        y_max_clipped = min(H, new_y_max)

        # Ensure the cropped area is still valid
        if x_min_clipped >= x_max_clipped or y_min_clipped >= y_max_clipped:
            # If the expansion caused an issue (very small original box), return original
            return bounding_box

        return np.array([x_min_clipped, y_min_clipped, x_max_clipped, y_max_clipped])

    def align_face_from_bounding_box(self, img: np.ndarray, bounding_box: np.ndarray) -> np.ndarray:
        target_width, target_height = self.target_size
        H_img, W_img = img.shape[:2]

        # 1. Expand Bounding Box
        x_min, y_min, x_max, y_max = bounding_box.astype(int)
        w, h = x_max - x_min, y_max - y_min

        expansion_amount = int(max(w, h) * (1 - self.face_factor))

        # Initial expanded box (might be outside image limits)
        x_min_exp = x_min - expansion_amount
        y_min_exp = y_min - expansion_amount
        x_max_exp = x_max + expansion_amount
        y_max_exp = y_max + expansion_amount

        # 2. Adjust Expanded Box to Match Target Aspect Ratio
        target_ar = target_width / target_height

        # Current expanded width and height
        current_w = x_max_exp - x_min_exp
        current_h = y_max_exp - y_min_exp

        # Calculate required width/height to match target_ar
        required_w = int(current_h * target_ar)
        required_h = int(current_w / target_ar)

        # Decide which dimension to expand further to match the aspect ratio
        if required_w > current_w:
            # Expand width to match aspect ratio
            w_diff = required_w - current_w
            x_min_final = x_min_exp - w_diff // 2
            x_max_final = x_max_exp + (w_diff - w_diff // 2)
            y_min_final = y_min_exp
            y_max_final = y_max_exp
        elif required_h > current_h:
            # Expand height to match aspect ratio
            h_diff = required_h - current_h
            y_min_final = y_min_exp - h_diff // 2
            y_max_final = y_max_exp + (h_diff - h_diff // 2)
            x_min_final = x_min_exp
            x_max_final = x_max_exp
        else:
            # Aspect ratio is already close enough or target AR is smaller. Use expanded box.
            x_min_final, y_min_final, x_max_final, y_max_final = x_min_exp, y_min_exp, x_max_exp, y_max_exp

        # 3. Calculate Padding and Clip Coordinates

        # Determine how much the box extends outside the image (the required padding)
        pad_left = max(0, -x_min_final)
        pad_top = max(0, -y_min_final)
        pad_right = max(0, x_max_final - W_img)
        pad_bottom = max(0, y_max_final - H_img)

        # Clip the final coordinates to the image boundaries
        x_min_clip = max(0, x_min_final)
        y_min_clip = max(0, y_min_final)
        x_max_clip = min(W_img, x_max_final)
        y_max_clip = min(H_img, y_max_final)

        # 4. Crop, Resize, and Apply Padding

        # Handle case where the box is invalid after clipping
        if x_min_clip >= x_max_clip or y_min_clip >= y_max_clip:
            # If clipping resulted in an empty box, return a blank target image
            return np.full((target_height, target_width, img.shape[2]), [0, 0, 0], dtype=img.dtype)

        # Crop the valid region
        cropped_img = img[y_min_clip:y_max_clip, x_min_clip:x_max_clip]

        # Pad the cropped image so its dimensions are what the original (unclipped) box would have been
        padded_cropped_img = cv2.copyMakeBorder(
            cropped_img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # Final step: Resize the padded crop to the target size.
        # Since the padded crop now has the exact aspect ratio of the target size,
        # this resize is a simple scaling with no letterboxing needed.
        resized_img = cv2.resize(
            padded_cropped_img,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )

        return resized_img

    def align_face(self, image: str, debug_path=None):
        if type(image) == str:
            img = cv2.imread(image)
        else:
            img = image
        if img is None:
            raise ValueError(f"Could not read image: {image}")

        result = self.detect_face(img)
        if result is None:
            print(f"[FaceAligner] No face detected")
            return None

        bbox, landmarks = result
        if self.alignment_method == FaceAligner.AlignmentMethod.AFFINE_TRANSFORM:
            aligned = self.align_face_from_landmarks(img, landmarks)
        elif self.alignment_method == FaceAligner.AlignmentMethod.BOUNDING_BOX_SCALING:
            aligned = self.align_face_from_bounding_box(img, bbox)
        else:
            print("Unknown AligmentMethod!")
        if aligned is not None and debug_path is not None:
            cv2.imwrite(debug_path, aligned)
            # print(f"[FaceAligner] Alignment successful → {debug_path}")
        return aligned


if __name__ == "__main__":
    model_path = "app/data/onnx_model_weights/retina_models/mobile0.25.onnx"
    image_path = "C:/Users/P00200439/Downloads/bellus_single_image/christoph.schultheiss@maurer-electronics.de/45d8dd1a313281a0f283c3476b5fc34f1058bd6b0_0_image.jpg"
    affine_aligner = FaceAligner(model_path, alignment_method=FaceAligner.AlignmentMethod.AFFINE_TRANSFORM)
    face1 = affine_aligner.align_face(image_path, "AffineAlignment.png")
    bb_aligner = FaceAligner(model_path, alignment_method=FaceAligner.AlignmentMethod.BOUNDING_BOX_SCALING)
    face2 = bb_aligner.align_face(image_path, "BoundingBoxAlignment.png")
    # image_path = r"C:\Development\BioID\data\original_images\b.jpg"
    # face2 = aligner.align_face(image_path)
