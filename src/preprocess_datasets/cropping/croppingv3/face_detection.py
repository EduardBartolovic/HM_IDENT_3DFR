from enum import Enum
import time
import cv2
import numpy as np
import sys
import os

from typing import Tuple, List

from src.preprocess_datasets.cropping.croppingv3.onnx_inference import InferenceType, ONNX_Inference_Windows

current_directory = os.getcwd()
sys.path.append(current_directory)

# ========================
# CONFIG
# ========================

STANDARD_LANDMARKS_5 = np.float32([
    [0.31556875000000000, 0.4615741071428571],
    [0.68262291666666670, 0.4615741071428571],
    [0.50026249999999990, 0.6405053571428571],
    [0.34947187500000004, 0.8246919642857142],
    [0.65343645833333330, 0.8246919642857142],
])


AURAFACE_LANDMARKS_5= np.array(
            [[38.2946, 51.6963],  #LEye
             [73.5318, 51.5014],         #REye
             [56.0252, 71.7366],         #Nose
             [41.5493, 92.3655],         #LLip
             [70.7299, 92.2041]],        #RLip
            dtype=np.float32)


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
    padded[top:top+new_h, left:left+new_w] = resized
    return padded, scale, top, left


def preprocess_image_for_retinaface(img: np.ndarray):
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
            priors[:, :2] + pred[:, i*2:(i+1)*2] * variances[0] * priors[:, 2:]
        )
    return np.concatenate(landmarks, axis=1)


def get_image_paths(data_dir,  extension= '.jpg'):
    folder_contents  = os.listdir(data_dir)
    image_paths = [f'{os.path.join(data_dir,f)}' for f in folder_contents if f.endswith(extension)]
    if image_paths:
        return image_paths
    else:
        raise FileNotFoundError(f"No JPG files found  in : {data_dir}")


# ========================
# FaceAligner Class
# ========================
class FaceAligner:
    class AlignmentMethod(Enum):
        AFFINE_TRANSFORM = 1
        BOUNDING_BOX_SCALING = 2
        AURA_FACE_ALIGNER = 3
        AURA_FROM_BOUNDING_BOX_SCALING = 4
        AFFINE_FROM_BOUNDING_BOX_SCALING = 5

    def __init__(self, model_path: str, target_size=(112, 112), max_side=512, conf_threshold=0.001,
                 inference_type=InferenceType.WINDOWS,device_type="cpu",
                 face_factor=0.8, bbox_offset =(0,0),
                 alignment_method=AlignmentMethod.AFFINE_TRANSFORM):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        #self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.device_type = device_type
        if inference_type == InferenceType.WINDOWS:
            self.onnx_inference = ONNX_Inference_Windows(model_path,device=self.device_type)

        self.target_size = target_size
        self.max_side = max_side
        self.conf_threshold = conf_threshold
        self.face_factor = face_factor
        self.bbox_offset = bbox_offset
        self.alignment_method = alignment_method
        self._init_landmarks_target()
        print(f"[FaceAligner] Model loaded from {model_path}")

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
        #input_name = self.session.get_inputs()[0].name
        #loc, conf, landms = self.session.run(None, {input_name: input_blob})
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


        bbox[0::2] -= left; bbox[1::2] -= top
        ldmk[:, 0] -= left; ldmk[:, 1] -= top
        bbox /= scale; ldmk /= scale

        return bbox, ldmk

    def detect_faces_batch(self, imgs: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Performs batched inference on a list of images.
        Returns a list of (bbox, landmarks) or None for each input image.
        """
        if not imgs:
            return []

        batch_blobs = []
        metadata = []

        # 1. Preprocess each image individually
        for img in imgs:
            padded, scale, top, left = resize_and_pad(img, self.max_side)
            input_blob = preprocess_image_for_retinaface(padded)  # Expected output: (1, C, H, W)
            batch_blobs.append(input_blob)

            # Store metadata needed to "un-scale" results later
            metadata.append({
                "scale": scale,
                "top": top,
                "left": left,
                "h_w": padded.shape[:2]
            })

        # 2. Stack all images into one batch tensor: (N, C, H, W)
        input_batch = np.concatenate(batch_blobs, axis=0)

        # 3. Perform Batched ONNX Inference
        # loc: (N, num_priors, 4), conf: (N, num_priors, 2), landms: (N, num_priors, 10)
        result = self.onnx_inference.perform_inference(input_batch)
        batch_loc, batch_conf, batch_landms = result

        batch_results = []

        # 4. Post-process each image in the batch
        # Note: We assume all images in the batch have the same padded dimensions (h, w)
        h, w = metadata[0]["h_w"]
        priors = generate_priors(h, w)

        for i in range(len(imgs)):
            meta = metadata[i]
            scores = batch_conf[i][:, 1]

            # Decode results for this specific image in the batch
            bboxes = decode_bboxes(batch_loc[i], priors)
            landmarks = decode_landmarks(batch_landms[i], priors)

            # Scale to padded dimensions
            bboxes[:, [0, 2]] *= w
            bboxes[:, [1, 3]] *= h
            landmarks[:, 0::2] *= w
            landmarks[:, 1::2] *= h

            # Thresholding
            inds = np.where(scores > self.conf_threshold)[0]
            if len(inds) == 0:
                batch_results.append(None)
                continue

            # Select best face
            best_idx = inds[np.argmax(scores[inds])]
            bbox = bboxes[best_idx]
            ldmk = landmarks[best_idx].reshape(-1, 2)

            # 5. Transform coordinates back to original image space
            # Remove padding offset
            bbox[0::2] -= meta["left"]
            bbox[1::2] -= meta["top"]
            ldmk[:, 0] -= meta["left"]
            ldmk[:, 1] -= meta["top"]

            # Rescale to original size
            bbox /= meta["scale"]
            ldmk /= meta["scale"]

            batch_results.append((bbox, ldmk))

        return batch_results

    def align_face_from_landmarks(self, img: np.ndarray, landmarks: np.ndarray):
        assert landmarks.shape == (5, 2)
        M, _ = cv2.estimateAffinePartial2D(landmarks, self.landmarks_target, method=cv2.LMEDS)
        if M is None:
            print("[FaceAligner] Could not estimate affine transform.")
            return None
        return cv2.warpAffine(img, M, self.target_size, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def init_aura_landmarks(self,img_size, ldmks, target=112):
        if img_size[0] != target:
            ldmks[:, 0] *= img_size[0] / target
            ldmks[:, 1] *= img_size[1] / target
        return ldmks

    def align_face_auraface(self,image, landmarks, ):

        assert landmarks.shape == (5, 2)

        dst = AURAFACE_LANDMARKS_5.copy()
        landmarks_target = self.init_aura_landmarks(img_size=self.target_size, ldmks=dst, target=self.target_size[0])

        # estimate similarity transform
        M, _ = cv2.estimateAffinePartial2D(landmarks, landmarks_target, method=cv2.LMEDS)

        if M is None:
            print("could not estimate affine transform with auraface ldmks")
            return None

        aligned = cv2.warpAffine(image, M, self.target_size, flags=cv2.INTER_LINEAR, borderValue=0)  # borderMode=cv2.BORDER_REPLICATE

        return aligned

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

    def add_bbox_offset(self, xmin,xmax,ymin,ymax):

        w = xmax -xmin
        h = ymax - ymin

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        # applying offset to center
        cx_shifted = cx + self.bbox_offset[0]
        cy_shifted = cy + self.bbox_offset[1]

        # bbox around shifted center
        xmin_shifted =  int(cx_shifted - w / 2)
        xmax_shifted =  int(cx_shifted + w / 2)
        ymin_shifted =  int(cy_shifted - h / 2)
        ymax_shifted =  int(cy_shifted + h / 2)

        return xmin_shifted,xmax_shifted,ymin_shifted,ymax_shifted

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

        # apply offset to bbox
        if not self.bbox_offset == (0,0):
            x_min_exp, x_max_exp, y_min_exp, y_max_exp = self.add_bbox_offset(x_min_exp, x_max_exp, y_min_exp, y_max_exp)
        
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
             return np.full((target_height, target_width, img.shape[2]), [0,0,0], dtype=img.dtype)

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
            value=[0,0,0]
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

    def auraface_align_from_bounding_box(self, img: np.ndarray, bounding_box: np.ndarray  ) -> np.ndarray:
        self.face_factor = 0.98
        self.bbox_offset = (0, -10)

        aligned_image = self.align_face_from_bounding_box(img, bounding_box)
        return aligned_image

    def affine_align_from_bounding_box(self, img: np.ndarray, bounding_box: np.ndarray  ) -> np.ndarray:
        self.face_factor = 0.88
        self.bbox_offset = (0, -10)

        aligned_image = self.align_face_from_bounding_box(img, bounding_box)
        return aligned_image

    def align_faces(self, image_paths: list,batch_size:int, debug_folder=''):
        aligned_images=[]

        # Collect results using the new batched method
        batch_results = []
        face_images = []
        face_image_names=[]
        start_batch = time.time()

        # Process images in chunks of batch_size
        for i in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[i:i + batch_size]
            chunk_imgs = [cv2.imread(p) for p in chunk_paths if cv2.imread(p) is not None]

            # Run the batch detector
            res_list =  self.detect_faces_batch(chunk_imgs)
            face_undetected_img_indexes = [idx  for idx, v in enumerate(res_list) if v==None]
            face_images.extend([im for idx, im in enumerate(chunk_imgs) if idx not in face_undetected_img_indexes])
            face_image_names.extend([os.path.basename(p) for idx, p in enumerate(chunk_paths) if idx not in face_undetected_img_indexes])

            batch_results.extend(res_list)

        time_batch = time.time() - start_batch
        print(f"⏱️ Batched detection took: {time_batch:.4f}s")
        if not batch_results:
            print(f"[FaceAligner] No face detected in input images")
            return None
        elif any([res==None for res in batch_results]):
            face_undetected_idx = [idx  for idx, v in enumerate(batch_results) if v==None]
            print(f"[FaceAligner] No face detected in {len(face_undetected_idx)} images")
            print(f"[FaceAligner] Applying alignment for face detected images")
            batch_results = [res for res in batch_results if res != None ]

        [bbox_lst, landmarks_lst] = [list(res) for res in zip(*batch_results) ]

        if self.alignment_method == FaceAligner.AlignmentMethod.AFFINE_TRANSFORM:
            aligned_images = [self.align_face_from_landmarks(img, landmarks)
                              for img, landmarks in zip(face_images,landmarks_lst)]

        elif self.alignment_method == FaceAligner.AlignmentMethod.AURA_FACE_ALIGNER:
            aligned_images = [self.align_face_auraface(img, landmarks)
                              for img, landmarks in zip(face_images,landmarks_lst)]

        elif self.alignment_method == FaceAligner.AlignmentMethod.BOUNDING_BOX_SCALING:
            aligned_images = [self.align_face_from_bounding_box(img, bbox)
                              for img, bbox in zip(face_images, bbox_lst)]

        elif self.alignment_method == FaceAligner.AlignmentMethod.AURA_FROM_BOUNDING_BOX_SCALING:
            aligned_images = [self.auraface_align_from_bounding_box(img, bbox)
                              for img, bbox in zip(face_images, bbox_lst)]

        elif self.alignment_method == FaceAligner.AlignmentMethod.AFFINE_FROM_BOUNDING_BOX_SCALING:
            aligned_images = [self.affine_align_from_bounding_box(img, bbox)
                              for img, bbox in zip(face_images, bbox_lst)]
        else:
            print("Unknown AligmentMethod!")

        if aligned_images:
            if not debug_folder == '':
                if not os.path.exists(debug_folder):
                    os.mkdir(debug_folder)
                else:
                    if len(os.listdir(debug_folder)) >= 1:
                        print(f'[FaceAligner] debug folder Not Empty!, files might be overwritten: {debug_folder}')

                [cv2.imwrite(f'{debug_folder}/{name}', aligned) for name, aligned in zip(face_image_names,aligned_images)]
                print(f"[FaceAligner] Alignment successful, results saved in  → {debug_folder}")

        return aligned_images

    def align_faces_single_batch(self, images: list, debug_folder=''):
        aligned_images=[]

        if not images:
            raise ValueError(f"empty input, num of images : {len(images)}")
        else:
            images = [cv2.imread(im) if isinstance(im, str) else im for im in images]

        results = self.detect_faces_batch(images)
        if not results:
            print(f"[FaceAligner] No face detected in input images")
            return None
        elif any([res==None for res in results]):
            face_undetected_img_indexes = [idx  for idx, v in enumerate(results) if v==None]
            print(f"[FaceAligner] No face detected in {len(face_undetected_img_indexes)} images")
            print(f"[FaceAligner] Applying alignment for face detected images")
            results = [res for res in results if res != None ]
            images =  [im for idx,im in images if idx not in face_undetected_img_indexes]

        [bbox_lst, landmarks_lst] = [list(res) for res in zip(*results) ]

        if self.alignment_method == FaceAligner.AlignmentMethod.AFFINE_TRANSFORM:
            aligned_images = [self.align_face_from_landmarks(img, landmarks)
                              for img, landmarks in zip(images,landmarks_lst)]

        elif self.alignment_method == FaceAligner.AlignmentMethod.AURA_FACE_ALIGNER:
            aligned_images = [self.align_face_auraface(img, landmarks)
                              for img, landmarks in zip(images,landmarks_lst)]

        elif self.alignment_method == FaceAligner.AlignmentMethod.BOUNDING_BOX_SCALING:
            aligned_images = [self.align_face_from_bounding_box(img, bbox)
                              for img, bbox in zip(images, bbox_lst)]

        elif self.alignment_method == FaceAligner.AlignmentMethod.AURA_FROM_BOUNDING_BOX_SCALING:
            aligned_images = [self.auraface_align_from_bounding_box(img, bbox)
                              for img, bbox in zip(images, bbox_lst)]

        elif self.alignment_method == FaceAligner.AlignmentMethod.AFFINE_FROM_BOUNDING_BOX_SCALING:
            aligned_images = [self.affine_align_from_bounding_box(img, bbox)
                              for img, bbox in zip(images, bbox_lst)]
        else:
            print("Unknown AligmentMethod!")

        if aligned_images:
            if not debug_folder == '':
                if not os.path.exists(debug_folder):
                    os.mkdir(debug_folder)
                else:
                    if len(os.listdir(debug_folder)) >= 1:
                        print(f'[FaceAligner] debug folder Not Empty!, files might be overwritten: {debug_folder}')

                [cv2.imwrite(f'{debug_folder}/{i}.jpg', aligned) for i, aligned in enumerate(aligned_images)]
                print(f"[FaceAligner] Alignment successful, results saved in  → {debug_folder}")

        return aligned_images

    def align_face(self, image: str, debug_path=None):
        if type(image) == str:
            img = cv2.imread(image)
        else: 
            img = image
        if img is None:
            raise ValueError(f"Could not read image: {image}")

        result = self.detect_face(img)
        if result is None:
            if type(image) == str:
                print(f"[FaceAligner] No face detected in {image}")
            else:
                print(f"[FaceAligner] No face detected")

            return None

        bbox, landmarks = result
        if self.alignment_method == FaceAligner.AlignmentMethod.AFFINE_TRANSFORM:
            aligned = self.align_face_from_landmarks(img, landmarks)
        elif self.alignment_method == FaceAligner.AlignmentMethod.AURA_FACE_ALIGNER:
            aligned = self.align_face_auraface(img, landmarks)
        elif self.alignment_method == FaceAligner.AlignmentMethod.BOUNDING_BOX_SCALING:
            aligned = self.align_face_from_bounding_box(img, bbox)
        elif self.alignment_method == FaceAligner.AlignmentMethod.AURA_FROM_BOUNDING_BOX_SCALING:
            aligned = self.auraface_align_from_bounding_box(img, bbox)
        elif self.alignment_method == FaceAligner.AlignmentMethod.AFFINE_FROM_BOUNDING_BOX_SCALING:
            aligned = self.affine_align_from_bounding_box(img, bbox)
        else:
            print("Unknown AligmentMethod!")
        if aligned is not None and debug_path is not None:
            cv2.imwrite(debug_path, aligned)
            #print(f"[FaceAligner] Alignment successful → {debug_path}")
        return aligned
