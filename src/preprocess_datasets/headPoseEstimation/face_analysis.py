from src.preprocess_datasets.headPoseEstimation.hpe_to_dataset import read_file
import os
import cv2
import mediapipe as mp

def process_landmarks(multi_face_landmarks, dst, r_pred_deg, image):
    for face_landmarks in multi_face_landmarks:
        # Get image dimensions
        h, w, _ = image.shape
        x_coords = [int(landmark.x * w) for landmark in face_landmarks.landmark]
        y_coords = [int(landmark.y * h) for landmark in face_landmarks.landmark]

        # Calculate bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Expand bounding box with padding
        padding = int(w*0.25)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Apply roll compensation to the full frame
        center_x = int(x_max - x_min) // 2
        center_y = int(y_max - y_min) // 2
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), r_pred_deg, 1.0)
        image = cv2.warpAffine(
            image,
            rotation_matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Crop and resize the face
        face_crop = image[y_min:y_max, x_min:x_max]
        resized_face = cv2.resize(face_crop, (224, 224))  # Adjust as needed

        cv2.imwrite(dst, resized_face)


def face_analysis(input_folder, output_folder, device):

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    for root, _, files in os.walk(input_folder):
        if "hpe" in root:
            for txt_file in files:
                if txt_file.endswith('hpe.txt'):
                    file_path = os.path.join(root, txt_file)
                    data = read_file(file_path, remove_header=False)

                    folder_image_path = os.path.join(root, "..", "frames_cropped")
                    folder_image_output_path = os.path.join(root, "..", "face_cropped")

                    os.makedirs(folder_image_output_path, exist_ok=True)

                    last_land_marks = None
                    for info in data:
                        r_pred_deg = int(info[2])
                        src = os.path.join(folder_image_path, info[3])
                        dst = os.path.join(folder_image_output_path, info[3])

                        image = cv2.imread(src)
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb_image)

                        if results.multi_face_landmarks:
                            process_landmarks(results.multi_face_landmarks, dst, r_pred_deg, image)
                            last_land_marks = results.multi_face_landmarks
                        else:
                            if last_land_marks:
                                print("NO LANDMARKS BUT USED PREV", dst)
                                process_landmarks(last_land_marks, dst, r_pred_deg, image)
                            else:
                                print("NO LANDMARKS")
                                raise Exception("NO LANDMARKS")
