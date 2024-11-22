from src.preprocess_datasets.headPoseEstimation.detect import expand_bbox


def cut_face(face_detector, image):

    bboxes, _ = face_detector.detect(image)
    worked_fine = True
    if len(bboxes) != 1:
        print(bboxes)
        worked_fine = False
        if len(bboxes) == 0:
            return image, worked_fine

    x_min, y_min, x_max, y_max = map(int, bboxes[0][:4])
    x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max)

    image_cut = image[y_min:y_max, x_min:x_max]

    return image_cut, worked_fine
