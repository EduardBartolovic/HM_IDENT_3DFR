
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


def expand_bbox(x_min, y_min, x_max, y_max, factor=0.25):
    """Expand the bounding box by a given factor and make it square."""
    width = x_max - x_min
    height = y_max - y_min

    # Expand the bbox dimensions
    x_min_new = x_min - int(factor * height)
    y_min_new = y_min - int(factor * width)
    x_max_new = x_max + int(factor * height)
    y_max_new = y_max + int(factor * width)

    # Ensure square by finding the max side length
    square_size = max(x_max_new - x_min_new, y_max_new - y_min_new)
    center_x = (x_min_new + x_max_new) // 2
    center_y = (y_min_new + y_max_new) // 2

    # Calculate new square bounding box
    half_size = square_size // 2
    x_min_square = max(0, center_x - half_size)
    y_min_square = max(0, center_y - half_size)
    x_max_square = x_min_square + square_size
    y_max_square = y_min_square + square_size

    return x_min_square, y_min_square, x_max_square, y_max_square
