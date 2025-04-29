import hashlib
import os

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm


class DepthCaptureCallback:
    def __init__(self, vis):
        self.vis = vis
        self.depth = None
        self.image = None

    def capture(self):
        self.vis.poll_events()
        self.vis.update_renderer()
        self.depth = np.asarray(self.vis.capture_depth_float_buffer())
        self.image = np.asarray(self.vis.capture_screen_float_buffer())

    def get_depth_image(self):
        return self.depth, self.image


def read_mesh(path, texture):
    mesh = o3d.io.read_triangle_mesh(path)
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = mesh.vertices
    new_mesh.triangles = mesh.triangles
    new_mesh.triangle_uvs = mesh.triangle_uvs
    new_mesh.triangle_material_ids = mesh.triangle_material_ids
    new_mesh.vertex_colors = mesh.vertex_colors

    if texture is None:
        try:
            new_mesh.textures = [mesh.textures[1], mesh.textures[1]]
        except:
            print('no texture')
    else:

        if not mesh.has_triangle_uvs():
            print("The mesh does not have UV coordinates.")
            raise Exception('')

        print(len(np.asarray(new_mesh.triangles)))
        v_uv = np.random.rand(len(np.asarray(new_mesh.triangles)) * 3, 2)
        new_mesh.triangle_uvs = o3d.open3d_pybind.utility.Vector2dVector(v_uv)

        new_mesh.textures = [texture, texture]  # TODO: DOESNT WORK YET

    return new_mesh


def render_rgbd(model, rotation_matrix):
    # Apply the rotation to the mesh
    model.rotate(rotation_matrix)

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(1920), height=int(1080))

    vis.add_geometry(model)

    view_ctl = vis.get_view_control()
    ZOOM_FACTOR = 2
    view_ctl.set_zoom(ZOOM_FACTOR)

    callback = DepthCaptureCallback(vis)
    callback.capture()
    depth, image = callback.get_depth_image()

    return depth, image


def postprocess_renderer_image(depth, image):
    # Threshold the image to get a binary mask where values greater than 0 are set to 255
    _, binary_mask = cv2.threshold(depth, 0, 255, cv2.THRESH_BINARY)

    # Convert the binary mask to CV_8UC1 format
    binary_mask = binary_mask.astype(np.uint8)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contours found in depth image")

    # Find the largest contour based on contour area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box around the contours
    b_box = cv2.boundingRect(largest_contour)

    # Ensure the bounding box is a square
    max_dim = max(b_box[2], b_box[3])
    center_x = b_box[0] + b_box[2] // 2
    center_y = b_box[1] + b_box[3] // 2
    new_x = center_x - max_dim // 2
    new_y = center_y - max_dim // 2
    b_box = (new_x, new_y, max_dim, max_dim)

    # Crop the image using the adjusted bounding box coordinates
    cropped_image = image[b_box[1]:b_box[1] + b_box[3], b_box[0]:b_box[0] + b_box[2]]
    cropped_depth = depth[b_box[1]:b_box[1] + b_box[3], b_box[0]:b_box[0] + b_box[2]]

    # plt.imshow(cropped_image, interpolation='nearest')
    # plt.show()
    return cropped_depth, cropped_image


def generate_rotation_matrices(angle_range):
    rotation_matrices = [
        (
            x,
            y,
            np.dot(
                np.array([[1, 0, 0], [0, np.cos(np.radians(x)), -np.sin(np.radians(x))],
                          [0, np.sin(np.radians(x)), np.cos(np.radians(x))]]),
                np.array([[np.cos(np.radians(y)), 0, np.sin(np.radians(y))], [0, 1, 0],
                          [-np.sin(np.radians(y)), 0, np.cos(np.radians(y))]])
            )
        )
        for x in angle_range
        for y in angle_range
    ]
    return rotation_matrices


def file_hash(file_path, algorithm='md5'):
    # Initialize hash object
    hasher = hashlib.new(algorithm)

    # Calculate hash
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def render(output_image_dir, headscan, flipped=False, render_angles=None):
    if render_angles is None:
        render_angles = [-10, 0, 10]

    file_abspath = headscan['obj_file_path']
    rotation_matrices = generate_rotation_matrices(render_angles)

    render_jobs = []
    for rotation_m in rotation_matrices:

        if flipped:  # Flip whole render by 180°
            rotation_m_flipped = rotation_m[2].copy()
            rotation_m_flipped[1] = -rotation_m_flipped[1]
            rotation_m_flipped[2] = -rotation_m_flipped[2]
            rotation_m = (rotation_m[0], rotation_m[1], rotation_m_flipped)

        path = os.path.join(output_image_dir,
                            headscan['user'],
                            headscan['date'],
                            headscan['scan_id'] + '_' + headscan['scan_name'])
        file_name = str(rotation_m[0]) + '_' + str(rotation_m[1])
        targetfile_depth = os.path.join(path, file_name + '_depth.jpg')
        targetfile_image = os.path.join(path, file_name + '_image.jpg')
        if not (os.path.exists(targetfile_depth) and os.path.exists(targetfile_image)):
            render_jobs.append((rotation_m[2], path, targetfile_image, targetfile_depth))

    if len(render_jobs) == 0:
        return 1

    print(file_abspath)
    # Read all render files
    if headscan.get('texture_path') is not None:
        print(headscan.get('texture_path'))
        # texture = o3d.io.read_image(headscan.get('texture_path'))
        texture = o3d.geometry.Image(cv2.cvtColor(cv2.imread(headscan.get('texture_path')), cv2.COLOR_BGR2RGB))
    else:
        texture = None

    model_ori = read_mesh(file_abspath, texture)

    for rotation_m, path, targetfile_image, targetfile_depth in tqdm(render_jobs, position=1, leave=True,
                                                                     desc="Render jobs"):

        os.makedirs(path, exist_ok=True)
        try:
            model = model_ori.__copy__()
            depth, image = render_rgbd(model, rotation_m)
        except Exception as e:
            print('Error at', path)
            raise e

        # plt.imshow(depth, interpolation='nearest')
        # plt.show()
        # plt.imshow(image, interpolation='nearest')
        # plt.show()

        depth, image = postprocess_renderer_image(depth, image)

        try:
            plt.imsave(targetfile_depth, np.asarray(depth), dpi=1, cmap='gray')
            plt.imsave(targetfile_image, np.asarray(image), dpi=1)
        except KeyboardInterrupt:
            print(f"\nInterrupted while saving image at {path}. Attempting to finish saving and exit cleanly.")
            try:
                # Try final save attempt before quitting
                plt.imsave(targetfile_depth, np.asarray(depth), dpi=1, cmap='gray')
                plt.imsave(targetfile_image, np.asarray(image), dpi=1)
            except Exception as save_err:
                print(f"Failed to save images due to: {save_err}")
            raise  # Re-raise to exit the loop cleanly
