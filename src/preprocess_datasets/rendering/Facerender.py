import os
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm


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


def generate_rotation_matrices_cross_x_y():
    # Vertical line
    vertical = [
        (
            x,
            0,
            np.dot(
                np.array([
                    [1, 0, 0],
                    [0, np.cos(np.radians(x)), -np.sin(np.radians(x))],
                    [0, np.sin(np.radians(x)),  np.cos(np.radians(x))]
                ]),
                np.eye(3)  # y=0 => identity rotation around y
            )
        )
        for x in range(-35, 36, 1)
    ]
    # horizontal line
    horizontal = [
        (
            0,
            y,
            np.dot(
                np.eye(3),  # x=0 => identity rotation around x
                np.array([
                    [np.cos(np.radians(y)), 0, np.sin(np.radians(y))],
                    [0, 1, 0],
                    [-np.sin(np.radians(y)), 0, np.cos(np.radians(y))]
                ])
            )
        )
        for y in range(-45, 46, 1)
    ]

    return vertical + horizontal


def generate_rotation_matrices_full_x_y():
    xs = range(-45, 46, 1)
    ys = range(-45, 46, 1)

    rotations = []
    for x in xs:
        # Rotation around X axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(x)), -np.sin(np.radians(x))],
            [0, np.sin(np.radians(x)),  np.cos(np.radians(x))]
        ])

        for y in ys:
            # Rotation around Y axis
            Ry = np.array([
                [np.cos(np.radians(y)), 0, np.sin(np.radians(y))],
                [0, 1, 0],
                [-np.sin(np.radians(y)), 0, np.cos(np.radians(y))]
            ])

            # Combined rotation: first x, then y
            R = np.dot(Ry, Rx)

            rotations.append((x, y, R))

    return rotations


def render(output_image_dir, headscan, flipped=False, render_angles=None):
    if render_angles is None:
        render_angles = [-10, 0, 10]

    file_abspath = headscan['obj_file_path']
    rotation_matrices = generate_rotation_matrices_cross_x_y() + generate_rotation_matrices(render_angles)

    render_jobs = []
    for rotation_m in rotation_matrices:
        if flipped:
            R = rotation_m[2].copy()
            R[1] = -R[1]
            R[2] = -R[2]
            rotation_m = (rotation_m[0], rotation_m[1], R)

        folder = os.path.join(
            output_image_dir,
            headscan['user'],
            headscan['date'],
            headscan['scan_id'] + '_' + headscan['scan_name']
        )

        fn = f"{rotation_m[0]}_{rotation_m[1]}"
        img_path = os.path.join(folder, fn + "_image.jpg")
        depth_path = os.path.join(folder, fn + "_depth.jpg")

        if not os.path.exists(img_path): # and os.path.exists(depth_path)):
            render_jobs.append((rotation_m[0], rotation_m[1], rotation_m[2], folder, img_path, depth_path))

    if len(render_jobs) == 0:
        return 1

    print(file_abspath)

    if headscan.get('texture_path') is not None:
        tex = o3d.geometry.Image(
            cv2.cvtColor(cv2.imread(headscan['texture_path']), cv2.COLOR_BGR2RGB)
        )
    else:
        tex = None

    base_mesh = read_mesh(file_abspath, tex)
    base_mesh.compute_vertex_normals()

    # Create one persistent visualizer
    W, H = 1920, 1080
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H, visible=False)
    vis.add_geometry(base_mesh)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.light_on = False

    # Camera setup
    ctr = vis.get_view_control()
    ctr.set_zoom(2.0)

    param = ctr.convert_to_pinhole_camera_parameters()
    original_extrinsic = param.extrinsic.copy()

    # Run through all render jobs
    for Xdeg, Ydeg, R_target, folder, img_path, depth_path in tqdm(render_jobs, desc="Render"):

        # Convert 3×3 to 4×4
        R4 = np.eye(4)
        R4[:3, :3] = R_target

        # orbit the camera instead of rotating the mesh
        param.extrinsic = original_extrinsic @ R4
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.poll_events()
        vis.update_renderer()

        # Depth
        depth_float = vis.capture_depth_float_buffer(do_render=True)
        depth_np = np.asarray(depth_float)

        # RGB
        rgb_float = vis.capture_screen_float_buffer(do_render=False)
        rgb_np = np.asarray(rgb_float)

        depth_np, rgb_np = postprocess_renderer_image(depth_np, rgb_np)

        os.makedirs(folder, exist_ok=True)
        # plt.imsave(depth_path, depth_np, cmap='gray', dpi=1)
        plt.imsave(img_path, rgb_np, dpi=1)

    vis.destroy_window()
